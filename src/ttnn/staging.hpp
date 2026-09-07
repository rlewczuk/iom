#pragma once

// Retained TTNN host-transfer staging facility (performance finding
// PF-004). Repeated synchronous host transfers used to allocate fresh
// staging on every operation: a zero-filled typed vector per owner plane
// per upload and one opaque byte buffer per download. This facility keeps
// one byte staging slot per supported native upload dtype per plane plus
// one byte staging buffer for downloads, all owned by the TTNN device, so
// after warm-up repeated same-or-smaller transfers allocate no host staging
// at all and larger shapes grow the retained buffers once and then reuse
// them.
//
// Ownership and ordering: every method is called only while the owning
// device's API mutex is held (region_from_host/region_to_host in copy.cpp
// run under it). A lease hands out one retained slot; the transfer returns
// it strictly after the region's queue finish consumed the data. A lease
// destroyed without an explicit release means the transfer failed: upload
// slots are discarded, a download whose drain finished is discarded, and a
// download whose drain could not be finished is retired (never reused,
// freed only with the device), so an asynchronous reader can never touch
// reused memory. Poisoned storage is never handed out again.

#include <atomic>
#include <array>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace iom::ttnn_detail {

    // Process-global host-transfer staging test seams (PF-004). Host
    // transfers run synchronously on the caller thread under the device API
    // mutex; the atomics mirror the seam style of the submission faults in
    // copy.cpp, where an arming release store from the test and an
    // acquire-loading transfer read never race.
    inline std::atomic<std::size_t> g_host_staging_allocations{0};
    inline std::atomic<bool> g_fail_next_host_staging_allocation{false};
    inline std::atomic<bool> g_host_staging_allocation_fault_consumed{false};

    class TtnnHostStaging final {
    public:
        // One upload slot list per supported native upload dtype, in the
        // order chosen by upload_slot_index in copy.cpp: BFLOAT16, FLOAT32,
        // UINT32, INT32, UINT16, UINT8.
        static constexpr std::size_t kUploadSlotListCount = 6;

        class UploadLease final {
        public:
            ~UploadLease() noexcept {
                // A lease destroyed without an explicit release means the
                // transfer failed: discard the slot so no poisoned bytes
                // are ever reused.
                if (pool_ != nullptr) {
                    pool_->release_upload(
                            list_index_, position_, /*poisoned=*/true);
                }
            }

            UploadLease(const UploadLease&) = delete;
            UploadLease& operator=(const UploadLease&) = delete;
            UploadLease(UploadLease&& other) noexcept
                    : pool_(std::exchange(other.pool_, nullptr)),
                      list_index_(other.list_index_),
                      position_(other.position_),
                      data_(other.data_),
                      keepalive_(std::move(other.keepalive_)) {}
            UploadLease& operator=(UploadLease&&) = delete;

            [[nodiscard]] std::byte* data() const noexcept { return data_; }

            // Aliasing keep-alive for the tt-metal HostBuffer pin: holds a
            // no-op-deleter shared_ptr to the slot's vector so the borrowed
            // host tensor can never free the facility's retained buffer.
            [[nodiscard]] std::shared_ptr<void> keepalive() const noexcept {
                return keepalive_;
            }

            // Returns the retained slot to the facility. Called only after
            // the region's queue finish has consumed the staged data.
            void release() noexcept {
                if (pool_ != nullptr) {
                    pool_->release_upload(
                            list_index_, position_, /*poisoned=*/false);
                    pool_ = nullptr;
                }
            }

        private:
            friend class TtnnHostStaging;

            UploadLease(
                    TtnnHostStaging& pool, std::size_t list_index,
                    std::size_t position, std::byte* data,
                    std::shared_ptr<void> keepalive) noexcept
                    : pool_(&pool), list_index_(list_index),
                      position_(position), data_(data),
                      keepalive_(std::move(keepalive)) {}

            TtnnHostStaging* pool_ = nullptr;
            std::size_t list_index_ = 0;
            std::size_t position_ = 0;
            std::byte* data_ = nullptr;
            std::shared_ptr<void> keepalive_;
        };

        class DownloadLease final {
        public:
            ~DownloadLease() noexcept {
                // A lease destroyed without a disposition is a failed
                // transfer: never hand the abandoned staging out again.
                if (pool_ != nullptr) {
                    pool_->release_download(
                            DownloadDisposition::Discard);
                }
            }

            DownloadLease(const DownloadLease&) = delete;
            DownloadLease& operator=(const DownloadLease&) = delete;
            DownloadLease(DownloadLease&&) = delete;
            DownloadLease& operator=(DownloadLease&&) = delete;

            [[nodiscard]] std::byte* data() const noexcept { return data_; }

            // The region completed: both the submissions and the single
            // finish drained; the staging returns to the retained pool.
            void release() noexcept {
                if (pool_ != nullptr) {
                    pool_->release_download(DownloadDisposition::Keep);
                    pool_ = nullptr;
                }
            }

            // A failed transfer whose drain finished: the staging may hold
            // partial bytes; discard it so the next transfer allocates
            // fresh, unpoisoned storage.
            void discard() noexcept {
                if (pool_ != nullptr) {
                    pool_->release_download(DownloadDisposition::Discard);
                    pool_ = nullptr;
                }
            }

            // A failed transfer whose drain could not finish: an
            // asynchronous reader may still touch the staging; retire it so
            // it is never reused and is freed only with the device.
            void retire() noexcept {
                if (pool_ != nullptr) {
                    pool_->release_download(DownloadDisposition::Retire);
                    pool_ = nullptr;
                }
            }

        private:
            friend class TtnnHostStaging;

            DownloadLease(
                    TtnnHostStaging& pool, std::byte* data) noexcept
                    : pool_(&pool), data_(data) {}

            TtnnHostStaging* pool_ = nullptr;
            std::byte* data_ = nullptr;
        };

        // Hands out one retained byte slot from the dtype's slot list,
        // growing the list and the storage only when no retained buffer
        // fits, and zero-initializing only the newly claimed tail.
        [[nodiscard]] UploadLease acquire_upload(
                std::size_t list_index, std::size_t required_bytes) {
            std::vector<UploadSlot>& slots =
                    upload_slots_.at(list_index);
            UploadSlot* free_slot = nullptr;
            std::size_t position = 0;
            for (std::size_t i = 0; i < slots.size(); ++i) {
                if (!slots[i].in_use) {
                    free_slot = &slots[i];
                    position = i;
                    break;
                }
            }
            if (free_slot == nullptr) {
                // The list grows by one retained slot for this plane; slot
                // positions (not pointers) identify leases, so the
                // reallocation below never invalidates an outstanding lease.
                slots.push_back(UploadSlot{});
                free_slot = &slots.back();
                position = slots.size() - 1;
            }
            grow(free_slot->data, required_bytes);
            free_slot->in_use = true;
            std::shared_ptr<std::vector<std::byte>> anchored(
                    &free_slot->data, [](std::vector<std::byte>*) {});
            return UploadLease{
                    *this, list_index, position, free_slot->data.data(),
                    std::shared_ptr<void>(std::move(anchored))};
        }

        // Hands out the retained byte staging buffer, growing it only when
        // no retained allocation fits.
        [[nodiscard]] DownloadLease acquire_download(
                std::size_t required_bytes) {
            if (download_slot_.in_use) {
                throw std::logic_error(
                        "TTNN download staging slot already leased");
            }
            grow(download_slot_.data, required_bytes);
            download_slot_.in_use = true;
            return DownloadLease{
                    *this, download_slot_.data.data()};
        }

    private:
        struct UploadSlot {
            std::vector<std::byte> data;
            bool in_use = false;
        };

        struct DownloadSlot {
            std::vector<std::byte> data;
            bool in_use = false;
        };

        enum class DownloadDisposition { Keep, Discard, Retire };

        // Grows the retained buffer when no retained allocation fits; the
        // claimed region beyond the current size is zero-initialized. Each
        // underlying storage allocation is counted for the testing seams.
        void grow(
                std::vector<std::byte>& data,
                std::size_t required_bytes) {
            if (required_bytes > data.capacity()) {
                consume_allocation_fault();
                data.resize(required_bytes);
                g_host_staging_allocations.fetch_add(
                        1, std::memory_order_relaxed);
            } else if (required_bytes > data.size()) {
                data.resize(required_bytes);
            }
        }

        void consume_allocation_fault() noexcept(false) {
            if (g_fail_next_host_staging_allocation.exchange(
                        false, std::memory_order_acq_rel)) {
                g_host_staging_allocation_fault_consumed.store(
                        true, std::memory_order_release);
                throw std::bad_alloc();
            }
        }

        void release_upload(
                std::size_t list_index, std::size_t position,
                bool poisoned) noexcept {
            std::vector<UploadSlot>& slots = upload_slots_[list_index];
            if (position >= slots.size() || !slots[position].in_use) {
                return;
            }
            slots[position].in_use = false;
            if (poisoned) {
                std::vector<std::byte>().swap(slots[position].data);
            }
        }

        void release_download(DownloadDisposition disposition) noexcept {
            DownloadSlot& slot = download_slot_;
            if (!slot.in_use) {
                return;
            }
            slot.in_use = false;
            switch (disposition) {
                case DownloadDisposition::Keep:
                    break;
                case DownloadDisposition::Discard:
                    std::vector<std::byte>().swap(slot.data);
                    break;
                case DownloadDisposition::Retire:
                    retired_.push_back(std::move(slot.data));
                    slot.data.clear();
                    break;
            }
        }

        std::array<std::vector<UploadSlot>, kUploadSlotListCount>
                upload_slots_;
        DownloadSlot download_slot_;
        std::vector<std::vector<std::byte>> retired_;

        friend class UploadLease;
        friend class DownloadLease;
    };

}  // namespace iom::ttnn_detail