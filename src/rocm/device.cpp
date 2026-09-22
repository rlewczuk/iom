#include "iom/rocm/device.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "copy.hpp"
#include "device_internal.hpp"
#include "driver.hpp"
#include "iom/detail/gpu_arena_config.hpp"
#include "../shared/standard_tiled_copy.hpp"

namespace iom {

    namespace {
        void free_status_cells(void* address) noexcept {
            if (address != nullptr) {
                (void)hipHostFree(address);
            }
        }

        // A native device-memory failure maps to std::bad_alloc; any other
        // runtime status keeps the established runtime category.
        void check_arena_alloc_hip(const char* operation, hipError_t status) {
            if (status == hipErrorOutOfMemory) {
                throw std::bad_alloc();
            }
            rocm_detail::check_hip(operation, status);
        }

        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, int device_count) {
            return std::invalid_argument(
                    "ROCm device ordinal " + std::to_string(ordinal)
                    + " is unavailable; device count is "
                    + std::to_string(device_count));
        }

    }  // namespace

    RocmDevice::RocmDevice(
            std::uint32_t ordinal,
            std::unique_ptr<ListAllocator> data_allocator,
            std::unique_ptr<FixedSizeAllocator> metadata_allocator,
            void* data_backing, std::size_t data_backing_bytes,
            void* metadata_backing)
            : ordinal_(ordinal),
              transfer_resource_(nullptr),
              data_allocator_(std::move(data_allocator)),
              metadata_allocator_(std::move(metadata_allocator)),
              data_backing_(data_backing),
              data_backing_bytes_(data_backing_bytes),
              metadata_backing_(metadata_backing),
              queue_slot_count_(metadata_allocator_->block_count()
                                / detail::kMaxLiveGpuQueues) {
        activate();
        transfer_resource_ = std::make_unique<RocmTransferResource>();
    }

    RocmDevice::~RocmDevice() {
        try {
            activate();
            transfer_resource_.reset();
        } catch (...) {
        }
        registry_state_.quarantine.drain();
        // Covering-proof every retained unknown-completion lease;
        // unproven bundles are discarded best-effort while the
        // ordinal is still current, before allocator bookkeeping
        // disappears.
        reclaim_retained_leases();
        discard_all_retained_leases();
        // Destroy allocator bookkeeping before releasing the native
        // backings so no live owner, lease, or retained quarantine
        // action can reference an allocator past this point; drain
        // above already freed every recognized range back into the
        // data allocator.
        data_allocator_.reset();
        metadata_allocator_.reset();
        if (data_backing_ != nullptr) {
            (void)rocm_detail::free_attempt(
                    data_backing_,
                    rocm_detail::AllocationClass::data_backing,
                    rocm_detail::AllocationPhase::post_publication);
            data_backing_ = nullptr;
        }
        if (metadata_backing_ != nullptr) {
            (void)rocm_detail::free_attempt(
                    metadata_backing_,
                    rocm_detail::AllocationClass::metadata_backing,
                    rocm_detail::AllocationPhase::post_publication);
            metadata_backing_ = nullptr;
        }
    }

    [[nodiscard]] BackendKind RocmDevice::backend_kind() const noexcept {
        return BackendKind::ROCM;
    }

    [[nodiscard]] std::uint32_t RocmDevice::backend_device() const noexcept {
        return ordinal_;
    }

    [[nodiscard]] std::span<const iom::DataType>
            RocmDevice::supported_data_types() const noexcept {
        return detail::standard_supported_data_types();
    }

    [[nodiscard]] std::uint32_t RocmDevice::ordinal() const noexcept {
        return ordinal_;
    }

    [[nodiscard]] std::unique_ptr<DeviceOps> RocmDevice::create_ops() {
        activate();
        // The native causal grouped-query SDPA capability is a runtime fact of
        // this exact device and of the image the HIP runtime loaded for it, so
        // it is resolved here, once per queue, and never re-read from a
        // submission thread or from the pure workspace query.
        return rocm_detail::make_queue(
                *this, *this, static_cast<int>(ordinal_), registry_state_,
                rocm_detail::sdpa_native_capability(
                        static_cast<int>(ordinal_)));
    }

    [[nodiscard]] std::size_t
            RocmDevice::queue_slot_count() const noexcept {
        return queue_slot_count_;
    }

    [[nodiscard]] detail::QueueResourceLease
            RocmDevice::reserve_queue_resources() {
        activate();
        std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
        const std::size_t slot_count = queue_slot_count_;
        std::vector<std::size_t> indices;
        indices.reserve(slot_count);
        try {
            for (std::size_t index = 0; index < slot_count; ++index) {
                void* block = metadata_allocator_->alloc(
                        detail::kMetadataSlotBytes);
                indices.push_back(metadata_allocator_->index_of(block));
            }
        } catch (...) {
            for (const std::size_t index : indices) {
                metadata_allocator_->free(
                        metadata_allocator_->ptr_from_index(index));
            }
            // Propagates std::bad_alloc for a fifth live queue.
            throw;
        }
        std::sort(indices.begin(), indices.end());
        const std::size_t base = indices.front();
        for (std::size_t index = 0; index < slot_count; ++index) {
            if (indices[index] != base + index
                    || base % slot_count != 0) {
                release_queue_resources_locked(base, slot_count);
                throw std::logic_error(
                        "metadata partition reservation lost the "
                        "device's C-slot geometry");
            }
        }
        const std::size_t mirror_bytes =
                slot_count * detail::kMetadataSlotBytes;
        std::unique_ptr<std::byte[]> host_mirrors;
        try {
            host_mirrors = std::make_unique<std::byte[]>(mirror_bytes);
        } catch (...) {
            release_queue_resources_locked(base, slot_count);
            throw;
        }
        if (slot_count > std::numeric_limits<std::size_t>::max()
                / sizeof(std::uint32_t)) {
            release_queue_resources_locked(base, slot_count);
            throw std::overflow_error(
                    "ROCm status-cell allocation size overflows");
        }
        const std::size_t status_bytes =
                slot_count * sizeof(std::uint32_t);
        void* status_cells = nullptr;
        const hipError_t status =
                hipHostMalloc(&status_cells, status_bytes, hipHostMallocDefault);
        if (status == hipErrorOutOfMemory) {
            free_status_cells(status_cells);
            release_queue_resources_locked(base, slot_count);
            throw std::bad_alloc();
        }
        if (status != hipSuccess) {
            free_status_cells(status_cells);
            release_queue_resources_locked(base, slot_count);
            throw rocm_detail::hip_error("hipHostMalloc", status);
        }
        return make_lease(
                *this, base, slot_count,
                metadata_allocator_->ptr_from_index(base),
                std::move(host_mirrors), status_cells, &free_status_cells);
    }

    void RocmDevice::release_queue_resources(
            std::size_t first_slot, std::size_t slot_count) noexcept {
        std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
        release_queue_resources_locked(first_slot, slot_count);
    }

    void RocmDevice::retain_unknown_lease(
            std::function<bool()> reclaim,
            std::function<void()> discard) {
        std::lock_guard<std::mutex> lock(retained_mutex_);
        if (retained_count_ >= detail::kMaxLiveGpuQueues) {
            // One retained lease per partition; this cannot exceed
            // the queue-count bound.
            throw std::logic_error(
                    "retained queue-lease quarantine is full");
        }
        retained_leases_[retained_count_].reclaim = std::move(reclaim);
        retained_leases_[retained_count_].discard = std::move(discard);
        ++retained_count_;
    }

    void RocmDevice::reclaim_retained_leases() {
        std::array<RetainedLease, detail::kMaxLiveGpuQueues> pending;
        std::size_t pending_count = 0;
        {
            std::lock_guard<std::mutex> lock(retained_mutex_);
            pending_count = retained_count_;
            for (std::size_t index = 0; index < pending_count; ++index) {
                pending[index] = std::move(retained_leases_[index]);
                retained_leases_[index] = {};
            }
            retained_count_ = 0;
        }
        // Callbacks synchronize retained streams: never under any device
        // lock.
        for (std::size_t index = 0; index < pending_count; ++index) {
            if (!pending[index].reclaim()) {
                std::lock_guard<std::mutex> lock(retained_mutex_);
                retained_leases_[retained_count_++] =
                        std::move(pending[index]);
            }
        }
    }

    void RocmDevice::activate() const {
        rocm_detail::check_hip(
                "hipSetDevice",
                hipSetDevice(static_cast<int>(ordinal_)));
    }

    [[nodiscard]] detail::RegistryState&
            RocmDevice::registry_state() noexcept {
        return registry_state_;
    }

    [[nodiscard]] void* RocmDevice::allocate_data(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
        return data_allocator_->alloc(bytes);
    }

    void RocmDevice::release_data(void* address) {
        std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
        data_allocator_->free(address);
    }

    void RocmDevice::release_workspace(
            const RawWorkspace* owner, void* address,
            std::size_t bytes) noexcept {
        bool retained = false;
        {
            std::lock_guard<std::mutex> lock(
                    registry_state_.allocation_mutex);
            retained = detail::workspace_range_retained(
                    registry_state_.workspace_leases, owner, address, bytes);
        }
        if (retained) {
            try {
                registry_state_.quarantine.emplace<detail::AllocatorCleanupAction>(
                        *data_allocator_, address, bytes);
            } catch (...) {
                // Keep failed storage unavailable if quarantine allocation
                // itself fails.
            }
            return;
        }
        release_data(address);
    }

    [[nodiscard]] iom::Allocator& RocmDevice::data_allocator() noexcept {
        return *data_allocator_;
    }

    [[nodiscard]] void* RocmDevice::data_backing() const noexcept {
        return data_backing_;
    }

    [[nodiscard]] std::size_t
            RocmDevice::data_backing_bytes() const noexcept {
        return data_backing_bytes_;
    }

    void RocmDevice::release_queue_resources_locked(
            std::size_t first_slot, std::size_t slot_count) noexcept {
        for (std::size_t index = 0; index < slot_count; ++index) {
            metadata_allocator_->free(
                    metadata_allocator_->ptr_from_index(
                            first_slot + index));
        }
    }

    void RocmDevice::discard_all_retained_leases() noexcept {
        std::lock_guard<std::mutex> lock(retained_mutex_);
        for (std::size_t index = 0; index < retained_count_; ++index) {
            retained_leases_[index].discard();
            retained_leases_[index] = {};
        }
        retained_count_ = 0;
    }
    namespace {
        // Factory-local RAII owner for one arena backing. Setup rollback
        // frees the backing through the same instrumented boundary used for
        // its reservation, in reverse acquisition order, while the ordinal
        // is still current; the original failure is preserved.
        class RocmBackingGuard final {
        public:
            RocmBackingGuard() = default;
            explicit RocmBackingGuard(
                    void* address, rocm_detail::AllocationClass classification)
                    : address_(address), classification_(classification) {}

            RocmBackingGuard(const RocmBackingGuard&) = delete;
            RocmBackingGuard& operator=(const RocmBackingGuard&) = delete;

            ~RocmBackingGuard() noexcept {
                if (address_ != nullptr) {
                    (void)rocm_detail::free_attempt(
                            address_, classification_,
                            rocm_detail::AllocationPhase::setup);
                }
            }

            void dismiss() noexcept { address_ = nullptr; }

        private:
            void* address_ = nullptr;
            rocm_detail::AllocationClass classification_ =
                    rocm_detail::AllocationClass::other_iom_setup;
        };
    }  // namespace

    std::unique_ptr<Device> make_rocm_device(
            std::uint32_t device_ordinal, DeviceMemoryConfig memory_config,
            QueueConfig queue_config) {
        const std::size_t data_bytes = detail::valid_tensor_arena_bytes(
                memory_config.tensor_arena_bytes);
        const std::size_t metadata_bytes =
                detail::standard_gpu_metadata_bytes(
                        queue_config.max_in_flight_per_queue);

        int device_count = 0;
        rocm_detail::check_hip(
                "hipGetDeviceCount", hipGetDeviceCount(&device_count));
        if (device_count <= 0
                || device_ordinal
                        >= static_cast<std::uint32_t>(device_count)) {
            throw invalid_ordinal(device_ordinal, device_count);
        }
        if (device_ordinal
                > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw invalid_ordinal(device_ordinal, device_count);
        }

        rocm_detail::check_hip(
                "hipSetDevice",
                hipSetDevice(static_cast<int>(device_ordinal)));

        // Reserve exactly two native backings on this exact ordinal: one
        // data arena and one metadata arena of checked capacity.
        void* data_backing = nullptr;
        check_arena_alloc_hip(
                "hipMalloc(data backing)",
                rocm_detail::allocation_attempt(
                        &data_backing, data_bytes,
                        rocm_detail::AllocationClass::data_backing,
                        rocm_detail::AllocationPhase::setup));
        RocmBackingGuard data_guard{
                data_backing, rocm_detail::AllocationClass::data_backing};

        void* metadata_backing = nullptr;
        check_arena_alloc_hip(
                "hipMalloc(metadata backing)",
                rocm_detail::allocation_attempt(
                        &metadata_backing, metadata_bytes,
                        rocm_detail::AllocationClass::metadata_backing,
                        rocm_detail::AllocationPhase::setup));
        RocmBackingGuard metadata_guard{
                metadata_backing,
                rocm_detail::AllocationClass::metadata_backing};

        // hipMalloc reserves 256-byte-aligned memory; verify the documented
        // 32-byte base contract before any allocator is constructed.
        if (reinterpret_cast<std::uintptr_t>(data_backing) % 32 != 0
                || reinterpret_cast<std::uintptr_t>(metadata_backing) % 32
                        != 0) {
            throw std::runtime_error(
                    "ROCm arena backing is not 32-byte aligned");
        }

        auto data_allocator = std::make_unique<ListAllocator>(
                data_backing, data_bytes, 32);
        auto metadata_allocator = std::make_unique<FixedSizeAllocator>(
                metadata_backing, metadata_bytes, 32, 512);
        if (metadata_allocator->block_count()
                != 4 * queue_config.max_in_flight_per_queue) {
            throw std::runtime_error(
                    "ROCm metadata arena block count does not match the "
                    "configured capacity");
        }

        auto result = std::make_unique<RocmDevice>(
                device_ordinal, std::move(data_allocator),
                std::move(metadata_allocator), data_backing, data_bytes,
                metadata_backing);
        data_guard.dismiss();
        metadata_guard.dismiss();
        return result;
    }

}  // namespace iom
