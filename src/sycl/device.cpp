#include "iom/sycl/device.hpp"
#include "device_internal.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "copy.hpp"
#include "runtime.hpp"
#include "iom/detail/gpu_arena_config.hpp"
#include "../shared/standard_tiled_copy.hpp"


namespace iom::sycl_detail {

    ContextCalls context_calls{};
    LaunchCalls launch_calls{};

#ifdef IOM_ENABLE_TESTING
    AllocationObserver allocation_observer{};
    AllocationCalls allocation_calls{};
#endif  // IOM_ENABLE_TESTING

}  // namespace iom::sycl_detail

namespace iom {

    namespace {
        [[nodiscard]] std::vector<sycl::device> eligible_devices() {
            std::vector<sycl::device> devices = sycl::device::get_devices();
            devices.erase(
                    std::remove_if(
                            devices.begin(), devices.end(),
                            [](const sycl::device& device) {
                                return !device.is_gpu() && !device.is_accelerator();
                            }),
                    devices.end());
            return devices;
        }

        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, std::size_t device_count) {
            return std::invalid_argument(
                    "SYCL device ordinal " + std::to_string(ordinal)
                    + " is unavailable; eligible accelerator count is "
                    + std::to_string(device_count));
        }
    }  // namespace

    SyclDevice::SyclDevice(
            std::uint32_t ordinal, sycl::device device,
            sycl::context context,
            std::unique_ptr<ListAllocator> data_allocator,
            std::unique_ptr<FixedSizeAllocator> metadata_allocator,
            void* data_backing, std::size_t data_backing_bytes,
            void* metadata_backing)
            : device_(std::move(device)),
              context_(std::move(context)),
              transfer_queue_(
                      std::in_place, *context_, device_,
                      sycl::property_list{
                              sycl::property::queue::in_order{}}),
              registry_state_(),
              ordinal_(ordinal),
              data_allocator_(std::move(data_allocator)),
              metadata_allocator_(std::move(metadata_allocator)),
              data_backing_(data_backing),
              data_backing_bytes_(data_backing_bytes),
              metadata_backing_(metadata_backing),
              queue_slot_count_(metadata_allocator_->block_count()
                                / detail::kMaxLiveGpuQueues) {
        if (sycl_detail::context_calls.context_ready != nullptr) {
            sycl_detail::context_calls.context_ready(*context_);
        }
    }

    SyclDevice::~SyclDevice() {
        try {
            transfer_queue_->wait_and_throw();
        } catch (...) {
        }
        registry_state_.quarantine.drain();
        // Covering-proof every retained unknown-completion lease;
        // unproven bundles are discarded best-effort while the
        // context is still valid, before allocator bookkeeping
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
            sycl_detail::free_attempt_device(
                    data_backing_, *context_,
                    sycl_detail::AllocationClass::data_backing,
                    sycl_detail::AllocationPhase::post_publication);
            data_backing_ = nullptr;
        }
        if (metadata_backing_ != nullptr) {
            sycl_detail::free_attempt_device(
                    metadata_backing_, *context_,
                    sycl_detail::AllocationClass::metadata_backing,
                    sycl_detail::AllocationPhase::post_publication);
            metadata_backing_ = nullptr;
        }
        transfer_queue_.reset();
        context_.reset();
        if (sycl_detail::context_calls.context_destroyed != nullptr) {
            sycl_detail::context_calls.context_destroyed();
        }
    }

    BackendKind SyclDevice::backend_kind() const noexcept {
        return BackendKind::SYCL;
    }

    std::uint32_t SyclDevice::backend_device() const noexcept {
        return ordinal_;
    }

    std::span<const iom::DataType>
            SyclDevice::supported_data_types() const noexcept {
        return detail::standard_supported_data_types();
    }

    std::unique_ptr<DeviceOps> SyclDevice::create_ops() {
        return sycl_detail::make_queue(
                *this, *this, *context_, device_, registry_state_);
    }

    std::size_t SyclDevice::queue_slot_count() const noexcept {
        return queue_slot_count_;
    }

    detail::QueueResourceLease SyclDevice::reserve_queue_resources() {
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
        return make_lease(
                *this, base, slot_count,
                metadata_allocator_->ptr_from_index(base),
                std::move(host_mirrors));
    }

    void SyclDevice::release_queue_resources(
            std::size_t first_slot, std::size_t slot_count) noexcept {
        std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
        release_queue_resources_locked(first_slot, slot_count);
    }

    void SyclDevice::retain_unknown_lease(
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

    void SyclDevice::reclaim_retained_leases() {
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
        // Callbacks wait on retained queues: never under any
        // device lock.
        for (std::size_t index = 0; index < pending_count; ++index) {
            if (!pending[index].reclaim()) {
                std::lock_guard<std::mutex> lock(retained_mutex_);
                retained_leases_[retained_count_++] =
                        std::move(pending[index]);
            }
        }
    }

    const sycl::context& SyclDevice::context() const noexcept {
        return *context_;
    }

    const sycl::device& SyclDevice::native_device() const noexcept {
        return device_;
    }

    sycl::queue& SyclDevice::transfer_queue() noexcept {
        return *transfer_queue_;
    }

    detail::RegistryState& SyclDevice::registry_state() noexcept {
        return registry_state_;
    }

    void* SyclDevice::allocate_data(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
        return data_allocator_->alloc(bytes);
    }

    void SyclDevice::release_data(void* address) {
        std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
        data_allocator_->free(address);
    }

    void SyclDevice::release_workspace(
            const RawWorkspace* owner, void* address,
            std::size_t bytes) noexcept {
        bool retained = false;
        {
            std::lock_guard<std::mutex> lock(
                    registry_state_.allocation_mutex);
            retained = detail::workspace_range_retained(
                    registry_state_.workspace_leases, owner,
                    address, bytes);
        }
        if (retained) {
            try {
                registry_state_.quarantine
                        .emplace<detail::AllocatorCleanupAction>(
                                *data_allocator_, address, bytes);
            } catch (...) {
                // Keep failed storage unavailable if quarantine
                // allocation itself fails.
            }
            return;
        }
        release_data(address);
    }

    iom::Allocator& SyclDevice::data_allocator() noexcept {
        return *data_allocator_;
    }

    void* SyclDevice::data_backing() const noexcept {
        return data_backing_;
    }

    std::size_t SyclDevice::data_backing_bytes() const noexcept {
        return data_backing_bytes_;
    }

    void SyclDevice::release_queue_resources_locked(
            std::size_t first_slot, std::size_t slot_count) noexcept {
        for (std::size_t index = 0; index < slot_count; ++index) {
            metadata_allocator_->free(
                    metadata_allocator_->ptr_from_index(
                            first_slot + index));
        }
    }

    void SyclDevice::discard_all_retained_leases() noexcept {
        std::lock_guard<std::mutex> lock(retained_mutex_);
        for (std::size_t index = 0; index < retained_count_; ++index) {
            retained_leases_[index].discard();
            retained_leases_[index] = {};
        }
        retained_count_ = 0;
    }


    namespace sycl_detail {

        std::size_t eligible_device_count() {
            return eligible_devices().size();
        }

    }  // namespace sycl_detail

    namespace {
        // Factory-local RAII owner for one device-USM arena backing. Setup
        // rollback frees the backing through the same instrumented boundary
        // used for its reservation, in reverse acquisition order, while the
        // context is still valid; the original failure is preserved.
        class SyclBackingGuard final {
        public:
            SyclBackingGuard() = default;
            SyclBackingGuard(
                    void* address, const sycl::context& context,
                    sycl_detail::AllocationClass classification)
                    : address_(address), context_(&context),
                      classification_(classification) {}

            SyclBackingGuard(const SyclBackingGuard&) = delete;
            SyclBackingGuard& operator=(const SyclBackingGuard&) = delete;

            ~SyclBackingGuard() noexcept {
                if (address_ != nullptr && context_ != nullptr) {
                    sycl_detail::free_attempt_device(
                            address_, *context_, classification_,
                            sycl_detail::AllocationPhase::setup);
                }
            }

            void dismiss() noexcept { address_ = nullptr; }

        private:
            void* address_ = nullptr;
            const sycl::context* context_ = nullptr;
            sycl_detail::AllocationClass classification_ =
                    sycl_detail::AllocationClass::other_iom_setup;
        };
    }  // namespace

    std::unique_ptr<Device> make_sycl_device(
            std::uint32_t device_ordinal, DeviceMemoryConfig memory_config,
            QueueConfig queue_config) {
        const std::size_t data_bytes = detail::valid_tensor_arena_bytes(
                memory_config.tensor_arena_bytes);
        const std::size_t metadata_bytes =
                detail::standard_gpu_metadata_bytes(
                        queue_config.max_in_flight_per_queue);

        std::vector<sycl::device> devices = eligible_devices();
        if (device_ordinal >= devices.size()) {
            throw invalid_ordinal(device_ordinal, devices.size());
        }
        sycl::device device = std::move(devices[device_ordinal]);
        sycl::context context(device);
        if (sycl_detail::context_calls.context_created != nullptr) {
            sycl_detail::context_calls.context_created();
        }

        // Reserve exactly two device-USM backings in this exact context and
        // device: one data arena and one metadata arena of checked capacity.
        void* data_backing = sycl_detail::alloc_attempt_device(
                data_bytes, device, context,
                sycl_detail::AllocationClass::data_backing,
                sycl_detail::AllocationPhase::setup);
        if (data_backing == nullptr) {
            throw std::bad_alloc();
        }
        SyclBackingGuard data_guard{
                data_backing, context,
                sycl_detail::AllocationClass::data_backing};

        void* metadata_backing = sycl_detail::alloc_attempt_device(
                metadata_bytes, device, context,
                sycl_detail::AllocationClass::metadata_backing,
                sycl_detail::AllocationPhase::setup);
        if (metadata_backing == nullptr) {
            throw std::bad_alloc();
        }
        SyclBackingGuard metadata_guard{
                metadata_backing, context,
                sycl_detail::AllocationClass::metadata_backing};

        // The SYCL USM store alignment is implementation-defined; verify the
        // documented 32-byte base contract before any allocator is
        // constructed and roll back cleanly when a platform cannot satisfy
        // it.
        if (reinterpret_cast<std::uintptr_t>(data_backing) % 32 != 0
                || reinterpret_cast<std::uintptr_t>(metadata_backing) % 32
                        != 0) {
            throw std::runtime_error(
                    "SYCL arena backing is not 32-byte aligned");
        }

        auto data_allocator = std::make_unique<ListAllocator>(
                data_backing, data_bytes, 32);
        auto metadata_allocator = std::make_unique<FixedSizeAllocator>(
                metadata_backing, metadata_bytes, 32, 512);
        if (metadata_allocator->block_count()
                != 4 * queue_config.max_in_flight_per_queue) {
            throw std::runtime_error(
                    "SYCL metadata arena block count does not match the "
                    "configured capacity");
        }

        auto result = std::make_unique<SyclDevice>(
                device_ordinal, std::move(device), context,
                std::move(data_allocator), std::move(metadata_allocator),
                data_backing, data_bytes, metadata_backing);
        data_guard.dismiss();
        metadata_guard.dismiss();
        return result;
    }

}  // namespace iom
