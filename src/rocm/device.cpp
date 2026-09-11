#include "iom/rocm/device.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "copy.hpp"
#include "driver.hpp"
#include "iom/detail/gpu_arena_config.hpp"
#include "../shared/standard_tiled_copy.hpp"

namespace iom {

    namespace {

        [[nodiscard]] std::runtime_error hip_error(
                const char* operation, hipError_t status) {
            return std::runtime_error(
                    std::string(operation) + " failed with "
                    + hipGetErrorName(status) + ": "
                    + hipGetErrorString(status));
        }

        void check_hip(const char* operation, hipError_t status) {
            if (status != hipSuccess) {
                throw hip_error(operation, status);
            }
        }

        // A native device-memory failure maps to std::bad_alloc; any other
        // runtime status keeps the established runtime category.
        void check_arena_alloc_hip(const char* operation, hipError_t status) {
            if (status == hipErrorOutOfMemory) {
                throw std::bad_alloc();
            }
            check_hip(operation, status);
        }

        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, int device_count) {
            return std::invalid_argument(
                    "ROCm device ordinal " + std::to_string(ordinal)
                    + " is unavailable; device count is "
                    + std::to_string(device_count));
        }

        // Validates that an arena suballocation is unmanaged device memory
        // on this exact ordinal and lies inside the owning arena backing.
        // Interior addresses are accepted: the identity checks pin the exact
        // ordinal, and membership in the device's own arena range binds the
        // address to this device's reservation without requiring it to be
        // the base of an independent native allocation (hipPointerGetAttributes
        // reports the queried address, not the containing allocation's base,
        // so base equality would reject every interior suballocation).
        void validate_native_storage(
                void* address, int ordinal, void* arena_base,
                std::size_t arena_bytes) {
            hipPointerAttribute_t attributes{};
            check_hip(
                    "hipPointerGetAttributes",
                    hipPointerGetAttributes(&attributes, address));
            const std::uintptr_t raw =
                    reinterpret_cast<std::uintptr_t>(address);
            const std::uintptr_t base =
                    reinterpret_cast<std::uintptr_t>(arena_base);
            if (attributes.type != hipMemoryTypeDevice
                    || attributes.isManaged != 0
                    || attributes.device != ordinal || raw < base
                    || raw - base >= arena_bytes) {
                throw std::runtime_error(
                        "ROCm tensor storage is incompatible with the owning "
                        "device");
            }
        }

        class RocmDevice final : public Device {
        public:
            RocmDevice(
                    std::uint32_t ordinal,
                    std::unique_ptr<ListAllocator> data_allocator,
                    std::unique_ptr<FixedSizeAllocator> metadata_allocator,
                    void* data_backing, std::size_t data_backing_bytes,
                    void* metadata_backing)
                    : ordinal_(ordinal),
                      transfer_pool_{},
                      data_allocator_(std::move(data_allocator)),
                      metadata_allocator_(std::move(metadata_allocator)),
                      data_backing_(data_backing),
                      data_backing_bytes_(data_backing_bytes),
                      metadata_backing_(metadata_backing) {}

            RocmDevice(const RocmDevice&) = delete;
            RocmDevice& operator=(const RocmDevice&) = delete;
            ~RocmDevice() override {
                try {
                    activate();
                    transfer_pool_.destroy();
                    staging_pool_.destroy();
                } catch (...) {
                }
                registry_state_.quarantine.drain();
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

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::ROCM;
            }

            [[nodiscard]] std::uint32_t backend_device() const noexcept override {
                return ordinal_;
            }
            [[nodiscard]] std::span<const iom::DataType>
                    supported_data_types() const noexcept override {
                return detail::standard_supported_data_types();
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec& spec) override;
            [[nodiscard]] std::unique_ptr<RawWorkspace> create_workspace(
                    std::size_t bytes) override;
            [[nodiscard]] std::uint32_t ordinal() const noexcept {
                return ordinal_;
            }

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                activate();
                return rocm_detail::make_queue(
                        *this, static_cast<int>(ordinal_), registry_state_);
            }

            void activate() const {
                check_hip(
                        "hipSetDevice",
                        hipSetDevice(static_cast<int>(ordinal_)));
            }

            [[nodiscard]] detail::RegistryState&
                    registry_state() noexcept {
                return registry_state_;
            }

            // Tensor and (later) raw-workspace data ranges suballocate from
            // the device data arena. Both operations serialize the
            // allocator's host bookkeeping at this device boundary and never
            // hold the lock across native work, waits, callbacks, or drains;
            // a fragmentation/contiguous-range failure is std::bad_alloc and
            // live addresses never move.
            [[nodiscard]] void* allocate_data(std::size_t bytes) {
                std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
                return data_allocator_->alloc(bytes);
            }

            void release_data(void* address) {
                std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
                data_allocator_->free(address);
            }

            // Returns one raw-workspace arena range at owner destruction.
            // A retained (live or quarantined) lease keeps the range out
            // of the allocator: it is deferred into the device quarantine
            // and freed only on the final drain, never reused. The lease
            // check releases the registry allocation mutex before any
            // allocator work.
            void release_workspace(
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

            [[nodiscard]] iom::Allocator& data_allocator() noexcept {
                return *data_allocator_;
            }

            [[nodiscard]] void* data_backing() const noexcept {
                return data_backing_;
            }

            [[nodiscard]] std::size_t data_backing_bytes() const noexcept {
                return data_backing_bytes_;
            }

        private:
            std::uint32_t ordinal_;
            detail::RegistryState registry_state_;
            rocm_detail::TransferStreamPool transfer_pool_;
            rocm_detail::StagingSlotPool staging_pool_;
            std::mutex bookkeeping_mutex_;
            std::unique_ptr<ListAllocator> data_allocator_;
            std::unique_ptr<FixedSizeAllocator> metadata_allocator_;
            void* data_backing_ = nullptr;
            std::size_t data_backing_bytes_ = 0;
            void* metadata_backing_ = nullptr;
            friend class RocmTensor;
        };

        /**
         * Owner of one raw-workspace arena suballocation. The base
         * RawWorkspace registers the exact identity with the device
         * before the body can fail; destruction hands the range back
         * through the device boundary, which keeps a retained lease out
         * of the allocator.
         */
        class RocmWorkspace final : public RawWorkspace {
        public:
            RocmWorkspace(
                    RocmDevice& device, void* address, std::size_t bytes)
                    : RawWorkspace(device, bytes),
                      device_(device),
                      address_(address) {}

            ~RocmWorkspace() override {
                if (address_ != nullptr) {
                    device_.release_workspace(this, address_, byte_size());
                }
            }

            RocmWorkspace(const RocmWorkspace&) = delete;
            RocmWorkspace& operator=(const RocmWorkspace&) = delete;

        private:
            [[nodiscard]] void* workspace_address() const noexcept override {
                return address_;
            }

            RocmDevice& device_;
            void* address_;
        };

        class RocmTensor final : public Tensor {
        public:
            RocmTensor(const TensorSpec& spec, RocmDevice& device)
                    : Tensor(spec, device), device_(device),
                      state_(&device.registry_state()) {
                address_ = device_.allocate_data(
                        view().spec().tiled_storage_nbytes());
                try {
                    device_.activate();
                    validate_native_storage(
                            address_, static_cast<int>(device_.ordinal()),
                            device_.data_backing(),
                            device_.data_backing_bytes());
                    check_hip(
                            "hipMemset",
                            hipMemset(
                                    address_, 0,
                                    view().spec().tiled_storage_nbytes()));
                    check_hip("hipDeviceSynchronize", hipDeviceSynchronize());
                } catch (...) {
                    void* rejected = std::exchange(address_, nullptr);
                    device_.release_data(rejected);
                    throw;
                }
            }

            ~RocmTensor() noexcept override {
                if (address_ == nullptr) {
                    return;
                }

                const std::size_t bytes =
                        view().spec().tiled_storage_nbytes();
                const auto quarantine_storage = [this, bytes]() noexcept {
                    try {
                        state_->quarantine
                                .emplace<detail::AllocatorCleanupAction>(
                                        device_.data_allocator(), address_,
                                        bytes);
                    } catch (...) {
                        // Keep failed storage unavailable if quarantine
                        // allocation itself fails.
                    }
                    address_ = nullptr;
                };
                const auto release_storage = [this]() noexcept {
                    try {
                        device_.release_data(address_);
                    } catch (...) {
                    }
                    address_ = nullptr;
                };
                iom::detail::release_or_quarantine(
                        state_->registry, address_, quarantine_storage,
                        release_storage);
            }

        private:
            [[nodiscard]] void* storage_handle() noexcept override {
                return address_;
            }

            void region_from_host(
                    const TensorView& destination,
                    std::span<const std::byte> source) override {
                device_.activate();
                rocm_detail::region_from_host(
                        device_.transfer_pool_, device_.staging_pool_,
                        static_cast<int>(device_.ordinal_), destination, source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                device_.activate();
                rocm_detail::region_to_host(
                        device_.transfer_pool_, device_.staging_pool_,
                        static_cast<int>(device_.ordinal_), source, destination);
            }

            RocmDevice& device_;
            detail::RegistryState* state_;
            void* address_ = nullptr;
        };

        std::unique_ptr<Tensor> RocmDevice::create_tensor(
                const TensorSpec& spec) {
            // The base Tensor ctor validates the spec before any device
            // interaction; the RocmTensor ctor then suballocates the data
            // arena.
            return std::make_unique<RocmTensor>(spec, *this);
        }

        std::unique_ptr<RawWorkspace> RocmDevice::create_workspace(
                std::size_t bytes) {
            if (bytes == 0) {
                // Valid empty owner: no arena suballocation, no native
                // call.
                return std::make_unique<RocmWorkspace>(
                        *this, nullptr, 0);
            }
            // Positive creation suballocates the already reserved data
            // arena under the allocator bookkeeping boundary; it never
            // reserves a new native backing. Exhaustion, including
            // fragmentation with no fitting contiguous range, is
            // std::bad_alloc with no fallback.
            void* address = allocate_data(bytes);
            try {
                activate();
                validate_native_storage(
                        address, ordinal_, data_backing_,
                        data_backing_bytes_);
                return std::make_unique<RocmWorkspace>(
                        *this, address, bytes);
            } catch (...) {
                release_data(address);
                throw;
            }
        }

    }  // namespace

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
        check_hip("hipGetDeviceCount", hipGetDeviceCount(&device_count));
        if (device_count <= 0
                || device_ordinal
                        >= static_cast<std::uint32_t>(device_count)) {
            throw invalid_ordinal(device_ordinal, device_count);
        }
        if (device_ordinal
                > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw invalid_ordinal(device_ordinal, device_count);
        }

        check_hip(
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
