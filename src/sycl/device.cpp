#include "iom/sycl/device.hpp"

#include <sycl/sycl.hpp>

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

        class SyclDevice final : public Device {
        public:
            SyclDevice(
                    std::uint32_t ordinal, sycl::device device,
                    sycl::context context,
                    std::unique_ptr<ListAllocator> data_allocator,
                    std::unique_ptr<FixedSizeAllocator> metadata_allocator,
                    void* data_backing, std::size_t data_backing_bytes,
                    void* metadata_backing)
                    : device_(std::move(device)),
                      context_(std::move(context)),
                      staging_pool_(*context_, device_),
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
                      metadata_backing_(metadata_backing) {
                if (sycl_detail::context_calls.context_ready != nullptr) {
                    sycl_detail::context_calls.context_ready(*context_);
                }
            }

            SyclDevice(const SyclDevice&) = delete;
            SyclDevice& operator=(const SyclDevice&) = delete;
            ~SyclDevice() override {
                try {
                    transfer_queue_->wait_and_throw();
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
                staging_pool_.destroy();
                transfer_queue_.reset();
                context_.reset();
                if (sycl_detail::context_calls.context_destroyed != nullptr) {
                    sycl_detail::context_calls.context_destroyed();
                }
            }

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::SYCL;
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
            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                return sycl_detail::make_queue(
                        *this, *context_, device_, registry_state_);
            }

            [[nodiscard]] const sycl::context& context() const noexcept {
                return *context_;
            }

            [[nodiscard]] const sycl::device& native_device() const noexcept {
                return device_;
            }
            [[nodiscard]] sycl_detail::StagingSlotPool&
                    staging_pool() noexcept {
                return staging_pool_;
            }

            [[nodiscard]] sycl::queue& transfer_queue() noexcept {
                return *transfer_queue_;
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
            sycl::device device_;
            std::optional<sycl::context> context_;
            sycl_detail::StagingSlotPool staging_pool_;
            std::optional<sycl::queue> transfer_queue_;
            detail::RegistryState registry_state_;
            std::uint32_t ordinal_;
            std::mutex bookkeeping_mutex_;
            std::unique_ptr<ListAllocator> data_allocator_;
            std::unique_ptr<FixedSizeAllocator> metadata_allocator_;
            void* data_backing_ = nullptr;
            std::size_t data_backing_bytes_ = 0;
            void* metadata_backing_ = nullptr;

            friend class SyclTensor;
        };
        class SyclTensor final : public Tensor {
        public:
            SyclTensor(
                    const TensorSpec& spec, SyclDevice& device,
                    detail::RegistryState& state)
                    : Tensor(spec, device),
                      device_(device),
                      state_(&state) {
                address_ = device_.allocate_data(
                        view().spec().tiled_storage_nbytes());

                try {
                    if (sycl::get_pointer_type(address_, device_.context())
                            == sycl::usm::alloc::unknown
                            || sycl::get_pointer_device(
                                       address_, device_.context())
                                    != device_.native_device()) {
                        throw std::runtime_error(
                                "SYCL tensor storage is incompatible with "
                                "the owning context");
                    }
                    auto& queue = device_.transfer_queue();
                    queue.memset(
                            address_, 0,
                            view().spec().tiled_storage_nbytes());
                    queue.wait_and_throw();
                } catch (...) {
                    void* rejected = std::exchange(address_, nullptr);
                    device_.release_data(rejected);
                    throw;
                }
            }

            ~SyclTensor() noexcept override {
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
                        // Leaking is safer than returning failed storage to
                        // the allocator when quarantine allocation fails.
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
                sycl_detail::region_from_host(
                        device_.staging_pool(), device_.transfer_queue(),
                        destination, address_, source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                sycl_detail::region_to_host(
                        device_.staging_pool(), device_.transfer_queue(),
                        source, address_, destination);
            }

            SyclDevice& device_;
            detail::RegistryState* state_;
            void* address_ = nullptr;
        };

        std::unique_ptr<Tensor> SyclDevice::create_tensor(
                const TensorSpec& spec) {
            return std::make_unique<SyclTensor>(
                    spec, *this, registry_state_);
        }

    }  // namespace

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
