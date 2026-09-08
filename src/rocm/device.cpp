#include "iom/rocm/device.hpp"

#include <hip/hip_runtime_api.h>

#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "copy.hpp"
#include "iom/detail/aligned_storage.hpp"
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

        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, int device_count) {
            return std::invalid_argument(
                    "ROCm device ordinal " + std::to_string(ordinal)
                    + " is unavailable; device count is "
                    + std::to_string(device_count));
        }

        void validate_native_storage(void* address, int ordinal) {
            hipPointerAttribute_t attributes{};
            check_hip(
                    "hipPointerGetAttributes",
                    hipPointerGetAttributes(&attributes, address));
            if (attributes.type != hipMemoryTypeDevice
                    || attributes.isManaged != 0
                    || attributes.device != ordinal
                    || attributes.devicePointer != address) {
                throw std::runtime_error(
                        "ROCm tensor storage is incompatible with the owning "
                        "device");
            }
        }

        class RocmDevice final : public Device {
        public:
            RocmDevice(std::uint32_t ordinal, Allocator& allocator)
                    : ordinal_(ordinal),
                      transfer_pool_{}, allocator_(allocator) {}

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

        private:
            std::uint32_t ordinal_;
            detail::RegistryState registry_state_;
            rocm_detail::TransferStreamPool transfer_pool_;
            rocm_detail::StagingSlotPool staging_pool_;
            Allocator& allocator_;
            friend class RocmTensor;
        };

        class RocmTensor final : public Tensor {
        public:
            RocmTensor(const TensorSpec& spec, RocmDevice& device,
                       Allocator& allocator)
                    : Tensor(spec, device), device_(device),
                      state_(&device.registry_state()), allocator_(allocator) {
                address_ = iom::detail::allocate_aligned_storage(
                        allocator_,
                        view().spec().tiled_storage_nbytes(),
                        [this] { device_.activate(); },
                        "ROCm tensor storage is not 32-byte aligned");
                try {
                    validate_native_storage(
                            address_, static_cast<int>(device_.ordinal()));
                } catch (...) {
                    void* rejected = std::exchange(address_, nullptr);
                    iom::detail::release_aligned_storage(
                            allocator_, rejected,
                            [this] { device_.activate(); });
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
                                        allocator_, address_, bytes,
                                        [device = &device_] {
                                            device->activate();
                                        });
                    } catch (...) {
                        // Keep failed storage unavailable if quarantine
                        // allocation itself fails.
                    }
                    address_ = nullptr;
                };
                const auto release_storage = [this]() noexcept {
                    iom::detail::release_aligned_storage(
                            allocator_, address_,
                            [this] { device_.activate(); });
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
            Allocator& allocator_;
            void* address_ = nullptr;
        };

        std::unique_ptr<Tensor> RocmDevice::create_tensor(
                const TensorSpec& spec) {
            // The base Tensor ctor validates the spec before any device
            // interaction; allocation-time activation is owned by the
            // RocmTensor ctor's pre_allocate callback.
            return std::make_unique<RocmTensor>(spec, *this, allocator_);
        }

    }  // namespace

    std::unique_ptr<Device> make_rocm_device(
            std::uint32_t device_ordinal, Allocator& allocator) {
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

        return std::make_unique<RocmDevice>(device_ordinal, allocator);
    }

}  // namespace iom
