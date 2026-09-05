#include "iom/rocm/device.hpp"

#include <hip/hip_runtime_api.h>

#include <array>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "copy.hpp"
#include "iom/detail/aligned_storage.hpp"
#include "registry_state.hpp"

namespace iom {

    namespace {

        constexpr std::array kRocmSupportedDataTypes = {
                iom::DataType::BOOL,
                iom::DataType::I2, iom::DataType::U2,
                iom::DataType::I4, iom::DataType::U4,
                iom::DataType::I8, iom::DataType::U8,
                iom::DataType::I16, iom::DataType::U16,
                iom::DataType::I32, iom::DataType::U32,
                iom::DataType::I64, iom::DataType::U64,
                iom::DataType::F4_E2M1,
                iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
                iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
                iom::DataType::F8_E8M0,
                iom::DataType::F16, iom::DataType::BF16,
                iom::DataType::F32, iom::DataType::F64,
        };

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

        class RocmDevice final : public Device {
        public:
            RocmDevice(std::uint32_t ordinal, Allocator& allocator)
                    : ordinal_(ordinal), allocator_(allocator) {}

            RocmDevice(const RocmDevice&) = delete;
            RocmDevice& operator=(const RocmDevice&) = delete;

            ~RocmDevice() override {
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
                return {kRocmSupportedDataTypes.data(),
                        kRocmSupportedDataTypes.size()};
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec& spec) override;

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

            [[nodiscard]] rocm_detail::RocmRegistryState&
                    registry_state() noexcept {
                return registry_state_;
            }

        private:
            std::uint32_t ordinal_;
            rocm_detail::RocmRegistryState registry_state_;
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
            }

            ~RocmTensor() noexcept override {
                if (address_ == nullptr) {
                    return;
                }
                std::vector<
                        detail::OutstandingWorkRegistry::EntrySnapshot>
                        snapshots;
                try {
                    snapshots = state_->registry.snapshot_for(address_);
                } catch (...) {
                    return;
                }

                bool safe_to_release = true;
                for (const auto& snapshot : snapshots) {
                    if (snapshot.state == detail::EntryState::Invalidated
                            || !snapshot.fence) {
                        safe_to_release = false;
                        continue;
                    }
                    try {
                        const detail::FenceResult result = snapshot.fence();
                        safe_to_release = safe_to_release
                                && result.succeeded && !result.failure;
                    } catch (...) {
                        safe_to_release = false;
                    }
                }

                if (safe_to_release) {
                    for (const auto& snapshot : snapshots) {
                        state_->registry.remove_entry_if_present(
                                snapshot.id, address_);
                    }
                    iom::detail::release_aligned_storage(
                            allocator_, address_,
                            [this] { device_.activate(); });
                    return;
                }

                try {
                    state_->quarantine.emplace<detail::AllocatorCleanupAction>(
                            allocator_, address_,
                            view().spec().tiled_storage_nbytes(),
                            [device = &device_] { device->activate(); });
                    address_ = nullptr;
                } catch (...) {
                }
                for (const auto& snapshot : snapshots) {
                    state_->registry.remove_entry_if_present(
                            snapshot.id,
                            address_ != nullptr ? address_ : snapshot.address);
                }
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
                        static_cast<int>(device_.ordinal_), destination,
                        source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                device_.activate();
                rocm_detail::region_to_host(
                        static_cast<int>(device_.ordinal_), source, destination);
            }

            RocmDevice& device_;
            rocm_detail::RocmRegistryState* state_;
            Allocator& allocator_;
            void* address_ = nullptr;
        };

        std::unique_ptr<Tensor> RocmDevice::create_tensor(
                const TensorSpec& spec) {
            activate();
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
