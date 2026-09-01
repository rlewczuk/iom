#include "iom/rocm/device.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "copy.hpp"

namespace iom {

    namespace {

        constexpr std::size_t kStorageAlignment = 32;

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

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::ROCM;
            }

            [[nodiscard]] std::uint32_t backend_device() const noexcept override {
                return ordinal_;
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec& spec) override;

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                activate();
                return rocm_detail::make_queue(
                        *this, static_cast<int>(ordinal_));
            }

            void activate() const {
                check_hip(
                        "hipSetDevice",
                        hipSetDevice(static_cast<int>(ordinal_)));
            }

        private:
            std::uint32_t ordinal_;
            Allocator& allocator_;

            friend class RocmTensor;
        };

        class RocmTensor final : public Tensor {
        public:
            RocmTensor(const TensorSpec& spec, RocmDevice& device,
                       Allocator& allocator)
                    : Tensor(spec, device), device_(device),
                      allocator_(allocator) {
                device_.activate();
                address_ = allocator_.alloc(view().spec().tiled_storage_nbytes());
                if (address_ == nullptr) {
                    throw std::bad_alloc();
                }
                if (reinterpret_cast<std::uintptr_t>(address_)
                                % kStorageAlignment
                        != 0) {
                    allocator_.free(address_);
                    address_ = nullptr;
                    throw std::runtime_error(
                            "ROCm tensor storage is not 32-byte aligned");
                }
            }

            ~RocmTensor() override {
                if (address_ == nullptr) {
                    return;
                }
                try {
                    device_.activate();
                } catch (...) {
                }
                try {
                    allocator_.free(address_);
                } catch (...) {
                }
                address_ = nullptr;
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
                        static_cast<int>(device_.ordinal_), source,
                        destination);
            }

            RocmDevice& device_;
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
