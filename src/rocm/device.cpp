#include "iom/rocm/device.hpp"

#include <hip/hip_runtime_api.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

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

        class RocmDevice final : public Device {
        public:
            RocmDevice(std::uint32_t ordinal, hipCtx_t context,
                       Allocator& allocator)
                    : ordinal_(ordinal), context_(context),
                      allocator_(allocator) {}

            RocmDevice(const RocmDevice&) = delete;
            RocmDevice& operator=(const RocmDevice&) = delete;

            ~RocmDevice() override {
                if (context_ != nullptr) {
                    // Destruction is deliberately unconditional and occurs
                    // once. HIP teardown errors cannot be reported by a
                    // noexcept destructor without terminating the caller.
                    (void)hipCtxDestroy(context_);
                    context_ = nullptr;
                }
            }

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::ROCM;
            }

            [[nodiscard]] std::uint32_t backend_device() const noexcept override {
                return ordinal_;
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec&) override {
                throw std::runtime_error(
                        "ROCm tensor storage is not implemented yet");
            }

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                throw std::runtime_error(
                        "ROCm operation queues are not implemented yet");
            }

        private:
            std::uint32_t ordinal_;
            hipCtx_t context_;
            Allocator& allocator_;
        };

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

        hipCtx_t context = nullptr;
        check_hip(
                "hipCtxCreate",
                hipCtxCreate(
                        &context, 0,
                        static_cast<hipDevice_t>(device_ordinal)));

        try {
            return std::make_unique<RocmDevice>(
                    device_ordinal, context, allocator);
        } catch (...) {
            (void)hipCtxDestroy(context);
            throw;
        }
    }

}  // namespace iom
