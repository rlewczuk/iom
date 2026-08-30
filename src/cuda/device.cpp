#include "iom/cuda/device.hpp"

#include <cuda.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace iom {

    namespace {

        [[nodiscard]] std::runtime_error cuda_error(
                const char* operation, CUresult status) {
            const char* name = nullptr;
            const char* description = nullptr;
            (void)cuGetErrorName(status, &name);
            (void)cuGetErrorString(status, &description);
            return std::runtime_error(
                    std::string(operation) + " failed with "
                    + (name != nullptr ? name : "unknown CUDA error") + ": "
                    + (description != nullptr ? description : "unknown error"));
        }

        void check_cuda(const char* operation, CUresult status) {
            if (status != CUDA_SUCCESS) {
                throw cuda_error(operation, status);
            }
        }

        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, int device_count) {
            return std::invalid_argument(
                    "CUDA device ordinal " + std::to_string(ordinal)
                    + " is unavailable; device count is "
                    + std::to_string(device_count));
        }

        class CudaDevice final : public Device {
        public:
            CudaDevice(std::uint32_t ordinal, CUcontext context,
                       Allocator& allocator)
                    : ordinal_(ordinal), context_(context),
                      allocator_(allocator) {}

            CudaDevice(const CudaDevice&) = delete;
            CudaDevice& operator=(const CudaDevice&) = delete;

            ~CudaDevice() override {
                if (context_ != nullptr) {
                    (void)cuCtxDestroy(context_);
                    context_ = nullptr;
                }
            }

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::CUDA;
            }

            [[nodiscard]] std::uint32_t backend_device() const noexcept override {
                return ordinal_;
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec&) override {
                throw std::runtime_error(
                        "CUDA tensor storage is not implemented in this scaffold");
            }

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                throw std::runtime_error(
                        "CUDA operations are not implemented in this scaffold");
            }

        private:
            std::uint32_t ordinal_;
            CUcontext context_;
            Allocator& allocator_;
        };

    }  // namespace

    std::unique_ptr<Device> make_cuda_device(
            std::uint32_t device_ordinal, Allocator& allocator) {
        check_cuda("cuInit", cuInit(0));

        int device_count = 0;
        CUresult count_status = cuDeviceGetCount(&device_count);
        if (count_status == CUDA_ERROR_NO_DEVICE) {
            throw invalid_ordinal(device_ordinal, 0);
        }
        check_cuda("cuDeviceGetCount", count_status);

        if (device_count <= 0
                || device_ordinal
                        >= static_cast<std::uint32_t>(device_count)
                || device_ordinal
                        > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw invalid_ordinal(device_ordinal, device_count);
        }

        CUdevice device = 0;
        check_cuda(
                "cuDeviceGet",
                cuDeviceGet(&device, static_cast<int>(device_ordinal)));

        CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
        check_cuda(
                "cuCtxCreate",
                cuCtxCreate(&context, nullptr, 0, device));
#else
        check_cuda("cuCtxCreate", cuCtxCreate(&context, 0, device));
#endif

        try {
            return std::make_unique<CudaDevice>(
                    device_ordinal, context, allocator);
        } catch (...) {
            (void)cuCtxDestroy(context);
            throw;
        }
    }

}  // namespace iom
