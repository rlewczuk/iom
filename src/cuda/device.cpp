#include "iom/cuda/device.hpp"

#include <cuda.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "copy.hpp"
#include "driver.hpp"
namespace iom {

    cuda_detail::DriverCalls cuda_detail::driver_calls{};

    namespace {
        constexpr std::size_t kStorageAlignment = 32;

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

        class PrimaryCtxGuard final {
        public:
            explicit PrimaryCtxGuard(CUdevice device) : device_(device) {}

            PrimaryCtxGuard(const PrimaryCtxGuard&) = delete;
            PrimaryCtxGuard& operator=(const PrimaryCtxGuard&) = delete;

            ~PrimaryCtxGuard() noexcept {
                if (armed_) {
                    (void)cuda_detail::driver_calls.primary_ctx_release(device_);
                }
            }

            void dismiss() noexcept { armed_ = false; }

        private:
            CUdevice device_;
            bool armed_ = true;
        };

        class CudaDevice final : public Device {
        public:
            CudaDevice(std::uint32_t ordinal, CUdevice device, CUcontext context,
                       Allocator& allocator)
                    : ordinal_(ordinal), device_(device), context_(context),
                      allocator_(allocator) {}

            CudaDevice(const CudaDevice&) = delete;
            CudaDevice& operator=(const CudaDevice&) = delete;

            ~CudaDevice() override {
                if (context_ != nullptr) {
                    (void)cuda_detail::driver_calls.primary_ctx_release(device_);
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
                    const TensorSpec& spec) override;

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                activate();
                return cuda_detail::make_queue(*this, context_);
            }

            void activate() const {
                check_cuda(
                        "cuCtxSetCurrent",
                        cuda_detail::driver_calls.ctx_set_current(context_));
            }

            [[nodiscard]] CUcontext context() const noexcept {
                return context_;
            }

        private:
            std::uint32_t ordinal_;
            CUdevice device_;
            CUcontext context_;
            Allocator& allocator_;

            friend class CudaTensor;
        };

        class CudaTensor final : public Tensor {
        public:
            CudaTensor(const TensorSpec& spec, CudaDevice& device,
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
                    void* misaligned = address_;
                    address_ = nullptr;
                    allocator_.free(misaligned);
                    throw std::runtime_error(
                        "CUDA tensor storage is not 32-byte aligned");
                }
            }

            ~CudaTensor() override {
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
                cuda_detail::region_from_host(
                        device_.context(), destination, source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                device_.activate();
                cuda_detail::region_to_host(
                        device_.context(), source, destination);
            }

            CudaDevice& device_;
            Allocator& allocator_;
            void* address_ = nullptr;
        };

    }  // namespace

    std::unique_ptr<Tensor> CudaDevice::create_tensor(
            const TensorSpec& spec) {
        activate();
        return std::make_unique<CudaTensor>(spec, *this, allocator_);
    }

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
        check_cuda(
                "cuDevicePrimaryCtxRetain",
                cuda_detail::driver_calls.primary_ctx_retain(&context, device));
        PrimaryCtxGuard context_guard{device};
        check_cuda(
                "cuCtxSetCurrent",
                cuda_detail::driver_calls.ctx_set_current(context));
        auto result = std::make_unique<CudaDevice>(
                device_ordinal, device, context, allocator);
        context_guard.dismiss();
        return result;
    }

}  // namespace iom
