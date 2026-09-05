#include "iom/cuda/device.hpp"

#include <cuda.h>

#include <array>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "copy.hpp"
#include "driver.hpp"
#include "iom/detail/aligned_storage.hpp"
#include "registry_state.hpp"
namespace iom {


    namespace {
        constexpr std::array kCudaSupportedDataTypes = {
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
                    try {
                        activate();
                        transfer_pool_.destroy();
                    } catch (...) {
                    }
                    registry_state_.quarantine.drain();
                    (void)cuda_detail::driver_calls.primary_ctx_release(device_);
                    context_ = nullptr;
                } else {
                    registry_state_.quarantine.drain();
                }
            }

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::CUDA;
            }

            [[nodiscard]] std::uint32_t backend_device() const noexcept override {
                return ordinal_;
            }
            [[nodiscard]] std::span<const iom::DataType>
                    supported_data_types() const noexcept override {
                return {kCudaSupportedDataTypes.data(),
                        kCudaSupportedDataTypes.size()};
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec& spec) override;

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                activate();
                return cuda_detail::make_queue(
                        *this, context_, registry_state_);
            }

            void activate() const {
                check_cuda(
                        "cuCtxSetCurrent",
                        cuda_detail::driver_calls.ctx_set_current(context_));
            }

            [[nodiscard]] CUcontext context() const noexcept {
                return context_;
            }

            [[nodiscard]] cuda_detail::CudaRegistryState&
                    registry_state() noexcept {
                return registry_state_;
            }

        private:
            std::uint32_t ordinal_;
            CUdevice device_;
            CUcontext context_;
            cuda_detail::CudaRegistryState registry_state_;
            cuda_detail::TransferStreamPool transfer_pool_;
            Allocator& allocator_;
            friend class CudaTensor;
        };

        class CudaTensor final : public Tensor {
        public:
            CudaTensor(const TensorSpec& spec, CudaDevice& device,
                       Allocator& allocator)
                    : Tensor(spec, device), device_(device),
                      state_(&device.registry_state()), allocator_(allocator) {
                address_ = iom::detail::allocate_aligned_storage(
                        allocator_,
                        view().spec().tiled_storage_nbytes(),
                        [this] { device_.activate(); },
                        "CUDA tensor storage is not 32-byte aligned");
            }

            ~CudaTensor() noexcept override {
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
                cuda_detail::region_from_host(
                        device_.transfer_pool_, device_.context(),
                        destination, source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                device_.activate();
                cuda_detail::region_to_host(
                        device_.transfer_pool_, device_.context(),
                        source, destination);
            }
            CudaDevice& device_;
            cuda_detail::CudaRegistryState* state_;
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
        check_cuda("cuInit", cuda_detail::driver_calls.init(0));

        int device_count = 0;
        CUresult count_status =
                cuda_detail::driver_calls.device_get_count(&device_count);
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
                cuda_detail::driver_calls.device_get(
                        &device, static_cast<int>(device_ordinal)));
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
