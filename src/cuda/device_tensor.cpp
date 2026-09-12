#include "device_internal.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "copy.hpp"
#include "driver.hpp"

namespace iom {
namespace {

// Validate the four driver attributes before checking the exact owner arena.
void validate_native_storage(
        void* address, CUcontext context, std::uint32_t ordinal,
        CUdeviceptr arena_base, std::size_t arena_bytes) {
    CUmemorytype memory_type{};
    int is_managed = 0;
    CUcontext pointer_context = nullptr;
    int pointer_ordinal = -1;
    const CUdeviceptr pointer = reinterpret_cast<CUdeviceptr>(address);
    check_cuda(
            "cuPointerGetAttribute(memory type)",
            cuPointerGetAttribute(
                    &memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer));
    check_cuda(
            "cuPointerGetAttribute(managed status)",
            cuPointerGetAttribute(
                    &is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, pointer));
    check_cuda(
            "cuPointerGetAttribute(context)",
            cuPointerGetAttribute(
                    &pointer_context, CU_POINTER_ATTRIBUTE_CONTEXT, pointer));
    check_cuda(
            "cuPointerGetAttribute(device ordinal)",
            cuPointerGetAttribute(
                    &pointer_ordinal,
                    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer));
    const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(address);
    const std::uintptr_t base = static_cast<std::uintptr_t>(arena_base);
    if (memory_type != CU_MEMORYTYPE_DEVICE || is_managed != 0
            || pointer_context != context
            || pointer_ordinal != static_cast<int>(ordinal)
            || raw < base || raw - base >= arena_bytes) {
        throw std::runtime_error(
                "CUDA tensor storage is incompatible with the owning device "
                "context");
    }
}

class CudaWorkspace final : public RawWorkspace {
public:
    CudaWorkspace(CudaDevice& device, void* address, std::size_t bytes)
            : RawWorkspace(device, bytes), device_(device), address_(address) {}

    ~CudaWorkspace() override {
        if (address_ != nullptr) {
            device_.release_workspace(this, address_, byte_size());
        }
    }

    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    CudaDevice& device_;
    void* address_;
};

}  // namespace

class CudaTensor final : public Tensor {
public:
    CudaTensor(const TensorSpec& spec, CudaDevice& device)
            : Tensor(spec, device), device_(device),
              state_(&device.registry_state()) {
        address_ = device_.allocate_data(view().spec().tiled_storage_nbytes());
        try {
            device_.activate();
            validate_native_storage(
                    address_, device_.context(), device_.ordinal(),
                    device_.data_backing(), device_.data_backing_bytes());
            cuda_detail::check_cuda_kernel(
                    "cudaMemset",
                    cudaMemset(
                            address_, 0, view().spec().tiled_storage_nbytes()));
            cuda_detail::check_cuda_kernel(
                    "cudaDeviceSynchronize", cudaDeviceSynchronize());
        } catch (...) {
            void* rejected = std::exchange(address_, nullptr);
            device_.release_data(rejected);
            throw;
        }
    }

    ~CudaTensor() noexcept override {
        if (address_ == nullptr) {
            return;
        }

        const std::size_t bytes = view().spec().tiled_storage_nbytes();
        const auto quarantine_storage = [this, bytes]() noexcept {
            try {
                state_->quarantine.emplace<detail::AllocatorCleanupAction>(
                        device_.data_allocator(), address_, bytes);
            } catch (...) {
                // Keep failed storage unavailable if quarantine allocation fails.
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
                state_->registry, address_, quarantine_storage, release_storage);
    }

private:
    [[nodiscard]] void* storage_handle() noexcept override { return address_; }

    void region_from_host(
            const TensorView& destination, std::span<const std::byte> source,
            RawWorkspaceView workspace) override {
        std::lock_guard<std::mutex> lock(device_.transfer_mutex_);
        cuda_detail::region_from_host(
                device_.transfer_stream(), device_.context_, device_,
                device_.registry_state_, destination, workspace, source,
                device_.transfer_resource_poisoned_);
    }

    void region_to_host(
            const TensorView& source, std::span<std::byte> destination,
            RawWorkspaceView workspace) const override {
        std::lock_guard<std::mutex> lock(device_.transfer_mutex_);
        cuda_detail::region_to_host(
                device_.transfer_stream(), device_.context_, device_,
                device_.registry_state_, source, workspace, destination,
                device_.transfer_resource_poisoned_);
    }

    CudaDevice& device_;
    detail::RegistryState* state_;
    void* address_ = nullptr;
};

std::unique_ptr<Tensor> CudaDevice::create_tensor(const TensorSpec& spec) {
    // Base Tensor validation precedes all device interaction.
    return std::make_unique<CudaTensor>(spec, *this);
}

std::unique_ptr<RawWorkspace> CudaDevice::create_workspace(std::size_t bytes) {
    if (bytes == 0) {
        return std::make_unique<CudaWorkspace>(*this, nullptr, 0);
    }
    void* address = allocate_data(bytes);
    try {
        activate();
        validate_native_storage(
                address, context_, ordinal_, data_backing_, data_backing_bytes_);
        return std::make_unique<CudaWorkspace>(*this, address, bytes);
    } catch (...) {
        release_data(address);
        throw;
    }
}

}  // namespace iom
