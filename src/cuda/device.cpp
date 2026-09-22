#include "iom/cuda/device.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "copy.hpp"
#include "device_internal.hpp"
#include "driver.hpp"
#include "iom/detail/gpu_arena_config.hpp"
#include "../shared/standard_tiled_copy.hpp"

namespace iom {

class CudaTransferResource final {
public:
    CudaTransferResource()
            : stream_(cuda_detail::gpu_policy::create_queue_stream()) {}
    ~CudaTransferResource() noexcept {
        (void)cuda_detail::gpu_policy::synchronize_stream_noexcept(stream_);
        cuda_detail::gpu_policy::destroy_queue_stream_noexcept(stream_);
    }

    CudaTransferResource(const CudaTransferResource&) = delete;
    CudaTransferResource& operator=(const CudaTransferResource&) = delete;

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

namespace {

void free_status_cells(void* address) noexcept {
    if (address != nullptr) {
        (void)cudaFreeHost(address);
    }
}

[[nodiscard]] std::invalid_argument invalid_ordinal(
        std::uint32_t ordinal, int device_count) {
    return std::invalid_argument(
            "CUDA device ordinal " + std::to_string(ordinal)
            + " is unavailable; device count is "
            + std::to_string(device_count));
}

void check_arena_alloc_cuda(const char* operation, CUresult status) {
    if (status == CUDA_ERROR_OUT_OF_MEMORY) {
        throw std::bad_alloc();
    }
    check_cuda(operation, status);
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

}  // namespace

CudaDevice::CudaDevice(
        std::uint32_t ordinal, CUdevice device, CUcontext context,
        std::unique_ptr<ListAllocator> data_allocator,
        std::unique_ptr<FixedSizeAllocator> metadata_allocator,
        CUdeviceptr data_backing, std::size_t data_backing_bytes,
        CUdeviceptr metadata_backing)
        : ordinal_(ordinal), device_(device), context_(context),
          transfer_resource_(nullptr), data_allocator_(std::move(data_allocator)),
          metadata_allocator_(std::move(metadata_allocator)),
          data_backing_(data_backing), data_backing_bytes_(data_backing_bytes),
          metadata_backing_(metadata_backing),
          queue_slot_count_(metadata_allocator_->block_count()
                            / detail::kMaxLiveGpuQueues) {
    activate();
    transfer_resource_ = std::make_unique<CudaTransferResource>();
}

CudaDevice::~CudaDevice() {
    if (context_ != nullptr) {
        try {
            activate();
            transfer_resource_.reset();
        } catch (...) {
        }
        registry_state_.quarantine.drain();
        reclaim_retained_leases();
        discard_all_retained_leases();
        data_allocator_.reset();
        metadata_allocator_.reset();
        if (data_backing_ != 0) {
            (void)cuda_detail::free_attempt(
                    data_backing_, cuda_detail::AllocationClass::data_backing,
                    cuda_detail::AllocationPhase::post_publication);
            data_backing_ = 0;
        }
        if (metadata_backing_ != 0) {
            (void)cuda_detail::free_attempt(
                    metadata_backing_,
                    cuda_detail::AllocationClass::metadata_backing,
                    cuda_detail::AllocationPhase::post_publication);
            metadata_backing_ = 0;
        }
        (void)cuda_detail::driver_calls.primary_ctx_release(device_);
        context_ = nullptr;
    } else {
        registry_state_.quarantine.drain();
    }
}

BackendKind CudaDevice::backend_kind() const noexcept {
    return BackendKind::CUDA;
}

std::uint32_t CudaDevice::backend_device() const noexcept {
    return ordinal_;
}

std::span<const iom::DataType>
        CudaDevice::supported_data_types() const noexcept {
    return detail::standard_supported_data_types();
}

std::unique_ptr<DeviceOps> CudaDevice::create_ops() {
    activate();
    // The native BF16 linear capability is a runtime fact of this exact device
    // and of the image the driver loaded for it, so it is resolved here, once
    // per queue, and never re-read from a submission thread.
    return cuda_detail::make_queue(
            *this, *this, context_, registry_state_,
            cuda_detail::linear_bf16_wmma_facility(device_));
}

std::size_t CudaDevice::queue_slot_count() const noexcept {
    return queue_slot_count_;
}

detail::QueueResourceLease CudaDevice::reserve_queue_resources() {
    activate();
    std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
    const std::size_t slot_count = queue_slot_count_;
    std::vector<std::size_t> indices;
    indices.reserve(slot_count);
    try {
        for (std::size_t index = 0; index < slot_count; ++index) {
            void* block = metadata_allocator_->alloc(detail::kMetadataSlotBytes);
            indices.push_back(metadata_allocator_->index_of(block));
        }
    } catch (...) {
        for (const std::size_t index : indices) {
            metadata_allocator_->free(metadata_allocator_->ptr_from_index(index));
        }
        throw;
    }
    std::sort(indices.begin(), indices.end());
    const std::size_t base = indices.front();
    for (std::size_t index = 0; index < slot_count; ++index) {
        if (indices[index] != base + index || base % slot_count != 0) {
            release_queue_resources_locked(base, slot_count);
            throw std::logic_error(
                    "metadata partition reservation lost the device's C-slot "
                    "geometry");
        }
    }
    const std::size_t mirror_bytes =
            slot_count * detail::kMetadataSlotBytes;
    std::unique_ptr<std::byte[]> host_mirrors;
    try {
        host_mirrors = std::make_unique<std::byte[]>(mirror_bytes);
    } catch (...) {
        release_queue_resources_locked(base, slot_count);
        throw;
    }

    if (slot_count > std::numeric_limits<std::size_t>::max()
            / sizeof(std::uint32_t)) {
        release_queue_resources_locked(base, slot_count);
        throw std::overflow_error("CUDA status-cell allocation size overflows");
    }
    const std::size_t status_bytes = slot_count * sizeof(std::uint32_t);
    void* status_cells = nullptr;
    const cudaError_t status = cudaHostAlloc(
            &status_cells, status_bytes, cudaHostAllocDefault);
    if (status == cudaErrorMemoryAllocation) {
        free_status_cells(status_cells);
        release_queue_resources_locked(base, slot_count);
        throw std::bad_alloc();
    }
    if (status != cudaSuccess) {
        free_status_cells(status_cells);
        release_queue_resources_locked(base, slot_count);
        throw std::runtime_error(
                std::string("cudaHostAlloc failed with ")
                + cudaGetErrorName(status) + ": "
                + cudaGetErrorString(status));
    }
    return make_lease(
            *this, base, slot_count,
            metadata_allocator_->ptr_from_index(base),
            std::move(host_mirrors), status_cells, &free_status_cells);
}

void CudaDevice::release_queue_resources(
        std::size_t first_slot, std::size_t slot_count) noexcept {
    std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
    release_queue_resources_locked(first_slot, slot_count);
}

void CudaDevice::retain_unknown_lease(
        std::function<bool()> reclaim, std::function<void()> discard) {
    std::lock_guard<std::mutex> lock(retained_mutex_);
    if (retained_count_ >= detail::kMaxLiveGpuQueues) {
        throw std::logic_error("retained queue-lease quarantine is full");
    }
    retained_leases_[retained_count_].reclaim = std::move(reclaim);
    retained_leases_[retained_count_].discard = std::move(discard);
    ++retained_count_;
}

void CudaDevice::reclaim_retained_leases() {
    std::array<RetainedLease, detail::kMaxLiveGpuQueues> pending;
    std::size_t pending_count = 0;
    {
        std::lock_guard<std::mutex> lock(retained_mutex_);
        pending_count = retained_count_;
        for (std::size_t index = 0; index < pending_count; ++index) {
            pending[index] = std::move(retained_leases_[index]);
            retained_leases_[index] = {};
        }
        retained_count_ = 0;
    }
    for (std::size_t index = 0; index < pending_count; ++index) {
        if (!pending[index].reclaim()) {
            std::lock_guard<std::mutex> lock(retained_mutex_);
            retained_leases_[retained_count_++] = std::move(pending[index]);
        }
    }
}

void CudaDevice::activate() const {
    check_cuda(
            "cuCtxSetCurrent",
            cuda_detail::driver_calls.ctx_set_current(context_));
}

CUcontext CudaDevice::context() const noexcept {
    return context_;
}

std::uint32_t CudaDevice::ordinal() const noexcept {
    return ordinal_;
}

detail::RegistryState& CudaDevice::registry_state() noexcept {
    return registry_state_;
}

void* CudaDevice::allocate_data(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
    return data_allocator_->alloc(bytes);
}

void CudaDevice::release_data(void* address) {
    std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
    data_allocator_->free(address);
}

void CudaDevice::release_workspace(
        const RawWorkspace* owner, void* address, std::size_t bytes) noexcept {
    bool retained = false;
    {
        std::lock_guard<std::mutex> lock(registry_state_.allocation_mutex);
        retained = detail::workspace_range_retained(
                registry_state_.workspace_leases, owner, address, bytes);
    }
    if (retained) {
        try {
            registry_state_.quarantine.emplace<detail::AllocatorCleanupAction>(
                    *data_allocator_, address, bytes);
        } catch (...) {
        }
        return;
    }
    release_data(address);
}

iom::Allocator& CudaDevice::data_allocator() noexcept {
    return *data_allocator_;
}

CUdeviceptr CudaDevice::data_backing() const noexcept {
    return data_backing_;
}

std::size_t CudaDevice::data_backing_bytes() const noexcept {
    return data_backing_bytes_;
}

cudaStream_t CudaDevice::transfer_stream() const noexcept {
    return transfer_resource_->stream();
}

void CudaDevice::release_queue_resources_locked(
        std::size_t first_slot, std::size_t slot_count) noexcept {
    for (std::size_t index = 0; index < slot_count; ++index) {
        metadata_allocator_->free(
                metadata_allocator_->ptr_from_index(first_slot + index));
    }
}

void CudaDevice::discard_all_retained_leases() noexcept {
    std::lock_guard<std::mutex> lock(retained_mutex_);
    for (std::size_t index = 0; index < retained_count_; ++index) {
        retained_leases_[index].discard();
        retained_leases_[index] = {};
    }
    retained_count_ = 0;
}

namespace {

class CudaBackingGuard final {
public:
    CudaBackingGuard() = default;
    explicit CudaBackingGuard(
            CUdeviceptr address, cuda_detail::AllocationClass classification)
            : address_(address), classification_(classification) {}

    CudaBackingGuard(const CudaBackingGuard&) = delete;
    CudaBackingGuard& operator=(const CudaBackingGuard&) = delete;

    ~CudaBackingGuard() noexcept {
        if (address_ != 0) {
            (void)cuda_detail::free_attempt(
                    address_, classification_,
                    cuda_detail::AllocationPhase::setup);
        }
    }

    void dismiss() noexcept { address_ = 0; }

private:
    CUdeviceptr address_ = 0;
    cuda_detail::AllocationClass classification_ =
            cuda_detail::AllocationClass::other_iom_setup;
};

}  // namespace

std::unique_ptr<Device> make_cuda_device(
        std::uint32_t device_ordinal, DeviceMemoryConfig memory_config,
        QueueConfig queue_config) {
    const std::size_t data_bytes = detail::valid_tensor_arena_bytes(
            memory_config.tensor_arena_bytes);
    const std::size_t metadata_bytes = detail::standard_gpu_metadata_bytes(
            queue_config.max_in_flight_per_queue);

    check_cuda("cuInit", cuda_detail::driver_calls.init(0));

    int device_count = 0;
    CUresult count_status =
            cuda_detail::driver_calls.device_get_count(&device_count);
    if (count_status == CUDA_ERROR_NO_DEVICE) {
        throw invalid_ordinal(device_ordinal, 0);
    }
    check_cuda("cuDeviceGetCount", count_status);

    if (device_count <= 0
            || device_ordinal >= static_cast<std::uint32_t>(device_count)
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

    CUdeviceptr data_backing = 0;
    check_arena_alloc_cuda(
            "cuMemAlloc(data backing)",
            cuda_detail::allocation_attempt(
                    &data_backing, data_bytes,
                    cuda_detail::AllocationClass::data_backing,
                    cuda_detail::AllocationPhase::setup));
    CudaBackingGuard data_guard{
            data_backing, cuda_detail::AllocationClass::data_backing};

    CUdeviceptr metadata_backing = 0;
    check_arena_alloc_cuda(
            "cuMemAlloc(metadata backing)",
            cuda_detail::allocation_attempt(
                    &metadata_backing, metadata_bytes,
                    cuda_detail::AllocationClass::metadata_backing,
                    cuda_detail::AllocationPhase::setup));
    CudaBackingGuard metadata_guard{
            metadata_backing,
            cuda_detail::AllocationClass::metadata_backing};

    if (data_backing % 32 != 0 || metadata_backing % 32 != 0) {
        throw std::runtime_error("CUDA arena backing is not 32-byte aligned");
    }

    auto data_allocator = std::make_unique<ListAllocator>(
            reinterpret_cast<void*>(data_backing), data_bytes, 32);
    auto metadata_allocator = std::make_unique<FixedSizeAllocator>(
            reinterpret_cast<void*>(metadata_backing), metadata_bytes, 32, 512);
    if (metadata_allocator->block_count()
            != 4 * queue_config.max_in_flight_per_queue) {
        throw std::runtime_error(
                "CUDA metadata arena block count does not match the configured "
                "capacity");
    }

    auto result = std::make_unique<CudaDevice>(
            device_ordinal, device, context, std::move(data_allocator),
            std::move(metadata_allocator), data_backing, data_bytes,
            metadata_backing);
    context_guard.dismiss();
    data_guard.dismiss();
    metadata_guard.dismiss();
    return result;
}

}  // namespace iom
