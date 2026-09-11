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
#include "driver.hpp"
#include "iom/detail/gpu_arena_config.hpp"
#include "../shared/standard_tiled_copy.hpp"
namespace iom {


    namespace {
        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, int device_count) {
            return std::invalid_argument(
                    "CUDA device ordinal " + std::to_string(ordinal)
                    + " is unavailable; device count is "
                    + std::to_string(device_count));
        }

        // A native device-memory failure maps to std::bad_alloc; any other
        // driver status keeps the established runtime category.
        void check_arena_alloc_cuda(const char* operation, CUresult status) {
            if (status == CUDA_ERROR_OUT_OF_MEMORY) {
                throw std::bad_alloc();
            }
            check_cuda(operation, status);
        }

        // Validates that an arena suballocation is unmanaged device memory
        // belonging to this exact context and ordinal and lies inside the
        // owning arena backing. Interior addresses are accepted: the
        // driver-attribute identity checks pin the exact context and
        // ordinal, and membership in the device's own arena range binds the
        // address to this device's reservation without requiring it to be
        // the base of an independent native allocation (the driver reports
        // the queried address, not the containing allocation's base, so
        // base equality would reject every interior suballocation).
        void validate_native_storage(
                void* address, CUcontext context, std::uint32_t ordinal,
                CUdeviceptr arena_base, std::size_t arena_bytes) {
            CUmemorytype memory_type{};
            int is_managed = 0;
            CUcontext pointer_context = nullptr;
            int pointer_ordinal = -1;
            const CUdeviceptr pointer =
                    reinterpret_cast<CUdeviceptr>(address);
            check_cuda(
                    "cuPointerGetAttribute(memory type)",
                    cuPointerGetAttribute(
                            &memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                            pointer));
            check_cuda(
                    "cuPointerGetAttribute(managed status)",
                    cuPointerGetAttribute(
                            &is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                            pointer));
            check_cuda(
                    "cuPointerGetAttribute(context)",
                    cuPointerGetAttribute(
                            &pointer_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                            pointer));
            check_cuda(
                    "cuPointerGetAttribute(device ordinal)",
                    cuPointerGetAttribute(
                            &pointer_ordinal,
                            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer));
            const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(address);
            const std::uintptr_t base =
                    static_cast<std::uintptr_t>(arena_base);
            if (memory_type != CU_MEMORYTYPE_DEVICE || is_managed != 0
                    || pointer_context != context
                    || pointer_ordinal != static_cast<int>(ordinal)
                    || raw < base || raw - base >= arena_bytes) {
                throw std::runtime_error(
                        "CUDA tensor storage is incompatible with the owning "
                        "device context");
            }
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

        class CudaTransferResource final {
        public:
            CudaTransferResource()
                    : stream_(cuda_detail::gpu_policy::create_queue_stream()) {}
            ~CudaTransferResource() noexcept {
                (void)cuda_detail::gpu_policy::synchronize_stream_noexcept(
                        stream_);
                cuda_detail::gpu_policy::destroy_queue_stream_noexcept(stream_);
            }

            CudaTransferResource(const CudaTransferResource&) = delete;
            CudaTransferResource& operator=(const CudaTransferResource&) = delete;

            [[nodiscard]] cudaStream_t stream() const noexcept {
                return stream_;
            }

        private:
            cudaStream_t stream_ = nullptr;
        };

        class CudaDevice final : public Device,
                                 public detail::QueueResourceProvider {
        public:
            CudaDevice(
                    std::uint32_t ordinal, CUdevice device, CUcontext context,
                    std::unique_ptr<ListAllocator> data_allocator,
                    std::unique_ptr<FixedSizeAllocator> metadata_allocator,
                    CUdeviceptr data_backing, std::size_t data_backing_bytes,
                    CUdeviceptr metadata_backing)
                    : ordinal_(ordinal), device_(device), context_(context),
                      transfer_resource_(nullptr),
                      data_allocator_(std::move(data_allocator)),
                      metadata_allocator_(std::move(metadata_allocator)),
                      data_backing_(data_backing),
                      data_backing_bytes_(data_backing_bytes),
                      metadata_backing_(metadata_backing),
                      queue_slot_count_(metadata_allocator_->block_count()
                                        / detail::kMaxLiveGpuQueues) {
                activate();
                transfer_resource_ =
                        std::make_unique<CudaTransferResource>();
            }

            CudaDevice(const CudaDevice&) = delete;
            CudaDevice& operator=(const CudaDevice&) = delete;

            ~CudaDevice() override {
                if (context_ != nullptr) {
                    try {
                        activate();
                        transfer_resource_.reset();
                    } catch (...) {
                    }
                    registry_state_.quarantine.drain();
                    // Covering-proof every retained unknown-completion
                    // lease; unproven bundles are discarded best-effort
                    // while the native context is still valid, before
                    // allocator bookkeeping disappears.
                    reclaim_retained_leases();
                    discard_all_retained_leases();
                    // Destroy allocator bookkeeping before releasing the
                    // native backings so no live owner, lease, or retained
                    // quarantine action can reference an allocator past this
                    // point; drain above already freed every recognized
                    // range back into the data allocator.
                    data_allocator_.reset();
                    metadata_allocator_.reset();
                    if (data_backing_ != 0) {
                        (void)cuda_detail::free_attempt(
                                data_backing_,
                                cuda_detail::AllocationClass::data_backing,
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
                    (void)cuda_detail::driver_calls.primary_ctx_release(
                            device_);
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
                return detail::standard_supported_data_types();
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec& spec) override;
            [[nodiscard]] std::unique_ptr<RawWorkspace> create_workspace(
                    std::size_t bytes) override;

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                activate();
                return cuda_detail::make_queue(
                        *this, *this, context_, registry_state_);
            }

            // -- detail::QueueResourceProvider: fixed queue-resource
            // geometry. All allocator bookkeeping is serialized here at the
            // device boundary and never held across native stream/event
            // creation, waits, submissions, callbacks, or drains.
            [[nodiscard]] std::size_t queue_slot_count() const noexcept
                    override {
                return queue_slot_count_;
            }

            [[nodiscard]] detail::QueueResourceLease
                    reserve_queue_resources() override {
                std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
                const std::size_t slot_count = queue_slot_count_;
                std::vector<std::size_t> indices;
                indices.reserve(slot_count);
                try {
                    for (std::size_t index = 0; index < slot_count;
                         ++index) {
                        void* block =
                                metadata_allocator_->alloc(
                                        detail::kMetadataSlotBytes);
                        indices.push_back(
                                metadata_allocator_->index_of(block));
                    }
                } catch (...) {
                    for (const std::size_t index : indices) {
                        metadata_allocator_->free(
                                metadata_allocator_->ptr_from_index(index));
                    }
                    // Propagates std::bad_alloc for a fifth live queue.
                    throw;
                }
                std::sort(indices.begin(), indices.end());
                const std::size_t base = indices.front();
                for (std::size_t index = 0; index < slot_count; ++index) {
                    if (indices[index] != base + index
                            || base % slot_count != 0) {
                        release_queue_resources_locked(base, slot_count);
                        throw std::logic_error(
                                "metadata partition reservation lost the "
                                "device's C-slot geometry");
                    }
                }
                const std::size_t mirror_bytes =
                        slot_count * detail::kMetadataSlotBytes;
                std::unique_ptr<std::byte[]> host_mirrors;
                try {
                    host_mirrors = std::make_unique<std::byte[]>(
                            mirror_bytes);
                } catch (...) {
                    release_queue_resources_locked(base, slot_count);
                    throw;
                }
                return make_lease(
                        *this, base, slot_count,
                        metadata_allocator_->ptr_from_index(base),
                        std::move(host_mirrors));
            }

            void release_queue_resources(
                    std::size_t first_slot,
                    std::size_t slot_count) noexcept override {
                std::lock_guard<std::mutex> lock(bookkeeping_mutex_);
                release_queue_resources_locked(first_slot, slot_count);
            }

            void retain_unknown_lease(
                    std::function<bool()> reclaim,
                    std::function<void()> discard) override {
                std::lock_guard<std::mutex> lock(retained_mutex_);
                if (retained_count_ >= detail::kMaxLiveGpuQueues) {
                    // One retained lease per partition; this cannot exceed
                    // the queue-count bound.
                    throw std::logic_error(
                            "retained queue-lease quarantine is full");
                }
                retained_leases_[retained_count_].reclaim =
                        std::move(reclaim);
                retained_leases_[retained_count_].discard =
                        std::move(discard);
                ++retained_count_;
            }

            void reclaim_retained_leases() override {
                std::array<RetainedLease, detail::kMaxLiveGpuQueues> pending;
                std::size_t pending_count = 0;
                {
                    std::lock_guard<std::mutex> lock(retained_mutex_);
                    pending_count = retained_count_;
                    for (std::size_t index = 0; index < pending_count;
                         ++index) {
                        pending[index] = std::move(retained_leases_[index]);
                        retained_leases_[index] = {};
                    }
                    retained_count_ = 0;
                }
                // Callbacks synchronize retained streams: never under any
                // device lock.
                for (std::size_t index = 0; index < pending_count; ++index) {
                    if (!pending[index].reclaim()) {
                        std::lock_guard<std::mutex> lock(retained_mutex_);
                        retained_leases_[retained_count_++] =
                                std::move(pending[index]);
                    }
                }
            }

            void activate() const {
                check_cuda(
                        "cuCtxSetCurrent",
                        cuda_detail::driver_calls.ctx_set_current(context_));
            }

            [[nodiscard]] CUcontext context() const noexcept {
                return context_;
            }
            [[nodiscard]] std::uint32_t ordinal() const noexcept {
                return ordinal_;
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

            // Returns one raw-workspace arena range at owner destruction.
            // A retained (live or quarantined) lease keeps the range out
            // of the allocator: it is deferred into the device quarantine
            // and freed only on the final drain, never reused. The lease
            // check releases the registry allocation mutex before any
            // allocator work.
            void release_workspace(
                    const RawWorkspace* owner, void* address,
                    std::size_t bytes) noexcept {
                bool retained = false;
                {
                    std::lock_guard<std::mutex> lock(
                            registry_state_.allocation_mutex);
                    retained = detail::workspace_range_retained(
                            registry_state_.workspace_leases, owner,
                            address, bytes);
                }
                if (retained) {
                    try {
                        registry_state_.quarantine
                                .emplace<detail::AllocatorCleanupAction>(
                                        *data_allocator_, address, bytes);
                    } catch (...) {
                        // Keep failed storage unavailable if quarantine
                        // allocation itself fails.
                    }
                    return;
                }
                release_data(address);
            }

            [[nodiscard]] iom::Allocator& data_allocator() noexcept {
                return *data_allocator_;
            }

            [[nodiscard]] CUdeviceptr data_backing() const noexcept {
                return data_backing_;
            }

            [[nodiscard]] std::size_t data_backing_bytes() const noexcept {
                return data_backing_bytes_;
            }

        private:
            struct RetainedLease {
                std::function<bool()> reclaim;
                std::function<void()> discard;
            };

            // Host bookkeeping only; caller holds bookkeeping_mutex_.
            void release_queue_resources_locked(
                    std::size_t first_slot, std::size_t slot_count) noexcept {
                for (std::size_t index = 0; index < slot_count; ++index) {
                    metadata_allocator_->free(
                            metadata_allocator_->ptr_from_index(
                                    first_slot + index));
                }
            }

            void discard_all_retained_leases() noexcept {
                std::lock_guard<std::mutex> lock(retained_mutex_);
                for (std::size_t index = 0; index < retained_count_;
                     ++index) {
                    retained_leases_[index].discard();
                    retained_leases_[index] = {};
                }
                retained_count_ = 0;
            }

            std::uint32_t ordinal_;
            CUdevice device_;
            CUcontext context_;
            detail::RegistryState registry_state_;
            std::unique_ptr<CudaTransferResource> transfer_resource_;
            mutable std::mutex transfer_mutex_;
            bool transfer_resource_poisoned_ = false;
            std::mutex bookkeeping_mutex_;
            std::unique_ptr<ListAllocator> data_allocator_;
            std::unique_ptr<FixedSizeAllocator> metadata_allocator_;
            CUdeviceptr data_backing_ = 0;
            std::size_t data_backing_bytes_ = 0;
            CUdeviceptr metadata_backing_ = 0;
            // Fixed queue-resource geometry: exactly C slots per partition,
            // four partitions, and at most one retained lease per credit.
            std::size_t queue_slot_count_ = 0;
            std::mutex retained_mutex_;
            std::array<RetainedLease, detail::kMaxLiveGpuQueues>
                    retained_leases_{};
            std::size_t retained_count_ = 0;
            friend class CudaTensor;
        };

        /**
         * Owner of one raw-workspace arena suballocation. The base
         * RawWorkspace registers the exact identity with the device
         * before the body can fail; destruction hands the range back
         * through the device boundary, which keeps a retained lease out
         * of the allocator.
         */
        class CudaWorkspace final : public RawWorkspace {
        public:
            CudaWorkspace(
                    CudaDevice& device, void* address, std::size_t bytes)
                    : RawWorkspace(device, bytes),
                      device_(device),
                      address_(address) {}

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

        class CudaTensor final : public Tensor {
        public:
            CudaTensor(const TensorSpec& spec, CudaDevice& device)
                    : Tensor(spec, device), device_(device),
                      state_(&device.registry_state()) {
                address_ = device_.allocate_data(
                        view().spec().tiled_storage_nbytes());
                try {
                    device_.activate();
                    validate_native_storage(
                            address_, device_.context(), device_.ordinal(),
                            device_.data_backing(),
                            device_.data_backing_bytes());
                    cuda_detail::check_cuda_kernel(
                            "cudaMemset",
                            cudaMemset(
                                    address_, 0,
                                    view().spec().tiled_storage_nbytes()));
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

                const std::size_t bytes =
                        view().spec().tiled_storage_nbytes();
                const auto quarantine_storage = [this, bytes]() noexcept {
                    try {
                        state_->quarantine
                                .emplace<detail::AllocatorCleanupAction>(
                                        device_.data_allocator(), address_,
                                        bytes);
                    } catch (...) {
                        // Keep failed storage unavailable if quarantine
                        // allocation itself fails.
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
                    std::span<const std::byte> source,
                    RawWorkspaceView workspace) override {
                std::lock_guard<std::mutex> lock(device_.transfer_mutex_);
                cuda_detail::region_from_host(
                        device_.transfer_resource_->stream(),
                        device_.context_, device_, device_.registry_state_,
                        destination, workspace, source,
                        device_.transfer_resource_poisoned_);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination,
                    RawWorkspaceView workspace) const override {
                std::lock_guard<std::mutex> lock(device_.transfer_mutex_);
                cuda_detail::region_to_host(
                        device_.transfer_resource_->stream(),
                        device_.context_, device_, device_.registry_state_,
                        source, workspace, destination,
                        device_.transfer_resource_poisoned_);
            }
            CudaDevice& device_;
            detail::RegistryState* state_;
            void* address_ = nullptr;
        };

    }  // namespace

    std::unique_ptr<Tensor> CudaDevice::create_tensor(
            const TensorSpec& spec) {
        // The base Tensor ctor validates the spec before any device
        // interaction; the CudaTensor ctor then suballocates the data arena.
        return std::make_unique<CudaTensor>(spec, *this);
    }

    std::unique_ptr<RawWorkspace> CudaDevice::create_workspace(
            std::size_t bytes) {
        if (bytes == 0) {
            // Valid empty owner: no arena suballocation, no native call.
            return std::make_unique<CudaWorkspace>(*this, nullptr, 0);
        }
        // Positive creation suballocates the already reserved data arena
        // under the allocator bookkeeping boundary; it never reserves a
        // new native backing. Exhaustion, including fragmentation with no
        // fitting contiguous range, is std::bad_alloc with no fallback.
        void* address = allocate_data(bytes);
        try {
            activate();
            validate_native_storage(
                    address, context_, ordinal_, data_backing_,
                    data_backing_bytes_);
            return std::make_unique<CudaWorkspace>(*this, address, bytes);
        } catch (...) {
            release_data(address);
            throw;
        }
    }

    namespace {
        // Factory-local RAII owner for one arena backing. Setup rollback
        // frees the backing through the same instrumented boundary used for
        // its reservation, in reverse acquisition order, while the retained
        // context is still valid; the original failure is preserved.
        class CudaBackingGuard final {
        public:
            CudaBackingGuard() = default;
            explicit CudaBackingGuard(
                    CUdeviceptr address,
                    cuda_detail::AllocationClass classification)
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
        const std::size_t metadata_bytes =
                detail::standard_gpu_metadata_bytes(
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

        // Reserve exactly two native backings in this exact context and
        // device: one data arena and one metadata arena of checked capacity.
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

        // cuMemAlloc reserves 256-byte-aligned memory; verify the documented
        // 32-byte base contract before any allocator is constructed.
        if (data_backing % 32 != 0 || metadata_backing % 32 != 0) {
            throw std::runtime_error(
                    "CUDA arena backing is not 32-byte aligned");
        }

        auto data_allocator = std::make_unique<ListAllocator>(
                reinterpret_cast<void*>(data_backing), data_bytes, 32);
        auto metadata_allocator = std::make_unique<FixedSizeAllocator>(
                reinterpret_cast<void*>(metadata_backing), metadata_bytes,
                32, 512);
        if (metadata_allocator->block_count()
                != 4 * queue_config.max_in_flight_per_queue) {
            throw std::runtime_error(
                    "CUDA metadata arena block count does not match the "
                    "configured capacity");
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
