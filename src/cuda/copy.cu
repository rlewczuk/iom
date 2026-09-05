#include "copy.hpp"
#include "transfer_pool.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <array>
#include <condition_variable>
#include <limits>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include "driver.hpp"

namespace iom::cuda_detail {
namespace {

std::atomic<SubmissionFault> g_submission_fault{SubmissionFault::none};

[[nodiscard]] bool consume_submission_fault(
        SubmissionFault point) noexcept {
    SubmissionFault expected = point;
    return g_submission_fault.compare_exchange_strong(
            expected, SubmissionFault::none, std::memory_order_acq_rel);
}


void check_cuda_kernel(const char* operation, cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
                std::string(operation) + " failed with "
                + cudaGetErrorName(status) + ": "
                + cudaGetErrorString(status));
    }
}

struct gpu_policy {
    using context_type = CUcontext;
    using stream_type = cudaStream_t;
    using event_type = cudaEvent_t;

    [[nodiscard]] static constexpr stream_type null_stream() noexcept {
        return nullptr;
    }

    static void activate(context_type context) {
        check_cuda(
                "cuCtxSetCurrent",
                driver_calls.ctx_set_current(context));
    }

    [[nodiscard]] static stream_type create_queue_stream() {
        stream_type stream = nullptr;
        check_cuda_kernel(
                "cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        return stream;
    }
    static void destroy_queue_stream_noexcept(stream_type stream) noexcept {
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
    }

    static void synchronize_stream(stream_type stream) {
        check_cuda_kernel("cudaStreamSynchronize", cudaStreamSynchronize(stream));
    }

    static void create_event(event_type* event) {
        if (consume_submission_fault(SubmissionFault::event_create)) {
            check_cuda_kernel(
                    "cudaEventCreateWithFlags", cudaErrorInvalidValue);
        }
        check_cuda_kernel(
                "cudaEventCreateWithFlags",
                cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }
    static void destroy_event_noexcept(event_type event) noexcept {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
        }
    }
    static void synchronize_event(event_type event) {
        check_cuda_kernel("cudaEventSynchronize", cudaEventSynchronize(event));
    }
    static void record_event(event_type event, stream_type stream) {
        cudaError_t status = cudaEventRecord(event, stream);
        if (consume_submission_fault(SubmissionFault::event_record)) {
            status = cudaErrorInvalidValue;
        }
        check_cuda_kernel("cudaEventRecord", status);
    }
    static void record_event_no_fault(
            event_type event, stream_type stream) noexcept {
        (void)cudaEventRecord(event, stream);
    }

    [[nodiscard]] static void* allocate(std::size_t bytes) {
        CUdeviceptr address = 0;
        check_cuda("cuMemAlloc", cuMemAlloc(&address, bytes));
        return reinterpret_cast<void*>(address);
    }
    [[nodiscard]] static void* staging_address(CUdeviceptr address) noexcept {
        return reinterpret_cast<void*>(address);
    }
    static void free_noexcept(void* address) noexcept {
        if (address != nullptr) {
            (void)cuMemFree(reinterpret_cast<CUdeviceptr>(address));
        }
    }

    static void copy_from_host(
            stream_type stream, void* destination, const void* source,
            std::size_t bytes) {
        check_cuda_kernel(
                "cudaMemcpyAsync HtoD",
                cudaMemcpyAsync(
                        destination, source, bytes,
                        cudaMemcpyHostToDevice, stream));
    }
    static void copy_to_host(
            stream_type, void* destination, const void* source,
            std::size_t bytes) {
        check_cuda_kernel(
                "cudaMemcpy DtoH",
                cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
    }
    static void memset(
            stream_type stream, void* destination, std::size_t bytes) {
        check_cuda_kernel(
                "cudaMemsetAsync",
                cudaMemsetAsync(destination, 0, bytes, stream));
    }

    static void check_kernel(const char* operation) {
        ::iom::cuda_detail::check_cuda_kernel(operation, cudaGetLastError());
    }
    static void after_copy_plane_launch(std::size_t plane_index) {
        if (plane_index == 2
                && consume_submission_fault(
                        SubmissionFault::third_plane_launch)) {
            ::iom::cuda_detail::check_cuda_kernel(
                    copy_kernel_operation(), cudaErrorInvalidValue);
        }
    }
    static void after_grid_stride_launch() {
        if (consume_submission_fault(SubmissionFault::third_plane_launch)) {
            ::iom::cuda_detail::check_cuda_kernel(
                    copy_kernel_operation(), cudaErrorInvalidValue);
        }
    }
    [[nodiscard]] static constexpr const char* scatter_kernel_operation() noexcept {
        return "CUDA scatter kernel launch";
    }
    [[nodiscard]] static constexpr const char* gather_kernel_operation() noexcept {
        return "CUDA gather kernel launch";
    }
    [[nodiscard]] static constexpr const char* copy_kernel_operation() noexcept {
        return "CUDA copy kernel launch";
    }

};

void synchronize_and_destroy_stream(cudaStream_t stream) noexcept {
    if (stream == nullptr) {
        return;
    }
    (void)cudaStreamSynchronize(stream);
    (void)cudaStreamDestroy(stream);
}


}  // namespace
}  // namespace iom::cuda_detail

#define IOM_GPU_DEVICE __device__
#define IOM_GPU_GLOBAL __global__
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(__VA_ARGS__)
namespace iom::cuda_detail {

struct CudaCopyMetadataHeader {
    std::uint64_t source_plane_offset;
    std::uint64_t destination_plane_offset;
    std::uint64_t rows;
    std::uint64_t columns;
    std::uint64_t plane_count;
    std::uint32_t bits;
    std::uint32_t leading_rank;
};
static_assert(std::is_trivially_copyable_v<CudaCopyMetadataHeader>);

}  // namespace iom::cuda_detail
#include "../shared/standard_tiled_copy.inl"
namespace iom::cuda_detail {
namespace {

IOM_GPU_GLOBAL void grid_stride_copy_kernel(
        const unsigned char* source, unsigned char* destination,
        const CudaCopyMetadataHeader* metadata) {
    const std::uint64_t padded_rows =
            (metadata->rows / TensorSpec::TILE
             + (metadata->rows % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_columns =
            (metadata->columns / TensorSpec::TILE
             + (metadata->columns % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_elements = padded_rows * padded_columns;
    const std::uint64_t plane_bits = padded_elements * metadata->bits;
    const std::uint64_t words_per_plane = (plane_bits + 31) / 32;
    const std::uint64_t total_words =
            metadata->plane_count * words_per_plane;
    const std::uint64_t* source_strides =
            reinterpret_cast<const std::uint64_t*>(metadata + 1);
    const std::uint64_t* destination_strides =
            source_strides + metadata->leading_rank;
    const std::uint64_t* leading_dimensions =
            destination_strides + metadata->leading_rank;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

    for (std::uint64_t word = IOM_GPU_GLOBAL_INDEX; word < total_words;
         word += stride) {
        const std::uint64_t logical_plane =
                word / words_per_plane;
        const std::uint64_t word_in_plane = word % words_per_plane;
        std::uint64_t source_plane = metadata->source_plane_offset;
        std::uint64_t destination_plane =
                metadata->destination_plane_offset;
        std::uint64_t rest = logical_plane;
        for (std::uint32_t axis = metadata->leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate =
                    rest % leading_dimensions[axis];
            rest /= leading_dimensions[axis];
            source_plane += coordinate * source_strides[axis];
            destination_plane += coordinate * destination_strides[axis];
        }
        detail::copy_tiled_to_tiled_word(
                source, destination, source_plane, destination_plane,
                word_in_plane, metadata->rows, metadata->columns,
                metadata->bits);
    }
}
}

}  // namespace iom::cuda_detail
#undef IOM_LAUNCH_KERNEL
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

namespace iom::cuda_detail {
void TransferStreamPool::Scope::poison() noexcept {
    poisoned_ = true;
}

std::size_t TransferStreamPool::idle_count_for_testing() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return idle_.size();
}

TransferStreamPool::Scope::~Scope() noexcept {
    const cudaError_t sync_status = cudaStreamSynchronize(stream_);
    const bool drop_stream = poisoned_ || (sync_status != cudaSuccess);
    if (drop_stream) {
        std::lock_guard<std::mutex> lock(pool_->mutex_);
        const auto it = pool_->in_use_.find(stream_);
        if (it != pool_->in_use_.end()) {
            pool_->in_use_.erase(it);
            pool_->cv_.notify_all();
        }
        (void)cudaStreamDestroy(stream_);
    } else {
        pool_->release(stream_);
    }
}

TransferStreamPool::Scope TransferStreamPool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closing_) {
        throw std::runtime_error(
                "cudaStreamCreateWithFlags failed with "
                "cudaErrorStreamDestroyed: TransferStreamPool is closing");
    }
    if (idle_.empty()) {
        cudaStream_t stream = nullptr;
        check_cuda_kernel(
                "cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            idle_.push_back(stream);
        } catch (...) {
            (void)cudaStreamDestroy(stream);
            throw;
        }
    }
    cudaStream_t stream = idle_.back();
    idle_.pop_back();
    try {
        in_use_.insert(stream);
    } catch (...) {
        idle_.push_back(stream);
        throw;
    }
    return Scope{*this, stream};
}

void TransferStreamPool::release(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = in_use_.find(stream);
    if (it != in_use_.end()) {
        in_use_.erase(it);
        try {
            idle_.push_back(stream);
        } catch (...) {
            (void)cudaStreamDestroy(stream);
        }
        cv_.notify_all();
    }
}

void TransferStreamPool::destroy() {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    cv_.wait(lock, [this] { return in_use_.empty(); });
    for (const cudaStream_t stream : idle_) {
        (void)cudaStreamDestroy(stream);
    }
    idle_.clear();
}


namespace {

[[nodiscard]] std::size_t checked_metadata_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_metadata_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t metadata_u64(
        std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::size_t padded_dimension(std::size_t value) {
    const std::size_t tiles =
            value / TensorSpec::TILE + (value % TensorSpec::TILE != 0);
    return checked_metadata_mul(
            tiles, TensorSpec::TILE, "metadata size overflows");
}

struct CudaMetadataLayout {
    std::size_t bytes;
    std::size_t total_words;
};

[[nodiscard]] CudaMetadataLayout metadata_layout(
        const TensorView& source, const TensorView& destination) {
    const std::span<const std::size_t> dimensions =
            source.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    if (leading_rank > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("metadata leading rank overflows");
    }
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = checked_metadata_mul(
                plane_count, dimensions[axis], "metadata plane count overflows");
    }
    const std::size_t padded_elements = checked_metadata_mul(
            padded_dimension(rows), padded_dimension(columns),
            "metadata element count overflows");
    const std::size_t plane_bits = checked_metadata_mul(
            padded_elements, detail::leaf_bits(source.spec().data_type),
            "metadata plane bits overflows");
    const std::size_t words_per_plane = checked_metadata_add(
            plane_bits, 31, "metadata word count overflows")
            / 32;
    const std::size_t total_words_size = checked_metadata_mul(
            plane_count, words_per_plane, "metadata total words overflows");
    const std::size_t array_count = checked_metadata_mul(
            leading_rank, 3, "metadata array count overflows");
    const std::size_t array_bytes = checked_metadata_mul(
            array_count, sizeof(std::uint64_t),
            "metadata array bytes overflows");
    const std::size_t bytes = checked_metadata_add(
            sizeof(CudaCopyMetadataHeader), array_bytes,
            "metadata allocation size overflows");
    (void)metadata_u64(rows, "metadata rows overflows");
    (void)metadata_u64(columns, "metadata columns overflows");
    (void)metadata_u64(plane_count, "metadata plane count overflows");
    (void)metadata_u64(total_words_size, "metadata total words overflows");
    for (const std::size_t stride : source.plane_strides()) {
        (void)metadata_u64(stride, "metadata source stride overflows");
    }
    for (const std::size_t stride : destination.plane_strides()) {
        (void)metadata_u64(stride, "metadata destination stride overflows");
    }
    (void)metadata_u64(
            source.plane_offset(), "metadata source offset overflows");
    (void)metadata_u64(
            destination.plane_offset(), "metadata destination offset overflows");
    return {bytes, total_words_size};
}

class CudaMetadataSlotPool final {
public:
    static constexpr std::size_t kMetadataSlotCount = 16;

    explicit CudaMetadataSlotPool(CUcontext context) : context_(context) {}

    ~CudaMetadataSlotPool() {
        try {
            gpu_policy::activate(context_);
        } catch (...) {
        }
        for (Slot& slot : slots_) {
            gpu_policy::free_noexcept(slot.device);
            slot.device = nullptr;
        }
    }

    CudaMetadataSlotPool(const CudaMetadataSlotPool&) = delete;
    CudaMetadataSlotPool& operator=(const CudaMetadataSlotPool&) = delete;

    [[nodiscard]] std::size_t acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        completion_.wait(lock, [this] {
            for (const Slot& slot : slots_) {
                if (!slot.in_use) {
                    return true;
                }
            }
            return false;
        });
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            if (!slots_[index].in_use) {
                slots_[index].in_use = true;
                return index;
            }
        }
        throw std::logic_error("metadata slot acquisition lost a free slot");
    }

    void release(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < slots_.size() && slots_[index].in_use) {
            slots_[index].in_use = false;
            completion_.notify_one();
        }
    }

    void ensure_slot_capacity(
            std::size_t index, std::size_t required_bytes,
            cudaStream_t stream) {
        Slot& slot = slots_.at(index);
        if (slot.capacity >= required_bytes) {
            return;
        }
        std::size_t capacity = slot.capacity == 0 ? 256 : slot.capacity;
        while (capacity < required_bytes) {
            if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
                capacity = required_bytes;
                break;
            }
            capacity *= 2;
        }
        gpu_policy::synchronize_stream(stream);
        std::unique_ptr<std::byte[]> replacement(
                new std::byte[capacity]);
        void* replacement_device = gpu_policy::allocate(capacity);
        void* old_device = slot.device;
        slot.host = std::move(replacement);
        slot.device = replacement_device;
        slot.capacity = capacity;
        gpu_policy::free_noexcept(old_device);
    }

    [[nodiscard]] std::byte* host_data(std::size_t index) {
        return slots_.at(index).host.get();
    }

    [[nodiscard]] void* device_data(std::size_t index) {
        return slots_.at(index).device;
    }

private:
    struct Slot {
        std::unique_ptr<std::byte[]> host;
        void* device = nullptr;
        std::size_t capacity = 0;
        bool in_use = false;
    };

    CUcontext context_;
    std::array<Slot, kMetadataSlotCount> slots_;
    std::mutex mutex_;
    std::condition_variable completion_;
};

void write_cuda_metadata(
        std::byte* storage, const TensorView& source,
        const TensorView& destination) {
    const std::span<const std::size_t> dimensions =
            source.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = checked_metadata_mul(
                plane_count, dimensions[axis], "metadata plane count overflows");
    }
    auto* header = reinterpret_cast<CudaCopyMetadataHeader*>(storage);
    header->source_plane_offset = metadata_u64(
            source.plane_offset(), "metadata source offset overflows");
    header->destination_plane_offset = metadata_u64(
            destination.plane_offset(), "metadata destination offset overflows");
    header->rows = metadata_u64(rows, "metadata rows overflows");
    header->columns = metadata_u64(columns, "metadata columns overflows");
    header->plane_count = metadata_u64(
            plane_count, "metadata plane count overflows");
    header->bits = static_cast<std::uint32_t>(
            detail::leaf_bits(source.spec().data_type));
    header->leading_rank = static_cast<std::uint32_t>(leading_rank);
    auto* values = reinterpret_cast<std::uint64_t*>(header + 1);
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        values[axis] = metadata_u64(
                source.plane_strides()[axis],
                "metadata source stride overflows");
        values[leading_rank + axis] = metadata_u64(
                destination.plane_strides()[axis],
                "metadata destination stride overflows");
        values[2 * leading_rank + axis] = metadata_u64(
                dimensions[axis], "metadata leading dimension overflows");
    }
}

struct CudaFenceResource {
    cudaEvent_t event = nullptr;
    std::size_t metadata_slot = 0;
    CudaMetadataSlotPool* pool = nullptr;
    CUcontext context = nullptr;
};

void destroy_cuda_resource_noexcept(CudaFenceResource* resource) noexcept {
    if (resource == nullptr) {
        return;
    }
    gpu_policy::destroy_event_noexcept(resource->event);
    resource->pool->release(resource->metadata_slot);
    delete resource;
}

void cuda_fence_complete(void* opaque) {
    auto* resource = static_cast<CudaFenceResource*>(opaque);
    gpu_policy::activate(resource->context);
    gpu_policy::synchronize_event(resource->event);
}

void cuda_fence_destroy(void* opaque) noexcept {
    auto* resource = static_cast<CudaFenceResource*>(opaque);
    if (resource == nullptr) {
        return;
    }
    try {
        gpu_policy::activate(resource->context);
    } catch (...) {
    }
    destroy_cuda_resource_noexcept(resource);
}

}  // namespace

namespace {

class CudaQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        const TensorView* source;
        TensorView* destination;
        cudaEvent_t event = nullptr;
        bool no_op;
        void* fence = nullptr;
    };

public:
    CudaQueue(const Device& device, CUcontext context)
            : device_(&device),
              context_(context),
              metadata_pool_(context_),
              worker_(
                      detail::StagedWorker<Task>::Callbacks{
                              [this](Task& task) {
                                  execute(task);
                              },
                              [](void* fence) {
                                  cuda_fence_complete(fence);
                              },
                              [](void* fence) {
                                  cuda_fence_destroy(fence);
                              },
                              [this](
                                      std::uint64_t sequence,
                                      std::exception_ptr failure) {
                                  complete(sequence, std::move(failure));
                              }},
                      detail::StagedWorker<Task>::PublishPolicy::Splice) {
        gpu_policy::activate(context_);
        try {
            stream_ = gpu_policy::create_queue_stream();
            worker_.start();
        } catch (...) {
            gpu_policy::destroy_queue_stream_noexcept(stream_);
            stream_ = gpu_policy::null_stream();
            throw;
        }
    }

    ~CudaQueue() override {
        worker_.shutdown_and_drain();
        try {
            gpu_policy::activate(context_);
            synchronize_and_destroy_stream(stream_);
        } catch (...) {
        }
        stream_ = gpu_policy::null_stream();
    }

    oid copy(const TensorView& source, TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        validate_copy(*device_, source, destination);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    worker_.submit_copy(
                            Task{sequence, &source, &destination, nullptr,
                                 no_op});
                });
    }

    oid add(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "add");
    }
    oid mul(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "mul");
    }
    oid silu(const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "silu");
    }
    oid linear(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "linear");
    }
    oid rmsnorm(const TensorView&, TensorView&, const TensorView&, float,
                size_t) override {
        throw unsupported("CUDA", "rmsnorm");
    }
    oid sdpa(const TensorView&, const TensorView&, const TensorView&,
             size_t, size_t, size_t, TensorView&) override {
        throw unsupported("CUDA", "sdpa");
    }

private:
    void execute(Task& task) {
        if (task.no_op) {
            task.event = nullptr;
            task.fence = nullptr;
            return;
        }

        gpu_policy::activate(context_);
        const std::size_t metadata_slot = metadata_pool_.acquire();
        cudaEvent_t event = nullptr;
        std::unique_ptr<CudaFenceResource> resource;
        bool kernel_enqueued = false;
        bool event_recorded = false;
        try {
            const CudaMetadataLayout layout =
                    metadata_layout(*task.source, *task.destination);
            metadata_pool_.ensure_slot_capacity(
                    metadata_slot, layout.bytes, stream_);
            write_cuda_metadata(
                    metadata_pool_.host_data(metadata_slot),
                    *task.source, *task.destination);
            gpu_policy::create_event(&event);
            resource = std::make_unique<CudaFenceResource>(
                    CudaFenceResource{
                            event, metadata_slot, &metadata_pool_, context_});
            gpu_policy::copy_from_host(
                    stream_, metadata_pool_.device_data(metadata_slot),
                    metadata_pool_.host_data(metadata_slot), layout.bytes);
            const std::size_t launch_words = checked_metadata_add(
                    layout.total_words, 255,
                    "metadata launch count overflows");
            const unsigned int blocks = static_cast<unsigned int>(
                    std::min<std::size_t>(launch_words / 256, 65535));
            grid_stride_copy_kernel<<<dim3(blocks), dim3(256), 0, stream_>>>(
                    static_cast<const unsigned char*>(
                            task.source->native_handle()),
                    static_cast<unsigned char*>(
                            task.destination->native_handle()),
                    static_cast<const CudaCopyMetadataHeader*>(
                            metadata_pool_.device_data(metadata_slot)));
            kernel_enqueued = true;
            gpu_policy::check_kernel(gpu_policy::copy_kernel_operation());
            gpu_policy::after_grid_stride_launch();
            gpu_policy::record_event(event, stream_);
            event_recorded = true;
            task.event = event;
            task.fence = resource.release();
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (kernel_enqueued) {
                if (!event_recorded && resource) {
                    gpu_policy::record_event_no_fault(event, stream_);
                    event_recorded = true;
                }
                if (event_recorded && resource) {
                    task.event = event;
                    task.fence = resource.release();
                    commit_failure(task.sequence, failure);
                    return;
                }
                commit_failure(task.sequence, failure);
                return;
            }
            if (resource) {
                resource->event = event;
                destroy_cuda_resource_noexcept(resource.release());
            } else {
                gpu_policy::destroy_event_noexcept(event);
                metadata_pool_.release(metadata_slot);
            }
            throw;
        }
    }

    const Device* device_;
    CUcontext context_;
    cudaStream_t stream_ = gpu_policy::null_stream();
    CudaMetadataSlotPool metadata_pool_;
    std::mutex submission_order_mutex_;
    detail::StagedWorker<Task> worker_;
};

}  // namespace


void inject_submission_fault_for_testing(
        SubmissionFault fault) noexcept {
    g_submission_fault.store(fault, std::memory_order_release);
}

void region_from_host(
        TransferStreamPool& transfer_pool, StagingSlotPool& staging_pool,
        CUcontext context, const TensorView& destination,
        std::span<const std::byte> source) {
    detail::synchronous_transfer<gpu_policy>(
            transfer_pool, staging_pool, context, destination, source, {},
            true);
}

void region_to_host(
        TransferStreamPool& transfer_pool, StagingSlotPool& staging_pool,
        CUcontext context, const TensorView& source,
        std::span<std::byte> destination) {
    detail::synchronous_transfer<gpu_policy>(
            transfer_pool, staging_pool, context, source, {}, destination,
            false);
}

std::unique_ptr<DeviceOps> make_queue(const Device& device, CUcontext context) {
    return std::make_unique<CudaQueue>(device, context);
}

}  // namespace iom::cuda_detail
