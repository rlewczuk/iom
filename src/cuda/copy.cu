#include "copy.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <limits>
#include <vector>

namespace iom::cuda_detail {
namespace {

constexpr std::size_t kTile = TensorSpec::TILE;
constexpr unsigned int kThreads = 256;
constexpr std::size_t kLaunchChunk = 1u << 20;

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

void check_kernel(const char* operation, cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
                std::string(operation) + " failed with "
                + cudaGetErrorName(status) + ": "
                + cudaGetErrorString(status));
    }
}

struct ContextGuard {
    explicit ContextGuard(CUcontext context) {
        check_cuda("cuCtxSetCurrent", cuCtxSetCurrent(context));
    }
};

struct PlanePair {
    std::size_t source;
    std::size_t destination;
};

[[nodiscard]] std::vector<std::size_t> view_planes(const TensorView& view) {
    const std::span<const std::size_t> dimensions = view.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::size_t plane_count =
            view.spec().shape.element_count() / (rows * columns);

    std::vector<std::size_t> planes;
    planes.reserve(plane_count);
    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        std::size_t rest = logical_plane;
        std::size_t plane = view.plane_offset();
        for (std::size_t k = leading_rank; k-- > 0;) {
            const std::size_t coordinate = rest % dimensions[k];
            rest /= dimensions[k];
            plane += coordinate * view.plane_strides()[k];
        }
        planes.push_back(plane);
    }
    return planes;
}

[[nodiscard]] std::vector<PlanePair> plane_pairs(
        const TensorView& source, const TensorView& destination) {
    const std::vector<std::size_t> source_planes = view_planes(source);
    const std::vector<std::size_t> destination_planes = view_planes(destination);
    std::vector<PlanePair> pairs;
    pairs.reserve(source_planes.size());
    for (std::size_t i = 0; i < source_planes.size(); ++i) {
        pairs.push_back({source_planes[i], destination_planes[i]});
    }
    return pairs;
}

[[nodiscard]] bool identical_window(
        const TensorView& source, const TensorView& destination) {
    return source.native_handle() == destination.native_handle()
            && source.plane_offset() == destination.plane_offset()
            && std::equal(
                    source.plane_strides().begin(), source.plane_strides().end(),
                    destination.plane_strides().begin(),
                    destination.plane_strides().end());
}

__device__ std::uint64_t read_bits(
        const unsigned char* base, std::uint64_t bit_offset,
        unsigned int bits) {
    std::uint64_t value = 0;
    for (unsigned int i = 0; i < bits; ++i) {
        const std::uint64_t bit = bit_offset + i;
        value |= static_cast<std::uint64_t>(
                         (base[bit / 8] >> (bit % 8)) & 1)
                << i;
    }
    return value;
}

__device__ void write_bits_atomic(
        unsigned char* base, std::uint64_t bit_offset, unsigned int bits,
        std::uint64_t value) {
    for (unsigned int i = 0; i < bits; ++i) {
        const std::uint64_t bit = bit_offset + i;
        auto* word = reinterpret_cast<unsigned int*>(
                base + (bit / 32) * sizeof(unsigned int));
        const unsigned int mask = 1u << (bit % 32);
        if ((value >> i) & 1) {
            atomicOr(word, mask);
        } else {
            atomicAnd(word, ~mask);
        }
    }
}

__device__ std::uint64_t plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_rows = (rows + kTile - 1) / kTile;
    const std::uint64_t tile_columns = (columns + kTile - 1) / kTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kTile) * tile_columns + column / kTile;
    return tile_index * kTile * kTile
            + (row % kTile) * kTile + column % kTile;
}

__device__ void copy_value(
        unsigned char* destination, std::uint64_t destination_bit,
        const unsigned char* source, std::uint64_t source_bit,
        unsigned int bits) {
    if (bits % 8 == 0) {
        const std::uint64_t destination_byte = destination_bit / 8;
        const std::uint64_t source_byte = source_bit / 8;
        for (unsigned int i = 0; i < bits / 8; ++i) {
            destination[destination_byte + i] = source[source_byte + i];
        }
        return;
    }
    write_bits_atomic(
            destination, destination_bit, bits,
            read_bits(source, source_bit, bits));
}

__global__ void scatter_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t destination_plane, std::uint64_t logical_base,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local =
            first + blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= first + count) {
        return;
    }
    const std::uint64_t row = local / columns;
    const std::uint64_t column = local % columns;
    const std::uint64_t destination_bit =
            plane_slot(destination_plane, row, column, rows, columns) * bits;
    copy_value(
            destination, destination_bit, source,
            (logical_base + local) * bits, bits);
}

__global__ void gather_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t logical_base,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local =
            first + blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= first + count) {
        return;
    }
    const std::uint64_t row = local / columns;
    const std::uint64_t column = local % columns;
    const std::uint64_t source_bit =
            plane_slot(source_plane, row, column, rows, columns) * bits;
    copy_value(
            destination, (logical_base + local) * bits, source, source_bit,
            bits);
}

__global__ void copy_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t destination_plane,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local =
            first + blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= first + count) {
        return;
    }
    const std::uint64_t row = local / columns;
    const std::uint64_t column = local % columns;
    const std::uint64_t source_bit =
            plane_slot(source_plane, row, column, rows, columns) * bits;
    const std::uint64_t destination_bit =
            plane_slot(destination_plane, row, column, rows, columns) * bits;
    copy_value(
            destination, destination_bit, source, source_bit, bits);
}

void launch_view_transfer(
        cudaStream_t stream, const TensorView& view, const void* source,
        void* destination, bool from_host) {
    const std::span<const std::size_t> dimensions = view.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::size_t plane_count =
            view.spec().shape.element_count() / (rows * columns);
    const unsigned int bits = static_cast<unsigned int>(
            detail::leaf_bits(view.spec().data_type));
    const std::vector<std::size_t> planes = view_planes(view);
    const std::size_t elements = rows * columns;
    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        for (std::size_t first = 0; first < elements; first += kLaunchChunk) {
            const std::size_t count = std::min(kLaunchChunk, elements - first);
            const unsigned int blocks = static_cast<unsigned int>(
                    (count + kThreads - 1) / kThreads);
            if (from_host) {
                const unsigned char* source_arg =
                        static_cast<const unsigned char*>(source);
                unsigned char* destination_arg =
                        static_cast<unsigned char*>(destination);
                const std::uint64_t plane_arg = planes[logical_plane];
                const std::uint64_t logical_base_arg =
                        logical_plane * elements;
                const std::uint64_t first_arg = first;
                const std::uint64_t count_arg = count;
                const std::uint64_t rows_arg = rows;
                const std::uint64_t columns_arg = columns;
                const unsigned int bits_arg = bits;
                scatter_plane_kernel<<<
                        dim3(blocks), dim3(kThreads), 0, stream>>>(
                        source_arg, destination_arg, plane_arg,
                        logical_base_arg, first_arg, count_arg, rows_arg,
                        columns_arg, bits_arg);
                check_kernel(
                        "CUDA scatter kernel launch", cudaGetLastError());
            } else {
                const unsigned char* source_arg =
                        static_cast<const unsigned char*>(source);
                unsigned char* destination_arg =
                        static_cast<unsigned char*>(destination);
                const std::uint64_t plane_arg = planes[logical_plane];
                const std::uint64_t logical_base_arg =
                        logical_plane * elements;
                const std::uint64_t first_arg = first;
                const std::uint64_t count_arg = count;
                const std::uint64_t rows_arg = rows;
                const std::uint64_t columns_arg = columns;
                const unsigned int bits_arg = bits;
                gather_plane_kernel<<<
                        dim3(blocks), dim3(kThreads), 0, stream>>>(
                        source_arg, destination_arg, plane_arg,
                        logical_base_arg, first_arg, count_arg, rows_arg,
                        columns_arg, bits_arg);
                check_kernel(
                        "CUDA gather kernel launch", cudaGetLastError());
            }
        }
    }
}

void launch_copy_plane(
        cudaStream_t stream, const PlanePair& pair, std::size_t rows,
        std::size_t columns, unsigned int bits, const void* source,
        void* destination) {
    const std::size_t elements = rows * columns;
    for (std::size_t first = 0; first < elements; first += kLaunchChunk) {
        const std::size_t count = std::min(kLaunchChunk, elements - first);
        const unsigned int blocks = static_cast<unsigned int>(
                (count + kThreads - 1) / kThreads);
        const unsigned char* source_arg =
                static_cast<const unsigned char*>(source);
        unsigned char* destination_arg =
                static_cast<unsigned char*>(destination);
        const std::uint64_t source_plane_arg = pair.source;
        const std::uint64_t destination_plane_arg = pair.destination;
        const std::uint64_t first_arg = first;
        const std::uint64_t count_arg = count;
        const std::uint64_t rows_arg = rows;
        const std::uint64_t columns_arg = columns;
        const unsigned int bits_arg = bits;
        copy_plane_kernel<<<
                dim3(blocks), dim3(kThreads), 0, stream>>>(
                source_arg, destination_arg, source_plane_arg,
                destination_plane_arg, first_arg, count_arg, rows_arg,
                columns_arg, bits_arg);
        check_kernel("CUDA copy kernel launch", cudaGetLastError());
    }
}

void synchronous_transfer(
        CUcontext context, const TensorView& view,
        std::span<const std::byte> source, std::span<std::byte> destination,
        bool from_host) {
    ContextGuard guard(context);
    const std::size_t logical_nbytes = view.spec().logical_nbytes();
    if (logical_nbytes
            > std::numeric_limits<std::size_t>::max()
                    - sizeof(unsigned int)) {
        throw std::overflow_error("CUDA transfer staging size overflows");
    }
    const std::size_t staging_nbytes =
            logical_nbytes + sizeof(unsigned int);
    CUdeviceptr staging = 0;
    try {
        check_cuda("cuMemAlloc", cuMemAlloc(&staging, staging_nbytes));
        if (from_host) {
            check_kernel(
                    "cudaMemcpy HtoD",
                    cudaMemcpy(
                            reinterpret_cast<void*>(staging), source.data(),
                            source.size(), cudaMemcpyHostToDevice));
            launch_view_transfer(
                    nullptr, view, reinterpret_cast<const void*>(staging),
                    const_cast<void*>(view.native_handle()), true);
        } else {
            check_kernel(
                    "cudaMemset",
                    cudaMemset(
                            reinterpret_cast<void*>(staging), 0,
                            staging_nbytes));
            launch_view_transfer(
                    nullptr, view, view.native_handle(),
                    reinterpret_cast<void*>(staging), false);
        }
        check_kernel(
                "cudaStreamSynchronize", cudaStreamSynchronize(nullptr));
        if (!from_host) {
            check_kernel(
                    "cudaMemcpy DtoH",
                    cudaMemcpy(
                            destination.data(), reinterpret_cast<void*>(staging),
                            destination.size(), cudaMemcpyDeviceToHost));
        }
    } catch (...) {
        if (staging != 0) {
            (void)cuMemFree(staging);
        }
        throw;
    }
    check_cuda("cuMemFree", cuMemFree(staging));
}

class CudaQueue final : public DeviceOps {
public:
    CudaQueue(const Device& device, CUcontext context)
            : device_(&device), context_(context) {
        ContextGuard guard(context_);
        check_kernel(
                "cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        try {
            worker_ = std::thread([this] { run(); });
        } catch (...) {
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
            throw;
        }
    }

    ~CudaQueue() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        completion_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
        try {
            ContextGuard guard(context_);
            std::lock_guard<std::mutex> lock(mutex_);
            while (!staged_.empty()) {
                (void)cudaEventDestroy(staged_.front().event);
                staged_.pop_front();
            }
            while (!tasks_.empty()) {
                (void)cudaEventDestroy(tasks_.front().event);
                tasks_.pop_front();
            }
            (void)cudaStreamDestroy(stream_);
        } catch (...) {
        }
        stream_ = nullptr;
    }

    oid copy(const TensorView& source, TensorView& destination) override {
        validate_copy(source, destination);
        const bool no_op = identical_window(source, destination);
        const std::vector<PlanePair> pairs =
                no_op ? std::vector<PlanePair>{}
                      : plane_pairs(source, destination);
        const std::span<const std::size_t> dimensions = source.spec().shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t columns = dimensions[dimensions.size() - 1];
        const unsigned int bits = static_cast<unsigned int>(
                detail::leaf_bits(source.spec().data_type));
        const void* source_handle = source.native_handle();
        void* destination_handle = destination.native_handle();
        const oid token = submit([&](std::uint64_t sequence) {
            ContextGuard guard(context_);
            cudaEvent_t event = nullptr;
            check_kernel(
                    "cudaEventCreateWithFlags",
                    cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            try {
                for (const PlanePair pair : pairs) {
                    launch_copy_plane(
                            stream_, pair, rows, columns, bits,
                            source_handle, destination_handle);
                }
                check_kernel("cudaEventRecord", cudaEventRecord(event, stream_));
                std::lock_guard<std::mutex> lock(mutex_);
                staged_.push_back({sequence, event});
            } catch (...) {
                (void)cudaEventDestroy(event);
                throw;
            }
        });
        publish_staged();
        return token;
    }

    oid add(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("add");
    }
    oid mul(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("mul");
    }
    oid silu(const TensorView&, TensorView&) override {
        throw unsupported("silu");
    }
    oid linear(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("linear");
    }
    oid rmsnorm(const TensorView&, TensorView&, const TensorView&, float, size_t) override {
        throw unsupported("rmsnorm");
    }
    oid sdpa(const TensorView&, const TensorView&, const TensorView&,
             size_t, size_t, size_t, TensorView&) override {
        throw unsupported("sdpa");
    }

private:
    struct Task {
        std::uint64_t sequence;
        cudaEvent_t event;
    };

    static std::runtime_error unsupported(const char* operation) {
        return std::runtime_error(
                std::string("CUDA backend does not implement ") + operation);
    }

    void validate_copy(
            const TensorView& source, const TensorView& destination) const {
        if (&source.device() != device_ || &destination.device() != device_) {
            throw std::invalid_argument(
                    "copy views must belong to the queue's own device");
        }
        if (!(source.spec() == destination.spec())) {
            throw std::invalid_argument(
                    "copy views must have identical shape, leaf type, and quantization");
        }
    }

    void publish_staged() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!staged_.empty()) {
            tasks_.push_back(std::move(staged_.front()));
            staged_.pop_front();
        }
        completion_.notify_one();
    }

    void run() {
        try {
            ContextGuard guard(context_);
            std::unique_lock<std::mutex> lock(mutex_);
            for (;;) {
                completion_.wait(lock, [this] {
                    return shutdown_ || !tasks_.empty();
                });
                if (shutdown_) {
                    return;
                }
                const Task task = tasks_.front();
                tasks_.pop_front();
                lock.unlock();

                std::exception_ptr failure;
                try {
                    check_kernel(
                            "cudaEventSynchronize",
                            cudaEventSynchronize(task.event));
                } catch (...) {
                    failure = std::current_exception();
                }
                (void)cudaEventDestroy(task.event);
                complete(task.sequence, failure);

                lock.lock();
            }
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            std::lock_guard<std::mutex> lock(mutex_);
            while (!tasks_.empty()) {
                const Task task = tasks_.front();
                tasks_.pop_front();
                (void)cudaEventDestroy(task.event);
                complete(task.sequence, failure);
            }
        }
    }

    const Device* device_;
    CUcontext context_;
    cudaStream_t stream_ = nullptr;
    std::mutex mutex_;
    std::condition_variable completion_;
    std::deque<Task> staged_;
    std::deque<Task> tasks_;
    bool shutdown_ = false;
    std::thread worker_;
};

}  // namespace

void region_from_host(
        CUcontext context, const TensorView& destination,
        std::span<const std::byte> source) {
    synchronous_transfer(context, destination, source, {}, true);
}

void region_to_host(
        CUcontext context, const TensorView& source,
        std::span<std::byte> destination) {
    synchronous_transfer(context, source, {}, destination, false);
}

std::unique_ptr<DeviceOps> make_queue(const Device& device, CUcontext context) {
    return std::make_unique<CudaQueue>(device, context);
}

}  // namespace iom::cuda_detail
