#include "copy.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace iom::rocm_detail {
namespace {

constexpr std::size_t kTile = TensorSpec::TILE;
constexpr std::size_t kThreads = 256;
constexpr std::size_t kLaunchChunk = 1u << 20;

[[nodiscard]] std::runtime_error hip_error(
        const char* operation, hipError_t status) {
    return std::runtime_error(
            std::string(operation) + " failed with "
            + hipGetErrorName(status) + ": " + hipGetErrorString(status));
}

void check_hip(const char* operation, hipError_t status) {
    if (status != hipSuccess) {
        throw hip_error(operation, status);
    }
}

struct DeviceGuard {
    explicit DeviceGuard(int device_ordinal) {
        check_hip("hipSetDevice", hipSetDevice(device_ordinal));
    }
};

struct PlanePair {
    std::size_t source;
    std::size_t destination;
};

[[nodiscard]] std::vector<PlanePair> plane_pairs(
        const TensorView& source, const TensorView& destination) {
    const std::span<const std::size_t> dimensions = source.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::size_t plane_elements = rows * columns;
    const std::size_t plane_count =
            source.spec().shape.element_count() / plane_elements;

    std::vector<PlanePair> pairs;
    pairs.reserve(plane_count);
    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        std::size_t source_rest = logical_plane;
        std::size_t destination_rest = logical_plane;
        std::size_t source_plane = source.plane_offset();
        std::size_t destination_plane = destination.plane_offset();
        for (std::size_t k = leading_rank; k-- > 0;) {
            const std::size_t source_coordinate = source_rest % dimensions[k];
            const std::size_t destination_coordinate =
                    destination_rest % dimensions[k];
            source_rest /= dimensions[k];
            destination_rest /= dimensions[k];
            source_plane += source_coordinate * source.plane_strides()[k];
            destination_plane +=
                    destination_coordinate * destination.plane_strides()[k];
        }
        pairs.push_back({source_plane, destination_plane});
    }
    return pairs;
}

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
        value |= static_cast<std::uint64_t>((base[bit / 8] >> (bit % 8)) & 1)
                << i;
    }
    return value;
}

__device__ void write_bits(
        unsigned char* base, std::uint64_t bit_offset, unsigned int bits,
        std::uint64_t value) {
    if (bits % 8 == 0) {
        unsigned char* destination = base + bit_offset / 8;
        for (unsigned int i = 0; i < bits / 8; ++i) {
            destination[i] = static_cast<unsigned char>(value >> (i * 8));
        }
        return;
    }
    for (unsigned int i = 0; i < bits; ++i) {
        const std::uint64_t bit = bit_offset + i;
        auto* word = reinterpret_cast<unsigned int*>(
                base + (bit / 32) * sizeof(unsigned int));
        const unsigned int mask =
                static_cast<unsigned int>(1u << (bit % 32));
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

__global__ void scatter_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t destination_plane, std::uint64_t logical_base,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= first + count) {
        return;
    }
    const std::uint64_t row = local / columns;
    const std::uint64_t column = local % columns;
    const std::uint64_t destination_bit =
            plane_slot(destination_plane, row, column, rows, columns) * bits;
    write_bits(destination, destination_bit, bits,
               read_bits(source, (logical_base + local) * bits, bits));
}

__global__ void gather_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t logical_base,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= first + count) {
        return;
    }
    const std::uint64_t row = local / columns;
    const std::uint64_t column = local % columns;
    const std::uint64_t source_bit =
            plane_slot(source_plane, row, column, rows, columns) * bits;
    write_bits(destination, (logical_base + local) * bits, bits,
               read_bits(source, source_bit, bits));
}

__global__ void copy_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t destination_plane,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= first + count) {
        return;
    }
    const std::uint64_t row = local / columns;
    const std::uint64_t column = local % columns;
    const std::uint64_t source_bit =
            plane_slot(source_plane, row, column, rows, columns) * bits;
    const std::uint64_t destination_bit =
            plane_slot(destination_plane, row, column, rows, columns) * bits;
    write_bits(destination, destination_bit, bits,
               read_bits(source, source_bit, bits));
}

void launch_view_transfer(
        hipStream_t stream, const TensorView& view, const void* source,
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
                hipLaunchKernelGGL(
                        scatter_plane_kernel, dim3(blocks), dim3(kThreads), 0,
                        stream, static_cast<const unsigned char*>(source),
                        static_cast<unsigned char*>(destination),
                        static_cast<std::uint64_t>(planes[logical_plane]),
                        static_cast<std::uint64_t>(logical_plane * elements),
                        static_cast<std::uint64_t>(first),
                        static_cast<std::uint64_t>(count),
                        static_cast<std::uint64_t>(rows),
                        static_cast<std::uint64_t>(columns), bits);
            } else {
                hipLaunchKernelGGL(
                        gather_plane_kernel, dim3(blocks), dim3(kThreads), 0,
                        stream, static_cast<const unsigned char*>(source),
                        static_cast<unsigned char*>(destination),
                        static_cast<std::uint64_t>(planes[logical_plane]),
                        static_cast<std::uint64_t>(logical_plane * elements),
                        static_cast<std::uint64_t>(first),
                        static_cast<std::uint64_t>(count),
                        static_cast<std::uint64_t>(rows),
                        static_cast<std::uint64_t>(columns), bits);
            }
            check_hip("HIP kernel launch", hipGetLastError());
        }
    }
}

void synchronous_transfer(
        int device_ordinal, const TensorView& view,
        std::span<const std::byte> source, std::span<std::byte> destination,
        bool from_host) {
    DeviceGuard guard(device_ordinal);
    void* staging = nullptr;
    hipStream_t stream = nullptr;
    try {
        check_hip("hipMalloc", hipMalloc(&staging, view.spec().logical_nbytes()));
        check_hip("hipStreamCreateWithFlags",
                  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        if (from_host) {
            check_hip("hipMemcpy HtoD",
                      hipMemcpy(staging, source.data(), source.size(),
                                hipMemcpyHostToDevice));
            launch_view_transfer(
                    stream, view, staging,
                    const_cast<void*>(view.native_handle()), true);
        } else {
            check_hip("hipMemset", hipMemset(
                                              staging,
                                              0,
                                              view.spec().logical_nbytes()));
            launch_view_transfer(stream, view, view.native_handle(), staging, false);
        }
        check_hip("hipStreamSynchronize", hipStreamSynchronize(stream));
        if (!from_host) {
            check_hip("hipMemcpy DtoH",
                      hipMemcpy(destination.data(), staging, destination.size(),
                                hipMemcpyDeviceToHost));
        }
    } catch (...) {
        if (stream != nullptr) {
            (void)hipStreamSynchronize(stream);
            (void)hipStreamDestroy(stream);
        }
        if (staging != nullptr) {
            (void)hipFree(staging);
        }
        throw;
    }
    check_hip("hipStreamDestroy", hipStreamDestroy(stream));
    check_hip("hipFree", hipFree(staging));
}

class RocmQueue final : public DeviceOps {
public:
    RocmQueue(const Device& device, int device_ordinal)
            : device_(&device), device_ordinal_(device_ordinal) {
        DeviceGuard guard(device_ordinal_);
        check_hip("hipStreamCreateWithFlags",
                  hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
        try {
            worker_ = std::thread([this] { run(); });
        } catch (...) {
            (void)hipStreamDestroy(stream_);
            stream_ = nullptr;
            throw;
        }
    }

    ~RocmQueue() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        completion_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
        try {
            DeviceGuard guard(device_ordinal_);
            std::lock_guard<std::mutex> lock(mutex_);
            while (!staged_.empty()) {
                (void)hipEventDestroy(staged_.front().event);
                staged_.pop_front();
            }
            while (!tasks_.empty()) {
                (void)hipEventDestroy(tasks_.front().event);
                tasks_.pop_front();
            }
            (void)hipStreamDestroy(stream_);
        } catch (...) {
            // Queue destruction is noexcept; callers must wait successful
            // submissions before destroying a queue.
        }
        stream_ = nullptr;
    }

    oid copy(const TensorView& source, TensorView& destination) override {
        validate_copy(source, destination);
        const bool no_op = identical_window(source, destination);
        const std::vector<PlanePair> pairs =
                no_op ? std::vector<PlanePair>{} : plane_pairs(source, destination);
        const std::span<const std::size_t> dimensions = source.spec().shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t columns = dimensions[dimensions.size() - 1];
        const unsigned int bits = static_cast<unsigned int>(
                detail::leaf_bits(source.spec().data_type));
        const void* source_handle = source.native_handle();
        void* destination_handle = destination.native_handle();
        const oid token = submit([&](std::uint64_t sequence) {
            DeviceGuard guard(device_ordinal_);
            hipEvent_t event = nullptr;
            check_hip("hipEventCreateWithFlags",
                      hipEventCreateWithFlags(&event, hipEventDisableTiming));
            try {
                for (const PlanePair pair : pairs) {
                    launch_copy_plane(
                            pair, rows, columns, bits,
                            source_handle, destination_handle);
                }
                check_hip("hipEventRecord", hipEventRecord(event, stream_));
                std::lock_guard<std::mutex> lock(mutex_);
                staged_.push_back({sequence, event});
            } catch (...) {
                (void)hipEventDestroy(event);
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
        hipEvent_t event;
    };

    static std::runtime_error unsupported(const char* operation) {
        return std::runtime_error(
                std::string("ROCm backend does not implement ") + operation);
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

    void launch_copy_plane(
            const PlanePair& pair, std::size_t rows, std::size_t columns,
            unsigned int bits, const void* source, void* destination) {
        const std::size_t elements = rows * columns;
        for (std::size_t first = 0; first < elements; first += kLaunchChunk) {
            const std::size_t count = std::min(kLaunchChunk, elements - first);
            const unsigned int blocks = static_cast<unsigned int>(
                    (count + kThreads - 1) / kThreads);
            hipLaunchKernelGGL(
                    copy_plane_kernel, dim3(blocks), dim3(kThreads), 0, stream_,
                    static_cast<const unsigned char*>(source),
                    static_cast<unsigned char*>(destination),
                    static_cast<std::uint64_t>(pair.source),
                    static_cast<std::uint64_t>(pair.destination),
                    static_cast<std::uint64_t>(first),
                    static_cast<std::uint64_t>(count),
                    static_cast<std::uint64_t>(rows),
                    static_cast<std::uint64_t>(columns), bits);
            check_hip("HIP kernel launch", hipGetLastError());
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
            DeviceGuard guard(device_ordinal_);
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
                    check_hip("hipEventSynchronize",
                              hipEventSynchronize(task.event));
                } catch (...) {
                    failure = std::current_exception();
                }
                (void)hipEventDestroy(task.event);
                complete(task.sequence, failure);

                lock.lock();
            }
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            std::lock_guard<std::mutex> lock(mutex_);
            while (!tasks_.empty()) {
                const Task task = tasks_.front();
                tasks_.pop_front();
                (void)hipEventDestroy(task.event);
                complete(task.sequence, failure);
            }
        }
    }

    const Device* device_;
    int device_ordinal_;
    hipStream_t stream_ = nullptr;
    std::mutex mutex_;
    std::condition_variable completion_;
    std::deque<Task> staged_;
    std::deque<Task> tasks_;
    bool shutdown_ = false;
    std::thread worker_;
};

}  // namespace

void region_from_host(
        int device_ordinal, const TensorView& destination,
        std::span<const std::byte> source) {
    synchronous_transfer(device_ordinal, destination, source, {}, true);
}

void region_to_host(
        int device_ordinal, const TensorView& source,
        std::span<std::byte> destination) {
    synchronous_transfer(device_ordinal, source, {}, destination, false);
}

std::unique_ptr<DeviceOps> make_queue(const Device& device, int device_ordinal) {
    return std::make_unique<RocmQueue>(device, device_ordinal);
}

}  // namespace iom::rocm_detail
