#include "copy.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace iom::sycl_detail {
namespace {

constexpr std::size_t kTile = TensorSpec::TILE;
constexpr std::size_t kTileSlots = kTile * kTile;

struct PlanePair {
    std::size_t source;
    std::size_t destination;
};

std::vector<std::size_t> view_planes(const TensorView& view) {
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::size_t plane_elements = rows * columns;
    const std::size_t plane_count =
            view.spec().shape.element_count() / plane_elements;

    std::vector<std::size_t> planes;
    planes.reserve(plane_count);
    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        std::size_t rest = logical_plane;
        std::size_t plane = view.plane_offset();
        for (std::size_t axis = leading_rank; axis-- > 0;) {
            plane += (rest % dimensions[axis])
                    * view.plane_strides()[axis];
            rest /= dimensions[axis];
        }
        planes.push_back(plane);
    }
    return planes;
}

std::vector<PlanePair> plane_pairs(
        const TensorView& source, const TensorView& destination) {
    const std::vector<std::size_t> source_planes = view_planes(source);
    const std::vector<std::size_t> destination_planes =
            view_planes(destination);
    std::vector<PlanePair> pairs;
    pairs.reserve(source_planes.size());
    for (std::size_t i = 0; i < source_planes.size(); ++i) {
        pairs.push_back({source_planes[i], destination_planes[i]});
    }
    return pairs;
}


std::uint64_t device_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_rows =
            (rows + kTile - 1) / kTile;
    const std::uint64_t tile_columns =
            (columns + kTile - 1) / kTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kTile) * tile_columns + column / kTile;
    return tile_index * kTileSlots
            + (row % kTile) * kTile + column % kTile;
}

std::uint64_t device_read_bits(
        const unsigned char* base, std::uint64_t bit_offset,
        unsigned int nbits) {
    std::uint64_t value = 0;
    for (unsigned int i = 0; i < nbits; ++i) {
        const std::uint64_t bit = bit_offset + i;
        value |= static_cast<std::uint64_t>(
                         (base[bit / 8] >> (bit % 8)) & 1u)
                << i;
    }
    return value;
}

void device_write_bits(
        unsigned char* base, std::uint64_t bit_offset,
        unsigned int nbits, std::uint64_t value) {
    if (nbits % 8 == 0) {
        auto* destination = base + bit_offset / 8;
        for (unsigned int i = 0; i < nbits / 8; ++i) {
            destination[i] = static_cast<unsigned char>(value >> (i * 8));
        }
        return;
    }

    for (unsigned int i = 0; i < nbits; ++i) {
        const std::uint64_t bit = bit_offset + i;
        auto* word = reinterpret_cast<std::uint32_t*>(
                base + (bit / 32) * sizeof(std::uint32_t));
        const std::uint32_t mask =
                static_cast<std::uint32_t>(1u << (bit % 32));
        sycl::atomic_ref<
                std::uint32_t, sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                atomic_word(*word);
        if ((value >> i) & 1u) {
            atomic_word.fetch_or(mask);
        } else {
            atomic_word.fetch_and(~mask);
        }
    }
}

void launch_view_transfer(
        sycl::queue& queue, const TensorView& view,
        const void* source, void* destination, bool from_host) {
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::size_t rows = dimensions[dimensions.size() - 2];
    const std::size_t columns = dimensions[dimensions.size() - 1];
    const std::size_t elements = rows * columns;
    const unsigned int bits = static_cast<unsigned int>(
            detail::leaf_bits(view.spec().data_type));
    const std::vector<std::size_t> planes = view_planes(view);

    for (std::size_t logical_plane = 0; logical_plane < planes.size();
         ++logical_plane) {
        const std::uint64_t plane = planes[logical_plane];
        const std::uint64_t logical_base =
                static_cast<std::uint64_t>(logical_plane * elements);
        if (from_host) {
            queue.submit([=](sycl::handler& handler) {
                handler.parallel_for(
                        sycl::range<1>(elements),
                        [=](sycl::id<1> index) {
                            const std::uint64_t local = index[0];
                            const std::uint64_t row = local / columns;
                            const std::uint64_t column = local % columns;
                            const std::uint64_t destination_bit =
                                    device_plane_slot(
                                            plane, row, column, rows,
                                            columns)
                                    * bits;
                            device_write_bits(
                                    static_cast<unsigned char*>(destination),
                                    destination_bit, bits,
                                    device_read_bits(
                                            static_cast<const unsigned char*>(
                                                    source),
                                            (logical_base + local) * bits,
                                            bits));
                        });
            });
        } else {
            queue.submit([=](sycl::handler& handler) {
                handler.parallel_for(
                        sycl::range<1>(elements),
                        [=](sycl::id<1> index) {
                            const std::uint64_t local = index[0];
                            const std::uint64_t row = local / columns;
                            const std::uint64_t column = local % columns;
                            const std::uint64_t source_bit =
                                    device_plane_slot(
                                            plane, row, column, rows,
                                            columns)
                                    * bits;
                            device_write_bits(
                                    static_cast<unsigned char*>(destination),
                                    (logical_base + local) * bits, bits,
                                    device_read_bits(
                                            static_cast<const unsigned char*>(
                                                    source),
                                            source_bit, bits));
                        });
            });
        }
    }
}

void synchronous_transfer(
        const sycl::context& context, const sycl::device& device,
        const TensorView& view, const void* storage,
        std::span<const std::byte> source,
        std::span<std::byte> destination, bool from_host) {
    sycl::queue queue(
            context, device,
            sycl::property_list{sycl::property::queue::in_order{}});
    const std::size_t logical_nbytes = view.spec().logical_nbytes();
    constexpr std::size_t kWordBytes = sizeof(std::uint32_t);
    if (logical_nbytes
            > std::numeric_limits<std::size_t>::max() - (kWordBytes - 1)) {
        throw std::overflow_error("SYCL transfer staging size overflows");
    }
    const std::size_t staging_nbytes =
            (logical_nbytes + (kWordBytes - 1)) / kWordBytes * kWordBytes;
    void* staging = sycl::malloc_shared(staging_nbytes, device, context);
    if (staging == nullptr) {
        throw std::bad_alloc();
    }

    try {
        if (from_host) {
            queue.memcpy(staging, source.data(), logical_nbytes);
            launch_view_transfer(
                    queue, view, staging, const_cast<void*>(storage), true);
            queue.wait_and_throw();
        } else {
            std::memset(staging, 0, logical_nbytes);
            launch_view_transfer(
                    queue, view, storage, staging, false);
            queue.wait_and_throw();
            queue.memcpy(destination.data(), staging, logical_nbytes)
                    .wait_and_throw();
        }
    } catch (...) {
        try {
            queue.wait_and_throw();
        } catch (...) {
        }
        try {
            sycl::free(staging, context);
        } catch (...) {
        }
        throw;
    }

    sycl::free(staging, context);
}

class SyclQueue final : public DeviceOps {
public:
    SyclQueue(
            const Device& device, const sycl::context& context,
            const sycl::device& native_device)
            : device_(&device),
              queue_(
                      context, native_device,
                      sycl::property_list{
                              sycl::property::queue::in_order{}}),
              worker_([this] { run(); }) {}

    ~SyclQueue() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        completion_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    oid copy(
            const TensorView& source,
            TensorView& destination) override {
        validate_copy(*device_, source, destination);
        const bool no_op = identical_window(source, destination);
        std::vector<PlanePair> pairs;
        if (!no_op) {
            pairs = plane_pairs(source, destination);
        }

        const std::span<const std::size_t> dimensions =
                source.spec().shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t columns = dimensions[dimensions.size() - 1];
        const unsigned int bits = static_cast<unsigned int>(
                detail::leaf_bits(source.spec().data_type));
        const void* source_handle = source.native_handle();
        void* destination_handle = destination.native_handle();

        const oid token = submit(
                [this, no_op, pairs = std::move(pairs), source_handle,
                 destination_handle, rows, columns, bits](
                        std::uint64_t sequence) mutable {
                    if (no_op) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        staged_.push_back(Task{sequence, std::nullopt});
                        return;
                    }

                    std::optional<sycl::event> last_event;
                    try {
                        for (const PlanePair pair : pairs) {
                            last_event = queue_.submit([=](sycl::handler& handler) {
                                handler.parallel_for(
                                        sycl::range<1>(rows * columns),
                                        [=](sycl::id<1> index) {
                                            const std::uint64_t local = index[0];
                                            const std::uint64_t row = local / columns;
                                            const std::uint64_t column = local % columns;
                                            const std::uint64_t source_bit =
                                                    device_plane_slot(
                                                            pair.source, row,
                                                            column, rows,
                                                            columns)
                                                    * bits;
                                            const std::uint64_t destination_bit =
                                                    device_plane_slot(
                                                            pair.destination,
                                                            row, column, rows,
                                                            columns)
                                                    * bits;
                                            device_write_bits(
                                                    static_cast<unsigned char*>(
                                                            destination_handle),
                                                    destination_bit, bits,
                                                    device_read_bits(
                                                            static_cast<const unsigned char*>(
                                                                    source_handle),
                                                            source_bit, bits));
                                        });
                            });
                        }
                    } catch (...) {
                        const std::exception_ptr failure =
                                std::current_exception();
                        if (!last_event.has_value()) {
                            throw;
                        }
                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            staged_.push_back(
                                    Task{sequence, std::move(last_event)});
                        }
                        commit_failure(sequence, failure);
                        return;
                    }

                    std::lock_guard<std::mutex> lock(mutex_);
                    staged_.push_back(Task{sequence, std::move(last_event)});
                });
        publish_staged();
        return token;
    }

    oid add(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("SYCL", "add");
    }

    oid mul(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("SYCL", "mul");
    }

    oid silu(const TensorView&, TensorView&) override {
        throw unsupported("SYCL", "silu");
    }

    oid linear(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("SYCL", "linear");
    }

    oid rmsnorm(
            const TensorView&, TensorView&, const TensorView&, float,
            size_t) override {
        throw unsupported("SYCL", "rmsnorm");
    }

    oid sdpa(
            const TensorView&, const TensorView&, const TensorView&, size_t,
            size_t, size_t, TensorView&) override {
        throw unsupported("SYCL", "sdpa");
    }

private:
    struct Task {
        std::uint64_t sequence;
        std::optional<sycl::event> event;
    };


    void publish_staged() {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.insert(
                tasks_.end(),
                std::make_move_iterator(staged_.begin()),
                std::make_move_iterator(staged_.end()));
        staged_.clear();
        completion_.notify_one();
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            completion_.wait(lock, [this] {
                return shutdown_ || !tasks_.empty();
            });
            if (shutdown_) {
                return;
            }
            Task task = std::move(tasks_.front());
            tasks_.pop_front();
            lock.unlock();

            std::exception_ptr failure;
            try {
                if (task.event.has_value()) {
                    task.event->wait_and_throw();
                }
            } catch (...) {
                failure = std::current_exception();
            }
            complete(task.sequence, failure);

            lock.lock();
        }
    }

    const Device* device_;
    sycl::queue queue_;
    std::mutex mutex_;
    std::condition_variable completion_;
    std::deque<Task> staged_;
    std::deque<Task> tasks_;
    bool shutdown_ = false;
    std::thread worker_;
};

}  // namespace

void region_from_host(
        const sycl::context& context, const sycl::device& device,
        const TensorView& destination, void* storage,
        std::span<const std::byte> source) {
    synchronous_transfer(
            context, device, destination, storage, source, {}, true);
}

void region_to_host(
        const sycl::context& context, const sycl::device& device,
        const TensorView& source, const void* storage,
        std::span<std::byte> destination) {
    synchronous_transfer(
            context, device, source, storage, {}, destination, false);
}

std::unique_ptr<DeviceOps> make_queue(
        const Device& device, const sycl::context& context,
        const sycl::device& native_device) {
    return std::make_unique<SyclQueue>(device, context, native_device);
}

}  // namespace iom::sycl_detail
