#include "standard_tiled_copy.hpp"

#include <algorithm>
#include <vector>

#include "iom/gpu_algorithm.hpp"

#ifndef IOM_GPU_DEVICE
#error "IOM_GPU_DEVICE must be defined before including standard_tiled_copy.inl"
#endif
#ifndef IOM_GPU_GLOBAL
#error "IOM_GPU_GLOBAL must be defined before including standard_tiled_copy.inl"
#endif
#ifndef IOM_GPU_GLOBAL_INDEX
#error "IOM_GPU_GLOBAL_INDEX must be defined before including standard_tiled_copy.inl"
#endif
#ifndef IOM_GPU_ATOMIC_OR
#error "IOM_GPU_ATOMIC_OR must be defined before including standard_tiled_copy.inl"
#endif
#ifndef IOM_GPU_ATOMIC_AND
#error "IOM_GPU_ATOMIC_AND must be defined before including standard_tiled_copy.inl"
#endif
#ifndef IOM_LAUNCH_KERNEL
#error "IOM_LAUNCH_KERNEL must be defined before including standard_tiled_copy.inl"
#endif

namespace iom::detail {
namespace {

constexpr std::size_t kTile = TensorSpec::TILE;
constexpr unsigned int kThreads = 256;
constexpr std::size_t kLaunchChunk = 1u << 20;

IOM_GPU_DEVICE std::uint64_t read_bits(
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

IOM_GPU_DEVICE void write_bits(
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
        const unsigned int mask = 1u << (bit % 32);
        if ((value >> i) & 1) {
            IOM_GPU_ATOMIC_OR(word, mask);
        } else {
            IOM_GPU_ATOMIC_AND(word, ~mask);
        }
    }
}

IOM_GPU_DEVICE std::uint64_t plane_slot(
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

IOM_GPU_DEVICE void copy_value(
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
    write_bits(
            destination, destination_bit, bits,
            read_bits(source, source_bit, bits));
}

IOM_GPU_GLOBAL void scatter_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t destination_plane, std::uint64_t logical_base,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local = first + IOM_GPU_GLOBAL_INDEX;
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

IOM_GPU_GLOBAL void gather_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t logical_base,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local = first + IOM_GPU_GLOBAL_INDEX;
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

IOM_GPU_GLOBAL void copy_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t destination_plane,
        std::uint64_t first, std::uint64_t count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t local = first + IOM_GPU_GLOBAL_INDEX;
    if (local >= first + count) {
        return;
    }
    const std::uint64_t row = local / columns;
    const std::uint64_t column = local % columns;
    const std::uint64_t source_bit =
            plane_slot(source_plane, row, column, rows, columns) * bits;
    const std::uint64_t destination_bit =
            plane_slot(destination_plane, row, column, rows, columns) * bits;
    copy_value(destination, destination_bit, source, source_bit, bits);
}

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
    const std::span<const std::size_t> dimensions =
            source.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::size_t plane_count =
            source.spec().shape.element_count() / (rows * columns);

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

template <typename Policy>
void launch_view_transfer(
        typename Policy::stream_type stream, const TensorView& view,
        const void* source, void* destination, bool from_host) {
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::size_t plane_count =
            view.spec().shape.element_count() / (rows * columns);
    const unsigned int bits =
            static_cast<unsigned int>(leaf_bits(view.spec().data_type));
    const std::vector<std::size_t> planes = view_planes(view);
    const std::size_t elements = rows * columns;
    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        for (std::size_t first = 0; first < elements; first += kLaunchChunk) {
            const std::size_t count = std::min(kLaunchChunk, elements - first);
            const unsigned int blocks = static_cast<unsigned int>(
                    (count + kThreads - 1) / kThreads);
            if (from_host) {
                IOM_LAUNCH_KERNEL(
                        scatter_plane_kernel, blocks, kThreads, stream,
                        static_cast<const unsigned char*>(source),
                        static_cast<unsigned char*>(destination),
                        static_cast<std::uint64_t>(planes[logical_plane]),
                        static_cast<std::uint64_t>(logical_plane * elements),
                        static_cast<std::uint64_t>(first),
                        static_cast<std::uint64_t>(count),
                        static_cast<std::uint64_t>(rows),
                        static_cast<std::uint64_t>(columns), bits);
                Policy::check_kernel(Policy::scatter_kernel_operation());
            } else {
                IOM_LAUNCH_KERNEL(
                        gather_plane_kernel, blocks, kThreads, stream,
                        static_cast<const unsigned char*>(source),
                        static_cast<unsigned char*>(destination),
                        static_cast<std::uint64_t>(planes[logical_plane]),
                        static_cast<std::uint64_t>(logical_plane * elements),
                        static_cast<std::uint64_t>(first),
                        static_cast<std::uint64_t>(count),
                        static_cast<std::uint64_t>(rows),
                        static_cast<std::uint64_t>(columns), bits);
                Policy::check_kernel(Policy::gather_kernel_operation());
            }
        }
    }
}

template <typename Policy>
void launch_copy_plane(
        typename Policy::stream_type stream, const PlanePair& pair,
        std::size_t plane_index, std::size_t rows, std::size_t columns,
        unsigned int bits, const void* source, void* destination) {
    const std::size_t elements = rows * columns;
    for (std::size_t first = 0; first < elements; first += kLaunchChunk) {
        const std::size_t count = std::min(kLaunchChunk, elements - first);
        const unsigned int blocks = static_cast<unsigned int>(
                (count + kThreads - 1) / kThreads);
        IOM_LAUNCH_KERNEL(
                copy_plane_kernel, blocks, kThreads, stream,
                static_cast<const unsigned char*>(source),
                static_cast<unsigned char*>(destination),
                static_cast<std::uint64_t>(pair.source),
                static_cast<std::uint64_t>(pair.destination),
                static_cast<std::uint64_t>(first),
                static_cast<std::uint64_t>(count),
                static_cast<std::uint64_t>(rows),
                static_cast<std::uint64_t>(columns), bits);
        Policy::check_kernel(Policy::copy_kernel_operation());
        Policy::after_copy_plane_launch(plane_index);
    }
}

template <typename Policy>
void synchronous_transfer_impl(
        typename Policy::context_type context, const TensorView& view,
        std::span<const std::byte> source, std::span<std::byte> destination,
        bool from_host) {
    const std::size_t logical_nbytes = view.spec().logical_nbytes();
    const std::size_t staging_nbytes =
            gpu_algorithm::compute_staging_size(logical_nbytes);
    Policy::activate(context);

    typename Policy::stream_type stream = Policy::create_transfer_stream();
    void* staging = nullptr;
    try {
        staging = Policy::allocate(staging_nbytes);
        if (from_host) {
            Policy::copy_from_host(
                    stream, staging, source.data(), source.size());
            launch_view_transfer<Policy>(
                    stream, view, staging,
                    const_cast<void*>(view.native_handle()), true);
        } else {
            Policy::memset(stream, staging, staging_nbytes);
            launch_view_transfer<Policy>(
                    stream, view, view.native_handle(), staging, false);
        }
        Policy::synchronize_stream(stream);
        if (!from_host) {
            Policy::copy_to_host(
                    stream, destination.data(), staging, destination.size());
        }
    } catch (...) {
        if (Policy::stream_is_valid(stream)) {
            Policy::synchronize_stream_noexcept(stream);
            Policy::destroy_transfer_stream_noexcept(stream);
        }
        if (staging != nullptr) {
            Policy::free_noexcept(staging);
        }
        throw;
    }
    Policy::destroy_transfer_stream(stream);
    Policy::free(staging);
}

template <typename Policy>
struct PendingEvent {
    typename Policy::event_type event = Policy::null_event();
    bool linked = false;

    ~PendingEvent() noexcept {
        if (Policy::event_is_valid(event) && !linked) {
            Policy::destroy_event_noexcept(event);
        }
    }
};

}  // namespace

template <typename Policy>
void synchronous_transfer(
        typename Policy::context_type context, const TensorView& view,
        std::span<const std::byte> source, std::span<std::byte> destination,
        bool from_host) {
    synchronous_transfer_impl<Policy>(
            context, view, source, destination, from_host);
}


}  // namespace iom::detail
