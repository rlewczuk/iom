#include "standard_tiled_copy.hpp"

#include <algorithm>

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



template <typename Policy>
void launch_view_transfer(
        typename Policy::stream_type stream, const TensorView& view,
        const void* source, void* destination, bool from_host) {
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::span<const std::size_t> plane_strides =
            view.plane_strides();
    const std::size_t plane_count =
            view.spec().shape.element_count() / (rows * columns);
    const unsigned int bits =
            static_cast<unsigned int>(leaf_bits(view.spec().data_type));
    const std::size_t elements = rows * columns;
    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        std::size_t rest = logical_plane;
        std::size_t plane = view.plane_offset();
        for (std::size_t k = leading_rank; k-- > 0;) {
            plane += (rest % dimensions[k]) * plane_strides[k];
            rest /= dimensions[k];
        }
        for (std::size_t first = 0; first < elements; first += kLaunchChunk) {
            const std::size_t count = std::min(kLaunchChunk, elements - first);
            const unsigned int blocks = static_cast<unsigned int>(
                    (count + kThreads - 1) / kThreads);
            if (from_host) {
                IOM_LAUNCH_KERNEL(
                        scatter_plane_kernel, blocks, kThreads, stream,
                        static_cast<const unsigned char*>(source),
                        static_cast<unsigned char*>(destination),
                        static_cast<std::uint64_t>(plane),
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
                        static_cast<std::uint64_t>(plane),
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

template <typename Policy, typename StreamPool, typename StagingPool>
void synchronous_transfer_impl(
        StreamPool& transfer_pool, StagingPool& staging_pool,
        typename Policy::context_type context, const TensorView& view,
        std::span<const std::byte> source, std::span<std::byte> destination,
        bool from_host) {
    const std::size_t logical_nbytes = view.spec().logical_nbytes();
    const std::size_t staging_nbytes =
            gpu_algorithm::compute_staging_size(logical_nbytes);
    const std::size_t logical_bits =
            view.spec().shape.element_count()
            * leaf_bits(view.spec().data_type);
    Policy::activate(context);

    auto staging_lease = staging_pool.acquire(staging_nbytes);
    try {
        auto stream_scope = transfer_pool.acquire();
        const typename Policy::stream_type stream = stream_scope.stream();
        void* staging = Policy::staging_address(staging_lease.staging());
        try {
            if (from_host) {
                Policy::copy_from_host(
                        stream, staging, source.data(), source.size());
                launch_view_transfer<Policy>(
                        stream, view, staging,
                        const_cast<void*>(view.native_handle()), true);
                Policy::after_copy_plane_launch(2);
            } else {
                const std::size_t tail_word_start =
                        (logical_bits / 32) * sizeof(std::uint32_t);
                const std::size_t tail_bytes =
                        staging_nbytes - tail_word_start;
                if (tail_bytes != 0) {
                    Policy::memset(
                            stream,
                            static_cast<std::byte*>(staging) + tail_word_start,
                            tail_bytes);
                }
                launch_view_transfer<Policy>(
                        stream, view, view.native_handle(), staging, false);
                Policy::after_copy_plane_launch(2);
            }
            Policy::synchronize_stream(stream);
            if (!from_host) {
                Policy::copy_to_host(
                        stream, destination.data(), staging,
                        destination.size());
            }
        } catch (...) {
            stream_scope.poison();
            staging_lease.poison();
            throw;
        }
    } catch (...) {
        staging_lease.poison();
        throw;
    }
}


}  // namespace

template <typename Policy, typename StreamPool, typename StagingPool>
void synchronous_transfer(
        StreamPool& transfer_pool, StagingPool& staging_pool,
        typename Policy::context_type context, const TensorView& view,
        std::span<const std::byte> source, std::span<std::byte> destination,
        bool from_host) {
    synchronous_transfer_impl<Policy>(
            transfer_pool, staging_pool, context, view, source, destination,
            from_host);
}


}  // namespace iom::detail
