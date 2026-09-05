#include "standard_tiled_copy.hpp"

#include <algorithm>
#include <cstdint>

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
#ifndef IOM_LAUNCH_KERNEL
#error "IOM_LAUNCH_KERNEL must be defined before including standard_tiled_copy.inl"
#endif

namespace iom::detail {
namespace {

constexpr std::uint64_t kTile = TensorSpec::TILE;
constexpr std::uint64_t kTileSlots = kTile * kTile;
constexpr unsigned int kThreads = 256;
constexpr unsigned int kMaxBlocks = 65535;

struct WordPair {
    std::uint32_t low;
    std::uint32_t high;
};

IOM_GPU_DEVICE std::uint32_t field_mask(unsigned int bits) {
    return bits == 32 ? 0xffffffffu : ((std::uint32_t{1} << bits) - 1);
}

IOM_GPU_DEVICE std::uint32_t read_field_extracted(
        const std::uint32_t* source_word_ptr, unsigned int bit_offset,
        unsigned int bits) {
    std::uint64_t joined = source_word_ptr[0];
    if (bit_offset + bits > 32) {
        joined |= static_cast<std::uint64_t>(source_word_ptr[1]) << 32;
    }
    return static_cast<std::uint32_t>(
            (joined >> bit_offset) & field_mask(bits));
}

IOM_GPU_DEVICE void read_field_pair_64(
        const std::uint32_t* low_word_ptr,
        const std::uint32_t* high_word_ptr, std::uint32_t& low,
        std::uint32_t& high) {
    low = *low_word_ptr;
    high = *high_word_ptr;
}

IOM_GPU_DEVICE void merge_field(
        std::uint32_t& destination_word, std::uint32_t value,
        unsigned int bit_offset, unsigned int bits) {
    const std::uint32_t mask = field_mask(bits) << bit_offset;
    destination_word =
            (destination_word & ~mask) | ((value << bit_offset) & mask);
}

IOM_GPU_DEVICE void store_word(
        std::uint32_t* destination_word_ptr,
        std::uint32_t destination_word) {
    *destination_word_ptr = destination_word;
}

IOM_GPU_DEVICE std::uint64_t plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_rows = (rows + kTile - 1) / kTile;
    const std::uint64_t tile_columns = (columns + kTile - 1) / kTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kTile) * tile_columns + column / kTile;
    return tile_index * kTileSlots
            + (row % kTile) * kTile + column % kTile;
}

struct PhysicalCoordinate {
    std::uint64_t row;
    std::uint64_t column;
};

IOM_GPU_DEVICE PhysicalCoordinate physical_coordinate(
        std::uint64_t slot, std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_columns = (columns + kTile - 1) / kTile;
    const std::uint64_t tile_index = slot / kTileSlots;
    const std::uint64_t in_tile = slot % kTileSlots;
    return PhysicalCoordinate{
            (tile_index / tile_columns) * kTile + in_tile / kTile,
            (tile_index % tile_columns) * kTile + in_tile % kTile};
}

IOM_GPU_DEVICE void merge_overlapping_field(
        std::uint32_t& destination_word, const unsigned char* source,
        std::uint64_t source_bit, std::uint64_t destination_bit,
        std::uint64_t word_first_bit, std::uint64_t word_end_bit,
        unsigned int bits) {
    const std::uint64_t field_end = destination_bit + bits;
    const std::uint64_t overlap_first =
            destination_bit > word_first_bit
            ? destination_bit
            : word_first_bit;
    const std::uint64_t overlap_end =
            field_end < word_end_bit ? field_end : word_end_bit;
    if (overlap_first >= overlap_end) {
        return;
    }
    const unsigned int destination_offset = static_cast<unsigned int>(
            overlap_first - word_first_bit);
    const unsigned int source_offset = static_cast<unsigned int>(
            overlap_first - destination_bit);
    const unsigned int part_bits = static_cast<unsigned int>(
            overlap_end - overlap_first);
    const auto* source_words =
            reinterpret_cast<const std::uint32_t*>(source);
    if (bits == 64) {
        std::uint32_t low = 0;
        std::uint32_t high = 0;
        const std::uint64_t source_word = source_bit / 32;
        read_field_pair_64(
                source_words + source_word, source_words + source_word + 1,
                low, high);
        const std::uint32_t value = source_offset < 32 ? low : high;
        merge_field(destination_word, value, destination_offset, part_bits);
        return;
    }
    const std::uint32_t value = read_field_extracted(
            source_words + source_bit / 32,
            static_cast<unsigned int>(source_bit % 32), bits);
    merge_field(
            destination_word, value >> source_offset, destination_offset,
            part_bits);
}

IOM_GPU_DEVICE void copy_tiled_to_tiled_word(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t destination_plane,
        std::uint64_t word_in_plane, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t padded_rows = (rows + kTile - 1) / kTile * kTile;
    const std::uint64_t padded_columns =
            (columns + kTile - 1) / kTile * kTile;
    const std::uint64_t padded_elements = padded_rows * padded_columns;
    const std::uint64_t plane_bits = padded_elements * bits;
    const std::uint64_t word_first_bit = word_in_plane * 32;
    const std::uint64_t word_end_bit =
            word_first_bit + 32 < plane_bits
            ? word_first_bit + 32
            : plane_bits;
    const std::uint64_t first_slot = word_first_bit / bits;
    const std::uint64_t last_slot =
            (word_end_bit + bits - 1) / bits;
    const std::uint64_t destination_base_word =
            plane_slot(destination_plane, 0, 0, rows, columns) * bits / 32;
    auto* destination_words =
            reinterpret_cast<std::uint32_t*>(destination)
            + destination_base_word + word_in_plane;
    std::uint32_t destination_word = *destination_words;
    for (std::uint64_t slot = first_slot; slot < last_slot; ++slot) {
        const PhysicalCoordinate coordinate =
                physical_coordinate(slot, rows, columns);
        if (coordinate.row >= rows || coordinate.column >= columns) {
            continue;
        }
        const std::uint64_t source_bit =
                plane_slot(
                        source_plane, coordinate.row, coordinate.column, rows,
                        columns)
                * bits;
        const std::uint64_t destination_bit =
                plane_slot(
                        destination_plane, coordinate.row, coordinate.column,
                        rows, columns)
                * bits;
        merge_overlapping_field(
                destination_word, source, source_bit, destination_bit,
                word_first_bit + destination_base_word * 32,
                word_end_bit + destination_base_word * 32, bits);
    }
    store_word(destination_words, destination_word);
}

IOM_GPU_DEVICE void copy_logical_to_tiled_word(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t destination_plane, std::uint64_t logical_base,
        std::uint64_t word_in_plane, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t padded_rows = (rows + kTile - 1) / kTile * kTile;
    const std::uint64_t padded_columns =
            (columns + kTile - 1) / kTile * kTile;
    const std::uint64_t padded_elements = padded_rows * padded_columns;
    const std::uint64_t plane_bits = padded_elements * bits;
    const std::uint64_t word_first_bit = word_in_plane * 32;
    const std::uint64_t word_end_bit =
            word_first_bit + 32 < plane_bits
            ? word_first_bit + 32
            : plane_bits;
    const std::uint64_t first_slot = word_first_bit / bits;
    const std::uint64_t last_slot =
            (word_end_bit + bits - 1) / bits;
    const std::uint64_t destination_base_word =
            plane_slot(destination_plane, 0, 0, rows, columns) * bits / 32;
    auto* destination_words =
            reinterpret_cast<std::uint32_t*>(destination)
            + destination_base_word + word_in_plane;
    std::uint32_t destination_word = *destination_words;
    for (std::uint64_t slot = first_slot; slot < last_slot; ++slot) {
        const PhysicalCoordinate coordinate =
                physical_coordinate(slot, rows, columns);
        if (coordinate.row >= rows || coordinate.column >= columns) {
            continue;
        }
        const std::uint64_t source_bit =
                (logical_base
                 + coordinate.row * columns + coordinate.column)
                * bits;
        const std::uint64_t destination_bit =
                plane_slot(
                        destination_plane, coordinate.row, coordinate.column,
                        rows, columns)
                * bits;
        merge_overlapping_field(
                destination_word, source, source_bit, destination_bit,
                word_first_bit + destination_base_word * 32,
                word_end_bit + destination_base_word * 32, bits);
    }
    store_word(destination_words, destination_word);
}

IOM_GPU_DEVICE void copy_tiled_to_logical_word(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t logical_base,
        std::uint64_t word, std::uint64_t rows, std::uint64_t columns,
        unsigned int bits) {
    const std::uint64_t elements = rows * columns;
    const std::uint64_t logical_first_bit = logical_base * bits;
    const std::uint64_t logical_end_bit =
            (logical_base + elements) * bits;
    const std::uint64_t word_first_bit = word * 32;
    if (word_first_bit >= logical_end_bit) {
        return;
    }
    const std::uint64_t word_end_bit =
            word_first_bit + 32 < logical_end_bit
            ? word_first_bit + 32
            : logical_end_bit;
    const std::uint64_t first_bit =
            word_first_bit > logical_first_bit
            ? word_first_bit
            : logical_first_bit;
    const std::uint64_t first_slot = first_bit / bits;
    const std::uint64_t last_slot =
            (word_end_bit + bits - 1) / bits;
    auto* destination_words =
            reinterpret_cast<std::uint32_t*>(destination);
    std::uint32_t destination_word = destination_words[word];
    for (std::uint64_t slot = first_slot; slot < last_slot; ++slot) {
        if (slot < logical_base || slot >= logical_base + elements) {
            continue;
        }
        const std::uint64_t local = slot - logical_base;
        const std::uint64_t row = local / columns;
        const std::uint64_t column = local % columns;
        const std::uint64_t source_bit =
                plane_slot(source_plane, row, column, rows, columns) * bits;
        const std::uint64_t destination_bit = slot * bits;
        merge_overlapping_field(
                destination_word, source, source_bit, destination_bit,
                word_first_bit, word_end_bit, bits);
    }
    store_word(destination_words + word, destination_word);
}

IOM_GPU_GLOBAL void scatter_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t destination_plane, std::uint64_t logical_base,
        std::uint64_t word_count, std::uint64_t rows,
        std::uint64_t columns, unsigned int bits) {
    const std::uint64_t stride =
            static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t word = IOM_GPU_GLOBAL_INDEX; word < word_count;
         word += stride) {
        copy_logical_to_tiled_word(
                source, destination, destination_plane, logical_base, word,
                rows, columns, bits);
    }
}

IOM_GPU_GLOBAL void gather_plane_kernel(
        const unsigned char* source, unsigned char* destination,
        std::uint64_t source_plane, std::uint64_t logical_base,
        std::uint64_t first_word, std::uint64_t word_count,
        std::uint64_t rows, std::uint64_t columns, unsigned int bits) {
    const std::uint64_t stride =
            static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (std::uint64_t index = IOM_GPU_GLOBAL_INDEX; index < word_count;
         index += stride) {
        copy_tiled_to_logical_word(
                source, destination, source_plane, logical_base,
                first_word + index, rows, columns, bits);
    }
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
    const std::size_t padded_rows =
            (rows + TensorSpec::TILE - 1) / TensorSpec::TILE
            * TensorSpec::TILE;
    const std::size_t padded_columns =
            (columns + TensorSpec::TILE - 1) / TensorSpec::TILE
            * TensorSpec::TILE;
    const std::size_t words_per_plane =
            padded_rows * padded_columns * bits / 32;
    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        std::size_t rest = logical_plane;
        std::size_t plane = view.plane_offset();
        for (std::size_t k = leading_rank; k-- > 0;) {
            plane += (rest % dimensions[k]) * plane_strides[k];
            rest /= dimensions[k];
        }
        const std::uint64_t logical_base =
                static_cast<std::uint64_t>(logical_plane * elements);
        std::size_t first_word = 0;
        std::size_t word_count = words_per_plane;
        if (!from_host) {
            first_word = logical_base * bits / 32;
            const std::size_t last_word =
                    (logical_base + elements) * bits / 32
                    + (((logical_base + elements) * bits) % 32 != 0);
            word_count = last_word - first_word;
        }
        const unsigned int blocks = static_cast<unsigned int>(
                std::min<std::size_t>(
                        (word_count + kThreads - 1) / kThreads,
                        kMaxBlocks));
        if (from_host) {
            IOM_LAUNCH_KERNEL(
                    scatter_plane_kernel, blocks, kThreads, stream,
                    static_cast<const unsigned char*>(source),
                    static_cast<unsigned char*>(destination),
                    static_cast<std::uint64_t>(plane), logical_base,
                    static_cast<std::uint64_t>(word_count),
                    static_cast<std::uint64_t>(rows),
                    static_cast<std::uint64_t>(columns), bits);
            Policy::check_kernel(Policy::scatter_kernel_operation());
        } else {
            IOM_LAUNCH_KERNEL(
                    gather_plane_kernel, blocks, kThreads, stream,
                    static_cast<const unsigned char*>(source),
                    static_cast<unsigned char*>(destination),
                    static_cast<std::uint64_t>(plane), logical_base,
                    static_cast<std::uint64_t>(first_word),
                    static_cast<std::uint64_t>(word_count),
                    static_cast<std::uint64_t>(rows),
                    static_cast<std::uint64_t>(columns), bits);
            Policy::check_kernel(Policy::gather_kernel_operation());
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
