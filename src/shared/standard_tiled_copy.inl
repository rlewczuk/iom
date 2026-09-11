#include "standard_tiled_copy.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "iom/gpu_algorithm.hpp"
#include "queue_resources.hpp"

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
#ifndef IOM_GPU_GLOBAL_STRIDE
#define IOM_GPU_GLOBAL_STRIDE \
    (static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
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
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
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
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    for (std::uint64_t index = IOM_GPU_GLOBAL_INDEX; index < word_count;
         index += stride) {
        copy_tiled_to_logical_word(
                source, destination, source_plane, logical_base,
                first_word + index, rows, columns, bits);
    }
}
struct CopyMetadataHeader {
    std::uint64_t source_plane_offset;
    std::uint64_t destination_plane_offset;
    std::uint64_t rows;
    std::uint64_t columns;
    std::uint64_t plane_count;
    std::uint32_t bits;
    std::uint32_t leading_rank;
};
inline constexpr std::size_t kInlineMetadataMaxRank = 8;

struct InlineCopyMetadata {
    CopyMetadataHeader header;
    std::uint64_t values[3 * kInlineMetadataMaxRank];
};
static_assert(std::is_trivially_copyable_v<InlineCopyMetadata>);
static_assert(sizeof(InlineCopyMetadata) == 240);
static_assert(sizeof(InlineCopyMetadata) % 16 == 0);
// Fixed-slot layout contract: the compiled rank-eight copy descriptor is
// 48 header bytes plus 3 * kInlineMetadataMaxRank stride words, and the
// rank-eight descriptor a submission actually writes is
// 48 + 24 * (max rank - 2) leading-rank bytes. Both the compiled
// representation and the rank-eight payload must fit one 512-byte metadata
// slot (alignment 32). A future overflow requires an intentional ABI/spec
// update, never automatic slot growth.
static_assert(
        sizeof(CopyMetadataHeader)
                + 3 * (kInlineMetadataMaxRank - 2) * sizeof(std::uint64_t)
        == 48 + 24 * 6);
static_assert(
        sizeof(CopyMetadataHeader)
                + 3 * (kInlineMetadataMaxRank - 2) * sizeof(std::uint64_t)
        <= kMetadataSlotBytes);
static_assert(sizeof(InlineCopyMetadata) <= kMetadataSlotBytes);
static_assert(
        alignof(InlineCopyMetadata) <= 32
        && kMetadataSlotBytes % alignof(InlineCopyMetadata) == 0);
static_assert(sizeof(InlineCopyMetadata) <= 1024);

static_assert(std::is_trivially_copyable_v<CopyMetadataHeader>);

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

struct CopyMetadataLayout {
    std::size_t bytes;
    std::size_t total_words;
};

[[nodiscard]] CopyMetadataLayout copy_metadata_layout(
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
            padded_elements, leaf_bits(source.spec().data_type),
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
            sizeof(CopyMetadataHeader), array_bytes,
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

void write_copy_metadata(
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
    auto* header = reinterpret_cast<CopyMetadataHeader*>(storage);
    header->source_plane_offset = metadata_u64(
            source.plane_offset(), "metadata source offset overflows");
    header->destination_plane_offset = metadata_u64(
            destination.plane_offset(), "metadata destination offset overflows");
    header->rows = metadata_u64(rows, "metadata rows overflows");
    header->columns = metadata_u64(columns, "metadata columns overflows");
    header->plane_count = metadata_u64(
            plane_count, "metadata plane count overflows");
    header->bits = static_cast<std::uint32_t>(
            leaf_bits(source.spec().data_type));
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
void write_copy_metadata(
        InlineCopyMetadata& storage, const TensorView& source,
        const TensorView& destination) {
    write_copy_metadata(
            reinterpret_cast<std::byte*>(&storage), source, destination);
}


IOM_GPU_DEVICE void copy_one_tiled_word(
        const unsigned char* source, unsigned char* destination,
        const CopyMetadataHeader& metadata,
        const std::uint64_t* values, std::uint64_t word) {
    const std::uint64_t padded_rows =
            (metadata.rows / TensorSpec::TILE
             + (metadata.rows % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_columns =
            (metadata.columns / TensorSpec::TILE
             + (metadata.columns % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_elements = padded_rows * padded_columns;
    const std::uint64_t plane_bits = padded_elements * metadata.bits;
    const std::uint64_t words_per_plane = (plane_bits + 31) / 32;
    const std::uint64_t logical_plane = word / words_per_plane;
    const std::uint64_t word_in_plane = word % words_per_plane;
    const std::uint64_t* source_strides = values;
    const std::uint64_t* destination_strides =
            source_strides + metadata.leading_rank;
    const std::uint64_t* leading_dimensions =
            destination_strides + metadata.leading_rank;
    std::uint64_t source_plane = metadata.source_plane_offset;
    std::uint64_t destination_plane =
            metadata.destination_plane_offset;
    std::uint64_t rest = logical_plane;
    for (std::uint32_t axis = metadata.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % leading_dimensions[axis];
        rest /= leading_dimensions[axis];
        source_plane += coordinate * source_strides[axis];
        destination_plane += coordinate * destination_strides[axis];
    }
    copy_tiled_to_tiled_word(
            source, destination, source_plane, destination_plane,
            word_in_plane, metadata.rows, metadata.columns,
            metadata.bits);
}

IOM_GPU_DEVICE void grid_stride_copy_body(
        const unsigned char* source, unsigned char* destination,
        const CopyMetadataHeader& metadata,
        const std::uint64_t* values) {
    const std::uint64_t padded_rows =
            (metadata.rows / TensorSpec::TILE
             + (metadata.rows % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_columns =
            (metadata.columns / TensorSpec::TILE
             + (metadata.columns % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_elements = padded_rows * padded_columns;
    const std::uint64_t plane_bits = padded_elements * metadata.bits;
    const std::uint64_t words_per_plane = (plane_bits + 31) / 32;
    const std::uint64_t total_words =
            metadata.plane_count * words_per_plane;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;

    for (std::uint64_t word = IOM_GPU_GLOBAL_INDEX; word < total_words;
         word += stride) {
        copy_one_tiled_word(source, destination, metadata, values, word);
    }
}

IOM_GPU_GLOBAL void grid_stride_copy_kernel(
        const unsigned char* source, unsigned char* destination,
        const CopyMetadataHeader* metadata) {
    grid_stride_copy_body(
            source, destination, *metadata,
            reinterpret_cast<const std::uint64_t*>(metadata + 1));
}

IOM_GPU_GLOBAL void grid_stride_copy_inline_kernel(
        const unsigned char* source, unsigned char* destination,
        InlineCopyMetadata metadata) {
    grid_stride_copy_body(
            source, destination, metadata.header, metadata.values);
}

template <typename Policy>
void launch_grid_stride_copy(
        typename Policy::stream_type stream, const unsigned char* source,
        unsigned char* destination, const CopyMetadataHeader* metadata,
        std::size_t total_words) {
    const std::size_t launch_words = checked_metadata_add(
            total_words, 255, "metadata launch count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::size_t>(launch_words / kThreads, kMaxBlocks));
    IOM_LAUNCH_KERNEL(
            grid_stride_copy_kernel, blocks, kThreads, stream,
            source, destination, metadata);
}

template <typename Policy>
void launch_grid_stride_copy(
        typename Policy::stream_type stream, const unsigned char* source,
        unsigned char* destination, const InlineCopyMetadata& metadata,
        std::size_t total_words) {
    const std::size_t launch_words = checked_metadata_add(
            total_words, 255, "metadata launch count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::size_t>(launch_words / kThreads, kMaxBlocks));
    IOM_LAUNCH_KERNEL(
            grid_stride_copy_inline_kernel, blocks, kThreads, stream,
            source, destination, metadata);
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

template <typename Policy>
void synchronous_transfer_impl(
        typename Policy::stream_type stream,
        typename Policy::context_type context, const TensorView& view,
        void* staging, std::span<const std::byte> source,
        std::span<std::byte> destination, bool from_host) {
    const std::size_t logical_nbytes = view.spec().logical_nbytes();
    const std::size_t staging_nbytes =
            gpu_algorithm::compute_staging_size(logical_nbytes);
    const std::size_t logical_bits =
            view.spec().shape.element_count()
            * leaf_bits(view.spec().data_type);
    if (staging == nullptr || staging_nbytes == 0) {
        throw std::invalid_argument("host transfer staging range is empty");
    }
    Policy::activate(context);
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
        const std::size_t tail_bytes = staging_nbytes - tail_word_start;
        if (tail_bytes != 0) {
            Policy::memset(
                    stream, static_cast<std::byte*>(staging) + tail_word_start,
                    tail_bytes);
        }
        launch_view_transfer<Policy>(
                stream, view, view.native_handle(), staging, false);
        Policy::after_copy_plane_launch(2);
    }
    Policy::synchronize_stream(stream);
    if (!from_host) {
        Policy::copy_to_host(
                stream, destination.data(), staging, destination.size());
    }
}

}  // namespace

template <typename Policy>
void synchronous_transfer(
        typename Policy::stream_type stream,
        typename Policy::context_type context, const TensorView& view,
        void* staging, std::span<const std::byte> source,
        std::span<std::byte> destination, bool from_host) {
    synchronous_transfer_impl<Policy>(
            stream, context, view, staging, source, destination, from_host);
}

}  // namespace iom::detail
