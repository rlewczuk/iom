#include "standard_tiled_embedding.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifndef IOM_GPU_DEVICE
#error "IOM_GPU_DEVICE must be defined before including standard_tiled_embedding.inl"
#endif
#ifndef IOM_GPU_GLOBAL
#error "IOM_GPU_GLOBAL must be defined before including standard_tiled_embedding.inl"
#endif
#ifndef IOM_GPU_GLOBAL_INDEX
#error "IOM_GPU_GLOBAL_INDEX must be defined before including standard_tiled_embedding.inl"
#endif
#ifndef IOM_LAUNCH_KERNEL
#error "IOM_LAUNCH_KERNEL must be defined before including standard_tiled_embedding.inl"
#endif
#ifndef IOM_GPU_GLOBAL_STRIDE
#define IOM_GPU_GLOBAL_STRIDE \
    (static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
#endif

namespace iom::detail {
namespace {

inline constexpr std::uint64_t kEmbeddingTile = TensorSpec::TILE;
inline constexpr std::uint64_t kEmbeddingTileSlots =
        kEmbeddingTile * kEmbeddingTile;
inline constexpr unsigned int kEmbeddingThreads = 256;
inline constexpr unsigned int kEmbeddingMaxBlocks = 65535;

IOM_GPU_DEVICE std::uint64_t embedding_mask(unsigned int bits) {
    return bits == 64
            ? ~std::uint64_t{0}
            : ((std::uint64_t{1} << bits) - 1);
}

IOM_GPU_DEVICE std::uint64_t embedding_read_bits(
        const unsigned char* source, std::uint64_t bit_offset,
        unsigned int bits) {
    const auto* words = reinterpret_cast<const std::uint32_t*>(source);
    const std::uint64_t word = bit_offset / 32;
    const unsigned int offset = static_cast<unsigned int>(bit_offset % 32);
    if (bits == 64) {
        const std::uint64_t low = words[word];
        const std::uint64_t high = words[word + 1];
        return low | (high << 32);
    }
    std::uint64_t joined = words[word];
    if (offset + bits > 32) {
        joined |= static_cast<std::uint64_t>(words[word + 1]) << 32;
    }
    return (joined >> offset) & embedding_mask(bits);
}

IOM_GPU_DEVICE std::uint64_t embedding_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_rows =
            (rows + kEmbeddingTile - 1) / kEmbeddingTile;
    const std::uint64_t tile_columns =
            (columns + kEmbeddingTile - 1) / kEmbeddingTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kEmbeddingTile) * tile_columns
            + column / kEmbeddingTile;
    return tile_index * kEmbeddingTileSlots
            + (row % kEmbeddingTile) * kEmbeddingTile
            + column % kEmbeddingTile;
}

struct EmbeddingPhysicalCoordinate {
    std::uint64_t row;
    std::uint64_t column;
};

IOM_GPU_DEVICE EmbeddingPhysicalCoordinate embedding_physical_coordinate(
        std::uint64_t slot, std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_columns =
            (columns + kEmbeddingTile - 1) / kEmbeddingTile;
    const std::uint64_t tile_index = slot / kEmbeddingTileSlots;
    const std::uint64_t in_tile = slot % kEmbeddingTileSlots;
    return EmbeddingPhysicalCoordinate{
            (tile_index / tile_columns) * kEmbeddingTile
                    + in_tile / kEmbeddingTile,
            (tile_index % tile_columns) * kEmbeddingTile
                    + in_tile % kEmbeddingTile};
}

IOM_GPU_DEVICE void embedding_merge_overlap(
        std::uint32_t& destination, std::uint64_t value,
        std::uint64_t destination_bit, std::uint64_t word_first_bit,
        std::uint64_t word_end_bit, unsigned int bits) {
    const std::uint64_t field_end = destination_bit + bits;
    const std::uint64_t overlap_first =
            destination_bit > word_first_bit ? destination_bit : word_first_bit;
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
    const std::uint32_t part = static_cast<std::uint32_t>(
            (value >> source_offset) & embedding_mask(part_bits));
    const std::uint32_t mask = static_cast<std::uint32_t>(
            embedding_mask(part_bits) << destination_offset);
    destination = (destination & ~mask) | ((part << destination_offset) & mask);
}

IOM_GPU_DEVICE void embedding_one_word(
        const unsigned char* table, const unsigned char* indices,
        unsigned char* output, std::uint32_t* status,
        EmbeddingMetadata metadata, std::uint64_t global) {
    const std::uint64_t logical_plane =
            global / metadata.header.words_per_plane;
    const std::uint64_t word_in_plane =
            global % metadata.header.words_per_plane;
    std::uint64_t index_plane = metadata.header.indices_plane;
    std::uint64_t output_plane = metadata.header.output_plane;
    std::uint64_t rest = logical_plane;
    for (std::uint32_t axis = metadata.header.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % metadata.dimensions[axis];
        rest /= metadata.dimensions[axis];
        index_plane += coordinate * metadata.index_strides[axis];
        output_plane += coordinate * metadata.output_strides[axis];
    }

    const std::uint64_t output_base = embedding_plane_slot(
            output_plane, 0, 0, metadata.header.run,
            metadata.header.features);
    auto* output_words = reinterpret_cast<std::uint32_t*>(output)
            + output_base * metadata.header.payload_bits / 32;
    std::uint32_t destination = output_words[word_in_plane];
    const std::uint64_t word_first_bit = word_in_plane * 32;
    const std::uint64_t word_end_bit = word_first_bit + 32;
    const std::uint64_t first_slot =
            word_first_bit / metadata.header.payload_bits;
    const std::uint64_t last_slot =
            (word_end_bit + metadata.header.payload_bits - 1)
            / metadata.header.payload_bits;
    for (std::uint64_t slot = first_slot; slot < last_slot; ++slot) {
        const EmbeddingPhysicalCoordinate coordinate =
                embedding_physical_coordinate(
                        slot, metadata.header.run,
                        metadata.header.features);
        if (coordinate.row >= metadata.header.run
                || coordinate.column >= metadata.header.features) {
            continue;
        }
        const std::uint64_t destination_bit = embedding_plane_slot(
                0, coordinate.row, coordinate.column,
                metadata.header.run, metadata.header.features)
                * metadata.header.payload_bits;
        const std::uint64_t index_bit = embedding_plane_slot(
                index_plane, 0, coordinate.row, 1,
                metadata.header.run) * metadata.header.index_bits;
        const std::uint64_t index = embedding_read_bits(
                indices, index_bit, metadata.header.index_bits);
        const bool negative = metadata.header.index_signed != 0
                && ((index >> (metadata.header.index_bits - 1)) & 1u) != 0;
        if (negative || index >= metadata.header.vocabulary) {
            atomicOr(status, std::uint32_t{1});
            continue;
        }
        const std::uint64_t source_bit = embedding_plane_slot(
                metadata.header.table_plane, index, coordinate.column,
                metadata.header.vocabulary,
                metadata.header.features)
                * metadata.header.payload_bits;
        const std::uint64_t value = embedding_read_bits(
                table, source_bit, metadata.header.payload_bits);
        embedding_merge_overlap(
                destination, value, destination_bit, word_first_bit,
                word_end_bit, metadata.header.payload_bits);
    }
    output_words[word_in_plane] = destination;
}

IOM_GPU_GLOBAL void embedding_word_kernel(
        const unsigned char* table, const unsigned char* indices,
        unsigned char* output, std::uint32_t* status,
        EmbeddingMetadata metadata) {
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    const std::uint64_t total_words =
            metadata.header.plane_count * metadata.header.words_per_plane;
    for (std::uint64_t global = IOM_GPU_GLOBAL_INDEX; global < total_words;
         global += stride) {
        embedding_one_word(
                table, indices, output, status, metadata, global);
    }
}

[[nodiscard]] bool embedding_index_is_signed(DataType type) noexcept {
    switch (type) {
        case DataType::I2: case DataType::I4: case DataType::I8:
        case DataType::I16: case DataType::I32: case DataType::I64:
            return true;
        default:
            return false;
    }
}

}  // namespace

template <typename Request>
EmbeddingMetadata write_embedding_metadata(
        void* host_storage, const void* device_storage,
        const Request& request) {
    const auto table_dimensions = request.table.spec.shape.dimensions();
    const auto index_dimensions = request.indices.spec.shape.dimensions();
    const auto output_dimensions = request.out.spec.shape.dimensions();
    const std::size_t rank = index_dimensions.size();
    const std::size_t leading = rank - 2;
    const std::size_t bytes = embedding_metadata_storage_bytes(rank);
    (void)bytes;

    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading; ++axis) {
        if (index_dimensions[axis] != output_dimensions[axis]) {
            throw std::invalid_argument(
                    "embedding metadata leading dimensions differ");
        }
        if (plane_count > std::numeric_limits<std::size_t>::max()
                / index_dimensions[axis]) {
            throw std::overflow_error(
                    "embedding metadata plane count overflows");
        }
        plane_count *= index_dimensions[axis];
    }
    const std::size_t padded_rows =
            (output_dimensions[rank - 2] + TensorSpec::TILE - 1)
            / TensorSpec::TILE * TensorSpec::TILE;
    const std::size_t padded_columns =
            (output_dimensions[rank - 1] + TensorSpec::TILE - 1)
            / TensorSpec::TILE * TensorSpec::TILE;
    const std::size_t padded_elements = padded_rows * padded_columns;
    const std::size_t words_per_plane =
            padded_elements * leaf_bits(request.out.spec.data_type) / 32;

    auto* host = static_cast<std::byte*>(host_storage);
    auto* header = reinterpret_cast<EmbeddingMetadataHeader*>(host);
    *header = EmbeddingMetadataHeader{
            static_cast<std::uint64_t>(request.table.plane_offset),
            static_cast<std::uint64_t>(request.indices.plane_offset),
            static_cast<std::uint64_t>(request.out.plane_offset),
            static_cast<std::uint64_t>(table_dimensions[0]),
            static_cast<std::uint64_t>(output_dimensions[rank - 2]),
            static_cast<std::uint64_t>(output_dimensions[rank - 1]),
            static_cast<std::uint64_t>(plane_count),
            static_cast<std::uint64_t>(words_per_plane),
            static_cast<std::uint32_t>(leaf_bits(request.out.spec.data_type)),
            static_cast<std::uint32_t>(leaf_bits(request.indices.spec.data_type)),
            static_cast<std::uint32_t>(request.out.spec.data_type),
            static_cast<std::uint32_t>(request.indices.spec.data_type),
            static_cast<std::uint32_t>(leading),
            embedding_index_is_signed(request.indices.spec.data_type) ? 1u : 0u};
    auto* values = reinterpret_cast<std::uint64_t*>(header + 1);
    for (std::size_t axis = 0; axis < leading; ++axis) {
        values[axis] = static_cast<std::uint64_t>(index_dimensions[axis]);
        values[leading + axis] = static_cast<std::uint64_t>(
                request.indices.plane_strides[axis]);
        values[2 * leading + axis] = static_cast<std::uint64_t>(
                request.out.plane_strides[axis]);
    }

    const auto* device = static_cast<const std::byte*>(device_storage);
    const auto* device_values = reinterpret_cast<const std::uint64_t*>(
            device + sizeof(EmbeddingMetadataHeader));
    return EmbeddingMetadata{
            *header, device_values, device_values + leading,
            device_values + 2 * leading};
}

template <typename Policy>
void launch_grid_stride_embedding(
        typename Policy::stream_type stream, const unsigned char* table,
        const unsigned char* indices, unsigned char* output,
        std::uint32_t* status, const EmbeddingMetadata& metadata) {
    const std::uint64_t total_words =
            metadata.header.plane_count * metadata.header.words_per_plane;
    const std::uint64_t launch_words = total_words + kEmbeddingThreads - 1;
    const unsigned int blocks = static_cast<unsigned int>(std::min<std::uint64_t>(
            launch_words / kEmbeddingThreads, kEmbeddingMaxBlocks));
    IOM_LAUNCH_KERNEL(
            embedding_word_kernel, blocks, kEmbeddingThreads, stream, table,
            indices, output, status, metadata);
}

}  // namespace iom::detail
