#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "queue_resources.hpp"
#include "iom/iom.hpp"

namespace iom::detail {

// Scalar metadata copied into one fixed queue slot before a CUDA/HIP gather.
// The three trailing arrays contain the leading extents, index-plane strides,
// and output-plane strides. They are device pointers only in the launch-local
// view and are never serialized into the slot.
struct EmbeddingMetadataHeader {
    std::uint64_t table_plane = 0;
    std::uint64_t indices_plane = 0;
    std::uint64_t output_plane = 0;
    std::uint64_t vocabulary = 0;
    std::uint64_t run = 0;
    std::uint64_t features = 0;
    std::uint64_t plane_count = 0;
    std::uint64_t words_per_plane = 0;
    std::uint32_t payload_bits = 0;
    std::uint32_t index_bits = 0;
    std::uint32_t payload_type = 0;
    std::uint32_t index_type = 0;
    std::uint32_t leading_rank = 0;
    std::uint32_t index_signed = 0;
};

struct EmbeddingMetadata {
    EmbeddingMetadataHeader header{};
    const std::uint64_t* dimensions = nullptr;
    const std::uint64_t* index_strides = nullptr;
    const std::uint64_t* output_strides = nullptr;
};

struct InlineEmbeddingMetadata {
    EmbeddingMetadataHeader header{};
    std::uint64_t values[3 * 6]{};
};

static_assert(std::is_trivially_copyable_v<EmbeddingMetadataHeader>);
static_assert(std::is_trivially_copyable_v<InlineEmbeddingMetadata>);
static_assert(sizeof(InlineEmbeddingMetadata) <= kMetadataSlotBytes);
static_assert(alignof(InlineEmbeddingMetadata) <= 32);


[[nodiscard]] inline std::size_t embedding_metadata_storage_bytes(
        std::size_t rank) {
    if (rank < 2 || rank > 8) {
        throw std::invalid_argument("embedding metadata rank is out of range");
    }
    const std::size_t leading = rank - 2;
    if (leading > std::numeric_limits<std::size_t>::max() / 3
            || leading * 3 > std::numeric_limits<std::size_t>::max()
                    / sizeof(std::uint64_t)) {
        throw std::overflow_error("embedding metadata size overflows");
    }
    const std::size_t array_bytes =
            leading * 3 * sizeof(std::uint64_t);
    if (array_bytes > std::numeric_limits<std::size_t>::max()
            - sizeof(EmbeddingMetadataHeader)) {
        throw std::overflow_error("embedding metadata size overflows");
    }
    const std::size_t result = sizeof(EmbeddingMetadataHeader) + array_bytes;
    if (result > kMetadataSlotBytes) {
        throw std::overflow_error("embedding metadata exceeds fixed slot");
    }
    return result;
}

template <typename Request>
[[nodiscard]] EmbeddingMetadata write_embedding_metadata(
        void* host_storage, const void* device_storage, const Request& request);

template <typename Policy>
void launch_grid_stride_embedding(
        typename Policy::stream_type stream, const unsigned char* table,
        const unsigned char* indices, unsigned char* output,
        std::uint32_t* status, const EmbeddingMetadata& metadata);

}  // namespace iom::detail
