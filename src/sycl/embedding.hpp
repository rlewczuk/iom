#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

#include "../shared/standard_tiled_embedding.hpp"

namespace iom::sycl_detail {

// Launches exactly one native SYCL work item for each destination 32-bit word.
// The reset event orders the gather after the caller's device status reset and
// metadata upload; the returned event is the gather completion dependency.
[[nodiscard]] sycl::event launch_embedding_words(
        sycl::queue& queue, const sycl::event& reset,
        const unsigned char* table, const unsigned char* indices,
        unsigned char* output, std::uint32_t* status,
        const detail::EmbeddingMetadata& metadata);

}  // namespace iom::sycl_detail
