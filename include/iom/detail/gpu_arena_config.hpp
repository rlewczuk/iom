#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace iom::detail {

    // Backend-neutral validation for the standard-GPU (CUDA/ROCm/SYCL)
    // factory configuration. The fixed tensor-data arena must have nonzero
    // capacity divisible by 32; the fixed metadata arena is exactly
    // 4 * C * 512 bytes with checked arithmetic. Invalid configuration
    // reports std::invalid_argument and checked sizing overflow reports
    // std::overflow_error, both before any native reservation. Concrete
    // backends keep native allocation and context handling local; this
    // shared sizing keeps the 4 * C * 512 geometry in one place.

    [[nodiscard]] inline std::size_t valid_tensor_arena_bytes(
            std::size_t tensor_arena_bytes) {
        if (tensor_arena_bytes == 0) {
            throw std::invalid_argument(
                    "tensor arena capacity must be nonzero");
        }
        if (tensor_arena_bytes % 32 != 0) {
            throw std::invalid_argument(
                    "tensor arena capacity must be divisible by 32");
        }
        return tensor_arena_bytes;
    }

    [[nodiscard]] inline std::size_t standard_gpu_metadata_bytes(
            std::size_t max_in_flight_per_queue) {
        constexpr std::size_t kQueueCount = 4;
        constexpr std::size_t kMetadataSlotBytes = 512;
        if (max_in_flight_per_queue == 0) {
            throw std::invalid_argument(
                    "max_in_flight_per_queue must be nonzero");
        }
        if (max_in_flight_per_queue
                > std::numeric_limits<std::size_t>::max() / kQueueCount) {
            throw std::overflow_error(
                    "metadata slot count overflows size range");
        }
        const std::size_t slot_count =
                kQueueCount * max_in_flight_per_queue;
        if (slot_count
                > std::numeric_limits<std::size_t>::max()
                        / kMetadataSlotBytes) {
            throw std::overflow_error(
                    "metadata backing size overflows size range");
        }
        return slot_count * kMetadataSlotBytes;
    }

}  // namespace iom::detail