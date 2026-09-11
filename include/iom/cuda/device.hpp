#pragma once

#include <cstdint>
#include <memory>

#include "iom/device.hpp"

namespace iom {

    /**
     * Creates one CUDA device with an owned driver context for the requested
     * backend-local ordinal. The device reserves one fixed tensor-data arena
     * of `memory_config.tensor_arena_bytes` bytes and one distinct metadata
     * arena checked at `4 * C * 512` bytes (C = `queue_config.
     * max_in_flight_per_queue`) during setup, then owns both arenas and
     * their allocators for its lifetime. Capacity must be nonzero and
     * divisible by 32; rejected configurations report
     * `std::invalid_argument`, checked sizing overflow reports
     * `std::overflow_error`, and native allocation failure reports
     * `std::bad_alloc`.
     */
    [[nodiscard]] std::unique_ptr<Device> make_cuda_device(
            std::uint32_t device_ordinal, DeviceMemoryConfig memory_config,
            QueueConfig queue_config = {});

}  // namespace iom
