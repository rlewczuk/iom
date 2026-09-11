#pragma once

#include <cstdint>
#include <memory>

#include "iom/device.hpp"

namespace iom {

    /**
     * Creates one SYCL device with an owned context for the requested
     * backend-local accelerator ordinal. The device reserves one fixed
     * device-USM tensor-data arena of `memory_config.tensor_arena_bytes`
     * bytes and one distinct metadata arena checked at `4 * C * 512` bytes
     * (C = `queue_config.max_in_flight_per_queue`) in that exact context,
     * then owns both arenas and their allocators for its lifetime. Capacity
     * must be nonzero and divisible by 32; rejected configurations report
     * `std::invalid_argument`, checked sizing overflow reports
     * `std::overflow_error`, and native allocation failure reports
     * `std::bad_alloc`.
     *
     * The backend-local ordinal enumerates eligible non-CPU SYCL devices in
     * the deterministic order defined by the SYCL backend implementation.
     */
    [[nodiscard]] std::unique_ptr<Device> make_sycl_device(
            std::uint32_t device_ordinal, DeviceMemoryConfig memory_config,
            QueueConfig queue_config = {});

}  // namespace iom
