#pragma once

#include <cstdint>
#include <memory>

#include "iom/alloc.hpp"
#include "iom/device.hpp"

namespace iom {

    /**
     * Creates one SYCL device with an owned context for the requested
     * backend-local accelerator ordinal. The allocator is borrowed and must
     * outlive the returned device and every resource created through it.
     *
     * The backend-local ordinal enumerates eligible non-CPU SYCL devices in
     * the deterministic order defined by the SYCL backend implementation.
     */
    [[nodiscard]] std::unique_ptr<Device> make_sycl_device(
            std::uint32_t device_ordinal, Allocator& allocator);

}  // namespace iom
