#pragma once

#include <cstdint>
#include <memory>

#include "iom/alloc.hpp"
#include "iom/device.hpp"

namespace iom {

    /**
     * Creates one CUDA device with an owned driver context for the requested
     * backend-local ordinal. The allocator is borrowed and must outlive the
     * returned device and every resource created through it. Allocator
     * allocations used for tensors must be unmanaged native device storage
     * belonging to this exact context and device, and its free operation must
     * remain valid while that context is active.
     */
    [[nodiscard]] std::unique_ptr<Device> make_cuda_device(
            std::uint32_t device_ordinal, Allocator& allocator);

}  // namespace iom
