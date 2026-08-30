#pragma once

#include <cstdint>
#include <memory>

#include "iom/alloc.hpp"
#include "iom/device.hpp"

namespace iom {

    /**
     * Creates one ROCm device with an owned HIP runtime context for the
     * requested backend-local ordinal. The allocator is borrowed and must
     * outlive the returned device and every resource created through it.
     */
    [[nodiscard]] std::unique_ptr<Device> make_rocm_device(
            std::uint32_t device_ordinal, Allocator& allocator);

}  // namespace iom
