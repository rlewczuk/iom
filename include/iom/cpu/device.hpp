#pragma once

#include <memory>

#include "iom/alloc.hpp"
#include "iom/device.hpp"

namespace iom {

    /**
     * Creates one CPU reference device over the caller-supplied storage
     * allocator. The device materializes every unquantized leaf type in the
     * standard 16x16 tiled layout, performs synchronous logical host
     * transfers, and serves any number of independent in-order asynchronous
     * queues. The allocator is borrowed and must outlive the returned
     * device and every resource created through it.
     */
    [[nodiscard]] std::unique_ptr<Device> make_cpu_device(Allocator& allocator);

}  // namespace iom
