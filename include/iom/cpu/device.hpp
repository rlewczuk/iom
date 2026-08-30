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
     * queues. The allocator and the returned device must outlive every
     * tensor and queue created through them.
     */
    [[nodiscard]] std::unique_ptr<Device> make_cpu_device(Allocator& allocator);

}  // namespace iom
