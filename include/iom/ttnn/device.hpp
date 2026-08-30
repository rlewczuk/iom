#pragma once

#include <cstdint>
#include <memory>

#include "iom/device.hpp"

namespace iom {

    /**
     * Creates one TTNN device with an owned native device context for the
     * requested backend-local ordinal. TTNN owns native tensor storage, so no
     * iom::Allocator is supplied.
     */
    [[nodiscard]] std::unique_ptr<Device> make_ttnn_device(
            std::uint32_t device_ordinal);

}  // namespace iom
