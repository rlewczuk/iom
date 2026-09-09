#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "iom/device.hpp"
#include "iom/tensor.hpp"

namespace iom {

    /**
     * TTNN storage accepts `BOOL` and all 21 numeric DataType values below
     * with `QuantizationFormat::NONE`; `F8_E8M0` storage is not required.
     * ADD uses the same full 21-leaf matrix. Non-native leaves may use an
     * internal UINT32 carrier, native 32x32 per-plane tiles, and staging or
     * emulation while preserving public logical shape, ownership, transfers,
     * copy, and queue semantics. Unsupported quantization remains rejected.
     */
    [[nodiscard]] std::span<const DataType> ttnn_supported_data_types() noexcept;

    /**
     * Creates one TTNN device with an owned native device context for the
     * requested backend-local ordinal. TTNN owns native tensor storage, so no
     * iom::Allocator is supplied.
     */
    [[nodiscard]] std::unique_ptr<Device> make_ttnn_device(
            std::uint32_t device_ordinal);

}  // namespace iom
