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
     * ADD, MUL, and SUB accept the same full 21-leaf matrix; DIV accepts the
     * nine floating leaves. Matching BOOL, F8_E8M0, non-NONE quantization,
     * and integer DIV are unsupported after common validation. Non-native
     * leaves may use an internal UINT32 carrier, native 32x32 per-plane tiles,
     * staging, or emulation while preserving public queue semantics.
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
