#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "iom/device.hpp"
#include "iom/tensor.hpp"

namespace iom {

    /**
     * The one explicit TTNN supported-leaf-type table. create_tensor accepts
     * exactly these DataType values with QuantizationFormat::NONE and rejects
     * every other leaf type before native allocation. The span covers
     * immutable storage; the table is never empty and always contains
     * DataType::BF16.
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
