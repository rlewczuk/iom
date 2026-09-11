#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "iom/iom.hpp"

namespace iom::detail {

    // The immutable 23-leaf capability policy shared by every standard
    // 16x16 tiled backend (CPU, CUDA, ROCm, SYCL): exactly the unquantized
    // leaf encodings the common standard-layout transfer implementation can
    // represent. Backend supported_data_types() overrides return spans into
    // this one array, so the advertised leaf set and its order cannot drift
    // between standard backends. The per-backend conformance expected
    // arrays and TTNN's native nine-leaf mapping stay independent so an
    // accidental narrowing or reordering remains detectable.
    inline constexpr std::array<DataType, 23> kStandardSupportedDataTypes = {
            DataType::BOOL,
            DataType::I2, DataType::U2,
            DataType::I4, DataType::U4,
            DataType::I8, DataType::U8,
            DataType::I16, DataType::U16,
            DataType::I32, DataType::U32,
            DataType::I64, DataType::U64,
            DataType::F4_E2M1,
            DataType::F6_E2M3, DataType::F6_E3M2,
            DataType::F8_E4M3FN, DataType::F8_E5M2,
            DataType::F8_E8M0,
            DataType::F16, DataType::BF16,
            DataType::F32, DataType::F64,
    };

    // Read-only span over the single standard capability array; no
    // allocation, conversion, synchronization, or runtime query.
    [[nodiscard]] constexpr std::span<const DataType>
            standard_supported_data_types() noexcept {
        return kStandardSupportedDataTypes;
    }

    template <typename Policy>
    void synchronous_transfer(
            typename Policy::stream_type stream,
            typename Policy::context_type context, const TensorView& view,
            void* staging, std::span<const std::byte> source,
            std::span<std::byte> destination, bool from_host);

}  // namespace iom::detail