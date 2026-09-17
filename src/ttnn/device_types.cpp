#include "device_internal.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>

namespace iom::ttnn_detail {

namespace {

// Native TTNN dtypes are used where they preserve the leaf width. Other NONE
// leaves use a UINT32 carrier; 64-bit leaves occupy two adjacent carrier
// columns per logical element. This keeps one stable TTNN-owned plane per
// leading allocation while conversion remains an internal detail of the
// transfer path.
constexpr auto kSupportedToNative = std::array{
        std::pair{DataType::BOOL, tt::tt_metal::DataType::UINT8},
        std::pair{DataType::I2, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::U2, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::I4, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::U4, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::I8, tt::tt_metal::DataType::UINT8},
        std::pair{DataType::U8, tt::tt_metal::DataType::UINT8},
        std::pair{DataType::I16, tt::tt_metal::DataType::UINT16},
        std::pair{DataType::U16, tt::tt_metal::DataType::UINT16},
        std::pair{DataType::I32, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::U32, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::I64, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::U64, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::F4_E2M1, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::F6_E2M3, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::F6_E3M2, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::F8_E4M3FN, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::F8_E5M2, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::F16, tt::tt_metal::DataType::UINT32},
        std::pair{DataType::BF16, tt::tt_metal::DataType::BFLOAT16},
        std::pair{DataType::F32, tt::tt_metal::DataType::FLOAT32},
        std::pair{DataType::F64, tt::tt_metal::DataType::UINT32},
};

constexpr auto kSupportedKeys = [] {
    std::array<DataType, kSupportedToNative.size()> keys{};
    for (std::size_t i = 0; i < kSupportedToNative.size(); ++i) {
        keys[i] = kSupportedToNative[i].first;
    }
    return keys;
}();

static_assert(kSupportedKeys.size() == kSupportedToNative.size());

constexpr bool kSupportedKeysUnique = [] {
    for (std::size_t i = 0; i < kSupportedToNative.size(); ++i) {
        for (std::size_t j = i + 1; j < kSupportedToNative.size(); ++j) {
            if (kSupportedToNative[i].first == kSupportedToNative[j].first) {
                return false;
            }
        }
    }
    return true;
}();
static_assert(kSupportedKeysUnique);

}  // namespace

bool is_supported(DataType type) {
    const std::span<const DataType> supported = kSupportedKeys;
    return std::find(supported.begin(), supported.end(), type)
           != supported.end();
}
bool rmsnorm_supported(DataType type) noexcept {
    switch (type) {
        case DataType::BF16:
        case DataType::F32:
            return true;
        case DataType::BOOL:
        case DataType::I2:
        case DataType::U2:
        case DataType::I4:
        case DataType::U4:
        case DataType::I8:
        case DataType::U8:
        case DataType::I16:
        case DataType::U16:
        case DataType::I32:
        case DataType::U32:
        case DataType::I64:
        case DataType::U64:
        case DataType::F4_E2M1:
        case DataType::F6_E2M3:
        case DataType::F6_E3M2:
        case DataType::F8_E4M3FN:
        case DataType::F8_E5M2:
        case DataType::F8_E8M0:
        case DataType::F16:
        case DataType::F64:
            return false;
    }
    return false;
}

std::size_t carrier_factor(DataType type) {
    return detail::leaf_bits(type) > 32 ? 2 : 1;
}

// Native tile dtype carrying the leaf encoding or its internal carrier.
// UINT32 is intentionally used for all non-native widths.
tt::tt_metal::DataType native_dtype(DataType type) {
    for (const auto& [supported_type, native_type] : kSupportedToNative) {
        if (supported_type == type) {
            return native_type;
        }
    }
    throw std::invalid_argument("DataType has no TTNN native tile dtype");
}

std::invalid_argument invalid_ordinal(
        std::uint32_t ordinal, std::size_t device_count) {
    return std::invalid_argument(
            "TTNN device ordinal " + std::to_string(ordinal)
            + " is unavailable; device count is "
            + std::to_string(device_count));
}

// Checked narrowing of a TTNN native extent. tt::tt_metal::Shape stores every
// dimension as uint32_t, so a logical dimension beyond that ceiling is
// unreachable on the native path and must be rejected before any native
// object is constructed.
std::uint32_t checked_to_uint32(
        std::size_t dimension, std::size_t index) {
    constexpr std::size_t kMaxNativeExtent =
            std::numeric_limits<std::uint32_t>::max();
    if (dimension > kMaxNativeExtent) {
        throw std::overflow_error(
                "TTNN native dimension " + std::to_string(index)
                + " (" + std::to_string(dimension)
                + ") exceeds the uint32_t native extent limit "
                + std::to_string(kMaxNativeExtent));
    }
    return static_cast<std::uint32_t>(dimension);
}

// Checked leading-plane count for the TTNN creation path: validates the final
// two dimensions against the native extent ceiling and multiplies the leading
// dimensions with checked overflow detection. Pure; runs before any TTNN
// native object or allocation exists.
std::size_t checked_plane_count(const TensorSpec& spec) {
    const std::span<const std::size_t> dimensions = spec.shape.dimensions();
    static_cast<void>(checked_to_uint32(
            dimensions[dimensions.size() - 2], dimensions.size() - 2));
    static_cast<void>(checked_to_uint32(
            dimensions[dimensions.size() - 1], dimensions.size() - 1));
    std::size_t plane_count = 1;
    for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
        const std::size_t dimension = dimensions[i];
        if (dimension != 0
                && plane_count
                        > std::numeric_limits<std::size_t>::max()
                                  / dimension) {
            throw std::overflow_error(
                    "TTNN leading-plane count overflows at dimension "
                    + std::to_string(i));
        }
        plane_count *= dimension;
    }
    return plane_count;
}

}  // namespace iom::ttnn_detail

namespace iom {

std::span<const DataType> ttnn_supported_data_types() noexcept {
    return {ttnn_detail::kSupportedKeys.data(),
            ttnn_detail::kSupportedKeys.size()};
}

}  // namespace iom
