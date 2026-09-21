#pragma once

#include "iom/tensor.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace iom::detail {

// Full tensor shapes span rank two through rank eight inclusive.
// Implementation-private: no public rank constant or query API exists, and a
// leading-dimension helper span is never a full shape.
inline constexpr std::size_t kMaxTensorRank = 8;

inline std::size_t checked_add(
        std::size_t lhs, std::size_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(what);
    }
    return lhs + rhs;
}

inline std::size_t checked_mul(
        std::size_t lhs, std::size_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(what);
    }
    return lhs * rhs;
}

inline std::size_t padded_extent(
        std::size_t extent, const char* what) {
    const std::size_t remainder = extent % TensorSpec::TILE;
    if (remainder == 0) {
        return extent;
    }
    return checked_add(extent, TensorSpec::TILE - remainder, what);
}

inline std::size_t bits_to_bytes(std::size_t bits, const char* what) {
    return checked_add(bits, 7, what) / 8;
}

struct UnsupportedOperation final : std::runtime_error {
    UnsupportedOperation()
            : std::runtime_error("operation is unsupported") {}
};

/*
 * Shared checked operand admission. Binary and the embedding facade admit an
 * operand's operation-neutral structural facts through this one bounded,
 * allocation-free path: recognized encodings, full-rank shape structure,
 * exact live owner and stable handle, leading-only view metadata,
 * selected-plane bounds, and checked plane/tile/element/bit/byte arithmetic.
 * A successful call builds no vector-backed shape, snapshot, request,
 * registration, or lease and has no backend effect, so a pure requirement
 * query can run it. Nothing here knows an operand role: matching shapes,
 * dtype/quantization policy, broadcasting, aliases, per-operation shapes,
 * and capability stay in the operation layers.
 */

// Declared leaf encodings and declared quantization formats. Both predicates
// are total over their enum and assign no operation support: a recognized but
// inapplicable value is rejected by the operation's capability layer, and an
// unknown value is std::invalid_argument.
[[nodiscard]] bool recognized_data_type(DataType value) noexcept;
[[nodiscard]] bool recognized_quantization(QuantizationFormat value) noexcept;

// Structure of one operand spec: recognized encoding enums, rank two through
// eight, and nonzero extents. `operation` tags rejection text only.
void validate_checked_spec(const TensorSpec& spec, const char* operation);

// Checked facts about one admitted operand view. Every value is produced by
// the validation pass itself with checked add/mul/rounding.
struct CheckedViewFacts {
    // Highest owner plane selected by the view offset and leading strides.
    std::size_t max_plane = 0;
    // Checked owner-relative byte extent through the last selected element.
    std::size_t addressed_bytes = 0;
    // Checked storage bytes of the whole owner in the standard tiled layout.
    std::size_t storage_bytes = 0;
    // Checked logical payload bytes of the view.
    std::size_t logical_bytes = 0;
    // Canonical backing facts, separate from the execution descriptor.
    StorageIdentity backing;
};

// Validate one operand view and report its checked facts: live owner identity
// on the exact queue `device`, stable native handle, view/owner final extents
// and encoding agreement, one positive leading stride per leading extent,
// selected-plane bounds inside the owner, and checked element/bit/byte
// arithmetic for the addressed range, owner storage, and logical payload.
// Rejects with std::invalid_argument and std::overflow_error; allocates,
// registers, leases, and submits nothing. `operation` tags rejection text
// only. Callers run validate_checked_spec first so spec errors keep their
// established precedence over view errors; this helper additionally re-checks
// the rank minimum whose matrix axes it indexes.
[[nodiscard]] CheckedViewFacts validate_checked_view(
        const Device& device, const TensorView& view, const char* operation);

// Defer address-end checks to the operation's existing overlap-validation
// stage: binary and RMSNorm deliberately have different alias policies.
[[nodiscard]] inline StorageRange checked_storage_range(
        const TensorView& view, std::size_t storage_bytes, const char* what) {
    return checked_storage_range(
            StorageAccess::identity(view), 0, storage_bytes, what);
}

}  // namespace iom::detail