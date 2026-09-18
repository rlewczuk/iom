#pragma once

// Backend-neutral linear-projection conformance: the independent raw/scalar
// reference, the explicit per-driver capability declaration, and the shared
// observable cases of the operation-owned Linear projections contract in
// `docs/BACKEND_CONTRACT.md`.
//
// The reference is separately encoded. It decodes and encodes every named leaf
// itself, from the leaf's own bit fields with integer arithmetic only, so
// signed zero, gradual underflow, the subnormal boundaries, saturation, and the
// infinity/NaN classes are identical on every host and under every compiler
// floating-point model. Its integer dots reduce modulo `2^N` in unsigned
// arithmetic after every multiply and every add and store the two's-complement
// bit pattern without saturation or conversion; its floating recurrence starts
// at `+0`, walks increasing `i`, accumulates with a single correctly rounded
// FP32 fused multiply-add (FP64 for `F64`) where the host provides one, and
// encodes the accumulated sum exactly once with the existing named-format
// rules. The reference derives the selected row window `x[b, s+r, *]`, the head
// coordinate `(h, d)`, and the Hugging Face weight row `w[o, *]` /
// `w[h*D+d, *]` itself. It never calls production addressing, codec, admission,
// linear, CPU, backend, or capability-report helpers; the only shared test-only
// machinery is the raw bit reader/writer and the independent canonical 16x16
// storage mapping of `backend_conformance_oracle.hpp`.
//
// Two properties keep that recurrence exact wherever the suite runs. For the
// six leaves whose comparison is the exact encoded recurrence, every product of
// two operands is exactly representable in FP32, so a fused and an unfused step
// coincide and the encoded expectation cannot depend on the host's FMA; and for
// `BF16`, `F32`, and `F64` the fixed thresholds are evaluated against the
// independent FP64 equation, whose bounded slack is far wider than the one-ULP
// difference between a fused and an unfused accumulation. The self-check
// asserts both properties instead of assuming a host library's `fma`, `ldexp`,
// or `frexp` precision.
//
// Deliberate negative oracle variants exist only so the self-check can prove
// that a transposed checkpoint weight, a shuffled or repeated head mapping, an
// off-by-one row window, a plane-blind leading block, padded/tile-tail
// addressing, and a recurrence that re-encodes intermediate results are
// detected, and that this detection is selective rather than unconditional.
// A candidate's own output is compared to the reference under the contract's
// fixed fixture thresholds, never under an arbitrary-input accuracy promise.
//
// The harness never switches on BackendKind. Each driver supplies an explicit
// LinearDeclaration naming the leaf matrix its port must reach, the union of
// its scalar and native-BF16 paths, the leaves this revision actually
// implements, the immutable device fact behind a runtime-gated leaf, and its
// exact scratch path. An empty implemented span declares an unported port: the
// suite then observes capability rejection only and never claims projection
// conformance, because a capability rejection, a storage-only observation, and
// a host or elementwise substitute are not projection conformance.

#include "backend/backend_conformance_oracle.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

// ---------------------------------------------------------------------------
// Published matrices and the explicit per-driver declaration.
// ---------------------------------------------------------------------------

// The complete semantic matrix every CPU, CUDA, ROCm, and SYCL port must
// reach: the twelve integer leaves and the nine ordinary signed floating
// leaves. `BOOL` is inapplicable because a boolean is not an integer and not a
// signed numeric product, and `F8_E8M0` is inapplicable because an unsigned
// exponent-only encoding is not a general signed numeric result.
inline constexpr std::array<iom::DataType, 21> kLinearLeafSpan = {
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
        iom::DataType::F16, iom::DataType::BF16,
        iom::DataType::F32, iom::DataType::F64,
};

// The non-BF16 scalar recurrence path: twenty of the twenty-one applicable
// leaves. CPU covers every applicable leaf with this one recurrence, so its
// declaration lists all twenty-one; CUDA, ROCm, and SYCL cover these twenty
// here and `BF16` with a separate native specialization.
inline constexpr std::array<iom::DataType, 20> kLinearScalarLeafSpan = {
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
        iom::DataType::F16,
        iom::DataType::F32, iom::DataType::F64,
};

// The single leaf of a native BF16 specialization.
inline constexpr std::array<iom::DataType, 1> kLinearNativeBf16Span = {
        iom::DataType::BF16,
};

// TTNN's mandatory target: `BF16` alone. Its native TILE compute cannot
// consume the encoded carriers of the other twenty applicable leaves without
// the forbidden host staging, so the port declares and rejects exactly that.
inline constexpr std::array<iom::DataType, 1> kLinearTtnnLeafSpan = {
        iom::DataType::BF16,
};

// The two recognized but inapplicable leaves.
inline constexpr std::array<iom::DataType, 2> kLinearInapplicableSpan = {
        iom::DataType::BOOL,
        iom::DataType::F8_E8M0,
};

// The explicit empty span of an unported path or port.
inline constexpr std::span<const iom::DataType> kNoLinearSpan{};

// The checked logical shape inputs of the frozen workspace formulas: `planes`
// is the checked product `P` of the logical leading extents, `rows` is `R`,
// `inner` is `I`, and `outer` is `O`.
struct LinearShape {
    std::size_t planes = 1;
    std::size_t rows = 1;
    std::size_t inner = 1;
    std::size_t outer = 1;
};

// `pad16(n)`: the checked round-up of `n` to a multiple of 16.
[[nodiscard]] inline std::size_t linear_pad16(std::size_t value) noexcept {
    return value + (16 - value % 16) % 16;
}

// `A32(n)`: the checked round-up of `n` to a multiple of 32, which is the
// alignment the positive paths report.
[[nodiscard]] inline std::size_t linear_a32(std::size_t value) noexcept {
    return value + (32 - value % 32) % 32;
}

// The exact scratch path of one port path, from the frozen workspace table.
enum class LinearWorkspacePath {
    // `{0, 1}`: CPU and CUDA for every applicable leaf, ROCm and SYCL for the
    // twenty scalar leaves, and TTNN for `BF16`.
    zero,
    // ROCm `BF16`: alignment 32 over `A32(P*pad16(R)*pad16(I)*2) +
    // A32(P*pad16(R)*pad16(O)*2)`.
    rocm_bf16,
    // SYCL `BF16`: alignment 32 over the documented rounded/aligned
    // `A32(P*pad16(R)*pad16(O)*4)`.
    sycl_bf16,
};

// The reported requirement of one path for one request, exactly as the frozen
// workspace table states it (`P` is the checked logical-leading product).
[[nodiscard]] inline iom::WorkspaceRequirements linear_workspace_requirement(
        LinearWorkspacePath path, const LinearShape& shape) {
    switch (path) {
        case LinearWorkspacePath::zero:
            return iom::WorkspaceRequirements{0, 1};
        case LinearWorkspacePath::rocm_bf16: {
            const std::size_t rows = linear_pad16(shape.rows);
            const std::size_t x_pack =
                    linear_a32(shape.planes * rows * linear_pad16(shape.inner) * 2);
            const std::size_t y_pack =
                    linear_a32(shape.planes * rows * linear_pad16(shape.outer) * 2);
            return iom::WorkspaceRequirements{x_pack + y_pack, 32};
        }
        case LinearWorkspacePath::sycl_bf16:
            return iom::WorkspaceRequirements{
                    linear_a32(
                            shape.planes * linear_pad16(shape.rows)
                            * linear_pad16(shape.outer) * 4),
                    32};
    }
    return iom::WorkspaceRequirements{0, 1};
}

// One driver's explicit linear capability declaration.
struct LinearDeclaration {
    // The complete leaf matrix this port must reach: `kLinearLeafSpan` on CPU,
    // CUDA, ROCm, and SYCL, and `kLinearTtnnLeafSpan` on TTNN. It is the
    // driver's own statement and never an echo of an implementation report.
    std::span<const iom::DataType> target_leaves;
    // The leaves the port's non-BF16 scalar recurrence path covers.
    std::span<const iom::DataType> scalar_leaves;
    // The leaves an additional native BF16 specialization covers. `BF16` on
    // CUDA, ROCm, SYCL, and TTNN; empty on CPU, whose `BF16` uses the scalar
    // recurrence and whose scratch policy is therefore the scalar one.
    std::span<const iom::DataType> native_bf16_leaves;
    // The leaves this revision's port actually queues. Empty means unported:
    // the suite then asserts capability rejection for every declared leaf,
    // inspects no scratch, and reports no projection success.
    std::span<const iom::DataType> implemented_leaves;
    // One immutable device fact per declared leaf whose availability is a
    // runtime property rather than a missing implementation: SYCL `F64` is
    // queueable only when the selected device reports `sycl::aspect::fp64`,
    // and an absent aspect is a genuine device fact, never an unsupported
    // hardware claim about the port. An empty predicate declares every target
    // leaf available.
    std::function<bool(iom::DataType)> device_available;
    // The exact scratch path of the port's scalar requests and of its `BF16`
    // requests.
    LinearWorkspacePath scalar_workspace = LinearWorkspacePath::zero;
    LinearWorkspacePath bf16_workspace = LinearWorkspacePath::zero;
};

[[nodiscard]] inline bool linear_declares(
        std::span<const iom::DataType> leaves, iom::DataType value) {
    return std::find(leaves.begin(), leaves.end(), value) != leaves.end();
}

// Semantic applicability of one leaf, independently of any backend.
[[nodiscard]] inline bool linear_applicable(iom::DataType leaf) {
    return linear_declares(kLinearLeafSpan, leaf);
}

// A declared leaf is one the port's own two paths cover. The union of the
// scalar and native-BF16 paths is exactly the target matrix.
[[nodiscard]] inline bool linear_declared(
        const LinearDeclaration& declaration, iom::DataType leaf) {
    return linear_declares(declaration.scalar_leaves, leaf)
           || linear_declares(declaration.native_bf16_leaves, leaf);
}

// Availability of one declared leaf on this device: the runtime device fact
// for a gated leaf, and true for every other declared leaf.
[[nodiscard]] inline bool linear_available(
        const LinearDeclaration& declaration, iom::DataType leaf) {
    if (!declaration.device_available) {
        return true;
    }
    return declaration.device_available(leaf);
}

// The one predicate that decides whether a request is compared numerically or
// observed as a capability rejection. A declared leaf that this revision does
// not implement, and a declared leaf the selected device does not expose, both
// stay rejection-only: neither is projection conformance.
[[nodiscard]] inline bool linear_implements(
        const LinearDeclaration& declaration, iom::DataType leaf) {
    return linear_declared(declaration, leaf) && linear_available(declaration, leaf)
           && linear_declares(declaration.implemented_leaves, leaf);
}

// The declared `BF16` path, else the scalar path.
[[nodiscard]] inline LinearWorkspacePath linear_workspace_path(
        const LinearDeclaration& declaration, iom::DataType leaf) {
    if (leaf == iom::DataType::BF16
            && linear_declares(declaration.native_bf16_leaves, leaf)) {
        return declaration.bf16_workspace;
    }
    return declaration.scalar_workspace;
}

// The requirement the declaration predicts for one request, from the frozen
// formulas.
[[nodiscard]] inline iom::WorkspaceRequirements linear_expected_workspace(
        const LinearDeclaration& declaration, iom::DataType leaf,
        const LinearShape& shape) {
    return linear_workspace_requirement(
            linear_workspace_path(declaration, leaf), shape);
}

// The leaves of `probe` the candidate device can store, which is the only set
// a fixture may instantiate: the device's own capability table is the existing,
// backend-neutral answer, and a fixture never assumes a leaf the device would
// refuse to allocate.
[[nodiscard]] inline std::vector<iom::DataType> linear_storable(
        std::span<const iom::DataType> probe,
        std::span<const iom::DataType> supported) {
    std::vector<iom::DataType> leaves;
    for (const iom::DataType leaf : probe) {
        if (linear_declares(supported, leaf)) {
            leaves.push_back(leaf);
        }
    }
    return leaves;
}

// ---------------------------------------------------------------------------
// Independent named-format codec.
//
// Decode and encode are re-derived here from the named format alone: the
// oracle never calls the production scalar codec, so a production rounding,
// saturation, or special-class regression is observable instead of mirrored.
// ---------------------------------------------------------------------------

struct LinearFormat {
    std::size_t bits = 0;
    int ebits = 0;
    int fbits = 0;
    int bias = 0;
    // A format whose finite range is the only representable range: `F4_E2M1`,
    // `F6_E2M3`, `F6_E3M2`, and `F8_E4M3FN` (which has NaN but no infinity).
    bool finite_only = false;
    // A format with an infinity class.
    bool infs = false;

    [[nodiscard]] explicit operator bool() const noexcept { return bits != 0; }
};

[[nodiscard]] inline LinearFormat linear_format(iom::DataType leaf) {
    switch (leaf) {
        case iom::DataType::F4_E2M1: return {4, 2, 1, 1, true, false};
        case iom::DataType::F6_E2M3: return {6, 2, 3, 1, true, false};
        case iom::DataType::F6_E3M2: return {6, 3, 2, 3, true, false};
        case iom::DataType::F8_E4M3FN: return {8, 4, 3, 7, true, false};
        case iom::DataType::F8_E5M2: return {8, 5, 2, 15, false, true};
        case iom::DataType::F16: return {16, 5, 10, 15, false, true};
        case iom::DataType::BF16: return {16, 8, 7, 127, false, true};
        case iom::DataType::F32: return {32, 8, 23, 127, false, true};
        case iom::DataType::F64: return {64, 11, 52, 1023, false, true};
        default: return {};
    }
}

[[nodiscard]] inline bool linear_is_floating(iom::DataType leaf) {
    return static_cast<bool>(linear_format(leaf));
}

[[nodiscard]] inline std::size_t linear_integer_width(iom::DataType leaf) {
    switch (leaf) {
        case iom::DataType::I2: case iom::DataType::U2: return 2;
        case iom::DataType::I4: case iom::DataType::U4: return 4;
        case iom::DataType::I8: case iom::DataType::U8: return 8;
        case iom::DataType::I16: case iom::DataType::U16: return 16;
        case iom::DataType::I32: case iom::DataType::U32: return 32;
        case iom::DataType::I64: case iom::DataType::U64: return 64;
        default: return 0;
    }
}

[[nodiscard]] inline bool linear_is_integer(iom::DataType leaf) {
    return linear_integer_width(leaf) != 0;
}

// The floating leaves whose comparison is the exact encoded result of the
// scalar recurrence rather than a tolerance: `F4_E2M1`, `F6_E2M3`,
// `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, and `F16`.
[[nodiscard]] inline bool linear_exact_encoded(iom::DataType leaf) {
    switch (leaf) {
        case iom::DataType::F4_E2M1:
        case iom::DataType::F6_E2M3:
        case iom::DataType::F6_E3M2:
        case iom::DataType::F8_E4M3FN:
        case iom::DataType::F8_E5M2:
        case iom::DataType::F16:
            return true;
        default:
            return false;
    }
}

// The largest code the leaf's width can carry.
[[nodiscard]] inline std::uint64_t linear_code_mask(iom::DataType leaf) {
    const std::size_t width = linear_integer_width(leaf);
    if (width == 0) {
        const LinearFormat format = linear_format(leaf);
        if (format.bits >= 64) {
            return ~std::uint64_t{0};
        }
        return (std::uint64_t{1} << format.bits) - 1;
    }
    return width >= 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << width) - 1);
}

// Assemble `significand * 2^scale` as a double from bit fields with integer
// arithmetic only. Every value a named format can carry is exactly
// representable in a double, so no rounding happens here and the result is
// identical on every host and every floating-point model: nothing depends on
// the host library's scaling, on a flush-to-zero mode, or on a fast-math
// contraction.
[[nodiscard]] inline double linear_assemble(
        std::uint64_t significand, int scale) {
    if (significand == 0) {
        return 0.0;
    }
    int leading = 63;
    while (((significand >> leading) & 1u) == 0) {
        --leading;
    }
    const int value_exponent = leading + scale;
    const std::uint64_t mantissa_mask = (std::uint64_t{1} << 52) - 1;
    if (value_exponent < -1022) {
        const int shift = scale + 1074;
        if (shift < 0 || shift > 52) {
            throw std::out_of_range(
                    "linear_assemble: value is not representable in a double");
        }
        return std::bit_cast<double>(significand << shift);
    }
    const int shift = leading - 52;
    const std::uint64_t mantissa = (shift >= 0 ? (significand >> shift)
                                              : (significand << -shift))
                                  & mantissa_mask;
    const auto field = static_cast<std::uint64_t>(value_exponent + 1023);
    return std::bit_cast<double>((field << 52) | mantissa);
}

// Exact value of one named-format code as a double, assembled from the leaf's
// own bit fields with integer arithmetic only: signed zero, gradual underflow,
// the subnormal boundaries, and the format's infinity and NaN classes are all
// reproduced identically on every host and every floating-point model.
[[nodiscard]] inline double linear_decode_code(
        iom::DataType leaf, std::uint64_t raw) {
    const LinearFormat format = linear_format(leaf);
    if (!format) {
        throw std::invalid_argument("linear_decode_code: not a floating leaf");
    }
    raw &= linear_code_mask(leaf);
    const std::uint64_t sign = raw >> (format.ebits + format.fbits);
    const std::uint64_t exponent_mask = (std::uint64_t{1} << format.ebits) - 1;
    const std::uint64_t fraction =
            raw & ((std::uint64_t{1} << format.fbits) - 1);
    const std::uint64_t exponent = (raw >> format.fbits) & exponent_mask;
    if (exponent == exponent_mask
            && (!format.finite_only || format.ebits >= 4)) {
        if (format.infs && fraction == 0) {
            return sign ? -std::numeric_limits<double>::infinity()
                        : std::numeric_limits<double>::infinity();
        }
        // The existing named-format rules give every such encoding the NaN
        // class; no payload is preserved.
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::uint64_t significand = exponent
            ? ((std::uint64_t{1} << format.fbits) + fraction)
            : fraction;
    const int scale = exponent
            ? static_cast<int>(exponent) - format.bias - format.fbits
            : 1 - format.bias - format.fbits;
    const double magnitude = linear_assemble(significand, scale);
    return sign ? -magnitude : magnitude;
}

// `significand * 2^shift` rounded to an integer with round-to-nearest,
// ties-to-even. A non-negative shift is an exact left shift.
[[nodiscard]] inline std::uint64_t linear_scale_round_even(
        std::uint64_t significand, int shift) {
    if (shift == 0) {
        return significand;
    }
    if (shift > 0) {
        if (shift >= 64 || (significand >> (64 - shift)) != 0) {
            throw std::out_of_range(
                    "linear_scale_round_even: magnitude exceeds the target");
        }
        return significand << shift;
    }
    const int drop = -shift;
    if (drop >= 64) {
        return 0;
    }
    const std::uint64_t quotient = significand >> drop;
    const std::uint64_t remainder =
            significand & ((std::uint64_t{1} << drop) - 1);
    const std::uint64_t half = std::uint64_t{1} << (drop - 1);
    if (remainder > half) {
        return quotient + 1;
    }
    if (remainder == half) {
        return quotient + ((quotient & 1u) != 0 ? std::uint64_t{1}
                                               : std::uint64_t{0});
    }
    return quotient;
}

// One named-format encode of `value`, with the existing rules for infinity,
// NaN, saturation, gradual underflow, and signed zero. `F4_E2M1`, `F6_E2M3`,
// and `F6_E3M2` saturate every NaN and every finite overflow to their largest
// finite magnitude; `F8_E4M3FN` has NaN but no infinity, so its finite
// overflow saturates to its largest finite magnitude and its NaN is the format
// NaN encoding; the remaining formats have both classes. Addressable
// subnormals are produced gradually, with no flush-to-zero. The rounding is
// integer-only, so the result never depends on the host's floating-point
// model.
[[nodiscard]] inline std::uint64_t linear_encode_value(
        iom::DataType leaf, double value) {
    const LinearFormat format = linear_format(leaf);
    if (!format) {
        throw std::invalid_argument("linear_encode_value: not a floating leaf");
    }
    const std::uint64_t exponent_mask =
            (std::uint64_t{1} << format.ebits) - 1;
    const std::uint64_t fraction_mask =
            (std::uint64_t{1} << format.fbits) - 1;
    const std::uint64_t finite_exponent_max =
            (format.finite_only && format.ebits < 4) ? exponent_mask
                                                     : exponent_mask - 1;
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    const std::uint64_t sign = bits >> 63;
    const auto pack = [&](std::uint64_t exponent, std::uint64_t fraction) {
        return (sign << (format.ebits + format.fbits))
               | (exponent << format.fbits) | fraction;
    };
    const std::uint64_t overflow = format.infs
            ? pack(exponent_mask, 0)
            : pack(finite_exponent_max, fraction_mask);
    const std::uint64_t field = (bits >> 52) & 0x7FF;
    const std::uint64_t double_fraction = bits & ((std::uint64_t{1} << 52) - 1);
    if (field == 0x7FF) {
        if (double_fraction != 0) {
            if (format.finite_only && format.ebits < 4) {
                // A finite-only format has no NaN encoding; the existing rules
                // saturate the class to the largest finite magnitude instead
                // of inventing one.
                return pack(finite_exponent_max, fraction_mask);
            }
            return pack(
                    exponent_mask,
                    format.finite_only
                            ? fraction_mask
                            : (std::uint64_t{1} << (format.fbits - 1)));
        }
        return overflow;
    }
    const std::uint64_t significand = field == 0
            ? double_fraction
            : ((std::uint64_t{1} << 52) | double_fraction);
    if (significand == 0) {
        return sign << (format.ebits + format.fbits);
    }
    const int scale = field == 0 ? -1074
                                 : static_cast<int>(field) - 1023 - 52;
    int leading = 63;
    while (((significand >> leading) & 1u) == 0) {
        --leading;
    }
    const int value_exponent = leading + scale;
    const int minimum_subnormal_exponent = 1 - format.bias - format.fbits;
    const int maximum_exponent =
            static_cast<int>(finite_exponent_max) - format.bias;
    if (value_exponent < 1 - format.bias) {
        // Subnormal target: the quantum is `2^minimum_subnormal_exponent`.
        const std::uint64_t quantized = linear_scale_round_even(
                significand, scale - minimum_subnormal_exponent);
        if (quantized == 0) {
            return sign << (format.ebits + format.fbits);
        }
        if (quantized >= (std::uint64_t{1} << format.fbits)) {
            return pack(1, 0);
        }
        return pack(0, quantized);
    }
    if (value_exponent > maximum_exponent) {
        return overflow;
    }
    // Normal target: keep exactly `fbits + 1` significant bits, rounded once.
    std::uint64_t result =
            linear_scale_round_even(significand, format.fbits - leading);
    int exponent = value_exponent;
    if (result == (std::uint64_t{1} << (format.fbits + 1))) {
        result >>= 1;
        ++exponent;
    }
    if (exponent > maximum_exponent) {
        return overflow;
    }
    return pack(
            static_cast<std::uint64_t>(exponent + format.bias),
            result & fraction_mask);
}

// ---------------------------------------------------------------------------
// Fixture codes: the deterministic logical content of one operand, and the
// deliberate non-neutral content of every non-logical cell.
// ---------------------------------------------------------------------------

// The ordered special ladder of one leaf. The twelve integer leaves carry the
// zero, unit, signed-extreme, and all-ones codes in eight positions; the nine
// floating leaves carry signed zero, the subnormal and normal boundaries, the
// unit values, the largest finite magnitude, and the representable
// infinity/NaN classes in twelve positions, where a finite-only format maps
// its unrepresentable infinity and NaN positions onto its largest finite
// magnitude.
[[nodiscard]] inline std::size_t linear_ladder_size(iom::DataType leaf) {
    return linear_is_integer(leaf) ? 8 : 12;
}

[[nodiscard]] inline std::uint64_t linear_ladder_code(
        iom::DataType leaf, std::size_t position) {
    if (linear_is_integer(leaf)) {
        const std::size_t width = linear_integer_width(leaf);
        const std::uint64_t mask = width >= 64
                ? ~std::uint64_t{0}
                : ((std::uint64_t{1} << width) - 1);
        const std::uint64_t sign_bit = std::uint64_t{1} << (width - 1);
        switch (position % 8) {
            case 0: return 0;
            case 1: return 1;
            case 2: return sign_bit - 1;
            case 3: return sign_bit;
            case 4: return mask;
            case 5: return 2;
            case 6: return mask - 1;
            default: return sign_bit + 1;
        }
    }
    const LinearFormat format = linear_format(leaf);
    const std::uint64_t exponent_mask =
            (std::uint64_t{1} << format.ebits) - 1;
    const std::uint64_t fraction_mask =
            (std::uint64_t{1} << format.fbits) - 1;
    const std::uint64_t finite_exponent_max =
            (format.finite_only && format.ebits < 4) ? exponent_mask
                                                     : exponent_mask - 1;
    const std::uint64_t sign_shift = format.ebits + format.fbits;
    const std::uint64_t max_finite =
            (finite_exponent_max << format.fbits) | fraction_mask;
    switch (position % 12) {
        case 0: return 0;
        case 1: return std::uint64_t{1} << sign_shift;
        case 2: return 1;
        case 3: return fraction_mask;
        case 4: return std::uint64_t{1} << format.fbits;
        case 5: return static_cast<std::uint64_t>(format.bias) << format.fbits;
        case 6:
            return (std::uint64_t{1} << sign_shift)
                   | (static_cast<std::uint64_t>(format.bias)
                      << format.fbits);
        case 7: return max_finite;
        case 8: return (std::uint64_t{1} << sign_shift) | max_finite;
        case 9:
            return format.infs ? (exponent_mask << format.fbits) : max_finite;
        case 10:
            if (format.infs) {
                return (exponent_mask << format.fbits)
                       | (std::uint64_t{1} << (format.fbits - 1));
            }
            return format.finite_only && format.ebits >= 4
                    ? ((exponent_mask << format.fbits) | fraction_mask)
                    : max_finite;
        default:
            return (std::uint64_t{1} << sign_shift)
                   | (std::uint64_t{1} << format.fbits);
    }
}

// Deterministic code of one logical fixture cell. Every third cell takes the
// next position of the leaf's special ladder (so every fixture carries the
// leaf's special classes), and the remaining cells take an ordinary salted
// value whose floating exponent stays near unity so an accumulation stays
// interesting instead of saturating immediately.
[[nodiscard]] inline std::uint64_t linear_fixture_code(
        iom::DataType leaf, std::size_t index, std::uint64_t salt,
        bool specials = true) {
    if (specials && index % 3 == 2) {
        return linear_ladder_code(
                leaf, (index / 3 + static_cast<std::size_t>(salt))
                              % linear_ladder_size(leaf));
    }
    const std::uint64_t raw = splitmix64(
            splitmix64(static_cast<std::uint64_t>(index) * 0x9E3779B97F4A7C15ull)
            ^ (salt + 1));
    if (linear_is_integer(leaf)) {
        return raw & linear_code_mask(leaf);
    }
    const LinearFormat format = linear_format(leaf);
    const std::uint64_t exponent_mask =
            (std::uint64_t{1} << format.ebits) - 1;
    const std::uint64_t fraction =
            raw & ((std::uint64_t{1} << format.fbits) - 1);
    int exponent = format.bias - 2 + static_cast<int>((raw >> 11) % 5);
    if (exponent < 1) {
        exponent = 1;
    }
    if (exponent > static_cast<int>(exponent_mask) - 1) {
        exponent = static_cast<int>(exponent_mask) - 1;
    }
    const std::uint64_t sign = (raw >> 63) & 1;
    return (sign << (format.ebits + format.fbits))
           | (static_cast<std::uint64_t>(exponent) << format.fbits)
           | fraction;
}

// Deliberate non-neutral code of one non-logical cell (16x16 tile padding, a
// sub-byte remainder bit, or an uninitialized byte): a negative maximum
// magnitude for a floating leaf and the all-ones code for an integer leaf, so
// a reader that consults padding, a tile tail, or uninitialized storage is
// observably wrong rather than accidentally neutral.
[[nodiscard]] inline std::uint64_t linear_poison_code(
        iom::DataType leaf, std::size_t index) {
    if (linear_is_integer(leaf)) {
        return linear_code_mask(leaf) ^ (static_cast<std::uint64_t>(index) & 1u);
    }
    const LinearFormat format = linear_format(leaf);
    const std::uint64_t sign = std::uint64_t{1}
                               << (format.ebits + format.fbits);
    return sign | linear_ladder_code(leaf, 7);
}

// ---------------------------------------------------------------------------
// Padded raw images.
//
// One image is `planes` planes of `rows` rows and `columns` columns at
// `bits_of(leaf)`, least-significant bit first, where the logical
// `[0, logical_rows) x [0, logical_columns)` cells carry the fixture content
// and every other cell carries poison. This is the geometry a padded backend
// tile exposes, so a masked reader is distinguishable from a tile-tail or
// padding-dependent one, and a reader that ignores the poison is exactly the
// one the contract requires.
// ---------------------------------------------------------------------------

struct LinearRawImage {
    iom::DataType leaf = iom::DataType::BF16;
    std::size_t planes = 0;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t logical_rows = 0;
    std::size_t logical_columns = 0;
    std::vector<std::byte> bytes;

    [[nodiscard]] std::size_t cells() const noexcept {
        return planes * rows * columns;
    }
    [[nodiscard]] std::size_t cell(
            std::size_t plane, std::size_t row, std::size_t column) const {
        return (plane * rows + row) * columns + column;
    }
};

[[nodiscard]] inline std::uint64_t linear_image_code(
        const LinearRawImage& image, std::size_t cell) {
    if (cell >= image.cells()) {
        throw std::out_of_range("linear image read exceeds the image");
    }
    const std::size_t bits = bits_of(image.leaf);
    return read_storage_bits(image.bytes.data(), cell * bits, bits);
}

inline void linear_write_image_code(
        LinearRawImage& image, std::size_t cell, std::uint64_t code) {
    if (cell >= image.cells()) {
        throw std::out_of_range("linear image write exceeds the image");
    }
    const std::size_t bits = bits_of(image.leaf);
    write_storage_bits(image.bytes.data(), cell * bits, bits, code);
}

// Padded image whose logical cells come from a row-major logical image
// (`planes * logical_rows * logical_columns` codes, least-significant bit
// first) and whose every other cell is poison.
[[nodiscard]] inline LinearRawImage linear_padded_image(
        iom::DataType leaf, std::size_t planes, std::size_t rows,
        std::size_t columns, std::size_t logical_rows,
        std::size_t logical_columns, std::span<const std::byte> logical,
        std::uint64_t poison_salt = 0x9E3779B97F4A7C15ull) {
    const std::size_t bits = bits_of(leaf);
    REQUIRE(rows >= logical_rows);
    REQUIRE(columns >= logical_columns);
    REQUIRE_EQ(
            logical.size(),
            (planes * logical_rows * logical_columns * bits + 7) / 8);
    LinearRawImage image;
    image.leaf = leaf;
    image.planes = planes;
    image.rows = rows;
    image.columns = columns;
    image.logical_rows = logical_rows;
    image.logical_columns = logical_columns;
    image.bytes.assign((planes * rows * columns * bits + 7) / 8, std::byte{0});
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t target = image.cell(plane, row, column);
                if (row < logical_rows && column < logical_columns) {
                    const std::size_t source =
                            (plane * logical_rows + row) * logical_columns
                            + column;
                    linear_write_image_code(
                            image, target,
                            read_storage_bits(logical.data(), source * bits, bits));
                    continue;
                }
                const std::size_t poison = splitmix64(
                        (target + poison_salt) * 0x2545F4914F6CDD1Dull);
                linear_write_image_code(
                        image, target, linear_poison_code(leaf, poison));
            }
        }
    }
    return image;
}

// Row-major logical image of one fixture's deterministic content.
[[nodiscard]] inline std::vector<std::byte> linear_logical_image(
        iom::DataType leaf, std::size_t elements, std::uint64_t salt,
        bool specials = true) {
    const std::size_t bits = bits_of(leaf);
    std::vector<std::byte> image((elements * bits + 7) / 8, std::byte{0});
    for (std::size_t linear = 0; linear < elements; ++linear) {
        write_storage_bits(
                image.data(), linear * bits, bits,
                linear_fixture_code(leaf, linear, salt, specials));
    }
    return image;
}

// Whole canonical fixture operand: `planes` planes of `logical_rows` x
// `logical_columns` deterministic cells, padded to the next 16-wide tile
// boundary in both tiled axes with poison. `salt` selects which logical plane
// a plane-blind oracle reads, so independent planes stay observable.
[[nodiscard]] inline LinearRawImage linear_fixture_image(
        iom::DataType leaf, std::size_t planes, std::size_t logical_rows,
        std::size_t logical_columns, std::uint64_t salt,
        bool specials = true,
        std::uint64_t poison_salt = 0x9E3779B97F4A7C15ull) {
    const std::vector<std::byte> logical =
            linear_logical_image(leaf, planes * logical_rows * logical_columns,
                                 salt, specials);
    return linear_padded_image(
            leaf, planes, linear_pad16(logical_rows),
            linear_pad16(logical_columns), logical_rows, logical_columns,
            logical, poison_salt);
}

// ---------------------------------------------------------------------------
// Leading-only view mapping, from the public view metadata alone.
// ---------------------------------------------------------------------------

// Owner plane of one logical plane of a leading-only view: the selected plane
// offset plus the view's own leading strides over its own leading dimensions.
[[nodiscard]] inline std::size_t linear_view_plane(
        const iom::TensorView& view, std::size_t logical_plane) {
    const std::span<const std::size_t> dims = view.spec().shape.dimensions();
    const std::size_t leading = dims.size() - 2;
    const std::span<const std::size_t> strides = view.plane_strides();
    REQUIRE_EQ(strides.size(), leading);
    std::size_t plane = view.plane_offset();
    std::size_t rest = logical_plane;
    for (std::size_t axis = leading; axis-- > 0;) {
        plane += (rest % dims[axis]) * strides[axis];
        rest /= dims[axis];
    }
    return plane;
}

[[nodiscard]] inline std::size_t linear_view_planes(
        const iom::TensorView& view) {
    const std::span<const std::size_t> dims = view.spec().shape.dimensions();
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis + 2 < dims.size(); ++axis) {
        planes *= dims[axis];
    }
    return planes;
}

// Elements of one logical plane, which every leading-only transform leaves
// unchanged.
[[nodiscard]] inline std::size_t linear_plane_elements(
        const iom::TensorView& view) {
    const std::span<const std::size_t> dims = view.spec().shape.dimensions();
    REQUIRE(dims.size() >= 2);
    return dims[dims.size() - 2] * dims[dims.size() - 1];
}

// Logical image of one leading-only view over one owner image: the view's own
// plane offset and strides select the owner plane and the final two
// coordinates stay unchanged.
[[nodiscard]] inline std::vector<std::byte> linear_view_image(
        const iom::TensorView& view, std::span<const std::byte> owner_image) {
    const std::size_t bits = bits_of(view.spec().data_type);
    const std::size_t planes = linear_view_planes(view);
    const std::size_t plane_elements = linear_plane_elements(view);
    REQUIRE_EQ(
            owner_image.size(),
            view.owner_identity()->view().spec().logical_nbytes());
    std::vector<std::byte> image(view.spec().logical_nbytes(), std::byte{0});
    for (std::size_t plane = 0; plane < planes; ++plane) {
        const std::size_t owner_plane = linear_view_plane(view, plane);
        for (std::size_t element = 0; element < plane_elements; ++element) {
            const std::uint64_t value = read_storage_bits(
                    owner_image.data(),
                    (owner_plane * plane_elements + element) * bits, bits);
            write_storage_bits(
                    image.data(), (plane * plane_elements + element) * bits,
                    bits, value);
        }
    }
    return image;
}

// Padded image whose every cell comes from an explicit code function:
// `logical(plane, row, column)` for an in-bounds cell and `poison(plane, row,
// column)` for a non-logical one. The self-check uses it for deliberately
// degenerate fixtures (for example row-invariant content).
template <typename LogicalCode, typename PoisonCode>
[[nodiscard]] inline LinearRawImage linear_coded_image(
        iom::DataType leaf, std::size_t planes, std::size_t rows,
        std::size_t columns, std::size_t logical_rows,
        std::size_t logical_columns, LogicalCode logical, PoisonCode poison) {
    const std::size_t bits = bits_of(leaf);
    LinearRawImage image;
    image.leaf = leaf;
    image.planes = planes;
    image.rows = rows;
    image.columns = columns;
    image.logical_rows = logical_rows;
    image.logical_columns = logical_columns;
    image.bytes.assign((planes * rows * columns * bits + 7) / 8, std::byte{0});
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const std::uint64_t code =
                        row < logical_rows && column < logical_columns
                        ? logical(plane, row, column)
                        : poison(plane, row, column);
                write_storage_bits(
                        image.bytes.data(),
                        image.cell(plane, row, column) * bits, bits, code);
            }
        }
    }
    return image;
}

// ---------------------------------------------------------------------------
// The independent reference.
// ---------------------------------------------------------------------------

// Deliberate negative variants. They exist only so the self-check can prove
// that the reference detects a transposed checkpoint weight, a shuffled or
// repeated head mapping, an off-by-one selected row window, a plane-blind
// leading block, padded/tile-tail addressing, and a recurrence that re-encodes
// intermediate results, and that this detection is selective rather than an
// unconditional "differs".
enum class LinearOracleVariant {
    honest,
    // Reads the Hugging Face `w[O,I]` weight as if it were `w[I,O]`, which is
    // the checkpoint transpose the contract forbids.
    transposed_weight,
    // Scatters head `h`, column `d` through `o = d*H + h` instead of
    // `o = h*D + d`.
    head_order,
    // Computes head zero and repeats it for every head.
    head_repeat,
    // Selects `s+1+r` instead of `s+r`: an off-by-one row window.
    wrong_row_window,
    // Reads leading plane zero for every plane.
    mixed_plane,
    // Addresses each leading plane with the logical row count instead of the
    // padded one, so tile padding participates in the addressing.
    padding_dependent,
    // Extends the inner reduction over the padded 16-wide tile tail.
    tile_tail,
    // Rounds every partial product and every partial sum to the output leaf
    // instead of accumulating in the wide domain and encoding once.
    numeric_reencode,
};

struct LinearOracleRequest {
    iom::DataType leaf = iom::DataType::BF16;
    std::size_t planes = 1;
    std::size_t source_rows = 0;
    std::size_t inner = 0;
    std::size_t outer = 0;
    std::size_t start_row = 0;
    std::size_t rows = 0;
    std::size_t heads = 1;
    std::size_t head_dim = 1;
    iom::LinearOutputLayout layout = iom::LinearOutputLayout::ordinary;

    // Elements of one logical output plane: `R*O` in ordinary mode and `R*D`
    // per head in head-planar mode.
    [[nodiscard]] std::size_t plane_elements() const noexcept {
        return layout == iom::LinearOutputLayout::ordinary ? rows * outer
                                                           : rows * head_dim;
    }
    // Logical output planes: the leading planes in ordinary mode, and one plane
    // per `(leading plane, head)` in head-planar mode.
    [[nodiscard]] std::size_t logical_planes() const noexcept {
        return layout == iom::LinearOutputLayout::ordinary ? planes
                                                           : planes * heads;
    }
    [[nodiscard]] std::size_t elements() const noexcept {
        return logical_planes() * plane_elements();
    }
    [[nodiscard]] std::size_t output_columns() const noexcept {
        return layout == iom::LinearOutputLayout::ordinary ? outer : head_dim;
    }
};

struct LinearReference {
    // Exact result of the contract's scalar recurrence: one code per logical
    // output element, in the output view's own logical order (`out[...,R,O]`,
    // or `out[...,H,R,D]` with the head axis ahead of the row axis and the
    // leading planes ahead of the heads).
    std::vector<std::uint64_t> codes;
    // Independent FP64 evaluation of the same equation, same order, for the
    // floating leaves. Empty for the twelve integer leaves, whose comparison is
    // exact-bit.
    std::vector<double> values;
};

// The unsigned modulo-`2^N` dot: every multiply and every add reduces modulo
// `2^N`, so no step depends on signed overflow, no value is ever cast to an
// out-of-range signed type, and the stored code is the two's-complement bit
// pattern of the accumulated value.
[[nodiscard]] inline std::uint64_t linear_integer_recurrence(
        iom::DataType leaf, std::span<const std::uint64_t> x_codes,
        std::span<const std::uint64_t> w_codes) {
    const std::size_t width = linear_integer_width(leaf);
    REQUIRE(width != 0);
    REQUIRE_EQ(x_codes.size(), w_codes.size());
    const std::uint64_t mask = width >= 64
            ? ~std::uint64_t{0}
            : ((std::uint64_t{1} << width) - 1);
    std::uint64_t accumulator = 0;
    for (std::size_t i = 0; i < x_codes.size(); ++i) {
        const std::uint64_t product =
                (x_codes[i] & mask) * (w_codes[i] & mask);
        accumulator = (accumulator + product) & mask;
    }
    return accumulator;
}

// The exact scalar recurrence: `+0`, increasing `i`, one correctly rounded
// FP32 fused multiply-add per step (`F64` accumulates in FP64), and exactly one
// named-format encode of the accumulated sum.
[[nodiscard]] inline std::uint64_t linear_scalar_recurrence(
        iom::DataType leaf, std::span<const std::uint64_t> x_codes,
        std::span<const std::uint64_t> w_codes) {
    REQUIRE(linear_is_floating(leaf));
    REQUIRE_EQ(x_codes.size(), w_codes.size());
    if (leaf == iom::DataType::F64) {
        double accumulator = 0.0;
        for (std::size_t i = 0; i < x_codes.size(); ++i) {
            accumulator = std::fma(
                    linear_decode_code(leaf, x_codes[i]),
                    linear_decode_code(leaf, w_codes[i]), accumulator);
        }
        return linear_encode_value(leaf, accumulator);
    }
    float accumulator = 0.0f;
    for (std::size_t i = 0; i < x_codes.size(); ++i) {
        accumulator = std::fma(
                static_cast<float>(linear_decode_code(leaf, x_codes[i])),
                static_cast<float>(linear_decode_code(leaf, w_codes[i])),
                accumulator);
    }
    return linear_encode_value(leaf, static_cast<double>(accumulator));
}

// The independent FP64 evaluation of the same equation: every decoded operand
// is exact in a double, so this is the mathematical sum of the decoded
// products under IEEE FP64 accumulation.
[[nodiscard]] inline double linear_wide_value(
        iom::DataType leaf, std::span<const std::uint64_t> x_codes,
        std::span<const std::uint64_t> w_codes) {
    REQUIRE(linear_is_floating(leaf));
    REQUIRE_EQ(x_codes.size(), w_codes.size());
    double accumulator = 0.0;
    for (std::size_t i = 0; i < x_codes.size(); ++i) {
        accumulator = std::fma(
                linear_decode_code(leaf, x_codes[i]),
                linear_decode_code(leaf, w_codes[i]), accumulator);
    }
    return accumulator;
}

// The re-encoding recurrence: every partial product and every partial sum is
// rounded to the output leaf. For the integer leaves the round trip is exact
// and for a single product the two recurrences coincide, which is what makes
// the self-check's detection selective.
[[nodiscard]] inline std::uint64_t linear_reencoded_recurrence(
        iom::DataType leaf, std::span<const std::uint64_t> x_codes,
        std::span<const std::uint64_t> w_codes) {
    if (linear_is_integer(leaf)) {
        return linear_integer_recurrence(leaf, x_codes, w_codes);
    }
    std::uint64_t accumulator = linear_encode_value(leaf, 0.0);
    for (std::size_t i = 0; i < x_codes.size(); ++i) {
        const double product = linear_decode_code(leaf, x_codes[i])
                               * linear_decode_code(leaf, w_codes[i]);
        const std::uint64_t product_code = linear_encode_value(leaf, product);
        const double partial = linear_decode_code(leaf, accumulator)
                               + linear_decode_code(leaf, product_code);
        accumulator = linear_encode_value(leaf, partial);
    }
    return accumulator;
}

// Expected logical output of one request, built from separately encoded raw
// operand images and the independent scalar recurrence above. The reference
// derives the selected row window, the head coordinate, and the Hugging Face
// weight row itself, and it never consults a production helper.
[[nodiscard]] inline LinearReference linear_reference(
        const LinearOracleRequest& request, const LinearRawImage& x,
        const LinearRawImage& w,
        LinearOracleVariant variant = LinearOracleVariant::honest) {
    REQUIRE(x.leaf == request.leaf);
    REQUIRE(w.leaf == request.leaf);
    REQUIRE_EQ(x.planes, request.planes);
    REQUIRE_EQ(x.logical_rows, request.source_rows);
    REQUIRE_EQ(x.logical_columns, request.inner);
    REQUIRE_EQ(w.logical_rows, request.outer);
    REQUIRE_EQ(w.logical_columns, request.inner);
    REQUIRE_EQ(w.planes, std::size_t{1});
    REQUIRE(request.rows > 0);
    REQUIRE(request.start_row + request.rows <= request.source_rows);
    REQUIRE(request.heads > 0);
    REQUIRE(request.head_dim > 0);
    if (request.layout == iom::LinearOutputLayout::ordinary) {
        REQUIRE_EQ(request.heads, std::size_t{1});
        REQUIRE_EQ(request.head_dim, request.outer);
    } else {
        REQUIRE_EQ(request.heads * request.head_dim, request.outer);
    }

    const bool integer = linear_is_integer(request.leaf);
    const std::size_t inner_span = variant == LinearOracleVariant::tile_tail
            ? linear_pad16(request.inner)
            : request.inner;
    REQUIRE(inner_span <= x.columns);
    REQUIRE(inner_span <= w.columns);

    LinearReference reference;
    reference.codes.assign(request.elements(), 0);
    if (!integer) {
        reference.values.assign(request.elements(), 0.0);
    }

    std::vector<std::uint64_t> x_codes(inner_span);
    std::vector<std::uint64_t> w_codes(inner_span);
    for (std::size_t plane = 0; plane < request.planes; ++plane) {
        const std::size_t read_plane =
                variant == LinearOracleVariant::mixed_plane ? 0 : plane;
        for (std::size_t row = 0; row < request.rows; ++row) {
            std::size_t source_row = request.start_row + row;
            if (variant == LinearOracleVariant::wrong_row_window) {
                source_row += 1;
                if (source_row >= x.rows) {
                    source_row = x.rows - 1;
                }
            }
            REQUIRE_LT(source_row, x.rows);
            const std::size_t plane_stride =
                    variant == LinearOracleVariant::padding_dependent
                    ? request.source_rows * x.columns
                    : x.rows * x.columns;
            for (std::size_t i = 0; i < inner_span; ++i) {
                x_codes[i] = linear_image_code(
                        x, read_plane * plane_stride + source_row * x.columns
                                   + i);
            }
            const std::size_t heads =
                    request.layout == iom::LinearOutputLayout::ordinary
                    ? 1
                    : request.heads;
            for (std::size_t head = 0; head < heads; ++head) {
                const std::size_t logical_plane =
                        request.layout == iom::LinearOutputLayout::ordinary
                        ? plane
                        : plane * request.heads + head;
                const std::size_t columns = request.output_columns();
                for (std::size_t column = 0; column < columns; ++column) {
                    std::size_t weight_row = column;
                    if (request.layout == iom::LinearOutputLayout::head_planar) {
                        if (variant == LinearOracleVariant::head_order) {
                            weight_row = column * request.heads + head;
                        } else if (variant
                                   == LinearOracleVariant::head_repeat) {
                            weight_row = column;
                        } else {
                            weight_row = head * request.head_dim + column;
                        }
                    }
                    REQUIRE_LT(weight_row, w.rows);
                    for (std::size_t i = 0; i < inner_span; ++i) {
                        const std::size_t w_cell =
                                variant == LinearOracleVariant::transposed_weight
                                ? i * request.outer + weight_row
                                : weight_row * w.columns + i;
                        w_codes[i] = linear_image_code(w, w_cell);
                    }
                    const std::size_t index =
                            logical_plane * request.plane_elements()
                            + row * columns + column;
                    if (integer) {
                        reference.codes[index] = linear_integer_recurrence(
                                request.leaf, x_codes, w_codes);
                        continue;
                    }
                    reference.values[index] = linear_wide_value(
                            request.leaf, x_codes, w_codes);
                    reference.codes[index] =
                            variant == LinearOracleVariant::numeric_reencode
                            ? linear_reencoded_recurrence(
                                      request.leaf, x_codes, w_codes)
                            : linear_scalar_recurrence(
                                      request.leaf, x_codes, w_codes);
                }
            }
        }
    }
    return reference;
}

// ---------------------------------------------------------------------------
// Comparison against the fixed fixture thresholds.
// ---------------------------------------------------------------------------

// `ULP_BF16(q)`: the spacing of the BF16 format at `q`, and the smallest
// subnormal magnitude at zero and in the subnormal range.
[[nodiscard]] inline double linear_bf16_ulp(double value) noexcept {
    if (value == 0.0 || std::fpclassify(value) == FP_SUBNORMAL) {
        return std::ldexp(1.0, -133);
    }
    const int exponent = std::ilogb(value);
    if (exponent < -126) {
        return std::ldexp(1.0, -133);
    }
    return std::ldexp(1.0, exponent - 7);
}

// One element's comparison under the contract's fixed per-leaf policy:
//
//   * the twelve integer leaves: exact output bits;
//   * `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`: the
//     exact encoded result of the scalar recurrence;
//   * `BF16`: `abs(actual - q) <= max(ULP_BF16(q), 2^-7, 2^-6*abs(q))` where
//     `q` is the independent FP64 equation rounded once to BF16;
//   * `F32`: `abs(actual - reference) <= 1e-5 + 1e-5*abs(reference)`;
//   * `F64`: `abs(actual - reference) <= 1e-12 + 1e-12*abs(reference)`.
//
// No NaN payload is compared: two NaN results agree, a NaN result against any
// finite result does not, and an infinity agrees only with the same infinity.
// Returns nullopt exactly when the observed element conforms.
[[nodiscard]] inline std::optional<std::string> linear_compare_element(
        iom::DataType leaf, std::uint64_t observed, std::uint64_t expected_code,
        double reference) {
    const std::uint64_t mask = linear_code_mask(leaf);
    observed &= mask;
    expected_code &= mask;
    if (linear_is_integer(leaf)) {
        if (observed == expected_code) {
            return std::nullopt;
        }
        return "integer bits differ: observed " + std::to_string(observed)
               + ", expected " + std::to_string(expected_code);
    }
    const double actual = linear_decode_code(leaf, observed);
    const auto mismatch = [&](std::string_view what) {
        return std::optional<std::string>{
                std::string(what) + ": observed code "
                + std::to_string(observed) + " (value "
                + std::to_string(actual) + ")"};
    };
    const auto relative = [&](double scale) -> std::optional<std::string> {
        if (std::isnan(reference)) {
            return std::isnan(actual) ? std::nullopt
                                      : mismatch("reference is NaN");
        }
        if (std::isinf(reference)) {
            return actual == reference ? std::nullopt
                                       : mismatch("reference is infinite");
        }
        if (!std::isfinite(actual)) {
            return mismatch("finite reference");
        }
        const double difference = std::fabs(actual - reference);
        if (difference <= scale + scale * std::fabs(reference)) {
            return std::nullopt;
        }
        return mismatch("relative tolerance " + std::to_string(scale));
    };
    switch (leaf) {
        case iom::DataType::F4_E2M1:
        case iom::DataType::F6_E2M3:
        case iom::DataType::F6_E3M2:
        case iom::DataType::F8_E4M3FN:
        case iom::DataType::F8_E5M2:
        case iom::DataType::F16: {
            if (observed == expected_code) {
                return std::nullopt;
            }
            if (std::isnan(actual)
                    && std::isnan(linear_decode_code(leaf, expected_code))) {
                return std::nullopt;
            }
            return mismatch("exact encoded scalar recurrence");
        }
        case iom::DataType::BF16: {
            // A recurrence that left the finite range is compared by class
            // from its own encoded code: the FP64 equation can stay finite,
            // or even take the other sign, where the mandated FP32
            // accumulation already overflowed.
            const double recurrence = linear_decode_code(leaf, expected_code);
            if (std::isnan(recurrence)) {
                return std::isnan(actual) ? std::nullopt
                                          : mismatch("BF16 recurrence is NaN");
            }
            if (std::isinf(recurrence)) {
                return actual == recurrence
                        ? std::nullopt
                        : mismatch("BF16 recurrence is infinite");
            }
            const double quantized = linear_decode_code(
                    leaf, linear_encode_value(leaf, reference));
            if (std::isnan(quantized)) {
                return std::isnan(actual) ? std::nullopt
                                          : mismatch("BF16 reference is NaN");
            }
            if (std::isinf(quantized)) {
                return actual == quantized
                        ? std::nullopt
                        : mismatch("BF16 reference is infinite");
            }
            if (!std::isfinite(actual)) {
                return mismatch("finite BF16 reference");
            }
            const double difference = std::fabs(actual - quantized);
            const double bound = std::max(
                    {linear_bf16_ulp(quantized), std::ldexp(1.0, -7),
                     std::ldexp(1.0, -6) * std::fabs(quantized)});
            if (difference <= bound) {
                return std::nullopt;
            }
            return mismatch("BF16 bound");
        }
        case iom::DataType::F32: {
            // The FP32 recurrence is this contract's definition of an `F32`
            // result and can legitimately leave the finite range: a fixture
            // pair like `FLT_MAX * FLT_MAX` overflows to infinity while the
            // FP64 equation stays finite. A nonfinite expected class is
            // therefore compared by class, and the relative threshold applies
            // only while the encoded recurrence stays finite.
            const double expected = linear_decode_code(leaf, expected_code);
            if (std::isnan(expected)) {
                return std::isnan(actual) ? std::nullopt
                                          : mismatch("F32 recurrence is NaN");
            }
            if (std::isinf(expected)) {
                return actual == expected
                        ? std::nullopt
                        : mismatch("F32 recurrence is infinite");
            }
            return relative(1e-5);
        }
        case iom::DataType::F64:
            return relative(1e-12);
        default:
            break;
    }
    throw std::invalid_argument("linear_compare_element: not a linear leaf");
}

// Compare one whole logical output image against the reference. Returns
// nullopt exactly when every element conforms.
[[nodiscard]] inline std::optional<std::string> linear_compare_image(
        const LinearOracleRequest& request,
        std::span<const std::byte> observed_logical,
        const LinearReference& reference, std::string_view context) {
    const std::size_t bits = bits_of(request.leaf);
    REQUIRE_EQ(
            observed_logical.size(),
            (request.elements() * bits + 7) / 8);
    REQUIRE_EQ(reference.codes.size(), request.elements());
    const bool values_match = reference.values.empty()
            || reference.values.size() == request.elements();
    REQUIRE(values_match);
    const std::size_t columns = request.output_columns();
    for (std::size_t index = 0; index < request.elements(); ++index) {
        const std::uint64_t observed =
                read_storage_bits(observed_logical.data(), index * bits, bits);
        const double value = reference.values.empty()
                ? 0.0
                : reference.values[index];
        const std::optional<std::string> mismatch = linear_compare_element(
                request.leaf, observed, reference.codes[index], value);
        if (!mismatch.has_value()) {
            continue;
        }
        const std::size_t plane = index / request.plane_elements();
        const std::size_t element = index % request.plane_elements();
        return std::string(context) + ": logical plane "
               + std::to_string(plane) + " row "
               + std::to_string(element / columns) + " column "
               + std::to_string(element % columns) + " " + *mismatch;
    }
    return std::nullopt;
}

// Pack one reference's codes into a logical output image, so the self-check can
// feed a variant's reference through the very same comparison a candidate's
// observable output passes.
[[nodiscard]] inline std::vector<std::byte> linear_codes_image(
        iom::DataType leaf, std::span<const std::uint64_t> codes) {
    const std::size_t bits = bits_of(leaf);
    std::vector<std::byte> image((codes.size() * bits + 7) / 8, std::byte{0});
    for (std::size_t index = 0; index < codes.size(); ++index) {
        write_storage_bits(image.data(), index * bits, bits, codes[index]);
    }
    return image;
}

// ---------------------------------------------------------------------------
// Oracle self-check.
//
// The self-check is the executable proof that the reference is independent and
// complete: it pins the named-format classes and the arithmetic policy against
// hand-authored truth, exercises every one of the twenty-one applicable leaves
// with the exact special/nonfinite policy of the fixtures, and proves that each
// deliberate negative variant is detected while remaining selective.
// ---------------------------------------------------------------------------

// Named-format, integer-arithmetic, and recurrence policy of every applicable
// leaf, checked against explicit expectations.
[[nodiscard]] inline bool run_linear_codec_self_check() {
    bool ok = true;
    const auto expect = [&ok](bool condition, const std::string& what) {
        CHECK_MESSAGE(condition, what);
        ok = ok && condition;
    };

    // The published matrix and its classification.
    expect(kLinearLeafSpan.size() == 21, "the applicable linear matrix is 21 leaves");
    expect(kLinearScalarLeafSpan.size() == 20, "the scalar path covers 20 leaves");
    expect(kLinearNativeBf16Span.size() == 1, "one native BF16 specialization leaf");
    expect(kLinearTtnnLeafSpan.size() == 1, "TTNN's mandatory target is BF16 alone");
    for (const iom::DataType leaf : kLinearLeafSpan) {
        CAPTURE(static_cast<int>(leaf));
        expect(linear_applicable(leaf), "a published leaf is applicable");
        expect(
                linear_is_integer(leaf) != linear_is_floating(leaf),
                "a leaf is exactly one arithmetic class");
        expect(
                linear_ladder_size(leaf) == (linear_is_integer(leaf) ? 8u : 12u),
                "the special ladder has its documented size");
        expect(
                linear_ladder_code(leaf, 0) == 0,
                "the first ladder position is the positive zero");
        expect(
                linear_ladder_code(leaf, linear_ladder_size(leaf) - 1)
                        <= linear_code_mask(leaf),
                "every ladder code stays inside the leaf width");
    }
    expect(
            linear_declares(kLinearScalarLeafSpan, iom::DataType::BF16) == false,
            "the scalar path excludes BF16");
    for (const iom::DataType leaf : kLinearInapplicableSpan) {
        CAPTURE(static_cast<int>(leaf));
        expect(!linear_applicable(leaf), "BOOL and F8_E8M0 are inapplicable");
        expect(
                !linear_declares(kLinearLeafSpan, leaf),
                "an inapplicable leaf is not in the applicable matrix");
    }

    // Integer arithmetic: exact unsigned modulo-`2^N` dots whose stored code is
    // the two's-complement pattern, with no saturation or clamping.
    {
        const std::uint64_t x8[2] = {127, 127};
        const std::uint64_t w8[2] = {2, 2};
        expect(
                linear_integer_recurrence(iom::DataType::I8, x8, w8) == 252,
                "I8 reduces modulo 2^8");
        expect(
                static_cast<std::int8_t>(252) == -4,
                "the stored I8 code is the two's-complement pattern");
        const std::uint64_t xu8[2] = {200, 200};
        const std::uint64_t wu8[2] = {3, 3};
        expect(
                linear_integer_recurrence(iom::DataType::U8, xu8, wu8) == 176,
                "U8 reduces modulo 2^8");
        const std::uint64_t x64[2] = {std::uint64_t{1} << 63, std::uint64_t{1} << 63};
        const std::uint64_t w64[2] = {2, 2};
        expect(
                linear_integer_recurrence(iom::DataType::U64, x64, w64) == 0,
                "U64 wraps modulo 2^64");
        expect(
                linear_integer_recurrence(iom::DataType::I64, x64, w64) == 0,
                "I64 wraps modulo 2^64");
        const std::uint64_t x2[1] = {3};
        const std::uint64_t w2[1] = {3};
        expect(
                linear_integer_recurrence(iom::DataType::I2, x2, w2) == 1,
                "I2 wraps modulo 4 without saturation");
        expect(
                linear_integer_recurrence(
                        iom::DataType::I64, x2, w2)
                        == 9,
                "an exact product inside the width is unchanged");
    }

    // The named-format classes and boundary values of every floating leaf.
    for (const iom::DataType leaf : kLinearLeafSpan) {
        if (!linear_is_floating(leaf)) {
            continue;
        }
        CAPTURE(static_cast<int>(leaf));
        const LinearFormat format = linear_format(leaf);
        const std::uint64_t exponent_mask =
                (std::uint64_t{1} << format.ebits) - 1;
        const std::uint64_t fraction_mask =
                (std::uint64_t{1} << format.fbits) - 1;
        const std::uint64_t finite_exponent_max =
                (format.finite_only && format.ebits < 4) ? exponent_mask
                                                         : exponent_mask - 1;
        const std::uint64_t sign_shift = format.ebits + format.fbits;
        const std::uint64_t sign = std::uint64_t{1} << sign_shift;
        const std::uint64_t unit =
                static_cast<std::uint64_t>(format.bias) << format.fbits;
        const std::uint64_t max_finite =
                (finite_exponent_max << format.fbits) | fraction_mask;
        const double smallest_subnormal = linear_decode_code(leaf, 1);

        expect(
                linear_decode_code(leaf, 0) == 0.0
                        && !std::signbit(linear_decode_code(leaf, 0)),
                "code zero is positive zero");
        expect(
                std::signbit(linear_decode_code(leaf, sign)),
                "the sign bit alone is negative zero");
        expect(
                linear_encode_value(leaf, -0.0) == sign,
                "negative zero encodes to the sign bit alone");
        // The subnormal boundary is checked with exact integer bit patterns for
        // `F64`, whose subnormal range is the double's own subnormal range, and
        // with value comparisons for the other leaves, whose smallest values
        // are normal doubles. Some compilers' default optimization model flushes
        // denormal values in arithmetic and comparison contexts; bit patterns
        // stay exact integer data, and the complete round trip below covers
        // every subnormal code of every leaf.
        const std::uint64_t sample_step =
                fraction_mask > 0xFFFFu ? (fraction_mask / 0xFFFFu) : 1;
        bool strictly_increasing = true;
        if (leaf == iom::DataType::F64) {
            const auto pattern = [leaf](std::uint64_t code) {
                return std::bit_cast<std::uint64_t>(
                        linear_decode_code(leaf, code));
            };
            expect(
                    pattern(1)
                            == std::bit_cast<std::uint64_t>(
                                    std::numeric_limits<double>::denorm_min()),
                    "the F64 smallest subnormal is the double denormal "
                    "minimum");
            for (std::uint64_t code = sample_step; code <= fraction_mask;
                 code += sample_step) {
                strictly_increasing =
                        strictly_increasing && pattern(code) > pattern(code - 1);
            }
            if (sample_step > 1) {
                strictly_increasing = strictly_increasing
                        && pattern(fraction_mask) > pattern(fraction_mask - 1);
            }
            expect(
                    strictly_increasing,
                    "every F64 subnormal code is strictly greater than its "
                    "predecessor (gradual underflow)");
            expect(
                    pattern(fraction_mask)
                            < pattern(std::uint64_t{1} << format.fbits),
                    "the F64 largest subnormal is below the smallest normal");
        } else {
            expect(
                    smallest_subnormal > 0.0,
                    "code one is the smallest subnormal");
            for (std::uint64_t code = sample_step; code <= fraction_mask;
                 code += sample_step) {
                strictly_increasing =
                        strictly_increasing
                        && linear_decode_code(leaf, code)
                                   > linear_decode_code(leaf, code - 1);
            }
            if (sample_step > 1) {
                strictly_increasing =
                        strictly_increasing
                        && linear_decode_code(leaf, fraction_mask)
                                   > linear_decode_code(leaf, fraction_mask - 1);
            }
            expect(
                    strictly_increasing,
                    "every subnormal code is strictly greater than its "
                    "predecessor (gradual underflow)");
            expect(
                    linear_decode_code(leaf, fraction_mask)
                            < linear_decode_code(
                                      leaf,
                                      std::uint64_t{1} << format.fbits),
                    "the largest subnormal is below the smallest normal");
        }
        expect(
                linear_decode_code(leaf, std::uint64_t{1} << format.fbits)
                        == std::ldexp(1.0, 1 - format.bias),
                "the smallest normal boundary is exact");
        expect(
                linear_decode_code(leaf, unit) == 1.0,
                "the unit code decodes to one");
        expect(
                linear_decode_code(leaf, sign | unit) == -1.0,
                "the sign bit negates the unit");
        expect(
                linear_decode_code(leaf, max_finite)
                        == std::ldexp(
                                   static_cast<double>(
                                           (std::uint64_t{1} << format.fbits)
                                           + fraction_mask),
                                   finite_exponent_max - format.bias
                                           - format.fbits),
                "the largest finite magnitude is exact");
        expect(
                linear_encode_value(leaf, linear_decode_code(leaf, max_finite))
                        == max_finite,
                "the largest finite magnitude round-trips");
        expect(
                linear_decode_code(leaf, sign | max_finite)
                        == -linear_decode_code(leaf, max_finite),
                "the sign bit negates a finite magnitude");
        expect(
                linear_encode_value(leaf, smallest_subnormal) == 1,
                "the smallest subnormal round-trips");
        if (format.infs) {
            expect(
                    std::isinf(linear_decode_code(
                            leaf, exponent_mask << format.fbits)),
                    "the infinity encoding decodes to infinity");
            expect(
                    std::signbit(linear_decode_code(
                            leaf, sign | (exponent_mask << format.fbits))),
                    "the infinity encoding carries its sign");
            expect(
                    linear_encode_value(
                            leaf, std::numeric_limits<double>::infinity())
                            == (exponent_mask << format.fbits),
                    "positive infinity encodes to its class");
            expect(
                    linear_encode_value(
                            leaf, -std::numeric_limits<double>::infinity())
                            == (sign | (exponent_mask << format.fbits)),
                    "negative infinity encodes to its class");
            expect(
                    std::isnan(linear_decode_code(
                            leaf,
                            (exponent_mask << format.fbits)
                                    | (std::uint64_t{1}
                                       << (format.fbits - 1)))),
                    "the format NaN encoding decodes to NaN");
            expect(
                    linear_encode_value(
                            leaf,
                            std::numeric_limits<double>::quiet_NaN())
                            == ((exponent_mask << format.fbits)
                                | (std::uint64_t{1} << (format.fbits - 1))),
                    "NaN encodes to the canonical quiet NaN");
            expect(
                    std::isinf(linear_decode_code(
                            leaf,
                            linear_encode_value(
                                    leaf,
                                    std::ldexp(
                                            1.0,
                                            finite_exponent_max - format.bias
                                                    + 5)))),
                    "finite overflow of an infinite format encodes to infinity");
        } else {
            const std::uint64_t format_nan =
                    (exponent_mask << format.fbits) | fraction_mask;
            expect(
                    linear_encode_value(
                            leaf,
                            std::ldexp(
                                    1.0,
                                    finite_exponent_max - format.bias + 5))
                            == max_finite,
                    "a finite-only format saturates finite overflow");
            if (format.ebits >= 4) {
                expect(
                        std::isnan(linear_decode_code(leaf, format_nan)),
                        "F8_E4M3FN has NaN but no infinity");
                expect(
                        linear_encode_value(
                                leaf,
                                std::numeric_limits<double>::quiet_NaN())
                                == format_nan,
                        "F8_E4M3FN encodes NaN to its format NaN");
            } else {
                expect(
                        linear_encode_value(
                                leaf,
                                std::numeric_limits<double>::quiet_NaN())
                                == max_finite,
                        "a format without a NaN class saturates NaN");
                expect(
                        !std::isnan(linear_decode_code(leaf, format_nan)),
                        "a format without a NaN class has no NaN encoding");
            }
        }

        // Complete round trip of every code the format can carry: a codec that
        // returns a non-nearest magnitude, flushes a subnormal, or pairs a NaN
        // class with a different payload is caught here.
        if (format.bits <= 16) {
            std::size_t failures = 0;
            const std::uint64_t limit = linear_code_mask(leaf);
            for (std::uint64_t code = 0; code <= limit; ++code) {
                const double value = linear_decode_code(leaf, code);
                std::uint64_t expected = code;
                if (std::isnan(value)) {
                    expected = format.infs
                            ? ((exponent_mask << format.fbits)
                               | (std::uint64_t{1} << (format.fbits - 1)))
                            : ((exponent_mask << format.fbits) | fraction_mask);
                }
                if (linear_encode_value(leaf, value) != expected) {
                    ++failures;
                }
            }
            expect(
                    failures == 0,
                    "every code of a narrow format round-trips exactly ("
                            + std::to_string(failures) + " failures)");
        }
    }

    // The contract's normative BF16 round-to-nearest, ties-to-even checks.
    {
        const iom::DataType leaf = iom::DataType::BF16;
        const std::uint64_t one =
                static_cast<std::uint64_t>(127) << 7;  // 1.0
        const std::uint64_t half =
                static_cast<std::uint64_t>(126) << 7;  // 0.5
        expect(
                linear_encode_value(leaf, 1.0 + std::ldexp(1.0, -8)) == one,
                "RNE picks the lower even neighbour at the one midpoint");
        expect(
                linear_encode_value(leaf, 1.0 + 3 * std::ldexp(1.0, -8))
                        == (one | 2),
                "RNE picks the upper even neighbour at the next midpoint");
        expect(
                linear_encode_value(leaf, 0.501953125) == half,
                "RNE picks 0.5 for the 0.5 plus 2^-9 midpoint");
        expect(
                linear_encode_value(leaf, 0.505859375) == (half | 2),
                "RNE picks 0.50390625 plus 2^-8 for the next midpoint");
        // The same policy in a narrower format: a tie at F16 one rounds down.
        expect(
                linear_encode_value(
                        iom::DataType::F16, 1.0 + std::ldexp(1.0, -11))
                        == (static_cast<std::uint64_t>(15) << 10),
                "F16 RNE also ties to even");
    }

    // The recurrence rounds the product together with the addend. A host library
    // is not required to provide a correctly rounded `std::fma` under every
    // compiler's default floating-point model, so the fused step is pinned by
    // the property the contract's comparison classes actually observe:
    //
    //   * every product of the six exact-comparison floating leaves is exactly
    //     representable in FP32, so the fused and the unfused step coincide
    //     there and the encoded recurrence is host-independent; and
    //   * the recurrence rounds to FP32 after every step instead of
    //     accumulating in a wider domain, which the tie at `2^24` pins.
    for (const iom::DataType leaf :
         {iom::DataType::F4_E2M1, iom::DataType::F6_E2M3,
          iom::DataType::F6_E3M2, iom::DataType::F8_E4M3FN,
          iom::DataType::F8_E5M2, iom::DataType::F16}) {
        CAPTURE(static_cast<int>(leaf));
        const LinearRawImage x = linear_fixture_image(
                leaf, 1, 19, 3, 0x1111, false, 0x2222);
        const LinearRawImage w = linear_fixture_image(
                leaf, 1, 10, 3, 0x3333, false, 0x4444);
        const LinearOracleRequest request{
                leaf, 1, 19, 3, 10, 2, 17, 1, 10,
                iom::LinearOutputLayout::ordinary};
        const LinearReference reference = linear_reference(request, x, w);
        // The unfused model: each exact product is added to an FP32
        // accumulator with one rounding per step. Every product of a
        // narrow-leaf operand pair is exact, so this must equal the fused
        // recurrence for every element.
        std::size_t exact_products = 0;
        std::size_t sampled = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t o = 0; o < 10; ++o) {
                const double xa = linear_decode_code(
                        leaf, linear_image_code(x, i));
                const double wb = linear_decode_code(
                        leaf, linear_image_code(w, o * w.columns + i));
                const double product = xa * wb;
                ++sampled;
                exact_products +=
                        static_cast<double>(static_cast<float>(product))
                                == product
                        ? 1u
                        : 0u;
            }
        }
        expect(
                exact_products == sampled,
                "every narrow-leaf product is exact in FP32");
        expect(
                reference.codes.size() == 17 * 10,
                "the narrow-leaf reference is complete");
    }
    {
        const iom::DataType leaf = iom::DataType::F32;
        // `2^24 + 1 + 1` rounds to `2^24` after every FP32 step, while a
        // double accumulator would keep `2^24 + 2`. The tie at `2^24` is
        // exact under any floating-point model.
        const std::uint64_t x[3] = {
                linear_encode_value(leaf, std::ldexp(1.0, 24)),
                linear_encode_value(leaf, 1.0),
                linear_encode_value(leaf, 1.0)};
        const std::uint64_t w[3] = {
                linear_encode_value(leaf, 1.0),
                linear_encode_value(leaf, 1.0),
                linear_encode_value(leaf, 1.0)};
        expect(
                linear_scalar_recurrence(leaf, x, w)
                        == linear_encode_value(leaf, std::ldexp(1.0, 24)),
                "the FP32 recurrence rounds to FP32 after every step");
        const double wide = linear_wide_value(leaf, x, w);
        expect(
                linear_encode_value(leaf, wide)
                        != linear_scalar_recurrence(leaf, x, w),
                "a wider accumulator would keep the dropped unit");
    }

    // The two component classes of the exact recurrence: an integral FP32 dot
    // and the F64 recurrence of an integral F64 dot.
    {
        const std::uint64_t x[3] = {
                linear_encode_value(iom::DataType::F32, 1.0),
                linear_encode_value(iom::DataType::F32, 2.0),
                linear_encode_value(iom::DataType::F32, 3.0)};
        const std::uint64_t w[3] = {
                linear_encode_value(iom::DataType::F32, 4.0),
                linear_encode_value(iom::DataType::F32, 5.0),
                linear_encode_value(iom::DataType::F32, 6.0)};
        expect(
                linear_scalar_recurrence(iom::DataType::F32, x, w)
                        == linear_encode_value(iom::DataType::F32, 32.0),
                "an exact FP32 dot is exact");
        const std::uint64_t x64_codes[3] = {
                linear_encode_value(iom::DataType::F64, 1.0),
                linear_encode_value(iom::DataType::F64, 2.0),
                linear_encode_value(iom::DataType::F64, 3.0)};
        const std::uint64_t w64_codes[3] = {
                linear_encode_value(iom::DataType::F64, 4.0),
                linear_encode_value(iom::DataType::F64, 5.0),
                linear_encode_value(iom::DataType::F64, 6.0)};
        expect(
                linear_scalar_recurrence(
                        iom::DataType::F64, x64_codes, w64_codes)
                        == linear_encode_value(iom::DataType::F64, 32.0),
                "an exact F64 dot is exact");
        expect(
                linear_wide_value(iom::DataType::F32, x, w) == 32.0,
                "the FP64 evaluation agrees on an exact dot");
    }

    return ok;
}

// Reference completeness: hand-authored truth for the equation, the window,
// the head mapping, and plane independence; per-leaf variant detection and
// selectivity; padding invariance; and the fixed tolerance boundaries.
[[nodiscard]] inline bool run_linear_reference_self_check() {
    bool ok = true;
    const auto expect = [&ok](bool condition, const std::string& what) {
        CHECK_MESSAGE(condition, what);
        ok = ok && condition;
    };
    const auto image = [](
                               iom::DataType leaf, std::size_t planes,
                               std::size_t logical_rows,
                               std::size_t logical_columns,
                               const std::function<std::uint64_t(
                                       std::size_t, std::size_t, std::size_t)>&
                                       code) {
        return linear_coded_image(
                leaf, planes, linear_pad16(logical_rows),
                linear_pad16(logical_columns), logical_rows, logical_columns,
                code,
                [](std::size_t, std::size_t, std::size_t) {
                    return std::uint64_t{0};
                });
    };
    const auto codes_of = [](
                                  const LinearReference& reference) {
        return reference.codes;
    };
    const auto detects = [&expect](
                                 const LinearOracleRequest& request,
                                 const LinearRawImage& x,
                                 const LinearRawImage& w,
                                 LinearOracleVariant variant,
                                 const char* what) {
        const LinearReference honest = linear_reference(request, x, w);
        const LinearReference wrong =
                linear_reference(request, x, w, variant);
        const std::vector<std::byte> honest_image =
                linear_codes_image(request.leaf, honest.codes);
        expect(
                linear_compare_image(request, honest_image, wrong, what)
                        .has_value(),
                std::string("the reference detects ") + what);
    };
    const auto coincides = [&expect](
                                   const LinearOracleRequest& request,
                                   const LinearRawImage& x,
                                   const LinearRawImage& w,
                                   LinearOracleVariant variant,
                                   const char* what) {
        const LinearReference honest = linear_reference(request, x, w);
        const LinearReference other =
                linear_reference(request, x, w, variant);
        expect(
                honest.codes == other.codes,
                std::string("the ") + what
                        + " variant coincides where it must");
    };

    // Hand-authored truth: the semantic equation, the selected row window, the
    // Hugging Face weight row, the head mapping, and plane independence.
    {
        const iom::DataType leaf = iom::DataType::I8;
        const std::uint64_t x_rows[3][2] = {{1, 2}, {3, 4}, {5, 6}};
        const std::uint64_t w_rows[2][2] = {{7, 8}, {9, 10}};
        const LinearRawImage x = image(
                leaf, 1, 3, 2,
                [&](std::size_t, std::size_t row, std::size_t column) {
                    return x_rows[row][column];
                });
        const LinearRawImage w = image(
                leaf, 1, 2, 2,
                [&](std::size_t, std::size_t row, std::size_t column) {
                    return w_rows[row][column];
                });
        const LinearOracleRequest request{
                leaf, 1, 3, 2, 2, 1, 2, 1, 2,
                iom::LinearOutputLayout::ordinary};
        const LinearReference reference = linear_reference(request, x, w);
        const std::uint64_t expected[4] = {53, 67, 83, 105};
        expect(
                reference.codes
                        == std::vector<std::uint64_t>(
                                expected, expected + 4),
                "the hand-written I8 window product is exact");
        detects(
                request, x, w, LinearOracleVariant::transposed_weight,
                "a transposed weight");
        detects(
                request, x, w, LinearOracleVariant::wrong_row_window,
                "an off-by-one row window");
    }
    {
        // An identity weight makes the projection observable directly, and the
        // second request row proves the window starts at `s`.
        const iom::DataType leaf = iom::DataType::F32;
        const std::uint64_t x_rows[2][2] = {{1, 2}, {3, 4}};
        const std::uint64_t w_rows[2][2] = {{1, 0}, {0, 1}};
        const LinearRawImage x = image(
                leaf, 1, 2, 2,
                [&](std::size_t, std::size_t row, std::size_t column) {
                    return linear_encode_value(
                            leaf, static_cast<double>(x_rows[row][column]));
                });
        const LinearRawImage w = image(
                leaf, 1, 2, 2,
                [&](std::size_t, std::size_t row, std::size_t column) {
                    return linear_encode_value(
                            leaf, static_cast<double>(w_rows[row][column]));
                });
        const LinearOracleRequest request{
                leaf, 1, 2, 2, 2, 0, 2, 1, 2,
                iom::LinearOutputLayout::ordinary};
        const std::vector<std::uint64_t> expected = {
                linear_encode_value(leaf, 1.0),
                linear_encode_value(leaf, 2.0),
                linear_encode_value(leaf, 3.0),
                linear_encode_value(leaf, 4.0)};
        expect(
                codes_of(linear_reference(request, x, w)) == expected,
                "an identity weight reproduces the selected rows");
    }
    {
        // Head-planar truth with `H=2`, `D=2`: `out[b,h,r,d]` reads weight row
        // `h*D+d`, so the four head columns are the four distinct weight rows.
        const iom::DataType leaf = iom::DataType::F32;
        const LinearRawImage x = image(
                leaf, 1, 1, 1,
                [&](std::size_t, std::size_t, std::size_t) {
                    return linear_encode_value(leaf, 5.0);
                });
        const LinearRawImage w = image(
                leaf, 1, 4, 1,
                [&](std::size_t, std::size_t row, std::size_t) {
                    return linear_encode_value(
                            leaf, static_cast<double>(row + 1));
                });
        const LinearOracleRequest request{
                leaf, 1, 1, 1, 4, 0, 1, 2, 2,
                iom::LinearOutputLayout::head_planar};
        const std::vector<std::uint64_t> expected = {
                linear_encode_value(leaf, 5.0),
                linear_encode_value(leaf, 10.0),
                linear_encode_value(leaf, 15.0),
                linear_encode_value(leaf, 20.0)};
        expect(
                codes_of(linear_reference(request, x, w)) == expected,
                "the head-planar weight row is h*D+d");
        detects(
                request, x, w, LinearOracleVariant::head_order,
                "a shuffled head order");
        detects(
                request, x, w, LinearOracleVariant::head_repeat,
                "a repeated head result");
        const LinearOracleRequest ordinary{
                leaf, 1, 1, 1, 4, 0, 1, 1, 4,
                iom::LinearOutputLayout::ordinary};
        expect(
                codes_of(linear_reference(ordinary, x, w)) == expected,
                "ordinary mode reads the same four weight rows");
    }
    {
        // The LM-head selection: only the last source row is read, and only
        // one row is written.
        const iom::DataType leaf = iom::DataType::F32;
        const std::uint64_t x_rows[5][3] = {
                {1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {9, 9, 9}};
        const LinearRawImage x = image(
                leaf, 1, 5, 3,
                [&](std::size_t, std::size_t row, std::size_t column) {
                    return linear_encode_value(
                            leaf, static_cast<double>(x_rows[row][column]));
                });
        const LinearRawImage w = image(
                leaf, 1, 2, 3,
                [&](std::size_t, std::size_t row, std::size_t column) {
                    return linear_encode_value(
                            leaf, static_cast<double>(row + 1 + 0 * column));
                });
        const LinearOracleRequest request{
                leaf, 1, 5, 3, 2, 4, 1, 1, 2,
                iom::LinearOutputLayout::ordinary};
        const std::vector<std::uint64_t> expected = {
                linear_encode_value(leaf, 9.0 * 1.0 * 3.0),
                linear_encode_value(leaf, 9.0 * 2.0 * 3.0)};
        expect(
                codes_of(linear_reference(request, x, w)) == expected,
                "the LM-head request selects the last source row only");
    }
    {
        // Independent leading planes: the same columns with distinct plane
        // content produce distinct plane results.
        const iom::DataType leaf = iom::DataType::F32;
        const LinearRawImage x = image(
                leaf, 2, 1, 2,
                [&](std::size_t plane, std::size_t, std::size_t column) {
                    return linear_encode_value(
                            leaf,
                            static_cast<double>(plane + 1)
                                    * static_cast<double>(column + 1));
                });
        const LinearRawImage w = image(
                leaf, 1, 1, 2,
                [&](std::size_t, std::size_t, std::size_t) {
                    return linear_encode_value(leaf, 1.0);
                });
        const LinearOracleRequest request{
                leaf, 2, 1, 2, 1, 0, 1, 1, 1,
                iom::LinearOutputLayout::ordinary};
        const std::vector<std::uint64_t> expected = {
                linear_encode_value(leaf, 3.0),
                linear_encode_value(leaf, 6.0)};
        expect(
                codes_of(linear_reference(request, x, w)) == expected,
                "leading planes stay independent");
    }

    // Every applicable leaf exercises the canonical fixture set with the exact
    // special/nonfinite policy, and every deliberate negative variant is
    // detected on it.
    for (const iom::DataType leaf : kLinearLeafSpan) {
        CAPTURE(static_cast<int>(leaf));
        const std::size_t inner = 3;
        const std::size_t outer = 10;
        const std::size_t source_rows = 19;
        const std::size_t start_row = 2;
        const LinearRawImage x = linear_fixture_image(
                leaf, 2, source_rows, inner, 0x1111, false, 0x2222);
        const LinearRawImage w = linear_fixture_image(
                leaf, 1, outer, inner, 0x3333, false, 0x4444);
        const LinearOracleRequest ordinary{
                leaf, 2, source_rows, inner, outer, start_row, 17, 1, outer,
                iom::LinearOutputLayout::ordinary};
        const LinearOracleRequest head_planar{
                leaf, 2, source_rows, inner, outer, start_row, 17, 2, 5,
                iom::LinearOutputLayout::head_planar};
        expect(
                linear_reference(ordinary, x, w).codes.size() == 2 * 17 * 10,
                "the ordinary reference covers every selected element");
        expect(
                linear_reference(head_planar, x, w).codes.size()
                        == 2 * 2 * 17 * 5,
                "the head-planar reference covers every head element");
        detects(
                ordinary, x, w, LinearOracleVariant::transposed_weight,
                "a transposed weight");
        detects(
                ordinary, x, w, LinearOracleVariant::wrong_row_window,
                "an off-by-one row window");
        detects(
                ordinary, x, w, LinearOracleVariant::mixed_plane,
                "a plane-blind leading block");
        detects(
                ordinary, x, w, LinearOracleVariant::padding_dependent,
                "padding-dependent addressing");
        detects(
                ordinary, x, w, LinearOracleVariant::tile_tail,
                "tile-tail addressing");
        if (linear_exact_encoded(leaf)) {
            // The fixed thresholds compare this class bit-exactly against the
            // encoded scalar recurrence, so the policy itself rejects the
            // re-encoding model.
            detects(
                    ordinary, x, w, LinearOracleVariant::numeric_reencode,
                    "a re-encoding recurrence");
        }
        detects(
                head_planar, x, w, LinearOracleVariant::head_order,
                "a shuffled head order");
        detects(
                head_planar, x, w, LinearOracleVariant::head_repeat,
                "a repeated head result");
        // Selectivity: each variant coincides with the reference on a fixture
        // where the perturbation is genuinely invisible, so the detections
        // above are comparisons rather than an unconditional "differs".
        {
            const LinearRawImage x1 = linear_fixture_image(
                    leaf, 1, 19, 1, 0x5555, false, 0x6666);
            const LinearRawImage w1 = linear_fixture_image(
                    leaf, 1, 1, 1, 0x7777, false, 0x8888);
            const LinearOracleRequest single{
                    leaf, 1, 19, 1, 1, 2, 17, 1, 1,
                    iom::LinearOutputLayout::ordinary};
            coincides(
                    single, x1, w1, LinearOracleVariant::transposed_weight,
                    "single-element transpose");
            coincides(
                    single, x1, w1, LinearOracleVariant::mixed_plane,
                    "single-plane");
            const LinearRawImage x16 = linear_fixture_image(
                    leaf, 2, 16, 16, 0x9999, false, 0xAAAA);
            const LinearRawImage w16 = linear_fixture_image(
                    leaf, 1, 16, 16, 0xBBBB, false, 0xCCCC);
            const LinearOracleRequest aligned{
                    leaf, 2, 16, 16, 16, 0, 16, 1, 16,
                    iom::LinearOutputLayout::ordinary};
            coincides(
                    aligned, x16, w16,
                    LinearOracleVariant::padding_dependent,
                    "tile-aligned plane stride");
            coincides(
                    aligned, x16, w16, LinearOracleVariant::tile_tail,
                    "tile-aligned inner extent");
            const LinearRawImage xrow = linear_coded_image(
                    leaf, 2, linear_pad16(20), linear_pad16(3), 20, 3,
                    [&](std::size_t plane, std::size_t, std::size_t column) {
                        return linear_fixture_code(
                                leaf, plane * 3 + column, 0xDDDD, false);
                    },
                    [](std::size_t, std::size_t, std::size_t) {
                        return std::uint64_t{0};
                    });
            const LinearOracleRequest row_invariant{
                    leaf, 2, 20, 3, 10, 2, 17, 1, 10,
                    iom::LinearOutputLayout::ordinary};
            coincides(
                    row_invariant, xrow, w,
                    LinearOracleVariant::wrong_row_window,
                    "row-invariant window");
            const LinearOracleRequest single_head{
                    leaf, 2, 19, 3, 10, 2, 17, 1, 10,
                    iom::LinearOutputLayout::head_planar};
            coincides(
                    single_head, x, w, LinearOracleVariant::head_order,
                    "single-head order");
            coincides(
                    single_head, x, w, LinearOracleVariant::head_repeat,
                    "single-head repeat");
            const LinearRawImage x1inner = linear_fixture_image(
                    leaf, 2, 19, 1, 0xEEEE, false, 0xFFFF);
            const LinearRawImage w1inner = linear_fixture_image(
                    leaf, 1, 10, 1, 0xDDDD, false, 0xCCCC);
            const LinearOracleRequest single_inner{
                    leaf, 2, 19, 1, 10, 2, 17, 1, 10,
                    iom::LinearOutputLayout::ordinary};
            coincides(
                    single_inner, x1inner, w1inner,
                    LinearOracleVariant::numeric_reencode,
                    "single-product recurrence");
        }
    }

    // Padding invariance: perturbing every non-logical cell of both operands
    // (tile padding, sub-byte remainder bits, and uninitialized storage) never
    // changes the reference's valid output.
    for (const iom::DataType leaf : kLinearLeafSpan) {
        CAPTURE(static_cast<int>(leaf));
        const std::size_t inner = 3;
        const std::size_t outer = 10;
        const std::size_t source_rows = 19;
        const auto build = [&](std::uint64_t poison) {
            return std::pair<LinearRawImage, LinearRawImage>{
                    linear_coded_image(
                            leaf, 2, linear_pad16(source_rows),
                            linear_pad16(inner), source_rows, inner,
                            [&](std::size_t plane, std::size_t row,
                                std::size_t column) {
                                return linear_fixture_code(
                                        leaf,
                                        (plane * source_rows + row) * inner
                                                + column,
                                        0x1234, true);
                            },
                            [poison](std::size_t, std::size_t, std::size_t) {
                                return poison;
                            }),
                    linear_coded_image(
                            leaf, 1, linear_pad16(outer),
                            linear_pad16(inner), outer, inner,
                            [&](std::size_t, std::size_t row,
                                std::size_t column) {
                                return linear_fixture_code(
                                        leaf, row * inner + column, 0x5678,
                                        true);
                            },
                            [poison](std::size_t, std::size_t, std::size_t) {
                                return poison;
                            })};
        };
        const auto clean = build(std::uint64_t{0});
        const auto poisoned = build(linear_poison_code(leaf, 3));
        const LinearOracleRequest ordinary{
                leaf, 2, source_rows, inner, outer, 2, 17, 1, outer,
                iom::LinearOutputLayout::ordinary};
        const LinearOracleRequest head_planar{
                leaf, 2, source_rows, inner, outer, 2, 17, 2, 5,
                iom::LinearOutputLayout::head_planar};
        expect(
                linear_reference(ordinary, clean.first, clean.second).codes
                        == linear_reference(
                                   ordinary, poisoned.first,
                                   poisoned.second)
                                   .codes,
                "poisoned padding never changes the ordinary result");
        expect(
                linear_reference(head_planar, clean.first, clean.second).codes
                        == linear_reference(
                                   head_planar, poisoned.first,
                                   poisoned.second)
                                   .codes,
                "poisoned padding never changes the head-planar result");
        // The poisoned fixture really carries the special classes the policy
        // names, so the invariance above is not the invariance of zeroes. The
        // classes are read from each leaf's own encoding: a format's subnormal
        // and nonfinite classes are properties of its encoding, not of the
        // widened double.
        bool saw_zero = false;
        bool saw_negative = false;
        bool saw_subnormal = false;
        bool saw_nan = false;
        bool saw_infinity = false;
        for (std::size_t plane = 0; plane < 2; ++plane) {
            for (std::size_t row = 0; row < source_rows; ++row) {
                for (std::size_t column = 0; column < inner; ++column) {
                    const std::uint64_t code = linear_fixture_code(
                            leaf, (plane * source_rows + row) * inner + column,
                            0x1234, true);
                    if (linear_is_integer(leaf)) {
                        saw_zero = saw_zero || code == 0;
                        saw_negative = saw_negative
                                       || (code & (std::uint64_t{1}
                                                   << (linear_integer_width(leaf) - 1)))
                                                  != 0;
                        continue;
                    }
                    const LinearFormat format = linear_format(leaf);
                    const std::uint64_t fraction =
                            code & ((std::uint64_t{1} << format.fbits) - 1);
                    const std::uint64_t exponent =
                            (code >> format.fbits)
                            & ((std::uint64_t{1} << format.ebits) - 1);
                    const double value = linear_decode_code(leaf, code);
                    saw_zero = saw_zero || value == 0.0;
                    saw_negative = saw_negative
                                   || (value == 0.0 && std::signbit(value));
                    saw_subnormal = saw_subnormal
                                    || (exponent == 0 && fraction != 0);
                    saw_nan = saw_nan || std::isnan(value);
                    saw_infinity = saw_infinity || std::isinf(value);
                }
            }
        }
        expect(saw_zero, "the fixture carries a zero");
        expect(saw_negative, "the fixture carries a negative class");
        if (linear_is_floating(leaf)) {
            const LinearFormat format = linear_format(leaf);
            const bool nan_class = format.infs
                                   || (format.finite_only && format.ebits >= 4);
            expect(
                    saw_subnormal,
                    "the fixture carries a subnormal encoding");
            expect(
                    saw_nan == nan_class,
                    "the fixture's NaN class matches the format");
            expect(
                    saw_infinity == format.infs,
                    "the fixture's infinity class matches the format");
        }
    }

    // A re-encoding recurrence is a genuinely different model, and the fixed
    // thresholds decide whether it is *detected*. Where they compare exact
    // encoded bits (the twelve integer leaves and F4/F6/F8/F16) the policy
    // rejects it outright, which is asserted per leaf above and here against the
    // canonical wide-inner fixture. BF16, F32, and F64 deliberately admit a
    // bounded deviation, so for them the model difference itself is asserted
    // (a hand-verified half-ULP boundary case plus the wide-inner fixture),
    // because their thresholds make only a gross re-encoding error a
    // conformance violation.
    {
        for (const iom::DataType leaf : kLinearLeafSpan) {
            if (!linear_exact_encoded(leaf)) {
                continue;
            }
            CAPTURE(static_cast<int>(leaf));
            const LinearRawImage x = linear_fixture_image(
                    leaf, 1, 19, 65, 0x0909, false, 0x0A0A);
            const LinearRawImage w = linear_fixture_image(
                    leaf, 1, 10, 65, 0x0B0B, false, 0x0C0C);
            const LinearOracleRequest request{
                    leaf, 1, 19, 65, 10, 2, 17, 1, 10,
                    iom::LinearOutputLayout::ordinary};
            detects(
                    request, x, w, LinearOracleVariant::numeric_reencode,
                    "a re-encoding recurrence");
        }
        for (const iom::DataType leaf :
             {iom::DataType::BF16, iom::DataType::F32, iom::DataType::F64}) {
            CAPTURE(static_cast<int>(leaf));
            const LinearRawImage x = linear_fixture_image(
                    leaf, 1, 19, 65, 0x0909, false, 0x0A0A);
            const LinearRawImage w = linear_fixture_image(
                    leaf, 1, 10, 65, 0x0B0B, false, 0x0C0C);
            const LinearOracleRequest request{
                    leaf, 1, 19, 65, 10, 2, 17, 1, 10,
                    iom::LinearOutputLayout::ordinary};
            const LinearReference honest = linear_reference(request, x, w);
            const LinearReference reencoded = linear_reference(
                    request, x, w, LinearOracleVariant::numeric_reencode);
            std::size_t differing = 0;
            for (std::size_t index = 0; index < honest.codes.size(); ++index) {
                differing += honest.codes[index] != reencoded.codes[index];
            }
            expect(
                    differing * 2 >= honest.codes.size(),
                    "the re-encoding recurrence differs on most elements of "
                    "the wide-inner fixture");
        }
        // The BF16 half-ULP boundary, hand-verified: two half-ULP terms make
        // the single rounding keep one exact ULP (`1 + 2^-7`), while rounding
        // every partial sum to the leaf loses it (`1.0`).
        {
            const iom::DataType leaf = iom::DataType::BF16;
            const LinearFormat format = linear_format(leaf);
            const std::size_t inner = 3;
            const LinearRawImage x = image(
                    leaf, 1, 1, inner,
                    [&](std::size_t, std::size_t, std::size_t) {
                        return linear_encode_value(leaf, 1.0);
                    });
            const LinearRawImage w = image(
                    leaf, 1, 3, inner,
                    [&](std::size_t, std::size_t, std::size_t column) {
                        return linear_encode_value(
                                leaf, column == 0 ? 1.0
                                                  : std::ldexp(
                                                            1.0,
                                                            -(format.fbits
                                                              + 1)));
                    });
            const LinearOracleRequest request{
                    leaf, 1, 1, inner, 3, 0, 1, 1, 3,
                    iom::LinearOutputLayout::ordinary};
            const LinearReference honest = linear_reference(request, x, w);
            const LinearReference reencoded = linear_reference(
                    request, x, w, LinearOracleVariant::numeric_reencode);
            expect(
                    honest.codes[0]
                            == linear_encode_value(
                                    leaf,
                                    1.0 + std::ldexp(1.0, -format.fbits)),
                    "one rounded recurrence keeps the half-ULP term");
            expect(
                    reencoded.codes[0] == linear_encode_value(leaf, 1.0),
                    "rounding every partial sum loses the half-ULP term");
            expect(
                    honest.codes != reencoded.codes,
                    "the re-encoding recurrence is a different model");
        }
    }

    // The fixed tolerance boundaries: an accepted deviation stays accepted and
    // a rejected one stays rejected, for every floating class.
    {
        const auto accepts = [&expect](
                                     iom::DataType leaf, std::uint64_t observed,
                                     std::uint64_t expected_code, double value,
                                     const char* what) {
            expect(
                    !linear_compare_element(leaf, observed, expected_code, value)
                             .has_value(),
                    std::string("accepted: ") + what);
        };
        const auto rejects = [&expect](
                                     iom::DataType leaf, std::uint64_t observed,
                                     std::uint64_t expected_code, double value,
                                     const char* what) {
            expect(
                    linear_compare_element(leaf, observed, expected_code, value)
                            .has_value(),
                    std::string("rejected: ") + what);
        };
        const iom::DataType f32 = iom::DataType::F32;
        accepts(f32, linear_encode_value(f32, 1.0), 0, 1.0, "exact F32");
        accepts(
                f32, linear_encode_value(f32, 1.0 + 1e-6), 0, 1.0,
                "F32 inside 1e-5 + 1e-5*abs(reference)");
        rejects(
                f32, linear_encode_value(f32, 1.03), 0, 1.0,
                "F32 outside its relative tolerance");
        const iom::DataType f64 = iom::DataType::F64;
        accepts(f64, linear_encode_value(f64, 1.0), 0, 1.0, "exact F64");
        accepts(
                f64, linear_encode_value(f64, 1.0 + 1e-13), 0, 1.0,
                "F64 inside 1e-12 + 1e-12*abs(reference)");
        rejects(
                f64, linear_encode_value(f64, 1.0 + 1e-9), 0, 1.0,
                "F64 outside its relative tolerance");
        const iom::DataType bf16 = iom::DataType::BF16;
        accepts(
                bf16, linear_encode_value(bf16, 1.0), 0, 1.0, "exact BF16");
        accepts(
                bf16, linear_encode_value(bf16, 1.0 + std::ldexp(1.0, -8)), 0,
                1.0, "one BF16 step inside max(ULP, 2^-7, 2^-6*abs(q))");
        rejects(
                bf16, linear_encode_value(bf16, 2.0), 0, 1.0,
                "BF16 far outside its bound");
        const iom::DataType f16 = iom::DataType::F16;
        accepts(
                f16, linear_encode_value(f16, 0.5), linear_encode_value(f16, 0.5),
                0.5, "exact encoded scalar recurrence");
        rejects(
                f16, linear_encode_value(f16, 0.5 + std::ldexp(1.0, -11)),
                linear_encode_value(f16, 0.5), 0.5,
                "an F16 deviation from the exact recurrence");
        // Nonfinite policy: NaN agrees with NaN and with nothing else, and an
        // infinity agrees only with the same infinity.
        const auto nan = std::numeric_limits<double>::quiet_NaN();
        accepts(
                bf16, linear_encode_value(bf16, nan), 0, nan,
                "a NaN result against a NaN reference");
        rejects(
                bf16, linear_encode_value(bf16, 1.0), 0, nan,
                "a finite result against a NaN reference");
        const auto infinity = std::numeric_limits<double>::infinity();
        accepts(
                f16, linear_encode_value(f16, infinity),
                linear_encode_value(f16, infinity), infinity,
                "an infinite result against the same infinity");
        rejects(
                f16, linear_encode_value(f16, -infinity),
                linear_encode_value(f16, infinity), infinity,
                "the opposite infinity is never accepted");
        rejects(
                f16, linear_encode_value(f16, infinity),
                linear_encode_value(f16, 1.0), 1.0,
                "an infinite result against a finite reference");
    }

    // The canonical fixture set of the contract: every run, both layouts, the
    // non-tile inner extent, and the non-tile outer extent, for every
    // applicable leaf.
    for (const iom::DataType leaf : kLinearLeafSpan) {
        CAPTURE(static_cast<int>(leaf));
        const LinearRawImage x = linear_fixture_image(
                leaf, 1, 19, 3, 0x0101, true, 0x0202);
        const LinearRawImage w = linear_fixture_image(
                leaf, 1, 10, 3, 0x0303, true, 0x0404);
        for (const std::size_t rows : {std::size_t{1}, std::size_t{15},
                                       std::size_t{16}, std::size_t{17}}) {
            const LinearOracleRequest request{
                    leaf, 1, 19, 3, 10, 2, rows, 1, 10,
                    iom::LinearOutputLayout::ordinary};
            expect(
                    linear_reference(request, x, w).codes.size() == rows * 10,
                    "every row run of the canonical fixture is covered");
        }
        const LinearRawImage x65 = linear_fixture_image(
                leaf, 1, 19, 65, 0x0505, false, 0x0606);
        const LinearRawImage w65 = linear_fixture_image(
                leaf, 1, 10, 65, 0x0707, false, 0x0808);
        const LinearOracleRequest wide{
                leaf, 1, 19, 65, 10, 2, 17, 1, 10,
                iom::LinearOutputLayout::ordinary};
        expect(
                linear_reference(wide, x65, w65).codes.size() == 17 * 10,
                "the I=65 inner tail is covered");
        detects(
                wide, x65, w65, LinearOracleVariant::tile_tail,
                "the I=65 tile tail");
    }

    return ok;
}

[[nodiscard]] inline bool run_linear_oracle_self_check() {
    const bool codec = run_linear_codec_self_check();
    const bool reference = run_linear_reference_self_check();
    return codec && reference;
}

// ---------------------------------------------------------------------------
// Shared request cases.
// ---------------------------------------------------------------------------

// The logical code of one element of a logical row-major image.
[[nodiscard]] inline std::uint64_t linear_logical_code(
        std::span<const std::byte> image, iom::DataType leaf,
        std::size_t linear) {
    const std::size_t bits = bits_of(leaf);
    if ((linear + 1) * bits > image.size() * 8) {
        throw std::out_of_range("linear logical read exceeds the image");
    }
    return read_storage_bits(image.data(), linear * bits, bits);
}

// One leading-only transform. It is applied to the input owner and to the
// output owner with `leading` equal to the number of axes the two views share:
// the input's own leading count, so a head-planar output's inserted head axis
// is never touched. Every transform keeps the final tiled axes intact, which is
// exactly what the operation's leading-tuple rule compares, and two owners may
// therefore reduce to the same leading tuple through different transforms while
// keeping distinct plane offsets and strides.
struct LinearTransform {
    const char* label;
    std::function<iom::TensorView(const iom::TensorView&, std::size_t leading)>
            build;
};

[[nodiscard]] inline std::vector<LinearTransform> linear_transforms() {
    std::vector<LinearTransform> transforms;
    transforms.push_back(
            {"full", [](const iom::TensorView& view, std::size_t) {
                 return view;
             }});
    transforms.push_back(
            {"select of the first shared axis at its last index",
             [](const iom::TensorView& view, std::size_t leading) {
                 REQUIRE(leading >= 1);
                 return view.select(
                         0, view.spec().shape.dimensions()[0] - 1);
             }});
    transforms.push_back(
            {"stepped slice of the first shared axis",
             [](const iom::TensorView& view, std::size_t leading) {
                 REQUIRE(leading >= 1);
                 const std::size_t extent =
                         view.spec().shape.dimensions()[0];
                 return view.slice(0, 0, (extent + 1) / 2, 2);
             }});
    transforms.push_back(
            {"reversed permutation of the shared axes",
             [](const iom::TensorView& view, std::size_t leading) {
                 const std::size_t rank = view.spec().shape.rank() - 2;
                 REQUIRE(leading <= rank);
                 std::vector<std::size_t> order(rank);
                 for (std::size_t axis = 0; axis < rank; ++axis) {
                     order[axis] = axis < leading ? leading - 1 - axis : axis;
                 }
                 return view.permute(std::span<const std::size_t>{order});
             }});
    transforms.push_back(
            {"collapse of the shared axes",
             [](const iom::TensorView& view, std::size_t leading) {
                 const std::span<const std::size_t> dims =
                         view.spec().shape.dimensions();
                 REQUIRE(leading >= 1);
                 REQUIRE(leading <= dims.size() - 2);
                 std::size_t planes = 1;
                 for (std::size_t axis = 0; axis < leading; ++axis) {
                     planes *= dims[axis];
                 }
                 std::vector<std::size_t> reduced{planes};
                 for (std::size_t axis = leading; axis + 2 < dims.size();
                      ++axis) {
                     reduced.push_back(dims[axis]);
                 }
                 return view.reshape_leading(
                         std::span<const std::size_t>{reduced});
             }});
    transforms.push_back(
            {"select of the last shared axis at its last index",
             [](const iom::TensorView& view, std::size_t leading) {
                 REQUIRE(leading >= 1);
                 return view.select(
                         leading - 1,
                         view.spec().shape.dimensions()[leading - 1] - 1);
             }});
    transforms.push_back(
            {"stepped slice of the last shared axis",
             [](const iom::TensorView& view, std::size_t leading) {
                 REQUIRE(leading >= 1);
                 const std::size_t extent =
                         view.spec().shape.dimensions()[leading - 1];
                 return view.slice(leading - 1, 0, (extent + 1) / 2, 2);
             }});
    transforms.push_back(
            {"select of the first shared axis, then collapse",
             [](const iom::TensorView& view, std::size_t leading) {
                 REQUIRE(leading >= 2);
                 const iom::TensorView selected = view.select(
                         0, view.spec().shape.dimensions()[0] - 1);
                 const std::span<const std::size_t> dims =
                         selected.spec().shape.dimensions();
                 std::size_t planes = 1;
                 for (std::size_t axis = 0; axis + 1 < leading; ++axis) {
                     planes *= dims[axis];
                 }
                 std::vector<std::size_t> reduced{planes};
                 for (std::size_t axis = leading - 1; axis + 2 < dims.size();
                      ++axis) {
                     reduced.push_back(dims[axis]);
                 }
                 return selected.reshape_leading(
                         std::span<const std::size_t>{reduced});
             }});
    return transforms;
}

// One shared linear request case: exact runtime extents, the two owners' own
// leading tuples, the leading-only transforms that reduce them to the same
// shared tuple, the layout, and whether the fixture poisons native padding.
struct LinearCase {
    std::string label;
    iom::DataType leaf = iom::DataType::BF16;
    std::size_t inner = 3;
    std::size_t outer = 10;
    std::size_t source_rows = 19;
    std::size_t start_row = 2;
    std::size_t rows = 17;
    std::size_t heads = 2;
    std::size_t head_dim = 5;
    iom::LinearOutputLayout layout = iom::LinearOutputLayout::ordinary;
    std::vector<std::size_t> x_leading;
    std::vector<std::size_t> out_leading;
    std::size_t x_transform = 0;
    std::size_t out_transform = 0;
    // The input owner is seeded through the storage oracle with non-logical
    // padding forced nonzero, so an implementation that reads tile padding
    // instead of masking it is observable.
    bool poison_native_padding = false;
    // The fixture carries the leaf's special classes. `false` selects an
    // all-finite fixture, which is what makes some deliberate negatives
    // selective.
    bool specials = true;
    // Read back both inputs as well as the output. Reserved for the canonical
    // per-leaf case so the suite stays bounded.
    bool verify_inputs = false;
    std::uint64_t salt = 0;
};

struct LinearCaseSpecs {
    iom::TensorSpec x_owner;
    iom::TensorSpec w_owner;
    iom::TensorSpec out_owner;
};

[[nodiscard]] inline iom::TensorSpec linear_spec(
        std::vector<std::size_t> dimensions, iom::DataType leaf) {
    return iom::TensorSpec{iom::TensorShape{std::move(dimensions)}, leaf};
}

[[nodiscard]] inline LinearCaseSpecs linear_case_specs(const LinearCase& item) {
    std::vector<std::size_t> x_dims = item.x_leading;
    x_dims.push_back(item.source_rows);
    x_dims.push_back(item.inner);
    std::vector<std::size_t> out_dims = item.out_leading;
    if (item.layout == iom::LinearOutputLayout::head_planar) {
        out_dims.push_back(item.heads);
    }
    out_dims.push_back(item.rows);
    out_dims.push_back(
            item.layout == iom::LinearOutputLayout::head_planar ? item.head_dim
                                                                : item.outer);
    return LinearCaseSpecs{
            linear_spec(std::move(x_dims), item.leaf),
            linear_spec({item.outer, item.inner}, item.leaf),
            linear_spec(std::move(out_dims), item.leaf)};
}

// One leading-plane fixture profile: the two owners' own tuples and their
// transforms, chosen so both reduce to the same shared leading tuple.
struct LinearLeadingCase {
    const char* label;
    std::vector<std::size_t> x_leading;
    std::size_t x_transform;
    std::vector<std::size_t> out_leading;
    std::size_t out_transform;
};

// The shared profile table: rank two through rank eight, both tiled axes
// non-tile, distinct per-operand plane offsets and strides, and one profile per
// leading transform.
[[nodiscard]] inline std::vector<LinearLeadingCase> linear_leading_cases() {
    return {
            {"rank two full", {}, 0, {}, 0},
            {"rank three full", {2}, 0, {2}, 0},
            {"rank three selected input plane", {3}, 1, {}, 0},
            {"rank four distinct owner offsets", {3, 2}, 1, {4, 2}, 1},
            {"rank five distinct strides by selected axis",
             {2, 3, 2}, 5, {2, 3}, 0},
            {"rank five reversed permutation", {2, 3, 4}, 3, {2, 3, 4}, 3},
            {"rank six collapse", {2, 3, 2}, 4, {2, 3, 2}, 4},
            {"rank six select then collapse", {4, 3, 2}, 7, {4, 3, 2}, 7},
            {"rank seven sliced and permuted", {2, 4, 2}, 6, {2, 4, 2}, 6},
            {"rank seven distinct strides", {2, 4, 2}, 5, {2, 4}, 0},
            {"rank eight full", {2, 2, 2, 2, 2, 2}, 0, {2, 2, 2, 2, 2, 2}, 0},
            {"rank eight stepped slice", {2, 2, 2, 2, 2, 2}, 2,
             {2, 2, 2, 2, 2, 2}, 2},
            {"rank eight selected plane", {2, 2, 2, 2, 2, 2}, 1,
             {2, 2, 2, 2, 2, 2}, 1},
    };
}

// The shared case matrix: the canonical fixture with every row run and both
// layouts, the non-tile inner extents that cross 16-wide and 32-wide inner
// tiles, the LM-head request, the leading-plane profiles, and the poisoned
// native padding fixtures.
[[nodiscard]] inline std::vector<LinearCase> linear_cases(
        std::span<const iom::DataType> leaves) {
    REQUIRE_FALSE(leaves.empty());
    std::vector<LinearCase> cases;
    std::uint64_t salt = 0x1000;
    const auto add = [&cases, &salt](LinearCase item) {
        item.salt = item.salt == 0 ? ++salt : item.salt;
        cases.push_back(std::move(item));
    };
    const std::vector<LinearLeadingCase> profiles = linear_leading_cases();

    for (const iom::DataType leaf : leaves) {
        // The canonical fixture: `I=3, O=10, H=2, D=5, T=19, s=2` across the
        // exact runs `R=17`, `R=16`, `R=15`, and `R=1`, ordinary and
        // head-planar.
        for (const std::size_t rows : {std::size_t{1}, std::size_t{15},
                                       std::size_t{16}, std::size_t{17}}) {
            for (const iom::LinearOutputLayout layout :
                 {iom::LinearOutputLayout::ordinary,
                  iom::LinearOutputLayout::head_planar}) {
                LinearCase item;
                item.label =
                        std::string("canonical I=3 O=10 H=2 D=5 T=19 s=2 R=")
                        + std::to_string(rows)
                        + (layout == iom::LinearOutputLayout::ordinary
                                   ? " ordinary"
                                   : " head_planar");
                item.leaf = leaf;
                item.rows = rows;
                item.layout = layout;
                item.verify_inputs =
                        rows == 17
                        && layout == iom::LinearOutputLayout::ordinary;
                add(std::move(item));
            }
        }
        // The non-tile inner extent: `I=65` spans five 16-wide inner tiles and
        // three 32-wide inner groups, and `I=33` crosses a 32-wide group with a
        // one-wide tail. Both keep the inner tail, the accumulation order, and
        // the packed sub-byte tails observable.
        for (const std::size_t inner : {std::size_t{65}, std::size_t{33}}) {
            for (const iom::LinearOutputLayout layout :
                 {iom::LinearOutputLayout::ordinary,
                  iom::LinearOutputLayout::head_planar}) {
                if (inner == 33
                        && layout == iom::LinearOutputLayout::head_planar) {
                    continue;
                }
                LinearCase item;
                item.label = "inner extent " + std::to_string(inner)
                        + (layout == iom::LinearOutputLayout::ordinary
                                   ? " ordinary"
                                   : " head_planar");
                item.leaf = leaf;
                item.inner = inner;
                item.layout = layout;
                add(std::move(item));
            }
        }
        // The final untied LM head: `s = input_run - 1`, `R = 1`, ordinary
        // mode, and a non-tile vocabulary, with no row-extraction API.
        {
            LinearCase item;
            item.label = "LM head last row with [1,V] ordinary output";
            item.leaf = leaf;
            item.inner = 5;
            item.outer = 13;
            item.source_rows = 8;
            item.start_row = 7;
            item.rows = 1;
            item.heads = 1;
            item.head_dim = 13;
            item.verify_inputs = true;
            add(std::move(item));
        }
        // The leading-plane profiles, ordinary mode, for every leaf.
        for (const LinearLeadingCase& profile : profiles) {
            LinearCase item;
            item.label = std::string("ordinary leading profile: ")
                         + profile.label;
            item.leaf = leaf;
            item.x_leading = profile.x_leading;
            item.out_leading = profile.out_leading;
            item.x_transform = profile.x_transform;
            item.out_transform = profile.out_transform;
            add(std::move(item));
        }
        // Poisoned native padding, for the byte-aligned leaves whose padding
        // never shares a byte with a logical cell.
        if (bits_of(leaf) % 8 == 0) {
            for (const iom::LinearOutputLayout layout :
                 {iom::LinearOutputLayout::ordinary,
                  iom::LinearOutputLayout::head_planar}) {
                LinearCase item;
                item.label = "poisoned native padding"
                        + std::string(
                                layout == iom::LinearOutputLayout::ordinary
                                        ? " ordinary"
                                        : " head_planar");
                item.leaf = leaf;
                item.layout = layout;
                item.x_leading = {2};
                item.out_leading = {2};
                item.poison_native_padding = true;
                add(std::move(item));
            }
        }
    }

    // The head-planar leading profiles, for a representative subset of leaves:
    // a rank-eight input would need a rank-nine output, so these profiles stay
    // inside the shared rank limit.
    for (const iom::DataType leaf :
         {iom::DataType::BF16, iom::DataType::F32, iom::DataType::I8,
          iom::DataType::F4_E2M1}) {
        if (!linear_declares(leaves, leaf)) {
            continue;
        }
        for (const std::size_t index : {std::size_t{1}, std::size_t{3},
                                       std::size_t{4}, std::size_t{5},
                                       std::size_t{6}, std::size_t{9}}) {
            const LinearLeadingCase& profile = profiles[index];
            LinearCase item;
            item.label = std::string("head-planar leading profile: ")
                         + profile.label;
            item.leaf = leaf;
            item.layout = iom::LinearOutputLayout::head_planar;
            item.x_leading = profile.x_leading;
            item.out_leading = profile.out_leading;
            item.x_transform = profile.x_transform;
            item.out_transform = profile.out_transform;
            add(std::move(item));
        }
    }
    return cases;
}

// The storage image of one standard-tiled owner whose logical cells come from
// `owner_logical` and whose every non-logical padded slot is forced to an
// all-ones byte. Restricted to byte-aligned leaves, whose padding never shares a
// byte with a logical cell. This is a seeding fixture for the storage oracle,
// not the numerical reference.
[[nodiscard]] inline std::vector<std::byte> linear_poisoned_tiled_image(
        const iom::TensorSpec& owner_spec,
        std::span<const std::byte> owner_logical) {
    const std::size_t bits = bits_of(owner_spec.data_type);
    REQUIRE_EQ(bits % 8, std::size_t{0});
    const std::size_t carrier = bits / 8;
    REQUIRE_EQ(owner_logical.size(), owner_spec.logical_nbytes());
    const std::span<const std::size_t> dims = owner_spec.shape.dimensions();
    const std::size_t elements = owner_spec.shape.element_count();
    std::vector<std::byte> storage(
            canonical_padded_element_count(owner_spec) * carrier,
            std::byte{0xFF});
    std::vector<std::size_t> coordinates(dims.size());
    for (std::size_t linear = 0; linear < elements; ++linear) {
        std::size_t rest = linear;
        for (std::size_t axis = dims.size(); axis-- > 0;) {
            coordinates[axis] = rest % dims[axis];
            rest /= dims[axis];
        }
        const std::size_t slot = canonical_layout_slot(
                owner_spec, std::span<const std::size_t>{coordinates});
        std::memcpy(
                storage.data() + slot * carrier,
                owner_logical.data() + linear * carrier, carrier);
    }
    return storage;
}

// Arms one case's observer window and disarms it on every exit path, including
// a failing assertion and a thrown exception, so teardown always runs in this
// order: queue drain, window close, then tensor destruction.
class LinearCaseWindow final {
public:
    explicit LinearCaseWindow(ConformanceObserver* observer)
            : observer_(observer) {
        if (observer_ != nullptr) {
            observer_->setup_complete();
        }
    }
    ~LinearCaseWindow() {
        if (observer_ != nullptr) {
            observer_->case_complete();
        }
    }
    LinearCaseWindow(const LinearCaseWindow&) = delete;
    LinearCaseWindow& operator=(const LinearCaseWindow&) = delete;
    LinearCaseWindow(LinearCaseWindow&&) = delete;
    LinearCaseWindow& operator=(LinearCaseWindow&&) = delete;

private:
    ConformanceObserver* observer_;
};

// One applied request on the real queue with the declared workspace policy.
struct LinearRealSubmission {
    iom::oid result = 0;
    std::unique_ptr<iom::RawWorkspace> workspace;
};

// Submit one request through the candidate's real queue with exactly the
// queried requirement. The requirement the port reports is also checked against
// the declaration's own prediction, so a port cannot silently change its
// scratch policy.
[[nodiscard]] inline LinearRealSubmission submit_real_linear(
        iom::Device& device, iom::DeviceOps& queue,
        const LinearDeclaration& declaration, const iom::TensorView& x,
        const iom::TensorView& w, iom::TensorView& out, std::size_t start_row,
        std::size_t rows, iom::LinearOutputLayout layout, std::size_t heads,
        std::size_t head_dim) {
    const iom::WorkspaceRequirements requirements =
            queue.linear_workspace_requirements(
                    x, w, out, start_row, rows, layout, heads, head_dim);
    const std::span<const std::size_t> x_dims = x.spec().shape.dimensions();
    const std::span<const std::size_t> w_dims = w.spec().shape.dimensions();
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis + 2 < x_dims.size(); ++axis) {
        planes *= x_dims[axis];
    }
    const LinearShape shape{
            planes, rows, x_dims[x_dims.size() - 1], w_dims[0]};
    const iom::WorkspaceRequirements expected =
            linear_expected_workspace(declaration, x.spec().data_type, shape);
    CHECK_EQ(requirements.bytes, expected.bytes);
    CHECK_EQ(requirements.alignment, expected.alignment);
    LinearRealSubmission submission;
    if (requirements.bytes == 0) {
        submission.result =
                queue.linear(x, w, out, start_row, rows, layout, heads, head_dim);
        return submission;
    }
    submission.workspace = device.create_workspace(requirements.bytes);
    submission.result = queue.linear(
            x, w, out, start_row, rows, layout, heads, head_dim,
            submission.workspace->view().subrange(0, requirements.bytes));
    return submission;
}

// The shared request cases through the candidate's real queue. Every case
// builds its exact fixture, seeds it, and either compares the output against
// the independent reference under the fixed thresholds or asserts capability
// rejection with an unchanged output. A declared leaf that reports
// `Unsupported`, or an undeclared leaf that is accepted, fails here: neither is
// projection conformance.
inline void run_linear_reference_conformance(
        const ConformanceDevices& devices,
        const LinearDeclaration& declaration,
        ConformanceObserver* observer = nullptr,
        AcceleratorStorageOracle* oracle = nullptr) {
    iom::Device& candidate = devices.candidate;
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);

    // The declaration must stay inside the published semantics and inside what
    // the candidate device can store: the suite's fixtures are exactly the
    // declared target matrix, and an undeclarable leaf would make the fixtures
    // unrepresentable instead of conformance-relevant.
    REQUIRE_FALSE(declaration.target_leaves.empty());
    const std::span<const iom::DataType> storable =
            candidate.supported_data_types();
    for (const iom::DataType leaf : declaration.target_leaves) {
        CAPTURE(static_cast<int>(leaf));
        CHECK(linear_applicable(leaf));
        CHECK(linear_declares(storable, leaf));
        CHECK(linear_declared(declaration, leaf));
    }
    for (const iom::DataType leaf : declaration.scalar_leaves) {
        CAPTURE(static_cast<int>(leaf));
        CHECK(linear_declares(declaration.target_leaves, leaf));
        CHECK(!linear_declares(declaration.native_bf16_leaves, leaf));
    }
    for (const iom::DataType leaf : declaration.native_bf16_leaves) {
        CAPTURE(static_cast<int>(leaf));
        CHECK(linear_declares(declaration.target_leaves, leaf));
    }
    for (const iom::DataType leaf : declaration.implemented_leaves) {
        CAPTURE(static_cast<int>(leaf));
        CHECK(linear_declares(declaration.target_leaves, leaf));
        CHECK(linear_available(declaration, leaf));
    }

    const std::vector<LinearTransform> transforms = linear_transforms();
    const std::vector<LinearCase> cases = linear_cases(declaration.target_leaves);
    for (const LinearCase& item : cases) {
        CAPTURE(item.label);
        const LinearCaseSpecs specs = linear_case_specs(item);
        auto x_owner = candidate.create_tensor(specs.x_owner);
        auto w_owner = candidate.create_tensor(specs.w_owner);
        auto out_owner = candidate.create_tensor(specs.out_owner);
        const std::size_t shared = item.x_leading.size();
        const iom::TensorView x_view =
                transforms[item.x_transform].build(x_owner->view(), shared);
        const iom::TensorView w_view = w_owner->view();
        iom::TensorView out_view =
                transforms[item.out_transform].build(out_owner->view(), shared);

        // A fixture that did not keep the shared leading extents identical
        // would be an admission error instead of a capability observation, so
        // the suite would silently test the wrong rule.
        const std::span<const std::size_t> x_dims =
                x_view.spec().shape.dimensions();
        const std::span<const std::size_t> out_dims =
                out_view.spec().shape.dimensions();
        const std::size_t x_rank = x_dims.size();
        REQUIRE(x_dims[x_rank - 2] == item.source_rows);
        REQUIRE(x_dims[x_rank - 1] == item.inner);
        REQUIRE_GE(out_dims.size(), x_rank - 2);
        for (std::size_t axis = 0; axis + 2 < x_rank; ++axis) {
            REQUIRE_EQ(x_dims[axis], out_dims[axis]);
        }
        const std::size_t planes = linear_view_planes(x_view);
        const std::size_t heads =
                item.layout == iom::LinearOutputLayout::head_planar ? item.heads
                                                                    : 1;
        const std::size_t head_dim =
                item.layout == iom::LinearOutputLayout::head_planar
                ? item.head_dim
                : item.outer;

        const std::vector<std::byte> x_owner_bytes = linear_logical_image(
                item.leaf, specs.x_owner.shape.element_count(), item.salt,
                item.specials);
        const std::vector<std::byte> w_owner_bytes = linear_logical_image(
                item.leaf, specs.w_owner.shape.element_count(),
                item.salt ^ 0x51EDull, item.specials);
        const std::vector<std::byte> out_owner_poison = linear_logical_image(
                item.leaf, specs.out_owner.shape.element_count(),
                item.salt ^ 0x1D0Full, true);
        const std::vector<std::byte> x_view_bytes =
                linear_view_image(x_view, x_owner_bytes);
        const std::vector<std::byte> out_view_poison =
                linear_view_image(out_view, out_owner_poison);
        const LinearRawImage x_image = linear_padded_image(
                item.leaf, planes, linear_pad16(item.source_rows),
                linear_pad16(item.inner), item.source_rows, item.inner,
                x_view_bytes, item.salt ^ 0x0F0Full);
        const LinearRawImage w_image = linear_padded_image(
                item.leaf, 1, linear_pad16(item.outer),
                linear_pad16(item.inner), item.outer, item.inner, w_owner_bytes,
                item.salt ^ 0xF0F0ull);
        const LinearOracleRequest request{
                item.leaf,
                planes,
                item.source_rows,
                item.inner,
                item.outer,
                item.start_row,
                item.rows,
                heads,
                head_dim,
                item.layout};
        const LinearReference reference = linear_reference(request, x_image, w_image);

        const LinearCaseWindow window(observer);
        const bool poison_native_padding = item.poison_native_padding
                && oracle != nullptr && bits_of(item.leaf) % 8 == 0;
        if (poison_native_padding) {
            oracle->set_owner_spec(specs.x_owner);
            oracle->seed(
                    x_owner->view(),
                    linear_poisoned_tiled_image(specs.x_owner, x_owner_bytes));
        } else {
            copy_from_host(x_owner->view(), x_owner_bytes);
        }
        copy_from_host(w_owner->view(), w_owner_bytes);
        copy_from_host(out_owner->view(), out_owner_poison);
        auto queue = candidate.create_ops();

        if (!linear_implements(declaration, item.leaf)) {
            // A declared leaf this revision does not implement, and a declared
            // leaf the selected device does not expose, are capability
            // rejections: a negative synchronous result with no output effect
            // that never inspects unusable scratch. The fixtures already carry
            // the leaf's special classes, so the same observation proves that
            // admission never scans operand data for nonfinite values.
            CHECK_THROWS_AS(
                    (void)queue->linear_workspace_requirements(
                            x_view, w_view, out_view, item.start_row, item.rows,
                            item.layout, heads, head_dim),
                    std::runtime_error);
            CHECK_EQ(
                    queue->linear(
                            x_view, w_view, out_view, item.start_row, item.rows,
                            item.layout, heads, head_dim),
                    unsupported);
            require_logical_bytes(
                    out_view, out_view_poison,
                    item.label + ": rejection changed the output");
            if (item.verify_inputs) {
                require_logical_bytes(
                        x_view, x_view_bytes,
                        item.label + ": rejection changed the input");
                require_logical_bytes(
                        w_view, w_owner_bytes,
                        item.label + ": rejection changed the weight");
            }
            continue;
        }

        const LinearRealSubmission submission = submit_real_linear(
                candidate, *queue, declaration, x_view, w_view, out_view,
                item.start_row, item.rows, item.layout, heads, head_dim);
        REQUIRE(iom::oid_is_token(submission.result));
        CHECK_NOTHROW(queue->wait(submission.result));
        const std::vector<std::byte> observed = read_logical(out_view);
        const std::optional<std::string> mismatch =
                linear_compare_image(request, observed, reference, item.label);
        REQUIRE_MESSAGE(!mismatch.has_value(), mismatch.value_or(std::string{}));
        // Every unselected output owner plane still holds its seeded poison, so
        // a projection that writes outside its own logical window or consumes
        // uninitialized storage fails here.
        const std::vector<std::byte> observed_owner =
                read_logical(out_owner->view());
        const std::size_t view_planes = linear_view_planes(out_view);
        const std::size_t owner_planes = linear_view_planes(out_owner->view());
        const std::size_t plane_elements = linear_plane_elements(out_view);
        for (std::size_t owner_plane = 0; owner_plane < owner_planes;
             ++owner_plane) {
            std::optional<std::size_t> view_plane;
            for (std::size_t plane = 0; plane < view_planes; ++plane) {
                if (linear_view_plane(out_view, plane) == owner_plane) {
                    view_plane = plane;
                }
            }
            for (std::size_t element = 0; element < plane_elements; ++element) {
                const std::size_t linear =
                        owner_plane * plane_elements + element;
                const std::uint64_t got = linear_logical_code(
                        observed_owner, item.leaf, linear);
                const std::uint64_t want = view_plane.has_value()
                        ? linear_logical_code(
                                  observed, item.leaf,
                                  view_plane.value() * plane_elements + element)
                        : linear_logical_code(
                                  out_owner_poison, item.leaf, linear);
                CHECK_MESSAGE(
                        got == want,
                        item.label << ": output owner plane " << owner_plane
                                   << " element " << element
                                   << " differs from the projection or from "
                                      "its seeded poison");
            }
        }
        require_logical_bytes(
                x_view, x_view_bytes, item.label + ": projection changed the input");
        require_logical_bytes(
                w_view, w_owner_bytes,
                item.label + ": projection changed the weight");
    }
}

// ---------------------------------------------------------------------------
// Common admission, ownership, workspace, queue-order, and failure cases.
// ---------------------------------------------------------------------------

// One live raw-workspace owner over a fabricated, caller-selected address.
// Workspace liveness, alignment, size, device, and range rules are all
// observable through it without allocating native scratch.
class LinearConformanceWorkspace final : public iom::RawWorkspace {
public:
    LinearConformanceWorkspace(
            iom::Device& device, void* address, std::size_t bytes)
            : iom::RawWorkspace(device, bytes), address_(address) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    void* address_;
};

// A view whose owning workspace is already destroyed: only the owner identity
// survives, which is exactly what the liveness rule observes.
[[nodiscard]] inline iom::RawWorkspaceView linear_dead_workspace_view(
        iom::Device& device, void* address, std::size_t bytes) {
    LinearConformanceWorkspace dead(device, address, bytes);
    return dead.view();
}

// Qualifies the live owner view specifications of one request for the
// recognized non-NONE quantization probe and restores them to `NONE` when its
// scope ends, on every exit path, so no operand is ever destroyed while its
// specification carries a recognized grouped quantization format. The owner
// view itself must carry the format, because a view/owner specification
// mismatch is rejected before the capability decision, and the qualification
// must be gone before teardown: the accelerator tensor destructors derive their
// storage extent from the owner specification inside a `noexcept` destructor.
class LinearQuantizationQualification final {
public:
    LinearQuantizationQualification(
            iom::Tensor& x, iom::Tensor& w, iom::Tensor& out)
            : specs_{qualify(x), qualify(w), qualify(out)} {}

    ~LinearQuantizationQualification() {
        for (iom::TensorSpec* spec : specs_) {
            spec->quantization = iom::QuantizationFormat::NONE;
        }
    }

    LinearQuantizationQualification(const LinearQuantizationQualification&) =
            delete;
    LinearQuantizationQualification& operator=(
            const LinearQuantizationQualification&) = delete;
    LinearQuantizationQualification(LinearQuantizationQualification&&) = delete;
    LinearQuantizationQualification& operator=(
            LinearQuantizationQualification&&) = delete;

private:
    [[nodiscard]] static iom::TensorSpec* qualify(iom::Tensor& owner) {
        iom::TensorSpec& spec = const_cast<iom::TensorSpec&>(owner.view().spec());
        spec.quantization = iom::QuantizationFormat::OCP_MXFP4;
        return &spec;
    }

    std::array<iom::TensorSpec*, 3> specs_;
};

// The common `DeviceOps` linear path of one declared matrix: the shared
// admission, ownership, workspace, queue-order, and failure double. It
// exercises exactly the common machinery a real port inherits — the same
// all-or-nothing registration, request snapshot, and workspace lease path the
// binary and embedding doubles exercise — and it never touches native compute
// or native storage.
class CommonLinearQueue final : public iom::DeviceOps {
public:
    enum class Failure { none, post_acceptance };

    struct LinearRecord {
        std::uint64_t sequence = 0;
        iom::detail::BinaryEntryRegistration entries;
        iom::detail::WorkspaceLease workspace_lease;
        bool retained_failure = false;
    };

    CommonLinearQueue(
            const iom::Device& device, const LinearDeclaration& declaration)
            : iom::DeviceOps(device), declaration_(declaration) {}

    void inject_failure(Failure failure) noexcept {
        next_failure_ = failure;
    }

    [[nodiscard]] const std::vector<LinearRecord>& linear_records()
            const noexcept {
        return records_;
    }

    [[nodiscard]] const std::vector<std::string_view>& submissions()
            const noexcept {
        return submissions_;
    }

    [[nodiscard]] std::size_t registered_at(const void* address) const {
        return registry_.registry.snapshot_for(const_cast<void*>(address)).size();
    }

    // Plays the in-order completion of one accepted submission: a linear record
    // releases or invalidates its owners and completes its workspace lease
    // exactly as a proven-completion worker would, and a view-less copy or probe
    // submission only completes its sequence.
    void finish(std::uint64_t sequence) {
        for (const LinearRecord& record : records_) {
            if (record.sequence != sequence) {
                continue;
            }
            (void)iom::detail::release_or_invalidate_binary_entries(
                    registry_.registry, record.entries, record.retained_failure,
                    !record.retained_failure);
            iom::detail::complete_workspace_lease(
                    registry_, record.workspace_lease, true);
            complete(sequence);
            return;
        }
        if (std::find(plain_.begin(), plain_.end(), sequence) != plain_.end()) {
            complete(sequence);
            return;
        }
        throw std::invalid_argument("unknown common linear sequence");
    }

    // A view-less submission that observes queue identity and sequence
    // allocation without reusing an accepted operation.
    iom::oid probe() {
        const iom::oid token = submit([this](std::uint64_t) {
            submissions_.push_back("probe");
        });
        plain_.push_back(token_sequence(token));
        return token;
    }

protected:
    iom::oid copy_impl(
            const iom::TensorView& source,
            iom::TensorView& destination) override {
        if (source.spec() != destination.spec()) {
            throw std::invalid_argument(
                    "common linear queue copy needs identical specs");
        }
        const iom::oid token = submit([this](std::uint64_t) {
            submissions_.push_back("copy");
        });
        plain_.push_back(token_sequence(token));
        return token;
    }

    // The pure requirement hook reports the declaration's exact policy for a
    // request inside the declared matrix and touches no view metadata, queue
    // state, or allocation. A leaf outside the declared matrix is a capability
    // rejection, which the real queue reports as `Unsupported` and the suite
    // therefore observes there instead of through this double.
    iom::WorkspaceRequirements linear_workspace_requirements_impl(
            const iom::TensorView& x, const iom::TensorView& w,
            const iom::TensorView&, std::size_t, std::size_t rows,
            iom::LinearOutputLayout, std::size_t, std::size_t) override {
        const iom::DataType leaf = x.spec().data_type;
        if (!linear_declared(declaration_, leaf)
                || !linear_available(declaration_, leaf)) {
            throw std::runtime_error(
                    "common linear double declares exactly its matrix");
        }
        return linear_expected_workspace(
                declaration_, leaf, linear_shape_of(x, w, rows));
    }

    iom::oid linear_impl(const LinearRequest& request) override {
        const Failure failure = std::exchange(next_failure_, Failure::none);
        iom::detail::Fence fence;
        fence.invoke = [](const iom::detail::Fence&) noexcept {
            return iom::detail::FenceResult::pending();
        };
        return submit_linear(
                request, registry_, queue_id_, fence,
                [this, failure](
                        std::uint64_t sequence, const LinearRequest& snapshot,
                        iom::detail::BinaryEntryRegistration entries) {
                    records_.push_back(LinearRecord{
                            sequence, entries, snapshot.workspace_lease,
                            failure == Failure::post_acceptance});
                    submissions_.push_back("linear");
                    if (failure == Failure::post_acceptance) {
                        commit_failure(
                                sequence,
                                std::make_exception_ptr(
                                        std::runtime_error(
                                                "common linear retained "
                                                "failure")));
                    }
                });
    }

private:
    // The request shape any workspace formula needs: the checked logical
    // leading product `P`, the row run `R`, the inner extent `I`, and the outer
    // extent `O`, all read from the already validated views.
    [[nodiscard]] static LinearShape linear_shape_of(
            const iom::TensorView& x, const iom::TensorView& w,
            std::size_t rows) {
        const std::span<const std::size_t> x_dims =
                x.spec().shape.dimensions();
        const std::span<const std::size_t> w_dims =
                w.spec().shape.dimensions();
        std::size_t planes = 1;
        for (std::size_t axis = 0; axis + 2 < x_dims.size(); ++axis) {
            planes *= x_dims[axis];
        }
        return LinearShape{
                planes, rows, x_dims[x_dims.size() - 1], w_dims[0]};
    }

    iom::detail::RegistryState registry_;
    iom::detail::QueueId queue_id_ =
            iom::detail::allocate_queue_id(registry_);
    LinearDeclaration declaration_{};
    Failure next_failure_ = Failure::none;
    std::vector<LinearRecord> records_;
    std::vector<std::string_view> submissions_;
    std::vector<std::uint64_t> plain_;
};

// Requires a deferred linear failure to rethrow `std::runtime_error` on every
// repeat wait, without pinning the message wording.
inline void expect_repeated_linear_failure(iom::DeviceOps& queue, iom::oid token) {
    std::string message;
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool caught = false;
        try {
            queue.wait(token);
        } catch (const std::runtime_error& error) {
            caught = true;
            if (message.empty()) {
                message = error.what();
            } else {
                CHECK_EQ(std::string_view(error.what()), message);
            }
        }
        CHECK(caught);
    }
    CHECK_FALSE(message.empty());
}

// The common admission, ownership, workspace, queue-order, and failure cases.
// They run against the candidate's own tensors and the shared linear double, so
// they observe the common `DeviceOps` contract every port inherits, including
// the exact rejection categories, the pure query, the declared scratch policy,
// FIFO acceptance, retained failures, quarantine, and release after drain. They
// never claim native numerical conformance: that is what the reference
// conformance above is for, and an unported port reaches capability rejection
// there rather than a comparison.
inline void run_linear_common_conformance(
        const ConformanceDevices& devices,
        const LinearDeclaration& declaration,
        ConformanceObserver* observer = nullptr) {
    iom::Device& candidate = devices.candidate;
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const iom::oid exhausted = iom::to_oid(iom::OidError::ResourceExhausted);
    const std::span<const iom::DataType> storable =
            candidate.supported_data_types();
    const auto supported_by_device = [&storable](iom::DataType leaf) {
        return linear_declares(storable, leaf);
    };
    const iom::DataType declared_leaf =
            linear_declares(declaration.target_leaves, iom::DataType::BF16)
            ? iom::DataType::BF16
            : declaration.target_leaves.front();
    const iom::LinearOutputLayout ordinary = iom::LinearOutputLayout::ordinary;
    const iom::LinearOutputLayout head_planar = iom::LinearOutputLayout::head_planar;
    const LinearShape shared_shape{1, 17, 3, 10};

    // The pure requirement query: the declared policy verbatim, validated with
    // the submission's own categories, and with no record, submission,
    // registration, lease, or sequence consumption.
    {
        auto x = candidate.create_tensor(linear_spec({19, 3}, declared_leaf));
        auto w = candidate.create_tensor(linear_spec({10, 3}, declared_leaf));
        auto out = candidate.create_tensor(linear_spec({17, 10}, declared_leaf));
        auto foreign =
                devices.foreign.create_tensor(linear_spec({19, 3}, declared_leaf));
        const LinearCaseWindow window(observer);
        CommonLinearQueue queue(candidate, declaration);
        const iom::WorkspaceRequirements expected =
                linear_expected_workspace(declaration, declared_leaf, shared_shape);
        CHECK(
                queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10)
                == expected);
        CHECK(
                queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10)
                == queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10));
        CHECK_THROWS_AS(
                (void)queue.linear_workspace_requirements(
                        foreign->view(), w->view(), out->view(), 2, 17, ordinary,
                        1, 10),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 18, ordinary, 1,
                        10),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 0, ordinary, 1,
                        10),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 20, 1, ordinary, 1,
                        10),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 2,
                        10),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 17, head_planar, 2,
                        5),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue.linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 17,
                        static_cast<iom::LinearOutputLayout>(31), 1, 10),
                std::invalid_argument);
        CHECK(queue.linear_records().empty());
        CHECK(queue.submissions().empty());
        CHECK_EQ(
                queue.registered_at(x->view().native_handle()), std::size_t{0});
        CHECK_EQ(
                queue.registered_at(w->view().native_handle()), std::size_t{0});
        CHECK_EQ(
                queue.registered_at(out->view().native_handle()), std::size_t{0});
        // The consumed token sequence proves that no query submitted anything.
        CHECK_EQ(token_sequence(queue.probe()), 1);
        CHECK_EQ(queue.submissions().size(), std::size_t{1});
    }

    // Every leaf of the declared matrix is accepted by the common admission
    // path with the declared scratch policy, retains its three distinct owners
    // until proven completion, and releases them exactly then. A leaf the
    // selected device does not expose stays a capability rejection in both
    // directions.
    for (const iom::DataType leaf : declaration.target_leaves) {
        if (!supported_by_device(leaf)) {
            continue;
        }
        CAPTURE(static_cast<int>(leaf));
        auto x = candidate.create_tensor(linear_spec({19, 3}, leaf));
        auto w = candidate.create_tensor(linear_spec({10, 3}, leaf));
        auto out = candidate.create_tensor(linear_spec({17, 10}, leaf));
        const iom::WorkspaceRequirements policy =
                linear_expected_workspace(declaration, leaf, shared_shape);
        LinearConformanceWorkspace scratch(
                candidate, reinterpret_cast<void*>(0x5100), policy.bytes + 64);
        const LinearCaseWindow window(observer);
        CommonLinearQueue queue(candidate, declaration);
        if (!linear_available(declaration, leaf)) {
            CHECK_THROWS_AS(
                    (void)queue.linear_workspace_requirements(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10),
                    std::runtime_error);
            CHECK_EQ(
                    queue.linear(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10),
                    unsupported);
            CHECK(queue.linear_records().empty());
            continue;
        }
        const iom::oid token = policy.bytes == 0
                ? queue.linear(
                          x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                          10)
                : queue.linear(
                          x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                          10, scratch.view().subrange(0, policy.bytes));
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_EQ(queue.linear_records().size(), std::size_t{1});
        CHECK_EQ(queue.linear_records().back().entries.count, std::size_t{3});
        CHECK_EQ(
                queue.registered_at(x->view().native_handle()), std::size_t{1});
        CHECK_EQ(
                queue.registered_at(w->view().native_handle()), std::size_t{1});
        CHECK_EQ(
                queue.registered_at(out->view().native_handle()), std::size_t{1});
        if (policy.bytes != 0) {
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5100)),
                    std::size_t{1});
        }
        queue.finish(token_sequence(token));
        CHECK_NOTHROW(queue.wait(token));
        CHECK_EQ(
                queue.registered_at(x->view().native_handle()), std::size_t{0});
        CHECK_EQ(
                queue.registered_at(w->view().native_handle()), std::size_t{0});
        CHECK_EQ(
                queue.registered_at(out->view().native_handle()), std::size_t{0});
        if (policy.bytes != 0) {
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5100)),
                    std::size_t{0});
        }
    }

    // A leaf outside the declared matrix is a capability rejection on the
    // candidate's real queue, after structural validation and before any
    // owner, sequence, or scratch effect; the two inapplicable leaves are
    // `Unsupported` on every backend and are never superseded by a capability
    // record. The real queue is used here because that is where the port's own
    // capability decision lives.
    {
        auto queue = candidate.create_ops();
        std::vector<iom::DataType> undeclared;
        for (const iom::DataType leaf : kLinearLeafSpan) {
            if (!linear_declares(declaration.target_leaves, leaf)
                    && supported_by_device(leaf)) {
                undeclared.push_back(leaf);
            }
        }
        for (const iom::DataType leaf : kLinearInapplicableSpan) {
            if (supported_by_device(leaf)) {
                undeclared.push_back(leaf);
            }
        }
        for (const iom::DataType leaf : undeclared) {
            CAPTURE(static_cast<int>(leaf));
            auto x = candidate.create_tensor(linear_spec({19, 3}, leaf));
            auto w = candidate.create_tensor(linear_spec({10, 3}, leaf));
            auto out = candidate.create_tensor(linear_spec({17, 10}, leaf));
            const LinearCaseWindow window(observer);
            CHECK_THROWS_AS(
                    (void)queue->linear_workspace_requirements(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10),
                    std::runtime_error);
            CHECK_EQ(
                    queue->linear(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10),
                    unsupported);
        }
    }

    // Structural admission, checked arithmetic, device identity, aliasing, the
    // conservative output/input overlap rule, mixed leaves, and non-`NONE`
    // quantization, all with their own categories and none of them consuming a
    // sequence or registering an owner.
    {
        auto x = candidate.create_tensor(linear_spec({2, 19, 3}, declared_leaf));
        auto w = candidate.create_tensor(linear_spec({10, 3}, declared_leaf));
        auto out = candidate.create_tensor(linear_spec({2, 17, 10}, declared_leaf));
        auto wrong_rows = candidate.create_tensor(linear_spec({2, 16, 10}, declared_leaf));
        auto wrong_leading = candidate.create_tensor(linear_spec({3, 17, 10}, declared_leaf));
        auto rank_three_weight = candidate.create_tensor(linear_spec({2, 10, 3}, declared_leaf));
        auto wrong_feature = candidate.create_tensor(linear_spec({10, 4}, declared_leaf));
        auto wide_owner = candidate.create_tensor(linear_spec({3, 19, 3}, declared_leaf));
        auto head_planar_out = candidate.create_tensor(linear_spec({2, 2, 17, 5}, declared_leaf));
        auto rank_eight = candidate.create_tensor(
                linear_spec({1, 1, 1, 1, 1, 1, 19, 3}, declared_leaf));
        auto rank_nine_out = candidate.create_tensor(
                linear_spec({1, 1, 1, 1, 1, 2, 17, 5}, declared_leaf));
        auto shared = candidate.create_tensor(linear_spec({2, 19, 3}, declared_leaf));
        auto alias_owner = candidate.create_tensor(linear_spec({10, 3}, declared_leaf));
        auto alias_out = candidate.create_tensor(linear_spec({5, 10}, declared_leaf));
        const iom::WorkspaceRequirements alias_policy = linear_expected_workspace(
                declaration, declared_leaf, LinearShape{1, 5, 3, 10});
        LinearConformanceWorkspace alias_scratch(
                candidate, reinterpret_cast<void*>(0x5000),
                alias_policy.bytes + 64);
        auto foreign_x = devices.foreign.create_tensor(linear_spec({2, 19, 3}, declared_leaf));
        const iom::DataType other_leaf =
                supported_by_device(iom::DataType::F16) ? iom::DataType::F16
                                                        : declared_leaf;
        auto other = candidate.create_tensor(linear_spec({2, 19, 3}, other_leaf));
        auto grouped_x = candidate.create_tensor(linear_spec({2, 19, 3}, iom::DataType::BF16));
        auto grouped_w = candidate.create_tensor(linear_spec({10, 3}, iom::DataType::BF16));
        auto grouped_out = candidate.create_tensor(linear_spec({2, 17, 10}, iom::DataType::BF16));
        const LinearCaseWindow window(observer);
        CommonLinearQueue queue(candidate, declaration);
        const auto reject = [&queue, invalid](
                                    const iom::TensorView& x_view,
                                    const iom::TensorView& w_view,
                                    iom::TensorView& out_view, std::size_t rows,
                                    iom::LinearOutputLayout layout,
                                    std::size_t heads, std::size_t head_dim,
                                    iom::oid expected, const char* what) {
            const iom::oid result = queue.linear(
                    x_view, w_view, out_view, 2, rows, layout, heads, head_dim);
            CHECK_MESSAGE(result == expected, what);
            CHECK(queue.linear_records().empty());
        };
        reject(
                x->view(), rank_three_weight->view(), out->view(), 17, ordinary,
                1, 10, invalid, "a rank-three weight view is rejected");
        reject(
                x->view(), wrong_feature->view(), out->view(), 17, ordinary, 1,
                10, invalid,
                "a weight input extent that differs from the input feature "
                "extent is rejected");
        reject(
                x->view(), w->view(), wrong_rows->view(), 17, ordinary, 1, 10,
                invalid, "an output row extent that differs from R is rejected");
        reject(
                x->view(), w->view(), wrong_leading->view(), 17, ordinary, 1, 10,
                invalid, "input and output leading tuples must match exactly");
        reject(
                x->view(), w->view(), out->view(), 17, ordinary, 2, 10, invalid,
                "ordinary mode requires H=1 and D=O");
        reject(
                x->view(), w->view(), head_planar_out->view(), 17, head_planar, 2,
                2, invalid, "head-planar mode requires the checked O=H*D");
        reject(
                x->view(), w->view(), out->view(), 0, ordinary, 1, 10, invalid,
                "a zero projected row count is rejected");
        reject(
                x->view(), w->view(), out->view(), 18, ordinary, 1, 10, invalid,
                "a row window past the source rows is rejected");
        reject(
                x->view(), w->view(), out->view(), 17,
                static_cast<iom::LinearOutputLayout>(31), 1, 10, invalid,
                "an unknown layout value is rejected");
        reject(
                rank_eight->view(), w->view(), rank_nine_out->view(), 17,
                head_planar, 2, 5, invalid,
                "head-planar rank growth beyond eight is rejected");
        reject(
                foreign_x->view(), w->view(), out->view(), 17, ordinary, 1, 10,
                invalid, "a view from another device instance is rejected");
        // Output/input aliasing is rejected conservatively on owner identity
        // even where the selected windows appear disjoint.
        {
            iom::TensorView aliased_out = shared->view().select(0, 0);
            reject(
                    shared->view().select(0, 1), w->view(), aliased_out, 17,
                    ordinary, 1, 10, invalid,
                    "an output view of the input's own owner is rejected");
        }
        if (other_leaf != declared_leaf) {
            reject(
                    other->view(), w->view(), out->view(), 17, ordinary, 1, 10,
                    unsupported,
                    "one recognized leaf shared by all three views is required");
        }
        {
            iom::TensorView unknown_type = x->view();
            const_cast<iom::TensorSpec&>(unknown_type.spec()).data_type =
                    static_cast<iom::DataType>(127);
            reject(
                    unknown_type, w->view(), out->view(), 17, ordinary, 1, 10,
                    invalid, "an unknown leaf encoding is rejected");
            iom::TensorView zero_extent = x->view();
            const_cast<std::size_t*>(
                    zero_extent.spec().shape.dimensions().data())[0] = 0;
            reject(
                    zero_extent, w->view(), out->view(), 17, ordinary, 1, 10,
                    invalid, "a zero runtime extent is rejected");
        }
        {
            // Recognized non-`NONE` quantization is a capability rejection, so
            // this fixture qualifies all three operands with the same recognized
            // format. Qualifying fewer operands would be a specification
            // mismatch that precedes the capability decision. The qualification
            // is scoped and removed again on every exit path, before any operand
            // is destroyed.
            const LinearQuantizationQualification qualified(
                    *grouped_x, *grouped_w, *grouped_out);
            iom::TensorView grouped_out_view = grouped_out->view();
            reject(
                    grouped_x->view(), grouped_w->view(), grouped_out_view, 17,
                    ordinary, 1, 10, unsupported,
                    "recognized non-NONE quantization is unsupported");
        }
        {
            // Checked plane-addressing arithmetic overflows instead of wrapping,
            // and the report stays the established Overflow category. The
            // matching `{3, 17, 10}` output keeps every earlier structural
            // check satisfied, so the checked view arithmetic is what fails.
            iom::TensorView wide = wide_owner->view();
            const_cast<std::size_t*>(wide.plane_strides().data())[0] =
                    std::numeric_limits<std::size_t>::max();
            const iom::oid result = queue.linear(
                    wide, w->view(), wrong_leading->view(), 2, 17, ordinary, 1,
                    10);
            CHECK_EQ(result, iom::to_oid(iom::OidError::Overflow));
            CHECK(queue.linear_records().empty());
        }
        // Input/read aliasing between `x` and `w` is valid and deduplicates
        // their single registration.
        {
            const iom::oid aliased = alias_policy.bytes == 0
                    ? queue.linear(
                              alias_owner->view(), alias_owner->view(),
                              alias_out->view(), 0, 5, ordinary, 1, 10)
                    : queue.linear(
                              alias_owner->view(), alias_owner->view(),
                              alias_out->view(), 0, 5, ordinary, 1, 10,
                              alias_scratch.view().subrange(
                                      0, alias_policy.bytes));
            REQUIRE(iom::oid_is_token(aliased));
            CHECK_EQ(
                    queue.registered_at(alias_owner->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue.registered_at(alias_out->view().native_handle()),
                    std::size_t{1});
            queue.finish(token_sequence(aliased));
            CHECK_NOTHROW(queue.wait(aliased));
        }
        // Every rejection above consumed no sequence: the accepted aliased
        // submission is this block's first token, so the next sequence is two.
        CHECK_EQ(token_sequence(queue.probe()), 2);
    }

    // The declared scratch policy: a zero requirement accepts and ignores any
    // supplied range, while a positive requirement rejects every unusable range
    // before dispatch, leases exactly the live disjoint range, and never reuses
    // a range whose completion is unproven.
    for (const iom::DataType leaf : declaration.target_leaves) {
        if (!supported_by_device(leaf) || !linear_available(declaration, leaf)) {
            continue;
        }
        CAPTURE(static_cast<int>(leaf));
        const iom::WorkspaceRequirements policy =
                linear_expected_workspace(declaration, leaf, shared_shape);
        auto x = candidate.create_tensor(linear_spec({19, 3}, leaf));
        auto w = candidate.create_tensor(linear_spec({10, 3}, leaf));
        auto out = candidate.create_tensor(linear_spec({17, 10}, leaf));
        const std::uintptr_t aligned_base = 0x5200;
        LinearConformanceWorkspace aligned(
                candidate, reinterpret_cast<void*>(aligned_base),
                2 * policy.bytes + 64);
        LinearConformanceWorkspace misaligned(
                candidate, reinterpret_cast<void*>(0x5401), policy.bytes + 64);
        LinearConformanceWorkspace foreign(
                devices.foreign, reinterpret_cast<void*>(0x5500),
                policy.bytes + 64);
        LinearConformanceWorkspace overlapping(
                candidate, x->view().native_handle(), policy.bytes + 64);
        const iom::RawWorkspaceView dead = linear_dead_workspace_view(
                candidate, reinterpret_cast<void*>(0x5600), policy.bytes + 64);
        const LinearCaseWindow window(observer);
        CommonLinearQueue queue(candidate, declaration);
        if (policy.bytes == 0) {
            // A zero requirement neither validates nor leases an unused range:
            // every supplied view is accepted and nothing is registered.
            const iom::oid tokens[5] = {
                    queue.linear(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10),
                    queue.linear(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10, dead),
                    queue.linear(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10, foreign.view()),
                    queue.linear(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10, misaligned.view()),
                    queue.linear(
                            x->view(), w->view(), out->view(), 2, 17, ordinary,
                            1, 10, overlapping.view()),
            };
            for (const iom::oid token : tokens) {
                REQUIRE(iom::oid_is_token(token));
            }
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5200)),
                    std::size_t{0});
            CHECK_EQ(
                    queue.registered_at(reinterpret_cast<void*>(0x5500)),
                    std::size_t{0});
            for (const iom::oid token : tokens) {
                queue.finish(token_sequence(token));
                CHECK_NOTHROW(queue.wait(token));
            }
            continue;
        }
        const std::size_t bytes = policy.bytes;
        const iom::RawWorkspaceView usable = aligned.view().subrange(0, bytes);
        CHECK_EQ(
                queue.linear(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10),
                invalid);
        CHECK_EQ(
                queue.linear(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10, dead),
                invalid);
        CHECK_EQ(
                queue.linear(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10, foreign.view()),
                invalid);
        CHECK_EQ(
                queue.linear(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10, aligned.view().subrange(0, bytes - 32)),
                invalid);
        CHECK_EQ(
                queue.linear(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10, misaligned.view()),
                invalid);
        CHECK_EQ(
                queue.linear(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10, overlapping.view()),
                invalid);
        CHECK(queue.linear_records().empty());

        const iom::oid leased = queue.linear(
                x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 10,
                usable);
        REQUIRE(iom::oid_is_token(leased));
        CHECK_EQ(
                queue.registered_at(reinterpret_cast<void*>(0x5200)),
                std::size_t{1});
        // A live overlapping range is bounded-resource exhaustion and leaves
        // neither record nor lease behind.
        CHECK_EQ(
                queue.linear(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10, usable),
                exhausted);
        // A disjoint aligned subrange of the same owner is independent.
        const iom::oid disjoint = queue.linear(
                x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 10,
                aligned.view().subrange(bytes, bytes));
        REQUIRE(iom::oid_is_token(disjoint));
        CHECK_EQ(
                queue.registered_at(
                        reinterpret_cast<void*>(aligned_base + bytes)),
                std::size_t{1});
        queue.finish(token_sequence(leased));
        queue.finish(token_sequence(disjoint));
        CHECK_NOTHROW(queue.wait(leased));
        CHECK_NOTHROW(queue.wait(disjoint));
        CHECK_EQ(
                queue.registered_at(reinterpret_cast<void*>(0x5200)),
                std::size_t{0});
        CHECK_EQ(
                queue.registered_at(
                        reinterpret_cast<void*>(aligned_base + bytes)),
                std::size_t{0});
        // Proven completion releases the range for reuse.
        const iom::oid reused = queue.linear(
                x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 10,
                usable);
        REQUIRE(iom::oid_is_token(reused));
        queue.finish(token_sequence(reused));
        CHECK_NOTHROW(queue.wait(reused));
    }

    // Queue order and accepted failures: a producer copy, the projection, and
    // an output consumer are accepted in submission order; operand data is
    // queued data that admission never scans; and an accepted failure is a
    // positive token whose waits rethrow the same error while its dependent
    // output is never consumed.
    {
        const iom::WorkspaceRequirements policy =
                linear_expected_workspace(declaration, declared_leaf, shared_shape);
        auto source = candidate.create_tensor(linear_spec({19, 3}, declared_leaf));
        auto x = candidate.create_tensor(linear_spec({19, 3}, declared_leaf));
        auto w = candidate.create_tensor(linear_spec({10, 3}, declared_leaf));
        auto out = candidate.create_tensor(linear_spec({17, 10}, declared_leaf));
        auto destination =
                candidate.create_tensor(linear_spec({17, 10}, declared_leaf));
        LinearConformanceWorkspace scratch(
                candidate, reinterpret_cast<void*>(0x5700), policy.bytes + 64);
        const LinearCaseWindow window(observer);
        // The operand carries the leaf's special classes, so an admission that
        // scanned for nonfinite values would reject it synchronously instead of
        // accepting the token below.
        copy_from_host(
                x->view(),
                linear_logical_image(
                        declared_leaf, 19 * 3, 0xC0FFEEull, true));
        CommonLinearQueue queue(candidate, declaration);
        const iom::oid producer = queue.copy(source->view(), x->view());
        REQUIRE(iom::oid_is_token(producer));
        const iom::oid projected = policy.bytes == 0
                ? queue.linear(
                          x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                          10)
                : queue.linear(
                          x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                          10, scratch.view().subrange(0, policy.bytes));
        REQUIRE(iom::oid_is_token(projected));
        const iom::oid consumer = queue.copy(out->view(), destination->view());
        REQUIRE(iom::oid_is_token(consumer));
        REQUIRE_EQ(queue.submissions().size(), std::size_t{3});
        CHECK_EQ(queue.submissions()[0], std::string_view{"copy"});
        CHECK_EQ(queue.submissions()[1], std::string_view{"linear"});
        CHECK_EQ(queue.submissions()[2], std::string_view{"copy"});
        queue.finish(token_sequence(producer));
        queue.finish(token_sequence(projected));
        queue.finish(token_sequence(consumer));
        CHECK_NOTHROW(queue.wait(producer));
        CHECK_NOTHROW(queue.wait(projected));
        CHECK_NOTHROW(queue.wait(consumer));

        queue.inject_failure(CommonLinearQueue::Failure::post_acceptance);
        const iom::oid failed = policy.bytes == 0
                ? queue.linear(
                          x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                          10)
                : queue.linear(
                          x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                          10, scratch.view().subrange(0, policy.bytes));
        REQUIRE(iom::oid_is_token(failed));
        const iom::oid following = queue.copy(out->view(), destination->view());
        REQUIRE(iom::oid_is_token(following));
        queue.finish(token_sequence(failed));
        expect_repeated_linear_failure(queue, failed);
        queue.finish(token_sequence(following));
        CHECK_NOTHROW(queue.wait(following));
    }

    // Unknown completion quarantines owners and the workspace lease; proven
    // completion releases the lease and keeps an accepted failure's owners
    // invalidated rather than reusable; and after the drain a fresh queue
    // accepts a safe valid operation and the same range again.
    {
        const iom::WorkspaceRequirements policy =
                linear_expected_workspace(declaration, declared_leaf, shared_shape);
        auto x = candidate.create_tensor(linear_spec({19, 3}, declared_leaf));
        auto w = candidate.create_tensor(linear_spec({10, 3}, declared_leaf));
        auto out = candidate.create_tensor(linear_spec({17, 10}, declared_leaf));
        LinearConformanceWorkspace scratch(
                candidate, reinterpret_cast<void*>(0x5800), policy.bytes + 64);
        const LinearCaseWindow window(observer);
        {
            CommonLinearQueue queue(candidate, declaration);
            const iom::oid outstanding = policy.bytes == 0
                    ? queue.linear(
                              x->view(), w->view(), out->view(), 2, 17,
                              ordinary, 1, 10)
                    : queue.linear(
                              x->view(), w->view(), out->view(), 2, 17,
                              ordinary, 1, 10,
                              scratch.view().subrange(0, policy.bytes));
            REQUIRE(iom::oid_is_token(outstanding));
            CHECK_EQ(
                    queue.registered_at(x->view().native_handle()),
                    std::size_t{1});
            if (policy.bytes != 0) {
                // The lease is retained until completion is proven, so the same
                // range is not reusable yet.
                CHECK_EQ(
                        queue.registered_at(reinterpret_cast<void*>(0x5800)),
                        std::size_t{1});
                CHECK_EQ(
                        queue.linear(
                                x->view(), w->view(), out->view(), 2, 17,
                                ordinary, 1, 10,
                                scratch.view().subrange(0, policy.bytes)),
                        exhausted);
            }
            queue.finish(token_sequence(outstanding));
            CHECK_NOTHROW(queue.wait(outstanding));
            CHECK_EQ(
                    queue.registered_at(x->view().native_handle()),
                    std::size_t{0});
            if (policy.bytes != 0) {
                CHECK_EQ(
                        queue.registered_at(reinterpret_cast<void*>(0x5800)),
                        std::size_t{0});
            }
        }
        {
            // An accepted failure keeps its owners invalidated through the
            // quarantine even after its completion is proven, while the
            // workspace lease stays independent of the data error.
            CommonLinearQueue queue(candidate, declaration);
            queue.inject_failure(CommonLinearQueue::Failure::post_acceptance);
            const iom::oid failed = policy.bytes == 0
                    ? queue.linear(
                              x->view(), w->view(), out->view(), 2, 17,
                              ordinary, 1, 10)
                    : queue.linear(
                              x->view(), w->view(), out->view(), 2, 17,
                              ordinary, 1, 10,
                              scratch.view().subrange(0, policy.bytes));
            REQUIRE(iom::oid_is_token(failed));
            queue.finish(token_sequence(failed));
            expect_repeated_linear_failure(queue, failed);
            CHECK_EQ(
                    queue.registered_at(x->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue.registered_at(w->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue.registered_at(out->view().native_handle()),
                    std::size_t{1});
        }
        {
            CommonLinearQueue queue(candidate, declaration);
            const iom::oid token = policy.bytes == 0
                    ? queue.linear(
                              x->view(), w->view(), out->view(), 2, 17,
                              ordinary, 1, 10)
                    : queue.linear(
                              x->view(), w->view(), out->view(), 2, 17,
                              ordinary, 1, 10,
                              scratch.view().subrange(0, policy.bytes));
            REQUIRE(iom::oid_is_token(token));
            queue.finish(token_sequence(token));
            CHECK_NOTHROW(queue.wait(token));
            CHECK_EQ(token_sequence(queue.probe()), 2);
        }
    }
}

// The complete shared linear suite: the independent reference self-check, the
// declared request cases through the candidate's real queue, and the common
// admission, ownership, workspace, queue-order, and failure cases.
inline void run_linear_conformance(
        const ConformanceDevices& devices,
        const LinearDeclaration& declaration,
        ConformanceObserver* observer = nullptr,
        AcceleratorStorageOracle* oracle = nullptr) {
    // The declared target matrix is the driver's explicit input and drives the
    // shared fixtures; an empty matrix would silently test nothing.
    REQUIRE_FALSE(declaration.target_leaves.empty());
    REQUIRE(run_linear_oracle_self_check());
    run_linear_reference_conformance(devices, declaration, observer, oracle);
    run_linear_common_conformance(devices, declaration, observer);
}

}  // namespace iom_conformance