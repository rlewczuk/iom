#pragma once
// Standard 16x16 tiled linear-projection metadata, device codec, device
// kernel, and launch boundary shared by the CUDA and ROCm translation units.
// No vendor types: the backend supplies only the IOM_GPU_* macros before
// including this file, and includes the backend-specific expansion of
// standard_tiled_copy.inl first for the shared tile primitives (kTile,
// kTileSlots, plane_slot).
//
// The device semantics are the frozen `Linear projections` contract of
// docs/BACKEND_CONTRACT.md: `x` is `[..., T, I]`, rank-two `w[O, I]` is the
// Hugging Face orientation shared unchanged by every independent leading
// plane, and `out` is `[..., R, O]` (ordinary) or `[..., H, R, D]`
// (head-planar) with the checked relation `O = H*D`. Every logical output
// element is `sum_i x[b, s+r, i] * w[o, i]` over the selected row window with
// increasing `i`: the unsigned modulo-`2^N` dot for the twelve integer leaves,
// and the `+0`-started correctly rounded FP32 fused-multiply-add recurrence
// (FP64 for `F64`) followed by exactly one named-format encode for the eight
// floating leaves. Leading planes, leading transforms, plane offsets, and
// plane strides are derived per operand through each view's own metadata, so
// no broadcast, implicit transpose, or persistent host transpose exists.
//
// Packed sub-byte and wide stores own whole 32-bit words: a writer reads the
// word it owns, replaces only the fields of its own logical cells, and stores
// it once, so a written field that crosses a word boundary can never race
// another writer, and every padding bit, remainder bit, and 16x16 tile
// padding slot keeps the value it had. Only `x` and `w` are ever read.
//
// This file neither launches nor enables either backend by itself: each
// backend's `gpu_policy` supplies its own `launch_linear` wrapper and its own
// immutable `linear_supported` leaf predicate.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "iom/iom.hpp"
#include "iom/tensor.hpp"

#include "queue_resources.hpp"
#include "scalar_binary_codec.hpp"

#ifndef IOM_GPU_DEVICE
#error "IOM_GPU_DEVICE must be defined before including standard_tiled_linear.inl"
#endif
#ifndef IOM_GPU_GLOBAL
#error "IOM_GPU_GLOBAL must be defined before including standard_tiled_linear.inl"
#endif
#ifndef IOM_GPU_GLOBAL_INDEX
#error "IOM_GPU_GLOBAL_INDEX must be defined before including standard_tiled_linear.inl"
#endif
#ifndef IOM_LAUNCH_KERNEL
#error "IOM_LAUNCH_KERNEL must be defined before including standard_tiled_linear.inl"
#endif
#ifndef IOM_GPU_GLOBAL_STRIDE
#define IOM_GPU_GLOBAL_STRIDE \
    (static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
#endif

namespace iom::detail {

// Immutable device-visible linear descriptor. Every field is a value copy of
// admitted request metadata: the three native handles, the selected row
// window, the source row extent, the inner and outer matrix extents, the head
// count and the output columns per row, the checked logical leading-plane
// count, the three selected plane offsets, the leaf width and type, the
// leading rank, the output mode, and the device addresses of the leading
// extents and of the `x` and `out` plane strides. Nothing here is a borrowed
// `TensorView`, and nothing here is host arithmetic.
struct LinearMetadata {
    const unsigned char* x;
    const unsigned char* w;
    unsigned char* out;
    const std::uint64_t* dims;
    const std::uint64_t* x_strides;
    const std::uint64_t* out_strides;
    std::uint64_t start_row;
    std::uint64_t rows;
    std::uint64_t source_rows;
    std::uint64_t inner;
    std::uint64_t outer;
    std::uint64_t heads;
    std::uint64_t columns;
    std::uint64_t plane_count;
    std::uint64_t x_offset;
    std::uint64_t w_offset;
    std::uint64_t out_offset;
    std::uint32_t bits;
    std::uint32_t type;
    std::uint32_t leading_rank;
    std::uint32_t head_planar;
};
static_assert(std::is_trivially_copyable_v<LinearMetadata>);
// Fixed-slot layout contract: rank eight is the widest full shape, so the
// compiled descriptor is this header plus at most 3 * (8 - 2) + 1 words — the
// leading extents, the `x` plane strides, the shared `out` plane strides, and
// the head-axis `out` stride — and must fit one 512-byte metadata slot
// (alignment 32). A future overflow requires an intentional ABI/spec update,
// never automatic slot growth.
static_assert(sizeof(LinearMetadata) == 152);
static_assert(
        sizeof(LinearMetadata) + (3 * (8 - 2) + 1) * sizeof(std::uint64_t)
        <= kMetadataSlotBytes);
static_assert(
        alignof(LinearMetadata) <= 32
        && kMetadataSlotBytes % alignof(LinearMetadata) == 0);

namespace {

constexpr unsigned int kLinearThreads = 256;
constexpr std::size_t kLinearMaxBlocks = 65535;

[[nodiscard]] std::size_t linear_checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t linear_checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t linear_metadata_u64(
        std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::uint64_t>(value);
}

// Byte length of the fixed-slot descriptor of one admitted request rank.
[[nodiscard]] std::size_t linear_metadata_storage_bytes(std::size_t rank) {
    if (rank < 2 || rank > 8) {
        throw std::invalid_argument(
                "LINEAR metadata rank is out of range");
    }
    const std::size_t leading_rank = rank - 2;
    const std::size_t words = linear_checked_add(
            linear_checked_mul(
                    leading_rank, 3,
                    "LINEAR metadata rank storage overflows"),
            1, "LINEAR metadata rank storage overflows");
    const std::size_t arrays = linear_checked_mul(
            words, sizeof(std::uint64_t),
            "LINEAR metadata rank storage overflows");
    const std::size_t result = linear_checked_add(
            sizeof(LinearMetadata), arrays,
            "LINEAR metadata storage size overflows");
    if (result > kMetadataSlotBytes) {
        throw std::overflow_error("LINEAR metadata exceeds fixed slot");
    }
    return result;
}

// Bounded checked descriptor of one admitted request. Shape, device, dtype,
// quantization, alias, capability, row-window, layout, and workspace
// admission stay with the common hook; this only converts already-admitted
// metadata into the fixed device descriptor and rejects the arithmetic that
// could not be represented.
template <typename Request>
[[nodiscard]] LinearMetadata make_linear_metadata(const Request& request) {
    const std::span<const std::size_t> x_dimensions =
            request.x.spec.shape.dimensions();
    const std::span<const std::size_t> w_dimensions =
            request.w.spec.shape.dimensions();
    if (x_dimensions.size() < 2 || w_dimensions.size() != 2) {
        throw std::invalid_argument(
                "LINEAR metadata operand rank is out of range");
    }
    const std::size_t leading_rank = x_dimensions.size() - 2;
    const bool head_planar =
            request.layout == LinearOutputLayout::head_planar;
    if (request.x.plane_strides.size() != leading_rank
            || request.out.plane_strides.size()
                    != leading_rank + (head_planar ? 1 : 0)) {
        throw std::invalid_argument(
                "LINEAR metadata leading strides do not match the rank");
    }
    (void)linear_metadata_storage_bytes(x_dimensions.size());

    LinearMetadata metadata{};
    metadata.x = static_cast<const unsigned char*>(
            request.x.native_handle);
    metadata.w = static_cast<const unsigned char*>(
            request.w.native_handle);
    metadata.out = static_cast<unsigned char*>(request.out.native_handle);
    metadata.start_row = linear_metadata_u64(
            request.start_row, "LINEAR metadata row start overflows");
    metadata.rows = linear_metadata_u64(
            request.rows, "LINEAR metadata row count overflows");
    metadata.source_rows = linear_metadata_u64(
            x_dimensions[leading_rank],
            "LINEAR metadata source row extent overflows");
    metadata.inner = linear_metadata_u64(
            x_dimensions[leading_rank + 1],
            "LINEAR metadata inner extent overflows");
    metadata.outer = linear_metadata_u64(
            w_dimensions[0], "LINEAR metadata outer extent overflows");
    metadata.heads = linear_metadata_u64(
            head_planar ? request.heads : 1,
            "LINEAR metadata head count overflows");
    metadata.columns = linear_metadata_u64(
            head_planar ? request.head_dim : w_dimensions[0],
            "LINEAR metadata output column extent overflows");
    metadata.x_offset = linear_metadata_u64(
            request.x.plane_offset, "LINEAR metadata x offset overflows");
    metadata.w_offset = linear_metadata_u64(
            request.w.plane_offset, "LINEAR metadata w offset overflows");
    metadata.out_offset = linear_metadata_u64(
            request.out.plane_offset, "LINEAR metadata out offset overflows");
    metadata.bits = static_cast<std::uint32_t>(
            leaf_bits(request.x.spec.data_type));
    metadata.type = static_cast<std::uint32_t>(request.x.spec.data_type);
    metadata.leading_rank = static_cast<std::uint32_t>(leading_rank);
    metadata.head_planar = head_planar ? 1u : 0u;

    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = linear_checked_mul(
                plane_count,
                linear_metadata_u64(
                        x_dimensions[axis],
                        "LINEAR metadata leading extent overflows"),
                "LINEAR metadata plane count overflows");
    }
    metadata.plane_count = linear_metadata_u64(
            plane_count, "LINEAR metadata plane count overflows");
    // The launch enumerates one logical chunk per output tile row, so the
    // product must be representable before it is handed to the device.
    const std::size_t chunks_per_row = linear_checked_add(
            linear_metadata_u64(
                    metadata.columns,
                    "LINEAR metadata output column extent overflows"),
            TensorSpec::TILE - 1,
            "LINEAR metadata chunk count overflows") / TensorSpec::TILE;
    (void)linear_checked_mul(
            linear_checked_mul(
                    plane_count,
                    head_planar ? request.heads : std::size_t{1},
                    "LINEAR metadata chunk count overflows"),
            linear_checked_mul(
                    request.rows, chunks_per_row,
                    "LINEAR metadata chunk count overflows"),
            "LINEAR metadata chunk count overflows");
    for (const std::size_t stride : request.x.plane_strides) {
        (void)linear_metadata_u64(
                stride, "LINEAR metadata x stride overflows");
    }
    for (const std::size_t stride : request.out.plane_strides) {
        (void)linear_metadata_u64(
                stride, "LINEAR metadata out stride overflows");
    }
    return metadata;
}

// Writes the descriptor header and its leading-plane arrays into the fixed
// metadata slot's host mirror. The arrays are addressed by the uploaded
// device slot, so the device kernel reads leading extents and plane strides
// without any device-to-host round trip. The trailing `out` word carries the
// head-axis stride in head-planar mode and is zero in ordinary mode.
template <typename Request>
void write_linear_metadata(
        void* host_storage, const void* device_storage,
        const Request& request) {
    const std::span<const std::size_t> dimensions =
            request.x.spec.shape.dimensions();
    LinearMetadata metadata = make_linear_metadata(request);
    const std::size_t leading_rank = dimensions.size() - 2;
    auto* host_bytes = static_cast<std::byte*>(host_storage);
    const auto* device_bytes =
            static_cast<const std::byte*>(device_storage);
    const std::size_t arrays_offset = sizeof(LinearMetadata);
    auto* host_dims = reinterpret_cast<std::uint64_t*>(
            host_bytes + arrays_offset);
    auto* host_x_strides = host_dims + leading_rank;
    auto* host_out_strides = host_x_strides + leading_rank;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        host_dims[axis] = linear_metadata_u64(
                dimensions[axis],
                "LINEAR metadata leading extent overflows");
        host_x_strides[axis] = linear_metadata_u64(
                request.x.plane_strides[axis],
                "LINEAR metadata x stride overflows");
        host_out_strides[axis] = linear_metadata_u64(
                request.out.plane_strides[axis],
                "LINEAR metadata out stride overflows");
    }
    host_out_strides[leading_rank] = metadata.head_planar != 0
            ? linear_metadata_u64(
                      request.out.plane_strides[leading_rank],
                      "LINEAR metadata out head stride overflows")
            : 0;
    const auto* device_dims = reinterpret_cast<const std::uint64_t*>(
            device_bytes + arrays_offset);
    metadata.dims = device_dims;
    metadata.x_strides = device_dims + leading_rank;
    metadata.out_strides = device_dims + leading_rank * 2;
    *reinterpret_cast<LinearMetadata*>(host_storage) = metadata;
}

// ---------------------------------------------------------------------------
// Accumulator traits. Both floating carriers drive the one shared named-format
// codec (src/shared/scalar_binary_codec.hpp) in their own IEEE domain: the
// seven non-F64 floating leaves accumulate in FP32 and `F64` stays FP64. The
// integer leaves use unsigned modulo-2^N arithmetic and never touch a floating
// carrier.
// ---------------------------------------------------------------------------

struct LinearFp32Traits {
    using carrier_type = float;

    IOM_GPU_DEVICE static carrier_type positive_infinity() noexcept {
        return static_cast<carrier_type>(__longlong_as_double(
                static_cast<long long>(0x7ff0000000000000ull)));
    }
    IOM_GPU_DEVICE static carrier_type quiet_nan() noexcept {
        return static_cast<carrier_type>(__longlong_as_double(
                static_cast<long long>(0x7ff8000000000000ull)));
    }
    IOM_GPU_DEVICE static carrier_type max_finite() noexcept {
        return 3.4028234663852886e+38F;
    }
    IOM_GPU_DEVICE static carrier_type fused_multiply_add(
            carrier_type multiplicand, carrier_type multiplier,
            carrier_type addend) noexcept {
        return ::fmaf(multiplicand, multiplier, addend);
    }
    IOM_GPU_DEVICE static bool isnan(carrier_type value) noexcept {
        return ::isnan(value);
    }
    IOM_GPU_DEVICE static bool isinf(carrier_type value) noexcept {
        return ::isinf(value);
    }
    IOM_GPU_DEVICE static bool signbit(carrier_type value) noexcept {
        return ::signbit(value);
    }
    IOM_GPU_DEVICE static carrier_type fabs(carrier_type value) noexcept {
        return ::fabs(value);
    }
    IOM_GPU_DEVICE static carrier_type floor(carrier_type value) noexcept {
        return ::floor(value);
    }
    IOM_GPU_DEVICE static carrier_type ldexp(
            carrier_type value, int exponent) noexcept {
        return ::ldexp(value, exponent);
    }
    IOM_GPU_DEVICE static carrier_type frexp(
            carrier_type value, int* exponent) noexcept {
        return ::frexp(value, exponent);
    }
};

struct LinearFp64Traits {
    using carrier_type = double;

    IOM_GPU_DEVICE static carrier_type positive_infinity() noexcept {
        return __longlong_as_double(
                static_cast<long long>(0x7ff0000000000000ull));
    }
    IOM_GPU_DEVICE static carrier_type quiet_nan() noexcept {
        return __longlong_as_double(
                static_cast<long long>(0x7ff8000000000000ull));
    }
    IOM_GPU_DEVICE static carrier_type max_finite() noexcept {
        return __longlong_as_double(
                static_cast<long long>(0x7fefffffffffffffull));
    }
    IOM_GPU_DEVICE static carrier_type fused_multiply_add(
            carrier_type multiplicand, carrier_type multiplier,
            carrier_type addend) noexcept {
        return ::fma(multiplicand, multiplier, addend);
    }
    IOM_GPU_DEVICE static bool isnan(carrier_type value) noexcept {
        return ::isnan(value);
    }
    IOM_GPU_DEVICE static bool isinf(carrier_type value) noexcept {
        return ::isinf(value);
    }
    IOM_GPU_DEVICE static bool signbit(carrier_type value) noexcept {
        return ::signbit(value);
    }
    IOM_GPU_DEVICE static carrier_type fabs(carrier_type value) noexcept {
        return ::fabs(value);
    }
    IOM_GPU_DEVICE static carrier_type floor(carrier_type value) noexcept {
        return ::floor(value);
    }
    IOM_GPU_DEVICE static carrier_type ldexp(
            carrier_type value, int exponent) noexcept {
        return ::ldexp(value, exponent);
    }
    IOM_GPU_DEVICE static carrier_type frexp(
            carrier_type value, int* exponent) noexcept {
        return ::frexp(value, exponent);
    }
};

template <typename Carrier>
struct LinearCodecOf;

template <>
struct LinearCodecOf<float> {
    using traits = LinearFp32Traits;
    using type = scalar_binary_codec_detail::Codec<traits>;
};

template <>
struct LinearCodecOf<double> {
    using traits = LinearFp64Traits;
    using type = scalar_binary_codec_detail::Codec<traits>;
};

// ---------------------------------------------------------------------------
// Packed-field primitives. Element slot `s` of a plane occupies bits
// `[s * bits, (s + 1) * bits)` of the plane's packed stream; 32-bit words are
// the unit of exclusive ownership, so a writer reads the owned word, replaces
// only the fields it owns, and stores once. A 6-bit field may cross a word
// boundary and a 64-bit value uses paired words without ever shifting by 64.
// ---------------------------------------------------------------------------

IOM_GPU_DEVICE std::uint32_t linear_field_mask(
        unsigned bits) noexcept {
    return bits == 32 ? 0xffffffffu
                      : ((std::uint32_t{1} << bits) - 1);
}

IOM_GPU_DEVICE std::uint32_t linear_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value;
    for (unsigned i = 0; i < 4; ++i) {
        reinterpret_cast<unsigned char*>(&value)[i] = base[word * 4 + i];
    }
    return value;
}

IOM_GPU_DEVICE void linear_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned i = 0; i < 4; ++i) {
        base[word * 4 + i] =
                reinterpret_cast<const unsigned char*>(&value)[i];
    }
}

IOM_GPU_DEVICE std::uint64_t linear_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned bits) noexcept {
    const std::uint64_t word = bit / 32;
    std::uint64_t joined = linear_load_word(base, word);
    if (bit % 32 + bits > 32) {
        joined |= static_cast<std::uint64_t>(
                          linear_load_word(base, word + 1))
                << 32;
    }
    return (joined >> (bit % 32))
            & (bits == 64
               ? ~std::uint64_t{}
               : ((std::uint64_t{1} << bits) - 1));
}

IOM_GPU_DEVICE std::uint64_t linear_integer_mask(
        unsigned bits) noexcept {
    return bits == 64 ? ~std::uint64_t{0}
                      : ((std::uint64_t{1} << bits) - 1);
}

// One logical output element: the frozen scalar recurrence over the whole
// logical inner extent of the selected row and Hugging Face weight row. The
// weight row is the output coordinate `o` — `h*D + d` in head-planar mode and
// the plain output column in ordinary mode — so the caller resolves the head
// mapping and this function never guesses it. The integer form reduces modulo
// 2^N in unsigned arithmetic after every multiply and every add, and the
// floating form starts at `+0`, walks increasing `i` with one correctly
// rounded fused multiply-add per step, and encodes the accumulated sum
// exactly once.
template <typename Carrier>
IOM_GPU_DEVICE std::uint64_t linear_projected_code(
        const LinearMetadata& m, std::uint64_t x_plane,
        std::uint64_t source_row, std::uint64_t weight_row,
        scalar_binary_codec_detail::Format format) noexcept {
    const unsigned bits = m.bits;
    if constexpr (std::is_integral_v<Carrier>) {
        (void)format;
        const std::uint64_t mask = linear_integer_mask(bits);
        std::uint64_t accumulator = 0;
        for (std::uint64_t index = 0; index < m.inner; ++index) {
            const std::uint64_t element = linear_load_bits(
                    m.x,
                    plane_slot(
                            x_plane, source_row, index, m.source_rows,
                            m.inner)
                            * bits,
                    bits);
            const std::uint64_t weight = linear_load_bits(
                    m.w,
                    plane_slot(m.w_offset, weight_row, index, m.outer, m.inner)
                            * bits,
                    bits);
            accumulator =
                    (accumulator + (element & mask) * (weight & mask)) & mask;
        }
        return accumulator;
    } else {
        using Traits = typename LinearCodecOf<Carrier>::traits;
        using Codec = typename LinearCodecOf<Carrier>::type;
        Carrier accumulator = static_cast<Carrier>(0);
        for (std::uint64_t index = 0; index < m.inner; ++index) {
            const Carrier element = Codec::decode(
                    linear_load_bits(
                            m.x,
                            plane_slot(
                                    x_plane, source_row, index,
                                    m.source_rows, m.inner)
                                    * bits,
                            bits),
                    format);
            const Carrier weight = Codec::decode(
                    linear_load_bits(
                            m.w,
                            plane_slot(
                                    m.w_offset, weight_row, index, m.outer,
                                    m.inner)
                                    * bits,
                            bits),
                    format);
            accumulator = Traits::fused_multiply_add(
                    element, weight, accumulator);
        }
        return Codec::encode(accumulator, format);
    }
}

// One thread owns one 16-column output chunk of one selected row of one
// output plane. The 16 slots of a tile row are exactly `bits / 2` whole
// 32-bit words for every applicable leaf width, so a chunk's word range is
// never shared with another row, another plane, or another chunk: ownership is
// exclusive and no observer sees a partially written word. Only logical cells
// are read and only logical cells are written; padding slots, remainder bits,
// and tile padding keep their value.
template <typename Carrier>
IOM_GPU_DEVICE void linear_tiled_body(const LinearMetadata& m) noexcept {
    scalar_binary_codec_detail::Format format{};
    if constexpr (!std::is_integral_v<Carrier>) {
        using Codec = typename LinearCodecOf<Carrier>::type;
        format = Codec::format(static_cast<DataType>(m.type));
    }
    const std::uint64_t chunks_per_row =
            (m.columns + kTile - 1) / kTile;
    const std::uint64_t rows_per_plane =
            m.heads * m.rows * chunks_per_row;
    const std::uint64_t total = m.plane_count * rows_per_plane;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    for (std::uint64_t unit = IOM_GPU_GLOBAL_INDEX; unit < total;
         unit += stride) {
        const std::uint64_t plane = unit / rows_per_plane;
        const std::uint64_t in_plane = unit % rows_per_plane;
        const std::uint64_t head =
                in_plane / (m.rows * chunks_per_row);
        const std::uint64_t in_head =
                in_plane % (m.rows * chunks_per_row);
        const std::uint64_t row = in_head / chunks_per_row;
        const std::uint64_t chunk = in_head % chunks_per_row;

        std::uint64_t x_plane = m.x_offset;
        std::uint64_t out_plane = m.out_offset;
        std::uint64_t rest = plane;
        for (std::uint32_t axis = m.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate = rest % m.dims[axis];
            rest /= m.dims[axis];
            x_plane += coordinate * m.x_strides[axis];
            out_plane += coordinate * m.out_strides[axis];
        }
        if (m.head_planar != 0) {
            out_plane += head * m.out_strides[m.leading_rank];
        }

        const std::uint64_t first_column = chunk * kTile;
        const std::uint64_t length =
                m.columns - first_column < kTile
                ? m.columns - first_column
                : kTile;
        const std::uint64_t out_bit = plane_slot(
                out_plane, row, first_column, m.rows, m.columns) * m.bits;
        const std::uint64_t first_word = out_bit / 32;
        const std::uint64_t last_word =
                (out_bit + length * m.bits - 1) / 32;
        const std::uint64_t source_row = m.start_row + row;
        for (std::uint64_t word = first_word; word <= last_word; ++word) {
            const std::uint64_t word_first_bit = word * 32;
            const std::uint64_t word_end_bit = word_first_bit + 32;
            const std::uint64_t first_index =
                    (word_first_bit > out_bit ? word_first_bit - out_bit : 0)
                    / m.bits;
            std::uint64_t last_index =
                    (word_end_bit - out_bit + m.bits - 1) / m.bits;
            if (last_index > length) {
                last_index = length;
            }
            std::uint32_t merged = linear_load_word(m.out, word);
            for (std::uint64_t index = first_index; index < last_index;
                 ++index) {
                const std::uint64_t field_bit = out_bit + index * m.bits;
                // The Hugging Face weight row is the output coordinate:
                // `h*D + d` in head-planar mode and the output column
                // `o` in ordinary mode, where `heads` is one and `head`
                // is always zero.
                const std::uint64_t weight_row =
                        head * m.columns + first_column + index;
                const std::uint64_t encoded =
                        linear_projected_code<Carrier>(
                                m, x_plane, source_row, weight_row, format);
                const std::uint64_t overlap_first =
                        field_bit > word_first_bit
                        ? field_bit
                        : word_first_bit;
                const std::uint64_t overlap_end =
                        field_bit + m.bits < word_end_bit
                        ? field_bit + m.bits
                        : word_end_bit;
                const unsigned count = static_cast<unsigned>(
                        overlap_end - overlap_first);
                const unsigned position = static_cast<unsigned>(
                        overlap_first - word_first_bit);
                const unsigned shift = static_cast<unsigned>(
                        overlap_first - field_bit);
                const std::uint32_t segment = static_cast<std::uint32_t>(
                        encoded >> shift) & linear_field_mask(count);
                merged = (merged
                          & ~(linear_field_mask(count) << position))
                        | (segment << position);
            }
            linear_store_word(m.out, word, merged);
        }
    }
}

// The shared kernel is a plain function so every including translation unit
// compiles the complete device operation, including the integer, FP32, and
// FP64 instantiations, before either backend wrapper lands. The shared codec
// reports no format exactly for the twelve integer leaves, so the leaf class
// is decided by the leaf encoding itself and never by a width heuristic.
IOM_GPU_GLOBAL void standard_tiled_linear_kernel(LinearMetadata m) {
    const scalar_binary_codec_detail::Format format =
            scalar_binary_codec_detail::Codec<LinearFp32Traits>::format(
                    static_cast<DataType>(m.type));
    if (format.bits == 0) {
        linear_tiled_body<std::uint64_t>(m);
    } else if (m.type == static_cast<std::uint32_t>(DataType::F64)) {
        linear_tiled_body<double>(m);
    } else {
        linear_tiled_body<float>(m);
    }
}

}  // namespace

// Policy-parameterized launch of the shared tiled device operation. The
// backend wrapper supplies the already-created nonblocking stream; no second
// stream, allocation, staging, or synchronization is introduced here.
template <typename Policy>
void launch_standard_tiled_linear(
        typename Policy::stream_type stream,
        const LinearMetadata& metadata) {
    const std::size_t chunks_per_row = linear_checked_add(
            static_cast<std::size_t>(metadata.columns),
            TensorSpec::TILE - 1,
            "LINEAR launch column count overflows") / TensorSpec::TILE;
    const std::size_t rows_per_plane = linear_checked_mul(
            linear_checked_mul(
                    static_cast<std::size_t>(metadata.heads),
                    static_cast<std::size_t>(metadata.rows),
                    "LINEAR launch row count overflows"),
            chunks_per_row, "LINEAR launch row count overflows");
    const std::size_t total = linear_checked_mul(
            static_cast<std::size_t>(metadata.plane_count),
            rows_per_plane, "LINEAR launch chunk count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            total < kLinearMaxBlocks ? total : kLinearMaxBlocks);
    IOM_LAUNCH_KERNEL(
            standard_tiled_linear_kernel, blocks, kLinearThreads, stream,
            metadata);
}

}  // namespace iom::detail
