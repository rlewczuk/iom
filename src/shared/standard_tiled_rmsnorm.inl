#pragma once
// Standard 16x16 tiled RMSNorm metadata, device codec, device kernel, and
// launch boundary shared by the CUDA and ROCm translation units. No vendor
// types: the backend supplies only the IOM_GPU_* macros before including this
// file, and includes the backend-specific expansion of
// standard_tiled_copy.inl first for the shared tile primitives (kTile,
// kTileSlots, plane_slot).
//
// The device semantics are the frozen `RMS normalization` contract of
// docs/BACKEND_CONTRACT.md: `x` and `out` are identical `[..., R, F]`, `scale`
// is exactly rank-two `[1, F]`, every leading plane and row is independent,
// only the logical `F` features of a row participate, and neither tile padding
// nor any other row or plane is read or written. `F4_E2M1`, `F6_E2M3`,
// `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, and `F32` decode to FP32
// and keep every square, reduction step, mean, epsilon addition, square root,
// reciprocal, normalization product, and scale product in FP32; `F64` keeps
// all of them in FP64. Each output element is encoded exactly once with the
// shared round-to-nearest-even named-format codec. The kernel performs no
// device-to-host round trip, no allocation, no host-side decoding, and no
// generic reduction framework; the only host work is the checked descriptor
// build and the launch geometry.
//
// This file neither launches nor enables either backend by itself: each
// backend's `gpu_policy` supplies its own `launch_rmsnorm` wrapper, and until
// that wrapper lands the policy reports `Unsupported`.

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
#error "IOM_GPU_DEVICE must be defined before including standard_tiled_rmsnorm.inl"
#endif
#ifndef IOM_GPU_GLOBAL
#error "IOM_GPU_GLOBAL must be defined before including standard_tiled_rmsnorm.inl"
#endif
#ifndef IOM_GPU_GLOBAL_INDEX
#error "IOM_GPU_GLOBAL_INDEX must be defined before including standard_tiled_rmsnorm.inl"
#endif
#ifndef IOM_LAUNCH_KERNEL
#error "IOM_LAUNCH_KERNEL must be defined before including standard_tiled_rmsnorm.inl"
#endif
#ifndef IOM_GPU_GLOBAL_STRIDE
#define IOM_GPU_GLOBAL_STRIDE \
    (static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
#endif
#ifndef IOM_GPU_SHARED
#error "IOM_GPU_SHARED must be defined before including standard_tiled_rmsnorm.inl"
#endif
#ifndef IOM_GPU_BARRIER
#error "IOM_GPU_BARRIER must be defined before including standard_tiled_rmsnorm.inl"
#endif

namespace iom::detail {

// Immutable device-visible RMSNorm descriptor. Every field is a value copy of
// admitted request metadata: the three native handles, the logical row and
// feature extents, the leading-plane count, the three selected plane offsets,
// the validated epsilon, the leaf width and type, the leading rank, and the
// device addresses of the leading extents and the `x`/`out` plane strides.
// Nothing here is a borrowed `TensorView`, and nothing here is host
// arithmetic.
struct RmsnormMetadata {
    const unsigned char* x;
    const unsigned char* scale;
    unsigned char* out;
    std::uint64_t rows;
    std::uint64_t columns;
    std::uint64_t plane_count;
    std::uint64_t x_offset;
    std::uint64_t scale_offset;
    std::uint64_t out_offset;
    float epsilon;
    std::uint32_t bits;
    std::uint32_t type;
    std::uint32_t leading_rank;
    const std::uint64_t* dims;
    const std::uint64_t* x_strides;
    const std::uint64_t* out_strides;
};
static_assert(std::is_trivially_copyable_v<RmsnormMetadata>);
// Fixed-slot layout contract: rank eight is the widest full shape, so the
// compiled descriptor is this header plus at most 3 * (8 - 2) leading-rank
// words and must fit one 512-byte metadata slot (alignment 32). A future
// overflow requires an intentional ABI/spec update, never automatic slot
// growth.
static_assert(sizeof(RmsnormMetadata) == 112);
static_assert(
        sizeof(RmsnormMetadata) + 3 * (8 - 2) * sizeof(std::uint64_t)
        <= kMetadataSlotBytes);
static_assert(
        alignof(RmsnormMetadata) <= 32
        && kMetadataSlotBytes % alignof(RmsnormMetadata) == 0);

namespace {

constexpr std::uint64_t kRmsnormTile = TensorSpec::TILE;
constexpr unsigned int kRmsnormThreads = 256;
constexpr unsigned int kRmsnormMaxBlocks = 65535;

[[nodiscard]] std::size_t rmsnorm_checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t rmsnorm_checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t rmsnorm_metadata_u64(
        std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::uint64_t>(value);
}

// Byte length of the fixed-slot descriptor of one admitted request rank.
[[nodiscard]] std::size_t rmsnorm_metadata_storage_bytes(std::size_t rank) {
    if (rank < 2) {
        throw std::invalid_argument(
                "RMSNORM rank below tiled matrix rank");
    }
    const std::size_t arrays = rmsnorm_checked_mul(
            rmsnorm_checked_mul(
                    rank - 2, sizeof(std::uint64_t),
                    "RMSNORM metadata rank storage overflows"),
            3, "RMSNORM metadata rank storage overflows");
    return rmsnorm_checked_add(
            sizeof(RmsnormMetadata), arrays,
            "RMSNORM metadata storage size overflows");
}

// Bounded checked descriptor of one admitted request. Shape, device, dtype,
// quantization, alias, workspace, and epsilon admission stay with the common
// hook; this only converts already-admitted metadata into the fixed device
// descriptor and rejects the arithmetic that could not be represented.
template <typename Request>
[[nodiscard]] RmsnormMetadata make_rmsnorm_metadata(const Request& request) {
    const std::span<const std::size_t> dimensions =
            request.x.spec.shape.dimensions();
    if (dimensions.size() < 2) {
        throw std::invalid_argument(
                "RMSNORM rank below tiled matrix rank");
    }
    const std::size_t leading_rank = dimensions.size() - 2;
    if (leading_rank
            > static_cast<std::size_t>(
                      std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
                "RMSNORM metadata leading rank representation overflows");
    }
    RmsnormMetadata metadata{};
    metadata.x = static_cast<const unsigned char*>(
            request.x.native_handle);
    metadata.scale = static_cast<const unsigned char*>(
            request.scale.native_handle);
    metadata.out = static_cast<unsigned char*>(request.out.native_handle);
    metadata.rows = rmsnorm_metadata_u64(
            dimensions[leading_rank],
            "RMSNORM metadata row extent overflows");
    metadata.columns = rmsnorm_metadata_u64(
            dimensions[leading_rank + 1],
            "RMSNORM metadata feature extent overflows");
    metadata.x_offset = rmsnorm_metadata_u64(
            request.x.plane_offset, "RMSNORM metadata x offset overflows");
    metadata.scale_offset = rmsnorm_metadata_u64(
            request.scale.plane_offset,
            "RMSNORM metadata scale offset overflows");
    metadata.out_offset = rmsnorm_metadata_u64(
            request.out.plane_offset,
            "RMSNORM metadata out offset overflows");
    metadata.epsilon = request.epsilon;
    metadata.bits = static_cast<std::uint32_t>(
            leaf_bits(request.x.spec.data_type));
    metadata.type = static_cast<std::uint32_t>(request.x.spec.data_type);
    metadata.leading_rank = static_cast<std::uint32_t>(leading_rank);
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = rmsnorm_checked_mul(
                plane_count,
                rmsnorm_metadata_u64(
                        dimensions[axis],
                        "RMSNORM metadata leading extent overflows"),
                "RMSNORM metadata plane count overflows");
    }
    metadata.plane_count = rmsnorm_metadata_u64(
            plane_count, "RMSNORM metadata plane count overflows");
    // The launch enumerates one block per logical row, so the product must be
    // representable before it is handed to the device.
    (void)rmsnorm_checked_mul(
            plane_count, metadata.rows,
            "RMSNORM metadata row count overflows");
    for (const std::size_t stride : request.x.plane_strides) {
        (void)rmsnorm_metadata_u64(
                stride, "RMSNORM metadata x stride overflows");
    }
    for (const std::size_t stride : request.out.plane_strides) {
        (void)rmsnorm_metadata_u64(
                stride, "RMSNORM metadata out stride overflows");
    }
    return metadata;
}

// Writes the descriptor header and its leading-plane arrays into the fixed
// metadata slot's host mirror. The arrays are addressed by the uploaded
// device slot, so the device kernel reads leading extents and plane strides
// without any device-to-host round trip.
template <typename Request>
void write_rmsnorm_metadata(
        void* host_storage, const void* device_storage,
        const Request& request) {
    const std::span<const std::size_t> dimensions =
            request.x.spec.shape.dimensions();
    RmsnormMetadata metadata = make_rmsnorm_metadata(request);
    const std::size_t leading_rank = dimensions.size() - 2;
    auto* host_bytes = static_cast<std::byte*>(host_storage);
    const auto* device_bytes =
            static_cast<const std::byte*>(device_storage);
    const std::size_t arrays_offset = sizeof(RmsnormMetadata);
    auto* host_dims = reinterpret_cast<std::uint64_t*>(
            host_bytes + arrays_offset);
    auto* host_x_strides = host_dims + leading_rank;
    auto* host_out_strides = host_x_strides + leading_rank;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        host_dims[axis] = rmsnorm_metadata_u64(
                dimensions[axis],
                "RMSNORM metadata leading extent overflows");
        host_x_strides[axis] = rmsnorm_metadata_u64(
                request.x.plane_strides[axis],
                "RMSNORM metadata x stride overflows");
        host_out_strides[axis] = rmsnorm_metadata_u64(
                request.out.plane_strides[axis],
                "RMSNORM metadata out stride overflows");
    }
    const auto* device_dims = reinterpret_cast<const std::uint64_t*>(
            device_bytes + arrays_offset);
    metadata.dims = device_dims;
    metadata.x_strides = device_dims + leading_rank;
    metadata.out_strides = device_dims + leading_rank * 2;
    *reinterpret_cast<RmsnormMetadata*>(host_storage) = metadata;
}

// ---------------------------------------------------------------------------
// Accumulator traits. Both carriers drive the one shared named-format codec
// (src/shared/scalar_binary_codec.hpp) in their own IEEE domain: the eight
// non-F64 leaves accumulate in FP32, `F64` stays FP64.
// ---------------------------------------------------------------------------

struct RmsnormFp32Traits {
    using carrier_type = float;

    // The bit-pattern constructors stay in the double domain the established
    // shared device codec already uses; narrowing preserves infinity and
    // NaN-ness, and the finite maximum is the exact FP32 maximum.
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
    IOM_GPU_DEVICE static carrier_type square_root(
            carrier_type value) noexcept {
        return ::sqrtf(value);
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

struct RmsnormFp64Traits {
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
    IOM_GPU_DEVICE static carrier_type square_root(
            carrier_type value) noexcept {
        return ::sqrt(value);
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
struct RmsnormCodecOf;

template <>
struct RmsnormCodecOf<float> {
    using traits = RmsnormFp32Traits;
    using type = scalar_binary_codec_detail::Codec<traits>;
};

template <>
struct RmsnormCodecOf<double> {
    using traits = RmsnormFp64Traits;
    using type = scalar_binary_codec_detail::Codec<traits>;
};

// ---------------------------------------------------------------------------
// Packed-field primitives. Element slot `s` of a plane occupies bits
// `[s * bits, (s + 1) * bits)` of the plane's packed stream; 32-bit words are
// the unit of exclusive ownership, so a writer reads the owned word, replaces
// only the fields it owns, and stores once. Padding bits and padding slots are
// never modified.
// ---------------------------------------------------------------------------

IOM_GPU_DEVICE std::uint32_t rmsnorm_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value;
    for (unsigned i = 0; i < 4; ++i) {
        reinterpret_cast<unsigned char*>(&value)[i] =
                base[word * 4 + i];
    }
    return value;
}

IOM_GPU_DEVICE void rmsnorm_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned i = 0; i < 4; ++i) {
        base[word * 4 + i] =
                reinterpret_cast<const unsigned char*>(&value)[i];
    }
}

IOM_GPU_DEVICE std::uint32_t rmsnorm_field_mask(
        unsigned bits) noexcept {
    return bits == 32 ? 0xffffffffu
                      : ((std::uint32_t{1} << bits) - 1);
}

IOM_GPU_DEVICE std::uint64_t rmsnorm_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned bits) noexcept {
    const std::uint64_t word = bit / 32;
    std::uint64_t joined = rmsnorm_load_word(base, word);
    if (bit % 32 + bits > 32) {
        joined |= static_cast<std::uint64_t>(
                          rmsnorm_load_word(base, word + 1))
                << 32;
    }
    return (joined >> (bit % 32))
            & (bits == 64
               ? ~std::uint64_t{}
               : ((std::uint64_t{1} << bits) - 1));
}

// ---------------------------------------------------------------------------
// Device kernel body. One block owns one logical row; every thread owns whole
// 16-feature chunks of that row. The 16 slots of one tile row are exactly
// `bits / 2` whole 32-bit words for every applicable leaf width, so a chunk's
// word range is never shared with another row, another tile, or another block:
// ownership is exclusive and no observer sees a partially written word.
//
// The reduction is two-phase: each thread sums the squares of its own chunks
// in FP32 (FP64 for `F64`), a fixed tree reduction combines the per-thread
// partials, thread 0 evaluates `1 / sqrt(sum / F + eps)` in the same domain,
// and the second pass normalizes, scales, and encodes each owned feature once.
// Reassociation across the tree stays inside the documented comparison policy;
// the special-value and signed-zero results do not depend on the order.
// ---------------------------------------------------------------------------

template <typename Carrier>
IOM_GPU_DEVICE void rmsnorm_tiled_row(
        const RmsnormMetadata& m, Carrier* partial) noexcept {
    using Codec = typename RmsnormCodecOf<Carrier>::type;
    const scalar_binary_codec_detail::Format format =
            Codec::format(static_cast<DataType>(m.type));
    const std::uint64_t global = IOM_GPU_GLOBAL_INDEX;
    const std::uint64_t row_stride =
            IOM_GPU_GLOBAL_STRIDE / kRmsnormThreads;
    const unsigned int thread =
            static_cast<unsigned int>(global % kRmsnormThreads);
    const std::uint64_t chunks =
            (m.columns + kRmsnormTile - 1) / kRmsnormTile;
    const unsigned bits = m.bits;
    const auto* x_bytes = m.x;
    const auto* scale_bytes = m.scale;
    auto* out_bytes = m.out;

    for (std::uint64_t row_index = global / kRmsnormThreads;
         row_index < m.plane_count * m.rows; row_index += row_stride) {
        const std::uint64_t row = row_index % m.rows;
        std::uint64_t x_plane = m.x_offset;
        std::uint64_t out_plane = m.out_offset;
        std::uint64_t rest = row_index / m.rows;
        for (std::uint32_t axis = m.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate = rest % m.dims[axis];
            rest /= m.dims[axis];
            x_plane += coordinate * m.x_strides[axis];
            out_plane += coordinate * m.out_strides[axis];
        }

        Carrier sum = static_cast<Carrier>(0);
        for (std::uint64_t chunk = thread; chunk < chunks;
             chunk += kRmsnormThreads) {
            const std::uint64_t first_feature = chunk * kRmsnormTile;
            const std::uint64_t length =
                    m.columns - first_feature < kRmsnormTile
                    ? m.columns - first_feature
                    : kRmsnormTile;
            const std::uint64_t bit = plane_slot(
                    x_plane, row, first_feature, m.rows, m.columns)
                    * bits;
            for (std::uint64_t index = 0; index < length; ++index) {
                const Carrier feature = Codec::decode(
                        rmsnorm_load_bits(
                                x_bytes, bit + index * bits, bits),
                        format);
                const Carrier square = feature * feature;
                sum = sum + square;
            }
        }
        partial[thread] = sum;
        IOM_GPU_BARRIER;
        for (unsigned int step = kRmsnormThreads / 2; step > 0;
             step >>= 1) {
            if (thread < step) {
                partial[thread] =
                        partial[thread] + partial[thread + step];
            }
            IOM_GPU_BARRIER;
        }
        if (thread == 0) {
            const Carrier mean =
                    partial[0] / static_cast<Carrier>(m.columns);
            const Carrier shifted =
                    mean + static_cast<Carrier>(m.epsilon);
            using Traits = typename RmsnormCodecOf<Carrier>::traits;
            partial[0] = static_cast<Carrier>(1)
                    / Traits::square_root(shifted);
        }
        IOM_GPU_BARRIER;
        const Carrier inverse = partial[0];

        for (std::uint64_t chunk = thread; chunk < chunks;
             chunk += kRmsnormThreads) {
            const std::uint64_t first_feature = chunk * kRmsnormTile;
            const std::uint64_t length =
                    m.columns - first_feature < kRmsnormTile
                    ? m.columns - first_feature
                    : kRmsnormTile;
            const std::uint64_t x_bit = plane_slot(
                    x_plane, row, first_feature, m.rows, m.columns)
                    * bits;
            const std::uint64_t out_bit = plane_slot(
                    out_plane, row, first_feature, m.rows, m.columns)
                    * bits;
            const std::uint64_t scale_bit = plane_slot(
                    m.scale_offset, 0, first_feature, 1, m.columns)
                    * bits;
            const std::uint64_t first_word = out_bit / 32;
            const std::uint64_t last_word =
                    (out_bit + length * bits - 1) / 32;
            for (std::uint64_t word = first_word; word <= last_word;
                 ++word) {
                const std::uint64_t word_first_bit = word * 32;
                const std::uint64_t word_end_bit = word_first_bit + 32;
                const std::uint64_t first_index =
                        (word_first_bit > out_bit
                         ? word_first_bit - out_bit
                         : 0)
                        / bits;
                std::uint64_t last_index =
                        (word_end_bit - out_bit + bits - 1) / bits;
                if (last_index > length) {
                    last_index = length;
                }
                std::uint32_t merged =
                        rmsnorm_load_word(out_bytes, word);
                for (std::uint64_t index = first_index;
                     index < last_index; ++index) {
                    const std::uint64_t field_bit =
                            out_bit + index * bits;
                    const Carrier feature = Codec::decode(
                            rmsnorm_load_bits(
                                    x_bytes, x_bit + index * bits, bits),
                            format);
                    const Carrier scale = Codec::decode(
                            rmsnorm_load_bits(
                                    scale_bytes,
                                    scale_bit + index * bits, bits),
                            format);
                    const Carrier normalized = feature * inverse;
                    Carrier scaled = normalized * scale;
                    // Contract clause 3 and 9: a NaN result is stored in its
                    // canonical positive form by clearing its sign before the
                    // single destination encode. NaN payloads and NaN signs
                    // are outside the contract, and only the positive
                    // canonical form also satisfies the published comparison
                    // for the narrow finite-only leaves, where a NaN
                    // saturates to a finite encoding instead of a NaN one.
                    // The CPU port applies the identical rule, so every
                    // backend encodes the same result independently of the
                    // device's invalid-operation NaN sign.
                    if (RmsnormCodecOf<Carrier>::traits::isnan(scaled)) {
                        scaled = RmsnormCodecOf<Carrier>::traits::fabs(scaled);
                    }
                    const std::uint64_t encoded =
                            Codec::encode(scaled, format);
                    const std::uint64_t overlap_first =
                            field_bit > word_first_bit
                            ? field_bit
                            : word_first_bit;
                    const std::uint64_t overlap_end =
                            field_bit + bits < word_end_bit
                            ? field_bit + bits
                            : word_end_bit;
                    const unsigned count = static_cast<unsigned>(
                            overlap_end - overlap_first);
                    const unsigned position = static_cast<unsigned>(
                            overlap_first - word_first_bit);
                    const unsigned shift = static_cast<unsigned>(
                            overlap_first - field_bit);
                    const std::uint32_t segment = static_cast<std::uint32_t>(
                            encoded >> shift)
                            & rmsnorm_field_mask(count);
                    merged = (merged
                              & ~(rmsnorm_field_mask(count) << position))
                            | (segment << position);
                }
                rmsnorm_store_word(out_bytes, word, merged);
            }
        }
        // A thread must not start the next row's partial before every thread
        // has read the broadcast reciprocal.
        IOM_GPU_BARRIER;
    }
}

// The shared kernel is a plain function so every including translation unit
// compiles the complete device operation, including both accumulator
// instantiations, before either backend wrapper lands.
IOM_GPU_GLOBAL void standard_tiled_rmsnorm_kernel(RmsnormMetadata m) {
    IOM_GPU_SHARED float partial_f32[kRmsnormThreads];
    IOM_GPU_SHARED double partial_f64[kRmsnormThreads];
    if (m.type == static_cast<std::uint32_t>(DataType::F64)) {
        rmsnorm_tiled_row<double>(m, partial_f64);
    } else {
        rmsnorm_tiled_row<float>(m, partial_f32);
    }
}

}  // namespace

// Policy-parameterized launch of the shared tiled device operation. The
// backend wrapper supplies the already-created nonblocking stream; no second
// stream, allocation, staging, or synchronization is introduced here. The
// kernel's thread/row mapping is derived from the block geometry, so a launch
// passes exactly kRmsnormThreads threads per block.
template <typename Policy>
void launch_standard_tiled_rmsnorm(
        typename Policy::stream_type stream,
        const RmsnormMetadata& metadata) {
    const std::size_t rows = rmsnorm_checked_mul(
            static_cast<std::size_t>(metadata.plane_count),
            static_cast<std::size_t>(metadata.rows),
            "RMSNORM launch row count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            rows < kRmsnormMaxBlocks ? rows : kRmsnormMaxBlocks);
    IOM_LAUNCH_KERNEL(
            standard_tiled_rmsnorm_kernel, blocks, kRmsnormThreads, stream,
            metadata);
}

}  // namespace iom::detail