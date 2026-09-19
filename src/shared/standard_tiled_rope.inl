#pragma once
// Standard 16x16 tiled split-half rotary-position-encoding metadata, device
// kernel, and launch boundary shared by the CUDA and ROCm translation units.
// The backend supplies only the IOM_GPU_* macros before including this file.
//
// The operation consumes the already-admitted immutable RopeRequest.  It does
// no validation of TensorView objects, no allocation, no host arithmetic over
// payloads, no host round trip, and no caller scratch.  The descriptor carries
// only value snapshots and pointers into one fixed queue metadata slot.  Each
// destination 32-bit word has one owning work item and one final store; logical
// tile padding and unowned output bits are never changed.

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
#error "IOM_GPU_DEVICE must be defined before including standard_tiled_rope.inl"
#endif
#ifndef IOM_GPU_GLOBAL
#error "IOM_GPU_GLOBAL must be defined before including standard_tiled_rope.inl"
#endif
#ifndef IOM_GPU_GLOBAL_INDEX
#error "IOM_GPU_GLOBAL_INDEX must be defined before including standard_tiled_rope.inl"
#endif
#ifndef IOM_LAUNCH_KERNEL
#error "IOM_LAUNCH_KERNEL must be defined before including standard_tiled_rope.inl"
#endif
#ifndef IOM_GPU_GLOBAL_STRIDE
#define IOM_GPU_GLOBAL_STRIDE \
    (static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
#endif

namespace iom::detail {

// Immutable device-visible descriptor.  `leading_rank` is the number of
// dimensions before the final two tiled matrix dimensions, so its last axis
// is H and the preceding axes are the independent outer leading planes.
// `dims`, `x_strides`, and `out_strides` point into the same fixed metadata
// slot's uploaded value array and contain exactly `leading_rank` words each.
struct RopeMetadata {
    const unsigned char* x;
    unsigned char* out;
    const std::uint64_t* dims;
    const std::uint64_t* x_strides;
    const std::uint64_t* out_strides;
    std::uint64_t heads;
    std::uint64_t rows;
    std::uint64_t width;
    std::uint64_t plane_count;
    std::uint64_t words_per_plane;
    std::uint64_t total_words;
    std::uint64_t x_offset;
    std::uint64_t out_offset;
    std::uint64_t position;
    double theta;
    std::uint32_t bits;
    std::uint32_t type;
    std::uint32_t leading_rank;
};
static_assert(std::is_trivially_copyable_v<RopeMetadata>);
// Rank eight is the widest admitted shape (six leading axes).  The complete
// descriptor and all three six-word arrays must stay inside one fixed slot.
static_assert(
        sizeof(RopeMetadata) + 3 * 6 * sizeof(std::uint64_t)
        <= kMetadataSlotBytes);
static_assert(
        alignof(RopeMetadata) <= 32
        && kMetadataSlotBytes % alignof(RopeMetadata) == 0);

namespace {

constexpr std::uint64_t kRopeTile = TensorSpec::TILE;
constexpr unsigned int kRopeThreads = 256;
constexpr unsigned int kRopeMaxBlocks = 65535;

[[nodiscard]] std::size_t rope_checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t rope_checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t rope_metadata_u64(
        std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::size_t rope_padded_extent(
        std::size_t value, const char* message) {
    const std::size_t bumped = rope_checked_add(
            value, static_cast<std::size_t>(kRopeTile) - 1, message);
    return bumped - bumped % static_cast<std::size_t>(kRopeTile);
}

// Byte length of the fixed-slot descriptor for one admitted rank.
[[nodiscard]] std::size_t rope_metadata_storage_bytes(std::size_t rank) {
    if (rank < 3 || rank > 8) {
        throw std::invalid_argument("ROPE metadata rank is out of range");
    }
    const std::size_t leading_rank = rank - 2;
    const std::size_t arrays = rope_checked_mul(
            rope_checked_mul(
                    leading_rank, sizeof(std::uint64_t),
                    "ROPE metadata rank storage overflows"),
            3, "ROPE metadata rank storage overflows");
    const std::size_t result = rope_checked_add(
            sizeof(RopeMetadata), arrays,
            "ROPE metadata storage size overflows");
    if (result > kMetadataSlotBytes) {
        throw std::overflow_error("ROPE metadata exceeds fixed slot");
    }
    return result;
}

// Convert the admitted immutable request into a checked descriptor.  Common
// admission owns shape, device, alias, parameter, dtype, and workspace
// validation; this function only checks descriptor representation and fixed
// launch arithmetic before the worker uploads it.
template <typename Request>
[[nodiscard]] RopeMetadata make_rope_metadata(const Request& request) {
    const std::span<const std::size_t> dimensions = {
            request.x.dimensions.data(), request.x.rank};
    if (dimensions.size() < 3 || dimensions.size() > 8
            || request.out.rank != dimensions.size()) {
        throw std::invalid_argument("ROPE metadata rank is out of range");
    }
    const std::size_t leading_rank = dimensions.size() - 2;
    (void)rope_metadata_storage_bytes(dimensions.size());

    RopeMetadata metadata{};
    metadata.x = static_cast<const unsigned char*>(request.x.native_handle);
    metadata.out = static_cast<unsigned char*>(request.out.native_handle);
    metadata.heads = rope_metadata_u64(
            dimensions[leading_rank - 1],
            "ROPE metadata head extent overflows");
    metadata.rows = rope_metadata_u64(
            dimensions[leading_rank],
            "ROPE metadata row extent overflows");
    metadata.width = rope_metadata_u64(
            dimensions[leading_rank + 1],
            "ROPE metadata width extent overflows");
    metadata.x_offset = rope_metadata_u64(
            request.x.plane_offset, "ROPE metadata x offset overflows");
    metadata.out_offset = rope_metadata_u64(
            request.out.plane_offset, "ROPE metadata output offset overflows");
    metadata.position = rope_metadata_u64(
            request.a, "ROPE metadata position overflows");
    metadata.theta = request.theta;
    metadata.bits = static_cast<std::uint32_t>(
            leaf_bits(request.x.data_type));
    metadata.type = static_cast<std::uint32_t>(request.x.data_type);
    metadata.leading_rank = static_cast<std::uint32_t>(leading_rank);

    // `plane_count` excludes H; H is mapped independently with its own view
    // stride so unequal leading views remain independent in every call.
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis + 1 < leading_rank; ++axis) {
        plane_count = rope_checked_mul(
                plane_count, dimensions[axis],
                "ROPE metadata leading plane count overflows");
    }
    metadata.plane_count = rope_metadata_u64(
            plane_count, "ROPE metadata leading plane count overflows");

    const std::size_t padded_rows = rope_padded_extent(
            dimensions[leading_rank], "ROPE metadata padded rows overflow");
    const std::size_t padded_width = rope_padded_extent(
            dimensions[leading_rank + 1],
            "ROPE metadata padded width overflow");
    const std::size_t padded_elements = rope_checked_mul(
            padded_rows, padded_width,
            "ROPE metadata padded plane size overflows");
    const std::size_t plane_bits = rope_checked_mul(
            padded_elements, static_cast<std::size_t>(metadata.bits),
            "ROPE metadata plane bits overflows");
    metadata.words_per_plane = rope_metadata_u64(
            rope_checked_add(
                    plane_bits, std::size_t{31},
                    "ROPE metadata plane word count overflows") / 32,
            "ROPE metadata plane word count overflows");
    const std::size_t total_planes = rope_checked_mul(
            plane_count, static_cast<std::size_t>(metadata.heads),
            "ROPE metadata total plane count overflows");
    metadata.total_words = rope_metadata_u64(
            rope_checked_mul(
                    total_planes,
                    static_cast<std::size_t>(metadata.words_per_plane),
                    "ROPE metadata total word count overflows"),
            "ROPE metadata total word count overflows");

    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        (void)rope_metadata_u64(
                request.x.plane_strides[axis],
                "ROPE metadata x stride overflows");
        (void)rope_metadata_u64(
                request.out.plane_strides[axis],
                "ROPE metadata output stride overflows");
    }
    return metadata;
}

// Write the descriptor and its rank-dependent arrays into one fixed host
// mirror.  The device pointers refer to the uploaded copy of these arrays,
// never to host memory.
template <typename Request>
void write_rope_metadata(
        void* host_storage, const void* device_storage,
        const Request& request) {
    RopeMetadata metadata = make_rope_metadata(request);
    const std::span<const std::size_t> dimensions = {
            request.x.dimensions.data(), request.x.rank};
    const std::size_t leading_rank = dimensions.size() - 2;
    auto* host_bytes = static_cast<std::byte*>(host_storage);
    const auto* device_bytes = static_cast<const std::byte*>(device_storage);
    const std::size_t arrays_offset = sizeof(RopeMetadata);
    auto* host_dims = reinterpret_cast<std::uint64_t*>(
            host_bytes + arrays_offset);
    auto* host_x_strides = host_dims + leading_rank;
    auto* host_out_strides = host_x_strides + leading_rank;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        host_dims[axis] = rope_metadata_u64(
                dimensions[axis], "ROPE metadata leading extent overflows");
        host_x_strides[axis] = rope_metadata_u64(
                request.x.plane_strides[axis],
                "ROPE metadata x stride overflows");
        host_out_strides[axis] = rope_metadata_u64(
                request.out.plane_strides[axis],
                "ROPE metadata output stride overflows");
    }
    const auto* device_values = reinterpret_cast<const std::uint64_t*>(
            device_bytes + arrays_offset);
    metadata.dims = device_values;
    metadata.x_strides = device_values + leading_rank;
    metadata.out_strides = device_values + leading_rank * 2;
    *reinterpret_cast<RopeMetadata*>(host_storage) = metadata;
}

// The eight non-F64 leaves use the binary32 domain for every operation before
// the single named-format encoding. F64 gets a separate binary64 domain and
// is never narrowed through the other trait.
struct RopeFp32Traits {
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
        return ::fabsf(value);
    }
    IOM_GPU_DEVICE static carrier_type floor(carrier_type value) noexcept {
        return ::floorf(value);
    }
    IOM_GPU_DEVICE static carrier_type ldexp(
            carrier_type value, int exponent) noexcept {
        return ::ldexpf(value, exponent);
    }
    IOM_GPU_DEVICE static carrier_type frexp(
            carrier_type value, int* exponent) noexcept {
        return ::frexpf(value, exponent);
    }
    IOM_GPU_DEVICE static carrier_type power(
            carrier_type value, carrier_type exponent) noexcept {
        return ::powf(value, exponent);
    }
    IOM_GPU_DEVICE static carrier_type cosine(carrier_type value) noexcept {
        return ::cosf(value);
    }
    IOM_GPU_DEVICE static carrier_type sine(carrier_type value) noexcept {
        return ::sinf(value);
    }
    IOM_GPU_DEVICE static carrier_type noncontracted_multiply(
            carrier_type left, carrier_type right) noexcept {
        // Explicit round-to-nearest intrinsics keep Rope's pair transform out
        // of FMA contraction; the shared linear facility retains its
        // intentional FMA policy.
        return __fmul_rn(left, right);
    }
    IOM_GPU_DEVICE static carrier_type noncontracted_add(
            carrier_type left, carrier_type right) noexcept {
        return __fadd_rn(left, right);
    }
    IOM_GPU_DEVICE static carrier_type noncontracted_subtract(
            carrier_type left, carrier_type right) noexcept {
        return __fsub_rn(left, right);
    }
};

struct RopeFp64Traits {
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
    IOM_GPU_DEVICE static carrier_type power(
            carrier_type value, carrier_type exponent) noexcept {
        return ::pow(value, exponent);
    }
    IOM_GPU_DEVICE static carrier_type cosine(carrier_type value) noexcept {
        return ::cos(value);
    }
    IOM_GPU_DEVICE static carrier_type sine(carrier_type value) noexcept {
        return ::sin(value);
    }
    IOM_GPU_DEVICE static carrier_type noncontracted_multiply(
            carrier_type left, carrier_type right) noexcept {
        return __dmul_rn(left, right);
    }
    IOM_GPU_DEVICE static carrier_type noncontracted_add(
            carrier_type left, carrier_type right) noexcept {
        return __dadd_rn(left, right);
    }
    IOM_GPU_DEVICE static carrier_type noncontracted_subtract(
            carrier_type left, carrier_type right) noexcept {
        return __dsub_rn(left, right);
    }
};

template <typename Carrier>
struct RopeCodecOf;

template <>
struct RopeCodecOf<float> {
    using traits = RopeFp32Traits;
    using type = scalar_binary_codec_detail::Codec<traits>;
};

template <>
struct RopeCodecOf<double> {
    using traits = RopeFp64Traits;
    using type = scalar_binary_codec_detail::Codec<traits>;
};

IOM_GPU_DEVICE std::uint32_t rope_field_mask(unsigned int bits) noexcept {
    return bits == 32 ? 0xffffffffu
                      : ((std::uint32_t{1} << bits) - 1);
}

IOM_GPU_DEVICE std::uint32_t rope_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    return reinterpret_cast<const std::uint32_t*>(base)[word];
}

IOM_GPU_DEVICE void rope_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    reinterpret_cast<std::uint32_t*>(base)[word] = value;
}

IOM_GPU_DEVICE std::uint64_t rope_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned int bits) noexcept {
    const std::uint64_t word = bit / 32;
    std::uint64_t joined = rope_load_word(base, word);
    if (bit % 32 + bits > 32) {
        joined |= static_cast<std::uint64_t>(
                          rope_load_word(base, word + 1))
                << 32;
    }
    return (joined >> (bit % 32))
            & (bits == 64
               ? ~std::uint64_t{}
               : ((std::uint64_t{1} << bits) - 1));
}

IOM_GPU_DEVICE void rope_merge_field(
        std::uint32_t& destination_word, std::uint64_t encoded,
        std::uint64_t field_bit, std::uint64_t word_first_bit,
        unsigned int bits) noexcept {
    const std::uint64_t word_end_bit = word_first_bit + 32;
    const std::uint64_t field_end_bit = field_bit + bits;
    const std::uint64_t overlap_first =
            field_bit > word_first_bit ? field_bit : word_first_bit;
    const std::uint64_t overlap_end =
            field_end_bit < word_end_bit ? field_end_bit : word_end_bit;
    if (overlap_first >= overlap_end) {
        return;
    }
    const unsigned int count = static_cast<unsigned int>(
            overlap_end - overlap_first);
    const unsigned int destination_shift = static_cast<unsigned int>(
            overlap_first - word_first_bit);
    const unsigned int source_shift = static_cast<unsigned int>(
            overlap_first - field_bit);
    const std::uint32_t segment = static_cast<std::uint32_t>(
            encoded >> source_shift) & rope_field_mask(count);
    destination_word = (destination_word
                        & ~(rope_field_mask(count) << destination_shift))
            | (segment << destination_shift);
}

// Evaluate one logical destination field.  Position zero returns the source
// payload directly, preserving signed zero and every admitted nonfinite or
// sub-byte payload bit-for-bit.  Nonzero positions capture both split-half
// partners before the caller's single destination-word store.
template <typename Carrier>
IOM_GPU_DEVICE std::uint64_t rope_encoded_field(
        const RopeMetadata& metadata, std::uint64_t x_plane,
        std::uint64_t row, std::uint64_t column,
        scalar_binary_codec_detail::Format format) noexcept {
    using Codec = typename RopeCodecOf<Carrier>::type;
    const unsigned int bits = metadata.bits;
    const std::uint64_t half = metadata.width / 2;
    const std::uint64_t pair_column =
            column < half ? column : column - half;
    const std::uint64_t first_bit = plane_slot(
            x_plane, row, pair_column, metadata.rows, metadata.width) * bits;
    const std::uint64_t second_bit = plane_slot(
            x_plane, row, pair_column + half,
            metadata.rows, metadata.width) * bits;
    const std::uint64_t first_raw = rope_load_bits(
            metadata.x, first_bit, bits);
    const std::uint64_t second_raw = rope_load_bits(
            metadata.x, second_bit, bits);
    const std::uint64_t absolute_position = metadata.position + row;
    if (absolute_position == 0) {
        return column < half ? first_raw : second_raw;
    }

    const Carrier first = Codec::decode(first_raw, format);
    const Carrier second = Codec::decode(second_raw, format);
    const Carrier position = static_cast<Carrier>(absolute_position);
    const Carrier theta = static_cast<Carrier>(metadata.theta);
    const Carrier exponent = static_cast<Carrier>(-2)
            * static_cast<Carrier>(pair_column)
            / static_cast<Carrier>(metadata.width);
    using Traits = typename RopeCodecOf<Carrier>::traits;
    const Carrier frequency = Traits::power(theta, exponent);
    const Carrier angle = position * frequency;
    const Carrier cosine = Traits::cosine(angle);
    const Carrier sine = Traits::sine(angle);

    const Carrier first_product =
            Traits::noncontracted_multiply(first, cosine);
    const Carrier second_product =
            Traits::noncontracted_multiply(second, sine);
    const Carrier rotated_first =
            Traits::noncontracted_subtract(first_product, second_product);
    const Carrier second_cosine_product =
            Traits::noncontracted_multiply(second, cosine);
    const Carrier first_sine_product =
            Traits::noncontracted_multiply(first, sine);
    const Carrier rotated_second = Traits::noncontracted_add(
            second_cosine_product, first_sine_product);
    return column < half
            ? Codec::encode(rotated_first, format)
            : Codec::encode(rotated_second, format);
}

template <typename Carrier>
IOM_GPU_DEVICE void rope_process_word(
        const RopeMetadata& metadata, std::uint64_t word) noexcept {
    using Codec = typename RopeCodecOf<Carrier>::type;
    const scalar_binary_codec_detail::Format format = Codec::format(
            static_cast<DataType>(metadata.type));
    const std::uint64_t plane_index = word / metadata.words_per_plane;
    const std::uint64_t word_in_plane = word % metadata.words_per_plane;
    const std::uint64_t head = plane_index % metadata.heads;
    std::uint64_t outer_plane = plane_index / metadata.heads;
    const std::uint32_t head_axis = metadata.leading_rank - 1;
    std::uint64_t x_plane = metadata.x_offset;
    std::uint64_t out_plane = metadata.out_offset;
    for (std::uint32_t axis = head_axis; axis-- > 0;) {
        const std::uint64_t coordinate = outer_plane % metadata.dims[axis];
        outer_plane /= metadata.dims[axis];
        x_plane += coordinate * metadata.x_strides[axis];
        out_plane += coordinate * metadata.out_strides[axis];
    }
    x_plane += head * metadata.x_strides[head_axis];
    out_plane += head * metadata.out_strides[head_axis];

    const std::uint64_t word_first_bit = word_in_plane * 32;
    const std::uint64_t first_slot = word_first_bit / metadata.bits;
    const std::uint64_t last_slot =
            (word_first_bit + 31) / metadata.bits;
    const std::uint64_t destination_base_word = plane_slot(
            out_plane, 0, 0, metadata.rows, metadata.width)
            * metadata.bits / 32;
    auto* destination = reinterpret_cast<std::uint32_t*>(metadata.out)
            + destination_base_word + word_in_plane;

    std::uint32_t merged = 0;
    bool touched = false;
    for (std::uint64_t slot = first_slot; slot <= last_slot; ++slot) {
        const PhysicalCoordinate coordinate = physical_coordinate(
                slot, metadata.rows, metadata.width);
        if (coordinate.row >= metadata.rows
                || coordinate.column >= metadata.width) {
            continue;
        }
        const std::uint64_t field_bit = slot * metadata.bits;
        if (field_bit + metadata.bits <= word_first_bit
                || field_bit >= word_first_bit + 32) {
            continue;
        }
        if (!touched) {
            // This is the only output load and the only eventual store for
            // this destination word; untouched fields, including padding,
            // retain their existing payload bits.
            merged = *destination;
            touched = true;
        }
        const std::uint64_t encoded = rope_encoded_field<Carrier>(
                metadata, x_plane, coordinate.row, coordinate.column, format);
        rope_merge_field(
                merged, encoded, field_bit, word_first_bit, metadata.bits);
    }
    if (touched) {
        rope_store_word(
                metadata.out, destination_base_word + word_in_plane, merged);
    }
}

template <typename Carrier>
IOM_GPU_DEVICE void rope_process_words(const RopeMetadata& metadata) noexcept {
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    for (std::uint64_t word = IOM_GPU_GLOBAL_INDEX;
         word < metadata.total_words; word += stride) {
        rope_process_word<Carrier>(metadata, word);
    }
}

IOM_GPU_GLOBAL void standard_tiled_rope_kernel(RopeMetadata metadata) {
    if (metadata.type == static_cast<std::uint32_t>(DataType::F64)) {
        rope_process_words<double>(metadata);
    } else {
        rope_process_words<float>(metadata);
    }
}

}  // namespace

// The backend wrapper supplies the queue's already-created in-order stream.
// No second stream, allocation, staging, or synchronization is introduced.
template <typename Policy>
void launch_standard_tiled_rope(
        typename Policy::stream_type stream,
        const RopeMetadata& metadata) {
    const std::size_t rounded = rope_checked_add(
            static_cast<std::size_t>(metadata.total_words),
            static_cast<std::size_t>(kRopeThreads) - 1,
            "ROPE launch word count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            rounded / kRopeThreads < kRopeMaxBlocks
            ? rounded / kRopeThreads : kRopeMaxBlocks);
    if (blocks != 0) {
        IOM_LAUNCH_KERNEL(
                standard_tiled_rope_kernel, blocks, kRopeThreads, stream,
                metadata);
    }
}

}  // namespace iom::detail
