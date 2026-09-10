#pragma once
// Standard 16x16 tiled ADD metadata and device kernel body shared by the
// CUDA and ROCm translation units. No vendor types: the backend supplies
// only the IOM_GPU_* macros before including this file.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <cmath>
#include "iom/tensor.hpp"
#include "iom/iom.hpp"

#include "scalar_binary_codec.hpp"
#ifndef IOM_GPU_DEVICE
#error "IOM_GPU_DEVICE must be defined before including standard_tiled_add.inl"
#endif
#ifndef IOM_GPU_GLOBAL
#error "IOM_GPU_GLOBAL must be defined before including standard_tiled_add.inl"
#endif
#ifndef IOM_GPU_GLOBAL_INDEX
#error "IOM_GPU_GLOBAL_INDEX must be defined before including standard_tiled_add.inl"
#endif
#ifndef IOM_GPU_GLOBAL_STRIDE
#define IOM_GPU_GLOBAL_STRIDE \
    (static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
#endif
#ifndef IOM_LAUNCH_KERNEL
#error "IOM_LAUNCH_KERNEL must be defined before including standard_tiled_add.inl"
#endif

namespace iom::detail {
namespace {


struct BinaryMetadata {
    std::uint64_t rows;
    std::uint64_t columns;
    std::uint64_t total_words;
    std::uint32_t bits;
    std::uint32_t type;
    std::uint32_t rank;
    std::uint64_t out_offset;
    std::uint64_t lhs_offset;
    std::uint64_t rhs_offset;
    const std::uint64_t* dims;
    const std::uint64_t* lhs_strides;
    const std::uint64_t* rhs_strides;
    const std::uint64_t* out_strides;
    std::uint64_t lhs_rows;
    std::uint64_t lhs_columns;
    std::uint64_t rhs_rows;
    std::uint64_t rhs_columns;
    std::uint32_t lhs_brow;
    std::uint32_t lhs_bcol;
    std::uint32_t rhs_brow;
    std::uint32_t rhs_bcol;
};


[[nodiscard]] std::size_t add_checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t add_checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}
[[nodiscard]] std::size_t binary_metadata_storage_bytes(
        std::size_t rank) {
    const std::size_t arrays = add_checked_mul(
            add_checked_mul(
                    rank, sizeof(std::uint64_t),
                    "ADD metadata rank storage overflows"),
            4, "ADD metadata rank storage overflows");
    return add_checked_add(
            sizeof(BinaryMetadata), arrays,
            "ADD metadata storage size overflows");
}

// ---------------------------------------------------------------------------
// Scalar ADD semantics, aligned with src/shared/scalar_add.hpp (the task-11
// independent oracle). Compact and wide floating formats are evaluated in
// double, which represents every operand of these formats exactly, so RNE
// encoding reproduces the reference encodings bit for bit.
// ---------------------------------------------------------------------------

struct DeviceTraits {
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
};

using DeviceCodec = scalar_binary_codec_detail::Codec<DeviceTraits>;
using DeviceBinaryOp = scalar_binary_codec_detail::BinaryOp;


// ---------------------------------------------------------------------------
// Packed-field primitives. Element slot s of a plane occupies bits
// [s*bits, (s+1)*bits) of the plane's packed stream; 32-bit words are the
// unit of exclusive ownership, so every writer merges its elements into one
// word and stores it once (no read-modify-write races, and untouched padding
// bits are preserved by starting from the observed word).
// ---------------------------------------------------------------------------

IOM_GPU_DEVICE std::uint32_t add_field_mask(unsigned int bits) noexcept {
    return bits == 32 ? 0xffffffffu : ((std::uint32_t{1} << bits) - 1);
}

IOM_GPU_DEVICE std::uint32_t add_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value;
    for (unsigned i = 0; i < 4; ++i) {
        reinterpret_cast<unsigned char*>(&value)[i] = base[word * 4 + i];
    }
    return value;
}

IOM_GPU_DEVICE void add_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned i = 0; i < 4; ++i) {
        base[word * 4 + i] = reinterpret_cast<const unsigned char*>(&value)[i];
    }
}

IOM_GPU_DEVICE std::uint64_t add_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned bits) noexcept {
    const std::uint64_t word = bit / 32;
    std::uint64_t joined = add_load_word(base, word);
    if (bit % 32 + bits > 32) {
        joined |= static_cast<std::uint64_t>(
                      add_load_word(base, word + 1))
                << 32;
    }
    return (joined >> (bit % 32))
            & (bits == 64
               ? ~std::uint64_t{}
               : ((std::uint64_t{1} << bits) - 1));
}


template <DeviceBinaryOp Op>
IOM_GPU_DEVICE void binary_body(
        const unsigned char* lhs, const unsigned char* rhs,
        unsigned char* out, const BinaryMetadata& m) noexcept {
    const std::uint64_t padded_rows =
            (m.rows + kTile - 1) / kTile * kTile;
    const std::uint64_t padded_columns =
            (m.columns + kTile - 1) / kTile * kTile;
    const std::uint64_t words_per_plane =
            (padded_rows * padded_columns * m.bits + 31) / 32;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;

    for (std::uint64_t word = IOM_GPU_GLOBAL_INDEX;
         word < m.total_words; word += stride) {
        const std::uint64_t plane = word / words_per_plane;
        const std::uint64_t word_in_plane = word % words_per_plane;
        const std::uint64_t base = word_in_plane * 32;

        // Logical coordinate mapping before physical tile-slot mapping:
        // decompose the leading plane index with the transformed strides.
        std::uint64_t lhs_plane = m.lhs_offset;
        std::uint64_t rhs_plane = m.rhs_offset;
        std::uint64_t out_plane = m.out_offset;
        std::uint64_t rest = plane;
        for (std::uint32_t axis = m.rank - 2; axis-- > 0;) {
            const std::uint64_t coordinate = rest % m.dims[axis];
            rest /= m.dims[axis];
            lhs_plane += coordinate * m.lhs_strides[axis];
            rhs_plane += coordinate * m.rhs_strides[axis];
            out_plane += coordinate * m.out_strides[axis];
        }

        // Exclusive ownership: read the existing word, merge only the bits
        // this word owns, and store once. Padding bits and padding slots
        // never change.
        std::uint32_t acc = add_load_word(out, word);
        const std::uint64_t first_slot = base / m.bits;
        const std::uint64_t last_slot = (base + 31) / m.bits;
        for (std::uint64_t slot = first_slot; slot <= last_slot; ++slot) {
            const std::uint64_t slot_bit = slot * m.bits;
            if (slot_bit + m.bits <= base || slot_bit >= base + 32) {
                continue;
            }
            const PhysicalCoordinate coordinate =
                    physical_coordinate(slot, m.rows, m.columns);
            const std::uint64_t row = coordinate.row;
            const std::uint64_t column = coordinate.column;
            if (row >= m.rows || column >= m.columns) {
                continue;  // padding slots are never read or written
            }
            const std::uint64_t lhs_row = m.lhs_brow ? 0 : row;
            const std::uint64_t lhs_column = m.lhs_bcol ? 0 : column;
            const std::uint64_t rhs_row = m.rhs_brow ? 0 : row;
            const std::uint64_t rhs_column = m.rhs_bcol ? 0 : column;
            const std::uint64_t lhs_slot = plane_slot(
                    lhs_plane, lhs_row, lhs_column,
                    m.lhs_rows, m.lhs_columns);
            const std::uint64_t rhs_slot = plane_slot(
                    rhs_plane, rhs_row, rhs_column,
                    m.rhs_rows, m.rhs_columns);
            // Capture both input values before any output store.
            const std::uint64_t a = add_load_bits(
                    lhs, lhs_slot * m.bits, m.bits);
            const std::uint64_t b = add_load_bits(
                    rhs, rhs_slot * m.bits, m.bits);
            const std::uint64_t value =
                    DeviceCodec::binary<Op>(
                            static_cast<DataType>(m.type), a, b);

            const std::uint64_t lo = slot_bit > base ? slot_bit : base;
            const std::uint64_t hi =
                    slot_bit + m.bits < base + 32 ? slot_bit + m.bits
                                                  : base + 32;
            const unsigned count = static_cast<unsigned>(hi - lo);
            const unsigned position = static_cast<unsigned>(lo - base);
            const unsigned shift = static_cast<unsigned>(lo - slot_bit);
            const std::uint32_t segment = static_cast<std::uint32_t>(
                    value >> shift) & add_field_mask(count);
            acc = (acc & ~(add_field_mask(count) << position))
                    | (segment << position);
        }
        add_store_word(out, word, acc);
    }
}

template <DeviceBinaryOp Op>
IOM_GPU_GLOBAL void grid_stride_binary_kernel(
        const unsigned char* lhs, const unsigned char* rhs,
        unsigned char* out, BinaryMetadata metadata) {
    binary_body<Op>(lhs, rhs, out, metadata);
}

template <typename Policy, DeviceBinaryOp Op>
void launch_grid_stride_binary(
        typename Policy::stream_type stream, const unsigned char* lhs,
        const unsigned char* rhs, unsigned char* out,
        const BinaryMetadata& metadata) {
    const std::uint64_t launch_words =
            add_checked_add(metadata.total_words, 255,
                    "ADD launch word count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            launch_words / 256 < 65535 ? launch_words / 256 : 65535);
    IOM_LAUNCH_KERNEL(
            grid_stride_binary_kernel<Op>, blocks, 256, stream, lhs, rhs, out,
            metadata);
}

template <typename Request>
[[nodiscard]] BinaryMetadata make_binary_metadata(const Request& request) {
    const auto dimensions = request.result_shape.dimensions();
    if (dimensions.size() < 2) {
        throw std::invalid_argument("ADD rank below tiled matrix rank");
    }
    if (dimensions.size()
            > static_cast<std::size_t>(
                      std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
                "ADD metadata rank representation overflows");
    }
    const std::size_t rank = dimensions.size();
    BinaryMetadata metadata{};
    metadata.rank = static_cast<std::uint32_t>(rank);
    metadata.rows = dimensions[rank - 2];
    metadata.columns = dimensions[rank - 1];
    metadata.bits = static_cast<std::uint32_t>(
            leaf_bits(request.out.spec.data_type));
    metadata.type =
            static_cast<std::uint32_t>(request.out.spec.data_type);
    metadata.out_offset = request.out.plane_offset;
    metadata.lhs_offset = request.lhs.plane_offset;
    metadata.rhs_offset = request.rhs.plane_offset;
    const auto lhs_dimensions = request.lhs.spec.shape.dimensions();
    const auto rhs_dimensions = request.rhs.spec.shape.dimensions();
    metadata.lhs_rows = lhs_dimensions[lhs_dimensions.size() - 2];
    metadata.lhs_columns = lhs_dimensions[lhs_dimensions.size() - 1];
    metadata.rhs_rows = rhs_dimensions[rhs_dimensions.size() - 2];
    metadata.rhs_columns = rhs_dimensions[rhs_dimensions.size() - 1];
    metadata.lhs_brow = request.lhs.broadcast_rows ? 1u : 0u;
    metadata.lhs_bcol = request.lhs.broadcast_columns ? 1u : 0u;
    metadata.rhs_brow = request.rhs.broadcast_rows ? 1u : 0u;
    metadata.rhs_bcol = request.rhs.broadcast_columns ? 1u : 0u;
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis + 2 < rank; ++axis) {
        plane_count = add_checked_mul(
                plane_count, static_cast<std::size_t>(dimensions[axis]),
                "ADD metadata plane count overflows");
    }
    const std::size_t padded_rows = add_checked_mul(
            (add_checked_add(
                    static_cast<std::size_t>(metadata.rows),
                    static_cast<std::size_t>(kTile) - 1,
                    "ADD metadata row padding overflows")
             / static_cast<std::size_t>(kTile)),
            static_cast<std::size_t>(kTile),
            "ADD metadata padded rows overflow");
    const std::size_t padded_columns = add_checked_mul(
            (add_checked_add(
                    static_cast<std::size_t>(metadata.columns),
                    static_cast<std::size_t>(kTile) - 1,
                    "ADD metadata column padding overflows")
             / static_cast<std::size_t>(kTile)),
            static_cast<std::size_t>(kTile),
            "ADD metadata column padding overflow");
    const std::size_t words_per_plane = add_checked_add(
            add_checked_mul(
                    add_checked_mul(
                            padded_rows, padded_columns,
                            "ADD metadata element count overflows"),
                    static_cast<std::size_t>(metadata.bits),
                    "ADD metadata plane bits overflows"),
            31, "ADD metadata word count overflows")
            / 32;
    metadata.total_words = add_checked_mul(
            plane_count, words_per_plane,
            "ADD metadata total words overflows");
    return metadata;
}

template <typename Request>
void write_binary_metadata(
        void* host_storage, const void* device_storage,
        const Request& request) {
    const auto dimensions = request.result_shape.dimensions();
    const std::size_t rank = dimensions.size();
    BinaryMetadata metadata = make_binary_metadata(request);
    auto* host_bytes = static_cast<std::byte*>(host_storage);
    const auto* device_bytes =
            static_cast<const std::byte*>(device_storage);
    const std::size_t arrays_offset = sizeof(BinaryMetadata);
    auto* host_dims = reinterpret_cast<std::uint64_t*>(
            host_bytes + arrays_offset);
    auto* host_lhs_strides = host_dims + rank;
    auto* host_rhs_strides = host_lhs_strides + rank;
    auto* host_out_strides = host_rhs_strides + rank;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        host_dims[axis] = dimensions[axis];
        host_lhs_strides[axis] = 0;
        host_rhs_strides[axis] = 0;
        host_out_strides[axis] = 0;
    }
    for (std::size_t axis = 0; axis + 2 < rank; ++axis) {
        host_lhs_strides[axis] =
                request.lhs.logical_plane_strides[axis];
        host_rhs_strides[axis] =
                request.rhs.logical_plane_strides[axis];
        host_out_strides[axis] =
                request.out.logical_plane_strides[axis];
    }
    const auto* device_dims = reinterpret_cast<const std::uint64_t*>(
            device_bytes + arrays_offset);
    metadata.dims = device_dims;
    metadata.lhs_strides = device_dims + rank;
    metadata.rhs_strides = device_dims + rank * 2;
    metadata.out_strides = device_dims + rank * 3;
    *reinterpret_cast<BinaryMetadata*>(host_storage) = metadata;
}

}  // namespace
}  // namespace iom::detail
