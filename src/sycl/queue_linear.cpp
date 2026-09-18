// SYCL linear-projection queue path.
//
// One native in-order `parallel_for` launch over the logical output rows of the
// `docs/BACKEND_CONTRACT.md` **Linear projections** contract. One work item
// owns one complete output row: it resolves that row's leading plane and head
// through the operand's own transformed plane offsets and strides, walks the
// logical inner extent `I` of the selected source row and of the weight row the
// output column names, and encodes exactly one result per logical output
// element. The integer leaves accumulate an unsigned modulo-`2^N` dot that
// reduces after every multiply and every add; the floating leaves start at `+0`
// and accumulate a correctly rounded fused multiply-add in FP32 (FP64 for
// `F64`). Nothing is staged, relocated, or computed on the host: admission
// already validated the operands, so this translation unit only snapshots the
// request into a bounded trivially copyable descriptor, submits the device
// kernel on the existing in-order queue, and projects completion through the
// existing fence, owner-registration, and quarantine machinery.
#include "queue_internal.hpp"

#include <sycl/sycl.hpp>

// The native `BF16` specialization below is the one linear leaf that queues a
// subgroup-16 `joint_matrix` BF16/BF16/FP32 MAD, so this translation unit
// requires the pinned oneAPI experimental matrix extension: the header, the
// `SYCL_EXT_ONEAPI_MATRIX` feature macro, and the subgroup-16 facility the
// runtime query validates before any submission. A SYCL-enabled configuration
// whose compiler/header contract lacks the extension fails here with that
// statement instead of silently dropping a leaf the driver declares; a
// recognized but different extension revision keeps the runtime `Unsupported`
// rejection path and compiles no matrix code.
#if !defined(SYCL_EXT_ONEAPI_MATRIX)
#error "The SYCL linear BF16 specialization requires the oneAPI experimental matrix extension (sycl/feature_test.hpp SYCL_EXT_ONEAPI_MATRIX), which the pinned oneAPI compiler provides."
#endif

#if SYCL_EXT_ONEAPI_MATRIX == 1
#define IOM_SYCL_BF16_MATRIX 1
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "runtime.hpp"
#include "../iom_internal.hpp"

namespace iom::sycl_detail {
namespace {

// The common admission contract admits rank two through eight, so neither the
// input's leading rank nor the output's — which may carry the one inserted head
// axis inside that bound — can exceed six; the bound is a descriptor capacity
// guard, not a second admission check.
inline constexpr std::size_t kLinearMaxLeadingRank = 6;
inline constexpr std::uint64_t kLinearTile = TensorSpec::TILE;
inline constexpr std::uint64_t kLinearTileSlots =
        static_cast<std::uint64_t>(TensorSpec::TILE)
        * static_cast<std::uint64_t>(TensorSpec::TILE);

// Immutable bounded snapshot of one admitted request: the logical row window,
// the exact leaf encoding and integer width, the head scalars, the explicit
// output mode, every operand plane offset, the output's own leading extents and
// transformed strides, and the input's transformed leading strides. It is
// captured by value into the device kernel, so the task retains no borrowed
// view and the queue needs no extra slot or allocation.
struct LinearMetadata {
    // Output planes: the product of the output view's own leading extents,
    // which already carries the head axis in head-planar mode.
    std::uint64_t plane_count = 0;
    std::uint64_t rows = 0;
    std::uint64_t source_rows = 0;
    std::uint64_t inner = 0;
    std::uint64_t outer = 0;
    std::uint64_t out_columns = 0;
    std::uint64_t head_dim = 0;
    std::uint64_t start_row = 0;
    std::uint64_t leading_rank = 0;
    std::uint64_t out_leading_rank = 0;
    std::uint64_t x_plane_offset = 0;
    std::uint64_t w_plane_offset = 0;
    std::uint64_t out_plane_offset = 0;
    std::uint64_t out_dimensions[kLinearMaxLeadingRank]{};
    std::uint64_t x_strides[kLinearMaxLeadingRank]{};
    std::uint64_t out_strides[kLinearMaxLeadingRank]{};
    std::uint32_t bits = 0;
    std::uint32_t data_type = 0;
    std::uint32_t integer_width = 0;
    std::uint32_t head_planar = 0;
};

static_assert(std::is_trivially_copyable_v<LinearMetadata>);

// Exact unsigned width of one integer leaf; zero for every floating leaf and
// for any leaf this port does not implement.
[[nodiscard]] std::uint32_t linear_integer_width(DataType data_type) {
    switch (data_type) {
        case DataType::I2: case DataType::U2: return 2;
        case DataType::I4: case DataType::U4: return 4;
        case DataType::I8: case DataType::U8: return 8;
        case DataType::I16: case DataType::U16: return 16;
        case DataType::I32: case DataType::U32: return 32;
        case DataType::I64: case DataType::U64: return 64;
        default:
            return 0;
    }
}

[[nodiscard]] LinearMetadata build_linear_metadata(
        const DeviceOps::LinearRequest& request) {
    const std::span<const std::size_t> x_dimensions =
            request.x.spec.shape.dimensions();
    const std::span<const std::size_t> w_dimensions =
            request.w.spec.shape.dimensions();
    const std::span<const std::size_t> out_dimensions =
            request.out.spec.shape.dimensions();
    const std::size_t leading_rank = x_dimensions.size() - 2;
    const std::size_t out_leading_rank = out_dimensions.size() - 2;
    if (leading_rank > kLinearMaxLeadingRank
            || out_leading_rank > kLinearMaxLeadingRank) {
        throw std::overflow_error(
                "SYCL linear leading rank exceeds its descriptor");
    }
    LinearMetadata metadata;
    metadata.rows = out_dimensions[out_leading_rank];
    metadata.source_rows = x_dimensions[leading_rank];
    metadata.inner = x_dimensions[leading_rank + 1];
    metadata.outer = w_dimensions[0];
    metadata.out_columns = out_dimensions[out_leading_rank + 1];
    metadata.head_dim = request.layout == LinearOutputLayout::head_planar
            ? request.head_dim : metadata.outer;
    metadata.start_row = request.start_row;
    metadata.leading_rank = leading_rank;
    metadata.out_leading_rank = out_leading_rank;
    metadata.x_plane_offset = request.x.plane_offset;
    metadata.w_plane_offset = request.w.plane_offset;
    metadata.out_plane_offset = request.out.plane_offset;
    metadata.bits = static_cast<std::uint32_t>(
            detail::leaf_bits(request.out.spec.data_type));
    metadata.data_type = static_cast<std::uint32_t>(
            request.out.spec.data_type);
    metadata.integer_width =
            linear_integer_width(request.out.spec.data_type);
    metadata.head_planar =
            request.layout == LinearOutputLayout::head_planar ? 1u : 0u;
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < out_leading_rank; ++axis) {
        metadata.out_dimensions[axis] = out_dimensions[axis];
        metadata.out_strides[axis] = request.out.plane_strides[axis];
        plane_count *= out_dimensions[axis];
    }
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        metadata.x_strides[axis] = request.x.plane_strides[axis];
    }
    metadata.plane_count = plane_count;
    return metadata;
}

// Element slot of (row, column) inside one physical plane of the standard
// 16x16 tiled layout. Device code cannot call `detail::standard_plane_slot`, so
// the same tile-slot mapping is expressed here once for the two matrix axes.
// Precondition: row and column address the logical matrix.
[[nodiscard]] inline std::uint64_t linear_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) noexcept {
    const std::uint64_t tile_rows = (rows + kLinearTile - 1) / kLinearTile;
    const std::uint64_t tile_columns =
            (columns + kLinearTile - 1) / kLinearTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kLinearTile) * tile_columns + column / kLinearTile;
    return tile_index * kLinearTileSlots
            + (row % kLinearTile) * kLinearTile
            + column % kLinearTile;
}

// 32-bit word of the packed plane stream. Little-endian byte assembly keeps the
// access unaligned and byte-granular, exactly like the shared tiled copy/add
// word helpers.
[[nodiscard]] inline std::uint32_t linear_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value = 0;
    for (unsigned int index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(base[word * 4 + index])
                << (8 * index);
    }
    return value;
}

inline void linear_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned int index = 0; index < 4; ++index) {
        base[word * 4 + index] = static_cast<unsigned char>(
                value >> (8 * index));
    }
}

// One packed carrier of `bits` bits at bit offset `bit` of a plane stream.
[[nodiscard]] inline std::uint64_t linear_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned int bits) noexcept {
    const std::uint64_t word = bit / 32;
    std::uint64_t joined = linear_load_word(base, word);
    if (bit % 32 + bits > 32) {
        joined |= static_cast<std::uint64_t>(linear_load_word(base, word + 1))
                << 32;
    }
    return (joined >> (bit % 32))
            & (bits == 64
               ? ~std::uint64_t{}
               : ((std::uint64_t{1} << bits) - 1));
}

// Writes one packed carrier, touching only the bits it owns: padding bits,
// neighbouring carriers, and padding slots keep their observed value. Every
// physical tile row occupies a whole number of 32-bit words for all applicable
// leaf widths, and one work item owns one complete logical output row inside
// one plane, so its carriers never share a word with another writer's carriers.
inline void linear_store_bits(
        unsigned char* base, std::uint64_t bit, unsigned int bits,
        std::uint64_t value) noexcept {
    const std::uint64_t word = bit / 32;
    const unsigned int offset = static_cast<unsigned int>(bit % 32);
    if (offset + bits <= 32) {
        const std::uint32_t mask =
                (bits == 32 ? 0xffffffffu : ((std::uint32_t{1} << bits) - 1))
                << offset;
        const std::uint32_t observed = linear_load_word(base, word);
        linear_store_word(
                base, word,
                (observed & ~mask)
                        | ((static_cast<std::uint32_t>(value) << offset)
                           & mask));
        return;
    }
    const unsigned int low_bits = 32 - offset;
    const std::uint32_t low_mask = 0xffffffffu << offset;
    const std::uint32_t low_observed = linear_load_word(base, word);
    linear_store_word(
            base, word,
            (low_observed & ~low_mask)
                    | ((static_cast<std::uint32_t>(value) << offset)
                       & low_mask));
    const unsigned int high_bits = bits - low_bits;
    const std::uint32_t high_mask = high_bits == 32
            ? 0xffffffffu
            : ((std::uint32_t{1} << high_bits) - 1);
    const std::uint32_t high_observed = linear_load_word(base, word + 1);
    linear_store_word(
            base, word + 1,
            (high_observed & ~high_mask)
                    | (static_cast<std::uint32_t>(value >> low_bits)
                       & high_mask));
}

// Accumulator-domain device traits for the shared named-format codec. The
// `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, and `F32`
// leaves use the FP32 carrier; `F64` keeps the FP64 carrier.
template <typename Accumulator>
struct LinearTraits {
    using carrier_type = Accumulator;

    static carrier_type positive_infinity() noexcept {
        return std::numeric_limits<carrier_type>::infinity();
    }
    static carrier_type quiet_nan() noexcept {
        return std::numeric_limits<carrier_type>::quiet_NaN();
    }
    static carrier_type max_finite() noexcept {
        return std::numeric_limits<carrier_type>::max();
    }
    static bool isnan(carrier_type value) noexcept {
        return sycl::isnan(value);
    }
    static bool isinf(carrier_type value) noexcept {
        return sycl::isinf(value);
    }
    static bool signbit(carrier_type value) noexcept {
        return sycl::signbit(value);
    }
    static carrier_type fabs(carrier_type value) noexcept {
        return sycl::fabs(value);
    }
    static carrier_type floor(carrier_type value) noexcept {
        return sycl::floor(value);
    }
    static carrier_type ldexp(carrier_type value, int exponent) noexcept {
        return sycl::ldexp(value, exponent);
    }
    static carrier_type frexp(carrier_type value, int* exponent) noexcept {
        return sycl::frexp(value, exponent);
    }
};

// One logical output row's fixed mapping: the transformed input plane, the
// transformed output plane, and the weight row group the head coordinate
// selects. The head coordinate is a leading coordinate of the output view, so
// it is resolved here once per row instead of once per element.
struct LinearRowMapping {
    std::uint64_t x_plane = 0;
    std::uint64_t out_plane = 0;
    std::uint64_t weight_row_base = 0;
    std::uint64_t row = 0;
};

[[nodiscard]] inline LinearRowMapping linear_row_mapping(
        const LinearMetadata& metadata,
        std::uint64_t out_plane_row) noexcept {
    LinearRowMapping mapping{
            metadata.x_plane_offset, metadata.out_plane_offset, 0, 0};
    const std::uint64_t plane = out_plane_row / metadata.rows;
    mapping.row = out_plane_row % metadata.rows;
    std::uint64_t rest = plane;
    for (std::uint64_t axis = metadata.out_leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % metadata.out_dimensions[axis];
        rest /= metadata.out_dimensions[axis];
        mapping.out_plane += coordinate * metadata.out_strides[axis];
        if (axis < metadata.leading_rank) {
            mapping.x_plane += coordinate * metadata.x_strides[axis];
            continue;
        }
        // The inserted head-planar axis: its coordinate is already part of the
        // output plane index and selects the weight row group `h*D`. Ordinary
        // mode has no such axis, so the base stays zero.
        mapping.weight_row_base = coordinate * metadata.head_dim;
    }
    return mapping;
}

// One logical output row of an integer leaf: the unsigned modulo-`2^N` dot of
// the selected source row and the named weight row, reduced after every
// multiply and every add, stored as its two's-complement bit pattern.
inline void linear_integer_row(
        const unsigned char* x, const unsigned char* w, unsigned char* out,
        const LinearMetadata& metadata,
        std::uint64_t out_plane_row) noexcept {
    const unsigned int width =
            static_cast<unsigned int>(metadata.integer_width);
    const std::uint64_t mask = width >= 64
            ? ~std::uint64_t{}
            : ((std::uint64_t{1} << width) - 1);
    const LinearRowMapping mapping =
            linear_row_mapping(metadata, out_plane_row);
    const std::uint64_t source_row = metadata.start_row + mapping.row;
    for (std::uint64_t column = 0; column < metadata.out_columns; ++column) {
        const std::uint64_t weight_row = mapping.weight_row_base + column;
        std::uint64_t sum = 0;
        for (std::uint64_t inner = 0; inner < metadata.inner; ++inner) {
            const std::uint64_t x_offset =
                    linear_plane_slot(
                            mapping.x_plane, source_row, inner,
                            metadata.source_rows, metadata.inner)
                    * metadata.bits;
            const std::uint64_t w_offset =
                    linear_plane_slot(
                            metadata.w_plane_offset, weight_row, inner,
                            metadata.outer, metadata.inner)
                    * metadata.bits;
            const std::uint64_t lhs =
                    linear_load_bits(x, x_offset, metadata.bits) & mask;
            const std::uint64_t rhs =
                    linear_load_bits(w, w_offset, metadata.bits) & mask;
            sum = (sum + lhs * rhs) & mask;
        }
        const std::uint64_t out_offset =
                linear_plane_slot(
                        mapping.out_plane, mapping.row, column, metadata.rows,
                        metadata.out_columns)
                * metadata.bits;
        linear_store_bits(out, out_offset, metadata.bits, sum);
    }
}

// One logical output row of a floating leaf: decode every operand carrier on
// device, accumulate from `+0` over increasing `i` with one correctly rounded
// fused multiply-add per step, and encode the accumulated sum exactly once.
template <typename Accumulator>
inline void linear_scalar_row(
        const unsigned char* x, const unsigned char* w, unsigned char* out,
        const LinearMetadata& metadata,
        std::uint64_t out_plane_row) noexcept {
    using Codec =
            detail::scalar_binary_codec_detail::Codec<
                    LinearTraits<Accumulator>>;
    const auto format =
            Codec::format(static_cast<DataType>(metadata.data_type));
    const LinearRowMapping mapping =
            linear_row_mapping(metadata, out_plane_row);
    const std::uint64_t source_row = metadata.start_row + mapping.row;
    for (std::uint64_t column = 0; column < metadata.out_columns; ++column) {
        const std::uint64_t weight_row = mapping.weight_row_base + column;
        Accumulator sum = static_cast<Accumulator>(0);
        for (std::uint64_t inner = 0; inner < metadata.inner; ++inner) {
            const std::uint64_t x_offset =
                    linear_plane_slot(
                            mapping.x_plane, source_row, inner,
                            metadata.source_rows, metadata.inner)
                    * metadata.bits;
            const std::uint64_t w_offset =
                    linear_plane_slot(
                            metadata.w_plane_offset, weight_row, inner,
                            metadata.outer, metadata.inner)
                    * metadata.bits;
            const Accumulator lhs = Codec::decode(
                    linear_load_bits(x, x_offset, metadata.bits), format);
            const Accumulator rhs = Codec::decode(
                    linear_load_bits(w, w_offset, metadata.bits), format);
            sum = sycl::fma(lhs, rhs, sum);
        }
        const std::uint64_t out_offset =
                linear_plane_slot(
                        mapping.out_plane, mapping.row, column, metadata.rows,
                        metadata.out_columns)
                * metadata.bits;
        linear_store_bits(
                out, out_offset, metadata.bits, Codec::encode(sum, format));
    }
}

[[nodiscard]] inline std::size_t linear_row_items(
        const LinearMetadata& metadata) noexcept {
    return static_cast<std::size_t>(
            metadata.plane_count * metadata.rows);
}

template <typename Accumulator>
[[nodiscard]] sycl::event launch_linear_scalar_rows(
        sycl::queue& queue, const unsigned char* x, const unsigned char* w,
        unsigned char* out, const LinearMetadata& metadata) {
    return queue.parallel_for(
            sycl::range<1>(linear_row_items(metadata)),
            [=](sycl::id<1> item) {
                linear_scalar_row<Accumulator>(
                        x, w, out, metadata, item[0]);
            });
}

// The one native launch of the operation: one work item per logical output row
// on the existing in-order queue, with the integer, FP32, and FP64 recurrences
// instantiated as separate kernels so no device path ever widens `F64` or
// borrows the integer dot.
[[nodiscard]] sycl::event launch_linear(
        sycl::queue& queue, const DeviceOps::LinearRequest& request,
        const LinearMetadata& metadata) {
    const auto* x = static_cast<const unsigned char*>(
            request.x.native_handle);
    const auto* w = static_cast<const unsigned char*>(
            request.w.native_handle);
    auto* out = static_cast<unsigned char*>(request.out.native_handle);
    if (metadata.integer_width != 0) {
        return queue.parallel_for(
                sycl::range<1>(linear_row_items(metadata)),
                [=](sycl::id<1> item) {
                    linear_integer_row(x, w, out, metadata, item[0]);
                });
    }
    if (request.out.spec.data_type == DataType::F64) {
        return launch_linear_scalar_rows<double>(queue, x, w, out, metadata);
    }
    return launch_linear_scalar_rows<float>(queue, x, w, out, metadata);
}

// ---------------------------------------------------------------------------
// The experimental device-native BF16 specialization.
//
// `BF16` is the one linear leaf a subgroup-16 `joint_matrix` BF16/BF16/FP32
// specialization owns. Admission has already validated the operands, so this
// path only snapshots the request into a bounded trivially copyable descriptor,
// assembles one `{M}x16` A tile and one `16x16` B tile per inner tile in
// subgroup-local memory with the logical extents as the only read bounds,
// accumulates the product in FP32 across every inner tile with
// `joint_matrix_mad`, stores the FP32 accumulator into the caller-owned product
// scratch, and then performs exactly one round-to-nearest-even BF16 pack and
// scatter per logical output element into the requested layout. No operand is
// relocated, transposed, staged on the host, or recomputed by a scalar,
// elementwise, host, or library substitute, and no allocation happens here: the
// only device work is the matrix MAD and the single final encode.
// ---------------------------------------------------------------------------

// The native `BF16` dispatch consumes a `16x16` physical tile at a time and
// requires exactly one subgroup per work-group, so the required facility is a
// subgroup-16 `joint_matrix` BF16/BF16/FP32 combination for both M shapes the
// row decomposition emits: a full `16x16x16` block and a single-row `1x16x16`
// tail. The FP32 map layout is the frozen `P*pad16(R)*pad16(O)*4` product
// scratch, one row-major padded matrix per logical leading plane.
inline constexpr std::uint64_t kBf16Subgroup = 16;
inline constexpr std::uint64_t kBf16Bits = 16;

[[nodiscard]] inline std::uint64_t bf16_pad16(
        std::uint64_t value, const char* what) {
    const std::uint64_t remainder = value % kLinearTile;
    if (remainder == 0) {
        return value;
    }
    return detail::checked_add(
            value, kLinearTile - remainder, what);
}

[[nodiscard]] inline std::uint64_t bf16_tiles(
        std::uint64_t value, const char* what) {
    const std::uint64_t remainder = value % kLinearTile;
    const std::uint64_t rounded = detail::checked_add(
            value, remainder == 0 ? 0 : kLinearTile - remainder, what);
    return rounded / kLinearTile;
}

#if defined(IOM_SYCL_BF16_MATRIX)
namespace matrix = sycl::ext::oneapi::experimental::matrix;
using Bf16 = sycl::ext::oneapi::bfloat16;

// Whether one queried combination covers the exact `M x 16 x 16`
// BF16/BF16/FP32 shape family this port queues. An exact `msize` names that
// single shape; a zero `msize` means every size up to `max_msize` is covered.
[[nodiscard]] bool bf16_combination_covers(
        const std::vector<matrix::combination>& combinations,
        std::uint64_t rows) {
    for (const matrix::combination& combination : combinations) {
        if (combination.atype != matrix::matrix_type::bf16
                || combination.btype != matrix::matrix_type::bf16
                || combination.ctype != matrix::matrix_type::fp32
                || combination.dtype != matrix::matrix_type::fp32) {
            continue;
        }
        const bool rows_covered = combination.msize == 0
                ? rows <= combination.max_msize
                : combination.msize == rows;
        const bool columns_covered = combination.nsize == 0
                ? kLinearTile <= combination.max_nsize
                : combination.nsize == kLinearTile;
        const bool inner_covered = combination.ksize == 0
                ? kLinearTile <= combination.max_ksize
                : combination.ksize == kLinearTile;
        if (rows_covered && columns_covered && inner_covered) {
            return true;
        }
    }
    return false;
}
#endif  // defined(IOM_SYCL_BF16_MATRIX)

// Immutable bounded snapshot of one admitted `BF16` request: the logical
// request shape, the output mode, the checked leading-plane product, every
// transformed operand plane stride and offset, the standard tiled geometry of
// the three operands, and the padded FP32 product geometry. It is captured by
// value into the kernels, so a task retains no borrowed view and the scratch
// contract stays exactly the frozen formula.
struct Bf16LinearMetadata {
    std::uint64_t planes = 0;
    std::uint64_t out_planes = 0;
    std::uint64_t rows = 0;
    std::uint64_t source_rows = 0;
    std::uint64_t inner = 0;
    std::uint64_t product_columns = 0;
    std::uint64_t out_columns = 0;
    std::uint64_t head_dim = 0;
    std::uint64_t start_row = 0;
    std::uint64_t leading_rank = 0;
    std::uint64_t out_leading_rank = 0;
    std::uint64_t pack_items = 0;
    std::uint64_t pack_segments = 0;
    std::uint64_t x_plane_offset = 0;
    std::uint64_t w_plane_offset = 0;
    std::uint64_t out_plane_offset = 0;
    std::uint64_t x_dimensions[kLinearMaxLeadingRank]{};
    std::uint64_t x_plane_weights[kLinearMaxLeadingRank]{};
    std::uint64_t x_strides[kLinearMaxLeadingRank]{};
    std::uint64_t out_dimensions[kLinearMaxLeadingRank]{};
    std::uint64_t out_strides[kLinearMaxLeadingRank]{};
    std::uint64_t x_tile_rows = 0;
    std::uint64_t x_tile_columns = 0;
    std::uint64_t w_tile_rows = 0;
    std::uint64_t w_tile_columns = 0;
    std::uint64_t padded_rows = 0;
    std::uint64_t padded_columns = 0;
    std::uint64_t inner_blocks = 0;
    std::uint64_t column_blocks = 0;
    std::uint64_t full_row_blocks = 0;
    std::uint64_t tail_rows = 0;
};

static_assert(std::is_trivially_copyable_v<Bf16LinearMetadata>);

// Absolute device plane of the input view for one logical leading-plane index
// of the row-major logical plane tuple.
[[nodiscard]] inline std::uint64_t bf16_x_plane(
        const Bf16LinearMetadata& metadata, std::uint64_t plane) noexcept {
    std::uint64_t result = metadata.x_plane_offset;
    for (std::uint64_t axis = metadata.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate = plane % metadata.x_dimensions[axis];
        plane /= metadata.x_dimensions[axis];
        result += coordinate * metadata.x_strides[axis];
    }
    return result;
}

// One output element's complete mapping: its absolute output plane, the
// logical product-scratch plane it reads, and its column inside that padded
// FP32 product matrix. The inserted head-planar axis selects the weight row
// group `h*D` inside the product and never an input plane.
struct Bf16OutputMapping {
    std::uint64_t out_plane = 0;
    std::uint64_t scratch_plane = 0;
    std::uint64_t product_column = 0;
};

[[nodiscard]] inline Bf16OutputMapping bf16_output_mapping(
        const Bf16LinearMetadata& metadata, std::uint64_t plane,
        std::uint64_t column) noexcept {
    Bf16OutputMapping mapping{metadata.out_plane_offset, 0, column};
    std::uint64_t rest = plane;
    for (std::uint64_t axis = metadata.out_leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate = rest % metadata.out_dimensions[axis];
        rest /= metadata.out_dimensions[axis];
        mapping.out_plane += coordinate * metadata.out_strides[axis];
        if (axis < metadata.leading_rank) {
            mapping.scratch_plane +=
                    coordinate * metadata.x_plane_weights[axis];
            continue;
        }
        mapping.product_column = coordinate * metadata.head_dim + column;
    }
    return mapping;
}

// The one BF16 encode of the accumulated FP32 sum: round-to-nearest-even from
// the FP32 bit pattern, so infinity, NaN, saturation at the BF16 range, and
// signed zero keep the existing named-format rules and no partial result is
// ever rounded through BF16.
[[nodiscard]] inline std::uint16_t bf16_rne_bits(float value) noexcept {
    const std::uint32_t bits = sycl::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

// One BF16 carrier of the packed tiled stream, reinterpreted bit-exactly.
[[nodiscard]] inline Bf16 bf16_load_bits(
        const unsigned char* base, std::uint64_t bit) noexcept {
    return sycl::bit_cast<Bf16>(
            static_cast<std::uint16_t>(linear_load_bits(base, bit, kBf16Bits)));
}

#if defined(IOM_SYCL_BF16_MATRIX)
// Kernel symbol of the M-row product tile kernel. The body is a plain function
// so the local-memory staging geometry stays explicit and the kernel keeps one
// stable, readable name in the runtime's profiler trace. The row-window tail
// carries its own kernel name because it is a distinct submitted kernel even
// where it shares the `M=16` tile shape.
template <std::uint64_t M>
struct LinearBf16MadKernel {};
template <std::uint64_t M>
struct LinearBf16TailKernel {};

// One work-group per `(logical plane, row block, product column block)`, one
// subgroup of exactly sixteen work items, and one `M x 16 x 16`
// BF16/BF16/FP32 MAD per inner tile. Each lane assembles one A row and one B
// row of the current inner tile from the operand's own transformed plane
// mapping, writing zero outside the logical extents, so no padded or
// uninitialized carrier is ever read and no logical result depends on physical
// tile padding. The FP32 accumulator is initialized once to `+0` and retained
// across every inner tile until the single store into the caller-owned product
// scratch.
template <std::uint64_t M>
inline void linear_bf16_mad_body(
        sycl::nd_item<1> item, const unsigned char* x,
        const unsigned char* w, unsigned char* scratch,
        const Bf16LinearMetadata& metadata, std::uint64_t row_blocks,
        std::uint64_t first_row_block, Bf16* a_local) {
        auto subgroup = item.get_sub_group();
        Bf16* const b_local = a_local + kLinearTileSlots;
        const std::uint64_t lane = item.get_local_id(0);
        const std::uint64_t group = item.get_group_linear_id();
        const std::uint64_t group_per_plane =
                row_blocks * metadata.column_blocks;
        const std::uint64_t plane_index = group / group_per_plane;
        const std::uint64_t within = group % group_per_plane;
        const std::uint64_t row_block =
                first_row_block + within / metadata.column_blocks;
        const std::uint64_t column_block = within % metadata.column_blocks;
        const std::uint64_t x_plane = bf16_x_plane(metadata, plane_index);
        const std::uint64_t m_index = row_block * kLinearTile + lane;
        const std::uint64_t source_row = metadata.start_row + m_index;
        const std::uint64_t n_index = column_block * kLinearTile + lane;
        const bool a_row_valid =
                lane < M && m_index < metadata.rows;
        const bool b_row_valid = n_index < metadata.product_columns;
        const std::uint64_t x_row_base =
                (x_plane * metadata.x_tile_rows + source_row / kLinearTile)
                * metadata.x_tile_columns;
        const std::uint64_t x_row_slot =
                (source_row % kLinearTile) * kLinearTile;
        const std::uint64_t w_row_base =
                (metadata.w_plane_offset * metadata.w_tile_rows
                 + n_index / kLinearTile)
                * metadata.w_tile_columns;
        const std::uint64_t w_row_slot =
                (n_index % kLinearTile) * kLinearTile;

        auto accumulator = matrix::joint_matrix<
                sycl::sub_group, float, matrix::use::accumulator, M,
                kLinearTile>();
        matrix::joint_matrix_fill(subgroup, accumulator, 0.0f);
        auto a_fragment = matrix::joint_matrix<
                sycl::sub_group, Bf16, matrix::use::a, M, kLinearTile,
                matrix::layout::row_major>();
        auto b_fragment = matrix::joint_matrix<
                sycl::sub_group, Bf16, matrix::use::b, kLinearTile,
                kLinearTile, matrix::layout::col_major>();
        for (std::uint64_t inner_block = 0;
             inner_block < metadata.inner_blocks; ++inner_block) {
            const std::uint64_t inner_base = inner_block * kLinearTile;
            const std::uint64_t legal_inner = metadata.inner - inner_base;
            for (std::uint64_t k = 0; k < kLinearTile; ++k) {
                const bool carrier = k < legal_inner;
                a_local[lane * kLinearTile + k] =
                        a_row_valid && carrier
                        ? bf16_load_bits(
                                  x,
                                  (x_row_base + inner_block) * kLinearTileSlots
                                                  * kBf16Bits
                                          + (x_row_slot + k) * kBf16Bits)
                        : sycl::bit_cast<Bf16>(static_cast<std::uint16_t>(0));
                b_local[lane * kLinearTile + k] =
                        b_row_valid && carrier
                        ? bf16_load_bits(
                                  w,
                                  (w_row_base + inner_block) * kLinearTileSlots
                                                  * kBf16Bits
                                          + (w_row_slot + k) * kBf16Bits)
                        : sycl::bit_cast<Bf16>(static_cast<std::uint16_t>(0));
            }
            sycl::group_barrier(item.get_group());
            matrix::joint_matrix_load(
                    subgroup, a_fragment,
                    sycl::multi_ptr<Bf16, sycl::access::address_space::local_space>(
                            a_local),
                    kLinearTile);
            matrix::joint_matrix_load(
                    subgroup, b_fragment,
                    sycl::multi_ptr<Bf16, sycl::access::address_space::local_space>(
                            b_local),
                    kLinearTile);
            matrix::joint_matrix_mad(
                    subgroup, accumulator, a_fragment, b_fragment,
                    accumulator);
            sycl::group_barrier(item.get_group());
        }
        auto* const tile = reinterpret_cast<float*>(
                scratch
                + ((plane_index * metadata.padded_rows
                    + row_block * kLinearTile)
                   * metadata.padded_columns
                   + column_block * kLinearTile)
                        * sizeof(float));
        matrix::joint_matrix_store(
                subgroup, accumulator,
                sycl::multi_ptr<float, sycl::access::address_space::global_space>(
                        tile),
                static_cast<std::size_t>(metadata.padded_columns),
                matrix::layout::row_major);
}

// Kernel symbol of the single RNE pack and scatter.
struct LinearBf16PackKernel {};

// One work item per logical output tile row: it owns the complete sixteen-slot
// physical row of one plane, so every 32-bit word of the packed output has
// exactly one writer and the read-modify-write of a packed 16-bit carrier never
// competes with a neighbouring carrier. It reads each logical element's FP32
// product from the caller-owned scratch and writes exactly one BF16 carrier
// into the requested output layout at that row's own transformed plane; padding
// slots and bits outside its logical columns keep their observed value.
inline void linear_bf16_pack_body(
        sycl::id<1> item, const unsigned char* scratch, unsigned char* out,
        const Bf16LinearMetadata& metadata) {
        const std::uint64_t index = item[0];
        const std::uint64_t per_plane =
                metadata.rows * metadata.pack_segments;
        const std::uint64_t plane_index = index / per_plane;
        const std::uint64_t within = index % per_plane;
        const std::uint64_t row = within / metadata.pack_segments;
        const std::uint64_t first_column =
                (within % metadata.pack_segments) * kLinearTile;
        const std::uint64_t last_column =
                first_column + kLinearTile < metadata.out_columns
                ? first_column + kLinearTile
                : metadata.out_columns;
        const Bf16OutputMapping mapping =
                bf16_output_mapping(metadata, plane_index, first_column);
        for (std::uint64_t column = first_column; column < last_column;
             ++column) {
            const std::uint64_t scratch_cell =
                    (mapping.scratch_plane * metadata.padded_rows + row)
                            * metadata.padded_columns
                    + mapping.product_column
                    + (column - first_column);
            const float value = sycl::bit_cast<float>(
                    linear_load_word(scratch, scratch_cell));
            const std::uint64_t slot = linear_plane_slot(
                    mapping.out_plane, row, column, metadata.rows,
                    metadata.out_columns);
            linear_store_bits(
                    out, slot * kBf16Bits, kBf16Bits, bf16_rne_bits(value));
        }
}
#endif  // defined(IOM_SYCL_BF16_MATRIX)

// The checked `A32(P*pad16(R)*pad16(O)*4)` product-scratch requirement of the
// `BF16` leaf: `P` is the checked product of the input view's logical leading
// extents, `pad16` the checked round-up to sixteen, and `A32` the checked
// round-up to the reported 32-byte alignment. It reads only the two admitted
// view specifications, so the pure requirement query allocates no request,
// snapshot, or vector.
[[nodiscard]] std::size_t bf16_linear_scratch_bytes(
        const TensorView& x, const TensorView& w, std::size_t rows) {
    const std::span<const std::size_t> x_dimensions =
            x.spec().shape.dimensions();
    const std::span<const std::size_t> w_dimensions =
            w.spec().shape.dimensions();
    const std::size_t leading_rank = x_dimensions.size() - 2;
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        planes = detail::checked_mul(
                planes, x_dimensions[axis],
                "SYCL BF16 linear scratch plane count overflows");
    }
    const std::size_t padded_rows = static_cast<std::size_t>(bf16_pad16(
            rows, "SYCL BF16 linear scratch row padding overflows"));
    const std::size_t padded_columns = static_cast<std::size_t>(bf16_pad16(
            w_dimensions[0],
            "SYCL BF16 linear scratch column padding overflows"));
    std::size_t bytes = detail::checked_mul(
            planes, padded_rows,
            "SYCL BF16 linear scratch element count overflows");
    bytes = detail::checked_mul(
            bytes, padded_columns,
            "SYCL BF16 linear scratch element count overflows");
    bytes = detail::checked_mul(
            bytes, sizeof(float),
            "SYCL BF16 linear scratch byte count overflows");
    const std::size_t remainder = bytes % 32;
    if (remainder == 0) {
        return bytes;
    }
    return detail::checked_add(
            bytes, 32 - remainder,
            "SYCL BF16 linear scratch alignment overflows");
}

// Bounded checked snapshot of one admitted `BF16` request. Every derived
// extent, tile count, block count, plane count, element count, and product is
// checked, so no address, index, or scratch offset below can overflow after
// acceptance.
[[nodiscard]] Bf16LinearMetadata build_bf16_linear_metadata(
        const DeviceOps::LinearRequest& request) {
    const std::span<const std::size_t> x_dimensions =
            request.x.spec.shape.dimensions();
    const std::span<const std::size_t> w_dimensions =
            request.w.spec.shape.dimensions();
    const std::span<const std::size_t> out_dimensions =
            request.out.spec.shape.dimensions();
    const std::size_t leading_rank = x_dimensions.size() - 2;
    const std::size_t out_leading_rank = out_dimensions.size() - 2;
    if (leading_rank > kLinearMaxLeadingRank
            || out_leading_rank > kLinearMaxLeadingRank) {
        throw std::overflow_error(
                "SYCL BF16 linear leading rank exceeds its descriptor");
    }
    const bool head_planar =
            request.layout == LinearOutputLayout::head_planar;
    Bf16LinearMetadata metadata;
    metadata.rows = out_dimensions[out_leading_rank];
    metadata.source_rows = x_dimensions[leading_rank];
    metadata.inner = x_dimensions[leading_rank + 1];
    metadata.product_columns = w_dimensions[0];
    metadata.out_columns = out_dimensions[out_leading_rank + 1];
    metadata.head_dim = head_planar ? request.head_dim
                                    : metadata.product_columns;
    metadata.start_row = request.start_row;
    metadata.leading_rank = leading_rank;
    metadata.out_leading_rank = out_leading_rank;
    metadata.x_plane_offset = request.x.plane_offset;
    metadata.w_plane_offset = request.w.plane_offset;
    metadata.out_plane_offset = request.out.plane_offset;
    std::uint64_t planes = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        metadata.x_dimensions[axis] =
                static_cast<std::uint64_t>(x_dimensions[axis]);
        metadata.x_strides[axis] =
                static_cast<std::uint64_t>(request.x.plane_strides[axis]);
        planes = detail::checked_mul(
                planes, x_dimensions[axis],
                "SYCL BF16 linear plane count overflows");
    }
    std::uint64_t plane_weight = 1;
    for (std::uint64_t axis = leading_rank; axis-- > 0;) {
        metadata.x_plane_weights[axis] = plane_weight;
        plane_weight = detail::checked_mul(
                plane_weight, metadata.x_dimensions[axis],
                "SYCL BF16 linear plane weight overflows");
    }
    for (std::size_t axis = 0; axis < out_leading_rank; ++axis) {
        metadata.out_dimensions[axis] =
                static_cast<std::uint64_t>(out_dimensions[axis]);
        metadata.out_strides[axis] =
                static_cast<std::uint64_t>(request.out.plane_strides[axis]);
    }
    metadata.planes = planes;
    metadata.out_planes = detail::checked_mul(
            planes, head_planar ? request.heads : 1,
            "SYCL BF16 linear output plane count overflows");
    // The pack grid is one work item per logical output tile row, which is the
    // physical write unit whose 32-bit words one item exclusively owns.
    metadata.pack_segments = bf16_tiles(
            metadata.out_columns,
            "SYCL BF16 linear output segment count overflows");
    metadata.pack_items = detail::checked_mul(
            detail::checked_mul(
                    metadata.out_planes, metadata.rows,
                    "SYCL BF16 linear pack item count overflows"),
            metadata.pack_segments,
            "SYCL BF16 linear pack item count overflows");
    metadata.padded_rows = bf16_pad16(
            metadata.rows, "SYCL BF16 linear scratch row padding overflows");
    metadata.padded_columns = bf16_pad16(
            metadata.product_columns,
            "SYCL BF16 linear scratch column padding overflows");
    metadata.x_tile_rows = bf16_tiles(
            metadata.source_rows,
            "SYCL BF16 linear input tile count overflows");
    metadata.x_tile_columns = bf16_tiles(
            metadata.inner, "SYCL BF16 linear input tile count overflows");
    metadata.w_tile_rows = bf16_tiles(
            metadata.product_columns,
            "SYCL BF16 linear weight tile count overflows");
    metadata.w_tile_columns = metadata.x_tile_columns;
    metadata.inner_blocks = metadata.x_tile_columns;
    metadata.column_blocks = bf16_tiles(
            metadata.product_columns,
            "SYCL BF16 linear product block count overflows");
    metadata.full_row_blocks = metadata.rows / kLinearTile;
    metadata.tail_rows = metadata.rows % kLinearTile;
    return metadata;
}

// Floor division of the leading product, used only by the launch geometry.
[[nodiscard]] std::size_t bf16_linear_group_count(
        const Bf16LinearMetadata& metadata, std::uint64_t row_blocks) {
    const std::size_t per_plane = static_cast<std::size_t>(
            row_blocks * metadata.column_blocks);
    const std::size_t groups = detail::checked_mul(
            static_cast<std::size_t>(metadata.planes), per_plane,
            "SYCL BF16 linear work-group count overflows");
    return groups;
}

// The three in-order submissions of one `BF16` request: the full M=16 row
// blocks, the row tail (an `1x16x16` M=1 tile when the window leaves exactly
// one row, otherwise a masked M=16 tile), and the single RNE pack and scatter.
// The caller-owned product scratch is written by the MAD kernels and read by
// the pack kernel only, and every submission rides the queue's one in-order
// stream, so completion of the last event proves the whole operation.
[[nodiscard]] sycl::event launch_linear_bf16(
        sycl::queue& queue, const DeviceOps::LinearRequest& request,
        const Bf16LinearMetadata& metadata) {
#if !defined(IOM_SYCL_BF16_MATRIX)
    (void)queue;
    (void)request;
    (void)metadata;
    throw detail::UnsupportedOperation();
#else
    const auto* x = static_cast<const unsigned char*>(
            request.x.native_handle);
    const auto* w = static_cast<const unsigned char*>(
            request.w.native_handle);
    auto* out = static_cast<unsigned char*>(request.out.native_handle);
    auto* scratch = static_cast<unsigned char*>(
            detail::WorkspaceValidation::address(request.workspace));
    if (scratch == nullptr || request.workspace.byte_size() == 0) {
        throw std::invalid_argument(
                "SYCL BF16 linear requires its product scratch");
    }
    sycl::event event;
    if (metadata.full_row_blocks != 0) {
        const std::size_t groups =
                bf16_linear_group_count(metadata, metadata.full_row_blocks);
        const std::uint64_t row_blocks = metadata.full_row_blocks;
        event = queue.submit([&](sycl::handler& handler) {
            sycl::local_accessor<Bf16, 1> stage{
                    sycl::range<1>(2 * kLinearTileSlots), handler};
            handler.parallel_for<LinearBf16MadKernel<kLinearTile>>(
                    sycl::nd_range<1>(
                            sycl::range<1>(groups * kBf16Subgroup),
                            sycl::range<1>(kBf16Subgroup)),
                    [=](sycl::nd_item<1> item)
                            [[sycl::reqd_sub_group_size(
                                    kBf16Subgroup)]] {
                                linear_bf16_mad_body<kLinearTile>(
                                        item, x, w, scratch, metadata,
                                        row_blocks, 0,
                                        stage.get_multi_ptr<
                                                        sycl::access::decorated::yes>()
                                                .get());
                            });
        });
    }
    if (metadata.tail_rows != 0) {
        const std::size_t groups = bf16_linear_group_count(metadata, 1);
        const std::uint64_t first_row_block = metadata.full_row_blocks;
        if (metadata.tail_rows == 1) {
            event = queue.submit([&](sycl::handler& handler) {
                sycl::local_accessor<Bf16, 1> stage{
                        sycl::range<1>(2 * kLinearTileSlots), handler};
                handler.parallel_for<LinearBf16TailKernel<1>>(
                        sycl::nd_range<1>(
                                sycl::range<1>(groups * kBf16Subgroup),
                                sycl::range<1>(kBf16Subgroup)),
                        [=](sycl::nd_item<1> item)
                                [[sycl::reqd_sub_group_size(
                                        kBf16Subgroup)]] {
                                    linear_bf16_mad_body<1>(
                                            item, x, w, scratch, metadata, 1,
                                            first_row_block,
                                            stage.get_multi_ptr<
                                                            sycl::access::decorated::yes>()
                                                    .get());
                                });
            });
        } else {
            event = queue.submit([&](sycl::handler& handler) {
                sycl::local_accessor<Bf16, 1> stage{
                        sycl::range<1>(2 * kLinearTileSlots), handler};
                handler.parallel_for<LinearBf16TailKernel<kLinearTile>>(
                        sycl::nd_range<1>(
                                sycl::range<1>(groups * kBf16Subgroup),
                                sycl::range<1>(kBf16Subgroup)),
                        [=](sycl::nd_item<1> item)
                                [[sycl::reqd_sub_group_size(
                                        kBf16Subgroup)]] {
                                    linear_bf16_mad_body<kLinearTile>(
                                            item, x, w, scratch, metadata, 1,
                                            first_row_block,
                                            stage.get_multi_ptr<
                                                            sycl::access::decorated::yes>()
                                                    .get());
                                });
            });
        }
    }
    event = queue.parallel_for<LinearBf16PackKernel>(
            sycl::range<1>(static_cast<std::size_t>(metadata.pack_items)),
            [=](sycl::id<1> item) {
                linear_bf16_pack_body(item, scratch, out, metadata);
            });
    return event;
#endif  // defined(IOM_SYCL_BF16_MATRIX)
}

}  // namespace

// Immutable device capability of the native `BF16` specialization: the device
// reports the Intel matrix extension, enumerates subgroup 16, and queries a
// BF16/BF16/FP32 combination covering both queued M shapes. The predicate
// inspects runtime device facts only — never the compilation or a bounded
// sample — and an absent extension revision, aspect, subgroup, or combination
// is a device fact reported as `Unsupported` before any queue effect.
bool bf16_linear_device_capable(const sycl::device& device) noexcept {
#if defined(IOM_SYCL_BF16_MATRIX)
    try {
        if (!device.has(sycl::aspect::ext_intel_matrix)) {
            return false;
        }
        const std::vector<std::size_t> subgroup_sizes =
                device.get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(
                    subgroup_sizes.begin(), subgroup_sizes.end(),
                    static_cast<std::size_t>(kBf16Subgroup))
                == subgroup_sizes.end()) {
            return false;
        }
        const std::vector<matrix::combination> combinations = device.get_info<
                sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();
        return bf16_combination_covers(combinations, kLinearTile)
                && bf16_combination_covers(combinations, 1);
    } catch (...) {
        // A device that cannot answer the query has not proven the facility.
        return false;
    }
#else
    (void)device;
    return false;
#endif  // defined(IOM_SYCL_BF16_MATRIX)
}

bool SyclQueue::linear_leaf_supported(DataType data_type) const noexcept {
    switch (data_type) {
        case DataType::I2: case DataType::U2:
        case DataType::I4: case DataType::U4:
        case DataType::I8: case DataType::U8:
        case DataType::I16: case DataType::U16:
        case DataType::I32: case DataType::U32:
        case DataType::I64: case DataType::U64:
        case DataType::F4_E2M1:
        case DataType::F6_E2M3: case DataType::F6_E3M2:
        case DataType::F8_E4M3FN: case DataType::F8_E5M2:
        case DataType::F16:
        case DataType::F32:
            return true;
        // The FP64 leaf keeps every product and every accumulation step in
        // FP64, so it is queueable exactly on a device that reports the FP64
        // aspect; it is never widened from FP32 or emulated here.
        case DataType::F64:
            return fp64_supported_;
        // Native BF16 is the one leaf the separate subgroup-16 `joint_matrix`
        // specialization owns, so it is deliberately not a scalar leaf here;
        // its capability is `bf16_linear_supported_` and the two recognized
        // inapplicable leaves have no linear semantics on any backend.
        case DataType::BF16:
        case DataType::BOOL:
        case DataType::F8_E8M0:
            return false;
    }
    return false;
}

WorkspaceRequirements SyclQueue::linear_workspace_requirements_impl(
        const TensorView& x, const TensorView& w, const TensorView& out,
        std::size_t s, std::size_t R, LinearOutputLayout layout, std::size_t H,
        std::size_t D) {
    (void)out;
    (void)s;
    (void)layout;
    (void)H;
    (void)D;
    // Pure capability decision: the query allocates nothing, constructs no
    // snapshot, registers or leases no owner, and mutates no queue state. A
    // leaf this port does not implement — `F64` on a device without the FP64
    // aspect, or `BF16` on a device without the queried subgroup-16
    // BF16/BF16/FP32 matrix facility — is a capability rejection before any
    // queue effect.
    if (x.spec().data_type == DataType::BF16) {
        if (!bf16_linear_supported_) {
            throw detail::UnsupportedOperation();
        }
        // The `BF16` specialization is the one positive linear requirement:
        // the exact checked `A32(P*pad16(R)*pad16(O)*4)` product scratch at
        // alignment 32, derived from the same views the submission consumes.
        return WorkspaceRequirements{bf16_linear_scratch_bytes(x, w, R), 32};
    }
    if (!linear_leaf_supported(x.spec().data_type)) {
        throw detail::UnsupportedOperation();
    }
    // The scalar leaves consume no raw workspace: the exact contract
    // requirement is the `{0, 1}` zero-scratch path.
    return WorkspaceRequirements{0, 1};
}

oid SyclQueue::linear_impl(const LinearRequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    if (consume_submission_fault(SubmissionFault::state_allocation)) {
        throw std::bad_alloc();
    }
    auto state = std::make_shared<SyclFenceState>();
    if (consume_submission_fault(SubmissionFault::fence_construction)) {
        throw std::bad_alloc();
    }
    detail::Fence fence = build_sycl_fence(state);
    return submit_linear(
            request, *state_, registry_queue_id_, fence,
            [this, state](
                    std::uint64_t sequence,
                    const LinearRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                Task task;
                task.sequence = sequence;
                task.state = state;
                task.fence = state.get();
                task.linear_request.emplace(captured);
                task.linear_entries = entries;
                worker_.submit_copy(std::move(task));
            });
}

void SyclQueue::execute_linear(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    {
        const auto [it, inserted] = outcomes_.emplace(
                task.sequence,
                SyclSequenceOutcome{
                        detail::SequenceOutcome{}, task.state, std::nullopt,
                        task.linear_request->workspace_lease, std::nullopt,
                        false, task.linear_entries});
        if (!inserted) {
            throw std::logic_error(
                    "duplicate SYCL outstanding-work sequence");
        }
    }

    bool native_attempted = false;
    try {
        const auto completion_slot = completion_pool_->try_acquire();
        if (!completion_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        task.state->set_completion_slot(
                *completion_pool_, *completion_slot);
        // Exactly one dispatch per admitted leaf: the `BF16` specialization
        // owns its own bounded snapshot and its three in-order submissions,
        // and the twenty scalar leaves keep the raw-word recurrence.
        const bool native_bf16 =
                task.linear_request->out.spec.data_type == DataType::BF16;
        const LinearMetadata metadata = native_bf16
                ? LinearMetadata{}
                : build_linear_metadata(*task.linear_request);
        const Bf16LinearMetadata bf16_metadata = native_bf16
                ? build_bf16_linear_metadata(*task.linear_request)
                : Bf16LinearMetadata{};
        if (consume_submission_fault(SubmissionFault::first_submit)) {
            throw std::runtime_error(
                    "injected SYCL first-submit failure");
        }
        // Once the native enqueue is attempted, the accepted outcome is
        // retained even if the enqueue throws, because the runtime may have
        // submitted work before reporting the error and only a successful
        // drain proves completion.
        native_attempted = true;
        sycl::event event = native_bf16
                ? launch_linear_bf16(
                          queue_, *task.linear_request, bf16_metadata)
                : launch_linear(queue_, *task.linear_request, metadata);
        if (launch_calls.kernel_launched != nullptr) {
            launch_calls.kernel_launched();
        }
        task.state->set_event(std::move(event));
        if (consume_submission_fault(SubmissionFault::post_launch)) {
            throw std::runtime_error(
                    "injected SYCL post-launch failure");
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        if (!native_attempted) {
            task.state->mark_completion_proven();
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                outcomes_.erase(task.sequence);
            }
            task.fence = nullptr;
            task.state.reset();
            throw;
        }
        task.state->set_failure(failure);
    }
}

}  // namespace iom::sycl_detail
