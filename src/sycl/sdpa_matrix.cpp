#include "sdpa_matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "../iom_internal.hpp"

#if defined(SYCL_EXT_ONEAPI_MATRIX) && SYCL_EXT_ONEAPI_MATRIX == 1
#define IOM_SYCL_BF16_MATRIX 1
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#endif

namespace iom::sycl_detail {
namespace {

constexpr std::uint64_t kTile = kSdpaMatrixTile;
constexpr std::uint64_t kTileSlots = kTile * kTile;
constexpr std::uint64_t kBf16Bits = 16;
constexpr std::size_t kScratchAlignment = 32;

[[nodiscard]] std::size_t checked_add(
        std::size_t lhs, std::size_t rhs, const char* what) {
    return detail::checked_add(lhs, rhs, what);
}

[[nodiscard]] std::size_t checked_mul(
        std::size_t lhs, std::size_t rhs, const char* what) {
    return detail::checked_mul(lhs, rhs, what);
}

[[nodiscard]] std::size_t align_up(
        std::size_t value, std::size_t alignment, const char* what) {
    if (alignment == 0) {
        throw std::invalid_argument("SYCL SDPA alignment is zero");
    }
    const std::size_t remainder = value % alignment;
    return remainder == 0
            ? value
            : checked_add(value, alignment - remainder, what);
}

[[nodiscard]] std::size_t padded(
        std::size_t value, const char* what) {
    return detail::padded_extent(value, what);
}


[[nodiscard]] std::uint64_t checked_u64_mul(
        std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(what);
    }
    return lhs * rhs;
}

[[nodiscard]] std::uint64_t checked_u64_add(
        std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(what);
    }
    return lhs + rhs;
}

[[nodiscard]] std::uint64_t round_up_tile(
        std::uint64_t value, const char* what) {
    const std::uint64_t remainder = value % kTile;
    return remainder == 0
            ? value
            : checked_u64_add(value, kTile - remainder, what);
}

// This is the device-side counterpart of detail::standard_plane_slot.  The
// logical rows/columns are always used for the tile grid; physical padding is
// addressed only through a tile whose neutral cells were explicitly staged.
[[nodiscard]] inline std::uint64_t plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) noexcept {
    const std::uint64_t tile_rows = (rows + kTile - 1) / kTile;
    const std::uint64_t tile_columns = (columns + kTile - 1) / kTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kTile) * tile_columns + column / kTile;
    return tile_index * kTileSlots
            + (row % kTile) * kTile + column % kTile;
}

[[nodiscard]] inline std::uint32_t load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t result = 0;
    for (unsigned int byte = 0; byte < 4; ++byte) {
        result |= static_cast<std::uint32_t>(base[word * 4 + byte])
                << (8 * byte);
    }
    return result;
}

[[nodiscard]] inline std::uint16_t load_u16(
        const unsigned char* base, std::uint64_t index) noexcept {
    const std::uint64_t bit = index * kBf16Bits;
    const std::uint32_t word = load_word(base, bit / 32);
    const unsigned int shift = static_cast<unsigned int>(bit % 32);
    return static_cast<std::uint16_t>((word >> shift) & 0xffffu);
}


[[nodiscard]] inline std::uint16_t bf16_daz_bits(
        std::uint16_t bits) noexcept {
    const std::uint16_t exponent = bits & 0x7f80u;
    const std::uint16_t fraction = bits & 0x007fu;
    if (exponent == 0 && fraction != 0) {
        return static_cast<std::uint16_t>(bits & 0x8000u);
    }
    return bits;
}

#if defined(IOM_SYCL_BF16_MATRIX)
namespace matrix = sycl::ext::oneapi::experimental::matrix;
using Bf16 = sycl::ext::oneapi::bfloat16;

[[nodiscard]] inline Bf16 bf16_from_bits(
        std::uint16_t bits) noexcept {
    return sycl::bit_cast<Bf16>(bits);
}

[[nodiscard]] inline Bf16 load_bf16_daz(
        const unsigned char* base, std::uint64_t slot) noexcept {
    return bf16_from_bits(bf16_daz_bits(load_u16(base, slot)));
}

[[nodiscard]] inline Bf16 load_scratch_bf16_daz(
        const unsigned char* base, std::uint64_t cell) noexcept {
    return bf16_from_bits(bf16_daz_bits(load_u16(base, cell)));
}

struct SdpaQkJointMatrixKernelTag {};
struct SdpaQkJointMatrixTailKernelTag {};
struct SdpaPvJointMatrixKernelTag {};
struct SdpaPvJointMatrixTailKernelTag {};

[[nodiscard]] inline std::uint64_t leading_plane(
        const SdpaMatrixGeometry& geometry,
        std::uint64_t logical_plane,
        const std::uint64_t* strides,
        std::uint64_t plane_offset) noexcept {
    std::uint64_t result = plane_offset;
    std::uint64_t rest = logical_plane;
    for (std::uint64_t axis = geometry.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % geometry.leading_dimensions[axis];
        rest /= geometry.leading_dimensions[axis];
        result += coordinate * strides[axis];
    }
    return result;
}

template <std::uint64_t M>
inline void sdpa_qk_body(
        sycl::nd_item<1> item, const SdpaQkRequest request,
        std::uint64_t row_blocks, std::uint64_t first_row_block,
        Bf16* local) {
    const SdpaMatrixGeometry& geometry = request.geometry;
    const sycl::sub_group subgroup = item.get_sub_group();
    const std::uint64_t lane = item.get_local_id(0);
    const std::uint64_t group = item.get_group_linear_id();
    const std::uint64_t key_blocks =
            (geometry.L + kTile - 1) / kTile;
    const std::uint64_t groups_per_head = row_blocks * key_blocks;
    const std::uint64_t query_plane = group / groups_per_head;
    const std::uint64_t within = group % groups_per_head;
    const std::uint64_t row_block =
            first_row_block + within / key_blocks;
    const std::uint64_t key_block = within % key_blocks;
    const std::uint64_t batch_plane = query_plane / geometry.Hq;
    const std::uint64_t query_head = query_plane % geometry.Hq;
    const std::uint64_t kv_head = query_head / geometry.grouping;
    const std::uint64_t q_plane = leading_plane(
            geometry, batch_plane, geometry.q_leading_strides,
            geometry.q_plane_offset)
            + query_head * geometry.q_head_stride;
    const std::uint64_t k_plane = leading_plane(
            geometry, batch_plane, geometry.k_leading_strides,
            geometry.k_plane_offset)
            + kv_head * geometry.k_head_stride;
    std::uint64_t row = 0;
    bool row_valid = false;
    if constexpr (M == kTile) {
        row = row_block * kTile + lane;
        row_valid = lane < M && row < geometry.R;
    } else {
        row = first_row_block + within / key_blocks;
        row_valid = lane == 0 && row < geometry.R;
    }
    const std::uint64_t key = key_block * kTile + lane;
    // M=1 launches are used for every logical row so this predicate is
    // uniform across the work-group and excludes causal futures before K is
    // loaded.  The M=16 template is retained for the direct matrix primitive
    // but is not launched for row-varying causal visibility.
    const bool key_valid = row < geometry.R && key < geometry.L
            && key <= geometry.a + row;

    auto accumulator = sycl::ext::oneapi::experimental::matrix::joint_matrix<
            sycl::sub_group, float, matrix::use::accumulator, M, kTile>();
    matrix::joint_matrix_fill(subgroup, accumulator, 0.0f);
    auto a_fragment = matrix::joint_matrix<
            sycl::sub_group, Bf16, matrix::use::a, M, kTile,
            matrix::layout::row_major>();
    auto b_fragment = matrix::joint_matrix<
            sycl::sub_group, Bf16, matrix::use::b, kTile, kTile,
            matrix::layout::col_major>();
    Bf16* const b_local = local + kTileSlots;

    for (std::uint64_t depth_block = 0;
         depth_block < (geometry.D + kTile - 1) / kTile; ++depth_block) {
        const std::uint64_t depth_base = depth_block * kTile;
        const std::uint64_t depth_remaining = geometry.D > depth_base
                ? geometry.D - depth_base : 0;
        for (std::uint64_t depth = 0; depth < kTile; ++depth) {
            const bool depth_valid = depth < depth_remaining;
            local[lane * kTile + depth] = row_valid && depth_valid
                    ? load_bf16_daz(
                              request.q,
                              plane_slot(
                                      q_plane, row, depth_base + depth,
                                      geometry.R, geometry.D))
                    : bf16_from_bits(0);
            b_local[lane * kTile + depth] = key_valid && depth_valid
                    ? load_bf16_daz(
                              request.k,
                              plane_slot(
                                      k_plane, key, depth_base + depth,
                                      geometry.C, geometry.D))
                    : bf16_from_bits(0);
        }
        sycl::group_barrier(item.get_group());
        matrix::joint_matrix_load(
                subgroup, a_fragment,
                sycl::multi_ptr<Bf16, sycl::access::address_space::local_space>(
                        local),
                kTile);
        matrix::joint_matrix_load(
                subgroup, b_fragment,
                sycl::multi_ptr<Bf16, sycl::access::address_space::local_space>(
                        b_local),
                kTile);
        matrix::joint_matrix_mad(
                subgroup, accumulator, a_fragment, b_fragment, accumulator);
        sycl::group_barrier(item.get_group());
    }

    const std::uint64_t output_row =
            M == kTile ? row_block * kTile : row;
    float* const tile = request.scores
            + query_plane * geometry.scores_head_stride
            + output_row * geometry.scores_row_stride
            + key_block * kTile;
    matrix::joint_matrix_store(
            subgroup, accumulator,
            sycl::multi_ptr<float, sycl::access::address_space::global_space>(
                    tile),
            static_cast<std::size_t>(geometry.scores_row_stride),
            matrix::layout::row_major);
}

template <std::uint64_t M>
inline void sdpa_pv_body(
        sycl::nd_item<1> item, const SdpaPvRequest request,
        std::uint64_t row_blocks, std::uint64_t first_row_block,
        Bf16* local) {
    const SdpaMatrixGeometry& geometry = request.geometry;
    const sycl::sub_group subgroup = item.get_sub_group();
    const std::uint64_t lane = item.get_local_id(0);
    const std::uint64_t group = item.get_group_linear_id();
    const std::uint64_t column_blocks =
            (geometry.D + kTile - 1) / kTile;
    const std::uint64_t groups_per_head = row_blocks * column_blocks;
    const std::uint64_t query_plane = group / groups_per_head;
    const std::uint64_t within = group % groups_per_head;
    const std::uint64_t row_block =
            first_row_block + within / column_blocks;
    const std::uint64_t column_block = within % column_blocks;
    const std::uint64_t batch_plane = query_plane / geometry.Hq;
    const std::uint64_t query_head = query_plane % geometry.Hq;
    const std::uint64_t kv_head = query_head / geometry.grouping;
    const std::uint64_t v_plane = leading_plane(
            geometry, batch_plane, geometry.v_leading_strides,
            geometry.v_plane_offset)
            + kv_head * geometry.v_head_stride;
    std::uint64_t row = 0;
    bool row_valid = false;
    if constexpr (M == kTile) {
        row = row_block * kTile + lane;
        row_valid = lane < M && row < geometry.R;
    } else {
        row = first_row_block + within / column_blocks;
        row_valid = lane == 0 && row < geometry.R;
    }
    auto accumulator = matrix::joint_matrix<
            sycl::sub_group, float, matrix::use::accumulator, M, kTile>();
    matrix::joint_matrix_fill(subgroup, accumulator, 0.0f);
    auto a_fragment = matrix::joint_matrix<
            sycl::sub_group, Bf16, matrix::use::a, M, kTile,
            matrix::layout::row_major>();
    // V is the persistent row-major [C,D] owner.  Its row-major values are
    // presented as the column-major B matrix [L,D] directly from the owner:
    // lane selects the output column and the inner slot selects the key row.
    // No KV-head copy or transpose is materialized.
    auto b_fragment = matrix::joint_matrix<
            sycl::sub_group, Bf16, matrix::use::b, kTile, kTile,
            matrix::layout::col_major>();
    Bf16* const b_local = local + kTileSlots;

    for (std::uint64_t key_block = 0;
         key_block < (geometry.L + kTile - 1) / kTile; ++key_block) {
        const std::uint64_t key_base = key_block * kTile;
        for (std::uint64_t inner = 0; inner < kTile; ++inner) {
            const std::uint64_t key = key_base + inner;
            const bool key_valid = row < geometry.R && key < geometry.L
                    && key <= geometry.a + row;
            local[lane * kTile + inner] = row_valid && key_valid
                    ? load_scratch_bf16_daz(
                              request.p_bf16,
                              query_plane * geometry.probability_head_stride
                                      + row * geometry.probability_row_stride
                                      + key)
                    : bf16_from_bits(0);
            const std::uint64_t v_column =
                    column_block * kTile + lane;
            b_local[lane * kTile + inner] = key_valid
                    && v_column < geometry.D
                    ? load_bf16_daz(
                              request.v,
                              plane_slot(
                                      v_plane, key, v_column,
                                      geometry.C, geometry.D))
                    : bf16_from_bits(0);
        }
        sycl::group_barrier(item.get_group());
        matrix::joint_matrix_load(
                subgroup, a_fragment,
                sycl::multi_ptr<Bf16, sycl::access::address_space::local_space>(
                        local),
                kTile);
        matrix::joint_matrix_load(
                subgroup, b_fragment,
                sycl::multi_ptr<Bf16, sycl::access::address_space::local_space>(
                        b_local),
                kTile);
        matrix::joint_matrix_mad(
                subgroup, accumulator, a_fragment, b_fragment, accumulator);
        sycl::group_barrier(item.get_group());
    }

    const std::uint64_t output_row =
            M == kTile ? row_block * kTile : row;
    float* const tile = request.pv_fp32
            + query_plane * geometry.pv_head_stride
            + output_row * geometry.pv_row_stride
            + column_block * kTile;
    matrix::joint_matrix_store(
            subgroup, accumulator,
            sycl::multi_ptr<float, sycl::access::address_space::global_space>(
                    tile),
            static_cast<std::size_t>(geometry.pv_row_stride),
            matrix::layout::row_major);
}
#endif  // defined(IOM_SYCL_BF16_MATRIX)

[[nodiscard]] std::size_t segment_elements(
        std::size_t planes, std::size_t Hq, std::size_t rows,
        std::size_t columns, const char* what) {
    return checked_mul(
            checked_mul(planes, Hq, what),
            checked_mul(rows, columns, what), what);
}

}  // namespace

bool sdpa_matrix_device_capable(const sycl::device& device) noexcept {
    // The SYCL linear BF16 leaf is the sole owner of the extension/aspect/
    // subgroup/combination query.  SDPA uses precisely the same M={1,16},
    // N=K=16 BF16/BF16-to-FP32 family, so a second probe would risk drift.
    return bf16_linear_device_capable(device);
}

SdpaMatrixScratchLayout sdpa_matrix_scratch_layout(
        std::size_t planes, std::size_t Hq, std::size_t R,
        std::size_t L, std::size_t D) {
    if (planes == 0 || Hq == 0 || R == 0 || L == 0 || D == 0) {
        throw std::invalid_argument(
                "SYCL SDPA scratch dimensions must be nonzero");
    }
    const std::size_t padded_rows = padded(
            R, "SYCL SDPA row padding overflows");
    const std::size_t padded_keys = padded(
            L, "SYCL SDPA key padding overflows");
    const std::size_t padded_depth = padded(
            D, "SYCL SDPA depth padding overflows");

    const std::size_t score_elements = segment_elements(
            planes, Hq, padded_rows, padded_keys,
            "SYCL SDPA score scratch size overflows");
    const std::size_t probability_elements = segment_elements(
            planes, Hq, padded_rows, padded_keys,
            "SYCL SDPA probability scratch size overflows");
    const std::size_t pv_elements = segment_elements(
            planes, Hq, padded_rows, padded_depth,
            "SYCL SDPA PV scratch size overflows");
    const std::size_t head_elements = segment_elements(
            planes, Hq, padded_rows, padded_depth,
            "SYCL SDPA head scratch size overflows");

    const std::size_t score_bytes = checked_mul(
            score_elements, sizeof(float),
            "SYCL SDPA score scratch bytes overflow");
    const std::size_t probability_bytes = checked_mul(
            probability_elements, sizeof(std::uint16_t),
            "SYCL SDPA probability scratch bytes overflow");
    const std::size_t pv_bytes = checked_mul(
            pv_elements, sizeof(float),
            "SYCL SDPA PV scratch bytes overflow");
    const std::size_t head_bytes = checked_mul(
            head_elements, sizeof(std::uint16_t),
            "SYCL SDPA head scratch bytes overflow");

    SdpaMatrixScratchLayout layout;
    layout.scores_offset = 0;
    layout.probability_offset = align_up(
            score_bytes, kScratchAlignment,
            "SYCL SDPA probability offset overflows");
    layout.pv_offset = align_up(
            checked_add(
                    layout.probability_offset, probability_bytes,
                    "SYCL SDPA PV offset overflows"),
            kScratchAlignment, "SYCL SDPA PV offset overflows");
    layout.head_offset = align_up(
            checked_add(
                    layout.pv_offset, pv_bytes,
                    "SYCL SDPA head offset overflows"),
            kScratchAlignment, "SYCL SDPA head offset overflows");
    layout.bytes = align_up(
            checked_add(layout.head_offset, head_bytes,
                        "SYCL SDPA scratch bytes overflow"),
            kScratchAlignment, "SYCL SDPA scratch bytes overflow");
    return layout;
}

void validate_sdpa_matrix_geometry(const SdpaMatrixGeometry& geometry) {
    if (geometry.planes == 0 || geometry.Hq == 0 || geometry.Hkv == 0
            || geometry.R == 0 || geometry.C == 0 || geometry.D == 0
            || geometry.L == 0 || geometry.grouping == 0) {
        throw std::invalid_argument(
                "SYCL SDPA matrix geometry contains a zero extent");
    }
    if (geometry.Hq % geometry.Hkv != 0
            || geometry.grouping != geometry.Hq / geometry.Hkv) {
        throw std::invalid_argument(
                "SYCL SDPA matrix geometry has invalid GQA grouping");
    }
    if (geometry.L > geometry.C || geometry.a >= geometry.C
            || geometry.R > geometry.C - geometry.a) {
        throw std::invalid_argument(
                "SYCL SDPA matrix geometry has an invalid causal range");
    }
    if (geometry.leading_rank > kSdpaMatrixMaxLeadingRank) {
        throw std::overflow_error(
                "SYCL SDPA leading rank exceeds matrix descriptor");
    }
    std::uint64_t plane_count = 1;
    for (std::uint64_t axis = 0; axis < geometry.leading_rank; ++axis) {
        if (geometry.leading_dimensions[axis] == 0) {
            throw std::invalid_argument(
                    "SYCL SDPA leading dimensions contain zero");
        }
        plane_count = checked_u64_mul(
                plane_count, geometry.leading_dimensions[axis],
                "SYCL SDPA leading plane product overflows");
    }
    if (plane_count != geometry.planes) {
        throw std::invalid_argument(
                "SYCL SDPA leading plane product does not match geometry");
    }
    const std::uint64_t expected_rows = round_up_tile(
            geometry.R, "SYCL SDPA row padding overflows");
    const std::uint64_t expected_keys = round_up_tile(
            geometry.L, "SYCL SDPA key padding overflows");
    const std::uint64_t expected_depth = round_up_tile(
            geometry.D, "SYCL SDPA depth padding overflows");
    if (geometry.padded_rows != expected_rows
            || geometry.padded_keys != expected_keys
            || geometry.padded_depth != expected_depth) {
        throw std::invalid_argument(
                "SYCL SDPA matrix geometry has inconsistent tile padding");
    }
    const std::uint64_t expected_score_row = expected_keys;
    const std::uint64_t expected_score_head = checked_u64_mul(
            expected_rows, expected_score_row,
            "SYCL SDPA score stride overflows");
    const std::uint64_t expected_score_plane = checked_u64_mul(
            geometry.Hq, expected_score_head,
            "SYCL SDPA score plane stride overflows");
    if (geometry.scores_row_stride != expected_score_row
            || geometry.scores_head_stride != expected_score_head
            || geometry.scores_plane_stride != expected_score_plane) {
        throw std::invalid_argument(
                "SYCL SDPA score strides do not match padded layout");
    }
    const std::uint64_t expected_probability_head = expected_score_head;
    const std::uint64_t expected_probability_plane = expected_score_plane;
    if (geometry.probability_row_stride != expected_score_row
            || geometry.probability_head_stride != expected_probability_head
            || geometry.probability_plane_stride != expected_probability_plane) {
        throw std::invalid_argument(
                "SYCL SDPA probability strides do not match padded layout");
    }
    const std::uint64_t expected_pv_row = expected_depth;
    const std::uint64_t expected_pv_head = checked_u64_mul(
            expected_rows, expected_pv_row,
            "SYCL SDPA PV stride overflows");
    const std::uint64_t expected_pv_plane = checked_u64_mul(
            geometry.Hq, expected_pv_head,
            "SYCL SDPA PV plane stride overflows");
    if (geometry.pv_row_stride != expected_pv_row
            || geometry.pv_head_stride != expected_pv_head
            || geometry.pv_plane_stride != expected_pv_plane) {
        throw std::invalid_argument(
                "SYCL SDPA PV strides do not match padded layout");
    }

    (void)checked_u64_mul(
            checked_u64_mul(
                    checked_u64_mul(geometry.planes, geometry.Hq,
                                    "SYCL SDPA matrix work size overflows"),
                    geometry.R, "SYCL SDPA matrix work size overflows"),
            geometry.D, "SYCL SDPA matrix work size overflows");
    (void)checked_u64_mul(
            checked_u64_mul(geometry.Hkv, geometry.C,
                            "SYCL SDPA KV work size overflows"),
            geometry.D, "SYCL SDPA KV work size overflows");
}

#if defined(IOM_SYCL_BF16_MATRIX)
namespace {

template <std::uint64_t M>
[[nodiscard]] sycl::event submit_qk(
        sycl::queue& queue, const SdpaQkRequest& request,
        std::uint64_t row_blocks, std::uint64_t first_row_block,
        std::size_t groups) {
    const std::size_t global_size = checked_mul(
            groups, static_cast<std::size_t>(kTile),
            "SYCL SDPA QK global range overflows");
    return queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<Bf16, 1> local(
                sycl::range<1>(2 * kTileSlots), handler);
        if constexpr (M == kTile) {
            handler.parallel_for<SdpaQkJointMatrixKernelTag>(
                    sycl::nd_range<1>(
                            sycl::range<1>(global_size),
                            sycl::range<1>(kTile)),
                    [=](sycl::nd_item<1> item)
                            [[sycl::reqd_sub_group_size(kTile)]] {
                                sdpa_qk_body<kTile>(
                                        item, request, row_blocks,
                                        first_row_block,
                                        local.get_multi_ptr<
                                                sycl::access::decorated::yes>()
                                                .get());
                            });
        } else {
            handler.parallel_for<SdpaQkJointMatrixTailKernelTag>(
                    sycl::nd_range<1>(
                            sycl::range<1>(global_size),
                            sycl::range<1>(kTile)),
                    [=](sycl::nd_item<1> item)
                            [[sycl::reqd_sub_group_size(kTile)]] {
                                sdpa_qk_body<1>(
                                        item, request, row_blocks,
                                        first_row_block,
                                        local.get_multi_ptr<
                                                sycl::access::decorated::yes>()
                                                .get());
                            });
        }
    });
}

template <std::uint64_t M>
[[nodiscard]] sycl::event submit_pv(
        sycl::queue& queue, const SdpaPvRequest& request,
        std::uint64_t row_blocks, std::uint64_t first_row_block,
        std::size_t groups) {
    const std::size_t global_size = checked_mul(
            groups, static_cast<std::size_t>(kTile),
            "SYCL SDPA PV global range overflows");
    return queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<Bf16, 1> local(
                sycl::range<1>(2 * kTileSlots), handler);
        if constexpr (M == kTile) {
            handler.parallel_for<SdpaPvJointMatrixKernelTag>(
                    sycl::nd_range<1>(
                            sycl::range<1>(global_size),
                            sycl::range<1>(kTile)),
                    [=](sycl::nd_item<1> item)
                            [[sycl::reqd_sub_group_size(kTile)]] {
                                sdpa_pv_body<kTile>(
                                        item, request, row_blocks,
                                        first_row_block,
                                        local.get_multi_ptr<
                                                sycl::access::decorated::yes>()
                                                .get());
                            });
        } else {
            handler.parallel_for<SdpaPvJointMatrixTailKernelTag>(
                    sycl::nd_range<1>(
                            sycl::range<1>(global_size),
                            sycl::range<1>(kTile)),
                    [=](sycl::nd_item<1> item)
                            [[sycl::reqd_sub_group_size(kTile)]] {
                                sdpa_pv_body<1>(
                                        item, request, row_blocks,
                                        first_row_block,
                                        local.get_multi_ptr<
                                                sycl::access::decorated::yes>()
                                                .get());
                            });
        }
    });
}

[[nodiscard]] std::size_t matrix_groups(
        const SdpaMatrixGeometry& geometry, std::uint64_t row_blocks,
        std::uint64_t column_blocks, const char* what) {
    const std::uint64_t per_head = checked_u64_mul(
            row_blocks, column_blocks, what);
    const std::uint64_t heads = checked_u64_mul(
            geometry.planes, geometry.Hq, what);
    const std::uint64_t groups = checked_u64_mul(heads, per_head, what);
    if (groups > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(what);
    }
    return static_cast<std::size_t>(groups);
}

}  // namespace
#endif  // defined(IOM_SYCL_BF16_MATRIX)

sycl::event launch_sdpa_qk(
        sycl::queue& queue, const SdpaQkRequest& request) {
    validate_sdpa_matrix_geometry(request.geometry);
    if (request.q == nullptr || request.k == nullptr
            || request.scores == nullptr) {
        throw std::invalid_argument(
                "SYCL SDPA QK stage received a null pointer");
    }
#if !defined(IOM_SYCL_BF16_MATRIX)
    (void)queue;
    (void)request;
    throw detail::UnsupportedOperation();
#else
    const std::uint64_t key_blocks =
            (request.geometry.L + kTile - 1) / kTile;
    // A causal row has its own visible prefix.  One M=1 work-group per
    // logical row keeps excluded future K values out of both the load and the
    // matrix product; using M=16 here would make that mask row-varying inside
    // one shared B fragment.
    const std::size_t groups = matrix_groups(
            request.geometry, request.geometry.R, key_blocks,
            "SYCL SDPA QK launch groups overflow");
    return submit_qk<1>(
            queue, request, request.geometry.R, 0, groups);
#endif
}

sycl::event launch_sdpa_pv(
        sycl::queue& queue, const SdpaPvRequest& request) {
    validate_sdpa_matrix_geometry(request.geometry);
    if (request.p_bf16 == nullptr || request.v == nullptr
            || request.pv_fp32 == nullptr) {
        throw std::invalid_argument(
                "SYCL SDPA PV stage received a null pointer");
    }
#if !defined(IOM_SYCL_BF16_MATRIX)
    (void)queue;
    (void)request;
    throw detail::UnsupportedOperation();
#else
    const std::uint64_t column_blocks =
            (request.geometry.D + kTile - 1) / kTile;
    const std::size_t groups = matrix_groups(
            request.geometry, request.geometry.R, column_blocks,
            "SYCL SDPA PV launch groups overflow");
    // Match QK's row-wise decomposition so excluded future V rows are never
    // loaded even when their probability is zero.
    return submit_pv<1>(
            queue, request, request.geometry.R, 0, groups);
#endif
}

}  // namespace iom::sycl_detail
