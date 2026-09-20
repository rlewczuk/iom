#include "sdpa_nonmatrix.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace iom::sycl_detail {
namespace {

constexpr std::size_t kAlignment = 32;
// Standard tensor tiling of the merged destination plane.  This stage keeps
// its own copy of the fixed 16x16 tile so it stays independent of the native
// matrix sibling.
constexpr std::size_t kTile = 16;
constexpr std::size_t kTileSlots = kTile * kTile;
constexpr std::uint32_t kF32ExponentMask = 0x7f800000u;
constexpr std::uint32_t kF32FractionMask = 0x007fffffu;
constexpr std::uint32_t kF32PositiveInfinity = 0x7f800000u;

[[nodiscard]] std::size_t checked_add(
        std::size_t lhs, std::size_t rhs, const char* message) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(message);
    }
    return lhs + rhs;
}

[[nodiscard]] std::size_t checked_mul(
        std::size_t lhs, std::size_t rhs, const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

[[nodiscard]] std::size_t indexed_extent(
        std::initializer_list<std::pair<std::size_t, std::size_t>> axes,
        const char* message) {
    std::size_t extent = 1;
    for (const auto [count, stride] : axes) {
        if (count == 0) {
            return 0;
        }
        extent = checked_add(
                extent,
                checked_mul(count - 1, stride, message),
                message);
    }
    return extent;
}

struct ByteRange final {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

template <typename Pointer>
[[nodiscard]] ByteRange make_range(
        Pointer pointer, std::size_t element_offset, std::size_t elements,
        const char* message) {
    if (pointer == nullptr || elements == 0) {
        throw std::invalid_argument(message);
    }
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(pointer);
    if (base % kAlignment != 0) {
        throw std::invalid_argument(message);
    }
    const std::size_t offset_bytes =
            checked_mul(element_offset, sizeof(*pointer), message);
    const std::size_t bytes = checked_mul(elements, sizeof(*pointer), message);
    const std::uintptr_t limit = std::numeric_limits<std::uintptr_t>::max();
    if (offset_bytes > limit - base) {
        throw std::overflow_error(message);
    }
    const std::uintptr_t begin = base + offset_bytes;
    if (begin % kAlignment != 0) {
        throw std::invalid_argument(message);
    }
    if (bytes > limit - begin) {
        throw std::overflow_error(message);
    }
    return {begin, begin + bytes};
}

[[nodiscard]] bool overlaps(ByteRange lhs, ByteRange rhs) noexcept {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

void reject_overlap(
        const char* lhs_name, ByteRange lhs,
        const char* rhs_name, ByteRange rhs) {
    if (overlaps(lhs, rhs)) {
        throw std::invalid_argument(
                (std::string("SYCL SDPA nonmatrix ranges overlap: ")
                 + lhs_name + " and " + rhs_name).c_str());
    }
}

[[nodiscard]] std::size_t row_visible_count(
        const SdpaNonmatrixRequest& request, std::size_t row) noexcept {
    const std::size_t causal_end = request.causal_offset + row + 1;
    return causal_end < request.initialized_length
            ? causal_end : request.initialized_length;
}


[[nodiscard]] inline SdpaBf16 float_to_bf16_rne(float value) noexcept {
    const std::uint32_t bits = sycl::bit_cast<std::uint32_t>(value);
    if (value == 0.0F) {
        // Stored numerical zeros are canonical +0, including a formed -0.
        return 0;
    }
    if ((bits & kF32ExponentMask) == kF32ExponentMask
            && (bits & kF32FractionMask) != 0) {
        // The payload/sign are unspecified, but the stored result must remain
        // a quiet NaN even when the source happened to be signaling.
        return static_cast<SdpaBf16>((bits >> 16) | 0x0040u);
    }
    const std::uint32_t rounding =
            0x7fffu + ((bits >> 16) & 1u);
    return static_cast<SdpaBf16>((bits + rounding) >> 16);
}

[[nodiscard]] inline SdpaBf16 quiet_nan_bf16() noexcept {
    return static_cast<SdpaBf16>(0x7fc0u);
}


[[nodiscard]] inline SdpaBf16 canonical_zero(SdpaBf16 bits) noexcept {
    return (bits & 0x7fffu) == 0 ? static_cast<SdpaBf16>(0) : bits;
}

[[nodiscard]] inline std::size_t score_index(
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t key) noexcept {
    return request.score_plane_offset
            + plane * request.score_plane_stride
            + head * request.score_head_stride
            + row * request.score_row_stride
            + key * request.score_key_stride;
}

[[nodiscard]] inline std::size_t p_index(
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t key) noexcept {
    return request.p_plane_offset
            + plane * request.p_plane_stride
            + head * request.p_head_stride
            + row * request.p_row_stride
            + key * request.p_key_stride;
}

[[nodiscard]] inline std::size_t pv_index(
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t feature) noexcept {
    return request.pv_plane_offset
            + plane * request.pv_plane_stride
            + head * request.pv_head_stride
            + row * request.pv_row_stride
            + feature * request.pv_feature_stride;
}

[[nodiscard]] inline std::size_t head_index(
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t feature) noexcept {
    return request.head_plane_offset
            + plane * request.head_plane_stride
            + head * request.head_head_stride
            + row * request.head_row_stride
            + feature * request.head_feature_stride;
}

[[nodiscard]] inline std::size_t merged_leading_plane_base(
        const SdpaNonmatrixRequest& request,
        std::size_t plane) noexcept {
    std::size_t result = request.merged_plane_offset;
    std::size_t rest = plane;
    for (std::size_t axis = request.merged_leading_rank; axis-- > 0;) {
        const std::size_t coordinate =
                rest % request.merged_leading_dimensions[axis];
        rest /= request.merged_leading_dimensions[axis];
        result += coordinate * request.merged_leading_strides[axis];
    }
    return result;
}

// Standard 16x16 tiled slot of one logical (row, column) output cell inside one
// destination plane, matching `detail::standard_plane_slot`.  Device code
// cannot call that host helper, so the formula is repeated here with checked
// host-side inputs exactly like the queue's other backend-private writers.
[[nodiscard]] inline std::size_t merged_plane_slot(
        std::size_t row, std::size_t column, std::size_t rows,
        std::size_t columns) noexcept {
    const std::size_t tile_columns = columns / kTile + (columns % kTile != 0);
    const std::size_t tile_index =
            (row / kTile) * tile_columns + column / kTile;
    return tile_index * kTileSlots + (row % kTile) * kTile + column % kTile;
}

[[nodiscard]] inline std::size_t merged_index(
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t feature) noexcept {
    return merged_leading_plane_base(request, plane)
            + merged_plane_slot(
                      row, head * request.head_dim + feature, request.rows,
                      request.query_heads * request.head_dim);
}

inline void store_zero_tail(
        SdpaBf16* probabilities,
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t first_key) noexcept {
    for (std::size_t key = first_key;
         key < request.initialized_length; ++key) {
        probabilities[p_index(request, plane, head, row, key)] = 0;
    }
}

inline void scale_softmax_row(
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row) noexcept {
    const std::size_t visible = row_visible_count(request, row);
    const float scale = request.score_scale != 0.0F
            ? request.score_scale
            : 1.0F / sycl::sqrt(static_cast<float>(request.head_dim));

    float maximum = sycl::bit_cast<float>(kF32PositiveInfinity);
    maximum = -maximum;
    bool has_nan = false;
    std::size_t positive_infinities = 0;
    std::size_t finite_values = 0;

    // Form and scale only the included score prefix.  Future keys are never
    // read, even when their caller-owned score cells contain nonfinite data.
    for (std::size_t key = 0; key < visible; ++key) {
        const std::size_t index = score_index(request, plane, head, row, key);
        const float value = request.scores[index] * scale;
        request.scores[index] = value;
        if (sycl::isnan(value)) {
            has_nan = true;
        } else if (sycl::isinf(value) && value > 0.0F) {
            ++positive_infinities;
        } else if (!sycl::isinf(value)) {
            ++finite_values;
            maximum = value > maximum ? value : maximum;
        }
    }

    SdpaBf16* const probabilities = request.p_bf16;
    if (has_nan || (positive_infinities == 0 && finite_values == 0)) {
        for (std::size_t key = 0; key < visible; ++key) {
            probabilities[p_index(request, plane, head, row, key)] =
                    quiet_nan_bf16();
        }
        store_zero_tail(probabilities, request, plane, head, row, visible);
        return;
    }

    if (positive_infinities != 0) {
        const float probability =
                1.0F / static_cast<float>(positive_infinities);
        const SdpaBf16 encoded = float_to_bf16_rne(probability);
        for (std::size_t key = 0; key < visible; ++key) {
            const float value = request.scores[
                    score_index(request, plane, head, row, key)];
            probabilities[p_index(request, plane, head, row, key)] =
                    sycl::isinf(value) && value > 0.0F ? encoded
                                                        : static_cast<SdpaBf16>(0);
        }
        store_zero_tail(probabilities, request, plane, head, row, visible);
        return;
    }

    float denominator = 0.0F;
    for (std::size_t key = 0; key < visible; ++key) {
        const float value = request.scores[
                score_index(request, plane, head, row, key)];
        if (sycl::isinf(value) && value < 0.0F) {
            continue;
        }
        denominator += sycl::exp(value - maximum);
    }

    for (std::size_t key = 0; key < visible; ++key) {
        const float value = request.scores[
                score_index(request, plane, head, row, key)];
        const float probability = sycl::isinf(value) && value < 0.0F
                ? 0.0F
                : sycl::exp(value - maximum) / denominator;
        probabilities[p_index(request, plane, head, row, key)] =
                float_to_bf16_rne(probability);
    }
    store_zero_tail(probabilities, request, plane, head, row, visible);
}


inline void merge_head_element(
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t feature) noexcept {
    const SdpaBf16 value = canonical_zero(request.head_bf16[
            head_index(request, plane, head, row, feature)]);
    request.merged_bf16[merged_index(
            request, plane, head, row, feature)] = value;
}

inline void convert_pv_element(
        const SdpaNonmatrixRequest& request, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t feature) noexcept {
    request.head_bf16[head_index(request, plane, head, row, feature)] =
            float_to_bf16_rne(request.pv_fp32[
                    pv_index(request, plane, head, row, feature)]);
}

template <typename Kernel>
[[nodiscard]] sycl::event submit_1d(
        sycl::queue& queue, std::size_t count,
        const sycl::event* dependency, Kernel kernel) {
    return queue.submit([&](sycl::handler& handler) {
        if (dependency != nullptr) {
            handler.depends_on(*dependency);
        }
        handler.parallel_for(sycl::range<1>(count), kernel);
    });
}

[[nodiscard]] std::size_t row_work_items(
        const SdpaNonmatrixRequest& request) {
    return checked_mul(
            checked_mul(request.plane_count, request.query_heads,
                        "SYCL SDPA nonmatrix row work overflows"),
            request.rows, "SYCL SDPA nonmatrix row work overflows");
}


[[nodiscard]] std::size_t pv_work_items(
        const SdpaNonmatrixRequest& request) {
    std::size_t result = checked_mul(
            request.plane_count, request.query_heads,
            "SYCL SDPA nonmatrix PV work overflows");
    result = checked_mul(
            result, request.rows,
            "SYCL SDPA nonmatrix PV work overflows");
    return checked_mul(
            result, request.head_dim,
            "SYCL SDPA nonmatrix PV work overflows");
}


void validate_stride(std::size_t stride, const char* message) {
    if (stride == 0) {
        throw std::invalid_argument(message);
    }
}

}  // namespace

void validate_sdpa_nonmatrix_request(
        const SdpaNonmatrixRequest& request) {
    if (request.plane_count == 0 || request.query_heads == 0
            || request.kv_heads == 0 || request.rows == 0
            || request.capacity == 0 || request.head_dim == 0
            || request.initialized_length == 0) {
        throw std::invalid_argument(
                "SYCL SDPA nonmatrix dimensions must be nonzero");
    }
    if (request.initialized_length > request.capacity
            || request.causal_offset >= request.capacity
            || request.rows > request.capacity - request.causal_offset) {
        throw std::invalid_argument(
                "SYCL SDPA nonmatrix causal range is invalid");
    }
    if (request.query_heads % request.kv_heads != 0
            || request.grouping != request.query_heads / request.kv_heads) {
        throw std::invalid_argument(
                "SYCL SDPA nonmatrix GQA grouping is invalid");
    }
    if (request.score_scale != 0.0F
            && (!std::isfinite(request.score_scale)
                || request.score_scale <= 0.0F)) {
        throw std::invalid_argument(
                "SYCL SDPA nonmatrix score scale is invalid");
    }
    if (request.head_dim > std::numeric_limits<float>::max()) {
        throw std::overflow_error(
                "SYCL SDPA nonmatrix head dimension cannot be scaled");
    }

    validate_stride(request.score_plane_stride,
                    "SYCL SDPA nonmatrix score plane stride is zero");
    validate_stride(request.score_head_stride,
                    "SYCL SDPA nonmatrix score head stride is zero");
    validate_stride(request.score_row_stride,
                    "SYCL SDPA nonmatrix score row stride is zero");
    validate_stride(request.score_key_stride,
                    "SYCL SDPA nonmatrix score key stride is zero");
    validate_stride(request.p_plane_stride,
                    "SYCL SDPA nonmatrix probability plane stride is zero");
    validate_stride(request.p_head_stride,
                    "SYCL SDPA nonmatrix probability head stride is zero");
    validate_stride(request.p_row_stride,
                    "SYCL SDPA nonmatrix probability row stride is zero");
    validate_stride(request.p_key_stride,
                    "SYCL SDPA nonmatrix probability key stride is zero");
    validate_stride(request.v_plane_stride,
                    "SYCL SDPA nonmatrix V plane stride is zero");
    validate_stride(request.v_head_stride,
                    "SYCL SDPA nonmatrix V head stride is zero");
    validate_stride(request.v_key_stride,
                    "SYCL SDPA nonmatrix V key stride is zero");
    validate_stride(request.v_feature_stride,
                    "SYCL SDPA nonmatrix V feature stride is zero");
    validate_stride(request.pv_plane_stride,
                    "SYCL SDPA nonmatrix PV plane stride is zero");
    validate_stride(request.pv_head_stride,
                    "SYCL SDPA nonmatrix PV head stride is zero");
    validate_stride(request.pv_row_stride,
                    "SYCL SDPA nonmatrix PV row stride is zero");
    validate_stride(request.pv_feature_stride,
                    "SYCL SDPA nonmatrix PV feature stride is zero");
    validate_stride(request.head_plane_stride,
                    "SYCL SDPA nonmatrix head plane stride is zero");
    validate_stride(request.head_head_stride,
                    "SYCL SDPA nonmatrix head stride is zero");
    validate_stride(request.head_row_stride,
                    "SYCL SDPA nonmatrix head row stride is zero");
    validate_stride(request.head_feature_stride,
                    "SYCL SDPA nonmatrix head feature stride is zero");
    if (request.merged_leading_rank > kSdpaNonmatrixMaxLeadingRank) {
        throw std::overflow_error(
                "SYCL SDPA nonmatrix merged leading rank is too large");
    }
    for (std::size_t axis = 0; axis < request.merged_leading_rank; ++axis) {
        if (request.merged_leading_dimensions[axis] == 0) {
            throw std::invalid_argument(
                    "SYCL SDPA nonmatrix merged leading dimension is zero");
        }
        validate_stride(
                request.merged_leading_strides[axis],
                "SYCL SDPA nonmatrix merged leading stride is zero");
    }
    for (std::size_t axis = request.merged_leading_rank;
         axis < kSdpaNonmatrixMaxLeadingRank; ++axis) {
        if (request.merged_leading_dimensions[axis] != 0
                || request.merged_leading_strides[axis] != 0) {
            throw std::invalid_argument(
                    "SYCL SDPA nonmatrix merged leading map is not bounded");
        }
    }

    const std::size_t score_elements = indexed_extent(
            {{request.plane_count, request.score_plane_stride},
             {request.query_heads, request.score_head_stride},
             {request.rows, request.score_row_stride},
             {request.initialized_length, request.score_key_stride}},
            "SYCL SDPA nonmatrix score range overflows");
    const std::size_t p_elements = indexed_extent(
            {{request.plane_count, request.p_plane_stride},
             {request.query_heads, request.p_head_stride},
             {request.rows, request.p_row_stride},
             {request.initialized_length, request.p_key_stride}},
            "SYCL SDPA nonmatrix probability range overflows");
    const std::size_t v_elements = indexed_extent(
            {{request.plane_count, request.v_plane_stride},
             {request.kv_heads, request.v_head_stride},
             {request.initialized_length, request.v_key_stride},
             {request.head_dim, request.v_feature_stride}},
            "SYCL SDPA nonmatrix V range overflows");
    const std::size_t pv_elements = indexed_extent(
            {{request.plane_count, request.pv_plane_stride},
             {request.query_heads, request.pv_head_stride},
             {request.rows, request.pv_row_stride},
             {request.head_dim, request.pv_feature_stride}},
            "SYCL SDPA nonmatrix PV range overflows");
    const std::size_t head_elements = indexed_extent(
            {{request.plane_count, request.head_plane_stride},
             {request.query_heads, request.head_head_stride},
             {request.rows, request.head_row_stride},
             {request.head_dim, request.head_feature_stride}},
            "SYCL SDPA nonmatrix head range overflows");
    // Conservative element-slot extent of the standard tiled destination: the
    // highest selected plane base plus the whole tiled plane span it can
    // address.  `merged_plane_offset` is already part of the extent, so the
    // range check below starts at offset zero.
    const std::size_t merged_output_width = checked_mul(
            request.query_heads, request.head_dim,
            "SYCL SDPA nonmatrix output width overflows");
    const std::size_t merged_plane_rows =
            request.rows / kTile + (request.rows % kTile != 0);
    const std::size_t merged_plane_columns =
            merged_output_width / kTile + (merged_output_width % kTile != 0);
    const std::size_t merged_plane_slots = checked_mul(
            checked_mul(
                    merged_plane_rows, merged_plane_columns,
                    "SYCL SDPA nonmatrix merged plane slots overflow"),
            kTileSlots, "SYCL SDPA nonmatrix merged plane slots overflow");
    std::size_t merged_elements = checked_add(
            request.merged_plane_offset, merged_plane_slots,
            "SYCL SDPA nonmatrix merged range overflows");
    for (std::size_t axis = 0; axis < request.merged_leading_rank; ++axis) {
        merged_elements = checked_add(
                merged_elements,
                checked_mul(
                        request.merged_leading_dimensions[axis] - 1,
                        request.merged_leading_strides[axis],
                        "SYCL SDPA nonmatrix merged range overflows"),
                "SYCL SDPA nonmatrix merged range overflows");
    }

    const ByteRange scores = make_range(
            request.scores, request.score_plane_offset, score_elements,
            "SYCL SDPA nonmatrix scores must be nonnull and 32-byte aligned");
    const ByteRange probabilities = make_range(
            request.p_bf16, request.p_plane_offset, p_elements,
            "SYCL SDPA nonmatrix probabilities must be nonnull and 32-byte aligned");
    const ByteRange v = make_range(
            request.v_bf16, request.v_plane_offset, v_elements,
            "SYCL SDPA nonmatrix V must be nonnull and 32-byte aligned");
    const ByteRange pv = make_range(
            request.pv_fp32, request.pv_plane_offset, pv_elements,
            "SYCL SDPA nonmatrix PV must be nonnull and 32-byte aligned");
    const ByteRange head = make_range(
            request.head_bf16, request.head_plane_offset, head_elements,
            "SYCL SDPA nonmatrix head staging must be nonnull and 32-byte aligned");
    const ByteRange merged = make_range(
            request.merged_bf16, 0, merged_elements,
            "SYCL SDPA nonmatrix output must be nonnull and 32-byte aligned");

    const ByteRange* ranges[] = {&scores, &probabilities, &v, &pv, &head, &merged};
    const char* names[] = {"scores", "probabilities", "V", "PV", "head", "merged"};
    for (std::size_t lhs = 0; lhs < 6; ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < 6; ++rhs) {
            reject_overlap(names[lhs], *ranges[lhs], names[rhs], *ranges[rhs]);
        }
    }
}

[[nodiscard]] sycl::event launch_sdpa_scale_softmax(
        sycl::queue& queue, const SdpaNonmatrixRequest& request) {
    validate_sdpa_nonmatrix_request(request);
    const SdpaNonmatrixRequest captured = request;
    const std::size_t work_items = row_work_items(captured);
    return submit_1d(
            queue, work_items, nullptr,
            [captured](sycl::id<1> item) {
                const std::size_t index = item[0];
                const std::size_t rows_per_plane =
                        captured.query_heads * captured.rows;
                const std::size_t plane = index / rows_per_plane;
                const std::size_t within = index % rows_per_plane;
                const std::size_t head = within / captured.rows;
                const std::size_t row = within % captured.rows;
                scale_softmax_row(captured, plane, head, row);
            });
}

[[nodiscard]] sycl::event launch_sdpa_scale_softmax(
        sycl::queue& queue, const SdpaNonmatrixRequest& request,
        const sycl::event& dependency) {
    validate_sdpa_nonmatrix_request(request);
    const SdpaNonmatrixRequest captured = request;
    const std::size_t work_items = row_work_items(captured);
    return submit_1d(
            queue, work_items, &dependency,
            [captured](sycl::id<1> item) {
                const std::size_t index = item[0];
                const std::size_t rows_per_plane =
                        captured.query_heads * captured.rows;
                const std::size_t plane = index / rows_per_plane;
                const std::size_t within = index % rows_per_plane;
                const std::size_t head = within / captured.rows;
                const std::size_t row = within % captured.rows;
                scale_softmax_row(captured, plane, head, row);
            });
}


[[nodiscard]] sycl::event launch_sdpa_merge_output(
        sycl::queue& queue, const SdpaNonmatrixRequest& request) {
    validate_sdpa_nonmatrix_request(request);
    const SdpaNonmatrixRequest captured = request;
    const std::size_t work_items = pv_work_items(captured);
    const sycl::event converted = submit_1d(
            queue, work_items, nullptr,
            [captured](sycl::id<1> item) {
                const std::size_t index = item[0];
                const std::size_t per_plane =
                        captured.query_heads * captured.rows
                        * captured.head_dim;
                const std::size_t plane = index / per_plane;
                const std::size_t within_plane = index % per_plane;
                const std::size_t per_head =
                        captured.rows * captured.head_dim;
                const std::size_t head = within_plane / per_head;
                const std::size_t within_head = within_plane % per_head;
                const std::size_t row = within_head / captured.head_dim;
                const std::size_t feature = within_head % captured.head_dim;
                convert_pv_element(captured, plane, head, row, feature);
            });
    return submit_1d(
            queue, work_items, &converted,
            [captured](sycl::id<1> item) {
                const std::size_t index = item[0];
                const std::size_t per_plane =
                        captured.query_heads * captured.rows
                        * captured.head_dim;
                const std::size_t plane = index / per_plane;
                const std::size_t within_plane = index % per_plane;
                const std::size_t per_head =
                        captured.rows * captured.head_dim;
                const std::size_t head = within_plane / per_head;
                const std::size_t within_head = within_plane % per_head;
                const std::size_t row = within_head / captured.head_dim;
                const std::size_t feature = within_head % captured.head_dim;
                merge_head_element(captured, plane, head, row, feature);
            });
}

[[nodiscard]] sycl::event launch_sdpa_merge_output(
        sycl::queue& queue, const SdpaNonmatrixRequest& request,
        const sycl::event& dependency) {
    validate_sdpa_nonmatrix_request(request);
    const SdpaNonmatrixRequest captured = request;
    const std::size_t work_items = pv_work_items(captured);
    const sycl::event converted = submit_1d(
            queue, work_items, &dependency,
            [captured](sycl::id<1> item) {
                const std::size_t index = item[0];
                const std::size_t per_plane =
                        captured.query_heads * captured.rows
                        * captured.head_dim;
                const std::size_t plane = index / per_plane;
                const std::size_t within_plane = index % per_plane;
                const std::size_t per_head =
                        captured.rows * captured.head_dim;
                const std::size_t head = within_plane / per_head;
                const std::size_t within_head = within_plane % per_head;
                const std::size_t row = within_head / captured.head_dim;
                const std::size_t feature = within_head % captured.head_dim;
                convert_pv_element(captured, plane, head, row, feature);
            });
    return submit_1d(
            queue, work_items, &converted,
            [captured](sycl::id<1> item) {
                const std::size_t index = item[0];
                const std::size_t per_plane =
                        captured.query_heads * captured.rows
                        * captured.head_dim;
                const std::size_t plane = index / per_plane;
                const std::size_t within_plane = index % per_plane;
                const std::size_t per_head =
                        captured.rows * captured.head_dim;
                const std::size_t head = within_plane / per_head;
                const std::size_t within_head = within_plane % per_head;
                const std::size_t row = within_head / captured.head_dim;
                const std::size_t feature = within_head % captured.head_dim;
                merge_head_element(captured, plane, head, row, feature);
            });
}

}  // namespace iom::sycl_detail
