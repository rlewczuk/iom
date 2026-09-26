#include "avx512_bf16.hpp"
#include "device_internal.hpp"
#include "transfer_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../iom_internal.hpp"
#include "../shared/scalar_binary_codec.hpp"
#include "avx512_bf16.hpp"

#if defined(IOM_AVX512_BF16_COMPILED)
// The BF16 QK stage entry point lives in the target-isolated translation unit
// registered through iom_add_avx512_bf16_source(); this baseline worker only
// calls it after the runtime eligibility check.
#include "avx512_bf16_sdpa_qk.hpp"
#endif

namespace iom::cpu_detail {
namespace {


using Bf16Codec = detail::scalar_binary_codec_detail::Codec<
        CpuCarrierTraits<float>>;
constexpr std::size_t kWorkspaceAlignment = 32;

std::atomic<bool> sdpa_failure_armed{false};


[[nodiscard]] float load_bf16(
        const SdpaView& view, std::size_t plane, std::size_t row,
        std::size_t column) noexcept {
    const std::span<const std::size_t> dimensions(
            view.dimensions.data(), view.rank);
    const std::size_t bit = cpu_detail::logical_element_bits(
            dimensions, DataType::BF16, plane, row, column);
    return Bf16Codec::decode(
            cpu_detail::load_bits(view.native_handle, bit, 16),
            Bf16Codec::format(DataType::BF16));
}

void store_bf16(
        const SdpaView& view, std::size_t plane, std::size_t row,
        std::size_t column, float value) noexcept {
    const std::span<const std::size_t> dimensions(
            view.dimensions.data(), view.rank);
    const std::size_t bit = cpu_detail::logical_element_bits(
            dimensions, DataType::BF16, plane, row, column);
    cpu_detail::store_bits(
            view.native_handle, bit, 16,
            Bf16Codec::encode(value, Bf16Codec::format(DataType::BF16)));
}

[[nodiscard]] float matrix_daz(float value) noexcept {
    return value != 0.0f && std::fabs(value) < 0x1p-126f
            ? std::copysign(0.0f, value) : value;
}

// Hand one PV row to the isolated AVX-512 BF16 stage. The stage exists only in
// a build that registered its source, and it may only run after the runtime
// eligibility check; anything else - including a row whose operand scale cannot
// rule out a subnormal result - leaves the row to the gradual scalar
// recurrence below, which stays the contract authority.
[[nodiscard]] bool try_native_sdpa_pv(
        const SdpaRequest& request, const unsigned char* probability_row,
        std::size_t v_head_plane, std::size_t out_plane, std::size_t row,
        std::size_t out_column, std::size_t visible_limit) noexcept {
#if defined(IOM_AVX512_BF16_COMPILED)
    return avx512_bf16_available()
            && sdpa_pv_bf16_row(
                       request, probability_row, v_head_plane, out_plane, row,
                       out_column, visible_limit);
#else
    (void)request;
    (void)probability_row;
    (void)v_head_plane;
    (void)out_plane;
    (void)row;
    (void)out_column;
    (void)visible_limit;
    return false;
#endif
}

[[nodiscard]] std::size_t workspace_elements(
        std::size_t leading_planes, std::size_t heads, std::size_t rows,
        std::size_t length) {
    return detail::checked_mul(
            detail::checked_mul(
                    detail::checked_mul(
                            leading_planes, heads,
                            "CPU SDPA workspace element count overflows"),
                    rows, "CPU SDPA workspace element count overflows"),
            length, "CPU SDPA workspace element count overflows");
}

[[nodiscard]] std::size_t score_bytes(std::size_t elements) {
    return detail::checked_mul(
            elements, sizeof(float), "CPU SDPA score workspace overflows");
}

[[nodiscard]] std::size_t probability_bytes(std::size_t elements) {
    return detail::checked_mul(
            elements, sizeof(std::uint16_t),
            "CPU SDPA probability workspace overflows");
}

void store_probability(
        unsigned char* workspace, std::size_t probability_base,
        std::size_t index, float value) noexcept {
    const std::size_t bit = (probability_base + index * sizeof(std::uint16_t))
            * 8;
    cpu_detail::store_bits(
            workspace, bit, 16,
            Bf16Codec::encode(value, Bf16Codec::format(DataType::BF16)));
}

[[nodiscard]] float load_probability(
        const unsigned char* workspace, std::size_t probability_base,
        std::size_t index) noexcept {
    const std::size_t bit = (probability_base + index * sizeof(std::uint16_t))
            * 8;
    return Bf16Codec::decode(
            cpu_detail::load_bits(workspace, bit, 16),
            Bf16Codec::format(DataType::BF16));
}

void sdpa_plane(
        const SdpaRequest& request, std::size_t q_plane,
        std::size_t k_plane, std::size_t v_plane, std::size_t out_plane,
        std::size_t scratch_plane, std::size_t probability_base,
        bool bf16_qk_target) {
#if defined(IOM_AVX512_BF16_COMPILED)
    // One dispatch decision per plane: eligibility is a process-level property,
    // so neither the row loop nor the probability loop re-queries it.
    const bool native_sdpa_softmax = avx512_bf16_available();
#endif
    const std::size_t q_head_stride =
            request.q.plane_strides[request.q.rank - 3];
    const std::size_t k_head_stride =
            request.k.plane_strides[request.k.rank - 3];
    const std::size_t v_head_stride =
            request.v.plane_strides[request.v.rank - 3];
    const float scale = std::sqrt(static_cast<float>(request.D));
    const float positive_infinity =
            std::numeric_limits<float>::infinity();
    const float quiet_nan = std::numeric_limits<float>::quiet_NaN();

    for (std::size_t head = 0; head < request.Hq; ++head) {
        const std::size_t q_head_plane =
                q_plane + head * q_head_stride;
        const std::size_t kv_head = head / request.grouping;
        const std::size_t k_head_plane =
                k_plane + kv_head * k_head_stride;
        const std::size_t v_head_plane =
                v_plane + kv_head * v_head_stride;
        for (std::size_t row = 0; row < request.R; ++row) {
            const std::size_t item =
                    ((scratch_plane * request.Hq + head) * request.R + row)
                    * request.L;
            auto* score_storage = reinterpret_cast<float*>(
                    request.workspace + item * sizeof(float));
            const std::size_t p_index = item;
            const std::size_t visible_limit =
                    std::min(request.L, request.a + row + 1);

            // QK: the isolated BF16 pair-dot stage when this build contains it
            // and this process passed the runtime eligibility check, otherwise
            // the portable scalar recurrence. The stage re-decides per row for
            // operands its native association cannot represent exactly and
            // reports that through a false result, after which this same scalar
            // loop recomputes every visible score of the row.
            bool native_scores = false;
#if defined(IOM_AVX512_BF16_COMPILED)
            if (bf16_qk_target) {
                native_scores = avx512_bf16_sdpa_qk_row(
                        request, q_head_plane, k_head_plane, row,
                        visible_limit, scale, score_storage);
            }
#else
            (void)bf16_qk_target;
#endif
            if (!native_scores) {
                for (std::size_t token = 0; token < request.L; ++token) {
                    if (token >= visible_limit) continue;
                    float dot = 0.0f;
                    for (std::size_t feature = 0; feature < request.D;
                         ++feature) {
                        const float q = matrix_daz(load_bf16(
                                request.q, q_head_plane, row, feature));
                        const float k = matrix_daz(load_bf16(
                                request.k, k_head_plane, token, feature));
                        const volatile float product = q * k;
                        const volatile float next = dot + product;
                        dot = next;
                    }
                    const volatile float scaled = dot / scale;
                    score_storage[token] = scaled;
                }
                // The compliant scalar worker runs as this stage's fallback
                // whenever a build containing the stage does not use the native
                // path for the row.
#if defined(IOM_AVX512_BF16_COMPILED) \
        && defined(IOM_AVX512_BF16_TESTING)
                avx512_bf16_test_record(
                        Avx512Bf16Stage::SdpaQk, Avx512Bf16Path::Fallback);
#endif
            }

            bool has_nan = false;
            std::size_t positive_infinities = 0;
            bool all_negative_infinity = true;
            for (std::size_t token = 0; token < visible_limit; ++token) {
                const float score = score_storage[token];
                has_nan = has_nan || std::isnan(score);
                positive_infinities +=
                        score == positive_infinity ? std::size_t{1} : 0;
                all_negative_infinity = all_negative_infinity
                        && std::isinf(score) && std::signbit(score);
            }

            if (has_nan || all_negative_infinity) {
                for (std::size_t token = 0; token < request.L; ++token) {
                    store_probability(
                            request.workspace, probability_base,
                            p_index + token,
                            token < visible_limit ? quiet_nan : 0.0f);
                }
                for (std::size_t feature = 0; feature < request.D;
                     ++feature) {
                    store_bf16(
                            request.out, out_plane, row,
                            head * request.D + feature, quiet_nan);
                }
                continue;
            }

            if (positive_infinities != 0) {
                const float probability =
                        1.0f / static_cast<float>(positive_infinities);
                for (std::size_t token = 0; token < visible_limit; ++token) {
                    store_probability(
                            request.workspace, probability_base,
                            p_index + token,
                            score_storage[token] == positive_infinity
                                    ? probability : 0.0f);
                }
                for (std::size_t token = visible_limit;
                     token < request.L; ++token) {
                    store_probability(
                            request.workspace, probability_base,
                            p_index + token, 0.0f);
                }
            } else {
                float maximum = -positive_infinity;
                for (std::size_t token = 0; token < visible_limit; ++token) {
                    if (score_storage[token] > maximum) {
                        maximum = score_storage[token];
                    }
                }
                float sum = 0.0f;
                for (std::size_t token = 0; token < visible_limit; ++token) {
                    if (std::isinf(score_storage[token])
                            && std::signbit(score_storage[token])) {
                        score_storage[token] = 0.0f;
                        continue;
                    }
                    const volatile float shifted =
                            score_storage[token] - maximum;
                    const volatile float exponent = std::exp(shifted);
                    const volatile float next = sum + exponent;
                    sum = next;
                    score_storage[token] = exponent;
                }
                // The isolated AVX-512 BF16 stage completes the whole visible
                // range of this row when this build contains the target sources
                // and the process may enter them; otherwise, and for an
                // ineligible process, the portable loop below keeps the row.
                // Both paths perform the same single BF16 RNE conversion.
                bool probabilities_stored = false;
#if defined(IOM_AVX512_BF16_COMPILED)
                if (native_sdpa_softmax) {
                    avx512_bf16_sdpa_softmax_probabilities(
                            score_storage, visible_limit, sum,
                            request.workspace, probability_base, p_index);
                    probabilities_stored = true;
                }
#endif
                if (!probabilities_stored) {
                    for (std::size_t token = 0; token < visible_limit;
                         ++token) {
                        const volatile float normalized =
                                score_storage[token] / sum;
                        store_probability(
                                request.workspace, probability_base,
                                p_index + token, normalized);
                    }
                }
                for (std::size_t token = visible_limit;
                     token < request.L; ++token) {
                    store_probability(
                            request.workspace, probability_base,
                            p_index + token, 0.0f);
                }
            }

            if (!try_native_sdpa_pv(
                        request,
                        request.workspace + probability_base
                                + p_index * sizeof(std::uint16_t),
                        v_head_plane, out_plane, row, head * request.D,
                        visible_limit)) {
                for (std::size_t feature = 0; feature < request.D;
                     ++feature) {
                    float result = 0.0f;
                    for (std::size_t token = 0; token < visible_limit;
                         ++token) {
                        const float probability = matrix_daz(load_probability(
                                request.workspace, probability_base,
                                p_index + token));
                        const float value = matrix_daz(load_bf16(
                                request.v, v_head_plane, token, feature));
                        const volatile float product = probability * value;
                        const volatile float next = result + product;
                        result = next;
                    }
                    // Canonicalize a -0 accumulator to +0 before the final BF16 store.
                    if (result == 0.0f) result = 0.0f;
                    store_bf16(
                            request.out, out_plane, row,
                            head * request.D + feature, result);
                }
            }
        }
    }

}

}  // namespace

WorkspaceRequirements sdpa_workspace_requirements(
        std::size_t leading_planes, std::size_t heads, std::size_t rows,
        std::size_t length) {
    const std::size_t elements =
            workspace_elements(leading_planes, heads, rows, length);
    const std::size_t scores = score_bytes(elements);
    const std::size_t probabilities = probability_bytes(elements);
    return {
            detail::checked_add(
                    scores, probabilities,
                    "CPU SDPA workspace total size overflows"),
            kWorkspaceAlignment};
}

void sdpa_elements(const SdpaRequest& request) {
    const std::size_t leading_rank = request.q.rank - 3;
    std::size_t leading_planes = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        leading_planes *= request.q.dimensions[axis];
    }
    const std::size_t elements = workspace_elements(
            leading_planes, request.Hq, request.R, request.L);
    const std::size_t score_segment = score_bytes(elements);

    // One dispatch decision for the whole accepted request: the isolated BF16
    // pair-dot stage exists in this build, and this process may enter its target
    // code. An ineligible CPU, OS, or portable build keeps the scalar worker for
    // every plane, and every plane of an eligible request selects the native
    // stage.
#if defined(IOM_AVX512_BF16_COMPILED)
    const bool bf16_qk_target = avx512_bf16_available();
#else
    const bool bf16_qk_target = false;
#endif

    auto visit = [&](auto&& self, std::size_t axis, std::size_t q_plane,
                     std::size_t k_plane, std::size_t v_plane,
                     std::size_t out_plane, std::size_t plane) -> void {
        if (axis == leading_rank) {
            sdpa_plane(
                    request, q_plane, k_plane, v_plane, out_plane, plane,
                    score_segment, bf16_qk_target);
            return;
        }
        for (std::size_t index = 0;
             index < request.q.dimensions[axis]; ++index) {
            self(
                    self, axis + 1,
                    q_plane + index * request.q.plane_strides[axis],
                    k_plane + index * request.k.plane_strides[axis],
                    v_plane + index * request.v.plane_strides[axis],
                    out_plane + index * request.out.plane_strides[axis],
                    plane * request.q.dimensions[axis] + index);
        }
    };
    visit(
            visit, 0, request.q.plane_offset, request.k.plane_offset,
            request.v.plane_offset, request.out.plane_offset, 0);
}

void arm_sdpa_failure() noexcept {
    sdpa_failure_armed.store(true, std::memory_order_release);
}

void clear_sdpa_failure() noexcept {
    sdpa_failure_armed.store(false, std::memory_order_release);
}

bool consume_sdpa_failure() noexcept {
    return sdpa_failure_armed.exchange(false, std::memory_order_acq_rel);
}

}  // namespace iom::cpu_detail
