#include "sdpa_softmax.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int kSdpaThreads = 256;
constexpr unsigned int kSdpaMaxBlocks = 65535;
constexpr std::uint32_t kFloatExponentMask = 0x7f800000u;
constexpr std::uint32_t kFloatMantissaMask = 0x007fffffu;
constexpr std::uint32_t kFloatSignMask = 0x80000000u;
constexpr std::uint32_t kQuietNanBits = 0x7fc00000u;
constexpr std::uint32_t kNegativeInfinityBits = 0xff800000u;

__device__ inline std::uint32_t sdpa_float_bits(float value) noexcept {
    return __float_as_uint(value);
}

__device__ inline float sdpa_float_from_bits(std::uint32_t bits) noexcept {
    return __uint_as_float(bits);
}

__device__ inline bool sdpa_is_nan(float value) noexcept {
    const std::uint32_t bits = sdpa_float_bits(value);
    return (bits & kFloatExponentMask) == kFloatExponentMask
            && (bits & kFloatMantissaMask) != 0;
}

__device__ inline bool sdpa_is_positive_infinity(float value) noexcept {
    return sdpa_float_bits(value) == 0x7f800000u;
}

__device__ inline bool sdpa_is_negative_infinity(float value) noexcept {
    return sdpa_float_bits(value) == kNegativeInfinityBits;
}

__device__ inline float sdpa_quiet_nan() noexcept {
    return sdpa_float_from_bits(kQuietNanBits);
}

__device__ inline float sdpa_negative_infinity() noexcept {
    return sdpa_float_from_bits(kNegativeInfinityBits);
}

// Keep the reduction in the declared FP32 domain.  The volatile materialize
// also prevents a compiler contraction or reassociation from moving the
// probability boundary away from the stable scalar recurrence.
__device__ inline float sdpa_add(float left, float right) noexcept {
    volatile float result = left + right;
    return result;
}

__device__ inline float sdpa_mul(float left, float right) noexcept {
    volatile float result = left * right;
    return result;
}

__device__ inline float sdpa_div(float left, float right) noexcept {
    volatile float result = left / right;
    return result;
}

// Explicit FP32 round-to-nearest-even conversion.  This is intentionally not
// a matrix DAZ conversion: subnormal probabilities remain present until this
// exact BF16 store, and the native PV stage owns its later fixed DAZ boundary.
__device__ inline std::uint16_t sdpa_bf16_rne(float value) noexcept {
    const std::uint32_t bits = sdpa_float_bits(value);
    if ((bits & ~kFloatSignMask) == 0) {
        return 0;
    }
    std::uint32_t upper = bits >> 16;
    const std::uint32_t lower = bits & 0xffffu;
    if (lower > 0x8000u || (lower == 0x8000u && (upper & 1u) != 0)) {
        ++upper;
    }
    return static_cast<std::uint16_t>(upper);
}

__device__ inline std::uint64_t sdpa_offset(
        const iom::rocm_detail::SdpaStageLayout& layout,
        std::uint64_t plane, std::uint64_t head,
        std::uint64_t row, std::uint64_t column) noexcept {
    return layout.plane_offset + plane * layout.plane_stride
            + head * layout.head_stride + row * layout.row_stride
            + column * layout.column_stride;
}

__device__ inline bool sdpa_is_included(
        const iom::rocm_detail::SdpaSoftmaxRequest& request,
        std::uint64_t row, std::uint64_t token) noexcept {
    if ((request.mask & iom::rocm_detail::sdpa_mask_initialized_prefix) != 0
            && token >= request.L) {
        return false;
    }
    if ((request.mask & iom::rocm_detail::sdpa_mask_causal) != 0
            && token > request.a + row) {
        return false;
    }
    return true;
}

__global__ void sdpa_softmax_kernel(
        iom::rocm_detail::SdpaSoftmaxRequest request) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
            * static_cast<std::uint64_t>(blockDim.x)
            + static_cast<std::uint64_t>(threadIdx.x);
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x)
            * static_cast<std::uint64_t>(gridDim.x);
    const std::uint64_t rows_per_plane =
            request.hq * request.probability_layout.rows;
    const std::uint64_t total = request.planes * rows_per_plane;

    for (std::uint64_t work = index; work < total; work += stride) {
        const std::uint64_t plane = work / rows_per_plane;
        const std::uint64_t in_plane = work % rows_per_plane;
        const std::uint64_t head = in_plane / request.probability_layout.rows;
        const std::uint64_t row = in_plane % request.probability_layout.rows;
        const bool logical_row = row < request.rows;

        bool has_nan = false;
        bool all_negative_infinity = logical_row;
        std::uint64_t positive_infinities = 0;
        if (logical_row) {
            for (std::uint64_t token = 0; token < request.L; ++token) {
                if (!sdpa_is_included(request, row, token)) continue;
                const float score = request.scores[sdpa_offset(
                        request.scores_layout, plane, head, row, token)];
                const float scaled = sdpa_mul(score, request.scale);
                has_nan = has_nan || sdpa_is_nan(scaled);
                positive_infinities +=
                        sdpa_is_positive_infinity(scaled) ? 1 : 0;
                all_negative_infinity = all_negative_infinity
                        && sdpa_is_negative_infinity(scaled);
            }
        }

        float maximum = sdpa_negative_infinity();
        float sum = 0.0F;
        if (logical_row && !has_nan && !all_negative_infinity
                && positive_infinities == 0) {
            for (std::uint64_t token = 0; token < request.L; ++token) {
                if (!sdpa_is_included(request, row, token)) continue;
                const float score = request.scores[sdpa_offset(
                        request.scores_layout, plane, head, row, token)];
                const float scaled = sdpa_mul(score, request.scale);
                if (scaled > maximum) maximum = scaled;
            }
            for (std::uint64_t token = 0; token < request.L; ++token) {
                if (!sdpa_is_included(request, row, token)) continue;
                const float score = request.scores[sdpa_offset(
                        request.scores_layout, plane, head, row, token)];
                const float scaled = sdpa_mul(score, request.scale);
                if (!sdpa_is_negative_infinity(scaled)) {
                    sum = sdpa_add(sum, expf(scaled - maximum));
                }
            }
        }

        for (std::uint64_t token = 0;
             token < request.probability_layout.columns; ++token) {
            std::uint16_t encoded = 0;
            if (logical_row && token < request.L
                    && sdpa_is_included(request, row, token)) {
                float probability = 0.0F;
                if (has_nan || all_negative_infinity) {
                    probability = sdpa_quiet_nan();
                } else if (positive_infinities != 0) {
                    const float score = request.scores[sdpa_offset(
                            request.scores_layout, plane, head, row, token)];
                    const float scaled = sdpa_mul(score, request.scale);
                    probability = sdpa_is_positive_infinity(scaled)
                            ? sdpa_div(
                                      1.0F,
                                      static_cast<float>(positive_infinities))
                            : 0.0F;
                } else {
                    const float score = request.scores[sdpa_offset(
                            request.scores_layout, plane, head, row, token)];
                    const float scaled = sdpa_mul(score, request.scale);
                    probability = sdpa_is_negative_infinity(scaled)
                            ? 0.0F
                            : sdpa_div(expf(scaled - maximum), sum);
                }
                encoded = sdpa_bf16_rne(probability);
            }
            request.p_bf16[sdpa_offset(
                    request.probability_layout, plane, head, row, token)] =
                    encoded;
        }
    }
}

__global__ void sdpa_merge_kernel(
        iom::rocm_detail::SdpaSoftmaxRequest request) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
            * static_cast<std::uint64_t>(blockDim.x)
            + static_cast<std::uint64_t>(threadIdx.x);
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x)
            * static_cast<std::uint64_t>(gridDim.x);
    const std::uint64_t head_width = request.hq * request.head_dim;
    const std::uint64_t row_width = head_width;
    const std::uint64_t rows_per_plane = request.rows * row_width;
    const std::uint64_t total = request.planes * rows_per_plane;

    for (std::uint64_t work = index; work < total; work += stride) {
        const std::uint64_t plane = work / rows_per_plane;
        const std::uint64_t in_plane = work % rows_per_plane;
        const std::uint64_t row = in_plane / row_width;
        const std::uint64_t in_row = in_plane % row_width;
        const std::uint64_t head = in_row / request.head_dim;
        const std::uint64_t feature = in_row % request.head_dim;
        const std::uint16_t value = request.pv_bf16[sdpa_offset(
                request.pv_layout, plane, head, row, feature)];
        const std::uint16_t canonical = (value & 0x7fffu) == 0 ? 0 : value;
        request.merged[sdpa_offset(
                request.merged_layout, plane, 0, row, in_row)] = canonical;
    }
}

}  // namespace

namespace iom::rocm_detail {
namespace {

[[nodiscard]] std::uint64_t checked_add(
        std::uint64_t left, std::uint64_t right, const char* message) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t checked_mul(
        std::uint64_t left, std::uint64_t right, const char* message) {
    if (right != 0
            && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

void validate_layout(
        const SdpaStageLayout& layout, std::uint64_t planes,
        std::uint64_t heads, std::uint64_t logical_rows,
        std::uint64_t logical_columns, const char* name,
        bool touch_physical_extent) {
    if (layout.rows < logical_rows || layout.columns < logical_columns) {
        throw std::invalid_argument(
                std::string(name) + " layout is smaller than its logical range");
    }
    const std::uint64_t touched_rows = touch_physical_extent
            ? layout.rows : logical_rows;
    const std::uint64_t touched_columns = touch_physical_extent
            ? layout.columns : logical_columns;
    if (layout.rows == 0 || layout.columns == 0
            || (planes > 1 && layout.plane_stride == 0)
            || (heads > 1 && layout.head_stride == 0)
            || (touched_rows > 1 && layout.row_stride == 0)
            || (touched_columns > 1 && layout.column_stride == 0)) {
        throw std::invalid_argument(
                std::string(name) + " layout has an invalid stride or extent");
    }
    std::uint64_t last = layout.plane_offset;
    if (planes != 0) {
        last = checked_add(
                last,
                checked_mul(
                        planes - 1, layout.plane_stride,
                        "SDPA layout plane offset overflows"),
                "SDPA layout offset overflows");
    }
    if (heads != 0) {
        last = checked_add(
                last,
                checked_mul(
                        heads - 1, layout.head_stride,
                        "SDPA layout head offset overflows"),
                "SDPA layout offset overflows");
    }
    if (touched_rows != 0) {
        last = checked_add(
                last,
                checked_mul(
                        touched_rows - 1, layout.row_stride,
                        "SDPA layout row offset overflows"),
                "SDPA layout offset overflows");
    }
    if (touched_columns != 0) {
        last = checked_add(
                last,
                checked_mul(
                        touched_columns - 1, layout.column_stride,
                        "SDPA layout column offset overflows"),
                "SDPA layout offset overflows");
    }
    (void)checked_add(last, 1, "SDPA layout extent overflows");
}

void validate_common(const SdpaSoftmaxRequest& request) {
    if (request.planes == 0 || request.hq == 0 || request.hkv == 0
            || request.rows == 0 || request.capacity == 0
            || request.head_dim == 0 || request.L == 0) {
        throw std::invalid_argument(
                "ROCm SDPA softmax dimensions must be nonzero");
    }
    if (request.hq % request.hkv != 0) {
        throw std::invalid_argument(
                "ROCm SDPA softmax query heads are not divisible by KV heads");
    }
    if (request.L > request.capacity || request.a >= request.capacity) {
        throw std::invalid_argument(
                "ROCm SDPA softmax initialized range is invalid");
    }
    if (request.rows > request.capacity - request.a) {
        throw std::invalid_argument(
                "ROCm SDPA softmax rows exceed the causal window");
    }
    (void)checked_add(
            request.a, request.rows - 1,
            "ROCm SDPA softmax causal position overflows");
    (void)checked_mul(
            request.hq, request.head_dim,
            "ROCm SDPA softmax output width overflows");
    if (!std::isfinite(request.scale) || request.scale <= 0.0F) {
        throw std::invalid_argument(
                "ROCm SDPA softmax scale must be finite and positive");
    }
    if ((request.mask & sdpa_required_mask) != sdpa_required_mask
            || (request.mask & ~sdpa_required_mask) != 0) {
        throw std::invalid_argument(
                "ROCm SDPA softmax requires initialized-prefix and causal masks");
    }
}

[[nodiscard]] unsigned int launch_blocks(std::uint64_t units) {
    const std::uint64_t rounded = checked_add(
            units, static_cast<std::uint64_t>(kSdpaThreads - 1),
            "ROCm SDPA launch count overflows");
    const std::uint64_t blocks = rounded / kSdpaThreads;
    return static_cast<unsigned int>(
            blocks < kSdpaMaxBlocks ? blocks : kSdpaMaxBlocks);
}

void check_launch(const char* operation) {
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        throw std::runtime_error(
                std::string(operation) + " failed with "
                + hipGetErrorName(status) + ": " + hipGetErrorString(status));
    }
}

}  // namespace

void validate_sdpa_softmax_request(const SdpaSoftmaxRequest& request) {
    validate_common(request);
    if (request.scores == nullptr || request.p_bf16 == nullptr) {
        throw std::invalid_argument(
                "ROCm SDPA softmax scores and probabilities are required");
    }
    validate_layout(
            request.scores_layout, request.planes, request.hq, request.rows,
            request.L, "ROCm SDPA scores", false);
    validate_layout(
            request.probability_layout, request.planes, request.hq,
            request.rows, request.L, "ROCm SDPA probability", true);
}

void validate_sdpa_merge_request(const SdpaSoftmaxRequest& request) {
    validate_common(request);
    if (request.pv_bf16 == nullptr || request.merged == nullptr) {
        throw std::invalid_argument(
                "ROCm SDPA PV and merged output pointers are required");
    }
    const std::uint64_t output_width = checked_mul(
            request.hq, request.head_dim,
            "ROCm SDPA merged output width overflows");
    validate_layout(
            request.pv_layout, request.planes, request.hq, request.rows,
            request.head_dim, "ROCm SDPA PV", false);
    validate_layout(
            request.merged_layout, request.planes, 1, request.rows,
            output_width, "ROCm SDPA merged output", false);
}

void launch_sdpa_softmax(
        hipStream_t stream, const SdpaSoftmaxRequest& request) {
    validate_sdpa_softmax_request(request);
    const std::uint64_t rows_per_plane = checked_mul(
            request.hq, request.probability_layout.rows,
            "ROCm SDPA softmax row count overflows");
    const std::uint64_t total = checked_mul(
            request.planes, rows_per_plane,
            "ROCm SDPA softmax work count overflows");
    hipLaunchKernelGGL(
            sdpa_softmax_kernel, dim3(launch_blocks(total)),
            dim3(kSdpaThreads), 0, stream, request);
    check_launch("ROCm SDPA softmax kernel launch");
}

void launch_sdpa_merge(
        hipStream_t stream, const SdpaSoftmaxRequest& request) {
    validate_sdpa_merge_request(request);
    const std::uint64_t output_width = checked_mul(
            request.hq, request.head_dim,
            "ROCm SDPA merged output width overflows");
    const std::uint64_t rows_per_plane = checked_mul(
            request.rows, output_width,
            "ROCm SDPA merge row count overflows");
    const std::uint64_t total = checked_mul(
            request.planes, rows_per_plane,
            "ROCm SDPA merge work count overflows");
    hipLaunchKernelGGL(
            sdpa_merge_kernel, dim3(launch_blocks(total)),
            dim3(kSdpaThreads), 0, stream, request);
    check_launch("ROCm SDPA merge kernel launch");
}

}  // namespace iom::rocm_detail
