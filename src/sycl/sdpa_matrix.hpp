#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace iom::sycl_detail {

// SDPA's native matrix path is deliberately backend-private.  The public
// DeviceOps facade owns validation, workspace leasing, and owner retention;
// these records carry only value-copied device metadata into the two direct
// joint_matrix stages.
inline constexpr std::uint64_t kSdpaMatrixTile = 16;
inline constexpr std::size_t kSdpaMatrixMaxLeadingRank = 6;

struct SdpaMatrixGeometry {
    std::uint64_t planes = 0;       // checked product of leading extents
    std::uint64_t Hq = 0;
    std::uint64_t Hkv = 0;
    std::uint64_t R = 0;
    std::uint64_t C = 0;
    std::uint64_t D = 0;
    std::uint64_t L = 0;
    std::uint64_t a = 0;
    std::uint64_t grouping = 0;
    std::uint64_t leading_rank = 0;
    std::uint64_t leading_dimensions[kSdpaMatrixMaxLeadingRank]{};

    // The three persistent operands use independent transformed leading-plane
    // maps.  The head axis is kept separate from the leading tuple so a stage
    // never infers a plane from cache capacity or physical padding.
    std::uint64_t q_plane_offset = 0;
    std::uint64_t k_plane_offset = 0;
    std::uint64_t v_plane_offset = 0;
    std::uint64_t q_leading_strides[kSdpaMatrixMaxLeadingRank]{};
    std::uint64_t k_leading_strides[kSdpaMatrixMaxLeadingRank]{};
    std::uint64_t v_leading_strides[kSdpaMatrixMaxLeadingRank]{};
    std::uint64_t q_head_stride = 0;
    std::uint64_t k_head_stride = 0;
    std::uint64_t v_head_stride = 0;

    // Physical caller-workspace layouts.  Each matrix is row-major and has a
    // checked 16-element tile padding in both axes.
    std::uint64_t padded_rows = 0;
    std::uint64_t padded_keys = 0;
    std::uint64_t padded_depth = 0;
    std::uint64_t scores_plane_stride = 0;
    std::uint64_t scores_head_stride = 0;
    std::uint64_t scores_row_stride = 0;
    std::uint64_t probability_plane_stride = 0;
    std::uint64_t probability_head_stride = 0;
    std::uint64_t probability_row_stride = 0;
    std::uint64_t pv_plane_stride = 0;
    std::uint64_t pv_head_stride = 0;
    std::uint64_t pv_row_stride = 0;
};

static_assert(std::is_trivially_copyable_v<SdpaMatrixGeometry>);

struct SdpaQkRequest {
    SdpaMatrixGeometry geometry{};
    const unsigned char* q = nullptr;
    const unsigned char* k = nullptr;
    float* scores = nullptr;
};

struct SdpaPvRequest {
    SdpaMatrixGeometry geometry{};
    const unsigned char* p_bf16 = nullptr;
    const unsigned char* v = nullptr;
    float* pv_fp32 = nullptr;
};

struct SdpaMatrixScratchLayout {
    std::size_t scores_offset = 0;
    std::size_t probability_offset = 0;
    std::size_t pv_offset = 0;
    std::size_t head_offset = 0;
    std::size_t bytes = 0;
};

static_assert(std::is_trivially_copyable_v<SdpaQkRequest>);
static_assert(std::is_trivially_copyable_v<SdpaPvRequest>);

// The linear leaf owns the single capability probe for the installed
// subgroup-16 BF16/BF16/FP32 joint_matrix family.  SDPA reuses that exact
// immutable device fact rather than maintaining a second probe.
[[nodiscard]] bool bf16_linear_device_capable(
        const sycl::device& device) noexcept;

[[nodiscard]] bool sdpa_matrix_device_capable(
        const sycl::device& device) noexcept;

// Checked physical workspace layout shared by the queue admission hook and
// the nonmatrix SDPA stages.  It contains FP32 scores, BF16 probabilities,
// FP32 PV, and BF16 head staging in that order, with every segment 32-byte
// aligned and disjoint.
[[nodiscard]] SdpaMatrixScratchLayout sdpa_matrix_scratch_layout(
        std::size_t planes, std::size_t Hq, std::size_t R,
        std::size_t L, std::size_t D);

// Validates the copied geometry before a queue sequence, owner registration,
// lease, output mutation, or native submission.  It performs no allocation or
// device effect and is also useful to a direct stage caller.
void validate_sdpa_matrix_geometry(const SdpaMatrixGeometry& geometry);

// Direct native QK^T and P V stages.  They do not allocate, wait, stage on the
// host, or emulate unsupported matrix combinations.  The queue is in-order;
// callers submit the nonmatrix scale/softmax and merge stages between these
// two calls on the same queue.
[[nodiscard]] sycl::event launch_sdpa_qk(
        sycl::queue& queue, const SdpaQkRequest& request);
[[nodiscard]] sycl::event launch_sdpa_pv(
        sycl::queue& queue, const SdpaPvRequest& request);

}  // namespace iom::sycl_detail
