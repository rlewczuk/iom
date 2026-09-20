#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace iom::cuda_detail {

// The public SDPA rank is at most eight.  The final three axes are
// [head,row,feature], so at most five independent leading planes remain.
inline constexpr std::size_t kSdpaMatrixMaxLeadingRank = 5;

// A value-owned snapshot of one standard-tiled operand.  `plane_offset` and
// `plane_strides` are in owner-plane units, not bytes; `head_stride` is the
// stride of the operand's explicit head axis.  No TensorView or borrowed
// shape metadata crosses the private matrix-stage boundary.
struct SdpaMatrixOperand {
    const unsigned char* data = nullptr;
    std::uint64_t plane_offset = 0;
    std::uint64_t plane_strides[kSdpaMatrixMaxLeadingRank]{};
    std::uint64_t head_stride = 0;
};

// The merged output has only the common leading tuple before [R,Hq*D].
struct SdpaMatrixOutput {
    unsigned char* data = nullptr;
    std::uint64_t plane_offset = 0;
    std::uint64_t plane_strides[kSdpaMatrixMaxLeadingRank]{};
};

// Checked, explicit dimensions shared by the QK and PV requests.  Rp/Lp/Dp
// are the fixed 16-cell physical extents used only for native staging and the
// caller-owned fixed score/probability layouts.
struct SdpaMatrixShape {
    std::uint32_t leading_rank = 0;
    std::uint64_t leading_dimensions[kSdpaMatrixMaxLeadingRank]{};
    std::uint64_t planes = 1;
    std::uint64_t Hq = 0;
    std::uint64_t Hkv = 0;
    std::uint64_t R = 0;
    std::uint64_t C = 0;
    std::uint64_t D = 0;
    std::uint64_t a = 0;
    std::uint64_t L = 0;
    std::uint64_t grouping = 0;
    std::uint64_t output_width = 0;
    std::uint64_t Rp = 0;
    std::uint64_t Lp = 0;
    std::uint64_t Dp = 0;
    // -1 permits a caller that already established the current CUDA device;
    // nonnegative values make the stage reject pointers from another device.
    std::int32_t device_ordinal = -1;
};

struct SdpaQkRequest {
    SdpaMatrixShape shape{};
    SdpaMatrixOperand q{};
    SdpaMatrixOperand k{};
    float* scores = nullptr;
};

struct SdpaPvRequest {
    SdpaMatrixShape shape{};
    const unsigned char* probability = nullptr;
    SdpaMatrixOperand v{};
    SdpaMatrixOutput out{};
};


}  // namespace iom::cuda_detail
