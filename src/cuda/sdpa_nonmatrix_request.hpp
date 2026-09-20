#pragma once

#include <cstdint>
#include <type_traits>

#include "iom/tensor.hpp"

namespace iom::cuda_detail {

inline constexpr std::uint32_t kSdpaNonmatrixMaxLeadingRank = 5;

// Immutable owner metadata needed by the final merged-output stage.  The
// output pointer is the owner allocation, while plane_offset/plane_strides
// are the snapshotted view coordinates captured before queue submission.
struct SdpaNonmatrixOutput {
    unsigned char* data = nullptr;
    std::uint64_t plane_offset = 0;
    std::uint64_t plane_strides[kSdpaNonmatrixMaxLeadingRank]{};
    std::uint64_t storage_bytes = 0;
};

// Backend-private, value-owned metadata for the CUDA SDPA stages that do not
// perform QK or PV.  Scores and probabilities are the two fixed contiguous
// caller-workspace segments [P,Hq,Rp,Lp].  Output remains in the caller's
// standard 16x16 tiled owner and is addressed only through this snapshot.
// No TensorView, owner, workspace, or queue object crosses this boundary.
struct SdpaNonmatrixRequest {
    float* scores = nullptr;
    unsigned char* probability = nullptr;
    SdpaNonmatrixOutput output{};

    std::uint32_t leading_rank = 0;
    std::uint64_t leading_dimensions[kSdpaNonmatrixMaxLeadingRank]{};
    std::uint64_t planes = 0;
    std::uint64_t Hq = 0;
    std::uint64_t Hkv = 0;
    std::uint64_t R = 0;
    std::uint64_t C = 0;
    std::uint64_t D = 0;
    std::uint64_t a = 0;
    std::uint64_t L = 0;
    std::uint64_t grouping = 0;
    std::uint64_t Rp = 0;
    std::uint64_t Lp = 0;
    std::uint32_t device_ordinal = 0;
    DataType data_type = DataType::BF16;
    QuantizationFormat quantization = QuantizationFormat::NONE;
};

static_assert(std::is_trivially_copyable_v<SdpaNonmatrixOutput>);
static_assert(std::is_trivially_copyable_v<SdpaNonmatrixRequest>);

}  // namespace iom::cuda_detail
