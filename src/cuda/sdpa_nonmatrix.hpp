#pragma once

#include <cuda_runtime_api.h>

#include "sdpa_nonmatrix_request.hpp"

namespace iom::cuda_detail {

// Validate dimensions, fixed segment geometry, exact-device pointers, and the
// snapshotted output range before any stage is launched.
void validate_sdpa_nonmatrix_request(const SdpaNonmatrixRequest& request);

// Scale only formed scores and apply the logical causal prefix predicate.
void launch_sdpa_scale_mask(
        cudaStream_t stream, const SdpaNonmatrixRequest& request);

// Reduce each logical row in FP32, apply the frozen special-value policy, and
// encode included probabilities to BF16 with round-to-nearest-even.
void launch_sdpa_softmax(
        cudaStream_t stream, const SdpaNonmatrixRequest& request);

// Canonicalize only the logical merged output [P,R,Hq*D]: numerical zero is
// +0 and a NaN is a quiet canonical BF16 NaN. Capacity and tile tails remain
// untouched.
void launch_sdpa_canonicalize_output(
        cudaStream_t stream, const SdpaNonmatrixRequest& request);

}  // namespace iom::cuda_detail
