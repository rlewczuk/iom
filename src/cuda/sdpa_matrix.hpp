#pragma once

#include "sdpa_matrix_request.hpp"

namespace iom::cuda_detail {

// Capability is a property of the selected CUDA device, not a host fallback
// decision.  The native BF16/FP32 WMMA route is available only at CC 8.0 or
// newer; a negative ordinal means use the current CUDA device.
[[nodiscard]] bool sdpa_matrix_wmma_supported(
        std::int32_t device_ordinal = -1) noexcept;

void launch_sdpa_qk(cudaStream_t stream, const SdpaQkRequest& request);
void launch_sdpa_pv(cudaStream_t stream, const SdpaPvRequest& request);

}  // namespace iom::cuda_detail
