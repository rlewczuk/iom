#include "sdpa.hpp"

#include "copy.hpp"

namespace iom::cuda_detail {

void sdpa_stage_checkpoint() {
    if (consume_submission_fault(SubmissionFault::third_plane_launch)) {
        check_cuda_kernel(
                "CUDA SDPA QK stage launch", cudaErrorInvalidValue);
    }
}

}  // namespace iom::cuda_detail
