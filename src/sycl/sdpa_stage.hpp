#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>

#include "iom/iom.hpp"
#include "sdpa_matrix.hpp"
#include "sdpa_nonmatrix.hpp"

namespace iom::sycl_detail {

// Immutable, fully lowered plan of one admitted SDPA request.  Every pointer is
// a device address derived from a value-copied view snapshot and the leased
// caller workspace; the two stage families receive only these records, so no
// borrowed `TensorView`, owner, or workspace reference survives admission.
//
// The plan is built, validated, and owned by the queue: `lower_sdpa_stages`
// performs every checked stride/pointer decision before owner registration,
// OID publication, output mutation, or native submission, and
// `launch_sdpa_stages` then submits the four in-order device stages on the
// queue's own in-order stream.
struct SdpaStagePlan final {
    SdpaQkRequest qk{};
    SdpaPvRequest pv{};
    SdpaNonmatrixRequest nonmatrix{};
};

// Checked, conservative alignment-32 caller-workspace requirement of the joined
// stages: distinct FP32 scores, BF16 probabilities, FP32 PV, and BF16 head
// staging, every segment 32-byte aligned and disjoint.  Pure: it performs no
// device query, allocation, registration, lease, sequence, submission, or
// operand access, so the common facade can consult it for both the call and the
// pure requirement query.  Checked-arithmetic overflow throws
// `std::overflow_error`.
[[nodiscard]] WorkspaceRequirements sdpa_stage_workspace_requirements(
        const DeviceOps::SdpaRequest& request);

// Lower one admitted request into the joined stage plan.  The exact padded
// scratch geometry, the three independent transformed operand leading-plane
// maps, the head grouping, the causal range, and the standard tiled output
// plane are all recomputed here in checked arithmetic and validated by both
// stage validators, so a request either becomes a complete plan or throws
// before any queue effect.  Performs no allocation, registration, lease,
// sequence, output mutation, or device work.
[[nodiscard]] SdpaStagePlan lower_sdpa_stages(
        const DeviceOps::SdpaRequest& request);

// Submit the joined sequence on one in-order queue and return the event of the
// last stage: native `ext_intel_matrix` subgroup-16 BF16/BF16-to-FP32 QK over
// the caller operands, device-local scale/mask/stable softmax/BF16 probability
// preparation, native PV over the DAZ-prepared V operand, then the single BF16
// RNE head merge into the caller's standard tiled output plane.  The queue is
// in order, so every stage is ordered by its explicit dependency on the
// previous one; no stage waits on the host.
[[nodiscard]] sycl::event launch_sdpa_stages(
        sycl::queue& queue, const SdpaStagePlan& plan);

}  // namespace iom::sycl_detail
