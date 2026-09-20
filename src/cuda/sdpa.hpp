#pragma once

#include "sdpa_matrix.hpp"
#include "sdpa_nonmatrix.hpp"

#include "iom/iom.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace iom::cuda_detail {

// Test-only fault injection is kept behind the integration seam so a single
// accepted queue task can fail after the native QK launch. Production builds
// observe a no-op checkpoint.
void sdpa_stage_checkpoint();

namespace sdpa_integration_detail {

[[nodiscard]] inline std::size_t checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] inline std::size_t checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (left != 0
            && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] inline std::size_t pad16(
        std::size_t value, const char* message) {
    return checked_mul(
            checked_add(value, std::size_t{15}, message) / 16,
            std::size_t{16}, message);
}

[[nodiscard]] inline std::size_t align32(
        std::size_t value, const char* message) {
    return checked_mul(
            checked_add(value, std::size_t{31}, message) / 32,
            std::size_t{32}, message);
}

}  // namespace sdpa_integration_detail

// Exact checked scratch geometry of one CUDA SDPA submission: the complete
// leading tuple, the three 16-cell padded matrix extents the private stages
// address, and the two segments of the single aligned caller workspace. The
// FP32 score segment starts at offset zero and the BF16 probability segment
// starts exactly at `score_bytes`, so the whole range is one 32-byte-aligned
// two-segment layout.
struct SdpaScratch {
    std::uint64_t planes = 0;
    std::uint64_t Rp = 0;
    std::uint64_t Lp = 0;
    std::uint64_t Dp = 0;
    std::size_t score_bytes = 0;
    std::size_t probability_bytes = 0;
};

// The common SDPA request is a nested type of `DeviceOps` that this backend
// namespace cannot name, so every entry point below is deducible over the
// caller's own request type: the shared queue passes its `const SdpaRequest&`
// straight through, and this header never declares a second public operation
// ABI. The consumed members are the validated snapshot fields only.
//
// Pure and deterministic: no allocation, registration, lease, OID/queue
// resource, device read, submission, or occupancy dependence. `sdpa_scratch`
// is the single source of the layout arithmetic behind both the pure query
// and the stage lowering, so the two can never drift apart. Checked
// arithmetic overflow reports `std::overflow_error`; a request that cannot be
// situated at all reports `std::invalid_argument`.
template <typename Request>
[[nodiscard]] SdpaScratch sdpa_scratch(const Request& request) {
    using namespace sdpa_integration_detail;

    if (request.q.rank < 3) {
        throw std::invalid_argument(
                "CUDA SDPA scratch requires a ranked query view");
    }
    const std::size_t leading_rank = request.q.rank - 3;
    const std::span<const std::size_t> dimensions =
            request.q.shape_dimensions();
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        planes = checked_mul(
                planes, dimensions[axis],
                "CUDA SDPA scratch plane count overflows");
    }

    SdpaScratch scratch{};
    scratch.planes = static_cast<std::uint64_t>(planes);
    scratch.Rp = static_cast<std::uint64_t>(pad16(
            request.R, "CUDA SDPA scratch padded row extent overflows"));
    scratch.Lp = static_cast<std::uint64_t>(pad16(
            request.L, "CUDA SDPA scratch padded key extent overflows"));
    // The feature extent is not one of the two segments, but its native
    // 16-cell staging extent is part of this layout: a successful query must
    // never leave an unchecked matrix dimension for the accepted stage chain.
    scratch.Dp = static_cast<std::uint64_t>(pad16(
            request.D, "CUDA SDPA scratch padded feature extent overflows"));

    const std::size_t score_elements = checked_mul(
            checked_mul(
                    checked_mul(
                            planes, request.Hq,
                            "CUDA SDPA scratch score extent overflows"),
                    static_cast<std::size_t>(scratch.Rp),
                    "CUDA SDPA scratch score extent overflows"),
            static_cast<std::size_t>(scratch.Lp),
            "CUDA SDPA scratch score extent overflows");
    scratch.score_bytes = align32(
            checked_mul(
                    score_elements, sizeof(float),
                    "CUDA SDPA scratch score bytes overflow"),
            "CUDA SDPA scratch score alignment overflows");
    scratch.probability_bytes = align32(
            checked_mul(
                    score_elements, sizeof(std::uint16_t),
                    "CUDA SDPA scratch probability bytes overflow"),
            "CUDA SDPA scratch probability alignment overflows");
    return scratch;
}

// The CUDA policy's pure workspace-requirements hook: the combined checked
// size of the two segments at the fixed 32-byte alignment. The probability
// segment begins at the aligned score segment, so one alignment covers both.
template <typename Request>
[[nodiscard]] WorkspaceRequirements sdpa_scratch_requirements(
        const Request& request) {
    using namespace sdpa_integration_detail;
    const SdpaScratch scratch = sdpa_scratch(request);
    return {
            checked_add(
                    scratch.score_bytes, scratch.probability_bytes,
                    "CUDA SDPA scratch bytes overflow"),
            32};
}

// Convert the common immutable request into the two private CUDA stage
// contracts and enqueue all stages on one already-created in-order stream.
// The chain owns only stage construction and native calls; owner retention,
// the workspace lease, and the completion event stay with the shared queue.
template <typename Request>
void launch_sdpa_stages(cudaStream_t stream, const Request& request) {
    using namespace sdpa_integration_detail;

    if (request.q.rank > 8 || request.k.rank != request.q.rank
            || request.v.rank != request.q.rank
            || request.out.rank + 1 != request.q.rank) {
        throw std::invalid_argument(
                "CUDA SDPA integration request has inconsistent ranks");
    }
    const std::size_t leading_rank = request.q.rank - 3;
    if (leading_rank > kSdpaMatrixMaxLeadingRank) {
        throw std::invalid_argument(
                "CUDA SDPA integration leading rank exceeds fixed capacity");
    }

    const SdpaScratch scratch = sdpa_scratch(request);
    const std::size_t required_bytes = checked_add(
            scratch.score_bytes, scratch.probability_bytes,
            "CUDA SDPA integration workspace bytes overflow");
    if (request.workspace_requirements.bytes != required_bytes
            || request.workspace_requirements.alignment != 32) {
        throw std::invalid_argument(
                "CUDA SDPA integration workspace layout is inconsistent");
    }
    void* const workspace_address =
            detail::WorkspaceValidation::address(request.workspace);
    if (workspace_address == nullptr
            || request.workspace.byte_size() < required_bytes) {
        throw std::invalid_argument(
                "CUDA SDPA integration workspace is unavailable");
    }

    const std::span<const std::size_t> q_dimensions =
            request.q.shape_dimensions();
    const std::uint32_t ordinal =
            request.q.device_identity == nullptr
            ? std::numeric_limits<std::uint32_t>::max()
            : request.q.device_identity->backend_device();
    if (ordinal > static_cast<std::uint32_t>(
                          std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
                "CUDA SDPA device ordinal exceeds matrix ABI");
    }

    SdpaMatrixShape shape{};
    shape.leading_rank = static_cast<std::uint32_t>(leading_rank);
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        shape.leading_dimensions[axis] =
                static_cast<std::uint64_t>(q_dimensions[axis]);
    }
    shape.planes = scratch.planes;
    shape.Hq = static_cast<std::uint64_t>(request.Hq);
    shape.Hkv = static_cast<std::uint64_t>(request.Hkv);
    shape.R = static_cast<std::uint64_t>(request.R);
    shape.C = static_cast<std::uint64_t>(request.C);
    shape.D = static_cast<std::uint64_t>(request.D);
    shape.a = static_cast<std::uint64_t>(request.a);
    shape.L = static_cast<std::uint64_t>(request.L);
    shape.grouping = static_cast<std::uint64_t>(request.grouping);
    shape.output_width = static_cast<std::uint64_t>(request.output_width);
    shape.Rp = scratch.Rp;
    shape.Lp = scratch.Lp;
    shape.Dp = scratch.Dp;
    shape.device_ordinal = static_cast<std::int32_t>(ordinal);

    const auto matrix_operand = [leading_rank](const auto& snapshot) {
        SdpaMatrixOperand operand{};
        operand.data = static_cast<const unsigned char*>(
                snapshot.native_handle);
        operand.plane_offset = static_cast<std::uint64_t>(
                snapshot.plane_offset);
        for (std::size_t axis = 0; axis < leading_rank; ++axis) {
            operand.plane_strides[axis] = static_cast<std::uint64_t>(
                    snapshot.plane_strides[axis]);
        }
        operand.head_stride = static_cast<std::uint64_t>(
                snapshot.plane_strides[leading_rank]);
        return operand;
    };

    SdpaMatrixOutput output{};
    output.data = static_cast<unsigned char*>(request.out.native_handle);
    output.plane_offset = static_cast<std::uint64_t>(request.out.plane_offset);
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        output.plane_strides[axis] = static_cast<std::uint64_t>(
                request.out.plane_strides[axis]);
    }

    auto* const workspace_bytes =
            static_cast<unsigned char*>(workspace_address);
    float* const scores = reinterpret_cast<float*>(workspace_bytes);
    unsigned char* const probability = workspace_bytes + scratch.score_bytes;

    SdpaQkRequest qk{};
    qk.shape = shape;
    qk.q = matrix_operand(request.q);
    qk.k = matrix_operand(request.k);
    qk.scores = scores;
    // The launch fault point precedes the native QK launch, so an injected
    // fault observes the same pre-launch error the copy, gather, and cache
    // families already report, inside one already-accepted queue task.
    sdpa_stage_checkpoint();
    launch_sdpa_qk(stream, qk);

    SdpaNonmatrixRequest nonmatrix{};
    nonmatrix.scores = scores;
    nonmatrix.probability = probability;
    nonmatrix.output.data = output.data;
    nonmatrix.output.plane_offset = output.plane_offset;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        nonmatrix.output.plane_strides[axis] = output.plane_strides[axis];
        nonmatrix.leading_dimensions[axis] = shape.leading_dimensions[axis];
    }
    nonmatrix.output.storage_bytes = static_cast<std::uint64_t>(
            request.out.storage_bytes);
    nonmatrix.leading_rank = static_cast<std::uint32_t>(leading_rank);
    nonmatrix.planes = scratch.planes;
    nonmatrix.Hq = static_cast<std::uint64_t>(request.Hq);
    nonmatrix.Hkv = static_cast<std::uint64_t>(request.Hkv);
    nonmatrix.R = static_cast<std::uint64_t>(request.R);
    nonmatrix.C = static_cast<std::uint64_t>(request.C);
    nonmatrix.D = static_cast<std::uint64_t>(request.D);
    nonmatrix.a = static_cast<std::uint64_t>(request.a);
    nonmatrix.L = static_cast<std::uint64_t>(request.L);
    nonmatrix.grouping = static_cast<std::uint64_t>(request.grouping);
    nonmatrix.Rp = scratch.Rp;
    nonmatrix.Lp = scratch.Lp;
    nonmatrix.device_ordinal = ordinal;
    nonmatrix.data_type = request.q.data_type;
    nonmatrix.quantization = request.q.quantization;

    launch_sdpa_scale_mask(stream, nonmatrix);
    launch_sdpa_softmax(stream, nonmatrix);

    SdpaPvRequest pv{};
    pv.shape = shape;
    pv.probability = probability;
    pv.v = matrix_operand(request.v);
    pv.out = output;
    launch_sdpa_pv(stream, pv);
    launch_sdpa_canonicalize_output(stream, nonmatrix);
}

}  // namespace iom::cuda_detail
