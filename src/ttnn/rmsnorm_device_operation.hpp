#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include <ttnn/common/queue_id.hpp>
#include <ttnn/device_operation.hpp>
#include <ttnn/tensor/tensor.hpp>

// The installed TT-NN package installs only the operation API headers; the
// layer-normalization program descriptor factory that this adapter reuses is
// declared by the SDK's own source at the installed version, so the TTNN build
// must expose that source tree (`${TT_METAL_HOME}/ttnn/cpp`). Every other
// TT-NN and Metalium header comes from the installed package.
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"

namespace tt::tt_metal::distributed {
    class MeshDevice;
}

namespace iom::ttnn_detail {

/**
 * Operands of one preallocated RMSNorm launch.
 *
 * One launch covers exactly one IOM leading plane: the native input plane, the
 * shared `[1,F]` gamma plane, and the caller's output plane. All three are
 * borrowed TTNN handles; `output` is already allocated by the caller and is
 * never replaced, relocated, or re-created by this adapter.
 */
struct RmsNormOperands {
    ttnn::Tensor input;
    ttnn::Tensor scale;
    ttnn::Tensor output;
};

/**
 * Program descriptor factory for the preallocated RMSNorm plane.
 *
 * The factory only forwards to the installed multi-core layer-normalization
 * factory, which owns the native program: the RMS compute kernel, its circular
 * buffers, and every reader, compute, and writer argument (including `eps` and
 * the gamma plane). It exists because the installed operation allocates its own
 * output through `create_device_tensor` while this adapter must bind the
 * caller's preallocated buffer instead.
 */
struct RmsNormPreallocatedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
            const ttnn::prim::LayerNormParams& attributes,
            const RmsNormOperands& operands,
            ttnn::Tensor& output,
            const std::optional<tt::tt_metal::CoreRangeSet>& core_range =
                    std::nullopt);
};

/**
 * Device operation that returns the caller's output tensor.
 *
 * It reuses the installed parameter, input, validation, and program-factory
 * types verbatim and overrides only the output-binding seam: custom
 * `create_output_tensors` hands the supplied output back, so the writer kernel
 * observes the caller's own `output.buffer()` and no SDK output storage is
 * allocated. The operation is backend-private; no IOM or TTNN public API is
 * extended by it.
 */
struct RmsNormPreallocatedDeviceOperation {
    using operation_attributes_t = ttnn::prim::LayerNormParams;
    using tensor_args_t = RmsNormOperands;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<RmsNormPreallocatedProgramFactory>;

    static program_factory_t select_program_factory(
            const operation_attributes_t& attributes,
            const tensor_args_t& operands);

    static void validate_on_program_cache_miss(
            const operation_attributes_t& attributes,
            const tensor_args_t& operands);

    static spec_return_value_t compute_output_specs(
            const operation_attributes_t& attributes,
            const tensor_args_t& operands);

    static tensor_return_value_t create_output_tensors(
            const operation_attributes_t& attributes,
            const tensor_args_t& operands);
};

/**
 * Build the installed layer-normalization attributes of one preallocated
 * RMSNorm plane.
 *
 * The attributes are the complete native contract this adapter hands to the
 * installed factory: `RMSNORM`, no distribution stage, the caller's `eps`, the
 * output plane's memory configuration, the default (non-Welford) program
 * configuration, and the fixed HiFi4/FP32-destination compute configuration.
 */
[[nodiscard]] ttnn::prim::LayerNormParams preallocated_plane_attributes(
        const ttnn::Tensor& input,
        const ttnn::Tensor& output,
        float eps);

/**
 * Validate the native preconditions of one preallocated RMSNorm plane and
 * reject everything else before any native work exists.
 *
 * Only structural and native-storage facts are checked: TILE layout and
 * standard 32x32 tiles, interleaved (non-sharded) storage, BF16 or F32 with
 * matching operand dtypes, the exact `[1,R,F]` input plane with a `[1,F]`
 * gamma plane and a shape-identical output plane, one selected device,
 * preallocated and disjoint output, tile-aligned padded extents covered by the
 * native buffers, and a finite nonnegative `eps`. Common IOM admission, error
 * precedence, and queue behavior belong to the caller.
 */
void validate_preallocated_rmsnorm(
        const ttnn::Tensor& input,
        const ttnn::Tensor& scale,
        const ttnn::Tensor& output,
        float eps,
        const tt::tt_metal::distributed::MeshDevice* device);

/**
 * Submit one asynchronous native RMSNorm launch for one IOM leading plane.
 *
 * `device` is the selected TTNN context, `command_queue` the selected native
 * command queue on that context. The launch reads the supplied input and gamma
 * planes, writes the preallocated caller output plane in place, and returns
 * the submitted operation's output handle, which is the supplied output
 * handle. It enqueues exactly one native launch, does not wait for completion,
 * allocates no output, operand, or staging storage, and consumes no external
 * scratch.
 */
[[nodiscard]] ttnn::Tensor launch_preallocated_rmsnorm(
        tt::tt_metal::distributed::MeshDevice& device,
        ttnn::QueueId command_queue,
        const ttnn::Tensor& input,
        const ttnn::Tensor& scale,
        ttnn::Tensor& output,
        float eps);

}  // namespace iom::ttnn_detail