#include "rmsnorm_device_operation.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>
#include <ttnn/core.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/normalization/layernorm/device/layernorm_common.hpp>

namespace iom::ttnn_detail {

    namespace {
        // IOM native planes use the fixed standard tile, the same tile the
        // installed multi-core layer-normalization factory assumes.
        constexpr std::uint32_t kStandardTileExtent = 32;

        [[nodiscard]] std::string native_dtype_name(
                tt::tt_metal::DataType dtype) {
            std::ostringstream stream;
            stream << dtype;
            return stream.str();
        }

        [[nodiscard]] std::invalid_argument invalid_plane(
                const char* role, std::string reason) {
            return std::invalid_argument(
                    std::string("RMSNorm ") + role + " " + reason);
        }

        // Structural and native-storage facts shared by every operand, checked
        // before any native program object exists.
        void validate_operand_storage(
                const ttnn::Tensor& tensor,
                const char* role,
                const tt::tt_metal::distributed::MeshDevice* device) {
            if (tensor.storage_type() != ttnn::StorageType::DEVICE
                    || !tensor.is_allocated() || tensor.buffer() == nullptr) {
                throw invalid_plane(
                        role, "must be an allocated device tensor");
            }
            if (tensor.device() != device) {
                throw invalid_plane(
                        role,
                        "must belong to the selected TTNN device");
            }
            if (tensor.is_sharded()) {
                throw invalid_plane(
                        role,
                        "must be interleaved; sharded storage is unsupported");
            }
            if (tensor.layout() != tt::tt_metal::Layout::TILE) {
                throw invalid_plane(role, "must use the TILE layout");
            }
            const tt::tt_metal::Tile& tile = tensor.tensor_spec().tile();
            if (tile.get_height() != kStandardTileExtent
                    || tile.get_width() != kStandardTileExtent) {
                throw invalid_plane(
                        role,
                        "must use the standard 32x32 tile of a TILE plane");
            }
            const tt::tt_metal::DataType dtype = tensor.dtype();
            if (dtype != tt::tt_metal::DataType::BFLOAT16
                    && dtype != tt::tt_metal::DataType::FLOAT32) {
                throw invalid_plane(
                        role,
                        "must be BF16 or F32, got "
                                + native_dtype_name(dtype));
            }
            // Every padded extent of a TILE plane is a whole number of tiles,
            // and the native buffer must already cover all of them: a smaller
            // or absent allocation would make the writer address memory
            // outside the caller's plane.
            const tt::tt_metal::Shape& padded = tensor.padded_shape();
            if (padded[-2] % kStandardTileExtent != 0
                    || padded[-1] % kStandardTileExtent != 0) {
                throw invalid_plane(
                        role,
                        "must have tile-aligned padded extents");
            }
            const std::uint64_t padded_cells = padded.volume();
            const std::uint32_t element_bytes = tensor.element_size();
            if (padded_cells
                    > std::numeric_limits<std::uint64_t>::max()
                            / element_bytes) {
                throw std::overflow_error(
                        std::string("RMSNorm ") + role
                        + " padded byte count overflows");
            }
            const std::uint64_t padded_bytes =
                    padded_cells * element_bytes;
            if (tensor.buffer()->size() < padded_bytes) {
                throw invalid_plane(
                        role,
                        "native buffer is smaller than its padded plane; the "
                        "operand must be preallocated for the whole plane");
            }
        }

        void validate_matching_dtype(
                const ttnn::Tensor& tensor,
                const ttnn::Tensor& input,
                const char* role) {
            if (tensor.dtype() != input.dtype()) {
                throw invalid_plane(
                        role,
                        "must match the input dtype, got "
                                + native_dtype_name(tensor.dtype()) + " for "
                                + native_dtype_name(input.dtype())
                                + " input");
            }
        }

        // One IOM leading plane: `[1,R,F]` with nonzero row and feature
        // extents. The last logical axis is the feature width reduced by
        // RMSNorm; tile padding never carries a logical row or feature.
        void validate_input_plane(const ttnn::Tensor& input) {
            const tt::tt_metal::Shape& shape = input.logical_shape();
            if (shape.rank() != 3 || shape[0] != 1) {
                throw invalid_plane(
                        "input",
                        "must be one leading plane shaped [1,R,F]");
            }
            if (shape[1] == 0 || shape[2] == 0) {
                throw invalid_plane(
                        "input", "must have nonzero row and feature extents");
            }
        }

        // The shared `[1,F]` gamma plane: exactly one row, no leading plane of
        // its own, and a feature extent that reaches the input's feature
        // extent in both logical and padded width.
        void validate_scale_plane(
                const ttnn::Tensor& scale, const ttnn::Tensor& input) {
            const tt::tt_metal::Shape& shape = scale.logical_shape();
            const std::size_t rank = shape.rank();
            if (rank != 2 && rank != 3) {
                throw invalid_plane(
                        "scale", "must be one [1,F] gamma plane");
            }
            for (std::size_t index = 0; index + 1 < rank; ++index) {
                if (shape[index] != 1) {
                    throw invalid_plane(
                            "scale",
                            "must carry exactly one row and no leading plane");
                }
            }
            if (shape[-1] != input.logical_shape()[-1]) {
                throw invalid_plane(
                        "scale",
                        "must cover the input feature extent exactly");
            }
            if (scale.padded_shape()[-1] != input.padded_shape()[-1]) {
                throw invalid_plane(
                        "scale",
                        "padded feature extent must match the input plane");
            }
            if (scale.padded_shape()[-2] != kStandardTileExtent) {
                throw invalid_plane(
                        "scale",
                        "must have exactly one padded tile row");
            }
        }

        // The output is the caller's own preallocated plane: identical
        // specification to the input and disjoint from both read operands.
        void validate_output_plane(
                const ttnn::Tensor& output,
                const ttnn::Tensor& input,
                const ttnn::Tensor& scale) {
            if (output.logical_shape() != input.logical_shape()
                    || output.padded_shape() != input.padded_shape()) {
                throw invalid_plane(
                        "output",
                        "must have the input plane's logical and padded "
                        "shape");
            }
            if (output.buffer() == input.buffer()) {
                throw invalid_plane(
                        "output",
                        "must not alias the input; in-place RMSNorm is "
                        "unsupported");
            }
            if (output.buffer() == scale.buffer()) {
                throw invalid_plane(
                        "output", "must not alias the scale plane");
            }
        }

        [[nodiscard]] ttnn::DeviceComputeKernelConfig
                rmsnorm_compute_kernel_config() {
            // Fixed accumulator-domain configuration of the inherited RMSNorm
            // arithmetic: HiFi4 with approximate math disabled and FP32
            // destination accumulation enabled.
            return ttnn::DeviceComputeKernelConfig{
                    .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                    .math_approx_mode = false,
                    .fp32_dest_acc_en = true};
        }
    }  // namespace

    ttnn::prim::LayerNormParams preallocated_plane_attributes(
            const ttnn::Tensor& input,
            const ttnn::Tensor& output,
            float eps) {
        const tt::tt_metal::Tile& tile = input.tensor_spec().tile();
        return ttnn::prim::LayerNormParams{
                .norm_type = ttnn::prim::LayerNormType::RMSNORM,
                .distributed_norm_stage =
                        ttnn::prim::DistributedLayerNormStage::NOT_DISTRIBUTED,
                .eps = eps,
                .output_mem_config = output.memory_config(),
                .program_config = ttnn::prim::create_layernorm_program_config(
                        std::nullopt, tile.get_height(), tile.get_width()),
                .compute_kernel_config = rmsnorm_compute_kernel_config(),
                .dtype = std::nullopt,
                .fused_activation = std::nullopt};
    }

    void validate_preallocated_rmsnorm(
            const ttnn::Tensor& input,
            const ttnn::Tensor& scale,
            const ttnn::Tensor& output,
            float eps,
            const tt::tt_metal::distributed::MeshDevice* device) {
        if (!std::isfinite(eps) || eps < 0.0f) {
            throw std::invalid_argument(
                    "RMSNorm epsilon must be finite and nonnegative");
        }
        validate_operand_storage(input, "input", device);
        validate_operand_storage(scale, "scale", device);
        validate_operand_storage(output, "output", device);
        validate_matching_dtype(output, input, "output");
        validate_matching_dtype(scale, input, "scale");
        validate_input_plane(input);
        validate_scale_plane(scale, input);
        validate_output_plane(output, input, scale);
    }

    tt::tt_metal::ProgramDescriptor
    RmsNormPreallocatedProgramFactory::create_descriptor(
            const ttnn::prim::LayerNormParams& attributes,
            const RmsNormOperands& operands,
            ttnn::Tensor& output,
            const std::optional<tt::tt_metal::CoreRangeSet>& core_range) {
        // The installed factory owns the whole native program. This adapter
        // only supplies the operands and the caller's output tensor, so the
        // RMS compute kernel, its circular buffers, the gamma plane, `eps`,
        // and the compute configuration travel into the native program
        // arguments unchanged.
        return ttnn::prim::LayerNormMultiCoreProgramFactory::create_descriptor(
                attributes,
                ttnn::prim::LayerNormInputs{
                        .input = operands.input,
                        .residual_input_tensor = std::nullopt,
                        .weight = operands.scale,
                        .bias = std::nullopt,
                        .stats = std::nullopt,
                        .recip_tensor = std::nullopt},
                output, core_range);
    }

    RmsNormPreallocatedDeviceOperation::program_factory_t
    RmsNormPreallocatedDeviceOperation::select_program_factory(
            const operation_attributes_t&, const tensor_args_t&) {
        return RmsNormPreallocatedProgramFactory{};
    }

    void RmsNormPreallocatedDeviceOperation::validate_on_program_cache_miss(
            const operation_attributes_t& attributes,
            const tensor_args_t& operands) {
        validate_preallocated_rmsnorm(
                operands.input, operands.scale, operands.output,
                attributes.eps, operands.input.device());
    }

    RmsNormPreallocatedDeviceOperation::spec_return_value_t
    RmsNormPreallocatedDeviceOperation::compute_output_specs(
            const operation_attributes_t&, const tensor_args_t& operands) {
        return operands.output.tensor_spec();
    }

    RmsNormPreallocatedDeviceOperation::tensor_return_value_t
    RmsNormPreallocatedDeviceOperation::create_output_tensors(
            const operation_attributes_t&, const tensor_args_t& operands) {
        // The caller owns the output plane. Handing the supplied handle back
        // is the entire custom output binding: no `create_device_tensor`
        // call, no replacement tensor, and no relocated storage, so the
        // writer binds the caller's own buffer.
        return operands.output;
    }

    ttnn::Tensor launch_preallocated_rmsnorm(
            tt::tt_metal::distributed::MeshDevice& device,
            ttnn::QueueId command_queue,
            const ttnn::Tensor& input,
            const ttnn::Tensor& scale,
            ttnn::Tensor& output,
            float eps) {
        validate_preallocated_rmsnorm(input, scale, output, eps, &device);
        const RmsNormOperands operands{input, scale, output};
        auto queue_scope = ttnn::core::with_command_queue_id(command_queue);
        return ttnn::device_operation::launch<
                RmsNormPreallocatedDeviceOperation>(
                preallocated_plane_attributes(input, output, eps), operands);
    }

}  // namespace iom::ttnn_detail