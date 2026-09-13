#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

#include "iom_internal.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace iom {

    using detail::bits_to_bytes;
    using detail::checked_add;
    using detail::checked_mul;
    using detail::kMaxTensorRank;
    using detail::UnsupportedOperation;

    namespace {
        bool recognized_quantization(QuantizationFormat value) noexcept {
            switch (value) {
                case QuantizationFormat::NONE:
                case QuantizationFormat::INT8_SYMMETRIC:
                case QuantizationFormat::INT8_ASYMMETRIC:
                case QuantizationFormat::INT4_SYMMETRIC:
                case QuantizationFormat::INT4_ASYMMETRIC:
                case QuantizationFormat::OCP_MXFP4:
                case QuantizationFormat::OCP_MXFP8_E4M3:
                case QuantizationFormat::OCP_MXFP8_E5M2:
                case QuantizationFormat::NVIDIA_NVFP4:
                case QuantizationFormat::GGML_Q4_0:
                case QuantizationFormat::GGML_Q4_1:
                case QuantizationFormat::GGML_Q5_0:
                case QuantizationFormat::GGML_Q5_1:
                case QuantizationFormat::GGML_Q8_0:
                case QuantizationFormat::GGML_Q2_K:
                case QuantizationFormat::GGML_Q3_K:
                case QuantizationFormat::GGML_Q4_K:
                case QuantizationFormat::GGML_Q5_K:
                case QuantizationFormat::GGML_Q6_K:
                case QuantizationFormat::TT_BFP2:
                case QuantizationFormat::TT_BFP2A:
                case QuantizationFormat::TT_BFP4:
                case QuantizationFormat::TT_BFP4A:
                case QuantizationFormat::TT_BFP8:
                case QuantizationFormat::TT_BFP8A:
                    return true;
            }
            return false;
        }

        bool add_numeric_leaf(DataType value) noexcept {
            switch (value) {
                case DataType::I2: case DataType::U2:
                case DataType::I4: case DataType::U4:
                case DataType::I8: case DataType::U8:
                case DataType::I16: case DataType::U16:
                case DataType::I32: case DataType::U32:
                case DataType::I64: case DataType::U64:
                case DataType::F4_E2M1: case DataType::F6_E2M3:
                case DataType::F6_E3M2: case DataType::F8_E4M3FN:
                case DataType::F8_E5M2: case DataType::F16:
                case DataType::BF16: case DataType::F32:
                case DataType::F64:
                    return true;
                case DataType::BOOL: case DataType::F8_E8M0:
                    return false;
            }
            return false;
        }

        bool div_numeric_leaf(DataType value) noexcept {
            switch (value) {
                case DataType::F4_E2M1: case DataType::F6_E2M3:
                case DataType::F6_E3M2: case DataType::F8_E4M3FN:
                case DataType::F8_E5M2: case DataType::F16:
                case DataType::BF16: case DataType::F32:
                case DataType::F64:
                    return true;
                default:
                    return false;
            }
        }

        void validate_binary_spec(const TensorSpec& spec) {
            (void)detail::leaf_bits(spec.data_type);
            if (!recognized_quantization(spec.quantization)) {
                throw std::invalid_argument("unknown quantization format");
            }
            if (spec.shape.rank() < 2) {
                throw std::invalid_argument("ADD requires rank at least two");
            }
            if (spec.shape.rank() > kMaxTensorRank) {
                throw std::invalid_argument("ADD requires rank at most eight");
            }
            for (const std::size_t dimension : spec.shape.dimensions()) {
                if (dimension == 0) {
                    throw std::invalid_argument(
                            "ADD dimensions must be nonzero");
                }
            }
        }

        std::size_t checked_binary_plane_count(const TensorShape& shape) {
            std::size_t planes = 1;
            const std::span<const std::size_t> dimensions =
                    shape.dimensions();
            for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                planes = checked_mul(
                        planes, dimensions[i],
                        "ADD plane count overflows");
            }
            return planes;
        }

        void validate_binary_view(
                const Device& device, const TensorView& view) {
            const Tensor* owner = view.owner_identity();
            if (owner == nullptr) {
                throw std::invalid_argument("ADD view has no owner");
            }
            if (&view.device() != &device
                    || &owner->view().device() != &device
                    || owner->view().owner_identity() != owner) {
                throw std::invalid_argument(
                        "ADD view owner belongs to another device");
            }
            const void* handle = view.native_handle();
            if (handle == nullptr
                    || handle != owner->view().native_handle()) {
                throw std::invalid_argument(
                        "ADD view has no stable owner handle");
            }

            const TensorSpec& spec = view.spec();
            const TensorSpec& owner_spec = owner->view().spec();
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::span<const std::size_t> owner_dimensions =
                    owner_spec.shape.dimensions();
            if (spec.data_type != owner_spec.data_type
                    || spec.quantization != owner_spec.quantization
                    || dimensions[dimensions.size() - 2]
                            != owner_dimensions[owner_dimensions.size() - 2]
                    || dimensions.back() != owner_dimensions.back()) {
                throw std::invalid_argument(
                        "ADD view specification does not match its owner");
            }

            const std::size_t leading = dimensions.size() - 2;
            const std::span<const std::size_t> strides =
                    view.plane_strides();
            if (strides.size() != leading) {
                throw std::invalid_argument("invalid ADD plane stride count");
            }
            std::size_t max_plane = view.plane_offset();
            for (std::size_t i = 0; i < leading; ++i) {
                if (strides[i] == 0) {
                    throw std::invalid_argument(
                            "ADD does not permit zero strides");
                }
                max_plane = checked_add(
                        max_plane,
                        checked_mul(
                                dimensions[i] - 1, strides[i],
                                "ADD plane address overflows"),
                        "ADD plane address overflows");
            }
            if (max_plane >= checked_binary_plane_count(owner_spec.shape)) {
                throw std::invalid_argument(
                        "ADD view addresses outside its owner");
            }

            const std::size_t last_slot = detail::standard_plane_slot(
                    spec, max_plane, dimensions[leading] - 1,
                    dimensions[leading + 1] - 1);
            const std::size_t addressed_bits = checked_mul(
                    checked_add(last_slot, 1, "ADD slot count overflows"),
                    detail::leaf_bits(spec.data_type),
                    "ADD view size overflows");
            const std::size_t addressed_bytes =
                    bits_to_bytes(addressed_bits, "ADD view byte size overflows");
            const std::size_t owner_bits = checked_mul(
                    owner_spec.standard_padded_shape().element_count(),
                    detail::leaf_bits(owner_spec.data_type),
                    "ADD owner storage size overflows");
            if (addressed_bytes
                    > bits_to_bytes(
                            owner_bits, "ADD owner storage size overflows")) {
                throw std::invalid_argument(
                        "ADD view exceeds its owner storage");
            }
            const std::size_t logical_bits = checked_mul(
                    spec.shape.element_count(),
                    detail::leaf_bits(spec.data_type),
                    "ADD logical size overflows");
            (void)bits_to_bytes(logical_bits, "ADD logical size overflows");
        }
    }  // namespace

    DeviceOps::BinaryViewSnapshot DeviceOps::snapshot_binary_view(
            const TensorView& view,
            std::span<const std::size_t> result_dimensions) {
        const std::span<const std::size_t> dimensions =
                view.spec().shape.dimensions();
        const std::size_t rank_offset =
                result_dimensions.size() - dimensions.size();
        const std::size_t result_leading =
                result_dimensions.size() - 2;
        std::vector<std::size_t> logical_plane_strides(
                result_leading, 0);
        bool broadcasts = dimensions.size() != result_dimensions.size();
        for (std::size_t axis = 0; axis < result_leading; ++axis) {
            if (axis < rank_offset) {
                broadcasts = broadcasts || result_dimensions[axis] != 1;
                continue;
            }
            const std::size_t source_axis = axis - rank_offset;
            if (dimensions[source_axis] == 1
                    && result_dimensions[axis] != 1) {
                broadcasts = true;
                continue;
            }
            logical_plane_strides[axis] =
                    view.plane_strides()[source_axis];
        }
        const bool broadcast_rows =
                dimensions[dimensions.size() - 2] == 1
                && result_dimensions[result_dimensions.size() - 2] != 1;
        const bool broadcast_columns =
                dimensions.back() == 1
                && result_dimensions.back() != 1;
        broadcasts = broadcasts || broadcast_rows || broadcast_columns;
        return DeviceOps::BinaryViewSnapshot{
                view.spec(), &view.device(), view.owner_identity(),
                const_cast<void*>(view.native_handle()), view.plane_offset(),
                {view.plane_strides().begin(), view.plane_strides().end()},
                std::move(logical_plane_strides), broadcast_rows,
                broadcast_columns, broadcasts};
    }

    DeviceOps::BinaryRequest DeviceOps::validate_binary(
            const Device& device, BinaryOperation operation,
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        validate_binary_spec(lhs.spec());
        validate_binary_spec(rhs.spec());
        validate_binary_spec(out.spec());
        if (lhs.spec().data_type != rhs.spec().data_type
                || lhs.spec().data_type != out.spec().data_type
                || lhs.spec().quantization != rhs.spec().quantization
                || lhs.spec().quantization != out.spec().quantization) {
            throw std::invalid_argument("binary specifications do not match");
        }
        validate_binary_view(device, lhs);
        validate_binary_view(device, rhs);
        validate_binary_view(device, out);
        const auto lhs_dims = lhs.spec().shape.dimensions();
        const auto rhs_dims = rhs.spec().shape.dimensions();
        const std::size_t rank = std::max(lhs_dims.size(), rhs_dims.size());
        std::vector<std::size_t> result(rank, 1);
        for (std::size_t i = 0; i < rank; ++i) {
            const std::size_t lhs_axis =
                    i < rank - lhs_dims.size() ? 1
                    : lhs_dims[i - (rank - lhs_dims.size())];
            const std::size_t rhs_axis =
                    i < rank - rhs_dims.size() ? 1
                    : rhs_dims[i - (rank - rhs_dims.size())];
            if (lhs_axis != rhs_axis && lhs_axis != 1 && rhs_axis != 1) {
                throw std::invalid_argument(
                        "binary shapes are not broadcast compatible");
            }
            result[i] = std::max(lhs_axis, rhs_axis);
        }
        if (out.spec().shape.dimensions().size() != result.size()
                || !std::equal(out.spec().shape.dimensions().begin(),
                               out.spec().shape.dimensions().end(),
                               result.begin())) {
            throw std::invalid_argument("binary output shape is incorrect");
        }
        // The computed broadcast result is itself a full tensor shape: it
        // must be validated before any snapshot, registration, sequence
        // reservation, token acceptance, metadata upload, or backend
        // dispatch exists.
        TensorShape result_shape{result};
        BinaryViewSnapshot lhs_snapshot = snapshot_binary_view(lhs, result);
        BinaryViewSnapshot rhs_snapshot = snapshot_binary_view(rhs, result);
        BinaryViewSnapshot out_snapshot = snapshot_binary_view(out, result);
        const auto exact_alias = [](const BinaryViewSnapshot& input,
                                    const BinaryViewSnapshot& output) {
            return !input.broadcasts
                    && input.owner_identity == output.owner_identity
                    && input.spec == output.spec
                    && input.plane_offset == output.plane_offset
                    && input.plane_strides == output.plane_strides
                    && input.logical_plane_strides
                            == output.logical_plane_strides
                    && input.broadcast_rows == output.broadcast_rows
                    && input.broadcast_columns == output.broadcast_columns;
        };
        if ((lhs_snapshot.owner_identity == out_snapshot.owner_identity
                    && !exact_alias(lhs_snapshot, out_snapshot))
                || (rhs_snapshot.owner_identity == out_snapshot.owner_identity
                    && !exact_alias(rhs_snapshot, out_snapshot))) {
            throw std::invalid_argument("binary input/output alias is forbidden");
        }
        if (lhs.spec().quantization != QuantizationFormat::NONE
                || lhs.spec().data_type == DataType::BOOL
                || lhs.spec().data_type == DataType::F8_E8M0
                || !add_numeric_leaf(lhs.spec().data_type)
                || (operation == BinaryOperation::Div
                    && !div_numeric_leaf(lhs.spec().data_type))) {
            throw UnsupportedOperation();
        }
        return BinaryRequest{operation, std::move(lhs_snapshot),
                             std::move(rhs_snapshot), std::move(out_snapshot),
                             std::move(result_shape), {}, {0, 1}, {}};
    }

    oid DeviceOps::binary_impl(const BinaryRequest&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::submit_binary_operation(
            BinaryOperation operation, const TensorView& lhs,
            const TensorView& rhs, TensorView& out,
            RawWorkspaceView workspace) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), operation, lhs, rhs, out);
            const WorkspaceRequirements requirements =
                    binary_workspace_requirements(request);
            const std::array<TensorView, 3> operands{lhs, rhs, out};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            queue_device(), workspace,
                            requirements.bytes, requirements.alignment,
                            operands);
            return invoke(binary_impl(BinaryRequest{
                    request.operation, request.lhs, request.rhs, request.out,
                    request.result_shape, validated_workspace, requirements,
                    {}}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::add(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        return submit_binary_operation(
                BinaryOperation::Add, lhs, rhs, out, workspace);
    }

    oid DeviceOps::mul(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        return submit_binary_operation(
                BinaryOperation::Mul, lhs, rhs, out, workspace);
    }

    oid DeviceOps::sub(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        return submit_binary_operation(
                BinaryOperation::Sub, lhs, rhs, out, workspace);
    }

    oid DeviceOps::div(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        return submit_binary_operation(
                BinaryOperation::Div, lhs, rhs, out, workspace);
    }

    WorkspaceRequirements
    DeviceOps::binary_workspace_requirements(const BinaryRequest&) {
        // CPU, TTNN, CUDA, and ROCm need no raw workspace for the binary
        // operations. SYCL overrides this hook with its checked
        // whole-plane staging sum.
        return {0, 1};
    }

    WorkspaceRequirements DeviceOps::add_workspace_requirements(
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        const BinaryRequest request = validate_binary(
                queue_device(), BinaryOperation::Add, lhs, rhs, out);
        return binary_workspace_requirements(request);
    }

    WorkspaceRequirements DeviceOps::mul_workspace_requirements(
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        const BinaryRequest request = validate_binary(
                queue_device(), BinaryOperation::Mul, lhs, rhs, out);
        return binary_workspace_requirements(request);
    }

    WorkspaceRequirements DeviceOps::sub_workspace_requirements(
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        const BinaryRequest request = validate_binary(
                queue_device(), BinaryOperation::Sub, lhs, rhs, out);
        return binary_workspace_requirements(request);
    }

    WorkspaceRequirements DeviceOps::div_workspace_requirements(
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        const BinaryRequest request = validate_binary(
                queue_device(), BinaryOperation::Div, lhs, rhs, out);
        return binary_workspace_requirements(request);
    }

}  // namespace iom
