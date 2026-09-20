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

    using detail::UnsupportedOperation;
    using detail::validate_checked_spec;
    using detail::validate_checked_view;

    namespace {
        // Established diagnostic tag of the four binary operations. The
        // shared checked-view admission path only interpolates it into
        // rejection text.
        constexpr const char* kAdmissionContext = "ADD";

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
        validate_checked_spec(lhs.spec(), kAdmissionContext);
        validate_checked_spec(rhs.spec(), kAdmissionContext);
        validate_checked_spec(out.spec(), kAdmissionContext);
        if (lhs.spec().data_type != rhs.spec().data_type
                || lhs.spec().data_type != out.spec().data_type
                || lhs.spec().quantization != rhs.spec().quantization
                || lhs.spec().quantization != out.spec().quantization) {
            throw std::invalid_argument("binary specifications do not match");
        }
        // The shared checked-view path validates each operand's live
        // owner/bounds and checked arithmetic; the broadcast, alias, and
        // capability rules below stay binary-owned.
        (void)validate_checked_view(device, lhs, kAdmissionContext);
        (void)validate_checked_view(device, rhs, kAdmissionContext);
        (void)validate_checked_view(device, out, kAdmissionContext);
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
        // CPU, CUDA, and ROCm need no raw workspace for the binary
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
