#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

#include "iom_internal.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace iom {

    using detail::UnsupportedOperation;
    using detail::validate_checked_spec;
    using detail::validate_checked_view;

    namespace {

        // Every supported RMS normalization implementation consumes no raw
        // workspace, so the queried and admitted requirement is exactly
        // `{0, 1}` and the only admissible supplied workspace is the empty
        // default view.
        constexpr WorkspaceRequirements kRmsnormWorkspaceRequirements{0, 1};

        // Established diagnostic tag of RMS normalization admission. The
        // shared checked-operand path only interpolates it into rejection
        // text.
        constexpr const char* kRmsnormContext = "RMSNORM";

        // Exact applicability: the nine ordinary signed floating leaves.
        // Recognized BOOL, integer, and exponent-only `F8_E8M0` leaves are
        // inapplicable everywhere; an unknown enumeration value is rejected
        // as invalid input by the shared spec check before this predicate is
        // consulted.
        bool rmsnorm_leaf(DataType value) noexcept {
            switch (value) {
                case DataType::F4_E2M1:
                case DataType::F6_E2M3:
                case DataType::F6_E3M2:
                case DataType::F8_E4M3FN:
                case DataType::F8_E5M2:
                case DataType::F16:
                case DataType::BF16:
                case DataType::F32:
                case DataType::F64:
                    return true;
                case DataType::BOOL:
                case DataType::I2:
                case DataType::U2:
                case DataType::I4:
                case DataType::U4:
                case DataType::I8:
                case DataType::U8:
                case DataType::I16:
                case DataType::U16:
                case DataType::I32:
                case DataType::U32:
                case DataType::I64:
                case DataType::U64:
                case DataType::F8_E8M0:
                    return false;
            }
            return false;
        }

        // Complete common admission validation in the frozen contract order.
        // Operation-neutral operand facts go through the shared
        // allocation-free checked-admission path exactly as the binary
        // operations do; the RMSNorm-owned layout, matching, alias,
        // applicability, and epsilon rules stay here. Allocation-free and
        // effect-free: it inspects live metadata only, so the pure
        // requirement query shares exactly these checks without building a
        // request or any owned snapshot.
        void validate_rmsnorm(
                const Device& device, const TensorView& x,
                const TensorView& scale, const TensorView& out, float eps) {
            validate_checked_spec(x.spec(), kRmsnormContext);
            validate_checked_spec(scale.spec(), kRmsnormContext);
            validate_checked_spec(out.spec(), kRmsnormContext);
            if (!(x.spec().shape == out.spec().shape)) {
                throw std::invalid_argument(
                        "RMSNORM x and out must have identical shape");
            }
            const std::span<const std::size_t> scale_dimensions =
                    scale.spec().shape.dimensions();
            if (scale.spec().shape.rank() != 2
                    || scale_dimensions[0] != 1
                    || scale_dimensions[1]
                            != x.spec().shape.dimensions().back()) {
                throw std::invalid_argument(
                        "RMSNORM scale must be exactly [1,F]");
            }
            // The shared checked-view path validates each operand's exact
            // device identity, live owner and stable handle, leading bounds
            // and strides, and checked plane, tile, element, and byte
            // arithmetic for the addressed range, owner storage, and logical
            // payload.
            (void)validate_checked_view(device, x, kRmsnormContext);
            (void)validate_checked_view(device, scale, kRmsnormContext);
            (void)validate_checked_view(device, out, kRmsnormContext);
            if (x.spec().data_type != scale.spec().data_type
                    || x.spec().data_type != out.spec().data_type
                    || x.spec().quantization != scale.spec().quantization
                    || x.spec().quantization != out.spec().quantization) {
                throw std::invalid_argument(
                        "RMSNORM specifications do not match");
            }
            // Read/read overlap between `x` and `scale` is valid, but output
            // storage must be disjoint from both inputs; the rejection is
            // conservative and applies even when transformed output windows
            // appear disjoint.
            if (out.owner_identity() == x.owner_identity()
                    || out.owner_identity() == scale.owner_identity()) {
                throw std::invalid_argument(
                        "RMSNORM output storage must be disjoint from its "
                        "inputs");
            }
            if (x.spec().quantization != QuantizationFormat::NONE
                    || !rmsnorm_leaf(x.spec().data_type)) {
                throw UnsupportedOperation();
            }
            if (!std::isfinite(eps) || eps < 0.0F) {
                throw std::invalid_argument(
                        "RMSNORM epsilon must be finite and nonnegative");
            }
        }

    }  // namespace

    // Default common hooks. A backend port replaces both: the capability
    // predicate reports its applicable leaves and `rmsnorm_impl` queues the
    // kernel for an admitted request. Until then a well-formed request is
    // explicitly unsupported and never reaches workspace inspection,
    // registration, sequence consumption, or execution.
    oid DeviceOps::rmsnorm_impl(const RmsnormRequest&) {
        throw UnsupportedOperation();
    }

    bool DeviceOps::rmsnorm_supported(DataType) const {
        return false;
    }

    oid DeviceOps::rmsnorm(
            const TensorView& x, const TensorView& scale, TensorView& out,
            float eps, RawWorkspaceView workspace) noexcept {
        try {
            validate_rmsnorm(queue_device(), x, scale, out, eps);
            if (!rmsnorm_supported(x.spec().data_type)) {
                throw UnsupportedOperation();
            }
            if (!workspace.empty()) {
                throw std::invalid_argument(
                        "RMSNORM consumes no raw workspace, so the supplied "
                        "workspace view must be empty");
            }
            return invoke(rmsnorm_impl(RmsnormRequest{
                    snapshot_copy_view(x), snapshot_copy_view(scale),
                    snapshot_copy_view(out), eps,
                    kRmsnormWorkspaceRequirements}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    WorkspaceRequirements DeviceOps::rmsnorm_workspace_requirements(
            const TensorView& x, const TensorView& scale,
            const TensorView& out, float eps) {
        validate_rmsnorm(queue_device(), x, scale, out, eps);
        if (!rmsnorm_supported(x.spec().data_type)) {
            throw UnsupportedOperation();
        }
        return kRmsnormWorkspaceRequirements;
    }

}  // namespace iom