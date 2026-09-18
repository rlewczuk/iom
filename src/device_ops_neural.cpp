#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

#include "iom_internal.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace iom {

    using detail::UnsupportedOperation;
    using detail::validate_checked_spec;
    using detail::validate_checked_view;

    namespace {

        // Established diagnostic tag of linear projection admission. The
        // shared checked-view path only interpolates it into rejection text.
        constexpr const char* kLinearContext = "LINEAR";

        // Exact applicability of one linear leaf: the twelve integer leaves
        // and the nine ordinary signed floating leaves. Recognized BOOL is
        // inapplicable because logical boolean encoding is not a linear
        // numeric operand, and `F8_E8M0` is inapplicable because its
        // unsigned exponent-only scale encoding cannot represent a general
        // signed linear result. An unknown enumeration value is rejected as
        // invalid input by the shared spec check before this predicate is
        // consulted.
        bool linear_leaf(DataType value) noexcept {
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

        // Reject one output/input relationship the projection must never
        // admit: the same owner identity anywhere, and — on standard-layout
        // backends, whose owners expose a storage base and exact reserved
        // extent — an identical storage handle or actual backing-range
        // intersection. The rule is deliberately conservative: an output
        // window that merely appears disjoint from the input's transformed
        // window is still rejected whenever their owners, handles, or
        // storage ranges coincide. Native backends address storage opaquely
        // and keep the owner-identity rule only. Input/read aliases between
        // `x` and `w` are never rejected here.
        void reject_output_overlap(
                BackendKind backend, const TensorView& out,
                std::size_t out_storage_bytes, const TensorView& input,
                std::size_t input_storage_bytes) {
            if (out.owner_identity() == input.owner_identity()) {
                throw std::invalid_argument(
                        "LINEAR output aliases an input owner");
            }
            if (backend == BackendKind::TTNN) {
                return;
            }
            const void* const out_handle = out.native_handle();
            const void* const input_handle = input.native_handle();
            if (out_handle == input_handle) {
                throw std::invalid_argument(
                        "LINEAR output shares an input storage handle");
            }
            const std::uintptr_t out_begin =
                    reinterpret_cast<std::uintptr_t>(out_handle);
            const std::uintptr_t input_begin =
                    reinterpret_cast<std::uintptr_t>(input_handle);
            const std::uintptr_t limit =
                    std::numeric_limits<std::uintptr_t>::max();
            if (out_storage_bytes > limit - out_begin
                    || input_storage_bytes > limit - input_begin) {
                throw std::overflow_error(
                        "LINEAR storage range overflows");
            }
            if (out_begin < input_begin + input_storage_bytes
                    && input_begin < out_begin + out_storage_bytes) {
                throw std::invalid_argument(
                        "LINEAR output storage range overlaps an input");
            }
        }

    }  // namespace

    DeviceOps::LinearViewSnapshot DeviceOps::snapshot_linear_view(
            const TensorView& view) {
        return DeviceOps::LinearViewSnapshot{
                view.spec(), &view.device(), view.owner_identity(),
                const_cast<void*>(view.native_handle()), view.plane_offset(),
                {view.plane_strides().begin(), view.plane_strides().end()}};
    }

    void DeviceOps::validate_linear(
            const Device& device, const TensorView& x, const TensorView& w,
            const TensorView& out, std::size_t s, std::size_t R,
            LinearOutputLayout layout, std::size_t H, std::size_t D) {
        validate_checked_spec(x.spec(), kLinearContext);
        validate_checked_spec(w.spec(), kLinearContext);
        validate_checked_spec(out.spec(), kLinearContext);

        // Host-known structure first: the explicit output mode is a
        // parameter and is never inferred from the output rank, so an
        // unrecognized value is invalid input before any shape decision.
        if (layout != LinearOutputLayout::ordinary
                && layout != LinearOutputLayout::head_planar) {
            throw std::invalid_argument(
                    "unknown LINEAR output layout value");
        }

        const std::span<const std::size_t> x_dims =
                x.spec().shape.dimensions();
        const std::span<const std::size_t> w_dims =
                w.spec().shape.dimensions();
        const std::span<const std::size_t> out_dims =
                out.spec().shape.dimensions();
        const std::size_t x_rank = x_dims.size();
        const std::size_t out_rank = out_dims.size();
        const std::size_t source_rows = x_dims[x_rank - 2];
        const std::size_t input_features = x_dims[x_rank - 1];

        // The HF-oriented weight is exactly the rank-two `[O,I]` matrix. Its
        // empty leading tuple is neither matched against the input's leading
        // tuple nor allowed to imply leading-state broadcast.
        if (w.spec().shape.rank() != 2) {
            throw std::invalid_argument(
                    "LINEAR weight must be the rank-two [O,I] matrix");
        }
        const std::size_t output_features = w_dims[0];
        if (w_dims[1] != input_features) {
            throw std::invalid_argument(
                    "LINEAR weight input extent must equal the input "
                    "feature extent");
        }
        if (H == 0 || D == 0) {
            throw std::invalid_argument(
                    "LINEAR head count and head width must be nonzero");
        }

        // Exact output shape, explicit mode relation, and rank growth. The
        // rank comparison short-circuits every indexed output extent below,
        // so no out-of-range read can precede the shape rejection.
        if (layout == LinearOutputLayout::ordinary) {
            if (H != 1 || D != output_features) {
                throw std::invalid_argument(
                        "LINEAR ordinary layout requires H=1 and D=O");
            }
            if (out_rank != x_rank || out_dims[x_rank - 2] != R
                    || out_dims[x_rank - 1] != output_features) {
                throw std::invalid_argument(
                        "LINEAR ordinary output must be [...,R,O]");
            }
        } else {
            // Checked `H*D` precedes the comparison with `O`, so an
            // overflowing head product is an Overflow rather than a
            // silently narrowed equality.
            const std::size_t head_elements = detail::checked_mul(
                    H, D, "LINEAR head product overflows");
            if (head_elements != output_features) {
                throw std::invalid_argument(
                        "LINEAR head-planar layout requires O=H*D");
            }
            if (x_rank + 1 > detail::kMaxTensorRank) {
                throw std::invalid_argument(
                        "LINEAR head-planar output rank exceeds eight");
            }
            if (out_rank != x_rank + 1 || out_dims[x_rank - 2] != H
                    || out_dims[x_rank - 1] != R
                    || out_dims[x_rank] != D) {
                throw std::invalid_argument(
                        "LINEAR head-planar output must be [...,H,R,D]");
            }
        }
        // Input and output leading tuples match extents exactly, excluding
        // the head axis head-planar mode inserts.
        for (std::size_t axis = 0; axis + 2 < x_rank; ++axis) {
            if (x_dims[axis] != out_dims[axis]) {
                throw std::invalid_argument(
                        "LINEAR input and output leading extents must "
                        "match");
            }
        }

        // The independently selected row window. `s <= T` is checked before
        // `T-s`, so the subtraction can never wrap and no out-of-window row
        // can be addressed.
        if (R == 0) {
            throw std::invalid_argument(
                    "LINEAR projected row count must be nonzero");
        }
        if (s > source_rows) {
            throw std::invalid_argument(
                    "LINEAR selected row start exceeds the source rows");
        }
        if (R > source_rows - s) {
            throw std::invalid_argument(
                    "LINEAR selected row count exceeds the available "
                    "source rows");
        }

        // The shared checked-view path validates each operand's exact
        // device identity, live owner and stable native handle, leading-only
        // view geometry, selected-plane bounds, and checked plane, tile,
        // element, bit, byte, stride, and address arithmetic.
        const detail::CheckedViewFacts x_facts =
                validate_checked_view(device, x, kLinearContext);
        const detail::CheckedViewFacts w_facts =
                validate_checked_view(device, w, kLinearContext);
        const detail::CheckedViewFacts out_facts =
                validate_checked_view(device, out, kLinearContext);

        // Conservative output/input overlap rejection precedes the leaf and
        // capability decisions, and permitted read/read overlap between `x`
        // and `w` is left alone.
        const BackendKind backend = device.backend_kind();
        reject_output_overlap(
                backend, out, out_facts.storage_bytes, x,
                x_facts.storage_bytes);
        reject_output_overlap(
                backend, out, out_facts.storage_bytes, w,
                w_facts.storage_bytes);

        // Only then the leaf classes: one applicable leaf and
        // `QuantizationFormat::NONE` shared by all three views. A recognized
        // but mixed, inapplicable, or unsupported leaf is a capability
        // rejection, which the operation's own hook reports as
        // `Unsupported`; unknown enumeration values were already rejected as
        // invalid input above.
        if (x.spec().data_type != w.spec().data_type
                || x.spec().data_type != out.spec().data_type
                || x.spec().quantization != w.spec().quantization
                || x.spec().quantization != out.spec().quantization) {
            throw UnsupportedOperation();
        }
        if (x.spec().quantization != QuantizationFormat::NONE
                || !linear_leaf(x.spec().data_type)) {
            throw UnsupportedOperation();
        }
    }

    oid DeviceOps::silu_impl(const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::linear_impl(const LinearRequest&) {
        throw UnsupportedOperation();
    }

    WorkspaceRequirements DeviceOps::linear_workspace_requirements_impl(
            const TensorView&, const TensorView&, const TensorView&,
            std::size_t, std::size_t, LinearOutputLayout, std::size_t,
            std::size_t) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::sdpa_impl(
            const TensorView&, const TensorView&, const TensorView&,
            size_t, size_t, size_t, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::silu(
            const TensorView& x, TensorView& y) noexcept {
        try {
            validate_views(queue_device(), {&x, &y});
            return invoke(silu_impl(x, y));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::linear(
            const TensorView& x, const TensorView& w, TensorView& out,
            std::size_t s, std::size_t R, LinearOutputLayout layout,
            std::size_t H, std::size_t D,
            RawWorkspaceView workspace) noexcept {
        try {
            const Device& device = queue_device();
            validate_linear(device, x, w, out, s, R, layout, H, D);
            // The capability decision precedes even the supplied workspace:
            // an unported backend reports `Unsupported` without inspecting
            // otherwise unusable or foreign scratch.
            const WorkspaceRequirements requirements =
                    linear_workspace_requirements_impl(
                            x, w, out, s, R, layout, H, D);
            // Supplied scratch is validated against the reported
            // requirement, including its whole-range nonoverlap with every
            // operand and the output. The shared zero-requirement policy
            // neither validates nor leases an unused range, exactly as the
            // binary and embedding facades behave.
            const std::array<TensorView, 3> operands{x, w, out};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            device, workspace, requirements.bytes,
                            requirements.alignment, operands);
            return invoke(linear_impl(LinearRequest{
                    snapshot_linear_view(x), snapshot_linear_view(w),
                    snapshot_linear_view(out), s, R, layout, H, D,
                    validated_workspace, requirements, {}}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    WorkspaceRequirements DeviceOps::linear_workspace_requirements(
            const TensorView& x, const TensorView& w, const TensorView& out,
            std::size_t s, std::size_t R, LinearOutputLayout layout,
            std::size_t H, std::size_t D) {
        validate_linear(queue_device(), x, w, out, s, R, layout, H, D);
        return linear_workspace_requirements_impl(
                x, w, out, s, R, layout, H, D);
    }

    oid DeviceOps::sdpa(
            const TensorView& q, const TensorView& k, const TensorView& v,
            size_t n_heads, size_t n_kv_heads, size_t head_dim,
            TensorView& attn_out) noexcept {
        try {
            validate_views(queue_device(), {&q, &k, &v, &attn_out});
            if (n_heads == 0 || n_kv_heads == 0 || head_dim == 0
                    || n_heads % n_kv_heads != 0) {
                throw std::invalid_argument("invalid sdpa parameters");
            }
            return invoke(sdpa_impl(
                    q, k, v, n_heads, n_kv_heads, head_dim, attn_out));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

}  // namespace iom