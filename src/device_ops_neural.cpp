#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

#include "iom_internal.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <cmath>

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

        constexpr const char* kRopeContext = "ROPE";
        constexpr std::size_t kRopeMaxPosition =
                (std::size_t{1} << 24) - 1;
        constexpr WorkspaceRequirements kRopeWorkspaceRequirements{0, 1};

        bool rope_leaf(DataType value) noexcept {
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

        constexpr WorkspaceRequirements kSiluWorkspaceRequirements{0, 1};
        constexpr const char* kSiluContext = "SILU";

        // SiLU accepts only the nine ordinary floating leaves. Every
        // recognized boolean, integer, and exponent-only scale leaf is
        // semantically inapplicable and therefore reports Unsupported after
        // all malformed host facts have been rejected.
        bool silu_leaf(DataType value) noexcept {
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


        void reject_rope_output_overlap(
                BackendKind backend,
                const TensorView& out,
                const detail::CheckedViewFacts& out_facts,
                const TensorView& input,
                const detail::CheckedViewFacts& input_facts) {
            if (out.owner_identity() == input.owner_identity()) {
                throw std::invalid_argument(
                        "ROPE output aliases an input owner");
            }
            const void* const out_handle = out.native_handle();
            const void* const input_handle = input.native_handle();
            if (out_handle == input_handle) {
                throw std::invalid_argument(
                        "ROPE output shares an input storage handle");
            }
            if (backend == BackendKind::TTNN) {
                return;
            }
            const std::uintptr_t out_begin =
                    reinterpret_cast<std::uintptr_t>(out_handle);
            const std::uintptr_t input_begin =
                    reinterpret_cast<std::uintptr_t>(input_handle);
            const std::uintptr_t limit =
                    std::numeric_limits<std::uintptr_t>::max();
            if (out_facts.storage_bytes > limit - out_begin
                    || input_facts.storage_bytes > limit - input_begin) {
                throw std::overflow_error("ROPE storage range overflows");
            }
            if (out_begin < input_begin + input_facts.storage_bytes
                    && input_begin < out_begin + out_facts.storage_bytes) {
                throw std::invalid_argument(
                        "ROPE output storage range overlaps an input");
            }
        }
        struct RopeValidationFacts {
            detail::CheckedViewFacts x;
            detail::CheckedViewFacts out;
        };

        /*
         * Complete common RoPE admission. This path intentionally only
         * inspects host metadata: no snapshots, registrations, leases,
         * sequence reservation, queue submission, or data access occur here.
         * Submission and the pure requirement query call this same validator
         * before their respective capability hooks.
         */
        RopeValidationFacts validate_rope(
                const Device& device, const TensorView& x,
                const TensorView& out, std::size_t a, double theta) {
            validate_checked_spec(x.spec(), kRopeContext);
            validate_checked_spec(out.spec(), kRopeContext);
            const std::span<const std::size_t> x_dimensions =
                    x.spec().shape.dimensions();
            const std::span<const std::size_t> out_dimensions =
                    out.spec().shape.dimensions();
            const std::size_t rank = x_dimensions.size();
            if (rank < 3 || rank > detail::kMaxTensorRank
                    || out_dimensions.size() != rank) {
                throw std::invalid_argument(
                        "ROPE requires rank three through eight");
            }
            for (std::size_t axis = 0; axis < rank; ++axis) {
                if (x_dimensions[axis] != out_dimensions[axis]) {
                    throw std::invalid_argument(
                            "ROPE input and output shapes must match");
                }
            }

            const std::size_t heads = x_dimensions[rank - 3];
            const std::size_t rows = x_dimensions[rank - 2];
            const std::size_t width = x_dimensions[rank - 1];
            if (heads == 0 || rows == 0 || width == 0) {
                throw std::invalid_argument(
                        "ROPE head, row, and width extents must be nonzero");
            }
            if ((width & 1U) != 0) {
                throw std::invalid_argument(
                        "ROPE final width must be positive and even");
            }

            // Exact queue/device, owner, native-handle, transformed-leading
            // bounds, and checked view arithmetic precede aliases and leaf
            // capability. The facts are also retained in each snapshot.
            const detail::CheckedViewFacts x_facts =
                    validate_checked_view(device, x, kRopeContext);
            const detail::CheckedViewFacts out_facts =
                    validate_checked_view(device, out, kRopeContext);

            // Repeat all operation-owned shape, tile, and logical-address
            // products explicitly so no backend adapter receives an
            // unchecked intermediate.
            std::size_t leading_planes = 1;
            for (std::size_t axis = 0; axis + 3 < rank; ++axis) {
                leading_planes = detail::checked_mul(
                        leading_planes, x_dimensions[axis],
                        "ROPE leading plane count overflows");
            }
            const std::size_t plane_count = detail::checked_mul(
                    leading_planes, heads, "ROPE plane count overflows");
            const std::size_t row_width = detail::checked_mul(
                    rows, width, "ROPE row element count overflows");
            const std::size_t logical_elements = detail::checked_mul(
                    plane_count, row_width,
                    "ROPE logical element count overflows");
            const std::size_t half_width = width / 2;
            (void)detail::checked_mul(
                    plane_count, detail::checked_mul(
                            rows, half_width,
                            "ROPE pair count overflows"),
                    "ROPE pair count overflows");
            const std::size_t padded_rows = detail::padded_extent(
                    rows, "ROPE padded row extent overflows");
            const std::size_t padded_width = detail::padded_extent(
                    width, "ROPE padded width extent overflows");
            const std::size_t padded_plane = detail::checked_mul(
                    padded_rows, padded_width,
                    "ROPE padded tile size overflows");
            (void)detail::checked_mul(
                    plane_count, padded_plane,
                    "ROPE padded storage element count overflows");
            (void)detail::checked_mul(
                    logical_elements, detail::leaf_bits(x.spec().data_type),
                    "ROPE logical bit count overflows");

            // Position arithmetic and its binary32 representability bound are
            // part of checked shape/address admission, before alias and theta
            // policy checks.
            const std::size_t last_position = detail::checked_add(
                    a, rows - 1, "ROPE position range overflows");
            if (last_position > kRopeMaxPosition) {
                throw std::invalid_argument(
                        "ROPE position exceeds the binary32 bound");
            }
            const float narrowed_position =
                    static_cast<float>(last_position);
            if (!std::isfinite(narrowed_position)) {
                throw std::overflow_error(
                        "ROPE position narrowing overflows");
            }

            reject_rope_output_overlap(
                    device.backend_kind(), out, out_facts, x, x_facts);

            if (!std::isfinite(theta)
                    || theta < 1.0
                    || theta
                            > static_cast<double>(
                                    std::numeric_limits<float>::max())) {
                throw std::invalid_argument(
                        "ROPE theta must be finite in [1,float_max]");
            }
            const float narrowed_theta = static_cast<float>(theta);
            if (!std::isfinite(narrowed_theta) || narrowed_theta < 1.0F) {
                throw std::overflow_error(
                        "ROPE theta narrowing overflows");
            }

            if (x.spec().data_type != out.spec().data_type
                    || x.spec().quantization != out.spec().quantization) {
                throw std::invalid_argument(
                        "ROPE input and output specifications must match");
            }
            if (x.spec().quantization != QuantizationFormat::NONE
                    || !rope_leaf(x.spec().data_type)) {
                throw UnsupportedOperation();
            }
            return {x_facts, out_facts};
        }
        void reject_silu_overlap(
                const TensorView& out, std::size_t out_storage_bytes,
                const TensorView& input, std::size_t input_storage_bytes) {
            if (out.owner_identity() == input.owner_identity()) {
                throw std::invalid_argument(
                        "SILU output aliases an input owner");
            }
            const std::uintptr_t out_begin =
                    reinterpret_cast<std::uintptr_t>(out.native_handle());
            const std::uintptr_t input_begin =
                    reinterpret_cast<std::uintptr_t>(input.native_handle());
            const std::uintptr_t limit =
                    std::numeric_limits<std::uintptr_t>::max();
            if (out_storage_bytes > limit - out_begin
                    || input_storage_bytes > limit - input_begin) {
                throw std::overflow_error("SILU storage range overflows");
            }
            const std::uintptr_t out_end = out_begin + out_storage_bytes;
            const std::uintptr_t input_end =
                    input_begin + input_storage_bytes;
            if (out_begin < input_end && input_begin < out_end) {
                throw std::invalid_argument(
                        "SILU output storage range overlaps an input");
            }
        }

        // Common SiLU admission is allocation-free and deliberately builds no
        // snapshot. Both the pure query and submission call this exact path
        // before backend capability, sequence, registration, or workspace
        // handling.
        void validate_silu(
                const Device& device, const TensorView& x,
                const TensorView& y) {
            validate_checked_spec(x.spec(), kSiluContext);
            validate_checked_spec(y.spec(), kSiluContext);
            if (x.spec().shape != y.spec().shape) {
                throw std::invalid_argument(
                        "SILU input and output shapes must match");
            }
            if (x.spec().data_type != y.spec().data_type
                    || x.spec().quantization != y.spec().quantization) {
                throw std::invalid_argument(
                        "SILU input and output specifications must match");
            }

            // Validate each complete owner specification before the shared
            // checked-view helper indexes its final two dimensions. This
            // keeps malformed owner metadata in the invalid-input path rather
            // than allowing an underflow while checking view bounds.
            if (x.owner_identity() != nullptr) {
                validate_checked_spec(
                        x.owner_identity()->view().spec(), kSiluContext);
            }
            if (y.owner_identity() != nullptr
                    && y.owner_identity() != x.owner_identity()) {
                validate_checked_spec(
                        y.owner_identity()->view().spec(), kSiluContext);
            }
            const detail::CheckedViewFacts x_facts =
                    validate_checked_view(device, x, kSiluContext);
            const detail::CheckedViewFacts y_facts =
                    validate_checked_view(device, y, kSiluContext);
            reject_silu_overlap(
                    y, y_facts.storage_bytes, x, x_facts.storage_bytes);

            if (x.spec().quantization != QuantizationFormat::NONE
                    || !silu_leaf(x.spec().data_type)) {
                throw UnsupportedOperation();
            }
        }

        void validate_silu_requirements(
                const WorkspaceRequirements& requirements) {
            if (requirements != kSiluWorkspaceRequirements) {
                throw std::invalid_argument(
                        "SILU workspace requirement must be {0, 1}");
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
    DeviceOps::RopeViewSnapshot DeviceOps::snapshot_rope_view(
            const TensorView& view,
            const detail::CheckedViewFacts& facts) {
        RopeViewSnapshot snapshot;
        snapshot.rank = view.spec().shape.rank();
        const std::span<const std::size_t> dimensions =
                view.spec().shape.dimensions();
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            snapshot.dimensions[axis] = dimensions[axis];
        }
        const std::span<const std::size_t> strides = view.plane_strides();
        for (std::size_t axis = 0; axis < strides.size(); ++axis) {
            snapshot.plane_strides[axis] = strides[axis];
        }
        snapshot.plane_offset = view.plane_offset();
        snapshot.data_type = view.spec().data_type;
        snapshot.quantization = view.spec().quantization;
        snapshot.device_identity = &view.device();
        snapshot.owner_identity = view.owner_identity();
        snapshot.native_handle = const_cast<void*>(view.native_handle());
        snapshot.max_plane = facts.max_plane;
        snapshot.addressed_bytes = facts.addressed_bytes;
        snapshot.storage_bytes = facts.storage_bytes;
        snapshot.logical_bytes = facts.logical_bytes;
        return snapshot;
    }


    DeviceOps::SiLUViewSnapshot DeviceOps::snapshot_silu_view(
            const TensorView& view) {
        SiLUViewSnapshot snapshot;
        const std::span<const std::size_t> dimensions =
                view.spec().shape.dimensions();
        snapshot.rank = dimensions.size();
        for (std::size_t index = 0; index < snapshot.rank; ++index) {
            snapshot.dimensions[index] = dimensions[index];
        }
        const std::span<const std::size_t> strides =
                view.plane_strides();
        for (std::size_t index = 0; index < strides.size(); ++index) {
            snapshot.plane_strides[index] = strides[index];
        }
        snapshot.plane_offset = view.plane_offset();
        snapshot.data_type = view.spec().data_type;
        snapshot.quantization = view.spec().quantization;
        snapshot.device_identity = &view.device();
        snapshot.owner_identity = view.owner_identity();
        snapshot.native_handle = const_cast<void*>(view.native_handle());

        return snapshot;
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

    oid DeviceOps::silu_impl(const SiLURequest&) {
        throw UnsupportedOperation();
    }

    WorkspaceRequirements
    DeviceOps::silu_workspace_requirements_impl(const SiLURequest&) {
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
    oid DeviceOps::rope_impl(const RopeRequest&) {
        throw UnsupportedOperation();
    }

    WorkspaceRequirements DeviceOps::rope_workspace_requirements(
            const RopeRequest&) {
        throw UnsupportedOperation();
    }


    oid DeviceOps::sdpa_impl(
            const TensorView&, const TensorView&, const TensorView&,
            size_t, size_t, size_t, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::silu(
            const TensorView& x, TensorView& y,
            RawWorkspaceView workspace) noexcept {
        (void)workspace;
        try {
            const Device& device = queue_device();
            validate_silu(device, x, y);
            SiLURequest request{
                    snapshot_silu_view(x), snapshot_silu_view(y),
                    kSiluWorkspaceRequirements};
            const WorkspaceRequirements requirements =
                    silu_workspace_requirements_impl(request);
            validate_silu_requirements(requirements);
            request.workspace_requirements = requirements;
            return invoke(silu_impl(request));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    WorkspaceRequirements DeviceOps::silu_workspace_requirements(
            const TensorView& x, const TensorView& y) {
        validate_silu(queue_device(), x, y);
        const SiLURequest request{
                snapshot_silu_view(x), snapshot_silu_view(y),
                kSiluWorkspaceRequirements};
        const WorkspaceRequirements requirements =
                silu_workspace_requirements_impl(request);
        validate_silu_requirements(requirements);
        return requirements;
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
    oid DeviceOps::rope(
            const TensorView& x, TensorView& out, std::size_t a,
            double theta, RawWorkspaceView workspace) noexcept {
        try {
            const Device& device = queue_device();
            const RopeValidationFacts validation =
                    validate_rope(device, x, out, a, theta);

            // Capability is intentionally queried before the supplied
            // workspace. The default unsupported hook therefore returns
            // Unsupported without inspecting an otherwise foreign or
            // malformed workspace.
            const RopeRequest capability_request{
                    snapshot_rope_view(x, validation.x),
                    snapshot_rope_view(out, validation.out), a, theta};
            const WorkspaceRequirements requirements =
                    rope_workspace_requirements(capability_request);
            if (requirements != kRopeWorkspaceRequirements) {
                throw std::invalid_argument(
                        "ROPE workspace requirement must be exactly {0,1}");
            }
            if (!workspace.empty()) {
                throw std::invalid_argument(
                        "ROPE consumes no raw workspace, so the supplied "
                        "workspace view must be empty");
            }
            const std::array<TensorView, 2> operands{x, out};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            device, workspace, requirements.bytes,
                            requirements.alignment, operands);
            return invoke(rope_impl(RopeRequest{
                    capability_request.x, capability_request.out, a, theta,
                    validated_workspace, requirements, {}}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    WorkspaceRequirements DeviceOps::rope_workspace_requirements(
            const TensorView& x, const TensorView& out, std::size_t a,
            double theta) {
        const RopeValidationFacts validation =
                validate_rope(queue_device(), x, out, a, theta);
        const RopeRequest request{
                snapshot_rope_view(x, validation.x),
                snapshot_rope_view(out, validation.out), a, theta};
        const WorkspaceRequirements requirements =
                rope_workspace_requirements(request);
        if (requirements != kRopeWorkspaceRequirements) {
            throw std::invalid_argument(
                    "ROPE workspace requirement must be exactly {0,1}");
        }
        return requirements;
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