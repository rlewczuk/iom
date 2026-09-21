#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

#include "iom_internal.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace iom {

    using detail::UnsupportedOperation;
    using detail::validate_checked_spec;
    using detail::validate_checked_view;

    namespace {
        // Established diagnostic tag of embedding admission. The shared
        // checked-view path only interpolates it into rejection text.
        constexpr const char* kAdmissionContext = "EMBEDDING";

        // The twelve integral ID leaves. BOOL and every floating leaf are
        // recognized but inapplicable as index semantics, which the operation
        // capability stage reports as Unsupported.
        bool embedding_index_leaf(DataType value) noexcept {
            switch (value) {
                case DataType::I2: case DataType::U2:
                case DataType::I4: case DataType::U4:
                case DataType::I8: case DataType::U8:
                case DataType::I16: case DataType::U16:
                case DataType::I32: case DataType::U32:
                case DataType::I64: case DataType::U64:
                    return true;
                case DataType::BOOL:
                case DataType::F4_E2M1: case DataType::F6_E2M3:
                case DataType::F6_E3M2:
                case DataType::F8_E4M3FN: case DataType::F8_E5M2:
                case DataType::F8_E8M0:
                case DataType::F16: case DataType::BF16:
                case DataType::F32: case DataType::F64:
                    return false;
            }
            return false;
        }

        // Keep embedding's output/input policy; only the backing facts differ
        // between addressable owners and opaque execution descriptors.
        void reject_output_overlap(
                const TensorView& out, const detail::CheckedViewFacts& out_facts,
                const TensorView& input, const detail::CheckedViewFacts& input_facts) {
            if (out.owner_identity() == input.owner_identity()) {
                throw std::invalid_argument(
                        "embedding output aliases an input owner");
            }
            if (out_facts.backing.key == input_facts.backing.key) {
                throw std::invalid_argument(
                        "embedding output shares an input storage handle");
            }
            const auto out_range = detail::checked_storage_range(
                    out_facts.backing, 0, out_facts.storage_bytes,
                    "embedding storage range overflows");
            const auto input_range = detail::checked_storage_range(
                    input_facts.backing, 0, input_facts.storage_bytes,
                    "embedding storage range overflows");
            if (detail::storage_ranges_overlap(out_range, input_range)) {
                throw std::invalid_argument(
                        "embedding output storage range overlaps an input");
            }
        }

        /*
         * Operation-owned embedding admission. Runs before any snapshot,
         * registration, lease, sequence reservation, token acceptance, or
         * backend capability decision, and allocates nothing, so the pure
         * requirement query can share it with submission. Rank, shape, and
         * role structure are embedding-owned; the operation-neutral checked
         * helpers in `iom_internal.hpp` supply recognized encodings,
         * structural validation, exact live owner and stable handle, leading
         * geometry, and checked arithmetic. Capability policy — the NONE-only
         * quantization rule and the twelve applicable ID leaves — is reported
         * as Unsupported here, and the backend hook decides the rest.
         */
        void validate_embedding(
                const Device& device, const TensorView& table,
                const TensorView& indices, const TensorView& out) {
            validate_checked_spec(table.spec(), kAdmissionContext);
            validate_checked_spec(indices.spec(), kAdmissionContext);
            validate_checked_spec(out.spec(), kAdmissionContext);

            const std::span<const std::size_t> table_dimensions =
                    table.spec().shape.dimensions();
            const std::span<const std::size_t> index_dimensions =
                    indices.spec().shape.dimensions();
            const std::span<const std::size_t> out_dimensions =
                    out.spec().shape.dimensions();
            if (table_dimensions.size() != 2) {
                throw std::invalid_argument(
                        "embedding table must have rank two");
            }
            // `indices` is `[..., 1, R]`: one dummy row axis precedes the
            // logical run axis, and there is no singleton output rank
            // inflation.
            if (index_dimensions[index_dimensions.size() - 2] != 1) {
                throw std::invalid_argument(
                        "embedding index view must have a single-row axis");
            }
            if (out_dimensions.size() != index_dimensions.size()
                    || out_dimensions[out_dimensions.size() - 2]
                            != index_dimensions.back()
                    || out_dimensions.back() != table_dimensions.back()
                    || !std::equal(
                            index_dimensions.begin(),
                            index_dimensions.end() - 2,
                            out_dimensions.begin(),
                            out_dimensions.end() - 2)) {
                throw std::invalid_argument(
                        "embedding index and output shapes do not match");
            }
            if (table.spec().data_type != out.spec().data_type
                    || table.spec().quantization != out.spec().quantization) {
                throw std::invalid_argument(
                        "embedding table and output specifications do not "
                        "match");
            }

            // Shared checked-view admission: exact queue device identity,
            // live owner, stable native handle, leading-only view geometry,
            // selected-plane bounds, and checked element, bit, byte, and
            // stride arithmetic. The reported owner storage extents feed the
            // backing-range rule below without allocating.
            const detail::CheckedViewFacts table_facts =
                    validate_checked_view(device, table, kAdmissionContext);
            const detail::CheckedViewFacts index_facts =
                    validate_checked_view(device, indices, kAdmissionContext);
            const detail::CheckedViewFacts out_facts =
                    validate_checked_view(device, out, kAdmissionContext);

            reject_output_overlap(
                    out, out_facts, table, table_facts);
            reject_output_overlap(
                    out, out_facts, indices, index_facts);

            if (table.spec().quantization != QuantizationFormat::NONE
                    || indices.spec().quantization
                            != QuantizationFormat::NONE
                    || out.spec().quantization != QuantizationFormat::NONE
                    || !embedding_index_leaf(indices.spec().data_type)) {
                throw UnsupportedOperation();
            }
        }

    }  // namespace

    oid DeviceOps::embedding_impl(const EmbeddingRequest&) {
        throw UnsupportedOperation();
    }

    WorkspaceRequirements DeviceOps::embedding_workspace_requirements_impl(
            const TensorView&, const TensorView&, const TensorView&) {
        throw UnsupportedOperation();
    }

    WorkspaceRequirements DeviceOps::embedding_workspace_requirements(
            const TensorView& table, const TensorView& indices,
            const TensorView& out) {
        validate_embedding(queue_device(), table, indices, out);
        return embedding_workspace_requirements_impl(table, indices, out);
    }

    oid DeviceOps::embedding(
            const TensorView& table, const TensorView& indices,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        try {
            const Device& device = queue_device();
            validate_embedding(device, table, indices, out);
            const WorkspaceRequirements requirements =
                    embedding_workspace_requirements_impl(
                            table, indices, out);
            // Supplied scratch is validated only for a positive
            // requirement: the common zero-requirement policy neither
            // validates nor leases an unused range.
            const std::array<TensorView, 3> operands{table, indices, out};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            device, workspace, requirements.bytes,
                            requirements.alignment, operands);
            return invoke(embedding_impl(EmbeddingRequest{
                    snapshot_copy_view(table), snapshot_copy_view(indices),
                    snapshot_copy_view(out), validated_workspace,
                    requirements, {}}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

}  // namespace iom