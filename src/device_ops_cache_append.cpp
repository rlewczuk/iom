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
    using detail::bits_to_bytes;
    using detail::checked_add;
    using detail::checked_mul;
    using detail::leaf_bits;
    using detail::recognized_data_type;
    using detail::recognized_quantization;
    using detail::standard_plane_slot;
    using detail::validate_checked_spec;
    using detail::validate_checked_view;

    namespace {

        constexpr const char* kAdmissionContext = "CACHE_APPEND";
        constexpr std::size_t kMinimumRank = 3;
        constexpr std::size_t kMaximumRank = 8;

        struct PhysicalRange {
            std::uintptr_t begin = 0;
            std::uintptr_t end = 0;
        };

        [[nodiscard]] std::size_t padded_extent(
                std::size_t extent, const char* what) {
            const std::size_t remainder = extent % TensorSpec::TILE;
            if (remainder == 0) {
                return extent;
            }
            return checked_add(
                    extent, TensorSpec::TILE - remainder, what);
        }

        [[nodiscard]] std::size_t checked_plane_count(
                std::span<const std::size_t> dimensions, const char* what) {
            std::size_t planes = 1;
            for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                planes = checked_mul(planes, dimensions[i], what);
            }
            return planes;
        }

        [[nodiscard]] PhysicalRange checked_storage_range(
                const TensorView& view, const detail::CheckedViewFacts& facts) {
            const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(
                    view.native_handle());
            if (facts.storage_bytes
                    > std::numeric_limits<std::uintptr_t>::max() - begin) {
                throw std::overflow_error(
                        "cache append storage range overflows");
            }
            return {begin, begin + facts.storage_bytes};
        }

        [[nodiscard]] bool ranges_overlap(
                const PhysicalRange& lhs, const PhysicalRange& rhs) noexcept {
            return lhs.begin < rhs.end && rhs.begin < lhs.end;
        }

        void validate_view_arithmetic(
                const TensorView& view, const detail::CheckedViewFacts& facts,
                std::size_t rows, std::size_t features,
                std::size_t append_row, const char* role) {
            const auto dimensions = view.spec().shape.dimensions();
            const std::size_t planes = checked_plane_count(
                    dimensions, "cache append plane count overflows");
            const std::size_t padded_rows = padded_extent(
                    dimensions[dimensions.size() - 2],
                    "cache append padded row count overflows");
            const std::size_t padded_features = padded_extent(
                    dimensions.back(),
                    "cache append padded feature count overflows");
            const std::size_t padded_elements = checked_mul(
                    checked_mul(planes, padded_rows,
                                "cache append padded tile count overflows"),
                    padded_features,
                    "cache append padded tile count overflows");
            const std::size_t bits = leaf_bits(view.spec().data_type);
            const std::size_t padded_bytes = bits_to_bytes(
                    checked_mul(padded_elements, bits,
                                "cache append padded byte count overflows"),
                    "cache append padded byte count overflows");
            if (padded_bytes > facts.storage_bytes) {
                throw std::invalid_argument(
                        "cache append view exceeds owner storage");
            }

            const std::size_t logical_elements = checked_mul(
                    checked_mul(planes, rows,
                                "cache append row count overflows"),
                    features,
                    "cache append feature count overflows");
            const std::size_t logical_bytes = bits_to_bytes(
                    checked_mul(logical_elements, bits,
                                "cache append byte count overflows"),
                    "cache append byte count overflows");
            if (logical_bytes > facts.storage_bytes) {
                throw std::invalid_argument(
                        "cache append logical range exceeds owner storage");
            }

            const std::size_t slot = standard_plane_slot(
                    view.spec(), facts.max_plane, append_row, features - 1);
            const std::size_t addressed_bytes = bits_to_bytes(
                    checked_mul(
                            checked_add(slot, 1,
                                        "cache append feature slot overflows"),
                            bits, "cache append row byte count overflows"),
                    "cache append row byte count overflows");
            if (addressed_bytes > facts.storage_bytes) {
                throw std::invalid_argument(
                        role == nullptr
                                ? "cache append row exceeds owner storage"
                                : role);
            }
        }


    }  // namespace
    DeviceOps::CacheAppendViewSnapshot
    DeviceOps::snapshot_cache_append_view(const TensorView& view) {
        DeviceOps::CacheAppendViewSnapshot snapshot;
        const auto dimensions = view.spec().shape.dimensions();
        const auto strides = view.plane_strides();
        snapshot.rank = dimensions.size();
        for (std::size_t index = 0; index < snapshot.rank; ++index) {
            snapshot.dimensions[index] = dimensions[index];
        }
        for (std::size_t index = 0; index < strides.size(); ++index) {
            snapshot.plane_strides[index] = strides[index];
        }
        snapshot.plane_offset = view.plane_offset();
        snapshot.data_type = view.spec().data_type;
        snapshot.quantization = view.spec().quantization;
        snapshot.device_identity = &view.device();
        snapshot.owner_identity = view.owner_identity();
        snapshot.native_handle =
                const_cast<void*>(view.native_handle());
        return snapshot;
    }

    DeviceOps::CacheAppendRequest DeviceOps::validate_cache_append(
            const Device& device, const TensorView& source,
            const TensorView& destination, std::size_t a) {
        // Keep this first pass allocation-free and before either snapshot is
        // constructed.  In particular, recognized non-NONE quantization is
        // allowed through structural validation so it can report the common
        // Unsupported category below rather than TensorSpec's old runtime
        // error.
        validate_checked_spec(source.spec(), kAdmissionContext);
        validate_checked_spec(destination.spec(), kAdmissionContext);
        if (source.spec().shape.rank() < kMinimumRank
                || source.spec().shape.rank() > kMaximumRank
                || destination.spec().shape.rank() < kMinimumRank
                || destination.spec().shape.rank() > kMaximumRank) {
            throw std::invalid_argument(
                    "cache append requires ranks three through eight");
        }

        const detail::CheckedViewFacts source_facts = validate_checked_view(
                device, source, kAdmissionContext);
        const detail::CheckedViewFacts destination_facts = validate_checked_view(
                device, destination, kAdmissionContext);

        const auto source_dimensions = source.spec().shape.dimensions();
        const auto destination_dimensions =
                destination.spec().shape.dimensions();
        if (source_dimensions.size() != destination_dimensions.size()) {
            throw std::invalid_argument(
                    "cache append leading ranks do not match");
        }
        const std::size_t rank = source_dimensions.size();
        for (std::size_t axis = 0; axis + 3 < rank; ++axis) {
            if (source_dimensions[axis] != destination_dimensions[axis]) {
                throw std::invalid_argument(
                        "cache append leading tuples do not match");
            }
        }
        if (source_dimensions[rank - 3]
                != destination_dimensions[rank - 3]) {
            throw std::invalid_argument(
                    "cache append head extents do not match");
        }
        if (source_dimensions.back() != destination_dimensions.back()) {
            throw std::invalid_argument(
                    "cache append feature extents do not match");
        }

        const std::size_t capacity = destination_dimensions[rank - 2];
        const std::size_t rows = source_dimensions[rank - 2];
        const std::size_t features = source_dimensions.back();
        // The order is contractual: do not subtract until a <= C has passed.
        if (a > capacity) {
            throw std::invalid_argument(
                    "cache append offset exceeds destination capacity");
        }
        const std::size_t available = capacity - a;
        if (rows > available) {
            throw std::invalid_argument(
                    "cache append source rows exceed destination capacity");
        }

        // Validate all operation-owned arithmetic, including both logical and
        // padded planes and the absolute destination row used by a+r.  The
        // checked-view pass above already validates transformed plane bounds
        // and owner storage; this pass closes the append-specific row range.
        const std::size_t end_row = checked_add(
                a, rows, "cache append row position overflows");
        const std::size_t last_destination_row = checked_add(
                a, rows - 1, "cache append row position overflows");
        (void)end_row;
        validate_view_arithmetic(
                source, source_facts, rows, features, rows - 1,
                "cache append source row exceeds owner storage");
        validate_view_arithmetic(
                destination, destination_facts, rows, features,
                last_destination_row,
                "cache append destination row exceeds owner storage");

        // Check complete owner ranges, not merely the logical windows.  This
        // conservatively covers every transformed leading plane and padded
        // tile owned by either operand.
        const PhysicalRange source_range = checked_storage_range(
                source, source_facts);
        const PhysicalRange destination_range = checked_storage_range(
                destination, destination_facts);

        if (source.spec().data_type != destination.spec().data_type
                || source.spec().quantization
                        != destination.spec().quantization) {
            throw std::invalid_argument(
                    "cache append operand specifications do not match");
        }
        if (source.owner_identity() == destination.owner_identity()
                || ranges_overlap(source_range, destination_range)) {
            throw std::invalid_argument(
                    "cache append source/destination overlap is forbidden");
        }

        if (!recognized_data_type(source.spec().data_type)
                || !recognized_quantization(source.spec().quantization)) {
            // The checked-spec pass normally catches this. Keep the explicit
            // guard adjacent to the operation policy so future enum changes
            // cannot silently widen the payload contract.
            throw std::invalid_argument("unknown cache append encoding");
        }
        if (source.spec().quantization != QuantizationFormat::NONE) {
            throw UnsupportedOperation();
        }

        return CacheAppendRequest{
                snapshot_cache_append_view(source),
                snapshot_cache_append_view(destination), a};
    }

    oid DeviceOps::cache_append(
            const TensorView& source, TensorView& destination, std::size_t a,
            RawWorkspaceView workspace) noexcept {
        try {
            CacheAppendRequest request = validate_cache_append(
                    queue_device(), source, destination, a);
            const WorkspaceRequirements requirements =
                    cache_append_workspace_requirements(request);
            const std::array<TensorView, 2> operands{source, destination};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            queue_device(), workspace, requirements.bytes,
                            requirements.alignment, operands);
            const CacheAppendRequest admitted{
                    request.source, request.destination, request.a,
                    validated_workspace, requirements};
            return invoke(cache_append_impl(admitted));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    WorkspaceRequirements DeviceOps::cache_append_workspace_requirements(
            const TensorView& source, const TensorView& destination,
            std::size_t a) {
        const CacheAppendRequest request = validate_cache_append(
                queue_device(), source, destination, a);
        return cache_append_workspace_requirements(request);
    }

    oid DeviceOps::cache_append_impl(const CacheAppendRequest&) {
        throw UnsupportedOperation();
    }

    WorkspaceRequirements DeviceOps::cache_append_workspace_requirements(
            const CacheAppendRequest&) {
        throw UnsupportedOperation();
    }

}  // namespace iom
