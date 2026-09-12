#include "iom/device.hpp"
#include "iom/tensor.hpp"
#include "iom/gpu_algorithm.hpp"

#include "iom_internal.hpp"

#include <utility>

namespace iom {

    using detail::checked_add;
    using detail::checked_mul;
    using detail::kMaxTensorRank;

    namespace {

        TensorSpec with_leading_dimensions(
                const TensorSpec& spec, std::vector<std::size_t> leading) {
            const std::span<const std::size_t> dims = spec.shape.dimensions();
            leading.insert(
                    leading.end(), dims.begin() + (dims.size() - 2), dims.end());
            return TensorSpec{
                    TensorShape{std::move(leading)},
                    spec.data_type,
                    spec.quantization};
        }

        std::vector<std::size_t> leading_dimensions_of(const TensorSpec& spec) {
            const std::span<const std::size_t> dims = spec.shape.dimensions();
            return {dims.begin(), dims.end() - 2};
        }

        std::vector<std::size_t> dense_plane_strides(const TensorShape& shape) {
            const std::span<const std::size_t> dims = shape.dimensions();
            const std::size_t leading_rank = dims.size() - 2;
            std::vector<std::size_t> strides(leading_rank, 1);
            // strides[i] is the product of the leading dimensions after i,
            // so dims.front() never participates in a stored stride.
            for (std::size_t i = leading_rank; i > 1; --i) {
                strides[i - 2] = checked_mul(
                        strides[i - 1], dims[i - 1],
                        "dense plane stride overflows");
            }
            return strides;
        }

        // The tensor constructor must reject an unsupported specification
        // before any part of the full view exists.
        std::vector<std::size_t> validated_dense_plane_strides(
                const TensorSpec& spec) {
            spec.validate();
            return dense_plane_strides(spec.shape);
        }

    }  // namespace

    TensorView::TensorView(
            Tensor& owner, TensorSpec spec, std::size_t plane_offset,
            std::vector<std::size_t> plane_strides)
            : owner_(&owner),
              spec_(std::move(spec)),
              plane_offset_(plane_offset),
              plane_strides_(std::move(plane_strides)) {}

    const Tensor* TensorView::owner_identity() const noexcept {
        return owner_;
    }
    const TensorSpec& TensorView::spec() const noexcept {
        return spec_;
    }

    const Device& TensorView::device() const noexcept {
        return *owner_->device_;
    }

    BackendKind TensorView::backend_kind() const noexcept {
        return device().backend_kind();
    }

    std::uint32_t TensorView::backend_device() const noexcept {
        return device().backend_device();
    }

    void* TensorView::native_handle() noexcept {
        return owner_->storage_handle();
    }

    const void* TensorView::native_handle() const noexcept {
        return owner_->storage_handle();
    }

    std::size_t TensorView::plane_offset() const noexcept {
        return plane_offset_;
    }

    std::span<const std::size_t> TensorView::plane_strides() const noexcept {
        return plane_strides_;
    }

    TensorView TensorView::slice(
            std::size_t dim, std::size_t first, std::size_t count,
            std::size_t step) const {
        const std::size_t leading_rank = spec_.shape.rank() - 2;
        if (dim >= leading_rank) {
            throw std::out_of_range("slice dimension is not a leading dimension");
        }
        if (count == 0) {
            throw std::invalid_argument("slice count must be nonzero");
        }
        if (step == 0) {
            throw std::invalid_argument("slice step must be nonzero");
        }
        const std::size_t stride = plane_strides_[dim];
        const std::size_t last = checked_add(
                first,
                checked_mul(count - 1, step, "slice last index overflows"),
                "slice last index overflows");
        if (last >= spec_.shape.dimension(dim)) {
            throw std::out_of_range(
                "slice last selected index exceeds the dimension");
        }

        std::vector<std::size_t> dims = leading_dimensions_of(spec_);
        dims[dim] = count;
        std::vector<std::size_t> strides{
                plane_strides_.begin(), plane_strides_.end()};
        strides[dim] = checked_mul(stride, step, "sliced plane stride overflows");
        const std::size_t offset = checked_add(
                plane_offset_,
                checked_mul(first, stride, "slice plane offset overflows"),
                "slice plane offset overflows");
        return TensorView{
                *owner_,
                with_leading_dimensions(spec_, std::move(dims)),
                offset,
                std::move(strides)};
    }

    TensorView TensorView::select(std::size_t dim, std::size_t index) const {
        const std::size_t leading_rank = spec_.shape.rank() - 2;
        if (dim >= leading_rank) {
            throw std::out_of_range("select dimension is not a leading dimension");
        }
        if (index >= spec_.shape.dimension(dim)) {
            throw std::out_of_range("select index exceeds the dimension");
        }
        const std::size_t stride = plane_strides_[dim];
        std::vector<std::size_t> dims = leading_dimensions_of(spec_);
        dims.erase(dims.begin() + static_cast<std::ptrdiff_t>(dim));
        std::vector<std::size_t> strides{
                plane_strides_.begin(), plane_strides_.end()};
        strides.erase(strides.begin() + static_cast<std::ptrdiff_t>(dim));
        const std::size_t offset = checked_add(
                plane_offset_,
                checked_mul(index, stride, "select plane offset overflows"),
                "select plane offset overflows");
        return TensorView{
                *owner_,
                with_leading_dimensions(spec_, std::move(dims)),
                offset,
                std::move(strides)};
    }

    TensorView TensorView::permute(
            std::span<const std::size_t> leading_order) const {
        const std::size_t leading_rank = spec_.shape.rank() - 2;
        if (leading_order.size() != leading_rank) {
            throw std::invalid_argument(
                "permutation must cover exactly the leading dimensions");
        }
        std::vector<bool> seen(leading_rank, false);
        for (const std::size_t axis : leading_order) {
            if (axis >= leading_rank) {
                throw std::invalid_argument(
                    "permutation index exceeds the leading rank");
            }
            if (seen[axis]) {
                throw std::invalid_argument("duplicate permutation index");
            }
            seen[axis] = true;
        }

        std::vector<std::size_t> dims(leading_rank);
        std::vector<std::size_t> strides(leading_rank);
        for (std::size_t i = 0; i < leading_rank; ++i) {
            dims[i] = spec_.shape.dimension(leading_order[i]);
            strides[i] = plane_strides_[leading_order[i]];
        }
        return TensorView{
                *owner_,
                with_leading_dimensions(spec_, std::move(dims)),
                plane_offset_,
                std::move(strides)};
    }

    TensorView TensorView::reshape_leading(
            std::span<const std::size_t> leading_dimensions) const {
        // The leading span is a helper, not itself a full shape; the
        // assembled full shape (span plus the two tiled matrix axes) must
        // stay inside the rank-two through rank-eight interval. Rank
        // increasing reshapes to rank nine are rejected even when the
        // source is contiguous and the leading plane count is unchanged.
        if (leading_dimensions.size() + 2 > kMaxTensorRank) {
            throw std::invalid_argument(
                    "reshape result rank exceeds the rank-eight limit");
        }
        const std::size_t leading_rank = spec_.shape.rank() - 2;

        std::size_t planes = 1;
        for (std::size_t i = 0; i < leading_rank; ++i) {
            planes = checked_mul(
                    planes, spec_.shape.dimension(i),
                    "leading plane count overflows");
        }
        std::size_t new_planes = 1;
        for (const std::size_t dimension : leading_dimensions) {
            if (dimension == 0) {
                throw std::invalid_argument("reshape dimension must be nonzero");
            }
            new_planes = checked_mul(
                    new_planes, dimension, "reshaped plane count overflows");
        }
        if (new_planes != planes) {
            throw std::invalid_argument("reshape changes the leading plane count");
        }

        // Contiguous in reverse leading-axis order: every dimension larger
        // than one must carry the next dense stride, beginning at one.
        // Size-one dimensions do not constrain their stored stride.
        std::size_t expected = 1;
        for (std::size_t i = leading_rank; i-- > 0;) {
            const std::size_t dimension = spec_.shape.dimension(i);
            if (dimension > 1) {
                if (plane_strides_[i] != expected) {
                    throw std::invalid_argument("reshape requires a contiguous view");
                }
                expected = checked_mul(
                        expected, dimension, "reshape contiguity overflows");
            }
        }

        std::vector<std::size_t> strides(leading_dimensions.size());
        std::size_t stride = 1;
        for (std::size_t i = strides.size(); i-- > 0;) {
            strides[i] = stride;
            stride = checked_mul(
                    stride, leading_dimensions[i],
                    "reshaped plane stride overflows");
        }
        return TensorView{
                *owner_,
                with_leading_dimensions(
                        spec_,
                        {leading_dimensions.begin(), leading_dimensions.end()}),
                plane_offset_,
                std::move(strides)};
    }

    void TensorView::copy_from_host(
            std::span<const std::byte> source, RawWorkspaceView workspace) {
        const std::size_t nbytes = spec_.logical_nbytes();
        if (source.size() != nbytes) {
            throw std::invalid_argument(
                "host source must hold exactly logical_nbytes bytes");
        }
        if (spec_.data_type == DataType::BOOL) {
            for (const std::byte value : source) {
                if (value != std::byte{0} && value != std::byte{1}) {
                    throw std::invalid_argument(
                        "BOOL host bytes must contain zero or one");
                }
            }
        }
        owner_->region_from_host(*this, source, workspace);
    }

    void TensorView::copy_to_host(
            std::span<std::byte> destination,
            RawWorkspaceView workspace) const {
        if (destination.size() != spec_.logical_nbytes()) {
            throw std::invalid_argument(
                "host destination must hold exactly logical_nbytes bytes");
        }
        owner_->region_to_host(*this, destination, workspace);
    }

    namespace {

        // Shared body of the two host-transfer requirement queries: the
        // checked logical-byte walk of the transfer itself plus the
        // backend's staging rule. Both current transfer directions share
        // the same preconditions; the data-dependent BOOL byte check is a
        // transfer-time predicate and intentionally not part of a pure
        // query. No allocation, registration, leasing, or native effect.
        WorkspaceRequirements host_transfer_workspace_requirements(
                const TensorView& view) {
            const std::size_t logical_nbytes =
                    view.spec().logical_nbytes();
            switch (view.backend_kind()) {
                case BackendKind::CPU:
                case BackendKind::TTNN:
                    return {0, 1};
                case BackendKind::CUDA:
                case BackendKind::ROCM:
                case BackendKind::SYCL:
                    return {gpu_algorithm::compute_staging_size(
                                    logical_nbytes),
                            32};
            }
            throw std::invalid_argument("tensor view has an unknown backend");
        }

    }  // namespace

    WorkspaceRequirements
            TensorView::copy_from_host_workspace_requirements() const {
        return host_transfer_workspace_requirements(*this);
    }

    WorkspaceRequirements
            TensorView::copy_to_host_workspace_requirements() const {
        return host_transfer_workspace_requirements(*this);
    }

    Tensor::Tensor(TensorSpec spec, Device& device)
            : device_(&device),
              full_view_(*this, spec, 0, validated_dense_plane_strides(spec)) {}

    TensorView& Tensor::view() noexcept {
        return full_view_;
    }

    const TensorView& Tensor::view() const noexcept {
        return full_view_;
    }

}  // namespace iom
