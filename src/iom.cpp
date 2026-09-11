#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

#include <algorithm>
#include <bitset>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace iom {

    namespace {

        constexpr std::size_t kTile = TensorSpec::TILE;
        constexpr std::size_t kTileSlots = kTile * kTile;
        // Full tensor shapes span rank two through rank eight inclusive.
        // Implementation-private: no public rank constant or query API
        // exists, and a leading-dimension helper span is never a full shape.
        constexpr std::size_t kMaxTensorRank = 8;

        std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* what) {
            if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
                throw std::overflow_error(what);
            }
            return lhs + rhs;
        }

        std::size_t checked_mul(std::size_t lhs, std::size_t rhs, const char* what) {
            if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
                throw std::overflow_error(what);
            }
            return lhs * rhs;
        }

        std::size_t bits_to_bytes(std::size_t bits, const char* what) {
            return checked_add(bits, 7, what) / 8;
        }

    }  // namespace

    TensorShape::TensorShape(std::vector<std::size_t> dimensions)
            : dimensions_(std::move(dimensions)) {
        if (dimensions_.size() < 2) {
            throw std::invalid_argument("tensor shape requires rank of at least two");
        }
        if (dimensions_.size() > kMaxTensorRank) {
            throw std::invalid_argument("tensor shape requires rank of at most eight");
        }
        for (const std::size_t dimension : dimensions_) {
            if (dimension == 0) {
                throw std::invalid_argument("tensor dimensions must be nonzero");
            }
        }
    }

    std::size_t TensorShape::rank() const noexcept {
        return dimensions_.size();
    }

    std::size_t TensorShape::dimension(std::size_t index) const {
        return dimensions_.at(index);
    }

    std::span<const std::size_t> TensorShape::dimensions() const noexcept {
        return dimensions_;
    }

    std::size_t TensorShape::element_count() const {
        std::size_t count = 1;
        for (const std::size_t dimension : dimensions_) {
            count = checked_mul(count, dimension, "tensor element count overflows");
        }
        return count;
    }

    void TensorSpec::validate() const {
        static_cast<void>(detail::leaf_bits(data_type));
        const int raw_quantization = static_cast<int>(quantization);
        if (raw_quantization < static_cast<int>(QuantizationFormat::NONE)
            || raw_quantization > static_cast<int>(QuantizationFormat::TT_BFP8A)) {
            throw std::invalid_argument("unknown QuantizationFormat value");
        }
        if (quantization != QuantizationFormat::NONE) {
            throw std::runtime_error("grouped quantization formats are not supported");
        }
        if (shape.rank() < 2) {
            throw std::invalid_argument(
                    "tensor spec requires rank of at least two");
        }
        if (shape.rank() > kMaxTensorRank) {
            throw std::invalid_argument(
                    "tensor spec requires rank of at most eight");
        }
    }

    TensorShape TensorSpec::standard_padded_shape() const {
        const std::span<const std::size_t> dimensions = shape.dimensions();
        std::vector<std::size_t> padded{dimensions.begin(), dimensions.end()};
        for (const std::size_t i : {padded.size() - 2, padded.size() - 1}) {
            const std::size_t remainder = padded[i] % kTile;
            if (remainder != 0) {
                padded[i] = checked_add(padded[i], kTile - remainder, "padded dimension overflows");
            }
        }
        return TensorShape{padded};
    }

    std::size_t TensorSpec::logical_nbytes() const {
        validate();
        const std::size_t bits = checked_mul(
                shape.element_count(), detail::leaf_bits(data_type), "logical bit count overflows");
        return bits_to_bytes(bits, "logical byte count overflows");
    }

    std::size_t TensorSpec::tiled_storage_nbytes() const {
        validate();
        const std::size_t bits = checked_mul(
                standard_padded_shape().element_count(),
                detail::leaf_bits(data_type),
                "tiled bit count overflows");
        return bits_to_bytes(bits, "tiled byte count overflows");
    }

    namespace detail {

        // Exact bit width of every declared leaf encoding.
        std::size_t leaf_bits(DataType type) {
            switch (type) {
                case DataType::BOOL: return 8;
                case DataType::I2:
                case DataType::U2: return 2;
                case DataType::I4:
                case DataType::U4:
                case DataType::F4_E2M1: return 4;
                case DataType::F6_E2M3:
                case DataType::F6_E3M2: return 6;
                case DataType::I8:
                case DataType::U8:
                case DataType::F8_E4M3FN:
                case DataType::F8_E5M2:
                case DataType::F8_E8M0: return 8;
                case DataType::I16:
                case DataType::U16:
                case DataType::F16:
                case DataType::BF16: return 16;
                case DataType::I32:
                case DataType::U32:
                case DataType::F32: return 32;
                case DataType::I64:
                case DataType::U64:
                case DataType::F64: return 64;
            }
            throw std::invalid_argument("unknown DataType value");
        }

        std::size_t standard_plane_slot(
                const TensorSpec& spec, std::size_t plane,
                std::size_t row, std::size_t column) {
            const std::span<const std::size_t> dimensions = spec.shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;

            const std::size_t tile_rows = checked_add(
                    dimensions[leading_rank], kTile - 1, "tile row count overflows") / kTile;
            const std::size_t tile_columns = checked_add(
                    dimensions[leading_rank + 1], kTile - 1, "tile column count overflows") / kTile;
            const std::size_t tiles_per_plane = checked_mul(
                    tile_rows, tile_columns, "tile count overflows");

            const std::size_t tile_index = checked_add(
                    checked_mul(plane, tiles_per_plane, "tile index overflows"),
                    checked_add(
                            checked_mul(row / kTile, tile_columns, "tile index overflows"),
                            column / kTile,
                            "tile index overflows"),
                    "tile index overflows");

            return checked_add(
                    checked_mul(tile_index, kTileSlots, "element slot overflows"),
                    checked_add(
                            checked_mul(row % kTile, kTile, "element slot overflows"),
                            column % kTile,
                            "element slot overflows"),
                    "element slot overflows");
        }

        std::size_t standard_layout_slot(
                const TensorSpec& spec, std::span<const std::size_t> coordinates) {
            spec.validate();

            const TensorShape& shape = spec.shape;
            if (coordinates.size() != shape.rank()) {
                throw std::invalid_argument("coordinate count must equal tensor rank");
            }
            for (std::size_t i = 0; i < coordinates.size(); ++i) {
                if (coordinates[i] >= shape.dimension(i)) {
                    throw std::out_of_range("coordinate exceeds tensor dimension");
                }
            }

            const std::span<const std::size_t> dimensions = shape.dimensions();
            const std::size_t leading_rank = shape.rank() - 2;

            std::size_t plane = 0;
            for (std::size_t i = 0; i < leading_rank; ++i) {
                plane = checked_add(
                        checked_mul(plane, dimensions[i], "plane index overflows"),
                        coordinates[i],
                        "plane index overflows");
            }

            return standard_plane_slot(
                    spec, plane, coordinates[leading_rank], coordinates[leading_rank + 1]);
        }

    }  // namespace detail

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

    void TensorView::copy_from_host(std::span<const std::byte> source) {
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
        owner_->region_from_host(*this, source);
    }

    void TensorView::copy_to_host(std::span<std::byte> destination) const {
        if (destination.size() != spec_.logical_nbytes()) {
            throw std::invalid_argument(
                "host destination must hold exactly logical_nbytes bytes");
        }
        owner_->region_to_host(*this, destination);
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

    namespace {

        // The only global queue state: the live eight-bit queue ids. Bit i
        // represents id i + 1. It never selects a backend, device, or
        // runtime context.
        std::mutex g_queue_ids_mutex;
        std::bitset<255> g_live_queue_ids;
        struct UnsupportedOperation final : std::runtime_error {
            UnsupportedOperation()
                    : std::runtime_error("operation is unsupported") {}
        };
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
                             std::move(result_shape)};
    }




    void DeviceOps::validate_copy(
            const Device& device, const TensorView& source,
            const TensorView& destination) {
        validate_views(device, {&source, &destination});
        if (!(source.spec() == destination.spec())) {
            throw std::invalid_argument(
                    "copy views must have identical shape, leaf type, "
                    "and quantization");
        }
    }

    bool DeviceOps::identical_window(
            const TensorView& source, const TensorView& destination) {
        return source.native_handle() == destination.native_handle()
                && source.plane_offset() == destination.plane_offset()
                && std::equal(
                        source.plane_strides().begin(),
                        source.plane_strides().end(),
                        destination.plane_strides().begin(),
                        destination.plane_strides().end());
    }

    std::runtime_error DeviceOps::unsupported(
            std::string_view backend, std::string_view operation) {
        std::string message(backend);
        message += " backend does not implement ";
        message += operation;
        return std::runtime_error(std::move(message));
    }
    DeviceOps::DeviceOps()
            : queue_id_(lease_queue_id()) {}

    DeviceOps::DeviceOps(const Device& device)
            : device_(&device), queue_id_(lease_queue_id()) {}


    const Device& DeviceOps::queue_device() const {
        if (device_ == nullptr) {
            throw std::logic_error("operation queue has no device");
        }
        return *device_;
    }
    void DeviceOps::validate_views(
            const Device& device,
            std::initializer_list<const TensorView*> views) {
        for (const TensorView* view : views) {
            if (view == nullptr || &view->device() != &device) {
                throw std::invalid_argument(
                        "operation views must belong to the queue's device");
            }
        }
    }
    oid DeviceOps::binary_impl(const BinaryRequest&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::copy_impl(
            const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::silu_impl(const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::linear_impl(
            const TensorView&, const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::rmsnorm_impl(
            const TensorView&, TensorView&, const TensorView&, float, size_t) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::sdpa_impl(
            const TensorView&, const TensorView&, const TensorView&,
            size_t, size_t, size_t, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::map_failure(std::exception_ptr failure) noexcept {
        try {
            if (failure != nullptr) {
                std::rethrow_exception(failure);
            }
        } catch (const UnsupportedOperation&) {
            return to_oid(OidError::Unsupported);
        } catch (const std::bad_alloc&) {
            return to_oid(OidError::ResourceExhausted);
        } catch (const std::overflow_error&) {
            return to_oid(OidError::Overflow);
        } catch (const std::invalid_argument&) {
            return to_oid(OidError::InvalidArgument);
        } catch (const std::exception&) {
            return to_oid(OidError::DeviceError);
        } catch (...) {
            return to_oid(OidError::InternalError);
        }
        return to_oid(OidError::InternalError);
    }

    oid DeviceOps::invoke_failure(std::exception_ptr failure) noexcept {
        return map_failure(std::move(failure));
    }

    oid DeviceOps::invoke(oid result) noexcept {
        return result;
    }

    oid DeviceOps::copy(
            const TensorView& source, TensorView& destination) noexcept {
        try {
            validate_copy(queue_device(), source, destination);
            return invoke(copy_impl(source, destination));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }
    oid DeviceOps::add(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), BinaryOperation::Add,
                                    lhs, rhs, out);
            return invoke(binary_impl(request));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::mul(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), BinaryOperation::Mul,
                                    lhs, rhs, out);
            return invoke(binary_impl(request));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::sub(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), BinaryOperation::Sub,
                                    lhs, rhs, out);
            return invoke(binary_impl(request));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::div(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), BinaryOperation::Div,
                                    lhs, rhs, out);
            return invoke(binary_impl(request));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
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
            const TensorView& x, const TensorView& w,
            TensorView& y) noexcept {
        try {
            validate_views(queue_device(), {&x, &w, &y});
            return invoke(linear_impl(x, w, y));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::rmsnorm(
            const TensorView& x, TensorView& y, const TensorView& w,
            float eps, size_t dim) noexcept {
        try {
            validate_views(queue_device(), {&x, &y, &w});
            if (!(eps >= 0.0F) || dim == 0) {
                throw std::invalid_argument("invalid rmsnorm parameters");
            }
            return invoke(rmsnorm_impl(x, y, w, eps, dim));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
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
    DeviceOps::~DeviceOps() {
        release_queue_id(queue_id_);
    }

    std::uint8_t DeviceOps::lease_queue_id() {
        std::lock_guard<std::mutex> lock(g_queue_ids_mutex);
        for (std::size_t i = 0; i < g_live_queue_ids.size(); ++i) {
            if (!g_live_queue_ids.test(i)) {
                g_live_queue_ids.set(i);
                return static_cast<std::uint8_t>(i + 1);
            }
        }
        throw std::runtime_error("all 255 DeviceOps queue ids are live");
    }

    void DeviceOps::release_queue_id(std::uint8_t queue_id) noexcept {
        std::lock_guard<std::mutex> lock(g_queue_ids_mutex);
        g_live_queue_ids.reset(queue_id - 1);
    }

    oid DeviceOps::encode_token(std::uint64_t sequence) const noexcept {
        const std::uint64_t encoded =
                (std::uint64_t{queue_id_} << kSequenceBits) | sequence;
        return static_cast<oid>(encoded);
    }

    void DeviceOps::wait(oid token) {
        if (token <= 0) {
            throw std::invalid_argument("oid must be a positive token");
        }
        const std::uint64_t encoded = static_cast<std::uint64_t>(token);
        const std::uint64_t id = encoded >> kSequenceBits;
        const std::uint64_t sequence = encoded & kSequenceMask;
        if (id == 0) {
            throw std::invalid_argument("oid queue id is zero");
        }
        if (sequence == 0) {
            throw std::invalid_argument("oid sequence is zero");
        }
        if (id != queue_id_) {
            throw std::invalid_argument("oid belongs to another queue");
        }
        std::unique_lock<std::mutex> lock(completion_mutex_);

        if (sequence >= next_sequence_) {
            throw std::invalid_argument("oid sequence was never submitted");
        }
        const auto skipped = skipped_sequences_.upper_bound(sequence);
        if (skipped != skipped_sequences_.begin()) {
            auto previous = skipped;
            --previous;
            if (previous->second > sequence) {
                throw std::invalid_argument("oid sequence was never submitted");
            }
        }
        completion_cv_.wait(lock, [&] {
            return completed_ >= sequence || failures_.count(sequence) != 0;
        });
        if (const auto failure = failures_.find(sequence);
                failure != failures_.end()) {
            std::rethrow_exception(failure->second);
        }
        lock.unlock();
        fence_through_sequence(sequence);
        lock.lock();
        if (const auto failure = failures_.find(sequence);
                failure != failures_.end()) {
            std::rethrow_exception(failure->second);
        }
    }

    void DeviceOps::fence_through_sequence(
            std::uint64_t /*sequence*/) noexcept {}

    void DeviceOps::record_post_completion_failure(
            std::uint64_t sequence, std::exception_ptr failure) {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (failures_.find(sequence) == failures_.end()) {
            failures_.emplace(sequence, std::move(failure));
            completion_cv_.notify_all();
        }
    }

    void DeviceOps::complete(
            std::uint64_t sequence, std::exception_ptr failure) {
        if (sequence == 0) {
            throw std::invalid_argument("completion sequence is zero");
        }
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (sequence >= next_sequence_) {
            throw std::invalid_argument(
                    "completion of a sequence that was never submitted");
        }
        if (const auto pending = pending_failures_.find(sequence);
                pending != pending_failures_.end()) {
            failure = std::move(pending->second);
            pending_failures_.erase(pending);
        }
        if (failure) {
            failures_[sequence] = std::move(failure);
        }
        if (completed_ < sequence) {
            completed_ = sequence;
        }
        completion_cv_.notify_all();
    }

    /**
     * Retains a backend failure for a reserved sequence without completing
     * it. A normally allocated sequence is accepted when
     * sequence < next_sequence_ && sequence > completed_. A sequence at or
     * beyond next_sequence_ was never submitted, while one at or below
     * completed_ has already completed; ranges skipped by the test seam are
     * also never submitted.
     */
    void DeviceOps::commit_failure(
            std::uint64_t sequence, std::exception_ptr failure) {
        if (sequence == 0) {
            throw std::invalid_argument("retained failure sequence is zero");
        }
        if (!failure) {
            throw std::invalid_argument("retained failure is empty");
        }
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (sequence >= next_sequence_) {
            throw std::invalid_argument(
                    "retained failure is not for a reserved submission");
        }
        if (sequence <= completed_) {
            throw std::invalid_argument(
                    "retained failure is for a sequence that has already been completed");
        }
        const auto skipped = skipped_sequences_.upper_bound(sequence);
        if (skipped != skipped_sequences_.begin()) {
            auto previous = skipped;
            --previous;
            if (previous->second > sequence) {
                throw std::invalid_argument(
                        "retained failure is not for a reserved submission");
            }
        }
        pending_failures_[sequence] = std::move(failure);
    }

    void DeviceOps::seek_next_sequence(std::uint64_t next_sequence) {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (next_sequence == 0 || next_sequence < next_sequence_
                || next_sequence > kMaxSequence + 1) {
            throw std::invalid_argument("queue sequence numbers only move forward");
        }
        if (next_sequence > next_sequence_) {
            skipped_sequences_.emplace(next_sequence_, next_sequence);
        }
        next_sequence_ = next_sequence;
    }

}  // namespace iom
