#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

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


        bool is_recognized(QuantizationFormat format) {
            switch (format) {
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
                case QuantizationFormat::TT_BFP8A: return true;
            }
            return false;
        }

    }  // namespace

    TensorShape::TensorShape(std::vector<std::size_t> dimensions)
            : dimensions_(std::move(dimensions)) {
        if (dimensions_.size() < 2) {
            throw std::invalid_argument("tensor shape requires rank of at least two");
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
        if (quantization == QuantizationFormat::NONE) {
            return;
        }
        if (is_recognized(quantization)) {
            throw std::runtime_error("grouped quantization formats are not supported");
        }
        throw std::invalid_argument("unknown QuantizationFormat value");
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

    }  // namespace

    DeviceOps::DeviceOps()
            : queue_id_(lease_queue_id()) {}

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
        return (static_cast<oid>(queue_id_) << kSequenceBits) | sequence;
    }

    void DeviceOps::wait(oid token) {
        const std::uint64_t id = token >> kSequenceBits;
        const std::uint64_t sequence = token & kSequenceMask;
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
        completion_cv_.wait(lock, [&] {
            return completed_ >= sequence || failures_.count(sequence) != 0;
        });
        if (const auto failure = failures_.find(sequence);
                failure != failures_.end()) {
            std::rethrow_exception(failure->second);
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

    void DeviceOps::commit_failure(
            std::uint64_t sequence, std::exception_ptr failure) {
        if (sequence == 0) {
            throw std::invalid_argument("retained failure sequence is zero");
        }
        if (!failure) {
            throw std::invalid_argument("retained failure is empty");
        }
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (sequence != next_sequence_) {
            throw std::invalid_argument(
                    "retained failure is not for the active submission");
        }
        pending_failures_[sequence] = std::move(failure);
    }

    void DeviceOps::seek_next_sequence(std::uint64_t next_sequence) {
        if (next_sequence == 0 || next_sequence < next_sequence_) {
            throw std::invalid_argument("queue sequence numbers only move forward");
        }
        std::lock_guard<std::mutex> lock(completion_mutex_);
        next_sequence_ = next_sequence;
    }
}  // namespace iom
