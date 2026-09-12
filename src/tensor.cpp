#include "iom/tensor.hpp"

#include "iom_internal.hpp"

#include <utility>

namespace iom {

    using detail::bits_to_bytes;
    using detail::checked_add;
    using detail::checked_mul;
    using detail::kMaxTensorRank;

    namespace {

        constexpr std::size_t kTile = TensorSpec::TILE;
        constexpr std::size_t kTileSlots = kTile * kTile;

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

}  // namespace iom
