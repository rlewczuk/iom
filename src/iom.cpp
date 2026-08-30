#include "iom/tensor.hpp"

#include <cstddef>
#include <limits>
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
        static_cast<void>(leaf_bits(data_type));
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
                shape.element_count(), leaf_bits(data_type), "logical bit count overflows");
        return bits_to_bytes(bits, "logical byte count overflows");
    }

    std::size_t TensorSpec::tiled_storage_nbytes() const {
        validate();
        const std::size_t bits = checked_mul(
                standard_padded_shape().element_count(),
                leaf_bits(data_type),
                "tiled bit count overflows");
        return bits_to_bytes(bits, "tiled byte count overflows");
    }

    namespace detail {

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

            const std::size_t tile_rows = checked_add(
                    dimensions[leading_rank], kTile - 1, "tile row count overflows") / kTile;
            const std::size_t tile_columns = checked_add(
                    dimensions[leading_rank + 1], kTile - 1, "tile column count overflows") / kTile;
            const std::size_t tiles_per_plane = checked_mul(
                    tile_rows, tile_columns, "tile count overflows");

            const std::size_t row = coordinates[leading_rank];
            const std::size_t column = coordinates[leading_rank + 1];
            const std::size_t tile_row = row / kTile;
            const std::size_t tile_column = column / kTile;

            const std::size_t tile_index = checked_add(
                    checked_mul(plane, tiles_per_plane, "tile index overflows"),
                    checked_add(
                            checked_mul(tile_row, tile_columns, "tile index overflows"),
                            tile_column,
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

    }  // namespace detail

}  // namespace iom
