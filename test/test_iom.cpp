#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "iom/tensor.hpp"

namespace {

iom::TensorShape make_shape(std::initializer_list<std::size_t> dimensions) {
    return iom::TensorShape{std::vector<std::size_t>{dimensions}};
}

std::vector<std::size_t> dimensions_of(const iom::TensorShape& shape) {
    return {shape.dimensions().begin(), shape.dimensions().end()};
}

iom::TensorSpec make_spec(
        std::vector<std::size_t> dimensions,
        iom::DataType data_type,
        iom::QuantizationFormat quantization = iom::QuantizationFormat::NONE) {
    return iom::TensorSpec{iom::TensorShape{std::move(dimensions)}, data_type, quantization};
}

std::size_t slot(
        const iom::TensorSpec& spec, std::initializer_list<std::size_t> coordinates) {
    return iom::detail::standard_layout_slot(spec, coordinates);
}

constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();

constexpr std::initializer_list<iom::DataType> kAllDataTypes = {
    iom::DataType::BOOL,
    iom::DataType::I2, iom::DataType::U2,
    iom::DataType::I4, iom::DataType::U4,
    iom::DataType::I8, iom::DataType::U8,
    iom::DataType::I16, iom::DataType::U16,
    iom::DataType::I32, iom::DataType::U32,
    iom::DataType::I64, iom::DataType::U64,
    iom::DataType::F4_E2M1,
    iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
    iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2, iom::DataType::F8_E8M0,
    iom::DataType::F16, iom::DataType::BF16,
    iom::DataType::F32, iom::DataType::F64,
};

static_assert(iom::TensorSpec::TILE == 16);

}  // namespace

TEST_CASE("TensorShape accepts ranks of two and above and round-trips dimensions") {
    const std::vector<std::vector<std::size_t>> cases = {
        {1, 1},
        {2, 3},
        {2, 3, 4},
        {2, 3, 4, 5},
        {2, 3, 4, 5, 6},
        {9, 7, 5, 3, 2, 1},
    };
    for (const auto& dimensions : cases) {
        CAPTURE(dimensions);
        const iom::TensorShape shape{dimensions};

        CHECK_EQ(shape.rank(), dimensions.size());
        std::size_t expected_count = 1;
        for (std::size_t i = 0; i < dimensions.size(); ++i) {
            CHECK_EQ(shape.dimension(i), dimensions[i]);
            expected_count *= dimensions[i];
        }
        CHECK_EQ(dimensions_of(shape), dimensions);
        CHECK_EQ(shape.element_count(), expected_count);
        CHECK_EQ(shape, iom::TensorShape{dimensions});
    }

    const iom::TensorShape shape = make_shape({2, 3});
    CHECK_NE(shape, make_shape({2, 4}));
    CHECK_NE(shape, make_shape({3, 2}));
    CHECK_NE(shape, make_shape({2, 3, 4}));
}

TEST_CASE("TensorShape rejects rank below two and zero dimensions") {
    const std::vector<std::size_t> empty;
    const std::vector<std::size_t> single{5};
    CHECK_THROWS_AS(iom::TensorShape{empty}, std::invalid_argument);
    CHECK_THROWS_AS(iom::TensorShape{single}, std::invalid_argument);

    for (std::size_t rank = 2; rank <= 5; ++rank) {
        for (std::size_t position = 0; position < rank; ++position) {
            std::vector<std::size_t> dimensions(rank, 1);
            dimensions[position] = 0;
            CAPTURE(rank);
            CAPTURE(position);
            CHECK_THROWS_AS(iom::TensorShape{dimensions}, std::invalid_argument);
        }
    }
}

TEST_CASE("TensorShape dimension throws out of range past the rank") {
    const iom::TensorShape shape = make_shape({2, 3});
    CHECK_THROWS_AS((void)shape.dimension(2), std::out_of_range);
    CHECK_THROWS_AS((void)shape.dimension(100), std::out_of_range);
}

TEST_CASE("TensorShape element count overflows instead of wrapping") {
    const iom::TensorShape shape = make_shape({kMax, 2});
    CHECK_THROWS_AS((void)shape.element_count(), std::overflow_error);
}

TEST_CASE("TensorSpec defaults quantization to NONE and compares by value") {
    const iom::TensorSpec spec = make_spec({2, 3}, iom::DataType::F16);
    CHECK(spec.quantization == iom::QuantizationFormat::NONE);
    CHECK(spec.shape == make_shape({2, 3}));

    iom::TensorSpec same = spec;
    CHECK(spec == same);
    same.quantization = iom::QuantizationFormat::GGML_Q4_0;
    CHECK(spec != same);
    same = spec;
    same.data_type = iom::DataType::F32;
    CHECK(spec != same);
    same = spec;
    same.shape = make_shape({2, 4});
    CHECK(spec != same);
}

TEST_CASE("TensorSpec validate accepts every declared DataType with NONE") {
    for (const iom::DataType data_type : kAllDataTypes) {
        CAPTURE(static_cast<int>(data_type));
        const iom::TensorSpec spec = make_spec({16, 16}, data_type);
        CHECK_NOTHROW(spec.validate());
        CHECK_NOTHROW((void)spec.logical_nbytes());
        CHECK_NOTHROW((void)spec.tiled_storage_nbytes());
    }
}

TEST_CASE("TensorSpec validate rejects every recognized grouped quantization format") {
    for (int value = 1; value <= 24; ++value) {
        CAPTURE(value);
        const auto quantization = static_cast<iom::QuantizationFormat>(value);
        const iom::TensorSpec spec = make_spec({16, 16}, iom::DataType::F32, quantization);
        CHECK_THROWS_AS(spec.validate(), std::runtime_error);
        CHECK_THROWS_AS((void)spec.logical_nbytes(), std::runtime_error);
        CHECK_THROWS_AS((void)spec.tiled_storage_nbytes(), std::runtime_error);
    }
}

TEST_CASE("TensorSpec validate rejects values outside either enum") {
    for (const int value : {25, 100, -1}) {
        CAPTURE(value);

        const auto quantization = static_cast<iom::QuantizationFormat>(value);
        const iom::TensorSpec grouped = make_spec({16, 16}, iom::DataType::F32, quantization);
        CHECK_THROWS_AS(grouped.validate(), std::invalid_argument);

        const auto data_type = static_cast<iom::DataType>(value);
        const iom::TensorSpec leaf = make_spec({16, 16}, data_type);
        CHECK_THROWS_AS(leaf.validate(), std::invalid_argument);
    }

    const auto unknown_data_type = static_cast<iom::DataType>(-1);
    const iom::TensorSpec both_invalid = make_spec(
            {16, 16}, unknown_data_type, iom::QuantizationFormat::GGML_Q4_0);
    CHECK_THROWS_AS(both_invalid.validate(), std::invalid_argument);
}

TEST_CASE("TensorSpec standard padded shape rounds only the final two dimensions") {
    struct PaddingCase {
        std::vector<std::size_t> logical;
        std::vector<std::size_t> padded;
    };
    const std::vector<PaddingCase> cases = {
        {{1, 17}, {16, 32}},
        {{2, 8, 31, 33}, {2, 8, 32, 48}},
        {{9, 7, 17, 1}, {9, 7, 32, 16}},
        {{3, 1, 4, 1, 5, 9}, {3, 1, 4, 1, 16, 16}},
        {{32, 32}, {32, 32}},
        {{16, 16}, {16, 16}},
    };
    for (const PaddingCase& padding_case : cases) {
        CAPTURE(padding_case.logical);
        const iom::TensorSpec spec{iom::TensorShape{padding_case.logical}, iom::DataType::I8};
        const iom::TensorShape padded = spec.standard_padded_shape();
        CHECK_EQ(dimensions_of(padded), padding_case.padded);
        CHECK(spec.shape == iom::TensorShape{padding_case.logical});
    }
}

TEST_CASE("TensorSpec padding overflow throws instead of wrapping") {
    const iom::TensorSpec spec = make_spec({2, kMax}, iom::DataType::I8);
    CHECK_THROWS_AS((void)spec.standard_padded_shape(), std::overflow_error);
}

TEST_CASE("TensorSpec reports exact leaf bit widths through byte counts") {
    const std::vector<std::pair<iom::DataType, std::size_t>> bits_per_element = {
        {iom::DataType::BOOL, 8},
        {iom::DataType::I2, 2}, {iom::DataType::U2, 2},
        {iom::DataType::I4, 4}, {iom::DataType::U4, 4}, {iom::DataType::F4_E2M1, 4},
        {iom::DataType::F6_E2M3, 6}, {iom::DataType::F6_E3M2, 6},
        {iom::DataType::I8, 8}, {iom::DataType::U8, 8},
        {iom::DataType::F8_E4M3FN, 8}, {iom::DataType::F8_E5M2, 8}, {iom::DataType::F8_E8M0, 8},
        {iom::DataType::I16, 16}, {iom::DataType::U16, 16},
        {iom::DataType::F16, 16}, {iom::DataType::BF16, 16},
        {iom::DataType::I32, 32}, {iom::DataType::U32, 32}, {iom::DataType::F32, 32},
        {iom::DataType::I64, 64}, {iom::DataType::U64, 64}, {iom::DataType::F64, 64},
    };
    for (const auto& [data_type, bits] : bits_per_element) {
        CAPTURE(static_cast<int>(data_type));

        const iom::TensorSpec eight_elements = make_spec({1, 8}, data_type);
        CHECK_EQ(eight_elements.logical_nbytes(), bits);

        const iom::TensorSpec one_tile = make_spec({16, 16}, data_type);
        const std::size_t tile_bytes = one_tile.tiled_storage_nbytes();
        CHECK_EQ(tile_bytes, 32 * bits);
        CHECK(tile_bytes % 32 == 0);
    }
}

TEST_CASE("TensorSpec logical byte counts round sub-byte elements up to bytes") {
    const std::vector<std::tuple<std::vector<std::size_t>, iom::DataType, std::size_t>> cases = {
        {{1, 3}, iom::DataType::I2, 1},
        {{1, 5}, iom::DataType::I2, 2},
        {{1, 1}, iom::DataType::I2, 1},
        {{1, 3}, iom::DataType::I4, 2},
        {{1, 3}, iom::DataType::F6_E2M3, 3},
        {{1, 1}, iom::DataType::BOOL, 1},
        {{1, 17}, iom::DataType::F32, 68},
        {{2, 8}, iom::DataType::BF16, 32},
    };
    for (const auto& [dimensions, data_type, expected_nbytes] : cases) {
        CAPTURE(dimensions);
        CAPTURE(static_cast<int>(data_type));
        const iom::TensorSpec spec = make_spec(dimensions, data_type);
        CHECK_EQ(spec.logical_nbytes(), expected_nbytes);
    }
}

TEST_CASE("TensorSpec tiled storage byte counts cover every padded slot") {
    const iom::TensorSpec rank_two = make_spec({1, 17}, iom::DataType::F32);
    CHECK_EQ(rank_two.logical_nbytes(), 68);
    CHECK_EQ(rank_two.tiled_storage_nbytes(), 16 * 32 * 4);

    const iom::TensorSpec rank_four = make_spec({2, 8, 31, 33}, iom::DataType::BF16);
    CHECK_EQ(rank_four.tiled_storage_nbytes(), 2 * 8 * 32 * 48 * 2);
}

TEST_CASE("TensorSpec logical byte count overflows instead of wrapping") {
    const iom::TensorSpec wide = make_spec({1, kMax}, iom::DataType::F32);
    CHECK_THROWS_AS((void)wide.logical_nbytes(), std::overflow_error);

    const std::size_t odd_bits = (std::size_t{1} << 63) - 1;
    const iom::TensorSpec round_up = make_spec({1, odd_bits}, iom::DataType::I2);
    CHECK(round_up.shape.element_count() == odd_bits);
    CHECK_THROWS_AS((void)round_up.logical_nbytes(), std::overflow_error);
}

TEST_CASE("TensorSpec tiled storage byte count overflows instead of wrapping") {
    const iom::TensorSpec spec = make_spec({1, std::size_t{1} << 56}, iom::DataType::F64);
    CHECK_NOTHROW((void)spec.shape.element_count());
    CHECK_NOTHROW((void)spec.standard_padded_shape().element_count());
    CHECK_THROWS_AS((void)spec.tiled_storage_nbytes(), std::overflow_error);
}

TEST_CASE("TensorSpec validates before calculating sizes") {
    const iom::TensorSpec grouped = make_spec(
            {1, std::size_t{1} << 62}, iom::DataType::F64, iom::QuantizationFormat::GGML_Q4_0);
    CHECK_THROWS_AS((void)grouped.logical_nbytes(), std::runtime_error);
    CHECK_THROWS_AS((void)grouped.tiled_storage_nbytes(), std::runtime_error);

    const auto unknown_data_type = static_cast<iom::DataType>(200);
    const iom::TensorSpec unknown = make_spec({1, std::size_t{1} << 62}, unknown_data_type);
    CHECK_THROWS_AS((void)unknown.logical_nbytes(), std::invalid_argument);
    CHECK_THROWS_AS((void)unknown.tiled_storage_nbytes(), std::invalid_argument);
}

TEST_CASE("Standard tiled layout addresses a 32x32 matrix in 16x16 tiles") {
    const iom::TensorSpec spec = make_spec({32, 32}, iom::DataType::F16);
    CHECK_EQ((slot(spec, {0, 0})), 0);
    CHECK_EQ((slot(spec, {0, 16})), 256);
    CHECK_EQ((slot(spec, {16, 0})), 512);
    CHECK_EQ((slot(spec, {16, 16})), 768);
    CHECK_EQ((slot(spec, {0, 15})), 15);
    CHECK_EQ((slot(spec, {15, 31})), 256 + 15 * 16 + 15);
    CHECK_EQ((slot(spec, {31, 31})), 3 * 256 + 15 * 16 + 15);
}

TEST_CASE("Standard tiled layout orders tiles within multiple leading planes") {
    const iom::TensorSpec spec = make_spec({2, 3, 16, 16}, iom::DataType::I8);
    CHECK_EQ((slot(spec, {0, 0, 0, 0})), 0);
    CHECK_EQ((slot(spec, {0, 0, 15, 15})), 255);
    CHECK_EQ((slot(spec, {0, 1, 0, 0})), 256);
    CHECK_EQ((slot(spec, {0, 2, 0, 0})), 512);
    CHECK_EQ((slot(spec, {1, 0, 0, 0})), 3 * 256);
    CHECK_EQ((slot(spec, {1, 2, 15, 15})), 5 * 256 + 255);
}

TEST_CASE("Standard tiled layout handles rank above four") {
    const iom::TensorSpec rank_five = make_spec({2, 2, 2, 16, 16}, iom::DataType::I8);
    CHECK_EQ((slot(rank_five, {1, 1, 1, 0, 0})), 7 * 256);
    CHECK_EQ((slot(rank_five, {0, 0, 1, 15, 15})), 256 + 255);

    const iom::TensorSpec rank_five_padded = make_spec({2, 3, 4, 17, 33}, iom::DataType::I8);
    CHECK_EQ((slot(rank_five_padded, {1, 2, 3, 16, 32})), 143 * 256);
}

TEST_CASE("Standard tiled layout handles non-square and padded matrices") {
    const iom::TensorSpec non_square = make_spec({20, 35}, iom::DataType::I8);
    CHECK_EQ((slot(non_square, {0, 0})), 0);
    CHECK_EQ((slot(non_square, {0, 16})), 256);
    CHECK_EQ((slot(non_square, {0, 34})), 2 * 256 + 2);
    CHECK_EQ((slot(non_square, {15, 34})), 2 * 256 + 15 * 16 + 2);
    CHECK_EQ((slot(non_square, {16, 0})), 3 * 256);

    const iom::TensorSpec padded = make_spec({2, 8, 31, 33}, iom::DataType::I8);
    CHECK_EQ((slot(padded, {0, 1, 0, 0})), 1 * 6 * 256);
    CHECK_EQ((slot(padded, {1, 0, 30, 32})), 53 * 256 + 14 * 16);
}

TEST_CASE("Standard tiled layout maps every logical element to a unique slot") {
    const std::vector<std::vector<std::size_t>> cases = {
        {20, 35},
        {2, 3, 17, 33},
    };
    for (const auto& logical : cases) {
        CAPTURE(logical);
        const iom::TensorSpec spec{iom::TensorShape{logical}, iom::DataType::I2};
        const iom::TensorShape padded = spec.standard_padded_shape();
        const std::size_t padded_count = padded.element_count();
        const std::size_t logical_count = spec.shape.element_count();
        REQUIRE(padded_count >= logical_count);

        std::vector<char> seen(padded_count, 0);
        std::size_t visited = 0;
        std::vector<std::size_t> coordinates(logical.size(), 0);
        for (;;) {
            const std::size_t address = iom::detail::standard_layout_slot(spec, coordinates);
            REQUIRE(address < padded_count);
            REQUIRE(seen[address] == 0);
            seen[address] = 1;
            ++visited;

            std::size_t axis = coordinates.size();
            for (; axis > 0; --axis) {
                ++coordinates[axis - 1];
                if (coordinates[axis - 1] < logical[axis - 1]) {
                    break;
                }
                coordinates[axis - 1] = 0;
            }
            if (axis == 0) {
                break;
            }
        }
        CHECK_EQ(visited, logical_count);
    }
}

TEST_CASE("Standard tiled layout validates coordinates") {
    const iom::TensorSpec spec = make_spec({2, 3, 16, 16}, iom::DataType::F32);

    CHECK_THROWS_AS((slot(spec, {0, 0, 0})), std::invalid_argument);
    CHECK_THROWS_AS((slot(spec, {0, 0, 0, 0, 0})), std::invalid_argument);

    CHECK_THROWS_AS((slot(spec, {2, 0, 0, 0})), std::out_of_range);
    CHECK_THROWS_AS((slot(spec, {0, 3, 0, 0})), std::out_of_range);
    CHECK_THROWS_AS((slot(spec, {0, 0, 16, 0})), std::out_of_range);
    CHECK_THROWS_AS((slot(spec, {0, 0, 0, 16})), std::out_of_range);
}

TEST_CASE("Standard tiled layout validates the spec before addressing") {
    const iom::TensorSpec grouped = make_spec(
            {2, 3, 16, 16}, iom::DataType::F32, iom::QuantizationFormat::OCP_MXFP4);
    CHECK_THROWS_AS((slot(grouped, {0, 0, 0, 0})), std::runtime_error);

    const auto unknown_data_type = static_cast<iom::DataType>(200);
    const iom::TensorSpec unknown = make_spec({2, 3, 16, 16}, unknown_data_type);
    CHECK_THROWS_AS((slot(unknown, {0, 0, 0, 0})), std::invalid_argument);
}

TEST_CASE("Standard tiled layout overflows instead of wrapping") {
    const iom::TensorSpec stride_overflow = make_spec({kMax / 2 + 1, 4, 16}, iom::DataType::I8);
    CHECK_THROWS_AS((slot(stride_overflow, {kMax / 2, 0, 0})), std::overflow_error);

    const iom::TensorSpec slot_overflow =
            make_spec({3, 16, std::size_t{1} << 60}, iom::DataType::I8);
    CHECK_THROWS_AS(
            (slot(slot_overflow, {2, 15, (std::size_t{1} << 60) - 1})), std::overflow_error);
}
