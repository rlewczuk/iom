#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/llama.hpp"
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

// ---------------------------------------------------------------------------
// Tensor owner and TensorView fixtures: a fake device and fake owner proving
// the common contract without any backend header or concrete storage.
// ---------------------------------------------------------------------------

namespace {

class FakeTensor final : public iom::Tensor {
public:
    FakeTensor(iom::TensorSpec spec, iom::Device& device)
            : iom::Tensor(std::move(spec), device) {}

    [[nodiscard]] void* storage_handle() noexcept override {
        return &storage_;
    }

    void region_from_host(
            const iom::TensorView& destination,
            std::span<const std::byte> source) override {
        ++from_host_calls;
        from_view_offset = destination.plane_offset();
        from_view_strides.assign(
                destination.plane_strides().begin(),
                destination.plane_strides().end());
        from_bytes.assign(source.begin(), source.end());
    }

    void region_to_host(
            const iom::TensorView& source,
            std::span<std::byte> destination) const override {
        ++to_host_calls;
        to_view_offset = source.plane_offset();
        to_view_strides.assign(
                source.plane_strides().begin(), source.plane_strides().end());
        to_destination_size = destination.size();
    }

    int storage_ = 0;

    std::size_t from_host_calls = 0;
    std::size_t from_view_offset = 0;
    std::vector<std::size_t> from_view_strides;
    std::vector<std::byte> from_bytes;

    mutable std::size_t to_host_calls = 0;
    mutable std::size_t to_view_offset = 0;
    mutable std::vector<std::size_t> to_view_strides;
    mutable std::size_t to_destination_size = 0;
};

// Deterministic deferred operation queue: submissions are recorded and the
// test thread plays the in-order worker by completing sequences through the
// common DeviceOps machinery. No threads, backends, or real work involved.
class FakeQueue final : public iom::DeviceOps {
public:
    FakeQueue() = default;

    using iom::DeviceOps::complete;
    using iom::DeviceOps::seek_next_sequence;

    struct Submission {
        std::uint64_t sequence;
        const char* op;
    };

    // Successfully queued operations in submission order.
    std::vector<Submission> submissions;

    bool silu_aliased = false;

    // A view-less submission used to observe queue identity and sequence
    // allocation directly.
    iom::oid probe() {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "probe"});
        });
    }

    iom::oid copy(const iom::TensorView& source, iom::TensorView& destination) override {
        if (source.spec() != destination.spec()) {
            throw std::invalid_argument("fake copy requires identical specs");
        }
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "copy"});
        });
    }

    iom::oid add(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "add"});
        });
    }

    iom::oid mul(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "mul"});
        });
    }

    iom::oid silu(const iom::TensorView& x, iom::TensorView& y) override {
        silu_aliased = &x == &y;
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "silu"});
        });
    }

    iom::oid linear(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "linear"});
        });
    }

    iom::oid rmsnorm(const iom::TensorView&, iom::TensorView&, const iom::TensorView&,
                     float, size_t) override {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "rmsnorm"});
        });
    }

    iom::oid sdpa(const iom::TensorView&, const iom::TensorView&, const iom::TensorView&,
                  size_t, size_t, size_t, iom::TensorView&) override {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "sdpa"});
        });
    }
};

class InlineQueue final : public iom::DeviceOps {
public:
    enum class Mode {
        complete,
        complete_with_failure,
        throw_before_complete,
        complete_then_throw,
        commit_failure_then_complete,
    };

    explicit InlineQueue(Mode mode = Mode::complete)
            : mode(mode) {}

    using iom::DeviceOps::commit_failure;
    using iom::DeviceOps::complete;
    using iom::DeviceOps::seek_next_sequence;
    using iom::DeviceOps::submit;

    iom::oid copy(
            const iom::TensorView& source,
            iom::TensorView& destination) override {
        if (source.spec() != destination.spec()) {
            throw std::invalid_argument("inline copy requires identical specs");
        }
        return submit([this](std::uint64_t sequence) {
            ++inline_calls;
            switch (mode) {
            case Mode::complete:
                complete(sequence);
                break;
            case Mode::complete_with_failure:
                complete(
                        sequence,
                        std::make_exception_ptr(
                                std::runtime_error("inline boom")));
                break;
            case Mode::throw_before_complete:
                throw std::runtime_error("event_create boom");
            case Mode::complete_then_throw:
                complete(sequence);
                throw std::runtime_error("post-complete boom");
            case Mode::commit_failure_then_complete:
                commit_failure(
                        sequence,
                        std::make_exception_ptr(
                                std::runtime_error("post-link boom")));
                complete(sequence);
                break;
            }
        });
    }

    iom::oid add(
            const iom::TensorView&, const iom::TensorView&, iom::TensorView&)
            override {
        throw unsupported("add");
    }

    iom::oid mul(
            const iom::TensorView&, const iom::TensorView&, iom::TensorView&)
            override {
        throw unsupported("mul");
    }

    iom::oid silu(const iom::TensorView&, iom::TensorView&) override {
        throw unsupported("silu");
    }

    iom::oid linear(
            const iom::TensorView&, const iom::TensorView&, iom::TensorView&)
            override {
        throw unsupported("linear");
    }

    iom::oid rmsnorm(
            const iom::TensorView&, iom::TensorView&, const iom::TensorView&,
            float, size_t) override {
        throw unsupported("rmsnorm");
    }

    iom::oid sdpa(
            const iom::TensorView&, const iom::TensorView&, const iom::TensorView&,
            size_t, size_t, size_t, iom::TensorView&) override {
        throw unsupported("sdpa");
    }

    Mode mode;
    std::size_t inline_calls = 0;

private:
    static std::runtime_error unsupported(const char* operation) {
        return std::runtime_error(
                std::string("inline queue does not implement ") + operation);
    }
};

constexpr std::uint64_t kTokenSequenceBits = 56;
constexpr std::uint64_t kTokenSequenceMask = (std::uint64_t{1} << kTokenSequenceBits) - 1;
constexpr std::uint64_t kMaxSequence = kTokenSequenceMask;

std::uint8_t token_queue(iom::oid token) {
    return static_cast<std::uint8_t>(token >> kTokenSequenceBits);
}

std::uint64_t token_sequence(iom::oid token) {
    return token & kTokenSequenceMask;
}

iom::oid make_token(std::uint64_t queue_id, std::uint64_t sequence) {
    return (queue_id << kTokenSequenceBits) | sequence;
}

std::vector<std::string_view> op_names(const FakeQueue& queue) {
    std::vector<std::string_view> names;
    for (const FakeQueue::Submission& entry : queue.submissions) {
        names.push_back(entry.op);
    }
    return names;
}

class FakeDevice final : public iom::Device {

public:
    [[nodiscard]] iom::BackendKind backend_kind() const noexcept override {
        return iom::BackendKind::CPU;
    }

    [[nodiscard]] std::uint32_t backend_device() const noexcept override {
        return 3;
    }

    [[nodiscard]] std::unique_ptr<iom::Tensor> create_tensor(
            const iom::TensorSpec& spec) override {
        return std::make_unique<FakeTensor>(spec, *this);
    }

    // The deterministic deferred queue stands in for the concrete backends.
    [[nodiscard]] std::unique_ptr<iom::DeviceOps> create_ops() override {
        return std::make_unique<FakeQueue>();
    }
};

FakeTensor make_tensor(
        iom::Device& device,
        std::vector<std::size_t> dimensions,
        iom::DataType data_type = iom::DataType::F32) {
    return FakeTensor{make_spec(std::move(dimensions), data_type), device};
}

std::vector<std::size_t> dims_of(const iom::TensorView& view) {
    return {view.spec().shape.dimensions().begin(),
            view.spec().shape.dimensions().end()};
}

std::vector<std::size_t> leading_of(const iom::TensorView& view) {
    const std::span<const std::size_t> dims = view.spec().shape.dimensions();
    return {dims.begin(), dims.end() - 2};
}

std::vector<std::size_t> strides_of(const iom::TensorView& view) {
    return {view.plane_strides().begin(), view.plane_strides().end()};
}

std::size_t view_plane(
        const iom::TensorView& view, std::span<const std::size_t> coordinates) {
    REQUIRE(coordinates.size() == view.plane_strides().size());
    std::size_t plane = view.plane_offset();
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        plane += coordinates[i] * view.plane_strides()[i];
    }
    return plane;
}

std::vector<std::vector<std::size_t>> permutations_of(std::size_t rank) {
    std::vector<std::vector<std::size_t>> result;
    std::vector<std::size_t> current(rank);
    std::iota(current.begin(), current.end(), std::size_t{0});
    do {
        result.push_back(current);
    } while (std::next_permutation(current.begin(), current.end()));
    return result;
}

// A span over a temporary initializer-list array; valid for the full
// expression that consumes it.
std::span<const std::size_t> span_of(std::initializer_list<std::size_t> values) {
    return {values.begin(), values.size()};
}

// Independent reference model of nested transforms. Every derived axis
// tracks the original leading axes it covers; selected axes are pinned to
// their chosen original coordinates. No iom code participates in the
// model's arithmetic.
struct ModelAxis {
    std::vector<std::size_t> axes;
    std::vector<std::size_t> dims;
    std::size_t scale = 1;
    std::size_t base = 0;
};

struct Model {
    std::vector<ModelAxis> axes;
    std::vector<std::pair<std::size_t, std::size_t>> fixed;
};

Model dense_model(std::span<const std::size_t> leading_dims) {
    Model model;
    for (std::size_t i = 0; i < leading_dims.size(); ++i) {
        model.axes.push_back(ModelAxis{{i}, {leading_dims[i]}, 1, 0});
    }
    return model;
}

void slice_model(
        Model& model, std::size_t axis, std::size_t first, std::size_t step) {
    model.axes[axis].base += first;
    model.axes[axis].scale *= step;
}

void select_model(Model& model, std::size_t axis, std::size_t index) {
    const ModelAxis selected = model.axes[axis];
    std::size_t remainder = selected.base + selected.scale * index;
    for (std::size_t k = selected.axes.size(); k-- > 0;) {
        model.fixed.emplace_back(selected.axes[k], remainder % selected.dims[k]);
        remainder /= selected.dims[k];
    }
    model.axes.erase(model.axes.begin() + static_cast<std::ptrdiff_t>(axis));
}

void permute_model(Model& model, std::span<const std::size_t> order) {
    std::vector<ModelAxis> reordered;
    reordered.reserve(order.size());
    for (const std::size_t axis : order) {
        reordered.push_back(model.axes[axis]);
    }
    model.axes = std::move(reordered);
}

// Merge consecutive model axes into groups, mirroring reshape_leading on a
// contiguous source: every grouped run of original axes becomes one derived
// axis. Parts must still be unscaled and unshifted, as they are directly
// after the dense reshape under test.
void reshape_model(Model& model, std::span<const std::size_t> group_sizes) {
    std::vector<ModelAxis> grouped;
    grouped.reserve(group_sizes.size());
    std::size_t next = 0;
    for (const std::size_t count : group_sizes) {
        REQUIRE(count >= 1);
        REQUIRE(next + count <= model.axes.size());
        ModelAxis merged;
        for (std::size_t k = 0; k < count; ++k) {
            const ModelAxis& part = model.axes[next + k];
            REQUIRE(part.scale == 1);
            REQUIRE(part.base == 0);
            merged.axes.insert(
                    merged.axes.end(), part.axes.begin(), part.axes.end());
            merged.dims.insert(
                    merged.dims.end(), part.dims.begin(), part.dims.end());
        }
        next += count;
        grouped.push_back(std::move(merged));
    }
    REQUIRE(next == model.axes.size());
    model.axes = std::move(grouped);
}

std::size_t model_plane(
        const Model& model,
        const std::vector<std::size_t>& dense_strides,
        std::span<const std::size_t> coordinates) {
    REQUIRE(coordinates.size() == model.axes.size());
    std::vector<std::size_t> mapped(dense_strides.size(), 0);
    std::vector<bool> pinned(dense_strides.size(), false);
    for (std::size_t j = 0; j < model.axes.size(); ++j) {
        const ModelAxis& axis = model.axes[j];
        std::size_t remainder = axis.base + axis.scale * coordinates[j];
        for (std::size_t k = axis.axes.size(); k-- > 0;) {
            mapped[axis.axes[k]] = remainder % axis.dims[k];
            pinned[axis.axes[k]] = true;
            remainder /= axis.dims[k];
        }
    }
    for (const auto& [axis, coordinate] : model.fixed) {
        mapped[axis] = coordinate;
        pinned[axis] = true;
    }
    std::size_t plane = 0;
    for (std::size_t i = 0; i < dense_strides.size(); ++i) {
        REQUIRE(pinned[i]);
        plane += mapped[i] * dense_strides[i];
    }
    return plane;
}

}  // namespace

TEST_CASE("Tensor owner constructs through a device with a stable full view") {
    static_assert(!std::is_default_constructible_v<iom::Tensor>);
    static_assert(!std::is_copy_constructible_v<iom::Tensor>);
    static_assert(!std::is_copy_assignable_v<iom::Tensor>);
    static_assert(!std::is_move_constructible_v<iom::Tensor>);
    static_assert(!std::is_move_assignable_v<iom::Tensor>);
    static_assert(std::is_abstract_v<iom::Device>);
    static_assert(std::is_copy_constructible_v<iom::TensorView>);
    static_assert(std::is_move_constructible_v<iom::TensorView>);
    static_assert(!std::is_copy_assignable_v<iom::TensorView>);
    static_assert(!std::is_move_assignable_v<iom::TensorView>);

    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 16, 16}, iom::DataType::F16);

    iom::TensorView& view = tensor.view();
    CHECK(&tensor.view() == &view);
    const iom::Tensor& frozen = tensor;
    CHECK(&frozen.view() == &view);

    CHECK(view.spec() == make_spec({2, 3, 16, 16}, iom::DataType::F16));
    CHECK_EQ(view.plane_offset(), 0);
    CHECK_EQ(strides_of(view), std::vector<std::size_t>({3, 1}));

    CHECK(view.native_handle() == tensor.storage_handle());
    CHECK(std::as_const(view).native_handle() == tensor.storage_handle());

    // Rank two has exactly one plane and an empty stride vector.
    FakeTensor rank_two = make_tensor(device, {4, 9}, iom::DataType::I8);
    CHECK(rank_two.view().plane_strides().empty());
    CHECK_EQ(rank_two.view().plane_offset(), 0);

    // Dense row-major plane strides at higher ranks.
    FakeTensor rank_five = make_tensor(device, {2, 3, 4, 5, 16, 16});
    CHECK_EQ(
            strides_of(rank_five.view()),
            std::vector<std::size_t>({60, 20, 5, 1}));

    std::unique_ptr<iom::Tensor> owned =
            device.create_tensor(make_spec({2, 3, 16, 16}, iom::DataType::F16));
    REQUIRE(owned != nullptr);
    CHECK(&owned->view() == &owned->view());
    CHECK(owned->view().native_handle() != nullptr);
}

TEST_CASE("Device owners are non-copyable and non-movable") {
    static_assert(!std::is_copy_constructible_v<FakeDevice>);
    static_assert(!std::is_copy_assignable_v<FakeDevice>);
    static_assert(!std::is_move_constructible_v<FakeDevice>);
    static_assert(!std::is_move_assignable_v<FakeDevice>);
}

TEST_CASE("Tensor owner validates its specification") {
    FakeDevice device;

    const iom::TensorSpec grouped = make_spec(
            {16, 16}, iom::DataType::F32, iom::QuantizationFormat::GGML_Q4_0);
    CHECK_THROWS_AS((FakeTensor{grouped, device}), std::runtime_error);

    const auto unknown = static_cast<iom::DataType>(77);
    CHECK_THROWS_AS(
            (FakeTensor{make_spec({16, 16}, unknown), device}),
            std::invalid_argument);
}

TEST_CASE("TensorView accessors report owner and view state") {
    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 17, 33}, iom::DataType::I8);

    const iom::TensorView& view = tensor.view();
    CHECK(&view.device() == &device);
    CHECK(view.backend_kind() == iom::BackendKind::CPU);
    CHECK_EQ(view.backend_device(), 3);
    CHECK(view.spec() == make_spec({2, 3, 17, 33}, iom::DataType::I8));

    // Copies stay pinned to the same owner and report its storage handle.
    iom::TensorView copy = view;
    CHECK(&copy.device() == &device);
    CHECK(copy.native_handle() == tensor.storage_handle());

    iom::TensorView derived = copy.slice(0, 1, 1).select(1, 2);
    CHECK(&derived.device() == &device);
    CHECK_EQ(derived.backend_device(), 3);
    CHECK(derived.native_handle() == tensor.storage_handle());
}

TEST_CASE("TensorView slice transforms one leading dimension") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {10, 20, 16, 16});
    const iom::TensorView full = tensor.view();

    const iom::TensorView middle = full.slice(0, 3, 4);
    CHECK_EQ(leading_of(middle), std::vector<std::size_t>({4, 20}));
    CHECK_EQ(strides_of(middle), std::vector<std::size_t>({20, 1}));
    CHECK_EQ(middle.plane_offset(), 60);

    const iom::TensorView stepped = full.slice(1, 2, 5, 3);
    CHECK_EQ(leading_of(stepped), std::vector<std::size_t>({10, 5}));
    CHECK_EQ(strides_of(stepped), std::vector<std::size_t>({20, 3}));
    CHECK_EQ(stepped.plane_offset(), 2);

    const iom::TensorView nested = full.slice(0, 2, 6).slice(1, 1, 3, 2);
    CHECK_EQ(leading_of(nested), std::vector<std::size_t>({6, 3}));
    CHECK_EQ(strides_of(nested), std::vector<std::size_t>({20, 2}));
    CHECK_EQ(nested.plane_offset(), 41);

    // A full-extent unstepped slice preserves the mapping exactly.
    const iom::TensorView identity = full.slice(1, 0, 20);
    CHECK_EQ(dims_of(identity), dims_of(full));
    CHECK_EQ(strides_of(identity), strides_of(full));
    CHECK_EQ(identity.plane_offset(), 0);

    // The last selected index may sit on the final coordinate.
    const iom::TensorView tail = full.slice(0, 9, 1, 1);
    CHECK_EQ(tail.plane_offset(), 180);
    CHECK_EQ(leading_of(tail), std::vector<std::size_t>({1, 20}));
}

TEST_CASE("TensorView select erases exactly one leading axis") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {2, 3, 4, 16, 16});
    const iom::TensorView full = tensor.view();

    const iom::TensorView picked = full.select(1, 2);
    CHECK_EQ(leading_of(picked), std::vector<std::size_t>({2, 4}));
    // The remaining strides are kept, not recomputed (dense would be {4, 1}).
    CHECK_EQ(strides_of(picked), std::vector<std::size_t>({12, 1}));
    CHECK_EQ(picked.plane_offset(), 8);

    const iom::TensorView reduced = picked.select(0, 1);
    CHECK_EQ(leading_of(reduced), std::vector<std::size_t>({4}));
    CHECK_EQ(strides_of(reduced), std::vector<std::size_t>({1}));
    CHECK_EQ(reduced.plane_offset(), 20);

    const iom::TensorView plane = reduced.select(0, 3);
    CHECK_EQ(plane.spec().shape.rank(), 2);
    CHECK(plane.plane_strides().empty());
    CHECK_EQ(plane.plane_offset(), 23);

    // Earlier strides survive a stepped slice untouched.
    const iom::TensorView stepped = full.slice(1, 0, 2, 1).select(2, 1);
    CHECK_EQ(leading_of(stepped), std::vector<std::size_t>({2, 2}));
    CHECK_EQ(strides_of(stepped), std::vector<std::size_t>({12, 4}));
    CHECK_EQ(stepped.plane_offset(), 1);
}

TEST_CASE("TensorView permute reorders leading axes exactly") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {2, 3, 4, 16, 16});
    const iom::TensorView full = tensor.view();

    const std::size_t dims[] = {2, 3, 4};
    const std::size_t strides[] = {12, 4, 1};
    for (const auto& order : permutations_of(3)) {
        CAPTURE(order);
        const iom::TensorView permuted = full.permute(order);
        CHECK_EQ(permuted.plane_offset(), 0);
        std::vector<std::size_t> want_dims(3);
        std::vector<std::size_t> want_strides(3);
        for (std::size_t i = 0; i < 3; ++i) {
            want_dims[i] = dims[order[i]];
            want_strides[i] = strides[order[i]];
        }
        CHECK_EQ(leading_of(permuted), want_dims);
        CHECK_EQ(strides_of(permuted), want_strides);
    }

    // The required rank-two empty permutation is a no-op.
    const FakeTensor rank_two = make_tensor(device, {4, 9});
    const iom::TensorView nothing = rank_two.view().permute(span_of({}));
    CHECK_EQ(dims_of(nothing), std::vector<std::size_t>({4, 9}));
    CHECK(nothing.plane_strides().empty());
    CHECK_EQ(nothing.plane_offset(), 0);

    // An offset survives permuting a sliced view.
    const iom::TensorView moved = full.slice(0, 1, 1).permute(span_of({2, 1, 0}));
    CHECK_EQ(moved.plane_offset(), 12);
    CHECK_EQ(leading_of(moved), std::vector<std::size_t>({4, 3, 1}));
    CHECK_EQ(strides_of(moved), std::vector<std::size_t>({1, 4, 12}));
}

TEST_CASE("TensorView reshape_leading splits merges and edits size-one axes") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {2, 3, 4, 5, 16, 16});
    const iom::TensorView full = tensor.view();

    struct ReshapeCase {
        std::vector<std::size_t> leading;
        std::vector<std::size_t> strides;
    };
    const std::vector<ReshapeCase> cases = {
        {{2, 3, 4, 5}, {60, 20, 5, 1}},          // identity
        {{6, 20}, {20, 1}},                      // merge
        {{120}, {1}},                            // full merge
        {{12, 10}, {10, 1}},                     // merge and split
        {{4, 6, 5}, {30, 5, 1}},                 // split
        {{2, 12, 5}, {60, 5, 1}},                // interior merge
        {{1, 2, 3, 4, 5}, {120, 60, 20, 5, 1}},  // size-one insert
    };
    for (const ReshapeCase& reshape_case : cases) {
        CAPTURE(reshape_case.leading);
        const iom::TensorView reshaped = full.reshape_leading(reshape_case.leading);
        CHECK_EQ(reshaped.plane_offset(), 0);
        CHECK_EQ(leading_of(reshaped), reshape_case.leading);
        CHECK_EQ(strides_of(reshaped), reshape_case.strides);
    }

    // The offset survives reshaping.
    const iom::TensorView moved =
            full.slice(0, 1, 1).reshape_leading(span_of({60}));
    CHECK_EQ(moved.plane_offset(), 60);
    CHECK_EQ(leading_of(moved), std::vector<std::size_t>({60}));
    CHECK_EQ(strides_of(moved), std::vector<std::size_t>({1}));

    // Size-one axes never constrain their stored stride.
    const iom::TensorView padded_axis =
            full.reshape_leading(span_of({1, 120})).slice(0, 0, 1, 7);
    CHECK_EQ(strides_of(padded_axis), std::vector<std::size_t>({840, 1}));
    const iom::TensorView merged = padded_axis.reshape_leading(span_of({120}));
    CHECK_EQ(leading_of(merged), std::vector<std::size_t>({120}));
    CHECK_EQ(strides_of(merged), std::vector<std::size_t>({1}));

    // Empty leading dimensions carry plane count one across the boundary.
    const FakeTensor rank_two = make_tensor(device, {4, 9});
    const iom::TensorView empty = rank_two.view().reshape_leading(span_of({}));
    CHECK_EQ(dims_of(empty), std::vector<std::size_t>({4, 9}));
    CHECK(empty.plane_strides().empty());

    const iom::TensorView single = rank_two.view().reshape_leading(span_of({1}));
    CHECK_EQ(dims_of(single), std::vector<std::size_t>({1, 4, 9}));
    CHECK_EQ(strides_of(single), std::vector<std::size_t>({1}));

    const iom::TensorView collapsed = single.reshape_leading(span_of({}));
    CHECK_EQ(dims_of(collapsed), std::vector<std::size_t>({4, 9}));
    CHECK(collapsed.plane_strides().empty());
}

TEST_CASE("TensorView reshape_leading maps every coordinate through the source layout") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {4, 3, 2, 16, 16});

    const iom::TensorView reshaped =
            tensor.view().reshape_leading(span_of({2, 12}));
    CHECK_EQ(leading_of(reshaped), std::vector<std::size_t>({2, 12}));
    CHECK_EQ(strides_of(reshaped), std::vector<std::size_t>({12, 1}));
    CHECK_EQ(reshaped.plane_offset(), 0);

    for (std::size_t c0 = 0; c0 < 2; ++c0) {
        for (std::size_t c1 = 0; c1 < 12; ++c1) {
            const std::size_t flat = c0 * 12 + c1;
            const std::size_t d0 = flat / 6;
            const std::size_t d1 = (flat / 2) % 3;
            const std::size_t d2 = flat % 2;
            CHECK_EQ(view_plane(reshaped, span_of({c0, c1})), d0 * 6 + d1 * 2 + d2);
        }
    }

    // A split with inserted and removed size-one axes keeps the same mapping.
    const iom::TensorView split =
            tensor.view().reshape_leading(span_of({1, 4, 1, 3, 2, 1}));
    CHECK_EQ(strides_of(split), std::vector<std::size_t>({24, 6, 6, 2, 1, 1}));
    for (std::size_t c1 = 0; c1 < 4; ++c1) {
        for (std::size_t c3 = 0; c3 < 3; ++c3) {
            for (std::size_t c4 = 0; c4 < 2; ++c4) {
                const std::size_t flat = (c1 * 3 + c3) * 2 + c4;
                CHECK_EQ(
                        view_plane(split, span_of({0, c1, 0, c3, c4, 0})),
                        flat);
            }
        }
    }
}

TEST_CASE("TensorView nested transforms match an independent reference plane map") {
    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 4, 5, 16, 16});
    const std::vector<std::size_t> dense_strides = {60, 20, 5, 1};

    Model model = dense_model(span_of({2, 3, 4, 5}));

    // Reshape groups consecutive original axes: {0}, {1}, {2, 3}.
    const iom::TensorView grouped =
            tensor.view().reshape_leading(span_of({2, 3, 20}));
    reshape_model(model, span_of({1, 1, 2}));

    const iom::TensorView sliced = grouped.slice(0, 1, 1, 1);
    slice_model(model, 0, 1, 1);

    const iom::TensorView picked = sliced.select(1, 1);
    select_model(model, 1, 1);

    const iom::TensorView stepped = picked.slice(1, 2, 6, 3);
    slice_model(model, 1, 2, 3);

    const iom::TensorView view = stepped.permute(span_of({1, 0}));
    permute_model(model, span_of({1, 0}));

    CHECK_EQ(leading_of(view), std::vector<std::size_t>({6, 1}));
    CHECK_EQ(strides_of(view), std::vector<std::size_t>({3, 60}));
    CHECK_EQ(view.plane_offset(), 82);

    std::vector<std::size_t> coordinates(view.spec().shape.rank() - 2, 0);
    std::size_t visited = 0;
    for (;;) {
        CAPTURE(coordinates);
        CHECK_EQ(
                view_plane(view, coordinates),
                model_plane(model, dense_strides, coordinates));
        ++visited;

        const std::vector<std::size_t> extents = dims_of(view);
        std::size_t axis = coordinates.size();
        for (; axis > 0; --axis) {
            ++coordinates[axis - 1];
            if (coordinates[axis - 1] < extents[axis - 1]) {
                break;
            }
            coordinates[axis - 1] = 0;
        }
        if (axis == 0) {
            break;
        }
    }
    CHECK_EQ(visited, 6);
}

TEST_CASE("TensorView rejects invalid transforms") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {2, 3, 4, 5, 16, 16});
    const iom::TensorView full = tensor.view();
    const FakeTensor rank_two = make_tensor(device, {4, 9});

    // Leading dimensions only: tiled dimensions and past-the-rank rejected.
    CHECK_THROWS_AS((void)full.slice(4, 0, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.slice(5, 0, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.slice(6, 0, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.select(4, 0), std::out_of_range);
    CHECK_THROWS_AS((void)full.select(6, 0), std::out_of_range);
    CHECK_THROWS_AS((void)rank_two.view().slice(0, 0, 1), std::out_of_range);
    CHECK_THROWS_AS((void)rank_two.view().select(0, 0), std::out_of_range);

    // Zero count or step.
    CHECK_THROWS_AS((void)full.slice(0, 0, 0), std::invalid_argument);
    CHECK_THROWS_AS((void)full.slice(0, 0, 1, 0), std::invalid_argument);

    // Out-of-range coordinates, including the last selected index boundary.
    CHECK_THROWS_AS((void)full.slice(0, 2, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.slice(0, 1, 2, 2), std::out_of_range);
    CHECK_THROWS_AS((void)full.slice(2, 1, 3, 2), std::out_of_range);
    CHECK_THROWS_AS((void)full.select(1, 3), std::out_of_range);
    CHECK_THROWS_AS((void)full.select(2, 4), std::out_of_range);

    // Permutations must be exact.
    CHECK_THROWS_AS((void)full.permute(span_of({0, 1, 2})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.permute(span_of({0, 1, 2, 3, 0})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.permute(span_of({0, 0, 1, 2})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.permute(span_of({0, 1, 2, 4})), std::invalid_argument);

    // Reshape rejections: zero dimensions, plane-count mismatch.
    CHECK_THROWS_AS(
            (void)full.reshape_leading(span_of({2, 0, 20})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.reshape_leading(span_of({119})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.reshape_leading(span_of({7, 18})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.reshape_leading(span_of({})), std::invalid_argument);

    // Reshape rejections: non-contiguous sources.
    const iom::TensorView stepped = full.slice(2, 0, 2, 2);
    CHECK_THROWS_AS(
            (void)stepped.reshape_leading(span_of({120})), std::invalid_argument);
    const iom::TensorView holed = full.select(1, 0);
    CHECK_THROWS_AS(
            (void)holed.reshape_leading(span_of({60})), std::invalid_argument);
    const iom::TensorView reordered = full.permute(span_of({0, 2, 1, 3}));
    CHECK_THROWS_AS(
            (void)reordered.reshape_leading(span_of({120})), std::invalid_argument);
}

TEST_CASE("TensorView transform arithmetic overflows instead of wrapping") {
    FakeDevice device;

    // The last-index arithmetic wraps before the bounds check can run.
    const FakeTensor small = make_tensor(device, {2, 16, 16});
    CHECK_THROWS_AS(
            (void)small.view().slice(0, kMax - 1, 2, 2), std::overflow_error);

    // Offset arithmetic overflows even for in-bounds coordinates.
    const FakeTensor wide = make_tensor(device, {kMax, 16, 16, 16});
    CHECK_THROWS_AS(
            (void)wide.view().slice(0, kMax - 1, 1), std::overflow_error);
    CHECK_THROWS_AS(
            (void)wide.view().select(0, kMax - 1), std::overflow_error);

    // Plane-count products overflow on both sides of the reshape.
    const FakeTensor huge_planes = make_tensor(device, {kMax, 2, 16, 16});
    CHECK_THROWS_AS(
            (void)huge_planes.view().reshape_leading(span_of({1})),
            std::overflow_error);
    CHECK_THROWS_AS(
            (void)huge_planes.view().reshape_leading(span_of({kMax, kMax})),
            std::overflow_error);
}

TEST_CASE("TensorView host transfers delegate the logical region to the owner") {
    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 16, 16}, iom::DataType::BOOL);
    const std::size_t nbytes = tensor.view().spec().logical_nbytes();
    REQUIRE(nbytes == 2 * 3 * 16 * 16);

    std::vector<std::byte> source(nbytes);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<std::byte>(i % 2);
    }

    tensor.view().copy_from_host(source);
    CHECK_EQ(tensor.from_host_calls, 1);
    CHECK(tensor.from_bytes == source);
    CHECK_EQ(tensor.from_view_offset, 0);
    CHECK_EQ(tensor.from_view_strides, std::vector<std::size_t>({3, 1}));

    // A derived view delegates itself, not the full view.
    iom::TensorView derived = tensor.view().slice(0, 1, 1).select(1, 2);
    CHECK_EQ(derived.plane_offset(), 5);
    CHECK_EQ(strides_of(derived), std::vector<std::size_t>({3}));
    const std::size_t derived_nbytes = derived.spec().logical_nbytes();
    std::vector<std::byte> small(derived_nbytes, std::byte{1});
    derived.copy_from_host(small);
    CHECK_EQ(tensor.from_host_calls, 2);
    CHECK(tensor.from_bytes == small);
    CHECK_EQ(tensor.from_view_offset, 5);
    CHECK_EQ(tensor.from_view_strides, std::vector<std::size_t>({3}));

    // Wrong byte counts are rejected before the owner hook runs.
    CHECK_THROWS_AS(
            tensor.view().copy_from_host(
                    std::span<const std::byte>(source.data(), nbytes - 1)),
            std::invalid_argument);
    CHECK_THROWS_AS(
            tensor.view().copy_from_host(
                    std::span<const std::byte>(source.data(), nbytes + 1)),
            std::invalid_argument);
    CHECK_EQ(tensor.from_host_calls, 2);

    // Non-canonical BOOL bytes are rejected before any backend write.
    std::vector<std::byte> invalid = source;
    invalid[0] = std::byte{2};
    CHECK_THROWS_AS(tensor.view().copy_from_host(invalid), std::invalid_argument);
    invalid = source;
    invalid[invalid.size() - 1] = std::byte{255};
    CHECK_THROWS_AS(tensor.view().copy_from_host(invalid), std::invalid_argument);
    CHECK_EQ(tensor.from_host_calls, 2);

    // Const reads delegate the same view metadata.
    std::vector<std::byte> sink(derived_nbytes);
    derived.copy_to_host(sink);
    CHECK_EQ(tensor.to_host_calls, 1);
    CHECK_EQ(tensor.to_view_offset, 5);
    CHECK_EQ(tensor.to_view_strides, std::vector<std::size_t>({3}));
    CHECK_EQ(tensor.to_destination_size, derived_nbytes);

    CHECK_THROWS_AS(
            derived.copy_to_host(
                    std::span<std::byte>(sink.data(), derived_nbytes + 1)),
            std::invalid_argument);
    CHECK_EQ(tensor.to_host_calls, 1);
}

TEST_CASE("TensorView transforms leave owner storage untouched") {
    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 4, 5, 16, 16});

    void* handle = tensor.view().native_handle();
    const iom::TensorView& full = tensor.view();

    iom::TensorView derived = full.slice(0, 1, 1)
                                      .select(1, 2)
                                      .reshape_leading(span_of({20}))
                                      .slice(0, 3, 4, 3)
                                      .permute(span_of({0}));
    CHECK_EQ(derived.plane_offset(), 103);
    CHECK(derived.native_handle() == handle);
    CHECK(derived.native_handle() == tensor.storage_handle());

    // The full view object kept its address; no hook touched storage.
    CHECK(&tensor.view() == &full);
    CHECK_EQ(tensor.from_host_calls, 0);
    CHECK_EQ(tensor.to_host_calls, 0);

    // Copies pin the same owner and handle.
    iom::TensorView copy = derived;
    CHECK(copy.native_handle() == handle);
    CHECK(&copy.device() == &device);
}

TEST_CASE("DeviceOps view signatures are exact and view-only") {
    using iom::DeviceOps;
    using iom::TensorView;

    static_assert(std::is_same_v<iom::oid, std::uint64_t>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::wait), void (DeviceOps::*)(iom::oid)>);

    static_assert(std::is_same_v<
        decltype(&DeviceOps::copy),
        iom::oid (DeviceOps::*)(const TensorView&, TensorView&)>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::add),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&, TensorView&)>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::mul),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&, TensorView&)>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::silu),
        iom::oid (DeviceOps::*)(const TensorView&, TensorView&)>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::linear),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&, TensorView&)>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::rmsnorm),
        iom::oid (DeviceOps::*)(const TensorView&, TensorView&, const TensorView&,
                                float, size_t)>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::sdpa),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&, const TensorView&,
                                size_t, size_t, size_t, TensorView&)>);

    // silu deliberately accepts the same window as const input and mutable
    // output; a const view is refused as an output.
    static_assert(std::is_invocable_v<
        decltype(&DeviceOps::silu), DeviceOps*, const TensorView&, TensorView&>);
    static_assert(!std::is_invocable_v<
        decltype(&DeviceOps::silu), DeviceOps*, TensorView&, const TensorView&>);

    // No Tensor operand overload survives the cutover: Tensor does not
    // convert to TensorView.
    static_assert(!std::is_invocable_v<
        decltype(&DeviceOps::copy), DeviceOps*, const iom::Tensor&, iom::Tensor&>);
    static_assert(!std::is_invocable_v<
        decltype(&DeviceOps::add),
        DeviceOps*, const iom::Tensor&, const iom::Tensor&, iom::Tensor&>);
    static_assert(!std::is_invocable_v<
        decltype(&DeviceOps::sdpa), DeviceOps*, const iom::Tensor&, const iom::Tensor&,
        const iom::Tensor&, size_t, size_t, size_t, iom::Tensor&>);

    static_assert(std::is_abstract_v<DeviceOps>);
    static_assert(!std::is_copy_constructible_v<DeviceOps>);
    static_assert(!std::is_move_constructible_v<DeviceOps>);
    static_assert(!std::is_copy_assignable_v<DeviceOps>);
    static_assert(!std::is_move_assignable_v<DeviceOps>);
}

TEST_CASE("DeviceOps view signatures accept stable owner views from callers") {
    FakeDevice device;
    FakeQueue queue;
    FakeTensor a = make_tensor(device, {2, 3, 16, 16});
    const FakeTensor& frozen = a;
    FakeTensor b = make_tensor(device, {2, 3, 16, 16});

    CHECK_NOTHROW((void)queue.copy(a.view(), b.view()));
    CHECK_NOTHROW((void)queue.add(frozen.view(), a.view(), b.view()));
    CHECK_NOTHROW((void)queue.mul(a.view(), frozen.view(), b.view()));
    CHECK_NOTHROW((void)queue.silu(b.view(), b.view()));
    CHECK(queue.silu_aliased);
    CHECK_NOTHROW((void)queue.linear(a.view(), b.view(), b.view()));
    CHECK_NOTHROW((void)queue.rmsnorm(a.view(), b.view(), b.view(), 1e-6F, 1));
    CHECK_NOTHROW((void)queue.sdpa(a.view(), b.view(), b.view(), 2, 1, 16, b.view()));

    // Derived views are equally acceptable operands.
    const iom::TensorView selected = a.view().select(1, 2);
    iom::TensorView selected_b = b.view().select(1, 2);
    CHECK_NOTHROW((void)queue.copy(selected, selected_b));

    REQUIRE(queue.submissions.size() == 8);
    CHECK_EQ(queue.submissions.back().sequence, 8);
}

TEST_CASE("DeviceOps queue ids lease exclusively across threads and are reused after release") {
    std::mutex live_mutex;
    std::vector<std::unique_ptr<FakeQueue>> live;
    std::atomic<int> rejections = 0;
    {
        std::vector<std::thread> builders;
        for (int builder = 0; builder < 8; ++builder) {
            builders.emplace_back([&] {
                for (;;) {
                    std::unique_ptr<FakeQueue> queue;
                    try {
                        queue = std::make_unique<FakeQueue>();
                    } catch (const std::runtime_error&) {
                        ++rejections;
                        return;
                    }
                    const std::lock_guard<std::mutex> lock(live_mutex);
                    live.push_back(std::move(queue));
                }
            });
        }
        for (std::thread& builder : builders) {
            builder.join();
        }
    }
    REQUIRE(live.size() == 255);
    CHECK_EQ(rejections.load(), 8);

    std::set<std::uint8_t> ids;
    for (const std::unique_ptr<FakeQueue>& queue : live) {
        CHECK(ids.insert(token_queue(queue->probe())).second);
    }
    REQUIRE(ids.size() == 255);

    // Releasing one live queue makes exactly its id reusable.
    const std::uint8_t released = token_queue(live.back()->probe());
    live.pop_back();
    {
        FakeQueue successor;
        CHECK_EQ(token_queue(successor.probe()), released);
        CHECK_THROWS_AS(FakeQueue{}, std::runtime_error);
    }

    // Every release is observed; an empty pool starts from a low id again.
    live.clear();
    FakeQueue fresh;
    CHECK_EQ(token_queue(fresh.probe()), 1);
}

TEST_CASE("DeviceOps queue tokens encode queue id above monotonic sequences") {
    FakeDevice device;
    FakeQueue first;
    FakeQueue second;
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid first_token = first.probe();
    const iom::oid second_token = first.copy(a.view(), b.view());
    const iom::oid other_token = second.probe();

    CHECK_EQ(token_sequence(first_token), 1);
    CHECK_EQ(token_sequence(second_token), 2);
    CHECK_EQ(token_sequence(other_token), 1);
    CHECK_NE(token_queue(first_token), 0);
    CHECK_NE(token_queue(first_token), token_queue(other_token));
    CHECK_EQ(first_token, make_token(token_queue(first_token), 1));

    // A validation failure consumes no sequence.
    FakeTensor wide = make_tensor(device, {5, 8});
    CHECK_THROWS_AS(first.copy(a.view(), wide.view()), std::invalid_argument);
    REQUIRE(first.submissions.size() == 2);
    const iom::oid next = first.probe();
    CHECK_EQ(token_sequence(next), 3);

    // An identical-window copy is still a queued operation.
    const iom::oid noop = first.copy(b.view(), b.view());
    CHECK_EQ(token_sequence(noop), 4);
    CHECK_EQ(first.submissions.back().sequence, 4);
    CHECK(std::string_view(first.submissions.back().op) == "copy");
}

TEST_CASE("DeviceOps queue sequences exhaust at the 56-bit boundary") {
    FakeDevice device;
    FakeQueue queue;
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    // The seam only moves forward and never allocates a sequence.
    CHECK_THROWS_AS(queue.seek_next_sequence(0), std::invalid_argument);
    CHECK_EQ(token_sequence(queue.probe()), 1);
    CHECK_THROWS_AS(queue.seek_next_sequence(1), std::invalid_argument);

    queue.seek_next_sequence(kMaxSequence);
    REQUIRE(queue.submissions.size() == 1);
    const iom::oid last = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(last), kMaxSequence);
    REQUIRE(queue.submissions.size() == 2);

    // Submission past the last sequence throws before queuing.
    CHECK_THROWS_AS(queue.copy(a.view(), b.view()), std::overflow_error);
    CHECK_EQ(queue.submissions.size(), 2);

    // The boundary submission is still waitable.
    queue.complete(kMaxSequence);
    CHECK_NOTHROW(queue.wait(last));
}

TEST_CASE("DeviceOps submit commits the sequence before queue_work runs and survives inline complete") {
    FakeDevice device;
    InlineQueue queue;
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid first = queue.copy(a.view(), b.view());
    CHECK_EQ(queue.inline_calls, 1);
    for (int i = 0; i < 3; ++i) {
        CHECK_NOTHROW(queue.wait(first));
    }

    const iom::oid second = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(second), token_sequence(first) + 1);
    CHECK_NOTHROW(queue.wait(second));

    queue.mode = InlineQueue::Mode::complete_with_failure;
    const iom::oid failed = queue.copy(a.view(), b.view());
    for (int i = 0; i < 3; ++i) {
        bool matched = false;
        try {
            queue.wait(failed);
        } catch (const std::runtime_error& error) {
            matched = std::string_view(error.what()) == "inline boom";
        }
        CHECK(matched);
    }

    InlineQueue exhausted;
    FakeTensor exhausted_a = make_tensor(device, {4, 8});
    FakeTensor exhausted_b = make_tensor(device, {4, 8});
    exhausted.seek_next_sequence(kMaxSequence + 1);
    const std::size_t calls_before = exhausted.inline_calls;
    CHECK_THROWS_AS(
            exhausted.copy(exhausted_a.view(), exhausted_b.view()),
            std::overflow_error);
    CHECK_EQ(exhausted.inline_calls, calls_before);
}

TEST_CASE("DeviceOps submit rolls back a synchronous queue_work failure") {
    FakeDevice device;
    InlineQueue queue(InlineQueue::Mode::throw_before_complete);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    bool matched = false;
    try {
        queue.copy(a.view(), b.view());
    } catch (const std::runtime_error& error) {
        matched = std::string_view(error.what()) == "event_create boom";
    }
    CHECK(matched);

    queue.mode = InlineQueue::Mode::complete;
    const iom::oid next = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(next), 1);
    CHECK_NOTHROW(queue.wait(next));
}

TEST_CASE("submit suppresses rollback when queue_work completes inline and then throws") {
    FakeDevice device;
    InlineQueue queue;
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});
    const iom::oid seed = queue.copy(a.view(), b.view());
    CHECK_NOTHROW(queue.wait(seed));
    queue.mode = InlineQueue::Mode::complete_then_throw;

    bool matched = false;
    try {
        queue.copy(a.view(), b.view());
    } catch (const std::runtime_error& error) {
        matched = std::string_view(error.what()) == "post-complete boom";
    }
    CHECK(matched);

    queue.mode = InlineQueue::Mode::complete;
    const iom::oid next = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(next), 3);
    CHECK_NOTHROW(queue.wait(next));
}

TEST_CASE("commit_failure accepts a reserved-but-not-completed sequence after submit reservation") {
    FakeDevice device;
    InlineQueue queue(InlineQueue::Mode::commit_failure_then_complete);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid token = queue.copy(a.view(), b.view());
    for (int i = 0; i < 3; ++i) {
        bool matched = false;
        try {
            queue.wait(token);
        } catch (const std::runtime_error& error) {
            matched = std::string_view(error.what()) == "post-link boom";
        }
        CHECK(matched);
    }
}

TEST_CASE("commit_failure accepts a reserved sequence under concurrent submit") {
    InlineQueue queue;
    std::promise<iom::oid> second_token_promise;
    std::future<iom::oid> second_token_future = second_token_promise.get_future();

    const iom::oid first = queue.submit([&](std::uint64_t sequence) {
        std::thread second([&] {
            try {
                second_token_promise.set_value(queue.submit(
                        [](std::uint64_t) {}));
            } catch (...) {
                second_token_promise.set_exception(std::current_exception());
            }
        });

        const iom::oid second_token = second_token_future.get();
        queue.commit_failure(
                sequence,
                std::make_exception_ptr(
                        std::runtime_error("concurrent boom")));
        queue.complete(sequence);
        queue.complete(token_sequence(second_token));
        second.join();
    });

    for (int i = 0; i < 3; ++i) {
        bool matched = false;
        try {
            queue.wait(first);
        } catch (const std::runtime_error& error) {
            matched = std::string_view(error.what()) == "concurrent boom";
        }
        CHECK(matched);
    }
}

TEST_CASE("commit_failure rejects a never-submitted sequence") {
    InlineQueue queue;
    const auto failure = std::make_exception_ptr(std::runtime_error("x"));
    queue.seek_next_sequence(3);

    for (const std::uint64_t sequence : {std::uint64_t{1}, std::uint64_t{2}}) {
        bool matched = false;
        try {
            queue.commit_failure(sequence, failure);
        } catch (const std::invalid_argument& error) {
            matched = std::string_view(error.what())
                    == "retained failure is not for a reserved submission";
        }
        CHECK(matched);
    }
}

TEST_CASE("commit_failure rejects an already-completed sequence") {
    FakeDevice device;
    InlineQueue queue;
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid token = queue.copy(a.view(), b.view());
    CHECK_NOTHROW(queue.wait(token));

    bool matched = false;
    try {
        queue.commit_failure(
                token_sequence(token),
                std::make_exception_ptr(std::runtime_error("late failure")));
    } catch (const std::invalid_argument& error) {
        matched = std::string_view(error.what())
                == "retained failure is for a sequence that has already been completed";
    }
    CHECK(matched);
}

TEST_CASE("DeviceOps queue waits reject invalid live tokens") {
    FakeDevice device;
    FakeQueue queue;
    FakeQueue foreign;
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid own = queue.probe();
    const iom::oid other = foreign.probe();

    CHECK_THROWS_AS(queue.wait(0), std::invalid_argument);
    CHECK_THROWS_AS(queue.wait(make_token(token_queue(own), 0)), std::invalid_argument);
    CHECK_THROWS_AS(queue.wait(other), std::invalid_argument);
    CHECK_THROWS_AS(queue.wait(make_token(200, 1)), std::invalid_argument);
    CHECK_THROWS_AS(queue.wait(make_token(token_queue(own), 2)), std::invalid_argument);

    // Rejections consumed nothing and completed nothing.
    queue.complete(1);
    CHECK_NOTHROW(queue.wait(own));
    CHECK_THROWS_AS(foreign.wait(own), std::invalid_argument);
}

TEST_CASE("DeviceOps queue waits are idempotent and retain per-sequence results") {
    FakeDevice device;
    FakeQueue queue;
    const iom::oid one = queue.probe();
    const iom::oid two = queue.probe();
    const iom::oid three = queue.probe();
    const iom::oid four = queue.probe();

    queue.complete(1);
    CHECK_NOTHROW(queue.wait(one));
    CHECK_NOTHROW(queue.wait(one));

    // In-order completion with a retained failure at two.
    queue.complete(2, std::make_exception_ptr(std::runtime_error("async boom")));
    queue.complete(3);
    CHECK_NOTHROW(queue.wait(one));
    bool rethrown = false;
    try {
        queue.wait(two);
    } catch (const std::runtime_error& error) {
        rethrown = std::string_view(error.what()) == "async boom";
    }
    CHECK(rethrown);
    CHECK_THROWS_AS(queue.wait(two), std::runtime_error);
    CHECK_NOTHROW(queue.wait(three));
    CHECK_THROWS_AS(queue.wait(two), std::runtime_error);

    queue.complete(4);
    CHECK_NOTHROW(queue.wait(four));
}

TEST_CASE("DeviceOps queue wait wakes a blocked waiter on completion") {
    FakeDevice device;
    FakeQueue queue;
    const iom::oid token = queue.probe();

    std::atomic<bool> returned = false;
    auto waiter = std::async(std::launch::async, [&] {
        queue.wait(token);
        returned.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK_FALSE(returned.load());
    queue.complete(1);
    REQUIRE(waiter.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(returned.load());
}

TEST_CASE("DeviceOps queue destruction neither waits nor cancels and releases the id") {
    FakeDevice device;
    std::uint8_t released = 0;
    {
        FakeQueue queue;
        FakeTensor a = make_tensor(device, {4, 8});
        FakeTensor b = make_tensor(device, {4, 8});
        released = token_queue(queue.copy(a.view(), b.view()));
        // Sequence one is left incomplete: destruction must neither block
        // on it nor cancel it, and the id must return to the pool.
    }
    FakeQueue successor;
    CHECK_EQ(token_queue(successor.probe()), released);
}

TEST_CASE("DeviceOps queue drives the llama models through owner views") {
    FakeDevice device;
    FakeQueue dev;

    constexpr std::size_t kCtxLen = 8;
    constexpr std::size_t kHeadDim = 4;
    constexpr double kTheta = 10000.0;

    FakeTensor sin = make_tensor(device, {kCtxLen, kHeadDim / 2});
    FakeTensor cos = make_tensor(device, {kCtxLen, kHeadDim / 2});
    iom::models::LlamaRoPE rope(dev, kCtxLen, kHeadDim, kTheta, sin, cos);

    // Initialization sent exactly one logical_nbytes host span each and
    // queued nothing.
    REQUIRE(sin.from_host_calls == 1);
    REQUIRE(cos.from_host_calls == 1);
    REQUIRE(sin.from_bytes.size() == sin.view().spec().logical_nbytes());
    REQUIRE(cos.from_bytes.size() == cos.view().spec().logical_nbytes());
    REQUIRE(dev.submissions.empty());

    // Exact float tables in row-major [ctx, head_dim / 2] order.
    std::vector<double> inv_freq(kHeadDim / 2, 0.0);
    for (std::size_t j = 0; j < kHeadDim / 2; ++j) {
        inv_freq[j] = 1.0 / std::pow(10000.0, static_cast<double>((2 * j) / kHeadDim));
    }
    auto decode_le_float = [](const std::byte* bytes) {
        std::uint32_t bits = 0;
        for (std::size_t b = 0; b < 4; ++b) {
            bits |= static_cast<std::uint32_t>(bytes[b]) << (8 * b);
        }
        float value = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    for (std::size_t i = 0; i < kCtxLen; ++i) {
        for (std::size_t j = 0; j < kHeadDim / 2; ++j) {
            CAPTURE(i);
            CAPTURE(j);
            const std::size_t offset = (i * kHeadDim / 2 + j) * 4;
            CHECK_EQ(decode_le_float(sin.from_bytes.data() + offset),
                     static_cast<float>(std::sin(kTheta * static_cast<double>(i) * inv_freq[j])));
            CHECK_EQ(decode_le_float(cos.from_bytes.data() + offset),
                     static_cast<float>(std::cos(kTheta * static_cast<double>(i) * inv_freq[j])));
        }
    }

    FakeTensor x = make_tensor(device, {16, 16});
    FakeTensor y = make_tensor(device, {16, 16});
    FakeTensor t = make_tensor(device, {16, 16});
    FakeTensor r = make_tensor(device, {16, 16});
    FakeTensor wq = make_tensor(device, {16, 16});
    FakeTensor wk = make_tensor(device, {16, 16});
    FakeTensor wv = make_tensor(device, {16, 16});
    FakeTensor wo = make_tensor(device, {16, 16});
    FakeTensor q = make_tensor(device, {16, 16});
    FakeTensor k = make_tensor(device, {16, 16});
    FakeTensor v = make_tensor(device, {16, 16});
    FakeTensor a = make_tensor(device, {16, 16});
    FakeTensor wu = make_tensor(device, {16, 16});
    FakeTensor wd = make_tensor(device, {16, 16});
    FakeTensor wg = make_tensor(device, {16, 16});
    FakeTensor u = make_tensor(device, {16, 16});
    FakeTensor g = make_tensor(device, {16, 16});
    FakeTensor wi = make_tensor(device, {16, 16});
    FakeTensor wm = make_tensor(device, {16, 16});
    FakeTensor wlm = make_tensor(device, {16, 16});
    FakeTensor wn = make_tensor(device, {16, 16});

    iom::models::LlamaAttention attn(dev, 16, 2, 1, wq, wk, wv, wo, q, k, v, a, rope);
    iom::models::LlamaMlp mlp(dev, wu, wd, wg, u, g);
    iom::models::LlamaDecoder decoder(dev, attn, mlp, wi, wm, t, r);
    std::vector<std::reference_wrapper<iom::models::LlamaDecoder>> decoders{decoder};
    iom::models::Llama2Model model(dev, decoders, wlm, wn, t);

    decoder.forward(x, y);
    const std::vector<std::string_view> decoder_ops = {
        "rmsnorm",
        "linear", "linear", "linear", "sdpa", "linear",
        "add",
        "copy",
        "rmsnorm",
        "linear", "linear", "silu", "mul", "linear",
        "add",
    };
    REQUIRE(dev.submissions.size() == decoder_ops.size());
    CHECK(op_names(dev) == decoder_ops);
    CHECK(dev.silu_aliased);
    CHECK_EQ(dev.submissions.back().sequence, decoder_ops.size());

    model.forward(x, y);
    // The model repeats the whole decoder, then closes with output norm
    // and the LM head projection.
    std::vector<std::string_view> expected_tail = decoder_ops;
    expected_tail.push_back("rmsnorm");
    expected_tail.push_back("linear");
    const std::vector<std::string_view> model_ops = op_names(dev);
    REQUIRE(model_ops.size() == decoder_ops.size() + expected_tail.size());
    CHECK(std::vector<std::string_view>(
              model_ops.begin() + static_cast<std::ptrdiff_t>(decoder_ops.size()),
              model_ops.end()) == expected_tail);
    CHECK_EQ(dev.submissions.back().sequence,
             decoder_ops.size() + expected_tail.size());
}
