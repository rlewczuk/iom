// Focused CPU cases for the isolated AVX-512 BF16 binary ADD/SUB route.
//
// The shared cross-backend binary conformance already proves the operation
// contract; these cases cover what it cannot express on a CPU backend: that
// ordinary BF16 ADD and SUB execute vector work, that the exceptional lanes
// stay with the scalar codec under their own stage labels, and that the
// directed BF16 rounding, subnormal, broadcast, exact-alias, and padding
// behavior survives the substitution.
//
// Every expectation is computed by the independent `add_oracle` that the
// shared conformance uses -- a long-double evaluation with a single
// round-to-nearest-even encode -- never by the code under test.
#include <doctest/doctest.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "backend/backend_conformance_add.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"

#include "../../src/cpu/avx512_bf16.hpp"

namespace {

constexpr std::size_t kLeafBits = 16;

class HostAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        return ::operator new(size, std::align_val_t(32));
    }

    void free(void* buffer) override {
        ::operator delete(buffer, std::align_val_t(32));
    }

    void reset() override {}
};

// One BF16 operand pair.
struct DirectedPair {
    std::uint16_t lhs;
    std::uint16_t rhs;
};

// Directed operand pairs. Each pair's exact result is representable in the
// binary32 carrier, so the single-rounding oracle and any correct
// decode/compute/one-RNE path must agree bit for bit.
constexpr DirectedPair kDirectedPairs[] = {
        // Operation-specific zeros: ADD keeps -0 only for two negative zeros,
        // SUB keeps the difference's own sign.
        {0x0000, 0x0000}, {0x8000, 0x8000}, {0x0000, 0x8000},
        {0x8000, 0x0000}, {0x0000, 0x3F80}, {0x8000, 0xBF80},
        // Cancellation to +0, and the ADD/SUB-separating infinity classes:
        // opposite signs are an invalid ADD, like signs an invalid SUB.
        {0x3F80, 0xBF80}, {0x7F80, 0x7F80}, {0xFF80, 0xFF80},
        {0x7F80, 0xFF80}, {0xFF80, 0x7F80}, {0x7F80, 0x3F80},
        {0xFF80, 0x4000},
        // NaN classes: a positive payload, a signed quiet NaN, and a NaN
        // beside an infinity.
        {0x7FC0, 0x3F80}, {0x3F80, 0xFFC0}, {0x7F81, 0x7F80},
        // Exact subnormal arithmetic, the subnormal/normal boundary carry, and
        // a subnormal difference formed from two nearly equal normals. A
        // flushed denormal mode changes every one of these results.
        {0x0001, 0x0001}, {0x0001, 0x8001}, {0x007F, 0x0001},
        {0x0080, 0x807F}, {0x0001, 0x0002}, {0x0080, 0x0001},
        // Rounding around half an ulp of the destination: below it, exactly on
        // it with both mantissa parities, and above it.
        {0x3F80, 0x3A80}, {0x3F80, 0x3B00}, {0x3F80, 0x3B80},
        {0x3F81, 0x3B00}, {0x3F81, 0x3B80},
        // Overflow: a finite overflow and a rounding that only the
        // destination's own overflow check turns into infinity.
        {0x7F7F, 0x7F7F}, {0xFF7F, 0xFF7F}, {0x7F7F, 0x7B00},
        {0x7F7F, 0x0080},
};

constexpr std::size_t kDirectedPairCount =
        sizeof(kDirectedPairs) / sizeof(kDirectedPairs[0]);

// Values for the broadcast and alias cases. Every pairwise sum of these is
// exactly representable in binary32, so the same oracle comparison applies.
constexpr std::uint16_t kSpreadValues[] = {
        0x3F80, 0x4000, 0x7F80, 0xFF80, 0x8000, 0x0000, 0x0001,
        0x0080, 0x7FC0, 0xBF80, 0x3F81, 0x007F,
};

constexpr std::size_t kSpreadCount =
        sizeof(kSpreadValues) / sizeof(kSpreadValues[0]);

// True when a BF16 pattern is a NaN, read from the format's own exponent and
// mantissa fields rather than through a host conversion.
[[nodiscard]] bool bf16_is_nan(std::uint16_t bits) noexcept {
    return (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0u;
}

// Compares one observed BF16 leaf with the oracle's answer. Every non-NaN
// expectation is required bit for bit: that is what pins gradual underflow,
// the signed-zero rules, a single encode, and the overflow class. The oracle's
// NaN sign follows the operand payloads while the codec canonicalizes every
// exceptional lane, so only the NaN class is shared between them.
void check_leaf(std::size_t index, std::uint16_t actual, std::uint16_t want) {
    CAPTURE(index);
    CAPTURE(actual);
    CAPTURE(want);
    if (bf16_is_nan(want)) {
        CHECK(bf16_is_nan(actual));
        return;
    }
    CHECK_EQ(actual, want);
}

// Answers the first tiled slot that no logical element occupies and whose two
// bytes are no longer the zero the CPU tensor storage was filled with, or
// SIZE_MAX when every padding slot is untouched. A span that stored its full
// vector width would write exactly these slots.
[[nodiscard]] std::size_t first_touched_padding_slot(
        const iom::TensorSpec& spec, const void* storage) {
    const auto dimensions = spec.shape.dimensions();
    const std::size_t rank = dimensions.size();
    const std::size_t rows = dimensions[rank - 2];
    const std::size_t columns = dimensions[rank - 1];
    const iom::TensorShape padded = spec.standard_padded_shape();
    const auto padded_dimensions = padded.dimensions();
    const std::size_t planes = spec.shape.element_count() / (rows * columns);
    const auto* bytes = static_cast<const unsigned char*>(storage);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < padded_dimensions[rank - 2]; ++row) {
            for (std::size_t column = 0; column < padded_dimensions[rank - 1];
                 ++column) {
                if (row < rows && column < columns) {
                    continue;
                }
                const std::size_t slot = iom::detail::standard_plane_slot(
                        spec, plane, row, column);
                const unsigned char* leaf = bytes + slot * (kLeafBits / 8);
                if (leaf[0] != 0 || leaf[1] != 0) {
                    return slot;
                }
            }
        }
    }
    return std::numeric_limits<std::size_t>::max();
}

void check_tiled_padding(const iom::TensorSpec& spec, const void* storage) {
    CHECK_EQ(first_touched_padding_slot(spec, storage),
             std::numeric_limits<std::size_t>::max());
}

// Encodes one operand's logical leaves by cycling through `pairs`.
std::vector<std::byte> directed_bytes(
        const iom::TensorSpec& spec, const DirectedPair* pairs,
        std::size_t pair_count, bool lhs) {
    std::vector<std::byte> bytes(spec.logical_nbytes());
    for (std::size_t index = 0; index < spec.shape.element_count(); ++index) {
        const DirectedPair& pair = pairs[index % pair_count];
        iom_conformance::write_bits(
                reinterpret_cast<unsigned char*>(bytes.data()),
                index * kLeafBits, kLeafBits, lhs ? pair.lhs : pair.rhs);
    }
    return bytes;
}

// Encodes one operand's logical leaves by cycling through `values`.
std::vector<std::byte> spread_bytes(const iom::TensorSpec& spec) {
    std::vector<std::byte> bytes(spec.logical_nbytes());
    for (std::size_t index = 0; index < spec.shape.element_count(); ++index) {
        iom_conformance::write_bits(
                reinterpret_cast<unsigned char*>(bytes.data()),
                index * kLeafBits, kLeafBits,
                kSpreadValues[index % kSpreadCount]);
    }
    return bytes;
}

// Submits one binary request through `queue` and returns its observed logical
// leaves. The output starts at a poison pattern so a leaf the worker never
// wrote stays observable. The queue is supplied so a caller can select the
// state its worker starts with, such as its floating-point control word.
std::vector<std::byte> run_binary_on(
        iom::Device& device, iom::DeviceOps& queue,
        iom_conformance::BinaryOperation operation,
        const iom::TensorSpec& lhs_spec, const iom::TensorSpec& rhs_spec,
        const iom::TensorSpec& out_spec,
        const std::vector<std::byte>& lhs_bytes,
        const std::vector<std::byte>& rhs_bytes) {
    auto lhs = device.create_tensor(lhs_spec);
    auto rhs = device.create_tensor(rhs_spec);
    auto out = device.create_tensor(out_spec);
    iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
    iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
    iom_conformance::copy_from_host(
            out->view(),
            std::vector<std::byte>(
                    out_spec.logical_nbytes(), std::byte{0xAA}));
    CHECK_EQ(iom_conformance::query_binary_workspace_requirements(
                     queue, operation, lhs->view(), rhs->view(), out->view())
                     .bytes,
             std::size_t{0});
    const iom::oid token = iom_conformance::submit_binary_operation(
            queue, operation, lhs->view(), rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue.wait(token));
    auto observed = iom_conformance::read_logical(out->view());
    check_tiled_padding(out_spec, out->view().native_handle());
    return observed;
}

std::vector<std::byte> run_binary(
        iom::Device& device, iom_conformance::BinaryOperation operation,
        const iom::TensorSpec& lhs_spec, const iom::TensorSpec& rhs_spec,
        const iom::TensorSpec& out_spec,
        const std::vector<std::byte>& lhs_bytes,
        const std::vector<std::byte>& rhs_bytes) {
    auto queue = device.create_ops();
    return run_binary_on(
            device, *queue, operation, lhs_spec, rhs_spec, out_spec, lhs_bytes,
            rhs_bytes);
}

// The source leaf index one output coordinate maps to when `dimensions` is the
// rank-3 shape of an operand whose axes are equal to the result's or one.
[[nodiscard]] std::size_t source_index(
        std::span<const std::size_t> dimensions, std::size_t plane,
        std::size_t row, std::size_t column) noexcept {
    const std::size_t p = dimensions[0] == 1 ? 0 : plane;
    const std::size_t r = dimensions[1] == 1 ? 0 : row;
    const std::size_t c = dimensions[2] == 1 ? 0 : column;
    return (p * dimensions[1] + r) * dimensions[2] + c;
}

[[nodiscard]] std::uint16_t leaf_at(
        const std::vector<std::byte>& bytes, std::size_t index) {
    return static_cast<std::uint16_t>(
            iom_conformance::add_read_bits(bytes, index, kLeafBits));
}

[[nodiscard]] std::uint16_t oracle_leaf(
        std::uint16_t lhs, std::uint16_t rhs,
        iom_conformance::BinaryOperation operation) {
    return static_cast<std::uint16_t>(iom_conformance::add_oracle::binary(
            iom::DataType::BF16, lhs, rhs,
            static_cast<iom_conformance::add_oracle::operation>(operation)));
}

[[nodiscard]] iom::TensorSpec bf16_spec(
        std::size_t planes, std::size_t rows, std::size_t columns) {
    return iom::TensorSpec{
            iom::TensorShape{{planes, rows, columns}}, iom::DataType::BF16};
}

}  // namespace

TEST_CASE("CPU AVX-512 BF16 binary ADD/SUB directed values and tiled mapping") {
    HostAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = bf16_spec(2, 17, 33);
    const auto lhs_bytes =
            directed_bytes(spec, kDirectedPairs, kDirectedPairCount, true);
    const auto rhs_bytes =
            directed_bytes(spec, kDirectedPairs, kDirectedPairCount, false);

    for (const auto operation : {iom_conformance::BinaryOperation::add,
                                 iom_conformance::BinaryOperation::sub}) {
        CAPTURE(static_cast<int>(operation));
        const auto observed = run_binary(
                *device, operation, spec, spec, spec, lhs_bytes, rhs_bytes);
        for (std::size_t index = 0; index < spec.shape.element_count();
             ++index) {
            check_leaf(
                    index, leaf_at(observed, index),
                    oracle_leaf(
                            leaf_at(lhs_bytes, index),
                            leaf_at(rhs_bytes, index), operation));
        }
    }
}

TEST_CASE("CPU AVX-512 BF16 binary ADD/SUB broadcast and mapped planes") {
    HostAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec out_spec = bf16_spec(2, 17, 33);
    // Both orientations: in one the leading operand broadcasts its planes and
    // rows while the trailing one broadcasts its single logical column, and in
    // the other the roles are swapped.
    for (const bool swapped : {false, true}) {
        const iom::TensorSpec lhs_spec = swapped ? bf16_spec(1, 17, 1)
                                                 : bf16_spec(2, 1, 33);
        const iom::TensorSpec rhs_spec = swapped ? bf16_spec(2, 1, 33)
                                                 : bf16_spec(1, 17, 1);
        const auto lhs_bytes = spread_bytes(lhs_spec);
        const auto rhs_bytes = spread_bytes(rhs_spec);
        const auto lhs_dimensions = lhs_spec.shape.dimensions();
        const auto rhs_dimensions = rhs_spec.shape.dimensions();
        CAPTURE(swapped);
        for (const auto operation : {iom_conformance::BinaryOperation::add,
                                     iom_conformance::BinaryOperation::sub}) {
            CAPTURE(static_cast<int>(operation));
            const auto observed = run_binary(
                    *device, operation, lhs_spec, rhs_spec, out_spec,
                    lhs_bytes, rhs_bytes);
            for (std::size_t plane = 0; plane < 2; ++plane) {
                for (std::size_t row = 0; row < 17; ++row) {
                    for (std::size_t column = 0; column < 33; ++column) {
                        const std::size_t index =
                                (plane * 17 + row) * 33 + column;
                        const std::uint16_t want = oracle_leaf(
                                leaf_at(
                                        lhs_bytes,
                                        source_index(
                                                lhs_dimensions, plane, row,
                                                column)),
                                leaf_at(
                                        rhs_bytes,
                                        source_index(
                                                rhs_dimensions, plane, row,
                                                column)),
                                operation);
                        CAPTURE(index);
                        CAPTURE(want);
                        const std::uint16_t actual = leaf_at(observed, index);
                        CAPTURE(actual);
                        // The mapping is the point of this case, so finite
                        // leaves keep the contract's one-ULP allowance; the
                        // directed case above is the bit-exact one.
                        if (bf16_is_nan(want)) {
                            CHECK(bf16_is_nan(actual));
                        } else {
                            CHECK(iom_conformance::add_non_f32_float_classes(
                                    iom::DataType::BF16, actual, want));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("CPU AVX-512 BF16 binary ADD/SUB exact in-place alias") {
    HostAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = bf16_spec(2, 17, 33);
    const auto lhs_bytes =
            directed_bytes(spec, kDirectedPairs, kDirectedPairCount, true);
    const auto rhs_bytes =
            directed_bytes(spec, kDirectedPairs, kDirectedPairCount, false);
    auto queue = device->create_ops();

    // All three operands are the same view, which the contract allows as an
    // exact unbroadcasted alias. A worker that stored any part of the output
    // before loading both inputs would report its own result as an operand.
    {
        auto tensor = device->create_tensor(spec);
        iom_conformance::copy_from_host(tensor->view(), lhs_bytes);
        const iom::oid token = iom_conformance::submit_binary_operation(
                *queue, iom_conformance::BinaryOperation::add,
                tensor->view(), tensor->view(), tensor->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        const auto observed = iom_conformance::read_logical(tensor->view());
        for (std::size_t index = 0; index < spec.shape.element_count();
             ++index) {
            check_leaf(
                    index, leaf_at(observed, index),
                    oracle_leaf(
                            leaf_at(lhs_bytes, index),
                            leaf_at(lhs_bytes, index),
                            iom_conformance::BinaryOperation::add));
        }
        check_tiled_padding(spec, tensor->view().native_handle());
    }

    // Subtracting the output from itself is exactly +0 in every non-NaN leaf,
    // which pins the sign of a zero difference under an alias.
    {
        auto tensor = device->create_tensor(spec);
        iom_conformance::copy_from_host(tensor->view(), lhs_bytes);
        const iom::oid token = iom_conformance::submit_binary_operation(
                *queue, iom_conformance::BinaryOperation::sub,
                tensor->view(), tensor->view(), tensor->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        const auto observed = iom_conformance::read_logical(tensor->view());
        for (std::size_t index = 0; index < spec.shape.element_count();
             ++index) {
            CAPTURE(index);
            const std::uint16_t leaf = leaf_at(observed, index);
            const std::uint16_t operand = leaf_at(lhs_bytes, index);
            const std::uint16_t expected = oracle_leaf(
                    operand, operand, iom_conformance::BinaryOperation::sub);
            // Same-sign infinities are the invalid SUB pair, so those leaves
            // are the only ones whose difference is not a zero.
            if (bf16_is_nan(expected)) {
                CHECK(bf16_is_nan(leaf));
                continue;
            }
            CHECK_EQ(expected, std::uint16_t{0x0000});
            CHECK_EQ(leaf, std::uint16_t{0x0000});
        }
        check_tiled_padding(spec, tensor->view().native_handle());
    }

    // The output is the leading operand and a distinct tensor is the trailing
    // one, the other exact alias the contract admits.
    {
        auto lhs = device->create_tensor(spec);
        auto rhs = device->create_tensor(spec);
        iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
        iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
        const iom::oid token = iom_conformance::submit_binary_operation(
                *queue, iom_conformance::BinaryOperation::sub,
                lhs->view(), rhs->view(), lhs->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        const auto observed = iom_conformance::read_logical(lhs->view());
        for (std::size_t index = 0; index < spec.shape.element_count();
             ++index) {
            check_leaf(
                    index, leaf_at(observed, index),
                    oracle_leaf(
                            leaf_at(lhs_bytes, index),
                            leaf_at(rhs_bytes, index),
                            iom_conformance::BinaryOperation::sub));
        }
        check_tiled_padding(spec, lhs->view().native_handle());
    }
}

#if defined(IOM_AVX512_BF16_TESTING)
namespace {

struct StagePathCounts {
    std::uint64_t add_native = 0;
    std::uint64_t add_fallback = 0;
    std::uint64_t sub_native = 0;
    std::uint64_t sub_fallback = 0;
};

StagePathCounts stage_path_counts() {
    const auto observation = [](iom::cpu_detail::Avx512Bf16Stage stage,
                                iom::cpu_detail::Avx512Bf16Path path) {
        return iom::cpu_detail::avx512_bf16_test_observation(stage, path);
    };
    return {
            observation(
                    iom::cpu_detail::Avx512Bf16Stage::BinaryAdd,
                    iom::cpu_detail::Avx512Bf16Path::Native),
            observation(
                    iom::cpu_detail::Avx512Bf16Stage::BinaryAdd,
                    iom::cpu_detail::Avx512Bf16Path::Fallback),
            observation(
                    iom::cpu_detail::Avx512Bf16Stage::BinarySub,
                    iom::cpu_detail::Avx512Bf16Path::Native),
            observation(
                    iom::cpu_detail::Avx512Bf16Stage::BinarySub,
                    iom::cpu_detail::Avx512Bf16Path::Fallback)};
}

}  // namespace

// The stage/path seam exists only in test builds, and the isolated worker
// exists only in a configuration that compiles it, so this case reads the
// eligibility answer instead of asserting a fixed one: on an eligible host
// ordinary operands must advance the native counter of their own operation and
// no other, directed operands must additionally record the codec lanes, and an
// ineligible host or portable configuration must leave all four counters
// untouched.
TEST_CASE("CPU AVX-512 BF16 binary ADD/SUB stage and path observations") {
    HostAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = bf16_spec(2, 17, 33);
    const DirectedPair ordinary[] = {{0x3F80, 0x4000}};
    const auto ordinary_lhs = directed_bytes(spec, ordinary, 1, true);
    const auto ordinary_rhs = directed_bytes(spec, ordinary, 1, false);
    const auto directed_lhs =
            directed_bytes(spec, kDirectedPairs, kDirectedPairCount, true);
    const auto directed_rhs =
            directed_bytes(spec, kDirectedPairs, kDirectedPairCount, false);
    const bool eligible = iom::cpu_detail::avx512_bf16_available();

    const auto run = [&](iom_conformance::BinaryOperation operation,
                         const std::vector<std::byte>& lhs_bytes,
                         const std::vector<std::byte>& rhs_bytes) {
        iom::cpu_detail::avx512_bf16_test_reset_observations();
        const auto observed = run_binary(
                *device, operation, spec, spec, spec, lhs_bytes, rhs_bytes);
        for (std::size_t index = 0; index < spec.shape.element_count();
             ++index) {
            check_leaf(
                    index, leaf_at(observed, index),
                    oracle_leaf(
                            leaf_at(lhs_bytes, index),
                            leaf_at(rhs_bytes, index), operation));
        }
        return stage_path_counts();
    };

    const StagePathCounts add_ordinary = run(
            iom_conformance::BinaryOperation::add, ordinary_lhs,
            ordinary_rhs);
    const StagePathCounts sub_ordinary = run(
            iom_conformance::BinaryOperation::sub, ordinary_lhs,
            ordinary_rhs);
    const StagePathCounts add_directed = run(
            iom_conformance::BinaryOperation::add, directed_lhs,
            directed_rhs);
    const StagePathCounts sub_directed = run(
            iom_conformance::BinaryOperation::sub, directed_lhs,
            directed_rhs);

    CAPTURE(eligible);
    if (!eligible) {
        // No target code exists for this host or configuration: every request
        // ran the scalar codec, so no stage may claim work.
        for (const StagePathCounts& counts :
             {add_ordinary, sub_ordinary, add_directed, sub_directed}) {
            CHECK_EQ(counts.add_native, std::uint64_t{0});
            CHECK_EQ(counts.add_fallback, std::uint64_t{0});
            CHECK_EQ(counts.sub_native, std::uint64_t{0});
            CHECK_EQ(counts.sub_fallback, std::uint64_t{0});
        }
        return;
    }

    // Ordinary finite BF16 operands: vector work for this operation only, no
    // lane reaching the codec.
    CHECK_GT(add_ordinary.add_native, std::uint64_t{0});
    CHECK_EQ(add_ordinary.add_fallback, std::uint64_t{0});
    CHECK_EQ(add_ordinary.sub_native, std::uint64_t{0});
    CHECK_EQ(add_ordinary.sub_fallback, std::uint64_t{0});
    CHECK_GT(sub_ordinary.sub_native, std::uint64_t{0});
    CHECK_EQ(sub_ordinary.sub_fallback, std::uint64_t{0});
    CHECK_EQ(sub_ordinary.add_native, std::uint64_t{0});
    CHECK_EQ(sub_ordinary.add_fallback, std::uint64_t{0});

    // Directed operands: the ordinary lanes keep executing vector arithmetic
    // and the exceptional lanes are attributed to the codec path, both under
    // the operation's own stage.
    CHECK_GT(add_directed.add_native, std::uint64_t{0});
    CHECK_GT(add_directed.add_fallback, std::uint64_t{0});
    CHECK_EQ(add_directed.sub_native, std::uint64_t{0});
    CHECK_EQ(add_directed.sub_fallback, std::uint64_t{0});
    CHECK_GT(sub_directed.sub_native, std::uint64_t{0});
    CHECK_GT(sub_directed.sub_fallback, std::uint64_t{0});
    CHECK_EQ(sub_directed.add_native, std::uint64_t{0});
    CHECK_EQ(sub_directed.add_fallback, std::uint64_t{0});
}

#if defined(__x86_64__)

// Rounding control (round-toward-negative-infinity), flush-to-zero, and
// denormals-are-zero: every MXCSR field the contract's arithmetic depends on.
constexpr std::uint32_t kAlteredControlBits = 0xC040u;

// The span kernel computes in binary32, where a BF16 subnormal is a binary32
// subnormal, so the worker's floating-point control word changes what the
// vector arithmetic does: denormals-are-zero drops subnormal operands,
// flush-to-zero drops subnormal results, and round-down changes the sign of an
// exact cancellation that the codec normalizes to positive zero. The queue
// worker inherits this thread's control word when the queue creates it, so
// selecting those fields here makes every request below run in that mode, while
// the contract still requires the scalar reference's values and the ordinary
// lanes must still be vectorized.
TEST_CASE("CPU AVX-512 BF16 binary ADD/SUB under an altered worker MXCSR") {
    HostAllocator allocator;
    const bool eligible = iom::cpu_detail::avx512_bf16_available();
    const iom::TensorSpec spec = bf16_spec(2, 17, 33);
    const auto lhs_bytes =
            directed_bytes(spec, kDirectedPairs, kDirectedPairCount, true);
    const auto rhs_bytes =
            directed_bytes(spec, kDirectedPairs, kDirectedPairCount, false);
    // Expectations are computed with this thread's control word untouched and
    // no queue existing yet.
    const auto expected_for = [&](iom_conformance::BinaryOperation operation) {
        std::vector<std::uint16_t> values(spec.shape.element_count());
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = oracle_leaf(
                    leaf_at(lhs_bytes, index), leaf_at(rhs_bytes, index),
                    operation);
        }
        return values;
    };
    const auto add_expected =
            expected_for(iom_conformance::BinaryOperation::add);
    const auto sub_expected =
            expected_for(iom_conformance::BinaryOperation::sub);

    const std::uint32_t saved_control =
            static_cast<std::uint32_t>(_mm_getcsr());
    struct RestoreControl {
        std::uint32_t control;
        ~RestoreControl() { _mm_setcsr(control); }
    } restore{saved_control};
    _mm_setcsr(saved_control | kAlteredControlBits);
    // The hazard is live in this mode: a plain binary32 sum of the two widened
    // BF16 minimum subnormals flushes to zero here, which is exactly what the
    // accelerated request must not return.
    volatile float widened_lhs =
            std::bit_cast<float>(std::uint32_t{0x00010000u});
    volatile float widened_rhs =
            std::bit_cast<float>(std::uint32_t{0x00010000u});
    const std::uint32_t flushed_sum =
            std::bit_cast<std::uint32_t>(widened_lhs + widened_rhs);

    const auto run_altered = [&](iom_conformance::BinaryOperation operation) {
        struct Result {
            std::vector<std::byte> observed;
            StagePathCounts counts;
        };
        auto device = iom::make_cpu_device(allocator);
        auto queue = device->create_ops();
        iom::cpu_detail::avx512_bf16_test_reset_observations();
        auto observed = run_binary_on(
                *device, *queue, operation, spec, spec, spec, lhs_bytes,
                rhs_bytes);
        return Result{std::move(observed), stage_path_counts()};
    };
    const auto add_run = run_altered(iom_conformance::BinaryOperation::add);
    const auto sub_run = run_altered(iom_conformance::BinaryOperation::sub);

    CHECK_EQ(flushed_sum, std::uint32_t{0});
    for (std::size_t index = 0; index < spec.shape.element_count(); ++index) {
        check_leaf(index, leaf_at(add_run.observed, index), add_expected[index]);
        check_leaf(index, leaf_at(sub_run.observed, index), sub_expected[index]);
    }
    CAPTURE(eligible);
    if (!eligible) {
        // A configuration or host without the isolated worker runs the scalar
        // reference for every leaf, so no stage may claim work.
        CHECK_EQ(add_run.counts.add_native, std::uint64_t{0});
        CHECK_EQ(add_run.counts.add_fallback, std::uint64_t{0});
        CHECK_EQ(sub_run.counts.sub_native, std::uint64_t{0});
        CHECK_EQ(sub_run.counts.sub_fallback, std::uint64_t{0});
        return;
    }
    // The control word does not move work off the vector path: every ordinary
    // lane still executes vector arithmetic and only the directed exceptional
    // lanes reach the codec, under each operation's own stage.
    CHECK_GT(add_run.counts.add_native, std::uint64_t{0});
    CHECK_GT(add_run.counts.add_fallback, std::uint64_t{0});
    CHECK_EQ(add_run.counts.sub_native, std::uint64_t{0});
    CHECK_EQ(add_run.counts.sub_fallback, std::uint64_t{0});
    CHECK_GT(sub_run.counts.sub_native, std::uint64_t{0});
    CHECK_GT(sub_run.counts.sub_fallback, std::uint64_t{0});
    CHECK_EQ(sub_run.counts.add_native, std::uint64_t{0});
    CHECK_EQ(sub_run.counts.add_fallback, std::uint64_t{0});
}

#endif  // __x86_64__
#endif  // IOM_AVX512_BF16_TESTING