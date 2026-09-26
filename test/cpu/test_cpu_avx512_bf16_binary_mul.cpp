// Focused evidence for the CPU AVX-512 BF16 MUL worker
// (src/cpu/avx512_bf16_binary_mul.cpp and its route in src/cpu/queue.cpp).
//
// The shared binary conformance already compares CPU MUL against an
// independent oracle for every applicable leaf, in both the portable and the
// AVX-512 configuration, so these cases cover what that matrix cannot decide:
//
//   * a directed BF16 MUL corpus whose leaf pairs are chosen for the classes an
//     accelerated tile row must not change -- rounding half-way values in both
//     directions, products that underflow to a subnormal or to zero, a product
//     at the finite ceiling, signed zeros, infinities, NaN, and `0 * infinity`
//     -- compared bit for bit against the scalar codec that is the operation's
//     semantic authority, with the independent oracle checked next to it;
//   * the same corpus behind a right-aligned broadcast operand, a one-element
//     broadcast operand, and the exact in-place alias the binary contract
//     allows;
//   * a leading-plane view over a larger owner, so the observed physical
//     storage proves that only the view's own leaves changed and that tile
//     padding and unselected planes kept the caller's bytes;
//   * an owner whose allocator hands out 32-byte-aligned but 64-byte-misaligned
//     bases, and whose observed base is checked, so no vector tile row may
//     assume a wider alignment than the device promises;
//   * the settled `BinaryMul` observation stages, which separate committed SIMD
//     work from the scalar authority instead of trusting a dispatch decision --
//     on a host or in a build without the isolated sources the same corpus must
//     still match, with no recorded work of either kind.
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "backend/backend_conformance_add.hpp"
#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_oracle.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"

#include "../../src/cpu/avx512_bf16.hpp"
#include "../../src/shared/scalar_add.hpp"

namespace {

using Bytes = std::vector<std::byte>;
using Codes = std::vector<std::uint16_t>;

// Every non-logical byte of a seeded owner: a value no MUL result produces, so
// a tile row that wrote outside its own logical leaves is observable.
constexpr std::byte kPaddingPoison{0xD7};

// One directed BF16 MUL leaf pair.
struct MulPair {
    std::uint16_t lhs;
    std::uint16_t rhs;
};

// Pairs whose exact product is a normal FP32 value below 2^127, which is the
// class an accelerated tile row commits. `0x3F81 * 1` and `0x3F83 * 1` are exact
// half-way values between BF16 neighbours (1.0 and 1.0 + 2^-7, and 1.0 + 2^-7
// and 1.0 + 2^-6), so each has to round to the even neighbour; `1 * 0x3F01` is
// the first of those with the operands exchanged. The last entry reaches the top
// of the committed class: 2^100 * 2^26 = 2^126.
constexpr MulPair kCommittedPairs[] = {
        {0x3F80, 0x3F80},   // 1 * 1
        {0x3F80, 0x4000},   // 1 * 2
        {0xBF80, 0x4000},   // -1 * 2
        {0x3F80, 0x3F00},   // 1 * 0.5
        {0x3F00, 0x4000},   // 0.5 * 2
        {0x3F81, 0x3F80},   // half-way: ties to even, 1.0
        {0x3F83, 0x3F80},   // half-way: ties to even, 1.0 + 2^-6
        {0x3F82, 0x3F80},   // exact 1.0 + 2^-7
        {0x3F80, 0x3F01},   // half-way with the operands exchanged
        {0xBF81, 0x4000},   // -2.0 - 2^-6, exact
        {0x0080, 0x3F80},   // smallest normal BF16
        {0x0081, 0x3F80},   // (1 + 2^-7) * 2^-126
        {0x3C80, 0x4080},   // 2^-6 * 4
        {0x3F80, 0x3E00},   // 1 * 0.125
        {0x7E00, 0x3F80},   // 2^125
        {0x7180, 0x4C80},   // 2^100 * 2^26 = 2^126
};

// Pairs whose product the accelerated tile row must not claim: a zero or
// subnormal FP32 product, an exceptional class, or a magnitude whose BF16
// result could leave the finite range. The scalar authority owns every one of
// them, so this group is the operation's fallback evidence. `0x0001 * 0x3F00`
// and `0x0003 * 0x3F00` are the subnormal-range half-way values 0.5 and 1.5
// times the smallest BF16 subnormal, which round to 0 and to 2^-132.
constexpr MulPair kAuthorityPairs[] = {
        {0x0000, 0x3F80},   // +0 * 1 = +0
        {0x8000, 0x3F80},   // -0 * 1 = -0
        {0x0000, 0xBF80},   // +0 * -1 = -0
        {0x8000, 0xBF80},   // -0 * -1 = +0
        {0x0000, 0x7F80},   // 0 * infinity = NaN
        {0x7F80, 0x0000},   // infinity * 0 = NaN
        {0x7F80, 0x3F80},   // +infinity
        {0xFF80, 0x3F80},   // -infinity
        {0x7F80, 0xFF80},   // -infinity
        {0x7FC0, 0x3F80},   // NaN
        {0x3F80, 0x7F81},   // NaN with a payload
        {0x0001, 0x3F80},   // 2^-133: smallest subnormal result
        {0x0001, 0x3F00},   // half-way to zero below 2^-133
        {0x0003, 0x3F00},   // half-way between 2^-133 and 2^-132
        {0x0040, 0x3E80},   // 2^-127 * 0.25 = 2^-129, a subnormal result
        {0x0040, 0x0002},   // 2^-127 * 2^-132, underflows to zero
        {0x7F7F, 0x3F80},   // the finite ceiling: exact, but not a committed lane
        {0x7F7F, 0x4000},   // overflows to infinity
};

// No column carries the exceptional pair, for the corpus that alternates whole
// ordinary rows with whole authority rows.
constexpr std::size_t kNoExceptionalColumn = static_cast<std::size_t>(-1);

// One allocator whose blocks are 32-byte aligned -- the CPU device's own
// contract -- and deliberately 64-byte misaligned. A vector tile row that
// assumed a wider base alignment than it was promised fails here instead of on
// a host whose allocator happens to align it.
class MisalignedBaseAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        void* block = ::operator new(size + kOffset, std::align_val_t{64});
        auto* address = static_cast<std::byte*>(block) + kOffset;
        live_.emplace(address, block);
        return address;
    }

    void free(void* buffer) override {
        const auto found = live_.find(buffer);
        REQUIRE_MESSAGE(found != live_.end(), "unknown block released");
        ::operator delete(found->second, std::align_val_t{64});
        live_.erase(found);
    }

    void reset() override {}

private:
    static constexpr std::size_t kOffset = 32;
    std::unordered_map<void*, void*> live_;
};

// One directed request: the three specifications, the operand leaves of the two
// inputs, the per-result-element operand sources that apply right-aligned
// broadcasting, and whether the result is the exact in-place alias of the left
// operand.
struct DirectedRequest {
    // TensorSpec carries no default constructor, and every case assigns all
    // three specifications before submitting, so this rank-two BF16 placeholder
    // only makes the request value-initializable.
    static iom::TensorSpec placeholder_spec() {
        return {iom::TensorShape{{1, 1}}, iom::DataType::BF16};
    }
    iom::TensorSpec lhs_spec = placeholder_spec();
    iom::TensorSpec rhs_spec = placeholder_spec();
    iom::TensorSpec out_spec = placeholder_spec();
    Codes lhs_codes;
    Codes rhs_codes;
    std::vector<std::size_t> lhs_source;
    std::vector<std::size_t> rhs_source;
    bool out_is_lhs = false;
    std::string_view label;
};

// The scalar codec's own leaf result. This is the CPU queue's authority for
// BF16 MUL, so an accelerated tile row and the authority must agree bit for bit.
[[nodiscard]] std::uint16_t authority_mul(
        std::uint16_t lhs, std::uint16_t rhs) {
    return static_cast<std::uint16_t>(
            iom::detail::scalar_mul(iom::DataType::BF16, lhs, rhs));
}

// The independent oracle's leaf result, matching the shared conformance's own
// comparison source.
[[nodiscard]] std::uint16_t independent_mul(
        std::uint16_t lhs, std::uint16_t rhs) {
    return static_cast<std::uint16_t>(iom_conformance::add_oracle::binary(
            iom::DataType::BF16, lhs, rhs,
            iom_conformance::add_oracle::operation::mul));
}

// The directed operand codes of one specification: rows alternate between a
// whole row of committed-class pairs and a whole row of authority-class pairs,
// so one accepted request contains both kinds of work. When
// `exceptional_column` names a column, every row carries the `infinity * 1`
// pair there, which moves that row's first tile-aligned chunk to the authority
// while its remaining chunk stays committed.
[[nodiscard]] Codes directed_codes(
        const iom::TensorSpec& spec, bool left, std::size_t exceptional_column) {
    const auto dimensions = spec.shape.dimensions();
    const std::size_t rows = dimensions[dimensions.size() - 2];
    const std::size_t columns = dimensions.back();
    const std::size_t count = spec.shape.element_count();
    Codes codes(count);
    for (std::size_t linear = 0; linear < count; ++linear) {
        const std::size_t column = linear % columns;
        const std::size_t row = (linear / columns) % rows;
        if (column == exceptional_column) {
            codes[linear] = left ? 0x7F80 : 0x3F80;
            continue;
        }
        const bool committed = row % 2 == 0;
        const MulPair* table = committed ? kCommittedPairs : kAuthorityPairs;
        const std::size_t table_size = committed
                ? sizeof(kCommittedPairs) / sizeof(kCommittedPairs[0])
                : sizeof(kAuthorityPairs) / sizeof(kAuthorityPairs[0]);
        const MulPair& pair = table[(row / 2 + column) % table_size];
        codes[linear] = left ? pair.lhs : pair.rhs;
    }
    return codes;
}

// Result element -> operand element for a right-aligned broadcast: an axis of
// extent one reads its only coordinate.
[[nodiscard]] std::vector<std::size_t> broadcast_sources(
        const iom::TensorSpec& operand, const iom::TensorSpec& result) {
    const auto operand_dimensions = operand.shape.dimensions();
    const auto result_dimensions = result.shape.dimensions();
    REQUIRE_EQ(operand_dimensions.size(), result_dimensions.size());
    const std::size_t count = result.shape.element_count();
    std::vector<std::size_t> sources(count);
    std::vector<std::size_t> coordinates(result_dimensions.size());
    for (std::size_t linear = 0; linear < count; ++linear) {
        std::size_t rest = linear;
        for (std::size_t axis = result_dimensions.size(); axis-- > 0;) {
            coordinates[axis] = rest % result_dimensions[axis];
            rest /= result_dimensions[axis];
        }
        std::size_t operand_linear = 0;
        for (std::size_t axis = 0; axis < result_dimensions.size(); ++axis) {
            const std::size_t coordinate =
                    operand_dimensions[axis] == 1 ? 0 : coordinates[axis];
            operand_linear =
                    operand_linear * operand_dimensions[axis] + coordinate;
        }
        sources[linear] = operand_linear;
    }
    return sources;
}

// The contiguous row-major host encoding of one logical leaf sequence, in the
// bit order the harness's independent encoder uses.
[[nodiscard]] Bytes logical_bytes(const Codes& codes) {
    Bytes buffer(codes.size() * sizeof(std::uint16_t), std::byte{0});
    auto* base = reinterpret_cast<unsigned char*>(buffer.data());
    for (std::size_t index = 0; index < codes.size(); ++index) {
        iom_conformance::write_bits(base, index * 16, 16, codes[index]);
    }
    return buffer;
}

// The physical storage of one owner whose logical leaves are `codes` and whose
// every padded byte is `padding`. Slots come from the harness's independent
// canonical encoder, which shares no code with the production mapper.
[[nodiscard]] Bytes storage_image(
        const iom::TensorSpec& spec, const Codes& codes, std::byte padding) {
    REQUIRE_EQ(codes.size(), spec.shape.element_count());
    const auto dimensions = spec.shape.dimensions();
    const std::size_t bits = iom_conformance::bits_of(spec.data_type);
    Bytes storage(spec.tiled_storage_nbytes(), padding);
    std::vector<std::size_t> coordinates(dimensions.size());
    for (std::size_t linear = 0; linear < codes.size(); ++linear) {
        std::size_t rest = linear;
        for (std::size_t axis = dimensions.size(); axis-- > 0;) {
            coordinates[axis] = rest % dimensions[axis];
            rest /= dimensions[axis];
        }
        iom_conformance::write_storage_bits(
                storage.data(),
                iom_conformance::canonical_layout_slot(spec, coordinates) * bits,
                bits, codes[linear]);
    }
    return storage;
}

void seed_storage(iom::Tensor& owner, std::span<const std::byte> image) {
    REQUIRE_EQ(image.size(), owner.view().spec().tiled_storage_nbytes());
    std::memcpy(owner.view().native_handle(), image.data(), image.size());
}

[[nodiscard]] Bytes observe_storage(iom::Tensor& owner) {
    const std::size_t bytes = owner.view().spec().tiled_storage_nbytes();
    Bytes image(bytes);
    std::memcpy(image.data(), owner.view().native_handle(), bytes);
    return image;
}

// The accepted work of one request, as the settled `BinaryMul` stages recorded
// it: committed SIMD tile rows, and BF16 MUL leaves the scalar authority
// recomputed.
struct DirectedWork {
    std::uint64_t native_rows = 0;
    std::uint64_t authority_leaves = 0;
};

[[nodiscard]] DirectedWork observed_work() {
    return {
            iom::cpu_detail::avx512_bf16_test_observation(
                    iom::cpu_detail::Avx512Bf16Stage::BinaryMul,
                    iom::cpu_detail::Avx512Bf16Path::Native),
            iom::cpu_detail::avx512_bf16_test_observation(
                    iom::cpu_detail::Avx512Bf16Stage::BinaryMul,
                    iom::cpu_detail::Avx512Bf16Path::Fallback)};
}

// The observed base of one owner, as the case that claims a misaligned base
// needs to check.
[[nodiscard]] std::uintptr_t owner_base(const iom::Tensor& owner) {
    return reinterpret_cast<std::uintptr_t>(owner.view().native_handle());
}

// Where two storage images first differ, so a failure names the offset instead
// of dumping both images.
[[nodiscard]] std::size_t first_difference(
        std::span<const std::byte> actual, std::span<const std::byte> expected) {
    const std::size_t shared =
            actual.size() < expected.size() ? actual.size() : expected.size();
    for (std::size_t index = 0; index < shared; ++index) {
        if (actual[index] != expected[index]) {
            return index;
        }
    }
    return shared == expected.size() ? actual.size() : shared;
}

void check_storage(
        std::string_view label, std::string_view what,
        std::span<const std::byte> actual, std::span<const std::byte> expected) {
    REQUIRE_MESSAGE(
            actual.size() == expected.size(),
            label << ": " << what << " has " << actual.size()
                  << " bytes instead of " << expected.size());
    const std::size_t difference = first_difference(actual, expected);
    CHECK_MESSAGE(
            difference == actual.size(),
            label << ": " << what << " first differs at byte " << difference
                  << " of " << actual.size());
}

// The authority's expected result codes of one directed request, with the
// independent oracle checked against it element by element.
[[nodiscard]] Codes checked_expected(const DirectedRequest& request) {
    const std::size_t count = request.out_spec.shape.element_count();
    REQUIRE_EQ(request.lhs_source.size(), count);
    REQUIRE_EQ(request.rhs_source.size(), count);
    REQUIRE_EQ(request.lhs_codes.size(), request.lhs_spec.shape.element_count());
    REQUIRE_EQ(request.rhs_codes.size(), request.rhs_spec.shape.element_count());
    Codes expected(count);
    for (std::size_t linear = 0; linear < count; ++linear) {
        const std::uint16_t lhs = request.lhs_codes[request.lhs_source[linear]];
        const std::uint16_t rhs = request.rhs_codes[request.rhs_source[linear]];
        expected[linear] = authority_mul(lhs, rhs);
        CAPTURE(linear);
        CAPTURE(lhs);
        CAPTURE(rhs);
        CHECK_MESSAGE(
                iom_conformance::add_non_f32_float_classes(
                        iom::DataType::BF16, expected[linear],
                        independent_mul(lhs, rhs)),
                request.label << ": the two oracles disagree at element "
                              << linear);
    }
    return expected;
}

// Submit one directed request through the real queue and check everything the
// contract observes: the authority's own result byte for byte in physical
// storage (logical leaves and every padded byte), the operand storage the
// operation must not touch, and the settled `BinaryMul` work of exactly this
// request. Returns that work.
[[nodiscard]] DirectedWork run_directed_request(
        iom::Device& device, iom::DeviceOps& queue,
        const DirectedRequest& request) {
    if (request.out_is_lhs) {
        REQUIRE(request.lhs_spec == request.out_spec);
    }
    const Codes expected = checked_expected(request);

    auto lhs = device.create_tensor(request.lhs_spec);
    auto rhs = device.create_tensor(request.rhs_spec);
    auto out = request.out_is_lhs
            ? nullptr : device.create_tensor(request.out_spec);
    const Bytes lhs_before =
            storage_image(request.lhs_spec, request.lhs_codes, kPaddingPoison);
    const Bytes rhs_before =
            storage_image(request.rhs_spec, request.rhs_codes, kPaddingPoison);
    const Bytes expected_image =
            storage_image(request.out_spec, expected, kPaddingPoison);
    seed_storage(*lhs, lhs_before);
    seed_storage(*rhs, rhs_before);
    if (out) {
        seed_storage(
                *out,
                Bytes(request.out_spec.tiled_storage_nbytes(), kPaddingPoison));
    }
    iom::TensorView& out_view = out ? out->view() : lhs->view();

    iom::cpu_detail::avx512_bf16_test_reset_observations();
    const iom::oid token = queue.mul(lhs->view(), rhs->view(), out_view);
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue.wait(token));
    const DirectedWork work = observed_work();

    check_storage(
            request.label, "the result storage",
            observe_storage(out ? *out : *lhs), expected_image);
    if (!request.out_is_lhs) {
        check_storage(
                request.label, "the left operand", observe_storage(*lhs),
                lhs_before);
        check_storage(
                request.label, "the right operand", observe_storage(*rhs),
                rhs_before);
    }
    return work;
}

}  // namespace

TEST_CASE("CPU AVX-512 BF16 MUL directed classes through the real queue") {
    MisalignedBaseAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    // The directed corpus: two leading planes, a row count that leaves a partial
    // tile, and a feature count that leaves a partial tile column, so committed
    // tile rows, declined tile rows, and the odd feature tail all appear in one
    // accepted request.
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 17, 33}}, iom::DataType::BF16};

    // Every CPU owner here is 32-byte aligned, the device's own promise, and
    // deliberately 64-byte misaligned, so an accelerated tile row must be
    // correct on a base misaligned for any wider access. The observed base is
    // checked, so the case cannot silently lose that property.
    {
        auto probe = device->create_tensor(spec);
        const std::uintptr_t base = owner_base(*probe);
        REQUIRE_EQ(base % 32, 0u);
        CHECK_NE(base % 64, 0u);
    }

    DirectedRequest request;
    request.lhs_spec = spec;
    request.rhs_spec = spec;
    request.out_spec = spec;
    request.lhs_codes = directed_codes(spec, true, kNoExceptionalColumn);
    request.rhs_codes = directed_codes(spec, false, kNoExceptionalColumn);
    request.lhs_source = broadcast_sources(spec, spec);
    request.rhs_source = broadcast_sources(spec, spec);
    request.label = "directed corpus";

    const DirectedWork work = run_directed_request(*device, *queue, request);
    if (iom::cpu_detail::avx512_bf16_available()) {
        // Ordinary rows must be committed vector tile rows, and the exceptional
        // rows must be recomputed by the scalar authority.
        CHECK(work.native_rows > 0);
        CHECK(work.authority_leaves > 0);
    } else {
        // Without this build's isolated sources, or on a host without the
        // required CPU and OS vector state, the whole request is authority work
        // and nothing may report otherwise.
        CHECK_EQ(work.native_rows, 0u);
        CHECK_EQ(work.authority_leaves, 0u);
    }
}

TEST_CASE("CPU AVX-512 BF16 MUL broadcast and exact in-place alias") {
    MisalignedBaseAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    const iom::TensorSpec row_spec{
            iom::TensorShape{{17, 33}}, iom::DataType::BF16};
    const Codes row_codes =
            directed_codes(row_spec, false, kNoExceptionalColumn);

    // A one-element broadcast left operand over the directed right operand: the
    // broadcast enters the vector tile row as a broadcast lane, and the right
    // operand still contributes committed and declined rows.
    {
        const iom::TensorSpec scalar_spec{
                iom::TensorShape{{1, 1}}, iom::DataType::BF16};
        DirectedRequest request;
        request.lhs_spec = scalar_spec;
        request.rhs_spec = row_spec;
        request.out_spec = row_spec;
        request.lhs_codes = Codes{0x3F80};
        request.rhs_codes = row_codes;
        request.lhs_source = broadcast_sources(scalar_spec, row_spec);
        request.rhs_source = broadcast_sources(row_spec, row_spec);
        request.label = "scalar broadcast";
        const DirectedWork work = run_directed_request(*device, *queue, request);
        if (iom::cpu_detail::avx512_bf16_available()) {
            CHECK(work.native_rows > 0);
            CHECK(work.authority_leaves > 0);
        }
    }

    // A column-broadcast left operand whose single value is an infinity: every
    // lane is exceptional, so no tile row may commit and the authority owns the
    // whole request.
    {
        const iom::TensorSpec column_spec{
                iom::TensorShape{{17, 1}}, iom::DataType::BF16};
        DirectedRequest request;
        request.lhs_spec = column_spec;
        request.rhs_spec = row_spec;
        request.out_spec = row_spec;
        request.lhs_codes.assign(17, static_cast<std::uint16_t>(0x7F80));
        request.rhs_codes = row_codes;
        request.lhs_source = broadcast_sources(column_spec, row_spec);
        request.rhs_source = broadcast_sources(row_spec, row_spec);
        request.label = "infinity broadcast";
        const DirectedWork work = run_directed_request(*device, *queue, request);
        CHECK_EQ(work.native_rows, 0u);
        if (iom::cpu_detail::avx512_bf16_available()) {
            CHECK(work.authority_leaves > 0);
        } else {
            CHECK_EQ(work.authority_leaves, 0u);
        }
    }

    // The exact in-place alias, with one exceptional column inside the first
    // tile column of every row: that row's first tile-aligned chunk is
    // recomputed by the authority while its second chunk is committed SIMD
    // work, and both read their operands before storing into the same storage.
    {
        DirectedRequest request;
        request.lhs_spec = row_spec;
        request.rhs_spec = row_spec;
        request.out_spec = row_spec;
        request.lhs_codes = directed_codes(row_spec, true, 7);
        request.rhs_codes = directed_codes(row_spec, false, 7);
        request.lhs_source = broadcast_sources(row_spec, row_spec);
        request.rhs_source = broadcast_sources(row_spec, row_spec);
        request.out_is_lhs = true;
        request.label = "in-place alias";
        const DirectedWork work = run_directed_request(*device, *queue, request);
        if (iom::cpu_detail::avx512_bf16_available()) {
            CHECK(work.native_rows > 0);
            CHECK(work.authority_leaves > 0);
        }
    }
}

TEST_CASE("CPU AVX-512 BF16 MUL leading-plane views keep other planes intact") {
    MisalignedBaseAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    // One owner per operand with two leading planes, of which one is selected by
    // a slice: the accelerated rows must follow the view's plane offset and
    // stride, and the unselected plane plus every padded byte must keep the
    // caller's bytes.
    const iom::TensorSpec owner_spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::BF16};
    const iom::TensorSpec view_spec{
            iom::TensorShape{{1, 3, 17, 33}}, iom::DataType::BF16};
    DirectedRequest request;
    request.lhs_spec = view_spec;
    request.rhs_spec = view_spec;
    request.out_spec = view_spec;
    request.lhs_codes = directed_codes(view_spec, true, kNoExceptionalColumn);
    request.rhs_codes = directed_codes(view_spec, false, kNoExceptionalColumn);
    request.lhs_source = broadcast_sources(view_spec, view_spec);
    request.rhs_source = broadcast_sources(view_spec, view_spec);
    request.label = "leading-plane view";
    const Codes expected = checked_expected(request);

    auto lhs_owner = device->create_tensor(owner_spec);
    auto rhs_owner = device->create_tensor(owner_spec);
    auto out_owner = device->create_tensor(owner_spec);
    const iom::TensorView lhs_view = lhs_owner->view().slice(0, 1, 1);
    const iom::TensorView rhs_view = rhs_owner->view().slice(0, 0, 1);
    iom::TensorView out_view = out_owner->view().slice(0, 1, 1);

    // Each owner is seeded from its own view's leaves over a poisoned image, so
    // the unselected plane holds the poison and cannot be written unnoticed.
    const Bytes lhs_before = [&] {
        Bytes image(owner_spec.tiled_storage_nbytes(), kPaddingPoison);
        iom_conformance::apply_standard_tiled_view(
                lhs_view, owner_spec, logical_bytes(request.lhs_codes), image);
        return image;
    }();
    const Bytes rhs_before = [&] {
        Bytes image(owner_spec.tiled_storage_nbytes(), kPaddingPoison);
        iom_conformance::apply_standard_tiled_view(
                rhs_view, owner_spec, logical_bytes(request.rhs_codes), image);
        return image;
    }();
    const Bytes out_seed(owner_spec.tiled_storage_nbytes(), kPaddingPoison);
    const Bytes out_expected = [&] {
        Bytes image(out_seed);
        iom_conformance::apply_standard_tiled_view(
                out_view, owner_spec, logical_bytes(expected), image);
        return image;
    }();
    seed_storage(*lhs_owner, lhs_before);
    seed_storage(*rhs_owner, rhs_before);
    seed_storage(*out_owner, out_seed);

    iom::cpu_detail::avx512_bf16_test_reset_observations();
    const iom::oid token =
            queue->mul(lhs_view, rhs_view, out_view);
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue->wait(token));
    const DirectedWork work = observed_work();

    check_storage(
            request.label, "the result owner", observe_storage(*out_owner),
            out_expected);
    check_storage(
            request.label, "the left owner", observe_storage(*lhs_owner),
            lhs_before);
    check_storage(
            request.label, "the right owner", observe_storage(*rhs_owner),
            rhs_before);
    if (iom::cpu_detail::avx512_bf16_available()) {
        CHECK(work.native_rows > 0);
        CHECK(work.authority_leaves > 0);
    } else {
        CHECK_EQ(work.native_rows, 0u);
        CHECK_EQ(work.authority_leaves, 0u);
    }
}