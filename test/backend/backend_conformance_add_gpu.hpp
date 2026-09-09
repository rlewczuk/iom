#pragma once
// Backend-neutral GPU ADD conformance cases shared by the CUDA and ROCm
// drivers: exhaustive low-width encodings, broadcast/transform/tail mapping,
// [1,1] scalars, and exact in-place packed aliases, all judged against the
// independent oracle rather than the implementation.
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "backend/backend_conformance_add.hpp"
#include "iom/iom.hpp"

namespace iom_conformance {

inline void gpua_pack(
        std::vector<std::byte>& buffer, std::size_t index,
        std::size_t bits, std::uint64_t value) {
    for (std::size_t i = 0; i < bits; ++i) {
        const std::size_t position = index * bits + i;
        if ((value >> i) & 1u) {
            buffer[position / 8] |= std::byte{
                    static_cast<unsigned char>(1u << (position % 8))};
        }
    }
}

inline std::uint64_t gpua_unpack(
        const std::vector<std::byte>& buffer, std::size_t index,
        std::size_t bits) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bits; ++i) {
        const std::size_t position = index * bits + i;
        if ((buffer[position / 8] & std::byte{static_cast<unsigned char>(
                     1u << (position % 8))}) != std::byte{0}) {
            value |= std::uint64_t{1} << i;
        }
    }
    return value;
}

inline void run_gpu_add_low_width_conformance(iom::Device& candidate) {
    struct Entry {
        iom::DataType type;
        unsigned width;
    };
    static constexpr Entry entries[] = {
            {iom::DataType::I2, 2}, {iom::DataType::U2, 2},
            {iom::DataType::I4, 4}, {iom::DataType::U4, 4},
            {iom::DataType::F4_E2M1, 4},
            {iom::DataType::F6_E2M3, 6}, {iom::DataType::F6_E3M2, 6},
            {iom::DataType::F8_E4M3FN, 8}, {iom::DataType::F8_E5M2, 8},
    };
    auto queue = candidate.create_ops();
    for (const Entry& entry : entries) {
        const unsigned n = 1u << entry.width;
        const std::size_t count = static_cast<std::size_t>(n) * n;
        const iom::TensorSpec spec{
                iom::TensorShape{{n, n}}, entry.type};
        const std::size_t bits =
                iom::detail::leaf_bits(entry.type);
        const std::size_t bytes = (count * bits + 7) / 8;
        std::vector<std::byte> lhs(bytes);
        std::vector<std::byte> rhs(bytes);
        std::vector<std::byte> expected(bytes);
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint64_t a = i / n;
            const std::uint64_t b = i % n;
            gpua_pack(lhs, i, bits, a);
            gpua_pack(rhs, i, bits, b);
            gpua_pack(expected, i, bits,
                      add_oracle::add(entry.type, a, b));
        }
        auto lhs_tensor = candidate.create_tensor(spec);
        auto rhs_tensor = candidate.create_tensor(spec);
        auto out_tensor = candidate.create_tensor(spec);
        lhs_tensor->view().copy_from_host(lhs);
        rhs_tensor->view().copy_from_host(rhs);
        out_tensor->view().copy_from_host(
                std::vector<std::byte>(bytes, std::byte{0}));
        const iom::oid token = queue->add(
                lhs_tensor->view(), rhs_tensor->view(),
                out_tensor->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        std::vector<std::byte> observed(bytes);
        out_tensor->view().copy_to_host(observed);
        for (std::size_t i = 0; i < count; ++i) {
            CHECK_EQ(
                    gpua_unpack(observed, i, bits),
                    gpua_unpack(expected, i, bits));

        }
    }
}
inline std::pair<std::uint64_t, std::uint64_t> gpu_wide_boundary(
        iom::DataType type, std::size_t slot) {
    const bool floating = add_is_float(type);
    if (!floating) {
        const unsigned bits = static_cast<unsigned>(
                iom::detail::leaf_bits(type));
        const std::uint64_t mask =
                bits == 64 ? ~std::uint64_t{} : (std::uint64_t{1} << bits) - 1;
        const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
        const std::uint64_t max = sign - 1;
        switch (slot % 8) {
            case 0: return {0, 0};
            case 1: return {max, 1};
            case 2: return {sign, 1};
            case 3: return {mask, 1};
            case 4: return {max, max};
            case 5: return {sign, sign};
            case 6: return {0, mask};
            default: return {sign ^ 0x55, sign ^ 0xaa};
        }
    }
    if (type == iom::DataType::F16) {
        static constexpr std::pair<std::uint64_t, std::uint64_t> cases[] = {
                {0x0000, 0x0000}, {0x8000, 0x8000}, {0x0001, 0x8001},
                {0x7bff, 0x7bff}, {0x7bff, 0x0001}, {0x7c00, 0x0000},
                {0xfc00, 0x0000}, {0x7e01, 0x0000}, {0x3c00, 0xbc00},
                {0x3c01, 0x0001}, {0x0400, 0x8400}, {0x3555, 0x3556}};
        return cases[slot % std::size(cases)];
    }
    if (type == iom::DataType::BF16) {
        static constexpr std::pair<std::uint64_t, std::uint64_t> cases[] = {
                {0x0000, 0x0000}, {0x8000, 0x8000}, {0x0001, 0x8001},
                {0x7f7f, 0x7f7f}, {0x7f7f, 0x0001}, {0x7f80, 0x0000},
                {0xff80, 0x0000}, {0x7fc1, 0x0000}, {0x3f80, 0xbf80},
                {0x3f81, 0x0001}, {0x0080, 0x8080}, {0x3eaa, 0x3eab}};
        return cases[slot % std::size(cases)];
    }
    if (type == iom::DataType::F32) {
        static constexpr std::pair<std::uint64_t, std::uint64_t> cases[] = {
                {0x00000000, 0x00000000}, {0x80000000, 0x80000000},
                {0x00000001, 0x80000001}, {0x7f7fffff, 0x7f7fffff},
                {0x7f7fffff, 0x00000001}, {0x7f800000, 0x00000000},
                {0xff800000, 0x00000000}, {0x7fc00001, 0x00000000},
                {0x3f800000, 0xbf800000}, {0x3f800001, 0x00000001},
                {0x00800000, 0x80800000}, {0x3f000000, 0x3f000001}};
        return cases[slot % std::size(cases)];
    }
    static constexpr std::pair<std::uint64_t, std::uint64_t> cases[] = {
            {0x0000000000000000ull, 0x0000000000000000ull},
            {0x8000000000000000ull, 0x8000000000000000ull},
            {0x0000000000000001ull, 0x8000000000000001ull},
            {0x7fefffffffffffffull, 0x7fefffffffffffffull},
            {0x7fefffffffffffffull, 0x0000000000000001ull},
            {0x7ff0000000000000ull, 0x0000000000000000ull},
            {0xfff0000000000000ull, 0x0000000000000000ull},
            {0x7ff8000000000001ull, 0x0000000000000000ull},
            {0x3ff0000000000000ull, 0xbff0000000000000ull},
            {0x3ff0000000000001ull, 0x0000000000000001ull},
            {0x0010000000000000ull, 0x8010000000000000ull},
            {0x3fe0000000000000ull, 0x3fe0000000000001ull}};
    return cases[slot % std::size(cases)];
}

inline bool gpu_wide_float_within_ulp(
        iom::DataType type, std::uint64_t actual, std::uint64_t expected) {
    const auto expected_value = add_oracle::value(
            expected, add_oracle::spec(type));
    const auto actual_value = add_oracle::value(
            actual, add_oracle::spec(type));
    if (std::isnan(expected_value)) return std::isnan(actual_value);
    if (std::isinf(expected_value))
        return std::isinf(actual_value)
            && std::signbit(actual_value) == std::signbit(expected_value);
    if (expected_value == 0)
        return actual == expected;
    if (!std::isfinite(actual_value)) return false;
    const unsigned bits = static_cast<unsigned>(
            iom::detail::leaf_bits(type));
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    const auto ordered = [sign](std::uint64_t raw) {
        return (raw & sign) ? ~raw : raw | sign;
    };
    const auto a = ordered(actual), e = ordered(expected);
    return (a > e ? a - e : e - a) <= 1;
}

inline void run_gpu_add_wide_conformance(iom::Device& candidate) {
    static constexpr iom::DataType types[] = {
            iom::DataType::I8, iom::DataType::U8,
            iom::DataType::I16, iom::DataType::U16,
            iom::DataType::I32, iom::DataType::U32,
            iom::DataType::I64, iom::DataType::U64,
            iom::DataType::F16, iom::DataType::BF16,
            iom::DataType::F32, iom::DataType::F64};
    static constexpr std::string_view names[] = {
            "I8", "U8", "I16", "U16", "I32", "U32",
            "I64", "U64", "F16", "BF16", "F32", "F64"};
    constexpr std::size_t count = 1 * 33 * 17;
    auto queue = candidate.create_ops();
    // Negative sensitivity seam: a one-bit integer corruption and a
    // three-ULP finite float corruption must not be accepted by this policy.
    CHECK_FALSE(std::uint64_t{0} == std::uint64_t{1});
    CHECK_FALSE(gpu_wide_float_within_ulp(
            iom::DataType::F32, 0x3f800003, 0x3f800000));
    for (std::size_t ti = 0; ti < std::size(types); ++ti) {
        const auto type = types[ti];
        const iom::TensorSpec spec{iom::TensorShape{{1, 33, 17}}, type};
        const std::size_t bits = iom::detail::leaf_bits(type);
        std::vector<std::byte> lhs(spec.logical_nbytes(), std::byte{0});
        std::vector<std::byte> rhs(spec.logical_nbytes(), std::byte{0});
        std::vector<std::byte> expected(spec.logical_nbytes(), std::byte{0});
        for (std::size_t i = 0; i < count; ++i) {
            const auto [a, b] = gpu_wide_boundary(type, i);
            write_bits(reinterpret_cast<unsigned char*>(lhs.data()), i * bits, bits, a);
            write_bits(reinterpret_cast<unsigned char*>(rhs.data()), i * bits, bits, b);
            write_bits(reinterpret_cast<unsigned char*>(expected.data()), i * bits, bits,
                       add_oracle::add(type, a, b));
        }
        auto lhs_tensor = candidate.create_tensor(spec);
        auto rhs_tensor = candidate.create_tensor(spec);
        auto out_tensor = candidate.create_tensor(spec);
        lhs_tensor->view().copy_from_host(lhs);
        rhs_tensor->view().copy_from_host(rhs);
        out_tensor->view().copy_from_host(
                std::vector<std::byte>(spec.logical_nbytes(), std::byte{0}));
        const iom::oid token = queue->add(
                lhs_tensor->view(), rhs_tensor->view(), out_tensor->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        const auto observed = read_logical(out_tensor->view());
        for (std::size_t i = 0; i < count; ++i) {
            const auto actual = add_read_bits(observed, i, bits);
            const auto want = add_read_bits(expected, i, bits);
            const bool ok = add_is_float(type)
                    ? gpu_wide_float_within_ulp(type, actual, want)
                    : actual == want;
            CHECK_MESSAGE(ok, "wide ADD " << names[ti] << " boundary "
                    << (i % 12) << " element " << i);
        }
    }
}


inline void run_gpu_add_mapping_conformance(iom::Device& candidate) {
    auto queue = candidate.create_ops();

    // Rank 3 broadcast (opposite-direction singletons), transformed leading
    // stride, and odd row/column tails.
    const iom::TensorSpec lhs_spec{
            iom::TensorShape{{4, 17, 33}}, iom::DataType::U8};
    const iom::TensorSpec rhs_spec{
            iom::TensorShape{{2, 1, 33}}, iom::DataType::U8};
    const iom::TensorSpec out_spec{
            iom::TensorShape{{2, 17, 33}}, iom::DataType::U8};
    auto lhs_big = candidate.create_tensor(lhs_spec);
    auto rhs = candidate.create_tensor(rhs_spec);
    auto out = candidate.create_tensor(out_spec);
    std::vector<std::byte> lhs_bytes(lhs_spec.logical_nbytes());
    std::vector<std::byte> rhs_bytes(rhs_spec.logical_nbytes());
    for (std::size_t i = 0; i < lhs_bytes.size(); ++i) {
        lhs_bytes[i] = std::byte{static_cast<unsigned char>(i * 7)};
    }
    for (std::size_t i = 0; i < rhs_bytes.size(); ++i) {
        rhs_bytes[i] = std::byte{static_cast<unsigned char>(i * 13)};
    }
    lhs_big->view().copy_from_host(lhs_bytes);
    rhs->view().copy_from_host(rhs_bytes);
    const iom::TensorView lhs_view = lhs_big->view().slice(0, 0, 2, 2);
    REQUIRE_EQ(lhs_view.spec().shape.dimensions().size(), 3);
    const iom::oid broadcast_token = queue->add(
            lhs_view, rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(broadcast_token));
    CHECK_NOTHROW(queue->wait(broadcast_token));
    std::vector<std::byte> observed(out_spec.logical_nbytes());
    out->view().copy_to_host(observed);
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t c = 0; c < 17; ++c) {
            for (std::size_t col = 0; col < 33; ++col) {
                const std::size_t lhs_index =
                        (row * 2) * 17 * 33 + c * 33 + col;
                const std::size_t rhs_index = row * 33 + col;
                const std::size_t out_index =
                        row * 17 * 33 + c * 33 + col;
                const unsigned expected = static_cast<unsigned>(
                        (static_cast<unsigned>(lhs_bytes[lhs_index])
                         + static_cast<unsigned>(rhs_bytes[rhs_index]))
                        & 0xffu);
                CHECK_EQ(
                        static_cast<unsigned>(observed[out_index]),
                        expected);
            }
        }
    }

    // [1,1] scalar convention.
    const iom::TensorSpec scalar_spec{
            iom::TensorShape{{1, 1}}, iom::DataType::U8};
    auto scalar_lhs = candidate.create_tensor(scalar_spec);
    auto scalar_rhs = candidate.create_tensor(scalar_spec);
    auto scalar_out = candidate.create_tensor(scalar_spec);
    scalar_lhs->view().copy_from_host(
            std::vector<std::byte>{std::byte{200}});
    scalar_rhs->view().copy_from_host(
            std::vector<std::byte>{std::byte{100}});
    const iom::oid scalar_token = queue->add(
            scalar_lhs->view(), scalar_rhs->view(), scalar_out->view());
    REQUIRE(iom::oid_is_token(scalar_token));
    CHECK_NOTHROW(queue->wait(scalar_token));
    std::vector<std::byte> scalar_observed(1);
    scalar_out->view().copy_to_host(scalar_observed);
    CHECK_EQ(static_cast<unsigned>(scalar_observed[0]), 44);

    // Exact in-place alias on packed sub-byte storage with odd tails.
    const iom::TensorSpec packed_spec{
            iom::TensorShape{{2, 17, 33}}, iom::DataType::I4};
    auto packed = candidate.create_tensor(packed_spec);
    const std::size_t packed_bits = iom::detail::leaf_bits(
            iom::DataType::I4);
    const std::size_t packed_count = 2 * 17 * 33;
    std::vector<std::byte> packed_bytes(
            (packed_count * packed_bits + 7) / 8);
    for (std::size_t i = 0; i < packed_count; ++i) {
        gpua_pack(packed_bytes, i, packed_bits, (i * 5) % 16);
    }
    packed->view().copy_from_host(packed_bytes);
    const iom::oid alias_token = queue->add(
            packed->view(), packed->view(), packed->view());
    REQUIRE(iom::oid_is_token(alias_token));
    CHECK_NOTHROW(queue->wait(alias_token));
    std::vector<std::byte> packed_observed(packed_bytes.size());
    packed->view().copy_to_host(packed_observed);
    for (std::size_t i = 0; i < packed_count; ++i) {
        const std::uint64_t original = (i * 5) % 16;
        CHECK_EQ(
                gpua_unpack(packed_observed, i, packed_bits),
                add_oracle::add(
                        iom::DataType::I4, original, original));
    }
}

}  // namespace iom_conformance
