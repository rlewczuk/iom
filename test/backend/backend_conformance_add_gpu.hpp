#pragma once
// Backend-neutral GPU ADD conformance cases shared by the CUDA and ROCm
// drivers: exhaustive low-width encodings, broadcast/transform/tail mapping,
// [1,1] scalars, and exact in-place packed aliases, all judged against the
// independent oracle rather than the implementation.
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
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
