#pragma once

// Backend-neutral standard-tiled binary conformance.  Drivers provide only
// device construction and call this once per selected operation.
#include <doctest/doctest.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend/backend_conformance_add.hpp"
#include "iom/iom.hpp"

namespace iom_conformance {

inline std::uint64_t gpu_read_bits(
        const std::vector<std::byte>& bytes, std::size_t index,
        std::size_t bits) {
    std::uint64_t value = 0;
    for (std::size_t bit = 0; bit < bits; ++bit) {
        const std::size_t position = index * bits + bit;
        if ((static_cast<unsigned char>(bytes[position / 8])
             >> (position % 8)) & 1u)
            value |= std::uint64_t{1} << bit;
    }
    return value;
}

inline void gpu_write_bits(
        std::vector<std::byte>& bytes, std::size_t index,
        std::size_t bits, std::uint64_t value) {
    write_bits(reinterpret_cast<unsigned char*>(bytes.data()), index * bits,
               bits, value);
}

inline void run_gpu_eltwise_conformance(
        iom::Device& candidate, BinaryOperation operation) {
    static constexpr iom::DataType types[] = {
            iom::DataType::I2, iom::DataType::U2, iom::DataType::I4,
            iom::DataType::U4, iom::DataType::I8, iom::DataType::U8,
            iom::DataType::I16, iom::DataType::U16, iom::DataType::I32,
            iom::DataType::U32, iom::DataType::I64, iom::DataType::U64,
            iom::DataType::F4_E2M1, iom::DataType::F6_E2M3,
            iom::DataType::F6_E3M2, iom::DataType::F8_E4M3FN,
            iom::DataType::F8_E5M2, iom::DataType::F16,
            iom::DataType::BF16, iom::DataType::F32, iom::DataType::F64};
    auto queue = candidate.create_ops();
    for (const auto type : types) {
        const bool integer = !add_is_float(type);
        if (operation == BinaryOperation::div && integer) continue;
        const iom::TensorSpec spec{iom::TensorShape{{1, 17, 33}}, type};
        const std::size_t bits = iom::detail::leaf_bits(type);
        const std::size_t count = spec.shape.element_count();
        std::vector<std::byte> lhs(spec.logical_nbytes());
        std::vector<std::byte> rhs(spec.logical_nbytes());
        std::vector<std::byte> expected(spec.logical_nbytes());
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint64_t a = element_pattern(type, i, 0x13579BDF);
            const std::uint64_t b = element_pattern(type, i, 0x2468ACE0);
            gpu_write_bits(lhs, i, bits, a);
            gpu_write_bits(rhs, i, bits, b);
            gpu_write_bits(expected, i, bits,
                           add_oracle::binary(type, a, b,
                                              static_cast<add_oracle::operation>(operation)));
        }
        auto l = candidate.create_tensor(spec);
        auto r = candidate.create_tensor(spec);
        auto out = candidate.create_tensor(spec);
        l->view().copy_from_host(lhs);
        r->view().copy_from_host(rhs);
        out->view().copy_from_host(std::vector<std::byte>(expected.size(), std::byte{0xAA}));
        const iom::oid token = submit_binary_operation(
                *queue, operation, l->view(), r->view(), out->view());
        REQUIRE(iom::oid_is_token(token));
        const auto actual = read_logical(out->view());
        for (std::size_t i = 0; i < count; ++i) {
            const auto got = gpu_read_bits(actual, i, bits);
            const auto want = gpu_read_bits(expected, i, bits);
            if (type == iom::DataType::F32)
                CHECK(add_f32_within_ulp(static_cast<std::uint32_t>(got),
                                         static_cast<std::uint32_t>(want)));
            else if (!integer)
                CHECK(add_non_f32_float_classes(type, got, want));
            else
                CHECK_EQ(got, want);
        }
    }
}

inline void run_gpu_eltwise_mapping_conformance(
        iom::Device& candidate, BinaryOperation operation) {
    const iom::TensorSpec lhs_spec{
            iom::TensorShape{{2, 1, 33}}, iom::DataType::U8};
    const iom::TensorSpec rhs_spec{
            iom::TensorShape{{1, 17, 1}}, iom::DataType::U8};
    const iom::TensorSpec out_spec{
            iom::TensorShape{{2, 17, 33}}, iom::DataType::U8};
    auto lhs = candidate.create_tensor(lhs_spec);
    auto rhs = candidate.create_tensor(rhs_spec);
    auto out = candidate.create_tensor(out_spec);
    lhs->view().copy_from_host(encode_logical(lhs_spec, 0x1111));
    rhs->view().copy_from_host(encode_logical(rhs_spec, 0x2222));
    auto queue = candidate.create_ops();
    const auto token = submit_binary_operation(
            *queue, operation, lhs->view(), rhs->view(), out->view());
    if (operation == BinaryOperation::div) {
        CHECK_EQ(token, iom::to_oid(iom::OidError::Unsupported));
        return;
    }
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue->wait(token));
    CHECK_EQ(read_logical(out->view()).size(), out_spec.logical_nbytes());
}

}  // namespace iom_conformance
