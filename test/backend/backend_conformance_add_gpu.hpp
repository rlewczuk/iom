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
        iom_conformance::copy_from_host(l->view(), lhs);
        iom_conformance::copy_from_host(r->view(), rhs);
        iom_conformance::copy_from_host(out->view(), std::vector<std::byte>(expected.size(), std::byte{0xAA}));
        const auto requirements = query_binary_workspace_requirements(
                *queue, operation, l->view(), r->view(), out->view());
        std::unique_ptr<iom::RawWorkspace> workspace_owner;
        if (requirements.bytes != 0) {
            workspace_owner = candidate.create_workspace(requirements.bytes);
        }
        iom::oid token = 0;
        if (workspace_owner) {
            const iom::RawWorkspaceView workspace = workspace_owner->view();
            token = submit_binary_operation(
                    *queue, operation, l->view(), r->view(), out->view(),
                    workspace);
        } else {
            token = submit_binary_operation(
                    *queue, operation, l->view(), r->view(), out->view());
        }
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
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
    const auto lhs_bytes = encode_logical(lhs_spec, 0x1111);
    const auto rhs_bytes = encode_logical(rhs_spec, 0x2222);
    auto lhs = candidate.create_tensor(lhs_spec);
    auto rhs = candidate.create_tensor(rhs_spec);
    auto out = candidate.create_tensor(out_spec);
    iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
    iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
    iom_conformance::copy_from_host(out->view(), std::vector<std::byte>(
            out_spec.logical_nbytes(), std::byte{0xAA}));
    auto queue = candidate.create_ops();
    const auto requirements = query_binary_workspace_requirements(
            *queue, operation, lhs->view(), rhs->view(), out->view());
    std::unique_ptr<iom::RawWorkspace> workspace_owner;
    if (requirements.bytes != 0) {
        workspace_owner = candidate.create_workspace(requirements.bytes);
    }
    iom::oid token = 0;
    if (workspace_owner) {
        const iom::RawWorkspaceView workspace = workspace_owner->view();
        token = submit_binary_operation(
                *queue, operation, lhs->view(), rhs->view(), out->view(),
                workspace);
    } else {
        token = submit_binary_operation(
                *queue, operation, lhs->view(), rhs->view(), out->view());
    }
    if (operation == BinaryOperation::div) {
        CHECK_EQ(token, iom::to_oid(iom::OidError::Unsupported));

        const iom::TensorSpec lhs_f32_spec{
                iom::TensorShape{{2, 1, 33}}, iom::DataType::F32};
        const iom::TensorSpec rhs_f32_spec{
                iom::TensorShape{{1, 17, 1}}, iom::DataType::F32};
        const iom::TensorSpec out_f32_spec{
                iom::TensorShape{{2, 17, 33}}, iom::DataType::F32};
        const auto f32_pattern = [](std::size_t linear, std::uint64_t salt) {
            const auto raw = element_pattern(
                    iom::DataType::F32, linear, salt);
            return std::uint64_t{0x3f000000u}
                    | (raw & std::uint64_t{0x007fffffu});
        };
        const std::size_t lhs_f32_count =
                lhs_f32_spec.shape.element_count();
        const std::size_t rhs_f32_count =
                rhs_f32_spec.shape.element_count();
        std::vector<std::byte> lhs_f32_bytes(lhs_f32_spec.logical_nbytes());
        std::vector<std::byte> rhs_f32_bytes(rhs_f32_spec.logical_nbytes());
        for (std::size_t i = 0; i < lhs_f32_count; ++i) {
            gpu_write_bits(
                    lhs_f32_bytes, i, 32, f32_pattern(i, 0x13579BDF));
        }
        for (std::size_t i = 0; i < rhs_f32_count; ++i) {
            gpu_write_bits(
                    rhs_f32_bytes, i, 32, f32_pattern(i, 0x2468ACE0));
        }
        std::vector<std::byte> expected_f32(out_f32_spec.logical_nbytes());
        for (std::size_t plane = 0; plane < 2; ++plane) {
            for (std::size_t row = 0; row < 17; ++row) {
                for (std::size_t column = 0; column < 33; ++column) {
                    const std::size_t output_index =
                            plane * 17 * 33 + row * 33 + column;
                    const auto a = gpu_read_bits(
                            lhs_f32_bytes, plane * 33 + column, 32);
                    const auto b = gpu_read_bits(rhs_f32_bytes, row, 32);
                    gpu_write_bits(
                            expected_f32, output_index, 32,
                            add_oracle::binary(
                                    iom::DataType::F32, a, b,
                                    add_oracle::operation::div));
                }
            }
        }

        auto lhs_f32 = candidate.create_tensor(lhs_f32_spec);
        auto rhs_f32 = candidate.create_tensor(rhs_f32_spec);
        auto out_f32 = candidate.create_tensor(out_f32_spec);
        iom_conformance::copy_from_host(lhs_f32->view(), lhs_f32_bytes);
        iom_conformance::copy_from_host(rhs_f32->view(), rhs_f32_bytes);
        iom_conformance::copy_from_host(out_f32->view(), std::vector<std::byte>(
                expected_f32.size(), std::byte{0xAA}));
        const auto f32_requirements =
                query_binary_workspace_requirements(
                        *queue, operation, lhs_f32->view(), rhs_f32->view(),
                        out_f32->view());
        std::unique_ptr<iom::RawWorkspace> f32_workspace_owner;
        if (f32_requirements.bytes != 0) {
            f32_workspace_owner =
                    candidate.create_workspace(f32_requirements.bytes);
        }
        iom::oid f32_token = 0;
        if (f32_workspace_owner) {
            const iom::RawWorkspaceView workspace =
                    f32_workspace_owner->view();
            f32_token = submit_binary_operation(
                    *queue, operation, lhs_f32->view(), rhs_f32->view(),
                    out_f32->view(), workspace);
        } else {
            f32_token = submit_binary_operation(
                    *queue, operation, lhs_f32->view(), rhs_f32->view(),
                    out_f32->view());
        }
        REQUIRE(iom::oid_is_token(f32_token));
        CHECK_NOTHROW(queue->wait(f32_token));
        const auto actual_f32 = read_logical(out_f32->view());
        const std::size_t output_count =
                out_f32_spec.shape.element_count();
        for (std::size_t i = 0; i < output_count; ++i) {
            const auto got = gpu_read_bits(actual_f32, i, 32);
            const auto want = gpu_read_bits(expected_f32, i, 32);
            CHECK(add_f32_within_ulp(
                    static_cast<std::uint32_t>(got),
                    static_cast<std::uint32_t>(want)));
        }
        return;
    }
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue->wait(token));
    const auto expected_count = out_spec.shape.element_count();
    std::vector<std::byte> expected(out_spec.logical_nbytes());
    for (std::size_t plane = 0; plane < 2; ++plane) {
        for (std::size_t row = 0; row < 17; ++row) {
            for (std::size_t column = 0; column < 33; ++column) {
                const std::size_t output_index =
                        plane * 17 * 33 + row * 33 + column;
                const auto a = gpu_read_bits(
                        lhs_bytes, plane * 33 + column, 8);
                const auto b = gpu_read_bits(rhs_bytes, row, 8);
                gpu_write_bits(
                        expected, output_index, 8,
                        add_oracle::binary(
                                iom::DataType::U8, a, b,
                                static_cast<add_oracle::operation>(operation)));
            }
        }
    }
    const auto actual = read_logical(out->view());
    for (std::size_t i = 0; i < expected_count; ++i) {
        CHECK_EQ(gpu_read_bits(actual, i, 8), gpu_read_bits(expected, i, 8));
    }
}

}  // namespace iom_conformance
