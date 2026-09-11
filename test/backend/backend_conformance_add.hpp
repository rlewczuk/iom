#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {
namespace add_oracle {
struct Spec { unsigned e, f; int bias; bool inf; bool finite; };
inline Spec spec(iom::DataType t) noexcept {
    switch (t) {
        case iom::DataType::F4_E2M1: return {2, 1, 1, 0, 1};
        case iom::DataType::F6_E2M3: return {2, 3, 1, 0, 1};
        case iom::DataType::F6_E3M2: return {3, 2, 3, 0, 1};
        case iom::DataType::F8_E4M3FN: return {4, 3, 7, 0, 1};
        case iom::DataType::F8_E5M2: return {5, 2, 15, 1, 0};
        case iom::DataType::F16: return {5, 10, 15, 1, 0};
        case iom::DataType::BF16: return {8, 7, 127, 1, 0};
        case iom::DataType::F32: return {8, 23, 127, 1, 0};
        default: return {11, 52, 1023, 1, 0};
    }
}
inline long double value(std::uint64_t r, Spec s) noexcept {
    const auto em = (std::uint64_t{1} << s.e) - 1, fm = (std::uint64_t{1} << s.f) - 1;
    const auto ex = (r >> s.f) & em, fr = r & fm;
    if (ex == em && (!s.finite || s.e == 4)) {
        if (s.inf && fr == 0) return (r >> (s.e + s.f)) ? -INFINITY : INFINITY;
        return std::numeric_limits<long double>::quiet_NaN();
    }
    const long double v = ex
            ? std::ldexp(static_cast<long double>((std::uint64_t{1} << s.f) + fr),
                         static_cast<int>(ex) - s.bias - static_cast<int>(s.f))
            : std::ldexp(static_cast<long double>(fr), 1 - s.bias - static_cast<int>(s.f));
    return (r >> (s.e + s.f)) ? -v : v;
}
inline std::uint64_t round_encode(long double x, Spec s) noexcept {
    const auto em = (std::uint64_t{1} << s.e) - 1, fm = (std::uint64_t{1} << s.f) - 1;
    const auto sg = std::signbit(x) ? std::uint64_t{1} : 0; x = std::fabs(x);
    const auto finite_emax = s.finite && s.e < 4 ? em : em - 1;
    const auto overflow = (sg << (s.e + s.f))
            | (s.inf ? (em << s.f) : (finite_emax << s.f) | fm);
    if (std::isnan(x))
        return (sg << (s.e + s.f)) | (em << s.f) |
               (s.finite ? fm : std::uint64_t{1} << (s.f - 1));
    if (std::isinf(x)) return overflow;
    if (x == 0) return sg << (s.e + s.f);
    int e = 0; std::frexp(x, &e); --e;
    const int minsub = 1 - s.bias - static_cast<int>(s.f);
    const int maxe = static_cast<int>(finite_emax - s.bias);
    auto rn = [](long double q) {
        const auto n = std::floor(q), r = q - n;
        return static_cast<std::uint64_t>(n + (r > .5L || (r == .5L && (static_cast<std::uint64_t>(n) & 1))));
    };
    if (e < minsub + static_cast<int>(s.f)) {
        const auto q = rn(std::ldexp(x, -minsub));
        return (sg << (s.e + s.f)) | (q == (std::uint64_t{1} << s.f) ? std::uint64_t{1} << s.f : q);
    }
    if (e > maxe) return overflow;
    auto q = rn(std::ldexp(x, static_cast<int>(s.f) - e) - (std::uint64_t{1} << s.f));
    if (q == (std::uint64_t{1} << s.f)) { ++e; q = 0; }
    if (e > maxe) return overflow;
    return (sg << (s.e + s.f)) | (static_cast<std::uint64_t>(e + s.bias) << s.f) | q;
}
enum class operation { add, mul, sub, div };
inline std::uint64_t binary(iom::DataType t, std::uint64_t a, std::uint64_t b, operation op) noexcept {
    unsigned w = 0;
    switch (t) {
        case iom::DataType::I2: case iom::DataType::U2: w = 2; break;
        case iom::DataType::I4: case iom::DataType::U4: w = 4; break;
        case iom::DataType::I8: case iom::DataType::U8: w = 8; break;
        case iom::DataType::I16: case iom::DataType::U16: w = 16; break;
        case iom::DataType::I32: case iom::DataType::U32: w = 32; break;
        case iom::DataType::I64: case iom::DataType::U64: w = 64; break;
        default: break;
    }
    if (w) {
        if (op == operation::div) return 0;
        const auto mask = w == 64 ? ~std::uint64_t{} : (std::uint64_t{1} << w) - 1;
        if (op == operation::mul) return (a * b) & mask;
        if (op == operation::sub) return (a - b) & mask;
        return (a + b) & mask;
    }
    const auto s = spec(t);
    const auto x = value(a, s), y = value(b, s);
    long double z;
    if (std::isnan(x) || std::isnan(y)) z = std::numeric_limits<long double>::quiet_NaN();
    else if (op == operation::add) {
        if (std::isinf(x) && std::isinf(y) && std::signbit(x) != std::signbit(y))
            z = std::numeric_limits<long double>::quiet_NaN();
        else if (x == 0 && y == 0) z = (std::signbit(x) && std::signbit(y)) ? -0.0L : 0.0L;
        else { z = x + y; if (z == 0) z = 0.0L; }
    } else if (op == operation::mul) {
        z = ((std::isinf(x) && y == 0) || (std::isinf(y) && x == 0))
            ? std::numeric_limits<long double>::quiet_NaN() : x * y;
    } else if (op == operation::sub) {
        z = (std::isinf(x) && std::isinf(y) && std::signbit(x) == std::signbit(y))
            ? std::numeric_limits<long double>::quiet_NaN() : x - y;
    } else {
        z = ((x == 0 && y == 0) || (std::isinf(x) && std::isinf(y)))
            ? std::numeric_limits<long double>::quiet_NaN() : x / y;
    }
    return round_encode(z, s);
}
inline std::uint64_t add(iom::DataType t, std::uint64_t a, std::uint64_t b) noexcept {
    return binary(t, a, b, operation::add);
}
}  // namespace add_oracle

inline std::uint64_t add_read_bits(const std::vector<std::byte>& bytes, std::size_t index, std::size_t bits) {
    std::uint64_t value = 0;
    for (std::size_t bit = 0; bit < bits; ++bit)
        if ((static_cast<unsigned char>(bytes[(index * bits + bit) / 8]) >> ((index * bits + bit) % 8)) & 1)
            value |= std::uint64_t{1} << bit;
    return value;
}
inline bool add_f32_within_ulp(std::uint32_t actual, std::uint32_t expected) {
    const float a = std::bit_cast<float>(actual), e = std::bit_cast<float>(expected);
    if (std::isnan(e)) return std::isnan(a);
    if (std::isinf(e) || e == 0.0f) return actual == expected;
    if (!std::isfinite(a)) return false;
    const auto ordered = [](std::uint32_t bits) { return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u; };
    const auto oa = ordered(actual), oe = ordered(expected);
    return (oa > oe ? oa - oe : oe - oa) <= 1;
}
inline bool add_is_float(iom::DataType type) {
    switch (type) {
        case iom::DataType::F4_E2M1: case iom::DataType::F6_E2M3:
        case iom::DataType::F6_E3M2: case iom::DataType::F8_E4M3FN:
        case iom::DataType::F8_E5M2: case iom::DataType::F16:
        case iom::DataType::BF16: case iom::DataType::F32:
        case iom::DataType::F64: return true;
        default: return false;
    }
}
inline bool add_non_f32_float_classes(iom::DataType type, std::uint64_t actual, std::uint64_t expected) {
    const auto s = add_oracle::spec(type);
    const auto a = add_oracle::value(actual, s);
    const auto e = add_oracle::value(expected, s);
    if (std::isnan(e)) return std::isnan(a);
    if (std::isinf(e)) return std::isinf(a) && std::signbit(a) == std::signbit(e);
    if (e == 0) return a == 0 && std::signbit(a) == std::signbit(e);
    if (!std::isfinite(a) || std::signbit(a) != std::signbit(e)) return false;
    const auto sign = std::uint64_t{1} << (s.e + s.f);
    const auto ordered = [sign](std::uint64_t bits) {
        return (bits & sign) ? ~bits : bits | sign;
    };
    const auto oa = ordered(actual), oe = ordered(expected);
    return (oa > oe ? oa - oe : oe - oa) <= 1;
}
inline void run_binary_mapping_value_conformance(
        iom::Device& candidate, BinaryOperation operation);
inline void run_binary_transformed_value_conformance(
        iom::Device& candidate, BinaryOperation operation);

inline void run_binary_value_conformance(
        iom::Device& candidate, BinaryOperation operation) {
    const iom::DataType types[] = {
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
        if (operation == BinaryOperation::div && !add_is_float(type))
            continue;
        const iom::TensorSpec spec{
                iom::TensorShape{{2, 17, 33}}, type};
        const std::size_t count = spec.shape.element_count();
        const std::size_t bits = iom::detail::leaf_bits(type);
        std::vector<std::byte> lhs(spec.logical_nbytes());
        std::vector<std::byte> rhs(spec.logical_nbytes());
        std::vector<std::byte> expected(spec.logical_nbytes());
        const auto float_value = [](iom::DataType float_type,
                                    std::size_t index,
                                    std::uint64_t salt) -> std::uint64_t {
            switch (float_type) {
                case iom::DataType::F16: {
                    static constexpr std::uint16_t values[] = {
                            0x3c00, 0x4000};
                    return values[(index + salt) % 2];
                }
                case iom::DataType::BF16: {
                    static constexpr std::uint16_t values[] = {
                            0x3f80, 0x4000};
                    return values[(index + salt) % 2];
                }
                case iom::DataType::F32: {
                    static constexpr std::uint32_t values[] = {
                            0x00000000, 0x80000000, 0x3f800000,
                            0xbf800000, 0x7f7fffff, 0x00800000,
                            0x007fffff, 0x7f800000, 0xff800000,
                            0x7fc00001};
                    return static_cast<std::uint64_t>(
                            values[(index + salt) % (sizeof(values)
                                                     / sizeof(values[0]))]);
                }
                case iom::DataType::F64: {
                    // Keep device arithmetic in the exact finite range for
                    // this cross-backend gate; F64 boundary/special probes
                    // remain covered by the backend-specific suites.
                    static constexpr std::uint64_t values[] = {
                            0x3ff0000000000000ull,
                            0x4000000000000000ull};
                    return values[(index + salt) % (sizeof(values)
                                                   / sizeof(values[0]))];
                }
                case iom::DataType::F4_E2M1: {
                    static constexpr std::uint8_t values[] = {2, 4};
                    return values[(index + salt) % 2];
                }
                case iom::DataType::F6_E2M3: {
                    static constexpr std::uint8_t values[] = {8, 16};
                    return values[(index + salt) % 2];
                }
                case iom::DataType::F6_E3M2: {
                    static constexpr std::uint8_t values[] = {12, 16};
                    return values[(index + salt) % 2];
                }
                case iom::DataType::F8_E4M3FN: {
                    static constexpr std::uint8_t values[] = {0x38, 0x40};
                    return values[(index + salt) % 2];
                }
                case iom::DataType::F8_E5M2: {
                    static constexpr std::uint8_t values[] = {0x3c, 0x40};
                    return values[(index + salt) % 2];
                }
                default:
                    return element_pattern(float_type, index, salt);
            }
        };
        const std::uint64_t integer_mask = bits == 64
                ? ~std::uint64_t{}
                : (std::uint64_t{1} << bits) - 1;
        for (std::size_t i = 0; i < count; ++i) {
            std::uint64_t a;
            std::uint64_t b;
            if (add_is_float(type)) {
                a = float_value(type, i, 0);
                b = float_value(type, i, operation == BinaryOperation::add
                                                ? 3
                                                : 7);
            } else {
                a = (i * 3 + 1) & integer_mask;
                b = (i * 5 + 2) & integer_mask;
            }
            if ((operation == BinaryOperation::sub
                 || operation == BinaryOperation::div)
                && a == b) {
                b = (b + 1) & integer_mask;
            }
            write_bits(reinterpret_cast<unsigned char*>(lhs.data()),
                       i * bits, bits, a);
            write_bits(reinterpret_cast<unsigned char*>(rhs.data()),
                       i * bits, bits, b);
            write_bits(
                    reinterpret_cast<unsigned char*>(expected.data()),
                    i * bits, bits,
                    add_oracle::binary(
                            type, a, b,
                            static_cast<add_oracle::operation>(operation)));
        }
        CAPTURE(static_cast<int>(type));
        CAPTURE(static_cast<int>(operation));
        auto lhs_tensor = candidate.create_tensor(spec);
        auto rhs_tensor = candidate.create_tensor(spec);
        auto out_tensor = candidate.create_tensor(spec);
        lhs_tensor->view().copy_from_host(lhs);
        rhs_tensor->view().copy_from_host(rhs);
        out_tensor->view().copy_from_host(
                std::vector<std::byte>(
                        expected.size(), std::byte{0xAA}));
        const auto requirements = query_binary_workspace_requirements(
                *queue, operation, lhs_tensor->view(), rhs_tensor->view(),
                out_tensor->view());
        std::unique_ptr<iom::RawWorkspace> workspace_owner;
        if (requirements.bytes != 0) {
            workspace_owner = candidate.create_workspace(requirements.bytes);
        }
        iom::oid token = 0;
        if (workspace_owner) {
            const iom::RawWorkspaceView workspace = workspace_owner->view();
            token = submit_binary_operation(
                    *queue, operation, lhs_tensor->view(), rhs_tensor->view(),
                    out_tensor->view(), workspace);
        } else {
            token = submit_binary_operation(
                    *queue, operation, lhs_tensor->view(), rhs_tensor->view(),
                    out_tensor->view());
        }
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        const auto observed = read_logical(out_tensor->view());
        for (std::size_t i = 0; i < count; ++i) {
            const auto actual = add_read_bits(observed, i, bits);
            const auto want = add_read_bits(expected, i, bits);
            CAPTURE(static_cast<std::size_t>(i));
            CAPTURE(actual);
            CAPTURE(want);
            if (type == iom::DataType::F32)
                CHECK(add_f32_within_ulp(
                        static_cast<std::uint32_t>(actual),
                        static_cast<std::uint32_t>(want)));
            else if (add_is_float(type))
                CHECK(add_non_f32_float_classes(type, actual, want));
            else
                CHECK_EQ(actual, want);
        }
    }
    run_binary_mapping_value_conformance(candidate, operation);
    run_binary_transformed_value_conformance(candidate, operation);
}

inline void run_binary_mapping_value_conformance(
        iom::Device& candidate, BinaryOperation operation) {
    const bool division = operation == BinaryOperation::div;
    const iom::DataType type = division ? iom::DataType::F32
                                        : iom::DataType::U8;
    const iom::TensorSpec lhs_spec{
            iom::TensorShape{{2, 1, 33}}, type};
    const iom::TensorSpec rhs_spec{
            iom::TensorShape{{1, 17, 1}}, type};
    const iom::TensorSpec out_spec{
            iom::TensorShape{{2, 17, 33}}, type};
    const std::size_t lhs_count = lhs_spec.shape.element_count();
    const std::size_t rhs_count = rhs_spec.shape.element_count();
    const std::size_t bits = iom::detail::leaf_bits(type);
    std::vector<std::byte> lhs_bytes(lhs_spec.logical_nbytes());
    std::vector<std::byte> rhs_bytes(rhs_spec.logical_nbytes());
    const auto mapped_value = [type](std::size_t index, std::uint64_t salt) {
        if (type == iom::DataType::F32) {
            const auto raw = element_pattern(type, index, salt);
            return std::uint64_t{0x3f000000u}
                    | (raw & std::uint64_t{0x007fffffu});
        }
        return element_pattern(type, index, salt);
    };
    for (std::size_t i = 0; i < lhs_count; ++i)
        write_bits(reinterpret_cast<unsigned char*>(lhs_bytes.data()),
                   i * bits, bits, mapped_value(i, 0x13579BDF));
    for (std::size_t i = 0; i < rhs_count; ++i)
        write_bits(reinterpret_cast<unsigned char*>(rhs_bytes.data()),
                   i * bits, bits, mapped_value(i, 0x2468ACE0));
    std::vector<std::byte> expected(out_spec.logical_nbytes());
    for (std::size_t plane = 0; plane < 2; ++plane) {
        for (std::size_t row = 0; row < 17; ++row) {
            for (std::size_t column = 0; column < 33; ++column) {
                const std::size_t output_index =
                        plane * 17 * 33 + row * 33 + column;
                const auto a = add_read_bits(
                        lhs_bytes, plane * 33 + column, bits);
                const auto b = add_read_bits(rhs_bytes, row, bits);
                write_bits(
                        reinterpret_cast<unsigned char*>(expected.data()),
                        output_index * bits, bits,
                        add_oracle::binary(
                                type, a, b,
                                static_cast<add_oracle::operation>(operation)));
            }
        }
    }
    auto lhs = candidate.create_tensor(lhs_spec);
    auto rhs = candidate.create_tensor(rhs_spec);
    auto out = candidate.create_tensor(out_spec);
    lhs->view().copy_from_host(lhs_bytes);
    rhs->view().copy_from_host(rhs_bytes);
    out->view().copy_from_host(
            std::vector<std::byte>(
                    expected.size(), std::byte{0xAA}));
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
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue->wait(token));
    const auto observed = read_logical(out->view());
    for (std::size_t i = 0; i < out_spec.shape.element_count(); ++i) {
        const auto actual = add_read_bits(observed, i, bits);
        const auto want = add_read_bits(expected, i, bits);
        if (type == iom::DataType::F32)
            CHECK(add_f32_within_ulp(
                    static_cast<std::uint32_t>(actual),
                    static_cast<std::uint32_t>(want)));
        else
            CHECK_EQ(actual, want);
    }
}

inline void run_binary_transformed_value_conformance(
        iom::Device& candidate, BinaryOperation operation) {
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 2, 16, 16}}, iom::DataType::F32};
    const std::size_t elements = spec.shape.element_count();
    std::vector<std::byte> lhs_bytes(spec.logical_nbytes());
    std::vector<std::byte> rhs_bytes(spec.logical_nbytes());
    for (std::size_t i = 0; i < elements; ++i) {
        const std::uint32_t lhs = 0x3F800000u + i;
        const std::uint32_t rhs = 0x40000000u + i;
        std::memcpy(lhs_bytes.data() + i * sizeof(lhs), &lhs, sizeof(lhs));
        std::memcpy(rhs_bytes.data() + i * sizeof(rhs), &rhs, sizeof(rhs));
    }
    auto lhs = candidate.create_tensor(spec);
    auto rhs = candidate.create_tensor(spec);
    auto out = candidate.create_tensor(spec);
    lhs->view().copy_from_host(lhs_bytes);
    rhs->view().copy_from_host(rhs_bytes);
    auto lhs_view = lhs->view().slice(0, 0, 1);
    auto rhs_view = rhs->view().slice(0, 0, 1);
    auto out_view = out->view().slice(0, 0, 1);
    out_view.copy_from_host(
            std::vector<std::byte>(
                    out_view.spec().logical_nbytes(), std::byte{0xAA}));
    auto queue = candidate.create_ops();
    const auto requirements = query_binary_workspace_requirements(
            *queue, operation, lhs_view, rhs_view, out_view);
    std::unique_ptr<iom::RawWorkspace> workspace_owner;
    if (requirements.bytes != 0) {
        workspace_owner = candidate.create_workspace(requirements.bytes);
    }
    iom::oid token = 0;
    if (workspace_owner) {
        const iom::RawWorkspaceView workspace = workspace_owner->view();
        token = submit_binary_operation(
                *queue, operation, lhs_view, rhs_view, out_view, workspace);
    } else {
        token = submit_binary_operation(
                *queue, operation, lhs_view, rhs_view, out_view);
    }
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue->wait(token));
    const auto observed = read_logical(out_view);
    for (std::size_t i = 0; i < 16 * 16; ++i) {
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        std::memcpy(&a, lhs_bytes.data() + i * sizeof(a), sizeof(a));
        std::memcpy(&b, rhs_bytes.data() + i * sizeof(b), sizeof(b));
        const auto want = static_cast<std::uint32_t>(
                add_oracle::binary(
                        iom::DataType::F32, a, b,
                        static_cast<add_oracle::operation>(operation)));
        const auto got = static_cast<std::uint32_t>(
                add_read_bits(observed, i, 32));
        CHECK(add_f32_within_ulp(got, want));
    }
}
}  // namespace iom_conformance
