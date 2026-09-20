#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "backend_conformance_rope.hpp"

namespace {

using namespace iom_conformance::rope_reference;

TEST_CASE("RoPE reference self-check is backend-free and self-falsifying") {
    const RopeReferenceSelfCheckReport report = rope_reference_self_check();
    for (const std::string& failure : report.failures) {
        INFO(failure);
        CHECK(false);
    }
    CHECK(report.checks > 0);
    CHECK(report.ok());
}

TEST_CASE("RoPE reference classifies every data type and support span") {
    CHECK(kRopeApplicableDataTypes.size() == 9);
    CHECK(kRopeUnsupportedDataTypes.size() == 14);
    CHECK(kRopeSyclExpectedSupported.size() == 8);
    CHECK(kRopeSyclExpectedSupported.back() == iom::DataType::F32);

    for (const iom::DataType type : kRopeApplicableDataTypes) {
        CHECK(rope_data_type_classification(type)
              == RopeDataTypeClassification::applicable);
    }
    for (const iom::DataType type : kRopeUnsupportedDataTypes) {
        CHECK(rope_data_type_classification(type)
              == RopeDataTypeClassification::unsupported);
        CHECK(rope_reference_cases(type).empty());
    }
}

TEST_CASE("RoPE position zero preserves raw signed zero and nonfinite payloads") {
    for (const iom::DataType type : kRopeApplicableDataTypes) {
        const auto cases = rope_reference_cases(type);
        const auto found = std::find_if(
                cases.begin(), cases.end(), [](const RopeReferenceCase& value) {
                    return value.kind == RopeReferenceCaseKind::position_zero;
                });
        REQUIRE(found != cases.end());
        const std::vector<RopeReferenceValue> values = evaluate(*found);
        REQUIRE(values.size() == found->input_bits.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            CHECK(values[index].bits == found->input_bits[index]);
            CHECK(matches(type, values[index].bits, values[index]));
        }
    }
}

TEST_CASE("RoPE reference uses split-half coefficients and absolute positions") {
    RopeReferenceCase reference_case;
    reference_case.kind = RopeReferenceCaseKind::theta_one;
    reference_case.data_type = iom::DataType::F32;
    reference_case.H = 1;
    reference_case.R = 3;
    reference_case.D = 4;
    reference_case.a = 15;
    reference_case.theta = 1.0;
    reference_case.input_bits = {
            0x3f800000, 0x40000000, 0x40400000, 0x40800000,
            0x3f800000, 0x40000000, 0x40400000, 0x40800000,
            0x3f800000, 0x40000000, 0x40400000, 0x40800000,
    };
    const auto values = evaluate(reference_case);
    REQUIRE(values.size() == reference_case.input_bits.size());
    // The first pair is (x[0], x[2]), not the adjacent pair (x[0], x[1]).
    const float expected_first = 1.0F * std::cos(15.0F)
                                 - 3.0F * std::sin(15.0F);
    CHECK(values[0].decoded
          == doctest::Approx(static_cast<double>(expected_first)).epsilon(1e-6));
    const float reset_first = 1.0F * std::cos(0.0F)
                              - 3.0F * std::sin(0.0F);
    CHECK(values[4].decoded
          != doctest::Approx(static_cast<double>(reset_first)).epsilon(1e-6));
}

TEST_CASE("RoPE comparison keeps special classes and frozen destination ULP") {
    RopeReferenceValue one;
    one.bits = 0x3f800000;
    one.decoded = 1.0;
    one.value_class = RopeReferenceClass::finite;
    CHECK(matches(iom::DataType::F32, 0x3f800001, one));
    CHECK_FALSE(matches(iom::DataType::F32, 0x3f800002, one));

    RopeReferenceValue nan;
    nan.bits = 0x7fc00001;
    nan.value_class = RopeReferenceClass::quiet_nan;
    CHECK(matches(iom::DataType::F32, 0x7fc01234, nan));
    CHECK_FALSE(matches(iom::DataType::F32, 0x3f800000, nan));

    RopeReferenceValue negative_zero;
    negative_zero.bits = 0x80000000;
    negative_zero.value_class = RopeReferenceClass::negative_zero;
    CHECK(matches(iom::DataType::F32, 0x80000000, negative_zero));
    CHECK_FALSE(matches(iom::DataType::F32, 0, negative_zero));
}

}  // namespace
