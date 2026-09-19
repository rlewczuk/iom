
#include <doctest/doctest.h>

#include "backend/backend_conformance_sdpa_reference.hpp"

TEST_CASE("SDPA reference fixes the causal GQA numerical contract") {
    const iom_conformance::SdpaReferenceSelfCheckReport report =
            iom_conformance::sdpa_reference_self_check();
    INFO("SDPA reference checks executed: " << report.checks);
    for (const std::string& failure : report.failures) {
        INFO(failure);
        CHECK(false);
    }
    CHECK(report.ok());
    CHECK(report.checks > 0);
}

TEST_CASE("SDPA reference rejects a deliberately perturbed expectation") {
    const iom::DataType type = iom::DataType::BF16;
    const iom_conformance::SdpaReferenceCase reference_case =
            iom_conformance::sdpa_oracle::make_ones_case(type);
    const std::vector<iom_conformance::SdpaReferenceValue> expected =
            iom_conformance::evaluate(reference_case);
    REQUIRE_FALSE(expected.empty());
    iom_conformance::SdpaReferenceValue perturbed = expected.front();
    perturbed.bits = iom_conformance::sdpa_oracle::value_bits(type, 1.0 + 3.0 * 0x1p-7);
    CHECK_FALSE(iom_conformance::matches(type, perturbed.bits, expected.front()));
}
