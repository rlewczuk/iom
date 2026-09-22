#pragma once

// Backend-neutral complete-model reference coverage.  The caller owns the
// device and allocator; this harness only materializes the frozen corpus,
// drives the private session seam, and compares completed BF16 boundaries.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "backend_conformance_common.hpp"
#include "iom/device.hpp"
#include "iom/session.hpp"
#include "iom/tensor.hpp"
#include "../../src/session_internal.hpp"
#include "../model/reference_fixture.hpp"

namespace iom_conformance {
namespace model_reference_detail {

[[nodiscard]] inline std::vector<float> read_bf16(
        const iom::TensorView& view) {
    REQUIRE(view.spec().data_type == iom::DataType::BF16);
    const std::size_t elements = view.spec().shape.element_count();
    std::vector<std::byte> bytes(elements * sizeof(std::uint16_t));
    iom_conformance::copy_to_host(view, bytes);
    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes.data() + index * sizeof(bits), sizeof(bits));
        values[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(bits) << 16);
    }
    return values;
}

[[nodiscard]] inline std::vector<std::size_t> record_shape(
        const nlohmann::json& record) {
    return record.at("shape").get<std::vector<std::size_t>>();
}

inline void compare_shape(
        const iom::TensorView& actual, const nlohmann::json& expected,
        std::string_view label) {
    const std::span<const std::size_t> dimensions =
            actual.spec().shape.dimensions();
    const std::vector<std::size_t> actual_shape(
            dimensions.begin(), dimensions.end());
    CHECK_MESSAGE(
            actual_shape == record_shape(expected),
            label << ": logical shape differs from the independent record");
}

inline void compare_tensor(
        const iom::TensorView& actual_view, std::span<const float> actual,
        const nlohmann::json& expected, double atol, double rtol,
        std::string_view label) {
    compare_shape(actual_view, expected, label);
    CHECK_NOTHROW(iom_model_reference::check_values(
            actual, expected, atol, rtol));
}

[[nodiscard]] inline nlohmann::json row_record(
        const nlohmann::json& tensor, std::size_t row) {
    const std::vector<std::size_t> shape = record_shape(tensor);
    REQUIRE(shape.size() == 2);
    const std::size_t rows = shape[0];
    const std::size_t width = shape[1];
    REQUIRE(row < rows);
    const std::vector<double> values =
            tensor.at("values").get<std::vector<double>>();
    const std::size_t offset = row * width;
    REQUIRE(values.size() == rows * width);
    return nlohmann::json{
            {"shape", {1, width}},
            {"values",
             std::vector<double>(
                     values.begin() + static_cast<std::ptrdiff_t>(offset),
                     values.begin()
                             + static_cast<std::ptrdiff_t>(offset + width))}};
}

[[nodiscard]] inline const iom::TensorView& last_layer_view(
        const iom::session_detail::RunBanks& banks,
        std::size_t layer_count) {
    const iom::Tensor* output = (layer_count & 1U) == 0
            ? banks.x.get()
            : banks.residual_after_mlp.get();
    REQUIRE(output != nullptr);
    return output->view();
}

inline void compare_case(
        iom::TinyLlamaSession& session, const nlohmann::json& case_record,
        std::size_t layer_count, double checkpoint_atol,
        double checkpoint_rtol, double logits_atol, double logits_rtol) {
    const nlohmann::json& input = case_record.at("input");
    const std::vector<std::size_t> prompt_ids =
            input.at("prompt_ids").get<std::vector<std::size_t>>();
    REQUIRE(!prompt_ids.empty());
    const nlohmann::json& snapshot = case_record.at("snapshots").at(0);
    const std::size_t run_length = prompt_ids.size();
    INFO(case_record.at("id").get<std::string>());

    iom::session_detail::SessionAccess::prepare_forward_request(
            session, run_length);
    const iom::session_detail::ForwardResult result =
            iom::session_detail::SessionAccess::forward_prefill(
                    session, prompt_ids);
    REQUIRE(result.logits != nullptr);
    REQUIRE(iom::oid_is_token(result.producer));
    iom::session_detail::SessionAccess::wait(session, result.producer);

    // The producer wait is the completion barrier for the full forward.  Read
    // only the final layer output: alternating RunBanks stores may have been
    // reused by an earlier layer, while this view is the completed final one.
    const iom::session_detail::RunBanks& banks =
            iom::session_detail::SessionAccess::prefill(session);
    const iom::TensorView& final_layer =
            last_layer_view(banks, layer_count);
    const std::vector<float> final_layer_values = read_bf16(final_layer);
    compare_tensor(
            final_layer, final_layer_values,
            snapshot.at("last_layer_output"), checkpoint_atol,
            checkpoint_rtol, "final layer residual_after_mlp");

    const std::vector<float> final_norm_values =
            read_bf16(banks.final_norm->view());
    compare_tensor(
            banks.final_norm->view(), final_norm_values,
            snapshot.at("final_norm"), checkpoint_atol, checkpoint_rtol,
            "final norm");

    const nlohmann::json expected_logits = row_record(
            snapshot.at("logits"), run_length - 1);
    const std::vector<float> logits = read_bf16(*result.logits);
    compare_tensor(
            *result.logits, logits, expected_logits, logits_atol, logits_rtol,
            "final logits");
}

}  // namespace model_reference_detail

inline void run_model_reference_conformance(iom::Device& device) {
    using iom_model_reference::SyntheticFixture;
    using iom_model_reference::load_synthetic_corpus;

    const nlohmann::json corpus = load_synthetic_corpus();
    const double checkpoint_atol =
            corpus.at("tolerances").at("checkpoint").at("atol").get<double>();
    const double checkpoint_rtol =
            corpus.at("tolerances").at("checkpoint").at("rtol").get<double>();
    const double logits_atol =
            corpus.at("tolerances").at("logits").at("atol").get<double>();
    const double logits_rtol =
            corpus.at("tolerances").at("logits").at("rtol").get<double>();

    constexpr std::array<std::size_t, 4> kRunLengths{1, 15, 16, 17};
    std::size_t total_cases = 0;
    for (const nlohmann::json& model_record : corpus.at("models")) {
        SyntheticFixture fixture(model_record);
        const std::unique_ptr<iom::TinyLlamaSession> session =
                iom::load_tinyllama_session(fixture.path(), device);
        REQUIRE(session != nullptr);
        const std::string model_id = model_record.at("id").get<std::string>();
        const std::size_t layer_count =
                model_record.at("config").at("num_hidden_layers")
                        .get<std::size_t>();
        std::size_t model_cases = 0;

        for (const nlohmann::json& case_record : corpus.at("cases")) {
            if (case_record.at("model_id") != model_id) continue;
            const nlohmann::json& input = case_record.at("input");
            if (input.at("mode") != "full_prompt") continue;
            const std::size_t run_length =
                    input.at("prompt_ids").size();
            if (std::find(
                        kRunLengths.begin(), kRunLengths.end(), run_length)
                    == kRunLengths.end()) {
                continue;
            }
            ++model_cases;
            ++total_cases;
            model_reference_detail::compare_case(
                    *session, case_record, layer_count, checkpoint_atol,
                    checkpoint_rtol, logits_atol, logits_rtol);
        }
        CHECK_MESSAGE(
                model_cases == kRunLengths.size(),
                model_id << ": expected one complete-model case at each tile boundary");
    }
    CHECK_EQ(total_cases, 2 * kRunLengths.size());
}

}  // namespace iom_conformance
