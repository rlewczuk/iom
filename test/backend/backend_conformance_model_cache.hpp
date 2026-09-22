#pragma once

// Backend-neutral full-model cache/reference coverage.  Backend setup stays in
// the owning test or driver; this harness only consumes an already-created
// Device and the frozen synthetic model corpus.

#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "iom/model.hpp"
#include "iom/session.hpp"
#include "../../src/session_internal.hpp"
#include "../model/reference_fixture.hpp"

namespace iom_conformance {
namespace model_cache_detail {

using iom_model_reference::SyntheticFixture;
using iom_model_reference::check_values;
using iom_model_reference::load_synthetic_corpus;

inline constexpr double kAtol = 0.05;
inline constexpr double kRtol = 0.02;
inline constexpr std::size_t kCapacity = 17;
inline constexpr std::size_t kInvarianceCutoff = 15;

[[nodiscard]] inline std::vector<float> read_bf16(const iom::TensorView& view) {
    REQUIRE(view.spec().data_type == iom::DataType::BF16);
    const std::size_t elements = view.spec().shape.element_count();
    std::vector<std::byte> bytes(elements * sizeof(std::uint16_t));
    const iom::WorkspaceRequirements requirement =
            view.copy_to_host_workspace_requirements();
    if (requirement.bytes == 0) {
        view.copy_to_host(bytes);
    } else {
        auto workspace = const_cast<iom::Device&>(view.device())
                                 .create_workspace(requirement.bytes);
        REQUIRE(workspace != nullptr);
        view.copy_to_host(bytes, workspace->view());
    }

    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes.data() + index * sizeof(bits), sizeof(bits));
        values[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(bits) << 16);
    }
    return values;
}

[[nodiscard]] inline nlohmann::json row_record(
        const nlohmann::json& tensor, std::size_t row) {
    const std::vector<std::size_t> shape =
            tensor.at("shape").get<std::vector<std::size_t>>();
    if (shape.size() != 2 || row >= shape[0]) {
        throw std::invalid_argument("synthetic reference row is out of range");
    }
    const std::vector<double> values =
            tensor.at("values").get<std::vector<double>>();
    const std::size_t width = shape[1];
    const std::size_t offset = row * width;
    if (offset > values.size() || width > values.size() - offset) {
        throw std::invalid_argument(
                "synthetic reference row exceeds tensor payload");
    }
    return nlohmann::json{
            {"shape", {1, width}},
            {"values", std::vector<double>(
                               values.begin()
                                       + static_cast<std::ptrdiff_t>(offset),
                               values.begin() + static_cast<std::ptrdiff_t>(
                                                        offset + width))}};
}

inline void compare_values(
        std::span<const float> actual, const nlohmann::json& expected,
        std::string_view label) {
    INFO(label);
    CHECK_NOTHROW(check_values(actual, expected, kAtol, kRtol));
}

inline void compare_row(
        std::span<const float> actual, const nlohmann::json& tensor,
        std::size_t row, std::string_view label) {
    compare_values(actual, row_record(tensor, row), label);
}

inline void compare_vector_rows(
        std::span<const float> actual, const nlohmann::json& tensor,
        std::size_t rows, std::size_t expected_start,
        std::string_view label) {
    const std::vector<std::size_t> shape =
            tensor.at("shape").get<std::vector<std::size_t>>();
    REQUIRE_EQ(shape.size(), std::size_t{2});
    REQUIRE_LE(expected_start, shape[0]);
    REQUIRE_LE(rows, shape[0] - expected_start);
    REQUIRE_EQ(actual.size(), rows * shape[1]);
    for (std::size_t row = 0; row < rows; ++row) {
        compare_row(
                actual.subspan(row * shape[1], shape[1]), tensor,
                expected_start + row,
                std::string(label) + " row " + std::to_string(expected_start + row));
    }
}

inline void compare_equal_values(
        std::span<const float> lhs, std::span<const float> rhs,
        std::string_view label) {
    REQUIRE_EQ(lhs.size(), rhs.size());
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        const float left = lhs[index];
        const float right = rhs[index];
        CHECK_MESSAGE(
                std::isfinite(left),
                label << ": base value is non-finite at index " << index);
        CHECK_MESSAGE(
                std::isfinite(right),
                label << ": future value is non-finite at index " << index);
        const double bound = kAtol + kRtol * std::abs(static_cast<double>(left));
        CHECK_MESSAGE(
                std::abs(static_cast<double>(right) - left) <= bound,
                label << ": mismatch at index " << index << " (bound "
                      << bound << ")");
    }
}

[[nodiscard]] inline const iom::TensorView& last_layer_view(
        const iom::session_detail::RunBanks& banks,
        std::size_t layer_count) {
    return (layer_count & 1U) == 0
            ? banks.x->view()
            : banks.residual_after_mlp->view();
}

inline void check_cache_lengths(
        iom::TinyLlamaSession& session, std::size_t expected,
        std::string_view label) {
    const std::span<iom::session_detail::CacheOwner> caches =
            iom::session_detail::SessionAccess::caches(session);
    REQUIRE_FALSE(caches.empty());
    for (std::size_t layer = 0; layer < caches.size(); ++layer) {
        CHECK_MESSAGE(
                caches[layer].initialized_length == expected,
                label << ": layer " << layer << " initialized length");
    }
}

inline void compare_banks(
        const iom::session_detail::RunBanks& banks,
        std::size_t layer_count, const nlohmann::json& snapshot,
        std::size_t rows, std::size_t expected_start,
        std::string_view label) {
    compare_vector_rows(
            read_bf16(last_layer_view(banks, layer_count)),
            snapshot.at("last_layer_output"), rows, expected_start,
            std::string(label) + " final-layer residual");
    compare_vector_rows(
            read_bf16(banks.final_norm->view()), snapshot.at("final_norm"),
            rows, expected_start, std::string(label) + " final norm");
}

inline void compare_logits(
        const iom::TensorView& logits, const nlohmann::json& snapshot,
        std::size_t position, std::string_view label) {
    const std::vector<float> values = read_bf16(logits);
    compare_row(
            values, snapshot.at("logits"), position,
            std::string(label) + " logits row " + std::to_string(position));
}

[[nodiscard]] inline const nlohmann::json& case_for(
        const nlohmann::json& corpus, std::string_view model_id,
        std::string_view suffix) {
    const std::string expected_id =
            std::string(model_id) + "-" + std::string(suffix);
    for (const nlohmann::json& item : corpus.at("cases")) {
        if (item.at("id").get<std::string>() == expected_id) return item;
    }
    throw std::invalid_argument("synthetic reference case is missing: " + expected_id);
}

[[nodiscard]] inline std::vector<float> fresh_full_logits(
        iom::Device& device, const std::filesystem::path& model_directory,
        std::span<const std::size_t> tokens) {
    std::unique_ptr<iom::TinyLlamaSession> session =
            iom::load_tinyllama_session(model_directory, device);
    REQUIRE(session != nullptr);
    iom::session_detail::SessionAccess::prepare_forward_request(
            *session, tokens.size());
    const iom::session_detail::ForwardResult result =
            iom::session_detail::SessionAccess::forward_prefill(*session, tokens);
    REQUIRE(result.logits != nullptr);
    iom::session_detail::SessionAccess::wait(*session, result.producer);
    return read_bf16(*result.logits);
}

struct FullPromptObservation {
    std::vector<float> residual;
    std::vector<float> final_norm;
    std::vector<float> logits;
};

[[nodiscard]] inline FullPromptObservation run_full_prompt(
        iom::Device& device, const SyntheticFixture& fixture,
        const nlohmann::json& case_record) {
    const std::vector<std::size_t> prompt =
            case_record.at("input").at("prompt_ids")
                    .get<std::vector<std::size_t>>();
    std::unique_ptr<iom::TinyLlamaSession> session =
            iom::load_tinyllama_session(fixture.path(), device);
    REQUIRE(session != nullptr);
    const std::size_t layer_count = session->config().num_hidden_layers;
    iom::session_detail::SessionAccess::prepare_forward_request(
            *session, prompt.size());
    const iom::session_detail::ForwardResult result =
            iom::session_detail::SessionAccess::forward_prefill(
                    *session, prompt);
    REQUIRE(result.logits != nullptr);
    iom::session_detail::SessionAccess::wait(*session, result.producer);
    check_cache_lengths(*session, prompt.size(), "full prompt");

    const auto& banks = iom::session_detail::SessionAccess::prefill(*session);
    FullPromptObservation observation{
            read_bf16(last_layer_view(banks, layer_count)),
            read_bf16(banks.final_norm->view()), read_bf16(*result.logits)};
    return observation;
}

inline void run_future_invariance(
        iom::Device& device, const SyntheticFixture& fixture,
        const nlohmann::json& base_case, const nlohmann::json& future_case) {
    const std::vector<std::size_t> base_prompt =
            base_case.at("input").at("prompt_ids")
                    .get<std::vector<std::size_t>>();
    const std::vector<std::size_t> future_prompt =
            future_case.at("input").at("prompt_ids")
                    .get<std::vector<std::size_t>>();
    REQUIRE_EQ(base_prompt.size(), kCapacity);
    REQUIRE_EQ(future_prompt.size(), kCapacity);
    REQUIRE_NE(base_prompt[15], future_prompt[15]);
    for (std::size_t index = 0; index < kInvarianceCutoff; ++index) {
        CHECK_EQ(base_prompt[index], future_prompt[index]);
    }

    const FullPromptObservation base =
            run_full_prompt(device, fixture, base_case);
    const FullPromptObservation future =
            run_full_prompt(device, fixture, future_case);
    const nlohmann::json& base_snapshot = base_case.at("snapshots").at(0);
    const nlohmann::json& future_snapshot = future_case.at("snapshots").at(0);

    compare_vector_rows(
            base.residual, base_snapshot.at("last_layer_output"), kCapacity, 0,
            "base full prompt residual");
    compare_vector_rows(
            future.residual, future_snapshot.at("last_layer_output"), kCapacity,
            0, "future full prompt residual");
    compare_vector_rows(
            base.final_norm, base_snapshot.at("final_norm"), kCapacity, 0,
            "base full prompt final norm");
    compare_vector_rows(
            future.final_norm, future_snapshot.at("final_norm"), kCapacity, 0,
            "future full prompt final norm");
    compare_row(
            base.logits, base_snapshot.at("logits"), kCapacity - 1,
            "base terminal");
    compare_row(
            future.logits, future_snapshot.at("logits"), kCapacity - 1,
            "future terminal");

    const std::vector<std::size_t> residual_shape =
            base_snapshot.at("last_layer_output")
                    .at("shape")
                    .get<std::vector<std::size_t>>();
    REQUIRE_EQ(residual_shape.size(), std::size_t{2});
    const std::size_t hidden = residual_shape[1];
    REQUIRE_EQ(base.residual.size(), kCapacity * hidden);
    REQUIRE_EQ(future.residual.size(), kCapacity * hidden);
    for (std::size_t row = 0; row < kInvarianceCutoff; ++row) {
        compare_equal_values(
                std::span<const float>(
                        base.residual.data() + row * hidden, hidden),
                std::span<const float>(
                        future.residual.data() + row * hidden, hidden),
                "future perturbation pre-15 residual row "
                        + std::to_string(row));
    }
}

inline void run_cached_case(
        iom::Device& device, const SyntheticFixture& fixture,
        const nlohmann::json& case_record) {
    const nlohmann::json& input = case_record.at("input");
    const std::vector<std::size_t> prefix =
            input.at("prefix_ids").get<std::vector<std::size_t>>();
    const std::vector<std::size_t> decode =
            input.at("decode_ids").get<std::vector<std::size_t>>();
    const std::vector<std::size_t> prompt =
            input.at("prompt_ids").get<std::vector<std::size_t>>();
    REQUIRE_EQ(prompt.size(), prefix.size() + decode.size());
    for (std::size_t index = 0; index < prefix.size(); ++index) {
        CHECK_EQ(prompt[index], prefix[index]);
    }
    for (std::size_t index = 0; index < decode.size(); ++index) {
        CHECK_EQ(prompt[prefix.size() + index], decode[index]);
    }

    std::unique_ptr<iom::TinyLlamaSession> cached =
            iom::load_tinyllama_session(fixture.path(), device);
    REQUIRE(cached != nullptr);
    const std::size_t layer_count = cached->config().num_hidden_layers;
    const nlohmann::json& cached_snapshot =
            case_record.at("snapshots").at(1);
    const nlohmann::json& full_snapshot = case_record.at("snapshots").at(0);

    iom::session_detail::SessionAccess::prepare_forward_request(
            *cached, prefix.size());
    const iom::session_detail::ForwardResult prefill =
            iom::session_detail::SessionAccess::forward_prefill(*cached, prefix);
    REQUIRE(prefill.logits != nullptr);
    iom::session_detail::SessionAccess::wait(*cached, prefill.producer);
    check_cache_lengths(*cached, prefix.size(), "cached prefill");
    compare_banks(
            iom::session_detail::SessionAccess::prefill(*cached), layer_count,
            cached_snapshot, prefix.size(), 0, "cached prefill");
    const std::size_t prefill_position = prefix.size() - 1;
    compare_logits(*prefill.logits, cached_snapshot, prefill_position,
                   "cached prefill");
    const std::vector<float> prefill_full = fresh_full_logits(
            device, fixture.path(),
            std::span<const std::size_t>(prompt.data(), prefix.size()));
    compare_values(
            prefill_full,
            row_record(full_snapshot.at("logits"), prefill_position),
            "fresh full prefill logits row "
                    + std::to_string(prefill_position));

    for (std::size_t step = 0; step < decode.size(); ++step) {
        const iom::session_detail::ForwardResult result =
                iom::session_detail::SessionAccess::forward_decode(
                        *cached, decode[step]);
        REQUIRE(result.logits != nullptr);
        iom::session_detail::SessionAccess::wait(*cached, result.producer);
        const std::size_t position = prefix.size() + step;
        check_cache_lengths(*cached, position + 1, "cached decode");
        compare_banks(
                iom::session_detail::SessionAccess::decode(*cached),
                layer_count, cached_snapshot, 1, position,
                "cached decode");
        compare_logits(*result.logits, cached_snapshot, position, "cached decode");

        const std::vector<float> fresh = fresh_full_logits(
                device, fixture.path(),
                std::span<const std::size_t>(prompt.data(), position + 1));
        compare_values(
                fresh, row_record(full_snapshot.at("logits"), position),
                "fresh full decode logits row " + std::to_string(position));
    }
}

inline void run_capacity_rejection(
        iom::Device& device, const SyntheticFixture& fixture,
        const nlohmann::json& full_case) {
    const std::vector<std::size_t> prompt =
            full_case.at("input").at("prompt_ids")
                    .get<std::vector<std::size_t>>();
    REQUIRE_EQ(prompt.size(), kCapacity);
    std::unique_ptr<iom::TinyLlamaSession> session =
            iom::load_tinyllama_session(fixture.path(), device);
    REQUIRE(session != nullptr);
    iom::session_detail::SessionAccess::prepare_forward_request(
            *session, kCapacity);
    const iom::session_detail::ForwardResult result =
            iom::session_detail::SessionAccess::forward_prefill(*session, prompt);
    REQUIRE(result.logits != nullptr);
    iom::session_detail::SessionAccess::wait(*session, result.producer);
    const std::vector<float> before = read_bf16(*result.logits);
    check_cache_lengths(*session, kCapacity, "capacity before rejection");

    CHECK_THROWS_AS(
            iom::session_detail::SessionAccess::forward_decode(*session, 4),
            std::invalid_argument);
    CHECK_FALSE(session->poisoned());
    check_cache_lengths(*session, kCapacity, "after capacity decode rejection");
    CHECK(read_bf16(*result.logits) == before);

    CHECK_THROWS_AS(
            iom::session_detail::SessionAccess::prepare_forward_request(
                    *session, kCapacity + 1),
            std::invalid_argument);
    CHECK_FALSE(session->poisoned());
    check_cache_lengths(*session, kCapacity, "after overlong prompt rejection");
    CHECK(read_bf16(*result.logits) == before);
}


}  // namespace model_cache_detail

inline void run_model_cache_reference_conformance(iom::Device& device) {
    using namespace model_cache_detail;
    const nlohmann::json corpus = load_synthetic_corpus();
    REQUIRE(corpus.at("kind") == "synthetic");
    REQUIRE(corpus.at("schema_version") == 1);
    REQUIRE(corpus.at("models").is_array());
    REQUIRE_EQ(corpus.at("models").size(), std::size_t{2});

    std::size_t model_count = 0;
    for (const nlohmann::json& model_record : corpus.at("models")) {
        const std::string model_id = model_record.at("id").get<std::string>();
        SyntheticFixture fixture(model_record);
        const nlohmann::json& base_case =
                case_for(corpus, model_id, "full-prefix-17");
        const nlohmann::json& future_case =
                case_for(corpus, model_id, "future-perturbation");
        const nlohmann::json& prefix_one_case =
                case_for(corpus, model_id, "teacher-forced-from-1");
        const nlohmann::json& prefix_fourteen_case =
                case_for(corpus, model_id, "teacher-forced-from-14");

        CHECK_EQ(
                model_record.at("config")
                        .at("max_position_embeddings")
                        .get<std::size_t>(),
                kCapacity);
        CHECK(
                model_record.at("config").at("hidden_size").get<std::size_t>()
                > 0);
        CHECK(
                model_record.at("config").at("num_attention_heads").get<std::size_t>()
                % model_record.at("config")
                                  .at("num_key_value_heads")
                                  .get<std::size_t>()
                == 0);

        run_future_invariance(device, fixture, base_case, future_case);
        run_cached_case(device, fixture, prefix_one_case);
        run_cached_case(device, fixture, prefix_fourteen_case);
        run_capacity_rejection(device, fixture, base_case);
        ++model_count;
    }
    CHECK_EQ(model_count, std::size_t{2});
}

}  // namespace iom_conformance
