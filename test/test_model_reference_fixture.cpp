#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/model.hpp"
#include "iom/session.hpp"
#include "../src/session_internal.hpp"
#include "model/reference_fixture.hpp"

namespace {

using iom_model_reference::SyntheticFixture;
using iom_model_reference::check_values;
using iom_model_reference::load_synthetic_corpus;
using iom_model_reference::margin_stable;

using iom::session_detail::SessionAccess;

[[nodiscard]] std::vector<float> read_bf16(const iom::TensorView& view) {
    REQUIRE(view.spec().data_type == iom::DataType::BF16);
    const std::size_t elements = view.spec().shape.element_count();
    std::vector<std::byte> bytes(elements * sizeof(std::uint16_t));
    view.copy_to_host(bytes);
    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes.data() + index * sizeof(bits), sizeof(bits));
        values[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(bits) << 16);
    }
    return values;
}

[[nodiscard]] nlohmann::json row_record(
        const nlohmann::json& tensor, std::size_t row, std::size_t width) {
    const std::vector<double> values =
            tensor.at("values").get<std::vector<double>>();
    const std::size_t offset = row * width;
    return nlohmann::json{
            {"shape", {1, width}},
            {"values",
             std::vector<double>(
                     values.begin() + static_cast<std::ptrdiff_t>(offset),
                     values.begin() + static_cast<std::ptrdiff_t>(offset + width))}};
}

[[nodiscard]] std::size_t lowest_argmax(std::span<const float> values) {
    REQUIRE(!values.empty());
    std::size_t winner = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] > values[winner]) {
            winner = index;
        }
    }
    return winner;
}

void compare_tensor(
        std::span<const float> actual,
        const nlohmann::json& expected,
        double atol,
        double rtol) {
    CHECK_NOTHROW(check_values(actual, expected, atol, rtol));
}

void check_certified_id(
        const nlohmann::json& snapshot,
        const nlohmann::json& expected_result,
        std::size_t position,
        std::span<const float> actual_logits) {
    const std::vector<std::size_t> certified =
            expected_result.at("reference_margin_certified_positions")
                    .get<std::vector<std::size_t>>();
    if (std::find(certified.begin(), certified.end(), position)
            == certified.end()) {
        return;
    }
    const std::size_t expected_id =
            snapshot.at("greedy_id").at(position).get<std::size_t>();
    CHECK(lowest_argmax(actual_logits) == expected_id);
}

[[nodiscard]] const iom::TensorView& last_layer_view(
        const iom::session_detail::RunBanks& banks,
        std::size_t layer_count) {
    return (layer_count & 1U) == 0
            ? banks.x->view()
            : banks.residual_after_mlp->view();
}

void compare_full_case(
        iom::TinyLlamaSession& session,
        const nlohmann::json& case_record,
        std::size_t layer_count,
        double checkpoint_atol,
        double checkpoint_rtol,
        double logits_atol,
        double logits_rtol) {
    const nlohmann::json& input = case_record.at("input");
    const std::vector<std::size_t> prompt =
            input.at("prompt_ids").get<std::vector<std::size_t>>();
    const nlohmann::json& full = case_record.at("snapshots").at(0);
    const nlohmann::json& expected = case_record.at("expected_result");
    INFO(case_record.at("id").get<std::string>() << " full");

    SessionAccess::prepare_forward_request(session, prompt.size());
    const iom::session_detail::ForwardResult result =
            SessionAccess::forward_prefill(session, prompt);
    REQUIRE(result.logits != nullptr);
    SessionAccess::wait(session, result.producer);
    const auto& banks = SessionAccess::prefill(session);
    compare_tensor(
            read_bf16(last_layer_view(banks, layer_count)),
            full.at("last_layer_output"), checkpoint_atol, checkpoint_rtol);
    compare_tensor(
            read_bf16(banks.final_norm->view()),
            full.at("final_norm"), checkpoint_atol, checkpoint_rtol);

    const std::vector<float> final_logits = read_bf16(*result.logits);
    compare_tensor(
            final_logits,
            row_record(full.at("logits"), prompt.size() - 1, 19),
            logits_atol, logits_rtol);
    check_certified_id(
            full, expected, prompt.size() - 1, final_logits);
    for (const auto& cache : SessionAccess::caches(session)) {
        CHECK(cache.initialized_length == prompt.size());
    }

    // The public forward prefill exposes only its final LM-head row. Re-run
    // each causal prefix to consume every full-reference logit row without
    // implementing another recurrence in this test.
    for (std::size_t position = 0; position < prompt.size(); ++position) {
        SessionAccess::prepare_forward_request(session, position + 1);
        const iom::session_detail::ForwardResult row_result =
                SessionAccess::forward_prefill(
                        session, std::span<const std::size_t>(
                                         prompt.data(), position + 1));
        REQUIRE(row_result.logits != nullptr);
        SessionAccess::wait(session, row_result.producer);
        const std::vector<float> row_logits = read_bf16(*row_result.logits);
        compare_tensor(
                row_logits,
                row_record(full.at("logits"), position, 19),
                logits_atol, logits_rtol);
        check_certified_id(full, expected, position, row_logits);
    }
}

void compare_cached_case(
        iom::TinyLlamaSession& session,
        const nlohmann::json& case_record,
        std::size_t layer_count,
        double checkpoint_atol,
        double checkpoint_rtol,
        double logits_atol,
        double logits_rtol) {
    const nlohmann::json& input = case_record.at("input");
    const std::vector<std::size_t> prompt =
            input.at("prompt_ids").get<std::vector<std::size_t>>();
    const nlohmann::json& cached = case_record.at("snapshots").at(1);
    const nlohmann::json& expected = case_record.at("expected_result");
    INFO(case_record.at("id").get<std::string>() << " cached");

    SessionAccess::prepare_forward_request(session, 1);
    for (std::size_t position = 0; position < prompt.size(); ++position) {
        iom::session_detail::ForwardResult result;
        if (position == 0) {
            const std::array<std::size_t, 1> first{prompt.front()};
            result = SessionAccess::forward_prefill(session, first);
        } else {
            result = SessionAccess::forward_decode(session, prompt[position]);
        }
        REQUIRE(result.logits != nullptr);
        SessionAccess::wait(session, result.producer);
        const auto& banks = position == 0
                ? SessionAccess::prefill(session)
                : SessionAccess::decode(session);
        compare_tensor(
                read_bf16(last_layer_view(banks, layer_count)),
                row_record(cached.at("last_layer_output"), position, layer_count == 1 ? 18 : 8),
                checkpoint_atol, checkpoint_rtol);
        compare_tensor(
                read_bf16(banks.final_norm->view()),
                row_record(cached.at("final_norm"), position, layer_count == 1 ? 18 : 8),
                checkpoint_atol, checkpoint_rtol);
        const std::vector<float> row_logits = read_bf16(*result.logits);
        compare_tensor(
                row_logits,
                row_record(cached.at("logits"), position, 19),
                logits_atol, logits_rtol);
        check_certified_id(cached, expected, position, row_logits);
    }
    for (const auto& cache : SessionAccess::caches(session)) {
        CHECK(cache.initialized_length == prompt.size());
    }
}

TEST_CASE("Model reference fixture materializes both pinned models without external dependencies") {
    const nlohmann::json corpus = load_synthetic_corpus();
    REQUIRE(corpus.at("kind") == "synthetic");
    REQUIRE(corpus.at("schema_version") == 1);
    REQUIRE(corpus.at("models").is_array());
    REQUIRE(corpus.at("models").size() == 2);

    for (const nlohmann::json& model_record : corpus.at("models")) {
        // The fixture owns the directory before any source/model owner is
        // created, and remains alive until both owners have drained below.
        SyntheticFixture fixture(model_record);
        const std::unique_ptr<iom::ModelSource> source =
                iom::load_tinyllama_safetensors(fixture.path());
        REQUIRE(source != nullptr);
        CHECK(source->weights().size() == model_record.at("weights").size());
        CHECK(source->config().vocab_size == 19);
        CHECK(source->config().max_position_embeddings == 17);
        CHECK(source->config().bos_token_id == 1);
        CHECK(source->config().eos_token_id == 2);

        alignas(32) std::array<std::byte, 1u << 20> arena{};
        iom::ListAllocator allocator(arena.data(), arena.size());
        const std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
        {
            const std::unique_ptr<iom::TinyLlamaModel> realized =
                    iom::load_tinyllama_model(fixture.path(), *device);
            REQUIRE(realized != nullptr);
            CHECK(realized->weights().size() == model_record.at("weights").size());
            CHECK(realized->config().hidden_size
                  == model_record.at("config").at("hidden_size").get<std::size_t>());
        }
        {
            const std::unique_ptr<iom::TinyLlamaSession> session =
                    iom::load_tinyllama_session(fixture.path(), *device);
            REQUIRE(session != nullptr);
            const std::array<std::size_t, 1> prompt{1};
            const iom::TokenGenerationResult result =
                    session->generate_tokens(prompt, 0);
            CHECK(result.token_ids.empty());
            CHECK(result.stop_reason == iom::GenerationStopReason::max_new_tokens);
        }
        // `fixture` still owns model.safetensors and tokenizer bytes here;
        // destruction of the source/device/model above cannot outlive it.
    }
}

TEST_CASE("Model reference fixture validates every independent corpus forward case") {
    const nlohmann::json corpus = load_synthetic_corpus();
    const double checkpoint_atol =
            corpus.at("tolerances").at("checkpoint").at("atol").get<double>();
    const double checkpoint_rtol =
            corpus.at("tolerances").at("checkpoint").at("rtol").get<double>();
    const double logits_atol =
            corpus.at("tolerances").at("logits").at("atol").get<double>();
    const double logits_rtol =
            corpus.at("tolerances").at("logits").at("rtol").get<double>();
    std::size_t total_cases = 0;

    for (const nlohmann::json& model_record : corpus.at("models")) {
        SyntheticFixture fixture(model_record);
        alignas(32) std::array<std::byte, 1u << 20> arena{};
        iom::ListAllocator allocator(arena.data(), arena.size());
        const std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
        const std::unique_ptr<iom::TinyLlamaSession> session =
                iom::load_tinyllama_session(fixture.path(), *device);
        REQUIRE(session != nullptr);
        const std::string model_id = model_record.at("id").get<std::string>();
        const std::size_t layer_count =
                model_record.at("config").at("num_hidden_layers").get<std::size_t>();
        std::size_t model_cases = 0;

        for (const nlohmann::json& case_record : corpus.at("cases")) {
            if (case_record.at("model_id") != model_id) {
                continue;
            }
            ++model_cases;
            ++total_cases;
            const nlohmann::json& input = case_record.at("input");
            const nlohmann::json& expected = case_record.at("expected_result");
            const std::vector<std::size_t> prefix_ids =
                    input.at("prefix_ids").get<std::vector<std::size_t>>();
            const std::vector<std::size_t> decode_ids =
                    input.at("decode_ids").get<std::vector<std::size_t>>();
            CHECK(expected.at("stop_reason") == "max_new_tokens");
            CHECK(expected.at("initialized_kv_length") == prefix_ids.size());
            CHECK(expected.at("token_ids")
                  == (input.at("mode") == "teacher_forced"
                              ? nlohmann::json(decode_ids)
                              : nlohmann::json::array()));

            compare_full_case(
                    *session, case_record, layer_count, checkpoint_atol,
                    checkpoint_rtol, logits_atol, logits_rtol);
            compare_cached_case(
                    *session, case_record, layer_count, checkpoint_atol,
                    checkpoint_rtol, logits_atol, logits_rtol);
        }
        CHECK(model_cases == 7);
    }
    CHECK(total_cases == 14);
}

TEST_CASE("Model reference fixture rejects malformed model records before materialization") {
    CHECK_THROWS_AS(
            [&] { SyntheticFixture fixture(nlohmann::json::array()); }(),
            std::invalid_argument);
    const nlohmann::json bad_id = {
            {"id", 7},
            {"config", nlohmann::json::object()},
            {"weights", nlohmann::json::array()},
    };
    CHECK_THROWS_AS(
            [&] { SyntheticFixture fixture(bad_id); }(),
            std::invalid_argument);
    const nlohmann::json bad_path_id = {
            {"id", "../unsafe"},
            {"config", nlohmann::json::object()},
            {"weights", nlohmann::json::array()},
    };
    CHECK_THROWS_AS(
            [&] { SyntheticFixture fixture(bad_path_id); }(),
            std::invalid_argument);
}

TEST_CASE("Model reference fixture checks every value and rejects corrupt payloads") {
    const nlohmann::json expected = nlohmann::json{
            {"shape", {2, 2}},
            {"values", {1.0, -2.0, 3.0, 4.0}},
    };
    const std::vector<float> within{1.0F, -2.0F, 3.04F, 3.96F};
    CHECK_NOTHROW(check_values(within, expected, 0.05, 0.02));

    const std::vector<float> wrong_size{1.0F, -2.0F, 3.0F};
    CHECK_THROWS_AS(check_values(wrong_size, expected, 0.05, 0.02),
                    std::invalid_argument);

    const nlohmann::json wrong_shape = nlohmann::json{
            {"shape", {2, 3}},
            {"values", {1.0, -2.0, 3.0, 4.0}},
    };
    CHECK_THROWS_AS(check_values(within, wrong_shape, 0.05, 0.02),
                    std::invalid_argument);

    std::vector<float> nonfinite = within;
    nonfinite[2] = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS_AS(check_values(nonfinite, expected, 0.05, 0.02),
                    std::invalid_argument);

    std::vector<float> one_bad = within;
    one_bad[0] = 1.2F;
    CHECK_THROWS_AS(check_values(one_bad, expected, 0.05, 0.02),
                    std::runtime_error);
}

TEST_CASE("Model reference fixture applies strict lowest-ID margin decisions") {
    const std::vector<float> stable{1.0F, 0.0F, -0.5F};
    CHECK(margin_stable(stable, 0.05, 0.02));

    // The lower ID wins an exact value tie for selection, but the frozen
    // interval rule correctly refuses to certify that winner.
    const std::vector<float> tie{1.0F, 1.0F, 0.0F};
    CHECK_FALSE(margin_stable(tie, 0.05, 0.02));

    const std::vector<float> near_tie{1.0F, 0.91F, 0.0F};
    CHECK_FALSE(margin_stable(near_tie, 0.05, 0.02));

    const std::vector<float> nonfinite{
            1.0F, std::numeric_limits<float>::infinity(), 0.0F};
    CHECK_FALSE(margin_stable(nonfinite, 0.05, 0.02));
}

}  // namespace
