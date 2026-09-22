#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/chat_format.hpp"
#include "iom/inference_metrics.hpp"
#include "iom/session.hpp"
#include "iom/tokenizer.hpp"
#include "../src/session_internal.hpp"
#include "inference_metrics_fixture.hpp"
#include "backend/backend_conformance_model_official.hpp"
#include "model/reference_fixture.hpp"

namespace {

using iom_model_reference::SyntheticFixture;
using iom_model_reference::load_synthetic_corpus;

const nlohmann::json& corpus_model(
        const nlohmann::json& corpus, std::string_view id) {
    const auto& models = corpus.at("models");
    const auto found = std::find_if(
            models.begin(), models.end(), [&](const nlohmann::json& model) {
                return model.at("id") == id;
            });
    if (found == models.end()) {
        throw std::runtime_error("synthetic corpus model is missing");
    }
    return *found;
}

const nlohmann::json& corpus_case(
        const nlohmann::json& corpus, std::string_view id) {
    const auto& cases = corpus.at("cases");
    const auto found = std::find_if(
            cases.begin(), cases.end(), [&](const nlohmann::json& test_case) {
                return test_case.at("id") == id;
            });
    if (found == cases.end()) {
        throw std::runtime_error("synthetic corpus case is missing");
    }
    return *found;
}

std::vector<std::size_t> size_ids(const nlohmann::json& values) {
    return values.get<std::vector<std::size_t>>();
}

iom_conformance::official_model_detail::OfficialCase
make_synthetic_official_case(
        const nlohmann::json& model, const nlohmann::json& source) {
    using namespace iom_conformance::official_model_detail;
    const auto& config = model.at("config");
    const auto& input = source.at("input");
    const std::vector<std::size_t> prompt = size_ids(input.at("prefix_ids"));
    const std::vector<std::size_t> decode = size_ids(input.at("decode_ids"));
    const std::size_t vocabulary =
            config.at("vocab_size").get<std::size_t>();
    const std::size_t layers =
            config.at("num_hidden_layers").get<std::size_t>();
    const auto& full_logits =
            source.at("snapshots").at(0).at("logits");
    CHECK(full_logits.at("shape").at(1) == vocabulary);
    const std::vector<float> values =
            full_logits.at("values").get<std::vector<float>>();
    const std::size_t first_row = prompt.size() - 1;
    REQUIRE(
            values.size()
            >= (first_row + decode.size()) * vocabulary);

    nlohmann::json snapshots = nlohmann::json::array();
    std::vector<std::size_t> prefix = prompt;
    for (std::size_t index = 0; index < decode.size(); ++index) {
        const auto begin = values.begin()
                + static_cast<std::ptrdiff_t>(
                        (first_row + index) * vocabulary);
        const std::vector<float> row(
                begin, begin + static_cast<std::ptrdiff_t>(vocabulary));
        snapshots.push_back({
                {"prefix_ids", prefix},
                {"logits", row},
                {"margin_stable",
                 reference_margin_stable(row, 0.05F, 0.02F)}});
        prefix.push_back(decode[index]);
    }

    std::vector<std::size_t> positions(prompt.size());
    std::iota(positions.begin(), positions.end(), 0);
    OfficialCase result;
    result.name = "synthetic-common-runner";
    result.record = {
            {"input", {{"positions", positions}}},
            {"snapshots", std::move(snapshots)},
            {"expected_result",
             {{"token_ids", decode},
              {"stop_reason", "max_new_tokens"},
              {"initialized_kv_length",
               prompt.size() + decode.size() - 1}}}};
    result.prompt = prompt;
    result.decode = decode;
    result.limit = decode.size();
    result.mode = "tokens";
    result.policy = "fixed-reference-continuation";
    result.layers = layers;
    result.vocabulary = vocabulary;
    result.absolute_tolerance = 0.05F;
    result.relative_tolerance = 0.02F;
    return result;
}

std::vector<std::size_t> widen_ids(
        std::span<const std::uint32_t> values) {
    return {values.begin(), values.end()};
}

nlohmann::json byte_values(std::string_view text) {
    nlohmann::json result = nlohmann::json::array();
    for (const unsigned char byte : text) result.push_back(byte);
    return result;
}

nlohmann::json official_snapshot(
        const std::vector<std::size_t>& prefix, std::size_t selected,
        std::size_t index) {
    std::vector<float> logits(
            iom_conformance::official_model_detail::kVocabulary, 0.0F);
    logits.front() = 2.0F;
    const std::size_t position_start =
            index == 0 ? 0 : prefix.size() - 1;
    const std::size_t run_length =
            index == 0 ? prefix.size() : 1;
    std::vector<std::size_t> absolute_positions(run_length);
    std::iota(
            absolute_positions.begin(), absolute_positions.end(),
            position_start);
    return {
            {"phase", index == 0 ? "prefill" : "cached-decode"},
            {"prefix_ids", prefix},
            {"position_start", position_start},
            {"run_length", run_length},
            {"absolute_positions", absolute_positions},
            {"logits", logits},
            {"greedy_id", std::size_t{0}},
            {"selected_id", selected},
            {"margin_stable", true}};
}

nlohmann::json official_reference_case(
        std::string name, std::string mode, std::string policy,
        std::vector<std::size_t> prompt,
        std::vector<std::size_t> decode) {
    const std::string rendered = mode == "raw"
            ? "The capital of France is"
            : "<|user|>\nHello.</s>\n<|assistant|>\n";
    std::vector<std::size_t> positions(prompt.size());
    std::iota(positions.begin(), positions.end(), 0);
    nlohmann::json input = {
            {"mode", mode},
            {"selection_policy", policy},
            {"prompt_ids", prompt},
            {"rendered_utf8", byte_values(rendered)},
            {"positions", positions},
            {"decode_ids", decode},
            {"max_new_tokens",
             decode.empty() ? std::size_t{0} : std::size_t{4}},
            {"bos_policy",
             {{"add_special_tokens", mode == "raw"},
              {"require_bos", mode == "raw"},
              {"bos_token_id", std::size_t{1}}}}};
    if (mode == "raw") {
        input["text"] = "The capital of France is";
    } else {
        input["messages"] =
                nlohmann::json::array(
                        {{{"role", "user"}, {"content", "Hello."}}});
    }
    nlohmann::json snapshots = nlohmann::json::array();
    std::vector<std::size_t> prefix = prompt;
    for (std::size_t index = 0; index < decode.size(); ++index) {
        snapshots.push_back(
                official_snapshot(prefix, decode[index], index));
        prefix.push_back(decode[index]);
    }
    return {
            {"id", std::move(name)},
            {"input", std::move(input)},
            {"snapshots", std::move(snapshots)},
            {"expected_result",
             {{"token_ids", decode},
              {"stop_reason", "max_new_tokens"},
              {"initialized_kv_length",
               decode.empty()
                       ? std::size_t{0}
                       : prompt.size() + decode.size() - 1}}}};
}

nlohmann::json complete_official_reference_pack() {
    using namespace iom_conformance::official_model_detail;
    const std::string artifact_id = "official-pack-regression";
    const nlohmann::json model_config = {
            {"num_hidden_layers", kLayers},
            {"hidden_size", kHidden},
            {"intermediate_size", kIntermediate},
            {"num_attention_heads", kQueryHeads},
            {"num_key_value_heads", kKeyValueHeads},
            {"head_dim", kHeadDim},
            {"vocab_size", kVocabulary},
            {"max_position_embeddings", kContext},
            {"model_type", "llama"},
            {"tie_word_embeddings", false},
            {"torch_dtype", "bfloat16"},
            {"model_id", "TinyLlama/reference-pack-regression"}};
    const std::vector<std::size_t> raw_prompt{1, 10};
    const std::vector<std::size_t> chat_prompt{10};
    const std::vector<std::size_t> greedy{0, 0, 0, 0};
    const std::vector<std::size_t> forced(
            kForcedIds.begin(), kForcedIds.end());
    nlohmann::json cases = nlohmann::json::array({
            official_reference_case(
                    "raw-production-greedy", "raw", "production-greedy",
                    raw_prompt, greedy),
            official_reference_case(
                    "raw-fixed-reference-continuation", "raw",
                    "fixed-reference-continuation", raw_prompt, forced),
            official_reference_case(
                    "chat-production-greedy", "chat",
                    "production-greedy", chat_prompt, greedy),
            official_reference_case(
                    "chat-fixed-reference-continuation", "chat",
                    "fixed-reference-continuation", chat_prompt, forced),
            official_reference_case(
                    "zero-new-token", "raw", "none", raw_prompt, {})});
    return {
            {"schema_version", 1},
            {"kind", "official"},
            {"provenance",
             {{"runtime",
               {{"implementation", "CPython"}, {"version", "3.11.16"}}},
              {"packages",
               {{"jinja2", "3.1.2"},
                {"numpy", "1.26.4"},
                {"safetensors", "0.4.1"},
                {"sentencepiece", "0.1.99"},
                {"tokenizers", "0.14.1"},
                {"torch", "2.1.2"},
                {"transformers", "4.35.0"}}},
              {"precision",
               {{"weights", "BF16"},
                {"activations", "BF16"},
                {"attention", "eager"},
                {"logits_storage", "BF16"},
                {"reference_arithmetic", "FP32"},
                {"device", "cpu"}}},
              {"generator_sha256", std::string(64, '0')},
              {"exporter_sha256", std::string(64, '1')},
              {"command", {"python3", "reference-pack-regression"}},
              {"case_payload_sha256", std::string(64, '2')},
              {"model_id", model_config.at("model_id")},
              {"revision", artifact_id},
              {"artifacts", nlohmann::json::array()}}},
            {"tolerances",
             {{"absolute", 0.25},
              {"relative", 0.02},
              {"formula", "abs(actual-ref) <= 0.25 + 0.02*abs(ref)"},
              {"tie_policy", "lowest-id"},
              {"exact_token", "reference-margin-certified-only"}}},
            {"model_config", model_config},
            {"artifact_id", artifact_id},
            {"cases", std::move(cases)}};
}

TEST_CASE("Official model harness executes the common tiny runner in every mode") {
    const nlohmann::json corpus = load_synthetic_corpus();
    const std::string model_id = "synthetic-h18-i22-hq3-hkv1-d6";
    const nlohmann::json& model = corpus_model(corpus, model_id);
    const nlohmann::json& source = corpus_case(
            corpus, model_id + "-teacher-forced-from-14");
    SyntheticFixture fixture(model);
    alignas(32) std::array<std::byte, 1u << 20> arena{};
    iom::ListAllocator allocator(arena.data(), arena.size());
    const std::unique_ptr<iom::Device> device =
            iom::make_cpu_device(allocator);
    const auto test_case = make_synthetic_official_case(model, source);

    const std::array<
            iom_conformance::official_model_detail::ModeObservation, 3>
            observations{
                    iom_conformance::official_model_detail::run_mode(
                            *device, fixture.path(), test_case, 0),
                    iom_conformance::official_model_detail::run_mode(
                            *device, fixture.path(), test_case, 1),
                    iom_conformance::official_model_detail::run_mode(
                            *device, fixture.path(), test_case, 2)};
    nlohmann::json evidence;
    CHECK_NOTHROW(
            iom_conformance::official_model_detail::compare_case(
                    test_case, observations, evidence));
    CHECK(evidence.at("mode_parity") == true);
    CHECK(evidence.at("reference_history_matched") == true);
    CHECK(evidence.at("result").at("token_ids") == test_case.decode);
    CHECK(evidence.at("trace").at("rows_dropped") == 0);
    CHECK_FALSE(evidence.at("trace").at("rows").empty());

    auto unexplained_history = observations;
    for (auto& observation : unexplained_history) {
        REQUIRE(!observation.histories.empty());
        REQUIRE(!observation.histories.front().empty());
        observation.histories.front().front() = 0;
    }
    auto greedy_case = test_case;
    greedy_case.policy = "production-greedy";
    nlohmann::json rejected_evidence;
    CHECK_THROWS_AS(
            iom_conformance::official_model_detail::compare_case(
                    greedy_case, unexplained_history, rejected_evidence),
            std::runtime_error);

    const auto& trace = observations[2].operations;
    REQUIRE(trace.size() == 9);
    CHECK(std::all_of(
            trace.begin(), trace.end(),
            [](const iom::InferenceTraceRecord& row) {
                return !row.decoder_layer.has_value();
            }));
    CHECK(
            std::count_if(
                    trace.begin(), trace.end(),
                    [&](const iom::InferenceTraceRecord& row) {
                        return row.phase == iom::InferencePhase::prefill
                                && row.position_start == 0
                                && row.run_length == test_case.prompt.size();
                    })
            == 2);
    CHECK(
            std::count_if(
                    trace.begin(), trace.end(),
                    [&](const iom::InferenceTraceRecord& row) {
                        return row.phase == iom::InferencePhase::prefill
                                && row.position_start + row.run_length
                                        == test_case.prompt.size()
                                && row.run_length == 1;
                    })
            == 1);
    for (std::size_t index = 0; index < 2; ++index) {
        CHECK(
                std::count_if(
                        trace.begin(), trace.end(),
                        [&](const iom::InferenceTraceRecord& row) {
                            return row.phase == iom::InferencePhase::decode
                                    && row.position_start
                                            == test_case.prompt.size() + index
                                    && row.run_length == 1;
                        })
                == 3);
    }
    for (const auto& row : evidence.at("trace").at("rows")) {
        CHECK(row.at("decoder_layer").is_null());
    }
}

TEST_CASE("Official model harness raw and chat execution use validated prompt IDs") {
    nlohmann::json config = iom_inference_metrics_test::h16_config();
    config["vocab_size"] = 300;
    config["max_position_embeddings"] = 512;
    iom_inference_metrics_test::SyntheticFixture fixture(
            "official-prompt-paths", config);
    const auto tokenizer = iom::load_tokenizer(fixture.directory.path());
    const auto formatter = iom::load_chat_formatter(fixture.directory.path());
    REQUIRE(tokenizer);
    REQUIRE(formatter);

    alignas(32) std::array<std::byte, 1u << 22> arena{};
    iom::ListAllocator allocator(arena.data(), arena.size());
    const std::unique_ptr<iom::Device> device =
            iom::make_cpu_device(allocator);

    using iom_conformance::official_model_detail::OfficialCase;
    using iom_conformance::official_model_detail::run_mode;
    const std::string raw_text = "The capital of France is";
    OfficialCase raw;
    raw.name = "synthetic-raw-prompt";
    raw.prompt = widen_ids(
            tokenizer->encode(raw_text, iom::EncodeOptions{true}));
    raw.decode = {3};
    raw.limit = 1;
    raw.mode = "raw";
    raw.policy = "fixed-reference-continuation";
    raw.layers = config.at("num_hidden_layers").get<std::size_t>();
    raw.vocabulary = config.at("vocab_size").get<std::size_t>();
    raw.record = {{"input", {{"text", raw_text}}}};
    const auto raw_observed =
            run_mode(*device, fixture.directory.path(), raw, 0);
    REQUIRE(raw_observed.histories.size() == 1);
    CHECK(raw_observed.histories.front() == raw.prompt);
    REQUIRE(!raw.prompt.empty());
    CHECK(raw.prompt.front() == 1);

    const std::array<iom::ChatMessageView, 1> messages{
            iom::ChatMessageView{"user", "Hello."}};
    const std::string rendered = formatter->format(messages, true);
    OfficialCase chat;
    chat.name = "synthetic-chat-prompt";
    chat.prompt = widen_ids(
            tokenizer->encode(rendered, iom::EncodeOptions{false}));
    chat.decode = {3};
    chat.limit = 1;
    chat.mode = "chat";
    chat.policy = "fixed-reference-continuation";
    chat.layers = raw.layers;
    chat.vocabulary = raw.vocabulary;
    chat.record = {
            {"input", {{"rendered_utf8", byte_values(rendered)}}}};
    REQUIRE(!chat.prompt.empty());
    CHECK(chat.prompt.front() != 1);
    const auto chat_observed =
            run_mode(*device, fixture.directory.path(), chat, 0);
    REQUIRE(chat_observed.histories.size() == 1);
    CHECK(chat_observed.histories.front() == chat.prompt);
}

TEST_CASE("Official model harness zero-token admission uses the common runner") {
    const nlohmann::json corpus = load_synthetic_corpus();
    const std::string model_id = "synthetic-h18-i22-hq3-hkv1-d6";
    const nlohmann::json& model = corpus_model(corpus, model_id);
    SyntheticFixture fixture(model);
    alignas(32) std::array<std::byte, 1u << 20> arena{};
    iom::ListAllocator allocator(arena.data(), arena.size());
    const std::unique_ptr<iom::Device> device =
            iom::make_cpu_device(allocator);
    auto test_case = make_synthetic_official_case(
            model, corpus_case(
                           corpus, model_id + "-teacher-forced-from-1"));
    test_case.name = "synthetic-common-zero-new";
    test_case.decode.clear();
    test_case.limit = 0;
    test_case.policy = "none";
    test_case.record["snapshots"] = nlohmann::json::array();
    test_case.record["expected_result"] = {
            {"token_ids", nlohmann::json::array()},
            {"stop_reason", "max_new_tokens"},
            {"initialized_kv_length", 0}};

    const std::array<
            iom_conformance::official_model_detail::ModeObservation, 3>
            observations{
                    iom_conformance::official_model_detail::run_mode(
                            *device, fixture.path(), test_case, 0),
                    iom_conformance::official_model_detail::run_mode(
                            *device, fixture.path(), test_case, 1),
                    iom_conformance::official_model_detail::run_mode(
                            *device, fixture.path(), test_case, 2)};
    nlohmann::json evidence;
    CHECK_NOTHROW(
            iom_conformance::official_model_detail::compare_case(
                    test_case, observations, evidence));
    CHECK(evidence.at("result").at("token_ids").empty());
    CHECK(evidence.at("metrics").at("generated_tokens") == 0);
    CHECK(evidence.at("trace").at("rows").empty());
}

TEST_CASE("Official model harness rejects missing, malformed, nonfinite, and escaping inputs") {
    alignas(32) std::array<std::byte, 1u << 20> arena{};
    iom::ListAllocator allocator(arena.data(), arena.size());
    const std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
    unsetenv("IOM_TEST_MODEL_DIR");
    unsetenv("IOM_TEST_MODEL_ID");
    unsetenv("IOM_TEST_MODEL_REFERENCE");
    unsetenv("IOM_TEST_MODEL_EVIDENCE");
    CHECK_THROWS_AS(iom_conformance::run_real_model_inference(*device), std::invalid_argument);

    const nlohmann::json malformed = nlohmann::json::object();
    CHECK_THROWS_AS(
            iom_conformance::official_model_detail::validate_reference_pack(
                    malformed, "missing"),
            std::invalid_argument);
    nlohmann::json nonfinite = nlohmann::json::array();
    for (std::size_t index = 0;
         index < iom_conformance::official_model_detail::kVocabulary; ++index) {
        nonfinite.push_back(index == 0
                ? std::numeric_limits<double>::quiet_NaN()
                : 0.0);
    }
    CHECK_THROWS_AS(
            iom_conformance::official_model_detail::finite_logits(
                    nonfinite, "nonfinite logits"),
            std::invalid_argument);
    CHECK_THROWS_AS(
            iom_conformance::official_model_detail::checked_relative_artifact(
                    ".", "../escape.json"),
            std::invalid_argument);
}

TEST_CASE("Official model harness accepts the complete frozen reference schema") {
    using namespace iom_conformance::official_model_detail;
    const nlohmann::json pack = complete_official_reference_pack();
    const auto cases =
            validate_reference_pack(pack, "official-pack-regression");
    REQUIRE(cases.size() == 5);
    CHECK(cases.front().name == "raw-production-greedy");
    CHECK(cases.back().name == "zero-new-token");

    auto name_alias = pack;
    name_alias["cases"][0]["name"] =
            name_alias["cases"][0]["id"];
    name_alias["cases"][0].erase("id");
    CHECK_THROWS_AS(
            validate_reference_pack(
                    name_alias, "official-pack-regression"),
            std::invalid_argument);

    auto mutated_window = pack;
    mutated_window["cases"][0]["snapshots"][0]["run_length"] =
            std::size_t{0};
    CHECK_THROWS_AS(
            validate_reference_pack(
                    mutated_window, "official-pack-regression"),
            std::invalid_argument);

    auto early_eos = pack;
    early_eos["cases"][0]["input"]["decode_ids"] =
            nlohmann::json::array({2, 0, 0, 0});
    CHECK_THROWS_AS(
            validate_reference_pack(
                    early_eos, "official-pack-regression"),
            std::invalid_argument);

    auto short_non_eos = pack;
    auto& short_case = short_non_eos["cases"][0];
    short_case["input"]["decode_ids"] =
            nlohmann::json::array({0, 0, 0});
    short_case["snapshots"].erase(
            short_case["snapshots"].end() - 1);
    short_case["expected_result"] = {
            {"token_ids", nlohmann::json::array({0, 0, 0})},
            {"stop_reason", "context_exhaustion"},
            {"initialized_kv_length", 4}};
    CHECK_THROWS_AS(
            validate_reference_pack(
                    short_non_eos, "official-pack-regression"),
            std::invalid_argument);

    auto context_capacity = short_non_eos;
    auto& capacity_case = context_capacity["cases"][0];
    nlohmann::json prompt = nlohmann::json::array();
    nlohmann::json positions = nlohmann::json::array();
    for (std::size_t index = 0; index < kContext - 3; ++index) {
        prompt.push_back(index == 0 ? 1 : 10);
        positions.push_back(index);
    }
    capacity_case["input"]["prompt_ids"] = prompt;
    capacity_case["input"]["positions"] = positions;
    for (std::size_t index = 0;
         index < capacity_case["snapshots"].size(); ++index) {
        nlohmann::json prefix = prompt;
        for (std::size_t decode = 0; decode < index; ++decode) {
            prefix.push_back(0);
        }
        auto& snapshot = capacity_case["snapshots"][index];
        snapshot["prefix_ids"] = prefix;
        if (index == 0) {
            snapshot["position_start"] = 0;
            snapshot["run_length"] = prompt.size();
            snapshot["absolute_positions"] = positions;
        } else {
            snapshot["position_start"] = prefix.size() - 1;
            snapshot["run_length"] = 1;
            snapshot["absolute_positions"] =
                    nlohmann::json::array({prefix.size() - 1});
        }
    }
    capacity_case["expected_result"]["initialized_kv_length"] =
            kContext - 1;
    CHECK_NOTHROW(
            validate_reference_pack(
                    context_capacity, "official-pack-regression"));
}

TEST_CASE("Official model harness protects inputs from evidence aliases") {
    const nlohmann::json corpus = load_synthetic_corpus();
    SyntheticFixture fixture(corpus.at("models").front());
    iom_model_loading::TempDir outputs(
            "official-evidence-path-separation");
    const std::filesystem::path reference =
            outputs.path() / "reference.json";
    {
        std::ofstream stream(reference, std::ios::binary);
        REQUIRE(stream);
        stream << "{\"schema_version\":";
    }
    const std::filesystem::path reference_symlink =
            outputs.path() / "reference-link.json";
    std::error_code error;
    std::filesystem::create_symlink(
            reference, reference_symlink, error);
    REQUIRE_FALSE(error);
    const std::filesystem::path artifact =
            fixture.path() / "model.safetensors";
    const std::filesystem::path artifact_hard_link =
            outputs.path() / "model-hard-link.safetensors";
    std::filesystem::create_hard_link(
            artifact, artifact_hard_link, error);
    REQUIRE_FALSE(error);
    const std::string reference_hash =
            iom_conformance::official_model_detail::sha256_file(reference);
    const std::string artifact_hash =
            iom_conformance::official_model_detail::sha256_file(artifact);

    REQUIRE(setenv("IOM_TEST_MODEL_DIR", fixture.path().c_str(), 1) == 0);
    REQUIRE(setenv("IOM_TEST_MODEL_ID", "path-separation", 1) == 0);
    alignas(32) std::array<std::byte, 1u << 20> arena{};
    iom::ListAllocator allocator(arena.data(), arena.size());
    const std::unique_ptr<iom::Device> device =
            iom::make_cpu_device(allocator);

    REQUIRE(setenv(
            "IOM_TEST_MODEL_REFERENCE",
            reference_symlink.c_str(), 1) == 0);
    const std::filesystem::path safe_evidence =
            outputs.path() / "safe-evidence.json";
    REQUIRE(setenv(
            "IOM_TEST_MODEL_EVIDENCE",
            safe_evidence.c_str(), 1) == 0);
    CHECK_THROWS_AS(
            iom_conformance::run_real_model_inference(*device),
            std::invalid_argument);
    REQUIRE(std::filesystem::exists(safe_evidence));
    const nlohmann::json symlink_failure =
            iom_conformance::official_model_detail::read_json(
                    safe_evidence, "symlink preflight evidence");
    CHECK(symlink_failure.at("status") == "failed");
    CHECK(
            symlink_failure.at("failure").get<std::string>().find(
                    "regular non-symlink")
            != std::string::npos);

    const std::filesystem::path missing_reference =
            outputs.path() / "missing-reference.json";
    const std::filesystem::path missing_evidence =
            outputs.path() / "missing-reference-evidence.json";
    {
        std::ofstream stale(missing_evidence, std::ios::binary);
        REQUIRE(stale);
        stale << "{\"status\":\"stale\"}\n";
    }
    REQUIRE(setenv(
            "IOM_TEST_MODEL_REFERENCE",
            missing_reference.c_str(), 1) == 0);
    REQUIRE(setenv(
            "IOM_TEST_MODEL_EVIDENCE",
            missing_evidence.c_str(), 1) == 0);
    CHECK_THROWS_AS(
            iom_conformance::run_real_model_inference(*device),
            std::invalid_argument);
    const nlohmann::json missing_failure =
            iom_conformance::official_model_detail::read_json(
                    missing_evidence, "missing-reference evidence");
    CHECK(missing_failure.at("status") == "failed");
    CHECK(
            missing_failure.at("failure").get<std::string>().find(
                    "regular non-symlink")
            != std::string::npos);

    REQUIRE(setenv(
            "IOM_TEST_MODEL_REFERENCE", reference.c_str(), 1) == 0);
    const std::filesystem::path temporary_alias_evidence =
            outputs.path() / "temporary-alias-evidence.json";
    const std::filesystem::path temporary_alias =
            temporary_alias_evidence.string() + ".tmp";
    std::filesystem::create_hard_link(
            artifact, temporary_alias, error);
    REQUIRE_FALSE(error);
    REQUIRE(setenv(
            "IOM_TEST_MODEL_EVIDENCE",
            temporary_alias_evidence.c_str(), 1) == 0);
    CHECK_THROWS_AS(
            iom_conformance::run_real_model_inference(*device),
            std::invalid_argument);
    CHECK_FALSE(
            std::filesystem::exists(temporary_alias_evidence));
    CHECK(
            iom_conformance::official_model_detail::sha256_file(artifact)
            == artifact_hash);
    CHECK(
            iom_conformance::official_model_detail::sha256_file(
                    temporary_alias)
            == artifact_hash);

    for (const std::filesystem::path& destination :
         {reference, artifact, artifact_hard_link,
          fixture.path() / "new-evidence.json"}) {
        REQUIRE(setenv(
                "IOM_TEST_MODEL_EVIDENCE",
                destination.c_str(), 1) == 0);
        CHECK_THROWS_AS(
                iom_conformance::run_real_model_inference(*device),
                std::invalid_argument);
        CHECK(
                iom_conformance::official_model_detail::sha256_file(reference)
                == reference_hash);
        CHECK(
                iom_conformance::official_model_detail::sha256_file(artifact)
                == artifact_hash);
    }
    CHECK(
            iom_conformance::official_model_detail::sha256_file(
                    artifact_hard_link)
            == artifact_hash);
    CHECK_FALSE(
            std::filesystem::exists(
                    fixture.path() / "new-evidence.json"));

    unsetenv("IOM_TEST_MODEL_DIR");
    unsetenv("IOM_TEST_MODEL_ID");
    unsetenv("IOM_TEST_MODEL_REFERENCE");
    unsetenv("IOM_TEST_MODEL_EVIDENCE");
}

TEST_CASE("Official model harness revalidates artifact bytes across load boundaries") {
    const nlohmann::json corpus = load_synthetic_corpus();
    SyntheticFixture fixture(corpus.at("models").front());
    for (const std::string_view name :
         {"generation_config.json", "special_tokens_map.json",
          "tokenizer.model"}) {
        std::ofstream output(fixture.path() / name, std::ios::binary);
        REQUIRE(output);
        output << "{}";
    }
    const std::array<std::string_view, 7> names{
            "config.json", "generation_config.json", "model.safetensors",
            "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
            "tokenizer_config.json"};
    nlohmann::json artifacts = nlohmann::json::array();
    for (const std::string_view name : names) {
        const std::filesystem::path path = fixture.path() / name;
        artifacts.push_back({
                {"path", name},
                {"size",
                 iom_conformance::official_model_detail::official_file_size(
                         path)},
                {"sha256",
                 iom_conformance::official_model_detail::sha256_file(path)}});
    }
    const nlohmann::json provenance{{"artifacts", artifacts}};
    CHECK_NOTHROW(
            iom_conformance::official_model_detail::validate_artifacts(
                    fixture.path(), provenance));

    {
        std::ofstream output(
                fixture.path() / "model.safetensors",
                std::ios::binary | std::ios::app);
        REQUIRE(output);
        output << "tampered";
    }
    CHECK_THROWS_AS(
            iom_conformance::official_model_detail::validate_artifacts(
                    fixture.path(), provenance),
            std::runtime_error);
}

TEST_CASE("Official model harness freezes arena, tie, and tolerance policy") {
    using namespace iom_conformance::official_model_detail;
    CHECK(parse_arena_bytes("32") == 32);
    CHECK(parse_arena_bytes("1024") == 1024);
    CHECK_THROWS_AS(parse_arena_bytes("0"), std::invalid_argument);
    CHECK_THROWS_AS(parse_arena_bytes("+32"), std::invalid_argument);
    CHECK_THROWS_AS(parse_arena_bytes("33"), std::invalid_argument);

    const std::array<float, 3> tie{1.0F, 1.0F, 0.0F};
    CHECK(lowest_argmax(tie) == 0);
    CHECK_FALSE(reference_margin_stable(tie));
    const std::array<float, 3> near_tie{1.0F, 0.99F, 0.0F};
    CHECK_FALSE(reference_margin_stable(near_tie));
    const std::array<float, 3> certified{2.0F, 0.0F, -1.0F};
    CHECK(reference_margin_stable(certified));

    const std::array<float, 2> reference{1.0F, -2.0F};
    const std::array<float, 2> accepted{1.27F, -2.29F};
    const std::array<float, 2> rejected{1.28F, -2.29F};
    CHECK_NOTHROW(compare_logits(accepted, reference, "accepted"));
    CHECK_THROWS_AS(
            compare_logits(rejected, reference, "rejected"),
            std::runtime_error);
}

TEST_CASE("Official model harness records corrupt reference failure") {
    const nlohmann::json corpus = load_synthetic_corpus();
    SyntheticFixture fixture(corpus.at("models").front());
    iom_model_loading::TempDir outputs(
            "official-corrupt-reference-evidence");
    const std::filesystem::path reference =
            outputs.path() / "corrupt-official-reference.json";
    const std::filesystem::path evidence =
            outputs.path() / "official-failure-evidence.json";
    {
        std::ofstream output(reference, std::ios::binary);
        REQUIRE(output);
        output << "{\"schema_version\":";
    }
    REQUIRE(setenv("IOM_TEST_MODEL_DIR", fixture.path().c_str(), 1) == 0);
    REQUIRE(setenv("IOM_TEST_MODEL_ID", "corrupt-reference", 1) == 0);
    REQUIRE(setenv("IOM_TEST_MODEL_REFERENCE", reference.c_str(), 1) == 0);
    REQUIRE(setenv("IOM_TEST_MODEL_EVIDENCE", evidence.c_str(), 1) == 0);

    alignas(32) std::array<std::byte, 1u << 20> arena{};
    iom::ListAllocator allocator(arena.data(), arena.size());
    const std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
    CHECK_THROWS_AS(
            iom_conformance::run_real_model_inference(*device),
            std::invalid_argument);
    const nlohmann::json recorded =
            iom_conformance::official_model_detail::read_json(
                    evidence, "failure evidence");
    CHECK(recorded.at("status") == "failed");
    CHECK(recorded.at("failure").get<std::string>().find("malformed")
          != std::string::npos);

    unsetenv("IOM_TEST_MODEL_DIR");
    unsetenv("IOM_TEST_MODEL_ID");
    unsetenv("IOM_TEST_MODEL_REFERENCE");
    unsetenv("IOM_TEST_MODEL_EVIDENCE");
}

}  // namespace
