#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "model_loading_fixture.hpp"

namespace {

using iom_model_loading::SafetensorsEntry;
using iom_model_loading::TempDir;
using iom_model_loading::as_destinations;
using iom_model_loading::bf16_entry;
using iom_model_loading::destination_pointers;
using iom_model_loading::deterministic_payload;
using iom_model_loading::dims_of;
using iom_model_loading::element_count;
using iom_model_loading::FakeDevice;
using iom_model_loading::kSupportedWithoutBf16;
using iom_model_loading::make_destinations;
using iom_model_loading::observed;
using iom_model_loading::one_layer_config;
using iom_model_loading::rejected_config_message;
using iom_model_loading::rejected_container_message;
using iom_model_loading::rejected_source_message;
using iom_model_loading::required_weight_entries;
using iom_model_loading::same_config;
using iom_model_loading::sentinel_storage;
using iom_model_loading::two_layer_config;
using iom_model_loading::WorkspacePolicy;
using iom_model_loading::write_config;
using iom_model_loading::write_file;
using iom_model_loading::write_safetensors_file;
using iom_model_loading::write_safetensors_header;

// Every configuration rejection names the file and the offending field.
void check_rejection(const std::string& message, const TempDir& dir,
                     std::string_view field) {
    REQUIRE_FALSE(message.empty());
    CHECK(message.find(field) != std::string::npos);
    CHECK(message.find(dir.path().string()) != std::string::npos);
}

TEST_CASE("Model loading configuration decodes the fixed two-layer boundary dimensions") {
    const TempDir dir("two-layer");
    write_config(dir.path(), two_layer_config());

    const iom::TinyLlamaConfig config = iom::load_tinyllama_config(dir.path());

    CHECK(config.num_hidden_layers == 2);
    CHECK(config.hidden_size == 8);
    CHECK(config.intermediate_size == 12);
    CHECK(config.num_attention_heads == 4);
    CHECK(config.num_key_value_heads == 2);
    CHECK(config.vocab_size == 19);
    CHECK(config.max_position_embeddings == 17);
    CHECK(config.head_dim == 2);
    CHECK(config.bos_token_id == 1);
    CHECK(config.eos_token_id == 2);
    CHECK(config.rms_norm_eps == 1e-5f);
    CHECK(config.rope_theta == 10000.0);
}

TEST_CASE("Model loading configuration decodes a valid one-layer configuration") {
    const TempDir dir("one-layer");
    write_config(dir.path(), one_layer_config());

    const iom::TinyLlamaConfig boundary = iom::load_tinyllama_config(dir.path());
    CHECK(boundary.num_hidden_layers == 1);
    CHECK(boundary.head_dim == 2);
    CHECK(boundary.vocab_size == 19);

    // A layer count of one with independent dimensions stays parameterized.
    const TempDir other_dir("one-layer-independent");
    nlohmann::json document = one_layer_config();
    document["hidden_size"] = 16;
    document["intermediate_size"] = 32;
    document["num_attention_heads"] = 4;
    document["num_key_value_heads"] = 1;
    document["vocab_size"] = 21;
    document["max_position_embeddings"] = 9;
    write_config(other_dir.path(), document);

    const iom::TinyLlamaConfig config =
            iom::load_tinyllama_config(other_dir.path());
    CHECK(config.num_hidden_layers == 1);
    CHECK(config.hidden_size == 16);
    CHECK(config.intermediate_size == 32);
    CHECK(config.num_key_value_heads == 1);
    CHECK(config.vocab_size == 21);
    CHECK(config.max_position_embeddings == 9);
    CHECK(config.head_dim == 4);
}

TEST_CASE("Model loading configuration accepts absent optionals that equal the documented defaults") {
    const TempDir defaults_dir("optional-defaults");
    write_config(defaults_dir.path(), two_layer_config());

    const TempDir explicit_dir("optional-explicit");
    nlohmann::json document = two_layer_config();
    document["attention_bias"] = false;
    document["mlp_bias"] = false;
    document["tie_word_embeddings"] = false;
    document["rope_scaling"] = nullptr;
    document["pretraining_tp"] = 1;
    write_config(explicit_dir.path(), document);

    const iom::TinyLlamaConfig absent =
            iom::load_tinyllama_config(defaults_dir.path());
    const iom::TinyLlamaConfig present =
            iom::load_tinyllama_config(explicit_dir.path());

    CHECK(same_config(absent, present));
    CHECK(present.head_dim == 2);
}

TEST_CASE("Model loading configuration accepts large representable dimensions") {
    const TempDir dir("large-dimensions");
    nlohmann::json document = two_layer_config();
    document["hidden_size"] = 1099511627776ULL;               // 2^40
    document["intermediate_size"] = 2199023255552ULL;          // 2^41
    document["num_attention_heads"] = 4;
    document["num_key_value_heads"] = 2;
    write_config(dir.path(), document);

    const iom::TinyLlamaConfig config = iom::load_tinyllama_config(dir.path());

    CHECK(config.hidden_size == 1099511627776ULL);
    CHECK(config.intermediate_size == 2199023255552ULL);
    CHECK(config.head_dim == 274877906944ULL);                 // 2^38
}

TEST_CASE("Model loading configuration accepts bounded representable epsilon and theta values") {
    const TempDir dir("numeric-boundaries");
    nlohmann::json document = two_layer_config();
    document["rms_norm_eps"] = 1.1754943508222875e-38;   // smallest normal float
    document["rope_theta"] = 1.7976931348623157e308;     // largest finite double
    write_config(dir.path(), document);

    const iom::TinyLlamaConfig config = iom::load_tinyllama_config(dir.path());

    CHECK(config.rms_norm_eps == 1.1754943508222875e-38f);
    CHECK(config.rope_theta == 1.7976931348623157e308);

    // Integral JSON numbers are valid positive numeric values.
    const TempDir integral_dir("integral-numerics");
    nlohmann::json integral = two_layer_config();
    integral["rms_norm_eps"] = 1;
    integral["rope_theta"] = 1;
    write_config(integral_dir.path(), integral);

    const iom::TinyLlamaConfig integral_config =
            iom::load_tinyllama_config(integral_dir.path());
    CHECK(integral_config.rms_norm_eps == 1.0f);
    CHECK(integral_config.rope_theta == 1.0);
}

TEST_CASE("Model loading configuration rejects unsupported optional values") {
    struct RejectedOptional {
        const char* field;
        nlohmann::json value;
    };
    const RejectedOptional rejected[] = {
            {"attention_bias", nlohmann::json(true)},
            {"attention_bias", nlohmann::json(0)},
            {"mlp_bias", nlohmann::json(true)},
            {"mlp_bias", nlohmann::json(nullptr)},
            {"tie_word_embeddings", nlohmann::json("false")},
            {"rope_scaling", nlohmann::json{{"type", "linear"}}},
            {"rope_scaling", nlohmann::json(true)},
            {"pretraining_tp", nlohmann::json(2)},
            {"pretraining_tp", nlohmann::json(1.0)},
            {"pretraining_tp", nlohmann::json(true)},
            {"pretraining_tp", nlohmann::json("1")},
    };

    for (const RejectedOptional& test_case : rejected) {
        CAPTURE(test_case.field);
        CAPTURE(test_case.value.dump());
        TempDir dir("optional-reject");
        nlohmann::json document = two_layer_config();
        document[test_case.field] = test_case.value;
        write_config(dir.path(), document);

        check_rejection(rejected_config_message(dir.path()), dir,
                        test_case.field);
    }
}

TEST_CASE("Model loading configuration reports a missing config.json as an I/O failure") {
    const TempDir dir("missing-config");

    bool rejected_as_argument = false;
    std::string message;
    try {
        const iom::TinyLlamaConfig config = iom::load_tinyllama_config(dir.path());
        (void)config;
    } catch (const std::invalid_argument&) {
        rejected_as_argument = true;
    } catch (const std::runtime_error& error) {
        message = error.what();
    }

    CHECK_FALSE(rejected_as_argument);
    REQUIRE_FALSE(message.empty());
    CHECK(message.find(dir.path().string()) != std::string::npos);

    // An absent model directory is the same I/O failure, not a missing field.
    CHECK_THROWS_AS(iom::load_tinyllama_config(dir.path() / "no-such-directory"),
                    std::runtime_error);
}

TEST_CASE("Model loading configuration rejects malformed, empty, and non-object JSON") {
    const std::string documents[] = {
            "",           "   \r\n", "{",      "not json",
            "[1, 2]",     "5",       "\"llama\"", "null",
            "{}",
    };

    for (const std::string& text : documents) {
        CAPTURE(text);
        TempDir dir("malformed-json");
        write_file(dir.path(), "config.json", text);

        check_rejection(rejected_config_message(dir.path()), dir, "config.json");
    }
}

TEST_CASE("Model loading configuration requires every documented field") {
    const std::string fields[] = {
            "architectures",       "model_type",
            "num_hidden_layers",   "hidden_size",
            "intermediate_size",   "num_attention_heads",
            "num_key_value_heads", "vocab_size",
            "max_position_embeddings", "hidden_act",
            "rms_norm_eps",        "rope_theta",
            "torch_dtype",         "bos_token_id",
            "eos_token_id",
    };

    for (const std::string& field : fields) {
        CAPTURE(field);
        TempDir dir("missing-field");
        nlohmann::json document = two_layer_config();
        REQUIRE(document.erase(field) == 1);
        write_config(dir.path(), document);

        const std::string message = rejected_config_message(dir.path());
        check_rejection(message, dir, field);
        CHECK(message.find("<missing>") != std::string::npos);
    }
}

TEST_CASE("Model loading configuration rejects non-integer and non-positive integer fields") {
    const std::string fields[] = {
            "num_hidden_layers",       "hidden_size",
            "intermediate_size",       "num_attention_heads",
            "num_key_value_heads",     "vocab_size",
            "max_position_embeddings", "bos_token_id",
            "eos_token_id",
    };
    const nlohmann::json wrong_kinds[] = {
            nlohmann::json(1.0),
            nlohmann::json(true),
            nlohmann::json("8"),
            nlohmann::json::array({8}),
            nlohmann::json(nullptr),
    };
    const nlohmann::json non_positive[] = {
            nlohmann::json(0),
            nlohmann::json(-1),
    };

    for (const std::string& field : fields) {
        CAPTURE(field);
        for (const nlohmann::json& value : wrong_kinds) {
            CAPTURE(value.dump());
            TempDir dir("integer-kind");
            nlohmann::json document = two_layer_config();
            document[field] = value;
            write_config(dir.path(), document);

            check_rejection(rejected_config_message(dir.path()), dir, field);
        }
        for (const nlohmann::json& value : non_positive) {
            CAPTURE(value.dump());
            TempDir dir("integer-positive");
            nlohmann::json document = two_layer_config();
            document[field] = value;
            write_config(dir.path(), document);

            check_rejection(rejected_config_message(dir.path()), dir, field);
        }
    }
}

TEST_CASE("Model loading configuration rejects unknown configuration keys") {
    const std::string keys[] = {
            "head_dim",
            "num_key_value_head",
            "rope_theta_scale",
            "tokenizer_class",
    };

    for (const std::string& key : keys) {
        CAPTURE(key);
        TempDir dir("unknown-key");
        nlohmann::json document = two_layer_config();
        document[key] = 4;
        write_config(dir.path(), document);

        check_rejection(rejected_config_message(dir.path()), dir, key);
    }
}

TEST_CASE("Model loading configuration rejects unsupported architecture, activation, and dtype strings") {
    struct RejectedString {
        const char* field;
        nlohmann::json value;
    };
    const RejectedString rejected[] = {
            {"architectures", nlohmann::json("LlamaForCausalLM")},
            {"architectures", nlohmann::json::array()},
            {"architectures", nlohmann::json::array({1})},
            {"architectures", nlohmann::json::array({"MistralForCausalLM"})},
            {"architectures",
             nlohmann::json::array({"LlamaForCausalLM", "MistralForCausalLM"})},
            {"model_type", nlohmann::json("gpt2")},
            {"model_type", nlohmann::json::array({"llama"})},
            {"model_type", nlohmann::json(nullptr)},
            {"hidden_act", nlohmann::json("gelu")},
            {"hidden_act", nlohmann::json(nullptr)},
            {"torch_dtype", nlohmann::json("float16")},
            {"torch_dtype", nlohmann::json::array({"bfloat16"})},
    };

    for (const RejectedString& test_case : rejected) {
        CAPTURE(test_case.field);
        CAPTURE(test_case.value.dump());
        TempDir dir("unsupported-string");
        nlohmann::json document = two_layer_config();
        document[test_case.field] = test_case.value;
        write_config(dir.path(), document);

        check_rejection(rejected_config_message(dir.path()), dir,
                        test_case.field);
    }
}

TEST_CASE("Model loading configuration rejects incompatible head ratios and odd head widths") {
    struct RejectedHeadPlan {
        const char* field;
        nlohmann::json value;
        const char* reported;
    };
    const RejectedHeadPlan rejected[] = {
            // hidden_size is not a multiple of num_attention_heads.
            {"hidden_size", nlohmann::json(10), "hidden_size"},
            {"num_attention_heads", nlohmann::json(3), "hidden_size"},
            // num_attention_heads is not a multiple of num_key_value_heads.
            {"num_key_value_heads", nlohmann::json(3), "num_attention_heads"},
            {"num_key_value_heads", nlohmann::json(8), "num_attention_heads"},
            // The derived per-head width is odd: 12 / 4 = 3, 8 / 8 = 1.
            {"hidden_size", nlohmann::json(12), "hidden_size"},
            {"num_attention_heads", nlohmann::json(8), "hidden_size"},
    };

    for (const RejectedHeadPlan& test_case : rejected) {
        CAPTURE(test_case.field);
        CAPTURE(test_case.value.dump());
        TempDir dir("head-plan");
        nlohmann::json document = two_layer_config();
        document[test_case.field] = test_case.value;
        write_config(dir.path(), document);

        check_rejection(rejected_config_message(dir.path()), dir,
                        test_case.reported);
    }
}

TEST_CASE("Model loading configuration enforces the token policy") {
    struct RejectedToken {
        const char* field;
        nlohmann::json value;
    };
    const RejectedToken rejected[] = {
            {"bos_token_id", nlohmann::json(0)},
            {"bos_token_id", nlohmann::json(2)},
            {"bos_token_id", nlohmann::json(1.0)},
            {"bos_token_id", nlohmann::json(true)},
            {"bos_token_id", nlohmann::json("1")},
            {"bos_token_id", nlohmann::json::array({1})},
            {"eos_token_id", nlohmann::json(1)},
            {"eos_token_id", nlohmann::json(3)},
            {"eos_token_id", nlohmann::json(2.0)},
            {"eos_token_id", nlohmann::json(nullptr)},
            {"eos_token_id", nlohmann::json::array({2})},
    };

    for (const RejectedToken& test_case : rejected) {
        CAPTURE(test_case.field);
        CAPTURE(test_case.value.dump());
        TempDir dir("token-policy");
        nlohmann::json document = two_layer_config();
        document[test_case.field] = test_case.value;
        write_config(dir.path(), document);

        check_rejection(rejected_config_message(dir.path()), dir,
                        test_case.field);
    }
}

TEST_CASE("Model loading configuration bounds token ids by the vocabulary size") {
    const TempDir eos_dir("vocab-eos-bound");
    nlohmann::json eos_document = two_layer_config();
    eos_document["vocab_size"] = 2;
    write_config(eos_dir.path(), eos_document);
    const std::string eos_message = rejected_config_message(eos_dir.path());
    check_rejection(eos_message, eos_dir, "eos_token_id");
    CHECK(eos_message.find("vocab_size") != std::string::npos);

    const TempDir bos_dir("vocab-bos-bound");
    nlohmann::json bos_document = two_layer_config();
    bos_document["vocab_size"] = 1;
    write_config(bos_dir.path(), bos_document);
    const std::string bos_message = rejected_config_message(bos_dir.path());
    check_rejection(bos_message, bos_dir, "bos_token_id");
    CHECK(bos_message.find("vocab_size") != std::string::npos);

    const TempDir valid_dir("vocab-valid");
    nlohmann::json valid_document = two_layer_config();
    valid_document["vocab_size"] = 3;
    write_config(valid_dir.path(), valid_document);
    const iom::TinyLlamaConfig config =
            iom::load_tinyllama_config(valid_dir.path());
    CHECK(config.vocab_size == 3);
    CHECK(config.eos_token_id == 2);
}

TEST_CASE("Model loading configuration rejects nonpositive or non-finite epsilon and theta values") {
    struct RejectedNumeric {
        const char* field;
        nlohmann::json value;
    };
    const RejectedNumeric rejected[] = {
            {"rms_norm_eps", nlohmann::json(0.0)},
            {"rms_norm_eps", nlohmann::json(-1e-5)},
            {"rms_norm_eps", nlohmann::json(1e-300)},  // underflows to zero
            {"rms_norm_eps", nlohmann::json(1e39)},    // overflows to infinity
            {"rms_norm_eps", nlohmann::json("1e-5")},
            {"rms_norm_eps", nlohmann::json(true)},
            {"rms_norm_eps", nlohmann::json(nullptr)},
            {"rms_norm_eps", nlohmann::json::array({1e-5})},
            {"rope_theta", nlohmann::json(0.0)},
            {"rope_theta", nlohmann::json(-10000.0)},
            {"rope_theta", nlohmann::json("10000")},
            {"rope_theta", nlohmann::json(false)},
            {"rope_theta", nlohmann::json(nullptr)},
    };

    for (const RejectedNumeric& test_case : rejected) {
        CAPTURE(test_case.field);
        CAPTURE(test_case.value.dump());
        TempDir dir("numeric-reject");
        nlohmann::json document = two_layer_config();
        document[test_case.field] = test_case.value;
        write_config(dir.path(), document);

        check_rejection(rejected_config_message(dir.path()), dir,
                        test_case.field);
    }
}

TEST_CASE("Model loading configuration ignores unrelated tokenizer, generation, and pad metadata") {
    const TempDir baseline_dir("metadata-baseline");
    write_config(baseline_dir.path(), two_layer_config());
    const iom::TinyLlamaConfig baseline =
            iom::load_tinyllama_config(baseline_dir.path());

    const nlohmann::json variants[] = {
            nlohmann::json{{"initializer_range", 0.02}},
            nlohmann::json{{"transformers_version", "4.31.0"}},
            nlohmann::json{{"use_cache", true}},
            nlohmann::json{{"pad_token_id", 0}},
            nlohmann::json{{"use_cache", false},
                           {"pad_token_id", 2},
                           {"initializer_range", 0.02},
                           {"transformers_version", "4.55.0"}},
    };

    for (const nlohmann::json& variant : variants) {
        CAPTURE(variant.dump());
        TempDir dir("metadata-variant");
        nlohmann::json document = two_layer_config();
        for (const auto& entry : variant.items()) {
            document[entry.key()] = entry.value();
        }
        write_config(dir.path(), document);

        CHECK(same_config(iom::load_tinyllama_config(dir.path()), baseline));
    }
}

TEST_CASE("Model loading configuration opens only config.json in the model directory") {
    const TempDir dir("only-config");
    write_config(dir.path(), two_layer_config());

    // Decoy artifacts and a corrupt checkpoint payload: a searching or
    // fallback-reading loader would observe them, the configuration reader
    // must not.
    write_file(dir.path(), "tokenizer.json", "not json at all");
    write_file(dir.path(), "tokenizer_config.json", "{\"chat_template\":");
    write_file(dir.path(), "generation_config.json", "{\"do_sample\":true}");
    write_file(dir.path(), "special_tokens_map.json", "{}");
    write_file(dir.path(), "model.safetensors", "garbage checkpoint payload");
    write_file(dir.path(), "config.json.bak", "{}");

    const iom::TinyLlamaConfig config = iom::load_tinyllama_config(dir.path());
    CHECK(config.hidden_size == 8);
    CHECK(config.head_dim == 2);

    // Without config.json the decoys are not a fallback source.
    std::error_code error;
    REQUIRE(std::filesystem::remove(dir.path() / "config.json", error));
    CHECK_THROWS_AS(iom::load_tinyllama_config(dir.path()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Mapped weight source
// ---------------------------------------------------------------------------

// One expected entry of the published inventory. The expectation is written
// independently of the implementation, so a wrong role, layer, order, or shape
// fails here instead of matching the production table by construction. `layer`
// is absent for the three global roles and carries the decoder layer for the
// nine layer-scoped roles.
struct ExpectedWeight {
    iom::ModelWeightRole role;
    std::optional<std::size_t> layer;
    std::vector<std::size_t> logical_shape;
};

// The nine layer-scoped roles of the boundary dimensions in canonical order.
std::vector<std::pair<iom::ModelWeightRole, std::vector<std::size_t>>>
layer_weight_plan() {
    return {
            {iom::ModelWeightRole::input_norm, {1, 8}},
            {iom::ModelWeightRole::post_attention_norm, {1, 8}},
            {iom::ModelWeightRole::query, {8, 8}},
            {iom::ModelWeightRole::key, {4, 8}},
            {iom::ModelWeightRole::value, {4, 8}},
            {iom::ModelWeightRole::attention_output, {8, 8}},
            {iom::ModelWeightRole::mlp_gate, {12, 8}},
            {iom::ModelWeightRole::mlp_up, {12, 8}},
            {iom::ModelWeightRole::mlp_down, {8, 12}},
    };
}

// The complete expected inventory of `N=2, H=8, I=12, Hq=4, Hkv=2, D=2, V=19`
// for `layers` decoder layers in canonical global/layer order.
std::vector<ExpectedWeight> expected_inventory(std::size_t layers) {
    std::vector<ExpectedWeight> expected{
            {iom::ModelWeightRole::token_embedding, std::nullopt, {19, 8}},
            {iom::ModelWeightRole::final_norm, std::nullopt, {1, 8}},
            {iom::ModelWeightRole::lm_head, std::nullopt, {19, 8}},
    };
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (const auto& role : layer_weight_plan()) {
            expected.push_back(ExpectedWeight{role.first, layer, role.second});
        }
    }
    return expected;
}

// Checks every published entry, and the selected metadata of every index,
// against the independent expectation.
void check_inventory(const iom::ModelSource& source,
                     const std::vector<ExpectedWeight>& expected) {
    const std::span<const iom::ModelWeightInfo> weights = source.weights();
    REQUIRE(weights.size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CAPTURE(index);
        const ExpectedWeight& want = expected[index];
        const iom::ModelWeightInfo& info = weights[index];
        CHECK(info.id.role == want.role);
        // The identity comparison covers both directions: a global that
        // wrongly carries a layer and a layer-scoped role that wrongly
        // omits one both fail here with the exact expected/actual layer.
        CHECK(info.id.layer == want.layer);
        CHECK(dims_of(info.logical_shape) == want.logical_shape);

        const iom::TensorSpec& spec = source.tensor_spec(index);
        CHECK(dims_of(spec.shape) == want.logical_shape);
        CHECK(spec.data_type == iom::DataType::BF16);
        CHECK(spec.quantization == iom::QuantizationFormat::NONE);
        CHECK(spec.logical_nbytes() == 2 * element_count(want.logical_shape));
    }
}

// Writes `document` and `entries` as one complete checkpoint directory.
void write_checkpoint(const TempDir& dir, const nlohmann::json& document,
                      const std::vector<SafetensorsEntry>& entries) {
    write_config(dir.path(), document);
    write_safetensors_file(dir.path(), "model.safetensors", entries);
}

// Loads a complete checkpoint directory and requires a published source.
std::unique_ptr<iom::ModelSource> load_source(const TempDir& dir) {
    std::unique_ptr<iom::ModelSource> source =
            iom::load_tinyllama_safetensors(dir.path());
    REQUIRE(source != nullptr);
    return source;
}

// Replaces the entry named `name`, leaving the store size unchanged.
void replace_entry(std::vector<SafetensorsEntry>& entries,
                   const std::string& name, SafetensorsEntry replacement) {
    for (SafetensorsEntry& entry : entries) {
        if (entry.name == name) {
            entry = std::move(replacement);
            return;
        }
    }
    REQUIRE_MESSAGE(false, "no fixture entry named " << name);
}

// Removes the entry named `name`, leaving the store size one smaller.
void erase_entry(std::vector<SafetensorsEntry>& entries,
                 const std::string& name) {
    const auto removed =
            std::remove_if(entries.begin(), entries.end(),
                           [&name](const SafetensorsEntry& entry) {
                               return entry.name == name;
                           });
    REQUIRE_MESSAGE(removed != entries.end(), "no fixture entry named " << name);
    entries.erase(removed, entries.end());
}

TEST_CASE("Model loading source selects the complete two-layer inventory in canonical order") {
    const TempDir dir("source-two-layer");
    const nlohmann::json document = two_layer_config();
    write_checkpoint(dir, document, required_weight_entries(document));

    const std::unique_ptr<iom::ModelSource> source = load_source(dir);

    CHECK(source->weights().size() == 21);
    CHECK(same_config(source->config(),
                      iom::load_tinyllama_config(dir.path())));
    check_inventory(*source, expected_inventory(2));

    // Selected metadata is addressed by inventory index under ordinary bounds.
    CHECK_THROWS_AS(source->tensor_spec(source->weights().size()),
                    std::out_of_range);
    CHECK_THROWS_AS(source->tensor_spec(4096), std::out_of_range);
}

TEST_CASE("Model loading source selects the complete one-layer inventory") {
    const TempDir dir("source-one-layer");
    const nlohmann::json document = one_layer_config();
    write_checkpoint(dir, document, required_weight_entries(document));

    const std::unique_ptr<iom::ModelSource> source = load_source(dir);

    CHECK(source->config().num_hidden_layers == 1);
    CHECK(source->weights().size() == 12);
    check_inventory(*source, expected_inventory(1));

    // A one-layer checkpoint cannot satisfy a two-layer configuration: the
    // obviously incomplete store is rejected before any weight is named.
    const TempDir mismatch("source-one-layer-checkpoint");
    write_checkpoint(mismatch, two_layer_config(),
                     required_weight_entries(one_layer_config()));
    const std::string message = rejected_source_message(mismatch.path());
    CHECK(message.find(mismatch.path().string()) != std::string::npos);
    CHECK(message.find("21") != std::string::npos);
    CHECK(message.find("model.embed_tokens.weight") == std::string::npos);
}

TEST_CASE("Model loading source adapts rank-one normalization metadata without changing bytes") {
    const TempDir dir("source-norms");
    const nlohmann::json document = two_layer_config();
    write_checkpoint(dir, document, required_weight_entries(document));

    const std::unique_ptr<iom::ModelSource> source = load_source(dir);

    // The global final norm and both per-layer scales are rank-one in the
    // checkpoint and rank-two `[1, H]` in the published metadata. Their mapped
    // span stays the exact `2*H` bytes the loader accepted, and only the
    // logical byte count and the standard 16x16 padded size are derived.
    const std::size_t norm_indices[] = {1, 3, 4, 12, 13};
    for (const std::size_t index : norm_indices) {
        CAPTURE(index);
        const iom::TensorSpec& spec = source->tensor_spec(index);
        CHECK(spec.shape.rank() == 2);
        CHECK(dims_of(spec.shape) == std::vector<std::size_t>{1, 8});
        CHECK(spec.logical_nbytes() == 16);
        CHECK(spec.tiled_storage_nbytes() == 512);
    }

    // Nonsquare projections keep the checkpoint's `[out, in]` orientation.
    CHECK(dims_of(source->tensor_spec(6).shape) ==
          std::vector<std::size_t>{4, 8});
    CHECK(dims_of(source->tensor_spec(11).shape) ==
          std::vector<std::size_t>{8, 12});
    CHECK(dims_of(source->tensor_spec(9).shape) ==
          std::vector<std::size_t>{12, 8});
    CHECK(source->tensor_spec(6).logical_nbytes() == 4 * 8 * 2);
    CHECK(source->tensor_spec(11).logical_nbytes() == 8 * 12 * 2);
}

TEST_CASE("Model loading source ignores valid extras and unrelated documents") {
    const nlohmann::json document = two_layer_config();
    const TempDir baseline("source-baseline");
    write_checkpoint(baseline, document, required_weight_entries(document));
    const std::unique_ptr<iom::ModelSource> baseline_source =
            load_source(baseline);

    const TempDir variant("source-extras");
    write_config(variant.path(), document);
    std::vector<SafetensorsEntry> entries = required_weight_entries(document);
    entries.push_back(SafetensorsEntry{"model.layers.0.self_attn.rotary_emb.inv_freq",
                                       "F32", {2}, deterministic_payload(8, 401)});
    entries.push_back(SafetensorsEntry{"extra.u8.table", "U8", {5},
                                       deterministic_payload(5, 402)});
    entries.push_back(bf16_entry("model.extra.unused.weight", {8, 8}, 403));
    write_safetensors_file(variant.path(), "model.safetensors", entries);
    // A second shard of unrelated tensors is valid and changes nothing either.
    write_safetensors_file(variant.path(), "extra-shard.safetensors",
                           {bf16_entry("shard.only.extra", {2, 2}, 404)});
    write_file(variant.path(), "tokenizer.json", "not json at all");
    write_file(variant.path(), "generation_config.json", "{\"do_sample\":true}");

    const std::unique_ptr<iom::ModelSource> source = load_source(variant);

    CHECK(source->weights().size() == baseline_source->weights().size());
    check_inventory(*source, expected_inventory(2));
    CHECK(same_config(source->config(), baseline_source->config()));
}

TEST_CASE("Model loading source rejects an empty or incomplete store before naming weights") {
    // No SafeTensors artifact at all.
    const TempDir empty("source-empty");
    write_config(empty.path(), two_layer_config());
    const std::string empty_message = rejected_source_message(empty.path());
    CHECK(empty_message.find(empty.path().string()) != std::string::npos);

    // A decoy artifact with another extension is not a store.
    write_file(empty.path(), "model.bin", "garbage checkpoint payload");
    CHECK_THROWS_AS(iom::load_tinyllama_safetensors(empty.path()),
                    std::invalid_argument);

    // Only the three global weights are present.
    const TempDir partial("source-partial");
    const nlohmann::json document = two_layer_config();
    write_config(partial.path(), document);
    std::vector<SafetensorsEntry> globals = required_weight_entries(document);
    globals.resize(3);
    write_safetensors_file(partial.path(), "model.safetensors", globals);
    const std::string partial_message = rejected_source_message(partial.path());
    CHECK(partial_message.find(partial.path().string()) != std::string::npos);
    CHECK(partial_message.find("21") != std::string::npos);
    CHECK(partial_message.find("model.embed_tokens.weight") == std::string::npos);

    // An empty store is rejected as empty even when the configuration itself
    // cannot be sized: only the checked required count precedes the mapping.
    const TempDir unsized("source-empty-unsized");
    nlohmann::json huge = two_layer_config();
    huge["hidden_size"] = 4611686018427387904ULL;  // 2^62
    huge["num_attention_heads"] = 1;
    huge["num_key_value_heads"] = 1;
    write_config(unsized.path(), huge);
    CHECK_THROWS_AS(iom::load_tinyllama_safetensors(unsized.path()),
                    std::invalid_argument);
}

TEST_CASE("Model loading source rejects a missing required weight by its checkpoint name") {
    const nlohmann::json document = two_layer_config();

    // The tensor count stays complete, so this rejection is the required-name
    // schema check rather than the incomplete-store guard.
    const TempDir layer_dir("source-missing-layer");
    write_config(layer_dir.path(), document);
    std::vector<SafetensorsEntry> entries = required_weight_entries(document);
    erase_entry(entries, "model.layers.1.mlp.down_proj.weight");
    entries.push_back(
            bf16_entry("model.layers.1.mlp.down_proj.unused", {8, 12}, 501));
    write_safetensors_file(layer_dir.path(), "model.safetensors", entries);
    const std::string message = rejected_source_message(layer_dir.path());
    CHECK(message.find("model.layers.1.mlp.down_proj.weight") !=
          std::string::npos);
    CHECK(message.find(layer_dir.path().string()) != std::string::npos);
    CHECK(message.find("<missing>") != std::string::npos);

    // A missing global role is rejected the same way.
    const TempDir global_dir("source-missing-global");
    write_config(global_dir.path(), document);
    std::vector<SafetensorsEntry> global_entries =
            required_weight_entries(document);
    erase_entry(global_entries, "lm_head.weight");
    global_entries.push_back(bf16_entry("lm_head.unused", {19, 8}, 502));
    write_safetensors_file(global_dir.path(), "model.safetensors",
                           global_entries);
    const std::string global_message =
            rejected_source_message(global_dir.path());
    CHECK(global_message.find("lm_head.weight") != std::string::npos);
    CHECK(global_message.find("<missing>") != std::string::npos);
}

TEST_CASE("Model loading source rejects wrong rank, shape, and dtype with schema context") {
    const nlohmann::json document = two_layer_config();

    // Rank: a rank-two normalization vector instead of the checkpoint's
    // rank-one form, in a container the parser accepts.
    const TempDir rank_dir("source-rank");
    write_config(rank_dir.path(), document);
    std::vector<SafetensorsEntry> rank_entries = required_weight_entries(document);
    replace_entry(rank_entries, "model.norm.weight",
                  bf16_entry("model.norm.weight", {1, 8}, 601));
    write_safetensors_file(rank_dir.path(), "model.safetensors", rank_entries);
    const std::string rank_message = rejected_source_message(rank_dir.path());
    CHECK(rank_message.find("model.norm.weight") != std::string::npos);
    CHECK(rank_message.find("rank 1") != std::string::npos);
    CHECK(rank_message.find("rank 2") != std::string::npos);
    CHECK(rank_message.find("[8]") != std::string::npos);
    CHECK(rank_message.find("[1, 8]") != std::string::npos);

    // Shape: a grouped key/value projection of the wrong source extent.
    const TempDir shape_dir("source-shape");
    write_config(shape_dir.path(), document);
    std::vector<SafetensorsEntry> shape_entries =
            required_weight_entries(document);
    replace_entry(shape_entries, "model.layers.0.self_attn.k_proj.weight",
                  bf16_entry("model.layers.0.self_attn.k_proj.weight", {5, 8},
                             602));
    write_safetensors_file(shape_dir.path(), "model.safetensors", shape_entries);
    const std::string shape_message = rejected_source_message(shape_dir.path());
    CHECK(shape_message.find("model.layers.0.self_attn.k_proj.weight") !=
          std::string::npos);
    CHECK(shape_message.find("[5, 8]") != std::string::npos);
    CHECK(shape_message.find("[4, 8]") != std::string::npos);

    // Dtype: the required shape in another encoding is still rejected.
    const TempDir dtype_dir("source-dtype");
    write_config(dtype_dir.path(), document);
    std::vector<SafetensorsEntry> dtype_entries =
            required_weight_entries(document);
    replace_entry(dtype_entries, "model.layers.0.self_attn.q_proj.weight",
                  SafetensorsEntry{"model.layers.0.self_attn.q_proj.weight",
                                   "F32", {8, 8},
                                   deterministic_payload(8 * 8 * 4, 603)});
    write_safetensors_file(dtype_dir.path(), "model.safetensors", dtype_entries);
    const std::string dtype_message = rejected_source_message(dtype_dir.path());
    CHECK(dtype_message.find("model.layers.0.self_attn.q_proj.weight") !=
          std::string::npos);
    CHECK(dtype_message.find("F32") != std::string::npos);
    CHECK(dtype_message.find("BF16") != std::string::npos);
}

TEST_CASE("Model loading source preserves parser behavior for malformed extras and containers") {
    const nlohmann::json document = two_layer_config();

    // An extra tensor whose payload does not match its declared shape is a
    // parser failure, so required-name filtering cannot hide it.
    const TempDir payload_dir("source-extra-payload");
    write_config(payload_dir.path(), document);
    std::vector<SafetensorsEntry> payload_entries =
            required_weight_entries(document);
    payload_entries.push_back(SafetensorsEntry{"extra.bad.payload", "BF16",
                                               {4, 4},
                                               deterministic_payload(10, 701)});
    write_safetensors_file(payload_dir.path(), "model.safetensors",
                           payload_entries);
    const std::string payload_message =
            rejected_container_message(payload_dir.path());
    CHECK(payload_message.find("extra.bad.payload") != std::string::npos);

    // An extra tensor with an unknown dtype spelling stays a parser failure.
    const TempDir extra_dtype_dir("source-extra-dtype");
    write_config(extra_dtype_dir.path(), document);
    std::vector<SafetensorsEntry> extra_dtype_entries =
            required_weight_entries(document);
    extra_dtype_entries.push_back(SafetensorsEntry{"extra.unknown.dtype", "NOPE",
                                                   {2},
                                                   deterministic_payload(4, 702)});
    write_safetensors_file(extra_dtype_dir.path(), "model.safetensors",
                           extra_dtype_entries);
    const std::string extra_dtype_message =
            rejected_container_message(extra_dtype_dir.path());
    CHECK(extra_dtype_message.find("NOPE") != std::string::npos);

    // The same required name in two shards stays a duplicate-shard failure.
    const TempDir duplicate_dir("source-duplicate");
    write_config(duplicate_dir.path(), document);
    write_safetensors_file(duplicate_dir.path(), "model.safetensors",
                           required_weight_entries(document));
    write_safetensors_file(
            duplicate_dir.path(), "model-00002.safetensors",
            {bf16_entry("model.embed_tokens.weight", {19, 8}, 703)});
    const std::string duplicate_message =
            rejected_container_message(duplicate_dir.path());
    CHECK(duplicate_message.find("duplicate") != std::string::npos);
    CHECK(duplicate_message.find("model.embed_tokens.weight") !=
          std::string::npos);

    // A container header that is not valid JSON is translated into a
    // contextual container failure at the model boundary.
    const TempDir header_dir("source-container-header");
    write_config(header_dir.path(), document);
    write_safetensors_header(header_dir.path(), "model.safetensors",
                             "{ not json", deterministic_payload(4, 704));
    const std::string header_message =
            rejected_container_message(header_dir.path());
    CHECK(header_message.find(header_dir.path().string()) != std::string::npos);

    // A header that is valid JSON but not an object keeps the parser's own
    // failure.
    const TempDir object_dir("source-container-object");
    write_config(object_dir.path(), document);
    write_safetensors_header(object_dir.path(), "model.safetensors", "[1, 2]",
                             deterministic_payload(4, 705));
    const std::string object_message =
            rejected_container_message(object_dir.path());
    CHECK(object_message.find("invalid safetensors header") != std::string::npos);
}

TEST_CASE("Model loading source rejects impossible weight sizes without allocating a checkpoint") {
    // Every configuration below passes configuration validation and fails
    // exactly one checked weight-sizing bound: the required tensor count, an
    // element product, a logical byte count, a standard tiled byte count, and
    // the repeated-layer byte total. Each directory carries one minimal
    // temporary SafeTensors tensor, so no huge checkpoint allocation and no
    // generated layer name list is ever involved.
    struct ImpossibleSize {
        const char* bound;
        nlohmann::json document;
    };

    std::vector<ImpossibleSize> cases;
    {
        nlohmann::json document = two_layer_config();
        document["num_hidden_layers"] = 2305843009213693952ULL;  // 2^61
        cases.push_back({"required tensor count", std::move(document)});
    }
    {
        nlohmann::json document = two_layer_config();
        document["hidden_size"] = 4611686018427387904ULL;  // 2^62
        document["num_attention_heads"] = 1;
        document["num_key_value_heads"] = 1;
        cases.push_back({"element count", std::move(document)});
    }
    {
        nlohmann::json document = two_layer_config();
        document["hidden_size"] = 600000000000000000ULL;  // 6e17
        document["num_attention_heads"] = 1;
        document["num_key_value_heads"] = 1;
        cases.push_back({"logical byte count", std::move(document)});
    }
    {
        nlohmann::json document = two_layer_config();
        document["vocab_size"] = 3;
        document["hidden_size"] = 72057594037927922ULL;  // 16 * 2^52 + 2
        document["num_attention_heads"] = 1;
        document["num_key_value_heads"] = 1;
        cases.push_back({"standard tiled byte count", std::move(document)});
    }
    {
        nlohmann::json document = two_layer_config();
        document["num_hidden_layers"] = 1152921504606846976ULL;  // 2^60
        cases.push_back({"repeated-layer byte total", std::move(document)});
    }

    for (const ImpossibleSize& test_case : cases) {
        CAPTURE(test_case.bound);
        CAPTURE(test_case.document.dump());
        TempDir dir("source-impossible");
        write_config(dir.path(), test_case.document);
        write_safetensors_file(dir.path(), "model.safetensors",
                               {bf16_entry("extra.dummy.weight", {1, 1}, 900)});

        CHECK_THROWS_AS(iom::load_tinyllama_safetensors(dir.path()),
                        std::overflow_error);
    }
}

TEST_CASE("Model loading source keeps published metadata valid after the checkpoint artifacts disappear") {
    const TempDir dir("source-retained");
    const nlohmann::json document = two_layer_config();
    write_checkpoint(dir, document, required_weight_entries(document));

    const std::unique_ptr<iom::ModelSource> source = load_source(dir);

    // The factory owns a mapping that outlives the files: removing the
    // artifacts after the complete inventory was published changes neither the
    // configuration nor the inventory metadata.
    std::error_code error;
    REQUIRE(std::filesystem::remove(dir.path() / "model.safetensors", error));
    REQUIRE(std::filesystem::remove(dir.path() / "config.json", error));

    CHECK(source->config().hidden_size == 8);
    CHECK(source->weights().size() == 21);
    check_inventory(*source, expected_inventory(2));

    // A new load of the same directory now fails: nothing is re-read lazily.
    CHECK_THROWS_AS(iom::load_tinyllama_safetensors(dir.path()),
                    std::runtime_error);
}

// ---------------------------------------------------------------------------
// Upload preflight
// ---------------------------------------------------------------------------

// Loads the complete two-layer boundary checkpoint every preflight case uses.
std::unique_ptr<iom::ModelSource> load_preflight_source(const TempDir& dir) {
    const nlohmann::json document = two_layer_config();
    write_checkpoint(dir, document, required_weight_entries(document));
    return load_source(dir);
}

TEST_CASE("Model loading upload preflight returns the maximum requirement of the complete binding") {
    const TempDir dir("preflight-maximum");
    const std::unique_ptr<iom::ModelSource> source = load_preflight_source(dir);

    FakeDevice device;
    std::vector<std::unique_ptr<iom::Tensor>> destinations =
            make_destinations(*source, device, WorkspacePolicy{2, 32});

    // One small role demands a wider alignment than the largest role, so the
    // byte and alignment maxima come from different owners.
    device.set_workspace_policy(WorkspacePolicy{1, 64});
    destinations[1] = device.create_tensor(source->tensor_spec(1));
    std::vector<iom::Tensor*> pointers = destination_pointers(destinations);

    const iom::WorkspaceRequirements requirements =
            source->upload_workspace_requirements(
                    device, as_destinations(pointers));

    // token_embedding and lm_head are the largest roles: 19 * 8 * 2 = 304
    // logical bytes, doubled by the fixture policy. Summing the mutually
    // exclusive per-role requirements would exceed 3000 bytes, so 608 can only
    // be the actual maximum of the owner queries.
    CHECK(requirements.bytes == 608);
    CHECK(requirements.alignment == 64);

    // Every owner was queried exactly once, and no owner received an upload or
    // changed its content.
    for (std::size_t index = 0; index < destinations.size(); ++index) {
        CAPTURE(index);
        CHECK(observed(destinations, index).requirement_queries() == 1);
        CHECK(observed(destinations, index).from_host_calls() == 0);
        CHECK(observed(destinations, index).uploaded_bytes().empty());
        CHECK(observed(destinations, index).storage() == sentinel_storage());
    }
    CHECK(device.workspace_creations() == 0);

    // The same complete list reports the same maxima again, and neither query
    // changed the published source metadata.
    const iom::WorkspaceRequirements repeated =
            source->upload_workspace_requirements(
                    device, as_destinations(pointers));
    CHECK(repeated == requirements);
    CHECK(source->weights().size() == 21);
    CHECK(dims_of(source->tensor_spec(20).shape) ==
          std::vector<std::size_t>{8, 12});
    CHECK(source->config().hidden_size == 8);
}

TEST_CASE("Model loading upload preflight keeps a zero-workspace binding at {0, 1}") {
    const TempDir dir("preflight-zero");
    const std::unique_ptr<iom::ModelSource> source = load_preflight_source(dir);

    FakeDevice device;
    std::vector<std::unique_ptr<iom::Tensor>> destinations =
            make_destinations(*source, device);
    std::vector<iom::Tensor*> pointers = destination_pointers(destinations);

    const iom::WorkspaceRequirements requirements =
            source->upload_workspace_requirements(
                    device, as_destinations(pointers));

    // The CPU/TTNN zero-workspace policy is reported exactly, never as a
    // positive allocation request.
    const iom::WorkspaceRequirements no_scratch{0, 1};
    CHECK(requirements == no_scratch);
    CHECK(requirements.bytes == 0);
    CHECK(requirements.alignment == 1);
    CHECK(device.workspace_creations() == 0);
    for (std::size_t index = 0; index < destinations.size(); ++index) {
        CAPTURE(index);
        CHECK(observed(destinations, index).requirement_queries() == 1);
        CHECK(observed(destinations, index).from_host_calls() == 0);
    }
}

TEST_CASE("Model loading upload preflight rejects an invalid binding before any owner query") {
    const TempDir dir("preflight-reject");
    const std::unique_ptr<iom::ModelSource> source = load_preflight_source(dir);

    FakeDevice device;
    std::vector<std::unique_ptr<iom::Tensor>> destinations =
            make_destinations(*source, device);
    std::vector<iom::Tensor*> pointers = destination_pointers(destinations);

    // One rejected preflight: the category is fixed and no owner was queried,
    // uploaded to, or provisioned for.
    const auto rejected =
            [&device](const iom::ModelSource& candidate,
                      std::span<iom::Tensor* const> binding,
                      const std::vector<std::unique_ptr<iom::Tensor>>& owners,
                      const char* what) {
                CAPTURE(what);
                CHECK_THROWS_AS(
                        (void)candidate.upload_workspace_requirements(
                                device, binding),
                        std::invalid_argument);
                for (std::size_t index = 0; index < owners.size(); ++index) {
                    CHECK(observed(owners, index).requirement_queries() == 0);
                    CHECK(observed(owners, index).from_host_calls() == 0);
                }
                CHECK(device.workspace_creations() == 0);
            };

    // A null owner at the last published index.
    std::vector<iom::Tensor*> null_list = pointers;
    null_list.back() = nullptr;
    rejected(*source, as_destinations(null_list), destinations,
             "null destination");

    // One destination too few, and one too many.
    std::vector<iom::Tensor*> short_list(pointers.begin(), pointers.end() - 1);
    rejected(*source, as_destinations(short_list), destinations,
             "missing destination");
    const std::unique_ptr<iom::Tensor> excess =
            device.create_tensor(source->tensor_spec(0));
    std::vector<iom::Tensor*> long_list = pointers;
    long_list.push_back(excess.get());
    rejected(*source, as_destinations(long_list), destinations,
             "excess destination");

    // One owner bound to two roles whose logical metadata is equal: both are
    // `[1, 8]` normalization scales, so only pointer identity can reject this.
    REQUIRE(source->tensor_spec(3) == source->tensor_spec(1));
    std::vector<iom::Tensor*> repeated = pointers;
    repeated[3] = repeated[1];
    rejected(*source, as_destinations(repeated), destinations,
             "repeated owner");

    // A wrong specification at the last published index.
    std::vector<std::unique_ptr<iom::Tensor>> wrong =
            make_destinations(*source, device);
    wrong.back() = device.create_tensor(iom::TensorSpec{
            iom::TensorShape{{8, 11}}, iom::DataType::BF16,
            iom::QuantizationFormat::NONE});
    std::vector<iom::Tensor*> wrong_pointers = destination_pointers(wrong);
    rejected(*source, as_destinations(wrong_pointers), wrong,
             "wrong final specification");

    // The complete first binding stayed unqueried through every rejection.
    for (std::size_t index = 0; index < destinations.size(); ++index) {
        CAPTURE(index);
        CHECK(observed(destinations, index).requirement_queries() == 0);
        CHECK(observed(destinations, index).storage() == sentinel_storage());
    }
}

TEST_CASE("Model loading upload preflight rejects a foreign device instance and an absent BF16 capability") {
    const TempDir dir("preflight-device");
    const std::unique_ptr<iom::ModelSource> source = load_preflight_source(dir);

    FakeDevice device;
    std::vector<std::unique_ptr<iom::Tensor>> destinations =
            make_destinations(*source, device);
    std::vector<iom::Tensor*> pointers = destination_pointers(destinations);

    // Two instances report the same backend kind and ordinal, so only exact
    // Device identity separates them.
    FakeDevice foreign;
    CHECK(foreign.backend_kind() == device.backend_kind());
    CHECK(foreign.backend_device() == device.backend_device());
    CHECK_THROWS_AS((void)source->upload_workspace_requirements(
                            foreign, as_destinations(pointers)),
                    std::invalid_argument);
    CHECK(device.workspace_creations() == 0);
    for (std::size_t index = 0; index < destinations.size(); ++index) {
        CAPTURE(index);
        CHECK(observed(destinations, index).requirement_queries() == 0);
    }

    // The device that created the binding accepts it.
    const iom::WorkspaceRequirements no_scratch{0, 1};
    CHECK(source->upload_workspace_requirements(
                  device, as_destinations(pointers)) == no_scratch);

    // A device whose storage capability omits BF16 cannot receive a published
    // weight even when it owns every destination.
    FakeDevice incapable{kSupportedWithoutBf16};
    std::vector<std::unique_ptr<iom::Tensor>> incapable_destinations =
            make_destinations(*source, incapable);
    std::vector<iom::Tensor*> incapable_pointers =
            destination_pointers(incapable_destinations);
    CHECK_THROWS_AS((void)source->upload_workspace_requirements(
                            incapable, as_destinations(incapable_pointers)),
                    std::invalid_argument);
    CHECK(incapable.workspace_creations() == 0);
    for (std::size_t index = 0; index < incapable_destinations.size(); ++index) {
        CAPTURE(index);
        CHECK(observed(incapable_destinations, index).requirement_queries() == 0);
        CHECK(observed(incapable_destinations, index).from_host_calls() == 0);
    }
}

TEST_CASE("Model loading upload preflight propagates checked sizing overflow without allocating storage") {
    const TempDir dir("preflight-overflow");
    const std::unique_ptr<iom::ModelSource> source = load_preflight_source(dir);

    FakeDevice device;
    std::vector<std::unique_ptr<iom::Tensor>> destinations =
            make_destinations(*source, device);

    // A destination whose declared shape cannot be sized at all: 2^62 BF16
    // elements overflow the checked logical bit count. The bounded owner holds
    // only its fixed sentinel, so the declaration allocates nothing.
    const iom::TensorSpec unsizeable{
            iom::TensorShape{{1, std::size_t{1} << 62}}, iom::DataType::BF16,
            iom::QuantizationFormat::NONE};
    destinations.back() = device.create_tensor(unsizeable);
    std::vector<iom::Tensor*> pointers = destination_pointers(destinations);

    CHECK_THROWS_AS((void)source->upload_workspace_requirements(
                            device, as_destinations(pointers)),
                    std::overflow_error);

    for (std::size_t index = 0; index < destinations.size(); ++index) {
        CAPTURE(index);
        CHECK(observed(destinations, index).requirement_queries() == 0);
        CHECK(observed(destinations, index).from_host_calls() == 0);
        CHECK(observed(destinations, index).storage() == sentinel_storage());
    }
    CHECK(device.workspace_creations() == 0);
    CHECK(source->weights().size() == 21);

    // The same checked failure at the first published index propagates too.
    std::vector<std::unique_ptr<iom::Tensor>> front =
            make_destinations(*source, device);
    front.front() = device.create_tensor(unsizeable);
    std::vector<iom::Tensor*> front_pointers = destination_pointers(front);
    CHECK_THROWS_AS((void)source->upload_workspace_requirements(
                            device, as_destinations(front_pointers)),
                    std::overflow_error);
}

}  // namespace