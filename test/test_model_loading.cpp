#include <doctest/doctest.h>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "model_loading_fixture.hpp"

namespace {

using iom_model_loading::TempDir;
using iom_model_loading::one_layer_config;
using iom_model_loading::rejected_config_message;
using iom_model_loading::same_config;
using iom_model_loading::two_layer_config;
using iom_model_loading::write_config;
using iom_model_loading::write_file;

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

}  // namespace