#include "iom/model.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "iom_internal.hpp"

namespace iom {
namespace {

// The loader reads exactly this file from the explicit model directory.
constexpr const char* kConfigFileName = "config.json";

// The supported checkpoint declares one HF architecture, one model type, one
// activation, and one source dtype; every other value is unsupported.
constexpr std::string_view kArchitecture = "LlamaForCausalLM";
constexpr std::string_view kModelType = "llama";
constexpr std::string_view kHiddenAct = "silu";
constexpr std::string_view kTorchDtype = "bfloat16";

// Complete accepted key set. Any other key -- including a head-dimension
// override -- is rejected instead of ignored, so an unsupported checkpoint
// cannot silently change a derived dimension.
constexpr std::string_view kSupportedKeys[] = {
        "architectures",
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "rms_norm_eps",
        "rope_theta",
        "attention_bias",
        "mlp_bias",
        "tie_word_embeddings",
        "rope_scaling",
        "pretraining_tp",
        "torch_dtype",
        "bos_token_id",
        "eos_token_id",
        // Accepted metadata that never changes configuration semantics.
        "initializer_range",
        "transformers_version",
        "use_cache",
        "pad_token_id",
};

constexpr std::string_view kPositiveInteger =
        "a positive integer representable as std::size_t";

// Every configuration rejection carries the exact file, the offending field,
// the constraint it violates, and its actual value or `<missing>`.
[[noreturn]] void reject(const std::filesystem::path& config_path,
                         std::string_view field, std::string_view requirement,
                         std::string_view actual) {
    throw std::invalid_argument("invalid TinyLlama configuration: " +
                                config_path.string() + " field '" +
                                std::string(field) + "' requires " +
                                std::string(requirement) + "; actual " +
                                std::string(actual));
}

std::string describe(const nlohmann::json& value) {
    return value.dump();
}

// Reads the configuration text. Open and read failures are contextual I/O
// failures, never missing-field rejections.
std::string read_config_text(const std::filesystem::path& config_path) {
    std::ifstream stream(config_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + config_path.string() + ": " +
                                 std::strerror(errno));
    }
    const std::string text{std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        throw std::runtime_error("cannot read " + config_path.string() + ": " +
                                 std::strerror(errno));
    }
    return text;
}

const nlohmann::json& require_field(const nlohmann::json& document,
                                    const std::filesystem::path& config_path,
                                    std::string_view field,
                                    std::string_view requirement) {
    const auto entry = document.find(std::string(field));
    if (entry == document.end()) {
        reject(config_path, field, requirement, "<missing>");
    }
    return *entry;
}

// Accepted integers are JSON integers only: booleans, strings, arrays, null,
// and every floating-point value including `1.0` are not integers.
bool is_positive_size(const nlohmann::json& value) {
    if (value.is_number_unsigned()) {
        const std::uint64_t raw = value.get<std::uint64_t>();
        return raw > 0 && raw <= std::numeric_limits<std::size_t>::max();
    }
    if (value.is_number_integer()) {
        const std::int64_t raw = value.get<std::int64_t>();
        return raw > 0 && static_cast<std::uint64_t>(raw) <=
                                  std::numeric_limits<std::size_t>::max();
    }
    return false;
}

// Valid only for a value accepted by `is_positive_size`.
std::size_t to_size(const nlohmann::json& value) {
    return value.is_number_unsigned()
                   ? static_cast<std::size_t>(value.get<std::uint64_t>())
                   : static_cast<std::size_t>(value.get<std::int64_t>());
}

std::size_t require_positive_size(const nlohmann::json& document,
                                  const std::filesystem::path& config_path,
                                  std::string_view field) {
    const nlohmann::json& value =
            require_field(document, config_path, field, kPositiveInteger);
    if (!is_positive_size(value)) {
        reject(config_path, field, kPositiveInteger, describe(value));
    }
    return to_size(value);
}

std::size_t require_exact_size(const nlohmann::json& document,
                               const std::filesystem::path& config_path,
                               std::string_view field, std::size_t expected) {
    const std::string requirement = "the integer " + std::to_string(expected);
    const nlohmann::json& value =
            require_field(document, config_path, field, requirement);
    if (!is_positive_size(value) || to_size(value) != expected) {
        reject(config_path, field, requirement, describe(value));
    }
    return expected;
}

void require_exact_string(const nlohmann::json& document,
                          const std::filesystem::path& config_path,
                          std::string_view field, std::string_view expected) {
    const std::string requirement =
            "the string \"" + std::string(expected) + "\"";
    const nlohmann::json& value =
            require_field(document, config_path, field, requirement);
    if (!value.is_string() || value.get<std::string>() != expected) {
        reject(config_path, field, requirement, describe(value));
    }
}

void require_supported_architectures(const nlohmann::json& document,
                                     const std::filesystem::path& config_path) {
    const std::string requirement =
            "the array [\"" + std::string(kArchitecture) + "\"]";
    const nlohmann::json& value = require_field(document, config_path,
                                                "architectures", requirement);
    if (!value.is_array() || value.size() != 1 ||
        !value.front().is_string() ||
        value.front().get<std::string>() != kArchitecture) {
        reject(config_path, "architectures", requirement, describe(value));
    }
}

// The epsilon must survive narrowing to `float`: a double that underflows to
// zero or overflows to infinity is not a usable normalization epsilon.
float require_positive_float(const nlohmann::json& document,
                             const std::filesystem::path& config_path,
                             std::string_view field) {
    constexpr std::string_view requirement =
            "a finite positive number that stays finite and positive as float";
    const nlohmann::json& value =
            require_field(document, config_path, field, requirement);
    if (value.is_number()) {
        const double raw = value.get<double>();
        const float narrowed = static_cast<float>(raw);
        if (std::isfinite(raw) && raw > 0.0 && std::isfinite(narrowed) &&
            narrowed > 0.0f) {
            return narrowed;
        }
    }
    reject(config_path, field, requirement, describe(value));
}

double require_positive_double(const nlohmann::json& document,
                               const std::filesystem::path& config_path,
                               std::string_view field) {
    constexpr std::string_view requirement =
            "a finite positive number representable as double";
    const nlohmann::json& value =
            require_field(document, config_path, field, requirement);
    if (value.is_number()) {
        const double raw = value.get<double>();
        if (std::isfinite(raw) && raw > 0.0) {
            return raw;
        }
    }
    reject(config_path, field, requirement, describe(value));
}

// The three bias keys default to false and accept only that exact boolean.
void require_absent_or_false(const nlohmann::json& document,
                             const std::filesystem::path& config_path,
                             std::string_view field) {
    const auto entry = document.find(std::string(field));
    if (entry == document.end()) {
        return;
    }
    if (!entry->is_boolean() || entry->get<bool>()) {
        reject(config_path, field, "either no value or the boolean false",
               describe(*entry));
    }
}

void require_absent_or_null(const nlohmann::json& document,
                            const std::filesystem::path& config_path,
                            std::string_view field) {
    const auto entry = document.find(std::string(field));
    if (entry != document.end() && !entry->is_null()) {
        reject(config_path, field, "either no value or null",
               describe(*entry));
    }
}

void require_absent_or_one(const nlohmann::json& document,
                           const std::filesystem::path& config_path,
                           std::string_view field) {
    const auto entry = document.find(std::string(field));
    if (entry == document.end()) {
        return;
    }
    if (!is_positive_size(*entry) || to_size(*entry) != 1) {
        reject(config_path, field, "either no value or the integer 1",
               describe(*entry));
    }
}

void require_supported_keys(const nlohmann::json& document,
                            const std::filesystem::path& config_path) {
    for (const auto& entry : document.items()) {
        const bool supported = std::any_of(
                std::begin(kSupportedKeys), std::end(kSupportedKeys),
                [&entry](std::string_view candidate) {
                    return candidate == entry.key();
                });
        if (!supported) {
            reject(config_path, entry.key(),
                   "a supported TinyLlama configuration key",
                   describe(entry.value()));
        }
    }
}

}  // namespace

TinyLlamaConfig load_tinyllama_config(
        const std::filesystem::path& model_directory) {
    const std::filesystem::path config_path = model_directory / kConfigFileName;
    const std::string text = read_config_text(config_path);

    nlohmann::json document;
    try {
        document = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(
                "invalid TinyLlama configuration: " + config_path.string() +
                " field '<config.json>' requires a valid JSON document; "
                "actual " +
                error.what());
    }

    // A success value is always a non-empty object carrying every field below.
    if (!document.is_object() || document.empty()) {
        reject(config_path, "<config.json>", "a non-empty JSON object",
               describe(document));
    }

    require_supported_keys(document, config_path);
    require_supported_architectures(document, config_path);
    require_exact_string(document, config_path, "model_type", kModelType);

    TinyLlamaConfig config{};
    config.num_hidden_layers = require_positive_size(
            document, config_path, "num_hidden_layers");
    config.hidden_size =
            require_positive_size(document, config_path, "hidden_size");
    config.intermediate_size =
            require_positive_size(document, config_path, "intermediate_size");
    config.num_attention_heads = require_positive_size(
            document, config_path, "num_attention_heads");
    config.num_key_value_heads = require_positive_size(
            document, config_path, "num_key_value_heads");
    config.vocab_size =
            require_positive_size(document, config_path, "vocab_size");
    config.max_position_embeddings = require_positive_size(
            document, config_path, "max_position_embeddings");

    require_exact_string(document, config_path, "hidden_act", kHiddenAct);
    config.rms_norm_eps =
            require_positive_float(document, config_path, "rms_norm_eps");
    config.rope_theta =
            require_positive_double(document, config_path, "rope_theta");

    require_absent_or_false(document, config_path, "attention_bias");
    require_absent_or_false(document, config_path, "mlp_bias");
    require_absent_or_false(document, config_path, "tie_word_embeddings");
    require_absent_or_null(document, config_path, "rope_scaling");
    require_absent_or_one(document, config_path, "pretraining_tp");
    require_exact_string(document, config_path, "torch_dtype", kTorchDtype);

    config.bos_token_id =
            require_exact_size(document, config_path, "bos_token_id", 1);
    config.eos_token_id =
            require_exact_size(document, config_path, "eos_token_id", 2);
    if (config.bos_token_id >= config.vocab_size) {
        reject(config_path, "bos_token_id",
               "an id below vocab_size " + std::to_string(config.vocab_size),
               std::to_string(config.bos_token_id));
    }
    if (config.eos_token_id >= config.vocab_size) {
        reject(config_path, "eos_token_id",
               "an id below vocab_size " + std::to_string(config.vocab_size),
               std::to_string(config.eos_token_id));
    }

    // The head plan is derived, never read: H divides by Hq, Hq divides by
    // Hkv, and the per-head width D = H / Hq must stay positive and even.
    if (config.hidden_size % config.num_attention_heads != 0) {
        reject(config_path, "hidden_size",
               "a multiple of num_attention_heads " +
                       std::to_string(config.num_attention_heads),
               std::to_string(config.hidden_size));
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        reject(config_path, "num_attention_heads",
               "a multiple of num_key_value_heads " +
                       std::to_string(config.num_key_value_heads),
               std::to_string(config.num_attention_heads));
    }
    config.head_dim = config.hidden_size / config.num_attention_heads;
    if (config.head_dim % 2 != 0) {
        reject(config_path, "hidden_size",
               "a positive even per-head width (hidden_size / "
               "num_attention_heads = " +
                       std::to_string(config.head_dim) + ")",
               std::to_string(config.hidden_size));
    }

    // The grouped K/V width is a derived checkpoint dimension with no stored
    // field, so it is checked here instead of at first use.
    static_cast<void>(
            detail::checked_mul(config.num_key_value_heads, config.head_dim,
                                "TinyLlama configuration num_key_value_heads "
                                "* head_dim overflows"));

    return config;
}

}  // namespace iom