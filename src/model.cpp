#include "iom/model.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "iom/safetensors.hpp"
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

// ---------------------------------------------------------------------------
// Mapped weight source
// ---------------------------------------------------------------------------

// Only BF16 with no grouped quantization is selected today; the selected
// encoding is declared once so the schema plan and every published entry
// cannot disagree about it.
constexpr DataType kSelectedDataType = DataType::BF16;
constexpr QuantizationFormat kSelectedQuantization = QuantizationFormat::NONE;

// Layer-scoped checkpoint names are `model.layers.{l}.<suffix>`.
constexpr std::string_view kLayerPrefix = "model.layers.";

constexpr const char* kRequiredCountOverflow =
        "TinyLlama weight schema required tensor count overflows";
constexpr const char* kWeightBytesOverflow =
        "TinyLlama weight schema weight byte total overflows";
constexpr const char* kGroupedWidthOverflow =
        "TinyLlama weight schema grouped key/value width overflows";

// One required role of the validated configuration. `source_shape` is the
// exact rank/shape the checkpoint must declare; `spec` is the selected
// logical destination metadata, which differs from the source only for the
// rank-one normalization vectors.
struct WeightRoleSchema {
    ModelWeightRole role;
    std::string_view name;
    bool layer_scoped;
    std::vector<std::size_t> source_shape;
    TensorSpec spec;
};

// Counts and byte totals of the complete required schema. Every value is
// checked while the plan is built, before any inventory metadata is expanded.
struct WeightPlan {
    std::size_t required_count = 0;
    std::size_t aggregate_logical_nbytes = 0;
    std::size_t aggregate_tiled_nbytes = 0;
    std::vector<WeightRoleSchema> globals;
    std::vector<WeightRoleSchema> layer_roles;
};

TensorSpec selected_spec(TensorShape logical_shape) {
    return TensorSpec{std::move(logical_shape), kSelectedDataType,
                      kSelectedQuantization};
}

// A checkpoint normalization vector is stored rank-one `[H]`. Only the
// logical metadata becomes `[1, H]`: the mapped span stays exactly `2*H`
// bytes, and no rank-one shape, final-axis view, transpose, or pre-tiled
// host copy is exposed.
std::vector<std::size_t> adapted_logical_shape(
        std::vector<std::size_t> source_shape) {
    if (source_shape.size() == 1) {
        source_shape.insert(source_shape.begin(), 1);
    }
    return source_shape;
}

WeightRoleSchema make_role_schema(ModelWeightRole role, std::string_view name,
                                  bool layer_scoped,
                                  std::vector<std::size_t> source_shape) {
    TensorSpec spec = selected_spec(
            TensorShape{adapted_logical_shape(source_shape)});
    return WeightRoleSchema{role, name, layer_scoped, std::move(source_shape),
                            std::move(spec)};
}

struct SchemaTotals {
    std::size_t logical_nbytes = 0;
    std::size_t tiled_nbytes = 0;
};

// Element, logical, and standard 16x16 tiled sizes of every role are checked
// with checked arithmetic before the corresponding metadata is expanded.
SchemaTotals checked_schema_totals(
        const std::vector<WeightRoleSchema>& schemas) {
    SchemaTotals totals;
    for (const WeightRoleSchema& schema : schemas) {
        static_cast<void>(schema.spec.shape.element_count());
        totals.logical_nbytes = detail::checked_add(
                totals.logical_nbytes, schema.spec.logical_nbytes(),
                kWeightBytesOverflow);
        totals.tiled_nbytes = detail::checked_add(
                totals.tiled_nbytes, schema.spec.tiled_storage_nbytes(),
                kWeightBytesOverflow);
    }
    return totals;
}

// The checked required tensor count `3 + 9*N` of a validated configuration.
// It is computed before any weight name exists.
std::size_t checked_required_count(const TinyLlamaConfig& config) {
    return detail::checked_add(
            3,
            detail::checked_mul(9, config.num_hidden_layers,
                                kRequiredCountOverflow),
            kRequiredCountOverflow);
}

// The complete required schema of a validated configuration. The repeated
// layers are bounded from the nine representative role sizes and the layer
// count instead of generating one name list per possible layer, so an
// impossible configuration fails without a huge source or name list.
WeightPlan build_weight_plan(const TinyLlamaConfig& config,
                             std::size_t required_count) {
    const std::size_t kv_width = detail::checked_mul(
            config.num_key_value_heads, config.head_dim,
            kGroupedWidthOverflow);

    WeightPlan plan;
    plan.required_count = required_count;

    plan.globals = {
            make_role_schema(ModelWeightRole::token_embedding,
                             "model.embed_tokens.weight", false,
                             {config.vocab_size, config.hidden_size}),
            make_role_schema(ModelWeightRole::final_norm, "model.norm.weight",
                             false, {config.hidden_size}),
            make_role_schema(ModelWeightRole::lm_head, "lm_head.weight", false,
                             {config.vocab_size, config.hidden_size}),
    };
    plan.layer_roles = {
            make_role_schema(ModelWeightRole::input_norm,
                             "input_layernorm.weight", true,
                             {config.hidden_size}),
            make_role_schema(ModelWeightRole::post_attention_norm,
                             "post_attention_layernorm.weight", true,
                             {config.hidden_size}),
            make_role_schema(ModelWeightRole::query,
                             "self_attn.q_proj.weight", true,
                             {config.hidden_size, config.hidden_size}),
            make_role_schema(ModelWeightRole::key,
                             "self_attn.k_proj.weight", true,
                             {kv_width, config.hidden_size}),
            make_role_schema(ModelWeightRole::value,
                             "self_attn.v_proj.weight", true,
                             {kv_width, config.hidden_size}),
            make_role_schema(ModelWeightRole::attention_output,
                             "self_attn.o_proj.weight", true,
                             {config.hidden_size, config.hidden_size}),
            make_role_schema(ModelWeightRole::mlp_gate,
                             "mlp.gate_proj.weight", true,
                             {config.intermediate_size, config.hidden_size}),
            make_role_schema(ModelWeightRole::mlp_up, "mlp.up_proj.weight",
                             true,
                             {config.intermediate_size, config.hidden_size}),
            make_role_schema(ModelWeightRole::mlp_down, "mlp.down_proj.weight",
                             true,
                             {config.hidden_size, config.intermediate_size}),
    };

    const SchemaTotals globals = checked_schema_totals(plan.globals);
    const SchemaTotals layers = checked_schema_totals(plan.layer_roles);
    plan.aggregate_logical_nbytes = detail::checked_add(
            globals.logical_nbytes,
            detail::checked_mul(layers.logical_nbytes,
                                config.num_hidden_layers,
                                kWeightBytesOverflow),
            kWeightBytesOverflow);
    plan.aggregate_tiled_nbytes = detail::checked_add(
            globals.tiled_nbytes,
            detail::checked_mul(layers.tiled_nbytes,
                                config.num_hidden_layers,
                                kWeightBytesOverflow),
            kWeightBytesOverflow);
    return plan;
}

// Diagnostics name the exact spelling the SafeTensors container uses.
std::string_view dtype_name(DataType type) {
    switch (type) {
        case DataType::BOOL: return "BOOL";
        case DataType::I2: return "I2";
        case DataType::U2: return "U2";
        case DataType::I4: return "I4";
        case DataType::U4: return "U4";
        case DataType::I8: return "I8";
        case DataType::U8: return "U8";
        case DataType::I16: return "I16";
        case DataType::U16: return "U16";
        case DataType::I32: return "I32";
        case DataType::U32: return "U32";
        case DataType::I64: return "I64";
        case DataType::U64: return "U64";
        case DataType::F4_E2M1: return "F4";
        case DataType::F6_E2M3: return "F6_E2M3";
        case DataType::F6_E3M2: return "F6_E3M2";
        case DataType::F8_E4M3FN: return "F8_E4M3";
        case DataType::F8_E5M2: return "F8_E5M2";
        case DataType::F8_E8M0: return "F8_E8M0";
        case DataType::F16: return "F16";
        case DataType::BF16: return "BF16";
        case DataType::F32: return "F32";
        case DataType::F64: return "F64";
    }
    return "unknown";
}

std::string describe_shape(const std::vector<std::size_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            text += ", ";
        }
        text += std::to_string(shape[i]);
    }
    return text + "]";
}

std::string describe_source(const std::vector<std::size_t>& shape) {
    return "rank " + std::to_string(shape.size()) + " source shape " +
           describe_shape(shape) + " and dtype BF16";
}

std::string describe_view(const SafeTensorView& view) {
    return "rank " + std::to_string(view.shape().size()) + " source shape " +
           describe_shape(view.shape()) + " and dtype " +
           std::string(dtype_name(view.dtype()));
}

// Every weight-schema rejection names the model directory, the logical
// checkpoint key, the required rank/shape/dtype, and the actual one.
[[noreturn]] void reject_weight(const std::filesystem::path& directory,
                                const std::string& name,
                                const std::string& requirement,
                                const std::string& actual) {
    throw std::invalid_argument("invalid TinyLlama weight schema: " +
                                directory.string() + " tensor '" + name +
                                "' requires " + requirement + "; actual " +
                                actual);
}

[[noreturn]] void reject_empty_store(
        const std::filesystem::path& directory) {
    throw std::invalid_argument(
            "invalid TinyLlama weight schema: " + directory.string() +
            " contains no SafeTensors tensors; the validated configuration "
            "requires a complete checkpoint");
}

// The required-count guard rejects an obviously incomplete store before any
// layer or weight name is enumerated.
[[noreturn]] void reject_incomplete_store(const std::filesystem::path& directory,
                                          std::size_t found,
                                          const WeightPlan& plan) {
    throw std::invalid_argument(
            "invalid TinyLlama weight schema: " + directory.string() +
            " contains " + std::to_string(found) +
            " SafeTensors tensors, but the validated configuration requires " +
            std::to_string(plan.required_count) + " (" +
            std::to_string(plan.aggregate_logical_nbytes) +
            " logical bytes, " +
            std::to_string(plan.aggregate_tiled_nbytes) +
            " standard tiled bytes) before any required weight is named");
}

std::string weight_name(const WeightRoleSchema& schema,
                        std::optional<std::size_t> layer) {
    if (!schema.layer_scoped) {
        return std::string(schema.name);
    }
    return std::string(kLayerPrefix) + std::to_string(*layer) + "." +
           std::string(schema.name);
}

// A required name that the validated container does not carry is a schema
// rejection, not the container's missing-name range error.
SafeTensorView required_view(const SafeTensorsStore& store,
                             const std::filesystem::path& directory,
                             const std::string& name,
                             const WeightRoleSchema& schema) {
    try {
        return store[name];
    } catch (const std::out_of_range&) {
        reject_weight(directory, name, describe_source(schema.source_shape),
                      "<missing>");
    }
}

// Validates one required role against the parsed source and appends its
// independent selected logical metadata.
void append_selected(const SafeTensorsStore& store,
                     const std::filesystem::path& directory,
                     const WeightRoleSchema& schema,
                     std::optional<std::size_t> layer,
                     std::vector<ModelWeightInfo>& weights,
                     std::vector<TensorSpec>& specs) {
    const std::string name = weight_name(schema, layer);
    const SafeTensorView view = required_view(store, directory, name, schema);
    const std::string requirement = describe_source(schema.source_shape);

    // Rank, exact shape, and dtype are compared as declared; the mapped span
    // must then be exactly the checked logical byte length of the selected
    // metadata, which for a BF16 role is `2 * product(shape)`.
    if (view.shape() != schema.source_shape ||
        view.dtype() != kSelectedDataType) {
        reject_weight(directory, name, requirement, describe_view(view));
    }
    if (view.nbytes() != schema.spec.logical_nbytes()) {
        reject_weight(directory, name, requirement,
                      describe_view(view) + " covering " +
                              std::to_string(view.nbytes()) + " bytes");
    }

    const std::span<const std::size_t> dimensions =
            schema.spec.shape.dimensions();
    weights.push_back(ModelWeightInfo{
            ModelWeightId{schema.role, layer},
            TensorShape{std::vector<std::size_t>(dimensions.begin(),
                                                 dimensions.end())}});
    // Each entry owns its own logical metadata, independent of every other
    // entry and of the retained mapping.
    specs.push_back(selected_spec(
            TensorShape{std::vector<std::size_t>(dimensions.begin(),
                                                 dimensions.end())}));
}

// Only a JSON-library escape from the container parser is translated here:
// the standalone SafeTensors API keeps its own behavior, while this boundary
// reports a contextual container runtime failure.
std::unique_ptr<SafeTensorsDir> open_mapped_store(
        const std::filesystem::path& model_directory) {
    try {
        return std::make_unique<SafeTensorsDir>(model_directory.string());
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("invalid TinyLlama weight container: " +
                                 model_directory.string() + ": " +
                                 error.what());
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

struct ModelSource::Impl {
    TinyLlamaConfig config;

    // The single owning mapping store is retained for as long as the
    // published source lives. Selected logical metadata never borrows it,
    // so destroying temporary parser or view objects cannot invalidate the
    // published inventory.
    std::unique_ptr<SafeTensorsDir> store;

    std::vector<ModelWeightInfo> weights;
    std::vector<TensorSpec> specs;
};

ModelSource::ModelSource(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {
}

ModelSource::~ModelSource() = default;

const TinyLlamaConfig& ModelSource::config() const noexcept {
    return impl_->config;
}

std::span<const ModelWeightInfo> ModelSource::weights() const noexcept {
    return impl_->weights;
}

const TensorSpec& ModelSource::tensor_spec(std::size_t index) const {
    return impl_->specs.at(index);
}

std::unique_ptr<ModelSource> load_tinyllama_safetensors(
        const std::filesystem::path& model_directory) {
    // The validated configuration supplies every required dimension, and a
    // rejected configuration stops this path before any mapping exists. The
    // checked required count is computed before any weight name is generated.
    const TinyLlamaConfig config = load_tinyllama_config(model_directory);
    const std::size_t required_count = checked_required_count(config);

    // Exactly one owning store: exact-extension discovery, every header,
    // byte, and range check, and duplicate-shard rejection happen here,
    // before any required-name filtering.
    std::unique_ptr<SafeTensorsDir> store = open_mapped_store(model_directory);
    if (store->size() == 0) {
        reject_empty_store(model_directory);
    }

    // Element products, logical byte counts, standard tiled byte counts, and
    // the repeated-layer aggregate are checked from the validated dimensions
    // before any required weight is named or inventory metadata is expanded.
    const WeightPlan plan = build_weight_plan(config, required_count);
    if (store->size() < plan.required_count) {
        reject_incomplete_store(model_directory, store->size(), plan);
    }

    std::vector<ModelWeightInfo> weights;
    std::vector<TensorSpec> specs;
    weights.reserve(plan.required_count);
    specs.reserve(plan.required_count);
    for (const WeightRoleSchema& schema : plan.globals) {
        append_selected(*store, model_directory, schema, std::nullopt, weights,
                        specs);
    }
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        for (const WeightRoleSchema& schema : plan.layer_roles) {
            append_selected(*store, model_directory, schema, layer, weights,
                            specs);
        }
    }

    // The published owner is created only after the complete inventory
    // passed. Every failure above cleaned up through RAII and left the source
    // artifacts unchanged, and no checkpoint payload byte was copied.
    auto impl = std::make_unique<ModelSource::Impl>();
    impl->config = config;
    impl->store = std::move(store);
    impl->weights = std::move(weights);
    impl->specs = std::move(specs);
    return std::unique_ptr<ModelSource>(new ModelSource(std::move(impl)));
}

}  // namespace iom