#pragma once

// Shared, network-free model-loading fixture.
//
// It owns one isolated temporary directory per test and writes deterministic
// `config.json` documents and SafeTensors checkpoints into it, so
// configuration and weight-loading scenarios reuse one directory helper and
// one checkpoint writer instead of adding per-backend copies. Each test
// constructs its own `TempDir`, whose path is removed on scope exit.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "iom/model.hpp"

namespace iom_model_loading {

// Owns a per-test directory under the system temp path; removed on scope exit.
class TempDir {
public:
    explicit TempDir(const std::string& tag)
        : path_(std::filesystem::temp_directory_path() /
                ("iom-model-loading-" + tag)) {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
        if (!std::filesystem::create_directories(path_)) {
            throw std::runtime_error(
                    "cannot create test directory: " + path_.string());
        }
    }

    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// Writes the exact bytes of `text` to `directory / filename` and returns the
// created path. A partial or failed write is a test-fixture error.
inline std::filesystem::path write_file(const std::filesystem::path& directory,
                                        const std::string& filename,
                                        const std::string& text) {
    const std::filesystem::path path = directory / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write test file: " + path.string());
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();
    if (!out) {
        throw std::runtime_error("cannot write test file: " + path.string());
    }
    return path;
}

// Writes `document` as the model directory's `config.json`.
inline std::filesystem::path write_config(const std::filesystem::path& directory,
                                          const nlohmann::json& document) {
    return write_file(directory, "config.json", document.dump());
}

// The fixed two-layer boundary configuration: N=2, H=8, I=12, Hq=4, Hkv=2,
// V=19, C=17, so the derived per-head width is D=2.
inline nlohmann::json two_layer_config() {
    return nlohmann::json{
            {"architectures", {"LlamaForCausalLM"}},
            {"model_type", "llama"},
            {"num_hidden_layers", 2},
            {"hidden_size", 8},
            {"intermediate_size", 12},
            {"num_attention_heads", 4},
            {"num_key_value_heads", 2},
            {"vocab_size", 19},
            {"max_position_embeddings", 17},
            {"hidden_act", "silu"},
            {"rms_norm_eps", 1e-5},
            {"rope_theta", 10000.0},
            {"torch_dtype", "bfloat16"},
            {"bos_token_id", 1},
            {"eos_token_id", 2},
    };
}

// The same dimensions with one decoder layer, which is equally supported.
inline nlohmann::json one_layer_config() {
    nlohmann::json document = two_layer_config();
    document["num_hidden_layers"] = 1;
    return document;
}

// True when both configurations carry identical runtime values.
inline bool same_config(const iom::TinyLlamaConfig& lhs,
                        const iom::TinyLlamaConfig& rhs) {
    return lhs.num_hidden_layers == rhs.num_hidden_layers &&
           lhs.hidden_size == rhs.hidden_size &&
           lhs.intermediate_size == rhs.intermediate_size &&
           lhs.num_attention_heads == rhs.num_attention_heads &&
           lhs.num_key_value_heads == rhs.num_key_value_heads &&
           lhs.vocab_size == rhs.vocab_size &&
           lhs.max_position_embeddings == rhs.max_position_embeddings &&
           lhs.head_dim == rhs.head_dim && lhs.bos_token_id == rhs.bos_token_id &&
           lhs.eos_token_id == rhs.eos_token_id &&
           lhs.rms_norm_eps == rhs.rms_norm_eps &&
           lhs.rope_theta == rhs.rope_theta;
}

// Loads `directory/config.json` and returns the message of the
// `std::invalid_argument` rejection it must produce.
inline std::string rejected_config_message(
        const std::filesystem::path& directory) {
    try {
        const iom::TinyLlamaConfig config = iom::load_tinyllama_config(directory);
        (void)config;
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    REQUIRE_MESSAGE(false,
                    "expected std::invalid_argument for " << directory.string());
    return std::string();
}

// ---------------------------------------------------------------------------
// SafeTensors checkpoint fixture
// ---------------------------------------------------------------------------

// One tensor entry of a written SafeTensors checkpoint. `payload` holds the
// declared bytes; an inexact payload is a deliberate malformed case.
struct SafetensorsEntry {
    std::string name;
    std::string dtype = "BF16";
    std::vector<std::size_t> shape;
    std::string payload;
};

// Deterministic, seed-dependent bytes so a wrong order, shape, or transposed
// orientation stays observable in the written checkpoint.
inline std::string deterministic_payload(std::size_t nbytes, std::size_t seed) {
    std::string payload(nbytes, '\0');
    for (std::size_t i = 0; i < nbytes; ++i) {
        payload[i] = static_cast<char>((i * 31 + seed * 7 + 3) & 0xFF);
    }
    return payload;
}

inline std::size_t element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        count *= dimension;
    }
    return count;
}

// One BF16 entry whose payload is unique to `seed`.
inline SafetensorsEntry bf16_entry(std::string name,
                                   std::vector<std::size_t> shape,
                                   std::size_t seed) {
    const std::size_t nbytes = 2 * element_count(shape);
    return SafetensorsEntry{std::move(name), "BF16", shape,
                            deterministic_payload(nbytes, seed)};
}

// Writes a container from a verbatim header, so malformed header cases keep
// the parser's own behavior.
inline std::filesystem::path write_safetensors_header(
        const std::filesystem::path& directory, const std::string& filename,
        const std::string& header, const std::string& payload) {
    std::string bytes(8, '\0');
    const auto header_len = static_cast<std::uint64_t>(header.size());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((header_len >> (8 * i)) & 0xFF);
    }
    bytes += header;
    bytes += payload;

    const std::filesystem::path path = directory / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write test safetensors file: " +
                                 path.string());
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) {
        throw std::runtime_error("cannot write test safetensors file: " +
                                 path.string());
    }
    return path;
}

// Writes one minimal valid SafeTensors file: an 8-byte little-endian header
// length, the JSON header, and the concatenated tensor payloads.
inline std::filesystem::path write_safetensors_file(
        const std::filesystem::path& directory, const std::string& filename,
        const std::vector<SafetensorsEntry>& entries) {
    std::string header = "{";
    std::string payload;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const SafetensorsEntry& entry = entries[i];
        if (i != 0) {
            header += ",";
        }
        header += "\"" + entry.name + "\":{\"dtype\":\"" + entry.dtype +
                  "\",\"shape\":[";
        for (std::size_t d = 0; d < entry.shape.size(); ++d) {
            if (d != 0) {
                header += ",";
            }
            header += std::to_string(entry.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(payload.size());
        payload += entry.payload;
        header += "," + std::to_string(payload.size()) + "]}";
    }
    header += "}";
    return write_safetensors_header(directory, filename, header, payload);
}

// The complete required TinyLlama inventory of `config` in canonical order:
// the three globals, then the nine layer-scoped roles of every configured
// decoder layer under `model.layers.{l}.`. Shapes are the checkpoint's own
// rank and `[out, in]` orientation, so the normalization vectors stay
// rank-one.
inline std::vector<SafetensorsEntry> required_weight_entries(
        const nlohmann::json& config) {
    const std::size_t layers = config.at("num_hidden_layers").get<std::size_t>();
    const std::size_t hidden = config.at("hidden_size").get<std::size_t>();
    const std::size_t intermediate =
            config.at("intermediate_size").get<std::size_t>();
    const std::size_t heads = config.at("num_attention_heads").get<std::size_t>();
    const std::size_t kv_heads =
            config.at("num_key_value_heads").get<std::size_t>();
    const std::size_t vocab = config.at("vocab_size").get<std::size_t>();
    const std::size_t kv_width = kv_heads * (hidden / heads);

    std::size_t seed = 1;
    std::vector<SafetensorsEntry> entries;
    entries.push_back(
            bf16_entry("model.embed_tokens.weight", {vocab, hidden}, seed++));
    entries.push_back(bf16_entry("model.norm.weight", {hidden}, seed++));
    entries.push_back(
            bf16_entry("lm_head.weight", {vocab, hidden}, seed++));
    for (std::size_t layer = 0; layer < layers; ++layer) {
        const std::string prefix =
                "model.layers." + std::to_string(layer) + ".";
        entries.push_back(
                bf16_entry(prefix + "input_layernorm.weight", {hidden}, seed++));
        entries.push_back(bf16_entry(prefix + "post_attention_layernorm.weight",
                                     {hidden}, seed++));
        entries.push_back(
                bf16_entry(prefix + "self_attn.q_proj.weight", {hidden, hidden},
                           seed++));
        entries.push_back(bf16_entry(prefix + "self_attn.k_proj.weight",
                                     {kv_width, hidden}, seed++));
        entries.push_back(bf16_entry(prefix + "self_attn.v_proj.weight",
                                     {kv_width, hidden}, seed++));
        entries.push_back(
                bf16_entry(prefix + "self_attn.o_proj.weight", {hidden, hidden},
                           seed++));
        entries.push_back(bf16_entry(prefix + "mlp.gate_proj.weight",
                                     {intermediate, hidden}, seed++));
        entries.push_back(bf16_entry(prefix + "mlp.up_proj.weight",
                                     {intermediate, hidden}, seed++));
        entries.push_back(bf16_entry(prefix + "mlp.down_proj.weight",
                                     {hidden, intermediate}, seed++));
    }
    return entries;
}

// Logical dimensions of a published inventory shape as a plain vector.
inline std::vector<std::size_t> dims_of(const iom::TensorShape& shape) {
    const std::span<const std::size_t> dimensions = shape.dimensions();
    return std::vector<std::size_t>(dimensions.begin(), dimensions.end());
}

// Loads `directory` as a mapped weight source and returns the message of the
// `std::invalid_argument` schema rejection it must produce.
inline std::string rejected_source_message(
        const std::filesystem::path& directory) {
    try {
        const std::unique_ptr<iom::ModelSource> source =
                iom::load_tinyllama_safetensors(directory);
        (void)source;
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    REQUIRE_MESSAGE(false,
                    "expected std::invalid_argument for " << directory.string());
    return std::string();
}

// Loads `directory` as a mapped weight source and returns the message of the
// `std::runtime_error` container rejection it must produce.
inline std::string rejected_container_message(
        const std::filesystem::path& directory) {
    try {
        const std::unique_ptr<iom::ModelSource> source =
                iom::load_tinyllama_safetensors(directory);
        (void)source;
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    REQUIRE_MESSAGE(false,
                    "expected std::runtime_error for " << directory.string());
    return std::string();
}

}  // namespace iom_model_loading