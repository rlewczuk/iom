#pragma once

// Shared, network-free model-loading fixture.
//
// It owns one isolated temporary directory per test and writes deterministic
// `config.json` documents into it, so configuration and later weight-loading
// scenarios reuse one directory helper instead of adding per-backend copies.
// Each test constructs its own `TempDir`, whose path is removed on scope exit.

#include <doctest/doctest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

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

}  // namespace iom_model_loading