#pragma once

#include <cstddef>
#include <filesystem>

namespace iom {

/**
 * Validated TinyLlama runtime configuration. Every field is a runtime value of
 * the explicit model directory, never a constant of one checkpoint: the number
 * of decoder layers, hidden, intermediate, head, key/value-head, vocabulary,
 * and context sizes, the two token ids, the RMSNorm epsilon, and the RoPE
 * theta. `head_dim` is derived as `hidden_size / num_attention_heads`, so an
 * independent head width is neither declared nor accepted from JSON.
 *
 * The loader rejects every missing, wrongly typed, unsupported, or
 * inconsistent value before any checkpoint, device, workspace, or weight work
 * begins, so a returned configuration is always complete and usable.
 */
struct TinyLlamaConfig {
    std::size_t num_hidden_layers;
    std::size_t hidden_size;
    std::size_t intermediate_size;
    std::size_t num_attention_heads;
    std::size_t num_key_value_heads;
    std::size_t vocab_size;
    std::size_t max_position_embeddings;
    std::size_t head_dim;
    std::size_t bos_token_id;
    std::size_t eos_token_id;
    float rms_norm_eps;
    double rope_theta;
};

/**
 * Reads exactly `model_directory/config.json` and returns its validated
 * configuration. No tokenizer, generation, distribution, or fallback file is
 * searched, and no mapping, device, tensor, or workspace is created.
 *
 * A missing or unreadable file is an I/O `std::runtime_error` naming the path.
 * Malformed, empty, or non-object JSON, a missing or unsupported field, an
 * invalid derived head ratio, and an invalid token policy are
 * `std::invalid_argument` naming the path, the field, the actual value or
 * `<missing>`, and the violated constraint. Arithmetic overflow remains
 * `std::overflow_error`; allocation failures remain `std::bad_alloc`.
 */
TinyLlamaConfig load_tinyllama_config(const std::filesystem::path& model_directory);

}  // namespace iom