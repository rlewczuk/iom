#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include "iom/tensor.hpp"

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

/**
 * Logical role of one required TinyLlama weight. The first three roles are
 * globals without a decoder layer; the remaining nine repeat once per
 * configured decoder layer in exactly this enum order.
 */
enum class ModelWeightRole {
    token_embedding,
    final_norm,
    lm_head,
    input_norm,
    post_attention_norm,
    query,
    key,
    value,
    attention_output,
    mlp_gate,
    mlp_up,
    mlp_down,
};

/**
 * Identity of one inventory entry: the logical role plus the decoder layer for
 * the nine layer-scoped roles. `layer` is absent for `token_embedding`,
 * `final_norm`, and `lm_head`, and is in `[0, num_hidden_layers)` otherwise.
 */
struct ModelWeightId {
    ModelWeightRole role;
    std::optional<std::size_t> layer;
};

/**
 * One selected weight of a published inventory: its logical identity and the
 * destination logical shape. A checkpoint normalization vector is declared
 * rank-one `[H]` and its logical shape is the adapted `[1, H]`; every other
 * required role keeps the checkpoint's rank and `[out, in]` orientation.
 *
 * Logical metadata stays independent of the selected encoding, the physical
 * row-major mapped bytes, and the private mapping owner that retains them.
 */
struct ModelWeightInfo {
    ModelWeightId id;
    TensorShape logical_shape;
};

/**
 * Immutable, completely validated TinyLlama weight source. It is published
 * only after the configuration and every required role passed, and it
 * privately retains exactly one owning SafeTensors mapping store for as long
 * as it lives. No checkpoint payload is copied and no device, tensor, or
 * workspace is created here.
 *
 * `tensor_spec(index)` exposes the selected BF16/NONE logical metadata of one
 * inventory entry, never host bytes or container types, and throws the
 * ordinary `std::out_of_range` for an invalid index.
 */
class ModelSource final {
public:
    ~ModelSource();
    ModelSource(const ModelSource&) = delete;
    ModelSource& operator=(const ModelSource&) = delete;
    ModelSource(ModelSource&&) = delete;
    ModelSource& operator=(ModelSource&&) = delete;

    const TinyLlamaConfig& config() const noexcept;
    std::span<const ModelWeightInfo> weights() const noexcept;
    const TensorSpec& tensor_spec(std::size_t index) const;

private:
    struct Impl;
    explicit ModelSource(std::unique_ptr<Impl> impl);
    friend std::unique_ptr<ModelSource> load_tinyllama_safetensors(
            const std::filesystem::path& model_directory);
    std::unique_ptr<Impl> impl_;
};

/**
 * Validates `model_directory/config.json` and then the complete required
 * SafeTensors weight schema of the same directory, and returns the owning
 * source only after every required role passed. The inventory holds the three
 * globals and the nine layer-scoped roles of every configured decoder layer in
 * canonical order.
 *
 * A missing or wrong weight is a model-schema `std::invalid_argument` naming
 * the logical checkpoint key, its required rank/shape/dtype, and the actual
 * rank/shape/dtype or `<missing>`. An empty or obviously incomplete store is
 * rejected before any required weight is named. Impossible counts or byte
 * totals are `std::overflow_error`; allocation failures remain
 * `std::bad_alloc`. Malformed containers, malformed extras, and duplicate
 * shard names keep their established container categories, and the standalone
 * SafeTensors API is unchanged.
 */
std::unique_ptr<ModelSource> load_tinyllama_safetensors(
        const std::filesystem::path& model_directory);

}  // namespace iom