#include "iom/session.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/device.hpp"
#include "iom/inference_metrics.hpp"
#include "iom/iom.hpp"
#include "iom/oid.hpp"
#include "iom_internal.hpp"
#include "session_internal.hpp"

namespace iom {
namespace {

using detail::checked_add;
using detail::checked_mul;
using session_detail::CacheOwner;
using session_detail::OperationContext;
using session_detail::RunBanks;
constexpr std::size_t kBaseWorkspaceAlignment = 32;
constexpr std::size_t kAcceptedOidsPerLayer = 24;
constexpr std::size_t kAcceptedOidBase = 32;


[[nodiscard]] bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

void validate_workspace_requirement(
        WorkspaceRequirements requirement, const char* label) {
    if (requirement.bytes == 0) {
        if (requirement.alignment != 1) {
            throw std::invalid_argument(
                    std::string(label) + " zero-byte alignment must be one");
        }
        return;
    }
    if (requirement.alignment < kBaseWorkspaceAlignment
            || !is_power_of_two(requirement.alignment)) {
        throw std::invalid_argument(
                std::string(label)
                + " positive workspace alignment must be a power of two"
                  " of at least 32 bytes");
    }
    // The factory takes exact bytes, but its aligned allocation and every
    // subsequent subrange must also be representable.
    static_cast<void>(checked_add(
            requirement.bytes, requirement.alignment - 1,
            "TinyLlama session workspace alignment overflows"));
}

[[nodiscard]] std::unique_ptr<Tensor> make_tensor(
        Device& device, std::vector<std::size_t> dimensions,
        DataType data_type = DataType::BF16) {
    TensorSpec spec{
            TensorShape{std::move(dimensions)}, data_type,
            QuantizationFormat::NONE};
    spec.validate();
    static_cast<void>(spec.logical_nbytes());
    static_cast<void>(spec.tiled_storage_nbytes());
    auto owner = device.create_tensor(spec);
    if (!owner || owner->view().spec() != spec
            || owner->view().owner_identity() != owner.get()) {
        throw std::invalid_argument(
                "TinyLlama session device returned an invalid tensor owner");
    }
    static_cast<void>(detail::validate_checked_view(
            device, owner->view(), "TinyLlama session"));
    return owner;
}

struct StorageBytes {
    std::size_t logical = 0;
    std::size_t tiled = 0;

    void add(const TensorSpec& spec, std::size_t count = 1) {
        logical = checked_add(
                logical, checked_mul(spec.logical_nbytes(), count,
                                     "session logical bytes overflow"),
                "session logical bytes overflow");
        tiled = checked_add(
                tiled, checked_mul(spec.tiled_storage_nbytes(), count,
                                   "session tiled bytes overflow"),
                "session tiled bytes overflow");
    }
};

// Preflight all distinct geometries and the full resident total before the
// first cache/bank allocation, using TensorSpec's standard checked sizing.
[[nodiscard]] StorageBytes run_storage_bytes(
        const TinyLlamaConfig& config, std::size_t R) {
    const std::size_t F = checked_mul(
            config.num_attention_heads, config.head_dim,
            "TinyLlama session merged width overflows");
    StorageBytes bytes;
    bytes.add({TensorShape{{1, R}}, DataType::U32});
    bytes.add({TensorShape{{R, F}}, DataType::BF16}, 9);
    bytes.add({TensorShape{{R, config.intermediate_size}}, DataType::BF16}, 4);
    bytes.add({TensorShape{{config.num_attention_heads, R, config.head_dim}},
               DataType::BF16}, 2);
    bytes.add({TensorShape{{config.num_key_value_heads, R, config.head_dim}},
               DataType::BF16}, 3);
    return bytes;
}

template <class T>
void validate_vector_capacity(std::size_t count) {
    static_cast<void>(checked_mul(
            count, sizeof(T), "TinyLlama session host bytes overflow"));
    if (count > std::vector<T>{}.max_size()) {
        throw std::overflow_error("TinyLlama session host capacity overflows");
    }
}

// The one checked accepted-OID capacity bound of a validated configuration:
// one prefill plus at most C-1 decode runs, including the per-run
// embedding/final/transfer work rather than merely the prompt's R rows.
// Request ledger sizing and explicit trace preparation share this bound, so a
// prepared operation trace can always hold one complete request.
[[nodiscard]] std::size_t accepted_oid_capacity(
        const TinyLlamaConfig& config, const char* overflow_message) {
    const std::size_t per_run = checked_add(
            checked_mul(config.num_hidden_layers, kAcceptedOidsPerLayer,
                        overflow_message),
            kAcceptedOidBase, overflow_message);
    const std::size_t capacity = checked_mul(
            config.max_position_embeddings, per_run, overflow_message);
    validate_vector_capacity<oid>(capacity);
    return capacity;
}

void validate_persistent_storage(const TinyLlamaConfig& config) {
    if (config.vocab_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("TinyLlama vocabulary exceeds U32 indices");
    }
    auto bytes = run_storage_bytes(config, 1);
    bytes.add({TensorShape{{1, config.vocab_size}}, DataType::BF16});
    bytes.add(
            {TensorShape{{config.num_key_value_heads,
                          config.max_position_embeddings, config.head_dim}},
             DataType::BF16},
            checked_mul(2, config.num_hidden_layers,
                        "TinyLlama session cache owner count overflows"));
    validate_vector_capacity<CacheOwner>(config.num_hidden_layers);
}

void validate_generation_prompt(
        const TinyLlamaConfig& config,
        std::span<const std::size_t> token_ids) {
    if (token_ids.empty()) {
        throw std::invalid_argument(
                "TinyLlama generation prompt must be nonempty");
    }
    if (token_ids.size() > config.max_position_embeddings) {
        throw std::invalid_argument(
                "TinyLlama generation prompt exceeds model context");
    }
    static_cast<void>(checked_mul(
            token_ids.size(), sizeof(std::uint32_t),
            "TinyLlama generation prompt bytes overflow"));
    validate_vector_capacity<std::size_t>(
            config.max_position_embeddings);
    for (const std::size_t token_id : token_ids) {
        if (token_id >= config.vocab_size) {
            throw std::invalid_argument(
                    "TinyLlama generation prompt token exceeds vocabulary");
        }
    }
}

[[nodiscard]] std::vector<std::size_t> widen_token_ids(
        std::span<const std::uint32_t> token_ids) {
    validate_vector_capacity<std::size_t>(token_ids.size());
    std::vector<std::size_t> widened;
    widened.reserve(token_ids.size());
    for (const std::uint32_t token_id : token_ids) {
        if constexpr (std::numeric_limits<std::size_t>::max()
                      < std::numeric_limits<std::uint32_t>::max()) {
            if (token_id > std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error(
                        "TinyLlama generation prompt token ID cannot be "
                        "represented by size_t");
            }
        }
        widened.push_back(static_cast<std::size_t>(token_id));
    }
    return widened;
}

[[nodiscard]] GenerationResult finish_text_generation(
        TinyLlamaSession& session, std::vector<std::uint32_t> encoded,
        std::size_t max_new_tokens) {
    std::vector<std::size_t> prompt = widen_token_ids(encoded);
    TokenGenerationResult generated =
            session.generate_tokens(prompt, max_new_tokens);

    validate_vector_capacity<std::uint32_t>(generated.token_ids.size());
    std::vector<std::uint32_t> generated_ids;
    generated_ids.reserve(generated.token_ids.size());
    for (const std::size_t token_id : generated.token_ids) {
        if (token_id > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                    "TinyLlama generated token ID cannot be represented by "
                    "the tokenizer");
        }
        generated_ids.push_back(static_cast<std::uint32_t>(token_id));
    }

    const std::string text = session.tokenizer().decode(
            generated_ids, DecodeOptions{.skip_special_tokens = true});
    return GenerationResult{
            std::move(generated.token_ids), text, generated.stop_reason};
}

// Raw/chat tokenization observation.
//
// `generate_raw` and `generate_chat` stage the attempt and the encode interval
// here, and attach the observation only after the nested generation returned
// or threw. Nothing is cleared at wrapper entry: `begin_generation` stages a
// new attempt without touching the last admitted request, so a rejected
// preprocessing attempt can neither steal nor overwrite the outgoing
// request's attribution. Only `tokenizer.encode` is measured; chat rendering
// completes before the measured begin, and generation, the post-generation ID
// conversion, and the output decoding all happen after the measured end. A
// null recorder disables every read and every write below.
class TokenizationObservation final {
public:
    explicit TokenizationObservation(InferenceMetrics* metrics) noexcept
            : metrics_(metrics) {
        if (metrics_ == nullptr) {
            return;
        }
        // Stage this preprocessing attempt before any rendering or encoding
        // work. The low-level generation stage replaces the recorded entry
        // instant and attempted cardinality for the same attempt, so TTFT
        // still starts at generation entry; staging here only makes this
        // attempt distinct from the last admitted request.
        metrics_->begin_generation(metrics_->now(), 0);
    }

    /** Begin the measured interval immediately before the encoder call. */
    void begin_encode() noexcept {
        if (metrics_ == nullptr) {
            return;
        }
        encode_begin_ = metrics_->now();
        encode_observed_ = true;
    }

    /** Capture the encode end instant of a completed or failed encode. */
    void end_encode() noexcept {
        if (metrics_ == nullptr) {
            return;
        }
        encode_end_ = metrics_->now();
    }

    // Report a failed formatter or encoder attempt. A formatter failure never
    // reached the encoder, so tokenization stays `not_run`; an observed
    // encoder failure is offered to the recorder as a failed tokenization
    // observation instead of a fabricated success. In both cases the failure
    // belongs to this attempt and never to a previously admitted request, and
    // no OID, selector call, or generated token is fabricated.
    void record_preprocessing_failure(std::exception_ptr failure) noexcept {
        if (metrics_ == nullptr) {
            return;
        }
        if (encode_observed_) {
            metrics_->record_tokenization(
                    encode_begin_, encode_end_, ObservationState::failed);
        }
        metrics_->record_preprocessing_failure(
                encode_end_, std::move(failure));
    }

    // Attach the completed encode interval to the generation attempt that owns
    // it. The recorder accepts it only while that attempt is the published
    // request, so a failed drain, a pre-publication validation/setup failure,
    // or a rejected preprocessing attempt leaves the outgoing request's
    // observation intact and its own interval unattributed.
    void attach() noexcept {
        if (metrics_ == nullptr) {
            return;
        }
        metrics_->record_tokenization(
                encode_begin_, encode_end_, ObservationState::succeeded);
    }

private:
    InferenceMetrics* metrics_ = nullptr;
    std::uint64_t encode_begin_ = 0;
    std::uint64_t encode_end_ = 0;
    bool encode_observed_ = false;
};

// Run one raw/chat encode under the observation above: only the encoder call
// itself lies between the captured begin and end instants, and a failed encode
// keeps its original exception.
template <class Encode>
[[nodiscard]] std::vector<std::uint32_t> observed_encode(
        TokenizationObservation& observation, Encode encode) {
    try {
        observation.begin_encode();
        std::vector<std::uint32_t> encoded = encode();
        observation.end_encode();
        return encoded;
    } catch (...) {
        observation.end_encode();
        observation.record_preprocessing_failure(std::current_exception());
        throw;
    }
}

// Complete a raw/chat attempt: attach the measured encode interval to the
// request the nested generation produced, whether it returned or threw.
[[nodiscard]] GenerationResult finish_text_generation(
        TinyLlamaSession& session, std::vector<std::uint32_t> encoded,
        std::size_t max_new_tokens, TokenizationObservation& observation) {
    try {
        GenerationResult result = finish_text_generation(
                session, std::move(encoded), max_new_tokens);
        observation.attach();
        return result;
    } catch (...) {
        observation.attach();
        throw;
    }
}

struct ForwardLayerPlan {
    session_detail::DecoderLayerForwardViews prefill;
    session_detail::DecoderLayerForwardViews decode;
    session_detail::DecoderLayerForwardParams prefill_params;
    session_detail::DecoderLayerForwardParams decode_params;
};

struct ForwardWorkspaceLayout {
    std::array<WorkspaceRequirements, 3> qkv_requirements{};
    WorkspaceRequirements attention_requirement{};
    session_detail::MlpWorkspaceRequirements mlp_requirements{};
    WorkspaceRequirements sequential_requirement{};
    std::array<std::size_t, 3> qkv_offsets{};
    std::size_t attention_offset = 0;
    std::size_t mlp_offset = 0;
    std::size_t sequential_offset = 0;
    WorkspaceRequirements total{};
};

struct ForwardPlan {
    std::vector<ForwardLayerPlan> layers;
    ForwardWorkspaceLayout workspace;
};

struct RequestState {
    std::size_t run_length = 0;
    RunBanks banks;
    std::unique_ptr<ForwardPlan> forward;
    std::unique_ptr<RawWorkspace> operation_workspace;
    std::vector<std::byte> selector_host_scratch;
    std::unique_ptr<RawWorkspace> selector_device_scratch;
    std::vector<std::uint32_t> prefill_indices;
    std::uint32_t decode_index = 0;
    std::vector<std::size_t> history;
    std::vector<std::size_t> results;
    std::vector<oid> accepted_oids;
    bool draining = false;
};


[[nodiscard]] RunBanks allocate_run_banks(
        Device& device, const TinyLlamaConfig& config, std::size_t R) {
    RunBanks banks;
    const std::size_t F = config.hidden_size;
    const std::size_t M = config.intermediate_size;
    const std::size_t Hq = config.num_attention_heads;
    const std::size_t Hkv = config.num_key_value_heads;
    const std::size_t D = config.head_dim;
    const std::size_t merged = checked_mul(
            Hq, D, "TinyLlama session merged width overflows");
    banks.token_indices = make_tensor(device, {1, R}, DataType::U32);
    banks.x = make_tensor(device, {R, F});
    banks.attention_norm = make_tensor(device, {R, F});
    banks.q = make_tensor(device, {Hq, R, D});
    banks.k = make_tensor(device, {Hkv, R, D});
    banks.v = make_tensor(device, {Hkv, R, D});
    banks.rotated_q = make_tensor(device, {Hq, R, D});
    banks.rotated_k = make_tensor(device, {Hkv, R, D});
    banks.attention_merged = make_tensor(device, {R, merged});
    banks.attention_output = make_tensor(device, {R, F});
    banks.residual_after_attention = make_tensor(device, {R, F});
    banks.mlp_norm = make_tensor(device, {R, F});
    banks.gate = make_tensor(device, {R, M});
    banks.up = make_tensor(device, {R, M});
    banks.silu = make_tensor(device, {R, M});
    banks.product = make_tensor(device, {R, M});
    banks.down = make_tensor(device, {R, F});
    banks.residual_after_mlp = make_tensor(device, {R, F});
    banks.final_norm = make_tensor(device, {R, F});
    return banks;
}

[[nodiscard]] std::array<const Tensor*, 19> bank_owners(const RunBanks& banks) {
    return {banks.token_indices.get(), banks.x.get(), banks.attention_norm.get(),

            banks.q.get(), banks.k.get(), banks.v.get(), banks.rotated_q.get(),
            banks.rotated_k.get(), banks.attention_merged.get(),
            banks.attention_output.get(), banks.residual_after_attention.get(),
            banks.mlp_norm.get(), banks.gate.get(), banks.up.get(),
            banks.silu.get(), banks.product.get(), banks.down.get(),
            banks.residual_after_mlp.get(), banks.final_norm.get()};
}
[[nodiscard]] WorkspaceRequirements requirement_max(
        WorkspaceRequirements lhs, WorkspaceRequirements rhs) {
    if (rhs.bytes == 0) {
        if (rhs.alignment != 1) {
            throw std::invalid_argument(
                    "TinyLlama forward zero-byte requirement has invalid "
                    "alignment");
        }
        return lhs;
    }
    if (rhs.alignment == 0
            || (rhs.alignment & (rhs.alignment - 1)) != 0
            || rhs.alignment < kBaseWorkspaceAlignment) {
        throw std::invalid_argument(
                "TinyLlama forward requirement has invalid alignment");
    }
    if (lhs.bytes == 0) {
        return {rhs.bytes, rhs.alignment};
    }
    return {std::max(lhs.bytes, rhs.bytes),
            std::max(lhs.alignment, rhs.alignment)};
}

void include_requirement(
        WorkspaceRequirements& destination, WorkspaceRequirements requirement) {
    destination = requirement_max(destination, requirement);
}

void include_mlp_requirements(
        session_detail::MlpWorkspaceRequirements& destination,
        const session_detail::MlpWorkspaceRequirements& requirement) {
    include_requirement(destination.gate, requirement.gate);
    include_requirement(destination.up, requirement.up);
    include_requirement(destination.mul, requirement.mul);
    include_requirement(destination.down, requirement.down);
    include_requirement(destination.residual, requirement.residual);
}

[[nodiscard]] std::size_t layer_weight_index(
        std::size_t layer, std::size_t role_offset) {
    return checked_add(
            3, checked_add(
                       checked_mul(
                               layer, std::size_t{9},
                               "TinyLlama forward layer weight index overflows"),
                       role_offset,
                       "TinyLlama forward layer weight index overflows"),
            "TinyLlama forward layer weight index overflows");
}

[[nodiscard]] session_detail::DecoderLayerForwardViews make_layer_views(
        const TinyLlamaModel& model, const RunBanks& banks,
        CacheOwner& cache, std::size_t layer) {
    const auto weight = [&](std::size_t role_offset) -> const TensorView& {
        return model.weight(layer_weight_index(layer, role_offset));
    };
    const TensorView& input = (layer & 1U) == 0
            ? banks.x->view()
            : banks.residual_after_mlp->view();
    const TensorView& output = (layer & 1U) == 0
            ? banks.residual_after_mlp->view()
            : banks.x->view();
    return session_detail::DecoderLayerForwardViews{
            session_detail::QkvRopeStageViews{
                    input, weight(0), weight(2), weight(3), weight(4),
                    banks.attention_norm->view(), banks.q->view(),
                    banks.k->view(), banks.v->view(),
                    banks.rotated_q->view(), banks.rotated_k->view()},
            session_detail::CacheAttentionStageRequest{
                    banks.rotated_q->view(), banks.rotated_k->view(),
                    banks.v->view(), cache.key->view(), cache.value->view(),
                    input, weight(5), banks.attention_merged->view(),
                    banks.attention_output->view(),
                    banks.residual_after_attention->view(), 0,
                    input.spec().shape.dimensions()[0],
                    model.config().max_position_embeddings},
            session_detail::MlpStageViews{
                    banks.residual_after_attention->view(), weight(1),
                    weight(6), weight(7), weight(8),
                    banks.mlp_norm->view(), banks.gate->view(), banks.up->view(),
                    banks.silu->view(), banks.product->view(), banks.down->view(),
                    output}};
}

[[nodiscard]] const TensorView& final_layer_output(
        const RunBanks& banks, std::size_t layer_count) {
    return (layer_count & 1U) == 0
            ? banks.x->view()
            : banks.residual_after_mlp->view();
}

[[nodiscard]] std::size_t align_up(
        std::size_t value, std::size_t alignment, const char* message) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(message);
    }
    return checked_add(value, alignment - 1, message) & ~(alignment - 1);
}

void reserve_slice(
        std::size_t& cursor, std::size_t& offset, WorkspaceRequirements req,
        const char* label) {
    if (req.bytes == 0) {
        offset = 0;
        return;
    }
    const std::size_t alignment =
            std::max(kBaseWorkspaceAlignment, req.alignment);
    offset = align_up(cursor, alignment, label);
    cursor = checked_add(offset, req.bytes, label);
}

[[nodiscard]] ForwardPlan make_forward_plan(
        const TinyLlamaModel& model, const RunBanks& prefill,
        const RunBanks& decode, const TensorView& logits,
        std::span<CacheOwner> caches, DeviceOps& operations,
        std::size_t run_length) {
    const TinyLlamaConfig& config = model.config();
    const std::size_t features = config.hidden_size;
    const std::size_t intermediate = config.intermediate_size;
    const std::size_t capacity = config.max_position_embeddings;
    const std::size_t query_heads = config.num_attention_heads;
    const std::size_t kv_heads = config.num_key_value_heads;
    const std::size_t head_dim = config.head_dim;
    const std::size_t vocabulary = config.vocab_size;
    ForwardPlan plan;
    plan.layers.reserve(config.num_hidden_layers);
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        plan.layers.emplace_back(ForwardLayerPlan{
                make_layer_views(model, prefill, caches[layer], layer),
                make_layer_views(model, decode, caches[layer], layer),
                session_detail::DecoderLayerForwardParams{
                        0, run_length, capacity, features, intermediate,
                        config.rope_theta, config.rms_norm_eps,
                        config.rms_norm_eps},
                session_detail::DecoderLayerForwardParams{
                        0, 1, capacity, features, intermediate,
                        config.rope_theta, config.rms_norm_eps,
                        config.rms_norm_eps}});
    }

    const auto add_sequential = [&](WorkspaceRequirements requirement) {
        include_requirement(plan.workspace.sequential_requirement, requirement);
    };
    const auto add_qkv = [&](std::size_t branch,
                             WorkspaceRequirements requirement) {
        include_requirement(plan.workspace.qkv_requirements[branch],
                            requirement);
    };
    const auto add_attention = [&](WorkspaceRequirements requirement) {
        include_requirement(plan.workspace.attention_requirement, requirement);
    };

    add_sequential(operations.embedding_workspace_requirements(
            model.weight(0), prefill.token_indices->view(), prefill.x->view()));
    add_sequential(prefill.token_indices->view()
                           .copy_from_host_workspace_requirements());
    add_sequential(operations.embedding_workspace_requirements(
            model.weight(0), decode.token_indices->view(), decode.x->view()));
    add_sequential(decode.token_indices->view()
                           .copy_from_host_workspace_requirements());

    for (ForwardLayerPlan& layer : plan.layers) {
        const auto query_run = [&](const RunBanks& banks,
                                   session_detail::DecoderLayerForwardViews& views,
                                   std::size_t rows, std::size_t a,
                                   std::size_t length,
                                   session_detail::DecoderLayerForwardParams&
                                           params) {
            params.a = a;
            views.attention.a = a;
            views.attention.R = rows;
            views.attention.C = capacity;

            add_sequential(operations.rmsnorm_workspace_requirements(
                    views.qkv.activation, views.qkv.attention_scale,
                    views.qkv.normalized, config.rms_norm_eps));
            add_qkv(0, operations.linear_workspace_requirements(
                              views.qkv.normalized, views.qkv.query_weight,
                              views.qkv.query, 0, rows,
                              LinearOutputLayout::head_planar, query_heads,
                              head_dim));
            add_qkv(1, operations.linear_workspace_requirements(
                              views.qkv.normalized, views.qkv.key_weight,
                              views.qkv.key, 0, rows,
                              LinearOutputLayout::head_planar, kv_heads,
                              head_dim));
            add_qkv(2, operations.linear_workspace_requirements(
                              views.qkv.normalized, views.qkv.value_weight,
                              views.qkv.value, 0, rows,
                              LinearOutputLayout::head_planar, kv_heads,
                              head_dim));
            add_sequential(operations.rope_workspace_requirements(
                    views.qkv.query, views.qkv.rotated_query, a,
                    config.rope_theta));
            add_sequential(operations.rope_workspace_requirements(
                    views.qkv.key, views.qkv.rotated_key, a,
                    config.rope_theta));
            add_sequential(operations.cache_append_workspace_requirements(
                    views.attention.rotated_k, views.attention.k_cache, a));
            add_sequential(operations.cache_append_workspace_requirements(
                    views.attention.rotated_v, views.attention.v_cache, a));
            add_attention(operations.sdpa_workspace_requirements(
                    views.attention.rotated_q, views.attention.k_cache,
                    views.attention.v_cache, views.attention.attention_merged,
                    a, length));
            add_attention(operations.linear_workspace_requirements(
                    views.attention.attention_merged,
                    views.attention.o_weight,
                    views.attention.attention_output, 0, rows,
                    LinearOutputLayout::ordinary, 1, features));
            add_attention(operations.add_workspace_requirements(
                    views.attention.residual_input,
                    views.attention.attention_output,
                    views.attention.residual_output));
            add_sequential(operations.rmsnorm_workspace_requirements(
                    views.mlp.x2, views.mlp.post_attention_scale,
                    views.mlp.n2, config.rms_norm_eps));
            add_sequential(operations.silu_workspace_requirements(
                    views.mlp.gate, views.mlp.activated_gate));

            const session_detail::MlpWorkspaceRequirements mlp =
                    session_detail::mlp_workspace_requirements(
                            operations, views.mlp, views.mlp);
            include_mlp_requirements(plan.workspace.mlp_requirements, mlp);
            add_sequential(operations.linear_workspace_requirements(
                    views.mlp.product, views.mlp.down_weight, views.mlp.down,
                    0, rows, LinearOutputLayout::ordinary, 1, features));
            add_sequential(operations.add_workspace_requirements(
                    views.mlp.x2, views.mlp.down, views.mlp.next_x));
            (void)banks;
        };

        query_run(prefill, layer.prefill, run_length, 0, run_length,
                  layer.prefill_params);
        query_run(
                decode, layer.decode, 1, capacity - 1, capacity,
                layer.decode_params);
    }

    const TensorView& prefill_input =
            final_layer_output(prefill, config.num_hidden_layers);
    add_sequential(operations.rmsnorm_workspace_requirements(
            prefill_input, model.weight(1), prefill.final_norm->view(),
            config.rms_norm_eps));
    add_sequential(operations.linear_workspace_requirements(
            prefill.final_norm->view(), model.weight(2), logits, run_length - 1,
            1, LinearOutputLayout::ordinary, 1, vocabulary));
    const TensorView& decode_input =
            final_layer_output(decode, config.num_hidden_layers);
    add_sequential(operations.rmsnorm_workspace_requirements(
            decode_input, model.weight(1), decode.final_norm->view(),
            config.rms_norm_eps));
    add_sequential(operations.linear_workspace_requirements(
            decode.final_norm->view(), model.weight(2), logits, 0, 1,
            LinearOutputLayout::ordinary, 1, vocabulary));

    const session_detail::MlpWorkspaceRequirements& mlp =
            plan.workspace.mlp_requirements;
    const WorkspaceRequirements mlp_total = mlp.total();
    std::size_t cursor = 0;
    for (std::size_t branch = 0; branch < plan.workspace.qkv_requirements.size();
         ++branch) {
        reserve_slice(cursor, plan.workspace.qkv_offsets[branch],
                      plan.workspace.qkv_requirements[branch],
                      "TinyLlama forward QKV workspace overflows");
    }
    reserve_slice(cursor, plan.workspace.attention_offset,
                  plan.workspace.attention_requirement,
                  "TinyLlama forward attention workspace overflows");
    reserve_slice(cursor, plan.workspace.mlp_offset, mlp_total,
                  "TinyLlama forward MLP workspace overflows");
    reserve_slice(cursor, plan.workspace.sequential_offset,
                  plan.workspace.sequential_requirement,
                  "TinyLlama forward sequential workspace overflows");
    if (cursor == 0) {
        plan.workspace.total = {0, 1};
    } else {
        plan.workspace.total = {
                cursor,
                std::max(
                        {kBaseWorkspaceAlignment,
                         plan.workspace.qkv_requirements[0].alignment,
                         plan.workspace.qkv_requirements[1].alignment,
                         plan.workspace.qkv_requirements[2].alignment,
                         plan.workspace.attention_requirement.alignment,
                         mlp_total.alignment,
                         plan.workspace.sequential_requirement.alignment})};
    }
    return plan;
}

void validate_workspace_owner(
        const RawWorkspace* owner, const Device& device,
        WorkspaceRequirements requirement, const char* label) {
    if (owner == nullptr) {
        throw std::invalid_argument(
                std::string(label) + " positive workspace returned null");
    }
    if (&owner->device() != &device
            || owner->byte_size() < requirement.bytes) {
        throw std::invalid_argument(
                std::string(label) + " workspace has the wrong device or size");
    }
    if (owner->byte_size() != requirement.bytes) {
        throw std::invalid_argument(
                std::string(label) + " workspace size is not exact");
    }
    static_cast<void>(detail::WorkspaceValidation::validated(
            device, owner->view(), requirement.bytes, requirement.alignment, {}));
}

}  // namespace

struct TinyLlamaSession::Impl {
    Device* device = nullptr;

    // Borrowed opt-in observation recorder. The caller owns it and keeps it
    // and its supplied clock context alive through session destruction and
    // drain; the session never owns, copies, replaces, or rejects it. A null
    // value means every observation hook is disabled.
    InferenceMetrics* metrics = nullptr;

    // Attribution copied into every accepted enqueue observation. The forward
    // entry points and their decoder-layer loops set these values from the
    // request and loop variables they already own; a submission outside any
    // forward scope keeps the default empty attribution. Disabled observation
    // keeps no bookkeeping here because the setter ignores a null recorder.
    OperationContext operation_context;

    // Declaration order is intentional.  The queue is destroyed first so it
    // can close and drain while every tensor/workspace owner remains alive.
    std::unique_ptr<TinyLlamaModel> model;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<ChatFormatter> formatter;
    std::unique_ptr<TokenSelector> selector;
    std::vector<CacheOwner> caches;
    RunBanks fixed;
    std::unique_ptr<Tensor> logits;
    std::unique_ptr<RequestState> request;
    std::unique_ptr<DeviceOps> queue;

    bool poisoned = false;

    ~Impl() noexcept {
        try {
            drain_request();
        } catch (...) {
            // Failed waits are not terminality proof. The backend queue closes
            // and drains/quarantines before any operand or scratch is released.
        }
        queue.reset();
    }

    [[nodiscard]] const TinyLlamaConfig& config() const noexcept {
        return model->config();
    }

    void require_request() const {
        if (!request || request->draining || poisoned) {
            throw std::logic_error(
                    "TinyLlama session request is not accepting work");
        }
    }

    // The published admitted request's ordinal, or zero when no observed
    // request is admitted. The accepted-OID ledger belongs to exactly that
    // request, so a token found in the ledger is owned by this ordinal.
    [[nodiscard]] std::uint64_t admitted_request_ordinal() const noexcept {
        if (metrics == nullptr || !metrics->snapshot().request_admitted) {
            return 0;
        }
        return metrics->snapshot().admitted.ordinal;
    }

    // Verify one exact accepted OID against the existing accepted-OID ledger.
    // Wait attribution and failure ownership come from that ledger, never from
    // the presence of a trace row: an OID this request never accepted - an
    // unknown sequence, an OID of another queue, or selector-internal work
    // submitted directly to the exposed queue - reports zero and can neither
    // invalidate the admitted request's rate nor fabricate copied request
    // context.
    [[nodiscard]] std::uint64_t owning_request_ordinal(
            oid token) const noexcept {
        const std::uint64_t ordinal = admitted_request_ordinal();
        if (ordinal == 0 || request == nullptr) {
            return 0;
        }
        const std::vector<oid>& accepted = request->accepted_oids;
        return std::find(accepted.begin(), accepted.end(), token)
                        != accepted.end()
                ? ordinal
                : 0;
    }

    // Observe one existing session-controlled wait outcome on the attached
    // recorder, immediately at that existing wait's return or catch. The
    // hooks are allocation-free and non-throwing: they add no wait, poll, or
    // synchronization work, they never replace or delay the existing error
    // path, and a null recorder performs no observation at all. A scalar-only
    // success needs neither a clock read nor a record; a scalar-only failure
    // may invalidate the verified owning request's rate without reading the
    // clock or registering a per-OID row. The recorder itself preserves the
    // first successful observation or the first retained failure of each
    // owned OID, so repeated waits, drain recursion, and destructor cleanup
    // add no row and select no different error.
    void observe_wait_success(oid token) noexcept {
        if (metrics == nullptr || !metrics->trace_prepared()) {
            return;
        }
        // A successful observation never invalidates a rate, so the owning
        // ordinal - consumed by the recorder only for a failed observation -
        // stays zero and this path needs no ledger proof.
        metrics->record_wait(token, 0, metrics->now(),
                             ObservationState::succeeded, nullptr);
    }

    void observe_wait_failure(oid token, std::exception_ptr failure) noexcept {
        if (metrics == nullptr) {
            return;
        }
        std::optional<InferenceMetrics::value_type> host_instant;
        if (metrics->trace_prepared()) {
            host_instant = metrics->now();
        }
        metrics->record_wait(token, owning_request_ordinal(token), host_instant,
                             ObservationState::failed, std::move(failure));
    }

    void drain_request() {
        if (!request) {
            return;
        }
        request->draining = true;
        std::exception_ptr first;
        for (const oid token : request->accepted_oids) {
            try {
                queue->wait(token);
            } catch (...) {
                // Save the original failure before any observation, poisoning,
                // drain, or cleanup activity; the wait observation and the
                // preserved first error both consume this exact exception.
                const std::exception_ptr failure = std::current_exception();
                observe_wait_failure(token, failure);
                if (!first) first = failure;
                poisoned = true;
                continue;
            }
            observe_wait_success(token);
        }
        request->draining = false;
        if (first) {
            std::rethrow_exception(first);
        }
        if (!poisoned) request->accepted_oids.clear();
    }

    void poison_and_drain() noexcept {
        poisoned = true;
        try {
            drain_request();
        } catch (...) {
            // The original generation failure remains the observable error;
            // drain_request has still attempted every accepted OID.
        }
    }

    void reset_cache_prefix() noexcept {
        for (CacheOwner& cache : caches) {
            cache.initialized_length = 0;
        }
    }

    void validate_scratch(
            const RequestState& candidate, WorkspaceRequirements operation,
            TokenSelectorScratchRequirements selector_requirements) const {
        const auto validate_owner = [&](const RawWorkspace* owner,
                                        WorkspaceRequirements requirement) {
            if (!owner) return;
            const auto against = [&](const TensorView& view) {
                static_cast<void>(detail::WorkspaceValidation::validated(
                        *device, owner->view(), requirement.bytes,
                        requirement.alignment, {&view, 1}));
            };
            for (std::size_t index = 0; index < model->weights().size(); ++index)
                against(model->weight(index));
            for (const auto& cache : caches) {
                against(cache.key->view());
                against(cache.value->view());
            }
            for (const Tensor* tensor : bank_owners(fixed)) against(tensor->view());
            for (const Tensor* tensor : bank_owners(candidate.banks))
                against(tensor->view());
            against(logits->view());
        };
        validate_owner(candidate.operation_workspace.get(), operation);
        validate_owner(candidate.selector_device_scratch.get(),
                       selector_requirements.device);

        const auto disjoint = [](const void* first, std::size_t first_bytes,
                                 const void* second, std::size_t second_bytes) {
            if (first_bytes == 0 || second_bytes == 0) return;
            const auto a = reinterpret_cast<std::uintptr_t>(first);
            const auto b = reinterpret_cast<std::uintptr_t>(second);
            const auto a_end = checked_add(a, first_bytes, "scratch end overflows");
            const auto b_end = checked_add(b, second_bytes, "scratch end overflows");
            if (a < b_end && b < a_end)
                throw std::invalid_argument("session scratch ranges overlap");
        };
        const RawWorkspace* op = candidate.operation_workspace.get();
        const RawWorkspace* selector_device = candidate.selector_device_scratch.get();
        const auto address = [](const RawWorkspace* owner) {
            return owner ? detail::WorkspaceValidation::address(owner->view())
                         : nullptr;
        };
        disjoint(address(op), operation.bytes,
                 address(selector_device), selector_requirements.device.bytes);
        for (const RawWorkspace* owner : {op, selector_device}) {
            if (owner)
                disjoint(address(owner), owner->byte_size(),
                         candidate.selector_host_scratch.data(),
                         candidate.selector_host_scratch.size());
        }
    }

    void prepare_request(
            std::size_t R, WorkspaceRequirements operation,
            TokenSelectorScratchRequirements selector_requirements) {
        if (R == 0) {
            throw std::invalid_argument(
                    "TinyLlama session request length must be nonzero");
        }
        if (R > config().max_position_embeddings) {
            throw std::invalid_argument(
                    "TinyLlama session request exceeds model context");
        }
        if (poisoned) {
            throw std::logic_error("TinyLlama session is poisoned");
        }
        validate_workspace_requirement(operation, "operation");
        validate_workspace_requirement(
                selector_requirements.device, "selector device");
        validate_vector_capacity<std::byte>(selector_requirements.host_bytes);
        validate_vector_capacity<std::size_t>(config().max_position_embeddings);
        static_cast<void>(run_storage_bytes(config(), R));
        const std::size_t oid_capacity = accepted_oid_capacity(
                config(), "TinyLlama session accepted-OID capacity overflows");

        // The caller must pass the exact requirement of this selector and the
        // fixed logical-R1 logits bank.  Querying is pure and happens before
        // any candidate request owner is constructed.
        const TokenSelectorScratchRequirements actual_selector =
                selector->scratch_requirements(
                        logits->view(), config().vocab_size);
        if (actual_selector.host_bytes != selector_requirements.host_bytes
                || actual_selector.device.bytes
                        != selector_requirements.device.bytes
                || actual_selector.device.alignment
                        != selector_requirements.device.alignment) {
            throw std::invalid_argument(
                    "TinyLlama selector scratch requirement does not match"
                    " the supplied request requirement");
        }

        // A replacement cannot be published while old queue work is live.
        // Allocation of the candidate starts only after a successful full
        // drain, while the old request remains intact if candidate setup throws.
        drain_request();

        auto candidate = std::make_unique<RequestState>();
        candidate->run_length = R;
        candidate->banks = allocate_run_banks(*device, config(), R);
        candidate->history.reserve(config().max_position_embeddings);
        candidate->results.reserve(config().max_position_embeddings);
        candidate->accepted_oids.reserve(oid_capacity);

        if (operation.bytes != 0) {
            candidate->operation_workspace =
                    device->create_workspace(operation.bytes);
            validate_workspace_owner(
                    candidate->operation_workspace.get(), *device, operation,
                    "operation");
        }
        candidate->selector_host_scratch.resize(
                selector_requirements.host_bytes);
        if (selector_requirements.device.bytes != 0) {
            candidate->selector_device_scratch = device->create_workspace(
                    selector_requirements.device.bytes);
            validate_workspace_owner(
                    candidate->selector_device_scratch.get(), *device,
                    selector_requirements.device, "selector device");
        }
        validate_scratch(*candidate, operation, selector_requirements);
        // Cache logical prefixes are reset only once the replacement is fully
        // constructed, so a synchronous setup failure leaves the old request
        // state untouched after its successful drain.
        reset_cache_prefix();

        // Publish only after every owner, range, and workspace is complete.
        request = std::move(candidate);

        // The recorder ordinal advances only here, after the request was
        // actually published: one publication replaces the admitted
        // observation, clears the outgoing operation rows without freeing
        // prepared trace storage, and publishes a staged attempt. Every
        // earlier return above - a failed drain, a rejected requirement, or a
        // failed candidate allocation - left the outgoing admitted observation
        // and its rows untouched.
        if (metrics != nullptr) {
            metrics->publish_generation();
        }
    }

    void prepare_operation_trace() {
        if (metrics == nullptr) {
            throw std::invalid_argument(
                    "TinyLlama session operation trace requires an attached"
                    " inference metrics recorder");
        }
        if (request != nullptr) {
            throw std::logic_error(
                    "TinyLlama session operation trace must be prepared before"
                    " the first request");
        }
        // The reservation reuses the request ledger's checked accepted-OID
        // bound, so a prepared trace always fits one complete request.
        // Capacity overflow and allocation failure propagate from here, before
        // any inference work exists, and tracing stays disabled because the
        // recorder enables it only after its reservation succeeded.
        metrics->prepare_trace(accepted_oid_capacity(
                config(), "TinyLlama session accepted-OID capacity overflows"));
    }
};

TinyLlamaSession::TinyLlamaSession(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

TinyLlamaSession::~TinyLlamaSession() = default;

const TinyLlamaModel& TinyLlamaSession::model() const noexcept {
    return *impl_->model;
}

const TinyLlamaConfig& TinyLlamaSession::config() const noexcept {
    return impl_->config();
}

const Tokenizer& TinyLlamaSession::tokenizer() const noexcept {
    return *impl_->tokenizer;
}

const ChatFormatter& TinyLlamaSession::formatter() const noexcept {
    return *impl_->formatter;
}

TokenSelector& TinyLlamaSession::selector() noexcept {
    return *impl_->selector;
}

const TokenSelector& TinyLlamaSession::selector() const noexcept {
    return *impl_->selector;
}

Device& TinyLlamaSession::device() const noexcept {
    return *impl_->device;
}

DeviceOps& TinyLlamaSession::queue() noexcept {
    return *impl_->queue;
}

const DeviceOps& TinyLlamaSession::queue() const noexcept {
    return *impl_->queue;
}

void TinyLlamaSession::prepare_request(
        std::size_t run_length, WorkspaceRequirements operation,
        TokenSelectorScratchRequirements selector_scratch) {
    impl_->prepare_request(run_length, operation, selector_scratch);
}

void TinyLlamaSession::prepare_operation_trace() {
    impl_->prepare_operation_trace();
}

TokenGenerationResult TinyLlamaSession::generate_tokens(
        std::span<const std::size_t> token_ids,
        std::size_t max_new_tokens) {
    // Low-level timing starts here, before validation and provisioning, and
    // stages the attempted input cardinality.  The attempt is published by
    // `prepare_forward_request` only after it replaced the request, so a
    // validation, setup, or old-drain failure is reported as this attempt's
    // failure and never overwrites the outgoing admitted request, its
    // completed counters, or its operation rows.
    InferenceMetrics* const metrics = impl_->metrics;
    std::uint64_t decode_begin = 0;
    bool decode_pending = false;
    bool current_from_decode = false;
    if (metrics != nullptr) {
        metrics->begin_generation(metrics->now(), token_ids.size());
    }
    try {
        const TinyLlamaConfig& config = impl_->config();
        validate_generation_prompt(config, token_ids);

        session_detail::SessionAccess::prepare_forward_request(
                *this, token_ids.size());
        auto& history = session_detail::SessionAccess::history(*this);
        auto& results = session_detail::SessionAccess::results(*this);
        const auto finish = [&](GenerationStopReason reason) {
            if (metrics != nullptr) {
                // The recorder stores no finish instant, so no clock read is
                // owed to one.
                metrics->end_generation(0, reason);
            }
            return TokenGenerationResult{std::move(results), reason};
        };

        // A zero limit still admits and provisions the nonempty prompt, but
        // never submits prefill work or asks the selector to inspect logits.
        if (max_new_tokens == 0) {
            return finish(GenerationStopReason::max_new_tokens);
        }

        const auto require_ready_result =
                [&](const session_detail::ForwardResult& result) {
                    const bool shape_ok = result.logits != nullptr
                            && result.logits == &impl_->logits->view()
                            && result.logits->spec().data_type == DataType::BF16
                            && result.logits->spec().quantization
                                    == QuantizationFormat::NONE
                            && result.logits->spec().shape.rank() == 2
                            && result.logits->spec().shape.dimensions()[0] == 1
                            && result.logits->spec().shape.dimensions()[1]
                                    == config.vocab_size
                            && &result.logits->device() == impl_->device
                            && result.logits->owner_identity()
                                    == impl_->logits.get();
                    const auto accepted =
                            session_detail::SessionAccess::accepted(*this);
                    const bool producer_ok = oid_is_token(result.producer)
                            && std::find(
                                       accepted.begin(), accepted.end(),
                                       result.producer)
                                    != accepted.end();
                    if (!shape_ok || !producer_ok) {
                        impl_->poison_and_drain();
                        throw std::logic_error(
                                "TinyLlama generation received an invalid "
                                "final-logit producer");
                    }
                };

        const auto select_next =
                [&](const session_detail::ForwardResult& result) {
                    require_ready_result(result);
                    try {
                        // Selection is synchronous.  The selector repeats this
                        // wait for the established seam, while this barrier
                        // also protects injected selectors that only observe
                        // their borrowed arguments.
                        session_detail::SessionAccess::wait(
                                *this, result.producer);
                        if (metrics != nullptr && decode_pending) {
                            // This is the existing first final-logits wait for
                            // the decode forward, so it ends both its
                            // completion-observed span and its successful
                            // readiness observation.  The interval opened
                            // before the forward stays open until the token
                            // that forward produces is committed.
                            const std::uint64_t ready = metrics->now();
                            metrics->record_decode(
                                    decode_begin, ready,
                                    ObservationState::succeeded);
                            metrics->record_decode_forward(ready);
                            decode_pending = false;
                        }
                        return impl_->selector->select(
                                *impl_->queue, *result.logits,
                                config.vocab_size, result.producer, history,
                                session_detail::SessionAccess::
                                        selector_scratch(*this));
                    } catch (...) {
                        impl_->poison_and_drain();
                        throw;
                    }
                };

        // Prefill is one completion-observed span per submitted request. It
        // begins immediately before the forward submission and ends after the
        // already-required initial final-logits readiness wait below, which is
        // the existing pre-history boundary: `forward_prefill` contains its own
        // correctness waits, so the span deliberately includes them and is
        // never host-enqueue-only time. A failed attempt records an incomplete
        // `failed` observation and rethrows the original exception unchanged,
        // without an added drain, second wait, callback, or altered error path.
        std::uint64_t prefill_begin = 0;
        if (metrics != nullptr) {
            prefill_begin = metrics->now();
        }
        session_detail::ForwardResult current;
        try {
            current = session_detail::SessionAccess::forward_prefill(
                    *this, token_ids);
            require_ready_result(current);
            session_detail::SessionAccess::wait(*this, current.producer);
        } catch (...) {
            if (metrics != nullptr) {
                metrics->record_prefill(
                        prefill_begin, metrics->now(),
                        ObservationState::failed);
            }
            throw;
        }
        if (metrics != nullptr) {
            metrics->record_prefill(prefill_begin, metrics->now());
        }
        history.assign(token_ids.begin(), token_ids.end());

        if (history.size() == config.max_position_embeddings) {
            return finish(GenerationStopReason::context_capacity);
        }

        for (;;) {
            const std::size_t selected = select_next(current);
            if (selected >= config.vocab_size) {
                impl_->poison_and_drain();
                throw std::invalid_argument(
                        "TinyLlama generation selector returned a token outside "
                        "the vocabulary");
            }

            // Request setup reserves both vectors to context capacity, so a
            // successful token commit performs no per-token allocation.
            history.push_back(selected);
            results.push_back(selected);
            if (metrics != nullptr) {
                // Every actual commit counts, including a terminal token; the
                // first one ends TTFT and a decode-produced one closes its
                // decode interval.
                metrics->commit_token(metrics->now(), current_from_decode);
            }

            // Stop precedence is deliberately independent of cache
            // publication: EOS wins, then the requested generation limit, then
            // context.
            if (selected == config.eos_token_id) {
                return finish(GenerationStopReason::eos);
            }
            if (results.size() >= max_new_tokens) {
                return finish(GenerationStopReason::max_new_tokens);
            }
            if (history.size() >= config.max_position_embeddings) {
                return finish(GenerationStopReason::context_capacity);
            }

            // A nonterminal token is the next decode input.  The decoder
            // updates initialized cache length only after both K/V appends
            // complete.
            if (metrics != nullptr) {
                // The decode interval opens immediately before the forward and
                // closes at the committed token that forward produces.
                decode_begin = metrics->now();
                metrics->record_decode_interval(decode_begin);
                decode_pending = true;
                current_from_decode = true;
            }
            current = session_detail::SessionAccess::forward_decode(
                    *this, selected);
        }
    } catch (...) {
        // The original exception category is rethrown unchanged.  A decode
        // interval whose readiness was never observed is a failed observation,
        // and a failed phase stores no duration, so its interval start is the
        // observation instant.
        if (metrics != nullptr) {
            if (decode_pending) {
                metrics->record_decode(
                        decode_begin, decode_begin, ObservationState::failed);
            }
            metrics->end_generation_failure(0, std::current_exception());
        }
        throw;
    }
}

GenerationResult TinyLlamaSession::generate_raw(
        std::string_view text, std::size_t max_new_tokens) {
    TokenizationObservation observation(
            session_detail::SessionAccess::metrics(*this));
    std::vector<std::uint32_t> encoded =
            observed_encode(observation, [&] {
                return tokenizer().encode(
                        text, EncodeOptions{.add_special_tokens = true});
            });
    return finish_text_generation(
            *this, std::move(encoded), max_new_tokens, observation);
}

GenerationResult TinyLlamaSession::generate_chat(
        std::span<const ChatMessageView> messages,
        std::size_t max_new_tokens) {
    TokenizationObservation observation(
            session_detail::SessionAccess::metrics(*this));
    std::string rendered;
    try {
        // The complete supported chat rendering precedes the measured encode.
        rendered = formatter().format(messages, true);
    } catch (...) {
        observation.record_preprocessing_failure(std::current_exception());
        throw;
    }
    std::vector<std::uint32_t> encoded =
            observed_encode(observation, [&] {
                return tokenizer().encode(
                        rendered, EncodeOptions{.add_special_tokens = true});
            });
    return finish_text_generation(
            *this, std::move(encoded), max_new_tokens, observation);
}

std::size_t TinyLlamaSession::request_length() const noexcept {
    return impl_->request == nullptr ? 0 : impl_->request->run_length;
}

bool TinyLlamaSession::poisoned() const noexcept {
    return impl_->poisoned;
}

std::unique_ptr<TinyLlamaSession> load_tinyllama_session(
        const std::filesystem::path& model_directory, Device& device) {
    return load_tinyllama_session(
            model_directory, device,
            std::make_unique<GreedyTokenSelector>());
}

std::unique_ptr<TinyLlamaSession> load_tinyllama_session(
        const std::filesystem::path& model_directory, Device& device,
        std::unique_ptr<TokenSelector> selector, InferenceMetrics* metrics) {
    // The instrumented interval starts here, before any validation, and ends
    // at successful publication or the failed unwind below. Outer
    // device/allocator setup and the construction of the caller-supplied
    // selector are outside it. A null recorder reads no clock at all.
    std::uint64_t load_begin = 0;
    if (metrics != nullptr) {
        load_begin = metrics->now();
    }
    try {
        // This is deliberately the first operation: a null selector is rejected
        // before the session, queue, model, tokenizer, formatter, tensor, cache,
        // workspace, history, result, or scratch allocation paths are touched.
        if (!selector) {
            throw std::invalid_argument(
                    "TinyLlama session selector must not be null");
        }
        const auto supported = device.supported_data_types();
        for (const DataType type : {DataType::BF16, DataType::U32}) {
            if (std::find(supported.begin(), supported.end(), type) == supported.end())
                throw std::invalid_argument(
                        "TinyLlama session requires BF16 and U32 storage");
        }

        auto impl = std::make_unique<TinyLlamaSession::Impl>();
        impl->device = &device;
        impl->metrics = metrics;
        impl->selector = std::move(selector);

        // The model factory is the sole source of mapped checkpoint ownership and
        // canonical uploaded weight owners.  No session-level weight copy exists.
        impl->model = load_tinyllama_model(model_directory, device);
        validate_persistent_storage(impl->model->config());
        impl->tokenizer = load_tokenizer(model_directory);
        impl->formatter = load_chat_formatter(model_directory);
        impl->queue = device.create_ops();
        if (!impl->queue) {
            throw std::runtime_error(
                    "TinyLlama session device returned a null operation queue");
        }
        if (&impl->queue->device() != &device) {
            throw std::invalid_argument(
                    "TinyLlama session queue belongs to another Device instance");
        }

        // Persistent cache owners and all fixed-R1 decode roles are constructed
        // only after model/configuration validation and exactly once per session.
        const TinyLlamaConfig& config = impl->model->config();
        impl->caches.reserve(config.num_hidden_layers);
        for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
            CacheOwner cache;
            cache.key = make_tensor(device, {config.num_key_value_heads,
                                             config.max_position_embeddings,
                                             config.head_dim});
            cache.value = make_tensor(device, {config.num_key_value_heads,
                                               config.max_position_embeddings,
                                               config.head_dim});
            impl->caches.push_back(std::move(cache));
        }
        impl->fixed = allocate_run_banks(device, config, 1);
        impl->logits = make_tensor(device, {1, config.vocab_size});

        auto session = std::unique_ptr<TinyLlamaSession>(
                new TinyLlamaSession(std::move(impl)));
        if (metrics != nullptr) {
            metrics->record_load(load_begin, metrics->now());
        }
        return session;
    } catch (...) {
        // The original exception category is rethrown unchanged; an enabled
        // failure is recorded with no session, no timestamp, and no duration.
        if (metrics != nullptr) {
            metrics->record_load(load_begin, metrics->now(),
                                 ObservationState::failed);
        }
        throw;
    }
}

}  // namespace iom
namespace iom::session_detail {

const RunBanks& SessionAccess::prefill(TinyLlamaSession& session) {
    session.impl_->require_request();
    return session.impl_->request->banks;
}

const RunBanks& SessionAccess::decode(TinyLlamaSession& session) {
    session.impl_->require_request();
    return session.impl_->fixed;
}

TensorView& SessionAccess::logits(TinyLlamaSession& session) {
    return session.impl_->logits->view();
}

std::span<CacheOwner> SessionAccess::caches(TinyLlamaSession& session) {
    session.impl_->require_request();
    return session.impl_->caches;
}

RawWorkspaceView SessionAccess::workspace(TinyLlamaSession& session) {
    session.impl_->require_request();
    const auto& owner = session.impl_->request->operation_workspace;
    return owner ? owner->view() : RawWorkspaceView{};
}

TokenSelectorScratch SessionAccess::selector_scratch(TinyLlamaSession& session) {
    session.impl_->require_request();
    auto& request = *session.impl_->request;
    return {request.selector_host_scratch,
            request.selector_device_scratch
                    ? request.selector_device_scratch->view()
                    : RawWorkspaceView{}};
}

std::vector<std::size_t>& SessionAccess::history(TinyLlamaSession& session) {
    session.impl_->require_request();
    return session.impl_->request->history;
}

std::vector<std::size_t>& SessionAccess::results(TinyLlamaSession& session) {
    session.impl_->require_request();
    return session.impl_->request->results;
}

std::span<const oid> SessionAccess::accepted(const TinyLlamaSession& session) {
    return session.impl_->request
            ? std::span<const oid>{session.impl_->request->accepted_oids}
            : std::span<const oid>{};
}

InferenceMetrics* SessionAccess::metrics(TinyLlamaSession& session) noexcept {
    return session.impl_->metrics;
}

void SessionAccess::set_operation_context(
        TinyLlamaSession& session, const OperationContext& context) noexcept {
    if (session.impl_->metrics == nullptr) {
        // Fully disabled observation keeps no attribution bookkeeping at all.
        return;
    }
    session.impl_->operation_context = context;
}

void SessionAccess::require_submission(TinyLlamaSession& session) {
    session.impl_->require_request();
    const auto& tokens = session.impl_->request->accepted_oids;
    if (tokens.size() == tokens.capacity())
        throw std::overflow_error("TinyLlama session submission bound exceeded");
    // The prepared trace table is the only storage for accepted observations,
    // and its fixed capacity is bounded by the same checked accepted-OID
    // bound as the ledger above. An accepted submission whose observation
    // could not be prepared is refused here with the same checked-bound
    // category instead of being dropped, grown, or silently lost after the
    // facade accepted it.
    const InferenceMetrics* recorder = session.impl_->metrics;
    if (recorder != nullptr && recorder->trace_prepared()
            && recorder->trace_capacity_remaining() == 0) {
        throw std::overflow_error(
                "TinyLlama session operation trace capacity exceeded");
    }
}

void SessionAccess::observe_submission(
        TinyLlamaSession& session, oid accepted,
        std::uint64_t enqueue_begin, std::uint64_t enqueue_end) noexcept {
    // Requires an attached recorder, which the submit template established
    // before it measured the facade call.
    const OperationContext& context = session.impl_->operation_context;
    session.impl_->metrics->record_enqueue(
            accepted, context.phase, context.decoder_layer,
            context.position_start, context.run_length, enqueue_begin,
            enqueue_end);
}

oid SessionAccess::record_submission(TinyLlamaSession& session, oid token) {
    if (oid_is_token(token)) {
        // submit() checked room before the facade accepted anything.
        session.impl_->request->accepted_oids.push_back(token);
        return token;
    }
    session.impl_->poisoned = true;
    try { session.impl_->drain_request(); } catch (...) {}
    switch (token) {
        case to_oid(OidError::InvalidArgument):
            throw std::invalid_argument("TinyLlama session submission rejected");
        case to_oid(OidError::Unsupported):
            throw detail::UnsupportedOperation{};
        case to_oid(OidError::Overflow):
            throw std::overflow_error("TinyLlama session submission overflow");
        case to_oid(OidError::ResourceExhausted):
            throw std::bad_alloc{};
        case to_oid(OidError::DeviceError):
            throw std::runtime_error("TinyLlama session device failure");
        default:
            throw std::logic_error("TinyLlama session invalid admission result");
    }
}

void SessionAccess::wait(TinyLlamaSession& session, oid token) {
    try {
        session.impl_->queue->wait(token);
    } catch (...) {
        // Save the original failure before any observation, poison, or drain
        // activity: the wait observation and the retained error both consume
        // this exact exception, and the existing error path stays unchanged.
        const std::exception_ptr failure = std::current_exception();
        session.impl_->observe_wait_failure(token, failure);
        session.impl_->poisoned = true;
        try { session.impl_->drain_request(); } catch (...) {}
        throw;
    }
    session.impl_->observe_wait_success(token);
}

void SessionAccess::drain(TinyLlamaSession& session) {
    session.impl_->drain_request();
}

namespace {

// The base alignment every device guarantees for a caller-owned raw workspace
// range, and the granularity at which the stage's fixed slices are reserved.
constexpr std::size_t kMlpSliceAlignment = 32;

[[nodiscard]] std::span<const std::size_t> mlp_extents(
        const TensorView& view) noexcept {
    return view.spec().shape.dimensions();
}

// Leading extents of one activation view: everything before the final two
// matrix axes. These stay independent and are never broadcast.
[[nodiscard]] std::span<const std::size_t> mlp_leading_extents(
        const TensorView& view) noexcept {
    const std::span<const std::size_t> dimensions = mlp_extents(view);
    return dimensions.first(dimensions.size() - 2);
}

// Final two extents of one activation view: the logical matrix axes.
[[nodiscard]] std::span<const std::size_t> mlp_matrix_extents(
        const TensorView& view) noexcept {
    const std::span<const std::size_t> dimensions = mlp_extents(view);
    return dimensions.subspan(dimensions.size() - 2);
}

[[nodiscard]] bool mlp_same_extents(
        std::span<const std::size_t> lhs,
        std::span<const std::size_t> rhs) noexcept {
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index] != rhs[index]) return false;
    }
    return true;
}

// One checked plain-BF16 view requirement. `expected` holds the required final
// two extents; `message` names the view and the violated boundary.
void mlp_require_view(
        const TensorView& view, std::span<const std::size_t> expected,
        const char* message) {
    if (view.spec().data_type != DataType::BF16
            || view.spec().quantization != QuantizationFormat::NONE
            || !mlp_same_extents(mlp_matrix_extents(view), expected)) {
        throw std::invalid_argument(message);
    }
}

void mlp_validate_views(
        const MlpStageViews& views, const MlpStageParams& params) {
    if (params.rows == 0 || params.features == 0
            || params.intermediate == 0) {
        throw std::invalid_argument(
                "TinyLlama MLP stage requires nonzero row, feature, and "
                "intermediate sizes");
    }
    if (!std::isfinite(params.rms_epsilon) || params.rms_epsilon < 0.0F) {
        throw std::invalid_argument(
                "TinyLlama MLP stage requires a finite nonnegative epsilon");
    }

    const std::size_t residual_extents[2] = {params.rows, params.features};
    const std::size_t projection_extents[2] = {
            params.rows, params.intermediate};
    const std::size_t scale_extents[2] = {1, params.features};
    const std::size_t gate_weight_extents[2] = {
            params.intermediate, params.features};
    const std::size_t down_weight_extents[2] = {
            params.features, params.intermediate};

    mlp_require_view(
            views.x2, residual_extents,
            "TinyLlama MLP stage first residual must be a plain BF16 [R,F] "
            "view");
    mlp_require_view(
            views.n2, residual_extents,
            "TinyLlama MLP stage norm store must be a plain BF16 [R,F] view");
    mlp_require_view(
            views.down, residual_extents,
            "TinyLlama MLP stage down store must be a plain BF16 [R,F] view");
    mlp_require_view(
            views.next_x, residual_extents,
            "TinyLlama MLP stage next-residual store must be a plain BF16 "
            "[R,F] view");
    mlp_require_view(
            views.gate, projection_extents,
            "TinyLlama MLP stage gate store must be a plain BF16 [R,M] view");
    mlp_require_view(
            views.up, projection_extents,
            "TinyLlama MLP stage up store must be a plain BF16 [R,M] view");
    mlp_require_view(
            views.activated_gate, projection_extents,
            "TinyLlama MLP stage activated-gate store must be a plain BF16 "
            "[R,M] view");
    mlp_require_view(
            views.product, projection_extents,
            "TinyLlama MLP stage product store must be a plain BF16 [R,M] "
            "view");
    mlp_require_view(
            views.post_attention_scale, scale_extents,
            "TinyLlama MLP stage post-attention scale must be a plain BF16 "
            "[1,F] view");
    mlp_require_view(
            views.gate_weight, gate_weight_extents,
            "TinyLlama MLP stage gate weight must be a plain BF16 [M,F] view");
    mlp_require_view(
            views.up_weight, gate_weight_extents,
            "TinyLlama MLP stage up weight must be a plain BF16 [M,F] view");
    mlp_require_view(
            views.down_weight, down_weight_extents,
            "TinyLlama MLP stage down weight must be a plain BF16 [F,M] view");

    const TensorView* const activations[8] = {
            &views.x2, &views.n2, &views.gate, &views.up,
            &views.activated_gate, &views.product, &views.down, &views.next_x};
    for (std::size_t index = 1; index < 8; ++index) {
        if (!mlp_same_extents(
                    mlp_leading_extents(*activations[0]),
                    mlp_leading_extents(*activations[index]))) {
            throw std::invalid_argument(
                    "TinyLlama MLP stage activation views must share one "
                    "independent leading tuple");
        }
    }
}

// Category of one negative admission OID, mapped back to the established
// exception class the facades map from. Negative OIDs are synchronous errors
// that must never be waited, and the mapped exception is constructed only when
// it is thrown.
[[noreturn]] void mlp_throw_admission_failure(oid token) {
    switch (token) {
        case to_oid(OidError::InvalidArgument):
            throw std::invalid_argument(
                    "TinyLlama MLP stage rejected an operand as invalid input");
        case to_oid(OidError::Unsupported):
            throw detail::UnsupportedOperation{};
        case to_oid(OidError::Overflow):
            throw std::overflow_error(
                    "TinyLlama MLP stage rejected an operand as overflowing");
        case to_oid(OidError::ResourceExhausted):
            throw std::bad_alloc{};
        case to_oid(OidError::DeviceError):
            throw std::runtime_error(
                    "TinyLlama MLP stage failed on its device");
        case to_oid(OidError::InternalError):
            throw std::logic_error("TinyLlama MLP stage failed internally");
        default:
            throw std::logic_error(
                    "TinyLlama MLP stage received an invalid operation "
                    "result");
    }
}

// First-failure preservation with complete draining. Every accepted OID is
// attempted even when an earlier wait throws, a later success never masks an
// earlier failure, and a negative admission OID is recorded without ever being
// waited.
class MlpFailureSpool {
public:
    explicit MlpFailureSpool(MlpStageFailure& record) noexcept
        : record_(&record) {}

    // Records one negative admission OID as the first actionable failure and
    // reports whether the token was accepted.
    [[nodiscard]] bool accept(oid token) noexcept {
        if (oid_is_token(token)) return true;
        if (first_ == nullptr) {
            admission_ = token;
        }
        return false;
    }

    void wait(DeviceOps& operations, oid token) noexcept {
        try {
            operations.wait(token);
        } catch (...) {
            capture(std::current_exception());
        }
    }

    [[nodiscard]] bool failed() const noexcept {
        return first_ != nullptr || admission_ != 0;
    }

    // Records the preserved first failure and rethrows it.
    [[noreturn]] void publish() {
        if (first_ != nullptr) {
            record_->record(first_);
            std::rethrow_exception(first_);
        }
        if (admission_ != 0) {
            record_->record_admission(admission_);
            mlp_throw_admission_failure(admission_);
        }
        throw std::logic_error(
                "TinyLlama MLP stage has no failure to publish");
    }

private:
    // A wait failure is only remembered while no earlier failure exists, so
    // the recorded failure is always the first actionable one.
    void capture(std::exception_ptr failure) noexcept {
        if (failure != nullptr && first_ == nullptr && admission_ == 0) {
            first_ = std::move(failure);
        }
    }

    MlpStageFailure* record_;
    std::exception_ptr first_;
    oid admission_ = 0;
};

// Rounds one positive byte count up to the fixed slice granularity with
// checked arithmetic; a zero count stays zero.
[[nodiscard]] std::size_t mlp_aligned_extent(std::size_t bytes) {
    if (bytes == 0) return 0;
    const std::size_t mask = kMlpSliceAlignment - 1;
    return detail::checked_add(
                   bytes, mask, "post-attention MLP workspace extent")
            & ~mask;
}

// Fixed member layout of the resolved workspace. The reused slice holds the
// gate projection and is reused by mul, down, and the second residual only
// after that producer and its readers completed; the up projection keeps its
// own disjoint slice because both projections are enqueued before either is
// waited.
struct MlpSliceLayout {
    std::size_t reused_bytes;
    std::size_t up_offset;
    std::size_t up_bytes;
};

[[nodiscard]] MlpSliceLayout mlp_slice_layout(
        const MlpWorkspaceRequirements& requirements) {
    const std::size_t reused = std::max(
            std::max(requirements.gate.bytes, requirements.mul.bytes),
            std::max(requirements.down.bytes, requirements.residual.bytes));
    return MlpSliceLayout{
            reused, mlp_aligned_extent(reused), requirements.up.bytes};
}

[[nodiscard]] WorkspaceRequirements mlp_wider(
        WorkspaceRequirements lhs, WorkspaceRequirements rhs) noexcept {
    return WorkspaceRequirements{
            std::max(lhs.bytes, rhs.bytes),
            std::max(lhs.alignment, rhs.alignment)};
}

// The five member requirements of one actual run bank. Every query runs the
// exact admission validation of its operation and reports the requirement
// without allocation, registration, lease, or submission.
[[nodiscard]] MlpWorkspaceRequirements mlp_bank_requirements(
        DeviceOps& operations, const MlpStageViews& views) {
    const std::span<const std::size_t> gate_extents =
            mlp_matrix_extents(views.gate);
    const std::size_t rows = gate_extents[0];
    const std::size_t intermediate = gate_extents[1];
    const std::size_t features = mlp_matrix_extents(views.down)[1];
    const std::size_t projection_extents[2] = {rows, intermediate};
    const std::size_t residual_extents[2] = {rows, features};
    mlp_require_view(
            views.gate, projection_extents,
            "TinyLlama MLP stage gate store must be a plain BF16 [R,M] view");
    mlp_require_view(
            views.down, residual_extents,
            "TinyLlama MLP stage down store must be a plain BF16 [R,F] view");

    return MlpWorkspaceRequirements{
            operations.linear_workspace_requirements(
                    views.n2, views.gate_weight, views.gate, 0, rows,
                    LinearOutputLayout::ordinary, 1, intermediate),
            operations.linear_workspace_requirements(
                    views.n2, views.up_weight, views.up, 0, rows,
                    LinearOutputLayout::ordinary, 1, intermediate),
            operations.mul_workspace_requirements(
                    views.activated_gate, views.up, views.product),
            operations.linear_workspace_requirements(
                    views.product, views.down_weight, views.down, 0, rows,
                    LinearOutputLayout::ordinary, 1, features),
            operations.add_workspace_requirements(
                    views.x2, views.down, views.next_x)};
}

// One member view of the reserved workspace. A zero-byte member keeps the
// empty default view, which is the only admissible workspace for the
// zero-requirement operations; a positive member must be a live, sufficient,
// aligned range of the exact device.
[[nodiscard]] RawWorkspaceView mlp_member_view(
        const Device& device, RawWorkspaceView workspace, std::size_t offset,
        WorkspaceRequirements requirement) {
    if (requirement.bytes == 0) return RawWorkspaceView{};
    return detail::WorkspaceValidation::validated(
            device, workspace.subrange(offset, requirement.bytes),
            requirement.bytes, requirement.alignment,
            std::span<const TensorView>{});
}

}  // namespace

void MlpStageFailure::record(std::exception_ptr failure) noexcept {
    if (failure != nullptr && admission_ == 0 && first_ == nullptr) {
        first_ = std::move(failure);
    }
}

void MlpStageFailure::record_admission(oid token) noexcept {
    if (token < 0 && admission_ == 0 && first_ == nullptr) {
        admission_ = token;
    }
}

[[noreturn]] void MlpStageFailure::rethrow_first() const {
    if (first_ != nullptr) {
        std::rethrow_exception(first_);
    }
    if (admission_ != 0) {
        mlp_throw_admission_failure(admission_);
    }
    throw std::logic_error("TinyLlama MLP stage failure record is empty");
}

[[nodiscard]] WorkspaceRequirements MlpWorkspaceRequirements::total() const {
    const MlpSliceLayout layout = mlp_slice_layout(*this);
    const std::size_t bytes = detail::checked_add(
            layout.up_offset, mlp_aligned_extent(layout.up_bytes),
            "post-attention MLP workspace requirement");
    if (bytes == 0) return WorkspaceRequirements{0, 1};
    const std::size_t alignment = std::max(
            {kMlpSliceAlignment, gate.alignment, up.alignment, mul.alignment,
             down.alignment, residual.alignment});
    if ((alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(
                "TinyLlama MLP stage requires a power-of-two workspace "
                "alignment");
    }
    return WorkspaceRequirements{bytes, alignment};
}

[[nodiscard]] MlpWorkspaceRequirements mlp_workspace_requirements(
        DeviceOps& operations, const MlpStageViews& prefill,
        const MlpStageViews& decode) {
    const MlpWorkspaceRequirements prefill_requirements =
            mlp_bank_requirements(operations, prefill);
    const MlpWorkspaceRequirements decode_requirements =
            mlp_bank_requirements(operations, decode);
    return MlpWorkspaceRequirements{
            mlp_wider(prefill_requirements.gate, decode_requirements.gate),
            mlp_wider(prefill_requirements.up, decode_requirements.up),
            mlp_wider(prefill_requirements.mul, decode_requirements.mul),
            mlp_wider(prefill_requirements.down, decode_requirements.down),
            mlp_wider(
                    prefill_requirements.residual,
                    decode_requirements.residual)};
}

[[nodiscard]] MlpWorkspace resolve_mlp_workspace(
        const Device& device, const MlpWorkspaceRequirements& requirements,
        RawWorkspaceView workspace) {
    const WorkspaceRequirements total = requirements.total();
    if (total.bytes == 0) {
        // Nothing is required, so the supplied range is neither validated nor
        // leased and every member keeps the empty default view.
        return MlpWorkspace{};
    }
    const MlpSliceLayout layout = mlp_slice_layout(requirements);
    (void)detail::WorkspaceValidation::validated(
            device, workspace, total.bytes, total.alignment,
            std::span<const TensorView>{});

    // Aggregate construction only: a workspace view is never reassignable, so
    // every member is built once over its reserved range.
    return MlpWorkspace{
            mlp_member_view(device, workspace, 0, requirements.gate),
            mlp_member_view(
                    device, workspace, layout.up_offset, requirements.up),
            mlp_member_view(device, workspace, 0, requirements.mul),
            mlp_member_view(device, workspace, 0, requirements.down),
            mlp_member_view(device, workspace, 0, requirements.residual)};
}

void run_mlp_stage(DeviceOps& operations, MlpStageViews& views,
                   const MlpStageParams& params, const MlpWorkspace& workspace,
                   std::span<const oid> readiness, MlpStageFailure& failure) {
    // A poisoned execution is never reused: the preserved first failure is
    // rethrown without submitting anything.
    if (failure.poisoned()) {
        failure.rethrow_first();
    }
    mlp_validate_views(views, params);
    for (const oid token : readiness) {
        if (!oid_is_token(token)) {
            throw std::invalid_argument(
                    "TinyLlama MLP stage readiness requires accepted producer "
                    "tokens");
        }
    }

    MlpFailureSpool spool(failure);

    // The supplied first-residual producer must complete before the stage
    // reads `x2`, and every readiness producer is attempted even when an
    // earlier one fails.
    for (const oid token : readiness) {
        spool.wait(operations, token);
    }
    if (spool.failed()) spool.publish();

    // Post-attention RMSNorm into its own store; RMSNorm consumes the empty
    // workspace under its zero-workspace contract.
    const oid norm = operations.rmsnorm(
            views.x2, views.post_attention_scale, views.n2,
            params.rms_epsilon);
    if (!spool.accept(norm)) spool.publish();
    spool.wait(operations, norm);
    if (spool.failed()) spool.publish();

    // Gate and up are independent branches of the same ready normalization:
    // both are submitted before either is waited, and both accepted branches
    // are attempted even when the first wait reports failure.
    const oid gate = operations.linear(
            views.n2, views.gate_weight, views.gate, 0, params.rows,
            LinearOutputLayout::ordinary, 1, params.intermediate,
            workspace.gate);
    const oid up = operations.linear(
            views.n2, views.up_weight, views.up, 0, params.rows,
            LinearOutputLayout::ordinary, 1, params.intermediate,
            workspace.up);
    const bool gate_accepted = spool.accept(gate);
    const bool up_accepted = spool.accept(up);
    if (gate_accepted) spool.wait(operations, gate);
    if (up_accepted) spool.wait(operations, up);
    if (spool.failed()) spool.publish();

    // Existing BF16 SiLU of the stored gate, then the existing multiply in the
    // `SiLU(gate), up` operand order. Neither boundary is fused.
    const oid activated = operations.silu(views.gate, views.activated_gate);
    if (!spool.accept(activated)) spool.publish();
    spool.wait(operations, activated);
    if (spool.failed()) spool.publish();

    const oid product = operations.mul(
            views.activated_gate, views.up, views.product, workspace.mul);
    if (!spool.accept(product)) spool.publish();
    spool.wait(operations, product);
    if (spool.failed()) spool.publish();

    // Ordinary down projection with the Hugging Face `[F,M]` weight, then the
    // second residual `x2 + down` into the next layer input.
    const oid down = operations.linear(
            views.product, views.down_weight, views.down, 0, params.rows,
            LinearOutputLayout::ordinary, 1, params.features, workspace.down);
    if (!spool.accept(down)) spool.publish();
    spool.wait(operations, down);
    if (spool.failed()) spool.publish();

    const oid residual = operations.add(
            views.x2, views.down, views.next_x, workspace.residual);
    if (!spool.accept(residual)) spool.publish();
    spool.wait(operations, residual);
    if (spool.failed()) spool.publish();
}
/*
 * Private TinyLlama session implementation.
 *
 * This translation unit owns the numerical stages of the session forward
 * path. Every stage is a private, allocation-free helper over caller-owned
 * views, the session's single `DeviceOps` queue, supplied parameters, and
 * caller-owned workspace; nothing here creates a public stage API, a graph,
 * a scheduler, or an owner.
 */

namespace {

    using detail::UnsupportedOperation;
    using detail::checked_add;
    using detail::checked_mul;

    // Supplied head-planar geometry `[H,R,D]` of one rank-three view.
    struct HeadPlane {
        std::size_t heads = 0;
        std::size_t rows = 0;
        std::size_t features = 0;
    };

    // Supplied matrix geometry `[R,F]` of one rank-two view.
    struct Matrix {
        std::size_t rows = 0;
        std::size_t features = 0;
    };

    // Checked stage facts derived before any submission.
    struct CheckedStage {
        std::size_t merged_width = 0;
        std::size_t initialized_length = 0;
    };

    // One supplied view of the stage, with the role name used in rejection
    // text and whether the stage writes it.
    struct StageOwnership {
        const TensorView* view;
        const char* role;
        bool store;
    };

    [[noreturn]] void reject(
            std::string_view role, std::string_view requirement) {
        throw std::invalid_argument(
                "cache attention stage " + std::string(role) + " "
                + std::string(requirement));
    }

    /**
     * Stage-wide conservative storage disjointness. The K/V cache appends,
     * SDPA, the output projection, and the first residual run in sequence over
     * the supplied views, so a store sharing storage with another supplied
     * view can destroy a value that an earlier or later phase still needs: V
     * rows would overwrite the K cache, the merged attention would overwrite
     * the residual input, and so on. Every store must therefore live in
     * distinct owner storage from every other supplied view, which is the same
     * conservative owner-overlap policy the operation facades apply locally.
     * Read/read aliasing stays legal, exactly as those contracts allow.
     */
    void require_distinct_stores(
            const std::array<StageOwnership, 10>& views) {
        for (std::size_t first = 0; first < views.size(); ++first) {
            for (std::size_t second = first + 1; second < views.size();
                 ++second) {
                if (!views[first].store && !views[second].store) continue;
                if (views[first].view->owner_identity()
                        != views[second].view->owner_identity()) {
                    continue;
                }
                reject(views[first].role,
                       std::string("must not share owner storage with the ")
                               + views[second].role);
            }
        }
    }

    [[nodiscard]] const TensorSpec& require_bf16(
            const Device& device, const TensorView& view,
            std::string_view role) {
        if (&view.device() != &device) {
            reject(role, "belongs to a different device");
        }
        const TensorSpec& spec = view.spec();
        if (spec.data_type != DataType::BF16
                || spec.quantization != QuantizationFormat::NONE) {
            reject(role, "must be an unquantized BF16 tensor");
        }
        return spec;
    }

    [[nodiscard]] HeadPlane require_heads(
            const Device& device, const TensorView& view,
            std::string_view role) {
        const std::span<const std::size_t> dimensions =
                require_bf16(device, view, role).shape.dimensions();
        if (dimensions.size() != 3) {
            reject(role, "must have rank three");
        }
        return {dimensions[0], dimensions[1], dimensions[2]};
    }

    [[nodiscard]] Matrix require_matrix(
            const Device& device, const TensorView& view,
            std::string_view role) {
        const std::span<const std::size_t> dimensions =
                require_bf16(device, view, role).shape.dimensions();
        if (dimensions.size() != 2) {
            reject(role, "must have rank two");
        }
        return {dimensions[0], dimensions[1]};
    }

    /**
     * Checked request admission. Everything here is derived from supplied
     * values and borrowed view metadata, so a rejection happens before the
     * stage submits an operation, mutates a cache row, or publishes a
     * length.
     */
    [[nodiscard]] CheckedStage validate_stage(
            const DeviceOps& ops, const CacheAttentionStageRequest& request,
            const CacheAttentionStageState& state) {
        if (state.failed) {
            throw std::logic_error(
                    "cache attention stage refuses a failed session state");
        }
        if (request.R == 0) {
            throw std::invalid_argument(
                    "cache attention stage requires nonzero rows");
        }
        if (request.C == 0) {
            throw std::invalid_argument(
                    "cache attention stage requires a nonzero cache "
                    "capacity");
        }
        // The order is contractual: the capacity relation is established
        // before `a+R` is formed, so the sum can never wrap.
        if (request.a > request.C) {
            throw std::invalid_argument(
                    "cache attention stage offset exceeds the cache "
                    "capacity");
        }
        const std::size_t initialized_length = checked_add(
                request.a, request.R,
                "cache attention stage initialized length overflows");
        if (initialized_length > request.C) {
            throw std::invalid_argument(
                    "cache attention stage rows exceed the cache capacity");
        }
        if (state.initialized_length != request.a) {
            throw std::invalid_argument(
                    "cache attention stage offset must continue the "
                "published cache prefix");
        }

        const Device& device = ops.device();
        const HeadPlane q =
                require_heads(device, request.rotated_q, "rotated Q");
        const std::size_t merged_width = checked_mul(
                q.heads, q.features,
                "cache attention stage merged attention width overflows");
        if (merged_width == 0) {
            reject("rotated Q", "must have nonzero head and feature extents");
        }
        if (q.rows != request.R) {
            reject("rotated Q", "must have exactly R rows");
        }

        const HeadPlane k =
                require_heads(device, request.rotated_k, "rotated K");
        const HeadPlane v =
                require_heads(device, request.rotated_v, "rotated V");
        if (k.heads != v.heads || k.rows != v.rows
                || k.features != v.features) {
            reject("rotated K",
                   "and rotated V must share head, row, and feature "
                   "extents");
        }
        const std::size_t kv_width = checked_mul(
                k.heads, k.features,
                "cache attention stage K/V merge width overflows");
        if (kv_width == 0) {
            reject("rotated K/V", "must have nonzero head and feature extents");
        }
        if (k.rows != request.R) {
            reject("rotated K/V", "must have exactly R rows");
        }
        if (k.features != q.features) {
            reject("rotated K/V", "must use the query head width");
        }
        if (q.heads % k.heads != 0) {
            reject("rotated Q",
                   "head count must be a multiple of the KV head count");
        }

        // Distinct append boundaries: each cache keeps its own owner,
        // head geometry, feature width, and capacity row extent.
        const HeadPlane k_cache =
                require_heads(device, request.k_cache, "K cache");
        const HeadPlane v_cache =
                require_heads(device, request.v_cache, "V cache");
        if (k_cache.heads != v_cache.heads
                || k_cache.rows != v_cache.rows
                || k_cache.features != v_cache.features) {
            reject("K cache",
                   "and V cache must share head, row, and feature extents");
        }
        if (k_cache.rows != request.C) {
            reject("K cache",
                   "row extent must equal the supplied cache capacity");
        }
        if (k_cache.heads != k.heads
                || k_cache.features != k.features) {
            reject("K cache", "geometry must match the appended K/V rows");
        }

        const Matrix input =
                require_matrix(device, request.residual_input,
                               "residual input");
        if (input.rows != request.R || input.features != merged_width) {
            reject("residual input", "must be [R,F]");
        }
        const Matrix weight =
                require_matrix(device, request.o_weight, "output weight");
        if (weight.rows != merged_width
                || weight.features != merged_width) {
            reject("output weight", "must be [F,F]");
        }
        const Matrix merged =
                require_matrix(device, request.attention_merged,
                               "merged attention");
        if (merged.rows != request.R || merged.features != merged_width) {
            reject("merged attention", "must be [R,F]");
        }
        const Matrix projected =
                require_matrix(device, request.attention_output,
                               "attention output");
        if (projected.rows != request.R
                || projected.features != merged_width) {
            reject("attention output", "must be [R,F]");
        }
        const Matrix residual =
                require_matrix(device, request.residual_output,
                               "residual output");
        if (residual.rows != request.R
                || residual.features != merged_width) {
            reject("residual output", "must be [R,F]");
        }

        // Every supplied view carries its stage role and whether the stage
        // writes it. Both caches are stores, so distinct K and V owners are
        // enforced here rather than by the individual append facades, and the
        // same check rejects every cross-phase alias that would let a later
        // store destroy an earlier input (for example the merged attention
        // overwriting the residual input before the closing residual).
        require_distinct_stores(std::array<StageOwnership, 10>{{
                {&request.rotated_q, "rotated Q", false},
                {&request.rotated_k, "rotated K", false},
                {&request.rotated_v, "rotated V", false},
                {&request.k_cache, "K cache", true},
                {&request.v_cache, "V cache", true},
                {&request.residual_input, "residual input", false},
                {&request.o_weight, "output weight", false},
                {&request.attention_merged, "merged attention", true},
                {&request.attention_output, "attention output", true},
                {&request.residual_output, "residual output", true}}});

        return {merged_width, initialized_length};
    }

    /**
     * Mirrors the facade's synchronous error mapping so a rejected
     * submission surfaces through the established exception categories
     * instead of a bare negative token.
     */
    [[noreturn]] void throw_oid_failure(
            oid value, std::string_view role) {
        if (!oid_is_error(value)) {
            throw std::logic_error(
                    "cache attention stage received a non-error oid");
        }
        const std::string what =
                "cache attention stage " + std::string(role);
        switch (static_cast<OidError>(value)) {
            case OidError::InvalidArgument:
                throw std::invalid_argument(
                        what + " rejected its operands");
            case OidError::Unsupported:
                throw UnsupportedOperation();
            case OidError::Overflow:
                throw std::overflow_error(what + " overflowed");
            case OidError::ResourceExhausted:
                throw std::bad_alloc();
            case OidError::DeviceError:
                throw std::runtime_error(what + " failed on its device");
            case OidError::InternalError:
                break;
        }
        throw std::runtime_error(what + " failed internally");
    }

    [[nodiscard]] std::exception_ptr oid_failure(
            oid value, std::string_view role) noexcept {
        try {
            throw_oid_failure(value, role);
        } catch (...) {
            return std::current_exception();
        }
    }

    /**
     * Waits one submitted step and returns its first failure without
     * throwing, so a caller can record one failure and still complete
     * every remaining mandatory wait.
     */
    [[nodiscard]] std::exception_ptr completed(
            DeviceOps& ops, oid token, std::string_view role) noexcept {
        if (oid_is_error(token)) {
            return oid_failure(token, role);
        }
        try {
            ops.wait(token);
        } catch (...) {
            return std::current_exception();
        }
        return nullptr;
    }

}  // namespace

void run_cache_attention_stage(
        DeviceOps& ops, CacheAttentionStageRequest& request,
        RawWorkspaceView workspace, CacheAttentionStageState& state) {
    const CheckedStage stage = validate_stage(ops, request, state);
    state.residual_output = 0;

    // Every downstream operation is preflighted with its actual operands
    // and scratch before the first submission. SDPA, the output projection,
    // and the first residual are three strictly sequential consumers of one
    // caller-owned range, so that range must satisfy the largest of their
    // queried requirements: a backend whose projection or residual needs
    // positive staging rejects an insufficient range here, before any cache
    // row is written or a prefix is published.
    const WorkspaceRequirements sdpa_requirement =
            ops.sdpa_workspace_requirements(
                    request.rotated_q, request.k_cache, request.v_cache,
                    request.attention_merged, request.a,
                    stage.initialized_length);
    const WorkspaceRequirements projection_requirement =
            ops.linear_workspace_requirements(
                    request.attention_merged, request.o_weight,
                    request.attention_output, 0, request.R,
                    LinearOutputLayout::ordinary, 1, stage.merged_width);
    const WorkspaceRequirements residual_requirement =
            ops.add_workspace_requirements(
                    request.residual_input, request.attention_output,
                    request.residual_output);
    const WorkspaceRequirements combined_requirement{
            std::max({sdpa_requirement.bytes, projection_requirement.bytes,
                      residual_requirement.bytes}),
            std::max({sdpa_requirement.alignment,
                      projection_requirement.alignment,
                      residual_requirement.alignment})};

    // The shared validator compares the caller range against one borrowed
    // operand at a time: its checks are pairwise, and copying a borrowed
    // `TensorView` would allocate its plane-stride vector. Every operand
    // read or written by those three operations is checked, so no later
    // rejection can leave accepted append work behind.
    const auto validate_against_operand = [&](const TensorView& operand) {
        const TensorView* borrowed = &operand;
        (void)detail::WorkspaceValidation::validated(
                ops.device(), workspace, combined_requirement.bytes,
                combined_requirement.alignment,
                std::span<const TensorView>(borrowed, 1));
    };
    validate_against_operand(request.rotated_q);
    validate_against_operand(request.k_cache);
    validate_against_operand(request.v_cache);
    validate_against_operand(request.attention_merged);
    validate_against_operand(request.o_weight);
    validate_against_operand(request.attention_output);
    validate_against_operand(request.residual_input);
    validate_against_operand(request.residual_output);

    // An operation whose queried requirement is zero consumes no scratch
    // and accepts only the empty default; each positive requirement gets the
    // validated reusable range, which stays live until its own wait.
    const RawWorkspaceView sdpa_workspace =
            sdpa_requirement.bytes == 0 ? RawWorkspaceView{} : workspace;
    const RawWorkspaceView projection_workspace =
            projection_requirement.bytes == 0 ? RawWorkspaceView{}
                                              : workspace;
    const RawWorkspaceView residual_workspace =
            residual_requirement.bytes == 0 ? RawWorkspaceView{}
                                            : workspace;

    // K and V have distinct cache owners and distinct OIDs. Both
    // submissions are attempted, and both accepted OIDs are waited, even
    // when the first result already failed.
    std::array<oid, 2> appends{0, 0};
    appends[0] =
            ops.cache_append(request.rotated_k, request.k_cache, request.a);
    appends[1] =
            ops.cache_append(request.rotated_v, request.v_cache, request.a);

    std::exception_ptr failure;
    for (std::size_t index = 0; index < appends.size(); ++index) {
        const std::string_view role =
                index == 0 ? "K append" : "V append";
        if (oid_is_error(appends[index])) {
            if (failure == nullptr) {
                failure = oid_failure(appends[index], role);
            }
            continue;
        }
        if (std::exception_ptr waited =
                    completed(ops, appends[index], role);
                waited != nullptr && failure == nullptr) {
            failure = std::move(waited);
        }
    }
    if (failure != nullptr) {
        // A possibly written physical cache row is never rolled back,
        // retried, or reused, and no new prefix is published.
        state.failed = true;
        std::rethrow_exception(failure);
    }

    // Both independent append barriers succeeded: publish exactly the
    // checked prefix and read no cache row beyond it.
    state.initialized_length = stage.initialized_length;

    failure = completed(
            ops,
            ops.sdpa(
                    request.rotated_q, request.k_cache, request.v_cache,
                    request.attention_merged, request.a,
                    stage.initialized_length, sdpa_workspace),
            "SDPA");
    if (failure == nullptr) {
        failure = completed(
                ops,
                ops.linear(
                        request.attention_merged, request.o_weight,
                        request.attention_output, 0, request.R,
                        LinearOutputLayout::ordinary, 1,
                        stage.merged_width, projection_workspace),
                "output projection");
    }
    if (failure == nullptr) {
        const oid residual = ops.add(
                request.residual_input, request.attention_output,
                request.residual_output, residual_workspace);
        failure = completed(ops, residual, "first residual");
        if (failure == nullptr) {
            // The composition uses the accepted, already-completed token as
            // the direct producer barrier for post-attention RMSNorm.
            state.residual_output = residual;
        }
    }
    if (failure != nullptr) {
        state.failed = true;
        std::rethrow_exception(failure);
    }
}

// ---------------------------------------------------------------------------
// Attention normalization, QKV projection, and Q/K RoPE stage
// (leaf 03-qkv-rope-stage).
//
// One private, allocation-free stage composes the existing RMSNorm, three
// head-planar linear, and two split-half RoPE facades over caller-supplied
// views and caller-provisioned scratch. It adds no public stage API, hidden
// owner, allocation, cache mutation, or schedule of its own: the producer
// order below is the frozen forward schedule, and every dependent
// submission happens only after its direct producer was waited
// successfully.
// ---------------------------------------------------------------------------

namespace {

// One diagnostic tag for every rejection this composition makes itself.
// Per-operation admission keeps its own operation tags.
constexpr const char* kQkvRopeStage = "TinyLlama QKV RoPE stage";

[[noreturn]] void qkv_rope_reject(const char* what) {
    throw std::invalid_argument(std::string(kQkvRopeStage) + ": " + what);
}

// Exact rank and extent agreement of one operand: a view with the same
// element count in another shape is not the same operand.
void qkv_rope_require_extents(const TensorView& view,
        std::span<const std::size_t> expected, const char* what) {
    const std::span<const std::size_t> actual =
            view.spec().shape.dimensions();
    if (actual.size() != expected.size()) qkv_rope_reject(what);
    for (std::size_t axis = 0; axis < expected.size(); ++axis) {
        if (actual[axis] != expected[axis]) qkv_rope_reject(what);
    }
}

// Head geometry of one invocation, read from the supplied views. The stage
// never assumes a configured head layout: storage decides the extents, and
// every relation below is checked against them.
struct QkvRopeGeometry {
    std::size_t rows = 0;
    std::size_t features = 0;
    std::size_t query_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t head_dim = 0;

    [[nodiscard]] std::size_t key_value_features() const {
        return detail::checked_mul(kv_heads, head_dim,
                "TinyLlama QKV RoPE stage Hkv*D overflows");
    }
};

QkvRopeGeometry prepare_qkv_rope_stage(const QkvRopeStageViews& views,
        const QkvRopeStageParams& params) {
    if (params.rows == 0) {
        qkv_rope_reject("row count must be nonzero");
    }

    const std::span<const std::size_t> activation =
            views.activation.spec().shape.dimensions();
    if (activation.size() != 2) {
        qkv_rope_reject("activation must be the rank-two [R,F] matrix");
    }
    QkvRopeGeometry geometry;
    geometry.rows = activation[0];
    geometry.features = activation[1];
    if (geometry.rows == 0 || geometry.features == 0) {
        qkv_rope_reject("activation extents must be nonzero");
    }
    if (geometry.rows != params.rows) {
        qkv_rope_reject("activation rows must equal the supplied row count");
    }

    const std::span<const std::size_t> query =
            views.query.spec().shape.dimensions();
    if (query.size() != 3) {
        qkv_rope_reject(
                "query projection must be the rank-three [Hq,R,D] tensor");
    }
    geometry.query_heads = query[0];
    geometry.head_dim = query[2];
    if (geometry.query_heads == 0 || geometry.head_dim == 0) {
        qkv_rope_reject("query head count and head width must be nonzero");
    }
    if ((geometry.head_dim & 1U) != 0) {
        qkv_rope_reject("head width must be positive and even");
    }
    if (query[1] != geometry.rows) {
        qkv_rope_reject("query rows must equal the stage row count");
    }

    const std::span<const std::size_t> key =
            views.key.spec().shape.dimensions();
    if (key.size() != 3) {
        qkv_rope_reject(
                "key projection must be the rank-three [Hkv,R,D] tensor");
    }
    geometry.kv_heads = key[0];
    if (geometry.kv_heads == 0) {
        qkv_rope_reject("key head count must be nonzero");
    }
    if (key[1] != geometry.rows || key[2] != geometry.head_dim) {
        qkv_rope_reject("key must be [Hkv,R,D] with the query head width");
    }

    const std::size_t expected_features = detail::checked_mul(
            geometry.query_heads, geometry.head_dim,
            "TinyLlama QKV RoPE stage Hq*D overflows");
    if (expected_features != geometry.features) {
        qkv_rope_reject("activation features must equal Hq*D");
    }
    if (geometry.query_heads % geometry.kv_heads != 0) {
        qkv_rope_reject(
                "query head count must be divisible by the KV head count");
    }

    const std::array<std::size_t, 2> activation_shape{
            geometry.rows, geometry.features};
    qkv_rope_require_extents(views.normalized, activation_shape,
            "normalized output must be [R,F]");
    const std::array<std::size_t, 2> scale_shape{1, geometry.features};
    qkv_rope_require_extents(views.attention_scale, scale_shape,
            "attention scale must be [1,F]");
    const std::array<std::size_t, 2> query_weight_shape{
            geometry.features, geometry.features};
    qkv_rope_require_extents(views.query_weight, query_weight_shape,
            "query weight must be [Hq*D,F]");
    const std::array<std::size_t, 2> kv_weight_shape{
            geometry.key_value_features(), geometry.features};
    qkv_rope_require_extents(views.key_weight, kv_weight_shape,
            "key weight must be [Hkv*D,F]");
    qkv_rope_require_extents(views.value_weight, kv_weight_shape,
            "value weight must be [Hkv*D,F]");
    const std::array<std::size_t, 3> query_shape{
            geometry.query_heads, geometry.rows, geometry.head_dim};
    qkv_rope_require_extents(views.rotated_query, query_shape,
            "rotated query must be [Hq,R,D]");
    const std::array<std::size_t, 3> kv_shape{
            geometry.kv_heads, geometry.rows, geometry.head_dim};
    qkv_rope_require_extents(views.value, kv_shape, "value must be [Hkv,R,D]");
    qkv_rope_require_extents(views.rotated_key, kv_shape,
            "rotated key must be [Hkv,R,D]");

    // The stage owns an exclusive end-position check in addition to the
    // operation's inclusive last-position admission.  This keeps `a + R`
    // representable for the caller's subsequent cache/window handoff even
    // when the RoPE operation itself would only need `a + R - 1`.
    (void)detail::checked_add(params.a, geometry.rows,
            "TinyLlama QKV RoPE stage absolute position range overflows");
    return geometry;
}

// Every operand of the stage belongs to the queue's device. The operation
// facades own the remaining per-operation admission: owners, handles,
// strides, alias, dtype, quantization, capability, and the rowwise and
// row-window rules.
void require_qkv_rope_device(
        const QkvRopeStageViews& views, const Device& device) {
    const std::array<const TensorView*, 11> operands{
            &views.activation, &views.attention_scale, &views.query_weight,
            &views.key_weight, &views.value_weight, &views.normalized,
            &views.query, &views.key, &views.value, &views.rotated_query,
            &views.rotated_key};
    for (const TensorView* operand : operands) {
        if (&operand->device() != &device) {
            qkv_rope_reject("stage operands must belong to the queue's "
                            "device");
        }
    }
}

// The established category of one synchronous OID rejection. The facades map
// exactly these classes, so a rejected dependent submission keeps the
// operation's own failure category instead of being relabelled by this
// composition. A zero or unknown result is the queue contract's invalid
// value and stays an internal failure.
[[nodiscard]] std::exception_ptr qkv_rope_admission_failure(
        oid value, const char* what) {
    const std::string context =
            std::string(kQkvRopeStage) + ": " + what + " was rejected";
    switch (value) {
        case to_oid(OidError::InvalidArgument):
            return std::make_exception_ptr(std::invalid_argument(
                    context + " as invalid input"));
        case to_oid(OidError::Unsupported):
            return std::make_exception_ptr(detail::UnsupportedOperation());
        case to_oid(OidError::Overflow):
            return std::make_exception_ptr(std::overflow_error(
                    context + " by checked arithmetic"));
        case to_oid(OidError::ResourceExhausted):
            return std::make_exception_ptr(std::bad_alloc());
        case to_oid(OidError::DeviceError):
        case to_oid(OidError::InternalError):
        default:
            return std::make_exception_ptr(std::runtime_error(
                    context + " by the backend"));
    }
}

// Observe one branch: a positive token is waited (and a retained failure is
// captured), while a negative result consumed no sequence and is never
// passed to `wait`. The returned failure is null exactly when the branch
// completed successfully.
[[nodiscard]] std::exception_ptr qkv_rope_branch_failure(
        DeviceOps& queue, oid value, const char* what) {
    if (!oid_is_token(value)) {
        return qkv_rope_admission_failure(value, what);
    }
    try {
        queue.wait(value);
    } catch (...) {
        return std::current_exception();
    }
    return nullptr;
}

// Attempt the wait of every branch in one simultaneously submitted group,
// even after an earlier branch already failed, and report the first failure
// only once every accepted OID of the group is terminal. A completed failure
// stays observable through its own token; a later successful branch never
// substitutes for it.
[[nodiscard]] std::exception_ptr drain_qkv_rope_branches(DeviceOps& queue,
        std::span<const oid> tokens, std::span<const char* const> roles) {
    std::exception_ptr first;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::exception_ptr failure =
                qkv_rope_branch_failure(queue, tokens[index], roles[index]);
        if (failure != nullptr && first == nullptr) {
            first = failure;
        }
    }
    return first;
}

// Caller scratch is admitted before the first submission. Each positive
// branch requirement is checked through the established workspace validator,
// which owns device identity, liveness, exact capacity, required alignment,
// and range arithmetic for the exact admitted range; the per-branch operand
// nonoverlap check stays with the submission facade, which receives the
// caller's own views and is authoritative for it. Two simultaneously
// submitted positive requirements must not share one caller range. A zero
// requirement neither validates nor leases a slice, exactly as the operation
// facades document, so the three branches are never serialized to reuse an
// overlapping caller range. The preflight copies no view and allocates
// nothing.
void admit_qkv_rope_scratch(DeviceOps& queue,
        const QkvRopeStageWorkspace& workspace,
        const std::array<WorkspaceRequirements, 3>& requirements) {
    const Device& device = queue.device();
    const std::array<const RawWorkspaceView*, 3> slices{
            &workspace.query, &workspace.key, &workspace.value};
    for (std::size_t branch = 0; branch < slices.size(); ++branch) {
        if (requirements[branch].bytes == 0) {
            continue;
        }
        (void)detail::WorkspaceValidation::validated(device,
                *slices[branch], requirements[branch].bytes,
                requirements[branch].alignment,
                std::span<const TensorView>{});
        for (std::size_t other = branch + 1; other < slices.size(); ++other) {
            if (requirements[other].bytes == 0) {
                continue;
            }
            const RawWorkspaceView& first = *slices[branch];
            const RawWorkspaceView& second = *slices[other];
            if (first.owner_identity() == second.owner_identity()
                    && first.range_begin() < second.range_end()
                    && second.range_begin() < first.range_end()) {
                qkv_rope_reject("simultaneously submitted projection "
                                "branches must not share caller scratch");
            }
        }
    }
}

}  // namespace

void run_qkv_rope_stage(DeviceOps& queue, QkvRopeStageViews views,
        const QkvRopeStageParams& params,
        const QkvRopeStageWorkspace& workspace) {
    const QkvRopeGeometry geometry = prepare_qkv_rope_stage(views, params);
    require_qkv_rope_device(views, queue.device());

    // Every operation query is pure and performs its own complete admission
    // validation.  Run all of them before RMSNorm can submit, so a malformed
    // RoPE scalar/view or a capability rejection cannot leave earlier stage
    // outputs published.
    const WorkspaceRequirements normalization_requirements =
            queue.rmsnorm_workspace_requirements(views.activation,
                    views.attention_scale, views.normalized, params.epsilon);
    if (normalization_requirements.bytes != 0) {
        qkv_rope_reject(
                "RMSNorm requires caller scratch unsupported by this stage");
    }
    const std::array<WorkspaceRequirements, 2> rotation_requirements{
            queue.rope_workspace_requirements(views.query,
                    views.rotated_query, params.a, params.theta),
            queue.rope_workspace_requirements(views.key, views.rotated_key,
                    params.a, params.theta)};
    for (const WorkspaceRequirements requirements : rotation_requirements) {
        if (requirements.bytes != 0) {
            qkv_rope_reject(
                    "RoPE requires caller scratch unsupported by this stage");
        }
    }

    // Pure requirement queries and caller-scratch admission precede every
    // queue effect. RMSNorm and RoPE consume no raw workspace on any retained
    // backend, so only the three projection branches carry a requirement
    // here.
    const std::array<WorkspaceRequirements, 3> requirements{
            queue.linear_workspace_requirements(views.normalized,
                    views.query_weight, views.query, 0, geometry.rows,
                    LinearOutputLayout::head_planar, geometry.query_heads,
                    geometry.head_dim),
            queue.linear_workspace_requirements(views.normalized,
                    views.key_weight, views.key, 0, geometry.rows,
                    LinearOutputLayout::head_planar, geometry.kv_heads,
                    geometry.head_dim),
            queue.linear_workspace_requirements(views.normalized,
                    views.value_weight, views.value, 0, geometry.rows,
                    LinearOutputLayout::head_planar, geometry.kv_heads,
                    geometry.head_dim)};
    admit_qkv_rope_scratch(queue, workspace, requirements);

    // 1. Attention normalization is the only producer of every projection
    //    input, so it is submitted and waited on its own.
    const oid normalization = queue.rmsnorm(views.activation,
            views.attention_scale, views.normalized, params.epsilon);
    if (const std::exception_ptr failure = qkv_rope_branch_failure(
                queue, normalization, "attention RMSNorm")) {
        std::rethrow_exception(failure);
    }

    // 2. Q, K, and V are independent branches over the completed
    //    normalization: HF `[out,in]` weights are consumed directly and
    //    head-planar mode writes [Hq,R,D] / [Hkv,R,D] without a transpose, a
    //    repeated KV head, or a final-axis view transform.
    const std::array<oid, 3> projections{
            queue.linear(views.normalized, views.query_weight, views.query, 0,
                    geometry.rows, LinearOutputLayout::head_planar,
                    geometry.query_heads, geometry.head_dim, workspace.query),
            queue.linear(views.normalized, views.key_weight, views.key, 0,
                    geometry.rows, LinearOutputLayout::head_planar,
                    geometry.kv_heads, geometry.head_dim, workspace.key),
            queue.linear(views.normalized, views.value_weight, views.value, 0,
                    geometry.rows, LinearOutputLayout::head_planar,
                    geometry.kv_heads, geometry.head_dim, workspace.value)};
    constexpr std::array<const char*, 3> kProjectionRoles{
            "query projection", "key projection", "value projection"};
    // 3. Every projection branch is drained before either rotation: a
    //    positive OID proves admission only, and a failed or rejected branch
    //    forbids the dependent rotations entirely.
    if (const std::exception_ptr failure =
                drain_qkv_rope_branches(queue, projections,
                        kProjectionRoles)) {
        std::rethrow_exception(failure);
    }

    // 4. Rotated Q and K are independent branches at the same absolute start
    //    with the same runtime theta. V stays unrotated.
    const std::array<oid, 2> rotations{
            queue.rope(views.query, views.rotated_query, params.a,
                    params.theta),
            queue.rope(views.key, views.rotated_key, params.a, params.theta)};
    constexpr std::array<const char*, 2> kRotationRoles{
            "query rotation", "key rotation"};
    if (const std::exception_ptr failure = drain_qkv_rope_branches(
                queue, rotations, kRotationRoles)) {
        std::rethrow_exception(failure);
    }
    // A normal return is the only publication point: every accepted OID of
    // this stage is terminal, and no owner, allocator, cache, or session
    // state was touched.
}
// ---------------------------------------------------------------------------
// Exactly-one-layer decoder composition (leaf 06-decoder-layer-forward).
//
// The three stage helpers above remain the numerical and failure boundaries.
// This layer only validates the cross-stage geometry/ownership seam, waits an
// optional input producer group, and calls the helpers in the fixed
// QKV/RoPE -> cache/attention -> MLP order.  No owner, view, vector, tensor,
// or workspace is created here.
// ---------------------------------------------------------------------------

namespace {

constexpr const char* kDecoderLayerStage =
        "TinyLlama decoder layer forward";

[[noreturn]] void decoder_reject(const char* what) {
    throw std::invalid_argument(
            std::string(kDecoderLayerStage) + ": " + what);
}

void decoder_require_shape(
        const TensorView& view, std::initializer_list<std::size_t> expected,
        const char* role) {
    const std::span<const std::size_t> actual =
            view.spec().shape.dimensions();
    if (actual.size() != expected.size()) decoder_reject(role);
    std::size_t axis = 0;
    for (const std::size_t extent : expected) {
        if (actual[axis++] != extent) decoder_reject(role);
    }
}

void decoder_require_bf16(
        const Device& device, const TensorView& view, const char* role) {
    if (&view.device() != &device) {
        decoder_reject("all stage operands must belong to one queue device");
    }
    if (view.spec().data_type != DataType::BF16
            || view.spec().quantization != QuantizationFormat::NONE) {
        decoder_reject(role);
    }
}

void decoder_require_distinct_stores(
        const DecoderLayerForwardViews& views) {
    const TensorView* const stores[] = {
            &views.qkv.normalized,
            &views.qkv.query,
            &views.qkv.key,
            &views.qkv.value,
            &views.qkv.rotated_query,
            &views.qkv.rotated_key,
            &views.attention.k_cache,
            &views.attention.v_cache,
            &views.attention.attention_merged,
            &views.attention.attention_output,
            &views.attention.residual_output,
            &views.mlp.n2,
            &views.mlp.gate,
            &views.mlp.up,
            &views.mlp.activated_gate,
            &views.mlp.product,
            &views.mlp.down,
            &views.mlp.next_x};
    const TensorView* const operands[] = {
            &views.qkv.activation,
            &views.qkv.attention_scale,
            &views.qkv.query_weight,
            &views.qkv.key_weight,
            &views.qkv.value_weight,
            &views.qkv.normalized,
            &views.qkv.query,
            &views.qkv.key,
            &views.qkv.value,
            &views.qkv.rotated_query,
            &views.qkv.rotated_key,
            &views.attention.rotated_q,
            &views.attention.rotated_k,
            &views.attention.rotated_v,
            &views.attention.k_cache,
            &views.attention.v_cache,
            &views.attention.residual_input,
            &views.attention.o_weight,
            &views.attention.attention_merged,
            &views.attention.attention_output,
            &views.attention.residual_output,
            &views.mlp.x2,
            &views.mlp.post_attention_scale,
            &views.mlp.gate_weight,
            &views.mlp.up_weight,
            &views.mlp.down_weight,
            &views.mlp.n2,
            &views.mlp.gate,
            &views.mlp.up,
            &views.mlp.activated_gate,
            &views.mlp.product,
            &views.mlp.down,
            &views.mlp.next_x};
    for (const TensorView* store : stores) {
        for (const TensorView* operand : operands) {
            if (store == operand) continue;
            // Q/K/V and the first residual intentionally cross stage seams;
            // those are the only producer-to-consumer aliases allowed here.
            if ((store == &views.qkv.rotated_query
                        && operand == &views.attention.rotated_q)
                    || (store == &views.qkv.rotated_key
                        && operand == &views.attention.rotated_k)
                    || (store == &views.qkv.value
                        && operand == &views.attention.rotated_v)
                    || (store == &views.attention.residual_output
                        && operand == &views.mlp.x2)) {
                continue;
            }
            if (store->owner_identity() == operand->owner_identity()) {
                decoder_reject(
                        "decoder layer output stores must be disjoint from "
                        "other operands");
            }
        }
    }
}

void decoder_validate(
        const DeviceOps& operations,
        const DecoderLayerForwardViews& views,
        const DecoderLayerForwardParams& params,
        const DecoderLayerForwardState& state) {
    if (state.failed) {
        throw std::logic_error(
                "TinyLlama decoder layer forward refuses poisoned state");
    }
    if (params.rows == 0 || params.capacity == 0 || params.features == 0
            || params.intermediate == 0) {
        decoder_reject("rows, capacity, features, and intermediate must be "
                       "nonzero");
    }
    if (!std::isfinite(params.theta) || params.theta <= 0.0) {
        decoder_reject("RoPE theta must be finite and positive");
    }
    if (!std::isfinite(params.attention_epsilon)
            || params.attention_epsilon < 0.0F
            || !std::isfinite(params.mlp_epsilon)
            || params.mlp_epsilon < 0.0F) {
        decoder_reject("RMSNorm epsilons must be finite and nonnegative");
    }
    if (params.a > params.capacity) {
        decoder_reject("absolute offset exceeds cache capacity");
    }
    const std::size_t initialized_length = detail::checked_add(
            params.a, params.rows,
            "TinyLlama decoder layer initialized length overflows");
    if (initialized_length > params.capacity) {
        decoder_reject("run rows exceed cache capacity");
    }
    if (state.initialized_length != params.a) {
        decoder_reject("cache prefix does not continue at the supplied offset");
    }
    const CacheAttentionStageRequest& attention = views.attention;
    if (attention.a != params.a || attention.R != params.rows
            || attention.C != params.capacity) {
        decoder_reject("stage positions and cache capacity disagree");
    }
    const QkvRopeStageViews& qkv = views.qkv;
    const std::span<const std::size_t> query_shape =
            qkv.query.spec().shape.dimensions();
    const std::span<const std::size_t> key_shape =
            qkv.key.spec().shape.dimensions();
    if (query_shape.size() != 3 || key_shape.size() != 3) {
        decoder_reject("Q and K must be rank-three head-planar views");
    }
    const std::size_t query_heads = query_shape[0];
    const std::size_t rows = query_shape[1];
    const std::size_t head_dim = query_shape[2];
    const std::size_t kv_heads = key_shape[0];
    if (query_heads == 0 || kv_heads == 0 || head_dim == 0
            || (head_dim & 1U) != 0 || rows != params.rows
            || key_shape[1] != params.rows || key_shape[2] != head_dim
            || query_heads % kv_heads != 0) {
        decoder_reject("Q/K head geometry is inconsistent");
    }
    const std::size_t merged_width = detail::checked_mul(
            query_heads, head_dim,
            "TinyLlama decoder layer feature width overflows");
    if (merged_width != params.features) {
        decoder_reject("configured feature width must equal Hq*D");
    }
    const std::size_t kv_width = detail::checked_mul(
            kv_heads, head_dim,
            "TinyLlama decoder layer KV width overflows");
    const Device& device = operations.device();
    const TensorView* const operands[] = {
            &qkv.activation, &qkv.attention_scale, &qkv.query_weight,
            &qkv.key_weight, &qkv.value_weight, &qkv.normalized, &qkv.query,
            &qkv.key, &qkv.value, &qkv.rotated_query, &qkv.rotated_key,
            &attention.rotated_q, &attention.rotated_k, &attention.rotated_v,
            &attention.k_cache, &attention.v_cache,
            &attention.residual_input, &attention.o_weight,
            &attention.attention_merged, &attention.attention_output,
            &attention.residual_output, &views.mlp.x2,
            &views.mlp.post_attention_scale, &views.mlp.gate_weight,
            &views.mlp.up_weight, &views.mlp.down_weight, &views.mlp.n2,
            &views.mlp.gate, &views.mlp.up, &views.mlp.activated_gate,
            &views.mlp.product, &views.mlp.down, &views.mlp.next_x};
    for (const TensorView* operand : operands) {
        decoder_require_bf16(device, *operand, "stage operands must be BF16");
    }

    decoder_require_shape(qkv.activation, {params.rows, params.features},
            "activation must be [R,F]");
    decoder_require_shape(qkv.attention_scale, {1, params.features},
            "attention scale must be [1,F]");
    decoder_require_shape(qkv.normalized, {params.rows, params.features},
            "normalized store must be [R,F]");
    decoder_require_shape(qkv.query_weight, {params.features, params.features},
            "query weight must be [F,F]");
    decoder_require_shape(qkv.key_weight, {kv_width, params.features},
            "key weight must be [Hkv*D,F]");
    decoder_require_shape(qkv.value_weight, {kv_width, params.features},
            "value weight must be [Hkv*D,F]");
    decoder_require_shape(qkv.query, {query_heads, params.rows, head_dim},
            "query must be [Hq,R,D]");
    decoder_require_shape(qkv.key, {kv_heads, params.rows, head_dim},
            "key must be [Hkv,R,D]");
    decoder_require_shape(qkv.value, {kv_heads, params.rows, head_dim},
            "value must be [Hkv,R,D]");
    decoder_require_shape(qkv.rotated_query,
            {query_heads, params.rows, head_dim},
            "rotated query must be [Hq,R,D]");
    decoder_require_shape(qkv.rotated_key, {kv_heads, params.rows, head_dim},
            "rotated key must be [Hkv,R,D]");

    decoder_require_shape(attention.rotated_q,
            {query_heads, params.rows, head_dim},
            "attention query must match QKV output");
    decoder_require_shape(attention.rotated_k,
            {kv_heads, params.rows, head_dim},
            "attention key must match QKV output");
    decoder_require_shape(attention.rotated_v,
            {kv_heads, params.rows, head_dim},
            "attention value must match QKV output");
    decoder_require_shape(attention.k_cache, {kv_heads, params.capacity,
                                               head_dim},
            "K cache must be [Hkv,C,D]");
    decoder_require_shape(attention.v_cache, {kv_heads, params.capacity,
                                               head_dim},
            "V cache must be [Hkv,C,D]");
    decoder_require_shape(attention.residual_input,
            {params.rows, params.features}, "residual input must be [R,F]");
    decoder_require_shape(attention.o_weight,
            {params.features, params.features}, "output weight must be [F,F]");
    decoder_require_shape(attention.attention_merged,
            {params.rows, params.features},
            "merged attention must be [R,F]");
    decoder_require_shape(attention.attention_output,
            {params.rows, params.features},
            "attention output must be [R,F]");
    decoder_require_shape(attention.residual_output,
            {params.rows, params.features},
            "first residual must be [R,F]");

    const MlpStageViews& mlp = views.mlp;
    decoder_require_shape(mlp.x2, {params.rows, params.features},
            "MLP input must be [R,F]");
    decoder_require_shape(mlp.post_attention_scale, {1, params.features},
            "post-attention scale must be [1,F]");
    decoder_require_shape(mlp.n2, {params.rows, params.features},
            "MLP norm store must be [R,F]");
    decoder_require_shape(mlp.gate_weight,
            {params.intermediate, params.features},
            "gate weight must be [M,F]");
    decoder_require_shape(mlp.up_weight,
            {params.intermediate, params.features},
            "up weight must be [M,F]");
    decoder_require_shape(mlp.down_weight,
            {params.features, params.intermediate},
            "down weight must be [F,M]");
    decoder_require_shape(mlp.gate, {params.rows, params.intermediate},
            "gate store must be [R,M]");
    decoder_require_shape(mlp.up, {params.rows, params.intermediate},
            "up store must be [R,M]");
    decoder_require_shape(mlp.activated_gate,
            {params.rows, params.intermediate},
            "activated gate must be [R,M]");
    decoder_require_shape(mlp.product, {params.rows, params.intermediate},
            "product store must be [R,M]");
    decoder_require_shape(mlp.down, {params.rows, params.features},
            "down store must be [R,F]");
    decoder_require_shape(mlp.next_x, {params.rows, params.features},
            "next residual must be [R,F]");

    if (qkv.activation.owner_identity()
                    != attention.residual_input.owner_identity()
            || qkv.rotated_query.owner_identity()
                    != attention.rotated_q.owner_identity()
            || qkv.rotated_key.owner_identity()
                    != attention.rotated_k.owner_identity()
            || qkv.value.owner_identity()
                    != attention.rotated_v.owner_identity()
            || attention.residual_output.owner_identity()
                    != mlp.x2.owner_identity()) {
        decoder_reject("stage boundaries must share the original X, Q/K/V, "
                       "and X2 owners");
    }
    decoder_require_distinct_stores(views);
}

[[nodiscard]] std::exception_ptr decoder_wait_readiness(
        DeviceOps& operations, std::span<const oid> readiness) {
    std::exception_ptr first;
    for (const oid token : readiness) {
        try {
            operations.wait(token);
        } catch (...) {
            if (first == nullptr) first = std::current_exception();
        }
    }
    return first;
}

}  // namespace

void run_decoder_layer_forward(
        DeviceOps& operations, DecoderLayerForwardViews& views,
        const DecoderLayerForwardParams& params,
        const DecoderLayerForwardWorkspace& workspace,
        std::span<const oid> readiness, DecoderLayerForwardState& state) {
    decoder_validate(operations, views, params, state);
    for (const oid token : readiness) {
        if (!oid_is_token(token)) {
            decoder_reject(
                    "readiness must contain accepted producer tokens");
        }
    }

    try {
        if (const std::exception_ptr failure =
                    decoder_wait_readiness(operations, readiness)) {
            std::rethrow_exception(failure);
        }

        run_qkv_rope_stage(
                operations, views.qkv,
                QkvRopeStageParams{params.a, params.rows, params.theta,
                                   params.attention_epsilon},
                workspace.qkv);

        CacheAttentionStageState attention_state{
                state.initialized_length, false, 0};
        run_cache_attention_stage(
                operations, views.attention, workspace.attention,
                attention_state);
        state.initialized_length = attention_state.initialized_length;
        if (!oid_is_token(attention_state.residual_output)) {
            throw std::logic_error(
                    "TinyLlama decoder layer forward lost first-residual "
                    "producer");
        }

        const std::span<const oid> residual_ready(
                &attention_state.residual_output, 1);
        MlpStageFailure mlp_failure;
        run_mlp_stage(
                operations, views.mlp,
                MlpStageParams{params.rows, params.features,
                               params.intermediate, params.mlp_epsilon},
                workspace.mlp, residual_ready, mlp_failure);
    } catch (...) {
        state.failed = true;
        throw;
    }
}

namespace {

void encode_token_ids(
        std::span<const std::size_t> source, std::size_t vocabulary,
        std::span<std::uint32_t> destination) {
    if (source.empty() || source.size() != destination.size()) {
        throw std::invalid_argument(
                "TinyLlama forward token sequence has invalid length");
    }
    static_cast<void>(checked_mul(
            source.size(), sizeof(std::uint32_t),
            "TinyLlama forward token upload bytes overflow"));
    for (std::size_t index = 0; index < source.size(); ++index) {
        if (source[index] >= vocabulary) {
            throw std::invalid_argument(
                    "TinyLlama forward token index exceeds vocabulary");
        }
        destination[index] = static_cast<std::uint32_t>(source[index]);
    }
}

[[nodiscard]] RawWorkspaceView forward_workspace_slice(
        const RequestState& request, std::size_t offset,
        WorkspaceRequirements requirement) {
    if (requirement.bytes == 0) return RawWorkspaceView{};
    if (!request.operation_workspace) {
        throw std::logic_error(
                "TinyLlama forward workspace owner is missing");
    }
    return request.operation_workspace->view().subrange(
            offset, requirement.bytes);
}

[[nodiscard]] DecoderLayerForwardWorkspace forward_layer_workspace(
        const Device& device, const RequestState& request,
        const ForwardPlan& plan) {
    const ForwardWorkspaceLayout& layout = plan.workspace;
    const QkvRopeStageWorkspace qkv{
            forward_workspace_slice(
                    request, layout.qkv_offsets[0],
                    layout.qkv_requirements[0]),
            forward_workspace_slice(
                    request, layout.qkv_offsets[1],
                    layout.qkv_requirements[1]),
            forward_workspace_slice(
                    request, layout.qkv_offsets[2],
                    layout.qkv_requirements[2])};
    const RawWorkspaceView attention = forward_workspace_slice(
            request, layout.attention_offset,
            layout.attention_requirement);
    const WorkspaceRequirements mlp_total =
            layout.mlp_requirements.total();
    const RawWorkspaceView mlp_range = forward_workspace_slice(
            request, layout.mlp_offset, mlp_total);
    const MlpWorkspace mlp = resolve_mlp_workspace(
            device, layout.mlp_requirements, mlp_range);
    return DecoderLayerForwardWorkspace{qkv, attention, mlp};
}


}  // namespace

void SessionAccess::prepare_forward_request(
        TinyLlamaSession& session, std::size_t run_length) {
    TinyLlamaSession::Impl& impl = *session.impl_;
    if (run_length == 0) {
        throw std::invalid_argument(
                "TinyLlama forward request length must be nonzero");
    }
    if (run_length > impl.config().max_position_embeddings) {
        throw std::invalid_argument(
                "TinyLlama forward request exceeds model context");
    }
    if (impl.poisoned) {
        throw std::logic_error("TinyLlama session is poisoned");
    }
    validate_vector_capacity<std::uint32_t>(run_length);
    validate_vector_capacity<std::size_t>(
            impl.config().max_position_embeddings);
    const TokenSelectorScratchRequirements selector_requirements =
            impl.selector->scratch_requirements(
                    impl.logits->view(), impl.config().vocab_size);
    validate_workspace_requirement(
            selector_requirements.device, "selector device");
    validate_vector_capacity<std::byte>(selector_requirements.host_bytes);

    // Drain the previous request before any replacement owner or query is
    // published.  A failed drain leaves the old request and poison state
    // intact, exactly as the public resource setup boundary promises.
    impl.drain_request();

    auto candidate = std::make_unique<RequestState>();
    candidate->run_length = run_length;
    candidate->banks = allocate_run_banks(
            *impl.device, impl.config(), run_length);
    candidate->prefill_indices.resize(run_length);
    candidate->history.reserve(impl.config().max_position_embeddings);
    candidate->results.reserve(impl.config().max_position_embeddings);
    const std::size_t oid_capacity = accepted_oid_capacity(
            impl.config(), "TinyLlama forward accepted-OID capacity overflows");
    candidate->accepted_oids.reserve(oid_capacity);

    ForwardPlan plan = make_forward_plan(
            *impl.model, candidate->banks, impl.fixed, impl.logits->view(),
            impl.caches, *impl.queue, run_length);
    candidate->forward =
            std::make_unique<ForwardPlan>(std::move(plan));

    const WorkspaceRequirements operation =
            candidate->forward->workspace.total;
    validate_workspace_requirement(operation, "operation");
    if (operation.bytes != 0) {
        candidate->operation_workspace =
                impl.device->create_workspace(operation.bytes);
        validate_workspace_owner(
                candidate->operation_workspace.get(), *impl.device, operation,
                "operation");
    }
    candidate->selector_host_scratch.resize(
            selector_requirements.host_bytes);
    if (selector_requirements.device.bytes != 0) {
        candidate->selector_device_scratch = impl.device->create_workspace(
                selector_requirements.device.bytes);
        validate_workspace_owner(
                candidate->selector_device_scratch.get(), *impl.device,
                selector_requirements.device, "selector device");
    }
    impl.validate_scratch(*candidate, operation, selector_requirements);
    impl.reset_cache_prefix();
    // Publish only after every owner, range, and workspace is complete.
    impl.request = std::move(candidate);

    // This second publication site carries the same recorder bridge as the
    // public request setup: the admitted request ordinal advances only here,
    // after the forward request was actually published, so one publication
    // replaces the admitted observation, clears the outgoing operation rows
    // without freeing prepared trace storage, and publishes a staged attempt.
    // Every earlier return above - a failed drain, a rejected length, or a
    // failed candidate allocation - left the outgoing admitted observation and
    // its rows untouched. Without this hook a raw/chat request could never
    // publish its staged tokenization attempt, and a direct low-level request
    // would keep inheriting the previous raw/chat tokenization observation.
    if (impl.metrics != nullptr) {
        impl.metrics->publish_generation();
    }
}

ForwardResult SessionAccess::forward_prefill(
        TinyLlamaSession& session, std::span<const std::size_t> token_ids) {
    TinyLlamaSession::Impl& impl = *session.impl_;
    impl.require_request();
    RequestState& request = *impl.request;
    if (token_ids.size() != request.run_length) {
        throw std::invalid_argument(
                "TinyLlama forward prefill length does not match setup");
    }
    encode_token_ids(
            token_ids, impl.config().vocab_size,
            std::span<std::uint32_t>(request.prefill_indices));
    // The accepted submissions of this prefill carry the whole-run prompt
    // window `[0,R)` and, inside a decoder-layer iteration, that configured
    // layer index. The closing normalization stays whole-run and the final
    // LM-head submission carries the one-row window `[R-1,R)`.
    SessionAccess::set_operation_context(
            session,
            OperationContext{InferencePhase::prefill, std::nullopt, 0,
                             request.run_length});
    try {
        const RawWorkspaceView sequential = forward_workspace_slice(
                request, request.forward->workspace.sequential_offset,
                request.forward->workspace.sequential_requirement);
        request.banks.token_indices->view().copy_from_host(
                std::as_bytes(std::span<const std::uint32_t>(
                        request.prefill_indices)),
                sequential);
        const oid embedding = SessionAccess::submit(
                session, [&](DeviceOps& operations) {
                    return operations.embedding(
                            impl.model->weight(0),
                            request.banks.token_indices->view(),
                            request.banks.x->view(), sequential);
                });
        std::span<const oid> readiness(&embedding, 1);
        for (std::size_t layer_index = 0;
             layer_index < request.forward->layers.size(); ++layer_index) {
            ForwardLayerPlan& layer =
                    request.forward->layers[layer_index];
            CacheOwner& cache = impl.caches[layer_index];
            layer.prefill_params.a = 0;
            layer.prefill.attention.a = 0;
            SessionAccess::set_operation_context(
                    session,
                    OperationContext{InferencePhase::prefill, layer_index, 0,
                                     request.run_length});
            DecoderLayerForwardState state{
                    cache.initialized_length, false};
            const DecoderLayerForwardWorkspace workspace =
                    forward_layer_workspace(
                            *impl.device, request, *request.forward);
            run_decoder_layer_forward(
                    *impl.queue, layer.prefill, layer.prefill_params,
                    workspace, readiness, state);
            cache.initialized_length = state.initialized_length;
            readiness = {};
        }

        SessionAccess::set_operation_context(
                session,
                OperationContext{InferencePhase::prefill, std::nullopt, 0,
                                 request.run_length});
        const oid final_norm = SessionAccess::submit(
                session, [&](DeviceOps& operations) {
                    const TensorView& input = final_layer_output(
                            request.banks,
                            impl.config().num_hidden_layers);
                    const WorkspaceRequirements requirement =
                            operations.rmsnorm_workspace_requirements(
                                    input, impl.model->weight(1),
                                    request.banks.final_norm->view(),
                                    impl.config().rms_norm_eps);
                    const RawWorkspaceView workspace =
                            requirement.bytes == 0 ? RawWorkspaceView{}
                                                   : sequential;
                    return operations.rmsnorm(
                            input, impl.model->weight(1),
                            request.banks.final_norm->view(),
                            impl.config().rms_norm_eps, workspace);
                });
        SessionAccess::wait(session, final_norm);
        SessionAccess::set_operation_context(
                session,
                OperationContext{InferencePhase::prefill, std::nullopt,
                                 request.run_length - 1, 1});
        const oid producer = SessionAccess::submit(
                session, [&](DeviceOps& operations) {
                    const TensorView& input =
                            request.banks.final_norm->view();
                    const TensorView& weight = impl.model->weight(2);
                    const WorkspaceRequirements requirement =
                            operations.linear_workspace_requirements(
                                    input, weight, impl.logits->view(),
                                    request.run_length - 1, 1,
                                    LinearOutputLayout::ordinary, 1,
                                    impl.config().vocab_size);
                    const RawWorkspaceView workspace =
                            requirement.bytes == 0 ? RawWorkspaceView{}
                                                   : sequential;
                    return operations.linear(
                            input, weight, impl.logits->view(),
                            request.run_length - 1, 1,
                            LinearOutputLayout::ordinary, 1,
                            impl.config().vocab_size, workspace);
                });
        return ForwardResult{&impl.logits->view(), producer};
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        impl.poisoned = true;
        try { impl.drain_request(); } catch (...) {}
        std::rethrow_exception(failure);
    }
}

ForwardResult SessionAccess::forward_decode(
        TinyLlamaSession& session, std::size_t token_id) {
    TinyLlamaSession::Impl& impl = *session.impl_;
    impl.require_request();
    RequestState& request = *impl.request;
    if (token_id >= impl.config().vocab_size) {
        throw std::invalid_argument(
                "TinyLlama forward decode token exceeds vocabulary");
    }
    if (impl.caches.empty()) {
        throw std::logic_error("TinyLlama forward session has no layers");
    }
    const std::size_t position = impl.caches.front().initialized_length;
    if (position >= impl.config().max_position_embeddings) {
        throw std::invalid_argument(
                "TinyLlama forward decode exceeds cache capacity");
    }
    for (const CacheOwner& cache : impl.caches) {
        if (cache.initialized_length != position) {
            throw std::logic_error(
                    "TinyLlama forward layer caches have different prefixes");
        }
    }
    request.decode_index = static_cast<std::uint32_t>(token_id);
    // The accepted submissions of this decode carry the current initialized
    // prefix as their absolute start with the fixed one-row run, and each
    // decoder-layer iteration adds its configured layer index. The one-row
    // run makes the closing normalization and the final LM-head row share the
    // same `[a,a+1)` window, so one no-layer attribution covers both.
    SessionAccess::set_operation_context(
            session,
            OperationContext{InferencePhase::decode, std::nullopt, position, 1});
    try {
        const RawWorkspaceView sequential = forward_workspace_slice(
                request, request.forward->workspace.sequential_offset,
                request.forward->workspace.sequential_requirement);
        impl.fixed.token_indices->view().copy_from_host(
                std::as_bytes(std::span<const std::uint32_t>(
                        &request.decode_index, 1)),
                sequential);
        const oid embedding = SessionAccess::submit(
                session, [&](DeviceOps& operations) {
                    return operations.embedding(
                            impl.model->weight(0),
                            impl.fixed.token_indices->view(),
                            impl.fixed.x->view(), sequential);
                });
        std::span<const oid> readiness(&embedding, 1);
        for (std::size_t layer_index = 0;
             layer_index < request.forward->layers.size(); ++layer_index) {
            ForwardLayerPlan& layer =
                    request.forward->layers[layer_index];
            CacheOwner& cache = impl.caches[layer_index];
            layer.decode_params.a = position;
            layer.decode.attention.a = position;
            SessionAccess::set_operation_context(
                    session,
                    OperationContext{InferencePhase::decode, layer_index,
                                     position, 1});
            DecoderLayerForwardState state{
                    cache.initialized_length, false};
            const DecoderLayerForwardWorkspace workspace =
                    forward_layer_workspace(
                            *impl.device, request, *request.forward);
            run_decoder_layer_forward(
                    *impl.queue, layer.decode, layer.decode_params,
                    workspace, readiness, state);
            cache.initialized_length = state.initialized_length;
            readiness = {};
        }

        SessionAccess::set_operation_context(
                session,
                OperationContext{InferencePhase::decode, std::nullopt, position,
                                 1});
        const oid final_norm = SessionAccess::submit(
                session, [&](DeviceOps& operations) {
                    const TensorView& input = final_layer_output(
                            impl.fixed,
                            impl.config().num_hidden_layers);
                    const WorkspaceRequirements requirement =
                            operations.rmsnorm_workspace_requirements(
                                    input, impl.model->weight(1),
                                    impl.fixed.final_norm->view(),
                                    impl.config().rms_norm_eps);
                    const RawWorkspaceView workspace =
                            requirement.bytes == 0 ? RawWorkspaceView{}
                                                   : sequential;
                    return operations.rmsnorm(
                            input, impl.model->weight(1),
                            impl.fixed.final_norm->view(),
                            impl.config().rms_norm_eps, workspace);
                });
        SessionAccess::wait(session, final_norm);
        const oid producer = SessionAccess::submit(
                session, [&](DeviceOps& operations) {
                    const TensorView& input = impl.fixed.final_norm->view();
                    const TensorView& weight = impl.model->weight(2);
                    const WorkspaceRequirements requirement =
                            operations.linear_workspace_requirements(
                                    input, weight, impl.logits->view(), 0, 1,
                                    LinearOutputLayout::ordinary, 1,
                                    impl.config().vocab_size);
                    const RawWorkspaceView workspace =
                            requirement.bytes == 0 ? RawWorkspaceView{}
                                                   : sequential;
                    return operations.linear(
                            input, weight, impl.logits->view(), 0, 1,
                            LinearOutputLayout::ordinary, 1,
                            impl.config().vocab_size, workspace);
                });
        return ForwardResult{&impl.logits->view(), producer};
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        impl.poisoned = true;
        try { impl.drain_request(); } catch (...) {}
        std::rethrow_exception(failure);
    }
}

}  // namespace iom::session_detail

