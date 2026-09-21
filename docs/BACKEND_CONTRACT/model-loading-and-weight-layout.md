# 10. Model loading and weight layout

Model ingestion begins with an explicit model directory and stays
backend-neutral. Reading a configuration never searches tokenizer, generation,
distribution, or fallback files, never creates a mapping, device, tensor, or
workspace, and never copies checkpoint payload bytes. Configuration validation
precedes every later loading stage, and the mapped-file -> `SafeTensors` ->
caller-created tensor flow with the host-transfer rules of
[section 4](view-and-transfer-contract.md#4-view-and-transfer-contract) is unchanged. Later loading stages
extend this section instead of defining competing contracts.

## Supported TinyLlama configuration

`iom::load_tinyllama_config(const std::filesystem::path& model_directory)` in
`include/iom/model.hpp` reads exactly `<model_directory>/config.json` and
returns a fully populated `iom::TinyLlamaConfig`. The public header declares no
JSON type, there is no partial or empty success value, and only runtime values
are stored:

```cpp
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
```

The supported fields, their defaults, and their required values are normative:

| JSON field | Rule | Result |
| --- | --- | --- |
| `architectures` | Required array, exactly `["LlamaForCausalLM"]` | no stored field |
| `model_type` | Required string, exactly `"llama"` | no stored field |
| `num_hidden_layers` | Required positive integer | `N` |
| `hidden_size` | Required positive integer | `H` |
| `intermediate_size` | Required positive integer | `I` |
| `num_attention_heads` | Required positive integer | `Hq` |
| `num_key_value_heads` | Required positive integer | `Hkv` |
| `vocab_size` | Required positive integer | `V` |
| `max_position_embeddings` | Required positive integer | `C` |
| `hidden_act` | Required string, exactly `"silu"` | no stored field |
| `rms_norm_eps` | Required finite positive number that stays finite and positive as `float` | `float` epsilon |
| `rope_theta` | Required finite positive number representable as `double` | `double` theta |
| `attention_bias` | Optional; absent means `false`, and a present value must be the boolean `false` | no stored field |
| `mlp_bias` | Optional with the `attention_bias` rule | no stored field |
| `tie_word_embeddings` | Optional with the `attention_bias` rule | no stored field |
| `rope_scaling` | Optional; absent means no scaling, and a present value must be `null` | no stored field |
| `pretraining_tp` | Optional; absent means the integer `1`, and a present value must be the integer `1` | no stored field |
| `torch_dtype` | Required string, exactly `"bfloat16"` | selected source dtype |
| `bos_token_id` | Required integer, exactly `1`, and below `V` | `1` |
| `eos_token_id` | Required integer, exactly `2`, and below `V` | `2` |
| `initializer_range`, `transformers_version`, `use_cache`, `pad_token_id` | Optional metadata: no validation and no merged default, and no effect on any stored value | not stored |
| any other key, including a head-dimension override | Rejected as unknown | none |

Every integer field accepts JSON integers only. Floating-point values including
`1.0`, booleans, strings, arrays, and `null` are rejected, and each accepted
value must be positive and representable as `std::size_t`; token ids are never
list-valued. The configuration declares no independent head width, and the head
plan is derived rather than read:

- `H % Hq == 0`, and `Hq % Hkv == 0`;
- `head_dim` is `D = H / Hq`, which must be positive and even;
- `Hkv * D` is the grouped K/V width and is computed with checked arithmetic.

`N`, `H`, `I`, `Hq`, `Hkv`, `D`, `V`, and `C` are the runtime parameters of
[TinyLlama forward layout — Session sizing and lifetime](tinyllama-forward-layout-session-sizing-and-lifetime.md#tinyllama-forward-layout--session-sizing-and-lifetime),
which owns the checked `F = Hq * D` equality and every storage consequence of
those values. This section adds no checkpoint-specific dimension limit: a
one-layer checkpoint and the two-layer `N=2, H=8, I=12, Hq=4, Hkv=2, V=19,
C=17` boundary fixture are both supported.

## Configuration failures

| Condition | Exception |
| --- | --- |
| Missing `config.json`, an unreadable file, or a failed read | `std::runtime_error` carrying the path and the system reason |
| Malformed, empty, or non-object JSON | `std::invalid_argument` carrying the path |
| Missing field, wrong type or integer kind, unsupported value, unknown key, nonrepresentable narrowing, invalid head ratio, odd `head_dim`, or invalid token policy | `std::invalid_argument` carrying the path, the field, the actual value or `<missing>`, and the violated constraint |
| Overflow of a checked derived width | `std::overflow_error` |
| Allocation failure while reading or decoding | `std::bad_alloc` |

A missing `config.json` is an I/O failure, never a missing-field error, and
decode exceptions from the JSON library are translated at this boundary only,
so the standalone SafeTensors parser keeps its own behavior. Configuration
reading allocates configuration and schema metadata alone: it creates no device
tensor, transfer workspace, queue, or checkpoint payload copy. The mapped
source, destination preflight, and synchronous realization stages that follow
extend this section with their own normative subsections.

## Mapped weight source and logical inventory

`iom::load_tinyllama_safetensors(const std::filesystem::path& model_directory)`
in `include/iom/model.hpp` reuses the configuration validation above and then
validates the complete required SafeTensors inventory of that same directory.
It returns an owning `iom::ModelSource` only after every required role passed,
and it still creates no device, tensor, or workspace:

```cpp
enum class ModelWeightRole { token_embedding, final_norm, lm_head,
    input_norm, post_attention_norm, query, key, value, attention_output,
    mlp_gate, mlp_up, mlp_down };
struct ModelWeightId { ModelWeightRole role; std::optional<std::size_t> layer; };
struct ModelWeightInfo { ModelWeightId id; TensorShape logical_shape; };
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
    WorkspaceRequirements upload_workspace_requirements(
        const Device& device, std::span<Tensor* const> destinations) const;
    void upload_weights(Device& device, std::span<Tensor* const> destinations,
        RawWorkspaceView workspace = {}) const;
private:
    struct Impl;
};
std::unique_ptr<ModelSource> load_tinyllama_safetensors(
    const std::filesystem::path& model_directory);
```

This ABI is fixed. `ModelSource` is non-copyable and non-movable, its private
constructor is defined out of line, and its `Impl` privately retains exactly one
owning `SafeTensorsDir` for the whole source lifetime, so the owner is
constructed only after the complete inventory passed. That `Impl` additionally
retains one borrowed mapped payload span per published entry, pointing into the
store it owns, so the synchronous realization below reads the retained mapping
instead of a retained host copy. `weights()` borrows the
published inventory for that lifetime; `tensor_spec(index)` returns the selected
`TensorSpec` of one entry and throws the ordinary `std::out_of_range` outside
`[0, weights().size())`.

**Inventory order.** `ModelWeightId.layer` is absent for the three globals and
is in `[0, N)` for the other nine roles. The published order is exactly the
globals `token_embedding`, `final_norm`, `lm_head`, followed by increasing
layer and, within each layer, `input_norm`, `post_attention_norm`, `query`,
`key`, `value`, `attention_output`, `mlp_gate`, `mlp_up`, `mlp_down`.

**Required checkpoint schema.** `N`, `H`, `I`, `Hq`, `Hkv`, `D`, `V`, and `C`
are the validated runtime parameters of
[TinyLlama forward layout — Session sizing and lifetime](tinyllama-forward-layout-session-sizing-and-lifetime.md#tinyllama-forward-layout--session-sizing-and-lifetime),
whose per-layer weight inventory these selected sizes reproduce. Every required
name below is mandatory, uniquely named, BF16, and of exact source rank/shape:

| SafeTensors name | Source shape | Role | Logical shape |
| --- | --- | --- | --- |
| `model.embed_tokens.weight` | `[V, H]` | `token_embedding` | source shape |
| `model.norm.weight` | `[H]` | `final_norm` | `[1, H]` |
| `lm_head.weight` | `[V, H]` | `lm_head` | source shape |
| `model.layers.{l}.input_layernorm.weight` | `[H]` | `input_norm` | `[1, H]` |
| `model.layers.{l}.post_attention_layernorm.weight` | `[H]` | `post_attention_norm` | `[1, H]` |
| `model.layers.{l}.self_attn.q_proj.weight` | `[H, H]` | `query` | source shape |
| `model.layers.{l}.self_attn.k_proj.weight` | `[Hkv*D, H]` | `key` | source shape |
| `model.layers.{l}.self_attn.v_proj.weight` | `[Hkv*D, H]` | `value` | source shape |
| `model.layers.{l}.self_attn.o_proj.weight` | `[H, H]` | `attention_output` | source shape |
| `model.layers.{l}.mlp.gate_proj.weight` | `[I, H]` | `mlp_gate` | source shape |
| `model.layers.{l}.mlp.up_proj.weight` | `[I, H]` | `mlp_up` | source shape |
| `model.layers.{l}.mlp.down_proj.weight` | `[H, I]` | `mlp_down` | source shape |

Every required role carries exactly the checked logical byte length
`2 * product(source shape)`. HF `[out, in]` orientation and row-major element
order are preserved unchanged: no transpose, host repack, or pre-tiled form is
introduced. A rank-one normalization vector becomes logical metadata `[1, H]`
only, its mapped span stays exactly `2*H` bytes, no rank-one `TensorShape` is
exposed, and no final-axis view or reshape is used. `ModelWeightInfo.logical_shape`
stays independent of the selected encoding, the physical row-major mapped
region, and the retaining mapping owner, and `tensor_spec(index)` exposes
selected logical metadata only — never host bytes, container types, or a native
allocation size.

**Checked sizing.** The validated configuration yields the checked required
count `3 + 9*N` before any name is generated. Counts, element products, logical
byte counts, standard 16x16 tiled byte counts, and the aggregate weight totals
are all computed with checked arithmetic; `logical_nbytes()` and
`tiled_storage_nbytes()` of each selected `TensorSpec` are the numbers used. The
repeated-layer aggregate is derived from the nine representative role sizes and
`N`, so an impossible configuration fails without allocating a huge payload or
generating a huge name list. Standard tiled totals are never presented as
backend-private sizes; native allocation, stride, and address checks remain
`Device`-factory responsibilities.

**Ownership and validation order.** The order is fixed: validate the
configuration, compute the checked required count `3 + 9*N`, construct exactly
one private `SafeTensorsDir` so exact-extension discovery, every tensor header,
byte, and range check, and duplicate-shard rejection happen before required-name
filtering, reject an empty store, check the element products, logical byte
counts, standard tiled byte counts, and repeated-layer aggregate from the
validated dimensions, reject a store whose tensor count is below `3 + 9*N`
before enumerating any layer, then validate each required role and publish.
Extras are ignored only after complete parser validation, and a valid
extra may use another dtype. No index JSON and no duplicate-JSON-key parsing is
added. `ModelSource` privately owns the mapping, no payload byte is traversed or
copied beyond container metadata validation, and no persistent second
checkpoint image or host copy exists. Every failure is cleaned up by RAII and
leaves the source artifacts unchanged.

**Source failures.** These categories are normative:

| Condition | Exception |
| --- | --- |
| Missing required weight, or a wrong source rank, shape, dtype, or byte length | `std::invalid_argument` naming the directory, the logical checkpoint key, the required rank/shape/dtype, and the actual rank/shape/dtype or `<missing>` |
| Empty store, or fewer than `3 + 9*N` stored tensors | `std::invalid_argument` naming the directory and the required and actual tensor counts, before any required weight is named |
| Impossible required count or checked byte total | `std::overflow_error` |
| Malformed container header JSON | `std::runtime_error` naming the directory; only this boundary translates that container leak, and the standalone SafeTensors API keeps its own behavior |
| Malformed extra tensor, invalid container fields, or a duplicate name across shards | the established container categories of the SafeTensors parser, unchanged |
| Allocation failure | `std::bad_alloc` |

A missing required name is therefore a model-schema rejection, not the store's
missing-name `std::out_of_range`, while `tensor_spec(index)` keeps ordinary
bounds behavior. The mapped-file -> SafeTensors -> caller-created tensor flow,
the host-transfer rules of [section 4](view-and-transfer-contract.md#4-view-and-transfer-contract), and the
realization ownership rules remain authoritative; upload, preflight, and
device realization are separate later stages that consume this published
source.

**Extension boundary.** This adapter is the TinyLlama SafeTensors/BF16 source
only. It must not expose container types, make Hugging Face file keys a
universal storage identity, or define a logical weight as permanently having
one BF16 host representation, so a later GGUF adapter or a container holding
several differently quantized or differently stored representations of one
logical role stays expressible. Such a re-selection must never retarget an
existing immutable owner or view or invalidate an in-flight user. Today's
read-only mmap and row-major bytes establish no direct-DMA, unified-memory,
or zero-copy capability, so this stage adds no DMA handle, registration API,
zero-copy guarantee, variant list, codec, selector, eviction policy, or format
registry, and a future aliased destination must independently retain its
backing. Existing copy/staging remains the current path.
