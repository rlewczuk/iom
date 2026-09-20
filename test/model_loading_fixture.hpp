#pragma once

// Shared, network-free model-loading fixture.
//
// It owns one isolated temporary directory per test and writes deterministic
// `config.json` documents and SafeTensors checkpoints into it, so
// configuration and weight-loading scenarios reuse one directory helper and
// one checkpoint writer instead of adding per-backend copies. Each test
// constructs its own `TempDir`, whose path is removed on scope exit.

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "iom/device.hpp"
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

// The one-layer tile-boundary configuration: N=1, H=16, I=20, Hq=4, Hkv=2,
// D=4, V=19, C=17. Its hidden size is exactly one 16x16 tile column, and its
// grouped K/V width `Hkv*D = 8` is half of `Hq*D = 16`.
inline nlohmann::json h16_config() {
    nlohmann::json document = two_layer_config();
    document["num_hidden_layers"] = 1;
    document["hidden_size"] = 16;
    document["intermediate_size"] = 20;
    document["num_attention_heads"] = 4;
    document["num_key_value_heads"] = 2;
    return document;
}

// The one-layer off-tile configuration: N=1, H=18, I=22, Hq=3, Hkv=1, D=6,
// V=19, C=17. Neither the hidden size nor the intermediate size is a multiple
// of the fixed 16x16 tile, so every selected weight and both normalization
// payloads cross a tile boundary.
inline nlohmann::json h18_config() {
    nlohmann::json document = two_layer_config();
    document["num_hidden_layers"] = 1;
    document["hidden_size"] = 18;
    document["intermediate_size"] = 22;
    document["num_attention_heads"] = 3;
    document["num_key_value_heads"] = 1;
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

// ---------------------------------------------------------------------------
// Independent inventory expectation
// ---------------------------------------------------------------------------

// One expected inventory entry: the logical identity and the selected logical
// shape of a published weight, encoded here independently of the
// implementation, so a wrong role, layer, order, or shape fails instead of
// matching the production table by construction. `layer` is absent for the
// three global roles and carries the decoder layer for the nine layer-scoped
// roles.
struct ExpectedWeight {
    iom::ModelWeightId id;
    std::vector<std::size_t> logical_shape;
};

// The nine layer-scoped roles of those dimensions in canonical order.
inline std::vector<std::pair<iom::ModelWeightRole, std::vector<std::size_t>>>
layer_weight_plan(std::size_t hidden, std::size_t intermediate,
                  std::size_t kv_width) {
    return {
            {iom::ModelWeightRole::input_norm, {1, hidden}},
            {iom::ModelWeightRole::post_attention_norm, {1, hidden}},
            {iom::ModelWeightRole::query, {hidden, hidden}},
            {iom::ModelWeightRole::key, {kv_width, hidden}},
            {iom::ModelWeightRole::value, {kv_width, hidden}},
            {iom::ModelWeightRole::attention_output, {hidden, hidden}},
            {iom::ModelWeightRole::mlp_gate, {intermediate, hidden}},
            {iom::ModelWeightRole::mlp_up, {intermediate, hidden}},
            {iom::ModelWeightRole::mlp_down, {hidden, intermediate}},
    };
}

// The complete expected inventory of `config` in canonical global/layer order:
// the three globals, then the nine layer-scoped roles of every configured
// decoder layer. The normalization roles are `[1, H]` here, which is the
// adapted logical metadata of their rank-one `[H]` checkpoint payloads.
inline std::vector<ExpectedWeight> expected_inventory(
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

    std::vector<ExpectedWeight> expected{
            {{iom::ModelWeightRole::token_embedding, std::nullopt},
             {vocab, hidden}},
            {{iom::ModelWeightRole::final_norm, std::nullopt}, {1, hidden}},
            {{iom::ModelWeightRole::lm_head, std::nullopt}, {vocab, hidden}},
    };
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (const auto& role :
             layer_weight_plan(hidden, intermediate, kv_width)) {
            expected.push_back(ExpectedWeight{{role.first, layer}, role.second});
        }
    }
    return expected;
}

// Checks every published entry, and the selected metadata of every index,
// against the independent expectation.
inline void check_inventory(const iom::ModelSource& source,
                            const std::vector<ExpectedWeight>& expected) {
    const std::span<const iom::ModelWeightInfo> weights = source.weights();
    REQUIRE(weights.size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CAPTURE(index);
        const ExpectedWeight& want = expected[index];
        const iom::ModelWeightInfo& info = weights[index];
        CHECK(info.id.role == want.id.role);
        // The identity comparison covers both directions: a global that
        // wrongly carries a layer and a layer-scoped role that wrongly
        // omits one both fail here with the exact expected/actual layer.
        CHECK(info.id.layer == want.id.layer);
        CHECK(dims_of(info.logical_shape) == want.logical_shape);

        const iom::TensorSpec& spec = source.tensor_spec(index);
        CHECK(dims_of(spec.shape) == want.logical_shape);
        CHECK(spec.data_type == iom::DataType::BF16);
        CHECK(spec.quantization == iom::QuantizationFormat::NONE);
        CHECK(spec.logical_nbytes() == 2 * element_count(want.logical_shape));
    }
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

// ---------------------------------------------------------------------------
// Destination preflight fixture
// ---------------------------------------------------------------------------

// Bounded storage capability tables: only BF16 membership matters to the
// preflight, so the fixture does not duplicate the 23-entry backend table.
inline constexpr std::array<iom::DataType, 2> kSupportedWithBf16 = {
        iom::DataType::BF16, iom::DataType::F32};
inline constexpr std::array<iom::DataType, 2> kSupportedWithoutBf16 = {
        iom::DataType::F16, iom::DataType::F32};

// The host-transfer workspace requirement policy of one bounded destination.
// A zero multiplier is the CPU `{0, 1}` policy; a positive one has the
// accelerator shape `{checked logical bytes * multiplier, alignment}`, so the
// preflight's maxima stay observable as actual owner query outputs.
struct WorkspacePolicy {
    std::size_t multiplier = 0;
    std::size_t alignment = 1;
};

// The established failure category one bounded destination reports from a
// chosen upload instead of recording its bytes. `none` records every upload.
enum class UploadFailure {
    none,
    runtime_error,
    allocation,
};

// The fixed sentinel content of every bounded destination. It is never sized
// from a declared specification, so an unsizeable destination stays
// constructible without allocating anything.
inline std::vector<std::byte> sentinel_storage() {
    return std::vector<std::byte>(4, std::byte{0xA5});
}

// One bounded destination owner without a backend, an arena, or native
// storage. It records the requirement queries and upload calls its owner
// receives, so a test can prove what a preflight or a synchronous realization
// did and did not do. A bounded upload failure policy additionally makes the
// chosen upload throw an established category instead of recording bytes, so a
// later synchronous realization failure, its propagation, and the untouched
// destinations after it stay observable without any production failure
// injection.
class FakeTensor final : public iom::Tensor {
public:
    FakeTensor(iom::TensorSpec spec, iom::Device& device,
               WorkspacePolicy policy = {})
        : iom::Tensor(std::move(spec), device),
          policy_(policy),
          storage_(sentinel_storage()) {}

    // Host-transfer requirement queries this owner has received.
    [[nodiscard]] std::size_t requirement_queries() const noexcept {
        return requirement_queries_;
    }
    // Host uploads this owner has received.
    [[nodiscard]] std::size_t from_host_calls() const noexcept {
        return from_host_calls_;
    }
    [[nodiscard]] const std::vector<std::byte>& uploaded_bytes() const noexcept {
        return uploaded_bytes_;
    }
    [[nodiscard]] const std::vector<std::byte>& storage() const noexcept {
        return storage_;
    }

    // Makes the upload whose 1-based ordinal on this owner matches `ordinal`
    // throw `category` instead of recording bytes. Every received upload is
    // still counted, other owners are unaffected, and no queue, retry, or
    // production injection seam is involved.
    void fail_upload_at(std::size_t ordinal, UploadFailure category) noexcept {
        failing_upload_ = ordinal;
        upload_failure_ = category;
    }

    [[nodiscard]] iom::WorkspaceRequirements
            host_transfer_workspace_requirements(
                    std::size_t checked_logical_nbytes) const override {
        ++requirement_queries_;
        if (policy_.multiplier == 0) {
            return iom::WorkspaceRequirements{0, 1};
        }
        return iom::WorkspaceRequirements{
                checked_logical_nbytes * policy_.multiplier,
                policy_.alignment};
    }

    [[nodiscard]] void* storage_handle() noexcept override {
        return storage_.data();
    }

    void region_from_host(const iom::TensorView&,
                          std::span<const std::byte> source,
                          iom::RawWorkspaceView) override {
        ++from_host_calls_;
        if (from_host_calls_ == failing_upload_) {
            if (upload_failure_ == UploadFailure::allocation) {
                throw std::bad_alloc();
            }
            if (upload_failure_ == UploadFailure::runtime_error) {
                throw std::runtime_error("bounded destination upload failure");
            }
        }
        uploaded_bytes_.assign(source.begin(), source.end());
    }

    void region_to_host(const iom::TensorView&, std::span<std::byte>,
                        iom::RawWorkspaceView) const override {
        // The bounded fixture persists no payload, so a readback must not
        // silently appear to succeed.
        throw std::logic_error(
                "the bounded destination fixture has no readable storage");
    }

private:
    WorkspacePolicy policy_;
    UploadFailure upload_failure_ = UploadFailure::none;
    std::size_t failing_upload_ = 0;
    mutable std::size_t requirement_queries_ = 0;
    std::size_t from_host_calls_ = 0;
    std::vector<std::byte> uploaded_bytes_;
    std::vector<std::byte> storage_;
};

/**
 * Test raw-workspace owner with no backing range. The fixture registers its
 * exact identity through the `RawWorkspace` base like every real owner, so
 * live/foreign/dead ownership checks have a real owner to observe.
 */
class FakeWorkspace final : public iom::RawWorkspace {
public:
    FakeWorkspace(iom::Device& device, std::size_t bytes)
        : iom::RawWorkspace(device, bytes) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return nullptr;
    }
};

/**
 * Bounded raw-workspace owner whose reported range base is supplied by the
 * caller. No backing storage is allocated: the fixture only needs a real base
 * address, a real capacity, and a real owner identity, so the shared workspace
 * validation observes genuine liveness, exact-device, capacity, alignment, and
 * operand-overlap ranges without a backend arena. Destroying it leaves exactly
 * the dead owner identity a released backend workspace leaves.
 */
class BoundedWorkspace final : public iom::RawWorkspace {
public:
    BoundedWorkspace(iom::Device& device, std::size_t bytes, void* address)
        : iom::RawWorkspace(device, bytes), address_(address) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    void* address_;
};

/**
 * Bounded backend-neutral device: no arena, no native storage, and no queue.
 * It records workspace provisioning so a test can prove the preflight
 * provisions nothing, and every instance reports the same backend kind and
 * ordinal so only exact `Device` identity can separate two instances.
 */
class FakeDevice final : public iom::Device {
public:
    explicit FakeDevice(
            std::span<const iom::DataType> supported = kSupportedWithBf16,
            iom::QueueConfig queue_config = {})
        : iom::Device(queue_config), supported_(supported) {}

    [[nodiscard]] iom::BackendKind backend_kind() const noexcept override {
        return iom::BackendKind::CPU;
    }

    [[nodiscard]] std::uint32_t backend_device() const noexcept override {
        return 3;
    }

    [[nodiscard]] std::span<const iom::DataType>
            supported_data_types() const noexcept override {
        return supported_;
    }

    [[nodiscard]] std::unique_ptr<iom::Tensor> create_tensor(
            const iom::TensorSpec& spec) override {
        ++tensor_creations_;
        return std::make_unique<FakeTensor>(spec, *this, policy_);
    }

    // Mirrors the CPU policy: zero bytes are a valid allocation-free owner,
    // and positive bytes are unsupported device scratch.
    [[nodiscard]] std::unique_ptr<iom::RawWorkspace> create_workspace(
            std::size_t bytes) override {
        ++workspace_creations_;
        if (bytes != 0) {
            throw std::invalid_argument(
                    "the bounded fixture device has no positive workspace");
        }
        return std::make_unique<FakeWorkspace>(*this, 0);
    }

    [[nodiscard]] std::unique_ptr<iom::DeviceOps> create_ops() override {
        throw std::logic_error(
                "the bounded fixture device creates no operation queue");
    }

    // The policy every subsequent `create_tensor` uses.
    void set_workspace_policy(WorkspacePolicy policy) noexcept {
        policy_ = policy;
    }

    // Raw-workspace provisioning attempted on this device.
    [[nodiscard]] std::size_t workspace_creations() const noexcept {
        return workspace_creations_;
    }

    // Destination owners this device created, so a realization pass that
    // hides its own tensor allocation stays observable.
    [[nodiscard]] std::size_t tensor_creations() const noexcept {
        return tensor_creations_;
    }

private:
    std::span<const iom::DataType> supported_;
    WorkspacePolicy policy_;
    std::size_t workspace_creations_ = 0;
    std::size_t tensor_creations_ = 0;
};

// Creates one bounded destination owner per published inventory entry, in
// exact inventory order, from the source's own selected specifications.
inline std::vector<std::unique_ptr<iom::Tensor>> make_destinations(
        const iom::ModelSource& source, FakeDevice& device,
        WorkspacePolicy policy = {}) {
    device.set_workspace_policy(policy);
    std::vector<std::unique_ptr<iom::Tensor>> destinations;
    destinations.reserve(source.weights().size());
    for (std::size_t index = 0; index < source.weights().size(); ++index) {
        destinations.push_back(device.create_tensor(source.tensor_spec(index)));
    }
    return destinations;
}

// The stable destination pointers of one binding, in inventory order.
inline std::vector<iom::Tensor*> destination_pointers(
        const std::vector<std::unique_ptr<iom::Tensor>>& destinations) {
    std::vector<iom::Tensor*> pointers;
    pointers.reserve(destinations.size());
    for (const std::unique_ptr<iom::Tensor>& destination : destinations) {
        pointers.push_back(destination.get());
    }
    return pointers;
}

// The exact borrowed destination list the preflight ABI takes.
inline std::span<iom::Tensor* const> as_destinations(
        std::vector<iom::Tensor*>& pointers) {
    return std::span<iom::Tensor* const>{pointers};
}

// The bounded owner of one destination index, for effect observation. Every
// destination of a `FakeDevice` is a `FakeTensor`.
inline const FakeTensor& observed(
        const std::vector<std::unique_ptr<iom::Tensor>>& destinations,
        std::size_t index) {
    return static_cast<const FakeTensor&>(*destinations[index]);
}

// The same owner for a bounded upload failure policy, so a test can choose the
// exact later upload of one destination that must fail.
inline FakeTensor& mutable_observed(
        const std::vector<std::unique_ptr<iom::Tensor>>& destinations,
        std::size_t index) {
    return static_cast<FakeTensor&>(*destinations[index]);
}

}  // namespace iom_model_loading