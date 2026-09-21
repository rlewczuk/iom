#include <doctest/doctest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/device.hpp"
#include "iom/inference_metrics.hpp"
#include "iom/iom.hpp"
#include "iom/session.hpp"
#include "iom/tensor.hpp"
#include "../src/session_internal.hpp"
#include "model_loading_fixture.hpp"
#include "tokenizer_fixture.hpp"
namespace {

using iom_model_loading::TempDir;
using iom_model_loading::one_layer_config;
using iom_model_loading::two_layer_config;
using iom_model_loading::required_weight_entries;
using iom_model_loading::write_config;
using iom_model_loading::write_file;
using iom_model_loading::write_safetensors_file;
using iom_tokenizer_test::write_tokenizer;
using iom::session_detail::SessionAccess;

class HeapAllocator final : public iom::Allocator {
public:
    std::size_t fail_after = std::numeric_limits<std::size_t>::max();
    std::size_t allocations = 0;
    std::size_t live = 0;
    std::size_t resets = 0;

    void* alloc(std::size_t bytes) override {
        if (allocations++ >= fail_after) {
            throw std::bad_alloc();
        }
        void* address = ::operator new(std::max<std::size_t>(bytes, 1),
                                       std::align_val_t{32});
        ++live;
        return address;
    }

    void free(void* address) override {
        --live;
        ::operator delete(address, std::align_val_t{32});
    }

    void reset() override { ++resets; }
};

[[nodiscard]] nlohmann::json tokenizer_config() {
    const auto metadata = [](const char* content) {
        return nlohmann::json{{"content", content},
                              {"lstrip", false},
                              {"normalized", false},
                              {"rstrip", false},
                              {"single_word", false},
                              {"special", true}};
    };
    return nlohmann::json{
            {"added_tokens_decoder",
             nlohmann::json{{"0", metadata("<unk>")},
                            {"1", metadata("<s>")},
                            {"2", metadata("</s>")}}},
            {"bos_token", "<s>"},
            {"chat_template",
             "{% for message in messages %}"
             "{% if message['role'] == 'system' %}"
             "{{ message['content'] + eos_token }}"
             "{% elif message['role'] == 'user' %}"
             "{{ message['content'] + eos_token }}"
             "{% elif message['role'] == 'assistant' %}"
             "{{ message['content'] + eos_token }}"
             "{% endif %}"
             "{% if loop.last and add_generation_prompt %}"
             "{{ '<|assistant|>' }}"
             "{% endif %}"
             "{% endfor %}"},
            {"clean_up_tokenization_spaces", false},
            {"eos_token", "</s>"},
            {"legacy", false},
            {"model_max_length", 2048},
            {"pad_token", "</s>"},
            {"padding_side", "right"},
            {"sp_model_kwargs", nlohmann::json::object()},
            {"tokenizer_class", "LlamaTokenizer"},
            {"unk_token", "<unk>"},
            {"use_default_system_prompt", false},
    };
}

struct SessionFixture {
    TempDir directory{"session-resources"};
    HeapAllocator allocator;
    std::unique_ptr<iom::Device> device;

    SessionFixture() {
        const nlohmann::json config = one_layer_config();
        write_config(directory.path(), config);
        write_safetensors_file(
                directory.path(), "model.safetensors",
                required_weight_entries(config));
        write_tokenizer(directory.path());
        write_file(directory.path(), "tokenizer_config.json",
                   tokenizer_config().dump());
        device = iom::make_cpu_device(allocator);
    }

    [[nodiscard]] std::unique_ptr<iom::TinyLlamaSession> load() {
        return iom::load_tinyllama_session(directory.path(), *device);
    }
};

[[nodiscard]] std::uint16_t forward_encode_bf16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7FFFU + lsb) >> 16);
}

[[nodiscard]] float forward_decode_bf16(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

[[nodiscard]] std::string forward_finite_payload(
        const std::vector<std::size_t>& shape, std::size_t entry) {
    const std::size_t elements = iom_model_loading::element_count(shape);
    std::string payload(elements * sizeof(std::uint16_t), '\0');
    for (std::size_t element = 0; element < elements; ++element) {
        float value = 0.015F
                * static_cast<float>(1 + ((entry + element) % 7));
        if (entry == 1
                || (entry >= 3 && ((entry - 3) % 9) < 2)) {
            value = 1.0F + 0.015F
                    * static_cast<float>((entry + element) % 3);
        }
        const std::uint16_t bits = forward_encode_bf16(value);
        std::memcpy(payload.data() + element * sizeof(bits), &bits,
                    sizeof(bits));
    }
    return payload;
}

void write_forward_weights(
        const std::filesystem::path& directory,
        const nlohmann::json& config) {
    auto entries = required_weight_entries(config);
    for (std::size_t entry = 0; entry < entries.size(); ++entry) {
        entries[entry].payload =
                forward_finite_payload(entries[entry].shape, entry);
    }
    write_safetensors_file(directory, "model.safetensors", entries);
}

struct ForwardFixture {
    TempDir directory;
    HeapAllocator allocator;
    std::unique_ptr<iom::Device> device;

    ForwardFixture(std::string tag, const nlohmann::json& config)
        : directory("session-forward-" + std::move(tag)) {
        write_config(directory.path(), config);
        write_forward_weights(directory.path(), config);
        write_tokenizer(directory.path());
        write_file(directory.path(), "tokenizer_config.json",
                   tokenizer_config().dump());
        device = iom::make_cpu_device(allocator);
    }

    [[nodiscard]] std::unique_ptr<iom::TinyLlamaSession> load() {
        return iom::load_tinyllama_session(directory.path(), *device);
    }
};

[[nodiscard]] nlohmann::json text_generation_config() {
    nlohmann::json config = one_layer_config();
    config["vocab_size"] = 32'000;
    config["max_position_embeddings"] = 64;
    return config;
}

[[nodiscard]] std::vector<float> read_forward_tensor(
        const iom::TensorView& view) {
    REQUIRE(view.spec().data_type == iom::DataType::BF16);
    const std::size_t elements = view.spec().shape.element_count();
    std::vector<std::byte> bytes(elements * sizeof(std::uint16_t));
    view.copy_to_host(bytes);
    std::vector<float> values(elements);
    for (std::size_t element = 0; element < elements; ++element) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes.data() + element * sizeof(bits), sizeof(bits));
        values[element] = forward_decode_bf16(bits);
    }
    return values;
}


class ProbeSelector final : public iom::TokenSelector {
public:
    explicit ProbeSelector(
            std::size_t* destructions = nullptr,
            iom::TokenSelectorScratchRequirements requirements = {0, {0, 1}})
        : destructions_(destructions), requirements_(requirements) {}

    ~ProbeSelector() override {
        if (destructions_) ++*destructions_;
    }

    [[nodiscard]] iom::TokenSelectorScratchRequirements scratch_requirements(
            const iom::TensorView&, std::size_t) const override {
        return requirements_;
    }

    [[nodiscard]] std::size_t select(
            iom::DeviceOps&, const iom::TensorView&, std::size_t, iom::oid,
            std::span<const std::size_t>, iom::TokenSelectorScratch) override {
        return 0;
    }
private:
    std::size_t* destructions_;
    iom::TokenSelectorScratchRequirements requirements_;
};

class SequenceSelector final : public iom::TokenSelector {
public:
    explicit SequenceSelector(std::vector<std::size_t> sequence)
        : sequence_(std::move(sequence)) {}

    [[nodiscard]] iom::TokenSelectorScratchRequirements scratch_requirements(
            const iom::TensorView&, std::size_t) const override {
        return {0, {0, 1}};
    }

    [[nodiscard]] std::size_t select(
            iom::DeviceOps&, const iom::TensorView&, std::size_t,
            iom::oid producer, std::span<const std::size_t> history,
            iom::TokenSelectorScratch) override {
        if (before_select) {
            before_select(calls);
        }
        ++calls;
        producers.push_back(producer);
        histories.emplace_back(history.begin(), history.end());
        if (throw_failure) {
            throw std::runtime_error("injected selector failure");
        }
        if (next >= sequence_.size()) {
            throw std::logic_error("injected selector sequence exhausted");
        }
        return sequence_[next++];
    }

    std::size_t calls = 0;
    std::vector<iom::oid> producers;
    std::vector<std::vector<std::size_t>> histories;
    bool throw_failure = false;
    // Optional case hook invoked with the number of selections already served,
    // so a case can act - for example arm a backend failure latch - at one
    // exact generation step.
    std::function<void(std::size_t)> before_select;

private:
    std::vector<std::size_t> sequence_;
    std::size_t next = 0;
};


// Use the existing bounded model owners to exercise genuine factory contract
// failures without allocating enormous tensors or requiring accelerator memory.
class ResourceProbeQueue final : public iom::DeviceOps {
public:
    explicit ResourceProbeQueue(const iom::Device& device) : DeviceOps(device) {}
};

class ResourceProbeDevice final : public iom::Device {
public:
    static constexpr std::array<iom::DataType, 2> kDefaultSupported{
            iom::DataType::BF16, iom::DataType::U32};

    std::size_t tensors = 0;
    std::size_t workspaces = 0;
    std::size_t queues = 0;
    mutable std::size_t capability_queries = 0;
    std::span<const iom::DataType> supported{kDefaultSupported};
    void* workspace_address = nullptr;
    std::size_t workspace_stride = 0;
    iom::Device* workspace_device = nullptr;
    std::optional<std::size_t> workspace_bytes;
    std::optional<iom::TensorSpec> wrong_tensor;
    iom::Device* queue_device = nullptr;

    iom::BackendKind backend_kind() const noexcept override {
        return iom::BackendKind::CPU;
    }
    std::uint32_t backend_device() const noexcept override { return 0; }
    std::span<const iom::DataType> supported_data_types() const noexcept override {
        ++capability_queries;
        return supported;
    }
    std::unique_ptr<iom::Tensor> create_tensor(const iom::TensorSpec& spec) override {
        ++tensors;
        return std::make_unique<iom_model_loading::FakeTensor>(
                wrong_tensor ? *wrong_tensor : spec, *this);
    }
    std::unique_ptr<iom::RawWorkspace> create_workspace(std::size_t bytes) override {
        ++workspaces;
        if (!workspace_address) throw std::invalid_argument("no device scratch");
        return std::make_unique<iom_model_loading::BoundedWorkspace>(
                workspace_device ? *workspace_device : *this,
                workspace_bytes.value_or(bytes),
                reinterpret_cast<void*>(
                        reinterpret_cast<std::uintptr_t>(workspace_address)
                        + (workspaces - 1) * workspace_stride));
    }
    std::unique_ptr<iom::DeviceOps> create_ops() override {
        ++queues;
        return std::make_unique<ResourceProbeQueue>(
                queue_device ? *queue_device : *this);
    }
};

void check_run_banks(const iom::session_detail::RunBanks& banks, std::size_t R) {
    const auto check = [](const iom::Tensor* tensor,
                          std::initializer_list<std::size_t> shape,
                          iom::DataType type = iom::DataType::BF16) {
        REQUIRE(tensor);
        CHECK(iom_model_loading::dims_of(tensor->view().spec().shape)
              == std::vector<std::size_t>(shape));
        CHECK(tensor->view().spec().data_type == type);
        CHECK(tensor->view().spec().quantization == iom::QuantizationFormat::NONE);
    };
    check(banks.token_indices.get(), {1, R}, iom::DataType::U32);
    const std::array feature_banks{
            banks.x.get(), banks.attention_norm.get(), banks.attention_merged.get(),
            banks.attention_output.get(), banks.residual_after_attention.get(),
            banks.mlp_norm.get(), banks.down.get(), banks.residual_after_mlp.get(),
            banks.final_norm.get()};
    for (const auto* tensor : feature_banks) check(tensor, {R, 8});
    for (const auto* tensor : {banks.gate.get(), banks.up.get(), banks.silu.get(),
                               banks.product.get()})
        check(tensor, {R, 12});
    for (const auto* tensor : {banks.q.get(), banks.rotated_q.get()})
        check(tensor, {4, R, 2});
    for (const auto* tensor : {banks.k.get(), banks.v.get(), banks.rotated_k.get()})
        check(tensor, {2, R, 2});
    const std::array owners{
            banks.token_indices.get(), banks.x.get(), banks.attention_norm.get(),
            banks.q.get(), banks.k.get(), banks.v.get(), banks.rotated_q.get(),
            banks.rotated_k.get(), banks.attention_merged.get(),
            banks.attention_output.get(), banks.residual_after_attention.get(),
            banks.mlp_norm.get(), banks.gate.get(), banks.up.get(), banks.silu.get(),
            banks.product.get(), banks.down.get(), banks.residual_after_mlp.get(),
            banks.final_norm.get()};
    for (std::size_t i = 0; i < owners.size(); ++i)
        for (std::size_t j = 0; j < i; ++j) CHECK(owners[i] != owners[j]);
}

// Deterministic supplied host clock of one attached observation recorder. The
// `exhausted` flag is the observable proof that no unexpected clock read
// happened: a session observation path that reads the clock without an
// instant to consume flips it instead of silently reading a wall clock.
struct RecordingClock {
    std::vector<std::uint64_t> instants{};
    std::size_t cursor = 0;
    bool exhausted = false;

    static std::uint64_t read(void* context) noexcept {
        auto* self = static_cast<RecordingClock*>(context);
        if (self->cursor >= self->instants.size()) {
            self->exhausted = true;
            return 0;
        }
        return self->instants[self->cursor++];
    }
};

[[nodiscard]] iom::HostClock make_clock(RecordingClock& recorder) {
    return iom::HostClock{&RecordingClock::read, &recorder};
}

// Deterministic supplied clock for the generation-path cases.
//
// The instrumented load consumes its fixed leading instants first.  Every later
// read returns the current window value, which the case advances at the
// generation observation points it owns: one window per request entry and one
// per selection boundary.  A pinned TTFT or decode-throughput delta therefore
// does not depend on how many reads the instrumented stages make inside one
// window - this leaf's commits and decode intervals, the prefill span, the
// per-facade enqueue sums, and the wait observations all share the same window.
// A read before the first window is armed is flagged instead of silently
// reading a wall clock.
struct ObservationWindowClock {
    std::vector<std::uint64_t> load_instants{};
    std::size_t load_cursor = 0;
    std::uint64_t window = 0;
    bool armed = false;
    bool unarmed_read = false;

    static std::uint64_t read(void* context) noexcept {
        auto* self = static_cast<ObservationWindowClock*>(context);
        if (self->load_cursor < self->load_instants.size()) {
            return self->load_instants[self->load_cursor++];
        }
        if (!self->armed) {
            self->unarmed_read = true;
        }
        return self->window;
    }

    // Enter a window at a generation observation point.
    void arm(std::uint64_t value) noexcept {
        window = value;
        armed = true;
    }
};

[[nodiscard]] iom::HostClock make_clock(ObservationWindowClock& recorder) {
    return iom::HostClock{&ObservationWindowClock::read, &recorder};
}

// Window script of one generation-path case: the case arms the window for each
// request entry with `enter`, and every selection advances to the next scripted
// window.  The served count stays observable so a case can pin how many
// selections and window steps it observed.
struct GenerationWindowScript {
    ObservationWindowClock& clock;
    std::vector<std::uint64_t> windows;
    std::size_t served = 0;

    void enter(std::uint64_t value) noexcept { clock.arm(value); }

    void on_select() noexcept {
        if (served < windows.size()) {
            clock.arm(windows[served]);
        }
        ++served;
    }
};

// Queue id 16, sequence `sequence`: one canonical positive accepted OID.
[[nodiscard]] iom::oid accepted_oid(std::uint64_t sequence) {
    return static_cast<iom::oid>((std::uint64_t{16} << 55) | sequence);
}

// The CPU post-acceptance cache append failure the accepted-failure fixtures
// inject: one armed latch is consumed by exactly the next accepted append, and
// the failure stays retained and repeatable through every later wait.
constexpr const char* kCacheAppendFailureMessage =
        "CPU cache append injected post-acceptance failure";

// One owned trace row of an observation recorder, or null when the recorder
// owns no row for that exact OID.
[[nodiscard]] const iom::InferenceTraceRecord* find_operation_record(
        const iom::InferenceMetrics& recorder, iom::oid operation) {
    for (const iom::InferenceTraceRecord& row : recorder.operations()) {
        if (row.operation == operation) return &row;
    }
    return nullptr;
}

// The copied operation context of one owned row: exactly the fields a wait
// observation must never rewrite. The wait outcome fields are deliberately
// excluded, so comparing them proves that an observed wait updates only its
// own exact positive OID's outcome and preserves the copied context.
void check_operation_context(
        const iom::InferenceTraceRecord& observed,
        const iom::InferenceTraceRecord& expected) {
    CHECK_EQ(observed.request_ordinal, expected.request_ordinal);
    CHECK_EQ(observed.operation, expected.operation);
    CHECK(observed.phase == expected.phase);
    CHECK(observed.decoder_layer == expected.decoder_layer);
    CHECK_EQ(observed.position_start, expected.position_start);
    CHECK_EQ(observed.run_length, expected.run_length);
    CHECK_EQ(observed.enqueue_begin_ns, expected.enqueue_begin_ns);
    CHECK_EQ(observed.enqueue_end_ns, expected.enqueue_end_ns);
}

// The message of one retained failure without consuming it: `exception_ptr`
// copies stay rethrowable, so the first retained exception is inspectable
// before and after its owning session is destroyed.
[[nodiscard]] std::string retained_failure_message(std::exception_ptr failure) {
    if (failure == nullptr) return std::string{"<none>"};
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return std::string{"<non-standard>"};
    }
}

// A generous deterministic instant sequence for an observed session. The two
// load instants come first; every observed wait then reads exactly one further
// instant, and each test asserts the read count it expects instead of relying
// on exhaustion.
[[nodiscard]] std::vector<std::uint64_t> wait_clock_instants() {
    std::vector<std::uint64_t> instants{10, 20};
    for (std::uint64_t step = 1; step <= 96; ++step) {
        instants.push_back(1'000 + step * 10);
    }
    return instants;
}

// Loads through the caller-supplied selector and returns the message of the
// failure the call must report, so an instrumented load's rethrown exception
// is comparable with the uninstrumented one.
[[nodiscard]] std::string load_failure_message(
        const std::filesystem::path& directory, iom::Device& device,
        std::unique_ptr<iom::TokenSelector> selector,
        iom::InferenceMetrics* metrics = nullptr) {
    try {
        static_cast<void>(iom::load_tinyllama_session(
                directory, device, std::move(selector), metrics));
    } catch (const std::exception& error) {
        return error.what();
    }
    REQUIRE_MESSAGE(false,
                    "expected a load failure for " << directory.string());
    return std::string();
}

// The one-layer off-tile configuration with a position count far outside any
// real context. No required weight shape depends on
// `max_position_embeddings`, and the bounded probe device allocates no cache
// storage, so a session whose accepted-OID bound or reservation cannot be
// represented is still loadable.
[[nodiscard]] nlohmann::json wide_position_config(
        nlohmann::json config, std::size_t positions) {
    config["max_position_embeddings"] = positions;
    return config;
}

}  // namespace

TEST_CASE("TinyLlama session resources expose one nonmovable owning boundary") {
    static_assert(!std::is_copy_constructible_v<iom::TinyLlamaSession>);
    static_assert(!std::is_copy_assignable_v<iom::TinyLlamaSession>);
    static_assert(!std::is_move_constructible_v<iom::TinyLlamaSession>);
    static_assert(!std::is_move_assignable_v<iom::TinyLlamaSession>);

    SessionFixture fixture;
    const std::unique_ptr<iom::TinyLlamaSession> session = fixture.load();

    REQUIRE(session);
    CHECK_EQ(&session->device(), fixture.device.get());
    CHECK_EQ(&session->queue().device(), fixture.device.get());
    CHECK_EQ(session->config().max_position_embeddings, 17);
    iom_model_loading::check_model_inventory(
            session->model(), iom_model_loading::expected_inventory(one_layer_config()));
    CHECK(dynamic_cast<iom::GreedyTokenSelector*>(&session->selector()) != nullptr);
    session->prepare_request(5, {0, 1}, {38, {0, 1}});
    check_run_banks(SessionAccess::prefill(*session), 5);
    check_run_banks(SessionAccess::decode(*session), 1);
    const auto caches = SessionAccess::caches(*session);
    REQUIRE_EQ(caches.size(), 1);
    CHECK(caches[0].key.get() != caches[0].value.get());
    for (const auto* tensor : {caches[0].key.get(), caches[0].value.get()}) {
        CHECK(iom_model_loading::dims_of(tensor->view().spec().shape)
              == std::vector<std::size_t>{2, 17, 2});
    }
    CHECK_EQ(caches[0].initialized_length, 0);
    CHECK(iom_model_loading::dims_of(SessionAccess::logits(*session).spec().shape)
          == std::vector<std::size_t>{1, 19});
    CHECK(session->request_length() == 5);
    CHECK_FALSE(session->poisoned());
}

TEST_CASE("TinyLlama session resources reject null selector before model work") {
    ResourceProbeDevice device;

    CHECK_THROWS_AS(
            iom::load_tinyllama_session(
                    std::filesystem::path("/definitely/missing/model"),
                    device, std::unique_ptr<iom::TokenSelector>{}),
            std::invalid_argument);
    CHECK_EQ(device.capability_queries, 0);
    CHECK_EQ(device.tensors, 0);
    CHECK_EQ(device.workspaces, 0);
    CHECK_EQ(device.queues, 0);
}

TEST_CASE("TinyLlama session resources retain an injected selector") {
    SessionFixture fixture;
    std::size_t destructions = 0;
    auto selector = std::make_unique<ProbeSelector>(&destructions);
    auto session =
            iom::load_tinyllama_session(
                    fixture.directory.path(), *fixture.device,
                    std::move(selector));

    session->prepare_request(
            1, iom::WorkspaceRequirements{0, 1},
            iom::TokenSelectorScratchRequirements{0, {0, 1}});
    CHECK_EQ(session->request_length(), 1);
    CHECK_EQ(destructions, 0);
    session.reset();
    CHECK_EQ(destructions, 1);
    CHECK_EQ(fixture.allocator.live, 0);
}

TEST_CASE("Inference metrics session load measures one instrumented success interval") {
    SessionFixture fixture;
    RecordingClock clock;
    clock.instants = {1'000, 1'750};
    iom::InferenceMetrics recorder(make_clock(clock));
    std::size_t destructions = 0;

    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<ProbeSelector>(&destructions), &recorder);

    REQUIRE(session);
    // Exactly the entry and the publication clock read: the caller-supplied
    // selector was constructed before the measured interval, and nothing else
    // in the factory reads the supplied clock.
    CHECK_EQ(clock.cursor, 2);
    CHECK_FALSE(clock.exhausted);
    CHECK(recorder.snapshot().load.state == iom::ObservationState::succeeded);
    CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 750);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 0);
    CHECK_FALSE(recorder.snapshot().request_admitted);
    CHECK_EQ(SessionAccess::metrics(*session), &recorder);
    CHECK_EQ(destructions, 0);

    // The recorder and its clock are borrowed, not owned: both stay intact and
    // usable through request preparation and the destructor drain, which
    // records nothing of its own.
    session->prepare_request(1, {0, 1}, {0, {0, 1}});
    const auto& banks = SessionAccess::prefill(*session);
    SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.copy(banks.x->view(), banks.attention_norm->view());
    });
    session.reset();
    CHECK_EQ(destructions, 1);
    CHECK_EQ(fixture.allocator.live, 0);
    CHECK_EQ(clock.cursor, 2);
    CHECK_FALSE(clock.exhausted);
    CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 750);
    CHECK(recorder.snapshot().request_admitted);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK(recorder.snapshot().attempt.outcome == iom::AttemptOutcome::none);

    // The two-argument factory and the selector overload without a recorder
    // publish sessions with no attached recorder and read no clock.
    CHECK(SessionAccess::metrics(*fixture.load()) == nullptr);
    const auto plain = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<ProbeSelector>(nullptr));
    CHECK(SessionAccess::metrics(*plain) == nullptr);
    CHECK_EQ(clock.cursor, 2);
    CHECK_FALSE(clock.exhausted);
}

TEST_CASE("Inference metrics session load records enabled failures and rethrows them") {
    // A null selector is still the rejected null-selector call: no capability
    // query and no owner, with the failure published to the recorder.
    {
        ResourceProbeDevice device;
        RecordingClock clock;
        clock.instants = {4, 9};
        iom::InferenceMetrics recorder(make_clock(clock));
        CHECK_THROWS_AS(
                iom::load_tinyllama_session(
                        std::filesystem::path("/definitely/missing/model"),
                        device, std::unique_ptr<iom::TokenSelector>{},
                        &recorder),
                std::invalid_argument);
        CHECK_EQ(device.capability_queries, 0);
        CHECK_EQ(device.tensors, 0);
        CHECK_EQ(device.workspaces, 0);
        CHECK_EQ(device.queues, 0);
        CHECK_EQ(clock.cursor, 2);
        CHECK_FALSE(clock.exhausted);
        CHECK(recorder.snapshot().load.state == iom::ObservationState::failed);
        // A failed load publishes no duration and no session.
        CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 0);
        CHECK_FALSE(recorder.snapshot().request_admitted);
        CHECK_EQ(recorder.snapshot().admitted.ordinal, 0);
    }

    // A device without the required storage capabilities is rejected before
    // any model work and still reports the original category.
    {
        SessionFixture fixture;
        ResourceProbeDevice device;
        device.supported = iom_model_loading::kSupportedWithoutBf16;
        RecordingClock clock;
        clock.instants = {11, 12};
        iom::InferenceMetrics recorder(make_clock(clock));
        std::size_t destructions = 0;
        CHECK_THROWS_AS(
                iom::load_tinyllama_session(
                        fixture.directory.path(), device,
                        std::make_unique<ProbeSelector>(&destructions),
                        &recorder),
                std::invalid_argument);
        CHECK_EQ(device.capability_queries, 1);
        CHECK_EQ(device.tensors, 0);
        CHECK_EQ(device.queues, 0);
        CHECK_EQ(destructions, 1);
        CHECK(recorder.snapshot().load.state == iom::ObservationState::failed);
        CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 0);
    }

    // A rejected configuration reports the same message with and without the
    // recorder, so an enabled failure rethrows the original exception
    // unchanged, and no owner survives it.
    {
        SessionFixture fixture;
        nlohmann::json rejected = one_layer_config();
        rejected["num_attention_heads"] = 3;
        write_config(fixture.directory.path(), rejected);
        RecordingClock clock;
        clock.instants = {20, 40};
        iom::InferenceMetrics recorder(make_clock(clock));
        const std::string without = load_failure_message(
                fixture.directory.path(), *fixture.device,
                std::make_unique<ProbeSelector>(nullptr));
        const std::string with = load_failure_message(
                fixture.directory.path(), *fixture.device,
                std::make_unique<ProbeSelector>(nullptr), &recorder);
        CHECK_EQ(with, without);
        CHECK_FALSE(with.empty());
        CHECK(recorder.snapshot().load.state == iom::ObservationState::failed);
        CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 0);
        // Only the instrumented call reads the supplied clock, exactly twice.
        CHECK_EQ(clock.cursor, 2);
        CHECK_FALSE(clock.exhausted);
        CHECK_EQ(fixture.allocator.live, 0);
    }

    // A setup allocation failure keeps its category and its cleanup, and is
    // published as a failed load with no duration.
    {
        SessionFixture fixture;
        RecordingClock clock;
        clock.instants = {5, 7, 5, 7};
        iom::InferenceMetrics recorder(make_clock(clock));
        for (const std::size_t after : {std::size_t{13}, std::size_t{33}}) {
            fixture.allocator.fail_after = fixture.allocator.allocations + after;
            CHECK_THROWS_AS(
                    iom::load_tinyllama_session(
                            fixture.directory.path(), *fixture.device,
                            std::make_unique<ProbeSelector>(nullptr),
                            &recorder),
                    std::bad_alloc);
            CHECK_EQ(fixture.allocator.live, 0);
            CHECK_EQ(fixture.allocator.resets, 0);
        }
        CHECK(recorder.snapshot().load.state == iom::ObservationState::failed);
        CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 0);
        CHECK_EQ(clock.cursor, 4);
        CHECK_FALSE(clock.exhausted);
    }
}

TEST_CASE("Inference metrics session load prepares trace storage before the first request") {
    SessionFixture fixture;
    RecordingClock clock;
    clock.instants = {10, 40};
    iom::InferenceMetrics recorder(make_clock(clock));
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<iom::GreedyTokenSelector>(), &recorder);
    REQUIRE(session);
    REQUIRE_EQ(SessionAccess::metrics(*session), &recorder);

    // Scalar observation requires no reservation at all.
    CHECK_FALSE(recorder.trace_prepared());
    CHECK(recorder.operations().empty());
    CHECK_EQ(recorder.trace_capacity_remaining(), 0);

    // The reservation reuses the request ledger's accepted-OID bound, so one
    // complete request always fits the prepared table.
    const std::size_t accepted_bound =
            session->config().max_position_embeddings
            * (session->config().num_hidden_layers * 24 + 32);
    const std::size_t allocations = fixture.allocator.allocations;
    session->prepare_operation_trace();

    CHECK(recorder.trace_prepared());
    CHECK(recorder.operations().empty());
    CHECK_GE(recorder.trace_capacity_remaining(), accepted_bound);
    // Host bookkeeping only: no device work and no clock read.
    CHECK_EQ(fixture.allocator.allocations, allocations);
    CHECK_EQ(clock.cursor, 2);
    CHECK_FALSE(clock.exhausted);

    // The request handoff then establishes the admitted observation context.
    session->prepare_request(1, {0, 1}, {38, {0, 1}});
    CHECK(recorder.snapshot().request_admitted);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
}

TEST_CASE("Inference metrics session load requires an attached recorder before the first request") {
    // An absent recorder is its own configuration failure: `load(path,
    // device, nullptr)` never means disabled instrumentation.
    {
        SessionFixture fixture;
        const auto session = fixture.load();
        CHECK_THROWS_AS(session->prepare_operation_trace(),
                        std::invalid_argument);
    }

    // Late configuration is rejected without partially enabling tracing and
    // without disturbing the published request.
    {
        SessionFixture fixture;
        RecordingClock clock;
        clock.instants = {3, 6};
        iom::InferenceMetrics recorder(make_clock(clock));
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device,
                std::make_unique<iom::GreedyTokenSelector>(), &recorder);
        REQUIRE(session);
        session->prepare_request(1, {0, 1}, {38, {0, 1}});
        CHECK_THROWS_AS(session->prepare_operation_trace(), std::logic_error);
        CHECK_FALSE(recorder.trace_prepared());
        CHECK(recorder.operations().empty());
        CHECK_EQ(recorder.trace_capacity_remaining(), 0);
        CHECK_EQ(session->request_length(), 1);
        CHECK(recorder.snapshot().request_admitted);
        CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
        CHECK_EQ(clock.cursor, 2);
        CHECK_FALSE(clock.exhausted);
    }
}

TEST_CASE("Inference metrics session load reports unrepresentable and unallocatable trace capacity") {
    // One third past the byte limit of the shared checked bound: the
    // reservation reports its own overflow failure before any inference work.
    // The single-plane off-tile geometry keeps its bf16 cache inside the
    // checked tiled bit count at this position count, so the failure is the
    // accepted-OID bound's own arithmetic and not a cache-size rejection.
    {
        SessionFixture fixture;
        const nlohmann::json base = iom_model_loading::h18_config();
        const std::size_t bound_bytes =
                std::numeric_limits<std::size_t>::max() / sizeof(iom::oid);
        write_config(
                fixture.directory.path(),
                wide_position_config(base, (bound_bytes / (24 + 32)) * 4 / 3));
        write_safetensors_file(fixture.directory.path(), "model.safetensors",
                               required_weight_entries(base));
        ResourceProbeDevice device;
        RecordingClock clock;
        clock.instants = {100, 220};
        iom::InferenceMetrics recorder(make_clock(clock));
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), device,
                std::make_unique<ProbeSelector>(nullptr), &recorder);
        REQUIRE(session);
        const std::size_t tensors = device.tensors;
        const std::size_t workspaces = device.workspaces;

        CHECK_THROWS_AS(session->prepare_operation_trace(),
                        std::overflow_error);
        CHECK_FALSE(recorder.trace_prepared());
        CHECK(recorder.operations().empty());
        CHECK_EQ(recorder.trace_capacity_remaining(), 0);
        // No device work and no clock read: the failure precedes inference.
        CHECK_EQ(device.tensors, tensors);
        CHECK_EQ(device.workspaces, workspaces);
        CHECK_EQ(device.queues, 1);
        CHECK_EQ(clock.cursor, 2);
        CHECK_FALSE(clock.exhausted);

        session.reset();
        CHECK_EQ(iom_model_loading::FixtureLiveness::shared().live_tensors, 0);
    }

    // A representable bound whose reservation cannot be satisfied reports the
    // allocation failure instead, with tracing still disabled.
    {
        SessionFixture fixture;
        write_config(fixture.directory.path(),
                     wide_position_config(
                             one_layer_config(),
                             std::numeric_limits<std::size_t>::max() >> 22));
        ResourceProbeDevice device;
        RecordingClock clock;
        clock.instants = {7, 9};
        iom::InferenceMetrics recorder(make_clock(clock));
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), device,
                std::make_unique<ProbeSelector>(nullptr), &recorder);
        REQUIRE(session);

        CHECK_THROWS_AS(session->prepare_operation_trace(), std::bad_alloc);
        CHECK_FALSE(recorder.trace_prepared());
        CHECK_EQ(recorder.trace_capacity_remaining(), 0);
        CHECK(recorder.operations().empty());
        CHECK_EQ(clock.cursor, 2);
        CHECK_FALSE(clock.exhausted);

        session.reset();
        CHECK_EQ(iom_model_loading::FixtureLiveness::shared().live_tensors, 0);
    }
}

TEST_CASE("Inference metrics session load publishes only the admitted request context") {
    SessionFixture fixture;
    RecordingClock clock;
    clock.instants = {1, 2};
    iom::InferenceMetrics recorder(make_clock(clock));
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<iom::GreedyTokenSelector>(), &recorder);
    REQUIRE(session);
    session->prepare_operation_trace();
    const std::size_t accepted_bound =
            session->config().max_position_embeddings
            * (session->config().num_hidden_layers * 24 + 32);

    // A direct request with no staged attempt establishes an empty admitted
    // context at the next ordinal.
    session->prepare_request(5, {0, 1}, {38, {0, 1}});
    CHECK(recorder.snapshot().request_admitted);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK_EQ(recorder.snapshot().attempt.ordinal, 1);
    CHECK(recorder.snapshot().attempt.outcome == iom::AttemptOutcome::none);
    CHECK_EQ(recorder.snapshot().admitted.prompt_tokens, 0);
    CHECK_FALSE(recorder.snapshot().admitted.stop_reason_valid);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, 0);

    // The rows and scalar spans belong to the admitted request until a later
    // successful publication replaces them.
    const iom::oid row = accepted_oid(1);
    recorder.record_enqueue(row, iom::InferencePhase::prefill, std::nullopt, 0,
                            5, 10, 30);
    recorder.record_tokenization(40, 90, iom::ObservationState::succeeded);
    REQUIRE_EQ(recorder.operations().size(), 1);
    CHECK_EQ(recorder.operations()[0].operation, row);

    // A failed candidate setup preserves the outgoing admitted request and its
    // rows and leaves the ordinal alone.
    fixture.allocator.fail_after = fixture.allocator.allocations + 3;
    CHECK_THROWS_AS(session->prepare_request(3, {0, 1}, {38, {0, 1}}),
                    std::bad_alloc);
    CHECK_EQ(session->request_length(), 5);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, 50);
    REQUIRE_EQ(recorder.operations().size(), 1);
    CHECK_EQ(recorder.operations()[0].operation, row);

    // Successful publication advances the ordinal, replaces the scalar spans,
    // and clears the rows without freeing the prepared capacity.
    fixture.allocator.fail_after = std::numeric_limits<std::size_t>::max();
    session->prepare_request(3, {0, 1}, {38, {0, 1}});
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 2);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, 0);
    CHECK(recorder.operations().empty());
    CHECK_GE(recorder.trace_capacity_remaining(), accepted_bound);

    // A staged attempt is published by the next successful handoff.
    recorder.begin_generation(70, 9);
    CHECK_EQ(recorder.snapshot().attempt.ordinal, 3);
    session->prepare_request(1, {0, 1}, {38, {0, 1}});
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 3);
    CHECK_EQ(recorder.snapshot().admitted.prompt_tokens, 9);
    CHECK_EQ(recorder.snapshot().attempt.ordinal, 3);
    CHECK(recorder.snapshot().attempt.outcome == iom::AttemptOutcome::none);
    CHECK_EQ(clock.cursor, 2);
    CHECK_FALSE(clock.exhausted);
}

TEST_CASE("Inference metrics session load preserves attribution after a poisoned failure") {
    SessionFixture fixture;
    RecordingClock clock;
    clock.instants = {1, 2};
    iom::InferenceMetrics recorder(make_clock(clock));
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<iom::GreedyTokenSelector>(), &recorder);
    REQUIRE(session);
    session->prepare_request(1, {0, 1}, {38, {0, 1}});
    const auto& banks = SessionAccess::prefill(*session);
    SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.copy(banks.x->view(), banks.attention_norm->view());
    });

    // A rejected synchronous submission poisons the session and keeps its
    // accepted OIDs. The recorder's outgoing admitted request and its scalar
    // spans survive the refusal, and the ordinal does not advance.
    CHECK_THROWS_AS(SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.copy(banks.x->view(), banks.gate->view());
    }), std::invalid_argument);
    CHECK(session->poisoned());
    recorder.record_tokenization(5, 25, iom::ObservationState::succeeded);
    CHECK_THROWS_AS(session->prepare_request(1, {0, 1}, {38, {0, 1}}),
                    std::logic_error);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, 20);
    CHECK_EQ(clock.cursor, 2);
    CHECK_FALSE(clock.exhausted);

    session.reset();
    CHECK_EQ(fixture.allocator.live, 0);
}

TEST_CASE("Inference metrics session load leaves generation behavior unchanged") {
    ForwardFixture fixture("metrics-session-generation", one_layer_config());
    const std::vector<std::size_t> prompt{0, 1};

    auto plain = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<SequenceSelector>(std::vector<std::size_t>{4, 2}));
    REQUIRE(plain);
    CHECK_EQ(SessionAccess::metrics(*plain), nullptr);
    const iom::TokenGenerationResult expected =
            plain->generate_tokens(prompt, 4);
    CHECK(expected.token_ids == std::vector<std::size_t>{4, 2});
    CHECK(expected.stop_reason == iom::GenerationStopReason::eos);

    ObservationWindowClock clock;
    clock.load_instants = {100, 200};
    iom::InferenceMetrics recorder(make_clock(clock));
    // Behavior parity and the observed outcomes only: the supplied clock
    // advances once per selection, so this case pins no generation instant.
    GenerationWindowScript script{clock, {21'000, 22'000}};
    auto selector =
            std::make_unique<SequenceSelector>(std::vector<std::size_t>{4, 2});
    selector->before_select = [&script](std::size_t) { script.on_select(); };
    auto observed = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector),
            &recorder);
    REQUIRE(observed);
    script.enter(20'000);
    const iom::TokenGenerationResult result =
            observed->generate_tokens(prompt, 4);
    CHECK(result.token_ids == expected.token_ids);
    CHECK(result.stop_reason == expected.stop_reason);
    CHECK_EQ(SessionAccess::caches(*observed)[0].initialized_length,
             SessionAccess::caches(*plain)[0].initialized_length);

    // The generation path observes exactly the admitted request it published:
    // its TTFT, its committed counts, and its single decode interval, while the
    // load interval keeps covering only the instrumented factory.
    const iom::InferenceSnapshot& snapshot = recorder.snapshot();
    REQUIRE(snapshot.request_admitted);
    CHECK_EQ(snapshot.load.host_nanoseconds, 100);
    CHECK_EQ(snapshot.admitted.ordinal, 1);
    CHECK_EQ(snapshot.admitted.prompt_tokens, 2);
    CHECK(snapshot.admitted.stop_reason_valid);
    CHECK(snapshot.admitted.stop_reason == iom::GenerationStopReason::eos);
    CHECK_EQ(snapshot.admitted.generated_tokens, 2);
    CHECK_EQ(snapshot.admitted.decode_forward_count, 1);
    CHECK_EQ(snapshot.admitted.decode_token_count, 1);
    CHECK(snapshot.admitted.time_to_first_token_valid);
    CHECK(snapshot.admitted.decode_throughput_valid);
    CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::succeeded);
    CHECK_EQ(script.served, 2);
    CHECK_FALSE(clock.unarmed_read);

    // A selector failure keeps its category and its poisoning with and without
    // an attached recorder, and is reported as its own failed attempt.
    const auto load_failing = [&](iom::InferenceMetrics* metrics) {
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{4});
        selector->throw_failure = true;
        return iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                metrics);
    };
    auto failing_plain = load_failing(nullptr);
    CHECK_THROWS_AS(failing_plain->generate_tokens(prompt, 2),
                    std::runtime_error);
    CHECK(failing_plain->poisoned());

    ObservationWindowClock failing_clock;
    failing_clock.load_instants = {30'000, 30'100};
    iom::InferenceMetrics failing_recorder(make_clock(failing_clock));
    auto failing_observed = load_failing(&failing_recorder);
    failing_clock.arm(31'000);
    CHECK_THROWS_AS(failing_observed->generate_tokens(prompt, 2),
                    std::runtime_error);
    CHECK(failing_observed->poisoned());
    const iom::InferenceSnapshot& failed = failing_recorder.snapshot();
    REQUIRE(failed.request_admitted);
    CHECK_EQ(failed.admitted.ordinal, 1);
    CHECK_EQ(failed.admitted.generated_tokens, 0);
    CHECK_FALSE(failed.admitted.stop_reason_valid);
    CHECK_FALSE(failed.admitted.decode_throughput_valid);
    CHECK(failed.admitted.failure != nullptr);
    CHECK(failed.attempt.outcome == iom::AttemptOutcome::failed);
    CHECK(failed.attempt.failure != nullptr);
    CHECK_FALSE(failing_clock.unarmed_read);
}


TEST_CASE("TinyLlama session resources preserve allocation failure cleanup") {
    SessionFixture fixture;
    // Fail after weights, in the cache pair and at the last decode bank.
    // Each partially constructed session must release every successful owner.
    for (const std::size_t after : {std::size_t{13}, std::size_t{33}}) {
        fixture.allocator.fail_after = fixture.allocator.allocations + after;
        CHECK_THROWS_AS(fixture.load(), std::bad_alloc);
        CHECK_EQ(fixture.allocator.live, 0);
        CHECK_EQ(fixture.allocator.resets, 0);
    }
}

TEST_CASE("TinyLlama session resources validate exact request bounds") {
    SessionFixture fixture;
    const std::unique_ptr<iom::TinyLlamaSession> session = fixture.load();
    const iom::WorkspaceRequirements operation{0, 1};
    const iom::TokenSelectorScratchRequirements selector{38, {0, 1}};
    const auto allocations = fixture.allocator.allocations;

    CHECK_THROWS_AS(session->prepare_request(0, operation, selector),
                    std::invalid_argument);
    CHECK_THROWS_AS(session->prepare_request(18, operation, selector),
                    std::invalid_argument);
    CHECK_EQ(fixture.allocator.allocations, allocations);

    session->prepare_request(17, operation, selector);
    CHECK_EQ(session->request_length(), 17);
    check_run_banks(SessionAccess::prefill(*session), 17);
    CHECK_FALSE(session->poisoned());

    session->prepare_request(1, operation, selector);
    CHECK_EQ(session->request_length(), 1);
    CHECK_FALSE(session->poisoned());
}

TEST_CASE("TinyLlama session resources reject guessed or malformed scratch") {
    SessionFixture fixture;
    const std::unique_ptr<iom::TinyLlamaSession> session = fixture.load();
    const iom::TokenSelectorScratchRequirements selector{38, {0, 1}};

    CHECK_THROWS_AS(
            session->prepare_request(
                    1, iom::WorkspaceRequirements{1, 16}, selector),
            std::invalid_argument);
    CHECK_THROWS_AS(
            session->prepare_request(
                    1, iom::WorkspaceRequirements{0, 0}, selector),
            std::invalid_argument);
    CHECK_THROWS_AS(
            session->prepare_request(
                    1, iom::WorkspaceRequirements{0, 1},
                    iom::TokenSelectorScratchRequirements{37, {0, 1}}),
            std::invalid_argument);
}

TEST_CASE("TinyLlama session resources reject host scratch overflow before request allocation") {
    SessionFixture fixture;
    const iom::TokenSelectorScratchRequirements requirements{
            std::numeric_limits<std::size_t>::max(), {0, 1}};
    const auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<ProbeSelector>(nullptr, requirements));
    const auto allocations = fixture.allocator.allocations;
    fixture.allocator.fail_after = allocations;
    CHECK_THROWS_AS(session->prepare_request(1, {0, 1}, requirements),
                    std::overflow_error);
    CHECK_EQ(fixture.allocator.allocations, allocations);
    CHECK_EQ(session->request_length(), 0);
}

TEST_CASE("TinyLlama session resources reject workspace overflow before allocation") {
    SessionFixture fixture;
    const auto session = fixture.load();
    const std::size_t allocations = fixture.allocator.allocations;
    fixture.allocator.fail_after = allocations;
    CHECK_THROWS_AS(
            session->prepare_request(
                    1, {std::numeric_limits<std::size_t>::max(), 32},
                    {38, {0, 1}}),
            std::overflow_error);
    CHECK_EQ(fixture.allocator.allocations, allocations);
    CHECK_EQ(session->request_length(), 0);
}


// Declared by the CPU driver in `src/cpu/queue.cpp` and by the SDPA port in
// `src/cpu/sdpa.cpp`: process-wide post-acceptance failure latches consumed
// inside an already accepted host task, before that task writes its logical
// output. The linear latch is the gate/up projection seam of the MLP stage and
// the SiLU latch is the existing activation seam of that later boundary; the
// SDPA and cache append latches are the attention and publication seams of the
// cache-attention stage, and the shared CPU conformance driver declares the
// same production seams, so no stage-specific fault API is introduced here.
namespace iom::cpu_detail {

void arm_linear_failure(std::size_t healthy_before, std::size_t failures) noexcept;
void clear_linear_failure() noexcept;
void arm_silu_failure() noexcept;
void clear_silu_failure() noexcept;
void arm_sdpa_failure() noexcept;
void clear_sdpa_failure() noexcept;
void arm_cache_append_failure() noexcept;
void clear_cache_append_failure() noexcept;
void arm_cache_append_wait_observation() noexcept;
void clear_cache_append_wait_observation() noexcept;
[[nodiscard]] std::size_t observed_cache_append_waits() noexcept;

}  // namespace iom::cpu_detail

TEST_CASE("Inference metrics generation measures TTFT and decode throughput") {
    ForwardFixture fixture("metrics-generation-throughput",
                           one_layer_config());
    const std::vector<std::size_t> prompt{0, 1};
    const std::vector<std::size_t> sequence{4, 5, 2};

    // The unobserved generation of the same request fixes the expected tokens,
    // stop reason, and cache transition.
    auto plain = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<SequenceSelector>(sequence));
    REQUIRE(plain);
    const iom::TokenGenerationResult expected =
            plain->generate_tokens(prompt, 4);
    CHECK(expected.token_ids == sequence);
    CHECK(expected.stop_reason == iom::GenerationStopReason::eos);

    ObservationWindowClock clock;
    clock.load_instants = {100, 200};
    iom::InferenceMetrics recorder(make_clock(clock));
    // One window per selection: the first covers the prefill-produced commit
    // and its decode interval, the following two close one decode interval
    // each.
    GenerationWindowScript script{clock, {1'200, 1'500, 1'900}};
    auto selector = std::make_unique<SequenceSelector>(sequence);
    selector->before_select = [&script](std::size_t) { script.on_select(); };
    auto observed = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector),
            &recorder);
    REQUIRE(observed);
    script.enter(1'000);
    const iom::TokenGenerationResult result =
            observed->generate_tokens(prompt, 4);
    CHECK(result.token_ids == expected.token_ids);
    CHECK(result.stop_reason == expected.stop_reason);
    CHECK_EQ(SessionAccess::caches(*observed)[0].initialized_length,
             SessionAccess::caches(*plain)[0].initialized_length);

    const iom::InferenceSnapshot& snapshot = recorder.snapshot();
    REQUIRE(snapshot.request_admitted);
    CHECK_EQ(snapshot.admitted.ordinal, 1);
    CHECK_EQ(snapshot.admitted.prompt_tokens, 2);
    CHECK(snapshot.admitted.stop_reason_valid);
    CHECK(snapshot.admitted.stop_reason == iom::GenerationStopReason::eos);
    CHECK_EQ(snapshot.admitted.generated_tokens, 3);
    CHECK_EQ(snapshot.admitted.decode_forward_count, 2);
    CHECK_EQ(snapshot.admitted.decode_token_count, 2);
    // TTFT starts at generation entry, before validation and setup, and ends
    // at the first commit; the prefill-produced token is not a decode token.
    CHECK(snapshot.admitted.time_to_first_token_valid);
    CHECK_EQ(snapshot.admitted.time_to_first_token_ns, 200);
    // The supplied clock advances only at a selection boundary, so a decode
    // forward's readiness observation reads the same window as its interval
    // start and the completion span is that window's zero elapsed time - which
    // also rules out a span that wrongly ended at the later commit instant.
    CHECK(snapshot.admitted.decode.state == iom::ObservationState::succeeded);
    CHECK_EQ(snapshot.admitted.decode.host_nanoseconds, 0);
    // The denominator sums both decode-start-to-commit intervals, which include
    // the synchronous selection time.
    CHECK(snapshot.admitted.decode_throughput_valid);
    CHECK_EQ(snapshot.admitted.decode_throughput_denominator_ns, 700);
    CHECK_EQ(snapshot.admitted.tokens_per_second,
             doctest::Approx(2.0 / (700.0 / 1'000'000'000.0)));
    CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::succeeded);
    // Every scripted window was consumed by exactly one selection.
    CHECK_EQ(script.served, 3);
    CHECK_FALSE(clock.unarmed_read);
}

TEST_CASE("Inference metrics generation counts a first-token EOS without decode work") {
    ForwardFixture fixture("metrics-generation-first-eos", one_layer_config());
    const std::vector<std::size_t> prompt{0, 1};
    const std::vector<std::size_t> sequence{2};

    auto plain = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<SequenceSelector>(sequence));
    REQUIRE(plain);
    const iom::TokenGenerationResult expected =
            plain->generate_tokens(prompt, 4);
    CHECK(expected.token_ids == sequence);
    CHECK(expected.stop_reason == iom::GenerationStopReason::eos);

    ObservationWindowClock clock;
    clock.load_instants = {100, 200};
    iom::InferenceMetrics recorder(make_clock(clock));
    GenerationWindowScript script{clock, {1'700}};
    auto selector = std::make_unique<SequenceSelector>(sequence);
    selector->before_select = [&script](std::size_t) { script.on_select(); };
    auto observed = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector),
            &recorder);
    REQUIRE(observed);
    script.enter(1'000);
    const iom::TokenGenerationResult result =
            observed->generate_tokens(prompt, 4);
    CHECK(result.token_ids == expected.token_ids);
    CHECK(result.stop_reason == expected.stop_reason);

    const iom::InferenceSnapshot& snapshot = recorder.snapshot();
    REQUIRE(snapshot.request_admitted);
    CHECK(snapshot.admitted.stop_reason_valid);
    CHECK(snapshot.admitted.stop_reason == iom::GenerationStopReason::eos);
    CHECK_EQ(snapshot.admitted.generated_tokens, 1);
    CHECK_EQ(snapshot.admitted.decode_forward_count, 0);
    CHECK_EQ(snapshot.admitted.decode_token_count, 0);
    CHECK(snapshot.admitted.time_to_first_token_valid);
    CHECK_EQ(snapshot.admitted.time_to_first_token_ns, 700);
    CHECK(snapshot.admitted.decode.state == iom::ObservationState::not_run);
    CHECK_EQ(snapshot.admitted.decode_throughput_denominator_ns, 0);
    CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
    CHECK_EQ(snapshot.admitted.tokens_per_second, 0.0);
    // A first-token EOS is counted without a decode forward or a cache append.
    CHECK_EQ(SessionAccess::caches(*observed)[0].initialized_length, 2);
    CHECK_EQ(SessionAccess::caches(*observed)[0].initialized_length,
             SessionAccess::caches(*plain)[0].initialized_length);
    CHECK_EQ(script.served, 1);
    CHECK_FALSE(clock.unarmed_read);
}

TEST_CASE("Inference metrics generation keeps terminal limit and context counting") {
    const std::vector<std::size_t> prompt{0, 1};
    const std::vector<std::size_t> sequence{4, 5, 6};

    SUBCASE("requested limit") {
        ForwardFixture fixture("metrics-generation-limit",
                               one_layer_config());
        ObservationWindowClock clock;
        clock.load_instants = {100, 200};
        iom::InferenceMetrics recorder(make_clock(clock));
        GenerationWindowScript script{clock, {1'200, 1'500}};
        auto selector = std::make_unique<SequenceSelector>(sequence);
        selector->before_select = [&script](std::size_t) { script.on_select(); };
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                &recorder);
        REQUIRE(session);
        script.enter(1'000);
        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 2);
        CHECK(result.token_ids == std::vector<std::size_t>{4, 5});
        CHECK(result.stop_reason == iom::GenerationStopReason::max_new_tokens);
        const iom::InferenceSnapshot& snapshot = recorder.snapshot();
        CHECK(snapshot.admitted.stop_reason_valid);
        CHECK(snapshot.admitted.stop_reason
              == iom::GenerationStopReason::max_new_tokens);
        CHECK_EQ(snapshot.admitted.generated_tokens, 2);
        CHECK_EQ(snapshot.admitted.decode_forward_count, 1);
        CHECK_EQ(snapshot.admitted.decode_token_count, 1);
        CHECK_EQ(snapshot.admitted.decode_throughput_denominator_ns, 300);
        CHECK(snapshot.admitted.decode_throughput_valid);
        CHECK_EQ(script.served, 2);
        CHECK_FALSE(clock.unarmed_read);
    }

    SUBCASE("context capacity") {
        nlohmann::json config = one_layer_config();
        config["max_position_embeddings"] = 4;
        ForwardFixture fixture("metrics-generation-context", config);
        ObservationWindowClock clock;
        clock.load_instants = {100, 200};
        iom::InferenceMetrics recorder(make_clock(clock));
        GenerationWindowScript script{clock, {1'200, 1'500}};
        auto selector = std::make_unique<SequenceSelector>(sequence);
        selector->before_select = [&script](std::size_t) { script.on_select(); };
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                &recorder);
        REQUIRE(session);
        script.enter(1'000);
        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 8);
        CHECK(result.token_ids == std::vector<std::size_t>{4, 5});
        CHECK(result.stop_reason
              == iom::GenerationStopReason::context_capacity);
        const iom::InferenceSnapshot& snapshot = recorder.snapshot();
        CHECK(snapshot.admitted.stop_reason
              == iom::GenerationStopReason::context_capacity);
        CHECK_EQ(snapshot.admitted.generated_tokens, 2);
        CHECK_EQ(snapshot.admitted.decode_forward_count, 1);
        CHECK_EQ(snapshot.admitted.decode_token_count, 1);
        // The terminal context token is counted without growing the cache.
        CHECK_EQ(SessionAccess::history(*session).size(), 4);
        CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length, 3);
        CHECK_EQ(script.served, 2);
        CHECK_FALSE(clock.unarmed_read);
    }
}

TEST_CASE("Inference metrics generation leaves zero-limit and zero-elapsed requests unrated") {
    const std::vector<std::size_t> prompt{0, 1};

    SUBCASE("zero limit") {
        ForwardFixture fixture("metrics-generation-zero-limit",
                               one_layer_config());
        ObservationWindowClock clock;
        clock.load_instants = {100, 200};
        iom::InferenceMetrics recorder(make_clock(clock));
        // A zero limit submits no selection at all, so no window is scripted.
        GenerationWindowScript script{clock, {}};
        auto selector =
                std::make_unique<SequenceSelector>(std::vector<std::size_t>{4});
        SequenceSelector* selector_observer = selector.get();
        selector->before_select = [&script](std::size_t) { script.on_select(); };
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                &recorder);
        REQUIRE(session);
        script.enter(1'000);
        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 0);
        CHECK(result.token_ids.empty());
        CHECK(result.stop_reason == iom::GenerationStopReason::max_new_tokens);
        const iom::InferenceSnapshot& snapshot = recorder.snapshot();
        REQUIRE(snapshot.request_admitted);
        CHECK_EQ(snapshot.admitted.prompt_tokens, 2);
        CHECK(snapshot.admitted.stop_reason_valid);
        CHECK(snapshot.admitted.stop_reason
              == iom::GenerationStopReason::max_new_tokens);
        CHECK_EQ(snapshot.admitted.generated_tokens, 0);
        CHECK_EQ(snapshot.admitted.decode_forward_count, 0);
        CHECK_EQ(snapshot.admitted.decode_token_count, 0);
        CHECK_FALSE(snapshot.admitted.time_to_first_token_valid);
        CHECK_EQ(snapshot.admitted.time_to_first_token_ns, 0);
        CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
        // The zero limit still provisions the prompt and submits no selection
        // work; the generation entry is the only added clock read.
        CHECK_EQ(selector_observer->calls, 0);
        CHECK_EQ(session->request_length(), 2);
        CHECK_FALSE(session->poisoned());
        CHECK_EQ(script.served, 0);
        CHECK_FALSE(clock.unarmed_read);
    }

    SUBCASE("zero elapsed time") {
        ForwardFixture fixture("metrics-generation-zero-elapsed",
                               one_layer_config());
        ObservationWindowClock clock;
        clock.load_instants = {5'000, 5'000};
        iom::InferenceMetrics recorder(make_clock(clock));
        GenerationWindowScript script{clock, {5'000, 5'000}};
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{4, 2});
        selector->before_select = [&script](std::size_t) { script.on_select(); };
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                &recorder);
        REQUIRE(session);
        script.enter(5'000);
        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 4);
        CHECK(result.token_ids == std::vector<std::size_t>{4, 2});
        const iom::InferenceSnapshot& snapshot = recorder.snapshot();
        CHECK(snapshot.admitted.time_to_first_token_valid);
        CHECK_EQ(snapshot.admitted.time_to_first_token_ns, 0);
        CHECK_EQ(snapshot.admitted.generated_tokens, 2);
        CHECK_EQ(snapshot.admitted.decode_forward_count, 1);
        CHECK_EQ(snapshot.admitted.decode_token_count, 1);
        CHECK(snapshot.admitted.decode.state == iom::ObservationState::succeeded);
        CHECK_EQ(snapshot.admitted.decode.host_nanoseconds, 0);
        // A zero denominator never fabricates a rate.
        CHECK_EQ(snapshot.admitted.decode_throughput_denominator_ns, 0);
        CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
        CHECK_EQ(snapshot.admitted.tokens_per_second, 0.0);
        CHECK_EQ(script.served, 2);
        CHECK_FALSE(clock.unarmed_read);
    }
}

TEST_CASE("Inference metrics generation refuses invalid selections after a completed decode") {
    const std::vector<std::size_t> prompt{0, 1};

    SUBCASE("out-of-range selection") {
        ForwardFixture fixture("metrics-generation-out-of-range",
                               one_layer_config());
        ObservationWindowClock clock;
        clock.load_instants = {100, 200};
        iom::InferenceMetrics recorder(make_clock(clock));
        GenerationWindowScript script{clock, {1'200, 1'300}};
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{std::size_t{4}, std::size_t{19}});
        selector->before_select = [&script](std::size_t) { script.on_select(); };
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                &recorder);
        REQUIRE(session);
        script.enter(1'000);
        CHECK_THROWS_AS(session->generate_tokens(prompt, 4),
                        std::invalid_argument);
        CHECK(session->poisoned());
        const iom::InferenceSnapshot& snapshot = recorder.snapshot();
        // The completed decode advanced only its completed-forward count; the
        // invalid selection committed nothing and disabled the overall rate.
        CHECK_EQ(snapshot.admitted.generated_tokens, 1);
        CHECK_EQ(snapshot.admitted.decode_forward_count, 1);
        CHECK_EQ(snapshot.admitted.decode_token_count, 0);
        CHECK(snapshot.admitted.decode.state == iom::ObservationState::succeeded);
        CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
        CHECK_EQ(snapshot.admitted.tokens_per_second, 0.0);
        CHECK(snapshot.admitted.failure != nullptr);
        CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::failed);
        CHECK(snapshot.attempt.failure != nullptr);
        CHECK_EQ(script.served, 2);
        CHECK_FALSE(clock.unarmed_read);
    }

    SUBCASE("selector failure") {
        ForwardFixture fixture("metrics-generation-selector-failure",
                               one_layer_config());
        ObservationWindowClock clock;
        clock.load_instants = {100, 200};
        iom::InferenceMetrics recorder(make_clock(clock));
        GenerationWindowScript script{clock, {1'200, 1'300}};
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{4, 5});
        SequenceSelector* selector_observer = selector.get();
        // The failure is injected at the second selection, after the decode
        // forward whose readiness was observed.
        selector_observer->before_select = [selector_observer, &script](
                                                   std::size_t served) {
            if (served == 1) {
                selector_observer->throw_failure = true;
            }
            script.on_select();
        };
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                &recorder);
        REQUIRE(session);
        script.enter(1'000);
        CHECK_THROWS_AS(session->generate_tokens(prompt, 4),
                        std::runtime_error);
        CHECK(session->poisoned());
        const iom::InferenceSnapshot& snapshot = recorder.snapshot();
        CHECK_EQ(snapshot.admitted.generated_tokens, 1);
        CHECK_EQ(snapshot.admitted.decode_forward_count, 1);
        CHECK_EQ(snapshot.admitted.decode_token_count, 0);
        CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
        CHECK(snapshot.admitted.failure != nullptr);
        CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::failed);
        CHECK(snapshot.attempt.failure != nullptr);
        CHECK_EQ(script.served, 2);
        CHECK_FALSE(clock.unarmed_read);
    }
}

TEST_CASE("Inference metrics generation records a failed decode phase and keeps counters") {
    ForwardFixture fixture("metrics-generation-decode-failure",
                           one_layer_config());
    const std::vector<std::size_t> prompt{0, 1};
    ObservationWindowClock clock;
    clock.load_instants = {100, 200};
    iom::InferenceMetrics recorder(make_clock(clock));
    GenerationWindowScript script{clock, {1'200, 1'500}};

    auto selector = std::make_unique<SequenceSelector>(
            std::vector<std::size_t>{4, 5, 6});
    // The one-shot CPU append latch fails the second decode forward after its
    // task was accepted, so the failure surfaces at that forward's existing
    // readiness wait.
    selector->before_select = [&script](std::size_t served) {
        if (served == 1) {
            iom::cpu_detail::arm_cache_append_failure();
        }
        script.on_select();
    };
    struct ClearLatch {
        ~ClearLatch() { iom::cpu_detail::clear_cache_append_failure(); }
    } clear_latch;
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector),
            &recorder);
    REQUIRE(session);
    script.enter(1'000);
    CHECK_THROWS_AS(session->generate_tokens(prompt, 4), std::runtime_error);
    CHECK(session->poisoned());

    const iom::InferenceSnapshot& snapshot = recorder.snapshot();
    // Completed work of the failed request stays inspectable; only the
    // unobserved decode forward is recorded as a failed phase.
    CHECK_EQ(snapshot.admitted.generated_tokens, 2);
    CHECK_EQ(snapshot.admitted.decode_forward_count, 1);
    CHECK_EQ(snapshot.admitted.decode_token_count, 1);
    CHECK(snapshot.admitted.decode.state == iom::ObservationState::failed);
    CHECK_EQ(snapshot.admitted.decode.host_nanoseconds, 0);
    CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
    CHECK(snapshot.admitted.failure != nullptr);
    CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::failed);
    CHECK(snapshot.attempt.failure != nullptr);
    CHECK_EQ(script.served, 2);
    CHECK_FALSE(clock.unarmed_read);

    session.reset();
    // The retained failing append keeps both operands in the device quarantine
    // until the device is released.
    CHECK_EQ(fixture.allocator.live, 2);
    fixture.device.reset();
    CHECK_EQ(fixture.allocator.live, 0);
    CHECK_EQ(fixture.allocator.resets, 0);
}

TEST_CASE("Inference metrics generation keeps outgoing attribution on a failed candidate") {
    const std::vector<std::size_t> prompt{0, 1};

    SUBCASE("validation rejection") {
        ForwardFixture fixture("metrics-generation-rejected-attempt",
                               one_layer_config());
        ObservationWindowClock clock;
        clock.load_instants = {100, 200};
        iom::InferenceMetrics recorder(make_clock(clock));
        GenerationWindowScript script{clock, {1'200, 2'900}};
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{4, 5});
        selector->before_select = [&script](std::size_t) { script.on_select(); };
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                &recorder);
        REQUIRE(session);
        session->prepare_operation_trace();
        script.enter(1'000);
        const iom::TokenGenerationResult first =
                session->generate_tokens(prompt, 1);
        CHECK(first.token_ids == std::vector<std::size_t>{4});
        const iom::oid row = accepted_oid(1);
        recorder.record_enqueue(row, iom::InferencePhase::prefill,
                                std::nullopt, 0, 2, 5, 9);
        REQUIRE_EQ(recorder.operations().size(), 1);

        // A rejected prompt stages its own attempt and consumes its entry
        // instant without publishing anything.
        const std::array<std::size_t, 0> empty{};
        CHECK_THROWS_AS(session->generate_tokens(empty, 1),
                        std::invalid_argument);
        CHECK_FALSE(session->poisoned());
        CHECK_EQ(session->request_length(), 2);
        const iom::InferenceSnapshot& rejected = recorder.snapshot();
        CHECK_EQ(rejected.attempt.ordinal, 2);
        CHECK(rejected.attempt.outcome == iom::AttemptOutcome::failed);
        CHECK(rejected.attempt.failure != nullptr);
        CHECK_EQ(rejected.attempt.prompt_tokens, 0);
        // The outgoing admitted request, its counters, and its rows survive.
        CHECK_EQ(rejected.admitted.ordinal, 1);
        CHECK_EQ(rejected.admitted.generated_tokens, 1);
        CHECK_EQ(rejected.admitted.time_to_first_token_ns, 200);
        CHECK(rejected.admitted.stop_reason_valid);
        CHECK(rejected.admitted.stop_reason
              == iom::GenerationStopReason::max_new_tokens);
        REQUIRE_EQ(recorder.operations().size(), 1);
        CHECK_EQ(recorder.operations()[0].operation, row);
        // The candidate consumed only its entry window and published nothing.
        CHECK_EQ(script.served, 1);
        CHECK_FALSE(clock.unarmed_read);

        // The next published request takes the next ordinal and resets the
        // request-scoped scalars and rows.
        script.enter(2'500);
        const iom::TokenGenerationResult second =
                session->generate_tokens(prompt, 1);
        CHECK(second.token_ids == std::vector<std::size_t>{5});
        const iom::InferenceSnapshot& published = recorder.snapshot();
        CHECK_EQ(published.admitted.ordinal, 2);
        CHECK_EQ(published.admitted.prompt_tokens, 2);
        CHECK_EQ(published.admitted.generated_tokens, 1);
        CHECK_EQ(published.admitted.time_to_first_token_ns, 400);
        CHECK(published.admitted.stop_reason
              == iom::GenerationStopReason::max_new_tokens);
        CHECK(published.attempt.outcome == iom::AttemptOutcome::succeeded);
        CHECK(recorder.operations().empty());
        // The next selection consumed the second scripted window.
        CHECK_EQ(script.served, 2);
        CHECK_FALSE(clock.unarmed_read);
    }

    SUBCASE("failed old drain") {
        ForwardFixture fixture("metrics-generation-failed-drain",
                               one_layer_config());
        ObservationWindowClock clock;
        clock.load_instants = {100, 200};
        iom::InferenceMetrics recorder(make_clock(clock));
        GenerationWindowScript script{clock, {1'500}};
        auto selector =
                std::make_unique<SequenceSelector>(std::vector<std::size_t>{4});
        selector->before_select = [&script](std::size_t) { script.on_select(); };
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector),
                &recorder);
        REQUIRE(session);
        session->prepare_operation_trace();
        script.enter(1'000);
        const iom::TokenGenerationResult first =
                session->generate_tokens(prompt, 1);
        CHECK(first.token_ids == std::vector<std::size_t>{4});
        const iom::oid row = accepted_oid(4);
        recorder.record_enqueue(row, iom::InferencePhase::decode, std::nullopt,
                                2, 1, 15, 45);
        REQUIRE_EQ(recorder.operations().size(), 1);

        // An accepted append fails while the candidate request drains it, so
        // the candidate fails before it publishes anything.
        const auto& banks = SessionAccess::prefill(*session);
        const auto caches = SessionAccess::caches(*session);
        std::vector<std::byte> bytes(banks.k->view().spec().logical_nbytes(),
                                     std::byte{0});
        banks.k->view().copy_from_host(bytes);
        banks.v->view().copy_from_host(bytes);
        struct ClearLatch {
            ~ClearLatch() { iom::cpu_detail::clear_cache_append_failure(); }
        } clear_latch;
        iom::cpu_detail::arm_cache_append_failure();
        SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
            return queue.cache_append(banks.k->view(), caches[0].key->view(),
                                      caches[0].initialized_length);
        });
        script.enter(2'000);
        CHECK_THROWS_AS(session->generate_tokens(prompt, 1),
                        std::runtime_error);
        CHECK(session->poisoned());

        const iom::InferenceSnapshot& failed = recorder.snapshot();
        CHECK_EQ(failed.attempt.ordinal, 2);
        CHECK(failed.attempt.outcome == iom::AttemptOutcome::failed);
        CHECK_EQ(failed.attempt.prompt_tokens, 2);
        CHECK(failed.attempt.failure != nullptr);
        // The failed old drain is attributed to the candidate attempt, never
        // to the admitted request, which keeps its counters and its rows.
        CHECK_EQ(failed.admitted.ordinal, 1);
        CHECK_EQ(failed.admitted.generated_tokens, 1);
        CHECK_EQ(failed.admitted.time_to_first_token_ns, 500);
        REQUIRE_EQ(recorder.operations().size(), 1);
        CHECK_EQ(recorder.operations()[0].operation, row);
        // The candidate consumed only its entry window and published nothing.
        CHECK_EQ(script.served, 1);
        CHECK_FALSE(clock.unarmed_read);

        session.reset();
        CHECK_EQ(fixture.allocator.live, 2);
        fixture.device.reset();
        CHECK_EQ(fixture.allocator.live, 0);
        CHECK_EQ(fixture.allocator.resets, 0);
    }
}

TEST_CASE("Inference metrics session load preserves attribution across a failed drain") {
    SessionFixture fixture;
    RecordingClock clock;
    clock.instants = wait_clock_instants();
    iom::InferenceMetrics recorder(make_clock(clock));
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<iom::GreedyTokenSelector>(), &recorder);
    REQUIRE(session);
    session->prepare_operation_trace();
    session->prepare_request(3, {0, 1}, {38, {0, 1}});
    const auto& banks = SessionAccess::prefill(*session);
    const auto caches = SessionAccess::caches(*session);
    std::vector<std::byte> bytes(banks.k->view().spec().logical_nbytes(),
                                 std::byte{0});
    banks.k->view().copy_from_host(bytes);
    banks.v->view().copy_from_host(bytes);
    recorder.record_tokenization(30, 70, iom::ObservationState::succeeded);
    const iom::oid row = accepted_oid(4);
    recorder.record_enqueue(row, iom::InferencePhase::prefill, std::nullopt, 0,
                            3, 15, 45);
    REQUIRE_EQ(recorder.operations().size(), 1);

    // The drain window below starts after the failing submission, so its
    // deltas cover exactly the existing drain waits.
    std::size_t drain_reads = 0;
    {
        // The process-wide CPU latch rejects the accepted append, so the
        // replacement's drain reports the queue's original failure instead of
        // publishing anything.
        struct ResetFailure {
            ~ResetFailure() { iom::cpu_detail::clear_cache_append_failure(); }
        } failure;
        iom::cpu_detail::arm_cache_append_failure();
        SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
            return queue.cache_append(banks.k->view(), caches[0].key->view(), 0);
        });
        drain_reads = clock.cursor;
        CHECK_THROWS_AS(session->prepare_request(1, {0, 1}, {38, {0, 1}}),
                        std::runtime_error);
    }
    // The replacement's drain observed exactly one existing wait: the failing
    // append's OID is a real accepted ledger entry whose wait observation reads
    // one host instant, while the unregistered preseeded row is never
    // fabricated into an additional record.
    CHECK_EQ(clock.cursor, drain_reads + 1);

    CHECK(session->poisoned());
    CHECK_EQ(session->request_length(), 3);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, 40);
    REQUIRE_EQ(recorder.operations().size(), 1);
    CHECK_EQ(recorder.operations()[0].operation, row);
    CHECK_FALSE(clock.exhausted);

    session.reset();
    // The destructor drain repeats that same wait and preserves both the
    // recorded row and its first observation state.
    CHECK_EQ(clock.cursor, drain_reads + 2);
    REQUIRE_EQ(recorder.operations().size(), 1);
    CHECK_EQ(recorder.operations()[0].operation, row);
    CHECK(recorder.operations()[0].wait_state == iom::WaitState::not_observed);
    // The CPU backend retains both operands of the failed append in its device
    // quarantine, so the session's own owners are released without resetting
    // the allocator.
    CHECK_EQ(fixture.allocator.live, 2);
    fixture.device.reset();
    CHECK_EQ(fixture.allocator.live, 0);
    CHECK_EQ(fixture.allocator.resets, 0);
}

// ---------------------------------------------------------------------------
// Wait observation cases.
//
// The recorder's wait hooks are the only observers of the existing
// session-controlled waits. These cases drive real accepted positive OIDs
// through the existing `SessionAccess` seam, preseed the already-owned trace
// rows with the recorder's core `record_enqueue` API and frozen copied
// context, and then exercise the real wait, repeated waits, an explicitly
// observed retained failure, the failure-driven drain, and destructor
// cleanup. They never require the enqueue-hook implementation: the preseeded
// row is exactly the already-owned record a duplicate acceptance ignores, and
// wait attribution and failure ownership come from the accepted-OID ledger,
// never from the presence of a row.
// ---------------------------------------------------------------------------

TEST_CASE("Inference metrics waits observe each accepted OID once across drains") {
    SessionFixture fixture;
    RecordingClock clock;
    clock.instants = wait_clock_instants();
    iom::InferenceMetrics recorder(make_clock(clock));
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<iom::GreedyTokenSelector>(), &recorder);
    REQUIRE(session);
    session->prepare_operation_trace();
    session->prepare_request(3, {0, 1}, {38, {0, 1}});
    REQUIRE(recorder.snapshot().request_admitted);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);

    // A completed decode observation gives the admitted request a valid rate,
    // so a retained wait failure's rate invalidation is observable against it.
    recorder.record_decode_interval(1'000);
    recorder.commit_token(2'000, true);
    recorder.end_generation(3'000, iom::GenerationStopReason::eos);
    REQUIRE(recorder.snapshot().admitted.decode_throughput_valid);

    const auto& banks = SessionAccess::prefill(*session);
    const auto caches = SessionAccess::caches(*session);
    std::vector<std::byte> zeros(banks.k->view().spec().logical_nbytes(),
                                 std::byte{0});
    banks.k->view().copy_from_host(zeros);
    banks.v->view().copy_from_host(zeros);

    // The first accepted append retains the CPU seam's post-acceptance
    // failure; the second accepted operation stays healthy. Both are real
    // positive OIDs in the accepted-OID ledger.
    struct ResetFailure {
        ~ResetFailure() { iom::cpu_detail::clear_cache_append_failure(); }
    } failure;
    iom::cpu_detail::arm_cache_append_failure();
    const iom::oid failed =
            SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
                return queue.cache_append(banks.k->view(),
                                          caches[0].key->view(), 0);
            });
    const iom::oid healthy =
            SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
                return queue.copy(banks.x->view(),
                                  banks.attention_norm->view());
            });
    REQUIRE(iom::oid_is_token(failed));
    REQUIRE(iom::oid_is_token(healthy));
    REQUIRE_EQ(SessionAccess::accepted(*session).size(), 2);
    CHECK(SessionAccess::accepted(*session)[0] == failed);
    CHECK(SessionAccess::accepted(*session)[1] == healthy);

    // The rows the wait hooks update are the already-owned core records of
    // those exact positive OIDs, carrying copied frozen context.
    recorder.record_enqueue(failed, iom::InferencePhase::decode, 4, 6, 1, 100,
                            140);
    recorder.record_enqueue(healthy, iom::InferencePhase::prefill,
                            std::nullopt, 0, 3, 200, 260);
    REQUIRE_EQ(recorder.operations().size(), 2);
    const iom::InferenceTraceRecord* const failed_before =
            find_operation_record(recorder, failed);
    const iom::InferenceTraceRecord* const healthy_before =
            find_operation_record(recorder, healthy);
    REQUIRE(failed_before != nullptr);
    REQUIRE(healthy_before != nullptr);
    CHECK_EQ(failed_before->request_ordinal, 1);
    CHECK_EQ(healthy_before->request_ordinal, 1);
    CHECK(failed_before->wait_state == iom::WaitState::not_observed);
    CHECK(healthy_before->wait_state == iom::WaitState::not_observed);
    const iom::InferenceTraceRecord failed_context = *failed_before;
    const iom::InferenceTraceRecord healthy_context = *healthy_before;

    // One successful observation: exactly one host instant is read and the
    // exact OID's first completion outcome is recorded without touching its
    // copied context.
    const std::size_t healthy_reads = clock.cursor;
    SessionAccess::wait(*session, healthy);
    CHECK_EQ(clock.cursor, healthy_reads + 1);
    const iom::InferenceTraceRecord* const healthy_after =
            find_operation_record(recorder, healthy);
    REQUIRE(healthy_after != nullptr);
    CHECK(healthy_after->wait_state == iom::WaitState::succeeded);
    CHECK_EQ(healthy_after->wait_observed_ns.value_or(0),
             clock.instants[healthy_reads]);
    check_operation_context(*healthy_after, healthy_context);
    REQUIRE_EQ(recorder.operations().size(), 2);

    // Repeated waits preserve the first successful observation: no row is
    // added and the first instant stays selected.
    SessionAccess::wait(*session, healthy);
    SessionAccess::wait(*session, healthy);
    REQUIRE_EQ(recorder.operations().size(), 2);
    const iom::InferenceTraceRecord* const healthy_repeat =
            find_operation_record(recorder, healthy);
    REQUIRE(healthy_repeat != nullptr);
    CHECK(healthy_repeat->wait_state == iom::WaitState::succeeded);
    CHECK_EQ(healthy_repeat->wait_observed_ns.value_or(0),
             clock.instants[healthy_reads]);
    check_operation_context(*healthy_repeat, healthy_context);

    // The failed wait is observed immediately at its catch: the original
    // exception is saved first, the exact OID retains it with its host
    // instant, and the existing poisoning and failure-driven drain follow
    // unchanged. This window contains three existing waits: the failing OID,
    // then the drain of the failing and the healthy ledger OIDs.
    const std::size_t failed_reads = clock.cursor;
    CHECK_THROWS_WITH_AS(SessionAccess::wait(*session, failed),
                         kCacheAppendFailureMessage, std::runtime_error);
    CHECK(session->poisoned());
    CHECK_EQ(clock.cursor, failed_reads + 3);
    REQUIRE_EQ(recorder.operations().size(), 2);
    const iom::InferenceTraceRecord* const failed_after =
            find_operation_record(recorder, failed);
    REQUIRE(failed_after != nullptr);
    CHECK(failed_after->wait_state == iom::WaitState::failed);
    CHECK_EQ(failed_after->wait_observed_ns.value_or(0),
             clock.instants[failed_reads]);
    CHECK_EQ(retained_failure_message(failed_after->wait_failure),
             kCacheAppendFailureMessage);
    check_operation_context(*failed_after, failed_context);

    // A later successful OID never certifies the earlier failed one, and the
    // healthy OID's first success survives the failure-driven drain.
    const iom::InferenceTraceRecord* const healthy_drained =
            find_operation_record(recorder, healthy);
    REQUIRE(healthy_drained != nullptr);
    CHECK(healthy_drained->wait_state == iom::WaitState::succeeded);
    CHECK_EQ(healthy_drained->wait_observed_ns.value_or(0),
             clock.instants[healthy_reads]);
    check_operation_context(*healthy_drained, healthy_context);

    // The retained wait failure invalidates the owning request's rate and
    // keeps every completed counter; it is no proof that the device failed.
    CHECK_FALSE(recorder.snapshot().admitted.decode_throughput_valid);
    CHECK_EQ(recorder.snapshot().admitted.tokens_per_second, 0.0);
    CHECK_EQ(recorder.snapshot().admitted.generated_tokens, 1);
    CHECK_EQ(recorder.snapshot().admitted.decode_token_count, 1);
    CHECK_EQ(recorder.snapshot().admitted.decode_throughput_denominator_ns,
             1'000);

    // A repeated wait of the failed OID rethrows the same retained exception
    // without adding a row, rewriting the first instant, or selecting another
    // error.
    const std::size_t repeat_reads = clock.cursor;
    CHECK_THROWS_WITH_AS(SessionAccess::wait(*session, failed),
                         kCacheAppendFailureMessage, std::runtime_error);
    CHECK_EQ(clock.cursor, repeat_reads + 3);
    REQUIRE_EQ(recorder.operations().size(), 2);
    const iom::InferenceTraceRecord* const failed_repeat =
            find_operation_record(recorder, failed);
    REQUIRE(failed_repeat != nullptr);
    CHECK(failed_repeat->wait_state == iom::WaitState::failed);
    CHECK_EQ(failed_repeat->wait_observed_ns.value_or(0),
             clock.instants[failed_reads]);
    CHECK_EQ(retained_failure_message(failed_repeat->wait_failure),
             kCacheAppendFailureMessage);

    // Destructor cleanup repeats the same two existing waits: both rows and
    // the first retained failure are preserved unchanged.
    const std::size_t destructor_reads = clock.cursor;
    session.reset();
    CHECK_EQ(clock.cursor, destructor_reads + 2);
    REQUIRE_EQ(recorder.operations().size(), 2);
    const iom::InferenceTraceRecord* const failed_final =
            find_operation_record(recorder, failed);
    REQUIRE(failed_final != nullptr);
    CHECK(failed_final->wait_state == iom::WaitState::failed);
    CHECK_EQ(failed_final->wait_observed_ns.value_or(0),
             clock.instants[failed_reads]);
    CHECK_EQ(retained_failure_message(failed_final->wait_failure),
             kCacheAppendFailureMessage);
    check_operation_context(*failed_final, failed_context);
    CHECK_FALSE(clock.exhausted);

    // The failed wait observation is not a device-terminality proof: the CPU
    // device keeps its own quarantine policy for the failed append.
    CHECK_EQ(fixture.allocator.live, 2);
    fixture.device.reset();
    CHECK_EQ(fixture.allocator.live, 0);
    CHECK_EQ(fixture.allocator.resets, 0);
}

TEST_CASE("Inference metrics waits never attribute unowned or scalar-only waits") {
    // Selector-internal work submitted directly to the exposed queue is
    // opaque: its real positive OID is never accepted by the ledger, so an
    // observed wait cannot fabricate a row, cannot rewrite the frozen context
    // of an existing row, and cannot invalidate the admitted request's rate
    // even though a trace row exists.
    {
        SessionFixture fixture;
        RecordingClock clock;
        clock.instants = wait_clock_instants();
        iom::InferenceMetrics recorder(make_clock(clock));
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device,
                std::make_unique<iom::GreedyTokenSelector>(), &recorder);
        REQUIRE(session);
        session->prepare_operation_trace();
        session->prepare_request(3, {0, 1}, {38, {0, 1}});
        const auto& banks = SessionAccess::prefill(*session);
        const auto caches = SessionAccess::caches(*session);
        std::vector<std::byte> zeros(banks.k->view().spec().logical_nbytes(),
                                     std::byte{0});
        banks.k->view().copy_from_host(zeros);

        recorder.record_decode_interval(1'000);
        recorder.commit_token(2'000, true);
        recorder.end_generation(3'000, iom::GenerationStopReason::eos);
        REQUIRE(recorder.snapshot().admitted.decode_throughput_valid);
        const double rate = recorder.snapshot().admitted.tokens_per_second;

        const iom::oid internal = session->queue().copy(
                banks.x->view(), banks.attention_norm->view());
        const iom::oid unrecorded = session->queue().copy(
                banks.x->view(), banks.attention_merged->view());
        REQUIRE(iom::oid_is_token(internal));
        REQUIRE(iom::oid_is_token(unrecorded));
        CHECK(SessionAccess::accepted(*session).empty());

        recorder.record_enqueue(internal, iom::InferencePhase::decode, 3, 9, 1,
                                21, 23);
        REQUIRE_EQ(recorder.operations().size(), 1);
        const iom::InferenceTraceRecord frozen = recorder.operations()[0];
        CHECK_EQ(frozen.operation, internal);

        const std::size_t internal_reads = clock.cursor;
        SessionAccess::wait(*session, internal);
        CHECK_EQ(clock.cursor, internal_reads + 1);
        const std::size_t unrecorded_reads = clock.cursor;
        SessionAccess::wait(*session, unrecorded);
        CHECK_EQ(clock.cursor, unrecorded_reads + 1);

        // The unrecorded OID acquired no row, and the owned row keeps exactly
        // the frozen context the test recorded for it.
        REQUIRE_EQ(recorder.operations().size(), 1);
        const iom::InferenceTraceRecord* const observed =
                find_operation_record(recorder, internal);
        REQUIRE(observed != nullptr);
        CHECK(observed->wait_state == iom::WaitState::succeeded);
        CHECK_EQ(observed->wait_observed_ns.value_or(0),
                 clock.instants[internal_reads]);
        check_operation_context(*observed, frozen);
        CHECK_EQ(observed->phase, iom::InferencePhase::decode);
        CHECK_EQ(observed->decoder_layer.value_or(0), 3);
        CHECK_EQ(observed->position_start, 9);
        CHECK_EQ(observed->run_length, 1);
        CHECK_EQ(observed->enqueue_begin_ns, 21);
        CHECK_EQ(observed->enqueue_end_ns, 23);

        // A successful unowned wait never touches the admitted request.
        CHECK(recorder.snapshot().admitted.decode_throughput_valid);
        CHECK_EQ(recorder.snapshot().admitted.tokens_per_second, rate);

        // A failed unowned wait keeps its own row's retained failure but never
        // invalidates the admitted request's rate: ownership comes from the
        // accepted-OID ledger, not from the presence of a trace row.
        struct ResetFailure {
            ~ResetFailure() { iom::cpu_detail::clear_cache_append_failure(); }
        } failure;
        iom::cpu_detail::arm_cache_append_failure();
        const iom::oid failed_internal = session->queue().cache_append(
                banks.k->view(), caches[0].key->view(), 0);
        REQUIRE(iom::oid_is_token(failed_internal));
        recorder.record_enqueue(failed_internal, iom::InferencePhase::decode,
                                3, 12, 1, 31, 33);
        REQUIRE_EQ(recorder.operations().size(), 2);
        const std::size_t failed_reads = clock.cursor;
        CHECK_THROWS_WITH_AS(SessionAccess::wait(*session, failed_internal),
                             kCacheAppendFailureMessage, std::runtime_error);
        // The direct wait observed one failure, and the failure-driven drain
        // had an empty accepted ledger to inspect.
        CHECK_EQ(clock.cursor, failed_reads + 1);
        REQUIRE_EQ(recorder.operations().size(), 2);
        const iom::InferenceTraceRecord* const failed_row =
                find_operation_record(recorder, failed_internal);
        REQUIRE(failed_row != nullptr);
        CHECK(failed_row->wait_state == iom::WaitState::failed);
        CHECK_EQ(failed_row->wait_observed_ns.value_or(0),
                 clock.instants[failed_reads]);
        CHECK_EQ(retained_failure_message(failed_row->wait_failure),
                 kCacheAppendFailureMessage);
        CHECK(recorder.snapshot().admitted.decode_throughput_valid);
        CHECK_EQ(recorder.snapshot().admitted.tokens_per_second, rate);

        session.reset();
        fixture.device.reset();
        CHECK_EQ(fixture.allocator.live, 0);
        CHECK_FALSE(clock.exhausted);
    }

    // Scalar-only mode: a successful wait adds neither a clock read nor a
    // record, while a failed wait of a verified owning request invalidates
    // that request's rate without a clock read, a row, or per-OID
    // registration.
    {
        SessionFixture fixture;
        RecordingClock clock;
        clock.instants = wait_clock_instants();
        iom::InferenceMetrics recorder(make_clock(clock));
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device,
                std::make_unique<iom::GreedyTokenSelector>(), &recorder);
        REQUIRE(session);
        REQUIRE_FALSE(recorder.trace_prepared());
        session->prepare_request(3, {0, 1}, {38, {0, 1}});
        const auto& banks = SessionAccess::prefill(*session);
        const auto caches = SessionAccess::caches(*session);
        std::vector<std::byte> zeros(banks.k->view().spec().logical_nbytes(),
                                     std::byte{0});
        banks.k->view().copy_from_host(zeros);

        recorder.record_decode_interval(1'000);
        recorder.commit_token(2'000, true);
        recorder.end_generation(3'000, iom::GenerationStopReason::eos);
        REQUIRE(recorder.snapshot().admitted.decode_throughput_valid);

        const iom::oid healthy =
                SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
                    return queue.copy(banks.x->view(),
                                      banks.attention_norm->view());
                });
        REQUIRE(iom::oid_is_token(healthy));
        // Scalar-only recording creates no per-OID storage at all.
        recorder.record_enqueue(healthy, iom::InferencePhase::prefill,
                                std::nullopt, 0, 3, 200, 260);
        const std::size_t healthy_reads = clock.cursor;
        SessionAccess::wait(*session, healthy);
        CHECK_EQ(clock.cursor, healthy_reads);
        CHECK(recorder.operations().empty());

        struct ResetFailure {
            ~ResetFailure() { iom::cpu_detail::clear_cache_append_failure(); }
        } failure;
        iom::cpu_detail::arm_cache_append_failure();
        const iom::oid failed =
                SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
                    return queue.cache_append(banks.k->view(),
                                              caches[0].key->view(), 0);
                });
        REQUIRE(iom::oid_is_token(failed));
        const std::size_t failed_reads = clock.cursor;
        CHECK_THROWS_WITH_AS(SessionAccess::wait(*session, failed),
                             kCacheAppendFailureMessage, std::runtime_error);
        CHECK_EQ(clock.cursor, failed_reads);
        CHECK(recorder.operations().empty());
        CHECK(session->poisoned());
        // The verified owning request loses its rate; completed counters stay.
        CHECK_FALSE(recorder.snapshot().admitted.decode_throughput_valid);
        CHECK_EQ(recorder.snapshot().admitted.tokens_per_second, 0.0);
        CHECK_EQ(recorder.snapshot().admitted.generated_tokens, 1);
        CHECK_EQ(recorder.snapshot().admitted.decode_token_count, 1);
        CHECK_EQ(recorder.snapshot().admitted.decode_throughput_denominator_ns,
                 1'000);

        // Destructor cleanup repeats the same scalar-only waits without
        // reading a clock or creating any row.
        const std::size_t destructor_reads = clock.cursor;
        session.reset();
        CHECK_EQ(clock.cursor, destructor_reads);
        CHECK(recorder.operations().empty());
        CHECK_EQ(fixture.allocator.live, 2);
        fixture.device.reset();
        CHECK_EQ(fixture.allocator.live, 0);
    }
}

TEST_CASE("Inference metrics waits preserve existing wait counts in both modes") {
    SessionFixture fixture;
    RecordingClock clock;
    clock.instants = wait_clock_instants();
    iom::InferenceMetrics recorder(make_clock(clock));
    auto plain = fixture.load();
    auto observed = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<iom::GreedyTokenSelector>(), &recorder);
    REQUIRE(plain);
    REQUIRE(observed);
    CHECK_EQ(SessionAccess::metrics(*plain), nullptr);
    CHECK_EQ(SessionAccess::metrics(*observed), &recorder);
    observed->prepare_operation_trace();
    plain->prepare_request(3, {0, 1}, {38, {0, 1}});
    observed->prepare_request(3, {0, 1}, {38, {0, 1}});

    struct Outcome {
        std::size_t accepted = 0;
        std::size_t waits = 0;
        std::string failure;
        bool poisoned = false;
        std::size_t initialized_length = 0;
    };
    const auto run = [](iom::TinyLlamaSession& session) {
        struct Reset {
            ~Reset() {
                iom::cpu_detail::clear_cache_append_failure();
                iom::cpu_detail::clear_cache_append_wait_observation();
            }
        } reset;
        iom::cpu_detail::arm_cache_append_failure();
        iom::cpu_detail::arm_cache_append_wait_observation();
        const auto& banks = SessionAccess::prefill(session);
        const auto caches = SessionAccess::caches(session);
        std::vector<std::byte> zeros(banks.k->view().spec().logical_nbytes(),
                                     std::byte{0});
        banks.k->view().copy_from_host(zeros);
        banks.v->view().copy_from_host(zeros);
        const iom::oid failed =
                SessionAccess::submit(session, [&](iom::DeviceOps& queue) {
                    return queue.cache_append(banks.k->view(),
                                              caches[0].key->view(), 0);
                });
        const iom::oid healthy =
                SessionAccess::submit(session, [&](iom::DeviceOps& queue) {
                    return queue.cache_append(banks.v->view(),
                                              caches[0].value->view(), 0);
                });
        Outcome outcome;
        outcome.accepted = SessionAccess::accepted(session).size();
        outcome.failure = "<none>";
        if (iom::oid_is_token(failed) && iom::oid_is_token(healthy)) {
            try {
                SessionAccess::wait(session, failed);
            } catch (const std::runtime_error& error) {
                outcome.failure = error.what();
            }
        }
        outcome.poisoned = session.poisoned();
        outcome.waits = iom::cpu_detail::observed_cache_append_waits();
        outcome.initialized_length = caches[0].initialized_length;
        return outcome;
    };

    const Outcome plain_outcome = run(*plain);
    const Outcome observed_outcome = run(*observed);

    // The existing waits, their counts, the retained error, poisoning, and the
    // published cache prefix are identical with and without the attached
    // recorder: the observation adds no wait of its own.
    CHECK_EQ(plain_outcome.accepted, 2);
    CHECK_EQ(observed_outcome.accepted, plain_outcome.accepted);
    CHECK_EQ(plain_outcome.waits, 1);
    CHECK_EQ(observed_outcome.waits, plain_outcome.waits);
    CHECK_EQ(plain_outcome.failure, kCacheAppendFailureMessage);
    CHECK_EQ(observed_outcome.failure, plain_outcome.failure);
    CHECK(plain_outcome.poisoned);
    CHECK_EQ(observed_outcome.poisoned, plain_outcome.poisoned);
    CHECK_EQ(plain_outcome.initialized_length, 0);
    CHECK_EQ(observed_outcome.initialized_length,
             plain_outcome.initialized_length);
    CHECK_FALSE(clock.exhausted);
}

TEST_CASE("TinyLlama session resources check all cache bytes before allocation") {
    SessionFixture fixture;
    auto config = one_layer_config();
    config["max_position_embeddings"] = std::numeric_limits<std::size_t>::max() / 16;
    write_config(fixture.directory.path(), config);
    ResourceProbeDevice device;
    std::size_t destructions = 0;
    CHECK_THROWS_AS(
            iom::load_tinyllama_session(
                    fixture.directory.path(), device,
                    std::make_unique<ProbeSelector>(&destructions)),
            std::overflow_error);
    // The sole canonical model loaded; no cache, bank, or queue was allocated.
    CHECK_EQ(device.tensors, 12);
    CHECK_EQ(device.queues, 0);
    CHECK_EQ(destructions, 1);
    CHECK_EQ(iom_model_loading::FixtureLiveness::shared().live_tensors, 0);
}

TEST_CASE("TinyLlama session resources reject foreign queues and wrong bank owners") {
    SessionFixture fixture;
    ResourceProbeDevice device;
    ResourceProbeDevice other;
    device.queue_device = &other;
    CHECK_THROWS_AS(
            iom::load_tinyllama_session(fixture.directory.path(), device),
            std::invalid_argument);
    CHECK_EQ(iom_model_loading::FixtureLiveness::shared().live_tensors, 0);
    device.queue_device = nullptr;
    auto session = iom::load_tinyllama_session(fixture.directory.path(), device);
    CHECK_EQ(device.queues, 2); // one rejected factory, one published session
    CHECK_EQ(device.tensors, 12 + 34);
    device.wrong_tensor = iom::TensorSpec{
            iom::TensorShape{{1, 1}}, iom::DataType::BF16};
    CHECK_THROWS_AS(session->prepare_request(3, {0, 1}, {38, {0, 1}}),
                    std::invalid_argument);
    CHECK_EQ(session->request_length(), 0);
    session.reset();
    CHECK_EQ(iom_model_loading::FixtureLiveness::shared().live_tensors, 0);
}

TEST_CASE("TinyLlama session resources provision exact separate scratch ranges") {
    SessionFixture fixture;
    ResourceProbeDevice device;
    alignas(128) std::array<std::byte, 1024> storage{};
    device.workspace_address = storage.data();
    device.workspace_stride = 256;
    const iom::TokenSelectorScratchRequirements requirements{13, {65, 32}};
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), device,
            std::make_unique<ProbeSelector>(nullptr, requirements));
    session->prepare_request(3, {129, 128}, requirements);
    const auto operation = SessionAccess::workspace(*session);
    const auto scratch = SessionAccess::selector_scratch(*session);
    CHECK_EQ(operation.byte_size(), 129);
    CHECK_EQ(scratch.host.size(), 13);
    CHECK_EQ(scratch.device.byte_size(), 65);
    CHECK(operation.owner_identity() != scratch.device.owner_identity());
    CHECK_EQ(device.workspaces, 2);
    session.reset();
    CHECK_EQ(iom_model_loading::FixtureLiveness::shared().live_workspaces, 0);
}

TEST_CASE("TinyLlama session resources validate supplied workspace ownership and ranges") {
    SessionFixture fixture;
    ResourceProbeDevice device;
    ResourceProbeDevice other;
    alignas(128) std::array<std::byte, 1024> storage{};
    device.workspace_address = storage.data();
    auto session = iom::load_tinyllama_session(fixture.directory.path(), device);
    session->prepare_request(1, {0, 1}, {38, {0, 1}});
    REQUIRE(SessionAccess::workspace(*session).empty());
    const auto* bank = SessionAccess::prefill(*session).x.get();

    SUBCASE("insufficient base alignment") {
        device.workspace_address = storage.data() + 1;
        CHECK_THROWS_AS(session->prepare_request(3, {64, 32}, {38, {0, 1}}),
                        std::invalid_argument);
    }
    SUBCASE("requirement exceeds the base guarantee") {
        device.workspace_address = storage.data() + 32;
        CHECK_THROWS_AS(session->prepare_request(3, {64, 128}, {38, {0, 1}}),
                        std::invalid_argument);
    }
    SUBCASE("foreign device owner") {
        device.workspace_device = &other;
        CHECK_THROWS_AS(session->prepare_request(3, {64, 32}, {38, {0, 1}}),
                        std::invalid_argument);
    }
    SUBCASE("inexact capacity") {
        device.workspace_bytes = 32;
        CHECK_THROWS_AS(session->prepare_request(3, {64, 32}, {38, {0, 1}}),
                        std::invalid_argument);
    }
    SUBCASE("address range overflow") {
        device.workspace_address = reinterpret_cast<void*>(
                std::numeric_limits<std::uintptr_t>::max() - 31);
        CHECK_THROWS_AS(session->prepare_request(3, {64, 32}, {38, {0, 1}}),
                        std::overflow_error);
    }
    SUBCASE("workspace overlaps canonical weight storage") {
        const auto weight = reinterpret_cast<std::uintptr_t>(
                session->model().weight(0).native_handle());
        device.workspace_address = reinterpret_cast<void*>(weight & ~std::uintptr_t{31});
        CHECK_THROWS_AS(session->prepare_request(3, {64, 32}, {38, {0, 1}}),
                        std::invalid_argument);
    }
    SUBCASE("backend rejects positive scratch") {
        device.workspace_address = nullptr;
        CHECK_THROWS_AS(session->prepare_request(3, {64, 32}, {38, {0, 1}}),
                        std::invalid_argument);
    }
    CHECK_EQ(session->request_length(), 1);
    CHECK(SessionAccess::prefill(*session).x.get() == bank);
    CHECK_EQ(iom_model_loading::FixtureLiveness::shared().live_workspaces, 0);
}

TEST_CASE("TinyLlama session resources reject overlapping operation and selector scratch") {
    SessionFixture fixture;
    ResourceProbeDevice device;
    alignas(32) std::array<std::byte, 512> storage{};
    device.workspace_address = storage.data();
    const iom::TokenSelectorScratchRequirements requirements{13, {64, 32}};
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), device,
            std::make_unique<ProbeSelector>(nullptr, requirements));
    CHECK_THROWS_AS(session->prepare_request(3, {64, 32}, requirements),
                    std::invalid_argument);
    CHECK_EQ(session->request_length(), 0);
    CHECK_EQ(iom_model_loading::FixtureLiveness::shared().live_workspaces, 0);
}

TEST_CASE("TinyLlama session resources keep every layer cache independent across requests") {
    SessionFixture fixture;
    const auto config = iom_model_loading::two_layer_config();
    write_config(fixture.directory.path(), config);
    write_safetensors_file(fixture.directory.path(), "model.safetensors",
                           required_weight_entries(config));
    auto session = fixture.load();
    session->prepare_request(17, {0, 1}, {38, {0, 1}});
    auto caches = SessionAccess::caches(*session);
    REQUIRE_EQ(caches.size(), 2);
    const std::array owners{caches[0].key.get(), caches[0].value.get(),
                            caches[1].key.get(), caches[1].value.get()};
    for (std::size_t i = 0; i < owners.size(); ++i)
        for (std::size_t j = 0; j < i; ++j) CHECK(owners[i] != owners[j]);
    for (auto& cache : caches) cache.initialized_length = 17;
    session->prepare_request(1, {0, 1}, {38, {0, 1}});
    CHECK(caches[0].key.get() == owners[0]);
    CHECK(caches[0].value.get() == owners[1]);
    CHECK(caches[1].key.get() == owners[2]);
    CHECK(caches[1].value.get() == owners[3]);
    for (const auto& cache : caches) CHECK_EQ(cache.initialized_length, 0);
}

TEST_CASE("TinyLlama session resources preserve the old request on partial allocation failure") {
    SessionFixture fixture;
    auto session = fixture.load();
    session->prepare_request(5, {0, 1}, {38, {0, 1}});
    const auto* bank = SessionAccess::prefill(*session).x.get();
    const auto* decode = SessionAccess::decode(*session).x.get();
    const auto* cache = SessionAccess::caches(*session)[0].key.get();
    SessionAccess::history(*session).push_back(7);
    SessionAccess::results(*session).push_back(8);
    SessionAccess::caches(*session)[0].initialized_length = 5;
    const std::size_t live = fixture.allocator.live;
    fixture.allocator.fail_after = fixture.allocator.allocations + 3;
    CHECK_THROWS_AS(session->prepare_request(3, {0, 1}, {38, {0, 1}}),
                    std::bad_alloc);
    CHECK_EQ(fixture.allocator.live, live);
    CHECK_EQ(session->request_length(), 5);
    CHECK(SessionAccess::prefill(*session).x.get() == bank);
    CHECK(SessionAccess::history(*session) == std::vector<std::size_t>{7});
    CHECK(SessionAccess::results(*session) == std::vector<std::size_t>{8});
    CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length, 5);

    fixture.allocator.fail_after = std::numeric_limits<std::size_t>::max();
    session->prepare_request(3, {0, 1}, {38, {0, 1}});
    CHECK(SessionAccess::prefill(*session).x.get() != bank);
    CHECK(SessionAccess::decode(*session).x.get() == decode);
    CHECK(SessionAccess::caches(*session)[0].key.get() == cache);
    CHECK(SessionAccess::history(*session).empty());
    CHECK(SessionAccess::results(*session).empty());
    CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length, 0);
    CHECK_EQ(fixture.allocator.resets, 0);
}

TEST_CASE("TinyLlama session resources drain both cache writers before different R replacement") {
    SessionFixture fixture;
    auto session = fixture.load();
    session->prepare_request(5, {0, 1}, {38, {0, 1}});
    const auto& banks = SessionAccess::prefill(*session);
    auto caches = SessionAccess::caches(*session);
    std::vector<std::byte> bytes(banks.k->view().spec().logical_nbytes(), std::byte{0});
    banks.k->view().copy_from_host(bytes);
    banks.v->view().copy_from_host(bytes);
    iom::cpu_detail::arm_cache_append_wait_observation();
    struct ResetObservation {
        ~ResetObservation() { iom::cpu_detail::clear_cache_append_wait_observation(); }
    } observation;
    SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.cache_append(banks.k->view(), caches[0].key->view(), 0);
    });
    SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.cache_append(banks.v->view(), caches[0].value->view(), 0);
    });
    CHECK_EQ(iom::cpu_detail::observed_cache_append_waits(), 0);
    session->prepare_request(3, {0, 1}, {38, {0, 1}});
    CHECK_EQ(iom::cpu_detail::observed_cache_append_waits(), 2);
    CHECK(SessionAccess::accepted(*session).empty());
    CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length, 0);
    check_run_banks(SessionAccess::prefill(*session), 3);
}

TEST_CASE("TinyLlama session resources retain accepted failures and drain before destruction") {
    SessionFixture fixture;
    auto session = fixture.load();
    session->prepare_request(3, {0, 1}, {38, {0, 1}});
    const auto& banks = SessionAccess::prefill(*session);
    auto caches = SessionAccess::caches(*session);
    std::vector<std::byte> bytes(banks.k->view().spec().logical_nbytes(), std::byte{0});
    banks.k->view().copy_from_host(bytes);
    banks.v->view().copy_from_host(bytes);
    iom::cpu_detail::arm_cache_append_wait_observation();
    iom::cpu_detail::arm_cache_append_failure();
    struct ResetFailure {
        ~ResetFailure() {
            iom::cpu_detail::clear_cache_append_failure();
            iom::cpu_detail::clear_cache_append_wait_observation();
        }
    } failure;
    const auto failed = SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.cache_append(banks.k->view(), caches[0].key->view(), 0);
    });
    const auto healthy = SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.cache_append(banks.v->view(), caches[0].value->view(), 0);
    });
    SUBCASE("observed failure poisons dependent work and retains every OID") {
        CHECK_THROWS_AS(SessionAccess::wait(*session, failed), std::runtime_error);
        CHECK_EQ(iom::cpu_detail::observed_cache_append_waits(), 1);
        CHECK(session->poisoned());
        REQUIRE_EQ(SessionAccess::accepted(*session).size(), 2);
        CHECK(SessionAccess::accepted(*session)[0] == failed);
        CHECK(SessionAccess::accepted(*session)[1] == healthy);
        const auto allocations = fixture.allocator.allocations;
        CHECK_THROWS_AS(session->prepare_request(1, {0, 1}, {38, {0, 1}}),
                        std::logic_error);
        bool submitted = false;
        CHECK_THROWS_AS(SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
            submitted = true;
            return queue.copy(banks.k->view(), banks.v->view());
        }), std::logic_error);
        CHECK_FALSE(submitted);
        CHECK_EQ(fixture.allocator.allocations, allocations);
        CHECK_THROWS_AS(SessionAccess::wait(*session, failed), std::runtime_error);
    }
    SUBCASE("destruction observes failures without an earlier caller wait") {
        CHECK_EQ(iom::cpu_detail::observed_cache_append_waits(), 0);
    }
    session.reset();
    CHECK_EQ(iom::cpu_detail::observed_cache_append_waits(), 1);
    // CPU retains both operands of the failed append in its device quarantine;
    // session teardown must not bypass that policy or reset the allocator.
    CHECK_EQ(fixture.allocator.live, 2);
    fixture.device.reset();
    CHECK_EQ(fixture.allocator.live, 0);
    CHECK_EQ(fixture.allocator.resets, 0);
}

TEST_CASE("TinyLlama session resources reject synchronous admission without losing prior work") {
    SessionFixture fixture;
    auto session = fixture.load();
    session->prepare_request(1, {0, 1}, {38, {0, 1}});
    const auto& banks = SessionAccess::prefill(*session);
    const auto accepted = SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.copy(banks.x->view(), banks.attention_norm->view());
    });
    CHECK_THROWS_AS(SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
        return queue.copy(banks.x->view(), banks.gate->view());
    }), std::invalid_argument);
    CHECK(session->poisoned());
    REQUIRE_EQ(SessionAccess::accepted(*session).size(), 1);
    CHECK_EQ(SessionAccess::accepted(*session)[0], accepted);
    session.reset();
    CHECK_EQ(fixture.allocator.live, 0);
}

TEST_CASE("TinyLlama session resources retain context sized buffers without decode allocation") {
    SessionFixture fixture;
    auto session = fixture.load();
    session->prepare_request(1, {0, 1}, {38, {0, 1}});
    const auto& banks = SessionAccess::decode(*session);
    auto& history = SessionAccess::history(*session);
    auto& results = SessionAccess::results(*session);
    const auto* history_data = history.data();
    const auto* result_data = results.data();
    const auto* oid_data = SessionAccess::accepted(*session).data();
    const auto allocations = fixture.allocator.allocations;
    const auto scratch = SessionAccess::selector_scratch(*session);
    for (std::size_t row = 0; row < session->config().max_position_embeddings; ++row) {
        history.push_back(row);
        results.push_back(row + 1);
        for (std::size_t operation = 0; operation < 24; ++operation) {
            const auto token = SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
                return queue.copy(banks.x->view(), banks.attention_norm->view());
            });
            SessionAccess::wait(*session, token);
        }
    }
    CHECK_EQ(history.back(), 16);
    CHECK_EQ(results.back(), 17);
    CHECK(history.data() == history_data);
    CHECK(results.data() == result_data);
    CHECK(SessionAccess::accepted(*session).data() == oid_data);
    CHECK(SessionAccess::selector_scratch(*session).host.data() == scratch.host.data());
    CHECK_EQ(fixture.allocator.allocations, allocations);
}

TEST_CASE("TinyLlama full model logits execute every configured layer") {
    ForwardFixture one_fixture("one-layer", one_layer_config());
    ForwardFixture two_fixture("two-layer", two_layer_config());
    auto one = one_fixture.load();
    auto two = two_fixture.load();

    SessionAccess::prepare_forward_request(*one, 2);
    SessionAccess::prepare_forward_request(*two, 2);
    const std::array<std::size_t, 2> prompt{0, 1};
    const auto one_result = SessionAccess::forward_prefill(*one, prompt);
    const auto two_result = SessionAccess::forward_prefill(*two, prompt);
    REQUIRE(iom::oid_is_token(one_result.producer));
    REQUIRE(iom::oid_is_token(two_result.producer));
    REQUIRE(one_result.logits);
    REQUIRE(two_result.logits);
    SessionAccess::wait(*one, one_result.producer);
    SessionAccess::wait(*two, two_result.producer);

    CHECK(iom_model_loading::dims_of(one_result.logits->spec().shape)
          == std::vector<std::size_t>{1, 19});
    CHECK(iom_model_loading::dims_of(two_result.logits->spec().shape)
          == std::vector<std::size_t>{1, 19});
    CHECK(one_result.logits == &SessionAccess::logits(*one));
    CHECK(two_result.logits == &SessionAccess::logits(*two));
    const auto one_logits = read_forward_tensor(*one_result.logits);
    const auto two_logits = read_forward_tensor(*two_result.logits);
    REQUIRE_EQ(one_logits.size(), 19);
    REQUIRE_EQ(two_logits.size(), 19);
    CHECK(std::all_of(one_logits.begin(), one_logits.end(),
                      [](float value) { return std::isfinite(value); }));
    CHECK(std::all_of(two_logits.begin(), two_logits.end(),
                      [](float value) { return std::isfinite(value); }));
    bool layer_changed_output = false;
    for (std::size_t index = 0; index < one_logits.size(); ++index) {
        if (std::abs(one_logits[index] - two_logits[index]) > 0.01F) {
            layer_changed_output = true;
            break;
        }
    }
    CHECK(layer_changed_output);
    CHECK_EQ(SessionAccess::caches(*one)[0].initialized_length, 2);
    CHECK_EQ(SessionAccess::caches(*two)[0].initialized_length, 2);
    REQUIRE_EQ(SessionAccess::caches(*two).size(), 2);
    CHECK_EQ(SessionAccess::caches(*two)[1].initialized_length, 2);
}

TEST_CASE("TinyLlama full model logits match cached decode at the same row") {
    ForwardFixture fixture("parity", two_layer_config());
    auto direct = fixture.load();
    auto cached = fixture.load();
    SessionAccess::prepare_forward_request(*direct, 2);
    SessionAccess::prepare_forward_request(*cached, 1);

    const std::array<std::size_t, 2> full_prompt{0, 1};
    const std::array<std::size_t, 1> prefix{0};
    const auto direct_result = SessionAccess::forward_prefill(
            *direct, full_prompt);
    const auto prefix_result = SessionAccess::forward_prefill(*cached, prefix);
    REQUIRE(iom::oid_is_token(direct_result.producer));
    REQUIRE(iom::oid_is_token(prefix_result.producer));
    SessionAccess::wait(*direct, direct_result.producer);
    SessionAccess::wait(*cached, prefix_result.producer);

    const auto decode_result = SessionAccess::forward_decode(*cached, 1);
    REQUIRE(iom::oid_is_token(decode_result.producer));
    REQUIRE(decode_result.logits);
    SessionAccess::wait(*cached, decode_result.producer);
    const auto direct_logits = read_forward_tensor(*direct_result.logits);
    const auto decode_logits = read_forward_tensor(*decode_result.logits);
    REQUIRE_EQ(direct_logits.size(), decode_logits.size());
    for (std::size_t index = 0; index < direct_logits.size(); ++index) {
        CHECK_MESSAGE(
                std::abs(direct_logits[index] - decode_logits[index]) < 0.02F,
                "cached decode differs at vocabulary index " << index);
    }
    CHECK(iom_model_loading::dims_of(decode_result.logits->spec().shape)
          == std::vector<std::size_t>{1, 19});
    CHECK_EQ(SessionAccess::caches(*cached)[0].initialized_length, 2);
    CHECK_EQ(SessionAccess::caches(*cached)[1].initialized_length, 2);
}

TEST_CASE("TinyLlama full model logits reject invalid token requests") {
    ForwardFixture fixture("validation", one_layer_config());
    auto session = fixture.load();
    CHECK_THROWS_AS(
            SessionAccess::prepare_forward_request(*session, 0),
            std::invalid_argument);
    CHECK_THROWS_AS(
            SessionAccess::prepare_forward_request(
                    *session, session->config().max_position_embeddings + 1),
            std::invalid_argument);

    SessionAccess::prepare_forward_request(*session, 1);
    const std::array<std::size_t, 0> empty{};
    CHECK_THROWS_AS(
            SessionAccess::forward_prefill(*session, empty),
            std::invalid_argument);
    const std::array<std::size_t, 1> invalid{
            session->config().vocab_size};
    CHECK_THROWS_AS(
            SessionAccess::forward_prefill(*session, invalid),
            std::invalid_argument);
    CHECK_THROWS_AS(
            SessionAccess::forward_decode(
                    *session, session->config().vocab_size),
            std::invalid_argument);
    CHECK_FALSE(session->poisoned());
    CHECK(SessionAccess::accepted(*session).empty());
}

TEST_CASE("TinyLlama full model logits reuse request owners") {
    ForwardFixture fixture("reuse", one_layer_config());
    auto session = fixture.load();
    SessionAccess::prepare_forward_request(*session, 2);
    const auto allocations = fixture.allocator.allocations;
    CHECK(iom_model_loading::dims_of(
                  SessionAccess::prefill(*session).x->view().spec().shape)
          == std::vector<std::size_t>{2, 8});
    CHECK(iom_model_loading::dims_of(
                  SessionAccess::decode(*session).x->view().spec().shape)
          == std::vector<std::size_t>{1, 8});

    const std::array<std::size_t, 2> prompt{0, 1};
    const auto prefill = SessionAccess::forward_prefill(*session, prompt);
    REQUIRE(iom::oid_is_token(prefill.producer));
    SessionAccess::wait(*session, prefill.producer);
    CHECK_EQ(fixture.allocator.allocations, allocations);
    const auto decode = SessionAccess::forward_decode(*session, 2);
    REQUIRE(iom::oid_is_token(decode.producer));
    SessionAccess::wait(*session, decode.producer);
    CHECK_EQ(fixture.allocator.allocations, allocations);
    const auto decode_logits = read_forward_tensor(*decode.logits);
    CHECK(std::all_of(
            decode_logits.begin(), decode_logits.end(),
            [](float value) { return std::isfinite(value); }));
}

TEST_CASE(
        "TinyLlama generation state machine validates admission and zero "
        "limit") {
    ForwardFixture fixture("generation-admission", one_layer_config());
    auto selector = std::make_unique<SequenceSelector>(
            std::vector<std::size_t>{4});
    SequenceSelector* selector_observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector));

    const std::array<std::size_t, 0> empty{};
    CHECK_THROWS_AS(session->generate_tokens(empty, 1),
            std::invalid_argument);

    const std::vector<std::size_t> over_capacity(
            session->config().max_position_embeddings + 1, 0);
    CHECK_THROWS_AS(session->generate_tokens(over_capacity, 1),
            std::invalid_argument);

    const std::array<std::size_t, 1> invalid{
            session->config().vocab_size};
    CHECK_THROWS_AS(session->generate_tokens(invalid, 1),
            std::invalid_argument);
    CHECK_EQ(selector_observer->calls, 0);
    CHECK_FALSE(session->poisoned());

    const std::array<std::size_t, 2> prompt{0, 1};
    const iom::TokenGenerationResult result =
            session->generate_tokens(prompt, 0);
    CHECK(result.token_ids.empty());
    CHECK(result.stop_reason == iom::GenerationStopReason::max_new_tokens);
    CHECK_EQ(selector_observer->calls, 0);
    CHECK(SessionAccess::history(*session).empty());
    CHECK(SessionAccess::accepted(*session).empty());
}

TEST_CASE(
        "TinyLlama generation state machine applies EOS and limit "
        "precedence") {
    {
        ForwardFixture fixture("generation-eos", one_layer_config());
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{2});
        SequenceSelector* observer = selector.get();
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector));
        const std::array<std::size_t, 2> prompt{0, 1};
        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 1);
        CHECK(result.token_ids == std::vector<std::size_t>{2});
        CHECK(result.stop_reason == iom::GenerationStopReason::eos);
        REQUIRE_EQ(observer->histories.size(), 1);
        CHECK(observer->histories[0] == std::vector<std::size_t>{0, 1});
        CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length, 2);
    }

    {
        ForwardFixture fixture("generation-limit", one_layer_config());
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{4});
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector));
        const std::array<std::size_t, 2> prompt{0, 1};
        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 1);
        CHECK(result.token_ids == std::vector<std::size_t>{4});
        CHECK(result.stop_reason == iom::GenerationStopReason::max_new_tokens);
        CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length, 2);
    }

    {
        ForwardFixture fixture("generation-context", one_layer_config());
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{4});
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector));
        const std::vector<std::size_t> prompt(
                session->config().max_position_embeddings - 1, 0);
        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 3);
        CHECK(result.token_ids == std::vector<std::size_t>{4});
        CHECK(result.stop_reason
              == iom::GenerationStopReason::context_capacity);
        CHECK_EQ(
                SessionAccess::caches(*session)[0].initialized_length,
                session->config().max_position_embeddings - 1);
    }

    {
        // EOS wins when the selected token simultaneously reaches both the
        // one-token limit and the final context row.
        ForwardFixture fixture("generation-precedence", one_layer_config());
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{2});
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector));
        const std::vector<std::size_t> prompt(
                session->config().max_position_embeddings - 1, 0);
        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 1);
        CHECK(result.token_ids == std::vector<std::size_t>{2});
        CHECK(result.stop_reason == iom::GenerationStopReason::eos);
    }
}

TEST_CASE(
        "TinyLlama generation state machine commits history and decodes "
        "only for nonterminal tokens") {
    ForwardFixture fixture("generation-transitions", one_layer_config());
    auto selector = std::make_unique<SequenceSelector>(
            std::vector<std::size_t>{4, 2});
    SequenceSelector* observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector));

    const std::array<std::size_t, 2> prompt{0, 1};
    const iom::TokenGenerationResult result =
            session->generate_tokens(prompt, 4);
    CHECK(result.token_ids == std::vector<std::size_t>{4, 2});
    CHECK(result.stop_reason == iom::GenerationStopReason::eos);
    REQUIRE_EQ(observer->calls, 2);
    REQUIRE_EQ(observer->histories.size(), 2);
    CHECK(observer->histories[0] == std::vector<std::size_t>{0, 1});
    CHECK(observer->histories[1] == std::vector<std::size_t>{0, 1, 4});
    REQUIRE_EQ(observer->producers.size(), 2);
    CHECK(iom::oid_is_token(observer->producers[0]));
    CHECK(iom::oid_is_token(observer->producers[1]));

    // The first generated token required one cached decode row.  The EOS
    // token was committed but did not grow the cache again.
    CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length, 3);

    // Returning the owned vector detaches it from the request that the next
    // call replaces.
    const std::array<std::size_t, 1> next_prompt{3};
    const iom::TokenGenerationResult zero =
            session->generate_tokens(next_prompt, 0);
    CHECK(zero.token_ids.empty());
    CHECK(result.token_ids == std::vector<std::size_t>{4, 2});
    CHECK(SessionAccess::history(*session).empty());
    CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length, 0);
}

TEST_CASE(
        "TinyLlama generation state machine rejects selector failures "
        "without reuse") {
    {
        ForwardFixture fixture("generation-range-failure", one_layer_config());
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{19});
        SequenceSelector* observer = selector.get();
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector));
        const std::array<std::size_t, 2> prompt{0, 1};
        CHECK_THROWS_AS(session->generate_tokens(prompt, 2),
                std::invalid_argument);
        CHECK(session->poisoned());
        CHECK_EQ(observer->calls, 1);
        REQUIRE_EQ(observer->histories.size(), 1);
        CHECK(observer->histories[0] == std::vector<std::size_t>{0, 1});
        CHECK_THROWS_AS(session->generate_tokens(prompt, 0),
                std::logic_error);
    }

    {
        ForwardFixture fixture("generation-selector-failure", one_layer_config());
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{4});
        SequenceSelector* observer = selector.get();
        observer->throw_failure = true;
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector));
        const std::array<std::size_t, 2> prompt{0, 1};
        CHECK_THROWS_AS(session->generate_tokens(prompt, 2),
                std::runtime_error);
        CHECK(session->poisoned());
        CHECK_EQ(observer->calls, 1);
        CHECK_THROWS_AS(session->generate_tokens(prompt, 0),
                std::logic_error);
    }
}

TEST_CASE(
        "TinyLlama generation state machine poisons and drains prefill "
        "failure") {
    ForwardFixture fixture("generation-prefill-failure", one_layer_config());
    auto selector = std::make_unique<SequenceSelector>(
            std::vector<std::size_t>{4});
    SequenceSelector* observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector));

    iom::cpu_detail::arm_cache_append_failure();
    struct ClearCacheAppendFailure {
        ~ClearCacheAppendFailure() {
            iom::cpu_detail::clear_cache_append_failure();
        }
    } clear_failure;

    const std::array<std::size_t, 2> prompt{0, 1};
    CHECK_THROWS_AS(session->generate_tokens(prompt, 2),
            std::runtime_error);
    CHECK(session->poisoned());
    CHECK_EQ(observer->calls, 0);
}

namespace {

// Strictly increasing, non-uniform supplied instants: consecutive differences
// grow monotonically, so a recorded phase window identifies the exact read pair
// it observed and a window that began or ended at the wrong read cannot
// reproduce the same difference. The supplied count is headroom: the wrapped
// call and every sibling observation between its endpoint reads consume
// instants, so the sequence is deliberately longer than the window asserted
// here.
void fill_distinct_instants(RecordingClock& clock, std::size_t count) {
    clock.instants.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        clock.instants[index] = 1'000 + 3 * index * index + 5 * index;
    }
}

// Message of a retained observation failure, rethrown from its owned handle.
[[nodiscard]] std::string retained_message(std::exception_ptr failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return error.what();
    }
    return std::string();
}

// The wrapped raw/chat attempt consumes the first three instants of its call
// (attempt entry, encode begin, encode end); the recorded span must be exactly
// the encode pair of that prefix.
[[nodiscard]] std::uint64_t window_after_entry(
        const RecordingClock& clock, std::size_t entry) {
    return clock.instants[entry + 2] - clock.instants[entry + 1];
}

}  // namespace

TEST_CASE("Inference metrics tokenization measures only the raw and chat encode span") {
    ForwardFixture fixture("metrics-tokenization-encode", text_generation_config());
    const iom::ChatMessageView messages[] = {
            {"system", "x"}, {"user", "x"}, {"assistant", "x"}};

    // Reference session without a recorder: the same deterministic fixture and
    // selector plan must produce identical generation behavior.
    RecordingClock detached;
    fill_distinct_instants(detached, 4);
    iom::InferenceMetrics detached_recorder(make_clock(detached));
    auto plain_selector =
            std::make_unique<SequenceSelector>(std::vector<std::size_t>{2, 2});
    SequenceSelector* plain_observer = plain_selector.get();
    auto plain = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(plain_selector));
    REQUIRE(plain);
    CHECK_EQ(SessionAccess::metrics(*plain), nullptr);
    const iom::GenerationResult plain_raw = plain->generate_raw("hello", 1);
    const iom::GenerationResult plain_chat = plain->generate_chat(messages, 1);
    CHECK(plain_raw.token_ids == std::vector<std::size_t>{2});
    CHECK(plain_chat.token_ids == std::vector<std::size_t>{2});
    // Disabled observation performs no clock read and touches no recorder.
    CHECK_EQ(detached.cursor, 0);
    CHECK_FALSE(detached.exhausted);
    CHECK_FALSE(detached_recorder.snapshot().request_admitted);

    RecordingClock clock;
    fill_distinct_instants(clock, 4096);
    iom::InferenceMetrics recorder(make_clock(clock));
    auto selector =
            std::make_unique<SequenceSelector>(std::vector<std::size_t>{2, 2, 2});
    SequenceSelector* observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector),
            &recorder);
    REQUIRE(session);
    CHECK_EQ(recorder.snapshot().load.host_nanoseconds,
             clock.instants[1] - clock.instants[0]);

    // Raw: the wrapper's first three reads are the attempt entry, the encode
    // begin, and the encode end, so the recorded span is exactly the encode
    // window - never entry-to-end and never extended into generation.
    const std::size_t raw_entry = clock.cursor;
    const iom::GenerationResult raw = session->generate_raw("hello", 1);
    REQUIRE(clock.cursor - raw_entry >= 3);
    CHECK(raw.token_ids == plain_raw.token_ids);
    CHECK(raw.text == plain_raw.text);
    CHECK(raw.stop_reason == plain_raw.stop_reason);
    CHECK(recorder.snapshot().request_admitted);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK(recorder.snapshot().admitted.tokenization.state
          == iom::ObservationState::succeeded);
    const std::uint64_t raw_span = window_after_entry(clock, raw_entry);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, raw_span);

    // Multi-message chat: rendering completes before the measured begin, and
    // the second admission replaces the span instead of accumulating it.
    const std::size_t chat_entry = clock.cursor;
    const iom::GenerationResult chat = session->generate_chat(messages, 1);
    REQUIRE(clock.cursor - chat_entry >= 3);
    CHECK(chat.token_ids == plain_chat.token_ids);
    CHECK(chat.text == plain_chat.text);
    CHECK(chat.stop_reason == plain_chat.stop_reason);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 2);
    CHECK(recorder.snapshot().admitted.tokenization.state
          == iom::ObservationState::succeeded);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds,
             window_after_entry(clock, chat_entry));

    // The encoded history and the KV prefix are identical with probes on/off.
    REQUIRE_EQ(observer->histories.size(), 2);
    REQUIRE_EQ(plain_observer->histories.size(), 2);
    CHECK(observer->histories[0] == plain_observer->histories[0]);
    CHECK(observer->histories[1] == plain_observer->histories[1]);
    CHECK(observer->histories[0] == std::vector<std::size_t>{1, 268});
    CHECK_EQ(session->request_length(), plain->request_length());
    CHECK_EQ(SessionAccess::caches(*session)[0].initialized_length,
             SessionAccess::caches(*plain)[0].initialized_length);
    CHECK_FALSE(session->poisoned());

    // A direct low-level request never inherits raw/chat tokenization: the
    // publication bridge clears the admitted span and no wrapper is involved.
    const std::vector<std::size_t> prompt{0, 1};
    const iom::TokenGenerationResult direct = session->generate_tokens(prompt, 1);
    CHECK(direct.token_ids == std::vector<std::size_t>{2});
    CHECK(direct.stop_reason == iom::GenerationStopReason::eos);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 3);
    CHECK(recorder.snapshot().admitted.tokenization.state
          == iom::ObservationState::not_run);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, 0);
}

TEST_CASE("Inference metrics tokenization attributes preprocessing failures to their own attempt") {
    ForwardFixture fixture("metrics-tokenization-failures",
                           text_generation_config());
    RecordingClock clock;
    fill_distinct_instants(clock, 4096);
    iom::InferenceMetrics recorder(make_clock(clock));
    auto selector =
            std::make_unique<SequenceSelector>(std::vector<std::size_t>{2, 2});
    SequenceSelector* observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector),
            &recorder);
    REQUIRE(session);

    // Request 1 is admitted with its own measured span.
    const std::size_t first_entry = clock.cursor;
    const iom::GenerationResult first = session->generate_raw("hello", 1);
    CHECK(first.token_ids == std::vector<std::size_t>{2});
    REQUIRE(recorder.snapshot().request_admitted);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    const std::uint64_t first_span = window_after_entry(clock, first_entry);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, first_span);
    const std::size_t calls_after_first = observer->calls;
    const std::size_t accepted_after_first =
            SessionAccess::accepted(*session).size();
    const std::size_t length_after_first = session->request_length();
    REQUIRE_EQ(calls_after_first, 1);

    // Encoder failure: the encoder itself was observed, so the attempt failed
    // after a captured encode window, while request 1 keeps its own span.
    std::string invalid_utf8(2, '\0');
    invalid_utf8[0] = static_cast<char>(0xC3);
    invalid_utf8[1] = static_cast<char>(0x28);
    const std::size_t encoder_entry = clock.cursor;
    CHECK_THROWS_AS(session->generate_raw(invalid_utf8, 1),
                    std::invalid_argument);
    CHECK_EQ(clock.cursor - encoder_entry, 3);
    CHECK_FALSE(clock.exhausted);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK(recorder.snapshot().admitted.tokenization.state
          == iom::ObservationState::succeeded);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, first_span);
    CHECK_EQ(recorder.snapshot().attempt.ordinal, 2);
    CHECK(recorder.snapshot().attempt.outcome == iom::AttemptOutcome::failed);
    CHECK_MESSAGE(
            retained_message(recorder.snapshot().attempt.failure)
                            .find("invalid UTF-8") != std::string::npos,
            retained_message(recorder.snapshot().attempt.failure));
    CHECK_EQ(observer->calls, calls_after_first);
    CHECK_EQ(SessionAccess::accepted(*session).size(), accepted_after_first);
    CHECK_EQ(session->request_length(), length_after_first);
    CHECK_FALSE(session->poisoned());

    // Formatter rejection: a distinct failed attempt that never reached the
    // encoder, so it reads no encode instants and records no tokenization.
    const iom::ChatMessageView invalid_role[] = {{"User", "hello"}};
    const std::size_t formatter_entry = clock.cursor;
    CHECK_THROWS_AS(session->generate_chat(invalid_role, 1),
                    std::invalid_argument);
    CHECK_EQ(clock.cursor - formatter_entry, 1);
    CHECK_FALSE(clock.exhausted);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK(recorder.snapshot().admitted.tokenization.state
          == iom::ObservationState::succeeded);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, first_span);
    CHECK(recorder.snapshot().attempt.outcome == iom::AttemptOutcome::failed);
    CHECK_MESSAGE(
            retained_message(recorder.snapshot().attempt.failure)
                            .find("requires the role string")
                    != std::string::npos,
            retained_message(recorder.snapshot().attempt.failure));
    CHECK_EQ(observer->calls, calls_after_first);
    CHECK_EQ(session->request_length(), length_after_first);
    CHECK_FALSE(session->poisoned());

    // Recovery replaces the admitted span with the new request's own window.
    const std::size_t recovered_entry = clock.cursor;
    const iom::GenerationResult recovered = session->generate_raw("hello", 1);
    CHECK(recovered.token_ids == std::vector<std::size_t>{2});
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 2);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds,
             window_after_entry(clock, recovered_entry));
    CHECK_EQ(observer->calls, calls_after_first + 1);
}

TEST_CASE("Inference metrics tokenization preserves prior attribution across a failed drain") {
    ForwardFixture fixture("metrics-tokenization-drain", text_generation_config());
    RecordingClock clock;
    fill_distinct_instants(clock, 4096);
    iom::InferenceMetrics recorder(make_clock(clock));
    auto selector =
            std::make_unique<SequenceSelector>(std::vector<std::size_t>{2, 2});
    SequenceSelector* observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector),
            &recorder);
    REQUIRE(session);

    const std::size_t first_entry = clock.cursor;
    const iom::GenerationResult first = session->generate_raw("hello", 1);
    CHECK(first.token_ids == std::vector<std::size_t>{2});
    REQUIRE(recorder.snapshot().request_admitted);
    const std::uint64_t first_span = window_after_entry(clock, first_entry);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, first_span);
    const std::size_t length_after_first = session->request_length();
    const std::size_t calls_after_first = observer->calls;

    // A pending accepted append that fails at its wait: the replacement's drain
    // reports the queue's original failure instead of publishing anything.
    const auto& banks = SessionAccess::prefill(*session);
    const auto caches = SessionAccess::caches(*session);
    std::vector<std::byte> bytes(banks.k->view().spec().logical_nbytes(),
                                 std::byte{0});
    banks.k->view().copy_from_host(bytes);
    banks.v->view().copy_from_host(bytes);
    {
        struct ResetFailure {
            ~ResetFailure() { iom::cpu_detail::clear_cache_append_failure(); }
        } failure;
        iom::cpu_detail::arm_cache_append_failure();
        SessionAccess::submit(*session, [&](iom::DeviceOps& queue) {
            return queue.cache_append(banks.k->view(), caches[0].key->view(), 0);
        });
        const std::size_t drain_entry = clock.cursor;
        CHECK_THROWS_AS(session->generate_raw("hello", 1), std::runtime_error);
        // The rejected candidate's encode window was measured but never
        // attributed: the failed drain precedes publication.
        CHECK(clock.cursor - drain_entry >= 3);
    }

    CHECK(session->poisoned());
    CHECK_EQ(session->request_length(), length_after_first);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    CHECK(recorder.snapshot().admitted.tokenization.state
          == iom::ObservationState::succeeded);
    CHECK_EQ(recorder.snapshot().admitted.tokenization.host_nanoseconds, first_span);
    CHECK_EQ(observer->calls, calls_after_first);
}

TEST_CASE("TinyLlama text chat generation composes raw tokenizer input") {
    ForwardFixture fixture("text-chat-raw-entrypoints",
                           text_generation_config());
    auto selector = std::make_unique<SequenceSelector>(
            std::vector<std::size_t>{2, 2, 2});
    SequenceSelector* observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector));

    const iom::GenerationResult hello = session->generate_raw("hello", 1);
    CHECK(hello.token_ids == std::vector<std::size_t>{2});
    CHECK(hello.text.empty());
    CHECK(hello.stop_reason == iom::GenerationStopReason::eos);
    REQUIRE_EQ(observer->histories.size(), 1);
    CHECK(observer->histories[0] == std::vector<std::size_t>{1, 268});

    std::string d7ff(3, '\0');
    d7ff[0] = static_cast<char>(0xED);
    d7ff[1] = static_cast<char>(0x9F);
    d7ff[2] = static_cast<char>(0xBF);
    const iom::GenerationResult byte_fallback =
            session->generate_raw(d7ff, 1);
    CHECK(byte_fallback.token_ids == std::vector<std::size_t>{2});
    CHECK(byte_fallback.text.empty());
    CHECK(byte_fallback.stop_reason == iom::GenerationStopReason::eos);
    REQUIRE_EQ(observer->histories.size(), 2);
    CHECK(observer->histories[1]
          == std::vector<std::size_t>{1, 259, 240, 162, 194});

    const iom::GenerationResult whitespace = session->generate_raw(" ", 1);
    CHECK(whitespace.token_ids == std::vector<std::size_t>{2});
    CHECK(whitespace.text.empty());
    CHECK(whitespace.stop_reason == iom::GenerationStopReason::eos);
    REQUIRE_EQ(observer->histories.size(), 3);
    CHECK(observer->histories[2] == std::vector<std::size_t>{1, 259, 259});
}

TEST_CASE("TinyLlama text chat generation renders roles and assistant prefix") {
    ForwardFixture fixture("text-chat-structured-entrypoints",
                           text_generation_config());
    auto selector = std::make_unique<SequenceSelector>(
            std::vector<std::size_t>{268, 2});
    SequenceSelector* observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector));

    const iom::ChatMessageView messages[] = {
            {"system", "x"}, {"user", "x"}, {"assistant", "x"}};
    CHECK(session->formatter().format(messages, true)
          == "x</s>x</s>x</s><|assistant|>");
    const iom::GenerationResult structured =
            session->generate_chat(messages, 1);
    CHECK(structured.token_ids == std::vector<std::size_t>{268});
    CHECK(structured.text == "hello");
    CHECK(structured.stop_reason == iom::GenerationStopReason::max_new_tokens);
    REQUIRE_EQ(observer->histories.size(), 1);
    CHECK(observer->histories[0]
          == std::vector<std::size_t>{
                  1,   259, 123, 2,   259, 123, 2,   259, 123, 2,
                  259, 63,  127, 100, 118, 118, 108, 118, 119, 100,
                  113, 119, 127, 65});

    const iom::ChatMessageView empty_user[] = {{"user", ""}};
    CHECK(session->formatter().format(empty_user, true)
          == "</s><|assistant|>");
    const iom::GenerationResult empty_content =
            session->generate_chat(empty_user, 1);
    CHECK(empty_content.token_ids == std::vector<std::size_t>{2});
    CHECK(empty_content.text.empty());
    CHECK(empty_content.stop_reason == iom::GenerationStopReason::eos);
    REQUIRE_EQ(observer->histories.size(), 2);
    CHECK(observer->histories[1]
          == std::vector<std::size_t>{
                  1, 2, 259, 63, 127, 100, 118, 118, 108, 118,
                  119, 100, 113, 119, 127, 65});
}

TEST_CASE("TinyLlama text chat generation forwards every stop reason") {
    {
        ForwardFixture fixture("text-chat-stop-eos", text_generation_config());
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device,
                std::make_unique<SequenceSelector>(
                        std::vector<std::size_t>{2}));
        const iom::GenerationResult result =
                session->generate_raw("hello", 3);
        CHECK(result.token_ids == std::vector<std::size_t>{2});
        CHECK(result.text.empty());
        CHECK(result.stop_reason == iom::GenerationStopReason::eos);
    }

    {
        ForwardFixture fixture("text-chat-stop-limit", text_generation_config());
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device,
                std::make_unique<SequenceSelector>(
                        std::vector<std::size_t>{268}));
        const iom::GenerationResult result =
                session->generate_raw("hello", 1);
        CHECK(result.token_ids == std::vector<std::size_t>{268});
        CHECK(result.text == "hello");
        CHECK(result.stop_reason == iom::GenerationStopReason::max_new_tokens);
    }

    {
        nlohmann::json config = text_generation_config();
        config["max_position_embeddings"] = 17;
        ForwardFixture fixture("text-chat-stop-context", config);
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device,
                std::make_unique<SequenceSelector>(
                        std::vector<std::size_t>{268}));
        std::string near_capacity = "hello";
        for (std::size_t index = 1; index < 15; ++index)
            near_capacity += " hello";
        const iom::GenerationResult result =
                session->generate_raw(near_capacity, 3);
        CHECK(result.token_ids == std::vector<std::size_t>{268});
        CHECK(result.text == "hello");
        CHECK(result.stop_reason == iom::GenerationStopReason::context_capacity);
    }

    {
        ForwardFixture fixture("text-chat-stop-zero", text_generation_config());
        auto selector = std::make_unique<SequenceSelector>(
                std::vector<std::size_t>{});
        SequenceSelector* observer = selector.get();
        auto session = iom::load_tinyllama_session(
                fixture.directory.path(), *fixture.device, std::move(selector));
        const iom::GenerationResult result = session->generate_raw("hello", 0);
        CHECK(result.token_ids.empty());
        CHECK(result.text.empty());
        CHECK(result.stop_reason == iom::GenerationStopReason::max_new_tokens);
        CHECK_EQ(observer->calls, 0);
        CHECK(session->request_length() == 2);
        CHECK(SessionAccess::history(*session).empty());
    }
}

TEST_CASE("TinyLlama text chat generation rejects preprocessing errors atomically") {
    ForwardFixture fixture("text-chat-invalid-input", text_generation_config());
    auto selector = std::make_unique<SequenceSelector>(
            std::vector<std::size_t>{2});
    SequenceSelector* observer = selector.get();
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device, std::move(selector));

    std::string invalid_utf8(2, '\0');
    invalid_utf8[0] = static_cast<char>(0xC3);
    invalid_utf8[1] = static_cast<char>(0x28);
    CHECK_THROWS_AS(session->generate_raw(invalid_utf8, 1),
                    std::invalid_argument);

    const iom::ChatMessageView invalid_role[] = {{"User", "hello"}};
    CHECK_THROWS_AS(session->generate_chat(invalid_role, 1),
                    std::invalid_argument);
    const iom::ChatMessageView invalid_content[] = {
            {"user", std::string_view("\xC3\x28", 2)}};
    CHECK_THROWS_AS(session->generate_chat(invalid_content, 1),
                    std::invalid_argument);

    CHECK_EQ(observer->calls, 0);
    CHECK(session->request_length() == 0);
    CHECK_FALSE(session->poisoned());

    const iom::GenerationResult recovered = session->generate_raw("hello", 1);
    CHECK(recovered.token_ids == std::vector<std::size_t>{2});
    CHECK(recovered.text.empty());
    CHECK(recovered.stop_reason == iom::GenerationStopReason::eos);
    CHECK_EQ(observer->calls, 1);
}

TEST_CASE("TinyLlama text chat generation owns results across session reuse") {
    ForwardFixture fixture("text-chat-result-ownership", text_generation_config());
    auto session = iom::load_tinyllama_session(
            fixture.directory.path(), *fixture.device,
            std::make_unique<SequenceSelector>(
                    std::vector<std::size_t>{268, 2}));

    const iom::GenerationResult first = session->generate_raw("hello", 1);
    REQUIRE(first.token_ids == std::vector<std::size_t>{268});
    REQUIRE(first.text == "hello");
    REQUIRE(first.stop_reason == iom::GenerationStopReason::max_new_tokens);

    const iom::GenerationResult second = session->generate_raw("hello", 1);
    CHECK(second.token_ids == std::vector<std::size_t>{2});
    CHECK(second.text.empty());
    CHECK(second.stop_reason == iom::GenerationStopReason::eos);

    CHECK(first.token_ids == std::vector<std::size_t>{268});
    CHECK(first.text == "hello");
    CHECK(first.stop_reason == iom::GenerationStopReason::max_new_tokens);
}
// ---------------------------------------------------------------------------
// Composed synthetic generation traces.  These cases deliberately build the
// expected logits without using a session forward seam or a production
// selector.  The fixture is small enough to run on every configured host but
// still exercises GQA, absolute RoPE positions, exact capacity, request
// replacement, and the accepted-failure poison boundary.
// ---------------------------------------------------------------------------

namespace synthetic_session_test {

constexpr float kSyntheticAtol = 0.05F;
constexpr float kSyntheticRtol = 0.02F;
constexpr float kSyntheticWinnerMargin = 0.50F;
constexpr std::size_t kSyntheticCapacity = 5;
constexpr std::size_t kSyntheticVocabulary = 8;
constexpr std::size_t kSyntheticEos = 2;

[[nodiscard]] float synthetic_round_bf16(float value) noexcept {
    return forward_decode_bf16(forward_encode_bf16(value));
}

[[nodiscard]] std::string synthetic_weight_payload(
        const std::vector<std::size_t>& shape, std::size_t entry) {
    const std::size_t elements = iom_model_loading::element_count(shape);
    const std::size_t hidden = shape.back();
    std::vector<float> values(elements);
    for (std::size_t element = 0; element < elements; ++element) {
        float value = 0.20F
                + 0.025F * static_cast<float>((entry + element) % 5);
        if (entry == 1
                || (entry >= 3 && ((entry - 3) % 9) < 2)) {
            value = 1.0F
                    + 0.02F * static_cast<float>((entry + element) % 3);
        } else if (entry == 2) {
            const std::size_t row = element / hidden;
            value = row == 0 ? 2.0F : row == 1 ? 0.25F : 0.0F;
        }
        values[element] = synthetic_round_bf16(value);
    }

    std::string payload(elements * sizeof(std::uint16_t), '\0');
    for (std::size_t element = 0; element < elements; ++element) {
        const std::uint16_t bits = forward_encode_bf16(values[element]);
        std::memcpy(payload.data() + element * sizeof(bits), &bits,
                    sizeof(bits));
    }
    return payload;
}

struct Fixture {
    TempDir directory;
    HeapAllocator allocator;
    std::unique_ptr<iom::Device> device;
    nlohmann::json config;

    Fixture(std::size_t layers, std::string tag)
        : directory("session-synthetic-" + std::move(tag)),
          config(two_layer_config()) {
        config["num_hidden_layers"] = layers;
        config["vocab_size"] = kSyntheticVocabulary;
        config["max_position_embeddings"] = kSyntheticCapacity;
        config["bos_token_id"] = 1;
        config["eos_token_id"] = kSyntheticEos;
        write_config(directory.path(), config);

        auto entries = required_weight_entries(config);
        for (std::size_t entry = 0; entry < entries.size(); ++entry) {
            entries[entry].payload =
                    synthetic_weight_payload(entries[entry].shape, entry);
        }
        write_safetensors_file(directory.path(), "model.safetensors", entries);
        write_tokenizer(directory.path());
        write_file(directory.path(), "tokenizer_config.json",
                   tokenizer_config().dump());
        device = iom::make_cpu_device(allocator);
    }

    [[nodiscard]] std::unique_ptr<iom::TinyLlamaSession> load(
            std::unique_ptr<iom::TokenSelector> selector) {
        return iom::load_tinyllama_session(
                directory.path(), *device, std::move(selector));
    }
};

class RecordingGreedySelector final : public iom::TokenSelector {
public:
    [[nodiscard]] iom::TokenSelectorScratchRequirements scratch_requirements(
            const iom::TensorView&, std::size_t) const override {
        return {0, {0, 1}};
    }

    [[nodiscard]] std::size_t select(
            iom::DeviceOps&, const iom::TensorView& logits, std::size_t,
            iom::oid producer, std::span<const std::size_t> history,
            iom::TokenSelectorScratch) override {
        const std::vector<float> row = read_forward_tensor(logits);
        logits_rows.push_back(row);
        producers.push_back(producer);
        histories.emplace_back(history.begin(), history.end());

        std::size_t selected = 0;
        for (std::size_t index = 1; index < row.size(); ++index) {
            if (row[index] > row[selected]) selected = index;
        }
        selected_ids.push_back(selected);
        return selected;
    }

    std::vector<std::vector<float>> logits_rows;
    std::vector<iom::oid> producers;
    std::vector<std::vector<std::size_t>> histories;
    std::vector<std::size_t> selected_ids;
};

struct ReferenceModel {
    iom::TinyLlamaConfig config;
    std::vector<std::vector<float>> weights;

    explicit ReferenceModel(const iom::TinyLlamaSession& session)
        : config(session.config()) {
        weights.reserve(session.model().weights().size());
        for (std::size_t index = 0;
             index < session.model().weights().size(); ++index) {
            weights.push_back(read_forward_tensor(session.model().weight(index)));
        }
    }

    [[nodiscard]] std::vector<float> logits(
            std::span<const std::size_t> tokens) const {
        const std::size_t rows = tokens.size();
        const std::size_t features = config.hidden_size;
        const std::size_t intermediate = config.intermediate_size;
        const std::size_t query_heads = config.num_attention_heads;
        const std::size_t kv_heads = config.num_key_value_heads;
        const std::size_t head_dim = config.head_dim;
        const std::size_t capacity = config.max_position_embeddings;
        const std::size_t grouping = query_heads / kv_heads;

        std::vector<float> hidden(rows * features);
        const std::vector<float>& embedding = weights[0];
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t feature = 0; feature < features; ++feature) {
                hidden[row * features + feature] =
                        embedding[tokens[row] * features + feature];
            }
        }

        for (std::size_t layer = 0; layer < config.num_hidden_layers;
             ++layer) {
            const auto layer_weight =
                    [&](std::size_t role) -> const std::vector<float>& {
                return weights[3 + layer * 9 + role];
            };
            const std::vector<float>& input_norm = layer_weight(0);
            const std::vector<float>& post_norm = layer_weight(1);
            const std::vector<float>& query_weight = layer_weight(2);
            const std::vector<float>& key_weight = layer_weight(3);
            const std::vector<float>& value_weight = layer_weight(4);
            const std::vector<float>& output_weight = layer_weight(5);
            const std::vector<float>& gate_weight = layer_weight(6);
            const std::vector<float>& up_weight = layer_weight(7);
            const std::vector<float>& down_weight = layer_weight(8);

            std::vector<float> key_cache(kv_heads * capacity * head_dim, 0.0F);
            std::vector<float> value_cache(
                    kv_heads * capacity * head_dim, 0.0F);
            std::vector<float> next(hidden.size());

            for (std::size_t row = 0; row < rows; ++row) {
                float square_sum = 0.0F;
                for (std::size_t feature = 0; feature < features; ++feature) {
                    const float value = hidden[row * features + feature];
                    square_sum += value * value;
                }
                const float inverse = 1.0F / std::sqrt(
                        square_sum / static_cast<float>(features)
                        + config.rms_norm_eps);
                std::vector<float> normalized(features);
                for (std::size_t feature = 0; feature < features; ++feature) {
                    normalized[feature] = synthetic_round_bf16(
                            hidden[row * features + feature] * inverse
                            * input_norm[feature]);
                }

                std::vector<float> query(query_heads * head_dim);
                std::vector<float> key(kv_heads * head_dim);
                std::vector<float> value(kv_heads * head_dim);
                const auto project = [&](std::vector<float>& output,
                                         const std::vector<float>& weight,
                                         std::size_t heads) {
                    for (std::size_t head = 0; head < heads; ++head) {
                        for (std::size_t element = 0; element < head_dim;
                             ++element) {
                            float sum = 0.0F;
                            const std::size_t weight_row =
                                    head * head_dim + element;
                            for (std::size_t feature = 0; feature < features;
                                 ++feature) {
                                sum = std::fma(
                                        normalized[feature],
                                        weight[weight_row * features + feature],
                                        sum);
                            }
                            output[weight_row] = synthetic_round_bf16(sum);
                        }
                    }
                };
                project(query, query_weight, query_heads);
                project(key, key_weight, kv_heads);
                project(value, value_weight, kv_heads);

                const auto rotate = [&](std::vector<float>& values,
                                        std::size_t heads) {
                    const std::size_t half = head_dim / 2;
                    for (std::size_t head = 0; head < heads; ++head) {
                        for (std::size_t pair = 0; pair < half; ++pair) {
                            const std::size_t first_index =
                                    head * head_dim + pair;
                            const std::size_t second_index =
                                    first_index + half;
                            const double exponent =
                                    (-2.0 * static_cast<double>(pair))
                                    / static_cast<double>(head_dim);
                            const float angle = static_cast<float>(
                                    static_cast<double>(row)
                                    * std::pow(config.rope_theta, exponent));
                            const float first = values[first_index];
                            const float second = values[second_index];
                            const volatile float first_product =
                                    first * std::cos(angle);
                            const volatile float second_product =
                                    second * std::sin(angle);
                            const volatile float first_output =
                                    first_product - second_product;
                            const volatile float second_cosine_product =
                                    second * std::cos(angle);
                            const volatile float first_sine_product =
                                    first * std::sin(angle);
                            const volatile float second_output =
                                    second_cosine_product + first_sine_product;
                            values[first_index] =
                                    synthetic_round_bf16(first_output);
                            values[second_index] =
                                    synthetic_round_bf16(second_output);
                        }
                    }
                };
                rotate(query, query_heads);
                rotate(key, kv_heads);

                for (std::size_t head = 0; head < kv_heads; ++head) {
                    for (std::size_t element = 0; element < head_dim;
                         ++element) {
                        const std::size_t cache_index =
                                (head * capacity + row) * head_dim + element;
                        key_cache[cache_index] =
                                key[head * head_dim + element];
                        value_cache[cache_index] =
                                value[head * head_dim + element];
                    }
                }

                std::vector<float> merged(features, 0.0F);
                for (std::size_t head = 0; head < query_heads; ++head) {
                    const std::size_t kv_head = head / grouping;
                    std::vector<float> scores(row + 1);
                    for (std::size_t token = 0; token <= row; ++token) {
                        float dot = 0.0F;
                        for (std::size_t element = 0; element < head_dim;
                             ++element) {
                            const volatile float product =
                                    query[head * head_dim + element]
                                    * key_cache[(kv_head * capacity + token)
                                                * head_dim + element];
                            const volatile float next_score = dot + product;
                            dot = next_score;
                        }
                        scores[token] =
                                dot / std::sqrt(static_cast<float>(head_dim));
                    }
                    float maximum = scores[0];
                    for (const float score : scores) {
                        maximum = std::max(maximum, score);
                    }
                    // The production SDPA stores the normalized probability
                    // only after the complete sum is known.
                    std::vector<float> exponentials(row + 1);
                    float sum = 0.0F;
                    for (std::size_t token = 0; token <= row; ++token) {
                        const volatile float shifted =
                                scores[token] - maximum;
                        const volatile float exponent = std::exp(shifted);
                        const volatile float next_sum = sum + exponent;
                        sum = next_sum;
                        exponentials[token] = exponent;
                    }
                    for (std::size_t token = 0; token <= row; ++token) {
                        scores[token] = synthetic_round_bf16(
                                exponentials[token] / sum);
                    }
                    for (std::size_t element = 0; element < head_dim;
                         ++element) {
                        float result = 0.0F;
                        for (std::size_t token = 0; token <= row; ++token) {
                            const volatile float product =
                                    scores[token]
                                    * value_cache[(kv_head * capacity + token)
                                                  * head_dim + element];
                            const volatile float next_result = result + product;
                            result = next_result;
                        }
                        merged[head * head_dim + element] =
                                synthetic_round_bf16(result);
                    }
                }

                std::vector<float> projected(features);
                for (std::size_t output = 0; output < features; ++output) {
                    float sum = 0.0F;
                    for (std::size_t feature = 0; feature < features;
                         ++feature) {
                        sum = std::fma(
                                merged[feature],
                                output_weight[output * features + feature],
                                sum);
                    }
                    projected[output] = synthetic_round_bf16(sum);
                }

                std::vector<float> residual(features);
                for (std::size_t feature = 0; feature < features; ++feature) {
                    residual[feature] = synthetic_round_bf16(
                            hidden[row * features + feature]
                            + projected[feature]);
                }

                square_sum = 0.0F;
                for (const float value : residual) square_sum += value * value;
                const float post_inverse = 1.0F / std::sqrt(
                        square_sum / static_cast<float>(features)
                        + config.rms_norm_eps);
                std::vector<float> normalized_post(features);
                for (std::size_t feature = 0; feature < features; ++feature) {
                    normalized_post[feature] = synthetic_round_bf16(
                            residual[feature] * post_inverse
                            * post_norm[feature]);
                }

                std::vector<float> gate(intermediate);
                std::vector<float> up(intermediate);
                for (std::size_t output = 0; output < intermediate; ++output) {
                    float gate_sum = 0.0F;
                    float up_sum = 0.0F;
                    for (std::size_t feature = 0; feature < features;
                         ++feature) {
                        gate_sum = std::fma(
                                normalized_post[feature],
                                gate_weight[output * features + feature],
                                gate_sum);
                        up_sum = std::fma(
                                normalized_post[feature],
                                up_weight[output * features + feature],
                                up_sum);
                    }
                    gate[output] = synthetic_round_bf16(gate_sum);
                    up[output] = synthetic_round_bf16(up_sum);
                }
                std::vector<float> product(intermediate);
                for (std::size_t index = 0; index < intermediate; ++index) {
                    const float activated = synthetic_round_bf16(
                            gate[index]
                            / (1.0F + std::exp(-gate[index])));
                    product[index] =
                            synthetic_round_bf16(activated * up[index]);
                }
                std::vector<float> down(features);
                for (std::size_t output = 0; output < features; ++output) {
                    float sum = 0.0F;
                    for (std::size_t index = 0; index < intermediate; ++index) {
                        sum = std::fma(
                                product[index],
                                down_weight[output * intermediate + index],
                                sum);
                    }
                    down[output] = synthetic_round_bf16(sum);
                    next[row * features + output] = synthetic_round_bf16(
                            residual[output] + down[output]);
                }
            }
            hidden.swap(next);
        }

        const std::vector<float>& final_norm = weights[1];
        const std::vector<float>& lm_head = weights[2];
        const std::size_t final_row = rows - 1;
        float square_sum = 0.0F;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const float value = hidden[final_row * features + feature];
            square_sum += value * value;
        }
        const float inverse = 1.0F / std::sqrt(
                square_sum / static_cast<float>(features)
                + config.rms_norm_eps);
        std::vector<float> normalized(features);
        for (std::size_t feature = 0; feature < features; ++feature) {
            normalized[feature] = synthetic_round_bf16(
                    hidden[final_row * features + feature] * inverse
                    * final_norm[feature]);
        }

        std::vector<float> result(config.vocab_size);
        for (std::size_t output = 0; output < config.vocab_size; ++output) {
            float sum = 0.0F;
            for (std::size_t feature = 0; feature < features; ++feature) {
                sum = std::fma(
                        normalized[feature],
                        lm_head[output * features + feature],
                        sum);
            }
            result[output] = synthetic_round_bf16(sum);
        }
        return result;
    }
};

[[nodiscard]] std::size_t reference_greedy(
        std::span<const float> logits, std::size_t eos) {
    REQUIRE(!logits.empty());
    std::size_t selected = 0;
    for (std::size_t index = 1; index < logits.size(); ++index) {
        if (logits[index] > logits[selected]) selected = index;
    }
    float next = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (index != selected) next = std::max(next, logits[index]);
    }
    const bool winner_ok =
            std::isfinite(logits[selected]) && std::isfinite(next)
            && logits[selected] - next >= kSyntheticWinnerMargin;
    REQUIRE_MESSAGE(
            winner_ok,
            "synthetic greedy winner has insufficient finite separation");
    (void)eos;
    return selected;
}

struct ExpectedTrace {
    std::vector<std::vector<float>> logits;
    std::vector<std::size_t> token_ids;
    iom::GenerationStopReason stop_reason =
            iom::GenerationStopReason::max_new_tokens;
};

[[nodiscard]] ExpectedTrace expected_trace(
        const ReferenceModel& reference,
        std::span<const std::size_t> prompt, std::size_t max_new_tokens) {
    ExpectedTrace expected;
    std::vector<std::size_t> prefix(prompt.begin(), prompt.end());
    for (;;) {
        expected.logits.push_back(reference.logits(prefix));
        const std::size_t selected =
                reference_greedy(expected.logits.back(),
                                 reference.config.eos_token_id);
        expected.token_ids.push_back(selected);
        prefix.push_back(selected);
        if (selected == reference.config.eos_token_id) {
            expected.stop_reason = iom::GenerationStopReason::eos;
            break;
        }
        if (expected.token_ids.size() >= max_new_tokens) {
            expected.stop_reason = iom::GenerationStopReason::max_new_tokens;
            break;
        }
        if (prefix.size() >= reference.config.max_position_embeddings) {
            expected.stop_reason =
                    iom::GenerationStopReason::context_capacity;
            break;
        }
    }
    return expected;
}

void compare_logits(
        std::span<const float> actual, std::span<const float> expected,
        const char* label) {
    REQUIRE_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const bool finite = std::isfinite(actual[index])
                && std::isfinite(expected[index]);
        REQUIRE_MESSAGE(
                finite,
                label << " has non-finite vocabulary index " << index);
        const float difference = std::abs(actual[index] - expected[index]);
        const float bound =
                kSyntheticAtol + kSyntheticRtol * std::abs(expected[index]);
        CHECK_MESSAGE(
                difference <= bound,
                label << " differs at vocabulary index " << index
                      << " by " << difference << " (bound " << bound << ")");
    }
}

void compare_trace(
        const ExpectedTrace& expected,
        const RecordingGreedySelector& selector,
        const iom::TokenGenerationResult& result,
        std::span<const std::size_t> prompt, const char* label) {
    REQUIRE_EQ(selector.logits_rows.size(), expected.logits.size());
    REQUIRE_EQ(selector.selected_ids, expected.token_ids);
    CHECK(result.token_ids == expected.token_ids);
    CHECK(result.stop_reason == expected.stop_reason);
    for (std::size_t step = 0; step < expected.logits.size(); ++step) {
        compare_logits(
                selector.logits_rows[step], expected.logits[step], label);
    }

    REQUIRE_EQ(selector.histories.size(), expected.logits.size());
    std::vector<std::size_t> history(prompt.begin(), prompt.end());
    for (std::size_t step = 0; step < selector.histories.size(); ++step) {
        CHECK(selector.histories[step] == history);
        history.push_back(expected.token_ids[step]);
    }
    for (const iom::oid producer : selector.producers) {
        CHECK(iom::oid_is_token(producer));
    }
}

}  // namespace synthetic_session_test

TEST_CASE(
        "TinyLlama synthetic session integration validates composed "
        "one-and two-layer parity at exact capacity") {
    using namespace synthetic_session_test;
    const std::array<std::size_t, 3> prompt{1, 3, 4};

    for (const std::size_t layers : {std::size_t{1}, std::size_t{2}}) {
        Fixture fixture(layers, "parity-" + std::to_string(layers));
        auto selector = std::make_unique<RecordingGreedySelector>();
        RecordingGreedySelector* observer = selector.get();
        auto session = fixture.load(std::move(selector));
        const ReferenceModel reference(*session);
        const ExpectedTrace expected = expected_trace(reference, prompt, 3);
        REQUIRE_EQ(expected.token_ids.size(), 2);
        REQUIRE(
                expected.stop_reason
                == iom::GenerationStopReason::context_capacity);

        const iom::TokenGenerationResult result =
                session->generate_tokens(prompt, 3);
        compare_trace(expected, *observer, result, prompt, "cached trace");
        CHECK_EQ(observer->logits_rows.size(), 2);
        CHECK_EQ(observer->histories[1].size(), 4);
        CHECK_EQ(observer->histories[1][3], expected.token_ids[0]);
        for (const auto& cache : SessionAccess::caches(*session)) {
            CHECK_EQ(cache.initialized_length, 4);
        }
        CHECK_EQ(SessionAccess::history(*session).size(), kSyntheticCapacity);
        CHECK(SessionAccess::results(*session).empty());
    }
}

TEST_CASE(
        "TinyLlama synthetic session integration isolates completed request "
        "reuse from a fresh session") {
    using namespace synthetic_session_test;
    const std::array<std::size_t, 3> first_prompt{1, 2, 3};
    const std::array<std::size_t, 1> second_prompt{5};

    for (const std::size_t layers : {std::size_t{1}, std::size_t{2}}) {
        Fixture fixture(layers, "reuse-" + std::to_string(layers));
        auto reused_selector = std::make_unique<RecordingGreedySelector>();
        RecordingGreedySelector* reused_observer = reused_selector.get();
        auto reused = fixture.load(std::move(reused_selector));
        const ReferenceModel reference(*reused);

        const ExpectedTrace first_expected =
                expected_trace(reference, first_prompt, 1);
        const iom::TokenGenerationResult first_result =
                reused->generate_tokens(first_prompt, 1);
        compare_trace(
                first_expected, *reused_observer, first_result, first_prompt,
                "first request");
        REQUIRE_EQ(first_result.token_ids.size(), 1);
        const std::vector<std::size_t> saved_first_result =
                first_result.token_ids;

        const ExpectedTrace second_expected =
                expected_trace(reference, second_prompt, 2);
        const std::size_t second_start = reused_observer->logits_rows.size();
        const iom::TokenGenerationResult reused_result =
                reused->generate_tokens(second_prompt, 2);
        RecordingGreedySelector reused_second_observer;
        reused_second_observer.logits_rows.assign(
                reused_observer->logits_rows.begin()
                        + static_cast<std::ptrdiff_t>(second_start),
                reused_observer->logits_rows.end());
        reused_second_observer.histories.assign(
                reused_observer->histories.begin()
                        + static_cast<std::ptrdiff_t>(second_start),
                reused_observer->histories.end());
        reused_second_observer.producers.assign(
                reused_observer->producers.begin()
                        + static_cast<std::ptrdiff_t>(second_start),
                reused_observer->producers.end());
        reused_second_observer.selected_ids.assign(
                reused_observer->selected_ids.begin()
                        + static_cast<std::ptrdiff_t>(second_start),
                reused_observer->selected_ids.end());
        compare_trace(
                second_expected, reused_second_observer, reused_result,
                second_prompt, "reused second request");
        CHECK(reused_observer->histories[second_start] == std::vector<std::size_t>{5});
        CHECK(reused_observer->histories[second_start + 1]
              == std::vector<std::size_t>{5, second_expected.token_ids[0]});
        CHECK(reused_result.stop_reason
              == iom::GenerationStopReason::max_new_tokens);
        CHECK(reused_result.token_ids == second_expected.token_ids);
        CHECK(first_result.token_ids == saved_first_result);

        auto fresh_selector = std::make_unique<RecordingGreedySelector>();
        RecordingGreedySelector* fresh_observer = fresh_selector.get();
        auto fresh = fixture.load(std::move(fresh_selector));
        const iom::TokenGenerationResult fresh_result =
                fresh->generate_tokens(second_prompt, 2);
        compare_trace(
                second_expected, *fresh_observer, fresh_result, second_prompt,
                "fresh second request");
        CHECK(fresh_result.token_ids == reused_result.token_ids);
        CHECK(fresh_result.stop_reason == reused_result.stop_reason);
        REQUIRE_EQ(fresh_observer->logits_rows.size(),
                   reused_second_observer.logits_rows.size());
        for (std::size_t step = 0; step < fresh_observer->logits_rows.size();
             ++step) {
            compare_logits(
                    reused_second_observer.logits_rows[step],
                    fresh_observer->logits_rows[step],
                    "reused/fresh logits");
        }
        for (const auto& cache : SessionAccess::caches(*reused)) {
            CHECK_EQ(cache.initialized_length, 2);
        }
        for (const auto& cache : SessionAccess::caches(*fresh)) {
            CHECK_EQ(cache.initialized_length, 2);
        }
    }
}

TEST_CASE(
        "TinyLlama synthetic session integration recovers after accepted "
        "failure at the poison boundary") {
    using namespace synthetic_session_test;
    const std::array<std::size_t, 3> failed_prompt{1, 3, 4};
    const std::array<std::size_t, 1> recovery_prompt{6};

    for (const std::size_t layers : {std::size_t{1}, std::size_t{2}}) {
        Fixture fixture(layers, "failure-" + std::to_string(layers));
        auto failed_selector = std::make_unique<RecordingGreedySelector>();
        RecordingGreedySelector* failed_observer = failed_selector.get();
        auto failed = fixture.load(std::move(failed_selector));

        iom::cpu_detail::arm_cache_append_wait_observation();
        iom::cpu_detail::arm_cache_append_failure();
        struct ClearFailureLatches {
            void clear() noexcept {
                iom::cpu_detail::clear_cache_append_failure();
                iom::cpu_detail::clear_cache_append_wait_observation();
            }

            ~ClearFailureLatches() { clear(); }
        } clear_latches;

        CHECK_THROWS_AS(
                failed->generate_tokens(failed_prompt, 2),
                std::runtime_error);
        CHECK(failed->poisoned());
        CHECK_EQ(failed->request_length(), failed_prompt.size());
        REQUIRE(!SessionAccess::accepted(*failed).empty());
        // The poisoned request remains inaccessible through normal resource
        // accessors; its accepted operations are retained for safe teardown.
        CHECK_THROWS_AS(SessionAccess::history(*failed), std::logic_error);
        CHECK_THROWS_AS(SessionAccess::results(*failed), std::logic_error);
        // K's accepted wait fails, but the independent V append is still
        // waited.  The CPU seam records each append sequence once even when
        // the session's later drain repeats that wait.
        CHECK_EQ(iom::cpu_detail::observed_cache_append_waits(), 1u);
        CHECK_THROWS_AS(
                failed->generate_tokens(recovery_prompt, 2),
                std::logic_error);
        clear_latches.clear();
        failed.reset();

        auto recovered_selector = std::make_unique<RecordingGreedySelector>();
        RecordingGreedySelector* recovered_observer = recovered_selector.get();
        auto recovered = fixture.load(std::move(recovered_selector));
        const ReferenceModel reference(*recovered);
        const ExpectedTrace expected =
                expected_trace(reference, recovery_prompt, 2);
        const iom::TokenGenerationResult recovered_result =
                recovered->generate_tokens(recovery_prompt, 2);
        compare_trace(
                expected, *recovered_observer, recovered_result,
                recovery_prompt, "recovered request");

        auto fresh_selector = std::make_unique<RecordingGreedySelector>();
        RecordingGreedySelector* fresh_observer = fresh_selector.get();
        auto fresh = fixture.load(std::move(fresh_selector));
        const iom::TokenGenerationResult fresh_result =
                fresh->generate_tokens(recovery_prompt, 2);
        compare_trace(
                expected, *fresh_observer, fresh_result, recovery_prompt,
                "fresh after failure");
        CHECK(fresh_result.token_ids == recovered_result.token_ids);
        CHECK(fresh_result.stop_reason == recovered_result.stop_reason);
        REQUIRE_EQ(fresh_observer->logits_rows.size(),
                   recovered_observer->logits_rows.size());
        for (std::size_t step = 0; step < fresh_observer->logits_rows.size();
             ++step) {
            compare_logits(
                    recovered_observer->logits_rows[step],
                    fresh_observer->logits_rows[step],
                    "recovered/fresh logits");
        }
    }
}

namespace {
using iom::session_detail::MlpStageFailure;
using iom::session_detail::MlpStageParams;

using iom::session_detail::MlpStageViews;
using iom::session_detail::MlpWorkspace;
using iom::session_detail::MlpWorkspaceRequirements;

// Value in every result store before a run, so a store that the stage must not
// publish is distinguishable from every reference value.
constexpr float kMlpUntouched = 64.0F;

class MlpHeapAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t bytes) override {
        return ::operator new(
                std::max<std::size_t>(bytes, 1), std::align_val_t{32});
    }

    void free(void* address) override {
        ::operator delete(address, std::align_val_t{32});
    }

    void reset() override {}
};

struct MlpCpuFixture {
    MlpHeapAllocator allocator;
    std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
};

// Disarms every armed post-acceptance occurrence on scope exit, so an
// interrupted case cannot leak a fault into a later one.
class MlpFailureLatches final {
public:
    MlpFailureLatches() = default;
    ~MlpFailureLatches() {
        iom::cpu_detail::clear_linear_failure();
        iom::cpu_detail::clear_silu_failure();
    }

    MlpFailureLatches(const MlpFailureLatches&) = delete;
    MlpFailureLatches& operator=(const MlpFailureLatches&) = delete;
};

// ---------------------------------------------------------------------------
// BF16 boundary helpers. The comparison policy below is the engine's fixed
// BF16 reference policy: an observed value agrees with the independently
// computed FP64 equation rounded once to BF16 when it stays inside one BF16
// ULP or the fixed absolute/relative allowance.
// ---------------------------------------------------------------------------

[[nodiscard]] std::uint16_t mlp_bf16_bits(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<std::uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}

[[nodiscard]] float mlp_bf16_value(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

[[nodiscard]] float mlp_bf16_round(double value) noexcept {
    return mlp_bf16_value(mlp_bf16_bits(static_cast<float>(value)));
}

[[nodiscard]] double mlp_bf16_ulp(double value) noexcept {
    if (value == 0.0 || std::fpclassify(value) == FP_SUBNORMAL) {
        return std::ldexp(1.0, -133);
    }
    const int exponent = std::ilogb(value);
    if (exponent < -126) {
        return std::ldexp(1.0, -133);
    }
    return std::ldexp(1.0, exponent - 7);
}

[[nodiscard]] bool mlp_agrees(float observed, float expected) noexcept {
    if (std::isnan(expected)) return std::isnan(observed);
    if (std::isinf(expected)) return observed == expected;
    if (!std::isfinite(observed)) return false;
    const double reference = static_cast<double>(expected);
    const double difference =
            std::fabs(static_cast<double>(observed) - reference);
    const double allowance = std::max(
            {mlp_bf16_ulp(reference), std::ldexp(1.0, -7),
             std::ldexp(1.0, -6) * std::fabs(reference)});
    return difference <= allowance;
}

// ---------------------------------------------------------------------------
// Tensor helpers.
// ---------------------------------------------------------------------------

[[nodiscard]] iom::TensorSpec mlp_spec(
        std::span<const std::size_t> extents) {
    return iom::TensorSpec{
            iom::TensorShape(std::vector<std::size_t>(
                    extents.begin(), extents.end())),
            iom::DataType::BF16};
}

[[nodiscard]] iom::TensorSpec mlp_spec(
        std::initializer_list<std::size_t> extents) {
    return mlp_spec(std::span<const std::size_t>(extents));
}

[[nodiscard]] std::vector<std::byte> mlp_encode(
        std::span<const float> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint16_t bits = mlp_bf16_bits(values[index]);
        bytes[2 * index] = static_cast<std::byte>(bits & 0xffu);
        bytes[2 * index + 1] = static_cast<std::byte>((bits >> 8) & 0xffu);
    }
    return bytes;
}

[[nodiscard]] std::vector<float> mlp_read(iom::Tensor& tensor) {
    const std::size_t elements = tensor.view().spec().shape.element_count();
    std::vector<std::byte> bytes(elements * sizeof(std::uint16_t));
    tensor.view().copy_to_host(bytes);
    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const auto low = static_cast<std::uint8_t>(bytes[2 * index]);
        const auto high = static_cast<std::uint8_t>(bytes[2 * index + 1]);
        values[index] = mlp_bf16_value(
                static_cast<std::uint16_t>(low)
                | static_cast<std::uint16_t>(high) << 8);
    }
    return values;
}

void mlp_write(iom::Tensor& tensor, std::span<const float> values) {
    tensor.view().copy_from_host(mlp_encode(values));
}

// Writes the fixed pre-run contents of every result store through the logical
// boundary, leaving tiled padding untouched.
void mlp_fill(iom::Tensor& tensor, float value) {
    const std::size_t elements = tensor.view().spec().shape.element_count();
    mlp_write(tensor, std::vector<float>(elements, value));
}

// Replaces every physical padded row/column element of one tensor with a
// sentinel, so a logical result that reads padding cannot agree with an
// independently computed reference.
void mlp_poison_padding(iom::Tensor& tensor, std::uint16_t sentinel) {
    const iom::TensorSpec spec = tensor.view().spec();
    const std::span<const std::size_t> dimensions = spec.shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const iom::TensorShape padded = spec.standard_padded_shape();
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        planes *= dimensions[axis];
    }
    auto* storage = static_cast<std::byte*>(tensor.view().native_handle());
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < padded.dimension(leading_rank); ++row) {
            for (std::size_t column = 0;
                 column < padded.dimension(leading_rank + 1); ++column) {
                if (row < rows && column < columns) continue;
                const std::size_t slot =
                        iom::detail::standard_plane_slot(
                                spec, plane, row, column);
                storage[2 * slot] = static_cast<std::byte>(sentinel & 0xffu);
                storage[2 * slot + 1] =
                        static_cast<std::byte>((sentinel >> 8) & 0xffu);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One synthetic MLP bank: the logical host inputs, the device tensors that
// hold them, the seven distinct result stores, and the independent reference
// of every operation boundary.
// ---------------------------------------------------------------------------

struct MlpBank {
    std::size_t planes = 1;
    std::size_t rows = 0;
    std::size_t features = 0;
    std::size_t intermediate = 0;
    float epsilon = 1e-5F;

    std::vector<float> x2_values;
    std::vector<float> scale_values;
    std::vector<float> gate_weight_values;
    std::vector<float> up_weight_values;
    std::vector<float> down_weight_values;

    std::unique_ptr<iom::Tensor> staging;
    std::unique_ptr<iom::Tensor> x2;
    std::unique_ptr<iom::Tensor> scale;
    std::unique_ptr<iom::Tensor> gate_weight;
    std::unique_ptr<iom::Tensor> up_weight;
    std::unique_ptr<iom::Tensor> down_weight;
    std::unique_ptr<iom::Tensor> n2;
    std::unique_ptr<iom::Tensor> gate;
    std::unique_ptr<iom::Tensor> up;
    std::unique_ptr<iom::Tensor> activated;
    std::unique_ptr<iom::Tensor> product;
    std::unique_ptr<iom::Tensor> down;
    std::unique_ptr<iom::Tensor> next_x;

    std::vector<float> ref_n2;
    std::vector<float> ref_gate;
    std::vector<float> ref_up;
    std::vector<float> ref_activated;
    std::vector<float> ref_product;
    std::vector<float> ref_down;
    std::vector<float> ref_next_x;
};

// Deterministic nonsymmetric BF16-exact fixture value: every value is `k/8`
// for a small integer `k`, so inputs and weights are exactly representable and
// swapping two operands cannot agree.
[[nodiscard]] float mlp_value(
        std::uint32_t tag, std::uint32_t row, std::uint32_t column) noexcept {
    const std::uint32_t mixed =
            tag * 31u + row * 17u + column * 11u + row * column * 5u;
    return static_cast<float>(static_cast<int>(mixed % 29u) - 14) / 8.0F;
}

[[nodiscard]] std::vector<float> mlp_activation_values(
        std::size_t planes, std::size_t rows, std::size_t width,
        std::uint32_t tag) {
    std::vector<float> values(planes * rows * width);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                // The plane index participates, so an independent leading
                // plane can never be satisfied by broadcast state.
                values[(plane * rows + row) * width + column] = mlp_value(
                        tag + static_cast<std::uint32_t>(plane) * 101u,
                        static_cast<std::uint32_t>(row),
                        static_cast<std::uint32_t>(column));
            }
        }
    }
    return values;
}

// The independent host reference of every stored operation boundary.
void mlp_compute_reference(MlpBank& bank);

[[nodiscard]] MlpBank mlp_make_bank(
        iom::Device& device, std::size_t planes, std::size_t rows,
        std::size_t features, std::size_t intermediate, std::uint32_t tag) {
    MlpBank bank;
    bank.planes = planes;
    bank.rows = rows;
    bank.features = features;
    bank.intermediate = intermediate;

    const std::vector<std::size_t> activation_extents =
            planes == 1
            ? std::vector<std::size_t>{rows, features}
            : std::vector<std::size_t>{planes, rows, features};
    const std::vector<std::size_t> projection_extents =
            planes == 1
            ? std::vector<std::size_t>{rows, intermediate}
            : std::vector<std::size_t>{planes, rows, intermediate};

    bank.x2_values = mlp_activation_values(planes, rows, features, tag);
    bank.scale_values = mlp_activation_values(1, 1, features, tag + 300u);
    bank.gate_weight_values =
            mlp_activation_values(1, intermediate, features, tag + 400u);
    bank.up_weight_values =
            mlp_activation_values(1, intermediate, features, tag + 500u);
    bank.down_weight_values =
            mlp_activation_values(1, features, intermediate, tag + 600u);

    bank.staging = device.create_tensor(mlp_spec(activation_extents));
    bank.x2 = device.create_tensor(mlp_spec(activation_extents));
    bank.scale = device.create_tensor(mlp_spec({1, features}));
    bank.gate_weight =
            device.create_tensor(mlp_spec({intermediate, features}));
    bank.up_weight = device.create_tensor(mlp_spec({intermediate, features}));
    bank.down_weight =
            device.create_tensor(mlp_spec({features, intermediate}));
    bank.n2 = device.create_tensor(mlp_spec(activation_extents));
    bank.gate = device.create_tensor(mlp_spec(projection_extents));
    bank.up = device.create_tensor(mlp_spec(projection_extents));
    bank.activated = device.create_tensor(mlp_spec(projection_extents));
    bank.product = device.create_tensor(mlp_spec(projection_extents));
    bank.down = device.create_tensor(mlp_spec(activation_extents));
    bank.next_x = device.create_tensor(mlp_spec(activation_extents));

    mlp_write(*bank.staging, bank.x2_values);
    mlp_write(*bank.scale, bank.scale_values);
    mlp_write(*bank.gate_weight, bank.gate_weight_values);
    mlp_write(*bank.up_weight, bank.up_weight_values);
    mlp_write(*bank.down_weight, bank.down_weight_values);
    mlp_fill(*bank.n2, kMlpUntouched);
    mlp_fill(*bank.gate, kMlpUntouched);
    mlp_fill(*bank.up, kMlpUntouched);
    mlp_fill(*bank.activated, kMlpUntouched);
    mlp_fill(*bank.product, kMlpUntouched);
    mlp_fill(*bank.down, kMlpUntouched);
    mlp_fill(*bank.next_x, kMlpUntouched);

    mlp_compute_reference(bank);
    return bank;
}

// The independent host reference of every stored operation boundary: the
// equations run in FP64, each boundary is rounded once to the BF16 store, and
// every consumer reads that stored value.
void mlp_compute_reference(MlpBank& bank) {
    const std::size_t features = bank.features;
    const std::size_t intermediate = bank.intermediate;
    const std::size_t rows = bank.rows;
    const double epsilon = static_cast<double>(bank.epsilon);
    const std::size_t activation_plane = rows * features;
    const std::size_t projection_plane = rows * intermediate;
    bank.ref_n2.assign(bank.planes * activation_plane, 0.0F);
    bank.ref_gate.assign(bank.planes * projection_plane, 0.0F);
    bank.ref_up.assign(bank.planes * projection_plane, 0.0F);
    bank.ref_activated.assign(bank.planes * projection_plane, 0.0F);
    bank.ref_product.assign(bank.planes * projection_plane, 0.0F);
    bank.ref_down.assign(bank.planes * activation_plane, 0.0F);
    bank.ref_next_x.assign(bank.planes * activation_plane, 0.0F);

    for (std::size_t plane = 0; plane < bank.planes; ++plane) {
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t activation_base =
                    plane * activation_plane + row * features;
            const std::size_t projection_base =
                    plane * projection_plane + row * intermediate;

            double square_sum = 0.0;
            for (std::size_t feature = 0; feature < features; ++feature) {
                const double value = bank.x2_values[activation_base + feature];
                square_sum += value * value;
            }
            const double inverse = 1.0
                    / std::sqrt(
                            square_sum / static_cast<double>(features)
                            + epsilon);
            for (std::size_t feature = 0; feature < features; ++feature) {
                bank.ref_n2[activation_base + feature] = mlp_bf16_round(
                        bank.x2_values[activation_base + feature] * inverse
                        * bank.scale_values[feature]);
            }

            for (std::size_t column = 0; column < intermediate; ++column) {
                double gate_sum = 0.0;
                double up_sum = 0.0;
                for (std::size_t feature = 0; feature < features; ++feature) {
                    const double normalized =
                            bank.ref_n2[activation_base + feature];
                    gate_sum += normalized
                            * bank.gate_weight_values[column * features
                                                      + feature];
                    up_sum += normalized
                            * bank.up_weight_values[column * features + feature];
                }
                bank.ref_gate[projection_base + column] =
                        mlp_bf16_round(gate_sum);
                bank.ref_up[projection_base + column] =
                        mlp_bf16_round(up_sum);
            }

            for (std::size_t column = 0; column < intermediate; ++column) {
                const double stored_gate = bank.ref_gate[projection_base + column];
                bank.ref_activated[projection_base + column] = mlp_bf16_round(
                        stored_gate / (1.0 + std::exp(-stored_gate)));
                bank.ref_product[projection_base + column] = mlp_bf16_round(
                        static_cast<double>(
                                bank.ref_activated[projection_base + column])
                        * static_cast<double>(
                                bank.ref_up[projection_base + column]));
            }

            for (std::size_t feature = 0; feature < features; ++feature) {
                double down_sum = 0.0;
                for (std::size_t column = 0; column < intermediate;
                     ++column) {
                    down_sum +=
                            static_cast<double>(
                                    bank.ref_product[projection_base + column])
                            * bank.down_weight_values[feature * intermediate
                                                      + column];
                }
                bank.ref_down[activation_base + feature] =
                        mlp_bf16_round(down_sum);
                bank.ref_next_x[activation_base + feature] = mlp_bf16_round(
                        bank.x2_values[activation_base + feature]
                        + static_cast<double>(
                                bank.ref_down[activation_base + feature]));
            }
        }
    }
}

[[nodiscard]] MlpStageViews mlp_make_views(
        iom::Tensor& x2, iom::Tensor& scale, iom::Tensor& gate_weight,
        iom::Tensor& up_weight, iom::Tensor& down_weight, iom::Tensor& n2,
        iom::Tensor& gate, iom::Tensor& up, iom::Tensor& activated,
        iom::Tensor& product, iom::Tensor& down, iom::Tensor& next_x) {
    return MlpStageViews{
            x2.view(),     scale.view(),   gate_weight.view(),
            up_weight.view(), down_weight.view(), n2.view(),
            gate.view(),   up.view(),      activated.view(),
            product.view(), down.view(),   next_x.view()};
}

[[nodiscard]] MlpStageViews mlp_views(const MlpBank& bank) {
    return mlp_make_views(
            *bank.x2, *bank.scale, *bank.gate_weight, *bank.up_weight,
            *bank.down_weight, *bank.n2, *bank.gate, *bank.up, *bank.activated,
            *bank.product, *bank.down, *bank.next_x);
}

[[nodiscard]] MlpStageParams mlp_params(const MlpBank& bank) noexcept {
    return MlpStageParams{
            bank.rows, bank.features, bank.intermediate, bank.epsilon};
}

// The request-scoped workspace of the two actual run banks. CPU reports the
// conventional zero requirement for every member, so the resolved members are
// the empty views the zero-workspace operations accept.
[[nodiscard]] MlpWorkspace mlp_workspace(
        iom::DeviceOps& operations, const MlpBank& prefill,
        const MlpBank& decode) {
    const MlpStageViews prefill_views = mlp_views(prefill);
    const MlpStageViews decode_views = mlp_views(decode);
    const MlpWorkspaceRequirements requirements =
            iom::session_detail::mlp_workspace_requirements(
                    operations, prefill_views, decode_views);
    return iom::session_detail::resolve_mlp_workspace(
            operations.device(), requirements, iom::RawWorkspaceView{});
}

// Submits the bank's first-residual producer: the exact copy of the
// caller-owned staging tensor into the residual store, which the stage must
// wait before it reads `x2`.
[[nodiscard]] iom::oid mlp_submit_first_residual(
        MlpBank& bank, iom::DeviceOps& operations) {
    const iom::oid producer =
            operations.copy(bank.staging->view(), bank.x2->view());
    REQUIRE(iom::oid_is_token(producer));
    return producer;
}

void mlp_drive(
        MlpBank& bank, iom::DeviceOps& operations, iom::oid producer,
        MlpWorkspace& workspace, MlpStageFailure& failure) {
    MlpStageViews views = mlp_views(bank);
    const MlpStageParams params = mlp_params(bank);
    const std::span<const iom::oid> readiness(&producer, 1);
    iom::session_detail::run_mlp_stage(
            operations, views, params, workspace, readiness, failure);
}

// Runs the stage and returns the failure message, or an empty string when the
// stage completed successfully.
[[nodiscard]] std::string mlp_failure_message(
        MlpBank& bank, iom::DeviceOps& operations, iom::oid producer,
        MlpWorkspace& workspace, MlpStageFailure& failure) {
    try {
        mlp_drive(bank, operations, producer, workspace, failure);
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

void mlp_check(
        iom::Tensor& store, std::span<const float> expected,
        const char* boundary) {
    const std::vector<float> observed = mlp_read(store);
    REQUIRE(observed.size() == expected.size());
    std::string first_mismatch;
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < observed.size(); ++index) {
        if (mlp_agrees(observed[index], expected[index])) continue;
        ++mismatches;
        if (first_mismatch.empty()) {
            first_mismatch = std::string(boundary) + " element "
                    + std::to_string(index) + " observed "
                    + std::to_string(observed[index]) + " expected "
                    + std::to_string(expected[index]);
        }
    }
    CHECK_MESSAGE(mismatches == 0, first_mismatch);
}

// One result store must still hold its exact pre-run contents after a failed
// stage: no dependent stage was published.
void mlp_check_untouched(iom::Tensor& store, const char* boundary) {
    const std::vector<float> observed = mlp_read(store);
    const std::uint16_t sentinel = mlp_bf16_bits(kMlpUntouched);
    std::string first_mismatch;
    for (std::size_t index = 0; index < observed.size(); ++index) {
        if (mlp_bf16_bits(observed[index]) == sentinel) continue;
        if (first_mismatch.empty()) {
            first_mismatch = std::string(boundary) + " element "
                    + std::to_string(index) + " changed to "
                    + std::to_string(observed[index]);
        }
    }
    CHECK_MESSAGE(first_mismatch.empty(), first_mismatch);
}

void mlp_check_reference(MlpBank& bank) {
    mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
    mlp_check(*bank.gate, bank.ref_gate, "gate projection");
    mlp_check(*bank.up, bank.ref_up, "up projection");
    mlp_check(*bank.activated, bank.ref_activated, "activated gate");
    mlp_check(*bank.product, bank.ref_product, "product");
    mlp_check(*bank.down, bank.ref_down, "down projection");
    mlp_check(*bank.next_x, bank.ref_next_x, "next residual");
}

// The first residual and every weight are read only: the stage never rewrites
// its inputs.
void mlp_check_inputs(const MlpBank& bank) {
    CHECK(mlp_read(*bank.x2) == bank.x2_values);
    CHECK(mlp_read(*bank.scale) == bank.scale_values);
    CHECK(mlp_read(*bank.gate_weight) == bank.gate_weight_values);
    CHECK(mlp_read(*bank.up_weight) == bank.up_weight_values);
    CHECK(mlp_read(*bank.down_weight) == bank.down_weight_values);
}

void mlp_fill_stores(MlpBank& bank) {
    mlp_fill(*bank.n2, kMlpUntouched);
    mlp_fill(*bank.gate, kMlpUntouched);
    mlp_fill(*bank.up, kMlpUntouched);
    mlp_fill(*bank.activated, kMlpUntouched);
    mlp_fill(*bank.product, kMlpUntouched);
    mlp_fill(*bank.down, kMlpUntouched);
    mlp_fill(*bank.next_x, kMlpUntouched);
}

[[nodiscard]] std::vector<float> mlp_store_image(const MlpBank& bank) {
    std::vector<float> image;
    const std::unique_ptr<iom::Tensor>* const stores[7] = {
            &bank.n2, &bank.gate, &bank.up, &bank.activated, &bank.product,
            &bank.down, &bank.next_x};
    for (const std::unique_ptr<iom::Tensor>* store : stores) {
        const std::vector<float> values = mlp_read(**store);
        image.insert(image.end(), values.begin(), values.end());
    }
    return image;
}

// ---------------------------------------------------------------------------
// Independent BF16 codec and host arithmetic
// ---------------------------------------------------------------------------

// Round-to-nearest-even narrowing of one finite float to BF16 bits. The
// fixtures stay finite and small, so NaN and infinity handling is not
// exercised here.
[[nodiscard]] std::uint16_t encode_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

[[nodiscard]] float decode_bf16(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

[[nodiscard]] float round_bf16(float value) {
    return decode_bf16(encode_bf16(value));
}

// Distance in BF16 units of the expected magnitude, with an absolute floor so
// elements that land close to zero are compared against the vector's own
// scale instead of an arbitrarily fine local unit.
[[nodiscard]] bool within_bf16_units(
        float actual, float expected, int units, float floor) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        return false;
    }
    const float ulp = expected == 0.0f
            ? 0.0f
            : std::ldexp(1.0f, std::ilogb(expected) - 7);
    return std::fabs(actual - expected)
            <= static_cast<float>(units) * ulp + floor;
}

void check_close(const std::vector<float>& actual,
                 const std::vector<float>& expected, int units, float floor,
                 const std::string& label) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK_MESSAGE(
                within_bf16_units(actual[index], expected[index], units, floor),
                label << " element " << index << ": " << actual[index]
                      << " vs " << expected[index]);
    }
}

void check_identical(const std::vector<float>& actual,
                     const std::vector<float>& expected,
                     const std::string& label) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK_MESSAGE(actual[index] == expected[index],
                      label << " element " << index << ": " << actual[index]
                            << " vs " << expected[index]);
    }
}

// Deterministic small values that are exactly representable in BF16.
class Values {
public:
    explicit Values(std::uint64_t seed) : state_(seed) {}

    float next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        const std::uint64_t mantissa = (state_ >> 11) & 0x1FFFFFull;
        const float unit =
                static_cast<float>(mantissa) / static_cast<float>(0x1FFFFF);
        return round_bf16((unit * 2.0f - 1.0f) * 0.75f);
    }

private:
    std::uint64_t state_;
};

// ---------------------------------------------------------------------------
// Fixed synthetic geometry and supplied tensor inventory
// ---------------------------------------------------------------------------

struct Geometry {
    std::size_t query_heads = 4;
    std::size_t kv_heads = 2;
    std::size_t head_width = 3;  // non-tile-aligned head width
    std::size_t rows = 2;
    std::size_t capacity = 7;
    std::size_t offset = 2;  // nonzero absolute cache prefix

    [[nodiscard]] std::size_t features() const noexcept {
        return query_heads * head_width;
    }
};

// One CPU device, one in-order queue, and the complete supplied tensor set of
// one stage call. Host copies of every seeded logical value stay available for
// the independent reference.
struct StageFixture {
    explicit StageFixture(const Geometry& geometry)
        : geometry(geometry),
          arena(1u << 20),
          allocator(arena.data(), arena.size(), 32),
          device(iom::make_cpu_device(allocator)),
          queue(device->create_ops()),
          rotated_q(create({geometry.query_heads, geometry.rows,
                            geometry.head_width})),
          rotated_k(
                  create({geometry.kv_heads, geometry.rows,
                          geometry.head_width})),
          rotated_v(
                  create({geometry.kv_heads, geometry.rows,
                          geometry.head_width})),
          k_cache(create({geometry.kv_heads, geometry.capacity,
                          geometry.head_width})),
          v_cache(create({geometry.kv_heads, geometry.capacity,
                          geometry.head_width})),
          residual_input(create({geometry.rows, geometry.features()})),
          o_weight(create({geometry.features(), geometry.features()})),
          attention_merged(create({geometry.rows, geometry.features()})),
          attention_output(create({geometry.rows, geometry.features()})),
          residual_output(create({geometry.rows, geometry.features()})) {
        const iom::WorkspaceRequirements requirements =
                queue->sdpa_workspace_requirements(
                        rotated_q->view(), k_cache->view(), v_cache->view(),
                        attention_merged->view(), geometry.offset,
                        geometry.offset + geometry.rows);
        sdpa_workspace = device->create_workspace(requirements.bytes);
    }

    [[nodiscard]] std::unique_ptr<iom::Tensor> create(
            std::vector<std::size_t> shape) const {
        return device->create_tensor(iom::TensorSpec{
                iom::TensorShape{std::move(shape)}, iom::DataType::BF16});
    }

    void upload(iom::Tensor& tensor, const std::vector<float>& values) {
        const iom::TensorSpec& spec = tensor.view().spec();
        REQUIRE(values.size() == spec.shape.element_count());
        std::vector<std::byte> bytes(spec.logical_nbytes());
        for (std::size_t index = 0; index < values.size(); ++index) {
            const std::uint16_t bits = encode_bf16(values[index]);
            std::memcpy(bytes.data() + index * 2, &bits, sizeof(bits));
        }
        tensor.view().copy_from_host(bytes);
    }

    [[nodiscard]] std::vector<float> read(const iom::Tensor& tensor) const {
        const iom::TensorSpec& spec = tensor.view().spec();
        std::vector<std::byte> bytes(spec.logical_nbytes());
        tensor.view().copy_to_host(bytes);
        std::vector<float> values(bytes.size() / 2);
        for (std::size_t index = 0; index < values.size(); ++index) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, bytes.data() + index * 2, sizeof(bits));
            values[index] = decode_bf16(bits);
        }
        return values;
    }

    // Seeds every supplied tensor: each cache keeps a published history prefix
    // below `offset`, a distinctive sentinel across the appended row window,
    // and a different sentinel through the uninitialized capacity tail.
    void seed_all(std::uint64_t seed) {
        Values generator(seed);
        rotated_q_values = sample(generator, rotated_q->view());
        rotated_k_values = sample(generator, rotated_k->view());
        rotated_v_values = sample(generator, rotated_v->view());
        residual_input_values = sample(generator, residual_input->view());
        o_weight_values = sample(generator, o_weight->view());

        k_cache_values.assign(k_cache->view().spec().shape.element_count(),
                              kHistorySentinel);
        v_cache_values.assign(v_cache->view().spec().shape.element_count(),
                              kHistorySentinel);
        for (std::size_t head = 0; head < geometry.kv_heads; ++head) {
            for (std::size_t row = 0; row < geometry.capacity; ++row) {
                for (std::size_t feature = 0; feature < geometry.head_width;
                     ++feature) {
                    const std::size_t index = cache_index(head, row, feature);
                    const bool history = row < geometry.offset;
                    const bool appended = row >= geometry.offset
                            && row < geometry.offset + geometry.rows;
                    k_cache_values[index] = history
                            ? generator.next()
                            : (appended ? kAppendedSentinel : kTailSentinel);
                    v_cache_values[index] = history
                            ? generator.next()
                            : (appended ? kAppendedSentinel : kTailSentinel);
                }
            }
        }

        attention_merged_values.assign(
                attention_merged->view().spec().shape.element_count(),
                kOutputSentinel);
        attention_output_values.assign(
                attention_output->view().spec().shape.element_count(),
                kOutputSentinel);
        residual_output_values.assign(
                residual_output->view().spec().shape.element_count(),
                kOutputSentinel);

        upload_all();
    }

    void upload_all() {
        upload(*rotated_q, rotated_q_values);
        upload(*rotated_k, rotated_k_values);
        upload(*rotated_v, rotated_v_values);
        upload(*k_cache, k_cache_values);
        upload(*v_cache, v_cache_values);
        upload(*residual_input, residual_input_values);
        upload(*o_weight, o_weight_values);
        upload(*attention_merged, attention_merged_values);
        upload(*attention_output, attention_output_values);
        upload(*residual_output, residual_output_values);
    }

    [[nodiscard]] std::size_t cache_index(
            std::size_t head, std::size_t row,
            std::size_t feature) const noexcept {
        return head * geometry.capacity * geometry.head_width
                + row * geometry.head_width + feature;
    }

    [[nodiscard]] std::vector<float> sample(
            Values& generator, const iom::TensorView& view) {
        std::vector<float> values(view.spec().shape.element_count());
        for (float& value : values) {
            value = generator.next();
        }
        return values;
    }

    struct Overrides {
        std::optional<std::size_t> offset;
        std::optional<std::size_t> rows;
        std::optional<std::size_t> capacity;
        const iom::TensorView* rotated_q = nullptr;
        const iom::TensorView* rotated_k = nullptr;
        const iom::TensorView* rotated_v = nullptr;
        const iom::TensorView* k_cache = nullptr;
        const iom::TensorView* v_cache = nullptr;
        const iom::TensorView* residual_input = nullptr;
        const iom::TensorView* attention_merged = nullptr;
        const iom::TensorView* attention_output = nullptr;
        const iom::TensorView* residual_output = nullptr;
    };

    [[nodiscard]] iom::session_detail::CacheAttentionStageRequest make_request()
            const {
        return make_request(Overrides{});
    }

    [[nodiscard]] iom::session_detail::CacheAttentionStageRequest make_request(
            const Overrides& overrides) const {
        return iom::session_detail::CacheAttentionStageRequest{
                .rotated_q = overrides.rotated_q != nullptr
                        ? *overrides.rotated_q
                        : rotated_q->view(),
                .rotated_k = overrides.rotated_k != nullptr
                        ? *overrides.rotated_k
                        : rotated_k->view(),
                .rotated_v = overrides.rotated_v != nullptr
                        ? *overrides.rotated_v
                        : rotated_v->view(),
                .k_cache = overrides.k_cache != nullptr
                        ? *overrides.k_cache
                        : k_cache->view(),
                .v_cache = overrides.v_cache != nullptr
                        ? *overrides.v_cache
                        : v_cache->view(),
                .residual_input = overrides.residual_input != nullptr
                        ? *overrides.residual_input
                        : residual_input->view(),
                .o_weight = o_weight->view(),
                .attention_merged = overrides.attention_merged != nullptr
                        ? *overrides.attention_merged
                        : attention_merged->view(),
                .attention_output = overrides.attention_output != nullptr
                        ? *overrides.attention_output
                        : attention_output->view(),
                .residual_output = overrides.residual_output != nullptr
                        ? *overrides.residual_output
                        : residual_output->view(),
                .a = overrides.offset.value_or(geometry.offset),
                .R = overrides.rows.value_or(geometry.rows),
                .C = overrides.capacity.value_or(geometry.capacity)};
    }

    [[nodiscard]] iom::RawWorkspaceView workspace() const {
        return sdpa_workspace->view();
    }

    [[nodiscard]] iom::DeviceOps& ops() const {
        return *queue;
    }

    static constexpr float kHistorySentinel = -9.5f;
    static constexpr float kAppendedSentinel = 3.5f;
    static constexpr float kTailSentinel = -7.5f;
    static constexpr float kOutputSentinel = 9.25f;

    Geometry geometry;
    std::vector<std::byte> arena;
    iom::ListAllocator allocator;
    std::unique_ptr<iom::Device> device;
    std::unique_ptr<iom::DeviceOps> queue;
    std::unique_ptr<iom::Tensor> rotated_q;
    std::unique_ptr<iom::Tensor> rotated_k;
    std::unique_ptr<iom::Tensor> rotated_v;
    std::unique_ptr<iom::Tensor> k_cache;
    std::unique_ptr<iom::Tensor> v_cache;
    std::unique_ptr<iom::Tensor> residual_input;
    std::unique_ptr<iom::Tensor> o_weight;
    std::unique_ptr<iom::Tensor> attention_merged;
    std::unique_ptr<iom::Tensor> attention_output;
    std::unique_ptr<iom::Tensor> residual_output;
    std::unique_ptr<iom::RawWorkspace> sdpa_workspace;

    std::vector<float> rotated_q_values;
    std::vector<float> rotated_k_values;
    std::vector<float> rotated_v_values;
    std::vector<float> k_cache_values;
    std::vector<float> v_cache_values;
    std::vector<float> residual_input_values;
    std::vector<float> o_weight_values;
    std::vector<float> attention_merged_values;
    std::vector<float> attention_output_values;
    std::vector<float> residual_output_values;
};

// ---------------------------------------------------------------------------
// Independent host reference of the documented stage equations
// ---------------------------------------------------------------------------

// Causal grouped-query SDPA over the published prefix `length`: scores and the
// max-subtracted softmax in FP32, one BF16 probability boundary, and one BF16
// store per merged output element.
[[nodiscard]] std::vector<float> reference_merged(
        const Geometry& geometry, std::size_t offset,
        const std::vector<float>& queries,
        const std::vector<float>& k_cache,
        const std::vector<float>& v_cache, std::size_t length) {
    const std::size_t heads = geometry.query_heads;
    const std::size_t kv_heads = geometry.kv_heads;
    const std::size_t width = geometry.head_width;
    const std::size_t rows = geometry.rows;
    const std::size_t features = geometry.features();
    const std::size_t group = heads / kv_heads;
    std::vector<float> merged(rows * features, 0.0f);
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t kv_head = head / group;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t visible = std::min(length, offset + row + 1);
            std::vector<float> scores(visible, 0.0f);
            for (std::size_t token = 0; token < visible; ++token) {
                float dot = 0.0f;
                for (std::size_t feature = 0; feature < width; ++feature) {
                    dot += queries[(head * rows + row) * width + feature]
                           * k_cache[(kv_head * geometry.capacity + token)
                                             * width
                                     + feature];
                }
                scores[token] = dot / std::sqrt(static_cast<float>(width));
            }
            float maximum = scores[0];
            for (const float score : scores) {
                maximum = std::max(maximum, score);
            }
            float sum = 0.0f;
            std::vector<float> probabilities(visible, 0.0f);
            for (std::size_t token = 0; token < visible; ++token) {
                probabilities[token] = std::exp(scores[token] - maximum);
                sum += probabilities[token];
            }
            for (float& probability : probabilities) {
                probability = round_bf16(probability / sum);
            }
            for (std::size_t feature = 0; feature < width; ++feature) {
                float accumulated = 0.0f;
                for (std::size_t token = 0; token < visible; ++token) {
                    accumulated +=
                            probabilities[token]
                            * v_cache[(kv_head * geometry.capacity + token)
                                              * width
                                      + feature];
                }
                merged[row * features + head * width + feature] =
                        round_bf16(accumulated);
            }
        }
    }
    return merged;
}

// Ordinary `[out,in]` output projection of one merged attention matrix.
[[nodiscard]] std::vector<float> reference_projected(
        const Geometry& geometry, const std::vector<float>& merged,
        const std::vector<float>& weight) {
    const std::size_t rows = geometry.rows;
    const std::size_t features = geometry.features();
    std::vector<float> projected(rows * features, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < features; ++out) {
            float accumulated = 0.0f;
            for (std::size_t feature = 0; feature < features; ++feature) {
                accumulated += merged[row * features + feature]
                               * weight[out * features + feature];
            }
            projected[row * features + out] = round_bf16(accumulated);
        }
    }
    return projected;
}

[[nodiscard]] std::vector<float> reference_residual(
        const std::vector<float>& input,
        const std::vector<float>& projected) {
    REQUIRE(input.size() == projected.size());
    std::vector<float> residual(input.size(), 0.0f);
    for (std::size_t index = 0; index < input.size(); ++index) {
        residual[index] = round_bf16(input[index] + projected[index]);
    }
    return residual;
}

// One cache with its appended row window replaced by `rows`, as [Hkv,R,D].
[[nodiscard]] std::vector<float> with_appended_rows(
        const Geometry& geometry, const std::vector<float>& cache,
        const std::vector<float>& rows) {
    std::vector<float> expected = cache;
    for (std::size_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::size_t row = 0; row < geometry.rows; ++row) {
            for (std::size_t feature = 0; feature < geometry.head_width;
                 ++feature) {
                expected[head * geometry.capacity * geometry.head_width
                         + (geometry.offset + row) * geometry.head_width
                         + feature] =
                        rows[(head * geometry.rows + row) * geometry.head_width
                             + feature];
            }
        }
    }
    return expected;
}

// The appended row window of one head-planar cache, in logical order.
[[nodiscard]] std::vector<float> cache_window(
        const Geometry& geometry, const std::vector<float>& rows) {
    const std::size_t width = geometry.head_width;
    std::vector<float> window(
            geometry.kv_heads * geometry.rows * width, 0.0f);
    for (std::size_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::size_t row = 0; row < geometry.rows; ++row) {
            for (std::size_t feature = 0; feature < width; ++feature) {
                window[(head * geometry.rows + row) * width + feature] =
                        rows[head * geometry.capacity * width
                             + (geometry.offset + row) * width + feature];
            }
        }
    }
    return window;
}

// Runs the private stage once and reports the mutated state plus the value of
// every supplied store afterwards.
struct StageOutcome {
    iom::session_detail::CacheAttentionStageState state{};
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    std::vector<float> merged;
    std::vector<float> projected;
    std::vector<float> residual;
    std::exception_ptr failure;
};

[[nodiscard]] StageOutcome run_stage(
        StageFixture& fixture,
        iom::session_detail::CacheAttentionStageRequest& request,
        iom::RawWorkspaceView workspace) {
    StageOutcome outcome;
    outcome.state.initialized_length = request.a;
    try {
        iom::session_detail::run_cache_attention_stage(
                fixture.ops(), request, workspace, outcome.state);
    } catch (...) {
        outcome.failure = std::current_exception();
    }
    outcome.k_cache = fixture.read(*fixture.k_cache);
    outcome.v_cache = fixture.read(*fixture.v_cache);
    outcome.merged = fixture.read(*fixture.attention_merged);
    outcome.projected = fixture.read(*fixture.attention_output);
    outcome.residual = fixture.read(*fixture.residual_output);
    return outcome;
}

// The largest scratch requirement of the three sequential downstream
// operations, computed exactly as the stage does from the actual operands.
[[nodiscard]] iom::WorkspaceRequirements combined_requirement(
        StageFixture& fixture) {
    const iom::session_detail::CacheAttentionStageRequest request =
            fixture.make_request();
    const iom::WorkspaceRequirements sdpa =
            fixture.ops().sdpa_workspace_requirements(
                    request.rotated_q, request.k_cache, request.v_cache,
                    request.attention_merged, request.a,
                    request.a + request.R);
    const iom::WorkspaceRequirements projection =
            fixture.ops().linear_workspace_requirements(
                    request.attention_merged, request.o_weight,
                    request.attention_output, 0, request.R,
                    iom::LinearOutputLayout::ordinary, 1,
                    fixture.geometry.features());
    const iom::WorkspaceRequirements residual =
            fixture.ops().add_workspace_requirements(
                    request.residual_input, request.attention_output,
                    request.residual_output);
    return {std::max({sdpa.bytes, projection.bytes, residual.bytes}),
            std::max({sdpa.alignment, projection.alignment,
                      residual.alignment})};
}

}  // namespace

TEST_CASE("TinyLlama MLP stage matches an independent reference for prefill and R1 decode") {
    struct Shape {
        std::size_t rows;
        std::size_t features;
        std::size_t intermediate;
        std::uint32_t tag;
    };
    // A non-tile prefill run, the fixed one-row decode run, and a run past the
    // 16x16 tile row boundary.
    const Shape shapes[] = {
            {15, 8, 12, 1}, {1, 8, 12, 2}, {17, 8, 12, 3}};

    for (const Shape& shape : shapes) {
        MlpCpuFixture fixture;
        MlpBank bank = mlp_make_bank(
                *fixture.device, 1, shape.rows, shape.features,
                shape.intermediate, shape.tag);
        auto operations = fixture.device->create_ops();
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        MlpStageFailure failure;
        mlp_drive(bank, *operations, producer, workspace, failure);

        CHECK_FALSE(failure.poisoned());
        mlp_check_reference(bank);
        mlp_check_inputs(bank);
    }
}

TEST_CASE("TinyLlama MLP stage isolates tiled padding and independent leading planes") {
    MlpCpuFixture fixture;
    // Two identical banks of independent leading planes whose physical padding
    // carries different sentinels: one is the nonfinite pattern a padding read
    // would poison, the other a finite pattern.
    MlpBank poisoned = mlp_make_bank(*fixture.device, 2, 15, 8, 12, 4);
    MlpBank finite = mlp_make_bank(*fixture.device, 2, 15, 8, 12, 4);
    const std::unique_ptr<iom::Tensor>* const poisoned_stores[7] = {
            &poisoned.n2, &poisoned.gate, &poisoned.up, &poisoned.activated,
            &poisoned.product, &poisoned.down, &poisoned.next_x};
    const std::unique_ptr<iom::Tensor>* const finite_stores[7] = {
            &finite.n2, &finite.gate, &finite.up, &finite.activated,
            &finite.product, &finite.down, &finite.next_x};
    for (std::size_t index = 0; index < 7; ++index) {
        mlp_poison_padding(**poisoned_stores[index], 0x7F80u);
        mlp_poison_padding(**finite_stores[index], 0x3F00u);
    }
    // The read-only operands carry sentinels too, so a weight or activation
    // result that consumes padded rows or features cannot agree either.
    const std::unique_ptr<iom::Tensor>* const poisoned_inputs[5] = {
            &poisoned.x2, &poisoned.scale, &poisoned.gate_weight,
            &poisoned.up_weight, &poisoned.down_weight};
    const std::unique_ptr<iom::Tensor>* const finite_inputs[5] = {
            &finite.x2, &finite.scale, &finite.gate_weight, &finite.up_weight,
            &finite.down_weight};
    for (std::size_t index = 0; index < 5; ++index) {
        mlp_poison_padding(**poisoned_inputs[index], 0x7F80u);
        mlp_poison_padding(**finite_inputs[index], 0x7F00u);
    }

    auto operations = fixture.device->create_ops();
    const iom::oid poisoned_producer =
            mlp_submit_first_residual(poisoned, *operations);
    const iom::oid finite_producer =
            mlp_submit_first_residual(finite, *operations);
    MlpWorkspace workspace = mlp_workspace(*operations, poisoned, finite);
    MlpStageFailure poisoned_failure;
    MlpStageFailure finite_failure;
    mlp_drive(poisoned, *operations, poisoned_producer, workspace, poisoned_failure);
    mlp_drive(finite, *operations, finite_producer, workspace, finite_failure);

    CHECK_FALSE(poisoned_failure.poisoned());
    CHECK_FALSE(finite_failure.poisoned());
    mlp_check_reference(poisoned);
    mlp_check_reference(finite);
    CHECK(mlp_store_image(poisoned) == mlp_store_image(finite));
    // Both planes carry independent state: the leading plane is not broadcast.
    CHECK(mlp_read(*poisoned.gate) == poisoned.ref_gate);
}

TEST_CASE("TinyLlama MLP stage rejects inconsistent views, aliases, and readiness before submission") {
    MlpCpuFixture fixture;
    MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 5);
    auto operations = fixture.device->create_ops();
    MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
    const MlpStageParams params = mlp_params(bank);
    const iom::oid producer = mlp_submit_first_residual(bank, *operations);

    SUBCASE("a mismatched weight view fails before any submission") {
        auto wrong_weight = fixture.device->create_tensor(
                mlp_spec({bank.intermediate + 1, bank.features}));
        MlpStageViews views = mlp_make_views(
                *bank.x2, *bank.scale, *wrong_weight, *bank.up_weight,
                *bank.down_weight, *bank.n2, *bank.gate, *bank.up,
                *bank.activated, *bank.product, *bank.down, *bank.next_x);
        MlpStageFailure failure;
        const std::span<const iom::oid> readiness(&producer, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        CHECK_FALSE(failure.poisoned());
        mlp_check_untouched(*bank.n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    SUBCASE("an in-place activation alias fails instead of overwriting a store") {
        MlpStageViews views = mlp_make_views(
                *bank.x2, *bank.scale, *bank.gate_weight, *bank.up_weight,
                *bank.down_weight, *bank.n2, *bank.gate, *bank.up, *bank.gate,
                *bank.product, *bank.down, *bank.next_x);
        MlpStageFailure failure;
        const std::span<const iom::oid> readiness(&producer, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        CHECK(failure.poisoned());
        // The dependent stores were never published, and the gate store was
        // never overwritten in place by the activation.
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    SUBCASE("an invalid readiness token is rejected without waiting it") {
        MlpStageFailure failure;
        MlpStageViews views = mlp_views(bank);
        const iom::oid invalid = 0;
        const std::span<const iom::oid> readiness(&invalid, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        CHECK_FALSE(failure.poisoned());
        mlp_check_untouched(*bank.n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    SUBCASE("a producer from another queue is drained as a failure") {
        auto foreign_operations = fixture.device->create_ops();
        const iom::oid foreign = foreign_operations->copy(
                bank.staging->view(), bank.x2->view());
        REQUIRE(iom::oid_is_token(foreign));
        MlpStageFailure failure;
        MlpStageViews views = mlp_views(bank);
        const std::span<const iom::oid> readiness(&foreign, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        // A readiness producer that fails its wait is an accepted-work
        // failure: the execution stays poisoned and is not reused.
        CHECK(failure.poisoned());
        mlp_check_untouched(*bank.n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    SUBCASE("the checked workspace requirement keeps gate and up disjoint") {
        // The two actual banks contribute their checked member maxima; CPU
        // reports the conventional zero requirement, so the resolved members
        // stay the empty views the zero-workspace operations accept.
        const MlpStageViews prefill_views = mlp_views(bank);
        const MlpStageViews decode_views = mlp_views(bank);
        const MlpWorkspaceRequirements requirements =
                iom::session_detail::mlp_workspace_requirements(
                        *operations, prefill_views, decode_views);
        CHECK(requirements.total() == iom::WorkspaceRequirements{0, 1});
        const MlpWorkspace resolved =
                iom::session_detail::resolve_mlp_workspace(
                        *fixture.device, requirements, iom::RawWorkspaceView{});
        CHECK(resolved.gate.empty());
        CHECK(resolved.up.empty());
        CHECK(resolved.mul.empty());
        CHECK(resolved.down.empty());
        CHECK(resolved.residual.empty());

        // A backend with positive member requirements reserves two permanent
        // slices at the fixed 32-byte granularity: the reused gate slice
        // (shared by mul, down, and the second residual after their producer
        // completed) and the disjoint up slice.
        const MlpWorkspaceRequirements synthetic{
                {40, 32}, {8, 32}, {0, 1}, {16, 32}, {4, 32}};
        CHECK(synthetic.total() == iom::WorkspaceRequirements{96, 32});
    }
}

TEST_CASE("TinyLlama MLP stage drains both projection branches after an accepted failure") {
    MlpCpuFixture fixture;
    auto operations = fixture.device->create_ops();
    MlpFailureLatches latches;

    // One failing gate branch while the up branch succeeds.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 6);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);

        iom::cpu_detail::arm_linear_failure(0, 1);
        MlpStageFailure failure;
        const std::string message =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK_MESSAGE(
                message.find("CPU linear projection injected post-acceptance "
                             "failure") != std::string::npos,
                message);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check(*bank.up, bank.ref_up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");

        // The poisoned execution is never reused: the preserved first failure
        // is rethrown without submitting anything, even though the latch is
        // already disarmed and the queue is healthy.
        const std::vector<float> before = mlp_store_image(bank);
        const std::string replay =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK(replay == message);
        CHECK(mlp_store_image(bank) == before);

        // Once every accepted OID has been drained, the same queue, tensors,
        // and resolved workspace are reusable.
        MlpStageFailure recovered;
        mlp_drive(bank, *operations, producer, workspace, recovered);
        CHECK_FALSE(recovered.poisoned());
        mlp_check_reference(bank);
    }

    // Both projection branches fail after acceptance.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 7);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);

        iom::cpu_detail::arm_linear_failure(0, 2);
        MlpStageFailure failure;
        const std::string message =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK_MESSAGE(
                message.find("CPU linear projection injected post-acceptance "
                             "failure") != std::string::npos,
                message);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check_untouched(*bank.up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    // One failing up branch while the gate branch succeeds: the armed plan
    // lets the gate projection complete and fails the sibling branch after its
    // own acceptance.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 8);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);

        iom::cpu_detail::arm_linear_failure(1, 1);
        MlpStageFailure failure;
        const std::string message =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK_MESSAGE(
                message.find("CPU linear projection injected post-acceptance "
                             "failure") != std::string::npos,
                message);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check(*bank.gate, bank.ref_gate, "gate projection");
        mlp_check_untouched(*bank.up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    // A rejected gate admission still submits, accepts, and drains the up
    // branch before the preserved first failure is rethrown.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 10);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);

        // A weight owner of another device is a plain BF16 [M,F] view, so the
        // rejection happens at the projection's own admission rather than in
        // the stage's local shape check.
        MlpHeapAllocator foreign_allocator;
        std::unique_ptr<iom::Device> foreign_device =
                iom::make_cpu_device(foreign_allocator);
        auto foreign_weight = foreign_device->create_tensor(
                mlp_spec({bank.intermediate, bank.features}));
        MlpStageViews views = mlp_make_views(
                *bank.x2, *bank.scale, *foreign_weight, *bank.up_weight,
                *bank.down_weight, *bank.n2, *bank.gate, *bank.up,
                *bank.activated, *bank.product, *bank.down, *bank.next_x);

        const iom::oid producer = mlp_submit_first_residual(bank, *operations);
        MlpStageFailure failure;
        const MlpStageParams params = mlp_params(bank);
        const std::span<const iom::oid> readiness(&producer, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check(*bank.up, bank.ref_up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    // A failing accepted activation boundary stops every dependent stage.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 9);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);

        iom::cpu_detail::arm_silu_failure();
        MlpStageFailure failure;
        const std::string message =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK_MESSAGE(
                message.find("CPU SiLU injected") != std::string::npos, message);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check(*bank.gate, bank.ref_gate, "gate projection");
        mlp_check(*bank.up, bank.ref_up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }
}

TEST_CASE(
        "TinyLlama cache attention stage publishes a checked prefix and "
        "forms the first residual") {
    StageFixture fixture{Geometry{}};
    fixture.seed_all(0x5F1D2B3C4D5E6F70ull);
    auto request = fixture.make_request();

    // Both accepted append OIDs must be waited by the stage itself before it
    // returns: the observation seam counts caller waits that observed an
    // accepted append completion, which row contents alone cannot show.
    iom::cpu_detail::arm_cache_append_wait_observation();
    const StageOutcome outcome =
            run_stage(fixture, request, fixture.workspace());
    CHECK_EQ(iom::cpu_detail::observed_cache_append_waits(), 2u);
    iom::cpu_detail::clear_cache_append_wait_observation();
    REQUIRE(outcome.failure == nullptr);

    // Exactly the checked prefix is published, and both appends succeeded.
    CHECK_EQ(outcome.state.initialized_length,
             fixture.geometry.offset + fixture.geometry.rows);
    CHECK_FALSE(outcome.state.failed);

    // Rotated K and V rows landed independently in their own caches, and no
    // other cache row changed.
    check_identical(cache_window(fixture.geometry, outcome.k_cache),
                    fixture.rotated_k_values, "appended K rows");
    check_identical(cache_window(fixture.geometry, outcome.v_cache),
                    fixture.rotated_v_values, "appended V rows");
    check_identical(outcome.k_cache,
                    with_appended_rows(fixture.geometry,
                                       fixture.k_cache_values,
                                       fixture.rotated_k_values),
                    "K cache after publication");
    check_identical(outcome.v_cache,
                    with_appended_rows(fixture.geometry, fixture.v_cache_values,
                                       fixture.rotated_v_values),
                    "V cache after publication");

    // Attention, output projection, and first residual match the independent
    // reference over the published prefix.
    const std::vector<float> expected_merged = reference_merged(
            fixture.geometry, fixture.geometry.offset, fixture.rotated_q_values,
            with_appended_rows(fixture.geometry, fixture.k_cache_values,
                               fixture.rotated_k_values),
            with_appended_rows(fixture.geometry, fixture.v_cache_values,
                               fixture.rotated_v_values),
            fixture.geometry.offset + fixture.geometry.rows);
    check_close(outcome.merged, expected_merged, 2, 1e-3f,
                "merged attention");

    const std::vector<float> expected_projected =
            reference_projected(fixture.geometry, outcome.merged,
                                fixture.o_weight_values);
    check_close(outcome.projected, expected_projected, 2, 1e-3f,
                "output projection");
    check_close(outcome.residual,
                reference_residual(fixture.residual_input_values,
                                   outcome.projected),
                2, 1e-3f, "first residual");

    // The complete chain, recomputed from the supplied inputs alone, is
    // compared within the propagated budget of the one stored boundary the
    // device may deviate by: two merged units per element through `F` weighted
    // terms. The tight checks above pin the exact stored values.
    const std::vector<float> chained_projected = reference_projected(
            fixture.geometry, expected_merged, fixture.o_weight_values);
    check_close(outcome.projected, chained_projected, 2, 2e-2f,
                "output projection from supplied inputs");
    check_close(outcome.residual,
                reference_residual(fixture.residual_input_values,
                                   chained_projected),
                2, 2e-2f, "first residual from supplied inputs");
}

TEST_CASE(
        "TinyLlama cache attention stage masks future tokens and maps grouped "
        "heads") {
    const Geometry geometry{};
    constexpr std::uint64_t seed = 0x0BADF00DCAFEF00Dull;
    StageFixture base{geometry};
    base.seed_all(seed);
    auto base_request = base.make_request();
    const StageOutcome base_outcome =
            run_stage(base, base_request, base.workspace());
    REQUIRE(base_outcome.failure == nullptr);

    // The appended row at cache row `offset+1` is a future token for query row
    // 0 and a visible token for query row 1: perturbing it must not move row 0
    // at all while it does move row 1.
    StageFixture future{geometry};
    future.seed_all(seed);
    std::vector<float> perturbed_k = future.rotated_k_values;
    std::vector<float> perturbed_v = future.rotated_v_values;
    for (std::size_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::size_t feature = 0; feature < geometry.head_width;
             ++feature) {
            const std::size_t index =
                    (head * geometry.rows + 1) * geometry.head_width + feature;
            perturbed_k[index] = round_bf16(perturbed_k[index] + 0.5f);
            perturbed_v[index] = round_bf16(perturbed_v[index] - 0.25f);
        }
    }
    REQUIRE(perturbed_k != future.rotated_k_values);
    future.rotated_k_values = perturbed_k;
    future.rotated_v_values = perturbed_v;
    future.upload_all();
    auto future_request = future.make_request();
    const StageOutcome future_outcome =
            run_stage(future, future_request, future.workspace());
    REQUIRE(future_outcome.failure == nullptr);

    const std::size_t features = geometry.features();
    bool row_one_changed = false;
    for (std::size_t index = 0; index < features; ++index) {
        CHECK_MESSAGE(future_outcome.merged[index] == base_outcome.merged[index],
                      "row 0 element " << index << ": "
                                       << future_outcome.merged[index]
                                       << " vs " << base_outcome.merged[index]);
        row_one_changed =
                row_one_changed
                || future_outcome.merged[features + index]
                           != base_outcome.merged[features + index];
    }
    CHECK(row_one_changed);

    // Distinct grouped-query head mapping: KV head 0 feeds exactly query heads
    // 0 and 1, and KV head 1 feeds exactly query heads 2 and 3.
    StageFixture grouped{geometry};
    grouped.seed_all(seed);
    std::vector<float> grouped_k = grouped.rotated_k_values;
    std::vector<float> grouped_v = grouped.rotated_v_values;
    for (std::size_t row = 0; row < geometry.rows; ++row) {
        for (std::size_t feature = 0; feature < geometry.head_width;
             ++feature) {
            const std::size_t index =
                    (geometry.rows + row) * geometry.head_width + feature;
            grouped_k[index] = round_bf16(grouped_k[index] + 0.5f);
            grouped_v[index] = round_bf16(grouped_v[index] - 0.25f);
        }
    }
    REQUIRE(grouped_k != grouped.rotated_k_values);
    grouped.rotated_k_values = grouped_k;
    grouped.rotated_v_values = grouped_v;
    grouped.upload_all();
    auto grouped_request = grouped.make_request();
    const StageOutcome grouped_outcome =
            run_stage(grouped, grouped_request, grouped.workspace());
    REQUIRE(grouped_outcome.failure == nullptr);

    const std::size_t group = geometry.query_heads / geometry.kv_heads;
    bool grouped_changed = false;
    for (std::size_t row = 0; row < geometry.rows; ++row) {
        for (std::size_t head = 0; head < geometry.query_heads; ++head) {
            for (std::size_t feature = 0; feature < geometry.head_width;
                 ++feature) {
                const std::size_t index =
                        (row * geometry.query_heads + head)
                                * geometry.head_width
                        + feature;
                if (head < group) {
                    CHECK_MESSAGE(grouped_outcome.merged[index]
                                          == base_outcome.merged[index],
                                  "row " << row << " head " << head
                                         << " feature " << feature << ": "
                                         << grouped_outcome.merged[index]
                                         << " vs "
                                         << base_outcome.merged[index]);
                } else if (grouped_outcome.merged[index]
                           != base_outcome.merged[index]) {
                    grouped_changed = true;
                }
            }
        }
    }
    CHECK(grouped_changed);

    // The uninitialized capacity tail is never read: a different tail sentinel
    // changes no output, projection, or residual value.
    StageFixture tail{geometry};
    tail.seed_all(seed);
    for (std::size_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::size_t row = geometry.offset + geometry.rows;
             row < geometry.capacity; ++row) {
            for (std::size_t feature = 0; feature < geometry.head_width;
                 ++feature) {
                tail.k_cache_values[tail.cache_index(head, row, feature)] =
                        21.75f;
                tail.v_cache_values[tail.cache_index(head, row, feature)] =
                        -18.5f;
            }
        }
    }
    tail.upload_all();
    auto tail_request = tail.make_request();
    const StageOutcome tail_outcome =
            run_stage(tail, tail_request, tail.workspace());
    REQUIRE(tail_outcome.failure == nullptr);
    check_identical(tail_outcome.merged, base_outcome.merged,
                    "merged attention with a perturbed uninitialized tail");
    check_identical(tail_outcome.projected, base_outcome.projected,
                    "output projection with a perturbed uninitialized tail");
    check_identical(tail_outcome.residual, base_outcome.residual,
                    "first residual with a perturbed uninitialized tail");
}

TEST_CASE(
        "TinyLlama cache attention stage fills exactly the published prefix at "
        "exact capacity and on a decode row") {
    // Prefill rows that exactly fill the persistent cache capacity.
    Geometry prefill;
    prefill.rows = 4;
    prefill.capacity = 5;
    prefill.offset = 1;
    StageFixture full{prefill};
    full.seed_all(0x1122334455667788ull);
    auto full_request = full.make_request();
    const StageOutcome full_outcome =
            run_stage(full, full_request, full.workspace());
    REQUIRE(full_outcome.failure == nullptr);
    CHECK_EQ(full_outcome.state.initialized_length, prefill.capacity);
    check_identical(cache_window(prefill, full_outcome.k_cache),
                    full.rotated_k_values, "exact-capacity K rows");
    check_identical(cache_window(prefill, full_outcome.v_cache),
                    full.rotated_v_values, "exact-capacity V rows");
    check_identical(full_outcome.k_cache,
                    with_appended_rows(prefill, full.k_cache_values,
                                       full.rotated_k_values),
                    "exact-capacity K cache");
    check_identical(full_outcome.v_cache,
                    with_appended_rows(prefill, full.v_cache_values,
                                       full.rotated_v_values),
                    "exact-capacity V cache");
    check_close(full_outcome.merged,
                reference_merged(
                        prefill, prefill.offset, full.rotated_q_values,
                        with_appended_rows(prefill, full.k_cache_values,
                                           full.rotated_k_values),
                        with_appended_rows(prefill, full.v_cache_values,
                                           full.rotated_v_values),
                        prefill.capacity),
                2, 1e-3f, "exact-capacity merged attention");

    // One decode row at a nonzero absolute position, with `L < C`.
    Geometry decode;
    decode.rows = 1;
    decode.capacity = 5;
    decode.offset = 3;
    StageFixture single{decode};
    single.seed_all(0xC0FFEE0BADF00D11ull);
    auto single_request = single.make_request();
    const StageOutcome single_outcome =
            run_stage(single, single_request, single.workspace());
    REQUIRE(single_outcome.failure == nullptr);
    CHECK_EQ(single_outcome.state.initialized_length, decode.offset + 1);
    check_identical(cache_window(decode, single_outcome.k_cache),
                    single.rotated_k_values, "decode K row");
    check_identical(cache_window(decode, single_outcome.v_cache),
                    single.rotated_v_values, "decode V row");
    check_identical(single_outcome.k_cache,
                    with_appended_rows(decode, single.k_cache_values,
                                       single.rotated_k_values),
                    "decode K cache");
    check_identical(single_outcome.v_cache,
                    with_appended_rows(decode, single.v_cache_values,
                                       single.rotated_v_values),
                    "decode V cache");
    check_close(single_outcome.merged,
                reference_merged(
                        decode, decode.offset, single.rotated_q_values,
                        with_appended_rows(decode, single.k_cache_values,
                                           single.rotated_k_values),
                        with_appended_rows(decode, single.v_cache_values,
                                           single.rotated_v_values),
                        decode.offset + 1),
                2, 1e-3f, "decode merged attention");
}

TEST_CASE(
        "TinyLlama cache attention stage refuses incomplete publication and "
        "never submits SDPA") {
    // A geometry whose append row window exactly covers the persistent cache
    // lets a rejected append borrow its own destination as the source, which
    // the cache append admission rejects before any effect.
    Geometry geometry;
    geometry.rows = 4;
    geometry.capacity = 4;
    geometry.offset = 0;
    constexpr std::uint64_t seed = 0x2468ACE013579BDFull;

    // An aliased K cache is rejected by the stage-wide disjointness check
    // before any submission: no cache row moves, no store is touched, no
    // prefix is published, the state is not poisoned, and neither armed latch
    // is consumed.
    {
        StageFixture fixture{geometry};
        fixture.seed_all(seed);
        // Both latches must survive the rejected run: a consumed append latch
        // would mean an append was submitted, and a consumed SDPA latch would
        // mean the stage submitted attention after a partial append success.
        iom::cpu_detail::arm_sdpa_failure();
        iom::cpu_detail::arm_cache_append_failure();
        const iom::TensorView aliased_k = fixture.k_cache->view();
        auto request = fixture.make_request(
                StageFixture::Overrides{.rotated_k = &aliased_k});

        iom::session_detail::CacheAttentionStageState state;
        state.initialized_length = request.a;
        CHECK_THROWS_AS(
                iom::session_detail::run_cache_attention_stage(
                        fixture.ops(), request, fixture.workspace(), state),
                std::invalid_argument);
        CHECK_FALSE(state.failed);
        CHECK_EQ(state.initialized_length, geometry.offset);

        check_identical(fixture.read(*fixture.k_cache),
                        fixture.k_cache_values,
                        "K cache after a rejected aliased request");
        check_identical(fixture.read(*fixture.v_cache),
                        fixture.v_cache_values,
                        "V cache after a rejected aliased request");
        check_identical(fixture.read(*fixture.attention_merged),
                        fixture.attention_merged_values,
                        "merged attention after a rejected aliased request");
        check_identical(fixture.read(*fixture.attention_output),
                        fixture.attention_output_values,
                        "attention output after a rejected aliased request");
        check_identical(fixture.read(*fixture.residual_output),
                        fixture.residual_output_values,
                        "residual output after a rejected aliased request");

        // The append latch is still armed, so no append was submitted: a
        // direct, otherwise valid append consumes it and fails its wait.
        const iom::oid append = fixture.ops().cache_append(
                fixture.rotated_k->view(), fixture.k_cache->view(),
                geometry.offset);
        REQUIRE(iom::oid_is_token(append));
        CHECK_THROWS_AS(fixture.ops().wait(append), std::runtime_error);
        iom::cpu_detail::clear_cache_append_failure();

        // The SDPA latch is still armed for the same reason.
        const iom::oid token = fixture.ops().sdpa(
                fixture.rotated_q->view(), fixture.k_cache->view(),
                fixture.v_cache->view(), fixture.attention_merged->view(),
                geometry.offset, geometry.offset + geometry.rows,
                fixture.workspace());
        REQUIRE(iom::oid_is_token(token));
        CHECK_THROWS_AS(fixture.ops().wait(token), std::runtime_error);
        iom::cpu_detail::clear_sdpa_failure();
    }

    // The mirror case: an aliased V cache is rejected before any submission
    // with the same no-effect and no-poison guarantees, and the armed SDPA
    // latch survives.
    {
        StageFixture fixture{geometry};
        fixture.seed_all(seed);
        iom::cpu_detail::arm_sdpa_failure();
        const iom::TensorView aliased_v = fixture.v_cache->view();
        auto request = fixture.make_request(
                StageFixture::Overrides{.rotated_v = &aliased_v});

        iom::session_detail::CacheAttentionStageState state;
        state.initialized_length = request.a;
        CHECK_THROWS_AS(
                iom::session_detail::run_cache_attention_stage(
                        fixture.ops(), request, fixture.workspace(), state),
                std::invalid_argument);
        CHECK_FALSE(state.failed);
        CHECK_EQ(state.initialized_length, geometry.offset);

        check_identical(fixture.read(*fixture.k_cache),
                        fixture.k_cache_values,
                        "K cache after a rejected aliased request");
        check_identical(fixture.read(*fixture.v_cache),
                        fixture.v_cache_values,
                        "V cache after a rejected aliased request");
        check_identical(fixture.read(*fixture.attention_merged),
                        fixture.attention_merged_values,
                        "merged attention after a rejected aliased request");
        check_identical(fixture.read(*fixture.attention_output),
                        fixture.attention_output_values,
                        "attention output after a rejected aliased request");
        check_identical(fixture.read(*fixture.residual_output),
                        fixture.residual_output_values,
                        "residual output after a rejected aliased request");

        const iom::oid token = fixture.ops().sdpa(
                fixture.rotated_q->view(), fixture.k_cache->view(),
                fixture.v_cache->view(), fixture.attention_merged->view(),
                geometry.offset, geometry.offset + geometry.rows,
                fixture.workspace());
        REQUIRE(iom::oid_is_token(token));
        CHECK_THROWS_AS(fixture.ops().wait(token), std::runtime_error);
        iom::cpu_detail::clear_sdpa_failure();
    }
}

TEST_CASE(
        "TinyLlama cache attention stage attempts both append waits after an "
        "append completion failure") {
    const Geometry geometry{};
    constexpr std::uint64_t seed = 0x0A11CE0BEEF01234ull;

    // K accepted and failed at completion while V was accepted and succeeded:
    // the K wait throws, the independent V wait still runs, no prefix is
    // published, and no downstream work is submitted.
    {
        StageFixture fixture{geometry};
        fixture.seed_all(seed);
        iom::cpu_detail::arm_cache_append_failure();
        // The SDPA latch survives only when no SDPA was submitted after the
        // append failure.
        iom::cpu_detail::arm_sdpa_failure();
        iom::cpu_detail::arm_cache_append_wait_observation();
        auto request = fixture.make_request();
        iom::session_detail::CacheAttentionStageState state;
        state.initialized_length = request.a;
        CHECK_THROWS_AS(
                iom::session_detail::run_cache_attention_stage(
                        fixture.ops(), request, fixture.workspace(), state),
                std::runtime_error);
        // K's throwing wait surfaces the injected completion failure, which
        // proves that wait ran; the seam then proves the independent V wait ran
        // too, because only a caller wait records V's accepted sequence.
        CHECK_EQ(iom::cpu_detail::observed_cache_append_waits(), 1u);
        iom::cpu_detail::clear_cache_append_wait_observation();
        iom::cpu_detail::clear_cache_append_failure();

        CHECK(state.failed);
        CHECK_EQ(state.initialized_length, geometry.offset);
        // The failing append wrote no row, while the independently accepted V
        // append completed.
        check_identical(fixture.read(*fixture.k_cache),
                        fixture.k_cache_values,
                        "K cache after a failed K completion");
        check_identical(fixture.read(*fixture.v_cache),
                        with_appended_rows(geometry, fixture.v_cache_values,
                                           fixture.rotated_v_values),
                        "V cache after a successful V append");
        check_identical(fixture.read(*fixture.attention_merged),
                        fixture.attention_merged_values,
                        "merged attention after an append completion failure");
        check_identical(fixture.read(*fixture.attention_output),
                        fixture.attention_output_values,
                        "attention output after an append completion failure");
        check_identical(fixture.read(*fixture.residual_output),
                        fixture.residual_output_values,
                        "residual output after an append completion failure");

        // The latch is still armed, so the stage submitted no SDPA.
        const iom::oid token = fixture.ops().sdpa(
                fixture.rotated_q->view(), fixture.k_cache->view(),
                fixture.v_cache->view(), fixture.attention_merged->view(),
                geometry.offset, geometry.offset + geometry.rows,
                fixture.workspace());
        REQUIRE(iom::oid_is_token(token));
        CHECK_THROWS_AS(fixture.ops().wait(token), std::runtime_error);
        iom::cpu_detail::clear_sdpa_failure();
    }
}

TEST_CASE(
        "TinyLlama cache attention stage preflights downstream scratch before "
        "any cache mutation") {
    const Geometry geometry{};
    StageFixture fixture{geometry};
    fixture.seed_all(0x5EED0F5C2A7B9D31ull);

    // SDPA, the output projection, and the first residual are admitted with
    // their actual operands before the first submission.
    const iom::WorkspaceRequirements combined = combined_requirement(fixture);
    REQUIRE(combined.bytes > 0);
    REQUIRE(combined.bytes <= fixture.workspace().byte_size());

    // A range that satisfies the largest requirement completes the stage.
    auto request = fixture.make_request();
    const StageOutcome outcome =
            run_stage(fixture, request, fixture.workspace());
    REQUIRE(outcome.failure == nullptr);
    CHECK_FALSE(outcome.state.failed);
    CHECK_EQ(outcome.state.initialized_length,
             geometry.offset + geometry.rows);

    // One byte less than the largest requirement is rejected before the first
    // append: no cache row moves, no output store is touched, and no prefix is
    // published or failure state recorded.
    StageFixture short_fixture{geometry};
    short_fixture.seed_all(0x5EED0F5C2A7B9D31ull);
    const std::unique_ptr<iom::RawWorkspace> short_workspace =
            short_fixture.device->create_workspace(combined.bytes - 1);
    auto short_request = short_fixture.make_request();
    iom::session_detail::CacheAttentionStageState state;
    state.initialized_length = short_request.a;
    CHECK_THROWS_AS(
            iom::session_detail::run_cache_attention_stage(
                    short_fixture.ops(), short_request,
                    short_workspace->view(), state),
            std::invalid_argument);
    CHECK_FALSE(state.failed);
    CHECK_EQ(state.initialized_length, geometry.offset);
    check_identical(short_fixture.read(*short_fixture.k_cache),
                    short_fixture.k_cache_values,
                    "K cache after a short scratch range");
    check_identical(short_fixture.read(*short_fixture.v_cache),
                    short_fixture.v_cache_values,
                    "V cache after a short scratch range");
    check_identical(short_fixture.read(*short_fixture.attention_merged),
                    short_fixture.attention_merged_values,
                    "merged attention after a short scratch range");
    check_identical(short_fixture.read(*short_fixture.attention_output),
                    short_fixture.attention_output_values,
                    "attention output after a short scratch range");
    check_identical(short_fixture.read(*short_fixture.residual_output),
                    short_fixture.residual_output_values,
                    "residual output after a short scratch range");
}

TEST_CASE(
        "TinyLlama cache attention stage poisons the state after an accepted "
        "SDPA failure") {
    const Geometry geometry{};
    StageFixture fixture{geometry};
    fixture.seed_all(0x13572468ACE0BDF1ull);

    iom::cpu_detail::arm_sdpa_failure();
    auto request = fixture.make_request();
    const StageOutcome outcome =
            run_stage(fixture, request, fixture.workspace());
    REQUIRE(outcome.failure != nullptr);
    CHECK_THROWS_AS(std::rethrow_exception(outcome.failure), std::runtime_error);
    iom::cpu_detail::clear_sdpa_failure();

    // Both appends completed, so the checked prefix is published, while the
    // failed SDPA submits no projection or residual.
    CHECK(outcome.state.failed);
    CHECK_EQ(outcome.state.initialized_length,
             geometry.offset + geometry.rows);
    check_identical(outcome.k_cache,
                    with_appended_rows(geometry, fixture.k_cache_values,
                                       fixture.rotated_k_values),
                    "K cache after SDPA failure");
    check_identical(outcome.v_cache,
                    with_appended_rows(geometry, fixture.v_cache_values,
                                       fixture.rotated_v_values),
                    "V cache after SDPA failure");
    check_identical(outcome.merged, fixture.attention_merged_values,
                    "merged attention after SDPA failure");
    check_identical(outcome.projected, fixture.attention_output_values,
                    "attention output after SDPA failure");
    check_identical(outcome.residual, fixture.residual_output_values,
                    "residual output after SDPA failure");

    // A poisoned state rejects reuse before any submission: no cache row is
    // appended twice and no published length changes.
    auto retry = fixture.make_request();
    iom::session_detail::CacheAttentionStageState poisoned;
    poisoned.initialized_length = retry.a;
    poisoned.failed = true;
    CHECK_THROWS_AS(
            iom::session_detail::run_cache_attention_stage(
                    fixture.ops(), retry, fixture.workspace(), poisoned),
            std::logic_error);
    CHECK_EQ(poisoned.initialized_length, geometry.offset);
    check_identical(fixture.read(*fixture.k_cache), outcome.k_cache,
                    "K cache after refused reuse");
    check_identical(fixture.read(*fixture.v_cache), outcome.v_cache,
                    "V cache after refused reuse");

    // The queue itself stays healthy for later independent work.
    const iom::oid token = fixture.ops().sdpa(
            fixture.rotated_q->view(), fixture.k_cache->view(),
            fixture.v_cache->view(), fixture.attention_merged->view(),
            geometry.offset, geometry.offset + geometry.rows,
            fixture.workspace());
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(fixture.ops().wait(token));
}

TEST_CASE(
        "TinyLlama cache attention stage rejects malformed requests before any "
        "submission") {
    enum class Category { Invalid, Overflow, Logic };

    const Geometry geometry{};
    StageFixture fixture{geometry};
    fixture.seed_all(0x77777777AAAAAAAAull);

    const iom::TensorView rank_two = fixture.residual_input->view();
    const iom::TensorView short_query =
            fixture.rotated_q->view().slice(0, 0, 3);
    const std::unique_ptr<iom::Tensor> narrow =
            fixture.create({geometry.rows, geometry.features() - 1});
    const iom::TensorView short_merged = narrow->view();
    const iom::TensorView merged_view = fixture.attention_merged->view();
    const iom::TensorView residual_input_view =
            fixture.residual_input->view();
    const iom::TensorView attention_output_view =
            fixture.attention_output->view();
    const iom::TensorView k_cache_view = fixture.k_cache->view();
    const iom::TensorView v_cache_view = fixture.v_cache->view();
    const iom::RawWorkspaceView empty_workspace{};
    const iom::WorkspaceRequirements combined = combined_requirement(fixture);
    REQUIRE(combined.bytes > 1);
    const std::unique_ptr<iom::RawWorkspace> short_workspace =
            fixture.device->create_workspace(combined.bytes - 1);
    const iom::RawWorkspaceView short_workspace_view =
            short_workspace->view();

    struct Rejection {
        const char* label;
        StageFixture::Overrides overrides;
        std::size_t initialized_length;
        bool poisoned_state;
        const iom::RawWorkspaceView* workspace;
        Category category;
    };
    const Rejection rejections[] = {
            {"zero rows", {.rows = 0}, geometry.offset, false, nullptr,
             Category::Invalid},
            {"zero capacity", {.capacity = 0}, geometry.offset, false, nullptr,
             Category::Invalid},
            {"offset beyond capacity", {.offset = geometry.capacity + 1},
             geometry.capacity + 1, false, nullptr, Category::Invalid},
            {"rows beyond capacity",
             {.offset = geometry.capacity - 1, .rows = geometry.rows + 1},
             geometry.capacity - 1, false, nullptr, Category::Invalid},
            {"appended row count overflow",
             {.offset = 1, .rows = std::numeric_limits<std::size_t>::max()}, 1,
             false, nullptr, Category::Overflow},
            {"rank-two query", {.rotated_q = &rank_two}, geometry.offset, false,
             nullptr, Category::Invalid},
            {"grouped head ratio", {.rotated_q = &short_query}, geometry.offset,
             false, nullptr, Category::Invalid},
            {"merged attention width", {.attention_merged = &short_merged},
             geometry.offset, false, nullptr, Category::Invalid},
            {"poisoned state", {}, geometry.offset, true, nullptr,
             Category::Logic},
            {"missing caller workspace", {}, geometry.offset, false,
             &empty_workspace, Category::Invalid},
            {"short caller workspace", {}, geometry.offset, false,
             &short_workspace_view, Category::Invalid},
            {"attention output aliases merged attention",
             {.attention_output = &merged_view}, geometry.offset, false,
             nullptr, Category::Invalid},
            {"shared K/V cache storage", {.v_cache = &k_cache_view},
             geometry.offset, false, nullptr, Category::Invalid},
            {"K cache aliases the V cache", {.k_cache = &v_cache_view},
             geometry.offset, false, nullptr, Category::Invalid},
            {"merged attention aliases residual input",
             {.attention_merged = &residual_input_view}, geometry.offset,
             false, nullptr, Category::Invalid},
            {"attention output aliases residual input",
             {.attention_output = &residual_input_view}, geometry.offset, false,
             nullptr, Category::Invalid},
            {"residual output aliases residual input",
             {.residual_output = &residual_input_view}, geometry.offset, false,
             nullptr, Category::Invalid},
            {"residual output aliases attention output",
             {.residual_output = &attention_output_view}, geometry.offset,
             false, nullptr, Category::Invalid},
            {"stale published prefix", {}, geometry.offset + 1, false, nullptr,
             Category::Invalid}};

    for (const Rejection& rejection : rejections) {
        auto request = fixture.make_request(rejection.overrides);
        iom::session_detail::CacheAttentionStageState state;
        state.initialized_length = rejection.initialized_length;
        state.failed = rejection.poisoned_state;
        const iom::RawWorkspaceView workspace = rejection.workspace != nullptr
                ? *rejection.workspace
                : fixture.workspace();
        bool rejected = false;
        try {
            iom::session_detail::run_cache_attention_stage(
                    fixture.ops(), request, workspace, state);
        } catch (const std::invalid_argument&) {
            rejected = rejection.category == Category::Invalid;
        } catch (const std::overflow_error&) {
            rejected = rejection.category == Category::Overflow;
        } catch (const std::logic_error&) {
            rejected = rejection.category == Category::Logic;
        }
        CHECK_MESSAGE(rejected, rejection.label);

        // A rejected request changes nothing: no cache row moved, no output
        // store was touched, and no prefix was published.
        CHECK_EQ(state.initialized_length, rejection.initialized_length);
        CHECK_EQ(state.failed, rejection.poisoned_state);
        check_identical(fixture.read(*fixture.k_cache),
                        fixture.k_cache_values, "rejected K cache");
        check_identical(fixture.read(*fixture.v_cache),
                        fixture.v_cache_values, "rejected V cache");
        check_identical(fixture.read(*fixture.attention_merged),
                        fixture.attention_merged_values,
                        "rejected merged attention");
        check_identical(fixture.read(*fixture.attention_output),
                        fixture.attention_output_values,
                        "rejected attention output");
        check_identical(fixture.read(*fixture.residual_output),
                        fixture.residual_output_values,
                        "rejected residual output");
    }
}

// ---------------------------------------------------------------------------
// Attention-normalization, QKV projection, and Q/K RoPE stage cases
// (leaf 03-qkv-rope-stage).
//
// Every case builds supplied BF16 tensors directly on a CPU device and calls
// the private stage through its implementation-private declaration. No
// session, model checkpoint, tokenizer, cache, selector, generation, or CLI
// resource is constructed, so the stage is proven usable before session
// resource ownership exists.
//
// The reference is independent: it decodes the supplied BF16 codes itself,
// re-derives the RMSNorm reduction, the Hugging Face `[out,in]` head-planar
// projection, and the split-half rotation from the operation-owned equations,
// and never calls production arithmetic. Each stage boundary is compared
// under the shared conformance harness' BF16 policy
// `max(one destination ULP, 2^-7, 2^-6 * |reference|)`; the downstream
// boundary references consume the *observed* upstream BF16 values, so a
// rounding at one boundary cannot be amplified into an unrelated mismatch.
// ---------------------------------------------------------------------------

namespace qkv_rope_stage_test {

    using iom::Device;
    using iom::DeviceOps;
    using iom::Tensor;
    using iom::TensorShape;
    using iom::TensorSpec;
    using iom::TensorView;
    using iom::oid;

    constexpr iom::DataType kLeaf = iom::DataType::BF16;

    // ------------------------------------------------------------------
    // Independent BF16 codec, reference arithmetic, and comparison policy.
    // ------------------------------------------------------------------

    [[nodiscard]] std::uint16_t bf16_code(double value) {
        const float narrowed = static_cast<float>(value);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &narrowed, sizeof(bits));
        const std::uint32_t sign = bits & 0x80000000u;
        const std::uint32_t magnitude = bits & 0x7fffffffu;
        // Round to nearest, ties to even, over the 16 discarded bits.
        const std::uint32_t rounded =
                magnitude + 0x7fffu + ((magnitude >> 16) & 1u);
        return static_cast<std::uint16_t>((sign >> 16) | (rounded >> 16));
    }

    [[nodiscard]] double bf16_value(std::uint16_t code) {
        const std::uint32_t bits = static_cast<std::uint32_t>(code) << 16;
        float narrowed = 0.0F;
        std::memcpy(&narrowed, &bits, sizeof(narrowed));
        return static_cast<double>(narrowed);
    }

    [[nodiscard]] double bf16_ulp(double value) {
        if (value == 0.0) {
            return std::ldexp(1.0, -133);
        }
        return std::ldexp(1.0, std::ilogb(std::fabs(value)) - 7);
    }

    // One stored BF16 operation boundary.
    [[nodiscard]] double stored(double value) {
        return bf16_value(bf16_code(value));
    }

    [[nodiscard]] std::vector<double> reference_normalization(
            std::span<const double> activation, std::size_t rows,
            std::size_t features, std::span<const double> scale,
            float epsilon) {
        std::vector<double> normalized(activation.size());
        const double divisor = static_cast<double>(features);
        for (std::size_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < features; ++feature) {
                const double value = activation[row * features + feature];
                sum += value * value;
            }
            const double inverse =
                    1.0 / std::sqrt(sum / divisor + static_cast<double>(epsilon));
            for (std::size_t feature = 0; feature < features; ++feature) {
                normalized[row * features + feature] =
                        stored(activation[row * features + feature] * inverse
                               * scale[feature]);
            }
        }
        return normalized;
    }

    // `out[h,r,d] = sum_i normalized[r,i] * weight[h*D+d,i]` with the
    // Hugging Face `[out,in]` weight consumed unchanged.
    [[nodiscard]] std::vector<double> reference_head_planar(
            std::span<const double> normalized, std::size_t rows,
            std::size_t features, std::span<const double> weight,
            std::size_t heads, std::size_t head_dim) {
        std::vector<double> projected(heads * rows * head_dim);
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t element = 0; element < head_dim;
                        ++element) {
                    const std::size_t output_row = head * head_dim + element;
                    double sum = 0.0;
                    for (std::size_t feature = 0; feature < features;
                            ++feature) {
                        sum += normalized[row * features + feature]
                                * weight[output_row * features + feature];
                    }
                    projected[(head * rows + row) * head_dim + element] =
                            stored(sum);
                }
            }
        }
        return projected;
    }

    // Split-half rotation at absolute position `a + r`.
    [[nodiscard]] std::vector<double> reference_rope(
            std::span<const double> projected, std::size_t heads,
            std::size_t rows, std::size_t head_dim, std::size_t a,
            double theta) {
        std::vector<double> rotated(projected.size());
        const std::size_t half = head_dim / 2;
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t row = 0; row < rows; ++row) {
                const double position = static_cast<double>(a + row);
                const std::size_t base = (head * rows + row) * head_dim;
                for (std::size_t pair = 0; pair < half; ++pair) {
                    const double angle = position
                            * std::pow(theta,
                                       -2.0 * static_cast<double>(pair)
                                               / static_cast<double>(head_dim));
                    const double cosine = std::cos(angle);
                    const double sine = std::sin(angle);
                    const double first = projected[base + pair];
                    const double second = projected[base + half + pair];
                    rotated[base + pair] =
                            stored(first * cosine - second * sine);
                    rotated[base + half + pair] =
                            stored(second * cosine + first * sine);
                }
            }
        }
        return rotated;
    }

    void require_reference(
            std::span<const double> observed,
            std::span<const double> reference, const char* label) {
        REQUIRE(observed.size() == reference.size());
        for (std::size_t index = 0; index < observed.size(); ++index) {
            const double expected = stored(reference[index]);
            const double bound = std::max(
                    {bf16_ulp(expected), std::ldexp(1.0, -7),
                     std::ldexp(1.0, -6) * std::fabs(expected)});
            INFO(label << " element " << index << ": observed "
                       << observed[index] << ", reference " << expected);
            REQUIRE(std::isfinite(observed[index]));
            REQUIRE(std::fabs(observed[index] - expected) <= bound);
        }
    }

    // ------------------------------------------------------------------
    // Supplied tensors, synthetic values, and caller-provisioned views.
    // ------------------------------------------------------------------

    class HeapAllocator final : public iom::Allocator {
    public:
        void* alloc(std::size_t bytes) override {
            return ::operator new(
                    std::max<std::size_t>(bytes, 1), std::align_val_t{32});
        }

        void free(void* address) override {
            ::operator delete(address, std::align_val_t{32});
        }

        void reset() override {}
    };

    struct CpuFixture {
        HeapAllocator allocator;
        std::unique_ptr<Device> device = iom::make_cpu_device(allocator);
        std::unique_ptr<DeviceOps> queue = device->create_ops();
    };

    // Head geometry of one invocation. `features` is `Hq*D` and the KV
    // weights have `Hkv*D` output rows, so grouped-query attention is
    // represented without any repeated KV head.
    struct Geometry {
        std::size_t rows = 0;
        std::size_t query_heads = 0;
        std::size_t kv_heads = 0;
        std::size_t head_dim = 0;

        [[nodiscard]] std::size_t features() const noexcept {
            return query_heads * head_dim;
        }

        [[nodiscard]] std::size_t key_value_outputs() const noexcept {
            return kv_heads * head_dim;
        }
    };

    // Nonsymmetric, role-specific values on a small dyadic grid, so a
    // swapped Q/K/V role, an off-by-one row, or a shared head mapping is
    // observably wrong and both operands of every product are exactly
    // representable in BF16.
    [[nodiscard]] double sample(std::size_t role, std::size_t index) {
        const std::size_t pattern =
                (role * 5 + index * 3 + (index / 7) * 11) % 23;
        const double magnitude =
                0.25 * static_cast<double>(pattern % 9 + 1);
        return (pattern % 2 == 0) ? magnitude : -magnitude;
    }

    [[nodiscard]] std::vector<double> sampled(
            std::size_t role, std::size_t count) {
        std::vector<double> values(count);
        for (std::size_t index = 0; index < count; ++index) {
            values[index] = sample(role, index);
        }
        return values;
    }

    [[nodiscard]] std::vector<double> constant(
            double value, std::size_t count) {
        return std::vector<double>(count, value);
    }
    [[nodiscard]] TensorSpec matrix_spec(
            std::size_t planes, std::size_t rows, std::size_t columns) {
        if (planes == 1) {
            return TensorSpec{TensorShape{{rows, columns}}, kLeaf};
        }
        return TensorSpec{
                TensorShape{{planes, rows, columns}}, kLeaf};
    }

    [[nodiscard]] TensorSpec head_spec(std::size_t planes,
            std::size_t heads, std::size_t rows, std::size_t head_dim) {
        if (planes == 1) {
            return TensorSpec{TensorShape{{heads, rows, head_dim}}, kLeaf};
        }
        return TensorSpec{
                TensorShape{{planes, heads, rows, head_dim}}, kLeaf};
    }

    // Owner storage of one invocation. `planes == 2` places every operand in
    // the second plane of a larger owner, so the stage receives genuinely
    // offset, transformed views.
    struct StageBundle {
        std::unique_ptr<Tensor> activation;
        std::unique_ptr<Tensor> attention_scale;
        std::unique_ptr<Tensor> query_weight;
        std::unique_ptr<Tensor> key_weight;
        std::unique_ptr<Tensor> value_weight;
        std::unique_ptr<Tensor> normalized;
        std::unique_ptr<Tensor> query;
        std::unique_ptr<Tensor> key;
        std::unique_ptr<Tensor> value;
        std::unique_ptr<Tensor> rotated_query;
        std::unique_ptr<Tensor> rotated_key;
        std::size_t planes = 1;

        StageBundle(Device& device, const Geometry& geometry,
                    std::size_t plane_count)
            : planes(plane_count) {
            const std::size_t features = geometry.features();
            const std::size_t key_value_outputs = geometry.key_value_outputs();
            activation = device.create_tensor(
                    matrix_spec(planes, geometry.rows, features));
            attention_scale = device.create_tensor(
                    matrix_spec(planes, 1, features));
            query_weight = device.create_tensor(
                    matrix_spec(planes, features, features));
            key_weight = device.create_tensor(
                    matrix_spec(planes, key_value_outputs, features));
            value_weight = device.create_tensor(
                    matrix_spec(planes, key_value_outputs, features));
            normalized = device.create_tensor(
                    matrix_spec(planes, geometry.rows, features));
            query = device.create_tensor(head_spec(planes,
                    geometry.query_heads, geometry.rows,
                    geometry.head_dim));
            key = device.create_tensor(head_spec(planes, geometry.kv_heads,
                    geometry.rows, geometry.head_dim));
            value = device.create_tensor(head_spec(planes,
                    geometry.kv_heads, geometry.rows, geometry.head_dim));
            rotated_query = device.create_tensor(head_spec(planes,
                    geometry.query_heads, geometry.rows,
                    geometry.head_dim));
            rotated_key = device.create_tensor(head_spec(planes,
                    geometry.kv_heads, geometry.rows, geometry.head_dim));
        }

        [[nodiscard]] std::size_t selected_plane() const noexcept {
            return planes - 1;
        }

        [[nodiscard]] TensorView plane_view(
                Tensor& owner, std::size_t plane) const {
            if (planes == 1) {
                return owner.view();
            }
            return owner.view().select(0, plane);
        }

        [[nodiscard]] TensorView selected(Tensor& owner) const {
            return plane_view(owner, selected_plane());
        }

        [[nodiscard]] iom::session_detail::QkvRopeStageViews views() {
            const std::size_t plane = selected_plane();
            return iom::session_detail::QkvRopeStageViews{
                    plane_view(*activation, plane),
                    plane_view(*attention_scale, plane),
                    plane_view(*query_weight, plane),
                    plane_view(*key_weight, plane),
                    plane_view(*value_weight, plane),
                    plane_view(*normalized, plane),
                    plane_view(*query, plane),
                    plane_view(*key, plane),
                    plane_view(*value, plane),
                    plane_view(*rotated_query, plane),
                    plane_view(*rotated_key, plane)};
        }
    };

    // Supplied host values of one invocation.
    struct StageCase {
        Geometry geometry;
        std::size_t a = 0;
        double theta = 10000.0;
        float epsilon = 1.0e-5F;
        std::vector<double> activation;
        std::vector<double> attention_scale;
        std::vector<double> query_weight;
        std::vector<double> key_weight;
        std::vector<double> value_weight;

        [[nodiscard]] iom::session_detail::QkvRopeStageParams params() const {
            return iom::session_detail::QkvRopeStageParams{
                    a, geometry.rows, theta, epsilon};
        }
    };

    [[nodiscard]] StageCase make_case(
            const Geometry& geometry, std::size_t a, double theta) {
        StageCase fixture;
        fixture.geometry = geometry;
        fixture.a = a;
        fixture.theta = theta;
        const std::size_t features = geometry.features();
        const std::size_t key_value_outputs = geometry.key_value_outputs();
        fixture.activation = sampled(1, geometry.rows * features);
        fixture.attention_scale = sampled(2, features);
        fixture.query_weight = sampled(3, features * features);
        fixture.key_weight = sampled(4, key_value_outputs * features);
        fixture.value_weight = sampled(5, key_value_outputs * features);
        return fixture;
    }

    void upload(TensorView view, std::span<const double> values) {
        REQUIRE(values.size() == view.spec().shape.element_count());
        std::vector<std::byte> bytes(values.size() * 2);
        for (std::size_t index = 0; index < values.size(); ++index) {
            const std::uint16_t code = bf16_code(values[index]);
            bytes[2 * index] = static_cast<std::byte>(code & 0xffu);
            bytes[2 * index + 1] = static_cast<std::byte>(code >> 8);
        }
        view.copy_from_host(bytes);
    }

    [[nodiscard]] std::vector<double> read(const TensorView& view) {
        std::vector<std::byte> bytes(view.spec().logical_nbytes());
        view.copy_to_host(bytes);
        std::vector<double> values(bytes.size() / 2);
        for (std::size_t index = 0; index < values.size(); ++index) {
            const std::uint16_t code = static_cast<std::uint16_t>(
                    std::to_integer<std::uint16_t>(bytes[2 * index]))
                    | static_cast<std::uint16_t>(
                            std::to_integer<std::uint16_t>(
                                    bytes[2 * index + 1]) << 8);
            values[index] = bf16_value(code);
        }
        return values;
    }

    // Fills every physical padding slot of one owner with `code`, so a
    // padded read is observable and a padding-independent result is proven.
    void poison_padding(Tensor& owner, std::uint16_t code) {
        const TensorSpec spec = owner.view().spec();
        const std::span<const std::size_t> dimensions =
                spec.shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t columns = dimensions[dimensions.size() - 1];
        const TensorShape padded = spec.standard_padded_shape();
        const std::size_t padded_rows =
                padded.dimension(dimensions.size() - 2);
        const std::size_t padded_columns =
                padded.dimension(dimensions.size() - 1);
        std::size_t planes = 1;
        for (std::size_t axis = 0; axis + 2 < dimensions.size(); ++axis) {
            planes *= dimensions[axis];
        }
        auto* storage =
                static_cast<std::uint8_t*>(owner.view().native_handle());
        for (std::size_t plane = 0; plane < planes; ++plane) {
            for (std::size_t row = 0; row < padded_rows; ++row) {
                for (std::size_t column = 0; column < padded_columns;
                        ++column) {
                    if (row < rows && column < columns) {
                        continue;
                    }
                    const std::size_t slot = iom::detail::standard_plane_slot(
                            spec, plane, row, column);
                    storage[2 * slot] =
                            static_cast<std::uint8_t>(code & 0xffu);
                    storage[2 * slot + 1] =
                            static_cast<std::uint8_t>(code >> 8);
                }
            }
        }
    }

    // Loads the real operands into the view's plane, a distinguishable decoy
    // into every other plane, and `padding_code` into every padding slot.
    void load_case(StageBundle& bundle, const StageCase& fixture,
                   std::uint16_t padding_code) {
        const std::size_t plane = bundle.selected_plane();
        upload(bundle.plane_view(*bundle.activation, plane),
               fixture.activation);
        upload(bundle.plane_view(*bundle.attention_scale, plane),
               fixture.attention_scale);
        upload(bundle.plane_view(*bundle.query_weight, plane),
               fixture.query_weight);
        upload(bundle.plane_view(*bundle.key_weight, plane),
               fixture.key_weight);
        upload(bundle.plane_view(*bundle.value_weight, plane),
               fixture.value_weight);
        if (bundle.planes == 2) {
            upload(bundle.plane_view(*bundle.activation, 0),
                   constant(9.0, fixture.activation.size()));
            upload(bundle.plane_view(*bundle.attention_scale, 0),
                   constant(7.5, fixture.attention_scale.size()));
            upload(bundle.plane_view(*bundle.query_weight, 0),
                   constant(-6.5, fixture.query_weight.size()));
            upload(bundle.plane_view(*bundle.key_weight, 0),
                   constant(5.5, fixture.key_weight.size()));
            upload(bundle.plane_view(*bundle.value_weight, 0),
                   constant(-4.5, fixture.value_weight.size()));
        }
        poison_padding(*bundle.activation, padding_code);
        poison_padding(*bundle.attention_scale, padding_code);
        poison_padding(*bundle.query_weight, padding_code);
        poison_padding(*bundle.key_weight, padding_code);
        poison_padding(*bundle.value_weight, padding_code);
        poison_padding(*bundle.normalized, padding_code);
        poison_padding(*bundle.query, padding_code);
        poison_padding(*bundle.key, padding_code);
        poison_padding(*bundle.value, padding_code);
        poison_padding(*bundle.rotated_query, padding_code);
        poison_padding(*bundle.rotated_key, padding_code);
    }

    struct StageObservation {
        std::vector<double> normalized;
        std::vector<double> query;
        std::vector<double> key;
        std::vector<double> value;
        std::vector<double> rotated_query;
        std::vector<double> rotated_key;
    };

    // Runs one invocation over the supplied tensors. RMSNorm and RoPE
    // consume no raw workspace on any retained backend, and this CPU device
    // cannot create positive scratch, so the caller-provisioned projection
    // slices are the admitted empty default.
    [[nodiscard]] StageObservation run_stage(DeviceOps& queue,
            StageBundle& bundle,
            const iom::session_detail::QkvRopeStageParams& params) {
        iom::session_detail::run_qkv_rope_stage(
                queue, bundle.views(), params,
                iom::session_detail::QkvRopeStageWorkspace{});
        StageObservation observed;
        observed.normalized = read(bundle.selected(*bundle.normalized));
        observed.query = read(bundle.selected(*bundle.query));
        observed.key = read(bundle.selected(*bundle.key));
        observed.value = read(bundle.selected(*bundle.value));
        observed.rotated_query = read(bundle.selected(*bundle.rotated_query));
        observed.rotated_key = read(bundle.selected(*bundle.rotated_key));
        return observed;
    }

    void require_differs(std::span<const double> first,
            std::span<const double> second, const char* label) {
        REQUIRE(first.size() == second.size());
        bool differs = false;
        for (std::size_t index = 0; index < first.size(); ++index) {
            if (std::fabs(first[index] - second[index]) > 1.0e-3) {
                differs = true;
            }
        }
        INFO(label);
        REQUIRE(differs);
    }

    void require_identical(std::span<const double> first,
            std::span<const double> second, const char* label) {
        REQUIRE(first.size() == second.size());
        for (std::size_t index = 0; index < first.size(); ++index) {
            INFO(label << " element " << index << ": " << first[index]
                       << " vs " << second[index]);
            REQUIRE(first[index] == second[index]);
        }
    }

    void require_all_equal(std::span<const double> values, double expected,
                           const char* label) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            INFO(label << " element " << index << ": " << values[index]);
            REQUIRE(values[index] == expected);
        }
    }

    // ------------------------------------------------------------------
    // Recording stub queue: observes the stage's submission order and its
    // branch waits, and injects a branch failure through the established
    // accepted-failure / admission-rejection seam.
    // ------------------------------------------------------------------

    class ProbeQueue final : public DeviceOps {
    public:
        explicit ProbeQueue(const Device& device) : DeviceOps(device) {}

        void defer(bool enabled) {
            std::lock_guard<std::mutex> lock(mutex_);
            deferred_ = enabled;
        }

        // `index` counts linear submissions. An empty message rejects that
        // branch synchronously; a message makes it an accepted branch whose
        // completion retains the failure.
        void fail_linear(std::size_t index, std::string message) {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_index_ = index;
            failure_message_ = std::move(message);
        }

        [[nodiscard]] std::vector<std::string> trace() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return trace_;
        }

        [[nodiscard]] std::size_t pending() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return pending_.size();
        }

        [[nodiscard]] std::vector<oid> accepted() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return accepted_;
        }

        [[nodiscard]] std::vector<oid> failed() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return failed_;
        }

        // Completes every currently pending deferred submission in
        // submission order.
        void release() {
            std::vector<std::uint64_t> pending;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                pending.swap(pending_);
            }
            for (const std::uint64_t sequence : pending) {
                complete(sequence);
            }
        }

        // Waits for one submission checkpoint: the expected number of
        // recorded submissions is present *and* the expected number of them
        // is still incomplete, so a missing producer wait is observable.
        [[nodiscard]] bool await_state(
                std::size_t trace_size, std::size_t pending) const {
            const auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < deadline) {
                if (trace().size() >= trace_size
                        && pending_count() == pending) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return trace().size() >= trace_size && pending_count() == pending;
        }

        [[nodiscard]] std::size_t pending_count() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return pending_.size();
        }

        // Keeps completing accepted work until the stage has nothing left in
        // flight. It bounds a failing implementation's ordering test instead
        // of letting a missed wait hang the joining thread.
        void settle() {
            const auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(10);
            std::size_t last = trace().size();
            while (std::chrono::steady_clock::now() < deadline) {
                defer(false);
                release();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                if (trace().size() == last && pending() == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    if (trace().size() == last && pending() == 0) {
                        return;
                    }
                }
                last = trace().size();
            }
        }

    protected:
        oid rmsnorm_impl(const RmsnormRequest&) override {
            record("rmsnorm");
            return submit_named(std::string{});
        }

        oid linear_impl(const LinearRequest&) override {
            bool reject = false;
            std::string message;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const std::size_t index = linear_submissions_++;
                reject = failed_index_ == index
                        && failure_message_.empty();
                if (failed_index_ == index) {
                    message = failure_message_;
                }
            }
            record("linear");
            if (reject) {
                throw std::invalid_argument(
                        "probe queue rejected a projection branch");
            }
            return submit_named(std::move(message));
        }

        oid rope_impl(const RopeRequest&) override {
            record("rope");
            return submit_named(std::string{});
        }

        [[nodiscard]] bool rmsnorm_supported(
                iom::DataType data_type) const override {
            return data_type == kLeaf;
        }

        [[nodiscard]] iom::WorkspaceRequirements
        linear_workspace_requirements_impl(const TensorView&,
                const TensorView&, const TensorView&, std::size_t,
                std::size_t, iom::LinearOutputLayout, std::size_t,
                std::size_t) override {
            return {0, 1};
        }

        [[nodiscard]] iom::WorkspaceRequirements rope_workspace_requirements(
                const RopeRequest&) override {
            return {0, 1};
        }

    private:
        void record(const char* name) {
            std::lock_guard<std::mutex> lock(mutex_);
            trace_.emplace_back(name);
        }

        [[nodiscard]] oid submit_named(std::string failure_message) {
            const bool retains_failure = !failure_message.empty();
            const oid token = submit(
                    [this, message = std::move(failure_message)](
                            std::uint64_t sequence) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (!message.empty()) {
                            commit_failure(
                                    sequence,
                                    std::make_exception_ptr(
                                            std::runtime_error(message)));
                            complete(sequence);
                            return;
                        }
                        pending_.push_back(sequence);
                        if (!deferred_) {
                            pending_.pop_back();
                            complete(sequence);
                        }
                    });
            std::lock_guard<std::mutex> lock(mutex_);
            accepted_.push_back(token);
            if (retains_failure) {
                failed_.push_back(token);
            }
            return token;
        }

        mutable std::mutex mutex_;
        std::vector<std::string> trace_;
        std::vector<std::uint64_t> pending_;
        std::vector<oid> accepted_;
        std::vector<oid> failed_;
        std::size_t linear_submissions_ = 0;
        std::size_t failed_index_ = static_cast<std::size_t>(-1);
        std::string failure_message_;
        bool deferred_ = false;
    };

    // ------------------------------------------------------------------
    // Cases.
    // ------------------------------------------------------------------

    TEST_CASE(
            "TinyLlama QKV RoPE stage prefill matches an independent "
            "reference") {
        CpuFixture fixture;
        const Geometry geometry{15, 4, 2, 4};
        const StageCase values = make_case(geometry, 3, 1234.5);
        StageBundle bundle{*fixture.device, geometry, 1};
        load_case(bundle, values, 0x7e00);
        const StageObservation observed =
                run_stage(*fixture.queue, bundle, values.params());

        require_reference(observed.normalized,
                reference_normalization(values.activation,
                        geometry.rows, geometry.features(),
                        values.attention_scale, values.epsilon),
                "normalized activation");
        // Downstream boundaries consume the observed upstream BF16 values,
        // so the reference isolates one operation boundary at a time.
        require_reference(observed.query,
                reference_head_planar(observed.normalized, geometry.rows,
                        geometry.features(), values.query_weight,
                        geometry.query_heads, geometry.head_dim),
                "query projection");
        require_reference(observed.key,
                reference_head_planar(observed.normalized, geometry.rows,
                        geometry.features(), values.key_weight,
                        geometry.kv_heads, geometry.head_dim),
                "key projection");
        require_reference(observed.value,
                reference_head_planar(observed.normalized, geometry.rows,
                        geometry.features(), values.value_weight,
                        geometry.kv_heads, geometry.head_dim),
                "value projection");
        require_reference(observed.rotated_query,
                reference_rope(observed.query, geometry.query_heads,
                        geometry.rows, geometry.head_dim, values.a,
                        values.theta),
                "rotated query");
        require_reference(observed.rotated_key,
                reference_rope(observed.key, geometry.kv_heads,
                        geometry.rows, geometry.head_dim, values.a,
                        values.theta),
                "rotated key");

        // Rotation is applied to Q and K, K and V rows are independent, and
        // the grouped-query head counts stay distinct.
        require_differs(observed.rotated_query, observed.query,
                "rotated query must differ from the query projection");
        require_differs(observed.rotated_key, observed.key,
                "rotated key must differ from the key projection");
        require_differs(
                std::span<const double>(observed.key).subspan(
                        0, geometry.kv_heads * geometry.head_dim),
                std::span<const double>(observed.value).subspan(
                        0, geometry.kv_heads * geometry.head_dim),
                "key and value rows must be distinct");
    }

    TEST_CASE(
            "TinyLlama QKV RoPE stage one-row decode uses the absolute "
            "position") {
        CpuFixture fixture;
        const Geometry geometry{1, 4, 2, 2};
        const StageCase values = make_case(geometry, 9, 10000.0);
        StageBundle bundle{*fixture.device, geometry, 1};
        load_case(bundle, values, 0x7c00);
        const StageObservation positioned =
                run_stage(*fixture.queue, bundle, values.params());

        require_reference(positioned.normalized,
                reference_normalization(values.activation,
                        geometry.rows, geometry.features(),
                        values.attention_scale, values.epsilon),
                "decode normalized activation");
        require_reference(positioned.rotated_query,
                reference_rope(positioned.query, geometry.query_heads,
                        geometry.rows, geometry.head_dim, values.a,
                        values.theta),
                "decode rotated query");
        require_reference(positioned.rotated_key,
                reference_rope(positioned.key, geometry.kv_heads,
                        geometry.rows, geometry.head_dim, values.a,
                        values.theta),
                "decode rotated key");

        StageCase origin = values;
        origin.a = 0;
        const StageObservation initial =
                run_stage(*fixture.queue, bundle, origin.params());
        // Only the rotation depends on the absolute position: the
        // projections are identical, the rotated results are not.
        require_identical(initial.normalized, positioned.normalized,
                "normalization must not depend on the position");
        require_identical(initial.query, positioned.query,
                "query projection must not depend on the position");
        require_identical(initial.key, positioned.key,
                "key projection must not depend on the position");
        require_differs(initial.rotated_query, positioned.rotated_query,
                "rotated query must depend on the absolute position");
        require_differs(initial.rotated_key, positioned.rotated_key,
                "rotated key must depend on the absolute position");
    }

    TEST_CASE(
            "TinyLlama QKV RoPE stage isolates padding and offset views") {
        CpuFixture fixture;
        const Geometry geometry{15, 4, 2, 4};
        const StageCase values = make_case(geometry, 2, 4096.0);
        StageBundle bundle{*fixture.device, geometry, 2};
        load_case(bundle, values, 0x7f00);
        const StageObservation first =
                run_stage(*fixture.queue, bundle, values.params());

        require_reference(first.normalized,
                reference_normalization(values.activation,
                        geometry.rows, geometry.features(),
                        values.attention_scale, values.epsilon),
                "offset normalized activation");
        require_reference(first.query,
                reference_head_planar(first.normalized, geometry.rows,
                        geometry.features(), values.query_weight,
                        geometry.query_heads, geometry.head_dim),
                "offset query projection");
        require_reference(first.rotated_key,
                reference_rope(first.key, geometry.kv_heads,
                        geometry.rows, geometry.head_dim, values.a,
                        values.theta),
                "offset rotated key");

        // Different physical padding and a different decoy plane may not
        // change one logical result.
        upload(bundle.plane_view(*bundle.activation, 0),
               constant(-3.25, values.activation.size()));
        upload(bundle.plane_view(*bundle.query_weight, 0),
               constant(11.5, values.query_weight.size()));
        for (Tensor* owner :
                {bundle.activation.get(), bundle.attention_scale.get(),
                 bundle.query_weight.get(), bundle.key_weight.get(),
                 bundle.value_weight.get(), bundle.normalized.get(),
                 bundle.query.get(), bundle.key.get(), bundle.value.get(),
                 bundle.rotated_query.get(), bundle.rotated_key.get()}) {
            poison_padding(*owner, 0x0001);
        }
        const StageObservation second =
                run_stage(*fixture.queue, bundle, values.params());
        require_identical(first.normalized, second.normalized,
                "normalization must ignore padding and other planes");
        require_identical(first.query, second.query,
                "query projection must ignore padding and other planes");
        require_identical(first.key, second.key,
                "key projection must ignore padding and other planes");
        require_identical(first.value, second.value,
                "value projection must ignore padding and other planes");
        require_identical(first.rotated_query, second.rotated_query,
                "rotated query must ignore padding and other planes");
        require_identical(first.rotated_key, second.rotated_key,
                "rotated key must ignore padding and other planes");
    }

    TEST_CASE(
            "TinyLlama QKV RoPE stage preflights scalar and position "
            "rejections") {
        CpuFixture fixture;
        const Geometry geometry{1, 2, 1, 2};
        const StageCase values = make_case(geometry, 0, 10000.0);
        StageBundle bundle{*fixture.device, geometry, 1};
        load_case(bundle, values, 0x7d00);
        ProbeQueue queue(*fixture.device);

        auto invalid_theta = values.params();
        invalid_theta.theta = 0.5;
        CHECK_THROWS_AS(run_stage(queue, bundle, invalid_theta),
                std::invalid_argument);
        CHECK(queue.trace().empty());
        CHECK(queue.pending() == 0);

        auto overflowing_position = values.params();
        overflowing_position.a =
                std::numeric_limits<std::size_t>::max();
        CHECK_THROWS_AS(run_stage(queue, bundle, overflowing_position),
                std::overflow_error);
        CHECK(queue.trace().empty());
        CHECK(queue.pending() == 0);
    }

    TEST_CASE(
            "TinyLlama QKV RoPE stage orders projection and rotation "
            "branches") {
        CpuFixture fixture;
        const Geometry geometry{4, 2, 1, 2};
        const StageCase values = make_case(geometry, 1, 10000.0);
        StageBundle bundle{*fixture.device, geometry, 1};
        for (Tensor* owner :
                {bundle.activation.get(), bundle.attention_scale.get(),
                 bundle.query_weight.get(), bundle.key_weight.get(),
                 bundle.value_weight.get()}) {
            upload(owner->view(), constant(1.0,
                    owner->view().spec().shape.element_count()));
        }
        ProbeQueue queue(*fixture.device);
        queue.defer(true);

        std::exception_ptr failure;
        std::thread runner([&] {
            try {
                run_stage(queue, bundle, values.params());
            } catch (...) {
                failure = std::current_exception();
            }
        });

        // The normalization is submitted and waited before any projection.
        const bool normalized_submitted = queue.await_state(1, 1);
        const std::vector<std::string> after_normalization = queue.trace();
        const std::size_t pending_after_normalization = queue.pending();
        queue.release();

        // All three projections are submitted and none completes before
        // either rotation is submitted.
        const bool projections_submitted = queue.await_state(4, 3);
        const std::vector<std::string> after_projections = queue.trace();
        const std::size_t pending_after_projections = queue.pending();
        queue.release();

        // Only then are the two independent rotations submitted.
        const bool rotations_submitted = queue.await_state(6, 2);
        const std::vector<std::string> after_rotations = queue.trace();
        const std::size_t pending_after_rotations = queue.pending();
        queue.release();
        queue.settle();
        runner.join();

        CHECK(normalized_submitted);
        CHECK(after_normalization == std::vector<std::string>{"rmsnorm"});
        CHECK(pending_after_normalization == 1);
        CHECK(projections_submitted);
        CHECK(after_projections
                == std::vector<std::string>{
                        "rmsnorm", "linear", "linear", "linear"});
        CHECK(pending_after_projections == 3);
        CHECK(rotations_submitted);
        CHECK(after_rotations
                == std::vector<std::string>{
                        "rmsnorm", "linear", "linear", "linear", "rope",
                        "rope"});
        CHECK(pending_after_rotations == 2);
        CHECK(failure == nullptr);
    }

    TEST_CASE(
            "TinyLlama QKV RoPE stage drains accepted branches after a "
            "projection failure") {
        CpuFixture fixture;
        const Geometry geometry{4, 2, 1, 2};
        const StageCase values = make_case(geometry, 5, 10000.0);
        StageBundle bundle{*fixture.device, geometry, 1};
        for (Tensor* owner :
                {bundle.activation.get(), bundle.attention_scale.get(),
                 bundle.query_weight.get(), bundle.key_weight.get(),
                 bundle.value_weight.get(), bundle.normalized.get(),
                 bundle.query.get(), bundle.key.get(), bundle.value.get(),
                 bundle.rotated_query.get(), bundle.rotated_key.get()}) {
            upload(owner->view(), constant(1.0,
                    owner->view().spec().shape.element_count()));
        }
        ProbeQueue queue(*fixture.device);
        queue.fail_linear(1, "key projection branch failed");

        bool threw = false;
        try {
            run_stage(queue, bundle, values.params());
        } catch (const std::exception& error) {
            threw = std::string(error.what())
                    == "key projection branch failed";
        }
        CHECK(threw);

        // No dependent rotation was submitted, and every accepted branch of
        // the stage is terminal.
        CHECK(queue.trace()
                == std::vector<std::string>{
                        "rmsnorm", "linear", "linear", "linear"});
        CHECK(queue.pending() == 0);
        const std::vector<oid> failed = queue.failed();
        REQUIRE(failed.size() == 1);
        // The retained failure stays observable on repeated waits, and the
        // drained branches stay repeat-waitable.
        CHECK_THROWS_AS(queue.wait(failed[0]), std::runtime_error);
        CHECK_THROWS_AS(queue.wait(failed[0]), std::runtime_error);
        for (const oid token : queue.accepted()) {
            if (token != failed[0]) {
                CHECK_NOTHROW(queue.wait(token));
            }
        }
        // No output was published as usable.
        require_all_equal(read(bundle.selected(*bundle.rotated_query)), 1.0,
                "rotated query must stay unpublished");
        require_all_equal(read(bundle.selected(*bundle.rotated_key)), 1.0,
                "rotated key must stay unpublished");
    }

    TEST_CASE(
            "TinyLlama QKV RoPE stage drains accepted branches after a "
            "rejected projection") {
        CpuFixture fixture;
        const Geometry geometry{4, 2, 1, 2};
        const StageCase values = make_case(geometry, 5, 10000.0);
        StageBundle bundle{*fixture.device, geometry, 1};
        for (Tensor* owner :
                {bundle.activation.get(), bundle.attention_scale.get(),
                 bundle.query_weight.get(), bundle.key_weight.get(),
                 bundle.value_weight.get(), bundle.normalized.get(),
                 bundle.query.get(), bundle.key.get(), bundle.value.get(),
                 bundle.rotated_query.get(), bundle.rotated_key.get()}) {
            upload(owner->view(), constant(1.0,
                    owner->view().spec().shape.element_count()));
        }
        ProbeQueue queue(*fixture.device);
        queue.fail_linear(1, std::string{});

        CHECK_THROWS_AS(run_stage(queue, bundle, values.params()),
                std::invalid_argument);

        CHECK(queue.trace()
                == std::vector<std::string>{
                        "rmsnorm", "linear", "linear", "linear"});
        CHECK(queue.pending() == 0);
        CHECK(queue.failed().empty());
        for (const oid token : queue.accepted()) {
            CHECK_NOTHROW(queue.wait(token));
        }
        require_all_equal(read(bundle.selected(*bundle.query)), 1.0,
                "query projection must stay unpublished");
        require_all_equal(read(bundle.selected(*bundle.rotated_query)), 1.0,
                "rotated query must stay unpublished");
        require_all_equal(read(bundle.selected(*bundle.rotated_key)), 1.0,
                "rotated key must stay unpublished");
    }

}  // namespace qkv_rope_stage_test

// ---------------------------------------------------------------------------
// Complete one-layer decoder composition (leaf 06-decoder-layer-forward).
//
// This fixture owns every stage boundary explicitly and compares only the
// final residual against an independent host reference.  It also exercises a
// cached one-row continuation, causal future-token isolation, exact capacity,
// and an accepted append failure that must prevent SDPA and the MLP.
// ---------------------------------------------------------------------------

namespace decoder_layer_forward_test {

using iom::DataType;
using iom::Device;
using iom::DeviceOps;
using iom::LinearOutputLayout;
using iom::RawWorkspace;
using iom::Tensor;
using iom::TensorShape;
using iom::TensorSpec;
using iom::TensorView;
using iom::oid;
using iom::session_detail::CacheAttentionStageRequest;
using iom::session_detail::DecoderLayerForwardParams;
using iom::session_detail::DecoderLayerForwardState;
using iom::session_detail::DecoderLayerForwardViews;
using iom::session_detail::DecoderLayerForwardWorkspace;
using iom::session_detail::MlpStageViews;
using iom::session_detail::MlpWorkspace;
using iom::session_detail::QkvRopeStageViews;
using iom::session_detail::QkvRopeStageWorkspace;

constexpr std::size_t kQueryHeads = 4;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kHeadDim = 2;
constexpr std::size_t kFeatures = kQueryHeads * kHeadDim;
constexpr std::size_t kIntermediate = 6;
constexpr float kEpsilon = 1.0e-5F;
constexpr double kTheta = 10000.0;
constexpr float kUntouched = 64.0F;

[[nodiscard]] float sample(
        std::uint32_t tag, std::size_t row, std::size_t column) {
    const std::uint32_t mixed =
            tag * 37u + static_cast<std::uint32_t>(row) * 17u
            + static_cast<std::uint32_t>(column) * 11u
            + static_cast<std::uint32_t>(row * column) * 5u;
    const float magnitude =
            static_cast<float>(mixed % 13u + 1u) / 4.0F;
    return round_bf16((mixed & 1u) == 0u ? magnitude : -magnitude);
}

[[nodiscard]] std::vector<float> sampled(
        std::uint32_t tag, std::size_t rows, std::size_t columns,
        std::size_t row_start = 0) {
    std::vector<float> values(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            values[row * columns + column] =
                    sample(tag, row_start + row, column);
        }
    }
    return values;
}

[[nodiscard]] std::unique_ptr<Tensor> make_tensor(
        Device& device, std::initializer_list<std::size_t> dimensions) {
    return device.create_tensor(TensorSpec{
            TensorShape{std::vector<std::size_t>(
                    dimensions.begin(), dimensions.end())},
            DataType::BF16});
}

void upload(Tensor& tensor, std::span<const float> values) {
    REQUIRE(values.size() == tensor.view().spec().shape.element_count());
    std::vector<std::byte> bytes(values.size() * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint16_t code = encode_bf16(values[index]);
        bytes[2 * index] = static_cast<std::byte>(code & 0xffu);
        bytes[2 * index + 1] = static_cast<std::byte>(code >> 8);
    }
    tensor.view().copy_from_host(bytes);
}

[[nodiscard]] std::vector<float> read(const Tensor& tensor) {
    const std::size_t elements = tensor.view().spec().shape.element_count();
    std::vector<std::byte> bytes(elements * sizeof(std::uint16_t));
    tensor.view().copy_to_host(bytes);
    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const std::uint16_t code =
                static_cast<std::uint16_t>(
                        std::to_integer<std::uint16_t>(bytes[2 * index]))
                | static_cast<std::uint16_t>(
                        std::to_integer<std::uint16_t>(bytes[2 * index + 1])
                        << 8);
        values[index] = decode_bf16(code);
    }
    return values;
}

void fill(Tensor& tensor, float value) {
    const std::size_t elements = tensor.view().spec().shape.element_count();
    upload(tensor, std::vector<float>(elements, value));
}

struct ReferenceResult {
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    std::vector<float> next_x;
};

struct Fixture {
    Fixture(std::size_t rows, std::size_t capacity, std::size_t offset,
            std::uint32_t tag, std::size_t input_row_start = 0)
        : rows(rows),
          capacity(capacity),
          offset(offset),
          tag(tag),
          input_row_start(input_row_start),
          input(sampled(tag + 1, rows, kFeatures, input_row_start)),
          attention_scale(sampled(tag + 2, 1, kFeatures)),
          post_attention_scale(sampled(tag + 3, 1, kFeatures)),
          query_weight(sampled(tag + 4, kFeatures, kFeatures)),
          key_weight(sampled(tag + 5, kKvHeads * kHeadDim, kFeatures)),
          value_weight(sampled(tag + 6, kKvHeads * kHeadDim, kFeatures)),
          output_weight(sampled(tag + 7, kFeatures, kFeatures)),
          gate_weight(sampled(tag + 8, kIntermediate, kFeatures)),
          up_weight(sampled(tag + 9, kIntermediate, kFeatures)),
          down_weight(sampled(tag + 10, kFeatures, kIntermediate)),
          k_cache_values(kKvHeads * capacity * kHeadDim, -7.5F),
          v_cache_values(kKvHeads * capacity * kHeadDim, 6.25F),
          cpu(),
          operations(cpu.device->create_ops()) {
        activation = make_tensor(*cpu.device, {rows, kFeatures});
        attention_norm = make_tensor(*cpu.device, {1, kFeatures});
        post_norm = make_tensor(*cpu.device, {1, kFeatures});
        query_weight_owner =
                make_tensor(*cpu.device, {kFeatures, kFeatures});
        key_weight_owner =
                make_tensor(*cpu.device, {kKvHeads * kHeadDim, kFeatures});
        value_weight_owner =
                make_tensor(*cpu.device, {kKvHeads * kHeadDim, kFeatures});
        output_weight_owner =
                make_tensor(*cpu.device, {kFeatures, kFeatures});
        gate_weight_owner =
                make_tensor(*cpu.device, {kIntermediate, kFeatures});
        up_weight_owner =
                make_tensor(*cpu.device, {kIntermediate, kFeatures});
        down_weight_owner =
                make_tensor(*cpu.device, {kFeatures, kIntermediate});

        normalized = make_tensor(*cpu.device, {rows, kFeatures});
        query = make_tensor(
                *cpu.device, {kQueryHeads, rows, kHeadDim});
        key = make_tensor(*cpu.device, {kKvHeads, rows, kHeadDim});
        value = make_tensor(*cpu.device, {kKvHeads, rows, kHeadDim});
        rotated_query = make_tensor(
                *cpu.device, {kQueryHeads, rows, kHeadDim});
        rotated_key = make_tensor(
                *cpu.device, {kKvHeads, rows, kHeadDim});
        k_cache = make_tensor(
                *cpu.device, {kKvHeads, capacity, kHeadDim});
        v_cache = make_tensor(
                *cpu.device, {kKvHeads, capacity, kHeadDim});
        merged = make_tensor(*cpu.device, {rows, kFeatures});
        attention_output = make_tensor(*cpu.device, {rows, kFeatures});
        first_residual = make_tensor(*cpu.device, {rows, kFeatures});

        n2 = make_tensor(*cpu.device, {rows, kFeatures});
        gate = make_tensor(*cpu.device, {rows, kIntermediate});
        up = make_tensor(*cpu.device, {rows, kIntermediate});
        activated_gate = make_tensor(*cpu.device, {rows, kIntermediate});
        product = make_tensor(*cpu.device, {rows, kIntermediate});
        down = make_tensor(*cpu.device, {rows, kFeatures});
        next_x = make_tensor(*cpu.device, {rows, kFeatures});

        upload(*activation, input);
        upload(*attention_norm, attention_scale);
        upload(*post_norm, post_attention_scale);
        upload(*query_weight_owner, query_weight);
        upload(*key_weight_owner, key_weight);
        upload(*value_weight_owner, value_weight);
        upload(*output_weight_owner, output_weight);
        upload(*gate_weight_owner, gate_weight);
        upload(*up_weight_owner, up_weight);
        upload(*down_weight_owner, down_weight);
        upload(*k_cache, k_cache_values);
        upload(*v_cache, v_cache_values);
        fill(*normalized, kUntouched);
        fill(*query, kUntouched);
        fill(*key, kUntouched);
        fill(*value, kUntouched);
        fill(*rotated_query, kUntouched);
        fill(*rotated_key, kUntouched);
        fill(*merged, kUntouched);
        fill(*attention_output, kUntouched);
        fill(*first_residual, kUntouched);
        fill(*n2, kUntouched);
        fill(*gate, kUntouched);
        fill(*up, kUntouched);
        fill(*activated_gate, kUntouched);
        fill(*product, kUntouched);
        fill(*down, kUntouched);
        fill(*next_x, kUntouched);

        const auto requirement = operations->sdpa_workspace_requirements(
                rotated_query->view(), k_cache->view(), v_cache->view(),
                merged->view(), offset, offset + rows);
        attention_workspace = cpu.device->create_workspace(requirement.bytes);
    }

    [[nodiscard]] DecoderLayerForwardViews views() const {
        return DecoderLayerForwardViews{
                QkvRopeStageViews{
                        activation->view(), attention_norm->view(),
                        query_weight_owner->view(), key_weight_owner->view(),
                        value_weight_owner->view(), normalized->view(),
                        query->view(), key->view(), value->view(),
                        rotated_query->view(), rotated_key->view()},
                CacheAttentionStageRequest{
                        rotated_query->view(), rotated_key->view(), value->view(),
                        k_cache->view(), v_cache->view(), activation->view(),
                        output_weight_owner->view(), merged->view(),
                        attention_output->view(), first_residual->view(),
                        offset, rows, capacity},
                MlpStageViews{
                        first_residual->view(), post_norm->view(),
                        gate_weight_owner->view(), up_weight_owner->view(),
                        down_weight_owner->view(), n2->view(), gate->view(),
                        up->view(), activated_gate->view(), product->view(),
                        down->view(), next_x->view()}};
    }

    [[nodiscard]] DecoderLayerForwardParams params() const noexcept {
        return DecoderLayerForwardParams{
                offset, rows, capacity, kFeatures, kIntermediate, kTheta,
                kEpsilon, kEpsilon};
    }

    [[nodiscard]] DecoderLayerForwardWorkspace workspace() const {
        return DecoderLayerForwardWorkspace{
                QkvRopeStageWorkspace{}, attention_workspace->view(),
                MlpWorkspace{}};
    }

    void replace_input(std::size_t row, std::span<const float> values) {
        REQUIRE(row < rows);
        REQUIRE(values.size() == kFeatures);
        std::copy(values.begin(), values.end(),
                input.begin() + static_cast<std::ptrdiff_t>(row * kFeatures));
        upload(*activation, input);
    }

    void set_cache(
            std::span<const float> new_k, std::span<const float> new_v) {
        REQUIRE(new_k.size() == k_cache_values.size());
        REQUIRE(new_v.size() == v_cache_values.size());
        k_cache_values.assign(new_k.begin(), new_k.end());
        v_cache_values.assign(new_v.begin(), new_v.end());
        upload(*k_cache, k_cache_values);
        upload(*v_cache, v_cache_values);
    }

    std::size_t rows;
    std::size_t capacity;
    std::size_t offset;
    std::uint32_t tag;
    std::size_t input_row_start;
    std::vector<float> input;
    std::vector<float> attention_scale;
    std::vector<float> post_attention_scale;
    std::vector<float> query_weight;
    std::vector<float> key_weight;
    std::vector<float> value_weight;
    std::vector<float> output_weight;
    std::vector<float> gate_weight;
    std::vector<float> up_weight;
    std::vector<float> down_weight;
    std::vector<float> k_cache_values;
    std::vector<float> v_cache_values;

    MlpCpuFixture cpu;
    std::unique_ptr<DeviceOps> operations;
    std::unique_ptr<Tensor> activation;
    std::unique_ptr<Tensor> attention_norm;
    std::unique_ptr<Tensor> post_norm;
    std::unique_ptr<Tensor> query_weight_owner;
    std::unique_ptr<Tensor> key_weight_owner;
    std::unique_ptr<Tensor> value_weight_owner;
    std::unique_ptr<Tensor> output_weight_owner;
    std::unique_ptr<Tensor> gate_weight_owner;
    std::unique_ptr<Tensor> up_weight_owner;
    std::unique_ptr<Tensor> down_weight_owner;
    std::unique_ptr<Tensor> normalized;
    std::unique_ptr<Tensor> query;
    std::unique_ptr<Tensor> key;
    std::unique_ptr<Tensor> value;
    std::unique_ptr<Tensor> rotated_query;
    std::unique_ptr<Tensor> rotated_key;
    std::unique_ptr<Tensor> k_cache;
    std::unique_ptr<Tensor> v_cache;
    std::unique_ptr<Tensor> merged;
    std::unique_ptr<Tensor> attention_output;
    std::unique_ptr<Tensor> first_residual;
    std::unique_ptr<Tensor> n2;
    std::unique_ptr<Tensor> gate;
    std::unique_ptr<Tensor> up;
    std::unique_ptr<Tensor> activated_gate;
    std::unique_ptr<Tensor> product;
    std::unique_ptr<Tensor> down;
    std::unique_ptr<Tensor> next_x;
    std::unique_ptr<RawWorkspace> attention_workspace;
};

[[nodiscard]] ReferenceResult reference(const Fixture& fixture) {
    const std::size_t rows = fixture.rows;
    const std::size_t features = kFeatures;
    const std::size_t width = kHeadDim;
    const std::size_t length = fixture.offset + rows;
    std::vector<float> normalized(rows * features);
    for (std::size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0F;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const float value = fixture.input[row * features + feature];
            square_sum += value * value;
        }
        const float inverse = 1.0F / std::sqrt(
                square_sum / static_cast<float>(features) + kEpsilon);
        for (std::size_t feature = 0; feature < features; ++feature) {
            normalized[row * features + feature] = round_bf16(
                    fixture.input[row * features + feature] * inverse
                    * fixture.attention_scale[feature]);
        }
    }

    std::vector<float> query(kQueryHeads * rows * width);
    std::vector<float> key(kKvHeads * rows * width);
    std::vector<float> value(kKvHeads * rows * width);
    auto project = [&](std::vector<float>& output,
                       std::span<const float> weight, std::size_t heads) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t element = 0; element < width; ++element) {
                    const std::size_t output_row = head * width + element;
                    float sum = 0.0F;
                    for (std::size_t feature = 0; feature < features;
                         ++feature) {
                        sum += normalized[row * features + feature]
                                * weight[output_row * features + feature];
                    }
                    output[(head * rows + row) * width + element] =
                            round_bf16(sum);
                }
            }
        }
    };
    project(query, fixture.query_weight, kQueryHeads);
    project(key, fixture.key_weight, kKvHeads);
    project(value, fixture.value_weight, kKvHeads);

    auto rotate = [](std::vector<float>& values, std::size_t heads,
                     std::size_t rows, std::size_t offset) {
        const std::size_t half = kHeadDim / 2;
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t row = 0; row < rows; ++row) {
                const float position =
                        static_cast<float>(offset + row);
                const std::size_t base = (head * rows + row) * kHeadDim;
                for (std::size_t pair = 0; pair < half; ++pair) {
                    const float angle = position * static_cast<float>(
                            std::pow(kTheta, -2.0 * static_cast<double>(pair)
                                                       / kHeadDim));
                    const float cosine = std::cos(angle);
                    const float sine = std::sin(angle);
                    const float first = values[base + pair];
                    const float second = values[base + half + pair];
                    values[base + pair] =
                            round_bf16(first * cosine - second * sine);
                    values[base + half + pair] =
                            round_bf16(second * cosine + first * sine);
                }
            }
        }
    };
    rotate(query, kQueryHeads, rows, fixture.offset);
    rotate(key, kKvHeads, rows, fixture.offset);

    ReferenceResult result;
    result.k_cache = fixture.k_cache_values;
    result.v_cache = fixture.v_cache_values;
    for (std::size_t head = 0; head < kKvHeads; ++head) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t feature = 0; feature < width; ++feature) {
                const std::size_t cache_index =
                        (head * fixture.capacity + fixture.offset + row)
                                * width + feature;
                const std::size_t run_index =
                        (head * rows + row) * width + feature;
                result.k_cache[cache_index] = key[run_index];
                result.v_cache[cache_index] = value[run_index];
            }
        }
    }

    std::vector<float> merged(rows * features, 0.0F);
    const std::size_t group = kQueryHeads / kKvHeads;
    for (std::size_t head = 0; head < kQueryHeads; ++head) {
        const std::size_t kv_head = head / group;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t visible =
                    std::min(length, fixture.offset + row + 1);
            std::vector<float> scores(visible);
            for (std::size_t token = 0; token < visible; ++token) {
                float dot = 0.0F;
                for (std::size_t feature = 0; feature < width; ++feature) {
                    dot += query[(head * rows + row) * width + feature]
                            * result.k_cache[
                                    (kv_head * fixture.capacity + token)
                                            * width + feature];
                }
                scores[token] = dot / std::sqrt(static_cast<float>(width));
            }
            float maximum = scores[0];
            for (const float score : scores) maximum = std::max(maximum, score);
            float probability_sum = 0.0F;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                probability_sum += score;
            }
            for (float& score : scores) {
                score = round_bf16(score / probability_sum);
            }
            for (std::size_t feature = 0; feature < width; ++feature) {
                float accumulated = 0.0F;
                for (std::size_t token = 0; token < visible; ++token) {
                    accumulated +=
                            scores[token]
                            * result.v_cache[
                                    (kv_head * fixture.capacity + token)
                                            * width + feature];
                }
                merged[row * features + head * width + feature] =
                        round_bf16(accumulated);
            }
        }
    }

    std::vector<float> projected(rows * features);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t output = 0; output < features; ++output) {
            float sum = 0.0F;
            for (std::size_t feature = 0; feature < features; ++feature) {
                sum += merged[row * features + feature]
                        * fixture.output_weight[output * features + feature];
            }
            projected[row * features + output] = round_bf16(sum);
        }
    }
    std::vector<float> first_residual(rows * features);
    for (std::size_t index = 0; index < first_residual.size(); ++index) {
        first_residual[index] =
                round_bf16(fixture.input[index] + projected[index]);
    }

    std::vector<float> mlp_norm(rows * features);
    for (std::size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0F;
        for (std::size_t feature = 0; feature < features; ++feature) {
            const float value = first_residual[row * features + feature];
            square_sum += value * value;
        }
        const float inverse = 1.0F / std::sqrt(
                square_sum / static_cast<float>(features) + kEpsilon);
        for (std::size_t feature = 0; feature < features; ++feature) {
            mlp_norm[row * features + feature] = round_bf16(
                    first_residual[row * features + feature] * inverse
                    * fixture.post_attention_scale[feature]);
        }
    }
    std::vector<float> gate(rows * kIntermediate);
    std::vector<float> up(rows * kIntermediate);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t output = 0; output < kIntermediate; ++output) {
            float gate_sum = 0.0F;
            float up_sum = 0.0F;
            for (std::size_t feature = 0; feature < features; ++feature) {
                gate_sum += mlp_norm[row * features + feature]
                        * fixture.gate_weight[output * features + feature];
                up_sum += mlp_norm[row * features + feature]
                        * fixture.up_weight[output * features + feature];
            }
            gate[row * kIntermediate + output] = round_bf16(gate_sum);
            up[row * kIntermediate + output] = round_bf16(up_sum);
        }
    }
    std::vector<float> product(rows * kIntermediate);
    for (std::size_t index = 0; index < product.size(); ++index) {
        const float activated =
                round_bf16(gate[index] / (1.0F + std::exp(-gate[index])));
        product[index] = round_bf16(activated * up[index]);
    }
    std::vector<float> down(rows * features);
    result.next_x.resize(rows * features);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t output = 0; output < features; ++output) {
            float sum = 0.0F;
            for (std::size_t intermediate = 0;
                 intermediate < kIntermediate; ++intermediate) {
                sum += product[row * kIntermediate + intermediate]
                        * fixture.down_weight[output * kIntermediate
                                               + intermediate];
            }
            down[row * features + output] = round_bf16(sum);
            result.next_x[row * features + output] = round_bf16(
                    first_residual[row * features + output]
                    + down[row * features + output]);
        }
    }
    return result;
}

[[nodiscard]] std::vector<float> row(
        std::span<const float> values, std::size_t row_index,
        std::size_t width) {
    return std::vector<float>(
            values.begin() + static_cast<std::ptrdiff_t>(row_index * width),
            values.begin()
                    + static_cast<std::ptrdiff_t>((row_index + 1) * width));
}

void require_untouched(const Tensor& tensor, const char* label) {
    const std::vector<float> values = read(tensor);
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK_MESSAGE(encode_bf16(values[index]) == encode_bf16(kUntouched),
                label << " element " << index << " was published");
    }
}

TEST_CASE(
        "TinyLlama decoder layer forward matches an independent reference "
        "and cached continuation") {
    Fixture full(3, 3, 0, 19);
    const ReferenceResult full_reference = reference(full);
    auto full_views = full.views();
    DecoderLayerForwardState full_state;
    CHECK_NOTHROW(iom::session_detail::run_decoder_layer_forward(
            *full.operations, full_views, full.params(), full.workspace(), {},
            full_state));
    CHECK_FALSE(full_state.failed);
    CHECK_EQ(full_state.initialized_length, 3u);
    check_close(read(*full.next_x), full_reference.next_x, 3, 5e-2F,
            "one-layer full output");

    Fixture prefill(2, 3, 0, 19);
    const ReferenceResult prefill_reference = reference(prefill);
    auto prefill_views = prefill.views();
    DecoderLayerForwardState prefill_state;
    CHECK_NOTHROW(iom::session_detail::run_decoder_layer_forward(
            *prefill.operations, prefill_views, prefill.params(),
            prefill.workspace(), {}, prefill_state));
    CHECK_EQ(prefill_state.initialized_length, 2u);
    check_close(read(*prefill.next_x), prefill_reference.next_x, 3, 5e-2F,
            "one-layer causal prefill");

    Fixture continuation(1, 3, 2, 19, 2);
    continuation.set_cache(read(*prefill.k_cache), read(*prefill.v_cache));
    auto continuation_views = continuation.views();
    DecoderLayerForwardState continuation_state;
    continuation_state.initialized_length = 2;
    CHECK_NOTHROW(iom::session_detail::run_decoder_layer_forward(
            *continuation.operations, continuation_views,
            continuation.params(), continuation.workspace(), {},
            continuation_state));
    CHECK_EQ(continuation_state.initialized_length, 3u);
    check_close(
            read(*continuation.next_x),
            row(read(*full.next_x), 2, kFeatures), 3, 5e-2F,
            "cached R=1 continuation versus full recomputation");
}

TEST_CASE(
        "TinyLlama decoder layer forward preserves causality and exact "
        "capacity") {
    Fixture base(2, 2, 0, 23);
    auto base_views = base.views();
    DecoderLayerForwardState base_state;
    CHECK_NOTHROW(iom::session_detail::run_decoder_layer_forward(
            *base.operations, base_views, base.params(), base.workspace(), {},
            base_state));
    CHECK_EQ(base_state.initialized_length, 2u);

    Fixture future(2, 2, 0, 23);
    std::vector<float> changed = row(future.input, 1, kFeatures);
    for (float& value : changed) value = round_bf16(value + 0.75F);
    future.replace_input(1, changed);
    auto future_views = future.views();
    DecoderLayerForwardState future_state;
    CHECK_NOTHROW(iom::session_detail::run_decoder_layer_forward(
            *future.operations, future_views, future.params(),
            future.workspace(), {}, future_state));
    const std::vector<float> base_output = read(*base.next_x);
    const std::vector<float> future_output = read(*future.next_x);
    CHECK(std::equal(base_output.begin(), base_output.begin() + kFeatures,
            future_output.begin()));
    CHECK(!std::equal(base_output.begin() + kFeatures, base_output.end(),
            future_output.begin() + kFeatures));

    // `a + R > C` is rejected by the composition before QKV can submit.
    Fixture beyond(2, 2, 0, 29);
    auto beyond_views = beyond.views();
    DecoderLayerForwardParams beyond_params = beyond.params();
    beyond_params.a = 1;
    beyond_views.attention.a = 1;
    DecoderLayerForwardState beyond_state;
    beyond_state.initialized_length = 1;
    CHECK_THROWS_AS(iom::session_detail::run_decoder_layer_forward(
                            *beyond.operations, beyond_views, beyond_params,
                            beyond.workspace(), {}, beyond_state),
            std::invalid_argument);
    CHECK_FALSE(beyond_state.failed);
    CHECK_EQ(beyond_state.initialized_length, 1u);
    require_untouched(*beyond.normalized, "rejected normalized output");
    require_untouched(*beyond.next_x, "rejected final output");
}

TEST_CASE(
        "TinyLlama decoder layer forward drains accepted append failure and "
        "does not submit dependent stages") {
    Fixture fixture(2, 2, 0, 31);
    const std::vector<float> initial_k = fixture.k_cache_values;
    const std::vector<float> initial_v = fixture.v_cache_values;
    auto views = fixture.views();
    DecoderLayerForwardState state;

    iom::cpu_detail::arm_cache_append_failure();
    iom::cpu_detail::arm_sdpa_failure();
    CHECK_THROWS_AS(iom::session_detail::run_decoder_layer_forward(
                            *fixture.operations, views, fixture.params(),
                            fixture.workspace(), {}, state),
            std::runtime_error);
    iom::cpu_detail::clear_cache_append_failure();
    CHECK(state.failed);
    CHECK_EQ(state.initialized_length, 0u);
    CHECK(read(*fixture.k_cache) == initial_k);
    CHECK(read(*fixture.v_cache) != initial_v);
    require_untouched(*fixture.merged, "failed merged attention");
    require_untouched(*fixture.next_x, "failed final residual");

    // The armed SDPA failure is still available only if the decoder submitted
    // no SDPA.  A direct probe consumes it and proves the append barrier.
    const DecoderLayerForwardWorkspace decoder_workspace =
            fixture.workspace();
    const oid probe = fixture.operations->sdpa(
            views.attention.rotated_q, views.attention.k_cache,
            views.attention.v_cache, views.attention.attention_merged, 0, 2,
            decoder_workspace.attention);
    REQUIRE(iom::oid_is_token(probe));
    CHECK_THROWS_AS(fixture.operations->wait(probe), std::runtime_error);
    iom::cpu_detail::clear_sdpa_failure();
}

}  // namespace decoder_layer_forward_test

namespace generation_cli_test {

#ifndef IOM_GENERATE_EXECUTABLE
#define IOM_GENERATE_EXECUTABLE ""
#endif

struct ProcessResult {
    int exit_code = -1;
    std::string stdout_text;
    std::string stderr_text;
};

[[nodiscard]] std::string read_capture(
        const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string(
            std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>());
}

[[nodiscard]] ProcessResult invoke(std::vector<std::string> arguments) {
    static std::size_t sequence = 0;
    const std::string tag = "iom-generate-cli-"
            + std::to_string(static_cast<long long>(::getpid()))
            + "-" + std::to_string(++sequence);
    const std::filesystem::path directory =
            std::filesystem::temp_directory_path();
    const std::filesystem::path stdout_path = directory / (tag + ".out");
    const std::filesystem::path stderr_path = directory / (tag + ".err");

    arguments.insert(arguments.begin(), IOM_GENERATE_EXECUTABLE);
    const pid_t child = ::fork();
    REQUIRE_MESSAGE(child >= 0, "fork failed: " << std::strerror(errno));
    if (child < 0) {
        return {};
    }

    if (child == 0) {
        const int stdout_fd = ::open(
                stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        const int stderr_fd = ::open(
                stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (stdout_fd < 0 || stderr_fd < 0
                || ::dup2(stdout_fd, STDOUT_FILENO) < 0
                || ::dup2(stderr_fd, STDERR_FILENO) < 0) {
            _exit(126);
        }
        ::close(stdout_fd);
        ::close(stderr_fd);

        std::vector<char*> child_arguments;
        child_arguments.reserve(arguments.size() + 1);
        for (std::string& argument : arguments) {
            child_arguments.push_back(argument.data());
        }
        child_arguments.push_back(nullptr);
        ::execv(child_arguments.front(), child_arguments.data());
        _exit(127);
    }

    int status = 0;
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }

    ProcessResult result;
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    }
    result.stdout_text = read_capture(stdout_path);
    result.stderr_text = read_capture(stderr_path);
    std::error_code error;
    std::filesystem::remove(stdout_path, error);
    std::filesystem::remove(stderr_path, error);
    return result;
}

[[nodiscard]] std::vector<std::string> required_cpu_arguments(
        const std::filesystem::path& model_directory) {
    return {
            "--model-dir", model_directory.string(),
            "--backend", "cpu",
            "--device", "0",
            "--max-new-tokens", "0",
    };
}

}  // namespace generation_cli_test

TEST_CASE("TinyLlama generation CLI rejects malformed process input") {
    using generation_cli_test::invoke;

    const auto missing = invoke({});
    CHECK_EQ(missing.exit_code, 2);
    CHECK(missing.stderr_text.find("usage/input") != std::string::npos);

    auto negative = invoke({
            "--model-dir", "/missing", "--backend", "cpu",
            "--device", "0", "--max-new-tokens", "-1",
            "--prompt", "hello"});
    CHECK_EQ(negative.exit_code, 2);
    CHECK(negative.stderr_text.find("usage/input") != std::string::npos);

    auto overflowing = invoke({
            "--model-dir", "/missing", "--backend", "cpu",
            "--device", "0", "--max-new-tokens",
            std::to_string(std::numeric_limits<std::size_t>::max()) + "0",
            "--prompt", "hello"});
    CHECK_EQ(overflowing.exit_code, 2);
    CHECK(overflowing.stderr_text.find("usage/input") != std::string::npos);

    auto wrong_cpu_ordinal = invoke({
            "--model-dir", "/missing", "--backend", "cpu",
            "--device", "1", "--max-new-tokens", "0",
            "--prompt", "hello"});
    CHECK_EQ(wrong_cpu_ordinal.exit_code, 2);
    CHECK(wrong_cpu_ordinal.stderr_text.find("usage/input")
          != std::string::npos);

    auto missing_arena = invoke({
            "--model-dir", "/missing", "--backend", "cuda",
            "--device", "0", "--max-new-tokens", "0",
            "--prompt", "hello"});
    CHECK_EQ(missing_arena.exit_code, 2);
    CHECK(missing_arena.stderr_text.find("usage/input")
          != std::string::npos);

    auto conflicting_input = invoke({
            "--model-dir", "/missing", "--backend", "cpu",
            "--device", "0", "--max-new-tokens", "0",
            "--prompt", "hello", "--message", "user", "world"});
    CHECK_EQ(conflicting_input.exit_code, 2);
    CHECK(conflicting_input.stderr_text.find("usage/input")
          != std::string::npos);
}

TEST_CASE("TinyLlama generation CLI classifies backend setup failures") {
    const auto result = generation_cli_test::invoke({
            "--model-dir", "/missing", "--backend", "cuda",
            "--device", "0", "--tensor-arena-bytes", "32",
            "--max-new-tokens", "0", "--prompt", "hello"});
    CHECK_EQ(result.exit_code, 3);
    CHECK(result.stdout_text.empty());
    CHECK(result.stderr_text.find("setup/load") != std::string::npos);
}

TEST_CASE(
        "TinyLlama generation CLI runs raw and structured chat process "
        "forms") {
    ForwardFixture fixture("generation-cli-process", text_generation_config());
    const std::vector<std::string> base =
            generation_cli_test::required_cpu_arguments(fixture.directory.path());

    std::vector<std::string> raw_arguments = base;
    raw_arguments.push_back("--prompt");
    raw_arguments.push_back("hello");
    const auto raw = generation_cli_test::invoke(std::move(raw_arguments));
    CHECK_EQ(raw.exit_code, 0);
    CHECK(raw.stdout_text.empty());
    CHECK(raw.stderr_text.empty());

    std::vector<std::string> chat_arguments = base;
    chat_arguments.insert(
            chat_arguments.end(),
            {"--message", "system", "Be concise.",
             "--message", "user", "hello"});
    const auto chat = generation_cli_test::invoke(std::move(chat_arguments));
    CHECK_EQ(chat.exit_code, 0);
    CHECK(chat.stdout_text.empty());
    CHECK(chat.stderr_text.empty());

    std::vector<std::string> unsupported_role = base;
    unsupported_role.insert(
            unsupported_role.end(), {"--message", "tool", "hello"});
    const auto rejected =
            generation_cli_test::invoke(std::move(unsupported_role));
    CHECK_EQ(rejected.exit_code, 2);
    CHECK(rejected.stdout_text.empty());
    CHECK(rejected.stderr_text.find("usage/input") != std::string::npos);
}
