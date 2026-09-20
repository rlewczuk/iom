#pragma once

#include "backend_conformance_other.hpp"
#include "model_loading_fixture.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "iom/token_selection.hpp"

namespace iom_conformance {

// A backend driver may expose an accepted transfer-failure seam.  The shared
// cases own the selector call and all assertions; a driver only arms and
// clears its existing native fault injection.
struct TokenSelectionNativeFailureSeam {
    std::function<void()> arm;
    std::function<void()> clear;
    std::string_view name;
    std::string_view unavailable_reason;

    [[nodiscard]] bool available() const noexcept {
        return static_cast<bool>(arm);
    }
};

struct TokenSelectionConformanceConfig {
    ConformanceDevices devices;
    std::span<const iom::DataType> supported_types;
    ConformanceObserver* observer = nullptr;
    AcceleratorStorageOracle* native_storage = nullptr;
    // CPU supplies {0, 1}; accelerator drivers can leave this unset and
    // assert their queried native staging requirement in their adapter.
    std::optional<iom::WorkspaceRequirements> expected_device_scratch;
    TokenSelectionNativeFailureSeam native_failure{};
};

namespace token_selection_detail {

constexpr std::byte kScratchSentinel{0xA5};
constexpr std::byte kPaddingLow{0x80};
constexpr std::byte kPaddingHigh{0x7F};

static_assert(std::is_same_v<
              decltype(&iom::TokenSelector::select),
              std::size_t (iom::TokenSelector::*)(
                      iom::DeviceOps&, const iom::TensorView&, std::size_t,
                      iom::oid, std::span<const std::size_t>,
                      iom::TokenSelectorScratch)>);

struct SelectionCase {
    std::string_view name;
    std::vector<std::uint16_t> values;
};

[[nodiscard]] inline std::vector<std::byte> encode_bf16(
        std::span<const std::uint16_t> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < values.size(); ++index) {
        bytes[2 * index] = static_cast<std::byte>(values[index] & 0xffu);
        bytes[2 * index + 1] =
                static_cast<std::byte>((values[index] >> 8) & 0xffu);
    }
    return bytes;
}

[[nodiscard]] inline float decode_bf16(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

// Independent oracle: it decodes the uploaded bit patterns and applies the
// public greedy policy, rather than sharing the production selector's scan.
[[nodiscard]] inline std::optional<std::size_t> expected_token(
        std::span<const std::uint16_t> values) {
    if (values.empty()) return std::nullopt;
    bool have_winner = false;
    float best = 0.0F;
    std::size_t winner = 0;
    for (std::size_t id = 0; id < values.size(); ++id) {
        const float value = decode_bf16(values[id]);
        if (!std::isfinite(value)) return std::nullopt;
        if (!have_winner || value > best) {
            have_winner = true;
            best = value;
            winner = id;
        }
    }
    return winner;
}

[[nodiscard]] inline std::vector<std::byte> poisoned_storage(
        const iom::TensorSpec& owner_spec, const iom::TensorView& view,
        std::span<const std::byte> logical) {
    std::vector<std::byte> storage(
            owner_spec.tiled_storage_nbytes(), std::byte{0});
    // Every physical BF16 slot outside the logical view is +inf.  The
    // independent tiled oracle then writes only logical slots, so a transfer
    // or selector that observes padding cannot accidentally pass.
    for (std::size_t offset = 0; offset + 1 < storage.size(); offset += 2) {
        storage[offset] = kPaddingLow;
        storage[offset + 1] = kPaddingHigh;
    }
    apply_standard_tiled_view(view, owner_spec, logical, storage);
    return storage;
}

[[nodiscard]] inline std::vector<std::byte> observe_storage(
        AcceleratorStorageOracle& native, const iom::Tensor& owner) {
    native.set_owner_spec(owner.view().spec());
    return native.observe(owner.view());
}

[[nodiscard]] inline iom::RawWorkspaceView scratch_view(
        const std::unique_ptr<iom::RawWorkspace>& owner) {
    return owner == nullptr ? iom::RawWorkspaceView{} : owner->view();
}

class DeviceLessQueue final : public iom::DeviceOps {
public:
    DeviceLessQueue() : iom::DeviceOps() {}
};

struct PreparedSelection {
    std::unique_ptr<iom::Tensor> source;
    std::unique_ptr<iom::Tensor> logits;
    std::unique_ptr<iom::DeviceOps> queue;
    std::unique_ptr<iom::RawWorkspace> device_scratch;
    iom::oid producer = 0;
    std::size_t vocabulary = 0;
    iom::TokenSelectorScratchRequirements requirements{};
    std::vector<std::byte> source_logical;
    std::vector<std::byte> host;
};

[[nodiscard]] inline std::vector<std::uint16_t> stale_values(
        std::size_t vocabulary) {
    std::vector<std::uint16_t> values(vocabulary, 0x3f80u);
    values[0] = 0x4000u;
    return values;
}

[[nodiscard]] inline PreparedSelection prepare_selection(
        iom::Device& candidate, AcceleratorStorageOracle& native,
        iom::GreedyTokenSelector& selector,
        std::span<const std::uint16_t> values) {
    PreparedSelection prepared;
    prepared.vocabulary = values.size();
    const iom::TensorSpec spec{
            iom::TensorShape{{1, prepared.vocabulary}}, iom::DataType::BF16};
    prepared.source = candidate.create_tensor(spec);
    prepared.logits = candidate.create_tensor(spec);
    REQUIRE(prepared.source != nullptr);
    REQUIRE(prepared.logits != nullptr);

    prepared.source_logical = encode_bf16(values);
    copy_from_host(prepared.source->view(), prepared.source_logical);

    const std::vector<std::uint16_t> stale = stale_values(prepared.vocabulary);
    const std::vector<std::byte> stale_logical = encode_bf16(stale);
    const std::vector<std::byte> stale_storage = poisoned_storage(
            spec, prepared.logits->view(), stale_logical);
    native.set_owner_spec(spec);
    native.seed(prepared.logits->view(), stale_storage);

    prepared.queue = candidate.create_ops();
    REQUIRE(prepared.queue != nullptr);
    prepared.producer = prepared.queue->copy(
            prepared.source->view(), prepared.logits->view());
    REQUIRE(iom::oid_is_token(prepared.producer));

    prepared.requirements = selector.scratch_requirements(
            prepared.logits->view(), prepared.vocabulary);
    REQUIRE_EQ(
            prepared.requirements.host_bytes,
            prepared.vocabulary * sizeof(std::uint16_t));
    if (prepared.requirements.device.bytes != 0) {
        prepared.device_scratch = candidate.create_workspace(
                prepared.requirements.device.bytes);
        REQUIRE(prepared.device_scratch != nullptr);
    }
    // Offset one deliberately makes the supplied host range unaligned.  The
    // selector contract has no host alignment requirement.
    prepared.host.assign(
            prepared.requirements.host_bytes + 9, kScratchSentinel);
    return prepared;
}

[[nodiscard]] inline PreparedSelection prepare_deferred_probe(
        iom::Device& candidate, AcceleratorStorageOracle& native,
        iom::GreedyTokenSelector& selector,
        std::span<const std::uint16_t> values) {
    PreparedSelection prepared;
    prepared.vocabulary = values.size();
    const iom::TensorSpec spec{
            iom::TensorShape{{1, prepared.vocabulary}}, iom::DataType::BF16};
    prepared.logits = candidate.create_tensor(spec);
    REQUIRE(prepared.logits != nullptr);
    const std::vector<std::byte> logical = encode_bf16(values);
    const std::vector<std::byte> storage = poisoned_storage(
            spec, prepared.logits->view(), logical);
    native.set_owner_spec(spec);
    native.seed(prepared.logits->view(), storage);

    auto deferred = std::make_unique<DeferredCopyQueue>(candidate);
    DeferredCopyQueue* deferred_ptr = deferred.get();
    prepared.producer = deferred_ptr->probe();
    REQUIRE(iom::oid_is_token(prepared.producer));
    prepared.queue = std::move(deferred);
    prepared.requirements = selector.scratch_requirements(
            prepared.logits->view(), prepared.vocabulary);
    REQUIRE_EQ(
            prepared.requirements.host_bytes,
            prepared.vocabulary * sizeof(std::uint16_t));
    if (prepared.requirements.device.bytes != 0) {
        prepared.device_scratch = candidate.create_workspace(
                prepared.requirements.device.bytes);
        REQUIRE(prepared.device_scratch != nullptr);
    }
    prepared.host.assign(
            prepared.requirements.host_bytes + 9, kScratchSentinel);
    prepared.source_logical = logical;
    return prepared;
}

[[nodiscard]] inline iom::TokenSelectorScratch scratch_for(
        PreparedSelection& prepared) {
    return iom::TokenSelectorScratch{
            std::span<std::byte>(
                    prepared.host.data() + 1, prepared.host.size() - 1),
            scratch_view(prepared.device_scratch)};
}

inline void check_host_scratch(
        const PreparedSelection& prepared,
        std::span<const std::byte> expected) {
    CHECK_EQ(prepared.host.front(), kScratchSentinel);
    CHECK_GE(prepared.host.size(), expected.size() + 1);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CHECK_EQ(prepared.host[index + 1], expected[index]);
    }
    for (std::size_t index = expected.size() + 1;
         index < prepared.host.size(); ++index) {
        CHECK_EQ(prepared.host[index], kScratchSentinel);
    }
}

inline void check_runtime_failure(std::exception_ptr failure) {
    REQUIRE(failure != nullptr);
    bool runtime = false;
    try {
        std::rethrow_exception(failure);
    } catch (const std::runtime_error&) {
        runtime = true;
    } catch (...) {
    }
    CHECK(runtime);
}

inline void finish_case(
        ConformanceObserver* observer) {
    if (observer != nullptr) observer->case_complete();
}

inline void begin_case(ConformanceObserver* observer) {
    if (observer != nullptr) observer->setup_complete();
}

inline void run_requirements_conformance(
        const TokenSelectionConformanceConfig& config,
        iom::GreedyTokenSelector& selector) {
    constexpr std::array<std::uint16_t, 17> values = {
            0x3f80u, 0x3f80u, 0x3f80u, 0x3f80u, 0x3f80u, 0x3f80u,
            0x3f80u, 0x3f80u, 0x4000u, 0x3f80u, 0x3f80u, 0x3f80u,
            0x3f80u, 0x3f80u, 0x3f80u, 0x3f80u, 0x3f80u};
    PreparedSelection prepared = prepare_selection(
            config.devices.candidate, *config.native_storage, selector,
            values);
    REQUIRE_NOTHROW(prepared.queue->wait(prepared.producer));
    const std::vector<std::byte> logits_before = observe_storage(
            *config.native_storage, *prepared.logits);

    DeferredCopyQueue queue(config.devices.candidate);
    const iom::oid probe = queue.probe();
    const std::uint64_t sequence = token_sequence(probe);
    CHECK_EQ(sequence, std::uint64_t{1});
    queue.complete(sequence);
    CHECK_NOTHROW(queue.wait(probe));
    const std::size_t queue_records_before = queue.records().size();

    begin_case(config.observer);
    const iom::TokenSelectorScratchRequirements first =
            selector.scratch_requirements(
                    prepared.logits->view(), prepared.vocabulary);
    const iom::TokenSelectorScratchRequirements second =
            selector.scratch_requirements(
                    prepared.logits->view(), prepared.vocabulary);
    finish_case(config.observer);

    CHECK_EQ(queue.records().size(), queue_records_before);

    CHECK_EQ(first.host_bytes, std::size_t{2} * prepared.vocabulary);
    CHECK_EQ(first.host_bytes, second.host_bytes);
    CHECK_EQ(first.device, second.device);
    if (config.expected_device_scratch.has_value()) {
        CHECK_EQ(first.device, *config.expected_device_scratch);
    }
    CHECK_EQ(
            observe_storage(*config.native_storage, *prepared.logits),
            logits_before);
}

inline void run_success_matrix(
        const TokenSelectionConformanceConfig& config,
        iom::GreedyTokenSelector& selector) {
    std::vector<SelectionCase> cases;
    cases.push_back({"V=1", {0x3f80u}});

    std::vector<std::uint16_t> first(17, 0xbf80u);
    first.front() = 0x7f7fu;
    cases.push_back({"unique maximum at first", std::move(first)});

    std::vector<std::uint16_t> interior(17, 0x3f80u);
    interior[8] = 0x4000u;
    cases.push_back({"unique maximum at interior", std::move(interior)});

    std::vector<std::uint16_t> last(16, 0x3f80u);
    last.back() = 0x4000u;
    cases.push_back({"unique maximum at last", std::move(last)});

    std::vector<std::uint16_t> all_negative(15, 0xbf80u);
    all_negative[7] = 0xc000u;
    cases.push_back({"all negative", std::move(all_negative)});

    std::vector<std::uint16_t> tied(17, 0x3f80u);
    tied[4] = 0x4000u;
    tied[11] = 0x4000u;
    cases.push_back({"equal maxima choose lowest ID", std::move(tied)});

    std::vector<std::uint16_t> signed_zero(17, 0x8000u);
    signed_zero[9] = 0x0000u;
    cases.push_back({"signed zero tie", std::move(signed_zero)});

    std::vector<std::uint16_t> finite_edges(17, 0xbf80u);
    finite_edges[0] = 0x0001u;
    finite_edges[1] = 0x8001u;
    finite_edges[2] = 0x7f7fu;
    finite_edges[3] = 0x7f7fu;
    finite_edges[4] = 0xff7fu;
    cases.push_back({"finite subnormal and extremes", std::move(finite_edges)});

    std::vector<std::uint16_t> practical(32000, 0xbf80u);
    practical[31999] = 0x4000u;
    cases.push_back({"practical vocabulary extent", std::move(practical)});

    for (const SelectionCase& test : cases) {
        CAPTURE(test.name);
        const std::optional<std::size_t> expected = expected_token(test.values);
        REQUIRE(expected.has_value());
        PreparedSelection prepared = prepare_selection(
                config.devices.candidate, *config.native_storage, selector,
                test.values);
        // Establish the post-producer physical image before arming the
        // allocator/traffic observation window.  The selector must not change
        // either logical logits or poisoned physical padding.
        REQUIRE_NOTHROW(prepared.queue->wait(prepared.producer));
        const std::vector<std::byte> logits_before = observe_storage(
                *config.native_storage, *prepared.logits);
        const iom::Tensor* owner = prepared.logits->view().owner_identity();
        const void* native_handle = prepared.logits->view().native_handle();
        const std::vector<std::size_t> history{11, 12, 13};
        const std::vector<std::size_t> history_before = history;

        std::size_t selected = std::numeric_limits<std::size_t>::max();
        std::exception_ptr failure;
        begin_case(config.observer);
        try {
            selected = selector.select(
                    *prepared.queue, prepared.logits->view(),
                    prepared.vocabulary, prepared.producer, history,
                    scratch_for(prepared));
        } catch (...) {
            failure = std::current_exception();
        }
        finish_case(config.observer);

        CHECK_FALSE(failure);
        CHECK_EQ(selected, *expected);
        check_host_scratch(prepared, prepared.source_logical);
        CHECK(history == history_before);
        CHECK_EQ(prepared.logits->view().owner_identity(), owner);
        CHECK_EQ(prepared.logits->view().native_handle(), native_handle);
        CHECK(
                read_logical(prepared.logits->view())
                == prepared.source_logical);
        CHECK_EQ(
                observe_storage(*config.native_storage, *prepared.logits),
                logits_before);
    }
}

inline void run_nonfinite_matrix(
        const TokenSelectionConformanceConfig& config,
        iom::GreedyTokenSelector& selector) {
    constexpr std::array<std::uint16_t, 3> nonfinite = {
            0x7f80u, 0xff80u, 0x7fc1u};
    for (const std::uint16_t bits : nonfinite) {
        CAPTURE(bits);
        std::vector<std::uint16_t> values(15, 0x3f80u);
        values[0] = 0x4000u;
        values.back() = bits;
        PreparedSelection prepared = prepare_selection(
                config.devices.candidate, *config.native_storage, selector,
                values);
        REQUIRE_NOTHROW(prepared.queue->wait(prepared.producer));
        const std::vector<std::byte> logits_before = observe_storage(
                *config.native_storage, *prepared.logits);
        const std::vector<std::byte> host_before = prepared.host;
        std::exception_ptr failure;
        begin_case(config.observer);
        try {
            static_cast<void>(selector.select(
                    *prepared.queue, prepared.logits->view(),
                    prepared.vocabulary, prepared.producer, {},
                    scratch_for(prepared)));
        } catch (...) {
            failure = std::current_exception();
        }
        finish_case(config.observer);
        check_runtime_failure(failure);
        CHECK(prepared.host != host_before);
        CHECK_EQ(
                observe_storage(*config.native_storage, *prepared.logits),
                logits_before);
        check_host_scratch(prepared, prepared.source_logical);
    }
}

inline void run_validation_conformance(
        const TokenSelectionConformanceConfig& config,
        iom::GreedyTokenSelector& selector) {
    constexpr std::size_t vocabulary = 17;
    const std::vector<std::uint16_t> values(vocabulary, 0x3f80u);
    const iom::TensorSpec spec{
            iom::TensorShape{{1, vocabulary}}, iom::DataType::BF16};
    auto logits = config.devices.candidate.create_tensor(spec);
    REQUIRE(logits != nullptr);
    const std::vector<std::byte> logical = encode_bf16(values);
    const std::vector<std::byte> storage = poisoned_storage(
            spec, logits->view(), logical);
    config.native_storage->set_owner_spec(spec);
    config.native_storage->seed(logits->view(), storage);

    DeferredCopyQueue queue(config.devices.candidate);
    const iom::oid producer = queue.probe();
    REQUIRE(iom::oid_is_token(producer));
    const auto requirements =
            selector.scratch_requirements(logits->view(), vocabulary);
    std::vector<std::byte> host(requirements.host_bytes + 9, kScratchSentinel);
    std::unique_ptr<iom::RawWorkspace> exact_workspace;
    if (requirements.device.bytes != 0) {
        exact_workspace = config.devices.candidate.create_workspace(
                requirements.device.bytes);
        REQUIRE(exact_workspace != nullptr);
    }
    const std::vector<std::byte> host_before = host;
    const std::vector<std::byte> logits_before = observe_storage(
            *config.native_storage, *logits);
    const std::vector<std::size_t> history{5, 6};
    const std::vector<std::size_t> history_before = history;

    const auto scratch = [&](iom::RawWorkspaceView device) {
        return iom::TokenSelectorScratch{
                std::span<std::byte>(host.data() + 1, host.size() - 1),
                device};
    };
    const auto reject = [&](iom::DeviceOps& target, const iom::TensorView& view,
                            std::size_t extent, iom::oid token,
                            iom::TokenSelectorScratch supplied) {
        bool rejected = false;
        try {
            static_cast<void>(selector.select(
                    target, view, extent, token, history, supplied));
        } catch (const std::invalid_argument&) {
            rejected = true;
        } catch (...) {
        }
        CHECK(rejected);
        CHECK(host == host_before);
        CHECK(history == history_before);
    };

    auto wrong_dtype = config.devices.candidate.create_tensor(
            iom::TensorSpec{iom::TensorShape{{1, vocabulary}},
                            iom::DataType::F32});
    auto wrong_rank = config.devices.candidate.create_tensor(
            iom::TensorSpec{iom::TensorShape{{1, 1, vocabulary}},
                            iom::DataType::BF16});
    auto wrong_extent = config.devices.candidate.create_tensor(
            iom::TensorSpec{iom::TensorShape{{1, vocabulary + 1}},
                            iom::DataType::BF16});
    auto quantized = config.devices.candidate.create_tensor(spec);
    REQUIRE(wrong_dtype != nullptr);
    REQUIRE(wrong_rank != nullptr);
    REQUIRE(wrong_extent != nullptr);
    REQUIRE(quantized != nullptr);
    auto& quantized_spec = const_cast<iom::TensorSpec&>(
            quantized->view().spec());
    quantized_spec.quantization = iom::QuantizationFormat::INT8_SYMMETRIC;

    auto foreign_logits = config.devices.foreign.create_tensor(spec);
    REQUIRE(foreign_logits != nullptr);

    DeferredCopyQueue foreign_queue(config.devices.candidate);
    const iom::oid foreign_token = foreign_queue.probe();
    REQUIRE(iom::oid_is_token(foreign_token));

    DeferredCopyQueue skipped_queue(config.devices.candidate);
    const iom::oid first = skipped_queue.probe();
    skipped_queue.complete(token_sequence(first));
    skipped_queue.seek_next_sequence(4);
    const iom::oid later = skipped_queue.probe();
    const std::size_t candidate_records_before = queue.records().size();
    const std::size_t foreign_records_before = foreign_queue.records().size();
    const std::size_t skipped_records_before = skipped_queue.records().size();

    std::unique_ptr<iom::RawWorkspace> foreign_workspace;
    std::unique_ptr<iom::RawWorkspace> undersized_workspace;
    std::unique_ptr<iom_model_loading::BoundedWorkspace>
            overlapping_workspace;
    std::unique_ptr<iom_model_loading::BoundedWorkspace>
            misaligned_workspace;
    std::optional<iom::RawWorkspaceView> stale_workspace;
    if (requirements.device.bytes != 0) {
        foreign_workspace = config.devices.foreign.create_workspace(
                requirements.device.bytes);
        REQUIRE(foreign_workspace != nullptr);
        if (requirements.device.bytes > 1) {
            undersized_workspace = config.devices.candidate.create_workspace(
                    requirements.device.bytes - 1);
            REQUIRE(undersized_workspace != nullptr);
        }
        overlapping_workspace =
                std::make_unique<iom_model_loading::BoundedWorkspace>(
                        config.devices.candidate, requirements.device.bytes,
                        logits->view().native_handle());
        misaligned_workspace =
                std::make_unique<iom_model_loading::BoundedWorkspace>(
                        config.devices.candidate, requirements.device.bytes,
                        reinterpret_cast<void*>(0x1001));
        auto stale_owner = config.devices.candidate.create_workspace(
                requirements.device.bytes);
        REQUIRE(stale_owner != nullptr);
        const auto stale_view = stale_owner->view();
        stale_workspace.emplace(stale_view);
        stale_owner.reset();
    }

    begin_case(config.observer);
    reject(queue, wrong_dtype->view(), vocabulary, producer,
           scratch(scratch_view(exact_workspace)));
    reject(queue, wrong_rank->view(), vocabulary, producer,
           scratch(scratch_view(exact_workspace)));
    reject(queue, wrong_extent->view(), vocabulary, producer,
           scratch(scratch_view(exact_workspace)));
    reject(queue, quantized->view(), vocabulary, producer,
           scratch(scratch_view(exact_workspace)));
    reject(queue, logits->view(), 0, producer,
           scratch(scratch_view(exact_workspace)));
    reject(queue, logits->view(), vocabulary - 1, producer,
           scratch(scratch_view(exact_workspace)));
    reject(queue, logits->view(), vocabulary, 0,
           scratch(scratch_view(exact_workspace)));
    reject(queue, logits->view(), vocabulary, -1,
           scratch(scratch_view(exact_workspace)));
    reject(queue, logits->view(), vocabulary, producer + 1,
           scratch(scratch_view(exact_workspace)));
    reject(queue, logits->view(), vocabulary,
           producer + 100, scratch(scratch_view(exact_workspace)));
    if (requirements.host_bytes > 0) {
        reject(
                queue, logits->view(), vocabulary, producer,
                iom::TokenSelectorScratch{
                        std::span<std::byte>(
                                host.data() + 1, requirements.host_bytes - 1),
                        scratch_view(exact_workspace)});
    }

    reject(queue, foreign_logits->view(), vocabulary, producer,
           scratch(scratch_view(exact_workspace)));
    reject(queue, logits->view(), vocabulary, foreign_token,
           scratch(scratch_view(exact_workspace)));
    reject(skipped_queue, logits->view(), vocabulary, later - 1,
           scratch(scratch_view(exact_workspace)));

    auto* overlapping_host = static_cast<std::byte*>(
            logits->view().native_handle());
    reject(
            queue, logits->view(), vocabulary, producer,
            iom::TokenSelectorScratch{
                    std::span<std::byte>(overlapping_host,
                                         requirements.host_bytes),
                    scratch_view(exact_workspace)});

    // Positive staging backends additionally prove exact-device, size,
    // alignment, overlap, and borrowed-owner validation.  CPU's requirement
    // is exactly zero and the default empty view is its complete contract.
    if (requirements.device.bytes != 0) {
        reject(
                queue, logits->view(), vocabulary, producer,
                scratch(iom::RawWorkspaceView{}));
        reject(queue, logits->view(), vocabulary, producer,
               scratch(foreign_workspace->view()));
        if (undersized_workspace != nullptr) {
            reject(queue, logits->view(), vocabulary, producer,
                   scratch(undersized_workspace->view()));
        }
        reject(queue, logits->view(), vocabulary, producer,
               scratch(overlapping_workspace->view()));
        reject(queue, logits->view(), vocabulary, producer,
               scratch(misaligned_workspace->view()));
        reject(queue, logits->view(), vocabulary, producer,
               scratch(*stale_workspace));
    }

    DeviceLessQueue no_device;
    reject(no_device, logits->view(), vocabulary, producer,
           scratch(scratch_view(exact_workspace)));
    CHECK_EQ(queue.records().size(), candidate_records_before);
    CHECK_EQ(foreign_queue.records().size(), foreign_records_before);
    CHECK_EQ(skipped_queue.records().size(), skipped_records_before);
    finish_case(config.observer);

    quantized_spec.quantization = iom::QuantizationFormat::NONE;
    CHECK_EQ(host, host_before);
    CHECK_EQ(
            observe_storage(*config.native_storage, *logits), logits_before);
    const iom::oid next = queue.probe();
    CHECK_EQ(token_sequence(next), token_sequence(producer) + 1);
    queue.complete(token_sequence(producer));
    queue.complete(token_sequence(next));
    CHECK_NOTHROW(queue.wait(producer));
    CHECK_NOTHROW(queue.wait(next));
    foreign_queue.complete(token_sequence(foreign_token));
    skipped_queue.complete(token_sequence(later));
}

inline void run_readiness_conformance(
        const TokenSelectionConformanceConfig& config,
        iom::GreedyTokenSelector& selector) {
    std::vector<std::uint16_t> values(17, 0xbf80u);
    values[12] = 0x4000u;
    PreparedSelection prepared = prepare_selection(
            config.devices.candidate, *config.native_storage, selector, values);
    const std::vector<std::size_t> history{31, 32, 33};
    const std::vector<std::size_t> history_before = history;
    begin_case(config.observer);
    std::size_t selected = std::numeric_limits<std::size_t>::max();
    std::exception_ptr failure;
    try {
        selected = selector.select(
                *prepared.queue, prepared.logits->view(),
                prepared.vocabulary, prepared.producer, history,
                scratch_for(prepared));
    } catch (...) {
        failure = std::current_exception();
    }
    finish_case(config.observer);
    CHECK_FALSE(failure);
    CHECK_EQ(selected, std::size_t{12});
    CHECK(history == history_before);
    check_host_scratch(prepared, prepared.source_logical);
    CHECK(
            read_logical(prepared.logits->view())
            == prepared.source_logical);
}

inline void run_repeated_failure_conformance(
        const TokenSelectionConformanceConfig& config,
        iom::GreedyTokenSelector& selector) {
    const std::vector<std::uint16_t> values(3, 0x3f80u);
    PreparedSelection prepared = prepare_deferred_probe(
            config.devices.candidate, *config.native_storage, selector, values);
    auto* deferred = dynamic_cast<DeferredCopyQueue*>(prepared.queue.get());
    REQUIRE(deferred != nullptr);
    const std::string_view expected_message = "retained producer failure";
    deferred->complete(
            token_sequence(prepared.producer),
            std::make_exception_ptr(
                    std::runtime_error(std::string(expected_message))));
    const std::vector<std::byte> host_before = prepared.host;
    const std::vector<std::byte> logits_before = observe_storage(
            *config.native_storage, *prepared.logits);
    begin_case(config.observer);
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::exception_ptr failure;
        try {
            static_cast<void>(selector.select(
                    *prepared.queue, prepared.logits->view(),
                    prepared.vocabulary, prepared.producer, {},
                    scratch_for(prepared)));
        } catch (...) {
            failure = std::current_exception();
        }
        check_runtime_failure(failure);
        try {
            std::rethrow_exception(failure);
        } catch (const std::runtime_error& error) {
            CHECK_EQ(std::string_view(error.what()), expected_message);
        } catch (...) {
            CHECK(false);
        }
        CHECK(prepared.host == host_before);
    }
    finish_case(config.observer);
    CHECK_EQ(
            observe_storage(*config.native_storage, *prepared.logits),
            logits_before);
    CHECK_THROWS_WITH(
            prepared.queue->wait(prepared.producer),
            "retained producer failure");
    CHECK_THROWS_WITH(
            prepared.queue->wait(prepared.producer),
            "retained producer failure");
}

inline void run_native_failure_conformance(
        const TokenSelectionConformanceConfig& config,
        iom::GreedyTokenSelector& selector) {
    if (!config.native_failure.available()) {
        // A backend whose testing seams cannot reach its configured transfer
        // path is a recorded coverage gap, never a silent pass: the driver
        // must state why, and the gap is printed with the exact missing seam.
        REQUIRE_MESSAGE(
                !config.native_failure.name.empty(),
                "an unavailable selector failure seam needs a stable adapter "
                "name");
        REQUIRE_MESSAGE(
                !config.native_failure.unavailable_reason.empty(),
                "an unavailable selector failure seam needs an explicit reason");
        std::printf(
                "token-selection-native-failure-record seam=%.*s "
                "coverage=unavailable reason=%.*s\n",
                static_cast<int>(config.native_failure.name.size()),
                config.native_failure.name.data(),
                static_cast<int>(
                        config.native_failure.unavailable_reason.size()),
                config.native_failure.unavailable_reason.data());
        return;
    }
    CHECK_MESSAGE(
            !config.native_failure.name.empty(),
            "a selector failure seam needs a stable adapter name");
    const std::array<std::uint16_t, 1> values = {0x3f80u};
    PreparedSelection prepared = prepare_selection(
            config.devices.candidate, *config.native_storage, selector, values);
    REQUIRE_NOTHROW(prepared.queue->wait(prepared.producer));
    const std::vector<std::byte> host_before = prepared.host;

    config.native_failure.arm();
    begin_case(config.observer);
    std::exception_ptr failure;
    try {
        static_cast<void>(selector.select(
                *prepared.queue, prepared.logits->view(),
                prepared.vocabulary, prepared.producer, {},
                scratch_for(prepared)));
    } catch (...) {
        failure = std::current_exception();
    }
    finish_case(config.observer);
    config.native_failure.clear();
    check_runtime_failure(failure);
    CHECK(prepared.host == host_before);
    CHECK_NOTHROW(prepared.queue->wait(prepared.producer));
}

inline void run_overflow_conformance() {
    iom_model_loading::FakeDevice bounded;
    const std::size_t huge = std::numeric_limits<std::size_t>::max();
    auto overflowing = bounded.create_tensor(iom::TensorSpec{
            iom::TensorShape{{1, huge}}, iom::DataType::BF16});
    REQUIRE(overflowing != nullptr);
    iom::GreedyTokenSelector selector;
    CHECK_THROWS_AS(
            static_cast<void>(
                    selector.scratch_requirements(overflowing->view(), huge)),
            std::overflow_error);
}

}  // namespace token_selection_detail

inline void run_token_selection_conformance(
        const TokenSelectionConformanceConfig& config) {
    REQUIRE(config.native_storage != nullptr);
    REQUIRE_FALSE(config.supported_types.empty());
    CHECK(
            std::find(
                    config.supported_types.begin(), config.supported_types.end(),
                    iom::DataType::BF16)
            != config.supported_types.end());

    iom::GreedyTokenSelector selector;
    token_selection_detail::run_requirements_conformance(config, selector);
    token_selection_detail::run_success_matrix(config, selector);
    token_selection_detail::run_nonfinite_matrix(config, selector);
    token_selection_detail::run_validation_conformance(config, selector);
    token_selection_detail::run_readiness_conformance(config, selector);
    token_selection_detail::run_repeated_failure_conformance(config, selector);
    token_selection_detail::run_native_failure_conformance(config, selector);
    token_selection_detail::run_overflow_conformance();
}

}  // namespace iom_conformance
