#pragma once

// Backend-neutral end-to-end conformance for the caller-owned inference
// recorder.  The driver supplies only its already-created candidate Device;
// this header owns the synthetic checkpoint, selector, mode join, and
// observation assertions.  No backend kind, runtime header, adapter, or host
// fallback is introduced here.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "backend/backend_conformance_common.hpp"
#include "inference_metrics_fixture.hpp"
#include "iom/inference_metrics.hpp"
#include "iom/session.hpp"
#include "../../src/session_internal.hpp"

namespace iom_conformance {
namespace inference_metrics_detail {
using iom_inference_metrics_test::SequenceSelector;
using iom_inference_metrics_test::SyntheticFixture;
using iom_inference_metrics_test::h16_config;

// The five branches deliberately exercise independent request shapes.  The
// failure plan is an additional error/poison join performed from the multi-step
// branch; it does not replace the normal successful decode branch.
struct ProbePlan {
    std::string tag;
    nlohmann::json config;
    std::vector<std::size_t> first_prompt;
    std::vector<std::size_t> first_sequence;
    std::size_t first_limit = 0;
    bool expect_logits = false;
    bool trace_rows_expected = false;
    bool first_raw = false;
    std::string first_text;
    bool reuse = false;
    std::vector<std::size_t> second_prompt;
    std::vector<std::size_t> second_sequence;
    std::size_t second_limit = 0;
};

enum class Mode { disabled, metrics_only, trace_enabled };

struct ModeObservation {
    bool threw = false;
    std::string failure_message;
    bool poisoned = false;
    bool trace_prepared = false;
    std::uint64_t trace_rows_dropped = 0;
    std::vector<std::vector<std::size_t>> generated_ids;
    std::vector<iom::GenerationStopReason> stop_reasons;
    std::vector<std::string> decoded_text;
    std::vector<std::vector<float>> logits;
    std::vector<std::vector<std::size_t>> cache_lengths;
    std::size_t request_length = 0;
    std::vector<iom::oid> selector_producers;
    std::vector<std::vector<std::size_t>> selector_histories;
    std::vector<iom::InferenceTraceRecord> operations;
    std::optional<iom::InferenceSnapshot> snapshot;
};

[[nodiscard]] inline std::string exception_message(
        const std::exception_ptr& failure) {
    if (failure == nullptr) return {};
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return "<non-standard>";
    }
}

[[nodiscard]] inline std::vector<float> read_bf16_logits(
        const iom::TensorView& view) {
    REQUIRE(view.spec().data_type == iom::DataType::BF16);
    REQUIRE(view.spec().shape.rank() == 2);
    REQUIRE_EQ(view.spec().shape.dimensions()[0], std::size_t{1});
    const std::size_t elements = view.spec().shape.element_count();
    std::vector<std::byte> bytes(view.spec().logical_nbytes());
    copy_to_host(view, bytes);
    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes.data() + index * sizeof(bits), sizeof(bits));
        values[index] = iom_inference_metrics_test::forward_decode_bf16(bits);
    }
    return values;
}

[[nodiscard]] inline std::string decode_ids(
        const iom::TinyLlamaSession& session,
        std::span<const std::size_t> ids) {
    std::vector<std::uint32_t> encoded;
    encoded.reserve(ids.size());
    for (const std::size_t id : ids) {
        encoded.push_back(static_cast<std::uint32_t>(id));
    }
    return session.tokenizer().decode(encoded, iom::DecodeOptions{});
}

inline void capture_cache_lengths(
        ModeObservation& observed, iom::TinyLlamaSession& session) {
    std::vector<std::size_t> lengths;
    for (const iom::session_detail::CacheOwner& cache
         : iom::session_detail::SessionAccess::caches(session)) {
        lengths.push_back(cache.initialized_length);
    }
    observed.cache_lengths.push_back(std::move(lengths));
    observed.request_length = session.request_length();
}

inline void configure_selector(
        SequenceSelector& selector, const ProbePlan& plan,
        bool failure_after_prefill) {
    if (!failure_after_prefill) return;
    selector.before_select = [&selector](std::size_t calls) {
        // The first selector call consumes the prefill result.  The second
        // call is reached only after a real decode forward and its readiness
        // wait, then retains the selector's original exception category.
        selector.throw_failure = calls >= 1;
    };
}

[[nodiscard]] inline ModeObservation run_mode(
        iom::Device& device, const std::filesystem::path& directory,
        const ProbePlan& plan, Mode mode, bool failure_after_prefill = false) {
    ModeObservation observed;
    std::unique_ptr<iom::InferenceMetrics> recorder;
    if (mode != Mode::disabled) {
        recorder = std::make_unique<iom::InferenceMetrics>();
    }

    auto selector = std::make_unique<SequenceSelector>(plan.first_sequence);
    SequenceSelector* const selector_state = selector.get();
    configure_selector(*selector_state, plan, failure_after_prefill);
    auto session = iom::load_tinyllama_session(
            directory, device, std::move(selector), recorder.get());
    REQUIRE(session != nullptr);
    if (mode == Mode::disabled) {
        CHECK(iom::session_detail::SessionAccess::metrics(*session) == nullptr);
    } else {
        REQUIRE(iom::session_detail::SessionAccess::metrics(*session)
                == recorder.get());
        if (mode == Mode::trace_enabled) {
            session->prepare_operation_trace();
            observed.trace_prepared = true;
        }
    }

    const auto record_generation = [&](std::span<const std::size_t> ids,
                                       iom::GenerationStopReason stop_reason,
                                       std::string text, bool has_forward) {
        observed.generated_ids.emplace_back(ids.begin(), ids.end());
        observed.stop_reasons.push_back(stop_reason);
        observed.decoded_text.push_back(std::move(text));
        if (has_forward) {
            observed.logits.push_back(
                    read_bf16_logits(
                            iom::session_detail::SessionAccess::logits(
                                    *session)));
        }
    };
    bool first_request = true;
    const auto run_one = [&](std::span<const std::size_t> prompt,
                             std::size_t limit) {
        try {
            if (first_request && plan.first_raw) {
                const iom::GenerationResult result =
                        session->generate_raw(plan.first_text, limit);
                record_generation(result.token_ids, result.stop_reason,
                                  result.text, limit != 0);
            } else {
                const iom::TokenGenerationResult result =
                        session->generate_tokens(prompt, limit);
                record_generation(result.token_ids, result.stop_reason,
                                  decode_ids(*session, result.token_ids),
                                  limit != 0);
            }
        } catch (...) {
            observed.threw = true;
            observed.failure_message = exception_message(
                    std::current_exception());
        }
        first_request = false;
        if (session->poisoned()) {
            // A poisoned request intentionally rejects private resource
            // accessors.  Preserve its public request length and let the
            // recorder retain the failure; do not turn the expected selector
            // exception into a secondary `require_request` logic_error.
            observed.request_length = session->request_length();
        } else {
            capture_cache_lengths(observed, *session);
        }
    };

    run_one(plan.first_prompt, plan.first_limit);
    if (!observed.threw && plan.reuse) {
        // The same session and selector are deliberately reused.  Successful
        // request publication must clear the old rows and reset KV prefixes;
        // no new session or backend queue is substituted for this branch.
        run_one(plan.second_prompt, plan.second_limit);
    }

    observed.poisoned = session->poisoned();
    observed.selector_producers = selector_state->producers;
    observed.selector_histories = selector_state->histories;
    // Destruction performs the session's existing queue drain.  Capture the
    // final trace state after that drain so rows whose first wait is owned by
    // cleanup retain their real completion state without adding any wait.
    session.reset();
    if (recorder) {
        observed.snapshot = recorder->snapshot();
        observed.trace_rows_dropped =
                observed.snapshot->trace_rows_dropped;
        const std::span<const iom::InferenceTraceRecord> rows =
                recorder->operations();
        observed.operations.assign(rows.begin(), rows.end());
    }
    return observed;
}

inline void compare_logits(
        std::span<const float> expected, std::span<const float> actual,
        std::string_view label) {
    REQUIRE_EQ(expected.size(), actual.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float lhs = expected[index];
        const float rhs = actual[index];
        const bool within_tolerance =
                (std::isfinite(lhs) && std::isfinite(rhs))
                ? std::abs(lhs - rhs)
                                <= 0.02F + 0.02F * std::abs(lhs)
                : lhs == rhs;
        CHECK_MESSAGE(
                within_tolerance,
                label << " logits differ at vocabulary index " << index);
    }
}

inline void compare_modes(
        const ProbePlan& plan, const ModeObservation& disabled,
        const ModeObservation& metrics_only,
        const ModeObservation& trace_enabled) {
    CHECK_EQ(disabled.threw, metrics_only.threw);
    CHECK_EQ(disabled.threw, trace_enabled.threw);
    CHECK_EQ(disabled.failure_message, metrics_only.failure_message);
    CHECK_EQ(disabled.failure_message, trace_enabled.failure_message);
    CHECK_EQ(disabled.poisoned, metrics_only.poisoned);
    CHECK_EQ(disabled.poisoned, trace_enabled.poisoned);
    CHECK(disabled.selector_histories == metrics_only.selector_histories);
    CHECK(disabled.selector_histories == trace_enabled.selector_histories);
    REQUIRE_EQ(disabled.selector_producers.size(),
               metrics_only.selector_producers.size());
    REQUIRE_EQ(disabled.selector_producers.size(),
               trace_enabled.selector_producers.size());
    for (const iom::oid producer : disabled.selector_producers) {
        CHECK(iom::oid_is_token(producer));
    }
    for (const iom::oid producer : metrics_only.selector_producers) {
        CHECK(iom::oid_is_token(producer));
    }
    for (const iom::oid producer : trace_enabled.selector_producers) {
        CHECK(iom::oid_is_token(producer));
    }

    CHECK(disabled.generated_ids == metrics_only.generated_ids);
    CHECK(disabled.generated_ids == trace_enabled.generated_ids);
    CHECK(disabled.stop_reasons == metrics_only.stop_reasons);
    CHECK(disabled.stop_reasons == trace_enabled.stop_reasons);
    CHECK(disabled.decoded_text == metrics_only.decoded_text);
    CHECK(disabled.decoded_text == trace_enabled.decoded_text);
    CHECK(disabled.cache_lengths == metrics_only.cache_lengths);
    CHECK(disabled.cache_lengths == trace_enabled.cache_lengths);
    CHECK_EQ(disabled.request_length, metrics_only.request_length);
    CHECK_EQ(disabled.request_length, trace_enabled.request_length);

    if (plan.expect_logits) {
        REQUIRE_EQ(disabled.logits.size(), metrics_only.logits.size());
        REQUIRE_EQ(disabled.logits.size(), trace_enabled.logits.size());
        for (std::size_t generation = 0;
             generation < disabled.logits.size(); ++generation) {
            compare_logits(disabled.logits[generation],
                           metrics_only.logits[generation],
                           "metrics-only");
            compare_logits(disabled.logits[generation],
                           trace_enabled.logits[generation],
                           "trace-enabled");
        }
    }

    REQUIRE(metrics_only.snapshot.has_value());
    REQUIRE(trace_enabled.snapshot.has_value());
    const iom::InferenceSnapshot& scalar = *metrics_only.snapshot;
    const iom::InferenceSnapshot& traced = *trace_enabled.snapshot;
    CHECK(scalar.load.state == iom::ObservationState::succeeded);
    CHECK(traced.load.state == iom::ObservationState::succeeded);
    CHECK(scalar.request_admitted == traced.request_admitted);
    CHECK(scalar.attempt.outcome == traced.attempt.outcome);
    CHECK_EQ(scalar.admitted.ordinal, traced.admitted.ordinal);
    CHECK_EQ(scalar.admitted.prompt_tokens, traced.admitted.prompt_tokens);
    CHECK_EQ(scalar.admitted.generated_tokens,
             traced.admitted.generated_tokens);
    CHECK_EQ(scalar.admitted.decode_forward_count,
             traced.admitted.decode_forward_count);
    CHECK_EQ(scalar.admitted.decode_token_count,
             traced.admitted.decode_token_count);
    CHECK(scalar.admitted.stop_reason_valid == traced.admitted.stop_reason_valid);
    if (scalar.admitted.stop_reason_valid) {
        CHECK(scalar.admitted.stop_reason == traced.admitted.stop_reason);
    }
    const bool scalar_failed = scalar.attempt.failure != nullptr;
    const bool traced_failed = traced.attempt.failure != nullptr;
    CHECK_EQ(scalar_failed, traced_failed);
    CHECK_EQ(scalar.admitted.tokenization.state,
             traced.admitted.tokenization.state);
    CHECK_EQ(scalar.admitted.prefill.state, traced.admitted.prefill.state);
    CHECK_EQ(scalar.admitted.decode.state, traced.admitted.decode.state);
    CHECK_EQ(scalar.admitted.time_to_first_token_valid,
             traced.admitted.time_to_first_token_valid);
    CHECK_EQ(scalar.admitted.decode_throughput_valid,
             traced.admitted.decode_throughput_valid);
    CHECK_EQ(metrics_only.operations.size(), std::size_t{0});
    CHECK_FALSE(metrics_only.trace_prepared);
    CHECK(trace_enabled.trace_prepared);
    CHECK_EQ(trace_enabled.trace_rows_dropped, std::uint64_t{0});
}

inline void check_snapshot_expectation(
        const ProbePlan& plan, const ModeObservation& observed,
        bool failure_after_prefill = false) {
    REQUIRE(observed.snapshot.has_value());
    const iom::InferenceSnapshot& snapshot = *observed.snapshot;
    REQUIRE(snapshot.request_admitted);
    REQUIRE(snapshot.admitted.ordinal != 0);
    CHECK_EQ(snapshot.attempt.ordinal, snapshot.admitted.ordinal);
    const std::vector<std::size_t>& prompt =
            plan.reuse ? plan.second_prompt : plan.first_prompt;
    CHECK_EQ(snapshot.admitted.prompt_tokens, prompt.size());
    if (plan.first_raw) {
        CHECK(snapshot.admitted.tokenization.state
              == iom::ObservationState::succeeded);
    } else {
        CHECK(snapshot.admitted.tokenization.state
              == iom::ObservationState::not_run);
    }
    if (failure_after_prefill) {
        CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::failed);
        CHECK(snapshot.attempt.failure != nullptr);
        CHECK(snapshot.admitted.failure != nullptr);
        CHECK_EQ(snapshot.admitted.generated_tokens, std::size_t{1});
        CHECK_EQ(snapshot.admitted.decode_forward_count, std::size_t{1});
        CHECK_EQ(snapshot.admitted.decode_token_count, std::size_t{0});
        CHECK_FALSE(snapshot.admitted.stop_reason_valid);
        CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
        return;
    }
    CAPTURE(plan.tag);
    CAPTURE(observed.threw);
    CAPTURE(observed.failure_message);
    CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::succeeded);
    REQUIRE_EQ(observed.generated_ids.size(), plan.reuse ? 2U : 1U);
    CHECK_EQ(snapshot.admitted.generated_tokens,
             observed.generated_ids.back().size());
    CHECK(snapshot.admitted.stop_reason_valid);
    CHECK(snapshot.admitted.stop_reason == observed.stop_reasons.back());
    if (plan.first_limit == 0 && !plan.reuse) {
        CHECK(snapshot.admitted.decode_forward_count == 0);
        CHECK(snapshot.admitted.decode_token_count == 0);
        CHECK(snapshot.admitted.decode.state == iom::ObservationState::not_run);
        CHECK_FALSE(snapshot.admitted.time_to_first_token_valid);
        CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
    }
    if (!plan.reuse && plan.config.at("max_position_embeddings")
                                   .get<std::size_t>() == prompt.size()) {
        CHECK(snapshot.admitted.stop_reason
              == iom::GenerationStopReason::context_capacity);
        CHECK_EQ(snapshot.admitted.generated_tokens, std::size_t{0});
        CHECK_EQ(snapshot.admitted.decode_forward_count, std::size_t{0});
        CHECK_EQ(snapshot.admitted.decode_token_count, std::size_t{0});
        CHECK(snapshot.admitted.prefill.state == iom::ObservationState::succeeded);
        CHECK(snapshot.admitted.decode.state == iom::ObservationState::not_run);
    }
}

inline void check_trace_rows(
        const ProbePlan& plan, const ModeObservation& observed,
        bool failure_after_prefill = false) {
    CHECK(observed.trace_prepared);
    CHECK_EQ(observed.trace_rows_dropped, std::uint64_t{0});
    const std::size_t prompt =
            (plan.reuse ? plan.second_prompt : plan.first_prompt).size();
    if (plan.first_limit == 0 && !plan.reuse) {
        CHECK(observed.operations.empty());
        return;
    }
    REQUIRE_FALSE(observed.operations.empty());
    const std::uint64_t ordinal = observed.snapshot->admitted.ordinal;
    bool saw_prefill = false;
    bool saw_decode = false;
    bool saw_terminal_window = false;
    std::vector<iom::oid> oids;
    for (const iom::InferenceTraceRecord& row : observed.operations) {
        if (row.wait_state == iom::WaitState::not_observed) {
            CHECK_FALSE(row.wait_observed_ns.has_value());
        } else {
            const bool wait_succeeded_or_failed =
                    row.wait_state == iom::WaitState::succeeded
                    || row.wait_state == iom::WaitState::failed;
            CHECK(wait_succeeded_or_failed);
            CHECK(row.wait_observed_ns.has_value());
        }
        const bool layer_is_in_range =
                !row.decoder_layer.has_value()
                || *row.decoder_layer
                        < plan.config.at("num_hidden_layers")
                                  .get<std::size_t>();
        CHECK(layer_is_in_range);
        CHECK(std::find(oids.begin(), oids.end(), row.operation) == oids.end());
        oids.push_back(row.operation);

        if (row.phase == iom::InferencePhase::prefill) {
            saw_prefill = true;
            CHECK(row.position_start + row.run_length <= prompt);
            if (row.run_length == prompt) {
                CHECK_EQ(row.position_start, std::size_t{0});
            } else {
                CHECK_EQ(row.run_length, std::size_t{1});
                CHECK_EQ(row.position_start + row.run_length, prompt);
                saw_terminal_window = true;
            }
        } else if (row.phase == iom::InferencePhase::decode) {
            saw_decode = true;
            CHECK_EQ(row.run_length, std::size_t{1});
            CHECK_GE(row.position_start, prompt);
        } else {
            CHECK_MESSAGE(false, "session operation row has an invalid phase");
        }
    }
    CHECK(saw_prefill);
    // Some accepted intermediate work is intentionally observed only by the
    // later dependency/destructor drain.  The row still carries copied layer
    // attribution when the submission has one; no extra wait is introduced by
    // this assertion.
    const bool terminal_window_or_single = saw_terminal_window || prompt == 1;
    CHECK(terminal_window_or_single);
    if (plan.reuse || failure_after_prefill) {
        CHECK(saw_decode);
    } else if (plan.first_sequence.size() == 1
               && plan.first_sequence.front() == 2) {
        CHECK_FALSE(saw_decode);
    } else if (plan.config.at("max_position_embeddings").get<std::size_t>()
               == prompt) {
        CHECK_FALSE(saw_decode);
    }
}

inline void run_success_branch(iom::Device& device, const ProbePlan& plan) {
    SyntheticFixture fixture(plan.tag, plan.config);
    const ModeObservation disabled = run_mode(
            device, fixture.directory.path(), plan, Mode::disabled);
    const ModeObservation metrics_only = run_mode(
            device, fixture.directory.path(), plan, Mode::metrics_only);
    const ModeObservation trace_enabled = run_mode(
            device, fixture.directory.path(), plan, Mode::trace_enabled);
    compare_modes(plan, disabled, metrics_only, trace_enabled);
    check_snapshot_expectation(plan, metrics_only);
    check_snapshot_expectation(plan, trace_enabled);
    if (plan.reuse) {
        CHECK_EQ(metrics_only.snapshot->admitted.ordinal, std::uint64_t{2});
        CHECK_EQ(trace_enabled.snapshot->admitted.ordinal, std::uint64_t{2});
    }
    check_trace_rows(plan, trace_enabled);
}

inline void run_failure_branch(iom::Device& device) {
    ProbePlan plan;
    plan.tag = "multi-step-decode-failure";
    plan.config = h16_config();
    plan.first_prompt = {0, 1};
    plan.first_sequence = {4, 5};
    plan.first_limit = 4;
    plan.expect_logits = false;

    SyntheticFixture fixture(plan.tag, plan.config);
    const ModeObservation disabled = run_mode(
            device, fixture.directory.path(), plan, Mode::disabled, true);
    const ModeObservation metrics_only = run_mode(
            device, fixture.directory.path(), plan, Mode::metrics_only, true);
    const ModeObservation trace_enabled = run_mode(
            device, fixture.directory.path(), plan, Mode::trace_enabled, true);
    compare_modes(plan, disabled, metrics_only, trace_enabled);
    CHECK(disabled.threw);
    CHECK(disabled.poisoned);
    check_snapshot_expectation(plan, metrics_only, true);
    check_snapshot_expectation(plan, trace_enabled, true);
    check_trace_rows(plan, trace_enabled, true);
}

}  // namespace inference_metrics_detail

inline void run_inference_instrumentation_conformance(iom::Device& device) {
    using namespace inference_metrics_detail;

    // First-token/EOS uses direct token IDs so the prefill and terminal
    // selection path are exercised without coupling it to tokenizer text.
    run_success_branch(device, ProbePlan{
            "first-token-eos", h16_config(), {0, 1}, {2}, 4, true,
            true});

    // Zero/new-token limit also covers raw tokenization.  The tokenizer adds
    // the BOS token to empty text, then the zero limit intentionally submits
    // no forward work.
    ProbePlan zero_limit{
            "zero-new-token-limit", h16_config(), {1}, {4}, 0, false,
            false};
    zero_limit.first_raw = true;
    zero_limit.first_text = "";
    run_success_branch(device, zero_limit);

    // Exact context: a full prompt submits prefill and returns before selector
    // work; its final-LM-head row is the absolute [R-1,R) window.
    nlohmann::json context_config = h16_config();
    context_config["max_position_embeddings"] = 4;
    run_success_branch(device, ProbePlan{
            "exact-context", std::move(context_config), {0, 1, 3, 4}, {4},
            4, true, true});

    // Multi-step decode: one prefill-produced token followed by two decode
    // forwards, including a terminal EOS commit.
    run_success_branch(device, ProbePlan{
            "multi-step-decode", h16_config(), {0, 1}, {4, 5, 2}, 4,
            true, true});
    run_failure_branch(device);

    // Two requests on one session: the second publication resets history/KV
    // prefixes and replaces the admitted ordinal and trace rows.
    ProbePlan reuse;
    reuse.tag = "two-requests-reuse";
    reuse.config = h16_config();
    reuse.first_prompt = {0, 1};
    reuse.first_sequence = {4, 2, 5, 2};
    reuse.first_limit = 4;
    reuse.second_prompt = {3};
    reuse.second_limit = 4;
    reuse.reuse = true;
    reuse.expect_logits = true;
    reuse.trace_rows_expected = true;
    run_success_branch(device, reuse);
}

}  // namespace iom_conformance
