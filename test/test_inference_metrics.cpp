// Focused coverage for the deterministic `InferenceMetrics` recorder.
//
// The tests exercise the recorder contract directly: supplied
// clock/event sequences, exact load/tokenization/prefill/decode sums, TTFT
// and throughput meanings, owned trace attribution after source-scope
// destruction, first-outcome retention, unknown/foreign/nonpositive
// observations, trace-capacity exhaustion, per-recorder snapshot ownership,
// and staged-attempt isolation. No session, device, or backend is involved,
// so the file remains backend-neutral and lives in the existing
// `iom_tests` target.

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "iom/inference_metrics.hpp"
#include "iom/oid.hpp"
#include "iom/session.hpp"

namespace {

struct RecordingClock {
    std::vector<std::uint64_t> instants{};
    std::size_t cursor = 0;
    std::atomic<bool> exhausted{false};

    static std::uint64_t read(void* context) noexcept {
        auto* self = static_cast<RecordingClock*>(context);
        if (self->cursor >= self->instants.size()) {
            self->exhausted.store(true, std::memory_order_relaxed);
            return 0;
        }
        return self->instants[self->cursor++];
    }
};

[[nodiscard]] iom::HostClock make_clock(RecordingClock& recorder) {
    return iom::HostClock{&RecordingClock::read, &recorder};
}

[[nodiscard]] auto find_row(
        std::span<const iom::InferenceTraceRecord> rows, iom::oid target) {
    return std::lower_bound(
            rows.begin(), rows.end(), target,
            [](const iom::InferenceTraceRecord& row, iom::oid value) noexcept {
                return row.operation < value;
            });
}

[[nodiscard]] iom::oid accepted_oid(std::uint64_t sequence) {
    // Queue id 16, sequence `sequence`: a canonical positive accepted token.
    return static_cast<iom::oid>((std::uint64_t{16} << 55) | sequence);
}

[[nodiscard]] std::string what_of(std::exception_ptr failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return "<non-standard>";
    }
}

}  // namespace

TEST_CASE("Inference metrics recorder uses the supplied monotonic clock") {
    RecordingClock clock_recorder{};
    clock_recorder.instants = {17, 19, 23, 29};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    CHECK_EQ(recorder.now(), 17);
    CHECK_EQ(recorder.now(), 19);
    CHECK_EQ(recorder.now(), 23);
    CHECK_EQ(recorder.now(), 29);
    // Falling off the supplied sequence returns zero rather than reading a
    // wall clock; the recorder is deterministic.
    CHECK_EQ(recorder.now(), 0);
    CHECK(clock_recorder.exhausted.load());
}

TEST_CASE("Inference metrics recorder defaults to steady_clock when no clock is supplied") {
    iom::InferenceMetrics recorder;
    const std::uint64_t first = recorder.now();
    const std::uint64_t second = recorder.now();
    CHECK_GE(second, first);
}

TEST_CASE("Inference metrics recorder is noncopyable and nonmovable") {
    CHECK_FALSE(std::is_copy_constructible_v<iom::InferenceMetrics>);
    CHECK_FALSE(std::is_copy_assignable_v<iom::InferenceMetrics>);
    CHECK_FALSE(std::is_move_constructible_v<iom::InferenceMetrics>);
    CHECK_FALSE(std::is_move_assignable_v<iom::InferenceMetrics>);
}

TEST_CASE("Inference metrics recorder own each snapshot reference") {
    RecordingClock first_clock{};
    RecordingClock second_clock{};
    iom::InferenceMetrics first(make_clock(first_clock));
    iom::InferenceMetrics second(make_clock(second_clock));
    first.begin_generation(0, 4);
    first.publish_generation();
    second.begin_generation(0, 9);
    second.publish_generation();
    const iom::InferenceSnapshot& first_view = first.snapshot();
    const iom::InferenceSnapshot& second_view = second.snapshot();
    CHECK_NE(&first_view, &second_view);
    CHECK_EQ(first_view.admitted.prompt_tokens, 4);
    CHECK_EQ(second_view.admitted.prompt_tokens, 9);
    // A later call on one recorder never rewrites another recorder's view,
    // and the borrowed reference still reports the same recorder's state.
    first.record_prefill(10, 30, iom::ObservationState::succeeded);
    CHECK_EQ(first_view.admitted.prefill.host_nanoseconds, 20);
    CHECK_EQ(second_view.admitted.prefill.state, iom::ObservationState::not_run);
}

TEST_CASE("Inference metrics recorder measures the load span from begin to end") {
    RecordingClock clock_recorder{};
    clock_recorder.instants = {1'000, 1'750};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    const std::uint64_t begin_instant = recorder.now();
    const std::uint64_t end_instant = recorder.now();
    recorder.record_load(begin_instant, end_instant,
                         iom::ObservationState::succeeded);
    const auto& snapshot = recorder.snapshot();
    CHECK(snapshot.load.state == iom::ObservationState::succeeded);
    // The stored field is the interval, never the absolute end instant.
    CHECK_EQ(snapshot.load.host_nanoseconds, 750);
    CHECK_LT(snapshot.load.host_nanoseconds, 1'000);
}

TEST_CASE("Inference metrics recorder records failed and not-run load spans") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.record_load(100, 220, iom::ObservationState::failed);
    CHECK(recorder.snapshot().load.state == iom::ObservationState::failed);
    CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 0);
    recorder.record_load(100, 300, iom::ObservationState::not_run);
    CHECK(recorder.snapshot().load.state == iom::ObservationState::not_run);
    CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 0);
    recorder.record_load(500, 560, iom::ObservationState::succeeded);
    CHECK_EQ(recorder.snapshot().load.host_nanoseconds, 60);
}

TEST_CASE("Inference metrics recorder exposes a borrowed scalar snapshot") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(recorder.now(), 4);
    recorder.publish_generation();
    recorder.record_tokenization(/*begin*/ 10, /*end*/ 30,
                                iom::ObservationState::succeeded);
    recorder.record_prefill(/*begin*/ 30, /*end*/ 80,
                            iom::ObservationState::succeeded);
    const auto& snapshot = recorder.snapshot();
    CHECK(snapshot.request_admitted);
    CHECK_EQ(snapshot.admitted.tokenization.host_nanoseconds, 20);
    CHECK_EQ(snapshot.admitted.prefill.host_nanoseconds, 50);
}

TEST_CASE("Inference metrics recorder measures completion observed phase spans") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 2);
    recorder.publish_generation();
    recorder.record_tokenization(0, 40, iom::ObservationState::succeeded);
    recorder.record_prefill(40, 140, iom::ObservationState::succeeded);
    recorder.record_decode(140, 200, iom::ObservationState::succeeded);
    const auto& snapshot = recorder.snapshot();
    CHECK_EQ(snapshot.admitted.tokenization.host_nanoseconds, 40);
    CHECK_EQ(snapshot.admitted.prefill.host_nanoseconds, 100);
    CHECK_EQ(snapshot.admitted.decode.host_nanoseconds, 60);
    // A failed span is never successful and reports no positive duration.
    recorder.record_decode(200, 260, iom::ObservationState::failed);
    CHECK(snapshot.admitted.decode.state == iom::ObservationState::failed);
    CHECK_EQ(snapshot.admitted.decode.host_nanoseconds, 0);
}

TEST_CASE("Inference metrics recorder measures TTFT from generation entry") {
    RecordingClock clock_recorder{};
    clock_recorder.instants = {100, 250};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(/*generation entry*/ 100, 3);
    recorder.publish_generation();
    recorder.commit_token(/*commit instant*/ 250,
                          /*from_decode_forward*/ false);
    const auto& snapshot = recorder.snapshot();
    CHECK(snapshot.admitted.time_to_first_token_valid);
    CHECK_EQ(snapshot.admitted.time_to_first_token_ns, 150);
    CHECK_EQ(snapshot.admitted.generated_tokens, 1);
    CHECK_EQ(snapshot.admitted.decode_token_count, 0);
}

TEST_CASE("Inference metrics recorder leaves TTFT unavailable with no commit") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 5);
    recorder.publish_generation();
    recorder.record_prefill(0, 20, iom::ObservationState::succeeded);
    const auto& snapshot = recorder.snapshot();
    CHECK_FALSE(snapshot.admitted.time_to_first_token_valid);
    CHECK_EQ(snapshot.admitted.time_to_first_token_ns, 0);
}

TEST_CASE("Inference metrics recorder ignores prefill-first commit in decode count") {
    RecordingClock clock_recorder{};
    clock_recorder.instants = {0, 100, 200};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 2);
    recorder.publish_generation();
    recorder.record_prefill(0, 100, iom::ObservationState::succeeded);
    // The prefill-produced first commit is not a decode token.
    recorder.commit_token(100, /*from_decode_forward*/ false);
    // The decode-produced commit counts as one decode token.
    recorder.commit_token(200, /*from_decode_forward*/ true);
    const auto& snapshot = recorder.snapshot();
    CHECK_EQ(snapshot.admitted.generated_tokens, 2);
    CHECK_EQ(snapshot.admitted.decode_token_count, 1);
}

TEST_CASE("Inference metrics recorder counts completion observed decode forwards") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    CHECK_EQ(recorder.snapshot().admitted.decode_forward_count, 0);
    recorder.record_decode_forward(120);
    recorder.record_decode_forward(180);
    CHECK_EQ(recorder.snapshot().admitted.decode_forward_count, 2);
    // A blocked staging attempt is not the admitted request and cannot
    // advance the admitted request's completed-forward count.
    recorder.begin_generation(200, 2);
    recorder.record_decode_forward(260);
    CHECK_EQ(recorder.snapshot().admitted.decode_forward_count, 2);
}

TEST_CASE("Inference metrics recorder captures EOS terminal commit") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_prefill(0, 50, iom::ObservationState::succeeded);
    recorder.commit_token(50, /*from_decode_forward*/ false);
    recorder.record_decode_interval(80);
    recorder.commit_token(120, /*from_decode_forward*/ true);
    recorder.record_decode_interval(140);
    recorder.commit_token(180, /*from_decode_forward*/ true);
    CHECK_FALSE(recorder.snapshot().admitted.decode_throughput_valid);
    CHECK_FALSE(recorder.snapshot().admitted.stop_reason_valid);
    recorder.end_generation(180, iom::GenerationStopReason::eos);
    const auto& snapshot = recorder.snapshot();
    CHECK_EQ(snapshot.admitted.generated_tokens, 3);
    CHECK_EQ(snapshot.admitted.decode_token_count, 2);
    CHECK(snapshot.admitted.stop_reason_valid);
    CHECK(snapshot.admitted.stop_reason == iom::GenerationStopReason::eos);
    CHECK(snapshot.admitted.decode_throughput_valid);
    // Two decode commits over a 40ns+40ns = 80ns denominator:
    //   2 tokens / 0.000000080 seconds = 25_000_000 tokens/s.
    CHECK_EQ(snapshot.admitted.tokens_per_second, doctest::Approx(25000000.0));
}

TEST_CASE("Inference metrics recorder records limit stop precedence") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_prefill(0, 50, iom::ObservationState::succeeded);
    recorder.commit_token(50, /*from_decode_forward*/ false);
    recorder.record_decode_interval(80);
    recorder.commit_token(120, /*from_decode_forward*/ true);
    recorder.end_generation(120, iom::GenerationStopReason::max_new_tokens);
    const auto& snapshot = recorder.snapshot();
    CHECK(snapshot.request_admitted);
    CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::succeeded);
    CHECK(snapshot.admitted.stop_reason_valid);
    CHECK(snapshot.admitted.stop_reason
          == iom::GenerationStopReason::max_new_tokens);
}

TEST_CASE("Inference metrics recorder handles context capacity stop") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.commit_token(10, /*from_decode_forward*/ false);
    recorder.end_generation(20, iom::GenerationStopReason::context_capacity);
    const auto& snapshot = recorder.snapshot();
    CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::succeeded);
    CHECK(snapshot.admitted.stop_reason_valid);
    CHECK(snapshot.admitted.stop_reason
          == iom::GenerationStopReason::context_capacity);
}

TEST_CASE("Inference metrics recorder handles zero-duration clocks") {
    RecordingClock clock_recorder{};
    clock_recorder.instants = {0, 0, 0, 0};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 2);
    recorder.publish_generation();
    recorder.record_prefill(0, 0, iom::ObservationState::succeeded);
    recorder.commit_token(0, /*from_decode_forward*/ false);
    recorder.record_decode_interval(0);
    recorder.commit_token(0, /*from_decode_forward*/ true);
    recorder.end_generation(0, iom::GenerationStopReason::max_new_tokens);
    const auto& snapshot = recorder.snapshot();
    CHECK_EQ(snapshot.admitted.prefill.host_nanoseconds, 0);
    CHECK_EQ(snapshot.admitted.generated_tokens, 2);
    // Zero denominator leaves the rate unavailable rather than infinity.
    CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
    CHECK_EQ(snapshot.admitted.tokens_per_second, 0.0);
}

TEST_CASE("Inference metrics recorder invalidates rate after retained wait failure") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_prefill(0, 50, iom::ObservationState::succeeded);
    recorder.commit_token(50, /*from_decode_forward*/ false);
    recorder.record_decode_interval(80);
    recorder.commit_token(150, /*from_decode_forward*/ true);
    recorder.end_generation(150, iom::GenerationStopReason::max_new_tokens);
    REQUIRE(recorder.snapshot().admitted.decode_throughput_valid);
    // A drain wait on the owning request's accepted OID retains the failure
    // and invalidates the already computed rate.
    recorder.record_wait(
            accepted_oid(1), /*owning ordinal*/ 1, /*host instant*/ 170,
            iom::ObservationState::failed,
            std::make_exception_ptr(std::runtime_error("boom")));
    const auto& snapshot = recorder.snapshot();
    // Counters preserved, rate invalidated and never recomputed as valid.
    CHECK_EQ(snapshot.admitted.generated_tokens, 2);
    CHECK_EQ(snapshot.admitted.decode_token_count, 1);
    CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
    CHECK_EQ(snapshot.admitted.tokens_per_second, 0.0);
    CHECK(snapshot.admitted.stop_reason_valid);
}

TEST_CASE("Inference metrics recorder preserves counters after generation failure") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_prefill(0, 50, iom::ObservationState::succeeded);
    recorder.commit_token(50, /*from_decode_forward*/ false);
    recorder.record_decode_interval(80);
    recorder.commit_token(120, /*from_decode_forward*/ true);
    const auto failure = std::make_exception_ptr(std::invalid_argument("oob"));
    recorder.end_generation_failure(140, failure);
    const auto& snapshot = recorder.snapshot();
    CHECK_EQ(snapshot.admitted.generated_tokens, 2);
    CHECK_EQ(snapshot.admitted.decode_token_count, 1);
    CHECK_EQ(snapshot.admitted.decode_forward_count, 0);
    CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
    CHECK_FALSE(snapshot.admitted.stop_reason_valid);
    CHECK_EQ(what_of(snapshot.attempt.failure), "oob");
}

TEST_CASE("Inference metrics recorder rejects unrepresentable trace capacity") {
    iom::InferenceMetrics recorder;
    CHECK_FALSE(recorder.trace_prepared());
    REQUIRE_THROWS_AS(
            recorder.prepare_trace(
                    std::size_t{std::numeric_limits<std::size_t>::max()}),
            std::overflow_error);
    CHECK_FALSE(recorder.trace_prepared());
    CHECK(recorder.operations().empty());
}

TEST_CASE("Inference metrics recorder appends trace rows without allocation") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(4);
    CHECK(recorder.trace_prepared());
    CHECK_EQ(recorder.trace_capacity_remaining(), 4);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_enqueue(
            accepted_oid(1), iom::InferencePhase::prefill,
            std::nullopt, 0, 1, 0, 5);
    recorder.record_enqueue(
            accepted_oid(2), iom::InferencePhase::decode,
            std::optional<std::size_t>{0}, 1, 1, 5, 11);
    const auto rows = recorder.operations();
    REQUIRE_EQ(rows.size(), 2);
    CHECK_EQ(rows.front().operation, accepted_oid(1));
    CHECK(rows.front().phase == iom::InferencePhase::prefill);
    CHECK_FALSE(rows.front().decoder_layer.has_value());
    CHECK_EQ(rows.front().position_start, 0);
    CHECK_EQ(rows.front().run_length, 1);
    CHECK_EQ(rows.front().enqueue_end_ns, 5);
    CHECK(rows.back().decoder_layer.has_value());
    CHECK_EQ(*rows.back().decoder_layer, 0);
    CHECK_EQ(recorder.trace_capacity_remaining(), 2);
    CHECK_EQ(recorder.snapshot().trace_rows_dropped, 0);
}

TEST_CASE("Inference metrics recorder counts every row dropped at prepared capacity") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(2);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 5);
    recorder.record_enqueue(accepted_oid(2), iom::InferencePhase::decode,
                            std::optional<std::size_t>{0}, 1, 1, 5, 9);
    CHECK_EQ(recorder.trace_capacity_remaining(), 0);
    recorder.record_enqueue(accepted_oid(3), iom::InferencePhase::decode,
                            std::optional<std::size_t>{0}, 2, 1, 9, 12);
    recorder.record_enqueue(accepted_oid(4), iom::InferencePhase::decode,
                            std::optional<std::size_t>{0}, 3, 1, 12, 15);
    // Exhaustion never allocates or throws; it is counted, not silent.
    CHECK_EQ(recorder.operations().size(), 2);
    CHECK_EQ(recorder.snapshot().trace_rows_dropped, 2);
    CHECK_EQ(recorder.trace_capacity_remaining(), 0);
}

TEST_CASE("Inference metrics recorder assigns one row per positive accepted OID") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(8);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 5);
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 6);
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 7);
    CHECK_EQ(recorder.operations().size(), 1);
    CHECK_EQ(recorder.operations().front().enqueue_end_ns, 5);
    CHECK_EQ(recorder.snapshot().trace_rows_dropped, 0);
}

TEST_CASE("Inference metrics recorder ignores nonpositive and foreign OIDs") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(4);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_enqueue(0, iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 1);
    recorder.record_enqueue(-1, iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 1);
    recorder.record_enqueue(iom::to_oid(iom::OidError::Overflow),
                            iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 1);
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 1);
    CHECK_EQ(recorder.operations().size(), 1);
    // A rejected submission is not a dropped accepted row.
    CHECK_EQ(recorder.snapshot().trace_rows_dropped, 0);
}

TEST_CASE("Inference metrics recorder preserves rows after source-scope destruction") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(2);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    {
        // Every value the row copies lives only inside this scope.
        const iom::oid operation = accepted_oid(7);
        const iom::InferencePhase phase = iom::InferencePhase::prefill;
        const std::optional<std::size_t> layer = std::nullopt;
        const std::size_t position_start = 0;
        const std::size_t run_length = 3;
        const std::uint64_t begin_instant = 41;
        const std::uint64_t end_instant = 59;
        recorder.record_enqueue(operation, phase, layer, position_start,
                                run_length, begin_instant, end_instant);
    }
    const auto rows = recorder.operations();
    REQUIRE_EQ(rows.size(), 1);
    CHECK_EQ(rows.front().operation, accepted_oid(7));
    CHECK(rows.front().phase == iom::InferencePhase::prefill);
    CHECK_EQ(rows.front().position_start, 0);
    CHECK_EQ(rows.front().run_length, 3);
    CHECK_EQ(rows.front().enqueue_begin_ns, 41);
    CHECK_EQ(rows.front().enqueue_end_ns, 59);
    CHECK_EQ(rows.front().request_ordinal, 1);
}

TEST_CASE("Inference metrics recorder retains first wait success and ignores repeats") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(2);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    const iom::oid accepted = accepted_oid(2);
    recorder.record_enqueue(accepted, iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 5);
    recorder.record_wait(accepted, /*owning ordinal*/ 1, /*host instant*/ 12,
                         iom::ObservationState::succeeded, nullptr);
    recorder.record_wait(accepted, /*owning ordinal*/ 1, /*host instant*/ 20,
                         iom::ObservationState::succeeded, nullptr);
    REQUIRE_EQ(recorder.operations().size(), 1);
    CHECK(recorder.operations().front().wait_state == iom::WaitState::succeeded);
    CHECK(recorder.operations().front().wait_observed_ns.has_value());
    CHECK_EQ(*recorder.operations().front().wait_observed_ns, 12);
}

TEST_CASE("Inference metrics recorder retains first failure and ignores later success") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(2);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    const iom::oid accepted = accepted_oid(3);
    recorder.record_enqueue(accepted, iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 5);
    const auto failure = std::make_exception_ptr(std::runtime_error("boom"));
    recorder.record_wait(accepted, 1, 12, iom::ObservationState::failed,
                         failure);
    recorder.record_wait(accepted, 1, 20, iom::ObservationState::succeeded,
                         nullptr);
    REQUIRE_EQ(recorder.operations().size(), 1);
    CHECK(recorder.operations().front().wait_state == iom::WaitState::failed);
    CHECK_EQ(what_of(recorder.operations().front().wait_failure), "boom");
}

TEST_CASE("Inference metrics recorder leaves not-run wait observations unobserved") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(2);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    const iom::oid accepted = accepted_oid(4);
    recorder.record_enqueue(accepted, iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 5);
    recorder.record_wait(accepted, 1, std::nullopt,
                         iom::ObservationState::not_run, nullptr);
    REQUIRE_EQ(recorder.operations().size(), 1);
    CHECK(recorder.operations().front().wait_state
          == iom::WaitState::not_observed);
    CHECK_FALSE(recorder.operations().front().wait_observed_ns.has_value());
    CHECK(recorder.operations().front().wait_failure == nullptr);
}

TEST_CASE("Inference metrics recorder preserves later OID success without erasing earlier failure") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(4);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    const iom::oid first = accepted_oid(4);
    const iom::oid second = accepted_oid(5);
    recorder.record_enqueue(first, iom::InferencePhase::decode,
                            std::optional<std::size_t>{0}, 0, 1, 0, 5);
    recorder.record_enqueue(second, iom::InferencePhase::decode,
                            std::optional<std::size_t>{1}, 1, 1, 5, 9);
    const auto failure = std::make_exception_ptr(std::runtime_error("first"));
    recorder.record_wait(first, 1, 7, iom::ObservationState::failed, failure);
    recorder.record_wait(second, 1, 9, iom::ObservationState::succeeded,
                         nullptr);
    const auto rows = recorder.operations();
    const auto first_row = find_row(rows, first);
    const auto second_row = find_row(rows, second);
    REQUIRE(first_row != rows.end());
    REQUIRE(second_row != rows.end());
    CHECK(first_row->wait_state == iom::WaitState::failed);
    CHECK(second_row->wait_state == iom::WaitState::succeeded);
}

TEST_CASE("Inference metrics recorder ignores wait on unknown OID") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(2);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_wait(accepted_oid(99), 1, 5,
                         iom::ObservationState::succeeded, nullptr);
    recorder.record_wait(0, 1, 5, iom::ObservationState::succeeded, nullptr);
    recorder.record_wait(-1, 1, 5, iom::ObservationState::failed,
                         std::make_exception_ptr(std::runtime_error("x")));
    CHECK(recorder.operations().empty());
}

TEST_CASE("Inference metrics recorder ignores foreign-OID failure attribution") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.commit_token(10, /*from_decode_forward*/ false);
    recorder.record_decode_interval(20);
    recorder.commit_token(40, /*from_decode_forward*/ true);
    recorder.end_generation(40, iom::GenerationStopReason::max_new_tokens);
    REQUIRE(recorder.snapshot().admitted.decode_throughput_valid);
    // An unowned failure (ordinal 0) and a foreign ordinal change nothing.
    recorder.record_wait(accepted_oid(99), 0, 50,
                         iom::ObservationState::failed,
                         std::make_exception_ptr(std::runtime_error("x")));
    recorder.record_wait(accepted_oid(99), 7, 50,
                         iom::ObservationState::failed,
                         std::make_exception_ptr(std::runtime_error("x")));
    CHECK(recorder.operations().empty());
    CHECK(recorder.snapshot().admitted.decode_throughput_valid);
}

TEST_CASE("Inference metrics recorder invalidates rate on unknown-OID failure") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.commit_token(10, /*from_decode_forward*/ false);
    recorder.record_decode_interval(20);
    recorder.commit_token(40, /*from_decode_forward*/ true);
    recorder.end_generation(40, iom::GenerationStopReason::max_new_tokens);
    REQUIRE(recorder.snapshot().admitted.decode_throughput_valid);
    // The session verified ownership from its accepted-OID ledger even
    // though no trace row exists: the owning rate is still invalidated.
    recorder.record_wait(accepted_oid(99), 1, 50,
                         iom::ObservationState::failed,
                         std::make_exception_ptr(std::runtime_error("x")));
    const auto& snapshot = recorder.snapshot();
    CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
    CHECK_EQ(snapshot.admitted.tokens_per_second, 0.0);
    CHECK_EQ(snapshot.admitted.generated_tokens, 2);
}

TEST_CASE("Inference metrics recorder clear_operations preserves capacity") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(3);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 5);
    recorder.record_enqueue(accepted_oid(2), iom::InferencePhase::decode,
                            std::optional<std::size_t>{0}, 1, 1, 5, 9);
    REQUIRE_EQ(recorder.operations().size(), 2);
    recorder.clear_operations();
    CHECK(recorder.operations().empty());
    CHECK_EQ(recorder.trace_capacity_remaining(), 3);
    recorder.record_enqueue(accepted_oid(3), iom::InferencePhase::decode,
                            std::optional<std::size_t>{0}, 2, 1, 9, 12);
    REQUIRE_EQ(recorder.operations().size(), 1);
    recorder.record_enqueue(accepted_oid(4), iom::InferencePhase::decode,
                            std::optional<std::size_t>{0}, 3, 1, 12, 15);
    CHECK_EQ(recorder.operations().size(), 2);
    CHECK_EQ(recorder.snapshot().trace_rows_dropped, 0);
}

TEST_CASE("Inference metrics recorder keeps zero scalar storage without prepare_trace") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    CHECK_FALSE(recorder.trace_prepared());
    CHECK(recorder.operations().empty());
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 5);
    // No reservation: scalar enqueue host time still accumulates, but no
    // row is created and nothing is counted as a dropped accepted row.
    CHECK(recorder.operations().empty());
    CHECK_EQ(recorder.snapshot().admitted.prefill_enqueue.host_nanoseconds, 5);
    CHECK_EQ(recorder.snapshot().trace_rows_dropped, 0);
    CHECK_EQ(recorder.trace_capacity_remaining(), 0);
}

TEST_CASE("Inference metrics recorder sums per-facade host enqueue work") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 4, 0, 7);
    recorder.record_enqueue(accepted_oid(2), iom::InferencePhase::prefill,
                            std::optional<std::size_t>{0}, 0, 4, 100, 108);
    recorder.record_enqueue(accepted_oid(3), iom::InferencePhase::decode,
                            std::optional<std::size_t>{0}, 4, 1, 200, 203);
    const auto& snapshot = recorder.snapshot();
    CHECK_EQ(snapshot.admitted.prefill_enqueue.host_nanoseconds, 15);
    CHECK_EQ(snapshot.admitted.decode_enqueue.host_nanoseconds, 3);
    // Host enqueue stays separate from the completion-observed span.
    CHECK(snapshot.admitted.prefill.state == iom::ObservationState::not_run);
}

TEST_CASE("Inference metrics recorder preserves admitted request across staged validation failure") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_prefill(0, 50, iom::ObservationState::succeeded);
    recorder.commit_token(50, /*from_decode_forward*/ false);
    recorder.record_decode_interval(80);
    recorder.commit_token(120, /*from_decode_forward*/ true);
    recorder.end_generation(120, iom::GenerationStopReason::max_new_tokens);
    REQUIRE(recorder.snapshot().admitted.decode_throughput_valid);
    // A staged attempt that never publishes must not write any
    // request-scoped span into the outgoing admitted request.
    recorder.begin_generation(200, 2);
    recorder.record_prefill(210, 260, iom::ObservationState::succeeded);
    recorder.record_tokenization(210, 240, iom::ObservationState::failed);
    recorder.record_preprocessing_failure(
            220, std::make_exception_ptr(std::invalid_argument("bad")));
    const auto& snapshot = recorder.snapshot();
    CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::failed);
    CHECK(snapshot.request_admitted);
    CHECK_EQ(snapshot.admitted.ordinal, 1);
    CHECK_EQ(snapshot.admitted.generated_tokens, 2);
    CHECK(snapshot.admitted.decode_throughput_valid);
    CHECK(snapshot.admitted.stop_reason_valid);
    CHECK_EQ(snapshot.admitted.prefill.host_nanoseconds, 50);
    // A formatter failure never reached the encoder: tokenization stays
    // not_run rather than fabricating a failed encode.
    CHECK(snapshot.admitted.tokenization.state
          == iom::ObservationState::not_run);
    CHECK_EQ(what_of(snapshot.attempt.failure), "bad");
}

TEST_CASE("Inference metrics recorder distinguishes attempt outcome from admitted request") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_prefill(0, 50, iom::ObservationState::succeeded);
    recorder.commit_token(50, /*from_decode_forward*/ false);
    recorder.commit_token(120, /*from_decode_forward*/ true);
    recorder.end_generation(120, iom::GenerationStopReason::max_new_tokens);
    // A new staged attempt that fails after a successful publication must
    // produce a distinct failed-attempt observation with its own ordinal.
    recorder.begin_generation(200, 2);
    recorder.publish_generation();
    recorder.record_preprocessing_failure(
            210, std::make_exception_ptr(std::invalid_argument("bad")));
    const auto& snapshot = recorder.snapshot();
    CHECK(snapshot.attempt.outcome == iom::AttemptOutcome::failed);
    CHECK(snapshot.request_admitted);
    CHECK_EQ(snapshot.admitted.ordinal, 2);
    CHECK_EQ(snapshot.admitted.generated_tokens, 0);
    CHECK_FALSE(snapshot.admitted.stop_reason_valid);
    CHECK_EQ(what_of(snapshot.attempt.failure), "bad");
}

TEST_CASE("Inference metrics recorder advances the ordinal across direct publication") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.begin_generation(0, 3);
    recorder.publish_generation();
    recorder.commit_token(10, /*from_decode_forward*/ false);
    recorder.end_generation(10, iom::GenerationStopReason::max_new_tokens);
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 1);
    // A direct publication with no staged generation establishes an empty
    // admitted context at the NEXT ordinal; it never regresses.
    recorder.publish_generation();
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 2);
    CHECK_EQ(recorder.snapshot().admitted.generated_tokens, 0);
    CHECK_FALSE(recorder.snapshot().admitted.time_to_first_token_valid);
    // The next staged request continues from the same monotonic sequence.
    recorder.begin_generation(50, 4);
    CHECK_EQ(recorder.snapshot().attempt.ordinal, 3);
    recorder.publish_generation();
    CHECK_EQ(recorder.snapshot().admitted.ordinal, 3);
    CHECK_EQ(recorder.snapshot().admitted.prompt_tokens, 4);
    // The direct publication left no inherited generation entry instant.
    recorder.commit_token(80, /*from_decode_forward*/ false);
    CHECK(recorder.snapshot().admitted.time_to_first_token_valid);
    CHECK_EQ(recorder.snapshot().admitted.time_to_first_token_ns, 30);
}

TEST_CASE("Inference metrics recorder republication clears previous rows and keeps capacity") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(4);
    recorder.begin_generation(0, 1);
    recorder.publish_generation();
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 0, 5);
    REQUIRE_EQ(recorder.operations().size(), 1);
    // Successful republication clears the previous request's rows while
    // keeping the prepared capacity.
    recorder.begin_generation(20, 1);
    recorder.publish_generation();
    CHECK(recorder.operations().empty());
    CHECK_EQ(recorder.trace_capacity_remaining(), 4);
    recorder.record_enqueue(accepted_oid(2), iom::InferencePhase::prefill,
                            std::nullopt, 0, 1, 20, 27);
    REQUIRE_EQ(recorder.operations().size(), 1);
    CHECK_EQ(recorder.operations().front().request_ordinal, 2);
}

TEST_CASE("Inference metrics recorder handles decode-only enqueue with no-decode commit") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(4);
    recorder.begin_generation(0, 2);
    recorder.publish_generation();
    recorder.record_prefill(0, 100, iom::ObservationState::succeeded);
    recorder.commit_token(100, /*from_decode_forward*/ false);
    recorder.end_generation(120, iom::GenerationStopReason::context_capacity);
    const auto& snapshot = recorder.snapshot();
    CHECK_EQ(snapshot.admitted.generated_tokens, 1);
    CHECK_EQ(snapshot.admitted.decode_token_count, 0);
    CHECK_FALSE(snapshot.admitted.decode_throughput_valid);
    CHECK(recorder.operations().empty());
}

TEST_CASE("Inference metrics recorder copies configured layer into trace rows") {
    RecordingClock clock_recorder{};
    iom::InferenceMetrics recorder(make_clock(clock_recorder));
    recorder.prepare_trace(6);
    recorder.begin_generation(0, 2);
    recorder.publish_generation();
    recorder.record_enqueue(accepted_oid(1), iom::InferencePhase::prefill,
                            std::nullopt, 0, 2, 0, 5);
    recorder.record_enqueue(accepted_oid(2), iom::InferencePhase::decode,
                            std::optional<std::size_t>{2}, 2, 1, 5, 9);
    recorder.record_enqueue(accepted_oid(3), iom::InferencePhase::decode,
                            std::optional<std::size_t>{5}, 3, 1, 9, 12);
    const auto rows = recorder.operations();
    REQUIRE_EQ(rows.size(), 3);
    // Final LM-head row uses the last decoded position as its single-row
    // window. The enqueue probe leaf wraps it with no decoder layer.
    CHECK_FALSE(rows[0].decoder_layer.has_value());
    CHECK(rows[1].decoder_layer.has_value());
    CHECK_EQ(*rows[1].decoder_layer, 2);
    CHECK_EQ(rows[1].position_start, 2);
    CHECK_EQ(rows[1].run_length, 1);
    CHECK(rows[2].decoder_layer.has_value());
    CHECK_EQ(*rows[2].decoder_layer, 5);
}

TEST_CASE("Inference metrics recorder constructor accepts no clock argument") {
    iom::InferenceMetrics first{};
    iom::InferenceMetrics second{};
    CHECK(first.snapshot().load.host_nanoseconds == 0);
    CHECK(second.snapshot().load.host_nanoseconds == 0);
}
