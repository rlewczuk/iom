// Deterministic inference observation recorder implementation.
//
// The recorder is a backend-neutral data contract: the supplied monotonic
// host clock is the only timing source, scalar recording is allocation-free
// and non-throwing, and the optional bounded trace table is reserved
// explicitly through `prepare_trace` before the first request. See
// `include/iom/inference_metrics.hpp` for the full contract and
// `docs/BACKEND_CONTRACT/inference-timing-and-observation.md` for the
// normative observation specification.

#include "iom/inference_metrics.hpp"

#include <algorithm>
#include <chrono>
#include <new>
#include <stdexcept>
#include <utility>

namespace iom {

namespace {

constexpr std::uint64_t kHostNanosecondsPerSecond = 1'000'000'000ULL;

[[nodiscard]] std::uint64_t steady_clock_now(void* /*context*/) noexcept {
    const auto duration =
            std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                    .count());
}

[[nodiscard]] std::uint64_t elapsed(
        std::uint64_t begin_instant, std::uint64_t end_instant) noexcept {
    return end_instant >= begin_instant ? end_instant - begin_instant : 0;
}

[[nodiscard]] std::vector<InferenceTraceRecord>::iterator lower_bound_record(
        std::vector<InferenceTraceRecord>& rows, oid target) noexcept {
    const auto compare =
            [](const InferenceTraceRecord& row, oid value) noexcept {
                return row.operation < value;
            };
    return std::lower_bound(rows.begin(), rows.end(), target, compare);
}

}  // namespace

InferenceMetrics::InferenceMetrics() noexcept = default;
InferenceMetrics::InferenceMetrics(HostClock clock) noexcept : clock_(clock) {}

InferenceMetrics::value_type InferenceMetrics::now() const noexcept {
    if (clock_.now != nullptr) {
        return clock_.now(clock_.context);
    }
    return steady_clock_now(nullptr);
}

const InferenceSnapshot& InferenceMetrics::snapshot() const noexcept {
    return state_;
}

std::span<const InferenceTraceRecord> InferenceMetrics::operations()
        const noexcept {
    return std::span<const InferenceTraceRecord>(operations_.data(),
                                                 operations_.size());
}

bool InferenceMetrics::trace_prepared() const noexcept {
    return trace_prepared_;
}

std::size_t InferenceMetrics::trace_capacity_remaining() const noexcept {
    return trace_prepared_ ? operations_.capacity() - operations_.size() : 0;
}

bool InferenceMetrics::writes_admitted_request() const noexcept {
    return state_.request_admitted
            && state_.attempt.ordinal == state_.admitted.ordinal;
}

void InferenceMetrics::reset_admitted_scalars(std::uint64_t ordinal) noexcept {
    state_.admitted = AdmittedObservation{};
    state_.admitted.ordinal = ordinal;
    rate_invalidated_ = false;
    decode_interval_open_ = false;
    decode_interval_begin_ns_ = 0;
}

void InferenceMetrics::begin_generation(
        value_type host_instant, std::size_t prompt_tokens) noexcept {
    generation_entry_ns_ = host_instant;
    generation_staged_ = true;
    state_.attempt = AttemptObservation{};
    state_.attempt.ordinal = state_.admitted.ordinal + 1;
    state_.attempt.outcome = AttemptOutcome::in_progress;
    state_.attempt.prompt_tokens = prompt_tokens;
    decode_interval_open_ = false;
    decode_interval_begin_ns_ = 0;
}

void InferenceMetrics::publish_generation() noexcept {
    const std::uint64_t next_ordinal = state_.admitted.ordinal + 1;
    const bool staged = generation_staged_;
    const std::size_t prompt_tokens =
            staged ? state_.attempt.prompt_tokens : 0;
    reset_admitted_scalars(next_ordinal);
    state_.admitted.prompt_tokens = prompt_tokens;
    state_.request_admitted = true;
    state_.attempt = AttemptObservation{};
    state_.attempt.ordinal = next_ordinal;
    generation_staged_ = false;
    if (!staged) {
        // A publication with no staged generation establishes an empty
        // admitted context: no generation entry instant may be inherited
        // from a previous request. A staged publication keeps the instant
        // captured at generation entry, which is the TTFT start.
        generation_entry_ns_ = 0;
    }
    operations_.clear();
}

void InferenceMetrics::end_generation(
        value_type /*host_instant*/, GenerationStopReason stop_reason) noexcept {
    state_.attempt.outcome = AttemptOutcome::succeeded;
    state_.attempt.failure = nullptr;
    if (writes_admitted_request()) {
        state_.admitted.stop_reason = stop_reason;
        state_.admitted.stop_reason_valid = true;
        if (!rate_invalidated_ && state_.admitted.decode_token_count != 0
                && state_.admitted.decode_throughput_denominator_ns != 0) {
            const double seconds = static_cast<double>(
                                           state_.admitted
                                                   .decode_throughput_denominator_ns)
                    / static_cast<double>(kHostNanosecondsPerSecond);
            state_.admitted.tokens_per_second =
                    static_cast<double>(state_.admitted.decode_token_count)
                    / seconds;
            state_.admitted.decode_throughput_valid = true;
        }
    }
    decode_interval_open_ = false;
    decode_interval_begin_ns_ = 0;
}

void InferenceMetrics::end_generation_failure(
        value_type /*host_instant*/, std::exception_ptr failure) noexcept {
    state_.attempt.outcome = AttemptOutcome::failed;
    state_.attempt.failure = failure;
    if (writes_admitted_request()) {
        state_.admitted.failure = failure;
        // A retained failure invalidates the owning request's rate for good;
        // completed counters stay inspectable.
        state_.admitted.decode_throughput_valid = false;
        state_.admitted.tokens_per_second = 0.0;
        rate_invalidated_ = true;
    }
    decode_interval_open_ = false;
    decode_interval_begin_ns_ = 0;
}

void InferenceMetrics::record_preprocessing_failure(
        value_type /*host_instant*/, std::exception_ptr failure) noexcept {
    state_.attempt.outcome = AttemptOutcome::failed;
    state_.attempt.failure = failure;
    // Tokenization stays `not_run`: a formatter failure never reached the
    // encoder, so no failed encode observation may be fabricated. When the
    // attempt is not the admitted request, nothing request-scoped changes.
    if (writes_admitted_request()) {
        state_.admitted.failure = failure;
        state_.admitted.decode_throughput_valid = false;
        state_.admitted.tokens_per_second = 0.0;
        rate_invalidated_ = true;
    }
}

void InferenceMetrics::record_load(
        value_type begin_instant, value_type end_instant,
        ObservationState outcome) noexcept {
    state_.load.state = outcome;
    state_.load.host_nanoseconds = outcome == ObservationState::succeeded
            ? elapsed(begin_instant, end_instant)
            : 0;
}

void InferenceMetrics::record_tokenization(
        value_type begin_instant, value_type end_instant,
        ObservationState outcome) noexcept {
    if (!writes_admitted_request()) {
        return;
    }
    state_.admitted.tokenization.state = outcome;
    state_.admitted.tokenization.host_nanoseconds =
            outcome == ObservationState::succeeded
            ? elapsed(begin_instant, end_instant)
            : 0;
}

void InferenceMetrics::record_prefill(
        value_type begin_instant, value_type end_instant,
        ObservationState outcome) noexcept {
    if (!writes_admitted_request()) {
        return;
    }
    state_.admitted.prefill.state = outcome;
    state_.admitted.prefill.host_nanoseconds =
            outcome == ObservationState::succeeded
            ? elapsed(begin_instant, end_instant)
            : 0;
}

void InferenceMetrics::record_decode(
        value_type begin_instant, value_type end_instant,
        ObservationState outcome) noexcept {
    if (!writes_admitted_request()) {
        return;
    }
    state_.admitted.decode.state = outcome;
    state_.admitted.decode.host_nanoseconds =
            outcome == ObservationState::succeeded
            ? elapsed(begin_instant, end_instant)
            : 0;
}

void InferenceMetrics::commit_token(
        value_type commit_instant, bool from_decode_forward) noexcept {
    if (!writes_admitted_request()) {
        return;
    }
    state_.admitted.generated_tokens += 1;
    if (!state_.admitted.time_to_first_token_valid) {
        state_.admitted.time_to_first_token_ns =
                elapsed(generation_entry_ns_, commit_instant);
        state_.admitted.time_to_first_token_valid = true;
    }
    if (!from_decode_forward) {
        return;
    }
    state_.admitted.decode_token_count += 1;
    if (decode_interval_open_) {
        state_.admitted.decode_throughput_denominator_ns +=
                elapsed(decode_interval_begin_ns_, commit_instant);
        decode_interval_open_ = false;
    }
}

void InferenceMetrics::record_decode_forward(
        value_type /*wait_observed_instant*/) noexcept {
    if (!writes_admitted_request()) {
        return;
    }
    state_.admitted.decode_forward_count += 1;
}

void InferenceMetrics::record_decode_interval(
        value_type begin_instant) noexcept {
    if (!writes_admitted_request()) {
        return;
    }
    decode_interval_open_ = true;
    decode_interval_begin_ns_ = begin_instant;
}

void InferenceMetrics::prepare_trace(std::size_t capacity) {
    constexpr std::size_t kMaxCapacity =
            static_cast<std::size_t>(-1) / sizeof(InferenceTraceRecord);
    if (capacity > kMaxCapacity) {
        throw std::overflow_error(
                "InferenceMetrics trace capacity is unrepresentable");
    }
    // Re-preparation clears previously recorded rows and reserves the new
    // capacity; tracing is (re-)enabled only when the reservation succeeds.
    operations_.clear();
    operations_.reserve(capacity);
    trace_prepared_ = true;
}

void InferenceMetrics::record_enqueue(
        oid operation,
        InferencePhase phase,
        std::optional<std::size_t> decoder_layer,
        std::size_t position_start,
        std::size_t run_length,
        value_type enqueue_begin_instant,
        value_type enqueue_end_instant) noexcept {
    if (!oid_is_token(operation)) {
        return;
    }
    const std::uint64_t enqueue_delta =
            elapsed(enqueue_begin_instant, enqueue_end_instant);
    const std::uint64_t request_ordinal = generation_staged_
            ? state_.attempt.ordinal
            : state_.admitted.ordinal;
    if (writes_admitted_request()) {
        if (phase == InferencePhase::prefill) {
            state_.admitted.prefill_enqueue.host_nanoseconds += enqueue_delta;
        } else if (phase == InferencePhase::decode) {
            state_.admitted.decode_enqueue.host_nanoseconds += enqueue_delta;
        }
    }
    if (!trace_prepared_) {
        // Scalar-only mode: never allocate per-OID trace storage.
        return;
    }
    // One row per positive accepted OID; locate the existing row through a
    // sorted exact-OID binary search and ignore the duplicate acceptance.
    auto existing = lower_bound_record(operations_, operation);
    if (existing != operations_.end() && existing->operation == operation) {
        return;
    }
    if (operations_.size() >= operations_.capacity()) {
        // The prepared capacity is fixed by `prepare_trace`, so exhaustion
        // cannot grow, throw, or terminate. The dropped row is observable
        // through `snapshot().trace_rows_dropped` rather than silent.
        ++state_.trace_rows_dropped;
        return;
    }
    InferenceTraceRecord record{};
    record.request_ordinal = request_ordinal;
    record.operation = operation;
    record.phase = phase;
    record.decoder_layer = decoder_layer;
    record.position_start = position_start;
    record.run_length = run_length;
    record.enqueue_begin_ns = enqueue_begin_instant;
    record.enqueue_end_ns = enqueue_end_instant;
    record.wait_state = WaitState::not_observed;
    operations_.insert(existing, record);
}

void InferenceMetrics::record_wait(
        oid operation,
        std::uint64_t owning_request_ordinal,
        std::optional<value_type> host_instant,
        ObservationState outcome,
        std::exception_ptr failure) noexcept {
    if (!oid_is_token(operation)) {
        return;
    }
    if (outcome == ObservationState::failed
            && owning_request_ordinal != 0 && state_.request_admitted
            && state_.admitted.ordinal == owning_request_ordinal) {
        // A retained failure invalidates the verified owning request's rate
        // without erasing completed counters, whether or not a row exists.
        state_.admitted.decode_throughput_valid = false;
        state_.admitted.tokens_per_second = 0.0;
        rate_invalidated_ = true;
    }
    if (outcome == ObservationState::not_run) {
        // No observation took place: the row stays `not_observed` rather
        // than being misclassified as a failure.
        return;
    }
    auto row = lower_bound_record(operations_, operation);
    if (row == operations_.end() || row->operation != operation) {
        // Unknown or unregistered OID: never fabricate attribution.
        return;
    }
    if (row->wait_state != WaitState::not_observed) {
        // Preserve the first successful observation or the first retained
        // failure; a repeat never appends a row or rewrites the outcome.
        return;
    }
    row->wait_state = outcome == ObservationState::succeeded
            ? WaitState::succeeded
            : WaitState::failed;
    row->wait_observed_ns = host_instant;
    if (outcome == ObservationState::failed) {
        row->wait_failure = std::move(failure);
    }
}

void InferenceMetrics::clear_operations() noexcept {
    operations_.clear();
    decode_interval_open_ = false;
    decode_interval_begin_ns_ = 0;
}

}  // namespace iom
