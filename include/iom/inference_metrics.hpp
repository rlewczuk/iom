#pragma once

// Deterministic inference observation recorder.
//
// One caller-owned, scalar-only by default, noncopyable, nonmovable recorder
// collects the host clock, request-attempt lifecycle, phase spans, and
// optional bounded operation attribution produced by the bound nonreentrant
// session. Scalar recording is allocation-free and non-throwing; an optional,
// explicit `prepare_trace` reservation adds a fixed-capacity owned trace
// table that hooks append into without ever allocating or throwing after
// acceptance, and whose exhaustion is observably counted.
//
// The recorder is a backend-neutral data contract. It does not become a
// callback, subscriber, telemetry service, or backend timing framework, and
// it never queries a vendor runtime for device duration: the supplied host
// clock is the only timing source. See
// `docs/BACKEND_CONTRACT/inference-timing-and-observation.md` for the
// normative observation contract and the honest CPU/CUDA/ROCm/SYCL
// limitations.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <span>
#include <vector>

#include "iom/oid.hpp"

namespace iom {

// Defined in `iom/session.hpp` (authoritative). Declared here with the same
// explicit underlying type so the recorder header stays free of
// `session.hpp` and its heavy transitive includes while any later change to
// the authoritative underlying type becomes a diagnosed conflict instead of
// a silent layout divergence.
enum class GenerationStopReason : int;

/**
 * Supplied monotonic host clock.
 *
 * The function pointer and context together replace the default
 * `std::chrono::steady_clock` source so tests can produce deterministic
 * instants without touching wall clocks, vendor runtimes, or backend
 * capabilities. The supplied function MUST be `noexcept` and MUST NOT
 * allocate; the returned value is interpreted as host nanoseconds since an
 * unspecified, monotonic epoch.
 */
struct HostClock {
    using value_type = std::uint64_t;
    using now_function = value_type (*)(void* context) noexcept;
    now_function now = nullptr;
    void* context = nullptr;
};

/**
 * Which inference stage produced one observation row or scalar field.
 *
 * `load` covers the instrumented factory span, `tokenization` covers the
 * encode-only call, `prefill` and `decode` cover the completion-observed
 * forward spans and their per-facade host-enqueue sums.
 */
enum class InferencePhase {
    load,
    tokenization,
    prefill,
    decode,
};

/**
 * Whether a phase or attempt observation ran, completed, or failed.
 */
enum class ObservationState {
    not_run,
    succeeded,
    failed,
};

/**
 * One captured duration expressed as a host-clock delta, with an explicit
 * not-run/succeeded/failed state. Every phase hook stores
 * `end_instant - begin_instant` here; a bare clock timestamp is never
 * stored in this field.
 */
struct PhaseTiming {
    ObservationState state = ObservationState::not_run;
    std::uint64_t host_nanoseconds = 0;
};

/**
 * Host-enqueue observation for one phase: the sum of the individual facade
 * call intervals that enqueue work for that phase, accumulated exclusively
 * from supplied host-clock readings taken immediately before and immediately
 * after each facade call. Runtime blocking inside a facade is part of the
 * host-enqueue total; later waits, selector work, and entire forward spans
 * are not.
 */
struct PhaseEnqueue {
    std::uint64_t host_nanoseconds = 0;
};

/**
 * Outcome of one wait on a positive OID.
 */
enum class WaitState {
    not_observed,
    succeeded,
    failed,
};

/**
 * One owned operation observation.
 *
 * The schema is the contract of the recorder; every field is copied by value
 * from the existing forward, queue, and accepted-OID context so the row
 * survives source-scope destruction of the original stage borrow. No string,
 * tensor/view pointer, borrowed stage structure, or callback is retained.
 */
struct InferenceTraceRecord {
    std::uint64_t request_ordinal = 0;
    oid operation = 0;
    InferencePhase phase = InferencePhase::load;
    std::optional<std::size_t> decoder_layer;
    std::size_t position_start = 0;
    std::size_t run_length = 0;
    std::uint64_t enqueue_begin_ns = 0;
    std::uint64_t enqueue_end_ns = 0;
    std::optional<std::uint64_t> wait_observed_ns;
    WaitState wait_state = WaitState::not_observed;
    std::exception_ptr wait_failure;
};

/**
 * Distinct scalar outcomes distinguished for the latest attempt and the last
 * admitted request.
 */
enum class AttemptOutcome {
    none,
    in_progress,
    succeeded,
    failed,
};

/**
 * Latest-attempt observation. `prompt_tokens` is the staged prompt count
 * captured at generation entry; the failure, when present, is the original
 * exception preserved verbatim.
 */
struct AttemptObservation {
    std::uint64_t ordinal = 0;
    AttemptOutcome outcome = AttemptOutcome::none;
    std::size_t prompt_tokens = 0;
    std::exception_ptr failure;
};

/**
 * Last admitted request observation.
 *
 * Tokenization, prefill, and decode retain the frozen completion-observed
 * spans separately from the host-enqueue sums. Tokenization is a single
 * encode facade call, so its span is its own host-enqueue interval and no
 * duplicate field is kept. `prompt_tokens`, `generated_tokens`,
 * `decode_forward_count`, and `decode_token_count` retain the frozen
 * token-count meanings; `time_to_first_token_ns`,
 * `decode_throughput_denominator_ns`, and `tokens_per_second` are produced
 * only when their denominators are valid, otherwise they stay `unavailable`
 * (zero numerator, zero denominator, failed request, or no commit), and
 * `time_to_first_token_valid`/`decode_throughput_valid` are the explicit
 * availability flags. `stop_reason_valid` is false until a successful
 * `end_generation` records the reused `GenerationStopReason`, so no
 * enumerator value is ever fabricated by default construction.
 */
struct AdmittedObservation {
    std::uint64_t ordinal = 0;
    PhaseTiming tokenization;
    PhaseTiming prefill;
    PhaseTiming decode;
    PhaseEnqueue prefill_enqueue;
    PhaseEnqueue decode_enqueue;
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t decode_forward_count = 0;
    std::size_t decode_token_count = 0;
    std::uint64_t time_to_first_token_ns = 0;
    std::uint64_t decode_throughput_denominator_ns = 0;
    double tokens_per_second = 0.0;
    bool time_to_first_token_valid = false;
    bool decode_throughput_valid = false;
    bool stop_reason_valid = false;
    GenerationStopReason stop_reason{};
    std::exception_ptr failure;
};

/**
 * Snapshot of the recorder's scalar state.
 *
 * The `load` field and `trace_rows_dropped` are session-scoped and persist
 * across admitted requests; the `attempt` and `admitted` fields let callers
 * distinguish the latest attempted request from the last successfully
 * published request, and the frozen `request_admitted` flag reports whether
 * the admitted fields are meaningful (set after at least one successful
 * `publish_generation`).
 */
struct InferenceSnapshot {
    PhaseTiming load;
    AttemptObservation attempt;
    AdmittedObservation admitted;
    std::uint64_t trace_rows_dropped = 0;
    bool request_admitted = false;
};

/**
 * One caller-owned, noncopyable, nonmovable `InferenceMetrics` recorder.
 *
 * The recorder is the common backend-neutral data contract for opt-in
 * inference observation. It does not become a callback, subscriber, telemetry
 * service, or backend timing framework. Scalar recording is allocation-free
 * and non-throwing; the optional bounded trace table is reserved explicitly
 * through `prepare_trace` before the first request and is never extended by
 * the observation hooks. The scalar fields live directly in one
 * `InferenceSnapshot` member, so `snapshot()` returns a borrowed reference to
 * this recorder's own state without copying and without sharing storage
 * between recorder instances.
 *
 * Request-scoped phase spans, counters, TTFT and rate fields belong to the
 * admitted request only: they are written exactly while the latest attempt is
 * the published one, so a staged-but-unpublished attempt can never overwrite
 * the outgoing admitted request, whose outcome is reported through
 * `snapshot().attempt` instead. Trace rows are owned by the positive OID
 * independently of that guard.
 */
class InferenceMetrics {
public:
    using value_type = std::uint64_t;

    InferenceMetrics() noexcept;
    explicit InferenceMetrics(HostClock clock) noexcept;

    InferenceMetrics(const InferenceMetrics&) = delete;
    InferenceMetrics& operator=(const InferenceMetrics&) = delete;
    InferenceMetrics(InferenceMetrics&&) = delete;
    InferenceMetrics& operator=(InferenceMetrics&&) = delete;

    ~InferenceMetrics() = default;

    /** Read the supplied monotonic host clock in host nanoseconds. */
    [[nodiscard]] value_type now() const noexcept;

    /**
     * Read this recorder's scalar state. The reference is borrowed from the
     * recorder itself and stays valid, and keeps reporting this recorder's
     * own values, for the recorder's whole lifetime; no copy is made and no
     * other recorder on the same thread can alias it.
     */
    [[nodiscard]] const InferenceSnapshot& snapshot() const noexcept;

    /**
     * Read the owned trace records. The span is empty until `prepare_trace`
     * succeeds; after reservation, hooks append without ever growing or
     * reallocating the underlying storage.
     */
    [[nodiscard]] std::span<const InferenceTraceRecord> operations()
            const noexcept;

    /** True once `prepare_trace` has successfully reserved the trace table. */
    [[nodiscard]] bool trace_prepared() const noexcept;

    /**
     * Rows still available before the prepared trace table is full. A caller
     * that must not drop accepted work preflights this before invoking the
     * facade that accepts an operation; exhaustion itself is counted in
     * `snapshot().trace_rows_dropped`.
     */
    [[nodiscard]] std::size_t trace_capacity_remaining() const noexcept;

    // -----------------------------------------------------------------
    // Lifecycle and trace preparation.
    // -----------------------------------------------------------------

    /**
     * Stage a new request attempt. Captures the generation-entry instant
     * (the TTFT start instant) and the attempted input cardinality without
     * touching the last admitted request or any prepared trace rows.
     */
    void begin_generation(
            value_type host_instant, std::size_t prompt_tokens) noexcept;

    /**
     * Publish the latest staged attempt, advancing the admitted ordinal by
     * exactly one, resetting tokenization/prefill/decode spans and
     * host-enqueue sums, counters, TTFT and rate fields, and clearing
     * previously recorded operation rows without freeing prepared capacity.
     * Called only after successful request publication. A direct publication
     * with no staged attempt establishes an empty admitted observation
     * context with the next ordinal and zeroed scalars.
     */
    void publish_generation() noexcept;

    /**
     * Mark the current attempt as successfully finished with the reused
     * stop reason. A successful attempt with zero generated tokens leaves
     * TTFT and throughput unavailable and does not fabricate a rate.
     */
    void end_generation(
            value_type host_instant, GenerationStopReason stop_reason) noexcept;

    /**
     * Mark the current attempt as failed with the original exception.
     * The exception is captured by reference-counted ownership and is
     * preserved verbatim through subsequent successful requests. Counts
     * accumulated before the failure remain inspectable; the owning
     * request's rate becomes unavailable and stays unavailable for that
     * request without erasing completed counters.
     */
    void end_generation_failure(
            value_type host_instant, std::exception_ptr failure) noexcept;

    /**
     * Record a failed preprocessing attempt (formatter or encoder failure
     * before low-level generation). Tokenization stays `not_run` unless the
     * encoder itself was observed, and the failure remains attributed to the
     * latest attempt.
     */
    void record_preprocessing_failure(
            value_type host_instant, std::exception_ptr failure) noexcept;

    /**
     * Capture the load interval. Only the `end_instant - begin_instant`
     * delta is stored; the load span spans only the instrumented factory
     * and never becomes a raw clock timestamp.
     */
    void record_load(
            value_type begin_instant, value_type end_instant,
            ObservationState outcome = ObservationState::succeeded) noexcept;

    /**
     * Capture a tokenization completion span. The encode call is a single
     * facade invocation, so this span is also the tokenization host-enqueue
     * interval.
     */
    void record_tokenization(
            value_type begin_instant, value_type end_instant,
            ObservationState outcome = ObservationState::succeeded) noexcept;

    /**
     * Capture a prefill completion span.
     */
    void record_prefill(
            value_type begin_instant, value_type end_instant,
            ObservationState outcome = ObservationState::succeeded) noexcept;

    /**
     * Capture a decode completion span.
     */
    void record_decode(
            value_type begin_instant, value_type end_instant,
            ObservationState outcome = ObservationState::succeeded) noexcept;

    /**
     * Record a committed token, including terminal EOS/limit/context tokens.
     * Establishes the TTFT end on the first commit; advances
     * `generated_tokens`; and, when the token came from a decode forward,
     * counts it toward `decode_token_count` and closes the open decode
     * interval into the throughput denominator. Never allocates and never
     * rewrites an already computed rate.
     */
    void commit_token(
            value_type commit_instant, bool from_decode_forward) noexcept;

    /**
     * Record a successful readiness observation for the owning decoder
     * forward. Advances `decode_forward_count`. Does not commit a token.
     */
    void record_decode_forward(
            value_type wait_observed_instant) noexcept;

    /**
     * Open the decode interval for the decoder forward that is about to be
     * submitted. The session decodes one token at a time, so exactly one
     * interval is open at a time; a later open overwrites an unclosed
     * predecessor. The interval closes at the matching valid
     * decode-produced `commit_token`, and its elapsed time contributes to
     * the throughput denominator.
     */
    void record_decode_interval(
            value_type begin_instant) noexcept;

    /**
     * Reserve `capacity` rows of trace storage before the first request.
     *
     * The reservation may throw `std::overflow_error` when the requested
     * capacity is unrepresentable and `std::bad_alloc` when allocation
     * fails; in either case the recorder is left with tracing disabled and
     * no per-OID rows are created. After successful reservation the
     * capacity is fixed: hooks never grow, never throw, and never terminate.
     * Exhaustion is not silent — it is counted in `trace_rows_dropped`.
     */
    void prepare_trace(std::size_t capacity);

    /**
     * Append one accepted session-submitted positive OID to the trace.
     * The recorder rejects unknown, unregistered, or nonpositive OIDs
     * without creating a row. After successful preparation, every later
     * accepted submission is appended in OID acceptance order until the
     * prepared capacity is full, after which the row is dropped and
     * counted.
     */
    void record_enqueue(
            oid operation,
            InferencePhase phase,
            std::optional<std::size_t> decoder_layer,
            std::size_t position_start,
            std::size_t run_length,
            value_type enqueue_begin_instant,
            value_type enqueue_end_instant) noexcept;

    /**
     * Record the first wait observation for an existing positive-OID row.
     * `owning_request_ordinal` (zero means unowned) identifies the verified
     * owning request so a failure can invalidate that request's rate;
     * `host_instant` (empty means no clock reading was taken) optionally
     * provides the first observed host instant. A failed wait observation
     * preserves the first failure across repeated waits; a later success
     * does not erase an earlier retained failure for the same OID. Repeated
     * successes preserve the first successful observation. Unknown OIDs are
     * ignored, and `ObservationState::not_run` leaves the row unobserved.
     */
    void record_wait(
            oid operation,
            std::uint64_t owning_request_ordinal,
            std::optional<value_type> host_instant,
            ObservationState outcome,
            std::exception_ptr failure) noexcept;

    /**
     * Clear every recorded operation row without freeing prepared
     * capacity. The trace capacity invariant established by
     * `prepare_trace` is preserved; subsequent hooks append again from
     * an empty trace table.
     */
    void clear_operations() noexcept;

private:
    // True when request-scoped scalars belong to the admitted request, i.e.
    // the latest attempt is the one that was published. Every request-scoped
    // scalar write is guarded on this so a staged-but-unpublished attempt can
    // never overwrite the outgoing admitted request's fields.
    [[nodiscard]] bool writes_admitted_request() const noexcept;
    void reset_admitted_scalars(std::uint64_t ordinal) noexcept;

    HostClock clock_{};
    InferenceSnapshot state_{};
    std::vector<InferenceTraceRecord> operations_{};
    bool trace_prepared_ = false;
    bool generation_staged_ = false;
    bool decode_interval_open_ = false;
    bool rate_invalidated_ = false;
    value_type generation_entry_ns_ = 0;
    value_type decode_interval_begin_ns_ = 0;
};

}  // namespace iom
