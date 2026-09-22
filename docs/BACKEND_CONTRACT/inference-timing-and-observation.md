# 14. Inference timing and observation

This section is the normative contract for opt-in inference observation.
It defines the caller-owned, backend-neutral `InferenceMetrics` recorder,
its scalar and trace schemas, the supplied monotonic host clock, the
backend-neutral timing meanings, and the honest CPU, CUDA, ROCm, and SYCL
limitations on genuine device duration. The recorder is a data contract;
it is not a callback, subscriber, telemetry service, or backend timing
framework. It does not introduce a universal timing API or a vendor-timing
adapter.

## General definitions

1. The recorder is exactly one caller-owned `InferenceMetrics` object per
   active nonreentrant session. The recorder is noncopyable and nonmovable,
   has no registry, no subscribers, no background tasks, no callbacks, and
   no ownership transfer. The supplied host-clock function and its context
   are borrowed and must outlive the recorder; the recorder is expected to
   outlive the bound session, including its destructor and ordinary
   drain.
2. Scalar recording is allocation-free and non-throwing. The recorder
   exposes `snapshot()` as a borrowed `const` reference to its own scalar
   snapshot and `operations()` as a borrowed
   `std::span<const InferenceTraceRecord>` over owned trace rows. The
   reference is stable for the recorder's lifetime, reports only that
   recorder's values, and is never shared between recorder instances. The
   recorder exposes `now()` to read the supplied host clock in host
   nanoseconds.
3. The recorder is optional. A null or unbound recorder disables every
   observation hook with no clock reads, no per-operation allocation, no
   registration, no wait, and no synchronization. The existing queue,
   session, model, kernel, and stop semantics are unchanged with the
   recorder absent.
4. `prepare_trace(capacity)` is an explicit, caller-controlled
   configuration operation. It is valid only before the first request and
   may throw `std::overflow_error` (unrepresentable capacity) or
   `std::bad_alloc` (allocation failure) before any inference work begins.
   Tracing is enabled only after reservation succeeds. Once reserved, the
   capacity is fixed: hooks never grow, never throw, and never terminate.
   A caller that must not lose an accepted row preflights
   `trace_capacity_remaining()`; if the table is nonetheless full the row
   is dropped and counted in `snapshot().trace_rows_dropped`, so exhaustion
   is always observable and never silent.
5. Scalar-only use creates no per-OID trace storage. `clear_operations`
   clears recorded rows without freeing prepared capacity.
6. The recorder never mutates logits, selected token IDs, history, KV
   contents, stop reason, queue ordering, or failure semantics. It does
   not add a wait, poll, or synchronization primitive that the existing
   queue/runtime does not already require.

## Attachment, lifetime, and load

1. At most one recorder is attached to one active nonreentrant session, and
   it is borrowed, never owned. The session stores the nullable pointer; it
   has no registry, callback, ownership transfer, queue wrapper, or
   concurrency support, and it never replaces or rejects an attached
   recorder. The caller owns the recorder and must keep it and its supplied
   clock context alive until the session's ordinary destruction and drain
   have completed. A null recorder means fully disabled observation: the
   session reads no clock, allocates nothing for observation, registers
   nothing, adds no wait, and adds no synchronization.
2. `load_tinyllama_session(path, Device&, std::unique_ptr<TokenSelector>,
   InferenceMetrics* = nullptr)` is the only instrumented factory. The
   two-argument factory and the selector overload without a recorded
   argument publish a session with no attached recorder. A null selector is
   still rejected with `std::invalid_argument` before any capability query,
   owner, or clock-dependent work: `load(path, device, nullptr)` is never a
   request to disable instrumentation.
3. The load span is measured exactly once per instrumented call, from that
   call's entry through successful session publication or the failed
   unwind. It covers the null-selector and capability validation, the
   synchronous mapped-model upload, the tokenizer, the formatter, the
   queue, the cache, scratch, and logits owners, and successful publication.
   Outer device or allocator setup and the construction of the
   caller-supplied selector are outside it. An enabled failure records the
   failed load state with no duration, publishes no session, and rethrows
   the original exception category unchanged; an enabled success records the
   interval.
4. `TinyLlamaSession::prepare_operation_trace()` is an explicit,
   caller-controlled configuration operation. It is valid only with an
   attached recorder and before the first request: an absent recorder is
   rejected with `std::invalid_argument`, and a call after request
   preparation is rejected with `std::logic_error`. The reservation reuses
   the request ledger's checked accepted-OID capacity bound, so one prepared
   trace always fits one complete request, and it may report
   `std::overflow_error` (unrepresentable bound or table) or
   `std::bad_alloc` (reservation failure) before any inference work exists.
   Tracing is enabled only after the reservation succeeds, so a failed
   preparation never partially enables it. The call performs no device work
   and adds no wait, and scalar observation never requires it.
5. Request publication is the only point that advances the admitted request
   ordinal. Immediately after a successful request publication the session
   publishes the staged attempt on the recorder, which replaces the admitted
   observation and clears the outgoing operation rows without freeing the
   prepared trace storage. A direct request with no staged attempt
   establishes an empty admitted observation context. Failed drains, refused
   poisoned requests, and failed candidate validation, allocation, or setup
   leave the outgoing admitted observation and its rows untouched; a staged
   attempt that never reached successful publication is reported as its own
   attempt without overwriting the outgoing admitted request.
6. Genuine device timing is unavailable with current CPU host execution,
   with CUDA completion events created `cudaEventDisableTiming`, with ROCm
   completion events created `hipEventDisableTiming`, and with SYCL queues
   constructed `in_order` without profiling. Session attachment and load
   observation therefore record host timing only: they add no vendor header,
   no backend-kind switch, no native timer or event adapter, and make no
   device-duration or kernel-time claim.

## Prefill observation

1. Prefill is one completion-observed span per request that submits a
   prefill forward. The begin instant is read immediately before the
   existing prefill forward submission and the end instant is read
   immediately after the already-required initial final-logits readiness
   wait for that forward's producer, which is the session's existing
   pre-history boundary: the wait precedes history publication and the
   selector. The forward method already contains its own correctness waits,
   so the span deliberately includes them and is never host-enqueue-only
   time; the per-facade host-enqueue sum of the `Submission attribution and
   host enqueue` formula is a separate observation and is neither
   implemented nor required by prefill completion observation.
2. The attribute value is written to the admitted request observation the
   request-publication bridge established for the request that submits the
   forward. Prefill observation adds no ordinal, attempt, or publication
   logic of its own, and it never advances, clears, or rewrites the
   published identity or the operation rows.
3. A prefill forward or readiness-wait failure records the incomplete
   attempt as `failed` with no duration at the catch and rethrows the
   original exception category unchanged. No drain, second wait, callback,
   or altered error path is added, and a failed attempt is never reported
   as a successful completion. Poisoning, retained accepted operations, and
   repeated-wait behavior stay exactly as the existing failure path defines
   them.
4. `require_ready_result` validation, history publication, and the
   full-context return keep their current order, so the recorded span ends
   before history publication. A selector failure after a completed prefill
   is not attributed to prefill: the span is already recorded as a
   successful completion.
5. A request with `max_new_tokens == 0` keeps its early return: no prefill
   forward is submitted, prefill stays `not_run`, and no clock read, submit,
   or wait is added. A full-capacity prompt with a nonzero limit still
   submits exactly one prefill and records its completion-observed span
   before the request returns `context_capacity` with no selected token.
6. Every prefill observation is guarded by the nullable `SessionAccess`
   recorder accessor, so a session with no attached recorder reads no clock
   and records nothing. The borrowed forward result, its readiness
   producer, and the numerical, KV, history, token, stop, scheduling, and
   poisoning behavior are unchanged with observation disabled and enabled.

## Host clock

1. The recorder's only timing source is a supplied monotonic host clock
   supplied as a `HostClock { std::uint64_t (*)(void*) noexcept now; void*
   context; }` pair. The function MUST be `noexcept` and MUST NOT
   allocate. The returned value is interpreted as host nanoseconds since
   an unspecified, monotonic epoch.
2. The default clock is `std::chrono::steady_clock` reported as nanoseconds
   since its epoch. When the caller supplies a clock, the recorder uses
   it for every host instant capture; when the caller does not, the
   recorder falls back to the default monotonic host clock.
3. The recorder never reads a vendor device timestamp, event, profiler, or
   backend-specific queue timer. The supplied host clock is the only
   legitimate timing source for all load, tokenization, prefill, decode,
   and throughput measurements.

## Tokenization

1. The raw and chat entry points measure exactly one interval: the
   `tokenizer.encode` call. `generate_raw` hands its text directly to the
   encoder; `generate_chat` completes the supported chat rendering and then
   encodes the rendered text, so rendering is never inside the measured
   interval. Low-level generation, the post-generation ID conversion, and
   the output text decoding all happen after the measured end.
   `EncodeOptions`, the supported templates, BOS/EOS behavior, encoded
   history, selected tokens, output text, and stop state are unchanged with
   or without a recorder.
2. The raw/chat wrapper stages its attempt at entry, before rendering or
   encoding, with the supplied clock's entry instant and the zero-token
   preprocessing cardinality. The low-level generation stage replaces that
   instant and cardinality for the same attempt, so time-to-first-token
   still starts at low-level generation entry. Staging never clears the
   outgoing admitted request: only a successful request publication
   replaces admitted fields.
3. The measured interval is attached to the attempt after the nested
   `generate_tokens` call returns or throws, and the recorder accepts it
   only while that attempt is the published request. A failed old-request
   drain, a pre-publication validation or setup failure, and a rejected
   preprocessing attempt therefore leave the outgoing request's
   tokenization observation intact and their own interval unattributed. Two
   successive raw/chat calls replace the admitted tokenization observation
   instead of accumulating it, and a direct low-level request never
   inherits a previous raw/chat measurement.
4. A formatter rejection is a failed preprocessing attempt with zero new
   generation: the encoder was never reached, so no tokenization
   observation is offered to the recorder and the admitted tokenization
   field is untouched, which leaves a session that never published at
   `not_run`. An encoder failure is also a failed preprocessing attempt,
   but the encoder itself was observed: the captured interval is offered as
   a failed tokenization observation instead of a fabricated success. A
   preprocessing failure never publishes a request, so the recorder, which
   attaches phase observations only to the published request, writes
   neither failure into the outgoing admitted request: both are reported
   through the attempt outcome and the retained original exception, which
   is rethrown unchanged, and neither fabricates a selector call, positive
   OID, or generated token.
5. A disabled recorder adds no clock read, no allocation, no registration,
   no wait, and no synchronization to either entry point, and leaves every
   numerical, token, history, stop, and failure behavior unchanged.

## Record schema

1. The recorder defines exactly four phases: `load`, `tokenization`,
   `prefill`, and `decode`. Every scalar observation refers to one of
   these phases or to the latest attempt.
2. Each scalar phase observation carries an explicit
   `ObservationState { not_run, succeeded, failed }`, distinguishing
   successful duration from failed observation and from not-run.
3. Each owned operation trace row contains, in this order, the copied
   request ordinal, the positive OID, the phase, an optional configured
   decoder-layer index, the absolute input-position start and run
   length, the host enqueue begin and end instants, the optional first
   wait-observed host instant, the wait state
   `{ not_observed, succeeded, failed }`, and the first retained
   `std::exception_ptr`. No string, tensor/view pointer, borrowed stage
   structure, or callback is retained. Embedding and final LM-head rows
   have no decoder-layer index.
4. The schema supports the canonical attribution windows used by the
   forward path: prefill `[0,R)`, decode `[a,a+1)`, and the final
   LM-head row `[a+R-1,a+R)`. Each accepted session-submitted positive
   OID has exactly one row, located by a sorted exact-OID binary search
   that does not require contiguous OIDs.
5. Repeated wait observations preserve the first successful observation
   or the first retained failure. A later OID success never proves an
   earlier OID succeeded, and a failed wait observation is not proof of
   device termination.

## Generation counts, TTFT, and decode

1. Low-level generation observation starts at `generate_tokens` entry,
   before prompt validation and request provisioning, and stages the
   attempted input cardinality with that entry instant. Validation order is
   unchanged. A request that is rejected, refused, or fails before
   publication is reported as its own attempted request and never
   overwrites the outgoing admitted observation, its completed counters, or
   its operation rows; a failed old-request drain is never attributed to the
   candidate request.
2. The staged attempt is published exactly once, by the session's request
   publication point, and only after the replacement request was actually
   published. That publication replaces the admitted observation, resets its
   tokenization, prefill, decode, host-enqueue, counter, TTFT, and rate
   fields, and clears the outgoing operation rows without freeing the
   prepared trace storage, so a new request never inherits the outgoing
   request's scalars.
3. Time-to-first-token starts at the generation entry instant and ends at
   the first validated committed token, captured after the existing history
   and result pushes. It therefore includes prompt validation, request
   setup, prefill, and the first selector completion, and excludes session
   load, tokenization, chat formatting, and output decoding. With no
   committed token, TTFT is unavailable, never zero.
4. `generated_tokens` advances by exactly one for every actual commit,
   including a terminal EOS, limit, or context token. An invalid or
   out-of-range selector result and a selector exception commit nothing and
   leave the count unchanged for that selection.
5. `decode_forward_count` advances once per successful one-row
   `forward_decode` whose final-logits readiness is observed at the existing
   first producer wait of the selection boundary. It is independent of a
   later selector success, and it adds no wait, poll, or synchronization.
6. `decode_token_count` advances only for committed tokens produced by a
   decode forward; the first prefill-produced token is excluded. Neither
   count is ever inferred from KV growth, a cache-length delta, or a later
   request.
7. Every decode interval opens immediately before its `forward_decode`, and
   the completion-observed decode span for that forward ends at the same
   existing first final-logits readiness wait. A successful decode-produced
   commit adds that decode-start-to-commit elapsed interval to the
   throughput denominator, intentionally including synchronous selector
   time; a prefill-produced commit closes no interval, and a forward or
   readiness failure leaves only a failed, duration-free decode
   observation.
8. Decode throughput is `decode_token_count / seconds(sum of successful
   decode-start-to-corresponding-valid-commit intervals)` and becomes
   available when the request completes successfully. It is unavailable -
   never infinity, NaN, or a fabricated zero - for a zero numerator, a zero
   denominator, a failed, incomplete, or otherwise unobserved request, and
   it makes no claim that unobserved operations succeeded.
9. A retained later failure invalidates the owning request's rate while
   keeping every completed counter inspectable, and an invalidated rate is
   never recomputed as valid. The original exception category, session
   poisoning, and drain behavior are preserved: a failed request rethrows
   the unchanged exception after recording the failed attempt.
10. Terminal counting is independent of cache growth: a terminal token is
    counted as generated without a decode append, and the existing stop
    precedence (`EOS > limit > context`), history, result, KV, and
    stop-reason behavior are unchanged.
11. A null recorder adds no clock read, no allocation, no registration, no
    wait, and no synchronization to low-level generation, and enabled
    observation changes no selected token, commit, KV or history transition,
    stop reason, or exception.

## Frozen formulas

1. The load span covers only the instrumented factory interval, captured
   by the recorder's load hook. Outer device/allocator setup and the
   construction of the caller-supplied selector are outside the measured
   span.
2. The tokenization span covers only the encoder call (`tokenizer.encode`).
   For raw and chat paths, chat rendering, low-level generation, the
   post-generation ID conversion, and the output decoding are all outside
   the measured span.
3. The prefill and decode completion-observed spans end at their existing
   final-logits readiness waits. The forward method already contains
   waits; its call-return wall time is not host-enqueue-only time. Failed
   spans are incomplete and never successful.
4. Host enqueue is the sum of individual facade-call intervals that
   enqueue work for a phase. Runtime blocking inside a facade is part of
   the host-enqueue total. Later waits, selector work, and entire
   forward spans are not.
5. Time-to-first-token starts at low-level generation entry before
   validation and setup and ends at the first validated committed
   output. With no committed token, TTFT is unavailable, never zero.
6. The generated token count includes committed EOS and terminal
   limit/context tokens. The decode-forward count advances on each
   successful one-row `forward_decode` whose final-logits readiness is
   observed, independently of later selector success. The decode-token
   count advances only for committed tokens produced from decode
   forwards; the first prefill-produced token is excluded. Counts are not
   inferred from KV growth.
7. Decode throughput is `decode_token_count / seconds(sum of successful
   decode-start-to-corresponding-valid-commit intervals)`. The session
   decodes one token at a time, so exactly one decode interval is open at
   a time: `record_decode_interval` opens it immediately before the
   decode forward and the matching decode-produced commit closes it,
   intentionally including synchronous selection time. The rate is
   computed once the request completes successfully, and is unavailable —
   never infinity, NaN, or a fabricated zero — for zero numerator, zero
   denominator, failed, incomplete, or rate-invalidated requests. A later
   retained failure invalidates the request rate without erasing
   completed counters, and an already invalidated rate is never
   recomputed as valid.
8. Stop precedence is `EOS > limit > context` and terminal tokens may be
   counted as generated without growing KV. The recorder preserves the
   reused `GenerationStopReason` and exposes it together with an explicit
   `stop_reason_valid` flag, which stays false until a successful
   `end_generation` records the reason, so no enumerator is ever
   fabricated by default construction. The recorder does not duplicate
   session stop policy.

## Submission attribution and host enqueue

1. Host enqueue is the sum of the individual facade-call intervals that
   enqueue work for a phase. The session measures it in the existing
   submission seam: with an attached recorder the supplied monotonic clock is
   read immediately before and immediately after the single
   `operation(session.queue())` invocation, and only that interval is
   accumulated. Runtime blocking inside the facade is part of the host-enqueue
   total; later producer waits, selector work, enqueue-independent host work,
   and an entire forward wall span are not. A null recorder reads no clock,
   constructs no record, registers nothing, allocates nothing, and performs no
   observation work at all.
2. The accumulated sums are the admitted request's
   `admitted.prefill_enqueue` and `admitted.decode_enqueue`. A staged but
   unpublished attempt never overwrites the outgoing admitted request's sums,
   and a published request starts from zero. Scalar-only mode, that is an
   attached recorder without explicitly prepared trace storage, records these
   sums and creates no per-OID row; `prepare_trace` is the only storage for
   rows, so no accepted observation is ever allocated during inference.
3. With tracing enabled, exactly one owned row is appended for each positive
   OID accepted from a session submission, in acceptance order, and located by
   the recorder's sorted exact-OID lookup. Negative admission results remain
   rejected submissions with their existing translation, poisoning, and drain
   behavior: they become no trace row and contribute no enqueue time. The
   accepted-OID ledger remains the correctness ledger; rows never drive
   scheduling, workspace lifetime, queue order, or publication.
4. The session-scoped operation context copied into each row is taken from the
   forward entry points and their decoder-layer loop variables, never from a
   timing sibling: the phase (`prefill` or `decode`), the optional configured
   decoder-layer index, the absolute input-position start, and the run length.
   The canonical windows are prefill whole-run `[0,R)` for embedding and the
   final normalization, prefill final LM head `[R-1,R)`, decode whole-run
   `[a,a+1)`, and the decode final LM head `[a,a+1)`, which is that run's own
   final row. A submission made inside a decoder-layer iteration carries that
   configured layer index; embedding, final normalization, and final LM-head
   operations carry none. A session submission outside any forward scope keeps
   the default attribution (`load` phase, no decoder layer, empty `[0,0)`
   window) instead of borrowing a stale phase, layer, or window.
5. Remaining prepared record capacity is preflighted with the bounded ledger
   before the facade is invoked, so an accepted submission whose observation
   could not be retained is refused with the same checked-bound category
   instead of being dropped, grown, or silently lost after acceptance; after a
   positive acceptance the append neither allocates nor throws. Rows of an
   outgoing request survive a failed drain or a failed candidate validation,
   setup, or preprocessing and are cleared or replaced only after a successful
   publication, so a later request never retroactively establishes an earlier
   success.
6. Only session-submitted positive OIDs are observed. In the current forward
   path those submissions are the embedding, the final normalization, and the
   final LM-head projection of one prefill or one decode. Selector-internal
   operations issued directly to an exposed queue, host-side transfers, and
   the private decoder-layer stage submissions of the session's own queue
   remain opaque; no generic queue instrumentation, queue registration, event,
   native timer, mutex, wait, or synchronization is added for observation.
7. Every value here is a host facade-call observation, never device or kernel
   execution time. Genuine device duration is unavailable under current CPU
   host execution, under CUDA completion events created
   `cudaEventDisableTiming`, under ROCm completion events created
   `hipEventDisableTiming`, and under SYCL queues constructed `in_order`
   without profiling. No optional native timer, adapter, or universal timing
   interface is introduced, and no enqueue sum may be reported as device time.

## Wait observations and retained failure

1. The recorder observes exactly the waits the session already performs: the
   session-controlled wait for one accepted producer, and every drain of the
   accepted-OID ledger, including the drain a failed wait triggers and the
   drain performed during session destruction. Observation happens
   immediately at that existing wait's return or catch. The hooks add no
   wait, poll, retry, event query, or synchronization primitive that the
   existing queue/runtime does not already require, and they leave existing
   per-producer waits inside a forward stage, duplicate selector readiness
   waits, and generic queue instrumentation unchanged.
2. A completion-observed wait outcome is host observation of the existing
   wait result. It is not a native device timestamp, not device duration, and
   not proof that all predecessor work is complete: a successful observation
   reports that the queue reported completion for that exact operation, and a
   failed observation reports only that this wait retained that failure.
3. For each owned positive OID at most one outcome is retained: the first
   successful observation or the first retained failure, with its supplied
   host instant and the copied request context captured at enqueue. Repeated
   waits, poison/drain recursion, and destructor cleanup preserve that first
   outcome and its retained `std::exception_ptr` instead of adding a row,
   rewriting the first instant, or selecting a different error. A later
   successful OID never proves that an earlier OID succeeded.
4. Failure ownership is established from the existing accepted-OID ledger,
   never from the presence of a trace row. A failed wait is observed
   immediately, and the original exception is saved before any subsequent
   drain, poisoning, or cleanup activity, so the existing error path keeps
   its original exception, first-error selection, poisoning, and rethrow
   behavior. Only a failure of the current admitted request's accepted
   ledger passes that verified ordinal to the recorder, which invalidates the
   owning request's rate while retaining its completed counters; an
   invalidated rate is never recomputed as valid.
5. An unowned failure never invalidates an unrelated request's rate, even
   when a trace row exists for the OID. Unknown, unregistered, foreign-queue,
   and selector-internal OIDs submitted directly to the exposed queue
   acquire no row and no copied context: the trace covers session-submitted
   positive OIDs only, their wait attribution comes from that exact ledger
   entry, and no generic queue instrumentation is added.
6. A failed wait observation is not a device-terminality proof. Registry and
   quarantine rules, resource destruction, and the queue's own retained
   failure semantics stay untouched.
7. Scalar-only mode needs no wait hook for a successful wait: it reads no
   clock and registers no OID. A retained failure in scalar-only mode
   invalidates the verified owning request's rate without a clock read and
   without per-OID registration. With tracing prepared, a wait observation
   may read the supplied host clock and update the already-owned row of that
   exact OID. A null recorder performs no observation work at all.
8. Genuine device timing stays unavailable for wait observations under the
   same limitations as every other observation: the CPU backend has no
   device clock, CUDA completion events are created `cudaEventDisableTiming`,
   ROCm completion events are created `hipEventDisableTiming`, and SYCL queues
   are constructed `in_order` without profiling. No optional native timer,
   event adapter, or universal timing adapter is introduced here.

## Honest backend limitations

1. **CPU.** The CPU backend has no device clock. Every measurement is a
   host-clock observation of the queue/runtime. No device duration is
   available, and the recorder does not synthesize one.
2. **CUDA.** CUDA completion events are created with
   `cudaEventDisableTiming`. Event-based device duration is unavailable
   by construction. The recorder reports host observations only; no
   native CUDA timer adapter is introduced here.
3. **ROCm.** ROCm completion events are created with
   `hipEventDisableTiming`. Event-based device duration is unavailable
   by construction. The recorder reports host observations only; no
   native HIP timer adapter is introduced here.
4. **SYCL.** SYCL queues are constructed `in_order` without profiling.
   The SYCL backend therefore exposes no kernel duration. The recorder
   reports host observations only; no native SYCL profiling adapter is
   introduced here.
5. Disabled observation mode adds no clocks, no per-operation allocation,
   no registration, no wait, and no synchronization that is not already
   required by the existing queue/runtime.
6. The recorder never labels a host duration as kernel or device time,
   never claims a throughput floor, and never substitutes host or
   elementwise work for unsupported native matrix evidence.

## CLI operation trace presentation

1. The CLI accepts a value-free `--trace` option. It rejects repeated,
   value-bearing, and unknown forms as usage/input errors (status 2).
   The option is independent of `--metrics`; trace alone creates exactly one
   scalar `InferenceMetrics` recorder, prepares its bounded operation table,
   and emits no metrics summary. When both options are present they share that
   one recorder and produce their independent presentations.
2. Trace preparation is performed explicitly after successful instrumented
   session load and before the first raw or chat generation request. It is
   outside inference and performs no device work or wait. A reservation or
   recorder-configuration failure is a setup failure (status 3), leaves
   tracing disabled, and never falls back to an untraced generation. With no
   observation option, the CLI creates no recorder or trace storage.
3. The recorder remains alive through ordinary session destruction. The CLI
   releases the session before formatting rows so its existing destructor drain
   can update final wait observations; reporting never adds a wait, poll, or
   synchronization. Trace formatting is secondary: it cannot replace the
   primary input, load, or execution status or diagnostic.
4. Each collected row is emitted on stderr in accepted positive-OID order as a
   bounded human-readable line containing the request ordinal, positive OID,
   phase, decoder layer (or `none`), absolute input-position start and run
   length, `host_enqueue` elapsed nanoseconds, and `device_time=unavailable`.
   When a wait was actually observed, the line also contains
   `completion_observed` elapsed host nanoseconds measured from enqueue begin;
   a pending row has no fabricated completion time. The wait state is exactly
   `not_observed`, `succeeded`, or `failed`.
5. The presentation reports only rows currently owned by the recorder.
   Unknown, rejected, and unregistered OIDs do not receive fabricated
   identity. A repeated wait retains the first observed success or failure,
   and a failed wait is reported as a failed observation rather than proof of
   native completion. Generated stdout bytes, normal stop reasons, primary
   diagnostics, and statuses 0, 2, 3, and 4 are unchanged.
