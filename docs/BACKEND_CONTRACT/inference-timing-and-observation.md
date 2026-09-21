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
