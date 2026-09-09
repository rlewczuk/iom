# Migrate backend OID conformance and coexistence tests

**Order:** 08
**Priority:** P0 — completes the backend-facing OID compatibility cutover before any ADD implementation work.
**Blocked by:** `03-migrate-cpu-copy`, `04-migrate-cuda-rocm-copy`, `05-migrate-sycl-copy`, `06-migrate-ttnn-copy`, `07-migrate-core-oid-tests`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

The shared backend conformance harness, every backend conformance driver, and the backend coexistence test exercise the signed `iom::oid` result contract end to end. They decode accepted tokens with the 55-bit sequence layout, inspect negative `OidError` results instead of expecting synchronous exceptions, and prove that invalid waits, pre-acceptance failures, accepted ordering, repeat waits, and retained asynchronous failures have the specified behavior on CPU, CUDA, ROCm, SYCL, TTNN, and mixed-backend processes.

## Scope

- Migrate all backend conformance fakes and scenarios from throwing OID-returning calls to checking negative `OidError` values for synchronous rejection and positive values for accepted submissions. The public OID-returning calls are non-throwing facades; `wait` remains the throwing operation for invalid values and retained asynchronous failures.
- Change the shared token helpers from a 56-bit sequence to the exact 55-bit sequence encoding: queue IDs `1..255` occupy bits `55..62`, sequence values are `1..2^55-1`, and zero is never a token.
- Cover waits on each of: a negative OID/error result, zero, a foreign queue token, a future token, a skipped sequence, a reserved-but-unsubmitted sequence, and an otherwise unsubmitted sequence. Each must immediately throw `std::invalid_argument` without completion, output/storage effects, or sequence/token side effects.
- Exercise the migrated `seek_next_sequence` fake seam exactly as a regression: advance past at least one sequence, submit a later sequence, complete that later sequence, then verify that the skipped value is still immediately rejected. Do not make skipped values valid merely because a later sequence completed.
- Preserve accepted in-order completion, repeatable successful waits, and repeatable rethrow of a retained post-acceptance asynchronous failure. Preserve queue ownership, queue identity, caller-owned storage, and coexistence behavior while changing only the OID contract assertions.
- Verify backend pre-acceptance failures through deterministic fakes or each backend's existing fault seams. Map invalid caller input to `InvalidArgument`, unavailable or unsupported work to `Unsupported`, checked overflow or sequence exhaustion to `Overflow`, allocation or bounded-resource failure to `ResourceExhausted`, pre-acceptance backend/runtime failure to `DeviceError`, and an otherwise unclassified internal failure to `InternalError`.
- Keep the preliminary all-255 queue-ID and sequence-exhaustion coverage intact, with the 55-bit boundary replacing the old 56-bit boundary. Keep non-ADD unimplemented operations (`mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`, and any still-unimplemented preliminary `add` expectation) at negative `OidError::Unsupported`; ADD-positive behavior is owned by later ADD tasks.
- Migrate every OID assertion in the coexistence cases around lines 399–688, including interleaved accepted copies, foreign-device rejection, queue-ID reuse, stale/unsubmitted waits, and repeat waits. Do not add generation bits or require distinguishing numerically identical tokens after legitimate queue-ID and sequence reuse.

## Implementation references

- **Modify:** `test/backend/backend_conformance_common.hpp` — `expect_repeated_runtime_failure`, `kTokenSequenceBits`, `kTokenSequenceMask`, `token_queue`, `token_sequence`, and the `DeviceOps` signature assertions; use the public OID helpers and 55-bit encoding without introducing backend switches.
- **Modify:** `test/backend/backend_conformance_other.hpp` — `DeferredCopyQueue`, `InstrumentedQueue`, `run_lifetime_conformance`, `run_compute_capability_conformance`, and `run_backend_conformance`; make fake failures return signed OID errors, retain post-acceptance failures, and add the complete invalid-wait and skipped-sequence coverage.
- **Modify:** `test/cpu/test_cpu_conformance.cpp` — CPU conformance driver and its shared-suite/capability cases; retain host allocator and preliminary storage/copy coverage while asserting signed results.
- **Modify:** `test/cuda/test_cuda_conformance.cpp` — CUDA conformance driver, submission fault cases, lifetime cases, and shared-suite/capability cases; qualify focused execution on a CUDA-capable host.
- **Modify:** `test/rocm/test_rocm_conformance.cpp` — ROCm conformance driver, watchdog/pre-acceptance and post-acceptance fault cases, lifetime cases, and shared-suite/capability cases; qualify focused execution on a ROCm-capable host.
- **Modify:** `test/sycl/test_sycl_conformance.cpp` — SYCL conformance driver, pre/post-enqueue fault cases, lifetime cases, and shared-suite/capability cases; qualify focused execution on an eligible SYCL accelerator host.
- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — TTNN conformance driver, retained batch-failure and lifetime cases, and shared-suite/capability cases; qualify focused execution on a TTNN-capable host.
- **Modify:** `test/backend/test_backend_coexistence.cpp` — cases around lines 399–688 (`enabled backends interleave`, `queues reject views from another device`, `second devices report their own ordinal`, and `queue ids release, reuse, and stay unique`) plus the remaining coexistence OID assertions; preserve mixed-backend queue-ID uniqueness, accepted ordering, repeat waits, foreign/stale/unsubmitted rejection, and reuse semantics.
- **Read:** `test/test_iom.cpp` — preliminary OID tests for all-255 queue IDs, sequence exhaustion, `seek_next_sequence`, invalid waits, and retained failures; this task consumes the migrated contract and must not remove or duplicate their ownership of core boundary coverage.
- **Read:** `test/CMakeLists.txt` — existing CTest target names and accelerator gating; no build-system change is required by this task.

## Requirements

- Use `iom::oid` as the result type in every migrated conformance assertion. A negative result is classified with `iom::oid_is_error`/the corresponding `OidError`, zero is invalid, and only a positive result is passed to `wait` as an accepted token.
- Decode queue and sequence fields using the public 55-bit contract. Assert that every accepted token has a nonzero sequence and that queue IDs remain representable through ID 255; do not retain any 56-bit mask or shift.
- Test invalid waits on both the queue that owns the relevant numeric ID and a different queue. Negative, zero, foreign, future, skipped, reserved-but-unsubmitted, and otherwise unsubmitted values must all be rejected immediately with `std::invalid_argument`, including after a later sequence has completed.
- Prove rejection timing: before and after each invalid call, the fake's submitted/completed records, next sequence, output bytes, and relevant allocator/storage observations are unchanged. A rejected OID result from an operation must likewise consume no sequence and perform no effect.
- Exercise every synchronous OID error category that the shared/backend seams can deterministically produce and assert the exact negative category. No synchronous exception may escape an OID-returning operation; only a post-acceptance failure is retained for `wait`.
- Prove accepted queue semantics: submissions receive positive tokens, complete in order even when a later sequence is completed first, successful waits are idempotent, and a retained asynchronous exception is rethrown with the same failure on repeated waits.
- In the `seek_next_sequence` regression, reserve a gap, submit and complete a later sequence, and then call `wait` on a token for the skipped sequence. The call must still immediately throw `std::invalid_argument`; it must not block, complete, or become valid through in-order completion.
- Keep non-ADD `Unsupported` expectations in all backend drivers and shared capability scenarios. Do not turn this migration task into ADD support or add a capability query/API.
- Keep coexistence's legitimate queue-ID reuse semantics: after a queue is destroyed and its ID and sequence are legitimately reused, numerically identical tokens are not required to be distinguishable. Continue rejecting sequences that the recreated queue did not submit.
- Do not add generation bits, token tags, public knobs, backend-specific OID APIs, or changes to production implementation, documentation, CMake, or unrelated tests.

## Non-goals

- Implementing ADD arithmetic, broadcast behavior, numeric codecs, TTNN storage expansion, or any other positive ADD behavior.
- Redesigning the OID encoding, adding generation information, or changing queue-ID allocation and reuse policy.
- Replacing the common facade, backend copy implementations, or core OID tests owned by the five blocking tasks.
- Changing non-OID `Device` queries, host-transfer semantics, allocator behavior, backend capability APIs, or coexistence architecture.
- Adding new CTest targets or running project-wide validation in this task.

## Acceptance criteria

- [ ] `backend_conformance_common.hpp` and `backend_conformance_other.hpp` contain no 56-bit token decoding or synchronous-exception expectations for OID-returning operations; their fakes and shared cases assert signed results, exact error categories, and repeatable retained failures.
- [ ] CPU, CUDA, ROCm, SYCL, and TTNN conformance drivers all compile against the migrated facade and independently cover pre-acceptance error mapping, invalid waits (negative/zero/foreign/future/skipped/reserved-but-unsubmitted/otherwise unsubmitted), accepted ordering, repeat waits, and retained asynchronous failures.
- [ ] `test_backend_coexistence.cpp` preserves the interleaving, foreign-device, ordinal, queue-ID reuse, and later coexistence cases with positive-token assertions and signed rejection assertions; its reuse test does not claim generation-based stale-token detection.
- [ ] The focused `seek_next_sequence` regression submits and completes a later sequence before proving the skipped token remains immediately invalid, and rejection leaves sequence/completion/output state unchanged.
- [ ] All-255 queue-ID and sequence-exhaustion tests remain owned by and compatible with the preliminary OID suite, now using the 55-bit boundary; every non-ADD unsupported expectation remains negative `OidError::Unsupported`.
- [ ] Focused CTest coverage is identified for the core host target `iom_tests`, CPU host target `iom_backend_conformance_cpu_tests`, CUDA accelerator-host target `iom_cuda_conformance_tests`, ROCm accelerator-host target `iom_rocm_conformance_tests`, SYCL accelerator-host target `iom_sycl_conformance_tests`, TTNN accelerator-host target `iom_ttnn_conformance_tests`, and mixed enabled-accelerator host target `iom_backend_coexistence_tests`.

## Verification

No gates are run for this mini-spec. The main integration pass may run these focused targets from an appropriately configured build directory:

- `ctest --test-dir <build> -R '^iom_tests$'` — core host OID compatibility and preliminary boundary coverage.
- `ctest --test-dir <build> -R '^iom_backend_conformance_cpu_tests$'` — CPU host conformance.
- `ctest --test-dir <build> -R '^iom_cuda_conformance_tests$'` — CUDA accelerator host (CUDA runtime/device required).
- `ctest --test-dir <build> -R '^iom_rocm_conformance_tests$'` — ROCm accelerator host (ROCm runtime/device required).
- `ctest --test-dir <build> -R '^iom_sycl_conformance_tests$'` — SYCL accelerator host (eligible SYCL device/runtime required).
- `ctest --test-dir <build> -R '^iom_ttnn_conformance_tests$'` — TTNN accelerator host (TTNN device/runtime required).
- `ctest --test-dir <build> -R '^iom_backend_coexistence_tests$'` — one process with every enabled accelerator backend and the CPU participant.
