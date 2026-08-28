# Cut over DeviceOps to views and implement queue tokens

**Order:** 04
**Priority:** P0 — the public operand contract and process-wide queue identity gate every concrete backend.
**Blocked by:** `01-repair-safetensors-dtypes`, `03-tensor-leading-views`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

Every `DeviceOps` tensor operand is a `TensorView`, current callers use stable owner views, and the common base provides process-unique queue IDs, monotonic submission tokens, repeatable waits, and deterministic validation for live queues.

## Scope

- Perform the source-breaking `Tensor`-to-`TensorView` operand cutover and migrate all in-repository callers.
- Centralize queue-ID leasing, sequence allocation, token validation, completion bookkeeping, and repeated-wait behavior in the common `DeviceOps` base.
- Use a deterministic deferred fake queue to verify the common machinery before concrete backends exist.

## Implementation references

- **Modify:** `include/iom/iom.hpp` — `DeviceOps`; retain `oid` as `std::uint64_t`, change every operand signature to views, make the SILU input const, and expose only the protected machinery concrete queues require.
- **Modify:** `src/iom.cpp` — implement the process-wide queue-ID pool and common token/completion behavior.
- **Modify:** `src/llama.cpp` — pass `tensor.view()` to every operation and replace both nonexistent `set_float` calls with byte spans passed to synchronous host transfer.
- **Modify:** `test/test_iom.cpp` — add the deterministic queue fake, token-boundary tests, signature compile checks, and caller cutover coverage.
- **Read:** `include/iom/llama.hpp` — preserve `Block` and model ownership signatures; only `DeviceOps` operands are cut over.

## Requirements

- Public signatures are exactly view-based for `copy`, `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`; inputs are const views and outputs are mutable views. `silu` takes a const input view even when callers deliberately pass the same window as output.
- `DeviceOps` leases one process-unique queue ID in `[1,255]` during construction and releases it during destruction. Construction throws `std::runtime_error` when all IDs are live. Construction and destruction from different threads must not race or duplicate an ID.
- Queue ID zero and sequence zero are invalid. An `oid` encodes `(queue_id << 56) | sequence`, where each queue's first successful submission is sequence one and the sequence occupies the low 56 bits.
- Validation or synchronous pre-queue failure consumes no sequence. Every successfully queued operation, including an identical-window copy no-op, consumes exactly one sequence.
- Submission after sequence `2^56 - 1` throws `std::overflow_error` before queuing. Provide a protected test seam that lets the deterministic fake begin near this limit without adding public queue controls.
- For a live queue, `wait` rejects zero, a live foreign queue ID, sequence zero, and a sequence never submitted by that queue with `std::invalid_argument`.
- A successful wait is idempotent and may use a completed watermark. An asynchronous failure is retained by sequence and the stored exception is rethrown on every later wait for that sequence. In-order completion of sequence `N` also completes earlier submissions while preserving their individual result.
- Releasing a queue permits its eight-bit ID to be reused. Per the direct stale-token decision, using an `oid` after its originating queue is destroyed is a caller error; the implementation does not add generation bits and is not required to distinguish a bit-identical stale token after ID reuse.
- The common pool is the only global queue state. It must not select a backend, active device, or runtime context.
- Calls on one queue remain caller-serialized. The base does not add internal concurrent-submission support, implicit cross-queue dependencies, destruction waits, or cancellation.
- In `src/llama.cpp`, use `std::as_bytes` over the existing float buffer for `copy_from_host`; do not add conversion or a compatibility `set_float` method.
- Remove all old `Tensor` operand overloads and aliases in the same cutover.

## Non-goals

- Implementing mathematical compute kernels or changing their contracts.
- A backend registry, global active backend, cross-queue event API, or wider `oid`.
- Detecting stale tokens after queue-ID reuse.
- Implicit waiting during queue destruction.

## Acceptance criteria

- [ ] Compile checks prove all seven operation families accept the specified view constness and no tensor overload remains.
- [ ] Every model operation passes stable `Tensor::view()` references, and RoPE initialization uses exact-size typed host bytes.
- [ ] Concurrent construction yields 255 unique live IDs; the 256th fails; destruction makes an ID reusable.
- [ ] Token bit layout, monotonic sequence allocation, no-consumption failures, and the 56-bit exhaustion boundary are deterministic.
- [ ] Success and failure waits are idempotent, in-order completion covers earlier sequences, and zero/unknown/live-foreign tokens fail.
- [ ] Fake queue destruction performs no wait or cancellation.
- [ ] Core-plus-current callers configure, compile, and run without the stale enum or `set_float` failures.

## Verification

- `cmake -S . -B build/core -DBUILD_TESTING=ON`
- `cmake --build build/core --target iom_tests`
- `./build/core/test/iom_tests --test-case="DeviceOps queue*,DeviceOps view signatures*"`
