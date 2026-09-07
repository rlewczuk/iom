# CUDA/ROCm host transfers synchronize the same stream twice

**Order:** 09
**Priority:** P1
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** PF-001
**Review area:** Performance
**Review severity:** medium
**Review verification:** strongly-supported, confidence 96
**Review scope:** whole-codebase
**Backend scope:** cuda, rocm
**Location:** `src/shared/standard_tiled_copy.inl:647-705` (`iom::detail::synchronous_transfer_impl`); `src/shared/transfer_pool.hpp:13-31` (`TransferStreamPool::Scope::~Scope`)

## Outcome

A successful CUDA or ROCm host transfer performs exactly one stream synchronization before its stream returns to the pool: the pooled `Scope` tracks explicit successful synchronization and skips the duplicate wait in its destructor, while throw/poisoned paths still synchronize (or drop safely) before reuse or destruction.

## Current problem

A completed synchronous host transfer must return with its transfer stream safe for reuse, while avoiding redundant critical-path device/stream waits.

`iom::detail::synchronous_transfer_impl` (`src/shared/standard_tiled_copy.inl:647-705`) launches the HtoD/transform path and explicitly calls `Policy::synchronize_stream(stream)` on the successful path (`:686-688`); only afterward does the DtoH copy occur for downloads (`:689-694`). On scope exit, `TransferStreamPool::Scope::~Scope` (`src/shared/transfer_pool.hpp:17-27`) unconditionally calls `Policy::synchronize_stream_noexcept(stream_)` before either destroying or releasing the stream. CUDA and ROCm implement both hooks as `cudaStreamSynchronize`/`hipStreamSynchronize` (`src/cuda/copy.hpp:71-80`, `src/rocm/copy.hpp:184-193`). Every successful CUDA/ROCm host upload/download therefore performs two stream-synchronization API calls; the destructor's second call is idle on the normal path. The only production callers of this pool are CUDA/ROCm `region_from_host`/`region_to_host` (`src/cuda/copy.cu:363-378`, `src/rocm/copy.hip:360-375`). Impact: a redundant runtime call and host-side latency on every accelerator host transfer, likely largest on small transfers; the wall-time fraction was not measured in the review.

## Scope

- Give `TransferStreamPool::Scope` an explicit successfully-synchronized state (or an equivalent `release_after_sync` operation).
- In `synchronous_transfer_impl`, mark the Scope immediately after the successful explicit synchronization.
- Make the destructor synchronize only when that state is not set, preserving synchronization before reuse/destruction on all throw and poisoned paths.

## Implementation references

- **Modify:** `src/shared/transfer_pool.hpp` — `TransferStreamPool::Scope` (`:13-31`); owns the release contract for pooled streams.
- **Modify:** `src/shared/standard_tiled_copy.inl` — `synchronous_transfer_impl` (`:647-705`); owns the successful-path explicit synchronization after which the Scope can be marked.
- **Read:** `src/cuda/copy.hpp:71-80`, `src/rocm/copy.hpp:184-193` — the policy hooks both paths invoke; `src/cuda/copy.cu:363-378`, `src/rocm/copy.hip:360-375` — the only production callers.
- **Tests:** `test/cuda/test_cuda_smoke.cpp:388-507`, `test/rocm/test_rocm_smoke.cpp:192-236` — host-transfer smoke scenarios to rerun unchanged.

## Requirements

- One stream synchronization per successful CUDA/ROCm host transfer; the destructor wait is skipped only for the explicit-successful synchronization state.
- All throw/poisoned paths must still synchronize (or drop) the stream before reuse/destruction exactly as today; pooled streams remain safe for reuse.
- Do not change staging layout, kernel launch count, host-buffer pinning, queue-copy events, host-transfer ordering, or logical bytes/error behavior.

## Non-goals

- No new pool or alternate synchronization mechanism; no changes to SYCL or TTNN transfer paths; no change to asynchronous queue semantics.

## Acceptance criteria

- [ ] A policy-call trace of successful CUDA and ROCm uploads and downloads reports exactly one stream synchronization per transfer (down from two), across small and large transfers and repeated pool reuse.
- [ ] Thrown/poisoned failed transfers still synchronize or drop safely before the stream is reused or destroyed, and existing storage/concurrency smoke checks remain green.

## Verification

`actual validation: none (read-only review); proposed gates below`.

- CUDA and ROCm (remote-host work per remote-development; requires the corresponding GPU hosts, configured separately): instrument the policy synchronization hooks, configure with testing enabled, build, and run `ctest --test-dir <build> -R 'iom_(cuda|rocm)_smoke_tests' --output-on-failure`.
- Acceptance observation: one synchronization API call per successful transfer, unchanged logical bytes/error behavior, and reduced or non-increased median transfer latency on a representative small-transfer loop.

- `ctest --test-dir <build> -R 'iom_(cuda|rocm)_smoke_tests' --output-on-failure`