# Grow pooled copy metadata without synchronizing the whole queue

**Order:** 08
**Priority:** P1 — remove redundant queue-wide waits from metadata slot growth
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase review of main at fb88959fb884b9bb495e055b9fdd0756e430dbde (clean working tree)`
**Finding:** `PF-002`
**Review area:** Performance
**Review severity:** medium
**Review verification:** strongly-supported, confidence 90
**Review scope:** whole-codebase
**Backend scope:** cuda; rocm; sycl
**Location:** `src/shared/metadata_slot_pool.hpp:62-91` (`MetadataSlotPool::ensure_slot_capacity`), `src/sycl/copy.cpp:103-145` (`SyclMetadataSlotPool::ensure_slot_capacity`), called from `src/shared/gpu_queue.hpp:198-228` and `src/sycl/copy.cpp:539-568`

## Outcome

Replacing a metadata slot uses its existing slot-local completion guarantee and does not add an explicit whole-stream or whole-queue wait. High-rank pooled bursts preserve metadata lifetime and copy correctness while avoiding queue-drain latency during capacity growth.

## Current problem

The invariant is that a metadata slot can be replaced only after its prior operation is complete, but replacement must not synchronize unrelated queue work once slot-local completion is already established. `src/shared/metadata_slot_pool.hpp:62-91` unconditionally calls `Policy::synchronize_stream(stream)` before allocating and freeing a CUDA/ROCm slot. `src/sycl/copy.cpp:103-145` similarly calls `queue_.wait_and_throw()` before replacement. Queue callbacks execute synchronously during submission through `StagedWorker::submit_copy` (`include/iom/iom.hpp:66-107`; `src/shared/gpu_queue.hpp:198-228`; `src/sycl/copy.cpp:539-568`), placing each growth wait directly on `queue->copy`.

The existing event ring and SYCL fence state wait their own event before releasing metadata (`src/shared/event_ring.hpp:146-186`; `src/sycl/copy.cpp:200-220,230-245`), so the slot being grown is already retired and no longer referenced by its prior operation. The redundant queue-wide wait drains unrelated work and serializes rank-11-or-higher pooled bursts. The corrected boundary is that rank-10 remains the inline control while rank-11 and above exercise pooled metadata growth.

## Scope

- Remove explicit stream/queue-wide synchronization from CUDA/ROCm and SYCL metadata capacity growth.
- Retain the existing slot-local event/fence completion guarantee before replacing or freeing a slot.
- If a backend allocator requires deferred reclamation, retire the old allocation against the already-completed slot event or existing cleanup mechanism, without introducing another hot-path wait.
- Preserve the 16-slot policy, inline threshold, metadata format, event/error behavior, and TTNN path.

## Implementation references

- **Modify:** `src/shared/metadata_slot_pool.hpp:62-91` — `MetadataSlotPool::ensure_slot_capacity`; remove `Policy::synchronize_stream(stream)` and use slot-local retirement.
- **Modify:** `src/sycl/copy.cpp:103-145` — `SyclMetadataSlotPool::ensure_slot_capacity`; remove `queue_.wait_and_throw()` and preserve fence-backed reclamation.
- **Read:** `src/shared/event_ring.hpp:146-186` and `src/sycl/copy.cpp:200-245` — existing event/fence completion and metadata release protocol.
- **Read:** `src/shared/gpu_queue.hpp:198-228` and `src/sycl/copy.cpp:539-568` — synchronous submission ordering and slot acquisition context.
- **Tests:** CUDA, ROCm, and SYCL smoke/conformance coverage; add a focused rank-10 inline control and rank-11-or-higher pooled no-wait burst with wait-count instrumentation.

## Requirements

- After a slot's associated event/fence has completed, growth/replacement must not call explicit whole-stream or whole-queue synchronize/wait.
- Preserve a slot-local completion proof before freeing/replacing old metadata; allocator reclamation must remain safe even if the allocator free itself is deferred.
- Verify the corrected boundary: rank-10 remains an inline control, while rank-11-or-higher enters the pooled metadata path.
- Keep capacity growth outside the steady-state path without adding another cache, allocation layer, or synchronization.
- Preserve copy results, error propagation, metadata lifetime, and all backend allocator APIs.

## Non-goals

- Do not change the 16-slot capacity policy, inline metadata threshold, kernel metadata format, backend allocator APIs, event/error semantics, or TTNN native path.
- Do not claim that vendor allocator frees are free of runtime cost; remove only the redundant explicit queue drain.
- Do not add a profiler-only behavior or alter ordinary token waits.

## Acceptance criteria

- [ ] A rank-10 inline control performs no metadata-pool growth, while rank-11-or-higher pooled copies grow slots without an explicit stream/queue wait after slot-local completion.
- [ ] A burst of at least 32 pooled copies without intermediate waits records zero capacity-growth `cudaStreamSynchronize`, `hipStreamSynchronize`, or SYCL `wait_and_throw` calls.
- [ ] The burst preserves exact logical/canonical storage results, metadata lifetime, error behavior, and safe allocator reclamation; the final token wait remains sufficient.
- [ ] Submit p50/p99 no longer includes an avoidable prior-queue drain, while first-use allocator cost may remain observable.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — local common/CPU gates remain green.
- `.agents/skills/remote-development/scripts/remote-sync cuda 08-PF-002-metadata-pool-growth-without-queue-sync`
- `.agents/skills/remote-development/scripts/remote-exec cuda 08-PF-002-metadata-pool-growth-without-queue-sync 'cmake -S . -B build-cuda -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-cuda --target iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-cuda --output-on-failure -R "iom_cuda_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — CUDA smoke, conformance, and coexistence pass.
- `.agents/skills/remote-development/scripts/remote-sync rocm 08-PF-002-metadata-pool-growth-without-queue-sync`
- `.agents/skills/remote-development/scripts/remote-exec rocm 08-PF-002-metadata-pool-growth-without-queue-sync 'cmake -S . -B build-rocm -DCUDA_ENABLED=OFF -DROCM_ENABLED=ON -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-rocm --target iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-rocm --output-on-failure -R "iom_rocm_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — ROCm smoke, conformance, and coexistence pass.
- `.agents/skills/remote-development/scripts/remote-sync sycl 08-PF-002-metadata-pool-growth-without-queue-sync`
- `.agents/skills/remote-development/scripts/remote-exec sycl 08-PF-002-metadata-pool-growth-without-queue-sync 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake -S . -B build-sycl -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-sycl --target iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-sycl --output-on-failure -R "iom_sycl_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — SYCL devices enumerate and smoke, conformance, and coexistence pass.
- Run the rank-10 control and rank-11+ burst with stream/queue wait counters and a CUDA Nsight Systems trace (`nsys profile --trace=cuda,nvtx,osrt`), ROCm HIP/API tracing, and SYCL queue/API tracing; expect zero explicit growth waits, preserved overlap until the final token, exact results, and no metadata lifetime errors.
