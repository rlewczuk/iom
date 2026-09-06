# Move the verbatim-duplicated `TrafficGate` and `expect_repeated_runtime_failure` conformance fixtures into the shared harness

**Order:** 79
**Priority:** P2 — required test-infrastructure finishing work; deletes four/three byte-identical fixture copies (one of them SYCL's) into the existing shared conformance harness so the next driver stops transcribing them. Low severity, gates no other work.
**Blocked by:** None
**Review source:** `cpp-inference-backend-simplicity` (Area 3 of `cpp-inference-code-review`) — whole-codebase working tree at `a9d8d0ccc3fb0d03e4746082481669dc969e7111`
**Finding:** `NT-005`
**Review area:** Numerical correctness & tests
**Review severity:** low
**Review verification:** verified, confidence 88
**Review scope:** whole-codebase
**Backend scope:** multi-backend
**Location:** `test/sycl/test_sycl_conformance.cpp:32-42` (`TrafficGate`), `:370-388` (`expect_repeated_runtime_failure`); identical copies at `test/cuda/test_cuda_conformance.cpp:29-39,229-247`, `test/rocm/test_rocm_conformance.cpp:38-48,277-295`, `test/cpu/test_cpu_conformance.cpp:22-32` (`TrafficGate` only)

## Outcome

The two backend-neutral conformance fixtures that every accelerator driver hand-copies — `TrafficGate` (a `iom_conformance::ConformanceObserver` that arms between `setup_complete`/`case_complete` so a gated allocator can assert no tensor storage is allocated or freed inside a transfer/operation) and `expect_repeated_runtime_failure` (asserts a deferred `DeviceOps::wait` failure rethrows the identical `runtime_error` message on every repeat call) — exist once, in the shared harness `test/backend/backend_conformance_common.hpp` under `namespace iom_conformance`. The four `TrafficGate` copies and three `expect_repeated_runtime_failure` copies (including SYCL's) are deleted; each driver refers to the shared definition. Conformance behavior on every backend is unchanged.

## Current problem

- **Invariant:** shared conformance behavior lives in the backend conformance suite, not copied into backend-specific drivers (AGENTS.md: "Keep shared behavior in the backend conformance suite and runtime setup in backend-specific drivers").
- **Failing path:** `TrafficGate` is transcribed byte-for-byte into four drivers and `expect_repeated_runtime_failure` into three. MD5 of the extracted `TrafficGate` class body is identical (`22d4a91c…`) across `test/{cuda,rocm,sycl,cpu}/test_*_conformance.cpp`; MD5 of `expect_repeated_runtime_failure` is identical (`0c33da5e…`) across `test/{cuda,rocm,sycl}/test_*_conformance.cpp`. Neither fixture references a backend type: `TrafficGate` derives from the shared `iom_conformance::ConformanceObserver` (`test/backend/backend_conformance_common.hpp:172-183`) and holds one `bool armed_`; `expect_repeated_runtime_failure` takes `iom::DeviceOps&` + `iom::oid` and uses only `doctest` macros and `std::string_view`.
- **Evidence:** the MD5 identities above; `grep -n "class TrafficGate\|expect_repeated_runtime_failure" test/backend/*.hpp` returns zero matches, confirming the shared harness does not already own them despite owning `ConformanceObserver` and the `iom_conformance` namespace they belong to.
- **Impact:** a fix to the traffic-gate arming discipline or the repeated-failure assertion (e.g. tightening it to also reject a second distinct message, or arming around a new scenario hook) must be applied in three or four files by hand; a driver that misses the edit silently keeps the weaker fixture. The 0002 per-operation test split adds more driver files, multiplying the transcription.

## Scope

- Define `TrafficGate` and `expect_repeated_runtime_failure` once in `test/backend/backend_conformance_common.hpp` inside `namespace iom_conformance`, with the exact behavior the current copies have.
- Delete the per-driver copies in `test/sycl/test_sycl_conformance.cpp`, `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`, and `test/cpu/test_cpu_conformance.cpp` (`TrafficGate` only — CPU has no `expect_repeated_runtime_failure`).
- Update each driver's references to the shared names. Drivers already include `backend/backend_conformance_common.hpp` and already qualify harness names with `iom_conformance::`, so the reference change is a namespace qualification, not a new include.
- Affected backends: sycl, cuda, rocm, cpu (test code only). No production code changes.

## Implementation references

- **Modify:** `test/backend/backend_conformance_common.hpp` — add, inside `namespace iom_conformance` (which already hosts `ConformanceObserver` at `:172`), an `inline`-equivalent `class TrafficGate final : public ConformanceObserver` reproducing the current 9-line body (`setup_complete`→`armed_=true`, `case_complete`→`armed_=false`, destructor→`armed_=false`, `armed()` accessor, `bool armed_=false`), and a `inline void expect_repeated_runtime_failure(iom::DeviceOps& queue, iom::oid token)` reproducing the current 19-line body. A class definition in a header is already `inline` for ODR purposes; the free function needs the `inline` specifier.
- **Modify:** `test/sycl/test_sycl_conformance.cpp` — delete `:32-42` and `:370-388`; replace the local `TrafficGate` uses (`SyclDevices::gate` at `:298`, the `iom_conformance::run_*` calls passing `&devices.gate`) with `iom_conformance::TrafficGate`, and the three `expect_repeated_runtime_failure(*queue, token)` calls with `iom_conformance::expect_repeated_runtime_failure(...)`.
- **Modify:** `test/cuda/test_cuda_conformance.cpp` — delete `:29-39` and `:229-247`; same reference updates (`CudaDevices::gate` at `:183`, `expect_repeated_runtime_failure` callers).
- **Modify:** `test/rocm/test_rocm_conformance.cpp` — delete `:38-48` and `:277-295`; same reference updates.
- **Modify:** `test/cpu/test_cpu_conformance.cpp` — delete `:22-32` (`TrafficGate`); update the `CpuDevices`/gated-allocator references. CPU has no `expect_repeated_runtime_failure` copy.
- **Read:** `test/backend/backend_conformance_common.hpp:172-183` — `ConformanceObserver`'s `setup_complete`/`case_complete` virtuals that `TrafficGate` overrides; the shared fixture sits beside them.
- **Read:** the per-driver gated allocators (`HostAllocator`, `SyclAllocator`, `CudaAllocator`, `HipAllocator`) — they take a `const TrafficGate&`/`const iom_conformance::TrafficGate&`; only the type's namespace qualification changes, not the allocator bodies.

## Requirements

- `test/backend/backend_conformance_common.hpp` contains exactly one definition of `TrafficGate` and one of `expect_repeated_runtime_failure`, both in `namespace iom_conformance`, with behavior identical to the deleted copies (arming transitions, `armed()` accessor, destructor disarms; the repeated-failure helper loops two `wait` attempts, requires both to throw `std::runtime_error`, requires the second message to equal the first, and requires a non-empty message).
- The gated allocators continue to take the gate by `const iom_conformance::TrafficGate&` and their `CHECK_MESSAGE(!gate_.armed(), …)` assertions are unchanged.
- Each driver's `run_storage_and_transfer_conformance`, `run_async_copy_conformance`, `run_lifetime_conformance`, and the deferred-failure cases compile against the shared fixture and produce identical pass/fail results.
- No production source under `src/` or `include/` is modified.

## Non-goals

- The per-backend `*Allocator`, `Reusing*Allocator`, `*StorageOracle`, `*Devices`, and `ContextCallsRestore`/`LaunchCallsRestore` fixtures — these legitimately reference backend types (`sycl::context`, `cudaMalloc`, `hipMalloc`) and stay in their drivers.
- The SYCL `SubmissionFault`/fault-injection cases, the ROCm watchdog subprocess harness, and any backend-specific scenario logic.
- `66-AR-007-factor-backend-test-targets` (the CMake `add_iom_backend_tests` extraction) — orthogonal; this task changes C++ fixture definitions, not build wiring.
- Adding new conformance scenarios or changing harness coverage; this is a pure relocation of two existing fixtures.

## Acceptance criteria

- [ ] `grep -rn "class TrafficGate" test/` returns exactly one match, in `test/backend/backend_conformance_common.hpp`; `grep -rn "void expect_repeated_runtime_failure" test/` returns exactly one match, in the same header.
- [ ] `grep -rn "TrafficGate\|expect_repeated_runtime_failure" test/sycl test/cuda test/rocm test/cpu` shows only `iom_conformance::`-qualified references, no local class/function definitions.
- [ ] `git diff --stat src include` is empty (no production change).
- [ ] On SYCL hardware, `iom_sycl_conformance_tests` passes unchanged, including the gated `run_storage_and_transfer_conformance` / `run_async_copy_conformance` cases and the deferred-failure cases that call `expect_repeated_runtime_failure`.
- [ ] On CUDA, ROCm, and CPU, the respective conformance suites pass unchanged with the shared fixtures.

## Verification

- `cd /home/rlew/iom/src/iom && cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build -j --target iom_backend_conformance_cpu_tests iom_tests && ctest --test-dir build --output-on-failure -R 'iom_backend_conformance_cpu_tests|iom_tests'` — local CPU baseline (the CPU driver is the only one of the four that builds without accelerator hardware).
- SYCL via `.agents/skills/remote-development`, `sycl` profile from `.remote-hosts.conf`: `remote-exec sycl <task-id> 'cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/sycl -j --target iom_sycl_conformance_tests && ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_conformance_tests$"'`.
- CUDA and ROCm, each on its own host per `.remote-hosts.conf`: build and run `iom_cuda_conformance_tests` / `iom_rocm_conformance_tests`; expect identical results to the pre-change baseline.
- Static audits: the `grep` criteria under Acceptance criteria.
