# Honor the documented ADD rank contract across all backends

**Order:** 01
**Priority:** P0 — public-contract correctness/memory-safety gating all backends
**Blocked by:** None
**Review source:** `cpp-inference-contract-correctness` — whole-codebase review of checked-out main HEAD `ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb` (`Update remote hosts`), clean tree at review start
**Finding:** CC-001
**Review area:** Contract & correctness
**Review severity:** high
**Review verification:** verified, confidence 99
**Review scope:** whole-codebase
**Backend scope:** multi-backend
**Location:** `src/iom.cpp:707-752` (`DeviceOps::validate_add`); `src/cpu/device.cpp:662-697,733-754` (`CpuQueue::add_elements/add_impl`); `src/sycl/copy.cpp:548-606` (`SyclQueue::add_elements`); `src/shared/standard_tiled_add.inl:33,411-415` (`kAddMaxRank/make_add_metadata`); `src/shared/gpu_queue.hpp:228-246` (`GpuQueue::execute`)

## Outcome

Every documented ADD request with rank at least two either executes correctly at every enabled backend, including ranks 8, 9, 16, and 17, or is rejected synchronously before sequence consumption, owner registration, output effects, and positive OID acceptance when setup is genuinely impossible. CPU, SYCL, CUDA, ROCm/HIP, and TTNN therefore share the documented rank contract without backend-local deferred failures or out-of-bounds coordinate access.

## Current problem

The public contract in `docs/BACKEND_CONTRACT.md:336-340` rejects only ranks below two and defines right-aligned broadcasting for rank at least two; it specifies no rank ceiling. `DeviceOps::validate_add` computes an arbitrary-rank result but does not impose a common upper bound. CPU and SYCL then use `std::array<std::size_t, 8>` coordinates: CPU throws `CPU ADD rank exceeds implementation limit` inside the `submit_add` completion callback, while SYCL indexes `coord[axis]` for every result axis. CUDA/ROCm separately use `kAddMaxRank = 16` and call `make_add_metadata` from the worker after `submit_add` has reserved work. The root-run CPU probe created a valid rank-9 U8 ADD, received positive token `36028797018963969`, and both waits failed with `CPU ADD rank exceeds implementation limit`; the remote SYCL ASan/UBSan host probe on Intel Arc Pro B60 caught a rank-9 `SyclQueue::execute` stack-buffer-overflow writing past the 64-byte local `coord` at offset 3072. Thus a conforming request can receive a positive OID and fail later, corrupt host memory, or diverge by backend, consuming sequence/owner state and violating arithmetic safety.

## Scope

- Preserve the documented rank-at-least-two ADD contract and right-aligned broadcast, view, alias, dtype, and arithmetic rules for CPU, CUDA, ROCm/HIP, SYCL, and TTNN.
- Replace fixed 8-axis CPU/SYCL coordinates and fixed 16-axis deferred GPU metadata rejection with rank-sized CPU/SYCL traversal and rank-dynamic GPU metadata, with no per-element allocation.
- Make metadata-size arithmetic checked and acquire all required metadata storage before sequence reservation and owner registration; map genuinely impossible setup to the documented `Overflow` or `ResourceExhausted` OID category synchronously, with no effects.
- Add the merged rank-boundary arithmetic oracle coverage (NT-005) and SYCL rank-9 OOB reproducer coverage (ST-002) here rather than creating separate tasks.

## Implementation references

- **Modify:** `src/iom.cpp` — `DeviceOps::validate_add`, `DeviceOps::add`, and submission path; own the single common pre-acceptance decision and synchronous OID mapping.
- **Modify:** `include/iom/iom.hpp` — `submit_add` reservation/owner-registration boundary; ensure metadata setup precedes sequence reservation without weakening rollback.
- **Modify:** `src/cpu/device.cpp` — `CpuQueue::add_elements/add_impl`; use one rank-sized coordinate buffer allocated once per ADD.
- **Modify:** `src/sycl/copy.cpp` — `SyclQueue::add_elements/execute`; remove the eight-axis buffer and retain checked queue/staging cleanup.
- **Modify:** `src/shared/standard_tiled_add.inl` and `src/shared/gpu_queue.hpp` — `AddMetadata`, `make_add_metadata`, and GPU ADD worker; replace `kAddMaxRank` rejection with rank-dynamic metadata and checked size/storage acquisition before acceptance.
- **Read:** `src/ttnn/copy.cpp` — TTNN ADD coordinate mapping; reuse its dynamic-coordinate behavior as a backend counterpart without changing TTNN performance policy.
- **Tests:** `test/backend/backend_conformance_add.hpp`, `test/backend/backend_conformance_common.hpp`, and CPU/CUDA/ROCm/SYCL/TTNN conformance drivers; add rank 8/9/16/17 execution, exact U8 oracle, rejection-boundary, owner/sequence, and sanitizer scenarios.

## Requirements

- Do not introduce a public rank cap. A valid rank 8, 9, 16, or 17 request must be accepted and complete on every enabled backend, including TTNN, with exact logical output and stable owner behavior.
- Use rank-sized coordinate/traversal state in CPU and SYCL, created once per operation without per-element allocation. Every coordinate access, plane calculation, row/column lookup, and packed load/store must be bounds-safe for the validated rank.
- Replace CUDA/ROCm's fixed `kAddMaxRank=16` semantic limit with rank-dynamic ADD metadata. Check every multiplication/addition used for metadata byte size and acquire storage successfully before any sequence reservation or owner registration.
- Preserve one common pre-acceptance decision point. Setup overflow, impossible metadata size, or failed metadata storage acquisition must throw through `DeviceOps::add` as the documented `Overflow` or `ResourceExhausted` OID, leave output and owner/registry state unchanged, and consume no sequence. No backend may convert this condition into a positive token followed by a wait-time exception.
- Keep ranks below two, incompatible shapes, invalid views, unsupported leaves, quantization, alias, and arithmetic semantics unchanged. Do not add a public capability query or fallback API.
- Exercise rank 8, 9, 16, and 17 on each enabled backend with nonuniform U8 output and rank-2 broadcast inputs, plus the documented rejection boundary for impossible setup. Run the rank-9 SYCL reproducer under host ASan/UBSan and assert no out-of-bounds report after correction.

## Non-goals

- Do not alter numerical codec semantics, public rank-0/rank-1/empty-tensor support, broadcasting rules, alias policy, or GPU event/lifetime behavior as a separate root cause.
- Do not narrow the contract to rank 8 or 16, add a backend-local rank exception, or hide an implementation limit behind deferred wait failure.
- Do not redesign TTNN performance, add generic capability/fallback APIs, or duplicate the merged NT-005/ST-002 findings elsewhere.

## Acceptance criteria

- [ ] Valid rank-8, rank-9, rank-16, and rank-17 numeric ADDs return positive OIDs, complete successfully on repeated waits, preserve owner/registry cleanup, and match an independent logical-byte oracle on every enabled CPU, SYCL, CUDA, ROCm/HIP, and TTNN backend.
- [ ] Rank-9 SYCL execution has no host ASan/UBSan out-of-bounds diagnostic and the worker remains usable for a subsequent ADD; rank-17 GPU metadata no longer fails after token acceptance.
- [ ] A checked metadata-size overflow or unavailable metadata allocation returns synchronous `Overflow` or `ResourceExhausted`, consumes no sequence, registers no owners, changes no output, and frees any partially acquired metadata exactly once.
- [ ] Rank-1 and malformed/incompatible ADD requests retain their existing synchronous `InvalidArgument`/`Unsupported` behavior, proving the new rank traversal does not weaken lower-bound and contract validation.
- [ ] No fixed 8-axis coordinate array or public 16-axis rank ceiling remains on the accepted execution path.

## Verification

- `cmake -S . -B build/review-cpu -G Ninja -DBUILD_TESTING=ON && cmake --build build/review-cpu --target iom_cpu_conformance_tests && ctest --test-dir build/review-cpu --output-on-failure -R '^iom_cpu_conformance_tests$'` — rank 8/9/16/17 oracle cases pass; impossible setup has the documented negative OID and no sequence/owner effects.
- `(remote-development: CUDA host)` `.agents/skills/remote-development/scripts/remote-sync cuda cc-001 && .agents/skills/remote-development/scripts/remote-exec cuda cc-001 'cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF && cmake --build build --target iom_cuda_conformance_tests && ctest --test-dir build --output-on-failure -R "^iom_cuda_conformance_tests$"'` — ranks 8/9/16/17 complete and rank metadata boundary is safe.
- `(remote-development: ROCm host)` `.agents/skills/remote-development/scripts/remote-sync rocm cc-001 && .agents/skills/remote-development/scripts/remote-exec rocm cc-001 'cmake -S . -B build -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF && cmake --build build --target iom_rocm_conformance_tests && ctest --test-dir build --output-on-failure -R "^iom_rocm_conformance_tests$"'` — the same rank oracle and synchronous setup-failure checks pass.
- `(remote-development: SYCL host)` sync and run the configured oneAPI/icpx SYCL conformance target through `remote-exec`, with rank-9 under host ASan/UBSan — no stack-buffer-overflow, positive token, successful repeated waits, and exact bytes.
- `(remote-development: TTNN host)` sync and run `cmake --build build --target iom_ttnn_conformance_tests && ctest --test-dir build --output-on-failure -R '^iom_ttnn_conformance_tests$'` — TTNN rank cases complete with the same oracle and owner checks.
