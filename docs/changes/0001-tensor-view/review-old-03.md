# Code Quality Review

## Review metadata

- **Scope:** `whole-codebase`
- **Target commit:** `87d74f9 Merge branch 'main' into spec-run-task/12784e9b...` (working tree reviewed at this HEAD; clean)
- **Baseline:** `n/a` (review began at `37f11e0`; commits `937674e` "fix: fence tensor destruction", `fba8770` "Use word-oriented CUDA and ROCm copy kernels", and the merge landed mid-review — all findings were re-verified against `87d74f9`)
- **Specification:** `docs/changes/0001-tensor-view` (root specification plus applied sub-specifications 01–50; retired `review-old-01.md`/`review-old-02.md` used as history context only, never as evidence)
- **Backends considered:** `cpu, cuda, rocm, sycl, ttnn` (SYCL implemented after the prior review and receiving its first review here)
- **Review coverage:** Five separated area passes (contract, stability, simplicity, numerical/testing, performance) over: all of `include/iom/**`; `src/{iom,alloc,safetensors,mmap,llama-excluded}.cpp`; `src/cpu/**`, `src/cuda/**`, `src/rocm/**`, `src/sycl/**`, `src/ttnn/**`, `src/shared/**`; both `CMakeLists.txt`; the full shared conformance harness (`test/backend/*.hpp`), all five backend drivers, `test/test_{iom,safetensors,alloc,mmap}.cpp`, `test/cpu/test_cpu_bench.cpp`. CUDA/ROCm host-side machinery inspected in full; CUDA/ROCm kernel bodies via the shared `standard_tiled_copy.inl`. `include/iom/llama.hpp` and `src/llama.cpp` excluded per the in-file scratchpad instruction and explicit user request. `build/`, `.work/`, `_local/` excluded as artifacts.
- **Validation performed:** CPU-only CMake configure + full build (GCC, Release) and complete `ctest` at both `37f11e0` and `87d74f9`: `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests` pass; `iom_cpu_bench` fails deterministically (see NT-001, reproduced 5/5). Two standalone probe programs linked against the built `libiom.a` directly inspected CPU tensor storage bytes against `detail::standard_plane_slot` (F32 and I4, `[1,32]`) — both demonstrate CC-001. Process note: the first parallel area-pass batch was lost to an external model-quota failure and was re-run to completion; every finding below was re-verified against the final HEAD by the orchestrator before acceptance. No CUDA/ROCm/SYCL/TTNN hardware, sanitizers, or profilers were available locally — accelerator claims are mechanism-verified from source only.

Finding IDs in this report are assigned by this review and are distinct from the historical change-spec IDs under `docs/changes/0001-tensor-view/<NN-*>`.

## 1. Contract & correctness

### CC-001 — CPU transfer/copy fast path writes a row-strip layout instead of the standard 16x16 tiled layout whenever columns is a multiple of 16 greater than 16

- **Severity:** critical
- **Verification:** verified
- **Confidence:** 95
- **Scope relation:** whole-codebase (introduced by the 41-PF-001 blocked-copy rewrite)
- **Backend scope:** cpu
- **Location:** `src/cpu/device.cpp:239-242` (`for_each_tile`) and `:351-357` (`for_each_tile_lockstep`) — the `columns % TILE == 0` whole-row run; materialized by `copy_tile_row_byte_aligned` `src/cpu/device.cpp:99-106` and `copy_tile_row_subbyte` `:128-134` (`elements > TILE` contiguous memcpy); consumers: `region_from_host` `:594-642`, `region_to_host` `:682-727`, `CpuQueue::copy_elements` `:910-952`
- **Invariant:** Spec §4: element `[..., row, column]` lives at slot `(plane*ceil(rows/16)*ceil(cols/16) + (row/16)*ceil(cols/16) + column/16)*256 + (row%16)*16 + column%16` — tile row, then tile column, then position in tile. `docs/changes/0001-tensor-view/spec.md:216-245` pins `[0,16] -> 256` explicitly.
- **Failure mode:** For any tensor whose last dimension is an exact multiple of 16 and larger than 16 (32, 48, …, 4096 — every tile-aligned model width, including both bench shapes), the tile walk emits one run of `columns` elements anchored at the row's start inside tile column 0 and the helper memcpys it contiguously. Row data is only contiguous within a single 16-element tile row; tile column k+1 begins 256 element slots after tile column k. Element `(0,16)` of a `[1,32]` F32 tensor is written to byte 64 (slot 16) instead of byte 1024 (slot 256), overwriting storage that belongs to rows 1–3, while the true tile-1 region is never written.
- **Evidence:** Direct storage inspection on the built library: `[1,32]` F32 seeded with element values 1000+i shows `storage[16] == 1016` (bug position) and `storage[256] == 0` (spec position untouched); `detail::standard_plane_slot(spec,0,0,16) == 256`. Reproduced for I4 (sub-byte whole-row branch) as well: element 16's nibble at slot 16, spec slots 256/257 read zero. Round trip `copy_from_host` → `copy_to_host` is byte-identical because both directions use the same wrong walk — the defect self-cancels in every round-trip and reference-vs-candidate comparison. All other backends are correct here: CUDA/ROCm word kernels (`src/shared/standard_tiled_copy.inl:71-96,140-186`) and SYCL (`src/sycl/copy.cpp:74-86`) map every element/word through the canonical slot math, so CPU physical bytes diverge from every accelerator for the same `TensorSpec`.
- **Impact:** The CPU reference backend physically violates the engine-defined layout for the most common real-model geometry. Spec §4 and the §11.2 worked examples are false for CPU storage; CPU tensors are byte-incompatible with CUDA/ROCm/SYCL tensors of identical spec, breaking every future direct-storage path (mapped weight upload into tiled storage, cross-backend staging, compute kernels on CPU tensors). Additionally the 41-PF-001 bench throughputs (24.3/85.6 GB/s F32, 19.5 GB/s I4) measure the invalid contiguous path and will drop when corrected (cross-referenced in NT-001's recalibration).
- **Recommended fix:** Do not patch the predicate alone. The structural cause is a second, divergent encoding of the tiled layout beside the one canonical `detail::standard_plane_slot` the walk already calls for row bases (flagged by the simplicity pass): gate the contiguous fast path on `tile_columns == 1` or, preferably, delete the `columns % TILE == 0` special case entirely so the existing per-tile-column loop (which is correct for exact multiples) always runs, and assert `elements <= 16` in both row helpers so the cross-tile memcpy branches can never fire.
- **Verification method:** Add `{16,32}` and `{2,3,16,48}`-class shapes to the shared owner-shape matrices (see CC-002); `run_storage_oracle_conformance` on CPU must fail pre-fix and pass post-fix; the orchestrator's probe (element `(0,16)` of `[1,32]` F32 at byte 1024) is the minimal regression; re-run `test_cpu_bench` to re-baseline honest throughputs.

### CC-002 — The direct-storage oracle matrix contains no shape with tile-aligned columns spanning ≥2 tile columns, so the suite was structurally blind to CC-001

- **Severity:** high
- **Verification:** verified
- **Confidence:** 93
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `test/backend/backend_conformance_copy_storage.hpp:168-178` (`transfer_owner_shapes`), `:255-263` (`copy_owner_shapes`), `:297-306` (oracle shape union); `test/cpu/test_cpu.cpp` physical-layout cases use columns ∈ {1,3,5,7,11,13,16,17,33} only
- **Invariant:** Spec §11.2: CPU tests "write independently chosen encoded byte patterns into padded tensors and inspect the allocation directly … prevents a matching pair of incorrect pack/unpack functions from passing through round-trip cancellation."
- **Failure mode:** Every oracle-matrix shape either has padded columns (17, 33 → correct per-tile-column path) or exactly one tile column ({16,16}, {2,3,16,16} → fast path degenerates to a correct 16-element run). No case has columns ∈ {32,48,…}. CC-001 therefore survived the entire suite — including the direct-allocation `CpuStorageOracle` and `test_iom.cpp`'s slot-math anchors (which do pin `slot({0,16}) == 256` for a 32×32 spec, but no transfer-level test consumes that shape).
- **Evidence:** Enumeration of both shape lists and the CPU-local layout cases; the oracle compares full native storage against `encode_standard_tiled_storage` built from `detail::standard_layout_slot` and would have caught CC-001 had an aligned multi-tile-column shape been present.
- **Impact:** The one acceptance mechanism explicitly designed to catch self-consistent layout errors is blind to the canonical tile-aligned geometry — the exact class of shape the engine will use in production.
- **Recommended fix:** Add at least one unpadded multi-tile-column shape (e.g. `{16,32}` and `{2,3,16,48}`) to both shape lists; the existing oracle machinery then enforces the layout on CPU and every accelerator with no new harness code.
- **Verification method:** With CC-001 unfixed, `run_storage_oracle_conformance` over the added shape fails; post-fix it passes on every backend.

### CC-003 — SYCL is fully excluded from the backend coexistence matrix, violating the umbrella spec's combined-build requirement

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 95
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `test/CMakeLists.txt:302` (`if(CUDA_ENABLED OR ROCM_ENABLED OR TTNN_ENABLED)`), `:322-361` (only `IOM_COEXIST_{CUDA,ROCM,TTNN}` definitions/links); `test/backend/test_backend_coexistence.cpp` (no SYCL branch anywhere)
- **Invariant:** Root spec goal 8 ("any combination of enabled backends can coexist in one build and process"), §6, §11.4, §11.7 ("a combined job enables and links all four optional backend libraries into one executable"), and sub-spec 15's outcome ("one executable links and uses CPU, CUDA, ROCm, SYCL, and TTNN devices together").
- **Failure mode:** The coexistence target never links `iom_sycl` under any configuration: with only SYCL enabled the target does not exist at all; with SYCL plus another accelerator the target exists but SYCL is not a participant. SYCL+X process-level coexistence (contexts alive together, interleaved submissions, the shared queue-id pool exercised with SYCL) is entirely untested.
- **Evidence:** `IOM_COEXIST_SYCL` appears nowhere in the tree. Sub-spec 15 is internally inconsistent — its outcome line includes SYCL while its requirements ("links libiom, iom_cuda, iom_rocm and iom_ttnn directly") and blocked-by list (08, 10, 14 — omitting 12) predate SYCL's implementation; the build followed the stale lines.
- **Impact:** A stated completion criterion of the change is unmet for the fourth optional backend; SYCL link-time collisions, context coexistence, and queue-id behavior under multi-backend processes are invisible to CI.
- **Recommended fix:** Add an `IOM_COEXIST_SYCL` branch (compile definition + `iom_sycl` link + a USM allocator participant) and extend the target gate to `SYCL_ENABLED`. The coexistence TU needs `-fsycl` compile/link options when the branch is active (or a separate SYCL participant TU with them).
- **Verification method:** Configure `-DSYCL_ENABLED=ON` with a second backend; build and run `iom_backend_coexistence_tests`; the interleaved-copy case must construct a SYCL device and match CPU bit-for-bit. Record a short spec-15 errata so the requirements text and tree agree.

### CC-004 — Public allocator-lifetime wording diverges across backend headers and from the applied deferred-free behavior

- **Severity:** low
- **Verification:** verified
- **Confidence:** 80
- **Scope relation:** whole-codebase (introduced by the 49-ST-003 deferred-quarantine behavior)
- **Backend scope:** common
- **Location:** `include/iom/cpu/device.hpp:11-17` vs `include/iom/cuda/device.hpp:12-15`, `include/iom/sycl/device.hpp:12-15`; quarantine drains at device destruction (`src/cpu/device.cpp:445-447` and twins); root spec §6–7
- **Invariant:** One consistent public allocator-lifetime contract; spec §7 "destruction calls `allocator.free(address)` exactly once."
- **Failure mode:** 49-ST-003 can defer a tensor's `free` to device destruction (quarantine). The CUDA/ROCm/SYCL headers were updated to require the allocator to outlive the returned device; the CPU header still promises only "must outlive every tensor and queue". A CPU caller following the CPU header (destroying the allocator after the last tensor/queue but before the device) hits a `free()` call on a destroyed allocator object whenever the quarantine holds an entry (failed op, or queue destroyed with unwaited work).
- **Evidence:** Header texts quoted; quarantine drains only from device destructors in all four registry backends.
- **Impact:** Narrow UB window (requires a prior failure/invalidations plus that destruction order), but the same engine states two different allocator-lifetime rules depending on which header the caller read, and the umbrella spec's "free exactly once at tensor destruction" text was never amended to permit the deferral.
- **Recommended fix:** Align the CPU header with the others; amend spec §6–7 (or the 49-ST-003 change doc) to record "free exactly once, possibly deferred to device destruction on the failure path."
- **Verification method:** Header text review; optional CPU unit test asserting the documented ordering requirement.

## 2. C++/GPU stability

### ST-001 — SYCL tensor destruction and queue teardown lack the 49-ST-003/30-AR-002 hardening the other four backends received: unconditional storage free, task drop at shutdown, and a non-transactional OOM submission path

- **Severity:** high
- **Verification:** verified
- **Confidence:** 90
- **Scope relation:** whole-codebase
- **Backend scope:** sycl
- **Location:** `src/sycl/device.cpp:154-156` (`~SyclTensor` bare release), `src/sycl/copy.cpp:259-268,399-424` (hand-rolled worker; `run()` returns on `shutdown_` without draining `tasks_`/`staged_`), `:289-349` (submission lambda: kernel submits precede `staged_.push_back`, which can throw)
- **Invariant:** Applied sub-change 49-ST-003's backend-generic outcome: a tensor destroyed while queued work references its storage must fence the work (or quarantine the storage) before release; applied 19-ST-002/30-AR-002: a submission that fails after backend work was enqueued must yield a waitable token or synchronously drain.
- **Failure mode:** (1) `copy(src, dst); destroy dst without wait` — the exact caller error 49-ST-003 hardens on CPU/CUDA/ROCm/TTNN — frees the USM allocation on SYCL while the in-flight kernel still reads/writes it: silent corruption or device fault. (2) `~SyclQueue` abandons published tasks without completing them; permitted in principle by the umbrella spec (destruction does not implicitly wait), but it diverges from the shared scaffold's drain-and-complete contract the other four backends follow. (3) If `staged_.push_back` (or the success-path `publish_staged` deque insert) throws after `queue_.submit` enqueued kernels, the caller receives an exception with no token while the kernels reference operand storage — the orphaned-work class 19-ST-002 eliminated elsewhere; here reachable only on allocation failure. Note: one candidate failure mode raised during review — "`wait()` deadlocks after queue destruction" — is impossible (a destroyed queue's `wait` cannot be called) and was rejected.
- **Evidence:** `src/sycl` contains no registry/quarantine/StagedWorker references (grep: zero matches); `~SyclTensor` is a bare `release_aligned_storage`; contrast the snapshot→fence→release-or-quarantine destructors in `src/cpu/device.cpp:493-553`, `src/cuda/device.cpp:153-220`, `src/rocm/device.cpp:133-200`, `src/ttnn/device.cpp:242-318`. No test anywhere destroys an owner before waiting (see NT-003).
- **Impact:** Backend-dependent memory safety for the identical caller error on the newest backend; the divergence also means the shared teardown/failure semantics have a fifth, weaker variant in the tree (root cause AR-001).
- **Recommended fix:** Port the 49-ST-003 pattern (registry state owned by `SyclDevice`, entries registered in the execute callback with a fence that waits the task's `sycl::event`, invalidate on queue teardown, quarantine in the destructor) and migrate `SyclQueue` onto `detail::StagedWorker`, which supplies drain-complete teardown and makes the OOM path transactional via staged-event-plus-`commit_failure`, exactly as CUDA/ROCm do.
- **Verification method:** SYCL analogue of the CUDA `ReusingCudaAllocator` regression: submit a copy, destroy an operand before wait, prove the freed address is not reused until the event completes, clean under a USM-aware sanitizer; plus a queue-destruction-with-pending case asserting every token completes or reports a retained failure.

### ST-002 — CUDA/ROCm `fence_destroy` dropped the event synchronize that spec 47-ST-001 mandates, regressing the P0 teardown fix

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 90
- **Scope relation:** whole-codebase (regression introduced by commit `387900b`, PF-002)
- **Backend scope:** cuda, rocm
- **Location:** `src/cuda/copy.cu:611-621` (`cuda_fence_destroy` → `destroy_cuda_resource_noexcept` → `destroy_event_noexcept`), mirrored at `src/rocm/copy.hip:591-600`; the drain path that depends on it: `include/iom/iom.hpp:157-186` (`process(task, false)` skips `fence_complete`)
- **Invariant:** Spec 47-ST-001 requirement 1: "Every non-null pending CUDA/HIP event is synchronized before exactly one destroy call, whether it is in AR-002's staged list, task list, normal worker path, or shutdown drain"; the spec's event-cleanup primitive is a no-throw synchronize-then-destroy, with the "second synchronize in `fence_destroy` … intentional because the same callback is also the helper's shutdown-drain cleanup boundary."
- **Failure mode:** On destructor-with-pending-tasks, `StagedWorker` drains via `process(task, false)`, which skips `fence_complete` and calls `fence_destroy` directly; the current `fence_destroy` only activates the context and destroys the event. Events are destroyed while their kernels may still be in flight; the queue stream is synchronized only afterward in `~CudaQueue`.
- **Evidence:** `git show 41ce130:src/cuda/copy.cu` contains the original `fence_and_destroy` (`(void)cudaEventSynchronize(event); (void)cudaEventDestroy(event);`) exactly as specified; commit `387900b` (PF-002 metadata-slot rework) replaced it with the sync-less destroy that survives at HEAD. Under CUDA's deferred busy-event-destroy semantics plus the trailing stream sync this is functionally safe today, which is why severity is medium and not high — but the repository's own accepted P0 contract is violated, and the safety now rests on undocumented-in-repo driver behavior rather than the specified fence.
- **Impact:** Spec-contract regression concentrated on teardown-with-pending-work; on any runtime that does not honor deferred event destruction, destructor ordering becomes unsafe.
- **Recommended fix:** Restore a per-backend no-throw synchronize-then-destroy inside `cuda_fence_destroy`/`hip_fence_destroy` (metadata-slot release stays as is); the normal-path double synchronize is cheap and was explicitly sanctioned by the spec.
- **Verification method:** CUDA/ROCm queue-destruction regression with pending copies under Compute Sanitizer / ROCm memory diagnostics: no busy-event diagnostics, each pending event synchronized before its destroy; source inspection confirms exactly one event cleanup boundary per event.

### ST-003 — `OutstandingWorkRegistry::register_entry` leaks the `by_id_` entry when the `by_address_` insertion throws

- **Severity:** low
- **Verification:** verified
- **Confidence:** 85
- **Scope relation:** whole-codebase (introduced by 49-ST-003)
- **Backend scope:** common
- **Location:** `include/iom/outstanding_work_registry.hpp:233-255` — the outer `catch (...) { throw; }` covers the `by_address_.emplace` statement; only the inner handlers erase `by_id_`
- **Invariant:** Registry mutators are strongly exception-safe: an exception during index insertion leaves the registry unchanged.
- **Failure mode:** If `by_address_.emplace` throws (node allocation failure), control reaches the outer handler, which rethrows without erasing the already-inserted `by_id_` entry. The entry stays `Live` with a fence capturing an event that the caller's cleanup then destroys, is unreachable from `by_address_` (so no tensor destructor will ever see it), and — because the submission failed, leaving no outcome record — is never released: a permanent map-node leak holding a dangling closure that is never invoked. One reviewer initially described this as use of an uninitialized iterator (UB); that reading is incorrect — the middle handler is only reachable when the iterator is initialized. The leak is the actual defect. Reachability is allocation-failure only.
- **Evidence:** Catch-nesting analysis at the cited lines; `register_registry_entries`' rollback cannot reach the leaked entry (its id was never returned).
- **Impact:** Bounded leak on an OOM path; no invocation of the dangling fence.
- **Recommended fix:** Erase `by_id_` in the outer handler — or simpler, restructure so the middle `try` encloses the address insertion (or initialize the rollback path from the `Entry` already held).
- **Verification method:** Fault-injecting allocator test asserting all four indexes return to pre-call state after a failed `register_entry`.

### ST-004 — TTNN's destructor terminates the process on quarantine-allocation failure while the sibling backends leak

- **Severity:** low
- **Verification:** verified
- **Confidence:** 95
- **Scope relation:** whole-codebase (introduced by 49-ST-003)
- **Backend scope:** multi-backend
- **Location:** `src/ttnn/device.cpp:243-270` (`std::terminate()` on `new (std::nothrow)` failure) vs `src/cpu/device.cpp:501-508`, `src/cuda/device.cpp:165-172`, `src/rocm/device.cpp:144-151` (swallow-and-leak)
- **Invariant:** One uniform failure policy for quarantine-add failure in `noexcept` tensor destructors.
- **Failure mode:** Identical allocation-failure conditions produce `std::terminate` on TTNN and a silent leak elsewhere; both policies are defensible but their coexistence is contract drift a maintainer must navigate.
- **Evidence:** Side-by-side of the four destructor quarantine lambdas.
- **Impact:** Process death vs continued-with-leak for the same fault class; currently unreachable in practice.
- **Recommended fix:** Pick one policy (leak matches three of four backends) and apply uniformly, or document why TTNN terminates.
- **Verification method:** Fault-injected quarantine allocation asserting identical observable behavior across backends.

### ST-005 — SYCL `synchronous_transfer` passes non-USM host pointers to `queue::memcpy`

- **Severity:** low
- **Verification:** verified (mechanism); impact conditional on runtime
- **Confidence:** 75
- **Scope relation:** whole-codebase
- **Backend scope:** sycl
- **Location:** `src/sycl/copy.cpp:218-231`
- **Invariant:** SYCL 2020 `queue::memcpy` requires USM allocations for both pointers; raw host pointers are out of contract on strict implementations.
- **Failure mode:** `source.data()`/`destination.data()` are ordinary host memory. DPC++ (the only compiler this build permits, enforced by `CMakeLists.txt:39-52`) stages such copies internally and works; other SYCL implementations the code never targets could fault or silently mishandle them. The surrounding staging logic is sound (word-rounded bounds, tail-bit zeroing, pad bytes never copied).
- **Evidence:** Call sites at the cited lines; CMake compiler restriction.
- **Impact:** Latent portability hazard bounded by the build's compiler gate; value correctness unaffected on DPC++.
- **Recommended fix:** Route host-side bytes through `sycl::malloc_host` staging or a `std::memcpy` on the host side of the transfer (host → shared staging is already a host `memcpy` candidate), keeping the synchronous contract.
- **Verification method:** Host-transfer conformance under DPC++ unchanged; if another runtime is ever supported, run the suite there.

**Verified clean (stability):** `CudaMetadataSlotPool::ensure_slot_capacity` synchronizes the stream before replacing the device buffer; `StagedWorker::submit_copy` is transactional (erase-then-rethrow; `std::list::splice` is noexcept) and CUDA/ROCm `execute` failure paths stage the event and `commit_failure` so every post-enqueue failure yields a waitable token; the `wait` → `fence_through_sequence` → `record_post_completion_failure` → `complete` lock graph is cycle-free (including TTNN's `fence_mutex_`/`api_mutex_` ordering); registry fences capture only primitive handles; the fba8770 word-merge kernels maintain single-writer-per-32-bit-word discipline with plane boundaries always word-aligned (256·bits ≡ 0 mod 32) and per-plane gather launches serialized on one in-order stream; SYCL staging atomics stay in bounds and relaxed ordering suffices (single writer per bit, in-order queue serializes the preceding `memcpy`); pool `poison`/`release`/`destroy` wait logic; `allocate_registry_id` overflow handling.

## 3. Backend architecture & simplicity

### AR-001 — The SYCL backend re-implements pre-remediation versions of every cross-backend queue/copy/destruction pattern instead of consuming the shared machinery

- **Severity:** high
- **Verification:** verified
- **Confidence:** 92
- **Scope relation:** whole-codebase
- **Backend scope:** sycl
- **Location:** `src/sycl/copy.cpp:88-129` (per-bit `atomic_ref` helpers — the pattern spec 46-PF-006:59 bans), `:131-245` (per-plane launches; per-call `sycl::queue` + `malloc_shared`/`sycl::free` staging), `:247-434` (hand-rolled ~190-line staged worker duplicating `detail::StagedWorker`, `include/iom/iom.hpp:28-196`), `:34-71` (per-submission `view_planes`/`plane_pairs` host vectors), `src/sycl/device.cpp:154-156` (no destruction fencing), `CMakeLists.txt:39-52` (global DPC++ compiler requirement vs spec 11's "target-local compiler/link settings"), `test/CMakeLists.txt:302` (coexistence exclusion, CC-003)
- **Invariant:** One semantic operation → one implementation: queue staging/completion/drain mechanics (30-AR-002), single-launch word-oriented copy (42-PF-002/46-PF-006), pooled staging and streams (43-PF-003), fenced tensor destruction (49-ST-003), and teardown fencing (47-ST-001) are backend-neutral contracts; AGENTS.md's clean common/backend boundary.
- **Failure mode:** ~310 of `src/sycl/copy.cpp`'s 460 lines duplicate machinery that exists exactly once elsewhere, at the pre-fix generation of each: the bespoke worker already diverges (drops pending tasks at shutdown; no `submission_order_mutex_`; non-transactional OOM path — ST-001), the data movement re-implements per-plane/per-bit patterns (PF-001), and the backend participates in none of the cross-backend hardening (ST-001, CC-003). Every future queue-semantics fix must now be applied in a sixth place or silently skipped.
- **Evidence:** Line-by-line citations above; `grep` for registry/Quarantine/StagedWorker in `src/sycl` returns zero matches; the divergence set matches the specs that explicitly deferred SYCL (30-AR-002 non-goal "Adding SYCL queue migration"; 46-PF-006 "Out of scope: … SYCL kernels") plus 49-ST-003, which omits SYCL silently — a deliberate-but-untracked deferral, which is precisely why nothing converges it.
- **Impact:** Five backends implement one contract at three different architecture generations; SYCL hardware behavior (teardown safety, throughput, coexistence) already differs observably and is unobserved by the coexistence suite.
- **Recommended fix:** One convergence change (mirroring the 29/30-AR-001/002 pattern): adopt `detail::StagedWorker`; consume `src/shared/standard_tiled_copy.inl` by defining the existing `IOM_GPU_*` macro surface for SYCL (the word-copy device functions are plain macro-prefixed code; SYCL needs only a launch wrapper) which deletes the per-bit helpers and per-plane launches; pool staging (`malloc_device`, not `malloc_shared`, plus one device-owned transfer queue); port the 49-ST-003 registry with the shared destructor helper (AR-002); fold SYCL into coexistence (CC-003). Legitimately SYCL-specific and worth keeping: device enumeration with GPU/accelerator filtering, the USM pointer-type compatibility check, in-order queue construction, kernel lambda syntax.
- **Verification method:** After convergence: `grep src/sycl` for `atomic_ref`/per-call `malloc_shared`/hand-rolled worker returns zero; SYCL conformance plus the destroy-before-wait regression passes on hardware; a SYCL+CPU coexistence configuration builds and runs.

### AR-002 — The tensor-destructor release-or-quarantine protocol and registry glue are duplicated near-verbatim across four backends

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 95
- **Scope relation:** whole-codebase (introduced by 49-ST-003)
- **Backend scope:** multi-backend
- **Location:** `src/cpu/device.cpp:493-553`, `src/cuda/device.cpp:153-220`, `src/rocm/device.cpp:133-200`, `src/ttnn/device.cpp:242-318` (the ~70-line snapshot→fence→classify→release-or-quarantine body, three of them identical including comments); `src/{cpu,cuda,rocm}/registry_state.hpp` identical modulo namespace; `SequenceOutcome`/`complete_task` twins in each queue
- **Invariant:** One source of truth for the destruction-fence protocol — the most lifetime-critical code the project owns.
- **Failure mode:** A fix to the protocol must be replicated four times; the copies already drift in small details; a missed backend silently keeps the old behavior.
- **Evidence:** Line-for-line comparison; only the quarantine-cleanup construction and the release action differ (exactly the two closures a helper would parameterize).
- **Impact:** ~180 duplicated lines of lifetime-critical logic plus ~85 lines of glue; the SYCL port (AR-001) would become a fifth copy.
- **Recommended fix:** One shared helper beside the registry — `detail::release_or_quarantine(registry, address, quarantine, make_cleanup, release)` encoding snapshot/fence-outside-locks/classify/remove — leaving each destructor ~12 lines; collapse the three identical `registry_state.hpp` files into one shared definition; keep only `TtnnNativeCleanupAction` backend-local.
- **Verification method:** The 49-ST-003 acceptance regressions pass unchanged on CPU and hardware backends; source audit shows a single definition.

### AR-003 — The outstanding-work registry is over-built for its two production consumers: a production-dead index, a duplicate snapshot type, public placement, and per-copy allocation cost that buys nothing on CPU

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 88
- **Scope relation:** whole-codebase (introduced by 49-ST-003)
- **Backend scope:** common
- **Location:** `include/iom/outstanding_work_registry.hpp:169-460` (four multimaps at `:455-459`; `invalidate_entries_for_sequence` `:297-312` and `by_sequence_`/`erase_sequence_index_locked` `:389-398` have no production caller — only `test/test_iom.cpp:2123`; `EntrySnapshot` `:180-187` repeats `Entry` field-for-field; `Quarantine` `:115-167` is a hand-rolled intrusive list); public placement: listed in `IOM_HEADERS` (`CMakeLists.txt:85`) despite the `iom::detail` namespace and spec 49-ST-003 calling it private; CPU registration at `src/cpu/device.cpp:831-863`
- **Invariant:** Registry state is the smallest structure serving its production queries — "entries at address" (destructor) and "entries of queue" (queue teardown); internal machinery stays internal; per-op submission cost stays small.
- **Failure mode:** Maintainers hold four insert/erase consistency invariants for a structure with two real queries; the dead sequence path can rot invisibly (its only test exercises a function nothing ships); every queued copy pays ~9–14 small allocations and ~10 lock operations (two entries × four multimaps, `outcomes_` map node, two `std::function` fence copies — see PF-003). On CPU the registration is provably pure overhead: `execute` runs the copy to completion on the caller thread before registering, so there is never in-flight work for a CPU entry to guard.
- **Evidence:** Consumer grep (only `snapshot_for`, `try_release_entry`, `invalidate_entries_for_queue`, `remove_entry*` are used in production); allocation/lock count traced through `register_registry_entries` and `complete_task`; CPU synchronous-execute fact from `include/iom/iom.hpp:81`.
- **Impact:** ~200 of 514 lines removable; measurable per-copy host overhead on the reference backend that the incoming 0002 elementwise work will multiply; internal machinery exported in the public include tree.
- **Recommended fix:** Reduce to `by_id_` + `by_address_` (queue teardown scans `by_id_`, bounded by genuinely outstanding entries, once per queue lifetime); delete the sequence index and its test (keep `sequence` as an unindexed diagnostic field); `using EntrySnapshot = Entry;`; replace the intrusive quarantine list with a `std::vector<std::unique_ptr<CleanupAction>>`; move the header under `include/iom/detail/` or `src/shared/` and out of `IOM_HEADERS`; let CPU opt out of registration entirely (the shared AR-002 helper handles the empty-entry case) or gate it until CPU has genuinely asynchronous work.
- **Verification method:** Registry unit tests updated; CPU conformance and the 16×16 bench before/after to show the per-copy overhead removal; CUDA/ROCm/TTNN 49-ST-003 hardware regressions unchanged.

### AR-004 — Residual CUDA/ROCm duplication: pools, metadata machinery, and the grid-stride kernel remain policy-mirrored twins (~695 identical lines) though the policy seam to share them already exists

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 90
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm
- **Location:** `src/cuda/copy.cu` vs `src/rocm/copy.hip` (475 changed lines of 932/931 → ~75% byte-identical): `gpu_policy` structs, metadata header + writer + `grid_stride_copy_kernel`, `MetadataSlotPool` classes, fence resources, queue shells; `src/{cuda,rocm}/{staging_pool.hpp,staging_pool.cpp,transfer_pool.hpp}` pairs
- **Invariant:** Duplicated responsibility is consolidated behind the existing policy seam; genuinely divergent units stay per-backend.
- **Failure mode:** The next copy-path fix again requires synchronized twin edits — the exact drift that produced ST-002 (a fix landed on one side of a rework) and the staging-padding divergence of the prior review.
- **Evidence:** Normalized diff; real adapter differences (CUcontext + injectable driver seam vs `hipSetDevice(int)`, error-string helpers, launch syntax) are already isolated inside `gpu_policy` and the `IOM_GPU_*` macros.
- **Impact:** ~450–550 duplicated lines in resource/pool/kernel/metadata code with zero semantic divergence; reviewer attention spent re-verifying twins.
- **Recommended fix:** Extend the proven `standard_tiled_copy.inl` pattern: move `grid_stride_copy_kernel` + the metadata header struct + writer into the `.inl` behind the existing macros; make `StagingSlotPool`/`TransferStreamPool`/`MetadataSlotPool` policy-parameterized shared headers. Do **not** template the whole queue classes — 0002-eltwise puts real per-backend compute implementations there, making queues the legitimate divergence boundary.
- **Verification method:** CUDA and ROCm conformance + fault-injection suites unchanged on hardware after extraction; source audit shows single definitions.

### AR-005 — Six compute-op rejection stubs hand-duplicated across five backends; default `DeviceOps` implementations would delete them and scale to the incoming 0002 op inventory

- **Severity:** low
- **Verification:** verified
- **Confidence:** 90
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `src/cpu/device.cpp:804-828`, `src/cuda/copy.cu:720-739`, `src/rocm/copy.hip:724-743`, `src/sycl/copy.cpp:354-380`, `src/ttnn/device.cpp:443-471` (~130 lines); base already owns the message constructor (`include/iom/iom.hpp:287-293`)
- **Invariant:** Capability rejection is backend policy, not per-method boilerplate; `DeviceOps` stays per-operation overridable.
- **Failure mode:** `docs/changes/0002-eltwise-binops/spec.md` defines ~35 elementwise operations with SYCL explicitly remaining blocked — without defaults, that scales to ~35 hand-written rejectors per lagging backend per op family.
- **Evidence:** Line counts above; 0002 spec op inventory.
- **Impact:** ~130 lines of noise now, growing with 0002.
- **Recommended fix:** Default `DeviceOps` bodies throwing `unsupported(backend_label(), "op")` backed by one protected `backend_label()` hook implemented once per queue; implementing backends override just their ops.
- **Verification method:** Existing unsupported-message assertions stay green; source audit shows zero hand-written rejectors outside intentional overrides.

### AR-006 — Vestigial and inconsistent queue/pool state misrepresents real invariants

- **Severity:** low
- **Verification:** verified
- **Confidence:** 92
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `src/cuda/staging_pool.hpp:46-51,76` and the ROCm twin (constructed `context_`/ordinal never read); `src/cuda/copy.cu:641-642`/`src/rocm/copy.hip:645-646` (`Task::{source,destination}` raw `TensorView*` dead after staging — read only by `execute` on the submitting thread); `SequenceOutcome::fence_succeeded` written in four queues but read only by TTNN; `submission_order_mutex_` present in four queues, absent in `SyclQueue` (folded into AR-001)
- **Invariant:** Stored state is read; task payloads reflect actual asynchronous consumers.
- **Failure mode:** Dead fields imply false invariants (staging bound to a context; queued tasks dereferencing caller views asynchronously) that future changes can be "fixed" against.
- **Evidence:** Usage greps per field.
- **Impact:** Local confusion cost; no behavioral defect.
- **Recommended fix:** Delete the dead pool members and constructor parameters; null the task view pointers after `execute` or document the one-line lifetime rule; drop `fence_succeeded` where unread.
- **Verification method:** Conformance suites unchanged; `-Wunused-private-field` clean where available.

### AR-007 — Five structurally identical per-backend CMake test blocks

- **Severity:** low
- **Verification:** verified
- **Confidence:** 80
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `test/CMakeLists.txt:52-108,109-156,158-227,250-300` (~170 lines; only names, sources, and options vary)
- **Invariant:** Build wiring for structurally identical targets is generated, not transcribed.
- **Failure mode:** The 0002 per-op test-file split (spec requires per-backend files under 500 lines) multiplies the blocks; inconsistencies creep in.
- **Evidence:** Block-by-block comparison of the four gated regions (rocm 52-108, cuda 109-156, sycl 158-227, ttnn 250-300): identical smoke/conformance executable structure, source property, include dirs, link lists, and add_test pairs; only target names, source files, and per-backend options differ. The 0002 spec's per-op test split (`docs/changes/0002-eltwise-binops/spec.md:108-109`) is the growth vector.
- **Impact:** Low at current scale.
- **Recommended fix:** An `add_iom_backend_tests()` function for the smoke/conformance pair; keep the coexistence block explicit. Natural trigger: the 0002 test files landing.
- **Verification method:** `ctest -N` target/test sets identical before and after.

**Verified clean (architecture):** common code is genuinely backend-neutral (no CUDA/HIP/SYCL/TTNN includes, no `BackendKind` switches, no active-backend state in core — the new registry header is type-clean, its placement being the AR-003 issue); the `standard_tiled_copy.{hpp,inl}` policy seam is the correct altitude and the natural landing zone for AR-004 and the AR-001 SYCL convergence; the two kernel families in the shared copy code (per-plane scatter/gather for pooled host transfers vs metadata-driven grid-stride for queued copies) are a justified division, not consolidation debt; the `DeviceOps` base owns exactly the right neutral state (queue-id pool, sequences, repeatable waits); TTNN's native per-plane storage behind the shared public contracts is legitimate divergence per AGENTS.md.

## 4. Numerical correctness & tests

### NT-001 — An absolute wall-clock latency gate with zero jitter margin keeps the always-run CPU suite permanently red on the reference host

- **Severity:** high
- **Verification:** verified
- **Confidence:** 95
- **Scope relation:** whole-codebase (gate introduced by 41-PF-001)
- **Backend scope:** common
- **Location:** `test/cpu/test_cpu_bench.cpp:115` (`CHECK_LE(queued_small_seconds, 6.0e-6)`), `:31-82` (median-of-5, one warmup, no isolation); `test/CMakeLists.txt:36-50` (unconditional `add_test`); threshold source `docs/changes/0001-tensor-view/41-PF-001-block-cpu-tile-copy/spec.md:171` ("36.3 µs reduces by ≥ 6×" → 6 µs)
- **Invariant:** A gate registered in the always-run ctest suite must pass on the reference machine under ordinary developer load, or the suite's exit-0 signal is meaningless.
- **Failure mode:** The measured quantity — submit+wait on a 16×16 copy — is dominated by two OS thread transitions (caller → `StagedWorker` → waiter), not by code; 7.1 µs median vs the 6 µs absolute bound fails deterministically on the development host (5/5 runs, load ≈ 1.2). The 6 µs number is a host-specific ratio frozen into an absolute gate with no margin.
- **Evidence:** Reproduced during this review at both reviewed commits; mechanism at `include/iom/iom.hpp:66-113` (staged publish/notify) and `src/cpu/device.cpp:766-790`; the conformance suites (`iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`) all pass — the suite's only red is this gate.
- **Impact:** Every applied sub-change spec requires "ctest exits 0"; that acceptance is structurally unreachable now. A permanently red always-run suite trains developers to ignore red, masking real regressions. Note the fix must also re-baseline the throughput floors once CC-001 lands, since the current 24/85 GB/s numbers measure the layout-invalid contiguous path.
- **Recommended fix:** Stop gating absolute submit+wait latency in the ctest-registered target: report it via `MESSAGE`, or make it self-relative (measure an identical-window no-op queued copy — the same condvar round trip without data movement — and gate `small_copy ≤ no_op + allowance`). Keep any absolute 6×-improvement gate in the remote/perf verification procedure, not the always-run suite; document the reference host next to the throughput floors.
- **Verification method:** 20 consecutive `ctest -R iom_cpu_bench` runs pass under ordinary load; an injected regression (e.g. reintroduced per-plane scratch) still trips the self-relative gate.

### NT-002 — SYCL queued copies are never observed against full native storage: the storage oracle exists but is not wired into the async-copy conformance

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 95
- **Scope relation:** whole-codebase
- **Backend scope:** sycl
- **Location:** `test/sycl/test_sycl_conformance.cpp:317-319` and `:357-360` (oracle argument omitted); contrast `test/cuda/test_cuda_conformance.cpp:320-324,366-371` (`&oracle` passed); spec `docs/changes/0001-tensor-view/38-NT-001-observe-queued-copy-storage/spec.md`
- **Invariant:** After a queued copy completes, the complete destination allocation equals its seed with only the window replaced, and the complete source allocation is unchanged.
- **Failure mode:** A SYCL copy kernel that writes padded tiles, clobbers a neighbor's bits with a wrong mask, or touches planes outside the destination window passes the entire SYCL suite: without the oracle, `run_async_copy_conformance` checks only destination-window logical bytes and never re-verifies the source. 38-NT-001 was created precisely to close this exposure on accelerators; SYCL — the newest backend, whose kernel is the only one still using per-bit atomics — is the only one left uncovered.
- **Evidence:** Direct argument-list comparison (verified independently during synthesis): `SyclStorageOracle` is implemented and used for host-transfer observation two tests above (`:306-313`) but the async-copy and full-suite calls omit it.
- **Impact:** Out-of-window or padding-corrupting writes by the SYCL copy kernel ship silently and would first surface as distant model corruption once compute ops arrive.
- **Recommended fix:** One-line cutover per site: construct `SyclStorageOracle(devices.candidate_allocator)` and pass it as the fourth argument, mirroring the CUDA driver.
- **Verification method:** On SYCL hardware: suite passes with the production kernel and fails at the first out-of-window byte with a deliberately widened kernel (parallel_for over the padded extent), as 38-NT-001 prescribes.

### NT-003 — 49-ST-003's destroy-while-queued guarantee has no regression test in the always-run suite; its own CPU acceptance criterion was never implemented

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 90
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `docs/changes/0001-tensor-view/49-ST-003-fence-tensor-destruction/spec.md:93` (unmet CPU criterion), `:97-98`; `test/test_iom.cpp:2097-2162` (unit-only coverage); no premature-owner-destruction case anywhere in `test/`
- **Invariant:** A tensor destroyed while queued work references its storage must be fenced or quarantined before free — the P0 mechanism 49-ST-003 built into four backends.
- **Failure mode:** No test destroys an owner before waiting (the shared harness waits every token first); consequently `~CpuTensor`'s snapshot/fence/quarantine flow executes zero times under test in its designed role, the queue-teardown → invalidated-entry → quarantine flow is covered only incidentally on accelerator hardware, and a regression to unconditional freeing passes the whole CPU suite. Unit gaps in the new machinery: `register_registry_entries`' rollback on second-entry failure (deterministically triggerable), `invalidate_entries_for_queue`, and the `try_release_entry`-vs-completion race have no direct tests. On SYCL the equivalent test cannot pass at all (ST-001).
- **Evidence:** Exhaustive grep for premature owner destruction; spec acceptance list vs tree; the two new unit cases cover only bookkeeping and double-drain idempotence.
- **Impact:** The P0 lifetime fix's user-visible guarantee is enforced only on machines with accelerators attached; the newest backend's lack of the mechanism stays invisible.
- **Recommended fix:** Implement the specified CPU poisoning/recycling test (submit, destroy source/destination before wait, assert no freed-address reuse until wait; queue-teardown variant asserting quarantine defers the free to device destruction); add the rollback and `invalidate_entries_for_queue` unit cases; mirror the destroy-before-wait case in the SYCL driver once AR-001/ST-001 land.
- **Verification method:** The CPU test fails when the destructor's snapshot/quarantine branch is disabled and passes enabled; the rollback test fails when the second-`register_entry` try/catch is removed.

### NT-004 — The shared copy matrix never pairs differing plane strides between source and destination, and lacks minimal-extent / row-tail-only shapes

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `test/backend/backend_conformance_copy_storage.hpp:193-253` (`copy_cases_for`), `:168-178` (`transfer_owner_shapes`), `:255-263` (`copy_owner_shapes`)
- **Invariant:** Conformance must exercise each backend's independent source-plane and destination-plane arithmetic (SYCL walks two stride vectors; CUDA/ROCm kernels carry separate stride arrays).
- **Failure mode:** Every shared `CopyCase` builds source and destination with mirrored transforms, so the two views always carry identical plane strides (only the offset differs). A backend that uses the source's stride vector for the destination walk passes the entire shared suite on every backend; differing-stride copies are exercised only by CPU-local tests, which cannot catch accelerator-only stride bugs. Secondary gaps in the same matrices: no `{1,1}` minimal-extent owner and no row-tail-only padded shape (`{17,16}`).
- **Evidence:** Inspection of all five case builders and both shape tables; the API forbids transforming the final two dimensions, so strides are the only copy-matrix degree of freedom beyond offset.
- **Impact:** A mirrored-stride regression in any accelerated backend's queued copy is invisible until a user copies between differently-strided views and gets silent plane misplacement.
- **Recommended fix:** Add two shared cases (contiguous-source → stepped-destination; permuted-source → contiguous-destination) plus `{17,16}` and `{1,1}` to the owner shapes; together with CC-002's additions the oracle path then physical-scope-checks them.
- **Verification method:** Inject the mirrored-stride bug into a backend's plane walk and confirm the extended suite fails while the current suite passes.

**Verified clean (numerical/tests):** the harness's host-encoding model is fully independent of backend code, with sentinel-prefilled readbacks that catch skipped bytes, padding leaks, and nonzero tail bits; the negative-oracle fixtures prove a perturbed slot map is detected on both CPU and SYCL with identity positive controls; SYCL's per-bit atomic RMW is value-correct (single writer per bit, in-order queue serialization, word-rounded staging bounds, tail-bit zeroing, pad bytes never copied); fail-not-skip holds everywhere (hardware `REQUIRE`d, no SKIP/MAYFAIL); compute-capability stubs are asserted on all five drivers; the safetensors and allocator fix-pinning tests (payload length, overlap, duplicate shard keys, `bad_alloc` exhaustion) are present and deterministic; tile-arithmetic anchors pin exact slots and injectivity.

## 5. Performance

### PF-001 — SYCL data movement reimplements the pre-PF-002/PF-003/PF-006 element-granular pattern: per-call queue + full-size `malloc_shared` staging/free + full shared-memory memset per download + per-bit element access, and per-plane submits with O(plane-count) host vectors and per-bit atomics on queued copies

- **Severity:** medium
- **Verification:** strongly-supported (mechanisms code-deterministic; magnitudes unmeasured — no SYCL runtime on this host)
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** sycl
- **Location:** `src/sycl/copy.cpp:197-245` (`synchronous_transfer`: per-call `sycl::queue` at `:202-204`, per-call `malloc_shared`/`sycl::free` at `:213,244`, full-size host memset of shared staging at `:225`), `:88-99` (`device_read_bits` — per-bit loop for all types including F32), `:101-129` (per-bit `atomic_ref` for sub-byte), `:270-352` (queued copy: `plane_pairs`/`view_planes` vectors per submission at `:275-278`, one `queue_.submit` per plane at `:301-331`, per-element slot recomputation)
- **Invariant:** Data movement approaches the design's near-DMA weight-upload goal (`docs/design/README.md:39`) with word-granular kernels, pooled staging/streams, and plane-count-independent submission — the standard the CUDA/ROCm backends now meet via `src/shared/standard_tiled_copy.inl` and the slot/stream pools.
- **Failure mode:** Every host transfer pays a runtime-queue construction and a full-size shared-memory alloc/free; on a discrete GPU the per-download memset and every per-element staging access traverse the interconnect (shared USM), and every element read is assembled bit-by-bit while sub-byte writes serialize per bit on contended words — for the sub-byte formats the design names as primary. Every queued copy scales submission cost with plane count (thousands of launches for `[1,4096,32,128]`-class views), exactly the pattern PF-002 eliminated elsewhere.
- **Evidence:** Side-by-side with `src/shared/standard_tiled_copy.inl` (word-ownership kernels, pooled streams/staging, tail-word-only device memset) — none of it present in `src/sycl`. Divergence set confirmed against the applied sub-changes (42/43/46 all list SYCL out of scope).
- **Impact:** Plausibly one-to-two orders below achievable device bandwidth for weight uploads on discrete hardware and plane-count-proportional submission latency; not a regression (SYCL never had the fast path) but a gap versus the engine's stated goal and its sibling backends. Root cause and fix vehicle: AR-001.
- **Recommended fix:** As AR-001: consume the shared `.inl` word kernels via the macro seam, pool `malloc_device` staging with one device-owned transfer queue, adopt grid-stride metadata-driven copies.
- **Verification method:** On SYCL hardware: transfer GB/s for F32 4096×4096 and I4 2048×2048 before/after; launch-count assertion (one submit per queued copy regardless of plane count); conformance unchanged.

### PF-002 — CPU and TTNN downloads pre-zero the entire destination buffer on every `copy_to_host` — one avoidable full pass, explaining the measured 24.3 vs 14.3 GB/s asymmetry

- **Severity:** medium
- **Verification:** strongly-supported (extra pass deterministic; asymmetry attribution consistent)
- **Confidence:** 80
- **Scope relation:** whole-codebase
- **Backend scope:** cpu, ttnn
- **Location:** `src/cpu/device.cpp:582` (`std::fill` before the tile walk), `src/ttnn/copy.cpp:213-215` (per-plane equivalent)
- **Invariant:** Download cost scales with logical bytes moved; only the tail bits of the final partial byte need canonical zeroing — the shared GPU path already models this (`src/shared/standard_tiled_copy.inl:409-418` memsets only the partial tail word).
- **Failure mode:** For byte-aligned types every zeroed byte is immediately overwritten — pure overhead (~40% of measured download time on CPU: 24.3 GB/s up vs 14.3 GB/s down on identical traversal). For sub-byte types only the final partial byte can retain garbage.
- **Evidence:** Measured asymmetry on the dev host; structural comparison with the tail-word-only GPU path.
- **Impact:** ~40% download-throughput loss on CPU; one extra host pass per TTNN download; downloads sit on every output-readback path.
- **Recommended fix:** Skip the fill when `bits % 8 == 0`; otherwise zero only the final partial byte (mirror the shared tail-word logic). Re-baseline after CC-001's fix.
- **Verification method:** `test_cpu_bench` F32 to_host approaches from_host parity; sub-byte tail-bit conformance cases (`{1,17}` I2/F6) keep passing.

### PF-003 — Per-submission fixed cost on every StagedWorker backend: ~14 small heap allocations and ~10–16 lock operations on the caller thread plus two thread handoffs per submit+wait

- **Severity:** medium
- **Verification:** strongly-supported (counts deterministic from code; time attribution anchored to the measured 7.1 µs round trip)
- **Confidence:** 78
- **Scope relation:** whole-codebase (the registry share introduced by 49-ST-003)
- **Backend scope:** cpu, cuda, rocm, ttnn
- **Location:** `include/iom/outstanding_work_registry.hpp:197-256,488-512` (8 multimap nodes + 2 `std::function` fence copies per copy), per-queue `outcomes_` map nodes, `include/iom/iom.hpp:66-113,308-329`, CPU/TTNN Task value-copies of two `TensorView`s (4 vector allocations), CUDA/ROCm fence-resource allocations
- **Invariant:** Submission overhead stays small relative to op cost; the future decode loop issues many ops per token.
- **Failure mode:** Every queued copy pays the bookkeeping stack on the caller thread (~14 allocations, ~8 lock pairs, plus ~8 more on the worker) and two condvar handoffs — plausibly 3–6 µs of the measured 7.1 µs small-copy round trip; every future compute op inherits it (0.1–0.2 ms/token at dozens of ops). Same data structure as AR-003 — not double-counted: AR-003 owns the design, this finding owns the cost.
- **Evidence:** Allocation/lock counts traced through the cited paths; measured anchor 7.1 µs for a 256-element copy whose data movement is sub-microsecond.
- **Impact:** Caps small-op throughput at ~140 K ops/s per queue today; multiplies across the 0002 op set.
- **Recommended fix:** In order: one registry entry per (sequence, queue) covering both addresses or lazy single-map lookups (AR-003); stateless function-pointer fences shared per task; reference-stored view descriptors in CPU/TTNN tasks (CUDA/ROCm already store pointers); let CPU complete inline when `execute` already ran the work (removes one thread hop).
- **Verification method:** Allocation/mutex counting per copy before/after; the 16×16 bench with headroom against the (recalibrated per NT-001) gate; registry conformance unchanged.

### PF-004 — CUDA/ROCm queued copies pay unpooled event create/destroy plus a metadata H2D memcpy per submission

- **Severity:** low
- **Verification:** strongly-supported (churn code-deterministic; per-call driver cost unmeasured)
- **Confidence:** 65
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm
- **Location:** `src/cuda/copy.cu:737,764` (`create_event` per execute; destroy in `fence_destroy`), `:741-743` (metadata H2D every submission); ROCm twin
- **Invariant:** Per-op driver-object lifecycle is amortized the way metadata slots already are.
- **Failure mode:** One `cudaEventCreateWithFlags` + `cudaEventDestroy` and one small `cudaMemcpyAsync` per non-no-op copy — same order as the whole small-op budget; negligible for large copies.
- **Evidence:** Code path; the adjacent metadata slots demonstrate the intended pooling pattern.
- **Impact:** Estimated ~1–4 µs driver overhead per submission in the small-op regime.
- **Recommended fix:** Pool a small event ring alongside the metadata slots (re-record after synchronize is legal); pass small metadata as kernel arguments when `leading_rank` is small.
- **Verification method:** GPU small-op latency with/without pooling; fault-injection suites re-pointed at the pooled acquire path.

### PF-005 — TTNN downloads serialize per plane: blocking `copy_to_host` per plane plus fresh host tensor and padded scratch vector per plane

- **Severity:** low
- **Verification:** strongly-supported (mechanism code-deterministic; magnitude needs hardware)
- **Confidence:** 70
- **Scope relation:** whole-codebase
- **Backend scope:** ttnn
- **Location:** `src/ttnn/copy.cpp:142-176` (`download_plane` blocking copy per plane), `:209-229` (per-plane loop), `:86` (per-plane padded vector); upload side already batches (`:206`)
- **Invariant:** Bulk host transfers pipeline device reads; host-buffer churn is O(1) per call.
- **Failure mode:** N serialized device round-trips instead of pipelined copies + one barrier; N fresh allocations per call. Bounded by SDK capabilities (a non-blocking `copy_to_host` may not exist).
- **Evidence:** PF-005-old pass-count fix verified present; PF-004-old watermark finish verified present — these are the residuals.
- **Impact:** Multi-plane download throughput scales with N × round-trip latency instead of bandwidth.
- **Recommended fix:** Submit all planes then finish once if the SDK allows; at minimum hoist one reusable scratch buffer per call.
- **Verification method:** On TT hardware: time a many-plane `region_to_host` before/after; count blocking copies per call.

### PF-006 — Grid-stride copy kernel recomputes plane/slot geometry with runtime divmods per word and per covered slot — likely ALU-bound below bandwidth for sub-byte types

- **Severity:** low
- **Verification:** hypothesis (arithmetic counts code-certain; materiality unmeasured)
- **Confidence:** 55
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm
- **Location:** `src/shared/standard_tiled_copy.inl:88-96` (`physical_coordinate` divmod per slot), `:140-186` (per-word slot walk; two full `plane_slot` recomputations per covered slot even for the same-spec identity case)
- **Invariant:** A pure data-movement kernel is memory-bandwidth-bound.
- **Failure mode:** Per 32-bit destination word: 3 runtime divmods by `bits`, plus per covered slot one tile-columns divmod and two slot recomputations — for I4 that is ~11 divmods plus 9 merge chains per 4 bytes (GPUs emulate 64-bit division); rough roofline puts sub-byte copies plausibly under half of achievable bandwidth. Coalescing and grid geometry are fine.
- **Evidence:** Instruction-shape analysis only; no profiler available.
- **Impact:** Potentially 1.5–3× below achievable copy bandwidth for I2/I4/F4/F6; minor for F32/BF16.
- **Recommended fix:** Precompute per-plane base word offsets into the metadata; specialize the same-spec case to straight word copies; strength-reduce `bits` divmods.
- **Verification method:** Nsight Compute on I4 and F32 4096×4096 copies versus a `cudaMemcpyDtoD` baseline; confirm SM-issue-bound before committing the rewrite.

**Verified clean (performance):** the CUDA/ROCm synchronous-transfer path matches the 43-PF-003/46-PF-006 structure (pooled streams, geometric pooled staging with poison-on-error, tail-word-only device memset, word-ownership kernels); TTNN's watermark finish fence suppresses repeated device drains; identical-window no-ops are genuinely cheap on every backend; completed-sequence waits take no condvar block and no device fence; the CPU partial-tile paths have correct slot math (the layout defect is exclusively the CC-001 run collapse, and the measured 7.1 µs / to-host asymmetry remain valid anchors for PF-003/PF-002).

## 6. Synthesis / overall assessment

### Overall assessment

The reviewed scope is **not acceptable as a completion of `0001-tensor-view` in its current state**, for one dominant reason and two structural ones. The dominant reason is CC-001: the CPU reference backend physically violates the engine's own tiled layout for every tile-aligned width above 16 — the canonical model-tensor geometry — and the violation is invisible to the entire test suite because every check either round-trips through the same wrong walk (self-cancelling) or never uses such a shape (CC-002). The reference backend being byte-incompatible with all four accelerators for the same `TensorSpec` undermines spec goal 7 ("CPU is the reference implementation") at exactly the layer the engine is built on. Second, the SYCL backend — implemented after the prior review — was built to the pre-remediation architecture of every cross-backend fix wave (AR-001) and consequently ships without the destruction-fencing safety property (ST-001), without coexistence participation (CC-003), without storage-oracle verification of its queued copies (NT-002), and with element-granular data movement (PF-001); the deferrals were partly spec-sanctioned but never tracked, so nothing converges them. Third, the always-run suite is deterministically red on the reference host (NT-001), which structurally blocks the acceptance workflow every sub-change spec relies on. Against that, the core metadata/view/token contracts are clean and strongly tested, the queue/token state machine held up under adversarial interleaving analysis, the shared `standard_tiled_copy` policy seam is genuinely good architecture, and the prior review's major findings (transactional submission, teardown, safetensors boundary, PF-001..006 on CPU/CUDA/ROCm/TTNN) are verifiably fixed — with one regression (ST-002) introduced by a later fix wave.

### Cross-area root causes

1. **A second, divergent encoding of the tiled layout in the CPU fast path (CC-001) + an oracle matrix blind to tile-aligned multi-tile-column shapes (CC-002) + bench gates measuring the invalid path (NT-001's recalibration clause).** The walk already calls the canonical `standard_plane_slot` for row bases; the `columns % TILE == 0` predicate re-derived contiguity incorrectly beside it. The fix must delete the second encoding and add the shape class, or the same class of bug returns with the 0002 elementwise walkers.
2. **SYCL non-convergence (AR-001 → ST-001, CC-003, PF-001, NT-002, ST-005).** One process root cause: the backend was implemented against the pre-fix architecture; specs 30/42/43/46 explicitly deferred SYCL and 49 omitted it, with no tracking item. One convergence change (StagedWorker + registry + shared `.inl` kernels + pools + coexistence) resolves five findings at once.
3. **The outstanding-work registry's altitude and duplication (AR-003 + AR-002 + PF-003 + ST-003 + CC-004 + NT-003).** A 514-line four-index structure with per-copy allocation cost, a production-dead index, and a destructor protocol copied four times — protecting a contract the umbrella spec assigns to callers, with its own specified regression tests never implemented. Slimming it to its two production queries and one shared destructor helper reduces risk on the exact machinery future fixes must touch.
4. **Fix waves overwriting each other without hardware acceptance re-runs (ST-002, and historically 18-ST-001).** PF-002's rework silently deleted 47-ST-001's synchronize-then-destroy one commit after it landed; no local hardware gate caught it. Cheap countermeasure: cite-and-verify prior fix specs in each new change's verification step.

### Residual risks / verification gaps

- All CUDA/ROCm/SYCL/TTNN claims are mechanism-verified from source only; no accelerator hardware, sanitizers, or profilers were available in this review. ST-002's practical severity, PF-001/PF-004/PF-005/PF-006 magnitudes, and the SYCL hardware suite (including the NT-002 oracle wiring) require the remote-development workflow.
- CC-001's full blast radius (e.g., interaction with 49-ST-003 quarantine paths on CPU, and the exact post-fix CPU throughput floor) is established only to the extent of the F32/I4 probes and the slot math; the shape-matrix additions in CC-002 are the systematic sweep.
- The mid-review landing of `937674e`/`fba8770` means the registry machinery and word kernels have had no hardware verification cycle recorded in-repo since their merge; NT-003's missing tests compound this for the destruction-fencing guarantee.
- NT-004's differing-stride exposure remains open on all accelerators until the matrix additions land.

### Suggested validation sequence

1. **Fix CC-001 structurally (delete the second layout encoding), add the CC-002/NT-004 matrix shapes, and recalibrate NT-001's gate** — all CPU-runnable today; the probe and oracle conformance are the acceptance evidence, then re-baseline the bench.
2. **One SYCL convergence change** (AR-001: StagedWorker, registry/ST-001, shared word kernels, pooled `malloc_device` staging, coexistence/CC-003, NT-002 oracle wiring) — verified on SYCL hardware via the remote workflow, including the destroy-before-wait regression.
3. **Restore 47-ST-001's synchronize-then-destroy (ST-002)** and run the CUDA/ROCm teardown battery under Compute Sanitizer / ROCm diagnostics.
4. **Slim the registry (AR-003), share the destructor helper (AR-002), opt CPU out of registration**, and land NT-003's specified regression tests.
5. **AR-004 extractions and AR-005 stub defaults** opportunistically alongside the 0002 elementwise work, which is their natural trigger.
