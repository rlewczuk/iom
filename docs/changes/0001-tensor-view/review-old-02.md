# Code Quality Review

## Review metadata

- **Scope:** `whole-codebase`
- **Target commit:** `6f16c5a Close up old review.`
- **Baseline:** `n/a`
- **Specification:** `docs/changes/0001-tensor-view` (requirements context)
- **Backends considered:** `cpu, cuda, rocm, ttnn` (SYCL specified but absent — see CC-002)
- **Review coverage:** Near-exhaustive across five parallel area passes (contract & correctness, C++/GPU stability, backend architecture & simplicity, numerical correctness & tests, performance): all of `include/iom/**`, `src/**` (core + four backends), both `CMakeLists.txt`, the full shared conformance harness, all backend drivers, and `test/test_{iom,safetensors,alloc,mmap}.cpp`. `src/llama.cpp`/`include/iom/llama.hpp` excluded per the in-file scratchpad instruction. GPU-runtime behavior is mechanism-verified from source only — no CUDA/ROCm/TTNN hardware runs were performed.
- **Validation performed:** Local CPU build + full CTest (3/3 pass: `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, ~31k assertions); standalone reproduction programs linked against `libiom.a` for CC-001/CC-003 (including a demonstrated out-of-bounds read crash, exit 135); measured CPU microbenchmarks for PF-001 on the review host (Ryzen AI 9 HX 370, g++ 15.2.0 `-O2`). Synthesis pass adversarially re-verified material claims against source; two candidates were rejected on factual grounds and several were demoted (see per-area rejection notes).

## 1. Contract & correctness

### CC-001 — SafeTensors parser never validates payload length against declared shape/dtype; inconsistent entries load silently and the weight flow reads out of bounds

- **Severity:** high
- **Verification:** verified
- **Confidence:** 93
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `src/safetensors.cpp:117-131`
- **Invariant:** Weight flow is mapped file → SafeTensors → caller tensors → queued ops, with shapes, dtypes, and buffer sizes validated before work begins (AGENTS.md). The SafeTensors layer is the untrusted-input boundary.
- **Failure mode:** An entry declares `F16 [4096,4096]` (32 MiB logical) but `data_offsets` span 4 bytes. Only `begin <= end <= data_size` is checked; `end - begin == ceil(prod(shape) × leaf_bits(dtype)/8)` and range disjointness are never verified. A consumer sizing from `shape()` reads ~32 MiB from a 4-byte region.
- **Evidence:** Reproduced against built `libiom.a`: inconsistent entry accepted (`nbytes=4` vs logical 33554432); sampling `logical_nbytes` bytes from `raw<T>()` killed the process at the end of the mmap (exit 135). Two overlapping entries also accepted. `test/test_safetensors.cpp` covers only well-formed payloads. Independently found by two area reviewers (merged; the numerical-testing candidate is folded here).
- **Impact:** Corrupt or hostile checkpoints produce silently wrong weights or an out-of-bounds host read on the primary ingestion path.
- **Recommended fix:** In the entry loop, compute expected payload bytes with checked arithmetic; reject mismatched, overlapping, or non-ordered ranges with the established invalid-input category, naming the tensor.
- **Verification method:** New `test_safetensors.cpp` cases: truncated, oversized, and overlapping entries each throw; well-formed files keep passing.

### CC-002 — Mandatory SYCL backend milestone is entirely absent while later milestones were executed

Note: SYCL backend has been implemented since, please skip this point.

- **Severity:** high
- **Verification:** verified
- **Confidence:** 97
- **Scope relation:** whole-codebase
- **Backend scope:** sycl (declared in `BackendKind`, `include/iom/tensor.hpp:78`)
- **Location:** `CMakeLists.txt:16-20` (options CUDA/ROCm/TTNN only)
- **Invariant:** `docs/changes/0001-tensor-view/spec.md` §6/§10/§12 declare SYCL one of four optional accelerator libraries with `make_sycl_device`, `SYCL_ENABLED`, and shared conformance; sub-specs `11-sycl-buildable-scaffold` (P0) and `12-sycl-storage-copy` (P1) are fully specified, and 13 states "Begin TTNN only after the SYCL implementation and conformance milestone is complete."
- **Failure mode:** `-DSYCL_ENABLED=ON` is silently ignored (no such option); no factory, no `src/sycl/`, no driver, no coexistence branch. Verified: zero SYCL matches in either CMakeLists.
- **Evidence:** Repo-wide search finds SYCL only in docs and the dangling enumerator; specs 11/12 lack the `task.md` every executed milestone carries; previously reported in `review-old-01.md` and still absent at HEAD while milestones 17–22 landed.
- **Impact:** The change's completion criteria are unmet; the declared backend matrix is false; milestone ordering was violated (TTNN started before SYCL conformance).
- **Recommended fix:** Implement specs 11/12 as written, or formally descope: amend the umbrella spec, remove `BackendKind::SYCL`, record the deviation. The current in-between state satisfies no contract (aligns with AR-007).
- **Verification method:** `-DSYCL_ENABLED=ON` configure must build `iom_sycl` + smoke tests or fail hard without the SDK; an all-backend coexistence run constructs a SYCL device (spec 11 verification commands).

### CC-003 — SafeTensorsDir silently resolves duplicate keys across shards and reports inconsistent `size()`/`keys()`

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 94
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `src/safetensors.cpp:160-167`
- **Invariant:** The store's `size()`, `keys()`, and `operator[]` describe one coherent collection; a sharded weight directory with conflicting names is invalid input and must not load silently.
- **Failure mode:** Two shards both define key `k`: `tensors_.emplace` keeps the first shard's tensor, `keys_.push_back` appends both → `size()==1`, `keys().size()==2`, `operator[]` returns shard A's payload.
- **Evidence:** Reproduced independently by two reviewers against `libiom.a` (merged; the numerical-testing candidate is folded here). Tests cover only disjoint shard key sets (`test/test_safetensors.cpp:234-279`).
- **Impact:** A mis-sharded model loads without diagnostic and runs with wrong tensor data; iteration double-visits the shadowed name.
- **Recommended fix:** Check the `emplace` result; throw the established invalid-input exception naming the key and both shard paths; append to `keys_` only on successful insertion.
- **Verification method:** New case: duplicate-name shards throw; distinct shards satisfy `size() == keys().size()`.

### CC-004 — Allocation-exhaustion exception category diverges across the Allocator family

- **Severity:** low
- **Verification:** verified
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `src/alloc.cpp:76-84` (`LinearAllocator` → `std::runtime_error`) vs `src/alloc.cpp:133-135,252-253` (`ListAllocator`/`FixedSizeAllocator` → `std::bad_alloc`)
- **Invariant:** Allocation failures use the established standard category; the tensor contract models exhaustion as `std::bad_alloc`.
- **Failure mode:** `create_tensor` through an exhausted `LinearAllocator` throws `runtime_error`; sibling allocators throw `bad_alloc`. A caller catching `bad_alloc` misses the Linear case. The divergence is pinned by `test/test_alloc.cpp:90` vs `:202,:314`.
- **Evidence:** Code at the cited lines; all three allocator-consuming backends surface whichever category the injected allocator throws.
- **Impact:** Bounded error-handling inconsistency; no memory-safety impact.
- **Recommended fix:** Throw `std::bad_alloc` from `LinearAllocator::alloc` on exhaustion and update the test; or document and align the category on `Allocator::alloc`.
- **Verification method:** `iom_tests` with the updated assertion; new CPU case asserting `bad_alloc` from `create_tensor` with an exhausted Linear allocator.

## 2. C++/GPU stability

### ST-001 — Backend queue teardown destroys events and streams without synchronizing in-flight work

- **Severity:** high
- **Verification:** strongly-supported
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm
- **Location:** `src/cuda/copy.cu:424-448` (destructor drain), `src/cuda/copy.cu:578-587` (worker exception drain), `src/rocm/copy.hip:370-386,544-553`
- **Invariant:** Event/stream fences may only be destroyed after the work they fence has completed.
- **Failure mode:** The worker returns at `copy.cu:558-560` on `shutdown_` with tasks still queued; the destructor then `cudaEventDestroy`s every pending event and `cudaStreamDestroy`s the stream with no synchronize. CUDA documents destroying a busy event as undefined behavior. Verified in source during synthesis: the drain loops at `copy.cu:436-444` and `581-586` contain no `cudaEventSynchronize`.
- **Evidence:** Code paths above; the conformance harness documents queue-destruction-with-pending-work as a supported pattern (`test/backend/backend_conformance_other.hpp:388-440`).
- **Impact:** UB at teardown; combined with ST-003 (tensor frees), in-flight kernels can execute against freed storage → corruption, driver faults, or hangs on context release.
- **Recommended fix:** Synchronize each event (`cudaEventSynchronize`/`hipEventSynchronize`) before destroy in both drains, and `cudaStreamSynchronize`/`hipStreamSynchronize` before stream destroy.
- **Verification method:** CUDA/ROCm conformance case destroying a queue with pending copies under Compute Sanitizer / ROCm ASan; no diagnostics after the fix. Exact busy-event-destroy semantics should be confirmed against the target CUDA/HIP versions during hardware verification.

### ST-002 — CPU and TTNN queue tasks store raw `TensorView*` to caller-owned temporaries

- **Severity:** high
- **Verification:** strongly-supported
- **Confidence:** 92
- **Scope relation:** whole-codebase
- **Backend scope:** cpu, ttnn
- **Location:** `src/cpu/device.cpp:283-288,320-325,394-401`; `src/ttnn/device.cpp:283-295,330-337,406-420`
- **Invariant:** A queued async operation may only reference resources that outlive every queued operation — derived `TensorView`s from `slice`/`select`/`permute`/`reshape_leading` are temporaries.
- **Failure mode:** `Task{sequence, &source, &destination, …}` stores the addresses of the `copy(const TensorView&, TensorView&)` parameters. `queue->copy(t->view().slice(0,0,1), u->view().slice(0,0,1))` — the natural usage pattern — leaves the worker dereferencing stack temporaries whose lifetime ended at the semicolon. UB. TTNN additionally stores raw `const ttnn::Tensor*` into the owner's `planes_` vector, dangling if the owner is destroyed while queued.
- **Evidence:** Verified in source during synthesis at `cpu/device.cpp:286`. The conformance harness binds every derived view to a named local, masking the hazard (`test/backend/backend_conformance_copy_storage.hpp:528-535`).
- **Impact:** Silent corruption or crash for any caller not binding views to named locals; undetectable by the current suite.
- **Recommended fix:** `Task` carries an owned view descriptor (owner pointer + offset/strides) instead of `TensorView*`; TTNN holds a shared handle to the owning `TtnnTensor`.
- **Verification method:** ASan (`detect_stack_use_after_return`) conformance case passing a temporary view directly; must trip pre-fix and pass post-fix.

### ST-003 — Backend tensor destructors free storage without fencing queued operations that reference it

- **Severity:** high
- **Verification:** strongly-supported
- **Confidence:** 88
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm, ttnn, cpu (write-after-reuse variant)
- **Location:** `src/cuda/device.cpp:141-154`, `src/rocm/device.cpp:101-114`, `src/cpu/device.cpp:183-185`, `src/ttnn/device.cpp:211-214`
- **Invariant:** Storage referenced by any in-flight queued operation stays valid until the operation is fenced and completed.
- **Failure mode:** `~CudaTensor` calls `allocator_.free(address_)` with no queue join, token wait, or stream sync (verified in source during synthesis). A caller destroying a tensor with outstanding copies frees storage out from under in-flight kernels. CPU's allocators don't unmap on `free`, but a freed slot can be re-handed to a new tensor's `alloc` while the worker is mid-copy — a write-after-reuse race. Severity depends on the caller-supplied allocator: a `cudaFree`-based free implicitly synchronizes, a `cudaFreeAsync`-based one does not.
- **Evidence:** Destructors above; nothing in the codebase connects tensor destruction to outstanding queue tokens. The `GatedAllocator` test fixtures (`test/cuda/test_cuda_conformance.cpp:92-126`) prove the suite expects no concurrent storage reuse, but the API does not enforce it.
- **Impact:** Silent corruption or hard fault when a tensor dies before its queued ops are waited; with ST-001, no teardown order is currently safe without manual `wait` discipline.
- **Recommended fix:** Document the "wait all tokens before destroying operands" requirement explicitly on `Tensor`/`DeviceOps`, and enforce where feasible: a device-level outstanding-work registry consulted by the destructor, or destructor-side stream synchronization.
- **Verification method:** CUDA case with `ReusingCudaAllocator`: submit copy, destroy tensor without `wait`, run under Compute Sanitizer memcheck — diagnostic pre-fix, clean post-fix; CPU ASan analogue with a poisoning allocator.

### ST-004 — SafeTensorView/mapping lifetime is undocumented; views dangle after `munmap`

- **Severity:** medium
- **Verification:** strongly-supported
- **Confidence:** 82
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `include/iom/safetensors.hpp:13-39` (no lifetime documentation), `src/safetensors.cpp:60-63,170-176`
- **Invariant:** The weight flow keeps the mapping alive until weights are staged into tensors; every other view type in the project (`TensorView`) documents its non-owning lifetime contract — `SafeTensorView` does not (verified in source during synthesis).
- **Failure mode:** `auto view = dir[name]; const float* w = view.raw<float>();` then the `SafeTensorsDir` goes out of scope → `w` points into unmapped memory; reused pages silently corrupt weights before upload.
- **Evidence:** `raw<T>()` returns a typed pointer with no ownership (`safetensors.hpp:22-26`); `SafeTensorsDir::operator[]` does not extend the file's lifetime.
- **Impact:** Use-after-unmap on the weight-ingestion path for any caller that doesn't infer the undocumented contract.
- **Recommended fix:** Smallest: document the "store must outlive all views" contract on `SafeTensorView`/`SafeTensorsStore` (matching the `TensorView` doc style). Structural option: `SafeTensorView` holds a `shared_ptr<const MappedFile>`.
- **Verification method:** ASan/poisoned-mapping test: obtain raw pointer, destroy store, read — must be prevented or diagnosed post-fix.

### ST-005 — CUDA host transfers all share the legacy default stream, serializing against each other

- **Severity:** low
- **Verification:** verified (code); impact restated after adversarial check
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** cuda
- **Location:** `src/cuda/copy.cu:354-405` (`synchronous_transfer`: all launches and `cudaStreamSynchronize(nullptr)` at :389-390 on the default stream)
- **Invariant:** Host-side synchronization scope no broader than necessary.
- **Failure mode:** The original candidate claim (default stream stalls until all queue work completes) is **rejected**: the queue stream is `cudaStreamNonBlocking` (`copy.cu:413-414`), and the legacy default stream interlocks only with *blocking* streams — so no whole-device stall. The residual defect: every CUDA host transfer funnels through the one process-wide legacy stream, so concurrent `copy_from_host`/`copy_to_host` calls serialize with each other and with any other legacy-stream activity; ROCm already demonstrates the intended shape (per-call `hipStreamNonBlocking`, `copy.hip:311-325`).
- **Evidence:** Verified in source during synthesis at `copy.cu:376-390` and `413-414`.
- **Impact:** Concurrent host transfers on CUDA serialize; bounded, since transfers are synchronous by contract.
- **Recommended fix:** Mirror ROCm: per-transfer dedicated non-blocking stream (or a cached per-device transfer stream, which also serves PF-003).
- **Verification method:** Two threads issuing concurrent host transfers under Nsight Systems: fixed build shows overlap; current build shows serialization.

### ST-006 — ROCm failure path records the completion event on an errored stream

- **Severity:** low
- **Verification:** hypothesis
- **Confidence:** 60
- **Scope relation:** whole-codebase
- **Backend scope:** rocm
- **Location:** `src/rocm/copy.hip:419-433`
- **Invariant:** A retained submission failure must become observable via `wait` in bounded time.
- **Failure mode:** After a first-plane launch failure, the catch block records the event on a stream that may be in an error state; whether `hipEventSynchronize` then returns promptly with the error or blocks is ROCm-version-dependent. If it blocks, `wait` hangs despite `commit_failure` having stored the error.
- **Evidence:** `src/rocm/copy.hip:419-433` vs `src/cuda/copy.cu:481-495`; the 19-ST-002 spec acceptance criteria are silent on the ROCm event-record failure path.
- **Impact:** Potential hang of `wait` on the fault-injection path on affected ROCm versions.
- **Recommended fix:** Check the `hipEventRecord` result in the catch; on failure, `complete(sequence, failure)` immediately without waiting on the event; or have the worker call `complete` with the captured failure when `hipEventSynchronize` returns an error, before `hipEventDestroy`.
- **Verification method:** ROCm conformance case with `SubmissionFault` injected on plane 1; assert `wait` rethrows within bounded time on the target ROCm version.

### ST-007 — `DeviceOps::submit` commits the sequence after `queue_work` runs; an inline-completing backend would break `complete`

- **Severity:** low
- **Verification:** hypothesis (latent — no current backend completes inline)
- **Confidence:** 75
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `include/iom/iom.hpp:83-100`, `src/iom.cpp:572-594`
- **Invariant:** A token returned by `submit` is always either waitable or a throw; the commit-before-complete ordering currently lives only in doc comments.
- **Failure mode:** `submit` allocates the sequence, releases the lock, runs `queue_work`, then increments `next_sequence_`. A future backend whose `queue_work` completes inline would hit `complete`'s `sequence >= next_sequence_` guard and throw a spurious `invalid_argument`. All four current backends stage, so this is unreachable today.
- **Evidence:** `src/iom.cpp:84-100` and `src/iom.cpp:572-594`.
- **Impact:** Latent contract gap; surfaces as a confusing spurious throw if a backend ever completes synchronously inside `queue_work`.
- **Recommended fix:** Increment `next_sequence_` before invoking `queue_work` (commit the reservation up front).
- **Verification method:** Unit test with a `FakeQueue` whose `queue_work` calls `complete(sequence)` inline; must pass post-fix.

**Rejected in this area (adversarial gate):** the CUDA `DriverCalls` table race (test-seam only, no production writers — no plausible failure scenario); unchecked `rows * columns` multiplication (self-admitted unreachable given the checked `element_count()`); pageable-memory `cudaMemcpyAsync` semantics (premise false — the code uses synchronous `cudaMemcpy`, verified at `copy.cu:372-375,393-396`); driver/runtime API mixing in `synchronous_transfer` (no invariant violated — CUDA guarantees `CUdeviceptr`/`void*` interop); `llama.cpp` missing synchronization (scratchpad, excluded by in-file instruction).

## 3. Backend architecture & simplicity

### AR-001 — CUDA and ROCm ship two ~600-line copy/queue implementations that are already drifting

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 95
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm
- **Location:** `src/cuda/copy.cu:19-624` vs `src/rocm/copy.hip:19-590`
- **Invariant:** Identical standard-tiled copy semantics on both backends: word-padded sub-byte staging, 2^20 launch chunking, staged submission, `commit_failure` retention, event lifecycle, identical-window no-op.
- **Failure mode:** ~80% line-identical after prefix normalization, yet already diverged in policy: staging padding (flat `+4` vs word-aligned), staging allocation (driver-API `cuMemAlloc` on default stream vs `hipMalloc` + dedicated stream — the divergence ST-005 addresses), and change `18-ST-001` landed a staging-bounds fix on ROCm only. Kernels match statement-for-statement.
- **Evidence:** `diff src/cuda/copy.cu src/rocm/copy.hip` after normalizing cuda/hip/cu prefixes; `docs/changes/0001-tensor-view/18-ST-001` (ROCm-only staging fix).
- **Impact:** Every fix to shared tiled-copy logic must be applied twice; the existing drift shows one already wasn't.
- **Recommended fix:** One shared implementation compiled into both `iom_cuda`/`iom_rocm` behind a thin CUDA/HIP API alias layer; keep only deliberate differences (stream choice, allocation API) as adapter hooks. Protected invariants: staging word padding, chunk bounds, staged-submit ordering, failure retention, event ownership.
- **Verification method:** CUDA+ROCm conformance suites unchanged-green; diff of the shared file against each former copy shows only adapter deltas.

### AR-002 — Staged worker-queue scaffolding, copy validation, and unsupported-op stubs are implemented four times

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 92
- **Scope relation:** whole-codebase
- **Backend scope:** cpu, cuda, rocm, ttnn
- **Location:** `src/cpu/device.cpp:253-443`, `src/ttnn/device.cpp:250-433`, `src/cuda/copy.cu:407-599`, `src/rocm/copy.hip:345-565`
- **Invariant:** The commit-before-complete staging invariant exists as four independently mutated copies carrying the same comment text; already divergent in policy (`publish_staged` complete-on-throw per task on CPU/TTNN vs plain splice on CUDA/ROCm).
- **Failure mode:** `validate_copy` verbatim ×4 (`cpu/device.cpp:333-346`, `ttnn/device.cpp:345-358`, `cuda/copy.cu:532-542`, `rocm/copy.hip:470-480`); identical-window predicate verbatim ×4; 24 unsupported-op stubs + 4 `unsupported()` helpers. A policy change to staging/ordering/failure handling must be replicated 4×; a missed replica produces backend-specific hangs or lost failures that only surface on hardware.
- **Evidence:** Side-by-side diffs of the four queue classes; identical comment text at `cpu/device.cpp:348-350` and `ttnn/device.cpp:360-362`.
- **Impact:** Maintenance hazard concentrated exactly on the ST-critical machinery (changes 19-ST-002/20-ST-003 touched this code four times).
- **Recommended fix:** Hoist into backend-neutral common code: `DeviceOps` protected `validate_copy`/`identical_window` helpers; a reusable staged worker-queue skeleton with a backend `execute(task)` hook; default compute-op implementations on `DeviceOps` throwing the capability error. Preserve per-queue serialization, no-destruction-wait, validation-before-sequence, no-op skip, and exact exception types/messages.
- **Verification method:** Full conformance suite unchanged-green on all backends; grep shows single definitions.

### AR-003 — Capability modeling is asymmetric: queryable only for TTNN, and only outside the Device contract

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** common, cpu, cuda, rocm, ttnn
- **Location:** `include/iom/device.hpp:31-36` (no capability surface) vs `include/iom/ttnn/device.hpp:19` (`ttnn_supported_data_types()`)
- **Invariant:** Capability declarations stay consistent with implementation and tests; unsupported formats rejected before work.
- **Failure mode:** CPU/CUDA/ROCm support is observable only by constructing tensors; TTNN exposes a backend-namespaced free function. Each conformance driver hardcodes its own 23-entry leaf list (`test/cpu/test_cpu_conformance.cpp:21-34`, `test/cuda/test_cuda_conformance.cpp:28-42`, `test/rocm/test_rocm_conformance.cpp:29-42`); each backend carries 2–3 parallel statements of what it supports. A driver list drifting from the implementation silently tests the wrong capability set.
- **Evidence:** Grep of `kCpuLeafTypes|kCudaLeafTypes|kRocmLeafTypes|kAllLeafTypes` (four hardcoded lists).
- **Impact:** Adding a leaf type or backend requires editing the backend plus N test drivers; drift fails nowhere until hardware runs.
- **Recommended fix:** One capability predicate on `Device` (e.g. `supported_data_types()`), implemented once per backend; drivers feed the query into the shared harness. Single predicate, not a framework; preserves reject-before-allocation and fail-don't-skip.
- **Verification method:** Conformance unchanged-green; mutation test removing one type from a backend's span visibly shrinks that driver's coverage.

### AR-004 — TTNN supported-type knowledge has two in-product sources of truth plus redundant re-validation

- **Severity:** low
- **Verification:** verified
- **Confidence:** 88
- **Scope relation:** whole-codebase
- **Backend scope:** ttnn
- **Location:** `src/ttnn/device.cpp:37-54` (`kSupportedDataTypes` + `is_supported`) vs `:57-72` (`native_dtype`); duplicate `checked_plane_count` at `:453` and `:188`
- **Invariant:** `create_tensor` accepts exactly the published set and rejects with the documented exception category before any native object exists.
- **Failure mode:** Adding a type to the table but not the switch passes the gate, then fails mid-construction with `invalid_argument` instead of the documented `runtime_error` capability rejection. `TtnnTensor` lives in an anonymous namespace with exactly one construction site, so the duplicate extent check protects no second entry path.
- **Evidence:** Both tables at `src/ttnn/device.cpp:37-72`; single call site `TtnnDevice::create_tensor` → `std::make_unique<TtnnTensor>`.
- **Impact:** Bounded: differently-typed rejection mid-construction rather than corruption; two lists updated in lockstep by hand.
- **Recommended fix:** One `constexpr` mapping from which both the span and `native_dtype` derive; drop the duplicate `checked_plane_count` (keep it in the constructor as last line of defense).
- **Verification method:** TTNN conformance capability cases (`test_ttnn_conformance.cpp:293-320`) green; static_assert that span and mapping agree.

### AR-005 — CUDA driver-call seam is partial and self-inconsistent; error helpers duplicated across two TUs

- **Severity:** low
- **Verification:** verified
- **Confidence:** 90
- **Scope relation:** whole-codebase
- **Backend scope:** cuda
- **Location:** `src/cuda/driver.hpp:7-14`; bypasses at `src/cuda/copy.cu:64` and `src/cuda/device.cpp:192-212`; duplicated `cuda_error`/`check_cuda` at `device.cpp:21-37` and `copy.cu:35-51`
- **Invariant:** The injection seam is the single dispatch path for the calls it claims to cover; error attribution keeps operation names for every failure.
- **Failure mode:** The seam covers three calls; `cuCtxSetCurrent` is invoked through the table at one site (`device.cpp:103`) and directly at another (`copy.cu:64`), so a test stubbing `ctx_set_current` sees the queue/transfer path bypass it. `cuInit`/`cuDeviceGetCount`/`cuDeviceGet` are direct-only.
- **Evidence:** Source comparison of the two TUs; grep of `cuCtxSetCurrent` call sites.
- **Impact:** Test-injection architecture is misleading; future driver-call additions arbitrarily pick one of two call styles; duplicated helpers drift.
- **Recommended fix:** Route every driver call through the table or delete it; hoist the duplicated error helpers into `driver.hpp`.
- **Verification method:** CUDA smoke suite green; grep shows no bypassing call sites.

### AR-006 — Allocator-backed tensor construction and the 32-byte alignment contract are reimplemented per backend

- **Severity:** low
- **Verification:** verified
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** cpu, cuda, rocm
- **Location:** `src/cpu/device.cpp:23,162-181`, `src/cuda/device.cpp:119-139`, `src/rocm/device.cpp:18,80-99`; alignment undocumented at `include/iom/alloc.hpp:8-14`
- **Invariant:** Caller-allocator storage rejected (bad_alloc/misalignment) before the tensor is live, freed exactly once.
- **Failure mode:** `kStorageAlignment = 32` triplicated; the allocate→check→free-and-throw dance exists in three structurally divergent variants (ROCm's free-through-member-then-null is one refactor from a double-free if `free` ever throws). The 32-byte obligation appears nowhere on the `Allocator` interface.
- **Evidence:** Side-by-side reading of the three constructors.
- **Impact:** Bounded today; the divergence is exactly where a lifetime bug would be introduced once.
- **Recommended fix:** One backend-neutral construction helper owning the allocate/validate/single-free contract (backends supply only context pre-activation); document alignment on `Allocator::alloc`.
- **Verification method:** Misalignment/bad-alloc conformance cases (e.g. `test_cuda_smoke.cpp:176-186`) unchanged-green; one alignment constant.

### AR-007 — Dead configuration surface: hello-world executable, unconsumed compile definitions, dangling SYCL enumerator, unused allocator predicate

- **Severity:** low
- **Verification:** verified
- **Confidence:** 98
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `src/main.cpp:1-5` + `CMakeLists.txt:168-175`; `CMakeLists.txt:91,125,163` (`IOM_*_ENABLED` referenced nowhere); `include/iom/tensor.hpp:78` (`BackendKind::SYCL`); `include/iom/alloc.hpp:26` (`owns()`, no callers)
- **Invariant:** Declared build/configuration surface corresponds to real behavior.
- **Failure mode:** The shipped `iom` executable prints `Hello world!` (executed, exit 0); the three `IOM_*_ENABLED` PRIVATE definitions are referenced by no source or test; `BackendKind::SYCL` has no backend/factory/test (see CC-002); `SingleBufferAllocatorBase::owns()` has no caller.
- **Evidence:** Repo-wide greps; executable run; `ls src/` shows no sycl directory.
- **Impact:** Maintainers and users infer behavior (a CLI, per-backend source switches, a SYCL roadmap) that does not exist.
- **Recommended fix:** Delete all four (no lifetime/synchronization invariant involved); the SYCL enumerator's fate is decided together with CC-002.
- **Verification method:** All configure/build/ctest combinations green; zero references.

### AR-008 — QuantizationFormat vocabulary duplicated in a mechanical recognizer while every non-NONE format is rejected

- **Severity:** low
- **Verification:** verified
- **Confidence:** 80
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `include/iom/tensor.hpp:34-72` (24 enumerators) vs `src/iom.cpp:42-71` (`is_recognized`) and `src/iom.cpp:107-116` (`validate` rejects all non-NONE)
- **Invariant:** Unknown enum values → `invalid_argument`; declared-but-unsupported → `runtime_error`.
- **Failure mode:** Adding an enumerator without updating `is_recognized` misclassifies "unsupported" as "unknown" — wrong exception category; two 24-entry lists updated in lockstep for a vocabulary no backend accepts.
- **Evidence:** The only effect of `is_recognized` is choosing the exception type of an unconditional rejection; no other consumer of any non-NONE enumerator exists.
- **Impact:** Bounded: misleading exception category on extension plus a two-list maintenance hazard.
- **Recommended fix:** Derive the recognizer mechanically from the enum, or reject `quantization != NONE` directly until a backend supports one.
- **Verification method:** Existing validation-category tests in `test_iom.cpp` unchanged-green.

### AR-009 — SafeTensorsFile and SafeTensorsDir duplicate the store container and its three operations

- **Severity:** low
- **Verification:** verified
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `src/safetensors.cpp:135-149` vs `:170-184`; members `include/iom/safetensors.hpp:71-75` vs `:93-96`
- **Invariant:** Identical lookup semantics (`out_of_range` with tensor name, key ordering) for files and directories.
- **Failure mode:** Byte-identical `operator[]`/`size`/`keys()` bodies over duplicated `tensors_`+`keys_` members; a lookup-policy change (e.g. the CC-003 duplicate-key fix) must be made twice — this is the exact spot where divergence already hurts.
- **Evidence:** Side-by-side of the two implementations; both derive from `SafeTensorsStore` yet share nothing below it.
- **Impact:** Small: ~30 duplicated lines on the weight-loading path; real risk only when lookup policy changes in one copy.
- **Recommended fix:** One private store base owning the map + ordered keys with `insert(key, view)`; the two classes become parsers feeding it. Preserve mapping-file ownership (keep `files_` in the dir class).
- **Verification method:** `test_safetensors.cpp` unchanged-green.

**Verified clean in this area:** no runtime headers/types, backend-kind switches, or global active-backend registry in common code (only the documented backend-neutral queue-id pool, `src/iom.cpp:514-535`); CMake fails configuration when an enabled backend's SDK is absent (`find_package ... REQUIRED` + explicit `FATAL_ERROR` checks); TTNN rejects unsupported formats before native allocation with a published capability table.

## 4. Numerical correctness & tests

### NT-001 — Accelerator conformance never physically observes storage after a queued copy; "writes only the destination window" is pinned only on CPU

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `test/backend/backend_conformance_copy_storage.hpp:552-557`
- **Invariant:** `copy` "copies logical values in view coordinate order … copies no padding" (spec.md:493-494) — nothing outside the destination window may change.
- **Failure mode:** The shared async-copy conformance checks only destination-window logical bytes via `require_logical_bytes`. The `AcceleratorStorageOracle` (change 22-NT-001) wraps `copy_from_host` only (`run_storage_oracle_conformance`, lines 313-390), never `DeviceOps::copy`. On CUDA/ROCm/TTNN a kernel writing full padded tiles, overshooting a `kLaunchChunk` tail, or touching planes outside the window passes every shared case. Only CPU pins full-storage copy behavior (`test/cpu/test_cpu.cpp:939-979`).
- **Evidence:** Post-copy checks in the copy-case loop are exclusively `require_logical_bytes`; no `oracle.observe` follows any queue copy. CUDA/ROCm observe full storage only on the injected-fault failure path (canary checks).
- **Impact:** The one implemented operation's physical write scope is untested on every accelerator; an out-of-window write corrupts sibling views and surfaces as wrong values, not a conformance failure.
- **Recommended fix:** Add an oracle-backed queued-copy scenario mirroring the CPU `StorageModel::copy_view` logic: seed both owners through the oracle, run one queued copy per `CopyCase`, wait, then observe the full destination-owner storage against `initial + window` and the source owner against `initial`.
- **Verification method:** On hardware, run with a deliberately widened copy kernel (copy full padded tiles): current suite passes, extended oracle must report the first out-of-window byte.

### NT-002 — The oracle-detects-perturbation negative fixture has no CPU-runnable instantiation

- **Severity:** low
- **Verification:** verified
- **Confidence:** 90
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `test/cpu/test_cpu_conformance.cpp:121-127` (positive only); `PermutingStorageOracle` instantiated only in the CUDA/ROCm/TTNN drivers (`test/cuda/test_cuda_conformance.cpp:278-284`, `test/rocm/test_rocm_conformance.cpp:248-253`, `test/ttnn/test_ttnn_conformance.cpp:434-439`)
- **Invariant:** The detection machinery (`PermutingStorageOracle`, `swap_first_adjacent_slots`, the `require_match=false` mismatch path) must itself be proven to fail on a wrong map; change 22-NT-001's acceptance requires the CPU perturbation fixture be generalized around the permutation hook.
- **Failure mode:** CPU-only CI — the only locally runnable configuration — never executes the negative-oracle path; a regression neutering detection (identity permutation, wrong `require_match` polarity) is invisible until hardware runs.
- **Evidence:** Grep over `test/`: `PermutingStorageOracle`/`swap_first_adjacent_slots` appear only in the three accelerator drivers.
- **Impact:** Harness self-check gap on the only broadly runnable backend.
- **Recommended fix:** CPU `TEST_CASE` wrapping `CpuStorageOracle` in `PermutingStorageOracle(swap_first_adjacent_slots)` with `CHECK_FALSE(...)`, mirroring the accelerator fixtures; this also discharges change 22's unmet acceptance item.
- **Verification method:** New case returns false with the swap map, passes with identity, via `ctest -R iom_backend_conformance_cpu_tests`.

### NT-003 — RoPE table test certifies integer-division-degenerate frequencies and would fail against correct RoPE math

- **Severity:** low
- **Verification:** verified
- **Confidence:** 95
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `test/test_iom.cpp:1531-1534`
- **Invariant:** A numerical-table test must encode the mathematical contract `inv_freq[j] = 1/θ^(2j/head_dim)`.
- **Failure mode:** `(2 * j) / kHeadDim` evaluates in integer division before the cast (verified in source during synthesis, line 1533); for `j < head_dim/2` the quotient is always 0, so every certified `inv_freq` is 1.0 — wrong values for `j ≥ 1` (correct: `1, 0.1, 0.01, …` for head_dim=8). `src/llama.cpp:9` divides identically, so fixing the scratchpad breaks this test. The constructor's `theta` parameter is also ignored for the exponent base (hardcoded `10000.0`).
- **Evidence:** Numerically verified for `head_dim=8`: truncated exponents yield `1,1,1,1`; correct exponents yield `1, 0.1, 0.01, 0.001`.
- **Impact:** The suite's only numerical-table case pins mathematically wrong tables and blocks a future correct implementation.
- **Recommended fix:** `2.0 * j / static_cast<double>(head_dim)` in both test and implementation; use the `theta` constructor argument instead of the `10000.0` literal; mark the case as tracking the scratchpad while the model layer stays excluded.
- **Verification method:** With corrected math, `j ≥ 1` columns differ from column 0 and the test rejects a collapsing implementation.

**Coverage note (tests actually run):** `ctest` 3/3 pass locally in the CPU-only configuration (`iom_tests` 0.06s, `iom_cpu_tests` 0.16-0.20s, `iom_backend_conformance_cpu_tests` ~29-30s), ~31k assertions across 108 cases. Only `copy` is implemented; all six compute ops (`add/mul/silu/linear/rmsnorm/sdpa`) reject capability with `std::runtime_error` in every backend, shared-tested via `run_compute_capability_conformance`. Every comparison is bit-exact — the correct contract for conversion-free moves; tolerance policy is moot until compute lands. 16×16 tiling boundaries, sub-byte LSB-first packing with odd lengths, tail-bit zeroing, BOOL canonicalization, and overflow rejection are covered. Fail-not-skip holds by construction: no `SKIP`/`MAYFAIL` anywhere in `test/`, and CMake fails configuration when an enabled SDK is absent. Safetensors ingestion test gaps have bug root causes and are folded into CC-001/CC-003.

## 5. Performance

### PF-001 — CPU transfers and copies are element-granular scalar loops, 280–2000× below the memory floor

- **Severity:** high
- **Verification:** verified (measured)
- **Confidence:** 95
- **Scope relation:** whole-codebase
- **Backend scope:** cpu
- **Location:** `src/cpu/device.cpp:407-434` (`copy_elements`), `:84-111` (`for_each_coordinate`), `:115-132` (`plane_at`), `:192-241` (host transfers), `src/iom.cpp:178-206` (`standard_plane_slot`)
- **Invariant:** Transfers/copy are the engine's only implemented data-movement surface; the design's core loop streams quantized expert weights per layer during decode and must run near memory bandwidth (`docs/design/README.md:14-18`).
- **Failure mode:** Per element: row/column divmods + leading-dim divmod loop, then `standard_plane_slot` (6 checked multiplies + 2 divmods) per element (twice for device-to-device), plus `plane_at` per element, plus a 4-byte `memcpy` per F32 element; sub-byte types go per-bit through `read_bits`/`write_bits`.
- **Evidence:** Measured on the review host (Ryzen AI 9 HX 370, single thread, g++ 15.2.0 `-O2`, `libiom.a`): F32 4096×4096 (64 MiB) — `memcpy` 24.9 GB/s vs `copy_from_host` 0.089 GB/s (~280×), `copy_to_host` 0.084 GB/s, queued same-device copy 0.030 GB/s (~750×); I4 2048×2048 `copy_from_host` 0.0084 GB/s (~2000×); 16×16 F32 copy 36.3 µs submit+wait of which ~30 µs is the 256-element scalar loop (~115 ns/element). Byte-aligned elements are provably contiguous in 16-element runs within a tile row (`src/iom.cpp:199-205`), so blocked `memcpy` is available but unused.
- **Impact:** A 14 GB F16 model spends ~2.5–5 min in pure host-upload conversion vs well under a second blocked; sub-byte expert streaming is infeasible on CPU as implemented.
- **Recommended fix:** Tile-blocked iteration: per plane, tile-row/tile-column loops with 16-element-run `memcpy` for byte-aligned types and word-oriented pack/unpack for sub-byte; hoist tile geometry out of inner loops; iterate source/destination plane lists in lockstep instead of `plane_at` per element. No API change.
- **Verification method:** Re-run the microbenchmark: expect ≥10 GB/s F32, ≥1 GB/s I4, queued copy ≈ host-transfer cost.

### PF-002 — CUDA/ROCm `copy` launches one kernel per plane and heap-allocates an O(plane-count) vector per submission

- **Severity:** medium
- **Verification:** strongly-supported (mechanism code-deterministic; magnitude unmeasured — no local GPU)
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm
- **Location:** `src/cuda/copy.cu:450-499,319-352,93-127`; mirrored at `src/rocm/copy.hip:392-425,482-509,79-134`
- **Invariant:** Submission cost and launch count scale with bytes, not with the product of leading dimensions; queues are async so submission must stay cheap enough to run per token.
- **Failure mode:** Each `copy` builds a `std::vector<PlanePair>` of P = ∏(leading dims) entries on the caller thread, then issues P × ⌈elements/2²⁰⌉ launches plus one event create/destroy per op (`copy.cu:470-473`, destroyed at `:573`). A `[1, 4096, 32, 128]` per-token view → ~131k launches + a 2 MB host vector per copy.
- **Evidence:** Launch/pair-enumeration structure at the cited lines; the design's per-token decode path (KV writes, activation copies over head/sequence leading dims) is exactly the many-plane case.
- **Impact:** Submission latency and launch overhead scale with plane count instead of data size; per-submit heap allocation churns the hot path (AGENTS.md keeps operands/outputs allocation-free; this reintroduces per-op allocation one level down).
- **Recommended fix:** One grid-stride kernel per copy (plane,element index decomposed in-kernel from dims/strides passed as arguments), eliminating the host-side pair vector; pool events per queue.
- **Verification method:** Nsight/ROCm timeline of a many-plane copy: fixed build shows 1 launch and no per-submit allocation; compare end-to-end copy time.

### PF-003 — Synchronous host transfers allocate/free full-size device staging per call (plus per-call stream create/destroy on ROCm, plus a full-size memset per download)

- **Severity:** medium
- **Verification:** verified (mechanism code-deterministic; magnitude needs GPU timing)
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm
- **Location:** `src/cuda/copy.cu:369,380-384,404`; `src/rocm/copy.hip:311-313,322,341-342`
- **Invariant:** The weight-flow upload path must scale to repeated per-layer expert streaming without per-call allocator/stream churn (design intends near-DMA weight loads, `docs/design/README.md:39`).
- **Failure mode:** Verified in source during synthesis: every transfer pays a full-size `cuMemAlloc`/`cuMemFree` (both synchronizing), every download pays a full-size `cudaMemset` when only the tail partial word needs zeroing, ROCm additionally creates/destroys a stream per call; staging bypasses the caller's `Allocator` entirely. Pageable `cudaMemcpy` from the mmap'd SafeTensors region additionally forces driver-internal staging.
- **Evidence:** Code-certain per-call allocation/free/stream-lifecycle at the cited lines.
- **Impact:** Per-tensor/per-expert transfer cost grows by two synchronizing runtime calls (four on ROCm), defeating direct-DMA streaming and serializing the device at each expert load during decode.
- **Recommended fix:** Cache a per-device staging buffer (grown geometrically) and one non-blocking stream; zero only the tail word on downloads. Keep the synchronous-at-return contract unchanged.
- **Verification method:** API trace over an N-tensor upload: alloc/free counts drop from N to ~0 after warmup; wall time compared.

### PF-004 — TTNN queue worker runs a full-device `finish()` after every task and holds the device-wide API mutex across every task

- **Severity:** medium
- **Verification:** strongly-supported (mechanism code-deterministic; latency unmeasured)
- **Confidence:** 88
- **Scope relation:** whole-codebase
- **Backend scope:** ttnn
- **Location:** `src/ttnn/device.cpp:406-416`, `:221-235`, `:161-165`
- **Invariant:** Queues are asynchronous and in-order; independent queues from one device make independent progress; submission stays cheap.
- **Failure mode:** Per task: `api_mutex_` held across the copy + `mesh_command_queue(0).finish()` (whole-device barrier). N queued copies cost N device drains with zero pipelining; any concurrent host transfer stalls all queues for the entire multi-plane transfer.
- **Evidence:** `finish()` per task at `device.cpp:415` (comment attributes it to per-sequence failure capture); `api_mutex_` is a single per-`TtnnDevice` mutex serializing tensors, queues, and transfers.
- **Impact:** Per-op latency floor equals device drain time; the async queue contract degenerates to synchronous execution with extra thread hops.
- **Recommended fix:** Enqueue back-to-back and `finish()` only at wait boundaries (track last-finished sequence) or per-op TTNN events where the SDK provides them; narrow the mutex to the TTNN API calls that require it.
- **Verification method:** On TT hardware, K copies with a single wait on the last token: barrier count drops from K to 1; timeline shows the drop.

### PF-005 — TTNN host transfer materializes every plane through ≥3 full-size host buffers/passes under the device-wide mutex

- **Severity:** medium
- **Verification:** verified (mechanism code-deterministic)
- **Confidence:** 85
- **Scope relation:** whole-codebase
- **Backend scope:** ttnn
- **Location:** `src/ttnn/copy.cpp:95-120,65-70,160-177`, `src/ttnn/device.cpp:221-227`
- **Invariant:** "Minimal memory footprint on both VRAM and host RAM" (`docs/design/README.md:4-5`); upload moves each byte ~once.
- **Failure mode:** Per plane: zero-filled padded vector → `memcpy` of rows → `make_host_buffer`→`typed_buffer` re-copy → `to_layout(ROW_MAJOR→TILE)` conversion tensor → `copy_to_device`, all under `api_mutex_`, then a final `finish()`. Downloads mirror this. ~3× peak host footprint and ≥3 passes per plane, × plane count.
- **Evidence:** Buffer allocations and full-size passes at the cited lines; the extra `typed_buffer` copy exists only to satisfy TTNN's typed `HostBuffer` view.
- **Impact:** Dominates upload time and transient host RAM for `[B,S,H,D]`-shaped weights/activations; conflicts with the stated minimal-footprint design goal.
- **Recommended fix:** Fill rows directly into the typed buffer; write TILE layout directly (row-segment writes into the tile-major buffer) to skip the `to_layout` pass; zero only padded tails.
- **Verification method:** On TT hardware: instrument bytes allocated per uploaded plane (~3× → ~1×); measure `region_from_host` wall time before/after.

### PF-006 — GPU copy kernels are bit-granular: per-bit atomics on contended words for sub-byte types; ROCm reads even byte-aligned types per-bit

- **Severity:** medium
- **Verification:** strongly-supported (mechanism code-deterministic; no GPU measurement)
- **Confidence:** 80
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, rocm
- **Location:** `src/cuda/copy.cu:139-166,180-195`; `src/rocm/copy.hip:146-180,207-244`
- **Invariant:** The design's primary weight formats are sub-byte ("8-bit, 4-bit, 1.6-bit to sub-1-bit", `docs/design/README.md:16`); copies of those formats must not serialize per bit.
- **Failure mode:** Sub-byte path: one `atomicOr`/`atomicAnd` per bit on a 32-bit word shared by 32/bits threads (4-bit: 8 threads × 4 atomics on the same word) — heavily serialized read-modify-write. ROCm has no byte-aligned read fast path: `read_bits` does 32 per-bit loads for an F32 element before `write_bits` writes whole bytes (~8× instruction multiplier on every copy); CUDA's `copy_value` has the direct path (`copy.cu:184-190`).
- **Evidence:** Cited kernel code; ROCm `write_bits` receives a value already gathered bit-by-bit.
- **Impact:** Device copies of the target quantized formats run at a small fraction of achievable bandwidth; all ROCm copies including F32/BF16 carry the per-bit read penalty.
- **Recommended fix:** One thread (or warp) per aligned word: load source word(s), shift/mask the field, store non-atomically within the owning thread; add the ROCm byte-aligned direct read path mirroring CUDA.
- **Verification method:** On hardware, I4 vs F32 same-device copy bandwidth vs a triad reference; word-oriented kernel brings I4 within ~2× of F32.

**Rejected in this area (adversarial gate):** 16×16 tile padding waste candidate — the reviewer itself recommended no structural change; the tile size is a sanctioned architectural invariant (WMMA-friendly) and standard LLM dims (multiples of 16) pay nothing. Recorded as a design tradeoff, not a defect: worst case 256× storage for 1×1, 16× for [1,N], 3.54× measured for 17×17; at most document the padding rule for callers shaping small tensors.

## 6. Synthesis / overall assessment

### Overall assessment

The architecture is in good shape: common code is genuinely backend-neutral (no leakage, no registry), the queue/token contract is well-specified and transactional submission was recently hardened (changes 18–21), validation discipline is strong in the tensor/view layer, and the local suite is green (3/3, ~31k assertions). The dominant risk themes are: **(1) teardown-without-fence** — queue and tensor destruction do not synchronize in-flight GPU work (ST-001/ST-003), the project's sharpest safety gap; **(2) an under-validated ingestion boundary** — SafeTensors accepts self-contradictory and duplicate inputs on the primary weight path (CC-001/CC-003); **(3) quadruplicated backend scaffolding** that has already drifted once (AR-001/AR-002) and will multiply the cost of fixing theme (1); **(4) the only implemented operation is far off bandwidth everywhere** — 280–2000× on CPU measured (PF-001), launch-explosion and churn patterns on the GPU backends (PF-002/PF-003/PF-006), and a synchronous-in-effect TTNN queue (PF-004). CC-002 (missing mandatory SYCL milestone) is a project-management contract gap, not a code defect.

### Cross-area root causes

- **Duplicated backend queue/copy machinery** (AR-001, AR-002) is the structural amplifier for ST-001/ST-005/ST-006 and PF-002/PF-003/PF-006: each of those fixes must currently be applied 2–4 times, and the drift the duplication predicts is already observable (staging padding, stream choice, publish policy).
- **SafeTensors ingestion boundary** (CC-001, CC-003, AR-009): one module, three findings — validation gaps plus a duplicated store that guarantees the duplicate-key fix lands in two places.
- **Missing physical-storage observability after queued copies** (NT-001) is exactly the harness gap that would catch kernel-level out-of-window writes suspected in the PF-006 rewrite — fix NT-001 before or alongside any kernel rework.

### Residual risks / verification gaps

- All CUDA/ROCm/TTNN runtime claims (ST-001, ST-003, ST-005, ST-006, PF-002…PF-006) are mechanism-verified from source but not hardware-reproduced; ST-001's exact severity depends on busy-event-destroy semantics of the target CUDA/HIP versions.
- ST-006 and ST-007 are hypothesis-grade by design (latent paths).
- The RoPE and llama numerics are scratchpad-adjacent; NT-003 fixes the test so the scratchpad can be fixed later.
- No sanitizer (ASan/TSan) or profiler runs were performed in this review beyond the CPU microbenchmarks and repro programs; GPU sanitizers require the remote hardware workflow.

### Suggested validation sequence

1. **Hardware teardown battery (ST-001/ST-003):** CUDA + ROCm conformance cases destroying queues/tensors with pending copies under Compute Sanitizer / ROCm ASan — settles the two high-severity stability findings.
2. **Local, cheap:** SafeTensors validation fix + tests (CC-001/CC-003), allocator category alignment (CC-004), NT-002 negative-oracle CPU fixture, NT-003 RoPE math — all CPU-runnable today.
3. **CPU blocked-copy rewrite (PF-001)** with the microbenchmark repeated — largest measured win, no hardware needed.
4. **Oracle-backed queued-copy conformance (NT-001) on hardware**, then PF-002/PF-003/PF-006 kernel work with Nsight/ROCprofiler timelines as the acceptance evidence.
5. **Decide SYCL (CC-002):** implement specs 11/12 or descope formally.
