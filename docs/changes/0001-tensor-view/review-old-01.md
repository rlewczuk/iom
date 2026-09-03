# Code Quality Review

## Review metadata

This review is outdated, please ignore it.

- **Scope:** whole-codebase, centered on the completed behavior claimed by `0001-tensor-view`
- **Target commit:** n/a
- **Baseline:** n/a
- **Specification:** `docs/changes/0001-tensor-view` (root specification plus ordered sub-specifications 01–16)
- **Backends considered:** CPU, CUDA, ROCm/HIP, SYCL, TTNN
- **Review coverage:** Exhaustive review of the public tensor/device/queue interfaces, core metadata/view/token implementation, CPU/CUDA/ROCm/TTNN factories, storage, transfer and copy paths, top-level/test CMake, shared conformance harness, backend drivers, and relevant focused tests. Safetensors dtype mapping and tensor-view model call sites were checked. `include/iom/llama.hpp` marks the model layer as a buggy scratchpad and explicitly requests omission from review, so its mathematical/model behavior was excluded. SYCL has specifications and an enum value but no implementation to inspect.
- **Validation performed:** CPU-only CMake configure and full build succeeded with GCC 15.2.0; CTest passed `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests` (3/3, 8.91 s). Configuring with `-DSYCL_ENABLED=ON` succeeded only with CMake's warning that `SYCL_ENABLED` was unused. A compile-only custom `Device` fixture proved that a concrete subclass is currently copy- and move-constructible. Source inspection found no remaining `hipCtx*` or `hipDevicePrimaryCtx*` use under `src/rocm`, `include/iom/rocm`, or `test/rocm`. Accelerator builds, hardware tests, device sanitizers, and profilers were not run.

## 1. Contract & correctness

### CC-001 — TTNN narrows and wraps dimensions before native allocation

- **Severity:** high
- **Verification:** verified
- **Confidence:** 98
- **Scope relation:** whole-codebase
- **Backend scope:** ttnn
- **Location:** `src/ttnn/device.cpp:132-157,390-397`
- **Invariant:** Shape, product, and backend-index conversions must be validated for overflow before work or allocation; a successfully created tensor must materialize storage matching its unchanged `std::size_t` logical metadata.
- **Failure mode:** `TtnnDevice::create_tensor` calls only `spec.validate()`. `TtnnTensor` then narrows the final dimensions with unchecked `static_cast<std::uint32_t>` and computes the leading-plane count with unchecked `plane_count *= dimensions[i]`. On a 64-bit host, supported `BF16` shape `{2^63, 2, 16, 16}` wraps the plane count to zero and returns a tensor with an empty native-plane vector. Shape `{1, 2^32 + 1}` retains that logical column count but creates a native tensor with one column; a later upload copies the logical row into storage sized from the truncated native shape (`src/ttnn/copy.cpp:95-110`).
- **Evidence:** Common `Tensor` construction calls `TensorSpec::validate()` and computes only leading strides (`src/iom.cpp:273-276,496-498`); it does not force `shape.element_count()`. TTNN's creation path likewise does not call an overflow-checking size method. The narrowing and unchecked plane product are direct at `src/ttnn/device.cpp:139-152`.
- **Impact:** Tensor creation can report success with no native storage or undersized native planes. Copies can silently do no work after a wrapped plane count; a narrowed matrix dimension can drive host-buffer overwrite or invalid native access. The logical metadata continues to advertise the original larger shape.
- **Recommended fix:** Before constructing any TTNN native object, call the checked common element-count path, compute the leading-plane count with checked multiplication, validate both tiled dimensions against `std::uint32_t` and TTNN runtime limits, and use checked casts. Reject unsupported native extents with the established exception category before allocation.
- **Verification method:** On TTNN hardware, assert that `{2^63, 2, 16, 16}` throws `std::overflow_error` with zero native plane allocations and that `{1, 2^32 + 1}` is rejected before native `TensorSpec` construction. Retain normal boundary cases at the largest accepted TTNN extent.

## 2. C++/GPU stability

### ST-001 — ROCm sub-byte host read writes past its staging allocation

- **Severity:** high
- **Verification:** verified
- **Confidence:** 99
- **Scope relation:** whole-codebase
- **Backend scope:** hip
- **Location:** `src/rocm/copy.hip:127-148,263-305`
- **Invariant:** Every device access must remain within the allocated object, including the word-sized atomic operations used to assemble sub-byte host output.
- **Failure mode:** `synchronous_transfer` allocates exactly `logical_nbytes()` bytes. During `copy_to_host`, `gather_plane_kernel` calls `write_bits`, which updates every sub-byte field through a 32-bit `atomicOr`/`atomicAnd` at `base + (bit / 32) * 4`. For shape `{1,17}` with `I2`, the logical allocation is five bytes; the final element starts at bit 32, so the atomic at byte offset four accesses bytes four through seven, three bytes beyond the allocation.
- **Evidence:** The shared ROCm type table includes `I2/U2/I4/U4/F4/F6` and the transfer matrix includes `{1,17}` (`test/rocm/test_rocm_conformance.cpp:24-37`; `test/backend/backend_conformance_copy_storage.hpp:169-177`). CUDA explicitly adds extra word space to its staging allocation (`src/cuda/copy.cu:322-329`); the corresponding ROCm allocation at `src/rocm/copy.hip:273` does not.
- **Impact:** Sub-byte ROCm host reads perform an out-of-bounds device read-modify-write. Depending on allocator granularity and runtime checking, this can corrupt adjacent state, raise a device fault, or remain latent while still violating the memory contract.
- **Recommended fix:** Use checked round-up to allocate and zero staging through the end of the last 32-bit atomic word, while copying only `logical_nbytes()` to the host. Preferably replace per-bit atomics with a packed-byte writer whose ownership granularity cannot cross the logical buffer bound.
- **Verification method:** Run the `{1,17}` `I2` and odd-length `F6` `copy_to_host` cases under supported HIP device AddressSanitizer, then verify exact bytes and zero tail bits. The current kernel must report the word access beyond the five-byte allocation; the corrected path must be clean.

### ST-002 — CUDA and ROCm can orphan submitted kernels without returning an oid

- **Severity:** high
- **Verification:** strongly-supported
- **Confidence:** 96
- **Scope relation:** whole-codebase
- **Backend scope:** cuda, hip
- **Location:** `include/iom/iom.hpp:83-100`; `src/cuda/copy.cu:426-446,493-500`; `src/rocm/copy.hip:364-383,452-458`
- **Invariant:** Once backend work can reference operand storage, the submission must have a waitable token or must be synchronously drained before an exception escapes. Synchronous submission failure must consume no sequence and modify no destination.
- **Failure mode:** The `DeviceOps::submit` template commits a sequence only after its callback returns. CUDA and ROCm callbacks launch one or more kernels and record an event before inserting the event into `staged_`. A later launch, event-record call, or `staged_.push_back` allocation can throw after earlier kernels are already queued. The callback then destroys the event and rethrows; `submit` does not advance the sequence and `copy()` returns no `oid`, but the stream can still reference source and destination storage. A separate `tasks_.push_back` allocation can also throw in `publish_staged` after the common sequence has been committed, again preventing the local token from reaching the caller.
- **Evidence:** Kernel launch precedes task publication in both backends. Their catch blocks destroy only the event and do not synchronize or otherwise retain the in-flight work. The common submit callback has no “work became externally visible” state. Shared lifetime/error tests use `DeferredCopyQueue`, not the concrete accelerator worker and publication path (`test/backend/backend_conformance_other.hpp:216-290`).
- **Impact:** After a reported synchronous failure, callers have no token requiring the operands to remain alive. Destruction or allocator reuse can race orphaned GPU writes, causing use-after-free, corruption, or an error attributed to a later unrelated operation. Post-commit publication failure also breaks the promised no-consumption rule.
- **Recommended fix:** Allocate and link all host task-tracking state before the first backend enqueue, then make publication non-throwing. Redesign the common submission boundary so any failure after the first possible enqueue is represented by a committed token and retained completion/failure, or synchronously drain the stream before throwing. Do not discard an event that is the only lifetime fence for queued work.
- **Verification method:** Add injectable CUDA/HIP wrappers and allocator fault points. Fail a later plane launch, event record, `staged_` insertion, and `tasks_` insertion after at least one successful enqueue. Each case must either return an `oid` whose repeated waits report the failure or prove the stream was drained before throwing; operand canaries must remain untouched after destruction/reuse.

### ST-003 — Device ownership is copyable and movable through the public base

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 98
- **Scope relation:** whole-codebase
- **Backend scope:** common
- **Location:** `include/iom/device.hpp:20-30`
- **Invariant:** Repository ownership rules require owners to be non-copyable and non-movable; tensors retain the address of their creating `Device`, which must remain stable and outlive them.
- **Failure mode:** `Device` declares only a virtual destructor and virtual operations, leaving copy construction and assignment available. A concrete user subclass with no non-copyable member is consequently copy- and move-constructible. Moving such a device does not retarget existing tensors' stored `Device*`; destroying the original object leaves their views dangling. Copying a subclass that owns a raw runtime handle can also duplicate ownership and teardown.
- **Evidence:** A compile-only concrete fixture implementing the four virtual methods passed `static_assert(std::is_copy_constructible_v<D>)` and `static_assert(std::is_move_constructible_v<D>)`. By contrast, `Tensor` and `DeviceOps` explicitly delete all copy/move operations. The repository invariant is explicit at `AGENTS.md:15-20`.
- **Impact:** The public subclassing surface permits exactly the owner relocation/duplication that stable tensor and asynchronous queue addresses are designed to prevent. Failures present as dangling device access or duplicate runtime-context release.
- **Recommended fix:** Delete copy and move construction and assignment on `Device` itself. Add compile-time traits for the custom-device fixture and each public owner type.
- **Verification method:** Require `!std::is_{copy,move}_{constructible,assignable}_v<ConcreteFixture>` in `test/test_iom.cpp`; all backend factories and existing tensor/view lifetime tests must continue to compile and pass.

### ST-004 — CUDA primary-context retain leaks when activation fails

- **Severity:** medium
- **Verification:** verified
- **Confidence:** 97
- **Scope relation:** whole-codebase
- **Backend scope:** cuda
- **Location:** `src/cuda/device.cpp:188-198`
- **Invariant:** Every successfully acquired backend resource must be released on every later construction failure.
- **Failure mode:** `make_cuda_device` successfully calls `cuDevicePrimaryCtxRetain`, then calls `cuCtxSetCurrent` before entering the `try` block that releases the primary context on failure. If activation fails, the function propagates immediately and never balances the retain.
- **Evidence:** The only `cuDevicePrimaryCtxRelease` failure cleanup is in the catch beginning after `cuCtxSetCurrent`; the normal owner has not yet been constructed at that point.
- **Impact:** A failed factory call leaks a primary-context retain, keeping CUDA process state and associated resources alive. Repeated failed construction can accumulate unmatched retains and make later teardown/configuration behavior nondeterministic.
- **Recommended fix:** Wrap the retained `(CUdevice, CUcontext)` in a local RAII guard immediately after `cuDevicePrimaryCtxRetain`; dismiss the guard only after `CudaDevice` takes ownership. Keep activation inside that guarded scope.
- **Verification method:** Through an injectable CUDA driver seam, make retain succeed and `cuCtxSetCurrent` fail; assert exactly one matching `cuDevicePrimaryCtxRelease` before the original exception propagates. Also cover `CudaDevice` allocation failure after successful activation.

## 3. Backend architecture & simplicity

### AR-001 — The required SYCL backend and build option are absent

- **Severity:** high
- **Verification:** verified
- **Confidence:** 100
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `CMakeLists.txt:16-21,63-165`; `test/CMakeLists.txt:36-128,152-267`
- **Invariant:** The change requires CPU plus four independently selectable accelerator libraries and factories, with every enabled combination represented in conformance and coexistence tests.
- **Failure mode:** `BackendKind::SYCL` and complete scaffold/storage specifications exist, but there is no `SYCL_ENABLED` option, public factory header, `src/sycl` implementation, SYCL library, smoke/conformance driver, or coexistence branch. Passing `-DSYCL_ENABLED=ON` configures the ordinary CPU build and reports that the variable was unused.
- **Evidence:** The top-level specification requires an independent SYCL library and factory (`docs/changes/0001-tensor-view/spec.md:428-439,550-558,653-669`), with concrete milestones in `11-sycl-buildable-scaffold` and `12-sycl-storage-copy`. `include/iom/tensor.hpp:74-80` exposes the enum value. Current CMake declares only CUDA, ROCm, and TTNN options; the configure probe emitted `Manually-specified variables were not used by the project: SYCL_ENABLED`.
- **Impact:** The claimed backend matrix and completion criteria are false. A requested SYCL build silently produces no SYCL target, and the all-backend coexistence test cannot construct or exercise the fourth accelerator.
- **Recommended fix:** Implement specs 11 and 12 as written: backend-neutral `make_sycl_device`, isolated `iom_sycl` target and dependency discovery, owned context/device, standard-layout allocation/transfers/copy, hardware smoke and shared conformance drivers, and SYCL participation in coexistence. A requested but unavailable SDK must fail configuration.
- **Verification method:** Run the specified SYCL-only smoke/conformance configuration, then an all-backend configuration that links and exercises CPU, CUDA, ROCm, SYCL, and TTNN in one process. Confirm disabling SYCL removes only its target, symbols, dependencies, and tests.

## 4. Numerical correctness & tests

### NT-001 — Accelerator layout conformance can pass through matching pack/unpack errors

- **Severity:** medium
- **Verification:** strongly-supported
- **Confidence:** 93
- **Scope relation:** whole-codebase
- **Backend scope:** multi-backend
- **Location:** `test/backend/backend_conformance_copy_storage.hpp:270-324,387-435`; `test/backend/backend_conformance_common.hpp:137-166`
- **Invariant:** Standard-layout backends must be checked against an oracle independent of both their write and read mappings; the specification explicitly requires a deliberately perturbed candidate layout or view map to fail rather than cancel through a round trip.
- **Failure mode:** The shared hardware suite writes expected logical bytes through the candidate's `copy_from_host` and observes them through that same candidate's `copy_to_host`. A consistently wrong plane map, tile map, sub-byte packing, or endian transform in both directions returns the original logical bytes and passes. Async-copy cases seed and read through the same transfer pair, so using the same wrong map in copy also cancels. CPU has direct physical-layout tests, but CUDA/ROCm/TTNN drivers provide no independent raw/native storage oracle.
- **Evidence:** `require_logical_bytes` always calls the candidate view's own `copy_to_host`. The CPU-only perturbation test (`test/cpu/test_cpu_conformance.cpp:167-212`) flips one byte of the correct CPU allocation; it does not instantiate a consistently perturbed accelerator mapping. CUDA and ROCm use their backend `view_planes` calculation for both scatter and gather; TTNN uses `owner_plane_at` for both directions.
- **Impact:** A backend can pass every shared logical conformance case while violating the mandated physical standard layout or TTNN logical-plane mapping. The defect appears later when a compute kernel or external native operation follows the specified mapping instead of the matching erroneous transfer implementation.
- **Recommended fix:** Give each backend driver an independent storage observer/seeder. CUDA and ROCm can copy the full native allocation as raw tiled bytes and compare it with a test-side standard-layout encoder; TTNN needs an independent native-plane adapter. Add a negative fixture whose read and write paths share the same deliberately permuted map and require the harness to fail.
- **Verification method:** Run the negative fixture with an identical nontrivial permutation in both candidate transfer directions; the current logical round trip passes, while the corrected physical/native oracle must identify the first wrong slot or plane. Then run the same oracle over all leaf widths, padded shapes, and transformed views.

## 5. Performance

No material findings.

The reviewed transfer paths contain plausible setup-cost hypotheses—per-call CUDA/ROCm staging allocation, per-call ROCm stream creation, TTNN per-plane host materialization, and TTNN queue-wide serialization—but the repository has no representative benchmark, baseline, performance budget, or profile. None is stated as an established regression. No hot-path performance simplification is recommended without measurement.

## 6. Synthesis / overall assessment

### Overall assessment

The CPU/core tensor-view implementation is coherent and its local and shared CPU suites pass, 
but the current tree is not an acceptable completion of `0001-tensor-view`. 
One required backend is wholly absent. The implemented accelerator paths include a deterministic ROCm out-of-bounds access, 
non-transactional CUDA/ROCm submission failure paths, and unchecked TTNN native-size conversion. 
These are contract and lifetime defects, not style concerns.

### Cross-area root causes

1. Backend-native bounds and ownership are validated after—or not at all before—narrowing, allocation, and enqueue side effects.
2. Common sequence commitment and backend task/event publication form no single transactional submission boundary.
3. The public ownership model is documented but not fully encoded in `Device`'s type traits.
4. Accelerator conformance treats matching logical round trips as sufficient evidence for an exact physical/native mapping.
5. The build matrix exposes SYCL in the contract and enum without implementing the backend boundary.

### Residual risks / verification gaps

- CUDA, ROCm, TTNN, and all-backend hardware builds/tests were not executed; no remote accelerator, device sanitizer, or profiler evidence was collected.
- Real-backend asynchronous failure injection is absent; the shared lifetime/error fixtures primarily verify the common base through fakes.
- TTNN runtime shape limits and native copy semantics were reviewed from call sites but not checked against the installed SDK implementation at runtime.
- No performance benchmark exists for model loading, host transfer, same-device copy, prompt processing, or token generation.
- The Llama scratchpad was intentionally excluded per `include/iom/llama.hpp:8-10`.

### Suggested validation sequence

1. Fix ST-001 and run odd-byte sub-byte ROCm host reads under device memory checking.
2. Make submission publication transactional, then fault-inject every post-enqueue CUDA/HIP failure point and verify token/lifetime behavior.
3. Reject TTNN overflow/narrowing cases before native calls and run the supported-type conformance suite.
4. Add independent accelerator storage/native-plane oracles and prove a consistently perturbed map fails.
5. Enforce non-copyable/non-movable `Device` ownership and verify CUDA retain cleanup through injected failures.
6. Implement and run the specified SYCL smoke, conformance, and five-backend coexistence matrix.
