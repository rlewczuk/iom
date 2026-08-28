# Build the shared storage and copy conformance harness

**Order:** 06
**Priority:** P0 — the reusable CPU-reference gate is required before starting any accelerator backend.
**Blocked by:** `05-cpu-storage-transfer-copy`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

A backend-neutral test harness proves storage, transfer, transformed-view, asynchronous-copy, error, and lifetime behavior against the CPU reference and is ready to be instantiated unchanged by each accelerator.

## Scope

- Extract cross-backend observable cases into a shared harness and instantiate it first with two independent CPU devices.
- Add deterministic deferred/error fixtures only for lifecycle and asynchronous behavior that a synchronous round trip cannot prove.
- Keep backend-specific construction and supported-type tables supplied by each backend test, not hidden in global selection logic.

## Implementation references

- **Create (planned):** `test/backend/backend_conformance.hpp` — reusable conformance entry points parameterized by reference device, candidate device, and an explicit supported `DataType` table.
- **Create (planned):** `test/cpu/test_cpu_conformance.cpp` — invoke the shared harness with independently allocated CPU reference and candidate devices and host buffers.
- **Modify:** `test/CMakeLists.txt` — add `iom_backend_conformance_cpu_tests` without folding hardware tests into the generic `iom_tests` executable.
- **Read:** `test/cpu/test_cpu.cpp` — reuse recording allocators and independent encoding helpers; do not move backend-specific allocator setup into the shared harness.
- **Read:** `test/test_iom.cpp` — reuse the independent logical-to-owner-plane reference map for transformed views.

## Requirements

- The harness accepts devices/factories supplied by the invoking backend test. It must not switch on `BackendKind`, construct a global active backend, or include accelerator headers.
- For each supported leaf type, create matching CPU-reference and candidate tensors, seed identical contiguous logical host bytes, exercise host transfers and same-device asynchronous copies, wait on the originating queue, read both results, and compare logical bytes bit-for-bit.
- CPU invocation covers every declared `DataType` with `QuantizationFormat::NONE`.
- Cases cover final-dimension padding, multiple leading ranks including rank greater than four, full tensors, stepped slices, selects, every relevant permutation, contiguous leading reshapes, nested transforms, and rank-two boundaries.
- Copy cases include different source/destination offsets and strides, identical-window no-op submission, metadata mismatch, foreign `Device` instances with the same backend ordinal, and validation before writes.
- Transfer cases use independently generated encodings, inspect logical bytes rather than padding, verify output tail bits, and include valid/invalid `BOOL` data.
- A deterministic deferred fake records the exact `TensorView`, owner, and native-handle addresses at submission and completion; it verifies those addresses remain stable until `wait` and that injected asynchronous failures are rethrown on repeated waits.
- A synchronous transfer failure fixture verifies metadata, owner identity, and native handle remain unchanged even when destination values become unspecified.
- Every successful submission is waited before queue destruction, and an instrumented queue proves destruction performs no implicit synchronization or cancellation.
- Compile-check every compute method's view signature and verify the CPU backend's unsupported compute methods fail before sequence consumption or output changes. Do not add numerical compute expectations.
- The shared harness must not silently skip a candidate because hardware or runtime is absent; backend enablement determines whether its executable is built and run.

## Non-goals

- Re-testing metadata arithmetic already covered by the core unit tests.
- Numerical add, multiply, SILU, linear, RMSNorm, or SDPA conformance.
- Backend discovery, runtime installation, performance thresholds, or implicit host staging.

## Acceptance criteria

- [ ] The CPU-reference invocation passes every shared case for every leaf type.
- [ ] A deliberately perturbed candidate layout or view map causes a bit-for-bit conformance failure rather than round-trip cancellation.
- [ ] The same harness API can be included by a backend-specific test without common-code changes or accelerator headers.
- [ ] Deferred success/failure tests prove stable view/owner/storage addresses, repeatable waits, in-order completion, and explicit queue lifetime.
- [ ] Cross-device, metadata, `BOOL`, span-size, and unsupported-compute failures occur before writes and sequence consumption.
- [ ] No conformance case allocates operands inside transfer, transform, or operation calls.

## Verification

- `cmake -S . -B build/cpu -DBUILD_TESTING=ON`
- `cmake --build build/cpu --target iom_backend_conformance_cpu_tests`
- `ctest --test-dir build/cpu --output-on-failure -R '^iom_backend_conformance_cpu_tests$'`
