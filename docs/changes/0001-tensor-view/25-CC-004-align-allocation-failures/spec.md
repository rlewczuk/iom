# Align LinearAllocator exhaustion with the Allocator family

**Order:** 25
**Priority:** P1 — bounded allocator error-category compatibility defect without broad gating
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `CC-004`
**Review severity:** low
**Review verification:** verified, confidence 85

## Outcome

Exhausting a `LinearAllocator` throws `std::bad_alloc`, matching `ListAllocator`, `FixedSizeAllocator`, and the tensor-storage exhaustion contract. A standard-layout device's `create_tensor` propagates that category unchanged, so callers can handle allocation exhaustion consistently regardless of which built-in allocator supplies storage.

## Current failure

`LinearAllocator::alloc` currently throws `std::runtime_error("LinearAllocator is out of memory")` when the next aligned allocation does not fit in its buffer. The sibling `ListAllocator::alloc` and `FixedSizeAllocator::alloc` exhaustion paths throw `std::bad_alloc`, and the tensor contract models allocation exhaustion as `std::bad_alloc`. Consequently, a caller that catches `std::bad_alloc` misses exhaustion when a CPU, CUDA, ROCm, or SYCL tensor is constructed with an exhausted `LinearAllocator`; each of those standard-layout backends propagates the allocator exception from `create_tensor` without translation. This is an error-category compatibility defect, not a memory-safety failure.

## Scope

- Change only the exhaustion category emitted by `LinearAllocator::alloc` when the requested allocation cannot fit after applying its configured alignment.
- Pin direct allocator parity across `LinearAllocator`, `ListAllocator`, and `FixedSizeAllocator`, and pin unchanged propagation through CPU `Device::create_tensor` using an actually exhausted `LinearAllocator`.
- Preserve all existing allocation, alignment, reset, free, ownership, and non-exhaustion error behavior. The common change naturally applies to every standard-layout backend that receives an `Allocator`; no accelerator-specific implementation change is required.

## Implementation references

- **Modify:** `src/alloc.cpp` — `LinearAllocator::alloc`; replace its out-of-memory `std::runtime_error` with the same `std::bad_alloc` category already used by the sibling allocators.
- **Read:** `src/alloc.cpp` — `ListAllocator::alloc` and `FixedSizeAllocator::alloc`; reuse their established exhaustion exception convention without changing their behavior.
- **Read:** `include/iom/alloc.hpp` — `Allocator`, `LinearAllocator`, `ListAllocator`, and `FixedSizeAllocator`; keep the public interface and allocator family structure unchanged.
- **Tests:** `test/test_alloc.cpp` — `LinearAllocator throws when out of memory`, `ListAllocator rejects invalid frees and out-of-memory allocation`, and `FixedSizeAllocator rejects invalid frees and exhausted allocations`; update the Linear assertion and retain the sibling parity assertions.
- **Tests:** `test/cpu/test_cpu.cpp` — CPU tensor allocation tests around `CPU tensors reject null and misaligned allocations exactly once` and `CPU create_tensor propagates allocator failures without freeing`; add the focused real-`LinearAllocator` exhaustion case beside these ownership and propagation tests.
- **Read:** `src/cpu/device.cpp` — `CpuTensor::CpuTensor` and `CpuDevice::create_tensor`; confirm allocator-thrown exceptions propagate unchanged and no failed allocation is freed.
- **Read:** `src/cuda/device.cpp` — `CudaTensor::CudaTensor`, `src/rocm/device.cpp` — `RocmTensor::RocmTensor`, and `src/sycl/device.cpp` — `SyclTensor::SyclTensor`; these are the other allocator-consuming tensor constructors and must continue to propagate the common allocator category without backend wrappers.
- **Read:** `docs/changes/0001-tensor-view/spec.md` — sections 7 and 11.4; preserve the contract that tensor allocation exhaustion is `std::bad_alloc` and allocator exceptions propagate after cleanup of resources actually acquired.
- **Read:** `test/CMakeLists.txt` — `iom_tests` and `iom_cpu_tests`; use the existing focused core allocator and CPU tensor test targets rather than adding a target.

## Requirements

- `LinearAllocator::alloc(sz)` must throw `std::bad_alloc` whenever its existing fit check determines that the aligned start is beyond the buffer end or fewer than `sz` bytes remain.
- The failed call must leave the allocator cursor unchanged. A subsequent `reset()` must retain its existing behavior and permit the buffer to be reused from its normalized beginning.
- Do not catch, wrap, or translate the `std::bad_alloc` in `CpuDevice::create_tensor`, `CpuTensor`, or any other backend. `create_tensor` must surface the allocator exception directly and must not call `Allocator::free` when `alloc` itself throws.
- Update the existing direct Linear exhaustion assertion in `test/test_alloc.cpp` from `std::runtime_error` to `std::bad_alloc`. Keep the existing List and FixedSize exhaustion assertions as parity coverage.
- Add a CPU regression in `test/cpu/test_cpu.cpp` that supplies `make_cpu_device` with a 32-byte-aligned `LinearAllocator`, successfully consumes its storage with one live tensor, and verifies that a second valid tensor requiring more storage throws `std::bad_alloc` from `create_tensor`. The test must keep the first tensor alive until after the failed construction so `LinearAllocator::free` cannot affect the exhaustion setup.
- The CPU regression must also demonstrate that the failed tensor construction returns no tensor and does not change the existing first tensor's storage or ownership. No special cleanup assertion is required beyond normal destruction because `LinearAllocator::free` is intentionally a no-op.
- Preserve `LinearAllocator`'s existing bump-pointer addresses, configured alignment, zero-size allocation behavior, no-op `free`, and `reset` semantics.
- Preserve every other exception category and diagnostic: invalid allocator construction remains `std::invalid_argument`; invalid List/FixedSize frees and index operations remain unchanged; FixedSize metadata corruption and tensor misalignment remain `std::runtime_error`; a null allocation returned by a custom allocator remains a tensor-side `std::bad_alloc`. Do not introduce a replacement custom exhaustion message or alter unrelated messages.
- Preserve propagation of arbitrary exceptions from caller-defined `Allocator` implementations. In particular, the existing injected `std::runtime_error` case in `CPU create_tensor propagates allocator failures without freeing` must continue to pass; this task aligns the built-in Linear exhaustion path rather than normalizing all allocator exceptions inside tensor construction.

## Non-goals

- Changing `Allocator::alloc`'s signature, documenting a new allocator-wide exception policy in the public header, or adding a status/result abstraction.
- Altering `ListAllocator` or `FixedSizeAllocator` allocation algorithms, zero-size semantics, free validation, diagnostics, or exhaustion behavior.
- Changing tensor allocation size, 32-byte alignment enforcement, allocation/free counts, backend resource handling, or tensor ownership and lifetime rules.
- Adding accelerator-specific tests or modifying CUDA, ROCm, SYCL, or TTNN code. TTNN uses native materialized storage rather than the injected `Allocator` and is outside this finding.
- Translating every exception from a custom `Allocator` to `std::bad_alloc` or changing non-exhaustion `std::runtime_error` paths.

## Acceptance criteria

- [ ] Direct allocation past the end of a `LinearAllocator` throws `std::bad_alloc`, just as exhausted `ListAllocator` and `FixedSizeAllocator` allocations do.
- [ ] The existing direct allocator suite continues to verify unchanged alignment, addresses, zero-size behavior, free/reset behavior, invalid-argument categories, and sibling allocator exhaustion categories.
- [ ] With one CPU tensor holding all available storage from a `LinearAllocator`, a second valid `create_tensor` call throws `std::bad_alloc`; the first tensor remains alive and valid through the failed call.
- [ ] CPU tensor construction does not wrap the Linear exception, does not return a tensor, and does not attempt to free storage for the failed allocation.
- [ ] The existing custom-allocator propagation test still observes its injected `std::runtime_error`, proving that only built-in Linear exhaustion was reclassified.
- [ ] No public API, allocator allocation behavior, exception category outside Linear exhaustion, or exception message outside the removed Linear exhaustion diagnostic changes.

## Verification

- `cmake -S . -B build/cc004 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF`
- `cmake --build build/cc004 --target iom_tests iom_cpu_tests`
- `./build/cc004/test/iom_tests --test-case="LinearAllocator*,ListAllocator rejects invalid frees and out-of-memory allocation,FixedSizeAllocator rejects invalid frees and exhausted allocations"` — all selected cases pass, with direct exhaustion reported as `std::bad_alloc` for all three built-in allocators.
- `./build/cc004/test/iom_cpu_tests --test-case="*create_tensor*allocator*,CPU tensors reject null and misaligned allocations exactly once"` — the new exhausted-Linear case observes `std::bad_alloc`, while the existing injected custom `std::runtime_error`, null-return, misalignment, and no-free-on-thrown-allocation behaviors remain unchanged.
- `ctest --test-dir build/cc004 --output-on-failure -R '^(iom_tests|iom_cpu_tests)$'`
