# Reject allocator arithmetic overflow during construction

**Order:** 07
**Priority:** P1 — turn representable allocator overflow into typed construction errors
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase review of main at fb88959fb884b9bb495e055b9fdd0756e430dbde (clean working tree)`
**Finding:** `CC-001`
**Review area:** Contract & correctness
**Review severity:** medium
**Review verification:** verified, confidence 99
**Review scope:** whole-codebase
**Backend scope:** common; cpu; cuda; rocm; sycl (TTNN does not use `iom::Allocator`)
**Location:** `src/alloc.cpp:22-36` (`SingleBufferAllocatorBase`), `src/alloc.cpp:229-232` (`FixedSizeAllocator`), `src/alloc.cpp:326-328` (`align_up_payload`)

## Outcome

Allocator construction either establishes a valid bounded aligned range and nonzero stride or deterministically throws a typed overflow/argument exception. No representable input can wrap address, size, alignment, or payload arithmetic into zero capacity/stride or an invalid range before validation.

## Current problem

The invariant is that caller-supplied allocator construction and tensor storage allocation establish a valid bounded aligned buffer/stride or fail with a typed error. `SingleBufferAllocatorBase` forms `raw_end_ = raw_begin_ + size` at `src/alloc.cpp:22-25` and computes aligned `begin_` through `align_up_addr` at `:31-36`; `align_up_addr` adds `mask` at `:41-44` without checking `uintptr_t` overflow. `FixedSizeAllocator` initializes `stride_` before its body at `:218-222`, computes `block_count_ = capacity() / stride_` at `:224-229`, and `align_up_payload` at `:326-328` uses `(payload_size + alignment - 1)` unchecked.

On the reviewed 64-bit target, `FixedSizeAllocator(buffer, size, 32, SIZE_MAX)` wraps `SIZE_MAX + 31`, aligns to zero, and divides by zero, terminating with status 136 instead of reporting bad configuration. A buffer range or align-up near the address-space ceiling can similarly wrap and make capacity/bounds checks operate on an invalid interval. The public allocator contract in `include/iom/alloc.hpp:9-13,22-35,87-110` does not constrain these maxima, and standard CPU/CUDA/ROCm/SYCL callers inherit the unchecked constructor.

## Scope

- Centralize checked range and alignment arithmetic in allocator-owned helpers.
- Validate null buffer/size and power-of-two alignment before deriving range state; preserve `std::invalid_argument` for invalid arguments.
- Checked-add raw range and address align-up, throwing `std::overflow_error` on `uintptr_t` overflow.
- Replace payload alignment with a checked helper that rejects zero payload as invalid and payload values exceeding `SIZE_MAX - (alignment - 1)` as overflow before `stride_` initialization.
- Keep dependent capacity/index arithmetic based on validated nonzero range and stride.

## Implementation references

- **Modify:** `src/alloc.cpp:22-44` — `SingleBufferAllocatorBase` constructor and `align_up_addr`; own checked raw-end and address alignment.
- **Modify:** `src/alloc.cpp:218-232,326-328` — `FixedSizeAllocator` constructor and `align_up_payload`; ensure checked stride construction before division.
- **Read:** `include/iom/alloc.hpp:9-13,22-35,87-110` and `include/iom/detail/aligned_storage.hpp:18-37` — public argument/error contract and caller-side limits; do not duplicate validators in backends.
- **Tests:** `test/test_alloc.cpp` — add constructor overflow/range cases beside ordinary invalid-argument and exhaustion coverage; backend tensor-construction smoke/conformance remains a compatibility check.

## Requirements

- Check `raw_begin + size` and every address align-up before storing derived range state; throw `std::overflow_error` on wrap.
- Check payload alignment before the `FixedSizeAllocator` member initializer can produce zero stride; reject zero payload and invalid alignment with `std::invalid_argument`, and payload addition overflow with `std::overflow_error`.
- Use one allocator-owned checked implementation rather than backend-specific or duplicated ad-hoc overflow guards.
- Preserve alignment, capacity, block counts, allocation/free behavior, and typed errors for all valid existing inputs.

## Non-goals

- Do not alter TTNN native allocation, `TensorSpec` layout arithmetic, allocator threading semantics, free/reset policy, or backend runtime ownership.
- Do not add backend-specific range validators or change allocator API shape beyond typed construction errors.
- Do not change valid allocation layout or exhaustion semantics.

## Acceptance criteria

- [ ] With a normal aligned buffer, `FixedSizeAllocator(buffer.data(), buffer.size(), 32, SIZE_MAX)` throws `std::overflow_error` before division or state exposure rather than terminating.
- [ ] Synthetic address/size and align-up operands that exceed `UINTPTR_MAX` throw `std::overflow_error` before `capacity()` is usable.
- [ ] Zero payload and invalid alignment continue to throw `std::invalid_argument`; all valid existing allocator cases preserve alignment, capacity, block counts, and allocation behavior.
- [ ] CPU/CUDA/ROCm/SYCL tensor construction continues to use the common checked allocator without backend-specific guards.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — local common/CPU gates pass, including the focused allocator cases.
- `.agents/skills/remote-development/scripts/remote-sync cuda 07-CC-001-allocator-checked-constructor-arithmetic`
- `.agents/skills/remote-development/scripts/remote-exec cuda 07-CC-001-allocator-checked-constructor-arithmetic 'cmake -S . -B build-cuda -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-cuda --target iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-cuda --output-on-failure -R "iom_cuda_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — CUDA tensor-construction compatibility passes.
- `.agents/skills/remote-development/scripts/remote-sync rocm 07-CC-001-allocator-checked-constructor-arithmetic`
- `.agents/skills/remote-development/scripts/remote-exec rocm 07-CC-001-allocator-checked-constructor-arithmetic 'cmake -S . -B build-rocm -DCUDA_ENABLED=OFF -DROCM_ENABLED=ON -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-rocm --target iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-rocm --output-on-failure -R "iom_rocm_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — ROCm compatibility passes.
- `.agents/skills/remote-development/scripts/remote-sync sycl 07-CC-001-allocator-checked-constructor-arithmetic`
- `.agents/skills/remote-development/scripts/remote-exec sycl 07-CC-001-allocator-checked-constructor-arithmetic 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake -S . -B build-sycl -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-sycl --target iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-sycl --output-on-failure -R "iom_sycl_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — SYCL devices enumerate and compatibility passes. The direct SIZE_MAX reproducer must now throw `std::overflow_error`, not signal 136.
