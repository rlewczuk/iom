# Zero-fill standard tensor storage before exposure

**Order:** 02
**Priority:** P0 — public ownership/creation invariant
**Blocked by:** None
**Review source:** `cpp-inference-contract-correctness` — whole-codebase review of checked-out main HEAD `ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb` (`Update remote hosts`), clean tree at review start
**Finding:** CC-002
**Review area:** Contract & correctness
**Review severity:** high
**Review verification:** verified, confidence 95
**Review scope:** whole-codebase
**Backend scope:** multi-backend
**Location:** `docs/BACKEND_CONTRACT.md:214-221`; `include/iom/detail/aligned_storage.hpp:20-35`; `src/cpu/device.cpp:397-404`; `src/cuda/device.cpp:170-178`; `src/rocm/device.cpp:121-129`; `src/sycl/device.cpp:142-152`

## Outcome

A standard tensor constructor establishes zero in every byte of its full tiled allocation, including logical elements, padding, and untouched regions, before exposing the Tensor. CPU, CUDA, ROCm/HIP, SYCL, and TTNN creation paths retain their existing allocator ownership and cleanup semantics while making fresh-tensor state deterministic even when caller-provided storage is nonzero-filled.

## Current problem

`docs/BACKEND_CONTRACT.md:214-221` requires a newly created standard allocation and every untouched region to remain zero-filled, but `iom::Allocator` promises allocation/free/alignment rather than zeroed bytes. `detail::allocate_aligned_storage` checks only allocation success and 32-byte alignment. `CpuTensor`, `CudaTensor`, `RocmTensor`, and `SyclTensor` constructors expose the allocated address without clearing the full `tiled_storage_nbytes()` range; backend-native memory requires a backend operation rather than host memset. The root-run CPU probe used a caller allocator that filled every allocation with `0xA5`, created a fresh U8 tensor, and observed immediate logical readback `165`, proving the missing initialization. Existing conformance cases seed storage before observing it and therefore do not protect fresh state. TTNN's native runtime initialization guarantee was not inspected, so it cannot be assumed.

## Scope

- Make zero initialization an explicit postcondition of standard tensor construction for CPU, CUDA, ROCm/HIP, SYCL, and TTNN.
- Clear the entire `view().spec().tiled_storage_nbytes()` allocation, not only logical elements, before construction returns or the Tensor becomes observable.
- Preserve allocator alignment and ownership contracts, existing device/context activation, partial-construction cleanup, error mapping, and native physical extent validation.
- Establish TTNN native-runtime behavior with a fresh-tensor readback/storage check; if the runtime does not guarantee zeroing before exposure, explicitly zero its native planes through the owning runtime before returning the Tensor.

## Implementation references

- **Modify:** `include/iom/detail/aligned_storage.hpp` — `allocate_aligned_storage` only if a narrowly scoped helper is needed for checked size/ownership; do not change the allocator contract or make allocation itself implicitly zeroing.
- **Modify:** `src/cpu/device.cpp` — `CpuTensor` constructor and cleanup; host-fill the complete allocation after aligned allocation and release it on failure.
- **Modify:** `src/cuda/device.cpp` and `src/rocm/device.cpp` — `CudaTensor`/`RocmTensor` constructors; activate the owning context/ordinal, validate the native pointer, and issue checked device memset over the full tiled byte count.
- **Modify:** `src/sycl/device.cpp` — `SyclTensor` constructor; validate USM pointer/context, issue checked queue memset, and wait before exposure.
- **Modify:** `src/ttnn/device.cpp` and its native tensor creation helper; establish and enforce the external runtime's zero guarantee without changing allocator ownership or staging policy.
- **Tests:** `test/cpu/test_cpu.cpp` `RecordingAllocator`/fresh tensor cases and backend storage-oracle/conformance setup; add nonzero allocator, padding, failure-cleanup, and native fresh-readback coverage.

## Requirements

- CPU must clear exactly the full tiled allocation immediately after successful aligned allocation, including physical padding and untouched planes. If the fill fails, release the allocation exactly once and expose no Tensor.
- CUDA and ROCm must validate the native pointer, activate the owning context/ordinal, and perform a checked device memset over `tiled_storage_nbytes()`. Construction must not return until the zeroing operation is complete and visible; failures must follow existing release/quarantine cleanup.
- SYCL must validate that storage belongs to the owning context, issue a checked queue `memset` over the full tiled allocation, and wait before construction returns. A failed memset or wait must release or quarantine according to existing cleanup and expose no partially usable Tensor.
- TTNN creation must include a targeted verification of native `create_device_tensor` initialization. If native storage is not authoritatively zero-filled before exposure, explicitly clear every native plane/physical region before returning; never rely on an unverified SDK default.
- Do not add caller-side clearing, alter `Allocator::alloc/free/reset`, or zero caller storage that is not materialized as a Tensor. Preserve native extent checks, alignment checks, and exactly-once ownership transfer.

## Non-goals

- Do not change logical host encoding, ADD arithmetic, async operation lifetime, model parsing, allocator API beyond the minimum ownership clarification, or TTNN performance/staging design.
- Do not change tensor view semantics or require zeroing storage that a caller owns but the backend never materializes.
- Do not mask initialization errors by exposing a Tensor before zeroing completion or by silently falling back to host memset for device-native storage.

## Acceptance criteria

- [ ] With a CPU allocator that fills new blocks with `0xA5`, a fresh standard tensor's immediate logical readback is zero for byte-aligned and sub-byte leaves, and a storage oracle confirms all padding/untouched bytes are zero.
- [ ] Fresh CUDA, ROCm, and SYCL tensors read back all-zero logical values before any host write; backend-native zeroing completes before construction returns and physical padding is zero where the storage oracle can observe it.
- [ ] Injected allocation, pointer-validation, memset, or wait failure releases/quarantines the block exactly once, returns the creation API's documented error, and exposes no usable Tensor.
- [ ] TTNN's native fresh-tensor test either proves the runtime zero guarantee or observes explicit plane clearing; no acceptance depends on an assumed external default.
- [ ] Existing allocator alignment, native extent, view, and ownership tests remain unchanged in behavior, and no caller-side clearing is required.

## Verification

- `cmake -S . -B build/review-cpu -G Ninja -DBUILD_TESTING=ON && cmake --build build/review-cpu --target iom_cpu_tests iom_cpu_conformance_tests && ctest --test-dir build/review-cpu --output-on-failure -R '^(iom_cpu_tests|iom_cpu_conformance_tests)$'` — nonzero allocator fresh-tensor and padding oracle cases are all zero; failure seams free exactly once.
- `(remote-development: CUDA host)` sync the workspace, then run the README CUDA configure/build commands and `ctest --test-dir build --output-on-failure -R '^iom_cuda_conformance_tests$'` — native fresh logical/padding readback is zero and injected cleanup preserves ownership.
- `(remote-development: ROCm host)` sync the workspace, then run the README ROCm configure/build commands and `ctest --test-dir build --output-on-failure -R '^iom_rocm_conformance_tests$'` — device memset is checked and fresh storage is zero.
- `(remote-development: SYCL host)` sync the workspace and run the configured oneAPI/icpx conformance target with fresh USM tensors — checked queue memset plus wait makes all logical and observable padding bytes zero before return; failed wait does not expose storage.
- `(remote-development: TTNN host)` run `cmake --build build --target iom_ttnn_conformance_tests && ctest --test-dir build --output-on-failure -R '^iom_ttnn_conformance_tests$'` with the native fresh-tensor scenario — native zero guarantee is verified or explicit plane clearing is observed.
