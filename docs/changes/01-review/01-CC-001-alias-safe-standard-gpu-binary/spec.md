# Make standard-GPU exact aliases race-free

**Order:** 01
**Priority:** P0 — silent accepted-operation corruption
**Blocked by:** None
**Review source:** `cpp-inference-contract-correctness` — whole-codebase review, branch `main`, HEAD `cb88619865ff9bcda0a4e3bac4bae4bd4bf510a8` (`Move cpp-inference-review skill to .omp`), baseline `origin/main`; clean at scope capture; upstream `origin/main` ahead 4/behind 0
**Finding:** CC-001
**Review area:** Contract & correctness
**Review severity:** high
**Review verification:** strongly-supported, confidence 85
**Review scope:** whole-codebase
**Backend scope:** common alias contract, CUDA, ROCm
**Location:** `src/device_ops_binary.cpp` — `DeviceOps::validate_binary`/`exact_alias`; `src/shared/standard_tiled_add.inl` — `add_load_bits` and `binary_body`; `src/shared/gpu_queue_operations.inl` — standard GPU binary dispatch

## Outcome

An admitted exact in-place CUDA or ROCm binary operation produces the same complete logical raw result as an independent operation, including fields crossing 32-bit words, while disjoint output retains the existing fast path and no external binary workspace is introduced.

## Current problem

The common validator intentionally admits an input/output relationship only when it is an exact same-owner, unbroadcasted alias with matching specification, plane offset, strides, logical mapping, and broadcast flags (`src/device_ops_binary.cpp:291-308`). The shared CUDA/ROCm kernel nevertheless reads and writes one 32-bit output word at a time: `add_load_bits` reads the current word and the next word for a field crossing a word boundary, then `binary_body` stores the merged output word (`src/shared/standard_tiled_add.inl:153-166, 191-220, 237-241`). CUDA and ROCm include this same kernel (`src/cuda/copy.cu:62-68`, `src/rocm/copy.hip:62-68`) and dispatch captured input/output handles directly (`src/shared/gpu_queue_operations.inl:127-179`). When output aliases `lhs` or `rhs`, an earlier word store can change a source word before a later work item captures the complete field. F6 fields crossing a 32-bit boundary and F64 fields spanning two words can therefore combine pre-store and post-store bits without an inter-work-item ordering guarantee. This is an accepted-operation correctness failure, not an output-output write race.

The corrected F64 witness is `lhs = 0x3ff0000000000000`, `rhs = 0x3feffffffffffffc`: independent RNE ADD is `0x3ffffffffffffffe`; the modeled torn result after a low-word store followed by a high-word reread is `0x40000000fffffffe`. For `F6_E2M3` physical slot 5 (bit offset 30), `lhs` raw 8 and `rhs` raw 14 independently produce raw 19, while rereading the aliased source after the low-word store can combine raw-19 low bits with a raw-20 high computation into raw 23. In the `{1,17,33}` padded geometry, slot 165 begins at bit 990 and crosses words 31/32 owned by warp 0 lane 31 and warp 1 lane 0; the same cross-warp construction applies to `F6_E3M2`. These are semantic interleaving witnesses, not hardware reproductions. Existing CUDA/ROCm arithmetic tests use distinct output tensors or non-crossing leaves, and the retained `lhs,lhs,out` fault cases are read/read overlap rather than output alias.

## Scope

- Correct the shared CUDA/ROCm execution path for exact `lhs == out`, exact `rhs == out`, both aliases, and valid exact alias plus broadcast of the other operand; capture complete source fields before any overlapping output store with race-free ownership/order.
- Preserve common exact-alias admission and rejection of non-exact same-owner windows, disjoint-output behavior and fast path, zero external binary workspace, named dtype encoding/RNE and operation semantics, logical mapping/broadcasting, untouched padding, owner registration/deduplication, queue ordering, lifetime, and capability/error behavior.

## Implementation references

- **Modify:** `src/shared/standard_tiled_add.inl` — `add_load_bits`, `binary_body`, and the alias-safe kernel strategy; this owns packed standard-GPU field reads/stores for both CUDA and ROCm.
- **Modify:** `src/shared/gpu_queue_operations.inl` — `GpuQueue<Policy>::binary_impl`/`execute` dispatch; select the alias-safe path only for already-validated exact aliases and retain the disjoint path.
- **Read:** `src/device_ops_binary.cpp` — `DeviceOps::validate_binary` and `exact_alias`; reuse its established alias predicate and do not broaden or narrow the public contract.
- **Read:** `src/cuda/copy.cu` and `src/rocm/copy.hip` — shared-kernel inclusion units; keep backend drivers mechanically equivalent.
- **Tests:** `test/backend/backend_conformance_add_gpu.hpp` — add one shared exact-raw alias helper; `test/cuda/test_cuda_conformance.cpp` and `test/rocm/test_rocm_conformance.cpp` — device setup/registration only; `test/backend/backend_conformance_add.hpp:15-155` — independent decode/RNE oracle.

## Requirements

- Add a standard-GPU alias execution strategy that snapshots complete `lhs` and `rhs` source fields before any store that can overlap either source, with a source-level race-free memory-order argument covering packed fields, divergent compact paths, and cross-warp/block ownership. A tile-local/shared-memory snapshot or an equivalently proven complete-source algorithm is required.
- Handle exact `lhs == out`, exact `rhs == out`, both aliases, and an exact alias with a separately broadcast operand. Do not reject a valid exact alias, relocate caller storage, or silently change the request to an independent output.
- Retain the current word-owned disjoint-output fast path and its zero external workspace requirement. Any alias staging must be internal to the kernel/execution strategy and must not become a general backend workspace contract.
- Preserve operation selection ADD/MUL/SUB/DIV, named dtype codec and RNE behavior, F64/F6/I64/U64 packing, logical mapping and transforms, untouched padding/rounding/mapping, queue and lease lifetime, and all existing validation/capability/error behavior.
- Add one shared CUDA/ROCm conformance helper that compares complete logical raw encodings to the independent oracle for exact aliases, including the corrected F64 and F6 cross-word witnesses; do not duplicate the conformance body in backend drivers.

## Non-goals

- Do not alter `DeviceOps::validate_binary`, common broadcasting, non-exact alias rejection, read/read overlap, dtype/capability policy, scalar codec, tile representation, or generic non-alias floating tolerance.
- Do not change CPU, SYCL, or TTNN execution, native carrier mapping, queue admission/resource ownership, external workspace APIs, or unrelated backend tests.
- Do not claim hardware reproduction from the source witnesses; no accelerator gate was available in the reviewed validation context.

## Acceptance criteria

- [ ] On real CUDA and ROCm devices, exact `lhs == out` and exact `rhs == out` for ADD/MUL/SUB/DIV with `F64`, `F6_E2M3`, and `F6_E3M2` match the independent oracle's complete logical raw encodings for shape `{1,17,33}`, including the corrected F64 witness and a slot-5/cross-word F6 witness; the modeled torn encodings are absent.
- [ ] Both-alias and valid alias-plus-broadcast cases, varied occupancy, transformed exact aliases admitted by common validation, untouched padding, disjoint-output fast path, zero external workspace, queue waits/repeats, and backend capability/error behavior remain unchanged; CPU/SYCL/TTNN differential controls remain oracle-correct.

## Verification

- `ctest --test-dir <build> --output-on-failure -R '^(iom_cuda_conformance_tests|iom_rocm_conformance_tests|iom_sycl_conformance_tests|iom_ttnn_conformance_tests|iom_backend_conformance_cpu_tests)$'` after adding the focused shared helper; CUDA/ROCm/other accelerator execution must use the configured remote hosts.
- Run the focused real CUDA and ROCm scenario at `{1,17,33}` for all four operations and the listed dtypes/orientations, with exact raw comparison, repeated occupancies, both aliases, both-alias, and alias-plus-broadcast cases. The supplied root context passed only local CPU targets `iom_tests`, `iom_scalar_add_tests`, and `iom_backend_conformance_cpu_tests` (3/3); CUDA/ROCm/SYCL/TTNN sync attempts timed out before build, so no hardware result is claimed here.
