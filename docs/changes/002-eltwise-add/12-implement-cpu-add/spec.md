# Implement CPU elementwise addition

**Order:** 12
**Priority:** P1 — required normal-path CPU ADD execution and the reusable backend conformance cases that prove its complete contract.
**Blocked by:** `10-validate-add-requests`, `11-implement-scalar-add`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

The CPU backend implements the post-OID `DeviceOps::add(const TensorView&, const TensorView&, TensorView&) noexcept` path for every required numeric `NONE` leaf, using the common request validation and view metadata snapshot produced by the ADD validation work and production scalar arithmetic from task 11. CPU ADD executes asynchronously according to the queue contract (inline completion is permitted), returns a positive token only for accepted work, and is covered by a backend-neutral ADD conformance matrix plus CPU-specific driver and focused tests.

## Scope

- Modify the CPU ADD extension behind the common non-throwing OID facade in `src/cpu/device.cpp`. Do not create a public support query, options object, signature knob, CPU `Device`, CPU tensor, or generic fallback framework.
- Execute exactly these 21 numeric leaves when `QuantizationFormat::NONE` is shared by all three recognized specs: `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`. `BOOL` and `F8_E8M0` return `Unsupported` after the three specs match; recognized non-`NONE` quantization is also `Unsupported`; malformed, unknown, mismatched, incompatible, or wrong-device requests are `InvalidArgument` (checked arithmetic overflow is `Overflow`) through the common boundary.
- Apply right-aligned multidirectional broadcasting for ranks at least two, including rank promotion, `[1,1]` as the only scalar convention, singleton final rows/columns, and tails in tiled final axes. Resolve logical coordinates before standard tile slots. Respect each view's independent leading-plane offset and strides, including nested transformed leading views; never read or write padding or planes outside the logical output.
- Produce exact low-`w`-bit two's-complement modulo-`2^w` integer sums, with ordinary binary unsigned interpretation. Decode each scalar floating format according to task 11's production arithmetic and encode once using its exact RNE reference; the accepted CPU result is the reference encoding or a finite encoding no more than one ULP away, with the specified NaN, infinity, gradual-underflow, cancellation, and zero-sign classes.
- Preserve caller-owned storage and handles. ADD must not allocate, replace, relocate, or resize operands or output. Any internal temporary required by the CPU implementation is private to the queue and must use the established failure mapping and cleanup rules.
- Capture both input values before each output store. Permit arbitrary `lhs`/`rhs` read/read overlap. Permit input/output overlap only for an exact same-owner alias with identical `TensorSpec`, plane offset, plane strides, and logical element mapping; exact `lhs`, exact `rhs`, and all-three exact aliases are valid. Reject every other same-owner input/output relationship, including disjoint windows, broadcast windows, and non-identical transformed windows, before effects or token acceptance.
- Preserve in-order submission, positive token encoding, repeat waits, retained asynchronous failures, three-owner lifetime tracking with exact-alias deduplication, temporary derived-view metadata snapshots, stable owner identity/handles, source values, and untouched physical padding/planes. CPU may complete accepted work inline but must still obey these observable rules.
- Extend the task-11 `test/backend/backend_conformance_add.hpp` oracle file with the backend-neutral case matrix. Keep shape, numeric, alias, lifetime, error, ordering, and wait cases backend-neutral; expose only narrow adapter seams for backend construction, transfers, queue submission, and observation so later backend drivers can reuse the matrix without redesign.
- Integrate the matrix from `test/cpu/test_cpu_conformance.cpp` using the existing `CpuDevices`, `ConformanceDevices`, allocator traffic gate, logical host encoding/readback helpers, and the independent scalar oracle delivered by task 11.
- Add focused CPU behavior tests in `test/cpu/test_cpu.cpp` where direct CPU storage/view and allocator seams are needed beyond the shared matrix. Change only ADD's compute expectation to supported; `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa` remain `Unsupported`.

## Implementation references

- **Modify:** `src/cpu/device.cpp` — `CpuQueue` and its tile/layout helpers (`for_each_tile`, `for_each_tile_lockstep`, `copy_elements`); add the CPU ADD execution path behind the migrated protected extension boundary, reusing common validation/metadata and queue submission/completion rather than retaining caller view references.
- **Read:** `include/iom/iom.hpp`, `include/iom/device.hpp`, and the post-OID common ADD validation/facade touchpoints — reuse the established no-throw OID mapping, `submit`, completion, owner/lifetime registration, and view metadata snapshot conventions; do not duplicate public validation in the CPU implementation.
- **Read:** `include/iom/tensor.hpp` and `src/iom.cpp` — use the existing rank, shape, leading-transform, offset, stride, owner, and standard tiled-storage semantics. Final axes are tiled and public broadcast/zero-stride views are not introduced.
- **Read:** `test/backend/backend_conformance_common.hpp` — reuse `ConformanceDevices`, `TrafficGate`, independent logical encoding/readback, token helpers, and backend-neutral test conventions.
- **Read:** `test/backend/backend_conformance_oracle.hpp` — reuse the independent storage/layout observation facilities; scalar expected values must come from task 11's independent exact oracle, never from CPU implementation helpers.
- **Modify:** `test/backend/backend_conformance_add.hpp` — extend task 11's independent scalar-oracle header with backend-neutral ADD matrix runners for all required numeric, broadcast, alias, error, lifetime, ordering, repeat-wait, metadata, and storage-preservation cases. Keep CPU-specific construction and fault seams in the driver or adapters.
- **Modify:** `test/cpu/test_cpu_conformance.cpp` — include and invoke the ADD matrix for the CPU candidate/reference/foreign devices; run exhaustive low-width raw-pair cases and representative wide/special vectors, then the shared behavioral matrix.
- **Modify:** `test/cpu/test_cpu.cpp` — focused CPU tests for production scalar/tiled execution, alias and transformed-view behavior, validation-before-effects, storage preservation, ownership stability, and CPU-only allocator/queue seams.
- **Read:** `test/CMakeLists.txt` — existing `iom_cpu_tests` and `iom_backend_conformance_cpu_tests` targets; no new dependency or test executable is needed.

## Requirements

- Route all three views through the common ADD validation and metadata snapshot before any CPU read, write, allocation, queue submission, sequence consumption, or token acceptance. Enforce exact queue `Device` identity, owner relationships, recognized/well-formed matching specs, exact broadcast output shape, checked shape/size/offset/stride arithmetic, and the input/output alias rules. Rejections leave output bytes, source bytes, allocator traffic, sequence state, and token state unchanged.
- Return a positive OID for accepted supported numeric work. Return `Unsupported` only for a matching excluded leaf (`BOOL` or `F8_E8M0`) or matching recognized non-`NONE` quantization; do not use storage support alone as a reason to reject a required numeric leaf. Other malformed or mismatched specifications are `InvalidArgument`.
- Implement all 21 leaves individually, including sub-byte packed storage and 64-bit values. For integer leaves, compute the exact sum modulo the leaf width without signed overflow or saturation. For floating leaves, use task 11's production scalar decode/sum/encode contract, not host-native arithmetic as an unbounded substitute.
- Map every logical output coordinate through right-aligned broadcast dimensions and each view's independent plane offset/stride. Correctly cover equal shapes, opposite-direction singleton axes, rank promotion, `[1,1]`, rank 2/3/6, final tiled row and column tails, and nested transformed leading views.
- Ensure exact aliases work without corrupting unread inputs by capturing both operand values before storing each result. Preserve source data for non-aliasing work, and preserve all unaddressed padding bits/bytes and other planes. Do not expose or construct public broadcast views or public zero strides.
- Reject incompatible operand shapes, an output shape other than the exact broadcast result, wrong-device views, and forbidden same-owner windows before submission. Read/read overlap is valid; exact in-place input/output aliases are valid only when all metadata and logical mapping are identical.
- Keep caller allocations and addresses stable throughout ADD. Track all three storages until accepted work completes, deduplicating exact aliases; snapshot temporary/derived view metadata so callers may destroy view temporaries immediately after submission. Preserve ordering, visibility, repeatable waits, and repeatable retained asynchronous failures. Never retry an accepted failure.
- In `test/backend/backend_conformance_add.hpp`, separate backend-neutral cases from CPU adapter code. The matrix must compare candidate output with one independent exact scalar oracle, test positive tokens and waits, and make every required behavior observable rather than asserting implementation details.
- In the CPU conformance driver, run every ordered raw-encoding pair for each of `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `I2`, `U2`, `I4`, and `U4` on CPU. Run representative boundary, subnormal, zero, infinity, NaN, cancellation, overflow, and wide-value vectors for the remaining required leaves. Check integers exactly modulo width and floating results against the independent reference's class and one-ULP envelope.
- Cover deterministic common error and applicable CPU fault seams, including pre-acceptance temporary/resource failure and post-acceptance retained failure where the existing queue seam permits it. Verify no effects or sequence/token side effects on rejection, cleanup of temporary allocations, and repeated throwing of an accepted asynchronous failure.
- Update the existing CPU compute-capability expectations so ADD is the sole newly supported compute hook; retain `Unsupported` expectations for every other unimplemented compute operation.

## Non-goals

- Do not implement CUDA, ROCm, SYCL, or TTNN ADD, TTNN storage expansion, accelerator emulation, or cross-backend coexistence; those are separate tasks.
- Do not alter the public API beyond the already-migrated three-view ADD facade, add `add_support`, expose backend path/capability queries, add options, or add caller-visible casts/promotion.
- Do not implement `sub`, `mul`, `silu`, `linear`, `rmsnorm`, `sdpa`, graphs, autograd, rank-0/rank-1/empty tensors, public broadcast views, zero strides, or non-`NONE` quantization arithmetic.
- Do not create a generic fallback/capability framework, a second scalar oracle, a CPU-specific public fallback API, or a new CPU queue/device/tensor abstraction.
- Do not make allocator behavior, native CPU vectorization, lookup-table choice, accumulator width, or other implementation mechanics part of the public contract; test observable results and ownership/lifetime behavior instead.
- Do not defer required backend-neutral cases to task 17 or treat the P2 coexistence task as CPU completeness.

## Acceptance criteria

- [ ] `src/cpu/device.cpp` accepts each of the 21 numeric `NONE` leaves and returns a positive token; matching `BOOL`, `F8_E8M0`, and recognized non-`NONE` requests return `Unsupported`, while malformed/mismatched/device/shape/view/alias requests return the specified negative OID without effects or sequence consumption.
- [ ] CPU output is exact modulo `2^w` for every integer width and matches task 11's independent floating reference or its finite one-ULP envelope, including required NaN, infinity, gradual-underflow, cancellation, and zero-sign behavior.
- [ ] The reusable matrix and CPU driver prove equal shapes, opposite singleton axes, rank promotion and `[1,1]`, final tiled row/column tails, ranks 2/3/6, nested transformed leading views, incompatible shapes, and wrong output shapes.
- [ ] The matrix and focused CPU tests prove exact `lhs`, exact `rhs`, and all-three aliases; reject forbidden same-owner windows; allow read/read overlap; capture operands before stores; preserve sources, untouched padding/planes, owners, handles, and caller allocation addresses; and retain temporary metadata/lifetimes until completion.
- [ ] Ordering, positive tokens, repeated waits, retained post-acceptance failures, deterministic pre-acceptance failures, cleanup, and no rejection side effects are observable. CPU may complete inline without weakening the queue contract.
- [ ] CPU conformance exhaustively executes all ordered raw pairs for the specified low-width integer and floating leaves and uses representative wide/special vectors for every other required leaf against one independent oracle.
- [ ] `test/cpu/test_cpu_conformance.cpp` integrates the backend-neutral matrix and `test/cpu/test_cpu.cpp` contains focused CPU coverage; `iom_cpu_tests` and `iom_backend_conformance_cpu_tests` cover the complete CPU ADD path, while all non-ADD compute hooks remain `Unsupported`.

## Verification

- `cmake --build <build-dir> --target iom_cpu_tests iom_backend_conformance_cpu_tests` (not run here; the assignment explicitly requires no gates).
- `ctest --test-dir <build-dir> --output-on-failure -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` (not run here; the assignment explicitly requires no gates).
