# CPU elementwise operations

**Order:** 04
**Priority:** P1 — delivers the first real backend implementation on the established common and scalar boundaries.
**Blocked by:** `01-common-binary-boundary`, `02-scalar-binary-arithmetic`, `03-shared-eltwise-conformance`
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Migrate the CPU ADD traversal to one operation-neutral standard-tiled traversal that executes ADD, MUL, SUB, and floating DIV. It must select arithmetic once per accepted request, preserve caller storage and inline completion/registry behavior, and pass shared real-queue conformance for all required mapping, alias, dtype, ordering, and failure cases.

## Scope

- Generalize `src/cpu/device.cpp:686-758` from ADD-only dispatch to the common operation identity and scalar operation-specialized path.
- Execute ADD/MUL/SUB for exactly the 21 NONE numeric leaves `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; execute DIV for exactly the nine floating leaves `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`.
- Reuse the existing standard-tiled logical/packed coordinate mapping, broadcast markers, transformed leading views, row/column tails, padding behavior, and operation submission/registry path. Load both operands before storing each exact aliased output element.
- Preserve CPU inline completion, positive OID sequencing, repeat waits, retained failures, no allocation or caller-storage replacement, and common rejection of BOOL/F8_E8M0/non-NONE quantization and integer DIV.
- Update `test/cpu/test_cpu.cpp:1140-1188,1360-1403`, `test/cpu/test_cpu_conformance.cpp:223-278` to invoke shared operation-neutral cases through the real CPU queue and replace stale MUL-unsupported assertions.

## Implementation references

- **Modify:** `src/cpu/device.cpp:686-758` — current ADD elementwise implementation/traversal; retain one loop and select an operation-specialized scalar callable outside it.
- **Modify:** `test/cpu/test_cpu.cpp` — CPU local tests nearest the existing ADD operation and storage behavior.
- **Modify:** `test/cpu/test_cpu_conformance.cpp:223-278` — runtime setup/driver registration; invoke shared four-operation harness.
- **Read:** `src/shared/scalar_add.hpp:11-113` — neutral scalar operation boundary from task 02.
- **Read:** `include/iom/iom.hpp:268-405` — immutable request and completion contract from task 01.

## Requirements

- Every supported CPU leaf and operation must produce independent-oracle values for addressed elements; integer MUL/SUB are exact modulo-$2^w$, and floating results obey one-ULP/special/zero/sign rules.
- Arithmetic selection must occur once per request, not as a per-element operation switch; do not duplicate coordinate mapping, packed-bit extraction, or traversal bodies.
- Preserve transformed nonzero offsets/strides, opposite-direction broadcasting, rank 2/3/6/8/9/16/17, final tiled tails, untouched padding/planes, exact lhs/out, rhs/out, and all-three aliases, and valid read/read overlap.
- Verify common errors before execution and ensure invalid MUL/SUB/DIV requests leave output, token sequence, and ownership untouched. Accepted failures retain owners and remain repeatable as specified.
- Keep CPU storage boundaries and completion/registry mechanics; do not add allocation, device fallback, or changes to unrelated operations.

## Non-goals

- Do not touch CUDA, ROCm, SYCL, or TTNN implementations or their drivers.
- Do not change public validation, scalar codec policy, integer DIV, storage ownership, or unrelated ADD semantics.
- Do not add a generic traversal framework, thread pool, fallback API, or CPU-specific dtype promotion.

## Acceptance criteria

- [ ] CPU real-queue tests return positive OIDs and correct oracle results for ADD/MUL/SUB on all 21 leaves and DIV on all nine floating leaves; valid MUL is no longer Unsupported.
- [ ] Mapping, rank, tail, transformed-view, padding, alias/read-overlap, no-allocation, ordering, repeat-wait, and retained-failure behavior remains observable and correct.
- [ ] Source review finds one standard-tiled traversal with operation selected outside logical loops and no duplicated ADD/MUL/SUB/DIV mapping or kernel body.
- [ ] Existing CPU ADD conformance and local behavior remain compatible.

## Verification

- `cmake --build build --target iom_backend_conformance_cpu_tests` — not run per assignment.
- `ctest --test-dir build -R '^iom_backend_conformance_cpu_tests$' --output-on-failure` — not run per assignment.
- Focused CPU scenario: interleave ADD, MUL, SUB, and F32 DIV on transformed/tail views, wait twice, and compare all addressed values and owner lifetime; not run per assignment.
