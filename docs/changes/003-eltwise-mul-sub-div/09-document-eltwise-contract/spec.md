# Document the elementwise contract

**Order:** 09
**Priority:** P2 — records the shipped four-operation behavior and removes stale public claims after integration is proven.
**Blocked by:** `08-eltwise-coexistence`
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Update architecture, backend-contract, README, and final public API comments to describe one consistent shipped contract for ADD, MUL, SUB, and DIV. Documentation must state exact signatures and differing dtype domains, arithmetic and special values, validation/error precedence, views/broadcasting/aliases, asynchronous ownership/failure behavior, backend capability boundaries, and where conformance coverage lives; stale MUL-unsupported/narrow claims must be removed.

## Scope

- Update `docs/ARCHITECTURE.md:58-63,112-124,200-240`, `docs/BACKEND_CONTRACT.md:157-165,250-390`, and relevant ADD/MUL operation sections in `README.md`.
- Update final public comments in `include/iom/iom.hpp`, `include/iom/device.hpp`, `include/iom/tensor.hpp`, and `include/iom/ttnn/device.hpp`; only comments are in scope in these headers unless required to accurately expose the already-shipped declarations.
- Document exact `noexcept` APIs: `oid add(const TensorView&, const TensorView&, TensorView&) noexcept`, and identical `mul`, `sub`, and `div`; no options, queries, promotion, fallback selector, public broadcast view, or public zero stride.
- Document ADD/MUL/SUB support for the 21 NONE numeric leaves `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; DIV support for the nine floating leaves `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; BOOL/F8_E8M0/non-NONE and integer DIV are Unsupported.
- Explain integer modulo-$2^w$ MUL/SUB without signed-overflow UB; floating decode/extended operation/one-step RNE encoding, gradual underflow, format-specific saturation/special rules, one-ULP finite envelope, and observable SUB/DIV operand order.
- Explain validation order and errors, right-aligned broadcasting including `[1,1]`, transformed views/tails/padding, exact in-place alias/read overlap, immutable metadata snapshots, three-owner deduplicated lifecycle, stable caller storage/handles, positive OIDs, ordering, repeat waits, and retained post-acceptance failures. Include backend capability tables and source map for shared scalar/common boundary, each backend conformance target, and coexistence target.

## Implementation references

- **Modify:** `docs/ARCHITECTURE.md:58-63,112-124,200-240` — architecture boundaries, operation support, shared GPU source, and backend-specific storage/staging descriptions.
- **Modify:** `docs/BACKEND_CONTRACT.md:157-165,250-390` — validation, dtype/capability, queue, ownership, errors, and conformance contract.
- **Modify:** `README.md` — user-facing operation/API compatibility and support summary.
- **Modify comments:** `include/iom/iom.hpp`, `include/iom/device.hpp`, `include/iom/tensor.hpp`, `include/iom/ttnn/device.hpp` — remove claims that ADD is the sole support signal or MUL is unsupported; keep declarations and unrelated APIs intact.
- **Read:** `test/CMakeLists.txt:1-17,19-36,263-363` and backend driver paths from tasks 04-08 to cite actual conformance source/target names.

## Requirements

- Documentation must be internally consistent: valid supported MUL now returns a positive token, SUB/DIV are three-view asynchronous operations, and existing ADD callers/results remain compatible; consumers must rebuild headers and no mixed-version ABI is promised.
- State exact validation precedence: recognized specs/rank/dim/device/owner/handle/view/storage/checked arithmetic, matching leaf/quantization, broadcasting/output shape, mapping snapshot, alias rule, then operation support; list `InvalidArgument`, `Overflow`, `ResourceExhausted`, `DeviceError`, `InternalError`, and `Unsupported` behavior without contradicting code.
- Describe CPU inline completion, accelerator queue/staging/emulation boundaries, no caller operand/output allocation or relocation, and no backend-specific SDK dtype narrowing of required numeric leaves.
- Include capability tables and conformance map for CPU, CUDA, ROCm, SYCL, and TTNN, plus shared scalar/common tests and one combined coexistence target.
- Remove stale MUL-unsupported, ADD-only, narrow-domain, public-query, promotion, fallback, or unrelated-operation claims. Do not document implementation options as API features.

## Non-goals

- Do not alter implementation, tests, public declarations, ABI, unrelated operations (`copy`, `silu`, `linear`, `rmsnorm`, `sdpa`), or tensor storage behavior.
- Do not add new documentation files, an index, options/query APIs, fallback/promotion concepts, or unsupported future features.
- Do not preserve contradictory historical wording merely to describe old behavior; mention the MUL compatibility cutover only as the shipped contract requires.

## Acceptance criteria

- [ ] Architecture, backend contract, README, and all named public comment surfaces agree on exact four-operation signatures, domains, arithmetic, validation, views/aliases, ownership, queues, failures, capabilities, and compatibility.
- [ ] No stale statement says MUL is unsupported, ADD is the sole binary operation, or required operations are limited by native SDK dtype support.
- [ ] Documentation identifies existing backend conformance targets and the combined coexistence target as the behavioral source map without inventing targets or features.
- [ ] Only the listed documentation/header comment files are changed; no implementation or parent-spec edits are made.

## Verification

- `git diff --check -- docs/ARCHITECTURE.md docs/BACKEND_CONTRACT.md README.md include/iom/iom.hpp include/iom/device.hpp include/iom/tensor.hpp include/iom/ttnn/device.hpp` — not run per assignment.
- Focused documentation review: search the listed files for `MUL unsupported`, ADD-only support claims, and contradictions between the capability table and the exact dtype lists; not run per assignment.
- `cmake --build build --target iom_backend_coexistence_tests` — not run per assignment; implementation/coexistence gates belong to prior tasks.
