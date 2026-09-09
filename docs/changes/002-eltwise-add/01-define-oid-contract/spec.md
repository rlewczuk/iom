# Define the public OID contract

**Order:** 01
**Priority:** P0 — establishes the signed public OID type, stable error values, and checked token boundaries required by every later migration.
**Blocked by:** None
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

The repository has one dedicated, public OID definition and one common queue-token contract. `oid` is signed, errors and tokens are distinguishable without exceptions, queue IDs and sequences encode losslessly through `INT64_MAX`, and common wait parsing and sequence state reject invalid values before any operation is accepted.

## Scope

- Create the planned public header `include/iom/oid.hpp` and define `using oid = std::int64_t`.
- Define the complete stable error enum and constexpr classification/conversion helpers in that header.
- Register `include/iom/oid.hpp` in the root `CMakeLists.txt` explicit `IOM_HEADERS` public-header list.
- Replace the current `typedef uint64_t oid` at `include/iom/iom.hpp:199-200` with the dedicated definition and use the new signed type throughout the common queue contract.
- Migrate the common token encoding, sequence bounds, queue-ID state, and wait-token parsing at `include/iom/iom.hpp:374-386` and `src/iom.cpp:483-545` to the checked 55-bit sequence format.

## Implementation references

- **Create (planned):** `include/iom/oid.hpp` — public `oid`, `OidError`, `to_oid`, `oid_is_error`, and `oid_is_token` definitions.
- **Modify:** `CMakeLists.txt` — root `IOM_HEADERS` list near lines 82-91; register the planned public header explicitly.
- **Modify:** `include/iom/iom.hpp` — current OID typedef near lines 199-200 and `DeviceOps` token constants/state near lines 374-386; retain queue completion and skipped-sequence state while changing its checked limits.
- **Modify:** `src/iom.cpp` — queue-ID lease/release and `DeviceOps::encode_token`/`wait` near lines 483-545; validate signed and structurally invalid OIDs before queue lookup or completion waiting.
- **Tests:** `test/test_iom.cpp:1695-1882` — focused `DeviceOps` signature, queue-ID, token-encoding, and sequence-boundary compile/runtime coverage; the broad core OID test migration is a later task. The focused test target is `iom_tests`.

## Requirements

- `include/iom/oid.hpp` MUST define exactly `using oid = std::int64_t` and MUST provide:
  - `enum class OidError : oid` with `InvalidArgument = -1`, `Unsupported = -2`, `Overflow = -3`, `ResourceExhausted = -4`, `DeviceError = -5`, and `InternalError = -6`.
  - `[[nodiscard]] constexpr oid to_oid(OidError error) noexcept`, returning the enum's underlying negative value.
  - `[[nodiscard]] constexpr bool oid_is_error(oid value) noexcept`, true exactly when `value < 0`.
  - `[[nodiscard]] constexpr bool oid_is_token(oid value) noexcept`, true exactly when `value > 0`.
- The six listed `OidError` values are the complete defined error set for this change. Every negative `oid` is reserved for errors. `0` is invalid and is neither an error code nor a token.
- Preserve all 255 queue IDs. For `q` in `[1, 255]` and `sequence` in `[1, 2^55 - 1]`, encode exactly as `static_cast<oid>((std::uint64_t{q} << 55) | sequence)`. Queue IDs occupy bits 55 through 62; bit 63 is never set.
- The maximum valid encoding MUST be `INT64_MAX`, produced by queue ID `255` and sequence `2^55 - 1`. The encoding MUST retain queue ID and sequence values without truncation, including queue ID 255.
- Sequence zero MUST never be submitted. Sequence allocation and any forward sequence-state manipulation MUST be checked against the inclusive maximum `2^55 - 1`; exhaustion MUST be detected synchronously before effects or token acceptance and exposed as a checked common condition that task 02 maps to `OidError::Overflow`. Checked state MUST not wrap.
- Common wait parsing MUST reject negative OIDs, zero, foreign queue IDs, queue ID zero, sequence zero, values above the signed positive boundary, future/unsubmitted sequences, and skipped or reserved-but-never-submitted sequences with `std::invalid_argument`. Such values MUST NOT wait, complete, consume a sequence, or mutate queue state.
- Positive structurally valid tokens for this queue MUST retain existing ordering, repeat-wait, visibility, and retained-failure behavior. The signed representation MUST not turn a valid token through `INT64_MAX` into an error.
- Keep the implementation backend-neutral: token bit arithmetic and validation belong to the common `DeviceOps` state, not to a backend or a public operation-specific facade.

## Non-goals

- Do not add public `add_support` or any equivalent capability query; `add` itself is the later support signal.
- Do not implement common public non-throwing facades, protected backend hooks, exception-to-`OidError` mapping, or submission/error-boundary migration for OID-returning operations.
- Do not migrate CPU, CUDA, ROCm, SYCL, or TTNN implementations, queues, fakes, or backend tests.
- Do not perform the broad core/shared/backend OID test migration, add new conformance coverage, or update architecture/backend/API documentation.
- Do not implement ADD behavior, storage expansion, numeric arithmetic, broadcasting, or any other operation.

## Acceptance criteria

- [ ] `include/iom/oid.hpp` exists as a public planned header with the exact signed alias, six exact negative constants, and three exact `constexpr noexcept` helpers; no unsigned public OID typedef remains at `include/iom/iom.hpp:199-200`.
- [ ] The root `IOM_HEADERS` list registers `include/iom/oid.hpp`, and common code includes/reuses that definition rather than declaring a second OID type.
- [ ] Observable classification distinguishes `oid` values as follows: `0` is neither error nor token; each defined negative enum converts through `to_oid` and satisfies `oid_is_error`; positive values satisfy `oid_is_token`; wait parsing additionally rejects zero and positive encodings with queue ID or sequence zero.
- [ ] Encoding and decoding preserve queue ID `1` and sequence `1`, queue ID `255`, and the maximum sequence `2^55 - 1`; queue ID `255` with that sequence yields exactly `INT64_MAX`. No valid encoding is negative or zero.
- [ ] Sequence exhaustion at `2^55` is detected before submission/effects without wrapping, while the boundary token at sequence `2^55 - 1` remains valid; task 02 owns conversion of that checked condition into the public `OidError::Overflow` result.
- [ ] The focused ranges in `test/test_iom.cpp:1695-1882` are recorded as task-07 migration touchpoints for signature, queue-ID, token-encoding, and boundary assertions; this task does not rewrite or require the pre-cutover `iom_tests` expectations.

## Verification

- `cmake --build <configured-build-dir> --target libiom`
- Inspect the public declarations and common encode/decode boundary for zero/error/token classification, queue ID 255, `INT64_MAX`, and checked sequence exhaustion; runtime test migration belongs to task 07.
