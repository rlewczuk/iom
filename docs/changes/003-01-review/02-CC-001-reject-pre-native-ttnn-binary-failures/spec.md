# Reject TTNN binary failures before native submission

**Order:** 02
**Priority:** P0 — restores public pre-acceptance transaction semantics and owner/sequence rollback
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** CC-001
**Review area:** Contract & correctness
**Review severity:** medium
**Review verification:** strongly-supported, confidence 95
**Review scope:** whole-codebase
**Backend scope:** TTNN
**Location:** `src/ttnn/device.cpp` — `TtnnQueue::binary_impl`, `TtnnQueue::execute`, and `execute_copy` counterpart; `src/ttnn/copy.cpp` — `ttnn_detail::binary_planes`; common `submit_binary` transaction in `include/iom/iom.hpp`

## Outcome

A TTNN binary failure before the first native output upload is a synchronous pre-acceptance error: it returns the mapped negative OID, consumes no sequence, registers no owners, and leaves output bytes unchanged. A failure after native mesh work has been submitted remains a positive retained failure with staging leases and owner protection until the existing TTNN completion proof.

## Current problem

The public invariant requires pre-acceptance errors to avoid output mutation, token/sequence consumption, and owner registration. `DeviceOps::submit_binary` reserves a sequence and registers the three distinct owners before synchronously invoking the staged callback. `TtnnQueue::execute` catches every binary exception and returns; `binary_planes` sets `any_submitted` only when the first output `copy_to_device` is reached, after host reads, scalar work, and output staging allocation. A fault from `TtnnHostStaging::acquire_upload` before that upload therefore is a no-native-work failure, but the catch retains it under a positive token and marks the sequence executed. The same broad catch can swallow outcome insertion failure before a `BinaryOutcome` exists. `execute_copy` already propagates its `!any_submitted` exception and rolls back as the correct counterpart. Root validation included TTNN smoke/conformance/coexistence passing, but no candidate-specific pre-native binary fault injection ran.

## Scope

- Distinguish the TTNN binary pre-native branch from the accepted post-upload branch using the existing `any_submitted` boundary.
- Propagate pre-native exceptions through `submit_binary`: erase only the provisional `BinaryOutcome` in the backend callback, then rethrow so the common transaction removes registered owners, rolls back the sequence, and maps the prescribed negative error.
- Preserve TTNN native per-plane storage, host scalar emulation, API mutex, positive retained failures after at least one upload, staging leases, and completion/quarantine behavior.

## Implementation references

- **Modify:** `src/ttnn/device.cpp` — `TtnnQueue::binary_impl`, `execute`, and `complete_task`; make no-native-work exceptions propagate and keep accepted failures retained.
- **Read:** `src/ttnn/device.cpp` — `execute_copy` around its `!any_submitted` rollback; reuse its transaction rule rather than inventing another error policy.
- **Read:** `src/ttnn/copy.cpp` — `ttnn_detail::binary_planes`; preserve the exact point where output `copy_to_device` establishes native submission.
- **Read:** `include/iom/iom.hpp` — `submit_binary` sequence/owner rollback and `BinaryEntryRegistration` cleanup.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` existing copy partial-plane/registration cases and retained-failure cases; add a binary first-upload staging-fault scenario.

## Requirements

- If an exception occurs before any output upload/native mesh command, erase any provisional `BinaryOutcome` and rethrow. `submit_binary` remains the single owner of registered-owner cleanup, and `submit` remains the single owner of sequence rollback and error mapping; do not remove either state twice.
- If one or more native uploads occurred, retain the positive token, failure, staging leases, and owner registrations until TTNN finish/quarantine proves completion; repeated waits must retain existing failure behavior.
- Use `DeviceOps::submit_binary` and the existing TTNN copy transaction as the owner of public error precedence. Do not move the native storage or mesh-finish boundary.

## Non-goals

- Do not change TTNN's 32x32 carrier mapping, supported operation domains, host-emulated arithmetic, staging layout, API mutex, mesh finish policy, accepted post-submission failure behavior, or common validation.
- Do not change GPU, SYCL, or CPU transaction paths except for a strictly necessary shared rollback API reuse.
- Do not turn numerical tolerance or throughput into this contract task.

## Acceptance criteria

- [ ] For every TTNN binary operation, a fault armed after input setup and before the first output upload returns the exact mapped pre-submit error (for example `OidError::ResourceExhausted`), not a positive token.
- [ ] The pre-submit fault leaves sentinel output bytes unchanged, leaves no owner entries, and makes the next valid request use token sequence one.
- [ ] A fault after at least one native upload still returns a positive token, rethrows the retained failure on repeated waits, and protects staging/owners until completion proof.
- [ ] Outcome insertion failure before native submission is also rejected synchronously rather than reported as successful completion.

## Verification

- `(remote-development: TTNN host)` arm `fail_next_host_transfer_staging_allocation_for_testing()` after inputs are loaded, submit fresh ADD/MUL/SUB/DIV cases, and assert the negative resource error, unchanged sentinel output, no registry entry, and next sequence one; repeat with any available pre-native outcome/runtime fault seam.
- Run `cmake --build <remote-build> --target iom_ttnn_conformance_tests iom_backend_coexistence_tests` and `ctest --test-dir <remote-build> --output-on-failure -R '^(iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$'` using the repository remote-development procedure.
- Actual validation available now is only the root-supplied TTNN smoke/conformance/coexistence pass; no candidate-specific fault injection or sanitizer run has occurred.
