# Enforce the rank-eight tensor contract

**Order:** 03
**Priority:** P0 — descriptor bounds and every backend's tensor, view, and binary behavior rely on a complete and consistent full-rank contract from rank 2 through rank 8.
**Blocked by:** None
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

The backend-neutral tensor contract accepts full tensor, owner-view, and binary-result shapes only at ranks 2–8 inclusive. Rank nine is rejected deterministically before a tensor allocation, view result is published, owner/operation registration occurs, a sequence or positive token is accepted, metadata is uploaded, or any native effect begins. Rank-eight owners, transformed views, broadcasts, copies, and supported binary operations retain their current behavior on CPU and every enabled backend.

## Scope

Update the common implementation and public contract comments only for this rank boundary. Cover full-shape creation and validation, every `TensorView` transform result, and common binary result validation. Migrate the named core, shared-conformance, CUDA, and ROCm tests so rank eight exercises real owner/view/broadcast behavior and rank nine, including rank-increasing reshapes, is rejected. Keep all dtype, quantization, broadcasting rules, aliases, codecs, arithmetic, storage layout, and operation support behavior unchanged.

## Implementation references

- `src/iom.cpp`: `TensorShape::TensorShape`, `TensorSpec::validate`, `Tensor::Tensor`, `with_leading_dimensions`, `TensorView::{slice,select,permute,reshape_leading}`, `validate_binary_spec`, `validate_binary`, and the `DeviceOps::{copy,add,mul,sub,div}` facades.
- `include/iom/tensor.hpp`: `TensorShape`, `TensorSpec`, `TensorView`, and `Tensor` comments; document that a full shape is rank 2–8 while a leading-dimension helper span is not itself a full shape.
- `include/iom/iom.hpp`: `DeviceOps` and its operation-facade comments; document invalid-rank mapping through `OidError::InvalidArgument` without changing the public signatures or protected operation representation.
- `test/test_iom.cpp`: existing `TensorShape`/`TensorSpec` validation, transform, and `ADD validation preserves error precedence and rejection effects` coverage.
- `test/backend/backend_conformance_common.hpp:467-487`: `run_binary_rank_boundary_conformance`; make rank-eight the successful boundary and rank-nine the rejected boundary.
- `test/cuda/test_cuda_conformance.cpp:721-749` and `test/rocm/test_rocm_conformance.cpp:889-929`: replace rank-10 success fixtures with rank-eight owner/view/broadcast coverage and rank-nine rejection coverage.

## Requirements

- Define the full-rank interval as 2 ≤ rank ≤ 8. Apply it to every full `TensorShape` held by a `TensorSpec`, every materialized `Tensor`, every `TensorView::spec()`, every transformed view result, and every computed binary result shape. Mark any implementation-only helper used for this check as private; do not add a public rank constant, operation enum, direction enum, or duplicate query API.
- Ensure tensor creation validates the complete `TensorSpec` before allocating or registering owner storage. `TensorSpec::validate()` must reject rank nine (and any rank below two) with `std::invalid_argument`; preserve its established dtype, quantization, dimension, overflow, and error-precedence behavior.
- Ensure `TensorView::slice`, `select`, `permute`, and `reshape_leading` validate the resulting full shape before returning it. For `reshape_leading`, validate `leading_dimensions.size() + 2`, so a rank-increasing reshape to rank nine is rejected even when the source view is contiguous and the leading plane count is unchanged. Empty and otherwise valid leading-dimension spans remain valid when their resulting full shape is rank two. Do not reject a helper span solely because its length is not a full tensor rank.
- Ensure all common view-consuming operation validation paths see only full ranks 2–8. In particular, binary validation must validate all three operand/output full specs and the computed broadcast result before snapshots, registration, sequence reservation, token acceptance, metadata upload, or backend dispatch. A rank-nine output/result is invalid input, not unsupported work or a post-acceptance failure.
- Throw `std::invalid_argument` from throwing creation, validation, and transform APIs. The `noexcept` OID facades retain `OidError::InvalidArgument` (OID -1) for invalid rank; they must not consume a sequence, create a positive token, append a submission/registry entry, mutate output storage, or invoke a backend/native hook.
- Keep existing validation precedence. Adding the rank bound must not reorder established unknown-dtype, quantization, zero-dimension, overflow, device/owner, broadcast/output-shape, alias, or capability outcomes. Do not flatten leading axes to make rank nine fit.
- Preserve rank-eight addressing and metadata: leading plane spans remain independent offsets/strides, the final two dimensions remain tiled matrix axes, transformed views remain non-retargeting, and broadcast snapshots retain their existing mappings and zero-stride handling internally.
- Migrate caller/test fixtures rather than weakening the bound. Every current rank greater than eight success fixture in the named files becomes rank-eight coverage or an explicit rank-nine rejection; no dtype or broadcast case is removed merely to avoid the boundary.

## Non-goals

- Do not change dtype or quantization support, right-aligned broadcasting, alias rules, codecs, tile layout, arithmetic, operation support, allocation policy, queue admission, metadata sizing, or error categories other than the specified invalid-rank result.
- Do not expose a public rank-validation helper or new public operation/direction type, and do not add an alternate compatibility mode or flattening fallback.
- Do not modify backends, factory contracts, documentation outside the listed public comments and this change's tests, or any unrelated task directory.

## Acceptance criteria

- Full tensor creation and `TensorSpec::validate()` accept every rank from 2 through 8 and reject rank nine with `std::invalid_argument` before owner allocation; valid rank-eight tensors retain stable storage and exact existing metadata.
- `slice`, `select`, and `permute` preserve correct rank-eight owner/view addressing and cannot publish an out-of-range full rank. `reshape_leading` preserves valid rank-eight transforms and rejects every result whose assembled full rank is nine or greater, without changing the source view or owner.
- `add`, `mul`, `sub`, and `div` retain rank-eight broadcast correctness, including transformed leading views and their existing result mappings. Invalid rank-nine operands, outputs, or computed result shapes return `OidError::InvalidArgument` and leave queue records, registry entries, sequence/token state, output bytes, metadata uploads, and native traffic unchanged.
- Existing validation-precedence assertions continue to pass for invalid dtype, quantization, dimensions, overflow, device identity, broadcast shape, aliases, and unsupported capabilities; rank rejection does not become `Unsupported`, `Overflow`, `ResourceExhausted`, or a retained asynchronous failure.
- `test/test_iom.cpp` covers rank-eight owners/views/broadcasts, rank-nine creation and binary rejection, rank-increasing reshape rejection, and no-side-effect/error-precedence behavior. `run_binary_rank_boundary_conformance` covers rank-eight success and rank-nine rejection on the shared CPU/reference path. The CUDA and ROCm boundary cases cover rank-eight owner/view/broadcast correctness and rank-nine plus rank-increasing-transform rejection without allocator/native effects. The same shared boundary behavior passes for every enabled backend, including SYCL and TTNN.

## Verification

The following are required implementation gates and were not run while writing this specification:

- Local core and CPU conformance:
  `cmake --build build --target iom_tests iom_backend_conformance_cpu_tests`
  `ctest --test-dir build --output-on-failure -R '^(iom_tests|iom_backend_conformance_cpu_tests)$'`
- For each enabled accelerator, run the matching conformance target through the `remote-development` workflow on configured hardware:
  `ctest --test-dir <build> --output-on-failure -R '^iom_cuda_conformance_tests$'`,
  `ctest --test-dir <build> --output-on-failure -R '^iom_rocm_conformance_tests$'`,
  `ctest --test-dir <build> --output-on-failure -R '^iom_sycl_conformance_tests$'`, and
  `ctest --test-dir <build> --output-on-failure -R '^iom_ttnn_conformance_tests$'`.
- Enabled-backend runs must fail rather than skip when hardware is unavailable; no build, test, lint, formatter, benchmark, or accelerator gate is part of writing this specification.
