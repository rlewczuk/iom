# Extract core tensor and view definitions

**Order:** 01
**Priority:** P0 — establishes private core linkage and removes the first oversized core responsibilities
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Split the tensor shape/spec/layout and tensor-view responsibilities out of the oversized core translation unit without changing the library contract. The resulting private linkage is explicit: `src/tensor.cpp` owns shape/spec/layout definitions, `src/tensor_view.cpp` owns view and tensor-owner/view definitions, and the remaining `src/iom.cpp` uses the minimal private `src/iom_internal.hpp` for helpers shared with those files. Every moved definition is emitted exactly once. The new translation units and private header must satisfy the 499-physical-line limit immediately; the intentionally retained intermediate `src/iom.cpp` is removed by `05-core-device-ops-cutover`, which enforces the final global cap.

## Scope

Only the following production files are planned for this task:

- `src/iom.cpp` — remove the moved definitions while retaining all remaining workspace and `DeviceOps` definitions; include and consume `src/iom_internal.hpp` for shared checked arithmetic, rank constants, and `UnsupportedOperation`.
- `src/tensor.cpp` — new file containing the moved shape/spec/layout definitions.
- `src/tensor_view.cpp` — new file containing the moved stride/view/tensor definitions.
- `src/iom_internal.hpp` — new private header containing only the shared checked-arithmetic helpers, rank constants, and `UnsupportedOperation`.
- `CMakeLists.txt` — add the two new `.cpp` files to `IOM_SOURCES`, while retaining `src/iom.cpp` for its remaining definitions.

`include/iom/tensor.hpp` and `test/test_iom.cpp` are references and must not be modified. No other path is in scope.

## Implementation references

- Current shape/spec/layout block: `src/iom.cpp:22-221`.
- Current tensor-view and tensor block: `src/iom.cpp:227-540`.
- Public declarations that define the required signatures and visibility: `include/iom/tensor.hpp`, especially `TensorShape`, `TensorSpec`, `iom::detail::standard_layout_slot`, `iom::detail::leaf_bits`, `iom::detail::standard_plane_slot`, `TensorView`, and `Tensor`.
- Current source registration: `CMakeLists.txt:74-90` (`IOM_SOURCES` and `IOM_HEADERS`).
- Existing observable behavior coverage: `test/test_iom.cpp`, including the shape/spec/layout cases and the Tensor/TensorView transform, host-transfer, arithmetic-overflow, ownership, and accessor cases.

## Requirements

### `src/tensor.cpp`

Move the definitions from `src/iom.cpp:22-221` that belong to tensor shape/spec/layout:

- `TensorShape::TensorShape`, `TensorShape::rank`, `TensorShape::dimension`, `TensorShape::dimensions`, and `TensorShape::element_count`.
- `TensorSpec::validate`, `TensorSpec::standard_padded_shape`, `TensorSpec::logical_nbytes`, and `TensorSpec::tiled_storage_nbytes`.
- `iom::detail::leaf_bits`.
- `iom::detail::standard_plane_slot` and `iom::detail::standard_layout_slot`.
- Tensor-local tile constants and layout arithmetic needed by those definitions.

Use the private checked arithmetic and rank definitions from `src/iom_internal.hpp` where shared. Preserve the standard 16-by-16 tiled layout, leading-plane order, final-two-dimension padding, exact leaf bit widths, and all overflow and coordinate/spec validation behavior.

### `src/tensor_view.cpp`

Move the definitions from `src/iom.cpp:227-540` that belong to leading dimensions, strides, views, host transfers, and tensor ownership:

- The anonymous helpers `with_leading_dimensions`, `leading_dimensions_of`, `dense_plane_strides`, and `validated_dense_plane_strides`.
- `TensorView::TensorView`.
- `TensorView::owner_identity`, `spec`, `device`, `backend_kind`, `backend_device`, both `native_handle` overloads, `plane_offset`, and `plane_strides`.
- `TensorView::slice`, `select`, `permute`, and `reshape_leading`.
- `TensorView::copy_from_host` and `copy_to_host`.
- The shared host-transfer requirement calculation and both
  `TensorView::copy_from_host_workspace_requirements` and
  `TensorView::copy_to_host_workspace_requirements`.
- `Tensor::Tensor` and both `Tensor::view` overloads.

Use the private checked arithmetic and rank definitions from `src/iom_internal.hpp` where shared. Preserve view owner identity, offsets, plane strides, transform validation, rank bounds, contiguity rules, host byte/BOOL validation, backend-specific workspace requirements, and delegation to the owner regions. Tensor construction must validate its specification before any full view or storage effect, as it does today.

### `src/iom_internal.hpp` and the remaining `src/iom.cpp`

Create `src/iom_internal.hpp` as a private backend-neutral linkage header. It may contain only:

- checked `std::size_t` addition, multiplication, and bit-to-byte conversion used by the core definitions;
- the implementation-private rank constant(s), including the rank-eight bound needed by shape/spec/view and remaining core validation;
- `UnsupportedOperation` and its existing message/exception behavior.

Do not turn this header into a miscellaneous utility collection or expose it through a public header. Update the remaining `src/iom.cpp` definitions to include and use it, removing duplicate local definitions while preserving all current validation, failure mapping, operation behavior, and exception-to-`OidError::Unsupported` mapping. Keep all workspace, registry, queue, and `DeviceOps` code not named above in `src/iom.cpp`.

### Build integration and linkage

Update `CMakeLists.txt` so `IOM_SOURCES` contains `src/tensor.cpp`, `src/tensor_view.cpp`, and the retained `src/iom.cpp` in addition to its existing sources. Do not remove `src/iom.cpp`, change test target source lists, alter public factory headers, or add the private header as a public API. Include the public declarations from `include/iom/tensor.hpp` without changing that header. No moved non-inline definition may remain in `src/iom.cpp`, and no new definition may be duplicated across the new translation units.

### Universal behavior invariants

This is source factoring only. Preserve all public headers, signatures, capabilities, error categories and validation order, ownership and lifetimes, allocation behavior, asynchronous in-order queues, repeatable waits and failures, registry/quarantine semantics, workspace/staging lease completion proof, context behavior, and numerics. Preserve private visibility and ODR behavior. The change must not introduce compatibility aliases or shims, backend switches, global registries, cross-backend abstractions, synchronization redesign, extra capability, or altered test registration.

Each new production `.cpp` or `.hpp` created by this task must be at most 499 physical lines after normal formatting. `src/iom.cpp` is an intermediate migration carrier and may remain oversized until `05-core-device-ops-cutover` removes it; the final integrated change still requires every retained original, touched, or new production `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl` under `src/` or `include/` to be at most 499 lines. Use the fewest responsibility-complete files.

## Non-goals

- Do not change `include/iom/tensor.hpp`, any public signature, public header availability, or public factory header.
- Do not change tensor/view arithmetic, layouts, validation order, exception categories/messages, host-transfer effects, ownership, allocation behavior, or backend capability.
- Do not move workspace, registry, queue, or unrelated `DeviceOps` responsibilities out of `src/iom.cpp` as part of this task.
- Do not add tests, modify `test/test_iom.cpp`, change test source lists, or add a permanent line-count test.
- Do not add backend switches, global registries, cross-backend abstractions, synchronization changes, compatibility aliases/shims, or speculative helper layers.
- Do not leave a compatibility translation unit or duplicate moved definitions.

## Acceptance criteria

1. `src/tensor.cpp` contains exactly one definition of every shape/spec/layout symbol listed in its scope, and `src/tensor_view.cpp` contains exactly one definition of every stride, TensorView, host-requirement, Tensor-constructor, and Tensor-view symbol listed in its scope.
2. `src/iom.cpp` no longer contains the definitions moved to either new file, retains its remaining definitions, and consumes the single private `src/iom_internal.hpp` definitions without duplicate local helpers or duplicate `UnsupportedOperation`.
3. `src/iom_internal.hpp` contains only checked arithmetic, rank constants, and `UnsupportedOperation`; it is private and does not change public header contents or visibility.
4. `CMakeLists.txt` builds all three core translation units (`src/iom.cpp`, `src/tensor.cpp`, and `src/tensor_view.cpp`) exactly once, with no test target or public factory source-list change.
5. Existing `iom_tests` behavior remains unchanged: rank and dimension rules, exact byte counts and tiled layout slots, overflow failures, view transforms and mappings, host transfer checks, workspace requirement values, owner/device/native-handle accessors, and construction-before-storage validation all remain observable as before. Universal ownership, lifetime, queue-order, registry/quarantine, lease-completion, context, error, and numerical invariants remain unchanged.
6. `include/iom/tensor.hpp` and `test/test_iom.cpp` are byte-for-byte untouched by the implementation.
7. `src/tensor.cpp`, `src/tensor_view.cpp`, and `src/iom_internal.hpp` are each at most 499 physical lines after normal formatting. The intermediate `src/iom.cpp` is removed by `05-core-device-ops-cutover`; it is not required to meet the final retained-file cap during this staged extraction.

## Verification

The future implementer must run these commands in an isolated CPU-only build tree after the cutover; they are instructions, not gates for this specification author:

```sh
cmake -S . -B build/split-cpu \
  -DBUILD_TESTING=ON \
  -DCUDA_ENABLED=OFF \
  -DROCM_ENABLED=OFF \
  -DSYCL_ENABLED=OFF \
  -DTTNN_ENABLED=OFF
cmake --build build/split-cpu --target libiom iom_tests
ctest --test-dir build/split-cpu --output-on-failure -R '^iom_tests$'
```
```sh
python3 - <<'PY'
from pathlib import Path

paths = [
    Path("src/tensor.cpp"),
    Path("src/tensor_view.cpp"),
    Path("src/iom_internal.hpp"),
]
for path in paths:
    lines = sum(1 for _ in path.open(encoding="utf-8"))
    if lines > 499:
        raise SystemExit(f"{path}: {lines} lines")
print("scoped new-file line-count check passed")
PY
```

The line-count command is a one-time post-extraction check, not a permanent test. Record the retained `src/iom.cpp` count without treating its intermediate size as this task's failure; `05-core-device-ops-cutover` removes it and runs the final tracked-production scan.
Also verify by source inspection or symbol search that each listed moved definition occurs exactly once, `src/iom.cpp` retains only its in-scope remainder, and the public header/test files were not modified. Hardware and non-CPU builds are outside this mini-spec’s focused verification commands; no formatter, linter, or additional build/test gate is required here.

