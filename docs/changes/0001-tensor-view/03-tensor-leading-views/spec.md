# Implement materialized tensor and leading-dimension views

**Order:** 03
**Priority:** P0 — this is the operand and ownership contract required by every device and queue.
**Blocked by:** `02-core-metadata-layout`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

`Tensor` is a stable non-copyable materialized owner, `TensorView` is a copyable but non-assignable non-owning operand, and every safe leading-dimension transform produces checked plane offsets and strides without moving or allocating tensor storage.

## Scope

- Add the backend-neutral `Device` interface and complete `Tensor`/`TensorView` ownership, accessors, host-transfer surface, and leading-dimension transformations.
- Test common behavior with small fake `Device` and `Tensor` subclasses; concrete CPU storage is outside this task.

## Implementation references

- **Modify:** `include/iom/tensor.hpp` — replace the placeholder `Tensor`, add `TensorView`, and forward-declare `Device`.
- **Create (planned):** `include/iom/device.hpp` — declare the subclassable backend-neutral `Device` contract and forward-declare `DeviceOps`.
- **Modify:** `src/iom.cpp` — implement the protected `Tensor` constructor, stable full view, accessors, transfer delegation, dense plane strides, and all four metadata transforms.
- **Modify:** `test/test_iom.cpp` — add fake owner/device fixtures and exhaustive transform/reference-map tests.
- **Read:** `docs/design/README.md` — reuse the established rule that only leading dimensions are view-transformable and tensor operations do not allocate outputs.

## Requirements

- `Device` exposes virtual `backend_kind()`, `backend_device()`, `create_tensor(const TensorSpec&)`, and `create_ops()`. Its public common header contains no CUDA, HIP, SYCL, or TTNN types and no registry or backend switch.
- `Tensor` has no default or empty state, validates its `TensorSpec` in its protected constructor, stores its creating `Device`, owns one stable full `TensorView`, and deletes copy and move construction and assignment.
- `Tensor::view()` always returns the same full-view object. Its plane offset is zero; its leading strides are dense row-major plane strides; rank two has an empty stride vector.
- `TensorView` stores a non-owning owner pointer plus its own spec, plane offset, and plane strides. It is copy/move constructible but neither copy- nor move-assignable.
- `spec()`, `device()`, `backend_kind()`, `backend_device()`, `native_handle()`, `plane_offset()`, and `plane_strides()` report the owner/view state exactly. A derived view's native handle is always its owner's storage handle.
- `slice(dim, first, count, step)` accepts leading dimensions only, requires nonzero count and step, checks the last selected index, advances the offset, and multiplies only that dimension's stride.
- `select(dim, index)` accepts leading dimensions only, advances the offset, and erases exactly the selected dimension and stride without recomputing the others.
- `permute(leading_order)` requires an exact permutation of all leading axes and reorders dimensions and matching strides. The required empty rank-two permutation is a no-op.
- `reshape_leading(new_dimensions)` preserves the final two dimensions and may split, merge, add, or remove size-one leading axes only when plane counts match and the source view is contiguous in reverse leading-axis order. Empty leading dimensions have product one; zero dimensions are invalid.
- Every offset, stride, product, and last-index operation is overflow-checked. Every result remains in-bounds and non-overlapping.
- The API exposes no final-two-dimension transform, raw `as_strided`, negative stride, or broadcast stride.
- `copy_from_host` and `copy_to_host` require exactly `spec().logical_nbytes()` and delegate the logical region to the owner hooks. `copy_from_host` rejects nonzero/non-one `BOOL` bytes before invoking a backend write.
- Host access does not implicitly synchronize operation queues. Before a host read, the caller waits for outstanding writes to participating storage; before a host write or owner destruction, the caller waits for every outstanding read/write involving that storage.
- Only metadata vectors may allocate during transforms. The owner and creating `Device` must outlive every derived view.

## Non-goals

- Concrete CPU or accelerator allocation, packing, kernels, or queues.
- Owning views, lifetime tracking, implicit synchronization, or compatibility inheritance from `TensorView`.
- Reshaping, slicing, selecting, or permuting either tiled dimension.

## Acceptance criteria

- [ ] A compile fixture implements a custom `Device` and fake `Tensor` without backend headers.
- [ ] Full-view identity, zero offset, dense strides, stable native handle, deleted tensor copy/move, and non-assignable views are compile- and runtime-checked.
- [ ] Stepped/unstepped slices, selects, every leading permutation, contiguous reshape split/merge, and size-one edits yield exact shape/offset/strides.
- [ ] Rank-two empty permutation and empty-product reshape boundaries work.
- [ ] Nested transforms at ranks above four map every logical leading coordinate to the same owner plane as an independent reference calculation.
- [ ] Non-leading dimensions, zero count/step, out-of-range coordinates, invalid permutations, non-contiguous reshapes, product mismatch, and arithmetic overflow throw the specified exception classes.
- [ ] Transform and transfer-delegation tests prove owner storage is neither allocated nor moved.

## Verification

- `c++ -std=c++20 -Iinclude -DDOCTEST_CONFIG_IMPLEMENT_WITH_MAIN src/iom.cpp test/test_iom.cpp -o /tmp/iom-tensor-view-tests && /tmp/iom-tensor-view-tests --test-case="Tensor owner*,TensorView*"`
