# Validate TTNN native extents before constructing any native tensor

**Order:** 17
**Priority:** P0 — high-severity correctness/memory-safety defect in the TTNN tensor-creation path (public tensor contract)
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `CC-001`
**Review severity:** high
**Review verification:** verified, confidence 98

## Outcome

`iom::Device::create_tensor` for `BackendKind::TTNN` rejects any specification whose leading-plane product overflows `std::size_t`, whose final two dimensions exceed the TTNN native-tile `std::uint32_t` extent, or whose resulting native plane count exceeds the TTNN runtime limit — all before constructing any `tt::tt_metal::TensorSpec`, `ttnn::Tensor`, or other native object — by throwing `std::overflow_error`. A successfully created `iom::Tensor` therefore always materializes exactly `plane_count` native planes matching its unchanged logical metadata.

## Current failure

The invariant is: shape, product, and backend-index conversions must be validated for overflow before any work or allocation; a successfully created tensor must materialize storage matching its unchanged `std::size_t` logical metadata. `TtnnDevice::create_tensor` at `src/ttnn/device.cpp:390-398` calls only `spec.validate()` and `is_supported(spec.data_type)` before forwarding to `TtnnTensor`. The `TtnnTensor` constructor at `src/ttnn/device.cpp:132-158` then narrows the final two dimensions with unchecked `static_cast<std::uint32_t>(dimensions[dimensions.size() - 2])` and `static_cast<std::uint32_t>(dimensions[dimensions.size() - 1])`, and computes the leading-plane count with the unchecked loop `plane_count *= dimensions[i]` (`src/ttnn/device.cpp:149-152`). The common `Tensor` constructor at `src/iom.cpp:496-498` likewise computes only the validated dense leading strides — it does not call `shape.element_count()`, so the product of leading dimensions is not checked anywhere on the TTNN path.

Concrete consequences (already realized in the current tree):

1. With `BF16` shape `{2^63, 2, 16, 16}` on a 64-bit host, `plane_count` wraps modulo `2^64` to zero. `planes_.reserve(0)` followed by `planes_.push_back(...)` zero times produces an empty native-plane vector; `create_tensor` returns a non-null `iom::Tensor` whose view advertises the original `2^63 * 2 * 16 * 16` logical planes. `ttnn::copy_to_device` is never invoked, and a later `copy_from_host`/`copy_to_host` does no work.
2. With shape `{1, 2^32 + 1}`, the unchecked `static_cast<std::uint32_t>` narrows the column count to `1`. The TTNN native tensor is constructed with one logical column, and `upload_plane` at `src/ttnn/copy.cpp:95-120` sizes `padded` from the truncated native `padded_shape()`. The logical row-major copy from `source` into the truncated `padded` buffer reads from the original `spec()`'s column count and writes past the truncated buffer.
3. Logical metadata continues to advertise the original larger shape through `TensorView::spec()` and `native_handle()`.

The established exception category for size-overflow rejections in this codebase is `std::overflow_error`, raised by the checked helpers `iom::detail::checked_mul` and `checked_add` (`src/iom.cpp:23-35`) and surfaced by `TensorShape::element_count()` (`src/iom.cpp:99-105`), `TensorSpec::logical_nbytes()` (`src/iom.cpp:130-135`), and `TensorSpec::tiled_storage_nbytes()` (`src/iom.cpp:137-144`). The TTNN path does not currently route through these helpers.

## Scope

- Pre-allocation rejection in `TtnnDevice::create_tensor` for every size-overflow condition that the TTNN native constructor cannot represent.
- Checked leading-plane product computation on the TTNN creation path that throws before any native object is constructed.
- Checked final-two-dimension validation against `std::uint32_t` and against the TTNN runtime's per-dimension native extent limit, on the TTNN creation path.
- A regression test that exercises the two overflow shapes called out in the review plus a largest-accepted-extent boundary case on real TTNN hardware.

## Implementation references

- **Modify:** `src/ttnn/device.cpp` — `TtnnDevice::create_tensor` (lines 390-398); reject the unsupported `DataType` first, then call `spec.shape.element_count()` (or an equivalent checked path) and throw `std::overflow_error` before constructing any native object when the product overflows `std::size_t`. Additionally validate each of the final two dimensions fits `std::uint32_t` and does not exceed the TTNN runtime's per-dimension native extent (`tt::tt_metal::Shape` accepts only `uint32_t` extents; the local TTNN runtime limit is `Shape::kMaxRank`-independent and bounded per axis by `std::numeric_limits<std::uint32_t>::max()`).
- **Modify:** `src/ttnn/device.cpp` — `TtnnTensor` constructor (lines 132-158); replace the unchecked `plane_count *= dimensions[i]` loop with a checked-multiplication pattern equivalent to `iom::detail::checked_mul` (defined at `src/iom.cpp:30-35`), throwing `std::overflow_error` with a message naming the leading-plane count overflow when the product wraps. Move that checked computation to a static anonymous-namespace helper (for example `checked_plane_count(const TensorSpec&)`) so `TtnnDevice::create_tensor` can call it before any native allocation.
- **Modify:** `src/ttnn/device.cpp` — `TtnnTensor` constructor (lines 136-148); replace the two unchecked `static_cast<std::uint32_t>` calls on `dimensions[dimensions.size() - 2]` and `dimensions[dimensions.size() - 1]` with checked conversions that throw `std::overflow_error` when the dimension exceeds `std::numeric_limits<std::uint32_t>::max()`. Reuse the same `checked_plane_count` (or a sibling `checked_to_uint32`) helper from the anonymous namespace.
- **Read:** `src/iom.cpp:23-35` — the existing `checked_mul`/`checked_add` helpers that throw `std::overflow_error`. Either mirror their bodies into the TTNN anonymous namespace (the TTNN translation unit is independent of common helpers' namespace structure) or move the helper into a shared internal header. Do not invent a second exception category.
- **Read:** `src/iom.cpp:99-105` — `TensorShape::element_count()` already throws `std::overflow_error` on leading-dimension overflow via `checked_mul`. Calling `spec.shape.element_count()` from `TtnnDevice::create_tensor` covers both the wrapped `plane_count` and any other leading-dimension overflow case; the TTNN-specific `checked_plane_count` exists only to validate before native allocation, not to redefine semantics.
- **Read:** `src/ttnn/device.cpp:74-80` — `invalid_ordinal` uses `std::invalid_argument`. The new size-rejection path uses `std::overflow_error`, matching the rest of the metadata and storage code paths and the failure mode called out in the review (which states "throws `std::overflow_error`").
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — extend the existing supported-type-rejection case with a new `TEST_CASE("TTNN rejects overflowing and narrowing extents before native allocation")` that follows the same `require_hardware()` + `make_ttnn_device(0)` pattern used in `TEST_CASE("TTNN supported-type table acceptance and rejection")` (lines 68-114). The new case asserts each of the three review-cited shapes throws `std::overflow_error` from `device->create_tensor` and never invokes a TTNN native constructor; it also asserts that the boundary case `{1, std::numeric_limits<std::uint32_t>::max()}` is accepted (since the largest accepted extent is retained per the review verification method).
- **Read:** `test/test_iom.cpp:124-127` and `test/test_iom.cpp:268-274` — the established pattern for `CHECK_THROWS_AS(..., std::overflow_error)` and `CHECK_NOTHROW` on `element_count()` / `logical_nbytes()`. The new TTNN tests use the same `CHECK_THROWS_AS` / `CHECK_NOTHROW` doctest macros for parity with the rest of the test suite.

## Requirements

- `TtnnDevice::create_tensor` (currently `src/ttnn/device.cpp:390-398`) must, after the existing `spec.validate()` and supported-`DataType` checks, call a checked leading-plane-count helper (or `spec.shape.element_count()`) and throw `std::overflow_error` if the product overflows `std::size_t` — all before constructing any `tt::tt_metal::TensorSpec`, `ttnn::Tensor`, or `ttnn::create_device_tensor`. The same path must also throw `std::overflow_error` when either final two dimensions exceeds `std::numeric_limits<std::uint32_t>::max()`. The exception message must name the offending dimension or product.
- The `TtnnTensor` constructor's unchecked `plane_count *= dimensions[i]` loop (`src/ttnn/device.cpp:149-152`) is replaced by a checked-multiplication path that throws `std::overflow_error` on the first overflow. The two unchecked `static_cast<std::uint32_t>` casts on the final two dimensions (`src/ttnn/device.cpp:139-142`) are replaced by checked conversions that throw `std::overflow_error` when the source dimension exceeds `std::uint32_t`. Both helper paths must produce `std::overflow_error`, not `std::invalid_argument`, to match the established exception category used by `TensorShape::element_count()`, `TensorSpec::logical_nbytes()`, and `TensorSpec::tiled_storage_nbytes()`.
- The runtime must never invoke `ttnn::create_device_tensor` (or any other TTNN native constructor that allocates) for a rejected specification. The pre-allocation rejection must happen while the device's `api_mutex_` is held only if necessary for the supported-`DataType` table; the size checks themselves are pure and must run before the lock if possible, matching the current ordering that locks only after the cheap validation. Whichever order is chosen, no TTNN runtime call is reached when an exception is thrown from this entry point.
- A tensor that does pass these checks must materialize exactly `plane_count` native planes via `ttnn::create_device_tensor`, and `plane_count` must equal the product of the validated leading dimensions. `TensorView::spec()` must continue to report the unchanged logical metadata.
- The `kSupportedDataTypes` table at `src/ttnn/device.cpp:37-47` and the `is_supported` helper at `src/ttnn/device.cpp:49-54` are unchanged in membership; the size checks are additive and run only after the type check.
- The new test cases run only when `TTNN_ENABLED=ON` and a real Tenstorrent device is present (`require_hardware()`), matching the established pattern at `test/ttnn/test_ttnn_conformance.cpp:58-60`. They must not silently skip.

## Non-goals

- Changing the supported-`DataType` table or adding new native types.
- Refactoring `TtnnTensor`'s destruction, region transfer, or queue plumbing.
- Generalizing the new size checks into a shared helper consumed by other backends.
- Adding new common-side checks to `TensorSpec`; the existing `TensorShape::element_count()` is sufficient and is reused.
- Computing or comparing the largest accepted extent at runtime from the TTNN SDK; the established `std::uint32_t` ceiling is the only additional axis bound, and the boundary case `{1, std::numeric_limits<std::uint32_t>::max()}` is accepted as the largest representable extent.
- Implementing or changing any `iom::Allocator` usage on the TTNN path; TTNN continues to own native storage without `iom::Allocator`.

## Acceptance criteria

- [ ] `device->create_tensor(spec)` for `BF16` shape `{2^63, 2, 16, 16}` throws `std::overflow_error` from `TtnnDevice::create_tensor`, and no `ttnn::Tensor` is constructed (no native-plane allocation occurs on the device's mesh).
- [ ] `device->create_tensor(spec)` for shape `{1, 2^32 + 1}` throws `std::overflow_error` from `TtnnDevice::create_tensor` before the `tt::tt_metal::TensorSpec` constructor is reached; no native tensor with a truncated column count is created.
- [ ] `device->create_tensor(spec)` for shape `{2, std::numeric_limits<std::size_t>::max() / 2 + 1, 16, 16}` (or any other shape whose `spec.shape.element_count()` overflows) throws `std::overflow_error` from `TtnnDevice::create_tensor` before the native `TensorSpec` is built.
- [ ] `device->create_tensor(spec)` for the boundary case `{1, std::numeric_limits<std::uint32_t>::max(), 16, 16}` with `BF16` does not throw and creates a TTNN tensor whose native padded shape matches the requested logical matrix; `TensorView::spec()` reports the unchanged logical metadata.
- [ ] Every existing TTNN conformance case in `test/ttnn/test_ttnn_conformance.cpp` and every smoke case in `test/ttnn/test_ttnn_smoke.cpp` continues to pass with no behavior change.
- [ ] No new failure paths reach a TTNN native constructor; the pre-allocation rejection is observable through the exception type and the absence of a returned `unique_ptr<Tensor>`.

## Verification

- Use the repository's remote-development procedure (`.agents/skills/remote-development`) to build and run against a remote Tenstorrent host. The local workspace is authoritative; sync via `.agents/skills/remote-development/scripts/remote-sync ttnn task-17-cc001` before each build or test, then execute remotely with `remote-exec`.
- Configure with TTNN enabled and every other accelerator disabled, matching the established TTNN isolation pattern (`13-ttnn-buildable-scaffold` and `14-ttnn-storage-copy`):

  ```bash
  .agents/skills/remote-development/scripts/remote-exec ttnn task-17-cc001 \
    'cmake -S . -B build/ttnn-cc001 -DBUILD_TESTING=ON -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF'
  .agents/skills/remote-development/scripts/remote-exec ttnn task-17-cc001 \
    'cmake --build build/ttnn-cc001 --target iom_ttnn_conformance_tests iom_ttnn_smoke_tests'
  .agents/skills/remote-development/scripts/remote-exec ttnn task-17-cc001 \
    'ctest --test-dir build/ttnn-cc001 --output-on-failure -R "^iom_ttnn_(conformance|smoke)_tests$"'
  ```

- Required observations: the new `TEST_CASE` in `test/ttnn/test_ttnn_conformance.cpp` reports three `CHECK_THROWS_AS(..., std::overflow_error)` passes for the `{2^63, 2, 16, 16}`, `{1, 2^32 + 1}`, and `{2, std::numeric_limits<std::size_t>::max() / 2 + 1, 16, 16}` shapes, and one `CHECK_NOTHROW` pass for `{1, std::numeric_limits<std::uint32_t>::max(), 16, 16}` with `BF16`. The existing supported-type-rejection and smoke cases continue to pass without skips.
- The remote test target must fail when the unchecked casts and unchecked `plane_count` product are present, and must pass after the fix — i.e., the test is observable evidence of the review's failure mode and of the corrected path.
- Clean up the remote mirror when done: `.agents/skills/remote-development/scripts/remote-clean ttnn task-17-cc001`.
