# TTNN host transfers allocate fresh staging per operation/plane

**Order:** 11
**Priority:** P1
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** PF-004
**Review area:** Performance
**Review severity:** medium
**Review verification:** strongly-supported, confidence 87
**Review scope:** whole-codebase
**Backend scope:** ttnn
**Location:** `src/ttnn/copy.cpp:78-142` (`ttnn_detail::upload_plane`); `src/ttnn/copy.cpp:230-280` (`ttnn_detail::region_to_host`)

## Outcome

TTNN host transfers reuse a device-owned staging facility held until the existing per-region queue finish: after warm-up, repeated uploads/downloads allocate no fresh host staging, padding-zero semantics are preserved, and failed transfers never reuse poisoned storage.

## Current problem

Repeated synchronous host transfers should separate reusable setup/staging storage from per-operation layout conversion and must not allocate avoidable large buffers on every transfer.

Each TTNN upload plane enters `upload_typed` and constructs a zero-filled `std::vector<T>` sized to the full padded plane (`src/ttnn/copy.cpp:89-91`), fills it, then constructs/moves a HostBuffer and host-tiled Tensor (`:111-116`) — one heap allocation and a full zero-fill per owner plane per upload, even when logical dimensions equal native padded dimensions and every element is subsequently overwritten by the tile conversion loop. Each download allocates a fresh `std::make_unique_for_overwrite<std::byte[]>(total_bytes)` for all padded planes (`src/ttnn/copy.cpp:251-252`) before issuing nonblocking downloads and a single finish (`:255-280`). No reusable TTNN host staging pool exists in the scoped source; CUDA/ROCm and SYCL have reusable staging pools. TTNN host-transfer methods hold the device API mutex and are synchronous (`src/ttnn/device.cpp:302-316`), so retained staging is safe until the existing finish. Impact: per-plane/per-operation heap allocations and full padded-buffer initialization on uploads, per-operation allocation and transient peak host memory on multi-plane downloads; the exact magnitude was not measured in the review.

## Scope

- Add a TTNN-device-owned reusable host staging facility under the existing `api_mutex`: typed tile buffers per supported native dtype for uploads and a byte staging buffer for downloads.
- Acquire staging per transfer and release it only after the existing queue finish; grow only when no retained buffer fits; initialize only the required padding (preserving padding-zero semantics).
- Poison/discard staging on transfer failure; keep the current one-finish-per-region ordering.

## Implementation references

- **Modify:** `src/ttnn/copy.cpp` — `upload_plane` (`:78-142`) and `region_to_host` (`:230-280`); own the per-transfer staging allocation sites.
- **Read:** `src/ttnn/device.cpp` — `TtnnTensor` host-transfer methods (`:302-316`) and the per-owner native `planes_` vector (`:226-258`); the API-mutex discipline and ownership model the staging facility must live under.
- **Read:** `src/shared/transfer_pool.hpp` and the CUDA/ROCm/SYCL staging pools — the reusable-pool convention to mirror in TTNN-shaped form.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — transfer/storage conformance anchors (`:468-519`) and shared storage matrices (aligned and padded rank-2 and multi-plane shapes).

## Requirements

- After warm-up, repeated transfers with the same or smaller padded shape perform no fresh host staging allocation (typed upload buffers and the download byte staging are reused); larger shapes grow staging once and then reuse it.
- All logical bytes and padding remain identical to today's behavior (zero-filled padded host tiles on upload; exact logical bytes on download); a transfer failure must not reuse poisoned staging.
- The facility must be safe under the existing API-mutex/finish ordering: retained buffers are reused only after the finish that consumed them; keep the one-finish-per-region ordering identical.

## Non-goals

- No change to native TTNN tensor ownership, plane mapping, API-mutex policy, queue-finish semantics, supported dtypes, or host logical encoding; no generic allocator/cache framework; no per-operation staging allocation unless lifetime requires it.

## Acceptance criteria

- [ ] An allocator counter shows zero fresh host staging allocations across repeated same-or-smaller padded uploads/downloads after warm-up, with larger shapes growing staging once then reusing it; bit-exact conformance for aligned and padded rank-2 and multi-plane transfers is unchanged.
- [ ] Injecting a transfer failure leaves the facility able to serve the next transfer with clean storage (no reuse of poisoned bytes), and the one-finish-per-region ordering is preserved.

## Verification

`actual validation: none (read-only review); proposed gates below`.

- TTNN (remote-host work per remote-development; requires TTNN hardware): configure with testing enabled, build, and run `ctest --test-dir <build> -R '^iom_ttnn_(smoke|conformance)_tests$' --output-on-failure`.
- Add a temporary allocation counter around repeated aligned/padded 1-plane and multi-plane host transfers and record peak resident host staging; expected: no fresh staging allocation after warm-up, identical logical/padding bytes, and no reuse of poisoned storage after an injected failure.

- `ctest --test-dir <build> -R '^iom_ttnn_(smoke|conformance)_tests$' --output-on-failure`