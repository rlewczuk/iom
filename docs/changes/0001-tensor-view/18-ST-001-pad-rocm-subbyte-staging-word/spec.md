# Pad ROCm sub-byte host-read staging through the last 32-bit atomic word

**Order:** 18
**Priority:** P0 — high-severity verified out-of-bounds device read-modify-write (memory safety)
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-001`
**Review severity:** high
**Review verification:** verified, confidence 99

## Outcome

Every 32-bit atomic word that `gather_plane_kernel` touches while assembling sub-byte host output lies entirely inside the ROCm staging allocation, the staging padding is zeroed before the gather so output tail bits read as zero, and only `logical_nbytes()` bytes ever reach host memory. A focused `{1,17}` sub-byte `copy_to_host` regression test passes cleanly under the ROCm device AddressSanitizer on ROCm hardware, where the uncorrected kernel reports an out-of-bounds write.

## Current failure

Invariant: every device access must remain within the allocated object, including the word-sized atomic operations used to assemble sub-byte host output.

`synchronous_transfer` in `src/rocm/copy.hip` (lines 263–306) allocates staging with `hipMalloc(&staging, view.spec().logical_nbytes())` (line 273) — exactly the logical byte count. During `copy_to_host` (`from_host == false`), it zeroes only those logical bytes (`hipMemset(staging, 0, view.spec().logical_nbytes())`, lines 282–285) and then `launch_view_transfer` runs `gather_plane_kernel` (lines 180–195), whose `write_bits` (lines 127–149) updates each sub-bit field through a 32-bit `atomicOr`/`atomicAnd` on the word at `base + (bit / 32) * sizeof(unsigned int)` (lines 139–146). When the last logical bit lives in a byte that is not the final byte of its 32-bit word, the atomic read-modify-write crosses the allocation bound:

- `{1,17}` with `I2`: 17 × 2 = 34 bits → `logical_nbytes()` = 5 bytes. The 17th element starts at bit 32, so its word covers bytes 4–7 — three bytes past the allocation.
- `{1,17}` with `F6_E2M3`/`F6_E3M2`: 17 × 6 = 102 bits → 13 logical bytes. The last word covers bytes 12–15 — three bytes past the allocation.

Every sub-byte `copy_to_host` whose logical byte count is not a multiple of 4 performs this out-of-bounds device read-modify-write. Depending on allocator granularity and runtime checking it corrupts adjacent device state, raises a device fault, or stays latent — in all cases violating the memory contract. The shared conformance matrix already exercises these exact cases (`transfer_owner_shapes()` includes `{1,17}` at `test/backend/backend_conformance_copy_storage.hpp:169-179`; `kRocmLeafTypes` includes `I2` and both `F6` variants at `test/rocm/test_rocm_conformance.cpp:24-38`) but passes without the ROCm device AddressSanitizer because the stray word lands inside allocator granularity. The CUDA analogue already solves this: `synchronous_transfer` in `src/cuda/copy.c…

## Scope

- Change the staging sizing and zeroing inside `synchronous_transfer` (`src/rocm/copy.hip`) only: both transfer directions allocate the word-rounded staging size, and the to-host branch zeroes the whole rounded size.
- Host-visible transfer behavior is unchanged and must remain compliant with `docs/changes/0001-tensor-view/08-rocm-storage-copy/spec.md`: exact logical byte counts, padding never exposed to the host, output tail bits zero.
- Affected backend: ROCm/HIP only. CUDA already pads (`src/cuda/copy.cu`); CPU and TTNN have no staging atomics. No public interface changes.
- Error behavior: add the same defensive `std::overflow_error` size guard the CUDA path already has, thrown before any allocation; every other exception, cleanup path, and category is unchanged.

## Implementation references

- **Modify:** `src/rocm/copy.hip` — `synchronous_transfer` (lines 263–306): the `hipMalloc` at line 273 and the to-host `hipMemset` at lines 282–285 are the only staging sizing/zeroing sites; the `hipMemcpy` DtoH at lines 290–292 already copies `destination.size()`. `write_bits` (lines 127–149) and `gather_plane_kernel` (lines 180–195) explain the word-granularity requirement but are not modified. The public wrappers `region_from_host`/`region_to_host` at the bottom of the file pass through unchanged.
- **Read:** `src/cuda/copy.cu` — `synchronous_transfer` (lines 317–368): the established same-problem pattern to mirror — overflow check (`std::overflow_error("CUDA transfer staging size overflows")`) before allocation, `staging_nbytes = logical_nbytes + sizeof(unsigned int)`, `cudaMemset` over the full `staging_nbytes` in the to-host branch only, DtoH copy of `destination.size()` only.
- **Read:** `src/iom.cpp` — `TensorView::copy_to_host` (lines 488–494) proves `destination.size() == spec().logical_nbytes()` on entry; `TensorSpec::logical_nbytes` (lines 130–135) is the ceil-bits-to-bytes logical size; `checked_add`/`checked_mul`/`bits_to_bytes` (lines 23–39) establish the `std::overflow_error` sizing convention.
- **Read:** `test/backend/backend_conformance_common.hpp` — `encode_logical` (lines 115–127), `kReadbackSentinel`/`read_logical` (lines 135–141), and `require_logical_bytes` (lines 158–166): the sentinel-prefilled readback that asserts exact bytes and zero tail bits; reuse, do not duplicate.
- **Tests:** `test/rocm/test_rocm_conformance.cpp` — add one focused `TEST_CASE` alongside the existing ones (lines 97–195), following the minimal device pattern of the "deferred queue lifetime" case (lines 162–169): `TrafficGate` + `HipAllocator` + `make_rocm_device(0, ...)`; the header `backend_conformance_common.hpp` is already included (line 15).

## Requirements

- `synchronous_transfer` computes `logical_nbytes = view.spec().logical_nbytes()` once and derives `staging_nbytes` as the smallest multiple of `sizeof(unsigned int)` that is `>= logical_nbytes` (checked round-up through the end of the last 32-bit atomic word, not an unconditional extra word). If the round-up would overflow `std::size_t`, throw `std::overflow_error("ROCm transfer staging size overflows")` before any `hipMalloc` or stream interaction, mirroring the guard in `src/cuda/copy.cu`.
- Both transfer directions allocate `staging_nbytes`: `hipMalloc(&staging, staging_nbytes)`.
- The to-host branch (`from_host == false`) zeroes the full rounded size: `hipMemset(staging, 0, staging_nbytes)`. The padding is what makes tail bits of the final partial word read as zero, because `atomicOr` can only set bits.
- Host-visible lengths are unchanged: the to-host `hipMemcpy` copies exactly `destination.size()` bytes (equal to `logical_nbytes`, validated upstream by `TensorView::copy_to_host`), and the from-host `hipMemcpyAsync` copies `source.size()` logical bytes into the padded staging. Staging padding never reaches host memory and the from-host branch needs no memset (the scatter path reads only logical bits).
- The corrected sizing must guarantee, for every leaf width and every shape, that all 32-bit words touched by `write_bits` during a gather lie fully inside `[staging, staging + staging_nbytes)`. Word alignment continues to rely on `hipMalloc`'s guaranteed alignment exactly as today; add no alignment handling.
- No other transfer semantics change: `write_bits` keeps its per-bit `atomicOr`/`atomicAnd` strategy and byte-aligned fast path, the scatter and queue copy kernels are untouched, and the existing `catch (...)` cleanup (stream synchronize/destroy, `hipFree`, rethrow) is unchanged.
- Add a regression `TEST_CASE` in `test/rocm/test_rocm_conformance.cpp` named `"ROCm conformance: sub-byte odd-length host reads stay within the staged atomic word"`. For shape `{1,17}` with `DataType::I2`, `DataType::F6_E2M3`, and `DataType::F6_E3M2` (logical sizes 5 and 13 bytes; final atomic words ending at bytes 7 and 15): create the tensor on the ROCm device, seed with `copy_from_host(encode_logical(spec, salt))` for a fixed salt, read back through `read_logical`, and assert with `require_logical_bytes` that the sentinel-prefilled readback matches the encoding bit-for-bit — which pins both exact bytes and zero tail bits. The case must run under the ROCm device AddressSanitizer per Verification.

## Non-goals

- Rejected alternative, recorded to prevent scope drift: replacing `write_bits`'s per-bit atomics with a packed-byte writer whose ownership granularity cannot cross the logical bound. The review lists it as preferable, but it redesigns a device helper shared by the scatter, gather, and copy kernels — whose other destinations are already bounds-safe padded tensor storage — while rounding up the staging allocation is the minimal fix with a proven analogue in `src/cuda/copy.cu`. Do not restructure `write_bits` in this task.
- No changes to the scatter (host→device) kernel path, `copy_plane_kernel`/queue copies, the CUDA backend (read-only reference), or device-side tensor storage sizing.
- No staging-allocation caching, transfer performance work, or benchmarking (the review records no established performance regression).
- No new public interface or error contract, and no forking of the shared conformance matrix; the focused regression case supplements it.

## Acceptance criteria

- [ ] On a HIP device, the focused test built with the ROCm SDK's documented device-ASan configuration (hipcc/clang `-fsanitize=address -fno-omit-frame-pointer` with the SDK's ASan runtime per the AMD ROCm docs; the implementer applies this recipe to `iom_rocm_conformance_tests` only — there is no project-wide ASan hook) reports zero memory errors and exits 0 under the ROCm device AddressSanitizer; the uncorrected kernel under the same build reports the out-of-bounds word write in `gather_plane_kernel`/`write_bits` past the 5-byte (`{1,17}` `I2`) and 13-byte (`{1,17}` `F6`) staging allocations.
- [ ] The focused test asserts bit-exact `{1,17}` round trips for `I2`, `F6_E2M3`, and `F6_E3M2` with sentinel-prefilled readbacks (exact bytes and zero tail bits), and the full `iom_rocm_conformance_tests` suite still passes on hardware.
- [ ] Staging is sized by checked round-up to a multiple of `sizeof(unsigned int)` with the `std::overflow_error` guard evaluated before allocation, and the to-host `hipMemset` covers the rounded size while the DtoH copy still transfers exactly `destination.size()` bytes.

## Verification

Use the repository remote-development procedure (`.agents/skills/remote-development`); all ROCm evidence comes from the configured remote HIP host, never CPU-only substitution. Wrap commands with `flock` per the procedure if the remote GPU is shared.

1. Sync and build the focused target with the ROCm SDK's documented device-ASan configuration applied to `iom_rocm_conformance_tests` only — there is no project-wide ASan hook. The recipe is hipcc/clang `-fsanitize=address -fno-omit-frame-pointer` (host + device, per the AMD ROCm docs), with the SDK's ASan runtime:

```bash
.agents/skills/remote-development/scripts/remote-sync rocm 18-st001-subbyte-staging
.agents/skills/remote-development/scripts/remote-exec rocm 18-st001-subbyte-staging \
  'cmake -S . -B build/rocm -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm \
     -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
     -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
     -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
   && cmake --build build/rocm --target iom_rocm_conformance_tests -j'
```

2. Reproduce the reviewed failure once on the uncorrected tree (before applying this task's edit, or with it stashed): the ROCm device AddressSanitizer must report the out-of-bounds global write and exit nonzero. Run the ASan-instrumented binary directly under `flock` for exclusive GPU access per the remote-development skill:

```bash
.agents/skills/remote-development/scripts/remote-exec rocm 18-st001-subbyte-staging \
  'flock /tmp/agent-gpu0.lock ./build/test/iom_rocm_conformance_tests -tc="*sub-byte odd-length host reads*"'
```

Expected on the uncorrected kernel: the ROCm device AddressSanitizer reports the invalid write from `gather_plane_kernel`'s atomic path past the 5-byte and 13-byte staging allocations; the run exits nonzero.

3. Apply the fix, re-sync, rebuild, and re-run step 2: expected all subcases pass, zero ROCm device AddressSanitizer reports, exit status 0.

4. Full hardware conformance still passes:

```bash
.agents/skills/remote-development/scripts/remote-exec rocm 18-st001-subbyte-staging \
  'ctest --test-dir build/rocm --output-on-failure -R "^iom_rocm_conformance_tests$"'
```

5. Remove the remote mirror when finished: `.agents/skills/remote-development/scripts/remote-clean rocm 18-st001-subbyte-staging`.
