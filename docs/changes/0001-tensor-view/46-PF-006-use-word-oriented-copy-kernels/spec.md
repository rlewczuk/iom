# Replace bit-granular GPU copies with destination-word ownership

**Order:** 46
**Priority:** P1 — medium, strongly-supported performance defect. Sub-byte copies currently use contended per-bit atomics, and ROCm reads byte-aligned values bit by bit.
**Blocked by:** `42-PF-002-grid-stride-gpu-copy`, `43-PF-003-cache-gpu-staging`
**Source:** `docs/changes/0001-tensor-view/review.md` — `PF-006`
**Review severity:** medium
**Review verification:** strongly-supported (mechanism code-deterministic; no GPU measurement), confidence 80

## Outcome

CUDA and ROCm copy kernels assign ownership by destination 32-bit storage word. Each owning thread loads its destination word once, extracts the source fields that overlap it with word loads plus shifts/masks, merges the fields in registers, and stores the destination word once with a non-atomic store. Byte-aligned widths use the same word path rather than per-bit extraction. The `bits == 64` path reads two source words and writes the two destination words through their respective owners.

The rewrite preserves the standard 16x16 tiled layout, LSB-first sub-byte packing, leading-plane stride mapping, host logical row-major order, exact logical byte counts, and identical-window no-op behavior. PF-002 owns the single grid-stride launch and metadata-slot lifetime; PF-003 owns cached host-transfer staging and streams. PF-006 changes only the device kernel helpers and their field-enumeration logic.

## Current failure

`src/cuda/copy.cu:139-166,180-195,235-253` and the equivalent ROCm code use `read_bits` and atomic per-bit writers. Multiple threads update the same destination word, so a sub-byte copy serializes on `atomicOr`/`atomicAnd`. ROCm also gathers byte-aligned values bit by bit before its byte writer, imposing the same cost on F32, F16, BF16, and integer widths.

The review identifies the violated invariant: primary weight formats must not serialize every bit, and CUDA/ROCm should have comparable word-oriented behavior. The fix must eliminate read/write atomics from the copy kernels without allowing two threads to update one destination word.

## Scope

- **Modify:** `src/cuda/copy.cu` and `src/rocm/copy.hip` at the post-PF-002 kernel and host-transfer kernel bodies. Replace `read_bits`, `write_bits_atomic`, and ROCm's bit-granular `write_bits` with TU-private word helpers and destination-word ownership.
- Keep PF-002's `grid_stride_copy_kernel` launch and metadata format. The device-to-device, scatter, and gather kernels decode the metadata supplied by the post-PF-002 slot pool.
- For each destination word, enumerate all logical fields whose destination bit range intersects that word, derive the source logical offset through the existing `plane_slot`/row-major mapping, merge the extracted field, and issue one final non-atomic store.
- Preserve `SubmissionFault::third_plane_launch` at the PF-002 post-launch wrapper if that seam remains in the implementation. PF-006 does not delete or rename fault-injection behavior owned by PF-002.

Out of scope: CPU/TTNN/SYCL kernels, tensor layout, tile size, metadata allocation, queue workers, stream pools, public APIs, and the PF-002/PF-003 ownership implementations.

## Implementation references

- **Modify:** `src/cuda/copy.cu` — replace bit helpers and update `grid_stride_copy_kernel`, scatter, and gather paths to use word ownership. Use existing CUDA launch/error helpers and context handling.
- **Modify:** `src/rocm/copy.hip` — mirror the CUDA helpers and mapping with HIP launch syntax. In particular, byte-aligned ROCm widths must use direct word extraction rather than `read_bits`.
- **Read/consume:** `docs/changes/0001-tensor-view/42-PF-002-grid-stride-gpu-copy/spec.md` — `CudaCopyMetadataHeader`/`HipCopyMetadataHeader`, dynamic leading-rank arrays, one grid-stride launch, event/fence resource, and `SubmissionFault` wrapper.
- **Read/consume:** `docs/changes/0001-tensor-view/43-PF-003-cache-gpu-staging/spec.md` — pooled `TransferStreamPool` and `StagingSlotPool` APIs. PF-006 must not name a per-call staging pool or invent a second stream allocator.
- **Read:** `include/iom/tensor.hpp` and `src/{cuda,rocm}/copy.*` — `TensorSpec::TILE == 16`, `plane_slot`, leaf bit-width table, LSB-first packing, and current host logical order.
- **Read/modify tests:** `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`, and shared copy conformance. Add focused word-width cases only if existing cases do not exercise 1, 2, 4, 6, 8, 16, 32, and 64-bit fields.

### Word helpers

Provide four TU-private device helpers with the same semantics in CUDA and ROCm:

```cpp
read_field_extracted(source_word_ptr, bit_offset, bits); // 1..32
read_field_pair_64(low_word_ptr, high_word_ptr, low, high); // 64 only
merge_field(destination_word, value, bit_offset, bits);
store_word(destination_word_ptr, destination_word);
```

`read_field_extracted` performs one or two 32-bit loads as needed, shifts the anchor bits, and masks without undefined shifts. `read_field_pair_64` returns the low and high 32-bit halves. `merge_field` performs an in-register mask-and-merge and never uses atomics. `store_word` performs the single final non-atomic store.

For a destination word `[w*32, w*32+31]`, the owning thread enumerates all fields whose bit ranges intersect that interval. Fully contained fields are merged in place. A field crossing a word boundary contributes its low part to the owner of `w` and its high part to the owner of `w+1`; each owner loads and stores only its own word. The code must handle widths 1, 2, 4, 6, 8, 16, 32, and 64 without rank or shape special cases.

The device-to-device and scatter paths enumerate destination tile fields from the PF-002 `global_word` decomposition. Gather enumerates logical row-major host words, then derives each source tiled slot with `plane_slot`. No host staging layout is changed. The launch grid remains based on the number of destination words, not logical element count.

## Requirements

1. No `atomicOr`, `atomicAnd`, per-bit read loop, or per-bit write loop remains in CUDA or ROCm copy kernels.
2. Exactly one work item owns each destination 32-bit word, and exactly one non-atomic store is issued for that word.
3. Destination words preserve untouched padding/neighbor bits by loading the pre-existing word before merging. No thread performs a read-modify-write on a word owned by another thread.
4. Widths 1, 2, 4, 6, 8, 16, 32, and 64 produce the same logical values as the CPU reference. Width 64 uses only the explicit two-word helper.
5. CUDA and ROCm use the same field extraction, mask, merge, and ownership rules; only SDK types, launch syntax, and error APIs differ.
6. Existing `plane_slot`, leading-plane stride traversal, tiled destination mapping, and host logical row-major mapping remain correct for arbitrary validated leading rank.
7. PF-002's metadata pointer and one-launch shape remain intact. PF-003's exact pooled stream/staging APIs remain intact; no per-call staging allocator, per-call stream abstraction, or backend-specific substitute API is introduced.
8. Existing transactional fault injection, including `SubmissionFault::third_plane_launch` where retained by PF-002, remains observable with the existing valid-token/repeated-wait semantics.
9. No public header, allocator, `DeviceOps` signature, queue worker, or tensor ownership contract changes.
10. The implementation contains no placeholder or second backend-specific copy of the algorithm; CUDA and ROCm helpers remain line-for-line policy mirrors where the runtime APIs permit.

## Non-goals

- Changing PF-002's grid-stride launch, metadata-slot pool, event pool, fence resource, or queue submission protocol.
- Changing PF-003's pooled transfer streams, staging slots, growth policy, or device ownership.
- Replacing the standard tile-major layout, changing bit order, adding conversion, or changing host-visible byte counts.
- CPU, TTNN, SYCL, numerical compute, public API, or ABI changes.
- Performance claims without hardware timing; correctness and instruction/atomic reduction are the required mechanism.

## Acceptance criteria

- [ ] Source inspection finds zero `atomicOr`/`atomicAnd` and zero `read_bits`/`write_bits` helpers in the CUDA and ROCm copy kernels.
- [ ] Focused CUDA and ROCm tests round-trip every required field width, including odd sub-byte widths and 64-bit values, through device-to-device and host scatter/gather paths.
- [ ] Existing shared conformance passes for all supported leaf types, owner shapes, transformed views, high ranks, and identical windows.
- [ ] Instrumented or disassembled kernels demonstrate one destination-word store per owning work item and no atomic write instruction.
- [ ] ROCm byte-aligned F32/BF16/F16/integer cases use word extraction and match CUDA logical output.
- [ ] Existing PF-002 fault-injection and PF-003 host-transfer tests retain their prior observable behavior.
- [ ] `src/cuda/copy.cu` and `src/rocm/copy.hip` contain no invented PF-003 symbol and no placeholder body.
- [ ] Public headers and exported backend symbols are unchanged.

## Verification

Follow `.agents/skills/remote-development` for CUDA and ROCm builds and execution. Run the backend conformance suites and focused width/round-trip cases on their respective hardware under exclusive access. Use CUDA disassembly or Nsight and ROCm ISA/profiling output to confirm absence of atomic writers and the one-word ownership shape. Compare I4 and F32 copy bandwidth only as measured evidence; do not treat a timing threshold as proof of correctness. Run source audits for the removed bit helpers and for exact PF-002/PF-003 symbol usage, then clean remote mirrors.
