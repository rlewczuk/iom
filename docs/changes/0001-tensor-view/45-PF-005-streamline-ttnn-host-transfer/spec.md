# TTNN host transfers fill a typed tile-major buffer in place and write TILE layout directly

**Order:** 45
**Priority:** P1 — required TTNN host-memory and pass-count performance remediation whose smallest complete fix lives entirely inside `src/ttnn/copy.cpp`, satisfies the §5.4 host-transfer contract verbatim, preserves the device-wide `api_mutex_` discipline and synchronous-at-return/error behavior, and is validated on real Tenstorrent hardware through `.agents/skills/remote-development`.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `PF-005`
**Review severity:** medium
**Review verification:** verified (mechanism code-deterministic), confidence 85

## Outcome

`iom::ttnn_detail::region_from_host` and `region_to_host` (`src/ttnn/copy.cpp:160-199`) allocate exactly **one** full-padded-shape typed `tt::tt_metal::HostBuffer` per plane (no zeroed `std::vector<std::byte>` intermediate, no `typed_buffer` reinterpretation copy, no `to_layout(ROW_MAJOR → TILE)` re-pack on the upload path, no `to_layout(TILE → ROW_MAJOR)` re-pack on the download path). The host tensor is constructed directly in `tt::tt_metal::Layout::TILE` from that single typed buffer; the upload `ttnn::Tensor` host-side constructor and the upload `to_layout` call disappear.

For each plane the rewrite does exactly three things:

1. Allocate one typed `std::vector<T>` of size `padded_rows * padded_columns` (where `padded_rows = plane.padded_shape()[-2]` and `padded_columns = plane.padded_shape()[-1]` in elements of size `element_size`). The default `tt::tt_metal::Tile` constructed at `src/ttnn/device.cpp:202` is 32×32 with `get_face_shape() == {16, 16}` and `get_num_faces() == 4`, so `padded_rows` and `padded_columns` are already multiples of 32 and the element count is a multiple of 1024. The default `std::vector<T>` constructor zero-initializes every element, covering the padding rows and the padding columns within each row.
2. Walk the source logical `(row, column)` lattice face-by-face. A logical row segment that fits inside one 16-column face is one copy segment; a logical row that crosses a 16-column face boundary (e.g. `columns = 32`, which spans two faces of one tile) splits into two copy segments — the source payload for the left face and the source payload for the right face. For each `(row, column)` element within a copy segment, the destination element index in the typed buffer is given by the TT-Metal documented `convert_to_tile` offset:
   ```text
   face_row      = (row % 32) / 16
   face_col      = (column % 32) / 16
   face_index    = face_row * 2 + face_col                       // 0..3
   element_row   = row % 16
   element_col   = column % 16
   tile_index    = (row / 32) * (padded_columns / 32) + (column / 32)
   element_index = tile_index * 32 * 32
                  + face_index * 16 * 16
                  + element_row * 16
                  + element_col
   byte_offset   = element_index * element_size
   ```
   Per copy segment the rewrite issues one `std::memcpy` of `segment_columns * element_size` source bytes into the contiguous range of `segment_columns` element indices that share the same `(tile_index, face_index, element_row)` and walk `element_col = 0..segment_columns-1`. The per-element offset formula above is only computed at the start of each copy segment; the inner step walks contiguous destination element indices and contiguous source bytes without re-evaluating `face_index` or `tile_index`. Each copy segment stays strictly inside one face: starting at column `col_start`, the segment length is `segment_columns = min(16 - (col_start % 16), columns - col_start)`, so the segment never crosses the 16-column face boundary of its current tile.
3. Wrap the typed vector in `tt::tt_metal::HostBuffer(std::move(typed))`, construct the host `ttnn::Tensor` with `plane.logical_shape()`, `plane.padded_shape()`, `plane.dtype()`, and `tt::tt_metal::Layout::TILE`, and call `ttnn::copy_to_device(host_tiled, plane)`. The download mirror reads the tile-major bytes through `host_buffer::get_host_buffer(host_tiled.host_tensor())` and inverse-maps each source element from the tile-major offset formula above to the row-major destination offset `(row * columns + column) * element_size` (the §5.4 row-major contract).

The per-plane host footprint drops from `~3× padded_plane_bytes` to `~1× padded_plane_bytes`. Per-plane passes drop from `≥3` full-size passes (zero-init + per-row row-major memcpy + typed-buffer re-pack + `to_layout` internal) to `1` typed-vector construction (the zero-init) plus per-segment `std::memcpy` calls that touch only the logical element positions. Per logical row the rewrite issues at most `ceil(columns / 16)` segments (one per 16-column face encountered along the row) and at least `ceil(columns / 16)` total bytes touched; for LLM-standard 16-aligned column counts each row is exactly one segment. The `device.mesh_command_queue(0).finish()` calls at `src/ttnn/copy.cpp:176` and `src/ttnn/device.cpp:415` are unchanged. The device-wide `api_mutex_` (held by `TtnnTensor::region_from_host`/`region_to_host` at `src/ttnn/device.cpp:221-235` and by the queue worker at `src/ttnn/device.cpp:409-410`) continues to serialize every TTNN runtime call; the rewrite only shortens the critical section. Synchronous-at-return, exception categories, error propagation, and the §5.4 row-major destination contract are unchanged.

## Current failure

Invariant (review §5 PF-005): "Minimal memory footprint on both VRAM and host RAM" (`docs/design/README.md:4-5`); host transfers move each byte ≈ once.

Per plane in `region_from_host` (`src/ttnn/copy.cpp:95-120`):

1. `std::vector<std::byte> padded(padded_rows * padded_columns * element_size, std::byte{0})` allocates the full padded-shape byte vector and zero-initializes it (one full-size pass).
2. `for (row) memcpy(padded + row * padded_columns * element_size, source + row * columns * element_size, columns * element_size)` copies each logical row into the padded buffer's row slot. The destination offset `row * padded_columns * element_size` is the **row-major** offset; this is correct only because the host tensor is about to be declared `Layout::ROW_MAJOR`. The destination and source bytes match for the `columns` payload columns; the remaining `padded_columns - columns` padded columns per row were zeroed in step 1.
3. `ttnn::Tensor host_row_major(make_host_buffer(plane.dtype(), padded), …, ROW_MAJOR)` invokes `make_host_buffer` (`copy.cpp:72-91`) → `typed_buffer<T>(bytes)` (`copy.cpp:65-70`), which `std::vector<T> typed(bytes.size() / sizeof(T))` allocates a second full-padded-shape vector, `std::memcpy(typed.data(), bytes.data(), bytes.size())` copies the entire padded buffer into it (a third full-size pass), and constructs the `HostBuffer` from `std::move(typed)`. The comment at `copy.cpp:62-64` confirms this is purely a reinterpretation.
4. `to_layout(host_row_major, Layout::TILE)` re-packs the row-major host tensor into TTNN's tile-major layout (a fourth full-size pass internal to TTNN). This is the pass that actually performs the face-by-face transpose per TT-Metal's documented `convert_to_tile` (`face_index * 16 * 16 + element_row * 16 + element_col`): the row-major bytes written in step 2 are split into 32×32 tiles, each tile is rearranged into four 16×16 faces stored face-row-major (F0 top-left, F1 top-right, F2 bottom-left, F3 bottom-right), and the result replaces `host_row_major`.
5. `ttnn::copy_to_device(host_tiled, plane)` ships the tile-major host tensor to the device.

`region_to_host` (`copy.cpp:122-146`) mirrors this on the download path: `allocate_tensor_on_host(plane.tensor_spec(), &device)` allocates a full-padded-shape tile-major host tensor, `copy_to_host` populates it, `to_layout(TILE → ROW_MAJOR)` inverts the face transpose via `convert_from_tile` and produces a row-major host tensor, `host_buffer::get_host_buffer(host_row_major.host_tensor())` extracts the row-major bytes, and the per-row `memcpy` copies only the logical columns into the row-major destination.

Concrete per-plane cost, for an `F32 [B, S, H, 4096]` activation plane (`rows × columns × 4 bytes`, `padded_columns` and `padded_rows` both multiples of 32):

- Footprint: `3 × padded_plane_bytes` peak (the `padded` byte vector plus the `typed<T>` vector plus TTNN's internal `to_layout` scratch).
- Passes: zero-init (1) + per-row logical `memcpy` (1, full padded buffer) + `typed_buffer` `memcpy` (1, full padded buffer) + `to_layout` (1, full padded buffer inside TTNN) = 4 full-size passes per plane, × `view_plane_count` planes.
- The `typed_buffer` round-trip is purely a reinterpretation — every byte is moved twice for no semantic reason.
- The `to_layout(ROW_MAJOR → TILE)` pass materializes the tile-major host tensor inside TTNN; the layout conversion exists only because the source tensor was constructed in `Layout::ROW_MAJOR`.

The remediation performs steps 2 and 4 in a single pass: write the logical source bytes directly into a `Layout::TILE` typed vector at the TT-Metal documented face-major offset, so no separate `to_layout` step is needed. Steps 1 (zero-init) and 3 (typed reinterpretation) collapse into the typed-vector constructor.

## Scope

- `src/ttnn/copy.cpp:95-120` `upload_plane` — rewrite the body to:
  1. Read `padded_rows = static_cast<std::size_t>(plane.padded_shape()[-2])` and `padded_columns = static_cast<std::size_t>(plane.padded_shape()[-1])` (same reads as today).
  2. `const std::size_t num_tile_cols = padded_columns / 32;` — new, derived directly from the `padded_shape()` reads. The default `tt::tt_metal::Tile()` constructed at `src/ttnn/device.cpp:202` is 32×32 with `get_face_shape() == {16, 16}`, `get_num_faces() == 4`, and `transpose_tile = false`, so the divisor 32 is exact.
  3. A single switch over `plane.dtype()` (same six cases as `make_host_buffer` at `copy.cpp:74-86`, same `logic_error("TTNN native dtype has no host element type")` default) constructs a `std::vector<T> typed(padded_rows * padded_columns, T{0})` where `T` is `bfloat16`, `float`, `std::uint32_t`, `std::int32_t`, `std::uint16_t`, or `std::uint8_t`. The vector's default constructor zero-initializes every element, so padding rows and padding columns within each row start at zero.
  4. A face-by-face segment walk over the source logical `(row, column)` lattice. For each `row = 0..rows-1`, partition the row into copy segments of up to 16 contiguous source columns, one segment per 16-column face. For each copy segment starting at `column = col_start` with length `seg_columns = min(16 - (col_start % 16), columns - col_start)`, compute the destination element index of the segment's first element via the formula in Outcome §2: `tile_index = (row / 32) * num_tile_cols + (col_start / 32); face_index = ((row % 32) / 16) * 2 + ((col_start % 32) / 16); first_element_index = tile_index * 1024 + face_index * 256 + (row % 16) * 16 + (col_start % 16);`. Then issue one `std::memcpy` of `seg_columns * element_size` source bytes from `source + (row * columns + col_start) * element_size` into `reinterpret_cast<std::byte*>(typed.data()) + first_element_index * element_size`. The destination indices `first_element_index + 1, ..., first_element_index + seg_columns - 1` are contiguous because the segment stays strictly inside one face (it never crosses a 16-column face boundary); no further `tile_index` / `face_index` arithmetic is performed in the inner step. A logical row whose `columns` count is 16 or less produces one segment per row; a 32-column logical row produces two segments (one at `face_index = 0`, one at `face_index = 1`); an `n * 16`-column row produces `n` segments.
  5. `tt::tt_metal::HostBuffer host_buffer(std::move(typed));` — same constructor used today at `copy.cpp:69`.
  6. `ttnn::Tensor host_tiled(std::move(host_buffer), plane.logical_shape(), plane.padded_shape(), plane.dtype(), tt::tt_metal::Layout::TILE);` — same constructor signature as today at `copy.cpp:112-115` but with `Layout::TILE` instead of `Layout::ROW_MAJOR`.
  7. `ttnn::copy_to_device(host_tiled, plane);` — unchanged from line 119.
  No `std::vector<std::byte> padded`, no `to_layout`, no intermediate row-major byte buffer, no per-element offset computation inside the inner step (the offset is computed once per copy segment).
- `src/ttnn/copy.cpp:122-146` `download_plane` — drop the `to_layout(TILE → ROW_MAJOR)` line at `copy.cpp:132-133`. `allocate_tensor_on_host(plane.tensor_spec(), &device)` already yields the native tile-major host tensor; `ttnn::copy_to_host(plane, host_tiled, /*blocking=*/true)` populates it; `host_buffer::get_host_buffer(host_tiled.host_tensor())` returns the tile-major bytes. The download mirrors the upload segment walk: for each logical row `row = 0..rows-1`, partition the row into segments of up to 16 contiguous source columns inside one face. For each segment starting at `col_start` with length `seg_columns = min(16 - (col_start % 16), columns - col_start)`, compute the source element index `first_element_index = tile_index * 1024 + face_index * 256 + (row % 16) * 16 + (col_start % 16)` (same formula as upload, applied to the tile-major source buffer), and issue one `std::memcpy` of `seg_columns * element_size` bytes from the tile-major source at `first_element_index * element_size` into the row-major destination at `(row * columns + col_start) * element_size` (the §5.4 row-major destination contract, identical to the destination offset today's code uses at `copy.cpp:142`). No `to_layout`.
- `src/ttnn/copy.cpp:160-199` `region_from_host` / `region_to_host` — no change to the per-plane loop structure, `view_plane_count` / `owner_plane_at` / `view_rows` / `view_columns` helpers, the `device.mesh_command_queue(0).finish()` call, or the caller's `std::fill` of the row-major destination.
- `src/ttnn/copy.hpp:25-36` — declarations unchanged (still require the caller to hold the owning device's API mutex; still synchronous at return).
- `src/ttnn/device.cpp:211-235` `TtnnTensor::region_from_host` / `region_to_host` — no change; the device-wide `api_mutex_` continues to serialize these calls exactly as today.
- The `typed_buffer` template at `copy.cpp:65-70` is preserved verbatim. `make_host_buffer` is removed: its dtype switch is inlined into `upload_plane` with the same body and the same `logic_error` site.
- No new types, no new headers, no new CMake entries, no public-API change, no `iom::Allocator` traffic.

## Implementation references

- **Modify:** `src/ttnn/copy.cpp` — `upload_plane` (`copy.cpp:95-120`). The new body, in order:
  1. `const std::size_t padded_rows = static_cast<std::size_t>(plane.padded_shape()[-2]);` — unchanged from line 99.
  2. `const std::size_t padded_columns = static_cast<std::size_t>(plane.padded_shape()[-1]);` — unchanged from line 101.
  3. `const std::size_t num_tile_cols = padded_columns / 32;` and `const std::size_t padded_elements = padded_rows * padded_columns;` — new, derived directly from the `padded_shape()` reads. The default `tt::tt_metal::Tile()` constructed at `src/ttnn/device.cpp:202` is 32×32 with `get_face_shape() == {16, 16}`, `get_num_faces() == 4`, and `transpose_tile = false`, so the divisor 32 is exact.
  4. A single switch over `plane.dtype()` (same six cases as `make_host_buffer` at `copy.cpp:74-86`, same `logic_error("TTNN native dtype has no host element type")` default) constructs a `std::vector<T> typed(padded_elements, T{0})` where `T` is `bfloat16`, `float`, `std::uint32_t`, `std::int32_t`, `std::uint16_t`, or `std::uint8_t`. The vector's default constructor zero-initializes every element, so padding rows and padding columns within each row start at zero.
  5. A face-by-face segment walk that copies one row-major segment of up to 16 contiguous source columns per face:
     ```cpp
     for (std::size_t row = 0; row < rows; ++row) {
         std::size_t col = 0;
         while (col < columns) {
             const std::size_t seg_columns =
                     std::min<std::size_t>(16 - (col % 16), columns - col);
             const std::size_t tile_index =
                     (row / 32) * num_tile_cols + (col / 32);
             const std::size_t face_index =
                     ((row % 32) / 16) * 2 + ((col % 32) / 16);
             const std::size_t first_element_index =
                     tile_index * 1024
                     + face_index * 256
                     + (row % 16) * 16
                     + (col % 16);
             std::memcpy(
                     reinterpret_cast<std::byte*>(typed.data())
                             + first_element_index * element_size,
                     source + (row * columns + col) * element_size,
                     seg_columns * element_size);
             col += seg_columns;
         }
     }
     ```
     The destination element indices `first_element_index, first_element_index + 1, ..., first_element_index + seg_columns - 1` are contiguous because the segment stays strictly inside one face: `seg_columns = min(16 - (col % 16), columns - col)` never crosses the 16-column face boundary. A logical row whose `columns` count is 16 or less produces one segment per row; a 32-column logical row produces two segments (one at `face_index = 0` with 16 source columns, one at `face_index = 1` with the remaining 16 source columns); an `n * 16`-column row produces `n` segments. `seg_columns` is bounded by 16 — never by 32 — so the segment count per row is `ceil(columns / 16)`, not `ceil(columns / 32)`.
  6. `tt::tt_metal::HostBuffer host_buffer(std::move(typed));` — same constructor used today at `copy.cpp:69`.
  7. `ttnn::Tensor host_tiled(std::move(host_buffer), plane.logical_shape(), plane.padded_shape(), plane.dtype(), tt::tt_metal::Layout::TILE);` — same constructor signature as today at `copy.cpp:112-115` but with `Layout::TILE` instead of `Layout::ROW_MAJOR`.
  8. `ttnn::copy_to_device(host_tiled, plane);` — unchanged from line 119.
  No `std::vector<std::byte> padded`, no `to_layout`, no intermediate row-major byte buffer, no `make_host_buffer` call.
- **Modify:** `src/ttnn/copy.cpp` — `download_plane` (`copy.cpp:122-146`). The new body, in order:
  1. `ttnn::Tensor host_tiled = ttnn::allocate_tensor_on_host(plane.tensor_spec(), &device);` — unchanged from line 129-130.
  2. `ttnn::copy_to_host(plane, host_tiled, /*blocking=*/true);` — unchanged from line 131. The result is in `Layout::TILE` and `host_tiled.padded_shape()[-1] == padded_columns` because `plane.tensor_spec()` carries `padded_shape()` and `Layout::TILE`.
  3. `const tt::tt_metal::HostBuffer buffer = tt::tt_metal::host_buffer::get_host_buffer(host_tiled.host_tensor());` — unchanged from line 134-136, but the local is the tile-major buffer.
  4. `const auto bytes = buffer.view_bytes();` and `const std::size_t num_tile_cols = static_cast<std::size_t>(host_tiled.padded_shape()[-1]) / 32;` — new read of `padded_columns` from the tile-major host tensor.
  5. The inverse segment walk:
     ```cpp
     for (std::size_t row = 0; row < rows; ++row) {
         std::size_t col = 0;
         while (col < columns) {
             const std::size_t seg_columns =
                     std::min<std::size_t>(16 - (col % 16), columns - col);
             const std::size_t tile_index =
                     (row / 32) * num_tile_cols + (col / 32);
             const std::size_t face_index =
                     ((row % 32) / 16) * 2 + ((col % 32) / 16);
             const std::size_t first_element_index =
                     tile_index * 1024
                     + face_index * 256
                     + (row % 16) * 16
                     + (col % 16);
             std::memcpy(
                     destination + (row * columns + col) * element_size,
                     bytes.data() + first_element_index * element_size,
                     seg_columns * element_size);
             col += seg_columns;
         }
     }
     ```
     The destination byte offset `(row * columns + col) * element_size` is the row-major offset that matches the §5.4 contract and is identical to the destination offset used today at `copy.cpp:142`. `seg_columns` is bounded by 16; each segment stays strictly inside one face.
  No `to_layout`.
- **Reuse (no edit):** `element_bytes` (`src/ttnn/copy.cpp:53-60`) — `element_size` is computed exactly once per `region_from_host` / `region_to_host` call and stays out of the inner loop. The byte-aligned invariant guarantees `element_size % sizeof(T) == 0` for every `T` in the typed dispatch.
- **Reuse (no edit):** `typed_buffer<T>` (`src/ttnn/copy.cpp:65-70`) — the reinterpretation helper stays in the translation unit as a documented helper; `upload_plane` no longer calls it. `make_host_buffer` (`copy.cpp:72-91`) is removed; its dtype switch and the `logic_error("TTNN native dtype has no host element type")` site are inlined into `upload_plane` verbatim.
- **Reuse (no edit):** `TtnnTensor::region_from_host` / `region_to_host` (`src/ttnn/device.cpp:221-235`) — the device-wide `api_mutex_` `std::lock_guard` and the `device_.mesh()` / `planes_.data()` handoff are unchanged. The locked section becomes shorter; the locking discipline does not.
- **Reuse (no edit):** `view_plane_count` (`copy.cpp:23-31`), `owner_plane_at` (`copy.cpp:37-49`), `view_rows` (`copy.cpp:148-151`), `view_columns` (`copy.cpp:153-156`). The per-plane loop in `region_from_host` and `region_to_host` keeps these helpers untouched.
- **Read:** `src/ttnn/copy.hpp:1-44` — confirm the contracts that do not change: "synchronous at return", "destination is pre-zeroed so unused tail bits read as zero", "no padding reaches the host buffer", and "values only; no host staging and no encoding conversion" for `copy_planes`. `region_from_host` / `region_to_host` retain their signatures and contracts.
- **Read:** `src/ttnn/device.cpp:189-209` — the `tt::tt_metal::TensorSpec` used by `ttnn::create_device_tensor` sets `tt::tt_metal::Tile()` (default 32×32, `transpose_tile = false`) on `PageConfig`, so `padded_shape()[-2]` and `padded_shape()[-1]` are guaranteed multiples of 32 for every supported leaf type. The `/32` and `/16` arithmetic in the offset formula is exact.
- **Read:** `test/ttnn/test_ttnn_conformance.cpp:142-220` `observe_plane_values` and `seed` — the conformance oracle reads through `plane.to_vector<T>()`, which TTNN populates by applying the inverse of the face-major formula internally. The test's pre-existing `ttnn::Tensor host_row_major(make_host_buffer(plane.dtype(), padded), …, ROW_MAJOR); to_layout(…, TILE); copy_to_device(…)` seed path stays as-is because that test seeds from a CPU reference and exercises the row-major→tile-major conversion explicitly; the production path does not.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:421-462` — `run_storage_and_transfer_conformance(devices.conformance(), iom::ttnn_supported_data_types())`, `run_storage_oracle_conformance(...)`, and `run_async_copy_conformance(...)` run unchanged against the post-fix `iom_ttnn` library; every leaf width in the TTNN supported table round-trips through the rewritten upload/download paths with byte-equivalent host buffers.

## Requirements

1. **One full-padded-shape typed buffer per plane.** The post-fix `upload_plane` allocates exactly one `std::vector<T>` of size `padded_rows * padded_columns` per plane (where `padded_rows` and `padded_columns` come from `plane.padded_shape()[-2]` / `[-1]`), typed via `tt::tt_metal::HostBuffer(std::move(typed))`, and immediately constructs the host `ttnn::Tensor` in `tt::tt_metal::Layout::TILE` with the plane's `logical_shape()` and `padded_shape()`. No `std::vector<std::byte> padded` allocation, no `to_layout(ROW_MAJOR → TILE)` call, no `make_host_buffer` call. The `logic_error("TTNN native dtype has no host element type")` site stays exactly where it is today (`copy.cpp:88-89`), inlined into the `upload_plane` dtype switch. Verified by `grep -n 'std::vector<std::byte>\|to_layout\|make_host_buffer' src/ttnn/copy.cpp` returning zero matches inside `upload_plane`.
2. **Direct tile-major fill using face-major offset, copied as per-face contiguous segments.** For each logical row `row = 0..rows-1`, partition the row into segments of up to 16 contiguous source columns, one segment per 16-column face. For each segment starting at column `col_start` with length `seg_columns = min(16 - (col_start % 16), columns - col_start)`, the rewrite computes the destination element index of the segment's first element by the TT-Metal documented `convert_to_tile` formula in Outcome §2 (`first_element_index = tile_index * 1024 + face_index * 256 + (row % 16) * 16 + (col_start % 16)`, where `tile_index = (row / 32) * num_tile_cols + (col_start / 32)` and `face_index = ((row % 32) / 16) * 2 + ((col_start % 32) / 16)`) and issues exactly one `std::memcpy` of `seg_columns * element_size` source bytes into the contiguous destination range starting at `first_element_index * element_size`. The segment is bounded by 16 source columns, so it stays strictly inside one face and the destination indices `first_element_index + 1, ..., first_element_index + seg_columns - 1` are contiguous without re-evaluating `tile_index` or `face_index` in the inner step. The segment count per row is `ceil(columns / 16)`: a 32-column row produces two segments (one at `face_index = 0`, one at `face_index = 1`); an `n * 16`-column row produces `n` segments. The typed vector's zero-initialized elements cover the padding rows and the padding columns within each row; the per-segment copies overwrite only the logical element positions and never touch padded bytes. Verified by inspection of the rewritten `upload_plane` body and by the conformance round-trip tests passing on hardware.
3. **Download skips `to_layout` and inverse-maps tile-major offsets to row-major.** `download_plane` allocates the host tensor through `allocate_tensor_on_host(plane.tensor_spec(), &device)`, calls blocking `ttnn::copy_to_host`, calls `host_buffer::get_host_buffer` on the tile-major result, and writes `seg_columns * element_size` destination bytes per logical-row segment from the tile-major source at the byte offset computed by the same `first_element_index * element_size` formula as the upload into the row-major destination offset `(row * columns + col_start) * element_size`. No `to_layout(TILE → ROW_MAJOR)` call. Verified by `grep -n 'to_layout' src/ttnn/copy.cpp` returning zero matches in `download_plane` (the seed code in `test/ttnn/test_ttnn_conformance.cpp:212-218` is unchanged and is the only `to_layout` left in the file).
4. **Padded-tail bytes stay zero.** The typed `std::vector<T>` constructor zero-initializes every element; the per-segment copies touch only logical `(row, col_start..col_start+seg_columns-1)` positions where `row < rows` and `col_start < columns`. Padding rows beyond `rows` and padding columns within each row past `columns` stay zero, matching the pre-fix `padded` vector's zero-initialization (`copy.cpp:103-105`). The conformance oracle at `test/ttnn/test_ttnn_conformance.cpp:142-220` continues to observe zero padding inside tile-major storage because the rewritten upload path writes zero to every padded element once (in the typed-vector constructor) and never overwrites it.
5. **`region_to_host` row-major zero-fill preserved.** `std::fill(destination.begin(), destination.end(), std::byte{0})` at `src/ttnn/copy.cpp:185` stays verbatim. The per-plane download path writes only `element_size` bytes per logical element into the row-major destination; any bytes the source does not write into the destination remain zero from the pre-fill. The conformance case `"TTNN conformance: storage oracle identifies perturbed transfer map"` (`test/ttnn/test_ttnn_conformance.cpp:425-447`) continues to pass: every logical byte the oracle compares is set by the rewrite, every uninitialized byte in the destination buffer is still zero.
6. **Synchronous-at-return and error behavior preserved.** `region_from_host` calls `device.mesh_command_queue(0).finish()` at `copy.cpp:176` after the per-plane loop; `region_to_host` performs no extra sync beyond the blocking `ttnn::copy_to_host` inside `download_plane` and the synchronous-on-return contract holds. Exception categories are unchanged: `std::logic_error` from `element_bytes` / the inlined dtype switch for unsupported byte widths or dtype mismatches; any TTNN runtime exception from `allocate_tensor_on_host`, `copy_to_host`, `copy_to_device`, or `to_layout` (in the seed path) propagates as today. No new exception sites, no swallowed exceptions.
7. **Device-wide mutex discipline unchanged.** `TtnnTensor::region_from_host` / `region_to_host` (`src/ttnn/device.cpp:221-235`) hold `api_mutex_` for the entire call into `iom::ttnn_detail::region_from_host` / `region_to_host`; `TtnnQueue::run` (`src/ttnn/device.cpp:388-424`) holds `api_mutex_` across `ttnn_detail::copy_planes` and `device_->mesh().mesh_command_queue(0).finish()`. The rewrite shortens the locked critical section proportionally to the per-plane work saved but does not drop the mutex, narrow it, or relax it. `copy_planes` is untouched (no host buffer traffic, no layout conversion).
8. **Typed dispatch and `DataType` set preserved.** `upload_plane` and `download_plane` accept every dtype `make_host_buffer` accepts today (`BFLOAT16`, `FLOAT32`, `UINT32`, `INT32`, `UINT16`, `UINT8`); the `logic_error("TTNN native dtype has no host element type")` site and the `logic_error("TTNN supported type is not byte-aligned")` site at `copy.cpp:56-57` are preserved. `iom::ttnn_supported_data_types()` and `src/ttnn/device.cpp:37-72` are untouched.
9. **Logical/native plane semantics preserved.** `region_from_host` writes `view_plane_count` planes in view-coordinate order through `owner_plane_at` (`copy.cpp:37-49`); `region_to_host` reads the same plane order. Each plane receives a host buffer whose logical shape equals `plane.logical_shape()` and whose padded shape equals `plane.padded_shape()` — identical to the pre-fix `ttnn::Tensor` construction at `copy.cpp:112-115`. TTNN runtime validates the `ttnn::copy_to_device` / `copy_to_host` argument against `plane` exactly as today.

## Non-goals

- Skipping the `device.mesh_command_queue(0).finish()` calls at `src/ttnn/copy.cpp:176` and `src/ttnn/device.cpp:415`. The `finish()` is what makes the host transfers synchronous-at-return; PF-004 owns the queue-side finish question and is a separate order.
- Changing `TtnnQueue::run` (`src/ttnn/device.cpp:388-424`), the `Task` record, `validate_copy` (`src/ttnn/device.cpp:345-358`), `publish_staged`, the staged worker-queue commit-before-complete ordering, or the asynchronous `copy_planes` path. The rewrite touches only the host-transfer paths inside `src/ttnn/copy.cpp`.
- Changing the CUDA/ROCm `synchronous_transfer` paths (`src/cuda/copy.cu:354-405`, `src/rocm/copy.hip:311-325`). PF-003 owns CUDA/ROCm full-size staging and PF-002 owns the per-plane GPU launch shape; both are separate orders.
- Per-bit sub-byte semantics or any change to `read_bits` / `write_bits` / `copy_value` on the GPU backends (PF-006, out of scope).
- Changing the supported-`DataType` table (`src/ttnn/device.cpp:37-72`) or `iom::ttnn_supported_data_types()`. The dispatch is the right primitive; only its caller is fixed.
- Replacing the typed `std::vector<T>` with `ttnn::owned_buffer` / `ttnn::borrowed_buffer` / `tt::tt_metal::distributed::MeshBuffer` or any other TTNN-side storage type. The existing `HostBuffer` API already accepts the typed vector and is what `allocate_tensor_on_host` returns on the download path; no new TTNN ownership is introduced.
- Reorganizing headers, renaming `upload_plane` / `download_plane`, moving them into shared translation units, or refactoring the anonymous namespace.
- Adding a benchmark dependency, microbenchmark infrastructure, CMake test target, TU-private accessor, timing hook, or allocation-counter seam. The post-fix verification runs only the existing public conformance round-trip on Tenstorrent hardware through `.agents/skills/remote-development`.
- Touching the test seed code at `test/ttnn/test_ttnn_conformance.cpp:142-220`. That path explicitly exercises the row-major→tile-major conversion because the test seeds from a CPU reference buffer; it is not the production path.

## Acceptance criteria

- [ ] `src/ttnn/copy.cpp`'s `upload_plane` (`copy.cpp:95-120`) builds one typed `std::vector<T>` per plane (no `std::vector<std::byte> padded`, no `to_layout` call, no `make_host_buffer` call) and constructs the host `ttnn::Tensor` directly in `tt::tt_metal::Layout::TILE` from the typed buffer. The per-row copy uses one `std::memcpy` per segment of up to 16 contiguous source columns inside one face, with the destination element index computed by the face-major formula in Outcome §2. `seg_columns = min(16 - (col % 16), columns - col)`; a 32-column row produces exactly two segments (one at `face_index = 0`, one at `face_index = 1`). Verified by `grep -n 'std::vector<std::byte>\|to_layout\|make_host_buffer' src/ttnn/copy.cpp` returning zero matches inside `upload_plane`.
- [ ] `src/ttnn/copy.cpp`'s `download_plane` (`copy.cpp:122-146`) drops the `to_layout(TILE → ROW_MAJOR)` line and inverse-maps each per-row segment from the tile-major `host_buffer::get_host_buffer` bytes to the row-major destination using the same face-major `first_element_index` formula and the same `seg_columns = min(16 - (col % 16), columns - col)` bound. Verified by `grep -n 'to_layout' src/ttnn/copy.cpp` returning zero matches in `download_plane` (the only remaining matches are in `test/ttnn/test_ttnn_conformance.cpp` seed code, untouched).
- [ ] The pre-existing public conformance round-trip at `test/ttnn/test_ttnn_conformance.cpp:421-462` (`run_storage_and_transfer_conformance`, `run_storage_oracle_conformance`, `run_async_copy_conformance`, `run_copy_error_conformance`) passes unchanged on TT hardware after the fix. Every leaf width in `iom::ttnn_supported_data_types()` round-trips over every shape in `copy_owner_shapes()` / `transfer_owner_shapes()` (including the `{17, 33}`, `{2, 3, 16, 16}`, `{2, 3, 4, 17, 33}`, `{2, 2, 2, 3, 17, 33}` matrices) and every transformed view from `view_cases_for`; the bytes observed post-write and post-read match the CPU reference bit-for-bit. The pre-existing assertion count baseline is preserved (no existing assertion, expected byte, or expected exception is changed).
- [ ] The pre-existing public perturbation oracle at `test/ttnn/test_ttnn_conformance.cpp:425-447` (`"TTNN conformance: storage oracle identifies perturbed transfer map"`) and at `test/ttnn/test_ttnn_conformance.cpp:445-447` (`run_storage_oracle_conformance` with the storage oracle) pass unchanged on TT hardware. These cases exercise the byte-level comparison the rewrite's offset formula must satisfy: every logical byte the oracle reads through `plane.to_vector<T>()` matches the encoded reference, and every padded byte the oracle reads matches zero.
- [ ] On Tenstorrent hardware, `iom::ttnn_detail::region_from_host` and `region_to_host` execute under the same `api_mutex_` discipline as today. The conformance case `"TTNN conformance: copy validation fails before writes and sequences"` (`test/ttnn/test_ttnn_conformance.cpp:457-462`) passes unchanged: a foreign-device view throws `std::invalid_argument` before any TTNN runtime call, a metadata-mismatched view throws `std::invalid_argument` before any write.
- [ ] The `device.mesh_command_queue(0).finish()` call at `src/ttnn/copy.cpp:176` and the corresponding call at `src/ttnn/device.cpp:415` are present and unchanged. The host transfers remain synchronous at return; the conformance case `"TTNN conformance: deferred queue lifetime and stability"` (the existing case name in `test/ttnn/test_ttnn_conformance.cpp` for queue-destruction-without-cancel behavior) continues to pass because `finish()` is still issued inside the `api_mutex_`-guarded critical section.
- [ ] No additional observable behavior change: the `iom_ttnn` exported symbol set includes exactly the TTNN public symbols declared by `include/iom/ttnn/device.hpp` (`make_ttnn_device`, `ttnn_supported_data_types`, the `iom::TtnnDevice` virtual-table implementations, and the C++ ABI helpers) plus the existing `iom::ttnn_detail::region_from_host`, `iom::ttnn_detail::region_to_host`, `iom::ttnn_detail::copy_planes` entry points declared in `src/ttnn/copy.hpp`. Verified by `nm -D --defined-only build/ttnn-pf005/libiom_ttnn.so | grep -E "make_ttnn_device|ttnn_supported_data_types|_ZN4iom12ttnn_support|TtnnDevice"` matching the same set produced by a fresh build with the pre-fix `src/ttnn/copy.cpp` checked out via `git checkout HEAD~1 -- src/ttnn/copy.cpp` in a separate task directory; the two `nm -D` outputs are byte-equivalent. Every file outside `src/ttnn/copy.cpp` is byte-identical except for comments. The unchanged set is `include/iom/ttnn/device.hpp` (the only TTNN public header — declares `make_ttnn_device` and `ttnn_supported_data_types`), `include/iom/tensor.hpp`, `include/iom/iom.hpp`, `src/ttnn/copy.hpp`, `src/ttnn/device.cpp`, and `test/ttnn/test_ttnn_conformance.cpp`. The only changed source file is `src/ttnn/copy.cpp`.

## Verification

Follow `.agents/skills/remote-development` (local workspace authoritative; unique remote task directory per backend). The fix's mechanism (one allocation per plane, tile-major typed buffer, no `to_layout` pass, face-major offset formula with per-tile contiguous segments) and the regression cases require real Tenstorrent hardware; CPU-only builds do not exercise the TTNN host-transfer path. The single verification path is the existing public conformance round-trip at `test/ttnn/test_ttnn_conformance.cpp:421-462` — no TU-private accessor, timing hook, or allocation-counter seam is added.

```bash
# Step 1: build the pre-fix control in a dedicated task directory on the remote host.
# src/ttnn/copy.cpp is a tracked source file, so checking out the parent commit's
# copy with `git checkout HEAD~1 -- src/ttnn/copy.cpp` is valid. Build artifacts are
# never read from `git show`; every `nm -D` invocation below targets a binary built
# inside its own task directory.
.agents/skills/remote-development/scripts/remote-sync tt-host pf005-ttnn-host-transfer-baseline
.agents/skills/remote-exec tt-host pf005-ttnn-host-transfer-baseline \
  'git checkout HEAD~1 -- src/ttnn/copy.cpp && cmake -S . -B build/ttnn-baseline -DBUILD_TESTING=OFF -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build/ttnn-baseline --target iom_ttnn -j && git checkout HEAD -- src/ttnn/copy.cpp && nm -D --defined-only build/ttnn-baseline/libiom_ttnn.so | sort > /tmp/symbols.baseline.txt && wc -l /tmp/symbols.baseline.txt'
.agents/skills/remote-clean tt-host pf005-ttnn-host-transfer-baseline

# Step 2: build the post-fix tree in the implementer's task directory.
# src/ttnn/copy.cpp is the post-fix content; every other file is HEAD.
.agents/skills/remote-development/scripts/remote-sync tt-host pf005-ttnn-host-transfer
.agents/skills/remote-exec tt-host pf005-ttnn-host-transfer \
  'cmake -S . -B build/ttnn-pf005 -DBUILD_TESTING=ON -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build/ttnn-pf005 --target iom_ttnn_conformance_tests iom_ttnn_smoke_tests'
.agents/skills/remote-exec tt-host pf005-ttnn-host-transfer \
  'ctest --test-dir build/ttnn-pf005 --output-on-failure -R "^iom_ttnn_(conformance|smoke)_tests$"'
.agents/skills/remote-exec tt-host pf005-ttnn-host-transfer \
  'grep -nE "std::vector<std::byte>|to_layout|make_host_buffer" src/ttnn/copy.cpp'
# ABI check: every public TTNN symbol declared in include/iom/ttnn/device.hpp
# (make_ttnn_device, ttnn_supported_data_types) must still be exported by the
# post-fix libiom_ttnn.so. The pre-fix baseline built in step 1 and the post-fix
# build in step 2 export the same set of defined symbols; the diff between
# /tmp/symbols.baseline.txt and the post-fix `nm -D` output is empty.
.agents/skills/remote-exec tt-host pf005-ttnn-host-transfer \
  'nm -D --defined-only build/ttnn-pf005/libiom_ttnn.so | sort > /tmp/symbols.postfix.txt && diff -u /tmp/symbols.baseline.txt /tmp/symbols.postfix.txt'
# Public header byte-identity: confirm no public header changed during this rewrite.
# These files are tracked source files (not build artifacts), so `git diff` against
# HEAD is valid and reproducible on any worktree.
.agents/skills/remote-exec tt-host pf005-ttnn-host-transfer \
  'git diff --stat HEAD -- include/iom/ttnn/device.hpp include/iom/tensor.hpp include/iom/iom.hpp src/ttnn/copy.hpp src/ttnn/device.cpp test/ttnn/test_ttnn_conformance.cpp'
.agents/skills/remote-clean tt-host pf005-ttnn-host-transfer
```

Expected observations on Tenstorrent hardware:

- `grep -nE "std::vector<std::byte>|to_layout|make_host_buffer" src/ttnn/copy.cpp` returns zero matches inside `upload_plane` and `download_plane`. The pre-fix control has the `std::vector<std::byte> padded(…, std::byte{0})` allocation at line 103, the `make_host_buffer` call at line 113, and the `to_layout` call at line 117.
- `ctest --test-dir build/ttnn-pf005 --output-on-failure -R "^iom_ttnn_(conformance|smoke)_tests$"` passes unchanged. The conformance cases `run_storage_and_transfer_conformance`, `run_storage_oracle_conformance`, `run_async_copy_conformance`, `run_copy_error_conformance`, `copy_validation_fails_before_writes_and_sequences`, and `storage_oracle_identifies_perturbed_transfer_map` all complete green with byte-equivalent host buffers; the conformance assertion totals match the pre-fix baseline.
- `diff -u /tmp/symbols.baseline.txt /tmp/symbols.postfix.txt` is empty. The pre-fix baseline is the `nm -D --defined-only` output of the binary built in step 1 with `git checkout HEAD~1 -- src/ttnn/copy.cpp` (tracked source file, valid `git` operation) and the post-fix output is the `nm -D --defined-only` of the binary built in step 2 with the post-fix `src/ttnn/copy.cpp`. Both binaries export the same set of defined symbols; no public ABI delta. The grep for `make_ttnn_device|ttnn_supported_data_types|_ZN4iom12ttnn_support|TtnnDevice` in the post-fix output returns non-empty (those symbols are present), and the diff confirms nothing extra leaked into the post-fix library because every internal helper in `src/ttnn/copy.cpp` lives in the anonymous namespace and the existing public namespace symbols `iom::ttnn_detail::region_from_host`, `iom::ttnn_detail::region_to_host`, `iom::ttnn_detail::copy_planes` (declared in `src/ttnn/copy.hpp`, all `inline`-free TU-private declarations) keep their existing mangled names. Build artifacts (`build/ttnn-baseline/libiom_ttnn.so`, `build/ttnn-pf005/libiom_ttnn.so`) are never read from `git show HEAD:build/...` because the repository carries no built binaries.

No formatters, linters, project-wide builds, or sibling-backend regressions are validated here; the task is TTNN-only and exercised entirely on Tenstorrent hardware through `remote-development`. PF-002/PF-003/PF-004/PF-006 carry their own validation on their own orders.
