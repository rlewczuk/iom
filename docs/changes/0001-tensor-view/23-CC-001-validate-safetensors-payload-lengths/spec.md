# Validate SafeTensors payload length and range disjointness in the entry loop

**Order:** 23
**Priority:** P0 — high-severity correctness and memory-safety defect at the untrusted weight-ingestion boundary
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `CC-001`
**Review severity:** high
**Review verification:** verified, confidence 93

## Outcome

`iom::SafeTensorsFile` rejects every header entry whose declared `shape` and `dtype` do not produce a payload byte count equal to its `data_offsets` span, and rejects entries whose `[begin, end)` ranges are not strictly ordered and disjoint across the whole file. Every rejection throws `std::runtime_error` naming the offending tensor and the specific mismatch (size mismatch, overlap, or out-of-order range). A well-formed file whose per-entry payload byte count matches `ceil(prod(shape) * leaf_bits(dtype) / 8)` loads identically and every existing test (with the one modified `dtype-map` case below) continues to pass.

## Current failure

The invariant is that the weight flow (`MappedFile` → `SafeTensorsFile` → `SafeTensorView` → caller tensors → queued ops) validates shape, dtype, and buffer sizes before any work begins (`AGENTS.md`). The SafeTensors layer is the untrusted-input boundary.

The entry loop at `src/safetensors.cpp:117-131` currently only checks `begin <= end <= data_size` (`src/safetensors.cpp:125`). It never checks that `end - begin == ceil(prod(shape) * leaf_bits(dtype) / 8)` and never enforces disjoint, ordered ranges across entries. A consumer that sizes reads from `view.shape()` walks `prod(shape)` elements, which can reach tens of millions of bytes past the actual allocation.

Concrete consequences (already realized in the current tree, reproduced against built `libiom.a` per the review):

1. An entry declaring `F16 [4096, 4096]` (logical `prod = 16777216` elements, `leaf_bits = 16`, `expected = 33554432` bytes) with `data_offsets = [0, 4]` is accepted; `SafeTensorView::nbytes()` reports `4` while `SafeTensorView::shape()` reports `[4096, 4096]`. Sampling `logical_nbytes` (33554432) bytes from `raw<T>()` segfaulted the host process at the end of the mmap (exit 135 per the review).
2. Two overlapping entries (e.g. ranges `[0, 100]` and `[50, 150]`) are accepted. A later `SafeTensorsDir` aggregation silently aliases one tensor onto another.
3. Two entries presented in reverse order (range `[200, 400]` before `[0, 100]`) are accepted. There is no contract that ranges are ordered, but downstream consumers have no way to detect inconsistency.

The established exception category for malformed/unsupported input at this boundary is `std::runtime_error`: see `src/safetensors.cpp:44` ("unsupported safetensors dtype"), `:49` (invalid shape dimension), `:53` (safetensors dimension too large), `:81` (header too large), `:86` (header exceeds file size), `:97` (invalid safetensors header), `:105` (invalid tensor entry), `:114` (invalid tensor fields), and `:126` (data offset out of range). The new checks use the same category and name the offending tensor in the message so a malformed checkpoint is diagnosed at ingestion.

## Scope

- Add three rejection checks inside the entry loop of `SafeTensorsFile::SafeTensorsFile` (`src/safetensors.cpp:100-132`):
  1. payload size match: `end - begin == ceil(prod(shape) * leaf_bits(dtype) / 8)` with checked arithmetic on both the element product and the bit-to-byte round-up;
  2. ordered ranges: `begin >= prev_end` for every entry after the first;
  3. non-overlapping ranges: enforced jointly with the ordered check, since `begin >= prev_end` combined with the existing `begin <= end` and the new size-match guarantee rules out overlap.

- Add three file-local checked-arithmetic helpers in the anonymous namespace at `src/safetensors.cpp:13-58`: `safetensors_checked_add` (mirrors the body of `iom::detail::checked_add` at `src/iom.cpp:23-28` and throws `std::runtime_error` on overflow to match the file-local exception category), `safetensors_checked_mul` (mirrors `iom::detail::checked_mul` at `src/iom.cpp:30-35`), and `safetensors_bits_to_bytes` (mirrors `iom::detail::bits_to_bytes` at `src/iom.cpp:37-39` and uses `safetensors_checked_add(bits, 7, what) / 8` to perform the standard ceil-bits-to-bytes round-up). Use these helpers instead of exposing the `src/iom.cpp` anonymous namespace through a new shared header. The three checked-arithmetic implementations are textually identical to the `src/iom.cpp` versions, so a future consolidation can fold them back without behavior change.

- Add one file-local helper `safetensors_expected_payload_bytes(DataType dtype, const std::vector<std::size_t>& shape, const std::string& name)` that returns `ceil(prod(shape) * leaf_bits(dtype) / 8)`. It calls `iom::detail::leaf_bits(dtype)` for the bit width, accumulates the element product through `safetensors_checked_mul`, multiplies the element count by the leaf bit width through `safetensors_checked_mul`, and converts bits to bytes through `safetensors_bits_to_bytes`. On any overflow it throws `std::runtime_error("safetensors tensor payload size overflows: " + name)` naming the tensor.

- `SafeTensorsDir` continues to delegate per-file construction to `SafeTensorsFile`, so the new checks automatically reject cross-shard inconsistency at the per-shard level. Cross-shard duplicate keys are addressed by task 24; cross-shard disjoint-but-overlapping ranges are impossible because each shard is a file with disjoint internal regions and per-file ordering.

- Public `SafeTensorView` shape, dtype, and nbytes accessors are unchanged. Well-formed payloads round-trip with identical observable behavior to today: same `dtype()`, same `shape()`, same `nbytes()` (which now equals `expected_nbytes`), same `raw<T>()` pointer.

- Throw category is the established `std::runtime_error`. No new exception type is introduced; the existing `std::out_of_range` from `operator[]` for missing keys is unchanged. The file-local checked-arithmetic helpers throw `std::runtime_error` (not `std::overflow_error`) so a single exception category covers every malformed-input path in this translation unit.

- The leaf-bit width helper `iom::detail::leaf_bits(DataType)` at `include/iom/tensor.hpp:123` and its definition at `src/iom.cpp:149-` (used by `TensorSpec::logical_nbytes()` at `src/iom.cpp:130-135`) is the single source of truth for the bit widths; the SafeTensors entry loop reuses it instead of duplicating the table.

- The size-check rejection fires before any `SafeTensorView` is constructed and before `keys_.push_back`/`tensors_.emplace`, so the half-constructed `SafeTensorsFile` is never observable.

- The existing `TEST_CASE("SafeTensors dtype maps every accepted string to its DataType")` at `test/test_safetensors.cpp:114-167` is in scope and must be modified: its per-entry payload length and `tensor.nbytes()` check currently hard-code 2 bytes for every dtype, which the new entry-loop size check rejects for every dtype whose `leaf_bits(dtype) * 2` does not round to 2 bytes. The case is modified to derive the payload byte count from the same integer ceil-to-bytes expression the production code uses, namely `(iom::detail::leaf_bits(cases[i].second) * 2 + 7) / 8`, and use that single `payload_bytes` value for the fixture payload size, the `tensor.nbytes()` check, the expected buffer passed to `memcmp`, and the entry's `data_offsets` — no separate or truncated calculation is permitted. The case's intent (every accepted safetensors dtype string maps to its `DataType`, the `shape` is reported correctly, and the raw payload round-trips) is preserved.

## Implementation references

- **Modify:** `src/safetensors.cpp` — anonymous namespace at lines 13-58: add four file-local helpers after `json_size` (line 56) and before the closing `}  // namespace` (line 58). The four helpers use `std::size_t` and `std::string` from existing includes; the `safetensors_checked_add` body uses `std::numeric_limits<std::size_t>` and therefore requires `<limits>`, which `src/safetensors.cpp:6` already includes.
  - `safetensors_checked_mul(std::size_t lhs, std::size_t rhs, const std::string& what) -> std::size_t`: identical body to `iom::detail::checked_mul` (`src/iom.cpp:30-35`) except the throw type is `std::runtime_error(what)` so the safetensors boundary keeps a single exception category. The string reference accepts both fixed diagnostics and the tensor-named overflow message.
  - `safetensors_checked_add(std::size_t lhs, std::size_t rhs, const std::string& what) -> std::size_t`: identical body to `iom::detail::checked_add` (`src/iom.cpp:23-28`) except the throw type is `std::runtime_error(what)`. The body compares `rhs > std::numeric_limits<std::size_t>::max() - lhs`.
  - `safetensors_bits_to_bytes(std::size_t bits, const std::string& what) -> std::size_t`: identical body to `iom::detail::bits_to_bytes` (`src/iom.cpp:37-39`), computed as `safetensors_checked_add(bits, 7, what) / 8`. This computes the standard ceil-bits-to-bytes round-up `(bits + 7) / 8`; a `safetensors_checked_mul(bits, 7, what) / 8` formulation is rejected because it would multiply by 7 instead of adding 7.
  - `safetensors_expected_payload_bytes(DataType dtype, const std::vector<std::size_t>& shape, const std::string& name) -> std::size_t`: first constructs `const std::string overflow_message = "safetensors tensor payload size overflows: " + name`, then calls `iom::detail::leaf_bits(dtype)`, multiplies each `shape[i]` through `safetensors_checked_mul`, multiplies the element count by the leaf bit width through `safetensors_checked_mul`, and converts bits to bytes through `safetensors_bits_to_bytes`, passing `overflow_message` to every helper.

- **Modify:** `src/safetensors.cpp` — `SafeTensorsFile::SafeTensorsFile` entry loop (lines 100-132). After the existing `begin > end || end > data_size` check (line 125) and before `keys_.push_back(name); tensors_.emplace(...)` (lines 129-131), insert three new checks that all throw `std::runtime_error` naming the tensor:
  1. `const size_t expected_nbytes = safetensors_expected_payload_bytes(parse_dtype(dtype_it->get<std::string>()), shape, name);`. Move the `parse_dtype` call to a local before the size check so its result is reused by the size check and the `SafeTensorView` constructor; the existing per-entry call is preserved by binding its return value once. If `(end - begin) != expected_nbytes`, throw `std::runtime_error("safetensors tensor payload size mismatch: " + name + " (expected " + std::to_string(expected_nbytes) + " bytes, got " + std::to_string(end - begin) + " bytes)")` with the actual decimal byte counts.
  2. Track `prev_end` (initially `0`); for every entry after the first, if `begin < prev_end`, throw `std::runtime_error("safetensors tensor range out of order or overlapping: " + name)`. This single check enforces both ordering and non-overlap against already-validated entries. The first entry cannot trigger this check.
  3. After a successful insertion, set `prev_end = end`.

  The new rejections must occur before `keys_.push_back` and `tensors_.emplace` so the partial state is discarded.

- **Read:** `include/iom/tensor.hpp:123` — `iom::detail::leaf_bits(DataType)` is declared here. Use the public detail function rather than redefining the table in `src/safetensors.cpp`.

- **Read:** `src/iom.cpp:23-39` — the reference bodies for the file-local `safetensors_checked_add`, `safetensors_checked_mul`, and `safetensors_bits_to_bytes`. The bodies are copied verbatim except the throw type is `std::runtime_error` in the safetensors copies so the boundary keeps one exception category.

- **Read:** `src/safetensors.cpp:117-131` — preserve the existing field-shape parsing pattern (`shape.reserve(shape_it->size());` plus the per-dimension `json_size` loop) unchanged; the new size check reads `shape` and `dtype` after parsing.

- **Read:** `src/safetensors.cpp:151-167` — `SafeTensorsDir::SafeTensorsDir` constructs `SafeTensorsFile` per shard inside `std::make_unique<SafeTensorsFile>(path.string())`. The per-file constructor now performs the rejection, so the dir needs no additional check beyond what already runs.

- **Tests:** `test/test_safetensors.cpp` — add one new `TEST_CASE("SafeTensorsFile rejects mismatched, overlapping, and out-of-order entries")`. The existing `write_safetensors_file` helper at lines 68-89 always emits sequential non-overlapping `data_offsets` derived from `payload.size()`, so it cannot construct the overlapping `[4, 12]` and out-of-order `[8, 16]` then `[0, 8]` fixtures this case requires. The implementer adds a sibling helper, `write_raw_safetensors_file`, alongside the existing one:

  ```cpp
  struct RawTensorEntry {
      std::string name;
      std::string dtype;
      std::vector<std::size_t> shape;
      std::size_t begin;
      std::size_t end;
  };

  std::string write_raw_safetensors_file(
          const std::filesystem::path& dir,
          const std::string& filename,
          const std::vector<RawTensorEntry>& entries,
          const std::string& payload);
  ```

  The new helper writes the same 8-byte little-endian header length, JSON object, and payload structure as `write_safetensors_file`, but each entry emits its caller-supplied `begin` and `end` directly into the `"data_offsets":[begin,end]` field, and the function takes the payload blob as a single argument rather than concatenating per-entry payloads. The helper does no validation; the test relies on the production `SafeTensorsFile` constructor to reject malformed ranges. The helper lives next to the existing one and reuses the same `TempDir` and `ofstream` plumbing at lines 27-112.

  Each subcase constructs its own `TempDir`, calls `write_raw_safetensors_file` to produce a file with the subcase's exact `data_offsets`, then constructs `iom::SafeTensorsFile` from the returned path. The case must not silently skip. Cover:

  - **Truncated entry:** one entry `{"a", "F32", {4096, 4096}, 0, 4}` with payload `deterministic_payload(4, 0)` (4 bytes total). Logical size is `4096 * 4096 * 4 = 67108864` bytes; declared span is `[0, 4]`. Assert `CHECK_THROWS_AS((void)iom::SafeTensorsFile(path), std::runtime_error)` and assert the exception `what()` contains both `"payload size mismatch"` and the tensor name `"a"`.

  - **Oversized entry:** one entry `{"b", "F16", {2, 2}, 0, 10}` with payload `std::string(10, '\0')` (10 bytes total). Logical size is `2 * 2 * 2 = 8` bytes; declared span is `[0, 10]`. Both offsets pass the existing `end <= data_size` check (since `data_size == 10`). Assert `CHECK_THROWS_AS((void)iom::SafeTensorsFile(path), std::runtime_error)` and assert the exception `what()` contains both `"payload size mismatch"` and the tensor name `"b"`.

  - **Overlapping entries:** two entries `{"c", "F32", {2}, 0, 8}` and `{"d", "F32", {2}, 4, 12}` with one shared payload `std::string(12, '\0')` (12 bytes total, large enough that both declared spans `[0, 8]` and `[4, 12]` lie entirely inside the file's data section). The second entry's `begin < prev_end` triggers the order/overlap check after the first entry was accepted. Assert `CHECK_THROWS_AS((void)iom::SafeTensorsFile(path), std::runtime_error)` and assert the exception `what()` contains both `"out of order or overlapping"` and one of the tensor names (`"c"` or `"d"`).

  - **Out-of-order entries:** two entries `{"e", "F32", {2}, 8, 16}` and `{"f", "F32", {2}, 0, 8}` with one shared payload `std::string(16, '\0')` (16 bytes total, large enough that both declared spans `[8, 16]` and `[0, 8]` lie entirely inside the file's data section). Assert `CHECK_THROWS_AS((void)iom::SafeTensorsFile(path), std::runtime_error)` and assert the exception `what()` contains both `"out of order or overlapping"` and the second tensor name `"f"`.

  - **Payload-size overflow guard:** one entry `{"g", "I64", {1ULL << 62, 2}, 0, 8}` with payload `std::string(8, '\0')`. Both offsets are within file bounds; the rejection comes from `safetensors_checked_mul` inside `safetensors_expected_payload_bytes`. Assert `CHECK_THROWS_AS((void)iom::SafeTensorsFile(path), std::runtime_error)` and assert the exception `what()` contains both `"payload size"` and the tensor name `"g"`.

  Every subcase must be observable through `write_raw_safetensors_file` and `TempDir`; the new helper is the only way to reach the order/overlap paths because `write_safetensors_file` cannot construct them.

- **Read:** `test/test_safetensors.cpp:114-167` — the existing well-formed `dtype-map` case iterates 19 accepted dtype strings with shape `{2}`, but it currently hard-codes `deterministic_payload(2, i)` and `CHECK(tensor.nbytes() == 2)` regardless of `DataType`. Under the new entry-loop size check, every dtype except `BOOL/U8/I8` (1 byte each) and the 4-bit `F4_E2M1` ceil-to-1 and the 6-bit `F6_*` ceil-to-2 cases would still misalign (e.g. `F32 {2}` = 8 bytes, `F64 {2}` = 16 bytes, `F16`/`BF16`/`U16`/`I16`/`I32`/`U32`/`F8_E5M2`/`F8_E4M3FN`/`F8_E8M0`/`I64`/`U64`/`I32`/`U32` = 4/8 bytes per element). The implementer must modify this case so the fixture is well-formed under the new check: introduce a single per-entry expression that computes the expected payload byte count from each `DataType`, namely `const std::size_t payload_bytes = (iom::detail::leaf_bits(cases[i].second) * 2 + 7) / 8;` (the `+ 7` is mandatory so the 4-bit `F4_E2M1` ceil-to-1 and the 6-bit `F6_*` ceil-to-2 round up correctly), then size the entry payload with `deterministic_payload(payload_bytes, i)`, assert `CHECK(tensor.nbytes() == payload_bytes)`, compare the raw bytes with `CHECK(std::memcmp(tensor.raw<uint8_t>(), want.data(), payload_bytes) == 0)` (using the same `want` produced from `payload_bytes`), and set the entry's `data_offsets` to that same `payload_bytes` value. Use the single expression for every check — do not compute a separate truncated `leaf_bits(...) / 8 * 2` value, because that formulation truncates the 4-bit and 6-bit dtypes and the new entry-loop size check would reject the file. The two-element fixture stays. The other existing cases at `:186-213` (`SafeTensorsFile exposes tensors from a generated file`), `:215-232` (`SafeTensorsFile satisfies SafeTensorsStore interface`), and `:234-279` (`SafeTensorsDir exposes tensors from every file in a directory`) already use disjoint ordered ranges with shapes whose per-element byte counts match their dtype and stay unchanged.
## Requirements


- Inside the entry loop, after the existing `begin > end || end > data_size` check, compute `expected_nbytes = safetensors_expected_payload_bytes(dtype, shape, name)` where `dtype` comes from the existing `parse_dtype(dtype_it->get<std::string>())` call. If `safetensors_expected_payload_bytes` throws on overflow, the throw propagates and the loop aborts without inserting the entry. If `(end - begin) != expected_nbytes`, throw `std::runtime_error("safetensors tensor payload size mismatch: " + name + " (expected " + std::to_string(expected_nbytes) + " bytes, got " + std::to_string(end - begin) + " bytes)")` with the actual decimal byte counts.
- Maintain a running `prev_end` (initial value `0`) across the loop. For every entry after the first, if `begin < prev_end`, throw `std::runtime_error("safetensors tensor range out of order or overlapping: " + name)`. Update `prev_end = end` only on a successful entry.
- All three checks must run before `keys_.push_back` and `tensors_.emplace`. A rejection therefore leaves the partially populated `keys_` and `tensors_` discarded (the `SafeTensorsFile` constructor has not yet returned).
- The shape-parsing loop at `src/safetensors.cpp:117-121` and the `data_offsets` parsing at `:123-124` are unchanged. The new check sits strictly between them and the final emplace.
- The leaf-bit width is read from `iom::detail::leaf_bits(dtype)` (`include/iom/tensor.hpp:123`, `src/iom.cpp:149-`). Bind the result of `parse_dtype(dtype_it->get<std::string>())` to a local before the size check so the `SafeTensorView` constructor at line 130 reuses the same value without a second parse.
- Throw category is `std::runtime_error` only, matching every other rejection in `src/safetensors.cpp`. No new exception type is introduced. The file-local `safetensors_checked_add` and `safetensors_checked_mul` each throw `std::runtime_error` instead of `std::overflow_error` so the boundary keeps one exception category; the message text each carries is preserved verbatim from the helpers at `src/iom.cpp:23-35`.
- The size-match check must compare exact byte counts; no tolerance, no off-by-one, no `>=` instead of `==`. A tensor with 1 byte too many or too few is rejected.
- `SafeTensorsFile::operator[]` (`src/safetensors.cpp:135-141`) and `SafeTensorsDir::operator[]` (`src/safetensors.cpp:170-176`) are unchanged; their `std::out_of_range("safetensors tensor not found: " + name)` continues to be the only post-construction failure path.
- `SafeTensorsDir` needs no change beyond the per-file construction it already performs; cross-shard duplicates are addressed by task 24.

- The existing `TEST_CASE("SafeTensors dtype maps every accepted string to its DataType")` at `test/test_safetensors.cpp:114-167` must be modified so each fixture tensor's payload byte count equals `(iom::detail::leaf_bits(cases[i].second) * 2 + 7) / 8` — the same integer ceil-to-bytes expression the production code uses (`safetensors_bits_to_bytes`). The modification uses the established `iom::detail::leaf_bits` helper (the same helper the production code reads), and the single `payload_bytes` value computed once per entry is reused for the fixture payload size, the `tensor.nbytes()` check, the expected buffer passed to `memcmp`, and the entry's `data_offsets` — no separate or truncated calculation is permitted. The modification preserves the case's intent: every accepted safetensors dtype string is mapped to its `DataType`, the `shape` is reported correctly, and the raw payload bytes round-trip exactly. The other three existing test cases (`SafeTensorsFile exposes tensors from a generated file`, `SafeTensorsFile satisfies SafeTensorsStore interface`, `SafeTensorsDir exposes tensors from every file in a directory`) need no change because their declared shapes and per-element byte counts already match their dtypes.
- No allocator, mapping, or backend path is touched. The check is purely in the parsing boundary and runs only against the already-parsed header.

## Non-goals

- Restructuring the SafeTensors entry loop into a separate parser class or introducing a streaming/zero-copy header decoder. The fix is additive and confined to the existing `for` loop body.
- Adding a new exception type, new `std::error_code`, or new error category. `std::runtime_error` is the chosen invalid-input category; the existing `std::out_of_range` for missing keys is preserved. The file-local `safetensors_checked_add` and `safetensors_checked_mul` mirror `iom::detail::checked_add` and `iom::detail::checked_mul` but throw `std::runtime_error` rather than `std::overflow_error`; this is a textual divergence confined to this translation unit and is not promoted to a general convention.
- Validating the `data_offsets` pair against the file size when the offsets themselves are well-ordered but extend past `data_size`. The existing `end > data_size` check at `src/safetensors.cpp:125` already covers that case.
- Cross-shard duplicate-key detection; that is the explicit subject of task 24 (`24-CC-003-reject-duplicate-shard-keys`).
- Resolving stale line numbers, removing the obsolete field-shape checks, or refactoring `json_size`/`parse_dtype`. The helper functions are reused unchanged.
- Adding a new public API surface, exposing the expected payload bytes, or providing a `try_parse` alternative constructor.
- Building performance, allocator integration, or mapping-mode changes. The fix is a constant-time check per entry.
- Tightening the header-level validation by validating every tensor's dtype is on the safetensors-accepted list before the loop. The accepted dtype list check is already enforced by `parse_dtype` at `src/safetensors.cpp:23-45`.
- Exposing `iom::detail::checked_mul`, `iom::detail::checked_add`, or `iom::detail::bits_to_bytes` through a shared internal header. The four file-local helpers in `src/safetensors.cpp` are the chosen implementation; a future consolidation can fold them back into `src/iom.cpp` without behavior change.

## Acceptance criteria

- [ ] A `SafeTensorsFile` whose single entry declares `F32 {4096, 4096}` with `data_offsets = [0, 4]` and a 4-byte payload buffer throws `std::runtime_error` from the `SafeTensorsFile` constructor; the message contains `"payload size mismatch"` and the tensor name `"a"`; no `SafeTensorView` is constructed and the `SafeTensorsFile`'s `keys_`/`tensors_` remain empty.
- [ ] A `SafeTensorsFile` whose single entry declares `F16 {2, 2}` with `data_offsets = [0, 10]` and a 10-byte payload buffer throws `std::runtime_error` from the `SafeTensorsFile` constructor; the message contains `"payload size mismatch"` and the tensor name `"b"`; no `SafeTensorView` is constructed.
- [ ] A `SafeTensorsFile` with two entries having `data_offsets = [0, 8]` and `[4, 12]` (overlapping by 4 bytes), each declaring `F32 {2}` with 8-byte payloads, throws `std::runtime_error`; the message contains `"out of order or overlapping"` and one of the tensor names (`"c"` or `"d"`); no `SafeTensorView` is constructed.
- [ ] A `SafeTensorsFile` with two entries having `data_offsets = [8, 16]` and `[0, 8]` (out of order), each declaring `F32 {2}` with 8-byte payloads, throws `std::runtime_error`; the message contains `"out of order or overlapping"` and the second tensor name `"f"`.
- [ ] A `SafeTensorsFile` whose entry declares `I64 {1ULL << 62, 2}` with `data_offsets = [0, 8]` and an 8-byte payload buffer throws `std::runtime_error` from `safetensors_checked_mul` (called by `safetensors_expected_payload_bytes`) before the size comparison; the message contains `"payload size"` and the tensor name `"g"`.
- [ ] The existing `TEST_CASE("SafeTensors dtype maps every accepted string to its DataType")` at `test/test_safetensors.cpp:114-167` is modified per the **Read** bullet above: for each entry the per-iteration expression is `const std::size_t payload_bytes = (iom::detail::leaf_bits(cases[i].second) * 2 + 7) / 8;` (the same integer ceil-to-bytes expression used by `safetensors_bits_to_bytes`); the entry payload is sized via `deterministic_payload(payload_bytes, i)`; the expected `data_offsets` span equals `payload_bytes`; the `tensor.nbytes()` check is `CHECK(tensor.nbytes() == payload_bytes)`; the expected buffer passed to `memcmp` is the same `deterministic_payload(payload_bytes, i)`; and the `memcmp` length is `payload_bytes`. The 19 dtype strings and their expected `iom::DataType` enumerators stay unchanged. No second formula (in particular no `leaf_bits(dtype) / 8 * 2`) appears anywhere in the case. The other three existing cases (`SafeTensorsFile exposes tensors from a generated file` at `:186-213`, `SafeTensorsFile satisfies SafeTensorsStore interface` at `:215-232`, `SafeTensorsDir exposes tensors from every file in a directory` at `:234-279`) continue to pass without any modification.
- [ ] The new `TEST_CASE` reports one `CHECK_THROWS_AS(..., std::runtime_error)` and one message-content `CHECK` per subcase (truncated, oversized, overlapping, out-of-order, overflow), and every subcase is constructed through the new `write_raw_safetensors_file` helper. Both range-violation subcases use a single shared payload buffer that is at least as large as the larger of the two declared spans (12 bytes for overlapping, 16 bytes for out-of-order) so each declared `data_offsets` pair remains inside the file's data section — i.e., the malformed ranges pass the existing `end <= data_size` guard at `src/safetensors.cpp:125` and reach the production code only as evidence for the new `begin < prev_end` ordering check, not as evidence of a file-bound violation.
- [ ] The new helper `write_raw_safetensors_file` and its `RawTensorEntry` struct live in `test/test_safetensors.cpp` next to the existing `write_safetensors_file` (lines 18-112), share the same `TempDir` and `ofstream` plumbing, write the same 8-byte little-endian header length and JSON object format, and emit the caller's per-entry `begin` and `end` directly into `"data_offsets":[begin,end]`. The helper does no validation; the test relies on the production `SafeTensorsFile` constructor to reject malformed ranges.
- [ ] No new failure path reaches `SafeTensorView` construction; `nbytes()` and `shape()` continue to agree (when both are present) for every well-formed entry.

## Verification

This is a CPU-only common-boundary fix. No accelerator hardware is required. The verification uses the repository's local CPU build and the existing `iom_tests` target that already exercises `test/test_safetensors.cpp`.

1. Configure with `BUILD_TESTING=ON` and every accelerator disabled, matching the established CPU isolation pattern (`docs/changes/0001-tensor-view/05-cpu-storage-transfer-copy`):

   ```bash
   cmake -S . -B build/cc001 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF
   cmake --build build/cc001 -j --target iom_tests
   ```

2. Run the focused suite first; the new case must run and all subcases must pass:

   ```bash
   ./build/test/iom_tests --test-case="SafeTensorsFile rejects mismatched, overlapping, and out-of-order entries"
   ```

   Expected: five `CHECK_THROWS_AS(..., std::runtime_error)` and message-content `CHECK` passes, no skips, exit status 0.
3. Reproduce the reviewed failure once on the uncorrected tree (before applying this task's edit, or with it stashed): the case must report a thrown `std::runtime_error` from the corrected path that the uncorrected code would not have thrown. The truncated `F32 {4096, 4096}` with `data_offsets = [0, 4]` is the documented reproduction; the oversized `F16 {2, 2}` with `data_offsets = [0, 10]`; the overlapping subcase using `write_raw_safetensors_file` with offsets `[0, 8]` and `[4, 12]` over a 12-byte payload buffer; the out-of-order subcase using `write_raw_safetensors_file` with offsets `[8, 16]` then `[0, 8]` over a 16-byte payload buffer; and the `I64 {1ULL << 62, 2}` overflow subcase each demonstrate a distinct rejection path. After the fix, every subcase passes.

4. Run the full `iom_tests` executable to confirm the modified existing `dtype-map` case, the three untouched existing safetensors cases, and the new case all pass together:

   ```bash
   ctest --test-dir build/cc001 --output-on-failure -R "^iom_tests$"
   ```

   Expected: all existing and new `test/test_safetensors.cpp` cases green, no skips, no new failures elsewhere in `iom_tests`.

5. Confirm the no-untargeted-side-effect invariant by running the broader CPU test executables in this build configuration (matching the patterns already used at `docs/changes/0001-tensor-view/05-cpu-storage-transfer-copy/spec.md`):

   ```bash
   ctest --test-dir build/cc001 --output-on-failure
   ```

   Expected: unchanged-green on every other test target; no other source file is modified by this fix.

The remote-development procedure is not required because the fix is in a translation unit that builds and runs purely on the CPU host, and the established test infrastructure for the SafeTensors layer (`iom_tests`) covers the change. GPU evidence is not applicable to this finding.
