# Unify TTNN supported-type publication and native-dtype conversion on a single constexpr mapping

**Order:** 32
**Priority:** P1 — bounded defect remediation that closes a hand-maintained capability list duplicated across the same TU without changing capability membership, rejection category, or ordering.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-004`
**Review severity:** low
**Review verification:** verified, confidence 88

## Outcome

`src/ttnn/device.cpp` defines exactly one hand-maintained `iom::DataType → tt::tt_metal::DataType` mapping for the TTNN backend. Both the public `ttnn_supported_data_types()` span and the internal `native_dtype` lookup are mechanically derived from that single mapping at compile time, so editing a pair (or adding/removing one) automatically changes the published capability set. The duplicate `checked_plane_count` call on the public `create_tensor` path is removed; the `TtnnTensor` constructor keeps its inline `checked_plane_count` call as the last line of defense.

## Current failure

`src/ttnn/device.cpp:37-54` declares the `kSupportedDataTypes` array of nine `iom::DataType` values and a separate `is_supported(DataType)` linear scan. `src/ttnn/device.cpp:57-72` then declares `native_dtype(DataType)`, an independent nine-arm `switch` enumerating the same nine types and mapping each to a `tt::tt_metal::DataType`. The two lists are typed by hand against each other: adding a type to one but not the other passes the gate (so `create_tensor` accepts the type), then fails mid-construction in `native_dtype`'s `default:` arm with `std::invalid_argument` instead of the documented `std::runtime_error` capability rejection at the documented pre-allocation point. `ttnn_supported_data_types()` and `native_dtype` are also forced to be maintained in lockstep by hand, which is exactly the drift mechanism the finding flags.

`checked_plane_count` is invoked on the public `create_tensor` path at `src/ttnn/device.cpp:453` and again inside the `TtnnTensor` constructor at `src/ttnn/device.cpp:188`. `TtnnTensor` lives in an anonymous namespace and has exactly one construction site (`TtnnDevice::create_tensor` at `src/ttnn/device.cpp:455`), so the public-side call protects no second entry path — it only redoes the same overflow/`uint32_t` ceiling check the constructor already performs.

The invariant the finding preserves: `create_tensor` accepts exactly the published `iom::DataType` set with `QuantizationFormat::NONE` and rejects every other leaf type with `std::runtime_error` before any native object or allocation exists.

## Scope

- Backend scope: `ttnn`. CPU/CUDA/ROCm are out of scope.
- Replaces the two parallel hand-maintained TTNN declarations with one constexpr `iom::DataType → tt::tt_metal::DataType` table from which both the public span and the internal lookup are derived.
- Removes the redundant `checked_plane_count` call from the public `create_tensor` path while preserving the same check inside the `TtnnTensor` constructor.
- Preserves the capability rejection category (`std::runtime_error` from `create_tensor`), the timing (before the API mutex is taken and before any native object exists), the exact capability membership (`BOOL`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `BF16`, `F32`), and the per-pair native mapping (`BOOL`/`U8`/`I8` → `UINT8`, `U16`/`I16` → `UINT16`, `U32` → `UINT32`, `I32` → `INT32`, `BF16` → `BFLOAT16`, `F32` → `FLOAT32`).
- Preserves the `static_assert` on `TTNN_ENABLED`-built code that the published key span and the mapping agree (no key in the span without a mapping pair; no mapping pair without a key in the span).
- Does **not** expose a new public capability API; does **not** touch `AR-003` (the public `Device::supported_data_types()` predicate) or any TTNN queue/copy/extent-validation logic beyond the redundant call deletion.

## Implementation references

- **Modify:** `src/ttnn/device.cpp` (anonymous namespace, lines 32–72) — collapse `kSupportedDataTypes` (array of nine `iom::DataType`) and the nine-arm `native_dtype` switch into one `constexpr std::array kSupportedToNative` of `{iom::DataType, tt::tt_metal::DataType}` pairs; replace `is_supported(DataType)` with a span membership check over a compile-time-derived key span; replace the body of `native_dtype(DataType)` with a linear scan over the same pairs. The derived key span and the mapping pair list must come from the same `kSupportedToNative` constant. Mechanical derivation must use a `constexpr` lambda or `std::index_sequence`/`std::apply` expansion that produces a `std::array<iom::DataType, N>` of the `.first` fields; no manual `DataType` literal may appear outside `kSupportedToNative`.
- **Modify:** `src/ttnn/device.cpp:443-446` — the `is_supported(spec.data_type)` check on the public path stays; only its implementation is replaced by the derived span membership test. The `std::runtime_error` thrown on rejection stays, with the same message text.
- **Modify:** `src/ttnn/device.cpp:453` — delete the `static_cast<void>(checked_plane_count(spec))` line. The check still runs inside the `TtnnTensor` constructor at `:188`, which remains the sole extent-validation gate before native allocation.
- **Modify:** `src/ttnn/device.cpp` (anonymous namespace, near `kSupportedToNative`) — add a `static_assert` that every `pair.first` is unique (no duplicate keys) and a `static_assert` that the size of the derived key span equals `kSupportedToNative.size()`. Both assertions guard against silently breaking the derivation.
- **Read:** `include/iom/ttnn/device.hpp` — `ttnn_supported_data_types()` returns `std::span<const DataType>`; the return type and contract are unchanged. The header carries no native TTNN dtype and is unaffected.
- **Read:** `src/ttnn/device.cpp:177-209` (`TtnnTensor` constructor) — keep the existing inline `checked_plane_count(spec)` call and the `native_dtype(spec.data_type)` call inside the constructor body. Their behavior is preserved verbatim.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:289-335` (`TEST_CASE("TTNN supported-type table acceptance and rejection")`) — exercises the public span (size, `BF16` membership) and rejects every non-member with `std::runtime_error`. This test continues to pass unchanged because the capability set and rejection category are preserved.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:341-416` (`TEST_CASE("TTNN rejects overflowing and narrowing extents before native allocation")`) — proves the constructor-side `checked_plane_count` still rejects overflow and `uint32_t` ceiling violations before native allocation. This test continues to pass unchanged because the constructor check is preserved.

## Requirements

1. `kSupportedToNative` is the one hand-maintained TTNN capability declaration: a `constexpr std::array` of `{iom::DataType, tt::tt_metal::DataType}` pairs, ordered as today (`BOOL`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `BF16`, `F32`), with the same `pair.second` values used today. No `iom::DataType` literal may appear in `src/ttnn/device.cpp` outside this constant.
2. `ttnn_supported_data_types()` returns a `std::span<const iom::DataType>` whose elements are mechanically extracted from `kSupportedToNative` at compile time (e.g. a `constexpr` helper that takes `std::index_sequence_for<...>` and returns a `std::array<iom::DataType, N>` of the `.first` fields; the span wraps that derived array). The returned span's element type, ordering, and exact membership must match the finding's published set.
3. `native_dtype(DataType)` looks up its argument in the same `kSupportedToNative` pairs (linear scan is acceptable) and returns the matching `.second`. The `default:` arm throws `std::invalid_argument` with the same message text as today. The function signature, `[[nodiscard]]` annotation, and throw category are preserved.
4. The public `create_tensor` rejection branch (the body of the `if (!is_supported(spec.data_type))` check at `:443-446`) uses the derived key span for membership and throws the same `std::runtime_error` with the same message text. The check runs before `spec.validate()`'s `api_mutex_` lock acquisition and before any native object exists, preserving the documented pre-allocation rejection timing.
5. The public `create_tensor` path no longer calls `checked_plane_count`; the call is removed from `:453`. The `TtnnTensor` constructor at `:188` continues to call `checked_plane_count(spec)` as the sole extent-validation gate before native allocation. Removing the public call must not change the set of `TensorSpec`s the backend accepts or the exception categories observed by callers.
6. A `static_assert` verifies that the derived key span has the same size as `kSupportedToNative` (i.e. `kSupportedKeys.size() == kSupportedToNative.size()`). A second `static_assert` verifies key uniqueness within `kSupportedToNative` (no two pairs share a `.first`). Both assertions live next to the constant.
7. The capability rejection category (`std::runtime_error`), timing (before any native object), exact capability membership, per-pair native mapping, span ordering, and exception message texts are preserved exactly. The public header `include/iom/ttnn/device.hpp` carries no new declarations and continues to expose only `ttnn_supported_data_types()` and `make_ttnn_device(std::uint32_t)`.

## Non-goals

- Adding, removing, or reordering any `iom::DataType` in the published capability set. The membership is fixed at the nine types listed above.
- Introducing a public `Device::supported_data_types()` capability predicate or otherwise touching `AR-003`. The free function `ttnn_supported_data_types()` remains the single public surface.
- Changing the constructor-side extent validation, the `native_dtype` throw category (`std::invalid_argument`), the `runtime_error` capability rejection text, or any TTNN queue/copy/asynchronous-failure logic.
- Sharing the mapping with CPU/CUDA/ROCm backends or moving it into a common header. The mapping is TTNN-private and lives only in the `ttnn` anonymous namespace.
- Building, formatting, linting, or running the conformance suite as part of this task. Verification commands belong to the implementer per the Verification section.

## Acceptance criteria

- [ ] `src/ttnn/device.cpp` declares exactly one capability constant, `kSupportedToNative`, and no `iom::DataType` literal appears in the file outside that constant. `grep` for `DataType::` in `src/ttnn/device.cpp` outside the constant returns only the constants used in pair definitions and pair-iteration code.
- [ ] `ttnn_supported_data_types()` returns the same nine-element span, in the same order, with the same elements (`BOOL`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `BF16`, `F32`).
- [ ] `native_dtype(BOOL|U8|I8) == tt::tt_metal::DataType::UINT8`; `native_dtype(U16|I16) == tt::tt_metal::DataType::UINT16`; `native_dtype(U32) == tt::tt_metal::DataType::UINT32`; `native_dtype(I32) == tt::tt_metal::DataType::INT32`; `native_dtype(BF16) == tt::tt_metal::DataType::BFLOAT16`; `native_dtype(F32) == tt::tt_metal::DataType::FLOAT32`; any other input throws `std::invalid_argument` with the same message text as before.
- [ ] The public `create_tensor` branch continues to reject non-supported `iom::DataType` with `std::runtime_error` and the same message text, before the `api_mutex_` lock is taken and before any native object exists.
- [ ] Editing one pair (or adding/removing a pair) in `kSupportedToNative` automatically changes `ttnn_supported_data_types()`'s returned span's size and membership, with no further source edits. Demonstrated by removing a pair in a throwaway change and observing the span shrink.
- [ ] `static_assert(kSupportedKeys.size() == kSupportedToNative.size())` and the uniqueness assertion compile. There is no `iom::DataType` literal repeated as a pair `.first`.
- [ ] The public `create_tensor` body does not call `checked_plane_count`; the `TtnnTensor` constructor still calls `checked_plane_count(spec)` and continues to throw `std::overflow_error` on overflow or `uint32_t` ceiling violation before native allocation. The acceptance of every representable spec is unchanged.
- [ ] `include/iom/ttnn/device.hpp` is unchanged: it still declares only `ttnn_supported_data_types()` and `make_ttnn_device(std::uint32_t)` with the same signatures and contracts.

## Verification

- `cmake --build build/ttnn --target iom_ttnn -j` — TTNN translation unit compiles with `TTNN_ENABLED=ON`. The build must fail (compile error) if the two `static_assert`s are violated.
- `ctest --test-dir build/ttnn --output-on-failure -R '^iom_ttnn_conformance_tests$' --subcase='TTNN supported-type table acceptance and rejection'` — existing capability matrix test remains green. It iterates all 22 declared `iom::DataType` values plus three grouped formats and asserts acceptance only for the nine supported types and rejection (`std::runtime_error`) for every other entry.
- `ctest --test-dir build/ttnn --output-on-failure -R '^iom_ttnn_conformance_tests$' --subcase='TTNN rejects overflowing and narrowing extents before native allocation'` — existing extent-validation test remains green. It proves the constructor-side `checked_plane_count` still rejects `std::size_t{1} << 63 * 2 * 16 * 16` (leading-plane overflow) and `(std::size_t{1} << 32) + 1` columns (`uint32_t` ceiling violation) before any native object is created.
- `grep -nE 'DataType::[A-Z0-9_]+' src/ttnn/device.cpp | grep -v kSupportedToNative` — the implementer runs this and observes only `iom::DataType` literals inside `kSupportedToNative` (the `.first` fields). Any literal outside that constant indicates a violation of the single-source invariant and must be folded back into the mapping.
- `grep -n checked_plane_count src/ttnn/device.cpp` — exactly one call site remains (inside the `TtnnTensor` constructor). Two call sites indicate the public-side call was not removed.
- TTNN hardware is required: the targeted build and conformance run use real Tenstorrent silicon. CPU-only builds cannot validate `native_dtype`'s `tt::tt_metal::DataType` return values; that mapping is enforced by the TTNN SDK headers at compile time and exercised on hardware by the conformance matrix.
