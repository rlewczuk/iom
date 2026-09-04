# Reject duplicate SafeTensors shard keys with both shard paths in the diagnostic

**Order:** 24
**Priority:** P1 — bounded invalid-input correctness defect in normal sharded weight loading; the established invalid-input category plus a new test close the gap without broadly gating other work
**Blocked by:** 23-CC-001-validate-safetensors-payload-lengths
**Source:** `docs/changes/0001-tensor-view/review.md` — `CC-003`
**Review severity:** medium
**Review verification:** verified, confidence 94

## Outcome

`iom::SafeTensorsDir` rejects any directory whose shards declare the same tensor key twice: the constructor throws `std::runtime_error` naming the duplicated key and the paths of both contributing shards, before the directory becomes observable. A directory with disjoint shard key sets loads identically to today: `size() == keys().size()`, `operator[]` returns the same payload as the originating `SafeTensorsFile`, and `keys()` keeps the insertion order used by the existing tests. The existing `std::out_of_range("safetensors tensor not found: " + name)` from `operator[]` is unchanged.

## Current failure

The invariant is that the store's `size()`, `keys()`, and `operator[]` describe one coherent collection; a sharded weight directory with conflicting names is invalid input and must not load silently.

The shard-aggregation loop at `src/safetensors.cpp:151-167` constructs each `SafeTensorsFile`, iterates the file's `keys()`, and runs `tensors_.emplace(key, (*file)[key]); keys_.push_back(key);`. The map insertion silently keeps the first shard's `SafeTensorView` on key collision, while `keys_.push_back(key)` unconditionally appends the duplicated name. After loading a directory with two shards that both define key `k`:

- `size()` returns `1` (the map has one entry — shard A's `SafeTensorView`),
- `keys().size()` returns `2` (both occurrences were appended),
- `operator[]("k")` returns shard A's payload.

The invariant `size() == keys().size()` is broken; iteration double-visits the shadowed name; a mis-sharded model loads without diagnostic and runs with the wrong tensor data.

Reproduced independently by two reviewers against built `libiom.a` (merged; the numerical-testing candidate is folded here per the review). The existing test at `test/test_safetensors.cpp:234-279` only covers disjoint shard key sets, so the regression slips through.

The established exception category for malformed/unsupported input at the SafeTensors boundary is `std::runtime_error` — see `src/safetensors.cpp:44,49,53,81,86,97,105,114,126` and the CC-001 (`23-CC-001-validate-safetensors-payload-lengths`) check at the same module. The duplicated-key rejection uses the same category and names the offending key plus both shard paths so a mis-sharded checkpoint is diagnosed at ingestion.

## Scope

- In `SafeTensorsDir::SafeTensorsDir` (`src/safetensors.cpp:151-167`), check the `std::unordered_map::emplace` result. On collision, throw `std::runtime_error` naming the duplicated key, the offending new shard path, and the already-loaded shard path whose entry is kept. The `keys_.push_back(key)` step runs strictly inside the successful-insertion branch so a rejected key never appears in `keys()`.
- The lookup of the kept shard's path uses a single, exact mechanism: a new `std::unordered_map<std::string, std::string> existing_shard_paths_` member on `SafeTensorsDir`, populated in lockstep with `tensors_` on every successful emplace. On collision, the diagnostic reads the kept shard's path from `existing_shard_paths_[key]`. There is no second source of truth and no recomputation from `files_`.
- The new member is declared in `include/iom/safetensors.hpp` alongside `files_`, `tensors_`, and `keys_`, in the same `private:` section. It is diagnostic-only and is not read by any public accessor (`operator[]`, `size()`, `keys()`).
- Successful insertion ordering is preserved exactly: shards are loaded in the sorted order produced by `std::sort(paths.begin(), paths.end())` (`src/safetensors.cpp:158`), and within a shard keys arrive in the file's own `keys()` order — unchanged from today. The new test relies on this order to assert deterministic key-sequence observations.
- Public surface unchanged: `SafeTensorsDir`'s constructor signature, `operator[]`, `size()`, `keys()`, the `std::out_of_range` from `operator[]`, the `SafeTensorsStore` virtual interface, and the `SafeTensorsFile` class are untouched. `SafeTensorsFile` continues to be the per-shard construction path and is the place where task 23's payload-length and ordering checks already run; this task does not duplicate that work.
- Throw category is `std::runtime_error`. No new exception type is introduced; the existing `std::out_of_range` for missing keys is preserved.
- Cross-shard duplicate keys are detected at the directory boundary; per-file duplicates cannot exist because each `SafeTensorsFile` already keys its own map from the unique header names parsed out of its own header (`src/safetensors.cpp:129-131`).
- Disjoint-shard compatibility: a directory whose shards declare disjoint key sets constructs identically to today. `size() == keys().size()`, `operator[]` returns the same payload as the originating `SafeTensorsFile` for every key, and CC-003's disjoint-shard test case at `test/test_safetensors.cpp:234-279` keeps passing without modification.

## Coordination with task 23 (CC-001)

CC-001 owns changes to `src/safetensors.cpp` and `test/test_safetensors.cpp`. To avoid merge collisions, this task serializes behind CC-001 (`Blocked by: 23-CC-001-validate-safetensors-payload-lengths`) and the merge protocol below applies:

- **CC-001 lands first.** Its edits land in `src/safetensors.cpp` (entry-loop size check, file-local checked-arithmetic helpers, anonymous-namespace helpers `safetensors_checked_mul` / `safetensors_checked_add` / `safetensors_bits_to_bytes` / `safetensors_expected_payload_bytes`), the file-local helpers above the constructor, and a one-fixture modification to the existing `dtype-map` test at `test/test_safetensors.cpp:114-167` so each accepted dtype's payload byte count equals `ceil(leaf_bits(DataType) * 2 / 8)`.
- **CC-003 lands second.** Its edits land in `include/iom/safetensors.hpp` (one new private member `existing_shard_paths_` on `SafeTensorsDir`) and in the inner aggregation loop of `SafeTensorsDir::SafeTensorsDir` in `src/safetensors.cpp:151-167`. CC-003 does not modify the entry loop, the new file-local checked-arithmetic helpers, or any test case CC-001 touched.
- **Merge boundary.** The two tasks' edits do not overlap. CC-001 edits `src/safetensors.cpp` lines 13-58 (anonymous namespace), lines 100-132 (entry loop), and the file-level helpers above the entry loop; CC-003 edits `src/safetensors.cpp` lines 151-167 (`SafeTensorsDir::SafeTensorsDir` body only) and `include/iom/safetensors.hpp:93-97` (`SafeTensorsDir` private members). The disjoint regions are non-adjacent; both can be applied in either order as long as CC-001 lands first.
- **CC-003 does not re-touch CC-001's edited tests.** The "every existing case remains unchanged" claim in this spec excludes the `dtype-map` case at `test/test_safetensors.cpp:114-167`, which CC-001 is allowed to modify (its payload byte count and `nbytes()` / `memcmp` assertions become dtype-dependent). The disjoint-shard case at `test/test_safetensors.cpp:234-279`, the missing-directory case at `:281-283`, the `dtype-reject` case at `:169-184`, the single-file case at `:186-213`, and the store-interface case at `:215-232` are not touched by either task and continue to pass without modification. CC-003 must run after CC-001; if the implementer applies CC-003 first, the dtype-map case will fail under CC-001's per-entry size check, which is the explicit dependency.

## Implementation references

- **Modify:** `include/iom/safetensors.hpp` — `SafeTensorsDir` private members (lines 93-97). Add a fourth member `std::unordered_map<std::string, std::string> existing_shard_paths_;` next to `files_`, `tensors_`, and `keys_`. The member is private, has no public accessor, and is used only by the constructor.

- **Modify:** `src/safetensors.cpp` — `SafeTensorsDir::SafeTensorsDir` (lines 151-167). Replace the current `tensors_.emplace(key, (*file)[key]); keys_.push_back(key);` two-liner with the explicit duplicate check that uses `existing_shard_paths_`:

  ```cpp
  const std::string path_str = path.string();
  for (const auto& key : file->keys()) {
      auto [it, inserted] = tensors_.emplace(key, (*file)[key]);
      if (!inserted) {
          throw std::runtime_error(
              "safetensors duplicate tensor key across shards: '" + key +
              "' in " + path_str +
              " (already loaded from " + existing_shard_paths_.at(key) + ")");
      }
      existing_shard_paths_.emplace(key, path_str);
      keys_.push_back(key);
  }
  ```

  Both `tensors_.emplace` and `existing_shard_paths_.emplace` are guarded by the same `inserted` flag: the new map entry is written only after the `SafeTensorView` was actually inserted. The throw reads the kept shard's path from `existing_shard_paths_.at(key)`, which is populated by an earlier successful emplace. The diagnostic-message string is built by direct concatenation of the three required substrings (`"duplicate tensor key across shards"`, the duplicated key, both shard paths) — no placeholder, no deferred formatting, no recomputation.

  The throw fires before `files_.push_back(std::move(file))` and before any further shard's keys are visited, so the partially populated `SafeTensorsDir` is never returned.

- **Read:** `src/safetensors.cpp:160-166` — the current aggregation loop. The new code replaces the body of the inner `for` loop; the outer file-iteration loop and the `paths` sort are unchanged. `std::sort(paths.begin(), paths.end())` at `src/safetensors.cpp:158` continues to define the shard order, which is what makes the new test's insertion-order assertion deterministic.

- **Read:** `src/safetensors.cpp:170-184` — `operator[]`, `size()`, `keys()` continue to read from `tensors_` and `keys_` only and are unchanged. After the fix, `size() == tensors_.size()` (already true) and `keys().size() == keys_.size()` remain equal because every successful emplace is paired with one push.

- **Read:** `src/safetensors.cpp:117-131` — the per-file parser already rejects entries whose key collisions happen inside a single shard's header (the `SafeTensorsFile` constructs a `std::unordered_map<std::string, SafeTensorView>` with unique keys by construction). Cross-shard duplicates are exactly what this task adds a check for; do not re-validate per-file header uniqueness.

- **Read:** `docs/changes/0001-tensor-view/23-CC-001-validate-safetensors-payload-lengths/spec.md` — the prerequisite task's scope, line ranges, and helper additions. CC-003 must serialize behind CC-001 per the coordination section above; the line ranges in this spec refer to post-CC-001 `src/safetensors.cpp`, where CC-001 has shifted the entry loop body by the inserted checked-arithmetic helpers but has not changed the line range 151-167 (`SafeTensorsDir::SafeTensorsDir` body).

- **Tests:** `test/test_safetensors.cpp` — extend with one new `TEST_CASE("SafeTensorsDir rejects duplicate keys across shards")`. Use the existing `TempDir`, `TensorEntry`, `deterministic_payload`, and `write_safetensors_file` helpers (lines 18-112). Cover at minimum:

  - **First-occurrence wins, second rejects:** write two shards `a.safetensors` (header key `k`, payload bytes `AAAA`, `F32 {1}`) and `b.safetensors` (header key `k`, payload bytes `BBBB`, `F32 {1}`). Constructing `iom::SafeTensorsDir(dir.path().string())` must throw `std::runtime_error`; the message must contain `"duplicate tensor key across shards"`, the key `"k"`, and both shard paths. The distinct 4-byte payloads make a passing aggregation observable: if the dir silently kept the second occurrence, `operator[]("k")` would expose `BBBB`. The expected post-condition on the failure path is that the directory never finished constructing, so no further assertions on its members are reachable from the throwing path.

  - **Disjoint keys still work (regression of CC-003's own disjoint-shard case):** construct a fresh `TempDir`, write `a.safetensors` (key `a1`), `b.safetensors` (keys `b1`, `b2`), and `c.safetensors` (key `c1`) using `write_safetensors_file`. Construct `iom::SafeTensorsDir(dir.path().string())`; assert `size() == keys().size() == 4`, assert `keys() == {"a1", "b1", "b2", "c1"}` (sorted shard filenames, within-shard key order preserved), and assert `operator[]` returns each shard's own payload for every key. This subcase uses its own `TempDir` so the regression proof is independent of the rejection subcase.

  - **Insertion order preserved on success:** inside the disjoint subcase above, additionally assert `keys()[0] == "a1"`, `keys()[1] == "b1"`, `keys()[2] == "b2"`, `keys()[3] == "c1"` (sorted shard filenames `a.safetensors`, `b.safetensors`, `c.safetensors` plus the file's own key order, which matches the existing `:234-279` test).

  - **Three-way collision also rejects:** construct a fresh `TempDir`, write three shards each declaring key `k` with distinct 4-byte payloads. Construction must throw `std::runtime_error`; the message names key `k` and at least the first two colliding shard paths (the kept shard path and the path that triggered the rejection — the second collision lands on the shard whose key was already in `existing_shard_paths_` from the first collision's success).

  The case uses `write_safetensors_file` and `TempDir` exactly like the existing cases; it does not silently skip.

- **Read:** `test/test_safetensors.cpp:234-279` — CC-003's own disjoint-shard test case remains unchanged and continues to pass without modification; that is the disjoint-path regression anchor proving the fix does not break the disjoint path. CC-003 does not touch this case.

- **Read:** `test/test_safetensors.cpp:281-283` — `SafeTensorsDir rejects missing directory` remains unchanged and continues to pass; the new check fires only when shards construct successfully, so a missing directory still throws earlier (the directory iterator construction itself throws).

- **Read:** `test/test_safetensors.cpp:114-167` — the existing `dtype-map` case is owned by CC-001 (`23-CC-001-validate-safetensors-payload-lengths`). CC-001 modifies the per-entry fixture byte count and `nbytes()` / `memcmp` assertions to derive from `iom::detail::leaf_bits(cases[i].second)`; CC-003 does not touch this case but relies on CC-001 having landed first.

## Requirements

- `SafeTensorsDir::SafeTensorsDir` (`src/safetensors.cpp:151-167`) checks the result of every `tensors_.emplace(key, ...)` call. On a collision (key already present), the constructor throws `std::runtime_error("safetensors duplicate tensor key across shards: '<key>' in <new shard path> (already loaded from <kept shard path>)")`. The message is built by direct string concatenation of the three required substrings: the literal prefix `"safetensors duplicate tensor key across shards: '"`, the duplicated key, the literal `"' in "`, the new shard path, the literal `" (already loaded from "`, the kept shard path read from `existing_shard_paths_.at(key)`, and the literal `")"`. No placeholder, no formatting helper, no deferred substitution.
- `existing_shard_paths_.emplace(key, path_str)` and `keys_.push_back(key)` run only inside the successful-insertion branch. The new map entry is written only after the corresponding `SafeTensorView` was actually inserted, so `existing_shard_paths_` and `tensors_` are kept in lockstep. A rejected key never appears in `keys_`, so post-construction `size() == keys().size()` is preserved for both the success path (every successful emplace pairs with one push) and the failure path (the constructor throws before returning).
- The duplicate check fires before `files_.push_back(std::move(file))` and before any further shard's keys are visited, so a partially constructed `SafeTensorsDir` is never returned to the caller. The current code's order — emplace, push_back, then `files_.push_back` — is reshaped so the new throw happens before any per-shard bookkeeping that would otherwise leave a half-loaded directory.
- Successful insertion ordering is preserved exactly: shards are visited in the sorted order produced by `std::sort(paths.begin(), paths.end())` (`src/safetensors.cpp:158`), and within a shard keys arrive in the file's own `keys()` order. No key is reordered on the success path. The disjoint subcase of the new test asserts this order explicitly.
- `SafeTensorsFile::SafeTensorsFile` (`src/safetensors.cpp:77-133`), `SafeTensorsFile::operator[]` (`:135-141`), `SafeTensorsFile::size()` (`:143-145`), `SafeTensorsFile::keys()` (`:147-149`), `SafeTensorsDir::operator[]` (`:170-176`), `SafeTensorsDir::size()` (`:178-180`), and `SafeTensorsDir::keys()` (`:182-184`) are unchanged.
- `include/iom/safetensors.hpp:93-97` gains exactly one new private member on `SafeTensorsDir`: `std::unordered_map<std::string, std::string> existing_shard_paths_;`. The `SafeTensorsStore` virtual interface (`:41-53`), the `SafeTensorsFile` declaration (`:55-75`), and the `SafeTensorsDir` public member functions, deleted copy/assignment, and override markers are unchanged. The new member has no public accessor and no friend declaration; it is used only by the constructor and (on the failure path) by the exception message.
- Throw category is `std::runtime_error` only. No new exception type, `std::error_code`, or error category is introduced. The existing `std::out_of_range` from `operator[]` for missing keys is preserved.
- `iom::SafeTensorsFile` continues to be the per-shard construction entry point; the duplicated-key check lives only in `SafeTensorsDir`. Task 23's payload-length, overlap, and ordering checks run inside the per-file constructor and remain the canonical place to reject inconsistent per-shard payloads; this task does not duplicate them.
- Disjoint-shard compatibility: a directory whose shards declare disjoint key sets constructs identically to today, with `size() == keys().size()`, `operator[]` returning the same payload as the originating `SafeTensorsFile` for every key, and insertion ordering matching the sorted-shard / within-shard-key pattern.
- Serialization: this task lands after `23-CC-001-validate-safetensors-payload-lengths` (line `**Blocked by:**` above). The merge protocol is non-overlapping: CC-003 edits `src/safetensors.cpp:151-167` and `include/iom/safetensors.hpp:93-97`; CC-001 edits `src/safetensors.cpp` lines 13-58 (anonymous namespace), 100-132 (entry loop body), and the file-level helpers above the entry loop. The two regions are disjoint and both edits apply to a clean checkout in CC-001-then-CC-003 order.

## Non-goals

- Resolving the duplicated-key condition by silently picking one shard. The fix is to reject, not to disambiguate; a mis-sharded checkpoint is an invalid input that callers must rename or merge themselves.
- Validating that the keys across shards form a logically consistent model (e.g. that every weight name the consumer expects is present). The fix is a structural duplicate-key check; full-model validation is out of scope.
- Adding a new exception type, a new `std::error_code`, or a new error category. `std::runtime_error` is the chosen invalid-input category.
- Restructuring the per-file parser or the `SafeTensorsFile` class. Cross-shard duplicates are exactly the case the `SafeTensorsDir` constructor owns; the per-file parser already keys its map from unique header names.
- Refactoring `SafeTensorsFile` and `SafeTensorsDir` into a shared store implementation (the AR-009 finding in `review.md`). Both files remain independent; this fix is a bounded additive change to `SafeTensorsDir` only.
- Adding new public API surface, exposing `existing_shard_paths_` as a member accessor, or providing a `try_load` alternative constructor.
- Changing the shard ordering rule, the directory-iteration pattern, or the file-extension filter (`src/safetensors.cpp:152-157`). The fix is strictly inside the inner `for (const auto& key : file->keys())` loop.
- Cross-shard overlap validation between distinct keys. Per-shard disjoint ranges are enforced by task 23; cross-shard range consistency is impossible because each shard is a separate file.
- Performance or mapping-mode changes. The new check is a constant-time `emplace` result inspection per key plus one parallel emplace into `existing_shard_paths_` and one `at` lookup on the failure path.
- Modifying `test/test_safetensors.cpp:114-167` (the `dtype-map` case) or any other case CC-001 owns. CC-003's disjoint-shard case at `test/test_safetensors.cpp:234-279` and the missing-directory case at `:281-283` are not modified by either task; they remain unchanged.

## Acceptance criteria

- [ ] Constructing `iom::SafeTensorsDir` over a directory containing `a.safetensors` and `b.safetensors`, both declaring key `k` with distinct `F32 {1}` payloads, throws `std::runtime_error` from the constructor; the message contains the substrings `"duplicate tensor key across shards"`, the key `"k"`, the path of `a.safetensors`, and the path of `b.safetensors`. No `SafeTensorsDir` instance is returned.
- [ ] Constructing `iom::SafeTensorsDir` over the same directory with the fix not yet applied produces a dir whose `size() == 1` and `keys().size() == 2` (the failure mode documented in the review) — i.e. the new test must fail on the uncorrected tree and pass on the corrected tree.
- [ ] Constructing `iom::SafeTensorsDir` over a directory with three shards each declaring key `k` with distinct payloads throws `std::runtime_error`; the message names key `k` and at least the first two colliding shard paths. No `SafeTensorsDir` instance is returned.
- [ ] Constructing `iom::SafeTensorsDir` over a directory with three shards `a.safetensors` (key `a1`), `b.safetensors` (keys `b1`, `b2`), and `c.safetensors` (key `c1`) — disjoint key sets — succeeds and satisfies `size() == 4`, `keys().size() == 4`, `keys() == {"a1", "b1", "b2", "c1"}` (sorted shard filenames, within-shard key order preserved), and `operator[]` returns each shard's own payload for every key.
- [ ] The existing `TEST_CASE("SafeTensorsDir exposes tensors from every file in a directory")` at `test/test_safetensors.cpp:234-279` continues to pass without modification; that is the disjoint-path regression anchor owned by CC-003.
- [ ] The existing `TEST_CASE("SafeTensorsDir rejects missing directory")` at `test/test_safetensors.cpp:281-283` continues to pass without modification; the new check fires only after the directory iterator has found `.safetensors` shards.
- [ ] The existing `TEST_CASE("SafeTensors dtype rejects unrecognized and nonstandard strings")` at `test/test_safetensors.cpp:169-184`, the `SafeTensorsFile exposes tensors from a generated file` case at `:186-213`, and the `SafeTensorsFile satisfies SafeTensorsStore interface` case at `:215-232` continue to pass without modification; CC-001 does not touch these cases and CC-003 must not touch them either.
- [ ] The existing `TEST_CASE("SafeTensors dtype maps every accepted string to its DataType")` at `test/test_safetensors.cpp:114-167` is allowed to be modified by CC-001 to derive its per-entry payload byte count from `iom::detail::leaf_bits(cases[i].second)`. CC-003 does not modify this case; CC-003's verification confirms the case still passes under CC-001's modification (i.e. CC-003 lands after CC-001 in the merge order, so the dtype-map case is already in its post-CC-001 shape when CC-003 runs). The case's intent — every accepted safetensors dtype string maps to its `DataType` — is preserved.
- [ ] The new `TEST_CASE` reports at minimum three `CHECK_THROWS_AS(..., std::runtime_error)` passes (first-occurrence wins, three-way collision, message-content checks) plus the disjoint success path, with no skips, and the message-content `CHECK` macros verify both shard paths appear in the diagnostic.

## Verification

This is a CPU-only common-boundary fix. No accelerator hardware is required. The verification uses the repository's local CPU build and the existing `iom_tests` target that already exercises `test/test_safetensors.cpp`. CC-003 lands after CC-001; the verification assumes CC-001 has already merged and the dtype-map case is in its post-CC-001 shape.

1. Configure with `BUILD_TESTING=ON` and every accelerator disabled, matching the established CPU isolation pattern (`docs/changes/0001-tensor-view/05-cpu-storage-transfer-copy`):

   ```bash
   cmake -S . -B build/cc003 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF
   cmake --build build/cc003 -j --target iom_tests
   ```

2. Run the focused suite first; the new case must run and all subcases must pass:

   ```bash
   ./build/test/iom_tests --test-case="SafeTensorsDir rejects duplicate keys across shards"
   ```

   Expected: three `CHECK_THROWS_AS(..., std::runtime_error)` passes (first-occurrence wins, three-way collision, message-content checks) plus the disjoint success path, with no skips, exit status 0.

3. Reproduce the reviewed failure once on the uncorrected CC-003 portion of the tree (CC-003 stashed, CC-001 still applied): the disjoint shard set `a.safetensors` + `b.safetensors` both declaring key `k` with distinct payloads currently constructs a `SafeTensorsDir` whose `size() == 1` and `keys().size() == 2` (the failure mode documented in the review). After CC-003 lands, the new case throws and the uncorrected assertion is no longer reachable.

4. Run the full `iom_tests` executable to confirm every existing safetensors test (post-CC-001 fixture for the dtype-map case, plus CC-003's disjoint-shard and missing-directory cases untouched) plus the new case pass together:

   ```bash
   ctest --test-dir build/cc003 --output-on-failure -R "^iom_tests$"
   ```

   Expected: all existing and new `test/test_safetensors.cpp` cases green, no skips, no new failures elsewhere in `iom_tests`.

5. Confirm the no-untargeted-side-effect invariant by running the broader CPU test executables if present in this build configuration (matching the patterns already used at `docs/changes/0001-tensor-view/05-cpu-storage-transfer-copy/spec.md`):

   ```bash
   ctest --test-dir build/cc003 --output-on-failure
   ```

   Expected: unchanged-green on every other test target; no other source file is modified by this fix.

The remote-development procedure is not required because the fix is in a translation unit that builds and runs purely on the CPU host, and the established test infrastructure for the SafeTensors layer (`iom_tests`) covers the change. GPU evidence is not applicable to this finding.
