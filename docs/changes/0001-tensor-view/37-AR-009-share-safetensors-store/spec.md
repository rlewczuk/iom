# Share one private SafeTensors store base between SafeTensorsFile and SafeTensorsDir

**Order:** 37
**Priority:** P1 — bounded common-boundary maintenance defect. The byte-identical `operator[]`/`size`/`keys()` bodies and the duplicated `tensors_`/`keys_` members mean any future lookup-policy change (CC-003 is the demonstrated case) must be applied twice; consolidating the store into one private base restores the invariant for both classes at once.
**Blocked by:** `24-CC-003-reject-duplicate-shard-keys`
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-009`
**Review severity:** low
**Review verification:** verified, confidence 85

## Outcome

The `SafeTensorsFile` and `SafeTensorsDir` classes share one private container base class `iom::detail::SafeTensorsStoreBase` that owns the keyed `SafeTensorView` map, the ordered insertion-order key vector, and the four operations both classes need: `insert(key, view)`, `at(key)` returning `const SafeTensorView&`, `size()`, and `keys()`. The two public classes become parsers that feed the shared base via `base_.insert(key, view)`; each class keeps only its own state (`MappedFile file_` for `SafeTensorsFile`; `std::vector<std::unique_ptr<SafeTensorsFile>> files_` plus the diagnostic-only `std::unordered_map<std::string, std::string> existing_shard_paths_` for `SafeTensorsDir`) and delegates the three `SafeTensorsStore` overrides to the corresponding public `SafeTensorsStoreBase` methods. The shared base's `tensors_` and `keys_` members remain private; the lookup/size/keys operations are concrete `public` methods on the base. Lookup semantics (`std::out_of_range("safetensors tensor not found: " + name)` on a missing key, insertion-ordered `keys()`, `size()` equal to `keys().size()`) are unchanged. The CC-003 duplicate-key rejection from task `24-CC-003-reject-duplicate-shard-keys` continues to fire in `SafeTensorsDir::SafeTensorsDir` through `base_.insert(...)`. The `SafeTensorsStore` virtual interface is unchanged. Every existing `TEST_CASE` in `test/test_safetensors.cpp` passes without modification.

## Current failure

The invariant under AR-009 is identical lookup semantics for files and directories: `std::out_of_range("safetensors tensor not found: " + name)` on a missing key, insertion-ordered `keys()`, and `size()` equal to the keyed collection.

`SafeTensorsFile` (`src/safetensors.cpp:135-149`, `include/iom/safetensors.hpp:71-75`) and `SafeTensorsDir` (`src/safetensors.cpp:170-184`, `include/iom/safetensors.hpp:93-97`) both derive from `SafeTensorsStore` yet share nothing below it. Each owns a private `std::unordered_map<std::string, SafeTensorView> tensors_` and `std::vector<std::string> keys_`. The three `SafeTensorsStore` overrides are byte-identical:

- `SafeTensorsFile::operator[](const std::string&)` (`src/safetensors.cpp:135-141`) and `SafeTensorsDir::operator[]` (`src/safetensors.cpp:170-176`) both run `tensors_.find(name)`, throw `std::out_of_range("safetensors tensor not found: " + name)` on miss, and return the view.
- `SafeTensorsFile::size()` (`src/safetensors.cpp:143-145`) and `SafeTensorsDir::size()` (`src/safetensors.cpp:178-180`) both return `tensors_.size()`.
- `SafeTensorsFile::keys()` (`src/safetensors.cpp:147-149`) and `SafeTensorsDir::keys()` (`src/safetensors.cpp:182-184`) both return `keys_`.

A lookup-policy change must be replicated in two places. CC-003 (`docs/changes/0001-tensor-view/24-CC-003-reject-duplicate-shard-keys/spec.md`) is the demonstrated cost: the duplicated-key rejection, the insertion-only `keys_.push_back`, and the `std::runtime_error` message live entirely in `SafeTensorsDir` because the file-level parser already keyed its map from unique header names, but the same `insert(key, view)` operation is the shape both classes want. The bug surface is small today (~30 lines) but it is exactly the spot where a divergence already costs a duplicate fix.

## Scope

- Add one private container base class `iom::detail::SafeTensorsStoreBase` declared in `include/iom/safetensors.hpp` inside `namespace iom::detail`. The class exposes four `public` operations and keeps two `private` members:
  - `public bool insert(const std::string& name, const SafeTensorView& view)` returning `true` on success (the key was new and `tensors_.emplace(name, view)` succeeded, then `keys_.push_back(name)` ran); `false` when the key was already present and the `tensors_.emplace` was a no-op (the existing view is preserved; `keys_` is not mutated).
  - `public const SafeTensorView& at(const std::string& name) const` returning the view by const-reference when the key is present, and throwing `std::out_of_range("safetensors tensor not found: " + name)` when the key is absent. The body performs `auto it = tensors_.find(name); if (it == tensors_.end()) { throw std::out_of_range("safetensors tensor not found: " + name); } return it->second;`. The reference return replaces the by-value return used by the previous per-class `operator[]` (`src/safetensors.cpp:135-141`, `:170-176`); the public `SafeTensorsStore::operator[]` overrides below copy the reference into a value to preserve the existing public-API signature exactly.
  - `public size_t size() const noexcept` returning `tensors_.size()`.
  - `public const std::vector<std::string>& keys() const noexcept` returning `keys_`.
  - `private std::unordered_map<std::string, SafeTensorView> tensors_`.
  - `private std::vector<std::string> keys_`.
  The `public`/`private` visibility is fixed: the four operations are reachable from the two `SafeTensorsStore` subclasses; the two data members are not reachable from outside `SafeTensorsStoreBase`.
- The three `SafeTensorsStore` overrides (`SafeTensorsFile::operator[]`, `SafeTensorsFile::size`, `SafeTensorsFile::keys`, and the corresponding three on `SafeTensorsDir`) become `inline` definitions inside `include/iom/safetensors.hpp` that delegate to the corresponding `public` `SafeTensorsStoreBase` method. The override bodies are exactly:
  - `operator[](const std::string& name) const` → `return base_.at(name);` (preserves the by-value `SafeTensorView` signature exactly; `at` returns `const SafeTensorView&` and the copy is the same one the previous per-class implementations performed implicitly when returning the map iterator's pair value).
  - `size() const` → `return base_.size();`
  - `keys() const` → `return base_.keys();`
  These three bodies live in `include/iom/safetensors.hpp` next to the class declarations. `src/safetensors.cpp` carries no override definitions. The override signatures, return types, `[[nodiscard]]` markers, and `override` keywords are unchanged.
- `SafeTensorsFile` retains `MappedFile file_` and `iom::detail::SafeTensorsStoreBase base_` as its private members. Its constructor body continues to parse the file at `src/safetensors.cpp:77-133` and feeds the base via `base_.insert(name, view)` for every entry that survives the per-file payload-size match, ordered-range, and overlap checks from task `23-CC-001-validate-safetensors-payload-lengths` (`src/safetensors.cpp:117-131`). The two terminal statements `keys_.push_back(name); tensors_.emplace(name, ...)` at `src/safetensors.cpp:129-131` become a single `base_.insert(name, view)` call whose return value the file path does not consume (per-file keys are unique by construction so the return is always `true`).
- `SafeTensorsDir` retains `std::vector<std::unique_ptr<SafeTensorsFile>> files_`, the diagnostic-only `std::unordered_map<std::string, std::string> existing_shard_paths_` (the exact member introduced by `24-CC-003-reject-duplicate-shard-keys`), and `iom::detail::SafeTensorsStoreBase base_` as its private members. Its constructor at `src/safetensors.cpp:151-167` continues to iterate the sorted shard paths, construct each `SafeTensorsFile`, and feed the base via `base_.insert(key, (*file)[key])`. The CC-003 duplicate-key check from task `24-CC-003-reject-duplicate-shard-keys` is the binding behavioral prerequisite: on `base_.insert(...) == false`, the constructor throws `std::runtime_error("safetensors duplicate tensor key across shards: '" + key + "' in " + new_shard_path + " (already loaded from " + existing_shard_paths_.at(key) + ")")` with both shard paths interpolated. `existing_shard_paths_` is populated on every successful `base_.insert(...)` and read on collision. The inner-loop body `tensors_.emplace(key, (*file)[key]); keys_.push_back(key);` at `src/safetensors.cpp:162-165` becomes `const std::string path_str = path.string(); for (const auto& key : file->keys()) { const bool inserted = base_.insert(key, (*file)[key]); if (inserted) { existing_shard_paths_.emplace(key, path_str); } else { throw std::runtime_error("safetensors duplicate tensor key across shards: '" + key + "' in " + path_str + " (already loaded from " + existing_shard_paths_.at(key) + ")"); } }`.
- `SafeTensorsStore`'s public virtual interface (`include/iom/safetensors.hpp:41-53`) — `virtual ~SafeTensorsStore() = default;`, `virtual SafeTensorView operator[](const std::string& name) const = 0;`, `virtual size_t size() const = 0;`, `virtual const std::vector<std::string>& keys() const = 0;` — is unchanged. The base class is not a `SafeTensorsStore` subclass; it is a member of both subclasses (composition, not inheritance). No public header outside `include/iom/safetensors.hpp` is changed.
- The `SafeTensorView` class (`include/iom/safetensors.hpp:13-39`) is unchanged.
- Throw categories are unchanged: `std::out_of_range` for missing keys (the base's `at(name)` throws it on miss; the public `operator[]` override propagates it), `std::runtime_error` for the per-file parser rejections (CC-001) and the cross-shard duplicate-key rejection (CC-003). No new exception type, no `std::error_code`, no new error category.
- Insertion order is preserved: `SafeTensorsFile`'s `keys()` keeps the file-header key order produced by the JSON iteration at `src/safetensors.cpp:100-132`; `SafeTensorsDir`'s `keys()` keeps the sorted-shard + within-shard-key order produced at `src/safetensors.cpp:151-167`. `SafeTensorsStoreBase::insert` performs `tensors_.emplace(name, view)` then `keys_.push_back(name)` on success, mirroring the order in `src/safetensors.cpp:129-131` and `:162-165`.
- `SafeTensorsFile` and `SafeTensorsDir` continue to be non-copyable and non-movable by virtue of the existing `SafeTensorsFile(const SafeTensorsFile&) = delete; SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;` declarations at `include/iom/safetensors.hpp:59-60` and the matching `SafeTensorsDir` declarations at `:81-82`. `iom::detail::SafeTensorsStoreBase` is implicitly non-copyable through its `std::unordered_map` and `std::vector` members.
- The CC-003 boundary is explicit and binding: this task owns the shared store base (the keyed map, the ordered keys vector, the four public operations `insert`, `at`, `size`, `keys`). Task `24-CC-003-reject-duplicate-shard-keys` owns the `SafeTensorsDir::SafeTensorsDir` throw site, the diagnostic message text, and the `existing_shard_paths_` member. Task `24-CC-003-reject-duplicate-shard-keys` is a behavioral prerequisite for this task — the `insert` operation's `false` return on collision is the mechanism through which CC-003's throw fires, and the `existing_shard_paths_` member is the diagnostic surface CC-003 reads to produce the kept-shard path. This task does not introduce the CC-003 throw or its test case; both live in task 24. Both specs are kept distinct; neither subsumes the other.

## Implementation references

- **Modify:** `include/iom/safetensors.hpp`. Add `namespace iom::detail { class SafeTensorsStoreBase { public: bool insert(const std::string& name, const SafeTensorView& view) { auto [it, inserted] = tensors_.emplace(name, view); if (inserted) { keys_.push_back(name); } return inserted; } const SafeTensorView& at(const std::string& name) const { auto it = tensors_.find(name); if (it == tensors_.end()) { throw std::out_of_range("safetensors tensor not found: " + name); } return it->second; } size_t size() const noexcept { return tensors_.size(); } const std::vector<std::string>& keys() const noexcept { return keys_; } private: std::unordered_map<std::string, SafeTensorView> tensors_; std::vector<std::string> keys_; }; }` declared before the `SafeTensorsFile` class at `include/iom/safetensors.hpp:55`. The four public operations are defined `inline` in the header so the source TU does not need to carry the definitions. The two data members are `private` and are reachable only from `SafeTensorsStoreBase`'s own member functions; the two subclasses (`SafeTensorsFile` and `SafeTensorsDir`) reach the data only through the four public operations. Then `SafeTensorsFile` (`include/iom/safetensors.hpp:55-75`) replaces its private members `std::unordered_map<std::string, SafeTensorView> tensors_;` and `std::vector<std::string> keys_;` with `iom::detail::SafeTensorsStoreBase base_;` and gains three `inline` override definitions next to its existing override declarations: `[[nodiscard]] SafeTensorView operator[](const std::string& name) const override { return base_.at(name); }`, `[[nodiscard]] size_t size() const override { return base_.size(); }`, `[[nodiscard]] const std::vector<std::string>& keys() const override { return base_.keys(); }`. `SafeTensorsDir` (`include/iom/safetensors.hpp:77-97`) replaces its private members `tensors_` and `keys_` with `iom::detail::SafeTensorsStoreBase base_;`, retains `std::vector<std::unique_ptr<SafeTensorsFile>> files_;` and the diagnostic-only `std::unordered_map<std::string, std::string> existing_shard_paths_;` (added by `24-CC-003-reject-duplicate-shard-keys`), and gains the same three `inline` override definitions as `SafeTensorsFile` delegating to `base_`. The public member function signatures, `[[nodiscard]]` markers, `override` markers, and `delete`d copy/assignment declarations are unchanged.
- **Modify:** `src/safetensors.cpp`. `SafeTensorsFile::SafeTensorsFile(const std::string& filename)` at `src/safetensors.cpp:77-133`: the two terminal statements `keys_.push_back(name); tensors_.emplace(name, SafeTensorView(parse_dtype(dtype_it->get<std::string>()), std::move(shape), data_begin + begin, end - begin));` at lines 129-131 become a single `base_.insert(name, SafeTensorView(parse_dtype(dtype_it->get<std::string>()), std::move(shape), data_begin + begin, end - begin));`. The per-file payload-size match, ordered-range, and overlap checks at `src/safetensors.cpp:117-131` are unchanged. `SafeTensorsFile::operator[]` (`src/safetensors.cpp:135-141`), `SafeTensorsFile::size()` (`src/safetensors.cpp:143-145`), and `SafeTensorsFile::keys()` (`src/safetensors.cpp:147-149`) are deleted; they live as `inline` overrides in `include/iom/safetensors.hpp`. `SafeTensorsDir::SafeTensorsDir(const std::string& dirname)` at `src/safetensors.cpp:151-167`: the inner-loop body at lines 162-165 becomes `const std::string path_str = path.string(); for (const auto& key : file->keys()) { const bool inserted = base_.insert(key, (*file)[key]); if (inserted) { existing_shard_paths_.emplace(key, path_str); } else { throw std::runtime_error("safetensors duplicate tensor key across shards: '" + key + "' in " + path_str + " (already loaded from " + existing_shard_paths_.at(key) + ")"); } }`. The aggregation loop's outer shell (directory iteration at `src/safetensors.cpp:152-157`, sort at `:158`, `files_.push_back(std::move(file))` at `:166`) is unchanged. `SafeTensorsDir::operator[]` (`src/safetensors.cpp:170-176`), `SafeTensorsDir::size()` (`src/safetensors.cpp:178-180`), and `SafeTensorsDir::keys()` (`src/safetensors.cpp:182-184`) are deleted; they live as `inline` overrides in `include/iom/safetensors.hpp`.
- **Read:** `include/iom/safetensors.hpp:41-53` — `SafeTensorsStore`'s virtual interface is read but not modified.
- **Read:** `include/iom/safetensors.hpp:13-39` — `SafeTensorView` is unchanged. `SafeTensorView` is passed by value into `base_.insert(...)`; `base_.at(name)` returns `const SafeTensorView&` which the override copies into the by-value return.
- **Read:** `src/safetensors.cpp:14-58` — anonymous-namespace helpers (`read_le_u64`, `parse_dtype`, `json_size`) are unchanged.
- **Read:** `docs/changes/0001-tensor-view/24-CC-003-reject-duplicate-shard-keys/spec.md` — the prerequisite task that owns the `existing_shard_paths_` member, the `std::runtime_error` message text, and the `SafeTensorsDir rejects duplicate keys across shards` test case. This task consumes task 24's `existing_shard_paths_` member to produce the kept-shard path; task 24's throw site is the only place that throws on duplicate-key detection, and it reads the `false` return from `base_.insert(...)`.
- **Read:** `docs/changes/0001-tensor-view/23-CC-001-validate-safetensors-payload-lengths/spec.md` — the per-file payload-size match, ordered-range, and overlap checks at `src/safetensors.cpp:117-131` continue to run before any `base_.insert(...)` call. Task 23 owns the per-file rejection logic and the `SafeTensorsFile rejects mismatched, overlapping, and out-of-order entries` test case.
- **Tests:** `test/test_safetensors.cpp` — every existing `TEST_CASE` passes without modification:
  - `TEST_CASE("SafeTensors dtype maps every accepted string to its DataType")` at `test/test_safetensors.cpp:114-167` exercises `SafeTensorsFile::size()`, `SafeTensorsFile::keys()`, and `SafeTensorsFile::operator[]`; the new header-side overrides delegate to `base_.size()`, `base_.keys()`, and `base_.at(name)` respectively and return identical results.
  - `TEST_CASE("SafeTensors dtype rejects unrecognized and nonstandard strings")` at `:169-184` throws from the constructor at the per-file rejection sites (CC-001 logic), not from `iom::detail::SafeTensorsStoreBase`.
  - `TEST_CASE("SafeTensorsFile exposes tensors from a generated file")` at `:186-213` checks `SafeTensorsFile::keys()` order, `SafeTensorsFile::size()`, `SafeTensorsFile::operator[]`, and the `std::out_of_range` missing-key case.
  - `TEST_CASE("SafeTensorsFile satisfies SafeTensorsStore interface")` at `:215-232` reads `SafeTensorsStore` through a reference; the virtual interface is unchanged so this case compiles and runs unchanged.
  - `TEST_CASE("SafeTensorsDir exposes tensors from every file in a directory")` at `:234-279` exercises `SafeTensorsDir` with disjoint shards; `iom::detail::SafeTensorsStoreBase` preserves insertion order and `size() == keys().size()`.
  - `TEST_CASE("SafeTensorsDir rejects missing directory")` at `:281-283` continues to throw from the directory iterator.
  - The `TEST_CASE("SafeTensorsDir rejects duplicate keys across shards")` introduced by task 24 passes without modification; its throw fires from the `base_.insert(...) == false` branch at `src/safetensors.cpp:162-165`, and its disjoint-subcase assertion (`size() == keys().size()`) still holds.
  - The `TEST_CASE("SafeTensorsFile rejects mismatched, overlapping, and out-of-order entries")` introduced by task 23 passes without modification; the per-file rejection sites at `src/safetensors.cpp:117-131` are unchanged.

## Requirements

- `iom::detail::SafeTensorsStoreBase` is a private class declared in `include/iom/safetensors.hpp` inside `namespace iom::detail`. It has `public` operations `bool insert(const std::string& name, const SafeTensorView& view)`, `const SafeTensorView& at(const std::string& name) const`, `size_t size() const noexcept`, `const std::vector<std::string>& keys() const noexcept`. It has `private` data members `std::unordered_map<std::string, SafeTensorView> tensors_` and `std::vector<std::string> keys_`. The four public operations are defined `inline` in the header.
  - `insert` performs `auto [it, inserted] = tensors_.emplace(name, view);`; on `inserted == true` it `keys_.push_back(name)` and returns `true`; on `inserted == false` it leaves both members unchanged and returns `false`. The order `emplace` then `push_back` matches the order in `src/safetensors.cpp:129-131` and `:162-165`, preserving insertion-order semantics.
  - `at` performs `auto it = tensors_.find(name); if (it == tensors_.end()) { throw std::out_of_range("safetensors tensor not found: " + name); } return it->second;`. The throw is identical to the existing `src/safetensors.cpp:138` / `:173` text.
  - `size` returns `tensors_.size()`.
  - `keys` returns `keys_`.
- `SafeTensorsFile` declares `iom::detail::SafeTensorsStoreBase base_;` and `MappedFile file_;` as its private members. Its three `SafeTensorsStore` overrides (`operator[]`, `size`, `keys`) are `inline` definitions in `include/iom/safetensors.hpp` delegating to `base_`. `operator[]` is `return base_.at(name);` (the by-value `SafeTensorView` return signature is preserved by copying the `const SafeTensorView&` returned from `at`); `size` is `return base_.size();`; `keys` is `return base_.keys();`. `SafeTensorsFile::SafeTensorsFile(const std::string& filename)` at `src/safetensors.cpp:77-133` ends the entry loop with a single `base_.insert(name, view)` call replacing the two terminal statements at `src/safetensors.cpp:129-131`. The per-file payload-size match, ordered-range, and overlap rejections from task 23 continue to run before the insert. `src/safetensors.cpp:135-149` is deleted.
- `SafeTensorsDir` declares `iom::detail::SafeTensorsStoreBase base_;`, `std::vector<std::unique_ptr<SafeTensorsFile>> files_;`, and `std::unordered_map<std::string, std::string> existing_shard_paths_;` as its private members. Its three `SafeTensorsStore` overrides are `inline` definitions in `include/iom/safetensors.hpp` delegating to `base_` exactly as `SafeTensorsFile`'s overrides do. `SafeTensorsDir::SafeTensorsDir(const std::string& dirname)` at `src/safetensors.cpp:151-167` ends the inner shard-iteration loop with the CC-003 throw on `base_.insert(...) == false`, reading `existing_shard_paths_.at(key)` for the kept-shard path; on `base_.insert(...) == true` it `existing_shard_paths_.emplace(key, path_str)` and continues. `src/safetensors.cpp:170-184` is deleted.
- The `SafeTensorsStore` virtual interface (`include/iom/safetensors.hpp:41-53`) and the public member function signatures of `SafeTensorsFile` and `SafeTensorsDir` are unchanged. `test/test_safetensors.cpp:215-232` (`SafeTensorsFile satisfies SafeTensorsStore interface`) reads through the interface; the test compiles unchanged because the interface is the same.
- Throw categories are unchanged: `std::out_of_range` for missing keys (raised inside `SafeTensorsStoreBase::at`, propagated through `SafeTensorsStore::operator[]`), `std::runtime_error` for every other rejection (CC-001 per-file validation, CC-003 cross-shard duplicate keys). No new exception type, `std::error_code`, or error category is introduced.
- `MappedFile file_` (`SafeTensorsFile`, `include/iom/safetensors.hpp:72`) and `std::vector<std::unique_ptr<SafeTensorsFile>> files_` (`SafeTensorsDir`, `include/iom/safetensors.hpp:94`) stay in their owning classes — the review requires mapping-file ownership preserved.
- `iom::detail::SafeTensorsStoreBase` is declared only in `include/iom/safetensors.hpp` inside `namespace iom::detail`. No new public header is added.
- `CMakeLists.txt` is unchanged: `src/safetensors.cpp` remains part of the `iom` library target at `CMakeLists.txt:79`; `include/iom/safetensors.hpp` remains a public include at `CMakeLists.txt:88`. No new TU, no new library target.
- `wc -l src/safetensors.cpp` shows the source file at most 175 lines (186 current minus ~6 deleted override bodies minus ~3 inner-loop simplification plus ~3 added `existing_shard_paths_.emplace` lines = ≤180; this target is 175). `wc -l include/iom/safetensors.hpp` shows the header at most 140 lines (100 current plus ~25 added `iom::detail::SafeTensorsStoreBase` declaration with four public operations plus ~6 added inline override definitions plus ~4 added `existing_shard_paths_` member).
- `iom::detail::SafeTensorsStoreBase` is not a `SafeTensorsStore` subclass. It is a private member, not a virtual base; `SafeTensorsFile` and `SafeTensorsDir` continue to derive directly from `SafeTensorsStore`. The four public operations on the base are reachable from the two subclasses because they are `public` members; the two data members remain `private` and are not reachable from the subclasses.

## Non-goals

- Restructuring the per-file parser or the `SafeTensorsFile` entry loop beyond the two terminal statements that become `base_.insert(...)`. The CC-001 payload-size match, ordered-range, and overlap checks (`docs/changes/0001-tensor-view/23-CC-001-validate-safetensors-payload-lengths/spec.md`) stay in `SafeTensorsFile::SafeTensorsFile` exactly as written; this task does not merge task 23's wording or alter the per-file rejection messages.
- Owning the CC-003 duplicate-key rejection message. The message text and the `SafeTensorsDir rejects duplicate keys across shards` test case are owned by `docs/changes/0001-tensor-view/24-CC-003-reject-duplicate-shard-keys/spec.md`. This task exposes the `insert(key, view) -> bool` operation whose `false` return task 24 reads; the `existing_shard_paths_` member is task 24's diagnostic surface.
- Adding a new exception type, `std::error_code`, or error category. `std::runtime_error` is the invalid-input category for every rejection on the SafeTensors boundary.
- Adding a CRTP helper, mixin, or template-base for the shared container. Composition with `iom::detail::SafeTensorsStoreBase base_;` is the chosen shape; templates are rejected.
- Changing the directory iteration, the file-extension filter, the sort order, or the `MappedFile` ownership pattern at `src/safetensors.cpp:152-167`.
- Changing the per-file JSON parsing, header length validation, dtype mapping, or per-entry rejection logic. Those live in `SafeTensorsFile::SafeTensorsFile` at `src/safetensors.cpp:77-133` and are owned by task 23.
- Changing the `SafeTensorView` shape at `include/iom/safetensors.hpp:13-39`, the `MappedFile` shape in `include/iom/mmap.hpp`, or the `DataType` enumerators. No public header outside `include/iom/safetensors.hpp` is touched.
- Adding a new TU, a new library target, or a CMake change. The shared base is header-only.
- Documentation, lifetime-annotation, or `SafeTensorView`/`SafeTensorsStore` lifetime contract work. The ST-004 finding is a separate review item; AR-009 addresses only the duplicated container.
- Cross-shard range consistency validation, payload-size match across shards, or any other ingestion correctness fix beyond AR-009's container-deduplication scope.
- Performance work, allocation pooling, or copy-elision in the parser.
- Promoting the header-only base to a separate header file. `include/iom/safetensors.hpp` is the only public header for this translation unit.

## Acceptance criteria

- [ ] `iom::detail::SafeTensorsStoreBase` exists as a private class declared in `include/iom/safetensors.hpp` inside `namespace iom::detail`. It exposes four `public` operations: `bool insert(const std::string& name, const SafeTensorView& view)`, `const SafeTensorView& at(const std::string& name) const`, `size_t size() const noexcept`, `const std::vector<std::string>& keys() const noexcept`. It keeps two `private` data members: `std::unordered_map<std::string, SafeTensorView> tensors_` and `std::vector<std::string> keys_`.
- [ ] The three `SafeTensorsStore` overrides on `SafeTensorsFile` (`operator[]`, `size`, `keys`) and the three on `SafeTensorsDir` are `inline` definitions in `include/iom/safetensors.hpp`, each delegating to the corresponding `public` `SafeTensorsStoreBase` method. The exact override bodies are: `operator[](const std::string& name) const override { return base_.at(name); }`; `size() const override { return base_.size(); }`; `keys() const override { return base_.keys(); }`. Six override bodies total, three unique bodies, each reachable through composition. `src/safetensors.cpp` carries zero override definitions; the only `src/safetensors.cpp` references to `operator[]`, `size`, or `keys` are the `base_.insert(...)` calls in the two constructor bodies and the CC-003 throw site in `SafeTensorsDir::SafeTensorsDir`.
- [ ] `grep -nE '\boperator\[\]\b|\bsize\(\)\b|\bkeys\(\)\b' src/safetensors.cpp` returns no matches for override definitions.
- [ ] `grep -nE '\btensors_\b|\bkeys_\b' src/safetensors.cpp` returns no matches. The standalone private members `tensors_` and `keys_` are absent from `src/safetensors.cpp`.
- [ ] `SafeTensorsFile::SafeTensorsFile` at `src/safetensors.cpp:77-133` ends with a single `base_.insert(name, SafeTensorView(parse_dtype(dtype_it->get<std::string>()), std::move(shape), data_begin + begin, end - begin));` call per entry; the `keys_.push_back(name); tensors_.emplace(name, ...)` two-liner at `src/safetensors.cpp:129-131` is gone. The per-file payload-size match, ordered-range, and overlap rejections continue to run before the insert.
- [ ] `SafeTensorsDir::SafeTensorsDir` at `src/safetensors.cpp:151-167` ends with `const std::string path_str = path.string(); for (const auto& key : file->keys()) { const bool inserted = base_.insert(key, (*file)[key]); if (inserted) { existing_shard_paths_.emplace(key, path_str); } else { throw std::runtime_error("safetensors duplicate tensor key across shards: '" + key + "' in " + path_str + " (already loaded from " + existing_shard_paths_.at(key) + ")"); } }`. The `tensors_.emplace(key, ...); keys_.push_back(key);` two-liner at `src/safetensors.cpp:162-165` is gone. The aggregation loop's outer shell (directory iteration at `:152-157`, sort at `:158`, `files_.push_back(std::move(file))` at `:166`) is unchanged.
- [ ] `wc -l src/safetensors.cpp` shows the source file at most 175 lines. `wc -l include/iom/safetensors.hpp` shows the header at most 140 lines.
- [ ] `grep -nE 'class SafeTensorsStore|virtual SafeTensorView|virtual size_t|virtual const std::vector<std::string>' include/iom/safetensors.hpp` returns exactly the four lines from `include/iom/safetensors.hpp:41-53` and no new ones. The `override` markers on the two subclasses' three overrides each appear once.
- [ ] The existing `TEST_CASE("SafeTensors dtype maps every accepted string to its DataType")` at `test/test_safetensors.cpp:114-167` passes without modification; `file.size() == std::size(cases)`, `file.keys().size() == std::size(cases)`, every `file[key_name(i)]` returns the correct `dtype()`/`shape()`/`nbytes()`/`raw<uint8_t>()` content.
- [ ] The existing `TEST_CASE("SafeTensors dtype rejects unrecognized and nonstandard strings")` at `:169-184` passes without modification; every rejected dtype throws `std::runtime_error` from the per-file parser.
- [ ] The existing `TEST_CASE("SafeTensorsFile exposes tensors from a generated file")` at `:186-213` passes without modification; `file.keys()[0] == "a"`, `file.keys()[1] == "b"`, both `file["a"]` and `file["b"]` return the expected payload, and `file["missing"]` throws `std::out_of_range`.
- [ ] The existing `TEST_CASE("SafeTensorsFile satisfies SafeTensorsStore interface")` at `:215-232` passes without modification; the `iom::SafeTensorsStore& store = file;` reference compiles and `store["x"]` returns the same view as `file["x"]`.
- [ ] The existing `TEST_CASE("SafeTensorsDir exposes tensors from every file in a directory")` at `:234-279` passes without modification; `shards.size() == 3`, `shards.keys().size() == 3`, every `shards[key]` matches the originating `SafeTensorsFile`'s payload, and `shards["no-such-tensor-anywhere"]` throws `std::out_of_range`.
- [ ] The existing `TEST_CASE("SafeTensorsDir rejects missing directory")` at `:281-283` passes without modification.
- [ ] The `TEST_CASE("SafeTensorsDir rejects duplicate keys across shards")` from task `24-CC-003-reject-duplicate-shard-keys` passes without modification; the duplicate-key throw fires from the `base_.insert(...) == false` branch at `src/safetensors.cpp:162-165`, the message text matches task 24's spec verbatim, and the disjoint-subcase assertion (`size() == keys().size()`) still holds.
- [ ] The `TEST_CASE("SafeTensorsFile rejects mismatched, overlapping, and out-of-order entries")` from task `23-CC-001-validate-safetensors-payload-lengths` passes without modification; the per-file rejection sites at `src/safetensors.cpp:117-131` are unchanged.
- [ ] No public header outside `include/iom/safetensors.hpp` is changed. `CMakeLists.txt` is unchanged. No new TU, no new library target, no new namespace symbol outside `iom::detail::SafeTensorsStoreBase`.

## Verification

This is a CPU-only common-boundary refactor. No accelerator hardware is required. The verification uses the repository's local CPU build and the existing `iom_tests` target that already exercises `test/test_safetensors.cpp`. The remote-development procedure is not required.

1. Configure with `BUILD_TESTING=ON` and every accelerator disabled, matching the established CPU isolation pattern (`docs/changes/0001-tensor-view/05-cpu-storage-transfer-copy`):

   ```bash
   cmake -S . -B build/ar009 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF
   cmake --build build/ar009 -j --target iom_tests
   ```

2. Confirm the source-file shrink and the header-side consolidation:

   ```bash
   wc -l src/safetensors.cpp include/iom/safetensors.hpp
   grep -nE '\boperator\[\]\b|\bsize\(\)\b|\bkeys\(\)\b' src/safetensors.cpp
   grep -nE '\btensors_\b|\bkeys_\b' src/safetensors.cpp
   ```

   Expected: `src/safetensors.cpp` ≤ 175 lines; `include/iom/safetensors.hpp` ≤ 140 lines; the `grep` for `operator[]`/`size()`/`keys()` in `src/safetensors.cpp` returns no override-definition matches; the `grep` for `tensors_`/`keys_` in `src/safetensors.cpp` returns no matches.

3. Confirm the virtual interface is unchanged:

   ```bash
   grep -nE 'class SafeTensorsStore|virtual SafeTensorView|virtual size_t|virtual const std::vector<std::string>' include/iom/safetensors.hpp
   ```

   Expected: exactly the four lines from `include/iom/safetensors.hpp:41-53`. The `override` markers on the two subclasses' three overrides each appear once.

4. Confirm the base's public/protected/private visibility is correct:

   ```bash
   grep -nE 'public:|private:|protected:' include/iom/safetensors.hpp
   ```

   Expected: in the `iom::detail::SafeTensorsStoreBase` block, exactly one `public:` label preceding the four operation declarations and exactly one `private:` label preceding the two data members; no `protected:` label. Outside that block, the existing `public:` / `private:` labels of `SafeTensorView`, `SafeTensorsStore`, `SafeTensorsFile`, `SafeTensorsDir` are unchanged.

5. Run the focused suite first; every existing case must pass without modification:

   ```bash
   ./build/test/iom_tests --test-case="SafeTensors*" -d
   ```

   Expected: six `TEST_CASE`s (dtype-map, dtype-reject, single-file, store-interface, directory-shards, missing-directory) plus task 23's `SafeTensorsFile rejects mismatched, overlapping, and out-of-order entries` and task 24's `SafeTensorsDir rejects duplicate keys across shards` all pass with no skips. Exit status 0. Each `CHECK`/`REQUIRE` line for `size()`, `keys()`, `operator[]`, and `out_of_range` matches today's output.

6. Run the full `iom_tests` executable to confirm every other case is unaffected by the header refactor:

   ```bash
   ctest --test-dir build/ar009 --output-on-failure -R "^iom_tests$"
   ```

   Expected: every existing `test_safetensors.cpp` case green plus every other `test_iom.cpp` case green; no skips; no new failures elsewhere in `iom_tests`.

7. Confirm the no-untargeted-side-effect invariant by running the broader CPU test executables:

   ```bash
   ctest --test-dir build/ar009 --output-on-failure
   ```

   Expected: unchanged-green on every other test target. The header refactor touches only `include/iom/safetensors.hpp` and `src/safetensors.cpp`; no other source file is modified.

8. Diff evidence — the shared base's four public operations are now the single source of truth for the keyed map:

   ```bash
   grep -nE 'bool insert|const SafeTensorView& at|size_t size|const std::vector<std::string>& keys' include/iom/safetensors.hpp
   grep -nE 'base_\.insert' src/safetensors.cpp
   ```

   Expected: in `include/iom/safetensors.hpp`, one `bool insert` declaration, one `const SafeTensorView& at` declaration, one `size_t size` declaration, and one `const std::vector<std::string>& keys` declaration inside `iom::detail::SafeTensorsStoreBase`, plus the two subclasses' three override declarations (six total) that each delegate to `base_.…`. In `src/safetensors.cpp`, two `base_.insert` call sites (one in `SafeTensorsFile::SafeTensorsFile`, one in `SafeTensorsDir::SafeTensorsDir`). No third insertion site.

The expected observation: every existing `test_safetensors.cpp` case passes unchanged, the override bodies live in exactly one place per class (the header) and delegate to `iom::detail::SafeTensorsStoreBase`'s four public operations, `src/safetensors.cpp` shrinks, `include/iom/safetensors.hpp` grows by the shared base declaration and inline overrides, and any future lookup-policy change (the exact pattern that motivated CC-003 and CC-001) now happens in one method body inside the shared base instead of in two duplicated bodies in the two subclasses.
