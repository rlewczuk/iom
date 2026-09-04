# Document SafeTensorView's non-owning mapping lifetime

**Order:** 50
**Priority:** P2 — medium documentation and API-use defect. `SafeTensorView` exposes raw pointers into an owning memory map without stating that the store must outlive every view and raw pointer.
**Blocked by:** `37-AR-009-share-safetensors-store`
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-004`
**Review severity:** medium
**Review verification:** strongly-supported, confidence 82

## Outcome

`SafeTensorView`, `SafeTensorsStore`, `SafeTensorsFile`, and `SafeTensorsDir` document their non-owning mapping contract in `include/iom/safetensors.hpp`. The comments state that the store owns the mapped bytes, that every `SafeTensorView` and raw pointer obtained from it is valid only while the store remains alive, and that a by-value view return does not extend that lifetime.

A focused existing-suite test demonstrates the safe weight-ingestion pattern: copy the bytes into caller-owned storage while the store is alive, then destroy the store and use only the copied bytes. No ownership or public API change is introduced. The structural alternative of putting `shared_ptr<const MappedFile>` into each view is not part of this task.

## Current failure

`SafeTensorView` (`include/iom/safetensors.hpp:13-39`) stores `const uint8_t* data_` and `size_t nbytes_` into a `MappedFile` owned by `SafeTensorsFile` or by one of the `SafeTensorsFile` objects held by `SafeTensorsDir`. `raw<T>()` returns a typed pointer to those bytes without retaining the store. `MappedFile::~MappedFile` unmaps the region.

Therefore this usage is invalid but currently undocumented:

```cpp
iom::SafeTensorView view = dir["layer.0.weight"];
const float* weights = view.raw<float>();
// dir is destroyed here; weights now points into an unmapped region.
```

The review verified the pointer-only implementation and the by-value `SafeTensorsDir::operator[]` return. The impact is use-after-unmap or silently corrupted weights when a caller destroys the store before staging the bytes into caller-owned tensors.

## Scope

- Add Doxygen-style non-owning lifetime comments immediately above `SafeTensorView`, `SafeTensorsStore`, `SafeTensorsFile`, and `SafeTensorsDir` in `include/iom/safetensors.hpp`.
- State the concrete owner in each comment: `MappedFile file_` for `SafeTensorsFile`, and `std::vector<std::unique_ptr<SafeTensorsFile>> files_` for `SafeTensorsDir`.
- State that `SafeTensorsStore` must outlive every view it produced and every raw pointer obtained from those views. State that copying a `SafeTensorView` copies only the pointer/metadata and does not extend lifetime.
- Add one test case to `test/test_safetensors.cpp` that copies a view's bytes while its store is alive, destroys the store, and verifies the copied bytes remain usable. The test must never dereference a raw pointer after store destruction.
- Keep `SafeTensorView` members, `raw<T>()`, `SafeTensorsStore` virtual methods, mapping ownership, and all production parsing unchanged.

## Implementation references

- **Modify:** `include/iom/safetensors.hpp:13` — document `SafeTensorView` as a copyable, non-owning byte view. Mention that `raw<T>()` and `nbytes()` borrow the enclosing store's mapped bytes, that the store must outlive the view and any raw pointer, and that destruction invalidates them.
- **Modify:** `include/iom/safetensors.hpp:41` — document the `SafeTensorsStore` lifetime rule for all views and raw pointers it returns.
- **Modify:** `include/iom/safetensors.hpp:55` — document that `SafeTensorsFile` owns `MappedFile file_` and must outlive views and raw pointers produced from it.
- **Modify:** `include/iom/safetensors.hpp:77` — document that `SafeTensorsDir` owns `files_`; its by-value `operator[]` does not extend the selected file's lifetime.
- **Read:** `include/iom/tensor.hpp:140-146` — mirror the existing `TensorView` non-owning comment style without copying its tensor-storage terminology.
- **Read:** `include/iom/mmap.hpp:8-26` and `src/mmap.cpp:44-51` — confirm `MappedFile` is non-copyable and unmaps its region in the destructor.
- **Modify:** `test/test_safetensors.cpp` — add one case after `SafeTensorsFile satisfies SafeTensorsStore interface`, using existing `TempDir`, `write_safetensors_file`, `deterministic_payload`, `SafeTensorsFile`, and `SafeTensorView` helpers.
- **Read/consume:** `docs/changes/0001-tensor-view/37-AR-009-share-safetensors-store` — apply comments to the post-refactor class declarations; do not modify the shared store base or parser in this task.

## Requirements

1. The `SafeTensorView` comment contains the exact phrase `non-owning`, identifies the mapped bytes as belonging to the enclosing store, says the store must outlive the view and raw pointer, and says copying the view does not retain the store.
2. The `SafeTensorsStore` comment states that the store owns the backing mapping resources and must outlive every view and raw pointer it produced.
3. The `SafeTensorsFile` comment names `MappedFile file_` as the owner and repeats the outlive rule.
4. The `SafeTensorsDir` comment names `std::vector<std::unique_ptr<SafeTensorsFile>> files_` as the owner and states that by-value `operator[]` does not extend the selected file lifetime.
5. The comments do not promise shared ownership, pointer stability after destruction, automatic invalidation diagnostics, or a structural mapping reference from the view.
6. The test writes one valid F32 `{4}` entry, constructs a `SafeTensorsFile`, obtains `SafeTensorView view = file["w"]`, and copies exactly `view.nbytes()` bytes into a caller-owned `std::vector<std::uint8_t>` while `file` is alive.
7. After the store scope ends, the test compares only the caller-owned copy with the original payload. It never dereferences, passes to `memcmp`, indexes, or otherwise reads the raw pointer captured from the destroyed store.
8. Existing file, directory, dtype, missing-key, and mapping tests remain unchanged and continue to observe their current exception categories and data.
9. No production source, mapping ownership, public signature, private data member, or CMake target changes beyond the four comments is allowed.
10. No sanitizer-dependent invalid read, signal expectation, memory poisoning, `mprotect`, or structural shared-ownership workaround is added. An invalid raw-pointer read after destruction is undefined behavior and cannot be made a portable documentation test.

## Non-goals

- Adding `shared_ptr<const MappedFile>`, `weak_ptr`, reference counting, or another ownership mechanism to `SafeTensorView`.
- Changing `SafeTensorsFile`, `SafeTensorsDir`, `SafeTensorsStore`, `SafeTensorView`, `MappedFile`, or `raw<T>()` signatures or storage.
- Changing parser validation, shard lookup, duplicate-key handling, tensor staging, or any production caller.
- Reading a raw pointer after the store is destroyed to provoke a fault. The review's “poisoned mapping” verification alternative is rejected here because it would add undefined behavior or platform-specific process handling to a documentation-only fix.
- Modifying `src/safetensors.cpp`, `src/mmap.cpp`, any allocator/backend file, any model file, or CMake.
- Making a returned `SafeTensorView` own or prolong a mapping.

## Acceptance criteria

- [ ] Four class-level comments exist in `include/iom/safetensors.hpp` immediately above `SafeTensorView`, `SafeTensorsStore`, `SafeTensorsFile`, and `SafeTensorsDir`.
- [ ] The comments are grep-checkable: `non-owning` appears on `SafeTensorView`; `MappedFile` appears in the file-level comment; `SafeTensorsFile` appears in the directory-level comment; `outlive` appears in all four comments.
- [ ] The comments explicitly state that the store owns the backing bytes and that destruction invalidates all produced views and raw pointers.
- [ ] One test named `SafeTensorView is non-owning: copy bytes before store destruction` exists in `test/test_safetensors.cpp` and passes through the existing `iom_tests` target.
- [ ] The test copies exactly the valid payload while the store is alive, destroys the store, and compares only the copied vector afterward; source inspection finds no dereference of a post-destruction pointer.
- [ ] Existing `test/test_safetensors.cpp` cases pass without production parser or mapping changes.
- [ ] `SafeTensorView` remains a raw-pointer view with no `shared_ptr`, `weak_ptr`, ownership field, or new public lifetime API.
- [ ] `git diff` for this task is limited to `include/iom/safetensors.hpp` comments and one test case in `test/test_safetensors.cpp`.

## Verification

Run the existing local CPU test target:

```bash
cmake --build build --target iom_tests
ctest --test-dir build --output-on-failure -R '^iom_tests$'
```

Confirm by source inspection that the four comments contain the required lifetime language and that the new test copies bytes before store destruction and never reads the destroyed store's raw pointer. Confirm `SafeTensorView` remains non-owning and that no file outside `include/iom/safetensors.hpp` and `test/test_safetensors.cpp` is touched. No accelerator or remote-development run is required; this task has no backend behavior.
