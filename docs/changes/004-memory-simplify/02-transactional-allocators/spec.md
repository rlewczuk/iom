# Make canonical allocators transactional

**Order:** 02
**Priority:** P0 — arena ownership cannot safely rely on bookkeeping that loses ranges on host allocation failure.
**Blocked by:** None
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

`ListAllocator` and `FixedSizeAllocator` retain exact ownership and free-space state when host bookkeeping throws, while preserving existing allocation, reuse, best-fit, coalescing, alignment, address, error-category, and fixed-slot behavior. Failed bookkeeping cannot lose or duplicate a range.

## Scope

- Make `ListAllocator::alloc`, `ListAllocator::free`, and free-range coalescing in `include/iom/alloc.hpp` / `src/alloc.cpp` strongly exception-safe. The current `alloc` path erases the chosen range before `insert_free_block` and `allocated_` insertion (`src/alloc.cpp:139-180`), and `free` erases `allocated_` before insertion/coalescing (`src/alloc.cpp:164-180`); replace these commit orderings with a transaction that either completes the whole state transition or leaves every ownership/free-range observable unchanged.
- Preserve best-fit selection, alignment padding and suffix splitting, sorted free ranges, adjacent-range coalescing, stable returned addresses, `std::bad_alloc` for exhaustion, `std::invalid_argument` for unknown/double frees, zero-allocation/null-free behavior, and the standalone public `Allocator` APIs.
- Make `FixedSizeAllocator::free` transactional around the state/push ordering currently at `src/alloc.cpp:265-282`: a failure to append the released index must leave the slot marked in use and owned by the caller; a successful release must append exactly one index and then make the slot reusable.
- Add focused behavioral fault/rollback coverage in `test/test_alloc.cpp` alongside the existing best-fit, coalescing, reuse, and fixed-slot cases (`test/test_alloc.cpp:110-348`).

## Implementation references

- **Modify:** `include/iom/alloc.hpp` — `ListAllocator` and `FixedSizeAllocator` private bookkeeping declarations only as needed to support transactional transitions without changing public standalone APIs.
- **Modify:** `src/alloc.cpp` — `ListAllocator::alloc`, `ListAllocator::free`, `ListAllocator::insert_free_block`, `ListAllocator::coalesce`, and `FixedSizeAllocator::free`; preserve existing allocator algorithms and error categories while preventing partial mutations.
- **Read:** `src/alloc.cpp:139-180` — current `ListAllocator` erase-before-insert allocation/free hazards to eliminate.
- **Read:** `src/alloc.cpp:265-282` — current `FixedSizeAllocator::free` state/push ordering and fixed-slot bookkeeping.
- **Tests:** `test/test_alloc.cpp:110-348` — existing allocator behavior and geometry tests to extend with deterministic bookkeeping-failure rollback scenarios.

## Requirements

- Before committing a `ListAllocator` allocation or free, prepare every potentially throwing vector/map operation needed for the complete transition, or provide exact rollback guards. A bookkeeping exception must leave allocated ownership, free ranges, `free_bytes()`, sorted/coalesced geometry, and all reusable addresses exactly as before the call.
- A successful `ListAllocator::alloc` removes exactly the chosen block, preserves nonzero padding and suffix ranges, and records the returned address exactly once. A successful `ListAllocator::free` removes exactly the recorded allocation, inserts exactly one range, and coalesces adjacent ranges without losing or duplicating bytes.
- A `FixedSizeAllocator::free` must not clear `in_use_[index]` until `free_indices_` can accept the index, and must not push an index more than once. If bookkeeping fails, repeated allocation/free attempts observe the original in-use state and fixed-slot geometry.
- Keep `LinearAllocator` semantics unchanged. Do not add internal allocator mutexes; synchronization for device-owned allocators belongs to task 04.
- Exercise fault points that cover allocation bookkeeping and release bookkeeping, then prove recovery by checking free-space/free-slot counts, successful reuse of the same addresses, no duplicate reuse, and no lost capacity. Keep existing best-fit, coalescing, alignment, invalid-free, exhaustion, reset, and fixed-slot geometry behavior covered.

## Non-goals

- Device-owned allocator locking or live arena reset policy; those belong to task 04.
- A new allocator implementation, compaction, paging, arena growth, or changes to `LinearAllocator`.
- Changes to public standalone allocator APIs, alignment rules, address placement, error categories, payload/stride geometry, or allocator ownership outside transactional bookkeeping.
- Backend arena setup, tensor/workspace migration, or native allocation policy.

## Acceptance criteria

- [ ] Injected host bookkeeping failure during `ListAllocator` allocation or free leaves `free_bytes()`, ownership, free-range geometry, and all subsequent allocation addresses unchanged; a later successful operation can reuse/coalesce the original range exactly once.
- [ ] Injected failure during `FixedSizeAllocator::free` leaves the slot in use and the free-slot count unchanged; after recovery, release and reuse return the slot exactly once with no duplicate index or lost slot.
- [ ] Existing best-fit, adjacent coalescing, alignment/address, exhaustion and invalid-free behavior remains unchanged, and fixed-slot block count, stride, payload size, and reuse ordering remain intact.
- [ ] Focused regression coverage resides in `test/test_alloc.cpp` and observes behavior through public allocator APIs rather than implementation-only state.

## Verification

- `cmake --build build --target iom_tests` (not run; supervisor gate).
- `ctest --test-dir build --output-on-failure -R '^iom_tests$'` (not run; supervisor gate).
