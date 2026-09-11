# Extract StagedWorker header

**Order:** 09
**Priority:** P0 — preserves a public umbrella contract while bringing iom.hpp under the line cap
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

`iom::detail::StagedWorker<Task>` is defined in the planned private detail header `include/iom/detail/staged_worker.hpp`, while `include/iom/iom.hpp` remains the public umbrella include that transitively provides the same template and all existing public declarations. The extraction is source factoring only: queue consumers, public signatures, capabilities, synchronization behavior, visibility, and ODR behavior remain unchanged, and both resulting headers are at most 499 physical lines.

## Scope

- Move the complete `iom::detail::StagedWorker` template currently at `include/iom/iom.hpp:31-203` to planned `include/iom/detail/staged_worker.hpp` without shortening, redesigning, or duplicating the implementation.
- Replace the removed class block at the same dependency point in `include/iom/iom.hpp` with the transitive include of `detail/staged_worker.hpp`, so existing consumers that include only `iom/iom.hpp` retain availability of `iom::detail::StagedWorker`.
- Keep `WorkspaceValidation`, `DeviceOps`, the `BinaryRequest` workspace fields, and `Block` together in `include/iom/iom.hpp`; do not move unrelated declarations as part of this extraction.
- Add the planned header to `IOM_HEADERS` in `CMakeLists.txt:82-90` so installation/build header tracking includes it.
- Preserve the existing queue consumers in `src/shared/gpu_queue.hpp`, `src/sycl/copy.cpp`, and `src/ttnn/device.cpp`, including their `StagedWorker<Task>::Callbacks`, `PublishPolicy`, `worker_`, and callback construction/use sites.

## Implementation references

- **Add (planned):** `include/iom/detail/staged_worker.hpp` — complete `iom::detail::StagedWorker<Task>` definition, including its callback aliases, `PublishPolicy`, `Callbacks`, constructors/destructor, submission, shutdown/drain, processing, worker loop, and synchronization members.
- **Modify:** `include/iom/iom.hpp:20-203` — include `detail/staged_worker.hpp` at the existing detail dependency point and remove only the inlined `StagedWorker` definition; retain `WorkspaceValidation`, `DeviceOps`, `BinaryRequest` workspace state, and `Block` in this header.
- **Modify:** `CMakeLists.txt:82-90` — add `include/iom/detail/staged_worker.hpp` to `IOM_HEADERS` without changing test source lists or public factory headers.
- **Read/consumer:** `src/shared/gpu_queue.hpp:168-219,649` — `make_worker_callbacks`, `StagedWorker<Task>` construction, and `worker_` member.
- **Read/consumer:** `src/sycl/copy.cpp:334-363,1065` — `SyclQueue` callback construction/start and `worker_` member.
- **Read/consumer:** `src/ttnn/device.cpp:582-598,1018` — `TtnnQueue` callback construction/start and `worker_` member.
- **Behavior references:** `test/test_iom.cpp`, `test/cpu/test_cpu.cpp`, and `test/cpu/test_cpu_conformance.cpp`, exercised by the existing `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests` targets.

## Requirements

- Move the complete template exactly once. Preserve `Execute`, `FenceComplete`, `FenceDestroy`, and `Complete` callback types; `PublishPolicy::{Splice,CompleteOnThrow}`; `Callbacks`; deleted copy/move operations; constructor/destructor behavior; `start`; `submit_copy`; and `shutdown_and_drain` signatures and error categories.
- Preserve `submit_copy` staging and publish ordering, `CompleteOnThrow` failure completion and rethrow behavior, notification timing, and all existing task/fence ownership and destruction behavior.
- Preserve `process`, `drain_list`, and `run` semantics, including in-order task processing, fence completion versus shutdown draining, repeatable completion/failure delivery, and the worker thread's shutdown behavior.
- Preserve `std::mutex`, `std::condition_variable`, staged/task list ownership, thread join, predicate waits, and all synchronization/lifetime ordering exactly. Do not introduce a new synchronization design, queue, global registry, backend switch, alias, shim, or cross-backend abstraction.
- Keep the template in `iom::detail` with the same private visibility and include it transitively from `iom/iom.hpp`; direct inclusion of the planned detail header must also be self-sufficient for the template's standard-library dependencies.
- Keep all existing queue consumers source-compatible without adding overloads or changing callback capture, `PublishPolicy`, task sequencing, workspace/staging lease completion, quarantine/registry behavior, or context behavior.
- Keep `include/iom/iom.hpp` and planned `include/iom/detail/staged_worker.hpp` at no more than 499 physical lines after normal formatting. Use the fewest responsibility-complete files and do not add a permanent line-count test.

## Non-goals

- No production behavior change, API redesign, public signature or capability change, error-category or validation-order change, ownership/lifetime change, allocation change, numerical change, or asynchronous ordering change.
- No movement of `WorkspaceValidation`, `DeviceOps`, `BinaryRequest` workspace fields, `Block`, or any other unrelated `iom.hpp` declaration.
- No compatibility aliases/shims, duplicate definition, backend-specific implementation, global registry, synchronization redesign, or new capability.
- No new tests, test source-list changes, public factory-header changes, formatter/line-count test, or unrelated CMake changes.

## Acceptance criteria

- [ ] The complete `iom::detail::StagedWorker<Task>` template exists exactly once in planned `include/iom/detail/staged_worker.hpp`, and `include/iom/iom.hpp` includes it transitively at the former dependency point.
- [ ] Existing consumers in `src/shared/gpu_queue.hpp`, `src/sycl/copy.cpp`, and `src/ttnn/device.cpp` compile without source changes to their callback, start, shutdown, drain, thread, or condition-variable behavior.
- [ ] `WorkspaceValidation`, `DeviceOps`, `BinaryRequest` workspace fields, and `Block` remain together in `include/iom/iom.hpp`; public headers/signatures/capabilities and private visibility/ODR behavior are unchanged.
- [ ] `include/iom/detail/staged_worker.hpp` is listed in `IOM_HEADERS` at `CMakeLists.txt:82-90`.
- [ ] `include/iom/iom.hpp` and `include/iom/detail/staged_worker.hpp` are each no more than 499 physical lines after normal formatting.
- [ ] Existing core and CPU tests provide behavior proof with no test additions, and a consumer that includes only `iom/iom.hpp` compiles and links successfully.

## Verification

- Configure the CPU-only build (not run while writing this specification):
  `cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF`.
- Build the affected library and existing core/CPU targets (not run while writing this specification):
  `cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests`.
- Run the existing core/CPU tests (not run while writing this specification):
  `ctest --test-dir build/split-cpu --output-on-failure -R '^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'`.
- Compile and link a throwaway consumer that includes only `#include <iom/iom.hpp>` against `build/split-cpu/libiom.a` with the project C++ standard and pthreads; do not add the consumer to the repository (not run while writing this specification).
- Run a scoped physical-line check over `include/iom/iom.hpp` and planned `include/iom/detail/staged_worker.hpp`, asserting each count is at most 499 (not run while writing this specification).
