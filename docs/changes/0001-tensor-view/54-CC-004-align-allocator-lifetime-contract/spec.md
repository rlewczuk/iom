# State one public allocator-lifetime contract across backend headers and the umbrella specification

**Order:** 54
**Priority:** P0 — public ownership/lifetime contract; the current split wording licenses a destruction order that calls `free` on a destroyed allocator object on every backend's failure path.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `CC-004`
**Review severity:** low
**Review verification:** verified, confidence 80

## Outcome

Every public surface that accepts an `Allocator&` states one identical lifetime rule — the allocator is borrowed and must outlive the returned device and every resource created through it — and the umbrella specification's ownership sections state the matching free rule: `allocator.free(address)` runs exactly once per successful tensor allocation, normally at tensor destruction and, for storage quarantined because a queued operation referencing it failed or was invalidated (the 49-ST-003 quarantine), deferred to device destruction after all of that device's tensors and queues are gone. After this change a CPU caller can no longer follow a CPU-only rule that permits destroying the allocator before the device, and the deferral the engines actually perform is written into the contract they are audited against.

## Current failure

The four allocator-taking backends state two different public rules:

- `include/iom/cpu/device.hpp:15-16` — "The allocator and the returned device must outlive every tensor and queue created through them." This does not require the allocator to outlive the `Device`.
- `include/iom/cuda/device.hpp:13-14`, `include/iom/rocm/device.hpp:14`, `include/iom/sycl/device.hpp:13-14` — "The allocator is borrowed and must outlive the returned device and every resource created through it."

The accelerator wording is the load-bearing one. Since 49-ST-003, a tensor destructor whose outstanding-work fence fails or was invalidated moves an `AllocatorCleanupAction` into the device-owned quarantine instead of freeing, and quarantine drains only from the device destructor: `registry_state_.quarantine.drain()` at `src/cpu/device.cpp:445-447` (`~CpuDevice`), `src/cuda/device.cpp:76-90` (`~CudaDevice`), `src/rocm/device.cpp:68-77` (`~RocmDevice`), and `src/ttnn/device.cpp:166-168` (`~TtnnDevice`). `AllocatorCleanupAction::run()` (`include/iom/outstanding_work_registry.hpp:69-87`) calls `allocator_.free(address_)` exactly once at that drain. Quarantine entries arise on the failure paths the review names — a failed or retained-failure operation, or a queue destroyed with unwaited work (queue teardown invalidates entries but keeps them address-indexed, so the later tensor destructor quarantines).

A CPU caller following the CPU header may therefore destroy tensors and queues, then the allocator, then the device. When the quarantine holds an entry, `~CpuDevice`'s drain invokes `free` on a destroyed allocator object: undefined behavior. The same engine thus publishes two allocator-lifetime rules depending on which header the caller read, and the umbrella spec was never amended: `docs/changes/0001-tensor-view/spec.md:449` still reads "destruction calls `allocator.free(address)` exactly once;" with no deferral permitted, and the §6 sentence at `spec.md:424` ("The `Device` and every caller-supplied allocator must outlive the tensors and queues that use them") permits the exact allocator-before-device order that the drain path punishes.

## Scope

- **Modify:** `include/iom/cpu/device.hpp` — replace the final sentence of the `make_cpu_device` doc comment with the exact accelerator wording so all four allocator-taking headers state one rule.
- **Modify:** `docs/changes/0001-tensor-view/spec.md` — §6 closing lifetime sentence and the §7 free-exactly-once bullet (plus one clarifying sentence after the §7 bullet list) so the published contract matches the applied deferred-free behavior.
- Documentation and public-wording change only: CPU header comment and umbrella-spec text. No runtime mechanism, registry, quarantine, destructor, or drain-order change on any backend.
- Backend coverage of the stated rule: CPU, CUDA, ROCm, SYCL (the factories that take `Allocator&`). TTNN supplies no allocator (`include/iom/ttnn/device.hpp:22-24`) and is untouched, as are the accelerator headers, whose wording is already the target.

## Implementation references

- **Modify:** `include/iom/cpu/device.hpp:10-17` — the `make_cpu_device` doc comment. Keep the first two sentences verbatim; replace only the final sentence (currently "The allocator and the returned device must outlive every tensor and queue created through them.") with:
  `The allocator is borrowed and must outlive the returned device and every resource created through it.`
- **Modify:** `docs/changes/0001-tensor-view/spec.md:424` — replace the sentence "The `Device` and every caller-supplied allocator must outlive the tensors and queues that use them." with:
  `The Device must outlive every tensor and queue it created, and every caller-supplied allocator must outlive its Device.`
- **Modify:** `docs/changes/0001-tensor-view/spec.md:449` — replace the bullet "- destruction calls `allocator.free(address)` exactly once;" with:
  `- destruction calls \`allocator.free(address)\` exactly once: at tensor destruction, or deferred to device destruction for storage quarantined because a queued operation referencing it failed or was invalidated;`
  and insert immediately after the §7 bullet list (before the TTNN paragraph at `spec.md:454`) the sentence:
  `Quarantined storage is released by the device destructor's quarantine drain, after all tensors and queues of that device are destroyed; this deferral is why the allocator must outlive the device.`
- **Read:** `include/iom/cuda/device.hpp:11-15`, `include/iom/rocm/device.hpp:11-15`, `include/iom/sycl/device.hpp:11-15` — the exact comment template the CPU header must converge to; do not reword these three.
- **Read:** `include/iom/device.hpp:13-19` and `include/iom/tensor.hpp:193-197` — the already-published "Device must outlive every tensor and queue it created" and "creating Device must outlive the tensor" rules; the replacement §6 sentence preserves both, so nothing is lost by dropping the old compound sentence.
- **Read:** `docs/changes/0001-tensor-view/49-ST-003-fence-tensor-destruction/spec.md` — requirements 6–7 and the tensor-destructor/device-drain paragraphs; the amended spec wording must use its vocabulary (`quarantine`, failed or invalidated fence, device destructor drains after all tensors and queues).
- **Read:** `src/cpu/device.cpp:493-553` (`~CpuTensor` release-or-quarantine), `include/iom/outstanding_work_registry.hpp:61-105` (`AllocatorCleanupAction`, one free via the `attempted_` guard), and the four drain sites listed under Current failure — the mechanisms the new sentences describe; audit-only, no edits.
- **Tests:** `test/cpu/test_cpu.cpp:27` (`RecordingAllocator`, whose `free` asserts every freed address was handed out) and the free-count cases at `:387-505` — the existing convention proving exactly-once free behavior. No test changes in this task: the behavioral quarantine-ordering regression that triggers a deferred `free` from device destruction belongs to the separate task `69-NT-003-test-destroy-while-queued-lifetime` (NT-003), which owns the always-run CPU regression for 49-ST-003's destroy-while-queued guarantee; CC-004 is documentation-only and does not add a CPU ordering unit test here. The deferred-free mechanism referenced by the amended wording is 49-ST-003's existing behavior, unchanged by this task.

## Requirements

1. `include/iom/cpu/device.hpp` states exactly: "The allocator is borrowed and must outlive the returned device and every resource created through it." as the final sentence of the `make_cpu_device` comment; the two preceding CPU-description sentences stay verbatim.
2. The four allocator-taking backend headers (`cpu`, `cuda`, `rocm`, `sycl`) then carry one identical lifetime sentence; no header other than the CPU one changes.
3. Parent spec §6 requires both orderings explicitly: `Device` outlives its tensors and queues, and the caller-supplied allocator outlives its `Device`.
4. Parent spec §7 keeps every existing ownership guarantee — null return throws `std::bad_alloc` without `free`, misaligned address freed once and rejected, construction-failure cleanup, no `alloc`/`free` from host transfers, device copies, compute submission, or view transforms, the non-throwing `free` obligation, and the `Allocator::reset()` invalidation rule — and adds only the deferral wording specified above.
5. The deferral wording names the trigger set exactly as 49-ST-003 implements it: a queued operation whose fence failed or was invalidated (failed/retained-failure op, or queue destroyed with unwaited work), storage released by the device destructor's quarantine drain.
6. No runtime file changes: registry, quarantine, cleanup actions, tensor/device destructors, and drain sites keep their current code; the spec text documents them, it does not redefine them.
7. `spec.md:623` (§11.4, "the documented device/allocator lifetime order") stays as-is; it refers to the amended §6–§7 text and needs no edit.
8. TTNN wording is unchanged everywhere: it owns native storage and takes no `Allocator&`, so the allocator rule does not apply to it.

## Non-goals

- Any runtime mechanism change: quarantine behavior, drain ordering, `AllocatorCleanupAction`, registry, or destructor code on any backend.
- Rewording the CUDA, ROCm, SYCL, or TTNN headers, `include/iom/device.hpp`, `include/iom/tensor.hpp`, or `include/iom/alloc.hpp`.
- Amending the applied 49-ST-003 change document — the review offered "spec §6–7 (or the 49-ST-003 change doc)" as alternatives; the living umbrella spec is the single source amended here.
- Implementing the destroy-while-queued CPU regression — that work is owned by the separate task `69-NT-003-test-destroy-while-queued-lifetime` (NT-003), not CC-004. ST-001/55-ST-001 SYCL hardening and ST-004 quarantine-failure policy unification are likewise excluded.
- Changing the `Allocator` interface, alignment contract, or `reset()` semantics.

## Acceptance criteria

- [ ] `include/iom/cpu/device.hpp` and the CUDA, ROCm, and SYCL headers each contain the identical sentence "The allocator is borrowed and must outlive the returned device and every resource created through it.", and the CPU header no longer contains "must outlive every tensor and queue created through them".
- [ ] `docs/changes/0001-tensor-view/spec.md` §6 states that the caller-supplied allocator must outlive its `Device`, and §7's free bullet states exactly-once free with the device-destruction deferral for quarantined storage plus the added drain sentence after the bullet list.
- [ ] All other §6–§7 guarantees enumerated in requirement 4 are byte-identical to their current text.
- [ ] The remediation diff touches exactly two files: `include/iom/cpu/device.hpp` and `docs/changes/0001-tensor-view/spec.md`; `git diff --stat` confirms no source, test, build, or other spec file changed.

## Verification

- `grep -n "outlive" include/iom/cpu/device.hpp include/iom/cuda/device.hpp include/iom/rocm/device.hpp include/iom/sycl/device.hpp` — each file's allocator sentence requires outliving "the returned device and every resource created through it"; the four allocator mentions are uniform.
- `grep -n "exactly once\|outlive its Device\|quarantine drain" docs/changes/0001-tensor-view/spec.md` — §6 line 424 and the §7 bullet/added sentence carry the new wording; no other §7 bullet changed.
- `grep -n "quarantine.drain\|quarantine_storage" src/cpu/device.cpp src/cuda/device.cpp src/rocm/device.cpp src/ttnn/device.cpp` — drain sites and destructor quarantine paths are unchanged (audit that documentation-only scope held).
- `git diff --stat` — exactly `include/iom/cpu/device.hpp` and `docs/changes/0001-tensor-view/spec.md` appear; no `.cpp`/`.cu`/`.hip`/test/CMake file listed.
- Optional non-hardware guard (comment-only change): configure and build the CPU-only target and run `ctest -R iom_cpu_tests` from the repository CMake workflow; all existing free-count and lifetime cases must pass unchanged. Behavioral deferred-free ordering is already owned by 49-ST-003's acceptance criteria and is not re-verified here.
