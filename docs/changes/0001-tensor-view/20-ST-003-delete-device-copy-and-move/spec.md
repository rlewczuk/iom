# Delete copy and move operations on Device to enforce owner address stability

**Order:** 20
**Priority:** P0 — enforces the public ownership invariant (owners non-copyable/non-movable) that tensor and queue address stability depends on
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-003`
**Review severity:** medium
**Review verification:** verified, confidence 98

## Outcome

`iom::Device` publicly deletes copy construction, copy assignment, move construction, and move assignment, so every concrete subclass — the CPU/CUDA/ROCm/TTNN factory devices and any user-defined device — is non-copyable and non-movable by construction. Compile-time trait assertions in `test/test_iom.cpp` prove the invariant for the concrete custom-device fixture `FakeDevice` and keep the existing assertions for `Tensor` and `DeviceOps` passing. All backend factories and all existing tensor/view lifetime tests compile and pass unchanged.

## Current failure

Repository ownership rules state that owners are non-copyable and non-movable and a creating `Device` must outlive its tensors and queues (`AGENTS.md`, "Architectural Invariants", line 16). `Tensor` and every view retain the address of the creating device (`Tensor` stores `Device* device_`, `include/iom/tensor.hpp:220`), and asynchronous queues hand that address to submitted work; those pointers are never retargeted.

`class Device` (`include/iom/device.hpp:20-30`) declares only a virtual destructor and four virtual operations; it leaves implicit copy/move construction and assignment in place. A concrete subclass with no non-copyable member is therefore copy- and move-constructible today: the repository's own test fixture `FakeDevice` (`test/test_iom.cpp:561-581`, stateless) satisfies `std::is_copy_constructible_v` / `std::is_move_constructible_v` (and the assignable traits) as true. By contrast, `Tensor` (`include/iom/tensor.hpp:201-204`) and `DeviceOps` (`include/iom/iom.hpp:30-33`) explicitly delete all four operations.

Failure mode: moving such a device does not retarget existing tensors' stored `Device*` — destroying the moved-from object leaves every tensor and view it created dangling. Copying a subclass that owns a raw runtime handle (for example `CudaDevice`'s `CUdevice`/`CUcontext` pair, `src/cuda/device.cpp:86-88`) duplicates ownership and teardown. These are exactly the relocation/duplication hazards the stable-address ownership model exists to prevent; failures surface as dangling device access or double release of a runtime context.

## Scope

- Public ownership surface of `iom::Device` only: four deleted special member functions in `include/iom/device.hpp`, plus a comment sentence stating the owner contract.
- Compile-time regression assertions for the concrete device fixture in `test/test_iom.cpp`.
- Backend scope (review): common — the change affects every `Device` subclass in all backends at compile time only; no runtime behavior changes anywhere.
- Compatibility: every existing construction must keep compiling without source changes — `make_cpu_device` (`src/cpu/device.cpp:453`), `make_cuda_device` (`src/cuda/device.cpp:165`), `make_rocm_device` (`src/rocm/device.cpp:152`), `make_ttnn_device` (`src/ttnn/device.cpp:404`) all construct via `std::make_unique<...Device>`, direct construction of `FakeDevice` on the stack, and every `create_tensor`/`create_ops` path. No in-tree code copies, moves, or passes a `Device` by value: devices are held as `std::unique_ptr<Device>`, `Device&`, or `Device*` (e.g. `ConformanceDevices` holds three references, `test/backend/backend_conformance_common.hpp:185-193`). Code that copy/moves a `Device` was already contract-invalid and does not exist in the tree.

## Implementation references

- **Modify:** `include/iom/device.hpp` — `class Device` (lines 20–30); the touchpoint: add the four deleted copy/move declarations to the public section and one comment sentence on the ownership contract.
- **Read:** `include/iom/tensor.hpp` — `class Tensor` (lines 198–204); the established four-deletion pattern and its comment wording ("Non-copyable and non-movable so the owner and full-view addresses stay valid for asynchronous operations") to mirror.
- **Read:** `include/iom/iom.hpp` — `class DeviceOps` (lines 28–33); the same pattern applied to the queue owner.
- **Tests:** `test/test_iom.cpp` — `FakeDevice` fixture (lines 561–581); the existing owner-trait assertion blocks to extend and keep green: `Tensor` traits in `TEST_CASE("Tensor owner constructs through a device with a stable full view")` (lines 737–747, including `static_assert(std::is_abstract_v<iom::Device>)` at line 743) and `DeviceOps` traits in `TEST_CASE("DeviceOps view signatures are exact and view-only")` (lines 1279–1283).
- **Read:** `src/cpu/device.cpp:136-155` (`CpuDevice`, member `Allocator& allocator_`), `src/cuda/device.cpp:44-92` (`CudaDevice`, already deletes copy construction/assignment at lines 51–52, owns raw `CUdevice`/`CUcontext`), `src/rocm/device.cpp:42-48` (`RocmDevice`, deletes copy construction/assignment at lines 47–48), `src/ttnn/device.cpp:82-92` (`TtnnDevice`, deletes copy construction/assignment at lines 89–90, owns `std::shared_ptr<ttnn::MeshDevice>`); these are all concrete `Device` subclasses in the tree and must compile unchanged.

## Requirements

- In `include/iom/device.hpp`, declare in the public section of `Device`, in this order after the defaulted virtual destructor (mirroring `Tensor`'s in-class layout):
  - `Device(const Device&) = delete;`
  - `Device& operator=(const Device&) = delete;`
  - `Device(Device&&) = delete;`
  - `Device& operator=(Device&&) = delete;`
- Make no other change to `Device`: keep `virtual ~Device() = default;` and the four virtual operations (`backend_kind`, `backend_device`, `create_tensor`, `create_ops`) exactly as they are, and do not add, remove, or protect any constructor — the implicit default constructor must remain so subclass constructors are unaffected.
- Extend the class documentation comment with one sentence stating that `Device` is non-copyable and non-movable so the owner address stays stable for the tensors and queues it created (mirror the `Tensor` comment at `include/iom/tensor.hpp:192-197`).
- In `test/test_iom.cpp`, add a dedicated `TEST_CASE` named `"Device owners are non-copyable and non-movable"` placed with the other owner-construction tests (immediately after `TEST_CASE("Tensor owner constructs through a device with a stable full view")`, which ends at line 780), containing exactly the established static_assert style:
  - `static_assert(!std::is_copy_constructible_v<FakeDevice>);`
  - `static_assert(!std::is_copy_assignable_v<FakeDevice>);`
  - `static_assert(!std::is_move_constructible_v<FakeDevice>);`
  - `static_assert(!std::is_move_assignable_v<FakeDevice>);`
  - These four assertions must fail to compile against the current (unfixed) `device.hpp`, since all four traits are true for the stateless `FakeDevice` today.
- Do not add trait assertions on `iom::Device` itself beyond the existing `static_assert(std::is_abstract_v<iom::Device>)` at line 743: copy/move traits of the abstract base are already false vacuously, so the concrete fixture is the meaningful probe. Do not modify the existing `Tensor`, `TensorView`, or `DeviceOps` trait assertions (lines 738–747, 1279–1283); they must keep passing unchanged.
- Change no backend source file: the existing per-subclass `= delete` copy declarations in `CudaDevice`, `RocmDevice`, and `TtnnDevice` become redundant but remain correct and stay as they are.
- The remediation is compilation-visible only: no runtime path, error category, ABI-facing factory signature, or test executable behavior other than the new assertions may change.

## Non-goals

- No changes to the `Tensor`, `TensorView`, or `DeviceOps` contracts — they already delete copy/move as required.
- No removal of the now-redundant copy-deletion declarations inside `CudaDevice`/`RocmDevice`/`TtnnDevice`; they are correct and touching backend sources is outside this finding.
- No new `swap`, `clone`, or transfer-of-ownership API on `Device`, and no registry/manager layer.
- No SYCL backend work and no work on other findings (for example ST-004's CUDA primary-context retain leak).

## Acceptance criteria

- [ ] `class Device` in `include/iom/device.hpp` publicly deletes copy construction, copy assignment, move construction, and move assignment, with the virtual destructor and virtual operations otherwise unchanged.
- [ ] `test/test_iom.cpp` contains the four negative trait assertions for `FakeDevice` in a `TEST_CASE`, and target `iom_tests` compiles and passes with them.
- [ ] With only the `include/iom/device.hpp` change reverted (test assertions kept), building target `iom_tests` fails to compile at the new `static_assert`s — proving the assertions defend the reviewed bug — and re-applying the header change restores a clean build.
- [ ] The pre-existing trait assertions for `Tensor`, `TensorView`, and `DeviceOps` (lines 738–747 and 1279–1283) are unmodified and still pass.
- [ ] All four factories and every existing tensor/view lifetime test compile and pass unchanged on CPU (`iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`).
- [ ] The CUDA, ROCm, and TTNN backend libraries and their hardware tests compile and pass unchanged on the remote accelerator hosts, with no skips.

## Verification

- CPU proof (local):
  ```bash
  cmake -S . -B build
  cmake --build build -j
  ctest --test-dir build -R '^iom_tests$|^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$' --output-on-failure
  ```
  Expected: configure/build succeed; all three targets pass.
- Negative compile control (local), after the full change is in place:
  ```bash
  git stash push -- include/iom/device.hpp   # remove only the four deletions
  cmake --build build --target iom_tests     # must FAIL at the new static_asserts
  git stash pop
  cmake --build build --target iom_tests     # must succeed again
  ctest --test-dir build -R '^iom_tests$' --output-on-failure
  ```
- Accelerator proof (all backends that own a `Device` subclass): follow the repository's remote-development procedure (`.agents/skills/remote-development` — sync the workspace to a unique remote task directory with `scripts/remote-sync`, run builds/tests only via `scripts/remote-exec` on the selected host). CPU-only evidence does not cover the CUDA/ROCm/TTNN factory compilations this finding's base-class change reaches.
  ```bash
  .agents/skills/remote-development/scripts/remote-sync <host> <task-id>
  .agents/skills/remote-development/scripts/remote-exec <host> <task-id> \
      'cmake -S . -B build -DCUDA_ENABLED=ON -DROCM_ENABLED=ON -DTTNN_ENABLED=ON && cmake --build build -j'
  .agents/skills/remote-development/scripts/remote-exec <host> <task-id> \
      "ctest --test-dir build -R '^iom_cuda_smoke_tests$|^iom_cuda_conformance_tests$|^iom_rocm_smoke_tests$|^iom_rocm_conformance_tests$|^iom_ttnn_smoke_tests$|^iom_ttnn_conformance_tests$|^iom_backend_coexistence_tests$' --output-on-failure"
  ```
  Enable each backend on a host whose SDK it requires if no single host has all three; every enabled backend must have its library (`iom_cuda`, `iom_rocm`, `iom_ttnn`) compile — each contains a concrete `Device` subclass instantiation — and its smoke/conformance tests plus `iom_backend_coexistence_tests` must pass with no skips.
