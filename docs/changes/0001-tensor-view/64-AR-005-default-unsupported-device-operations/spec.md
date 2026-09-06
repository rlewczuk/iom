# Default unsupported compute operations in `DeviceOps` and delete the per-backend rejection stubs

**Order:** 64
**Priority:** P1 — required scalable capability rejection without backend boilerplate; keeps the incoming 0002 operation inventory from multiplying hand-written rejectors in every lagging backend.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-005`
**Review severity:** low
**Review verification:** verified, confidence 90

## Outcome

Calling any of the six compute operations — `add`, `mul`, `silu`, `linear`, `rmsnorm`, `sdpa` — on a queue that does not implement it throws `std::runtime_error` with the exact message `<LABEL> backend does not implement <op>` before submitting work, consuming a sequence, or writing an output. The rejection comes from one default `DeviceOps` body per operation backed by a single protected `backend_label()` hook implemented once per queue. All thirty hand-written backend rejectors (six methods x five backends) and the duplicate test-fake rejectors are deleted; `copy` remains pure virtual. Observable unsupported behavior on every backend is unchanged, and a backend that later implements an operation overrides exactly that method — which is what lets the ~35-operation 0002 inventory scale without per-backend rejector families.

## Current failure

- `DeviceOps` declares the six compute operations pure virtual (`include/iom/iom.hpp:233-253`), so every queue must spell out a rejector even when it implements none of them. The five implementations are identical except for the label literal:
  - `CpuQueue` — `src/cpu/device.cpp:804-828` (`throw unsupported("CPU", "<op>");`)
  - `CudaQueue` — `src/cuda/copy.cu:720-739`
  - `RocmQueue` — `src/rocm/copy.hip:724-743`
  - `SyclQueue` — `src/sycl/copy.cpp:354-380`
  - `TtnnQueue` — `src/ttnn/device.cpp:442-471`
- The message constructor already lives in the common base — `DeviceOps::unsupported(backend, operation)` at `include/iom/iom.hpp:287-293` builds `"<backend> backend does not implement <operation>"` — yet every queue re-supplies both strings by hand, ~130 lines in total.
- Test fakes repeat the same boilerplate with ad-hoc messages: `InlineQueue` (`test/test_iom.cpp:593-625` plus a private message helper at `631-634`), `DeferredCopyQueue` (`test/backend/backend_conformance_other.hpp:130-149`), and `InstrumentedQueue` (`test/backend/backend_conformance_other.hpp:206-225`). No test asserts any of those fake messages.
- Impact: pure noise today, growing with 0002. `docs/changes/0002-eltwise-binops/spec.md` defines ~35 elementwise operations with SYCL explicitly remaining blocked; without defaults that scales to ~35 hand-written rejectors per lagging backend per op family, and every rejection-message fix must be applied once per stub instead of once in the base.

## Scope

- Change exactly the six compute methods (`add`, `mul`, `silu`, `linear`, `rmsnorm`, `sdpa`) from pure virtual to virtual with default bodies that throw `unsupported(backend_label(), "<op>")`. `copy` stays pure virtual.
- Add one protected pure-virtual `backend_label()` hook to `DeviceOps`; each concrete queue and each test fake implements it exactly once.
- Delete the five backend stub blocks and the three test-fake rejector groups; keep `FakeQueue`'s six recording overrides — they genuinely implement queueing for sequence-allocation tests and are the review's "intentional overrides".
- Pin the label/message contract observably without breaking the hidden suite-level callsite: append a defaulted `std::string_view backend_label = {}` parameter to `run_compute_capability_conformance` (after the existing `observer`, so the suite-level `run_backend_conformance` call stays untouched) and run exact-message `CHECK_THROWS_WITH_AS` assertions only when `backend_label` is nonempty. When empty, the scenario falls back to the existing `CHECK_THROWS_AS` (category-only) behavior, preserving the current relative matcher contract.
- The observable rejection contract is preserved everywhere: category `std::runtime_error`, message format, throw-before-submission timing, zero sequence consumption, untouched outputs.

## Implementation references

- **Modify:** `include/iom/iom.hpp` — `DeviceOps` compute declarations at lines 233-253 and the protected section from line 255; default bodies replace `= 0`, and the new hook is declared beside the existing `unsupported` helper at 287-293.
- **Modify:** `src/cpu/device.cpp` — `CpuQueue` stub block at 804-828; delete it and add the label override.
- **Modify:** `src/cuda/copy.cu` — `CudaQueue` stub block at 720-739; same cutover.
- **Modify:** `src/rocm/copy.hip` — `RocmQueue` stub block at 724-743; same cutover.
- **Modify:** `src/sycl/copy.cpp` — `SyclQueue` stub block at 354-380; same cutover.
- **Modify:** `src/ttnn/device.cpp` — `TtnnQueue` stub block at 442-471; same cutover.
- **Modify:** `test/backend/backend_conformance_other.hpp` — `run_compute_capability_conformance` (lines 446-503) gains the label parameter and exact-message assertions; `DeferredCopyQueue` (stubs at 130-149) and `InstrumentedQueue` (stubs at 206-225) drop their stubs and add label overrides.
- **Modify:** `test/test_iom.cpp` — `InlineQueue` drops stubs 593-625 and the private helper at 631-634; `InlineQueue` (class at 541) and `FakeQueue` (class at 467) each add a label override; `FakeQueue`'s recording overrides (from line 501) stay unchanged.
- **Modify:** the five direct driver call sites — `test/cpu/test_cpu_conformance.cpp:272`, `test/cuda/test_cuda_conformance.cpp:356`, `test/rocm/test_rocm_conformance.cpp:503`, `test/sycl/test_sycl_conformance.cpp:349`, `test/ttnn/test_ttnn_conformance.cpp:504` — keep the existing first arguments and append the backend label literal as the new trailing argument (the only position available without disturbing the hidden suite callsite). The TTNN direct call is currently two-argument (no observer) and becomes three-argument; CPU/CUDA/ROCm/SYCL each currently pass an observer last and become four-argument.
- **Read:** `test/cpu/test_cpu.cpp:1005-1042` — "CPU compute operations reject capability without submitting" already asserts type-level rejection, untouched storage, and zero sequence consumption; it remains valid unchanged and needs no edit.
- **Read:** `docs/changes/0002-eltwise-binops/spec.md` — the operation inventory that motivates defaults; no 0002 operation is implemented or declared here.

## Requirements

- In `include/iom/iom.hpp`, replace `= 0` with an inline default body on each of `add`, `mul`, `silu`, `linear`, `rmsnorm`, `sdpa`, keeping each method's doc comment and signature byte-identical (parameter names and types are public API):
  ```cpp
  /** Addition: c = a + b **/
  virtual oid add(const TensorView& a, const TensorView& b, TensorView& c) {
      throw unsupported(backend_label(), "add");
  }
  ```
  The op-name string in each body is exactly the lowercase method name used today: `"add"`, `"mul"`, `"silu"`, `"linear"`, `"rmsnorm"`, `"sdpa"`.
- Add one hook to the protected section adjacent to `unsupported`:
  ```cpp
  [[nodiscard]] virtual std::string_view backend_label() const noexcept = 0;
  ```
  It must be pure virtual: the compiler, not a source audit, enforces "implemented once per queue". `std::string_view` is already included (`<string_view>` at `include/iom/iom.hpp:13`).
- Each queue implements the hook once, returning a static literal exactly matching today's label: `CpuQueue` -> `"CPU"`, `CudaQueue` -> `"CUDA"`, `RocmQueue` -> `"ROCm"`, `SyclQueue` -> `"SYCL"`, `TtnnQueue` -> `"TTNN"`. With these literals the default bodies reproduce every current message byte-for-byte (for example `"CPU backend does not implement add"` and `"TTNN backend does not implement sdpa"`).
- Delete all six stub overrides in each of the five backend queues. After the cutover no file under `src/` declares or overrides any of the six compute methods.
- Test fakes: `InlineQueue`, `DeferredCopyQueue`, and `InstrumentedQueue` delete their rejector overrides and private message helpers and inherit the defaults; each implements `backend_label()` returning a static literal (`"inline"`, `"deferred"`, `"instrumented"` — these labels are test-internal and unasserted). `FakeQueue` keeps its six recording overrides unchanged and adds its own label override (for example `"fake"`).
- Extend `run_compute_capability_conformance` by appending one trailing parameter `std::string_view backend_label = {}` after `observer` (the second-existing-parameter position is unavailable because `run_backend_conformance` at `test/backend/backend_conformance_other.hpp:517` invokes `run_compute_capability_conformance` with the two-argument `(candidate, supported_types, observer)` shape; adding a required parameter there would break the suite callsite). Branch on `backend_label.empty()`:
  - **Empty (default, suite-driven path):** keep every existing `CHECK_THROWS_AS` assertion exactly as today. This preserves the relative `THROWS_AS` matcher behavior and means `run_backend_conformance` callers see no change.
  - **Nonempty (direct driver path):** replace each of the six primary `CHECK_THROWS_AS` lines with the stricter exact-message form below. The two transformed-operand checks remain `CHECK_THROWS_AS` (type-level — no message string is built for those). The post-failure probe (`token_sequence(probe) == 1`) and output-bytes assertions are unchanged either way.
- Exact-message assertion form (selected as the stricter behavior the empty-label path opts out of):
  ```cpp
  const std::string expected =
          std::string(backend_label) + " backend does not implement add";
  CHECK_THROWS_WITH_AS(queue->add(x->view(), x->view(), y->view()),
                       expected.c_str(), std::runtime_error);
  ```
  This doctest build is 2.4.12: `THROWS_WITH` matchers bind to `doctest::String`/`doctest::Contains`, so pass `expected.c_str()`, never a bare `std::string` (it does not compile). The exact-message path is explicit: it replaces the relative `THROWS_AS` matcher with `THROWS_WITH_AS` for those six calls; no `THROWS_WITH` (string-only) variant is used anywhere in this change, and the previous relative-matcher behavior is preserved by the empty-label branch.
- Update the five driver call sites to pass `"CPU"`, `"CUDA"`, `"ROCm"`, `"SYCL"`, `"TTNN"` respectively.
- The hook is `const` and `noexcept`, returns a view of static-literal storage, and performs no locking, allocation, or backend-state access. Default bodies throw before `submit` is called, so no sequence is consumed and no output is written — the property `run_compute_capability_conformance` and `test/cpu/test_cpu.cpp:1005` both pin.
- Add one class-level sentence to the `DeviceOps` doc comment (lines 202-211) stating that compute operations default to throwing `unsupported(backend_label(), op)` and that backends override only implemented operations. Do not duplicate that sentence in each per-operation comment.

## Non-goals

- No default body for `copy`: every backend implements it, and a defaulting copy would silently weaken the queue contract.
- No 0002 operation is implemented or declared, and no per-operation capability/dtype matrix is introduced; capability validation beyond "backend does not implement" belongs to the operation changes.
- No behavioral change to rejection: same category, message text, timing, sequence consumption, and storage guarantees.
- No conversion of `backend_label()` into a public accessor, enum, dispatch key, or device-level property; common code must not switch on backend identity (AGENTS.md architectural invariants).
- No changes to queue-id leasing, submission sequences, wait tokens, `StagedWorker`, the outstanding-work registry, or any copy path.
- No SYCL compute enablement and no change to SYCL's blocked status for 0002; the SYCL architecture convergence is AR-001's task.
- No rework of `test_iom.cpp`'s common `DeviceOps` machinery tests beyond the fixture cutover listed above.

## Acceptance criteria
- [ ] On CUDA, ROCm, SYCL, and TTNN hardware, `run_compute_capability_conformance` passes with each backend's exact label in every message and still proves no sequence consumption (`token_sequence(probe) == 1`) and untouched outputs.
- [ ] `run_backend_conformance` (the suite-level driver) calls `run_compute_capability_conformance` with no extra argument, takes the empty-label branch, and runs identically to today — every shared suite case still passes on every backend without any callsite edit.
- [ ] On CPU (local), each of the six compute calls on a `CpuQueue` throws `std::runtime_error` with the exact message `"CPU backend does not implement <op>"`, proven by the extended conformance case; the pre-existing `test/cpu/test_cpu.cpp:1005` case still passes unchanged, including zero sequence consumption and untouched storage.
- [ ] Source audit: `grep -rn "throw unsupported" src/ test/` returns zero matches — the only throw sites are the six default bodies in `include/iom/iom.hpp`.
- [ ] Source audit: `grep -rnE "oid (add|mul|silu|linear|rmsnorm|sdpa)\(" src/` returns zero matches; the only tree-wide overrides of the six methods are `FakeQueue`'s six recording stubs in `test/test_iom.cpp`.
- [ ] A queue implementing none of the compute operations compiles with only `copy` plus `backend_label()` — demonstrated by the converted `InlineQueue`/`DeferredCopyQueue`/`InstrumentedQueue` fixtures.
- [ ] `iom_tests` and `iom_cpu_tests` pass, and all five backend smoke and conformance targets pass on their supported runtimes.

## Verification

- CPU-only build and tests via the repository CMake/CTest workflow (GCC, Release): `cmake -S . -B build -DCPU_ENABLED=ON && cmake --build build -j && ctest --test-dir build --output-on-failure -R "iom_tests|iom_cpu_tests"`. The `iom_cpu_bench` gate is excluded because its absolute threshold requires the NT-001 re-baselining, which is out of scope here and reported separately.
- Hardware backends follow `.agents/skills/remote-development`. Build and run the focused targets `iom_cuda_conformance_tests`, `iom_rocm_conformance_tests`, `iom_sycl_conformance_tests`, and `iom_ttnn_conformance_tests` (plus the per-backend smoke targets) under exclusive accelerator access. The capability case must pass with the exact `CHECK_THROWS_WITH_AS` messages for each backend's label — this is the hardware-level proof that the labels and messages survived the cutover.
- Static audits (run anywhere, no build needed):
  - `grep -rn "throw unsupported" src/ test/` — expect zero matches.
  - `grep -rnE "oid (add|mul|silu|linear|rmsnorm|sdpa)\(" src/` — expect zero matches.
  - `grep -rn "backend_label" src/ include/iom/iom.hpp` — expect exactly the pure-virtual declaration and its default-body call sites in the header plus one short definition per queue implementation file.
