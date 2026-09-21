# RMS normalization

This is the operation-owned contract for `DeviceOps::rmsnorm`. The common
facade, its admission rules, and its pure requirement query are declared and
frozen here. The CPU port queues all nine applicable floating leaves at the
exact `{0, 1}` zero-workspace requirement, CUDA and ROCm launch the shared
tiled core on their existing nonblocking streams, and SYCL queues its native
row kernel for the eight non-`F64` leaves and for `F64` exactly where the
device reports `aspect::fp64`. Each port owns its own capability predicate, and
an unsupported or limited leaf stays `Unsupported` exactly as section 9 states
above rather than counting as numerical conformance. The exact ABI is:

```cpp
oid rmsnorm(const TensorView& x, const TensorView& scale, TensorView& out,
            float eps, RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements rmsnorm_workspace_requirements(
        const TensorView& x, const TensorView& scale,
        const TensorView& out, float eps);
```

The redundant `dim` argument is removed from declarations, definitions,
callers, tests, and documentation in one clean cutover; no overload, alias,
compatibility shim, or re-export remains. The feature extent is the
operation's `F`, and no operation-specific host span, implicit conversion,
integer norm, quantization, or storage-format staging is introduced.

1. **Layout.** `x` and `out` have identical logical shape `[...,R,F]` with
   rank two through eight and nonzero extents. `scale` is exactly rank-two
   `[1,F]`, shared explicitly across every independent leading plane and row;
   RMSNorm MUST NOT accept a rank-one scale or introduce a general
   hidden-state or leading-plane broadcast. Every plane and row is
   independent: the reduction covers exactly the `F` logical features of its
   own row, and tiled padding, other rows, other planes, and other requests
   MUST NOT contribute. Selected plane offsets and strides are honored, and
   final-axis transforms or shape inflation are rejected. All three views use
   the same applicable leaf type and `QuantizationFormat::NONE`, output
   storage is disjoint from both inputs, and read/read overlap between `x` and
   `scale` remains valid.
2. **Arithmetic.** For every leading-plane index `b`, row `r`, and feature
   `f`, RMSNorm is row-local and evaluates
   `out[b,r,f] = x[b,r,f] * rsqrt(sum(i=0..F-1, x[b,r,i] * x[b,r,i]) / F + eps)
   * scale[0,f]` in the direct accumulator domain of its leaf, never against
   an unbounded-real oracle. `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`,
   `F8_E5M2`, `F16`, `BF16`, and `F32` decode to FP32, and every square,
   product, and reduction step is FP32, with FP32 overflow/underflow and
   permitted reduction reassociation within the comparison policy below.
   `F64` stays FP64 throughout. The mean, epsilon addition, reciprocal square
   root, normalization, and scale multiplication use that accumulator, and
   the result is encoded exactly once to the output leaf using
   round-to-nearest, ties-to-even together with the existing named-format
   special-value and saturation rules.
3. **Special values and epsilon.** `eps` MUST be finite and nonnegative.
   `eps == 0` on an all-zero row produces quiet NaNs with no NaN payload
   promise. NaN in `x` makes the shared row reduction NaN and poisons that
   row. Infinite `x` makes the reduction infinite: finite features normalize
   to signed zero, and infinite features become quiet NaN before the scale
   multiply. A nonfinite `scale` affects only its own feature after the shared
   norm. A NaN result — including the all-zero-row decision — is stored in its
   canonical positive form: the sign bit of a NaN result is cleared before the
   single destination encode, so a leaf without a NaN encoding saturates to
   its positive maximum and no device's invalid-operation NaN sign can change
   a stored result. Admission MUST NOT scan for nonfinite data, and no
   RMS-specific queued data failure is added.
4. **Capability matrix.** Applicability is exactly the nine ordinary signed
   floating leaves; the remaining fourteen leaves have no integer, boolean,
   or exponent-only normalization contract. The matrix below is the complete
   leaf-by-backend capability record.

   | Leaf | Contract | CPU | CUDA | ROCm | SYCL |
   | --- | --- | --- | --- | --- | --- |
   | `BOOL` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I2`, `U2` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I4`, `U4` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I8`, `U8` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I16`, `U16` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I32`, `U32` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I64`, `U64` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `F4_E2M1` | applicable | supported | supported | supported | supported |
   | `F6_E2M3` | applicable | supported | supported | supported | supported |
   | `F6_E3M2` | applicable | supported | supported | supported | supported |
   | `F8_E4M3FN` | applicable | supported | supported | supported | supported |
   | `F8_E5M2` | applicable | supported | supported | supported | supported |
   | `F8_E8M0` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `F16` | applicable | supported | supported | supported | supported |
   | `BF16` | applicable | supported | supported | supported | supported |
   | `F32` | applicable | supported | supported | supported | supported |
   | `F64` | applicable | supported | supported | supported | `aspect::fp64` only |

CPU, CUDA, and ROCm support all nine applicable floating leaves. SYCL supports
the eight non-`F64` leaves and supports `F64` only when the device reports
`aspect::fp64`; otherwise `F64` is `Unsupported`. On every backend, an unknown
dtype enumeration value is `InvalidArgument`, while a recognized inapplicable
or unsupported leaf and a recognized non-`NONE` quantization format are
`Unsupported`.
5. **Admission order.** Before capability dispatch or any queue effect, a
   submission validates in exactly this order: (1) rank, nonzero extents,
   identical `[...,R,F]` shape, and `scale == [1,F]`; (2) exact queue `Device`
   identity for every view, stable live owner registration and native handle,
   selected plane and leading bounds, and stride and view metadata; (3)
   checked element, byte, address, stride, plane, feature, and tile
   arithmetic, rejecting overflow before narrowing or pointer calculation;
   (4) exact same leaf type and `QuantizationFormat::NONE`, output
   disjointness, and conservative output/input alias rejection; (5) finite
   nonnegative epsilon; (6) immutable backend capability; and (7) supplied
   workspace validation and lease. This is the generic facade order of
   *TinyLlama forward layout — Workspace and execution* above, split so that
   leaf applicability and conservative aliasing are host-checkable admission,
   epsilon precedes immutable capability, and supplied-workspace admission
   follows capability and therefore never masks an unported backend. Read/read
   `x`/`scale` aliases are allowed, but any output alias is rejected even when
   transformed windows appear disjoint, and no data is inspected to admit a
   request.
6. **Requirement query and workspace.** The throwing query is pure and
   deterministic: it allocates no host or native metadata, constructs no
   request or vector snapshot, registers or leases no owner, mutates no
   queue, token, or state, reads no data, and submits nothing, so it stays
   pure while the queue is occupied. Its result depends only on the validated
   views, `eps`, and immutable capability, and every supported implementation
   returns exactly `{0, 1}` and consumes no `RawWorkspace`. Because the
   requirement is `{0, 1}`, only the empty `RawWorkspaceView{}` is admissible;
   any supplied workspace with an owner is `InvalidArgument` before dispatch,
   with no hidden allocation, relocation, host arithmetic, or host roundtrip.
   Unsupported capability and malformed query inputs surface as the
   established throwing exceptions instead of OID mapping.
7. **Request snapshot and ownership.** Successful admission constructs one
   immutable `RmsnormRequest` that owns value-copied metadata for `x`,
   `scale`, and `out` (shape, leaf type, quantization, plane offset, plane
   strides, and validated bounds), their exact live owner and native-handle
   identities, the validated epsilon, and the admitted `{0, 1}` requirement.
   No borrowed `TensorView` or caller-owned metadata is retained, and
   snapshot values do not change when caller views or their backing metadata
   are mutated or destroyed. Read/read owner identities are deduplicated;
   output storage stays disjoint. Every required owner is registered and
   retained through proven in-order completion by the existing
   prepare/register/dispatch/rollback, OID sequencing, completion-release,
   and quarantine machinery — this contract adds no second registry or
   workspace framework. Temporary caller views may die immediately after the
   call returns.
8. **Errors and default hooks.** Host-checkable shape, metadata, device,
   alias, dtype, epsilon, workspace, and overflow failures are admission
   failures: the `noexcept` facade maps them to the established negative OIDs
   without consuming a token, registering an owner, submitting work, or
   mutating queue or output state, and successful accepted work returns a
   positive token. A runtime failure after acceptance leaves output unusable,
   and every wait for that token repeats the same failure. The default common
   RMSNorm hooks keep a well-formed request `Unsupported`, so a valid-shape
   request on an unported backend reaches explicit capability rejection
   before workspace inspection or execution and is never counted as
   successful conformance. A backend port replaces only its own capability
   predicate and implementation.
9. **Comparison policy.** The conformance policy is fixed before measurement:
   special-value class and signed-zero checks are exact while NaN payloads
   are ignored; finite `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`,
   `F8_E5M2`, `F16`, and `BF16` outputs must lie within two adjacent
   destination encodings of the reference; `F32` requires
   `abs_err <= 1e-6 + 2e-5 * abs(reference)`; and `F64` requires
   `abs_err <= 1e-15 + 1e-12 * abs(reference)`. The independent reference
   rounds every FP32 intermediate and uses `double` for `F64`; production
   code MUST NOT serve as its own oracle.

**Shared CUDA/ROCm queue and tiled-kernel core.** The two accelerator
backends share one queue branch and one tiled device operation.
`src/shared/gpu_queue.hpp` carries the immutable RMSNorm task and completion
record beside the copy and binary variants, and
`src/shared/gpu_queue_operations.inl` submits, dispatches, and completes it on
the same fixed partition as every accelerator copy: the exactly `C` eagerly
created completion resources, the fixed 512-byte metadata slots, the
preallocated worker, the common owner registration output, and the `{0, 1}`
zero-workspace requirement are reused with no queue growth, per-call
allocation, second stream, or submission-side wait, and all terminal paths
go through the existing completion release and quarantine rules. Immutable
request scalars, view snapshots, owner registrations, the fixed metadata
lease, and the event-ring submission are captured by value before the worker
runs; no borrowed view, workspace lease, or metadata slot outlives proven
completion. `src/shared/standard_tiled_rmsnorm.inl` is the complete
backend-parameterized CUDA/HIP device kernel and device codec: independent
leading planes and rows, a logical-`F`-only traversal that never reads or
writes tile padding, the frozen FP32 accumulator (FP64 for `F64`) with the
special-value and signed-zero rules above, and exactly one
round-to-nearest-even output encode with no host decode, hidden transfer, or
hidden allocation. Each backend contributes only its own `gpu_policy` launcher,
which must use the queue's already-created nonblocking stream. The CUDA
launcher in `src/cuda/copy.cu` calls the shared
`launch_standard_tiled_rmsnorm` kernel directly on that stream, while
`src/cuda/copy.hpp` advertises all nine applicable signed floating leaves.
The CUDA, ROCm, and SYCL runtime identities and focused gate results are
recorded in the retained-backend closure evidence below. The shared CUDA/ROCm
core and the SYCL row kernel are exercised through their real queues; SYCL
`F64` remains conditional on `aspect::fp64`.

**ROCm capability and implementation evidence.** `src/rocm/copy.hpp` exposes
the ROCm policy's immutable RMSNorm capability for the nine applicable
floating leaves. `src/rocm/copy.hip` invokes the shared
`launch_standard_tiled_rmsnorm` HIP kernel on the queue's existing
nonblocking stream, preserving the device-local logical-`F` reduction,
FP32/FP64 accumulator domains, one destination encode, and zero-workspace
queue protocol.
 
**Retained-backend closure evidence.** The four focused gates were run from the
prepared closure worktree on branch
`run-task/006-tinyllama--05-rms-normalization--11-all-backend-rmsnorm-closure`,
based at revision `e6d834726bc80d8d8786ace6a965dd5c9c2c380e`. Accelerator
commands used the configured remote profiles and a fresh sync immediately
before each execution; the complete command transcripts are retained in the
task evidence file
`.cswd/tasks/006-tinyllama/05-rms-normalization/11-all-backend-rmsnorm-closure/remote.log`.
The accelerator mirrors were `impl11closure-cuda`, `impl11closure-rocm`, and
`impl11closure-sycl`. Accelerator executable and CTest rows used
`flock -w 120 /tmp/iom-<backend>-gpu.lock timeout --kill-after=30s 900s`
around the listed command; SYCL additionally sourced
`/opt/intel/oneapi/setvars.sh` and ran the full-path `sycl-ls` inventory
before each remote command.

| Backend and runtime identity | Focused gate evidence |
| --- | --- |
| CPU: local `x86_64`, AMD Ryzen AI 9 HX 370 w/ Radeon 890M | `cmake --build build --target iom_backend_conformance_cpu_tests iom_tests iom_cpu_tests -j2`; `ctest --test-dir build --output-on-failure --timeout 300 -R '^(iom_backend_conformance_cpu_tests\|iom_tests\|iom_cpu_tests)$'` — all 3 tests passed. |
| CUDA profile `bv1`: NVIDIA GeForce RTX 5090, driver `595.71.05`, CUDA `13.2`, `nvcc` `V13.2.78` | `./build/test/iom_cuda_conformance_tests --test-case=*RMSNorm*` — 1 test, 2054 assertions passed; `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_cuda_conformance_tests$'` — 1/1 passed. |
| ROCm profile `bv2`: AMD Radeon AI PRO R9700 `gfx1201` (with `gfx1036` enumerated), HIP `7.15.26333` | `./build/test/iom_rocm_conformance_tests --test-case=*RMSNorm*` — 1 test, 2053 assertions passed; `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_rocm_conformance_tests$'` — 1/1 passed. |
| SYCL profile `bv2`: two Intel Arc Pro B60 Level Zero GPUs, oneAPI compiler `2026.1.0` | `./build/test/iom_sycl_conformance_tests --test-case=*RMSNorm*` after `sycl-ls` — 1 test, 1734 assertions passed; `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_sycl_conformance_tests$'` — 1/1 passed. The enumerated device has `aspect::fp64`, so the conditional `F64` leaf ran. |

The automated driver links are [`test/cpu/test_cpu_conformance.cpp`](../../test/cpu/test_cpu_conformance.cpp),
[`test/cuda/test_cuda_conformance.cpp`](../../test/cuda/test_cuda_conformance.cpp),
[`test/rocm/test_rocm_conformance.cpp`](../../test/rocm/test_rocm_conformance.cpp),
and [`test/sycl/test_sycl_conformance.cpp`](../../test/sycl/test_sycl_conformance.cpp).

The exact target links are `iom_backend_conformance_cpu_tests`,
`iom_cuda_conformance_tests`, `iom_rocm_conformance_tests`, and
`iom_sycl_conformance_tests`. The matrix does not count unsupported probes as
contract-exact numerical or accepted-failure coverage.


The common owner is `src/device_ops_rmsnorm.cpp` behind
`include/iom/iom.hpp`. `test/test_iom.cpp` owns signature/cutover, query
purity, validation precedence, rejection-effect, snapshot-lifetime,
registration-lifetime, workspace, and repeat-wait coverage.
`test/backend/backend_conformance_rmsnorm.hpp` is the single shared owner of
every observable conformance scenario — the independent oracle's numeric
matrix for each declared leaf, the rank-two-through-eight and boundary
geometries, padded-physical invariance, and the common query, workspace,
admission, alias, overflow, capability, ownership, ordering, and retained
failure cases — and the CPU, CUDA, ROCm, and SYCL conformance drivers each
invoke its one dispatcher with their own device setup and declared supported
leaf span. The four focused target links are
`iom_backend_conformance_cpu_tests` (`test/cpu/test_cpu_conformance.cpp`),
`iom_cuda_conformance_tests` (`test/cuda/test_cuda_conformance.cpp`),
`iom_rocm_conformance_tests` (`test/rocm/test_rocm_conformance.cpp`), and
`iom_sycl_conformance_tests` (`test/sycl/test_sycl_conformance.cpp`).
`test/cpu/test_cpu.cpp` retains CPU-local RMSNorm probes, while
`test/backend/backend_conformance_other.hpp` keeps unrelated neural
capability probes without a second RMSNorm suite. Backend kernels and their
backend-specific launch code remain owned by their own ports.
