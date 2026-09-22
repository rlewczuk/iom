# ROCm pinned synthetic model-reference and causal-cache evidence

This page records the ROCm integration of the pinned, offline synthetic TinyLlama
reference corpus. The ROCm conformance driver owns device creation and lifetime;
the backend-neutral consumers in
`test/backend/backend_conformance_model_reference.hpp` and
`test/backend/backend_conformance_model_cache.hpp` own the frozen numerical and
cache assertions. No official checkpoint, network access, runtime weight
readback, or copied C++ reference arithmetic is part of these cases.

## Registered cases and frozen coverage

`iom_rocm_conformance_tests` registers these exact Doctest cases:

- `ROCm pinned model reference` calls
  `iom_conformance::run_model_reference_conformance(Device&)`. It exercises the
  two synthetic models at full-prompt lengths 1, 15, 16, and 17, including the
  one- and two-layer BF16 intermediate/final checkpoints and final logits.
- `ROCm pinned causal cache` calls
  `iom_conformance::run_model_cache_reference_conformance(Device&)`. It
  exercises non-tile GQA geometry, future-token perturbation invariance,
  nonzero cached decode from both prefixes, finite-value checks, and exact
  capacity admission/rejection.

Both cases use the driver-created ROCm `Device` and the committed corpus at
`test/model/synthetic_reference.json`, selected by the target-only
`IOM_MODEL_REFERENCE_DIR` definition. The shared consumers retain the frozen
per-value comparison policy (`abs(actual - reference) <= 0.05 + 0.02 *
abs(reference)`) and the recorded corpus identity; no backend-specific
relaxation or aggregate-only comparison is allowed.

The committed corpus file SHA-256 is
`e19efde2f9360159d244093f5a28c989e8a54a906f10f0a6bd26d51aad9c0db5`;
its recorded independent payload SHA-256 is
`d34876ad407a4955b644c8d747a3c658e9159ed96cf0bfda2d8760e7acfd4d4a`.

## Source map

- Driver registrations: `test/rocm/test_rocm_conformance.cpp`.
- Target data definition: `test/CMakeLists.txt`,
  `IOM_MODEL_REFERENCE_DIR` on `iom_rocm_conformance_tests` only.

## Verification receipt

The checks ran from the prepared worktree through the configured `rocm`
profile. The selected accelerator was ROCm ordinal 0, the `gfx1201` agent
(`AMD Radeon AI PRO R9700`, wavefront size 32; HSA node 1). The runtime
reported HSA version 1.21 (extension 1.30); `hipcc --version` reported HIP
7.15.26333-0000000 and AMD Clang 23.0.0git from commit
8f497e0992fb7513f7f78a6f6b6f1056c375e961 under `/opt/rocm/core-10.0`.
The committed corpus file SHA-256 was
`e19efde2f9360159d244093f5a28c989e8a54a906f10f0a6bd26d51aad9c0db5`; the
recorded independent payload SHA-256 is
`d34876ad407a4955b644c8d747a3c658e9159ed96cf0bfda2d8760e7acfd4d4a`.

The inventory command, after a fresh sync to the same mirror, was
`csw-remote-exec rocm rocm-synthetic-12-focused "flock -w 120 /tmp/agent-gpu0.lock timeout --kill-after=30s 900s sh -c 'hipcc --version; hipconfig --full; rocminfo; sha256sum test/model/synthetic_reference.json'"`;
the complete stdout/stderr is retained in `remote.log`; the command exited 0,
with the non-fatal diagnostic `sh: 1: /opt/rocm/core-10.0/lib/llvm/bin/llc:
not found` from `hipconfig --full`.

- Standard suite, mirror `rocm-synthetic-12-attempt1-rocm`: with
  `CSW_REMOTE_TASK_DIR=.cswd/tasks/006-tinyllama/14-integration-reference-validation/12-rocm-synthetic-integration`
  and the exact assigned `CSW_REMOTE_WORKSPACE`, ran
  `/home/rlew/iom/src/iom/.omp/csw/bin/test_rocm rocm-synthetic-12-attempt1`.
  Configure/build succeeded and CTest reported 4/4 tests passed:
  `iom_rocm_smoke_tests`, `iom_rocm_conformance_tests`,
  `iom_rocm_sdpa_nonmatrix_tests`, and
  `iom_backend_coexistence_tests`; cleanup removed the mirror.
- Focused setup, mirror `rocm-synthetic-12-focused`: after fresh sync, the
  bounded remote configure
  `timeout --kill-after=30s 1800s cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DIOM_TEST_REAL_MODEL_LOADING=OFF -DCUDA_ENABLED=OFF -DROCM_ENABLED=ON -DSYCL_ENABLED=OFF`
  and build
  `timeout --kill-after=30s 1800s cmake --build build --parallel "$(nproc)"`
  both exited 0 and built `iom_rocm_conformance_tests`.
- Focused named cases, after another fresh sync, ran
  `flock -w120 /tmp/agent-gpu0.lock timeout --kill-after=30s 900s ./build/test/iom_rocm_conformance_tests --test-case="ROCm pinned model reference,ROCm pinned causal cache"`.
  Doctest reported 2/2 cases and 2157/2157 assertions passed, 0 failed, with
  49 unrelated cases skipped; no finite-value failures were reported. The
  focused mirror was removed successfully after the run.
- The initial focused attempt on the fresh, unbuilt mirror is retained in
  `remote.log`: the specified executable path exited 127 with
  `timeout: failed to execute process: No such file or directory`. The
  bounded configure/build above repaired the missing remote build path; the
  exact focused command then passed. All owned remote processes and mirrors
  were cleaned up.

The all-backend `csw_verify` gate is owned by the integration verifier and is
intentionally not run by this leaf.

