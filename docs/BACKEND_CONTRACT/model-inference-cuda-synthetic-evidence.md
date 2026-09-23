# CUDA pinned synthetic model-reference evidence

This page records the CUDA execution evidence for the two driver-owned synthetic
model cases. The cases use the shared complete-model checkpoint and causal-cache
consumers; they do not add a CUDA copy of the numerical or cache algorithms and
do not require an official checkpoint, model-artifact environment, or network.

## Registered cases and frozen coverage

`iom_cuda_conformance_tests` registers these exact Doctest cases:

- `CUDA pinned model reference` calls
  `iom_conformance::run_model_reference_conformance(Device&)`. It exercises the
  two synthetic models at full-prompt lengths 1, 15, 16, and 17, including the
  one- and two-layer BF16 intermediate and final checkpoints and the final
  logits.
- `CUDA pinned causal cache` calls
  `iom_conformance::run_model_cache_reference_conformance(Device&)`. It
  exercises non-tile GQA geometry, future-token perturbation invariance,
  nonzero cached decode from both prefixes, finite-value checks, and exact
  capacity admission/rejection.

Both cases construct the existing `CudaDevices` driver fixture, pass its
driver-owned CUDA `Device&` unchanged to the shared consumer, and assert that
the fixture's traffic gate never armed. The cases require a live CUDA device:
`cuInit(0)` and device creation fail the case instead of skipping, falling back,
or substituting a synthetic input.

The target's aggregate CTest registration passes only
`--test-case-exclude=CUDA real model inference`, which keeps the opt-in
official-model case out of the artifact-free selection; both synthetic cases
above remain inside it. The default standard CUDA run therefore reaches them
with no model artifact, no model-artifact environment, and no network, using
only the committed corpus read through `IOM_MODEL_REFERENCE_DIR`.

The shared consumers retain the frozen per-value comparison policy
(`abs(actual - reference) <= 0.05 + 0.02 * abs(reference)` for checkpoint and
final-logit values) and reject non-finite actual or reference values. No
backend-specific relaxation, candidate-fitted bound, aggregate-only comparison,
or near-tie exact-token requirement is introduced: greedy IDs are compared
exactly only where the frozen interval rule certifies the margin, and
teacher-forced logits stay prefix-identical.

## Frozen corpus identity

The committed offline corpus is `test/model/synthetic_reference.json`, selected
for this target only by `IOM_MODEL_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model"`.

| Item | Recorded identity |
| --- | --- |
| Corpus file SHA-256 | `e19efde2f9360159d244093f5a28c989e8a54a906f10f0a6bd26d51aad9c0db5` |
| Recorded payload SHA-256 | `d34876ad407a4955b644c8d747a3c658e9159ed96cf0bfda2d8760e7acfd4d4a` |
| Generator SHA-256 | `3a230cb9f3c75471ecdc775d9b6ad553c2e6ff64dffa01018c6c4f7a28c75d36` |
| Generator command | `python3 test/reference/generate_model_oracles.py --output /tmp/iom-model-reference.json` |
| Generator runtime | CPython 3.11.16 |
| Reference and package pins | `transformers.models.llama.modeling_llama.LlamaForCausalLM`; `torch` 2.1.2+cpu; `numpy` 1.26.4; `safetensors` 0.4.1 |
| Precision | BF16 weights/activations/stored results; FP32 reductions and softmax; BF16 softmax probability storage before PV |
| Comparison bounds | checkpoint and logits `atol=0.05`, `rtol=0.02`; `abs(actual-ref) <= atol + rtol * abs(ref)` |

The corpus carries two pinned geometries
(`synthetic-h18-i22-hq3-hkv1-d6` and `synthetic-h8-i12-hq4-hkv2-d2`) and
fourteen cases across `full_prompt`, `future_perturbation`, and
`teacher_forced` modes. The shared consumers own every assertion; neither case
reads an IOM library, runtime weight readback, or copied C++ recurrence as an
oracle.

## Source map

- Driver registrations: `test/cuda/test_cuda_conformance.cpp`, cases
  `CUDA pinned model reference` and `CUDA pinned causal cache`.
- Shared checkpoint consumer:
  `test/backend/backend_conformance_model_reference.hpp`.
- Shared cache consumer: `test/backend/backend_conformance_model_cache.hpp`.
- Target data definition: `test/CMakeLists.txt`, `IOM_MODEL_REFERENCE_DIR` on
  `iom_cuda_conformance_tests` only.

## Verification receipt

The checks ran from the prepared worktree through the configured `cuda` profile
with both required environment values, and every remote execution was preceded
by a fresh sync of that exact worktree. Retained transcript with syncs, full
build output, test counts, exit statuses, and cleanup: `remote.log` in the task
metadata directory
`.cswd/tasks/006-tinyllama/14-integration-reference-validation/11-cuda-synthetic-integration`.

- Standard suite, mirror `cuda-synthetic-11-recover1-cuda`:
  `CSW_REMOTE_TASK_DIR=<this leaf's task directory> CSW_REMOTE_WORKSPACE=<this leaf's worktree> /home/rlew/iom/src/iom/.omp/csw/bin/test_cuda cuda-synthetic-11-recover1`.
  Configure reported `CUDAToolkit 13.2.78` at
  `/usr/local/cuda/targets/x86_64-linux/include`, the CUDA compiler
  `NVIDIA 13.2.78` with host compiler `GNU 13.3.0`, and the CXX compiler
  `GNU 13.3.0`; the build of `iom_cuda_conformance_tests` succeeded. CTest
  reported `100% tests passed, 0 tests failed out of 4`:
  `iom_cuda_smoke_tests` (0.31 s), `iom_cuda_conformance_tests` (30.25 s),
  `iom_cuda_sdpa_nonmatrix_tests` (0.17 s), and
  `iom_backend_coexistence_tests` (0.19 s); total real time 30.91 s. The
  runner removed mirror `cuda-synthetic-11-recover1-cuda` afterwards.
- Device, toolchain, and corpus identity, mirror `cuda-synthetic-11-recover2`,
  after a fresh sync:
  `csw-remote-exec cuda cuda-synthetic-11-recover2 'timeout --kill-after=30s 300s bash -c "nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader; nvcc --version; nvidia-smi | sed -n 1,4p; sha256sum test/model/synthetic_reference.json"'`
  exited 0 and reported ordinal `0`, `NVIDIA GeForce RTX 5090`, driver
  `595.71.05`; `nvcc` release `13.2`, `V13.2.78`, built
  `Thu_Mar_19_11:12:51_PM_PDT_2026`; the driver banner
  `NVIDIA-SMI 595.71.05 | Driver Version: 595.71.05 | CUDA Version: 13.2`;
  and `e19efde2f9360159d244093f5a28c989e8a54a906f10f0a6bd26d51aad9c0db5` for
  the corpus file, matching the committed and recorded identity above.
- Focused setup, mirror `cuda-synthetic-11-recover2`: after a fresh sync, the
  bounded remote configure
  `timeout --kill-after=30s 1800s cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DIOM_TEST_REAL_MODEL_LOADING=OFF -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DCUDA_ENABLED=ON`
  and the bounded target build
  `timeout --kill-after=30s 1800s cmake --build build --target iom_cuda_conformance_tests --parallel "$(nproc)"`
  both exited 0, with CUDAToolkit/nvcc `13.2.78` and host compiler `GNU 13.3.0`.
- Named synthetic cases, mirror `cuda-synthetic-11-recover2`, after another
  fresh sync:
  `flock -w120 /tmp/agent-gpu0.lock timeout --kill-after=30s 900s ./build/test/iom_cuda_conformance_tests --test-case="CUDA pinned model reference,CUDA pinned causal cache"`
  exited 0 and reported `test cases: 2 | 2 passed | 0 failed | 47 skipped` and
  `assertions: 2159 | 2159 passed | 0 failed`, doctest `Status: SUCCESS!`.
  Both named cases therefore ran on the CUDA device rather than being filtered:
  the interval checkpoint/final-logit comparisons, finite-value checks, the
  future-perturbation invariance, the nonzero cached decode from prefixes 1 and
  14, and the exact-capacity admission/rejection all reported no failure.
- Cleanup: `csw-remote-clean cuda cuda-synthetic-11-recover2` removed the
  focused mirror, and the standard runner removed its own mirror. A final
  bounded check confirmed no owned test process remained, no
  `cuda-synthetic-11` mirror was left behind, and `/tmp/agent-gpu0.lock` was
  free; the throwaway mirror used for that check was then removed as well.

The all-backend `csw_verify` gate and the official-model cases are owned by the
integration verifier and are intentionally not run or claimed by this leaf.
