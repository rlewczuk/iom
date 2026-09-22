# SYCL pinned synthetic model integration evidence

This page records the SYCL execution of the pinned offline model-reference and
causal-cache consumers. It is conformance evidence for the existing
`iom_sycl_conformance_tests` target; it is not official-model, network, or
profiling evidence.

## Implementation boundary

The driver registers these separate Doctest cases:

- `SYCL pinned model reference`
- `SYCL pinned causal cache`

Both cases construct the existing `SyclDevices` fixture and pass its
driver-owned candidate `iom::Device&` to the unchanged shared consumers
`iom_conformance::run_model_reference_conformance(Device&)` and
`iom_conformance::run_model_cache_reference_conformance(Device&)`. The existing
SYCL arena, context-capture seam, and device lifetimes remain in the driver.
The target-only `IOM_MODEL_REFERENCE_DIR` definition points at the committed
`test/model` directory. No official artifact or network input is read.

The shared frozen comparison policy is used unchanged:
`abs(actual-ref) <= 0.05 + 0.02 * abs(ref)` for checkpoint and final-logit
values. The fixture keeps BF16 storage/RNE boundaries and FP32 reductions and
softmax. Non-finite actual or reference values are failures, not filtered
observations.

## Corpus identity and provenance

The committed corpus is `test/model/synthetic_reference.json`.

| Field | Recorded identity |
| --- | --- |
| Corpus file SHA-256 | `e19efde2f9360159d244093f5a28c989e8a54a906f10f0a6bd26d51aad9c0db5` |
| Corpus payload SHA-256 | `d34876ad407a4955b644c8d747a3c658e9159ed96cf0bfda2d8760e7acfd4d4a` |
| Generator SHA-256 | `3a230cb9f3c75471ecdc775d9b6ad553c2e6ff64dffa01018c6c4f7a28c75d36` |
| Reference runtime | CPython 3.11.16; `transformers.models.llama.modeling_llama.LlamaForCausalLM` |
| Pinned packages | `torch 2.1.2+cpu`, `numpy 1.26.4`, `safetensors 0.4.1`, `transformers 4.35.0`, `tokenizers 0.14.1`, `sentencepiece 0.1.99`, `jinja2 3.1.2` |
| Corpus generation command | `python3 test/reference/generate_model_oracles.py --output /tmp/iom-model-reference.json` |
| Frozen bounds | checkpoint/logits `atol=0.05`, `rtol=0.02`; `abs(actual-ref) <= atol + rtol * abs(ref)` |

## Device and toolchain

The checks used the configured `sycl` profile on `bv2`, with the selected
SYCL driver ordinal `0`. Every supplemental execution used an outside-checkout
profile override with an empty `REMOTE_SETUP`, then initialized oneAPI with
nounset disabled before enabling nounset again. `sycl-ls` was run before each
supplemental build, test, or identity command and required Level Zero GPU
output.

| Item | Observed value |
| --- | --- |
| Level Zero device ordinal 0 | Intel(R) Arc(TM) Pro B60 Graphics; oneAPI Unified Runtime over Level-Zero V2 20.1.0; driver `1.15.38646+7` |
| Level Zero device ordinal 1 | Intel(R) Arc(TM) Pro B60 Graphics; oneAPI Unified Runtime over Level-Zero V2 20.1.0; driver `1.15.38646+7` |
| DPC++ compiler | Intel(R) oneAPI DPC++/C++ Compiler 2026.1.0 (`2026.1.0.20260617`), `icpx` |
| `ocloc` | `26.22.38646.7` |
| Standard mirror | `tinyllama14-sycl-synth-13-a1-sycl` |
| Focused mirror | `tinyllama14-sycl-synth-13-focus1` |
| Evidence log | `.cswd/tasks/006-tinyllama/14-integration-reference-validation/13-sycl-synthetic-integration/remote.log` |

## Executed gates

The local standard-runner invocation was:

```text
CSW_REMOTE_TASK_DIR=/home/rlew/iom/src/iom/.work/006-tinyllama/14-integration-reference-validation/13-sycl-synthetic-integration/.cswd/tasks/006-tinyllama/14-integration-reference-validation/13-sycl-synthetic-integration \
CSW_REMOTE_WORKSPACE=/home/rlew/iom/src/iom/.work/006-tinyllama/14-integration-reference-validation/13-sycl-synthetic-integration \
/home/rlew/iom/src/iom/.omp/csw/bin/test_sycl tinyllama14-sycl-synth-13-a1
```

The standard runner freshly synchronized, configured a Release SYCL-only
build, built the complete configured target set, enumerated Level Zero GPUs,
and ran:

```text
ctest --test-dir build/csw-runner-sycl --no-tests=error --output-on-failure --timeout 300 --parallel 1 -R '^iom_sycl_.*tests$|^iom_backend_coexistence_tests$'
```

Observed result: `3/3` tests passed (`iom_sycl_smoke_tests`,
`iom_sycl_conformance_tests`, and `iom_backend_coexistence_tests`), `0` failed,
with total CTest time `20.14 s`. The SYCL conformance target therefore
executed both newly registered cases in the ordinary standard suite. The
standard mirror was removed successfully.

After a fresh sync, the focused build used the required nounset-safe setup and
Level Zero check before this bounded build command:

```text
set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; set -o pipefail; timeout --kill-after=30s 120s sycl-ls | tee build/sycl-devices-focus.log; if ! grep -Eiq "level[-_ ]zero.*gpu|gpu.*level[-_ ]zero" build/sycl-devices-focus.log; then echo "No Level Zero GPU found" >&2; exit 1; fi; timeout --kill-after=30s 1800s cmake --build build --target iom_sycl_conformance_tests --parallel "$(nproc)"
```

The focused test then freshly synchronized again and ran under the bounded
shared accelerator lock:

```text
set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; set -o pipefail; timeout --kill-after=30s 120s sycl-ls | tee build/sycl-devices-focus.log; if ! grep -Eiq "level[-_ ]zero.*gpu|gpu.*level[-_ ]zero" build/sycl-devices-focus.log; then echo "No Level Zero GPU found" >&2; exit 1; fi; flock -w120 /tmp/agent-gpu0.lock timeout --kill-after=30s 900s ./build/test/iom_sycl_conformance_tests --test-case="SYCL pinned model reference,SYCL pinned causal cache"
```

Observed focused result:

```text
[doctest] test cases:    2 |    2 passed | 0 failed | 41 skipped
[doctest] assertions: 2163 | 2163 passed | 0 failed |
[doctest] Status: SUCCESS!
```

The two selected cases reported zero failed finite-value or comparison
assertions (non-finite observations: `0`); all `2163` assertions passed. The
focused mirror was removed successfully with `csw-remote-clean` (exit `0`).
The complete sync, setup, build, test, identity, and cleanup transcript is
retained in `remote.log`.

The final `csw_verify` all-backend integration gate is owned by the verifier
and was not run by this leaf.
