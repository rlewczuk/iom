# SYCL official TinyLlama inference evidence

## Evidence boundary

This is the SYCL-owned, opt-in evidence path for the shared [official-model
inference harness](opt-in-real-model-inference.md). It executes the production
TinyLlama session on the caller-selected SYCL device and binds the complete
numerical, state, ownership, instrumentation, runtime, and native-facility
record to the verified model and exported `official_reference` pack. It is not
ordinary artifact-free CTest, synthetic-model evidence, a sample command, a
loading-only check, or evidence for another backend.

`IOM_TEST_REAL_MODEL_INFERENCE_SYCL` defaults to `OFF`. Enabling it requires
`SYCL_ENABLED=ON`, compiles the guarded `SYCL real model inference` case into
`iom_sycl_conformance_tests`, and registers exactly
`iom_sycl_real_model_inference_tests`. The ordinary aggregate
`iom_sycl_conformance_tests` registration excludes the guarded case, so the
Python wrapper remains the only artifact-validation and process-supervision
route. The registered test has the `official-model` label and an 1800-second
CTest timeout. `IOM_TEST_REAL_MODEL_LOADING` remains independent.

## Bound caller inputs and artifacts

Every enabled run must provide all five values explicitly:

```text
IOM_TEST_MODEL_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0
IOM_TEST_MODEL_ID=fe8a4ea1ffedaf415f4da2f062534de366a451e6
IOM_TEST_MODEL_REFERENCE=/path/to/tinyllama-official-reference.json
IOM_TEST_MODEL_EVIDENCE=/path/to/sycl-official-evidence.json
IOM_TEST_MODEL_ARENA_BYTES=4294967296
```

The canonical retained distribution and reference pack are:

- model directory: `/home/rlew/models/TinyLlama-1.1B-Chat-v1.0`;
- model revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`;
- reference pack: `/home/rlew/agent-work/iom-reference/tinyllama14-official-053125.json`;
- reference-pack SHA-256: `8eba795ed856e2eeef3f92d2db05163b4df5094ce05a72c12979be0010af043a`;
- reference case-payload SHA-256: `21816a453e20142711c5c04ff44ff94912ce13bb2bb274272b3fb107843cac01`;
- reference generator SHA-256: `207008c8afce174a1c57c365a4e16fb4aecebb6cd5b498220643cad1c29cfdd4`.

The wrapper and the C++ runner recompute the complete model inventory and
reference identity; names and asserted revisions are not trusted. The
reference pack binds the following model files:

| Relative artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 608 | `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9` |
| `generation_config.json` | 124 | `18046d04f5bd8b4998095ecabdd17a1bf0053d9acdccead4a05be4a3575f3c5c` |
| `model.safetensors` | 2200119864 | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` |
| `special_tokens_map.json` | 551 | `82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db` |
| `tokenizer.json` | 1842767 | `bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e` |
| `tokenizer.model` | 499723 | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` |
| `tokenizer_config.json` | 1289 | `7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a` |

The model directory remains immutable through each independent load and final
evidence publication. Evidence is outside the model and reference trees and
must not alias any protected file by path, canonical path, or file identity.

## Exact build and execution

The remote SYCL command must initialize oneAPI with nounset disabled for every
remote process and must show a Level Zero GPU before configuration:

```sh
set +u
source /opt/intel/oneapi/setvars.sh >/tmp/iom-official-sycl-setvars.log 2>&1
set -u
sycl-ls
cmake -S . -B build-official -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF \
  -DSYCL_ENABLED=ON -DIOM_TEST_REAL_MODEL_INFERENCE_SYCL=ON
cmake --build build-official --target iom_sycl_conformance_tests

IOM_TEST_MODEL_DIR=/home/rlew/models/TinyLlama-1.1B-Chat-v1.0 \
IOM_TEST_MODEL_ID=fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
IOM_TEST_MODEL_REFERENCE=/home/rlew/agent-work/iom-reference/tinyllama14-official-053125.json \
IOM_TEST_MODEL_EVIDENCE=/home/rlew/agent-work/iom-reference/tinyllama14-14-sycl-official-evidence.json \
IOM_TEST_MODEL_ARENA_BYTES=4294967296 \
flock -w120 /tmp/agent-gpu0.lock \
timeout --kill-after=30s 1800s ctest --test-dir build-official \
  --output-on-failure --no-tests=error --timeout 1800 \
  -R '^iom_sycl_real_model_inference_tests$'
```

The CTest registration invokes `test/model/run_official_inference.py` with
`--backend sycl --executable $<TARGET_FILE:iom_sycl_conformance_tests>`.
That wrapper then launches only the named Doctest case through an argument
vector, never a shell. Invalid or missing model/reference/digest inputs,
invalid arena size,
unsupported device/configuration, an unavailable Level Zero GPU, and a failed
enabled case are failures, never skips or fallbacks.

## Numerical and behavioral coverage

The fixed official geometry is untied BF16 Llama
`N22,H2048,I5632,Hq32,Hkv4,D64,V32000,C2048`. The five cases are raw
production greedy, raw fixed-reference continuation, structured-chat
production greedy, structured-chat fixed-reference continuation, and zero new
tokens. Nonzero cases perform actual prefill and repeated single-token cached
R1 decode up to four tokens. The fixed continuation is `3,4,5,6`; it is
teacher-forced reference input, not a fabricated production result. Raw and
chat prompts retain their exact tokenizer/rendering bytes and logical
positions.

Weights, activations, stored logits, and softmax probabilities use BF16 storage
with round-to-nearest-even boundaries. Reductions and softmax use FP32,
including the BF16 probability round before PV. Every logical and compared
full-vocabulary value is finite. For every vocabulary index `i`, the frozen
bound is:

```
abs(actual_i - reference_i) <= 0.53125 + 0.02 * abs(reference_i)
```

A greedy ID is exact only when the reference winner's lower bound is strictly
greater than every competing upper bound; lowest-ID ties are recorded rather
than treated as certified. Teacher-forced logits compare identical prefixes
regardless of margin. The record retains full logits, IDs, text, stop reason,
generated count, history, initialized KV length, and per-layer cache
advancement. Disabled, metrics-only, and trace modes must have equal finite
logits, IDs, text, stop/count/KV state, and poisoning outcomes. A zero-new run
performs no forward, selector, or trace operation and records no invented
latency.

The SYCL driver creates ordinal `0` from a caller-validated positive arena
before invoking the shared runner. Device, allocator, runtime, queue,
workspace, model/session, activation, cache, scratch, and temporary views stay
owned within the case lifetime. Initialized KV length remains distinct from
committed history when a terminal ID is retained without another cache append.

## SYCL, UR, matrix, and timing evidence

Machine-readable official JSON records the selected SYCL ordinal, runtime and
toolchain identity, BF16/FP32 policy, exact command/environment, artifact
digests, per-case outcomes, and host-only timing capability. It does not claim
UR device fields, subgroup or `ext_intel_matrix` facility observations, or
trace attribution. Those observations live in the separately retained
`sycl-ls` and `sycl-trace` payload records. Current SYCL configurations do not
expose a genuine native device timer; host enqueue/completion spans are
observations only and must never be reported as kernel duration or throughput.

The SYCL linear BF16 path uses subgroup-16 `ext_intel_matrix` joint-matrix
BF16/BF16/FP32 MAD when the runtime capability query proves it. The SYCL SDPA
path uses the corresponding joint-matrix QK and PV stages. Official evidence
must connect those actual facilities to the executed prefill and logical R1
cached-decode work, including the linear projection and attention QK/PV
products; the existing operation-only matrix record is not a substitute.
Evidence may claim the facility, runtime, toolchain, and traced kernel
attribution, but must not claim hardware instruction counters or device time
unless independently observed. If the selected device lacks a native matrix
facility, the evidence records that facility as unsupported while the BF16
inference result still has to pass; no CPU fallback or fabricated device
record is permitted.

The bounded supplemental profiler payload frozen by the specification is
exactly:

```sh
sycl-trace --print-format=verbose --ur.call \
  python3 test/model/run_official_inference.py \
  --backend sycl --executable build-official/test/iom_sycl_conformance_tests
```

Run it only after a fresh SYCL sync, oneAPI initialization, and acquisition of
`/tmp/agent-gpu0.lock` outside the bounded command. Retain `sycl-ls`, stdout,
stderr, exit status, result JSON, trace output, and named counter/facility
limitations beside the task evidence. A trace that identifies the linear and
QK/PV joint-matrix launches proves attribution, not unavailable ISA counters or
device duration.

### Launcher root cause and traced payloads

Three bounded controls on the pinned oneAPI `2026.1` toolchain and the same
Level Zero host isolate the recorded exit `255` to the launcher's target
argument: `--ur.call /bin/echo hi` exited `0`, `--ur.call python3 -c "print(1)"`
reproduced `Failed to launch target application. Error code -1` with exit
`255`, and `--ur.call /usr/bin/python3 -c "print(1)"` exited `0`. `sycl-trace`
therefore launches an absolute target path under this pinned runtime, and the
frozen payload above is retained verbatim as the specification's form.

The corrected form below keeps the wrapper on the path and runs to completion,
but records zero Unified Runtime calls: the traced process is the interpreter,
and the payload does not follow the conformance grandchild that owns the device
work. Observed on the pinned host: exit `0`, wrapper measurement
`status=passed` with all five cases, and `0` recorded Unified Runtime
kernel-launch calls.

```sh
sycl-trace --print-format=verbose --ur.call \
  /usr/bin/python3 test/model/run_official_inference.py \
  --backend sycl --executable build-official/test/iom_sycl_conformance_tests
```

Native attribution therefore comes from the bounded supplementary payload
below, which traces the guarded conformance binary that the wrapper's child
would run, under the same caller environment and the same case filter:

```sh
IOM_TEST_MODEL_DIR=... IOM_TEST_MODEL_ID=... \
IOM_TEST_MODEL_REFERENCE=... \
IOM_TEST_MODEL_EVIDENCE=/path/to/sycl-trace-supplementary.json \
IOM_TEST_MODEL_ARENA_BYTES=4294967296 \
sycl-trace --print-format=verbose --ur.call \
  ./build-official/test/iom_sycl_conformance_tests \
  --test-case="SYCL real model inference"
```

Use a separate evidence destination: the registered wrapper run remains the
canonical artifact-validation, supervision, and publication record, and the
supplementary run only adds dispatch attribution. Its driver performs the same
reference schema and provenance validation, checks the complete artifact
size/SHA-256 inventory before every mode and before publication, and rechecks
the reference identity, so the traced dispatches stay bound to the official
artifacts. It deliberately does not add the wrapper's executable-alias
protection, process supervision, or final wrapper JSON, and it is not a
substitute for the registered CTest route.

## Observed implementation evidence and limitations

The implementer gate used the unique mirrors
`tinyllama-17-sycl-official-inference-owner2-sycl` for the standard suite and
`tinyllama-17-sycl-official-inference-official2` for the opt-in case. The
standard SYCL runner configured and exercised the default-off tree: `3/3`
tests passed (`iom_sycl_smoke_tests`, `iom_sycl_conformance_tests`, and
`iom_backend_coexistence_tests`) in `20.17` seconds. The explicit opt-in
configuration found IntelLLVM `2026.1.0`, built
`iom_sycl_conformance_tests`, and the artifact-bound wrapper passed the exact
five-case official run in `109.26` seconds. The wrapper's final machine result
was `status=passed`, `backend=sycl`, `measurement.device=0`,
`measurement.cases=5`, reference SHA-256
`8eba795ed856e2eeef3f92d2db05163b4df5094ce05a72c12979be0010af043a`, and
case-payload SHA-256
`21816a453e20142711c5c04ff44ff94912ce13bb2bb274272b3fb107843cac01`.
The retained measurement reports build host `bh2`, runtime host `bh2`,
IntelLLVM `2026.1.0`, Release `-O3 -DNDEBUG`, and processor
`8 core AMD Ryzen 7 9700X 8-Core Processor`.

`sycl-ls` enumerated two Level Zero GPU devices:
`Intel(R) Arc(TM) Pro B60 Graphics`, Unified Runtime
`1.15.38646+7`. The shared official JSON intentionally leaves both wrapper
and nested measurement `native_capability` as `unclaimed`; it provides no
UR-facility or kernel attribution field. The JSON therefore claims the
numerical BF16 inference, state, ownership, and instrumentation parity, while
the native joint-matrix dispatch attribution comes from the retained
`sycl-trace` payload documented below. No hardware counter, ISA instruction
count, or device duration is claimed anywhere.

The frozen bounded profiler payload was attempted repeatedly after fresh syncs
and the same GPU lock:

```text
sycl-trace --print-format=verbose --ur.call python3 test/model/run_official_inference.py --backend sycl --executable build-official/test/iom_sycl_conformance_tests
```

Each attempt enumerated the same Level Zero GPUs but `sycl-trace` returned
exit `255` with `Failed to launch target application. Error code -1`. The
complete invocations and diagnostics remain in the task `remote.log`; the
bounded controls above identify bare-name target resolution as the cause, and
the corrected and supplementary payloads below then produced the retained
trace. No ISA instruction count or device-time claim is made.

The supplementary payload exited `0` on mirrors
`tinyllama-17-sycl-official-inference-owner3-trace` and
`tinyllama-17-sycl-official-inference-owner4-official` after fresh syncs under
the same GPU lock. Both retained verbose traces have `1,448,611` lines; the
post-repair trace is retained as SHA-256
`0ccec1ea948b19ae815dc25f985101fbcae9d737108bb64fcd3764ea5b33a342` and the
earlier one as
`b2b493d6d9664c2aadc1230572110b7801382b8c37bb9459d939a04dd377fee0`. Each
trace creates exactly six native kernels — `LinearBf16MadKernel<16>`,
`LinearBf16TailKernel<16>`, `LinearBf16TailKernel<1>`, `LinearBf16PackKernel`,
`SdpaQkJointMatrixTailKernelTag`, and `SdpaPvJointMatrixTailKernelTag` — and
each records exactly `44,355` Unified Runtime kernel-launch records in total,
of which the six native kernels hold these attributed shares:

| Kernel | Launch records (trace 1 / trace 2) |
| --- | --- |
| `SdpaQkJointMatrixTailKernelTag` | `1,056` / `1,056` |
| `SdpaPvJointMatrixTailKernelTag` | `1,056` / `1,056` |
| `LinearBf16PackKernel` | `7,439` / `7,440` |
| `LinearBf16TailKernel<1>` | `5,400` / `5,423` |
| `LinearBf16TailKernel<16>` | `1,825` / `1,841` |
| `LinearBf16MadKernel<16>` | `912` / `922` |

The `1,056` QK and `1,056` PV records are exactly `22` layers times `48`
executed forwards (4 executed cases, 3 modes, 1 prefill plus 3 logical `R=1`
cached decodes), and the linear joint-matrix MAD, tail, and pack records are
the native projection dispatches of those same forwards: the pack path holds
`7,440` and `7,439` records against the `7,440` linear products of those `48`
forwards (`48 x 155`). Counts are launch records attributed by kernel handle;
the runtime recycles handles within a run, so the six native identities, the
`44,355` total, and the QK/PV totals are the stable facts, and the linear
per-kernel split moves by a few records between independent traces. Each
traced invocation wrote the official measurement retained beside it with
`kind=official-inference-evidence`, `status=passed`, `backend=sycl`, and
exactly the case IDs `raw-production-greedy`,
`raw-fixed-reference-continuation`, `chat-production-greedy`,
`chat-fixed-reference-continuation`, and `zero-new-token`; the trace also
records the Level Zero adapter and platform backend. The linear and QK/PV
joint-matrix dispatches are therefore tied to the executed official prefill and
logical `R=1` cached decode of that run, not to the separate operation-level
analogue. The trace reports call records only, so no counter, occupancy, ISA
instruction count, or device-duration claim is made, and the shared JSON still
reports `native_capability` as `unclaimed` because it has no
facility-publication field.

The invalid-arena negative scenario also ran through the registered wrapper:
`IOM_TEST_MODEL_ARENA_BYTES=31` failed in `0.04` seconds with the required
positive-decimal/divisible-by-32 diagnostic and CTest exit `8` (wrapper exit
`2`). Its failure is retained in the same task `remote.log`; the final
successful official run was repeated afterward so the published evidence
remained a passing result.

## Diagnostics and status classes

If an enabled run discovers a cryptic CLI or runtime diagnostic, retain the
exact invocation, exit status, stdout, stderr, and remote log, then repair the
smallest owning reporting path and rerun the same reproduction plus a
successful case. Existing useful diagnostics need no wording change. CLI
status classes remain usage/input `2`, setup/load `3`, and execution `4`;
diagnostics remain on stderr and generated bytes remain on stdout. Fatal
signals require root-cause repair rather than a native-facility claim.
