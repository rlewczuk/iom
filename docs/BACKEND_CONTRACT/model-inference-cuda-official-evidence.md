# CUDA official TinyLlama inference evidence

## Evidence boundary

This is the CUDA-owned, opt-in evidence path for the shared [official-model
inference harness](opt-in-real-model-inference.md). It executes the production
TinyLlama session on the selected CUDA ordinal with a caller-selected tensor
arena. It is not ordinary artifact-free CTest, a synthetic model, a sample
command, a download path, a CPU fallback, or evidence for another backend.

The CMake option `IOM_TEST_REAL_MODEL_INFERENCE_CUDA` defaults to `OFF`.
Enabling it compiles the guarded `CUDA real model inference` case into
`iom_cuda_conformance_tests`. The ordinary aggregate registration passes
`--test-case-exclude=CUDA real model inference`, while exactly
`iom_cuda_real_model_inference_tests` selects that case through the Python
wrapper. The wrapper test is labelled `official-model` with an 1800-second CTest
timeout. `IOM_TEST_REAL_MODEL_LOADING` remains independent, and enabling the
option without `CUDA_ENABLED=ON` fails configuration with a `FATAL_ERROR`
instead of silently registering an unusable case.

## Bound official inputs

The retained official input set is immutable for the complete run:

- model directory: `/home/rlew/models/TinyLlama-1.1B-Chat-v1.0`;
- model identity/revision:
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6`;
- reference pack:
  `/home/rlew/agent-work/iom-reference/tinyllama14-official-053125.json`;
- reference-pack SHA-256:
  `8eba795ed856e2eeef3f92d2db05163b4df5094ce05a72c12979be0010af043a`;
- reference case-payload SHA-256:
  `21816a453e20142711c5c04ff44ff94912ce13bb2bb274272b3fb107843cac01`;
- reference generator SHA-256:
  `207008c8afce174a1c57c365a4e16fb4aecebb6cd5b498220643cad1c29cfdd4`.

The wrapper and the C++ runner hash these caller files instead of trusting the
names above. The pack binds this exact inventory:

| Relative artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 608 | `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9` |
| `generation_config.json` | 124 | `18046d04f5bd8b4998095ecabdd17a1bf0053d9acdccead4a05be4a3575f3c5c` |
| `model.safetensors` | 2200119864 | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` |
| `special_tokens_map.json` | 551 | `82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db` |
| `tokenizer.json` | 1842767 | `bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e` |
| `tokenizer.model` | 499723 | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` |
| `tokenizer_config.json` | 1289 | `7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a` |

The independent export used CPython 3.12.3 with `torch 2.3.1+cpu`,
`transformers 4.41.2`, `tokenizers 0.19.1`, `safetensors 0.4.3`,
`sentencepiece 0.2.0`, `numpy 1.26.4`, and `jinja2 3.1.6`.

## Exact invocation

The registered route passes the caller environment to the wrapper, which
verifies the model and reference identity before launching only
`CUDA real model inference` in `iom_cuda_conformance_tests` through an argument
vector; it never invokes a shell, downloads an artifact, discovers a default, or
substitutes a backend.

```sh
cmake -S . -B build-official \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=ON \
  -DROCM_ENABLED=OFF \
  -DSYCL_ENABLED=OFF \
  -DIOM_TEST_REAL_MODEL_INFERENCE_CUDA=ON
cmake --build build-official --target iom_cuda_conformance_tests

IOM_TEST_MODEL_DIR=/home/rlew/models/TinyLlama-1.1B-Chat-v1.0 \
IOM_TEST_MODEL_ID=fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
IOM_TEST_MODEL_REFERENCE=/home/rlew/agent-work/iom-reference/tinyllama14-official-053125.json \
IOM_TEST_MODEL_EVIDENCE=/home/rlew/agent-work/iom-reference/tinyllama15-cuda-official-evidence.json \
IOM_TEST_MODEL_ARENA_BYTES=4294967296 \
flock -w 120 /tmp/agent-gpu0.lock \
timeout --kill-after=30s 1800s \
  ctest --test-dir build-official --output-on-failure --no-tests=error \
    --timeout 1800 -R '^iom_cuda_real_model_inference_tests$'
```

A missing model/reference/evidence input, a wrong revision or digest, an
unsupported CUDA configuration or device, an invalid arena, and a failed enabled
run are failures. The arena is a device-construction value: it must be a
positive decimal byte count divisible by 32 that covers the checked aggregate
standard weight total plus the queried scratch maximum, and it is validated
before any device, context, or allocation exists. There is no skip, fallback,
default, download, sampling feature, or tolerance change.

The nested deadlines are deliberate. The registered CTest deadline is 1800
seconds and the documented shell supervisor allows the same 1800 seconds plus
its 30-second kill-after interval, while the wrapper retains its shared
3600-second child budget, 30-second `SIGTERM` grace, and one shared 30-second
post-`SIGKILL` reap deadline for the case where it supervises without an outer
CTest deadline. Every accelerator payload acquires the host-wide
`/tmp/agent-gpu0.lock` outside its bounded timeout.

Each enabled failure class was exercised and failed instead of skipping: a
missing caller model directory, a reference pack whose recorded `config.json`
size no longer matches the model (exit `2`, `artifact identity mismatch for
config.json`), and an arena byte count of `31`, which is not divisible by 32
(exit `2`) were all rejected by the wrapper with fresh
`status: "failed"` evidence, and the guarded driver itself exited nonzero for
the same misaligned arena before any device existed and for the missing model
directory, where it published failure evidence naming `IOM_TEST_MODEL_DIR` and
printed its `result          : FAIL` line. No cryptic bare-native-status,
`bad_alloc`, or unknown-phase failure was observed in these runs, so no
error-reporting repair was required.

## CUDA execution record

The guarded driver records the selected device and toolchain facts on standard
output, and the wrapper retains that text verbatim in its result JSON beside the
shared harness measurement. The retained run reported:

```text
iom cuda official model inference evidence
  invocation      : ["/home/rlew/agent-work/iom/csw00614-15-cuda-a1/build-official/test/iom_cuda_conformance_tests","--test-case=*real model inference*"]
  backend kind    : cuda
  backend ordinal : 0
  arena bytes     : 4294967296
  device name     : NVIDIA GeForce RTX 5090
  compute capability: 12.0
  cuda runtime    : 13020
  cuda driver     : 13020
  native bf16 wmma: supported, compiled image arch 1200
  result          : PASS
```

The measurement's execution identity names runtime and build host `beha`, the
configured processor name `Unknown AMD family`, description `16 core AMD Ryzen 9
9950X 16-Core Processor`, architecture `x86_64`, compiler GNU `13.3.0`, build
type `Release` with effective C++ flags `-O3 -DNDEBUG`, and no OpenMP
variables; the backend/device fields report `cuda` and ordinal `0`, and the
precision fields report BF16 weights, activations, and logits with FP32
accumulation. The registered attempt passed in `107.61` seconds inside the
1800-second CTest deadline, and the wrapper supervision record reports the
3600/30/30-second child/TERM/reap budgets with `timed_out=false`, no signals,
and a normal reap.

## Numerical, behavioral, and ownership coverage

The reference and implementation use BF16 weights, activations, and stored
logits with round-to-nearest-even conversion boundaries. Reductions and softmax
arithmetic are FP32, with softmax probabilities rounded to BF16 before the PV
product. Every logical value and every compared full-vocabulary logit must be
finite. For each logit $i$, the frozen envelope is

$$
|\mathrm{actual}_i-\mathrm{reference}_i| \le
0.53125 + 0.02|\mathrm{reference}_i|.
$$

A production greedy ID is exact only when the reference winner's lower bound is
strictly greater than every competing upper bound; lowest-ID ties are recorded
separately. Teacher-forced reference-prefix continuations compare full logits at
identical prefixes even without a certified margin, while production IDs are
enforced only when margin-certified. No CUDA-specific tolerance relaxation is
permitted.

The five fixed cases cover raw and structured-chat production greedy paths, raw
and chat fixed-reference continuations, and zero new tokens. Nonzero cases run
real prefill and repeated single-token cached decode up to four new tokens. If
production greedy reaches EOS early, the fixed continuation still exercises
repeated cached decode. The evidence retains full finite logits, selected IDs,
text, stop reason, generated count, request/history state, initialized KV
length, per-layer cache advancement, disabled/metrics/trace parity, and trace
phase/position/operation attribution. The zero-new case must perform no forward,
selection, or trace operation and must not invent timing.

The retained CUDA run reproduced the reference history for all five cases and
compared full BF16 vocabularies against the independently exported reference:

| Case | Selected IDs | Text | Initialized KV | History | Decode forwards | Logit rows | Trace rows |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `raw-production-greedy` | `3681,29889,13,13` | `Paris.\n\n` | 9 | 10 | 3 | 4 | 12 |
| `raw-fixed-reference-continuation` | `3,4,5,6` | forced IDs | 9 | 10 | 3 | 4 | 12 |
| `chat-production-greedy` | `29902,29915,29885,451` | `I'm not` | 21 | 22 | 3 | 4 | 12 |
| `chat-fixed-reference-continuation` | `3,4,5,6` | forced IDs | 21 | 22 | 3 | 4 | 12 |
| `zero-new-token` | none | empty | 0 | 0 | 0 | 0 | 0 |

Every nonzero case stopped on the four-token limit with TTFT available, reported
one whole-prompt prefill plus one terminal prefill row and a decode row for each
of its three cached forwards, dropped no trace rows, and reported equal
disabled/metrics/trace outcomes. The zero-new case performed no forward,
selection, or trace work and reported an unavailable time-to-first-token instead
of an invented rate. Initialized KV length stays one below committed history in
the four-token cases because the terminal selected token is retained in history
without another cache append.

The selected `std::unique_ptr<iom::Device>` owns the CUDA context and both
arenas for the whole case lifetime; it is created after the arena is validated
and destroyed before the case returns, so no device, allocator, queue,
model/session owner, activation or cache storage, or temporary view escapes the
case. Initialized KV length remains distinct from committed history when a
selected terminal token is retained without another cache append. The three
instrumentation modes must have equal BF16 logits, IDs, text, stop/count/KV
state, and poisoning outcome.

## Native BF16 matrix evidence

The CUDA device path executes the production TinyLlama session, so its linear
projections and attention QK/PV stages are submitted through the real queue/OID
path. On a device with the BF16 WMMA facility (compute capability 8.0 or newer)
the projections are executed by the native BF16 matrix specialization
`standard_tiled_linear_bf16_kernel`, and the attention QK/PV stages by
`sdpa_qk_kernel` and `sdpa_pv_kernel`.

The profiler payload wraps the wrapper itself, so its observed launches belong
to exactly the official prefill and logical `R=1` cached-decode submissions of
the five fixed cases. The retained payload was

```sh
flock -w 120 /tmp/agent-gpu0.lock \
timeout --kill-after=30s 1800s \
env IOM_TEST_MODEL_DIR=/home/rlew/models/TinyLlama-1.1B-Chat-v1.0 \
    IOM_TEST_MODEL_ID=fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
    IOM_TEST_MODEL_REFERENCE=/home/rlew/agent-work/iom-reference/tinyllama14-official-053125.json \
    IOM_TEST_MODEL_EVIDENCE=/home/rlew/agent-work/iom-reference/tinyllama15-cuda-official-nsys-evidence.json \
    IOM_TEST_MODEL_ARENA_BYTES=4294967296 \
  nsys profile --force-overwrite=true \
    -o /home/rlew/agent-work/iom-reference/cuda-model \
    python3 test/model/run_official_inference.py --backend cuda \
      --executable build-official/test/iom_cuda_conformance_tests
nsys stats --report cuda_gpu_kern_sum \
  /home/rlew/agent-work/iom-reference/cuda-model.nsys-rep
```

The profiled run passed the same numerical gate and published the same
`status: passed` summary with reference SHA-256
`8eba795ed856e2eeef3f92d2db05163b4df5094ce05a72c12979be0010af043a`. The
retained 14,209,954-byte report
`/home/rlew/agent-work/iom-reference/cuda-model.nsys-rep` attributes these
executed launches:

| Kernel | Instances | Total device time (ns) |
| --- | ---: | ---: |
| `iom::cuda_detail::standard_tiled_linear_bf16_kernel` | 7,440 | 4,200,553,774 |
| `iom::detail::silu_kernel` | 1,056 | 74,862,605 |
| `iom::detail::scatter_plane_kernel` | 3,063 | 69,090,605 |
| `iom::detail::standard_tiled_rmsnorm_kernel` | 2,160 | 45,739,803 |
| `iom::cuda_detail::sdpa_qk_kernel` | 1,056 | 8,288,403 |
| `iom::detail::standard_tiled_rope_kernel` | 2,112 | 6,988,546 |
| `iom::detail::cache_append_kernel` | 2,112 | 6,284,101 |
| `iom::cuda_detail::sdpa_pv_kernel` | 1,056 | 5,554,789 |
| `iom::cuda_detail::softmax_kernel` | 1,056 | 5,180,929 |
| `iom::cuda_detail::scale_mask_kernel` | 1,056 | 1,052,934 |
| `iom::cuda_detail::canonicalize_output_kernel` | 1,056 | 917,445 |
| `iom::detail::embedding_word_kernel` | 48 | 113,248 |
| `iom::detail::gather_plane_kernel` | 48 | 70,304 |

The counts are the execution attribution. Four nonzero cases in three
instrumentation modes give twelve independent runs, each with one prefill and
three single-token cached forwards: 22 layers times seven projections plus one
LM head per forward is 155 native BF16 projection launches per forward, and
`12 x 4 x 155 = 7,440` exactly matches the observed instances. Attention QK and
PV are likewise `12 x 4 x 22 = 1,056` instances each, one per layer per forward,
so the native linear and QK/PV evidence is tied to the official prefill and
logical `R=1` decode rather than to a separate operation benchmark.

`ncu` is installed on this host (`/usr/local/cuda/bin/ncu`, Nsight Compute
2026.4.1), but it cannot produce counter evidence here. The documented payload
is rejected by this version with `==ERROR== --print-source option requires
--page source.`, and the minimally corrected invocation

```sh
flock -w120 /tmp/agent-gpu0.lock \
timeout --kill-after=30s 1800s \
env <the four caller inputs and IOM_TEST_MODEL_ARENA_BYTES> \
  ncu --target-processes all \
      --kernel-name regex:standard_tiled_linear_bf16_kernel --launch-count 8 \
      --page source --print-source sass \
      python3 test/model/run_official_inference.py --backend cuda \
        --executable build-official/test/iom_cuda_conformance_tests
```

connected to the guarded child and reported

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

without collecting counters or SASS. That run nevertheless completed the
official gate with `status: "passed"` for the same reference SHA-256, and its
log is retained at
`/home/rlew/agent-work/iom-reference/tinyllama15-cuda-ncu.log`. No counter,
occupancy, or disassembly-level claim is made: the native BF16 matrix evidence
above comes from executed Nsight Systems launches tied to the official prefill
and logical `R=1` decode.

## Capability and timing limitations

- Native BF16 matrix support: supported and observed on the retained device
  (compute capability 12.0, compiled image arch 1200, 7,440 executed
  `standard_tiled_linear_bf16_kernel` launches plus 1,056 each of
  `sdpa_qk_kernel` and `sdpa_pv_kernel` in the profiled official run).
- Launch-level profiling: available through Nsight Systems 2026.4.1.
- Counter-level profiling: unavailable — `ERR_NVGPUCTRPERM` restricts NVIDIA
  GPU performance counters to administrators on this host, and the documented
  `ncu` payload is additionally rejected by Nsight Compute 2026.4.1 unless
  `--page source` is supplied.
- CPU sampling: unavailable — this host's configuration disables CPU
  IP/backtrace sampling and CPU context-switch tracing, so the report contains
  GPU kernel activity only.
- Session timing: CUDA exposes no genuine per-phase device timer through the
  public session API, so the harness records host enqueue and completion
  observations only; they are never described as kernel or device time, and no
  throughput threshold or accelerator timing claim is made. Profiler and
  counter availability is reported separately from numerical acceptance, and
  every facility above is stated as observed rather than inferred.

A bare native status, `bad_alloc`, unknown phase, fatal signal, or similarly
cryptic failure encountered by this enabled run must retain invocation, exit
status, stdout, and stderr and be repaired at the smallest owning reporting
boundary before this page can report success. CLI status classes remain
usage/input `2`, setup/load `3`, and execution `4`, with diagnostics on stderr
and generated bytes on stdout.
