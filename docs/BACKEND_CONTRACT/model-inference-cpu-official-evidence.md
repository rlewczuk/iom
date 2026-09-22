# CPU official TinyLlama inference evidence

## Evidence boundary

This is the CPU-owned, opt-in evidence path for the shared [official-model
inference harness](opt-in-real-model-inference.md). It executes the production
TinyLlama session on CPU device ordinal `0`, using the allocator owned by the CPU
conformance driver. It is not ordinary artifact-free CTest, a synthetic model, a
sample command, a download path, or evidence for an accelerator backend.

The CMake option `IOM_TEST_REAL_MODEL_INFERENCE_CPU` defaults to `OFF`. Enabling
it compiles the guarded `CPU real model inference` case into
`iom_backend_conformance_cpu_tests`. The ordinary aggregate registration
explicitly excludes that case, while exactly
`iom_cpu_real_model_inference_tests` selects it through the Python wrapper. The
wrapper test is labelled `official-model` with a 3900-second CTest timeout.
`IOM_TEST_REAL_MODEL_LOADING` remains independent.

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

The pre-cutover historical `tinyllama14-02-official-a4.json` pack remains
retained under its original name and bytes. It is not an alias for the
regenerated canonical pack.

The wrapper and C++ runner hash the caller files rather than trusting these
names. The pack binds this exact inventory:

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

The CPU host receives all four caller inputs explicitly. The wrapper is
standard-library-only and launches the already-built backend executable through
an argument vector; it never invokes a shell, downloads an artifact, discovers
a default, or substitutes a backend.

```sh
cmake -S . -B build-official -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -fopenmp" \
  -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF \
  -DSYCL_ENABLED=OFF -DIOM_TEST_REAL_MODEL_INFERENCE_CPU=ON
cmake --build build-official --target iom_backend_conformance_cpu_tests

OMP_NUM_THREADS=32 \
IOM_TEST_MODEL_DIR=/home/rlew/models/TinyLlama-1.1B-Chat-v1.0 \
IOM_TEST_MODEL_ID=fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
IOM_TEST_MODEL_REFERENCE=/home/rlew/agent-work/iom-reference/tinyllama14-official-053125.json \
IOM_TEST_MODEL_EVIDENCE=/home/rlew/agent-work/iom-reference/tinyllama14-14-cpu-official-053125-evidence.json \
timeout --kill-after=30s 4200s \
  ctest --test-dir build-official --output-on-failure --no-tests=error \
    --timeout 3900 -R '^iom_cpu_real_model_inference_tests$'
```

A missing model/reference/evidence input, a wrong revision or digest, an
unsupported configuration, an empty test selection, and a failed enabled run
are failures. The CPU invocation does not set `IOM_TEST_MODEL_ARENA_BYTES`;
tensor storage comes from the driver-owned allocator.

The nested deadlines deliberately leave ownership of timeout evidence and
cleanup with the wrapper: its child receives 3600 seconds, followed by at most
30 seconds of `SIGTERM` grace and one 30-second post-`SIGKILL` reap deadline;
CTest receives 3900 seconds including validation, and the outer shell receives
4200 seconds plus its 30-second kill-after interval. The separately registered
fast timeout regressions finish inside a 10-second CTest deadline. One covers a
SIGTERM-ignoring leader; the other proves that when a leader exits and closes
its pipes while a same-group descendant ignores SIGTERM, the wrapper preserves
the leader output, sends SIGKILL to the residual original group, observes both
processes disappear within the one post-kill deadline, and publishes that
escalation in fresh machine evidence.

The retained measurement records `OMP_NUM_THREADS` and any supplied
`OMP_DYNAMIC`, `OMP_PROC_BIND`, and `OMP_PLACES` values. Its
`execution_identity` replaces the former process-local device address with the
runtime/build host, configured processor name/description/architecture,
compiler ID/version, build type/effective C++ flags, and CPU device ordinal.

The retained attempt-4 evidence at
`/home/rlew/agent-work/iom-reference/tinyllama14-14-cpu-official-053125-evidence.json`
records these exact execution values:

- runtime host and build host: `beha`;
- processor name: `Unknown AMD family`;
- processor description: `16 core AMD Ryzen 9 9950X 16-Core Processor`;
- processor architecture: `x86_64`;
- compiler: GNU `13.3.0`;
- build: `Release`, effective C++ flags
  `-O3 -DNDEBUG -march=native -fopenmp`;
- OpenMP: `OMP_NUM_THREADS=32`; `OMP_DYNAMIC`, `OMP_PROC_BIND`, and
  `OMP_PLACES` absent;
- backend/device: `cpu`, ordinal `0`.

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
separately. Teacher-forced continuations compare full logits at identical
prefixes even without a certified margin. No CPU-specific tolerance relaxation
is permitted.

The regenerated reference-margin bits, in snapshot order, are
`false,false,false,true` for raw production greedy,
`false,false,false,false` for raw fixed continuation,
`false,false,true,false` for chat production greedy, and
`false,false,true,false` for chat fixed continuation; the zero-token case has
no snapshots. They are generated by the pinned exporter from the same
binary-exact `0.53125` envelope, not hand-edited.

The five fixed cases cover raw and structured-chat production greedy paths,
raw and chat fixed-reference continuations, and zero new tokens. Nonzero cases
run real prefill and repeated single-token cached decode up to four new tokens.
If production greedy reaches EOS early, the fixed continuation still exercises
repeated cached decode. The evidence retains full finite logits, selected IDs,
text, stop reason, generated count, request/history state, initialized KV
length, per-layer cache advancement, disabled/metrics/trace parity, and trace
phase/position/operation attribution. The zero-new case must perform no forward,
selection, or trace operation and must not invent timing.

An additional CPU quality probe on `bv2` invoked `iom_generate` with the
official chat template, user message `What is the capital of Poland ?`, and a
16-token ceiling. It completed with status `0` and emitted exactly
`The capital of Poland is Warsaw.`. This is understandable grammatical English;
the terminal period and early stop are ordinary model output rather than a
postprocessing repair.

Device, allocator, queue, model/session owners, activation and cache storage,
and temporary views stay within the case lifetime. Initialized KV length remains
distinct from committed history when a selected terminal token is retained
without another cache append. The three instrumentation modes must have equal
BF16 logits, IDs, text, stop/count/KV state, and poisoning outcome.

## CPU capability and timing interpretation

The machine result binds its measurements and command environment to the
verified model/reference identities, reports backend `cpu` and device ordinal
`0`, and records the shared host-observation fields. CPU has no accelerator
profiling facility in this path: native accelerator timing and native matrix
support are **not applicable**, not skipped numerical checks. Only host enqueue
and completion observations are available, and they must not be described as
kernel/device time. No performance threshold or accelerator capability claim is
made.

The registered attempt-4 five-case gate completed successfully in `480.89`
seconds inside the 3900-second CTest boundary. Its wrapper supervision record
reported the 3600/30/30-second child/TERM/reap budgets, no timeout, no signal,
and normal reap.

A bare native status, `bad_alloc`, unknown phase, fatal signal, or similarly
cryptic failure encountered by this enabled run must retain invocation, exit
status, stdout, and stderr and be repaired at the smallest owning reporting
boundary before this page can report success. Useful existing diagnostics need
no wording churn. CLI status classes remain usage/input `2`, setup/load `3`, and
execution `4`, with diagnostics on stderr and generated bytes on stdout.
