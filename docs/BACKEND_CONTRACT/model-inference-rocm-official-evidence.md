# ROCm official TinyLlama inference evidence

## Evidence boundary

This is the ROCm-owned, opt-in evidence path for the shared [official-model
inference harness](opt-in-real-model-inference.md). It executes the production
TinyLlama session on ROCm device ordinal `0`, backed by the caller-selected
tensor arena. It is not ordinary artifact-free CTest, a synthetic model, a
sample command, a download path, a fallback to CPU, or evidence for another
backend.

The CMake option `IOM_TEST_REAL_MODEL_INFERENCE_ROCM` defaults to `OFF`.
Enabling it compiles the guarded `ROCm real model inference` case into the
existing `iom_rocm_conformance_tests` executable, and the aggregate
registration of that executable passes
`--test-case-exclude=ROCm real model inference`, so the Python wrapper is the
sole registered route to the case. Exactly
`iom_rocm_real_model_inference_tests` is registered, with label
`official-model` and a 1800-second CTest timeout; it invokes
`test/model/run_official_inference.py --backend rocm --executable
$<TARGET_FILE:iom_rocm_conformance_tests>`. An enabled configuration without
`ROCM_ENABLED=ON` fails at configure time instead of compiling nothing, and
`IOM_TEST_REAL_MODEL_LOADING` remains independent: the case reads no
loading-only symbol, shares no option, and adds no loader.

Because the tensor arena is a device-construction value, the ROCm driver reads
and validates `IOM_TEST_MODEL_ARENA_BYTES` as a positive decimal byte count
divisible by 32 **before** any device, allocator, or runtime exists, then
constructs exactly one `make_rocm_device(0, DeviceMemoryConfig{arena})`. The
selected ordinal reaches the machine evidence through
`Device::backend_device()`. A missing, empty, non-decimal, zero, or misaligned
arena is a failure rather than a skip, and the synthetic conformance reserve is
never substituted for it. There is no skip return for the enabled case.

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

The wrapper and the C++ runner hash the caller files rather than trusting these
names. The pack binds this exact inventory, independently of the ROCm backend:

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
`sentencepiece 0.2.0`, `numpy 1.26.4`, and `jinja2 3.1.6`; that pin describes
the reference generator, not the ROCm wrapper interpreter, which is
standard-library only and hashes the pack itself.

## Exact invocation

The ROCm host receives all five caller inputs explicitly, including the arena
that only the accelerator path needs. The wrapper validates the model,
reference pack, artifact digests, arena, and requested backend/executable
before launching exactly one already-built backend executable through a
subprocess argument vector; it never invokes a shell, downloads an artifact,
selects a default, or substitutes a backend.

```sh
cmake -S . -B build/rocm-official \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=OFF \
  -DROCM_ENABLED=ON \
  -DSYCL_ENABLED=OFF \
  -DIOM_TEST_REAL_MODEL_INFERENCE_ROCM=ON
cmake --build build/rocm-official \
  --target iom_rocm_conformance_tests

IOM_TEST_MODEL_DIR=/home/rlew/models/TinyLlama-1.1B-Chat-v1.0 \
IOM_TEST_MODEL_ID=fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
IOM_TEST_MODEL_REFERENCE=/home/rlew/agent-work/iom-reference/tinyllama14-official-053125.json \
IOM_TEST_MODEL_EVIDENCE=/home/rlew/agent-work/iom-reference/tinyllama16-rocm-official-evidence.json \
IOM_TEST_MODEL_ARENA_BYTES=4294967296 \
flock -w120 /tmp/agent-gpu0.lock \
timeout --kill-after=30s 1800s \
  ctest --test-dir build/rocm-official --output-on-failure \
    --no-tests=error --timeout 1800 \
    -R '^iom_rocm_real_model_inference_tests$'
```

A missing model/reference/evidence input, a wrong revision or digest, a
missing or misaligned arena, an unsupported configuration or device, an empty
test selection, and a failed enabled run are failures. The registered CTest
deadline of 1800 seconds bounds the complete wrapper invocation; the wrapper's
own supervision budget is the shared 3600-second child deadline with a
30-second `SIGTERM` grace interval and a bounded 30-second post-`SIGKILL` reap,
so the CTest deadline is the binding limit for this registration and the
wrapper still owns timeout evidence and child cleanup.

## Numerical, behavioral, and ownership coverage

The frozen pack and shared runner, not this driver, own the comparisons. The
enforced envelope is

$$
|\mathrm{actual}_i-\mathrm{reference}_i| \le
0.53125 + 0.02|\mathrm{reference}_i|
$$

for every full-vocabulary BF16 logit, with BF16 weights, activations, and
stored logits at round-to-nearest-even boundaries, FP32 reductions and softmax,
and BF16 softmax probabilities before the PV product. Every logical value and
every compared logit must be finite. A production greedy ID is exact only when
the reference winner's lower bound strictly exceeds every competing upper
bound; lowest-ID ties are recorded separately, and teacher-forced continuations
compare full logits at identical prefixes regardless of margin. No ROCm-specific
tolerance relaxation is permitted and none is applied.

The five fixed cases, in order, are raw production greedy, raw fixed-reference
continuation, chat production greedy, chat fixed-reference continuation, and
zero new tokens. The raw prompt is `The capital of France is` with special
tokens and a required BOS (prompt IDs
`1,450,7483,310,3444,338`); the chat case uses the distribution template for
`[{"role":"user","content":"Hello."}]` including the assistant prefix and adds
no BOS. Nonzero cases use a four-token limit and script the continuation as IDs
`3,4,5,6`; those IDs are never reported as greedy success. The frozen expected
results are `3681,29889,13,13` for raw production greedy (initialized KV `9`)
and `29902,29915,29885,451` for chat production greedy (initialized KV `21`),
both stopping at the four-token limit rather than EOS, so every case exercises
prefill plus three repeated single-token (`R1`) cached decode forwards. The
reference margin bits are `false,false,false,true` (raw production greedy),
`false,false,false,false` (raw fixed continuation),
`false,false,true,false` (chat production greedy), and
`false,false,true,false` (chat fixed continuation); the zero-token case has no
snapshots and must not perform a forward, selection, or trace operation or
invent timing.

Each nonzero case keeps its own request, history, and cache state: initialized
KV length stays distinct from committed history when the selected terminal
token is retained without another append, every one of the 22 decoder-layer
cache states advances to the same terminal length, and stop precedence remains
EOS, new-token limit, then context capacity. Device, allocator, runtime,
queue, model/session owners, activation and cache storage, and temporary views
stay inside the case, and the caller arena is returned with the device. The
disabled, metrics-only, and prepared-trace modes must agree on finite BF16
logits, selected and generated IDs, decoded text, stop and count state,
committed history, cache lengths, accepted operations, and poisoning outcome.
Generic dtype, shape, alias, device, workspace, BF16, and padding behavior
remains owned by the existing operation conformance; this case implements no
second model recurrence and duplicates none of it.

## ROCm capability, native attribution, and timing interpretation

The machine result records backend `rocm` and device ordinal `0`, which is the
`gfx1201` agent (AMD Radeon AI PRO R9700, wavefront 32). The retained
toolchain is HIP `7.15.26333-0000000` with AMD clang `23.0.0git`
(`8f497e0992fb7513f7f78a6f6b6f1056c375e961`) under `/opt/rocm/core-10.0`.
The measurement's precision block records BF16 weights, activations, and
logits with FP32 accumulation.

Native attribution is not inferred from a kernel name or a disassembly. The
official run executes the production `linear` and SDPA paths, which on this
device reach the native BF16 matrix kernels
`linear_bf16_pack_kernel`, `linear_bf16_wmma_kernel`,
`linear_bf16_scatter_kernel`, `sdpa_q_pack_kernel`, `sdpa_k_pack_kernel`,
`sdpa_qk_wmma_kernel`, `sdpa_softmax_kernel`, `sdpa_v_pack_kernel`,
`sdpa_pv_wmma_kernel`, and `sdpa_merge_kernel`; the bounded `rocprofv3`
payload below records their dispatches during the same prefill and repeated
`R1` decode forwards whose logits, tokens, stop state, and per-layer cache
progress the harness compares. The operation-level analogue with its own
frozen fixtures, workspace declaration, and sampled WMMA facility probe is
[linear projections](linear-projections.md) and the ROCm SDPA cases in
`test/rocm/test_rocm_conformance.cpp`; those cases own the exact per-operation
evidence, while this page records that the official prefill/decode execution
reaches the same native kernels.

This path has no genuine native device timer: `device_time` remains
`unavailable`, and host enqueue and completion observations must not be
described as kernel or device time. `rocprofv3` 1.3.5 on this `gfx1201` agent
supports neither PC sampling nor SPM counter collection, so hardware
instruction counters are honestly unavailable rather than fabricated, and the
facility statement rests on the executed probe and the traced kernel dispatches
of the submitted work. No performance threshold, accelerator speedup, or
counter-derived claim is made.

## Bounded profiler payload

The exact preserved command, run against the same verified inputs through the
ROCm host, is:

```sh
flock -w120 /tmp/agent-gpu0.lock \
timeout --kill-after=30s 1800s \
rocprofv3 --kernel-trace --hip-trace --sys-trace -f csv \
  -d "$IOM_EVIDENCE_DIR/rocm-model" -- \
  python3 test/model/run_official_inference.py --backend rocm \
    --executable build-official/test/iom_rocm_conformance_tests
```

## Verification receipt

Everything below was executed from the prepared worktree through the configured
`rocm` profile (`bv2`), with `CSW_REMOTE_TASK_DIR` set to this task directory
and `CSW_REMOTE_WORKSPACE` set to the assigned worktree. The helper output for
these attempts is retained in the task's `remote.log`, whose entries for this
leaf begin at line 3544.

- **Configure and build**, mirror `tinyllama16-official-a1`: after a fresh
  sync, the bounded remote configure
  `cmake -S . -B build-official -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DIOM_TEST_REAL_MODEL_INFERENCE_ROCM=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=ON -DSYCL_ENABLED=OFF`
  exited `0`, and
  `cmake --build build-official --target iom_rocm_conformance_tests --parallel 16`
  exited `0` after compiling the guarded case into the existing ROCm
  conformance executable.
- **Official gate**, mirror `tinyllama16-official-a1`, fresh sync, accelerator
  lock held outside the bounded run:

  ```sh
  flock -w120 /tmp/agent-gpu0.lock \
  timeout --kill-after=30s 1800s \
  env IOM_TEST_MODEL_DIR=/home/rlew/models/TinyLlama-1.1B-Chat-v1.0 \
      IOM_TEST_MODEL_ID=fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
      IOM_TEST_MODEL_REFERENCE=/home/rlew/agent-work/iom-reference/tinyllama14-official-053125.json \
      IOM_TEST_MODEL_EVIDENCE=/home/rlew/agent-work/iom-reference/tinyllama16-rocm-official-a1-evidence.json \
      IOM_TEST_MODEL_ARENA_BYTES=4294967296 \
      ctest --test-dir build-official --output-on-failure --no-tests=error \
        --timeout 1800 -R '^iom_rocm_real_model_inference_tests$'
  ```

  observed `1/1 Test #13: iom_rocm_real_model_inference_tests ... Passed 114.77 sec`,
  `100% tests passed, 0 tests failed out of 1`, exit `0`. The retained wrapper
  evidence records wrapper `status: passed`, backend `rocm`, return code `0`,
  supervision `timed_out: false`, `reap: normal`, `signals_sent: []`, and the
  measurement `status: passed` with device ordinal `0`, BF16 weights,
  activations, and logits with FP32 accumulation, reference SHA-256
  `8eba795e…43a`, and all five fixed cases in order. Per case the measurement
  records: raw production greedy `token_ids 3681,29889,13,13`,
  `stop_reason max_new_tokens`, cache lengths `9` for all 22 layers,
  `decode_forward_count 3`, `time_to_first_token 49222253 ns`,
  `tokens_per_second 21.87`, mode parity `true`; raw fixed continuation
  `3,4,5,6` with cache `9` and parity `true`; chat production greedy
  `29902,29915,29885,451` with cache `21` and parity `true`; chat fixed
  continuation `3,4,5,6` with cache `21` and parity `true`; zero new tokens
  with empty history, empty cache lengths, no forward, `time_to_first_token`
  unavailable, and parity `true`. Every case recorded finite full-vocabulary
  logits (4 snapshots of 32000 values for the four nonzero cases, none for the
  zero-token case), and the trace rows retain unique operation IDs with
  `prefill`/`decode` phases, position windows, run lengths `6`/`18` then `1`,
  and `succeeded` wait states.
- **Bounded profiler payload**, mirror `tinyllama16-official-a1`, fresh sync,
  the exact command in the section above with
  `IOM_EVIDENCE_DIR=/home/rlew/agent-work/iom-reference/tinyllama16-rocm-profile`
  and `IOM_TEST_MODEL_EVIDENCE=/home/rlew/agent-work/iom-reference/tinyllama16-rocm-official-profiled-evidence.json`:
  exit `0` in 127 seconds including profiling, the wrapper again reporting
  `status: passed` for backend `rocm` with the same reference digest, and the
  profiled measurement again passing all five cases with mode parity. The
  kernel trace is retained at
  `/home/rlew/agent-work/iom-reference/tinyllama16-rocm-profile/rocm-model/bh2/3834784_kernel_trace.csv`
  (65598 dispatches, 14 MB) beside the HIP, HSA, memory, and agent traces.

  Its dispatch counts decompose exactly into the executed forwards:
  `linear_bf16_pack_kernel`, `linear_bf16_wmma_kernel`, and
  `linear_bf16_scatter_kernel` `7440` each, `sdpa_q_pack_kernel`,
  `sdpa_k_pack_kernel`, `sdpa_qk_wmma_kernel`, `sdpa_softmax_kernel`,
  `sdpa_v_pack_kernel`, `sdpa_pv_wmma_kernel`, `sdpa_merge_kernel`, and
  `sdpa_output_store_kernel` `1056` each, `standard_tiled_rmsnorm_kernel`
  `2160`, `standard_tiled_rope_kernel` `2112`, `cache_append_kernel` `2112`,
  the ADD form of `grid_stride_binary_kernel` `2112`, its MUL form and
  `silu_kernel` `1056` each, and `embedding_word_kernel` and
  `gather_plane_kernel` `48` each. With `7440 = 48 × 155` (seven projections
  per layer across 22 layers plus the untied LM head), `1056 = 48 × 22`, and
  `2160 = 48 × 45`, these counts describe exactly 48 forwards, and the 48
  embedding dispatches partition the trace, in submission order, into twelve
  groups of four: one prefill forward followed by three single-token (`R1`)
  decode forwards per group, for the four nonzero cases across the three
  instrumentation modes, with the zero-token case contributing no forward.
  Inside every one of those 48 forwards the trace contains exactly 155 linear
  WMMA dispatches and 22 QK plus 22 PV WMMA dispatches, so native matrix work is
  tied to the executed prefill and `R1` decode forwards rather than to a kernel
  name: the six chat-run prefill forwards are also distinguishable in the
  trace by their larger launch grids (`sdpa_qk_wmma_kernel` grid 4096 and
  `linear_bf16_wmma_kernel` grids 8192/1024/22528) from the `R1` decode
  forwards (grids 2048 and 4096/512/11264).
- **Registration and configuration checks**, mirror `tinyllama16-official-a2`:
  with the option `ON`, the generated `CTestTestfile.cmake` registers
  `iom_rocm_real_model_inference_tests` as
  `<python3> test/model/run_official_inference.py --backend rocm --executable <build>/test/iom_rocm_conformance_tests`
  with `LABELS official-model` and `TIMEOUT 1800`, and the aggregate
  `iom_rocm_conformance_tests` registration carries
  `--test-case-exclude=ROCm real model inference`; with the option `OFF`
  (default), configure still exits `0`, discovers no Python3 interpreter, and
  contains no `iom_rocm_real_model_inference_tests` registration at all; with
  the option `ON` and `ROCM_ENABLED=OFF`, configure exits `1` with the intended
  `FATAL_ERROR` at `test/CMakeLists.txt:317`.
- **Negative caller inputs**, mirror `tinyllama16-official-a1`, by the temporary
  probe `_local/rocm16-negatives.sh` (removed before the final commit), each
  under the same accelerator lock and bounded timeout: an unavailable model
  directory exits `2` with `caller model directory /nonexistent-model-dir is
  unavailable`; a tampered checkpoint inventory (pinned small files, truncated
  `model.safetensors`) exits `2` with `artifact identity mismatch for
  model.safetensors`; the superseded `0.25`-envelope reference pack exits `2`
  with `official comparison policy differs from the frozen policy`; a missing
  `IOM_TEST_MODEL_ARENA_BYTES` fails the case before device creation, with the
  child reporting `test case THREW exception: IOM_TEST_MODEL_ARENA_BYTES must
  be supplied explicitly` and the wrapper publishing exit `1` plus the failed
  child status; a misaligned arena (`100`) exits `2` with
  `IOM_TEST_MODEL_ARENA_BYTES must be a positive decimal byte count divisible
  by 32`. Each negative published a `failed` evidence file; no input produced a
  skip, a fallback, or a false pass, and all diagnostics named the failing
  caller input and stage without further repair being needed.
- **Standard ROCm suite**, default configuration, mirror
  `tinyllama16-official-std-rocm` (runner-owned and removed on success):
  `/home/rlew/iom/src/iom/.omp/csw/bin/test_rocm tinyllama16-official-std-rocm`
  configured and built the ordinary tree with `IOM_TEST_REAL_MODEL_LOADING=OFF`
  and the official option absent, then reported 4/4 CTest entries passed —
  `iom_rocm_smoke_tests` (1.00 s), `iom_rocm_conformance_tests` (39.81 s, the
  aggregate registration that now carries the exclude argument with no such
  case compiled), `iom_rocm_sdpa_nonmatrix_tests` (0.12 s), and
  `iom_backend_coexistence_tests` (0.24 s) — `100% tests passed, 0 tests failed
  out of 4`, total `41.17 s`, `test_rocm passed`. The default configuration
  therefore needs no model artifact, no reference pack, no arena, and no
  Python model library, and the ROCm suite is unchanged by this leaf.

