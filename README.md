# Inference of Oversized Models

Installing dependencies (Ubuntu):

```
sudo apt install build-essential cmake doctest-dev doxygen clangd lldb-20 nlohmann-json3-dev
```

Install ROCm 7.2 or newer separately when accelerator builds are needed. The
ROCm SDK must provide its HIP CMake package, normally under `/opt/rocm`.

Install the CUDA Toolkit separately when NVIDIA accelerator builds are
needed. The default toolkit path is `/usr/local/cuda`.

Build with CUDA support:

```sh
cmake -S . -B build/cuda \
  -DBUILD_TESTING=ON \
  -DCUDA_ENABLED=ON \
  -DROCM_ENABLED=OFF \
  -DCUDA_PATH=/usr/local/cuda
cmake --build build/cuda --target iom_cuda
```

Set `-DCUDA_PATH=/path/to/cuda` when the toolkit is installed elsewhere.
Enabling CUDA fails configuration if the toolkit or its driver library is
unavailable; it does not silently disable the backend. The CUDA smoke test
requires a usable NVIDIA GPU/runtime and does not skip when CUDA is enabled:

```sh
cmake --build build/cuda --target iom_cuda_smoke_tests
ctest --test-dir build/cuda --output-on-failure \
  -R '^iom_cuda_smoke_tests$'
```
Build:

```sh
cmake -S . -B build
cmake --build build
```

Build with ROCm support:

```sh
cmake -S . -B build/rocm \
  -DBUILD_TESTING=ON \
  -DROCM_ENABLED=ON \
  -DCUDA_ENABLED=OFF \
  -DROCM_PATH=/opt/rocm
cmake --build build/rocm --target iom_rocm
```

Set `-DROCM_PATH=/path/to/rocm` when ROCm is installed elsewhere. Enabling
ROCm fails configuration if the SDK or its HIP package is unavailable; it does
not silently disable the backend. The ROCm smoke test requires a usable AMD
GPU/runtime and does not skip when ROCm is enabled:

```sh
cmake --build build/rocm --target iom_rocm_smoke_tests
ctest --test-dir build/rocm --output-on-failure \
  -R '^iom_rocm_smoke_tests$'
```

## Build with all retained backends together

`CUDA_ENABLED`, `ROCM_ENABLED`, and `SYCL_ENABLED` are independent: no
option disables another, and each controls only its own library,
dependencies, and tests. `libiom` always contains the common code and the
CPU backend, so any combination of CUDA, ROCm, and SYCL accelerator options
configures and builds alongside it:

```sh
cmake -S . -B build/all \
  -DBUILD_TESTING=ON \
  -DCUDA_ENABLED=ON \
  -DROCM_ENABLED=ON \
  -DSYCL_ENABLED=ON \
  -DCUDA_PATH=/usr/local/cuda \
  -DROCM_PATH=/opt/rocm
cmake --build build/all --target iom_backend_coexistence_tests
```

The coexistence test links `libiom` and every enabled backend library into
one executable, constructs devices and queues from CPU, CUDA, ROCm, and SYCL
in one process, interleaves ADD (representative `I32`/`F32`) and asynchronous
`BF16` copy work across the queues, waits on each originating queue, and
compares bit-identical logical results. It fails (never skips) when an
enabled backend has no usable device:

```sh
ctest --test-dir build/all --output-on-failure \
  -R '^iom_backend_coexistence_tests$'
```

## TinyLlama generation CLI

`iom_generate` performs one explicit TinyLlama generation request. It never
chooses a model directory from the current directory or an environment
variable, downloads a checkpoint, or uses a hardcoded path. Every invocation
must name the model directory, backend, device ordinal, and generation limit,
and must provide exactly one raw prompt or one or more ordered chat messages:

```text
iom_generate --model-dir DIR --backend cpu|cuda|rocm|sycl \
  --device ORDINAL --max-new-tokens N --prompt TEXT

iom_generate --model-dir DIR --backend cpu|cuda|rocm|sycl \
  --device ORDINAL --max-new-tokens N \
  --message ROLE CONTENT [--message ROLE CONTENT ...]
```

CPU accepts only `--device 0` and does not use a tensor arena option. CUDA,
ROCm, and SYCL require an explicit `--tensor-arena-bytes N` value in addition
to the options above; it must be nonzero and divisible by 32:

```text
iom_generate --model-dir ./tinyllama --backend cuda --device 0 \
  --tensor-arena-bytes 1073741824 --max-new-tokens 32 \
  --prompt "Write one sentence."

iom_generate --model-dir ./tinyllama --backend cpu --device 0 \
  --max-new-tokens 32 --message system "Be concise." \
  --message user "Summarize this input."
```

On success, stdout contains only the owned decoded generated text. The CLI
does not add a diagnostic label or a trailing newline. Diagnostics are written
to stderr. Exit status 2 denotes usage or input validation (including
malformed numbers, conflicting forms, and unsupported chat roles), 3 denotes
backend, model, tokenizer, allocator, device, or session setup/load failure,
and 4 denotes queue, generation, result, token-selection, or output-execution
failure. The normal EOS, maximum-token, and context-capacity stop reasons all
return status 0.

### Session input and request boundaries

`--prompt` is the raw-input path: the value is passed directly to the
session-owned tokenizer and bypasses chat formatting. Repeated
`--message ROLE CONTENT` values are ordered structured input: the session
renders them with the selected chat template, appends its generation prompt,
and then passes the rendered bytes to the tokenizer. At the formatter
boundary, an engaged template override is authoritative, even when it is
empty or invalid; it never falls back to `chat_template`. With no engaged
override, the model's configured `tokenizer_config.json` `chat_template` is
selected.

The session owns the model, tokenizer, formatter, selector, KV/cache storage,
and request storage. Final-logit selection is synchronous: the queue, the
borrowed logits view and its owner, the complete history, and caller-provided
scratch must remain live and unchanged through the call, and the selector
retains none of them. History contains the full prompt (including any
policy-added special tokens) followed by successfully committed generated
IDs. A successful result owns its returned vectors and text: generated
`token_ids` retain a committed EOS ID, while decoded `text` skips the
special IDs `0`, `1`, and `2`.

Every selected ID is committed before stopping. Stop precedence is
`EOS > max-new-tokens > context capacity`. The generated-token count and
initialized KV length are distinct: a nonterminal token is processed by the
next decode only after both K/V appends succeed, while a terminal token
remains in the result and history without a KV append merely for bookkeeping.
With `--max-new-tokens 0`, the prompt is still validated, tokenized, and
provisioned, but no forward or selector call is made; the result is empty and
the request stops at the new-token limit.

Starting a second request first drains every accepted operation from the
current request. Only a successful drain permits a fresh request to be
published, with a reset cache prefix and new history; a previously returned
result remains owned independently of that reset. If any accepted operation
reports a retained failure, the old request remains owned, the session is
poisoned, and no replacement is published. A failed selection or out-of-range
selected ID likewise commits no token; poisoning then rejects later work while
drain still attempts every accepted operation and preserves the first failure.
The ownership and cache boundaries are specified by [final-logit selection
and ownership](docs/BACKEND_CONTRACT/tinyllama-forward-layout-final-logits-selection-and-ownership.md#tinyllama-forward-layout--final-logits-selection-and-ownership)
and [session sizing and lifetime](docs/BACKEND_CONTRACT/tinyllama-forward-layout-session-sizing-and-lifetime.md#forward-stores-and-live-ranges),
including its [producer schedule, cache publication, and abort](docs/BACKEND_CONTRACT/tinyllama-forward-layout-session-sizing-and-lifetime.md#producer-schedule-cache-publication-and-abort).

Existing behavioral evidence covers these boundaries: [chat composition
tests](test/test_chat_format.cpp) (`Chat format composes raw and structured
owner paths` and `Chat format composes formatter override and generation
policy`); [session tests](test/test_model_session.cpp) (`TinyLlama text chat
generation composes raw tokenizer input`, `TinyLlama text chat generation
renders roles and assistant prefix`, `TinyLlama text chat generation forwards
every stop reason`, `TinyLlama text chat generation owns results across session
reuse`, `TinyLlama generation state machine applies EOS and limit precedence`,
`TinyLlama generation state machine commits history and decodes only for
nonterminal tokens`, and `TinyLlama generation state machine rejects selector
failures without reuse`); and [selector lifetime tests](test/test_token_selection.cpp)
(`token selection readiness and lifetime rejects invalid producer OIDs before
effects`, `token selection readiness and lifetime waits for deferred producer
and releases borrowed inputs`, and `token selection readiness and lifetime
retains failures and reuses scratch`). The CLI process case `TinyLlama
generation CLI runs raw and structured chat process forms`, the selector
injection/no-commit case, and `TinyLlama synthetic session integration
recovers after accepted failure at the poison boundary` provide the remaining
raw/chat, no-commit, and drain evidence; these are existing executable tests,
not a second harness.

Use `--trace` to opt into one human-readable operation line per accepted
positive OID:

```text
iom_generate --model-dir ./tinyllama --backend cpu --device 0 \
  --max-new-tokens 32 --prompt "Write one sentence." --trace
```

Trace lines are written only to stderr; generated text on stdout remains
byte-for-byte identical, including its lack of an added trailing newline.
Each line reports the request ordinal, positive OID, inference phase, decoder
layer (`none` for embedding and final-logit operations), absolute input
position and run length, host enqueue elapsed nanoseconds, and the elapsed
host time from enqueue begin to the first observed wait when one exists.
It also reports the wait state (`not_observed`, `succeeded`, or `failed`) and
always identifies device timing as `unavailable`. A failed wait is an
observation of the retained failure, not proof that the native operation
completed.

Trace storage is reserved explicitly after session load and before the first
generation request. A reservation failure is a setup failure (status 3) and
does not fall back to untraced generation. The recorder remains alive through
ordinary session destruction so the existing final drain can update rows; the
trace report never adds a wait. `--trace` is independent of `--metrics`, and
trace alone emits no metrics summary. With no observation option, no recorder
or trace storage is created.

### Optional scalar metrics

Pass the value-free `--metrics` option to print one bounded scalar report on
stderr. The generated text remains the only stdout payload and is byte-for-byte
unchanged, including its lack of a trailing newline:

```text
iom_generate --model-dir ./tinyllama --backend cpu --device 0 \
  --max-new-tokens 32 --prompt "Write one sentence." --metrics
```

The report includes explicit `ns` units for load, encode/tokenization, the
completion-observed prefill and decode phases, and each phase's host-enqueue
sum. It also reports prompt, generated, decode-forward, and decode-token
counts; TTFT; the decode-throughput numerator, denominator, and tokens/second;
and the observed request status and stop reason. A phase that did not run, a
zero-token or zero-denominator result, an incomplete or failed rate, and TTFT
without a committed token are printed as `unavailable`. Current CPU, CUDA,
ROCm, and SYCL configurations have no genuine native device timer, so
`device_time=unavailable` is always explicit. Host enqueue and completion
spans MUST NOT be read as kernel or device time, and no throughput floor is
implied.

The load interval excludes outer device/allocator setup and caller selector
construction. Encode/tokenization covers only the tokenizer's encode call:
chat rendering and output decoding are outside it. Prefill and decode complete
at the existing final-logits readiness waits, while host enqueue is the sum of
the individual facade-call intervals, including blocking inside those calls.
The report does not prepare operation-trace storage, add a wait, or change the
existing stop behavior and exit statuses. Duplicate or value-bearing forms of
`--metrics` remain usage errors (status 2); setup/load and execution failures
retain their existing status and primary diagnostic while the report exposes
only observations actually collected.

## CUDA CLI samples

These sample scripts are repository-root examples for the configured CUDA host
`bv1`; they do not download a model or provide remote orchestration. On `bv1`,
`/home/rlew/cuda_env.sh` supplies the CUDA toolkit environment. From the IOM
checkout root, run:

```sh
./samples/build-cuda.sh
./samples/run-cuda.sh
```

The build script configures a Release `build-cuda` tree with testing disabled,
CUDA enabled, and ROCm and SYCL disabled, then builds only `iom_generate`. The
run script loads `/home/rlew/models/TinyLlama-1.1B-Chat-v1.0` on CUDA device 0
with a 4 GiB tensor arena and up to 16 new tokens. It supplies the unchanged
question `What is the capital of Poland ?` as a user message so the production
CLI applies TinyLlama's official chat formatting and assistant-generation
prefix; raw `--prompt` behavior remains unchanged. A successful normal stop
exits with status 0 and writes only generated text to stdout; wording may vary.
Diagnostics go to stderr. Usage/input errors exit 2, setup/load errors exit 3,
and execution errors exit 4. These fixed paths and settings describe only the
configured `bv1` example and are not library or CLI defaults.

## Elementwise operation contract

`DeviceOps` exposes four exact three-view asynchronous facades:

```cpp
oid add(const TensorView&, const TensorView&, TensorView&) noexcept;
oid mul(const TensorView&, const TensorView&, TensorView&) noexcept;
oid sub(const TensorView&, const TensorView&, TensorView&) noexcept;
oid div(const TensorView&, const TensorView&, TensorView&) noexcept;
```

ADD, MUL, and SUB accept `QuantizationFormat::NONE` and the 21 numeric leaves
`I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,
F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`. DIV accepts only the nine floating
leaves `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`.
BOOL, F8_E8M0, non-NONE quantization, and integer DIV are `Unsupported`;
matching-type validation happens first. There is no promotion, public query,
fallback selector, or SDK dtype narrowing.

Operations use right-aligned broadcasting (with `[1,1]` as the scalar
convention), transformed leading views, tiled tails without padding reads, and
exact in-place aliases only; read/read overlap is valid. Integer MUL/SUB are
modulo `2^w`; floating operations decode, compute once in extended precision,
and encode once with RNE, gradual underflow, format-specific special-value
rules, and a one-ULP finite envelope. Operand order is `lhs-rhs` and `lhs/rhs`.
Positive OIDs are accepted in-order queue work (CPU may complete inline);
waits are repeatable and retained failures rethrow. Callers provide stable
storage and owners—operations never allocate or relocate operands/results.
SUB and DIV are additive APIs, while valid MUL now accepts work; rebuild
consumers and do not mix header/library versions (no mixed-version ABI).

Run:

```sh
./build/iom
```

Test:

```
cmake --build build --target iom_tests
ctest --test-dir build --output-on-failure
```
