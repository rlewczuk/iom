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
