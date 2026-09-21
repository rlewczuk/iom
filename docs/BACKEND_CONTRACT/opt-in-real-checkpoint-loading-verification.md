# Opt-in real-checkpoint loading verification

The synthetic four-driver integration above stays required and unchanged. In
addition, one explicit, default-off check loads a caller-supplied official
checkpoint on the selected device of each retained backend conformance
executable. `test/CMakeLists.txt` owns the option and defines
`IOM_TEST_REAL_MODEL_LOADING=1` for those four executables only:

```cmake
option(IOM_TEST_REAL_MODEL_LOADING
    "Compile the opt-in real-checkpoint loading cases into the existing backend conformance tests"
    OFF
)
```

With the default `OFF`, an ordinary configuration compiles and runs the whole
suite with no model artifact, no hardcoded path, and no download: the entire
real-loading section of `test/backend/backend_conformance_model_loading.hpp` is
compiled out, so no environment read, checkpoint path, or real-loading code
exists in that build. With the option `ON`, each existing executable compiles
exactly one additional case, and no new executable, harness, or test target is
created:

| Backend | Existing executable | Added case |
| --- | --- | --- |
| CPU | `iom_backend_conformance_cpu_tests` | `CPU real model loading` |
| CUDA | `iom_cuda_conformance_tests` | `CUDA real model loading` |
| ROCm | `iom_rocm_conformance_tests` | `ROCm real model loading` |
| SYCL | `iom_sycl_conformance_tests` | `SYCL real model loading` |

**Shared case.** The one case is
`iom_conformance::run_real_model_loading(iom::Device&)` and the standard-GPU
environments are read by `iom_conformance::real_model_memory_config()`, which
returns the exact `iom::DeviceMemoryConfig` the driver passes to its factory,
because the arena is a device-construction value. The production model API of
the previous subsections is unchanged, and this check adds no loader, selector,
digest, downloader, or device-selection API.

**Environment intake.** Nothing is defaulted, and no directory is searched:

| Variable | Required | Rule |
| --- | --- | --- |
| `IOM_TEST_MODEL_DIR` | every backend | Nonempty path of the caller-supplied official checkpoint directory; the case requires an existing directory and reads exactly that directory. |
| `IOM_TEST_MODEL_ID` | every backend | Nonempty pinned identity of that artifact (a Hugging Face revision or a digest-manifest identity); recorded verbatim in the evidence block and never interpreted by the loader. |
| `IOM_TEST_MODEL_ARENA_BYTES` | CUDA, ROCm, SYCL | Positive decimal byte count divisible by 32, passed through `DeviceMemoryConfig.tensor_arena_bytes`; it must cover the checked aggregate standard weight total plus the queried scratch maximum, and the synthetic conformance arena is deliberately not reused. |
| `IOM_TEST_MODEL_SHA256` | optional | The externally collected `sha256sum` manifest of that directory, recorded verbatim; when it is absent the evidence block records the exact collection command instead. |

**Fixed order.** The case reads the environment, loads and validates the
checkpoint, checks the pinned identity and complete inventory, computes the
checked aggregate standard tiled weight total, rejects an arena below that
total before any data tensor exists, creates one destination per published role
from `tensor_spec(index)` on the selected device, queries the real maximum
serial workspace requirement of that binding, rejects an arena below the weight
total plus that queried maximum, provisions caller scratch only when the
requirement is positive, and then synchronously uploads every role. Only the
normal `void` return of that upload is success: there is no readback,
computation, logits, generation, token, or matrix-performance claim, and no
partial model is loaded to fit memory.

**Pinned reference.** The case validates `N22, H2048, I5632, Hq32, Hkv4, D64,
V32000, C2048` and exactly `3 + 9*N = 201` required BF16 roles against an
expectation encoded independently of the loader: the globals `token_embedding`
`[V, H]`, `final_norm` `[1, H]` from its rank-one `[H]` source, and untied
`lm_head` `[V, H]`, plus the nine per-layer roles `input_norm` and
`post_attention_norm` `[1, H]`, `query` `[H, H]`, `key` and `value`
`[Hkv*D, H]`, `attention_output` `[H, H]`, `mlp_gate` and `mlp_up` `[I, H]`,
and `mlp_down` `[H, I]`, each BF16/NONE with `2 * product(logical shape)`
logical bytes. No subset is accepted, and the identity is settled before the
first destination exists.

**Failure policy.** There is no skip, fallback, default, download, empty
success, or synthetic substitute on this path. A missing or non-directory
`IOM_TEST_MODEL_DIR` fails the case before loading; an empty `IOM_TEST_MODEL_ID`
fails it before loading; a malformed or wrong checkpoint keeps the established
schema, container, overflow, and allocation categories of this section; a
missing, nonnumeric, zero, misaligned, or too-small `IOM_TEST_MODEL_ARENA_BYTES`
fails before the first data tensor exists, and any later native exhaustion
(`std::bad_alloc` from destination creation or scratch provisioning) propagates
unchanged; an unavailable enabled device fails its factory; and a synchronous
upload failure keeps that backend's established category.

**Ownership and storage.** Only the driver's selected device is used: CPU
creates its destinations through the driver's own allocator, and the retained
accelerators use the caller-selected arena or queried workspace. No reference
or foreign device receives a second copy of the full inventory, no equally
large reference payload is built, and the source, the destinations, and any
scratch stay alive through every call.

**Evidence record.** Each case prints one evidence block to the run's standard
output: the exact invocation (the process argv), the backend kind, the backend
ordinal, the `Device` instance identity, the model directory, the pinned
identity, the artifact manifest or its exact collection command, the arena bytes
or `native storage`, the required role count and dtype, the checked aggregate
standard weight total, the queried scratch requirement, and the `PASS`/`FAIL`
result, which reports failure unless the synchronous upload returned. The
manifest is collected outside the test and retained with that block:

```text
sha256sum "$IOM_TEST_MODEL_DIR/config.json" "$IOM_TEST_MODEL_DIR"/*.safetensors
```

**Invocation.** Configure with the existing SDK arguments and the option, then
build and run that host's existing conformance target; the focused filter must
report a nonempty selection. Reconfigure with `-DIOM_TEST_REAL_MODEL_LOADING=OFF`
for the ordinary suite, which needs none of these variables.

| Backend | Build target | Focused enabled runtime command |
| --- | --- | --- |
| CPU | `iom_backend_conformance_cpu_tests` | `./build/test/iom_backend_conformance_cpu_tests --test-case="*real model loading*"` |
| CUDA | `iom_cuda_conformance_tests` | `./build/test/iom_cuda_conformance_tests --test-case="*real model loading*"` |
| ROCm | `iom_rocm_conformance_tests` | `./build/test/iom_rocm_conformance_tests --test-case="*real model loading*"` |
| SYCL | `iom_sycl_conformance_tests` | `./build/test/iom_sycl_conformance_tests --test-case="*real model loading*"` |

All accelerator commands execute only in their synchronized remote mirrors
through `skill://csw-remote`, with the SYCL toolchain initialization rules above
unchanged. Missing artifacts, missing hardware, or an unrun required gate
prevents closure; none of them is a skip reason.
