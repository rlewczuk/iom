# CPU loading conformance and observable inventory

`test/backend/backend_conformance_model_loading.hpp` owns the one shared,
backend-neutral loading scenario,
`iom_conformance::run_model_loading_conformance(const ConformanceDevices&)`. It
is the only model-loading conformance surface: it contains no backend-kind
switch, no accelerator header, no failure-injection seam, and no second
inventory generator, so every retained backend driver calls it unchanged with
its own devices, in the CPU, CUDA, ROCm, and SYCL integration order.
`test/cpu/test_cpu_conformance.cpp` registers it as
`iom_backend_conformance_cpu_tests` case `CPU model loading*`, which is the CPU
invocation:

```text
cmake --build build --target iom_backend_conformance_cpu_tests
./build/test/iom_backend_conformance_cpu_tests --test-case="CPU model loading*"
```

The CPU case loads four independent synthetic checkpoints through
`load_tinyllama_config` and `load_tinyllama_safetensors` only: the two-layer
`N=2, H=8, I=12, Hq=4, Hkv=2, D=2, V=19, C=17` boundary, its one-layer form,
and the H16 `N=1, H=16, I=20, Hq=4, Hkv=2, D=4` and H18
`N=1, H=18, I=22, Hq=3, Hkv=1, D=6` normalization/tile-boundary forms. The
observable inventory is exactly `3 + 9*N` selected roles — 12 for each one-layer
checkpoint and 21 for the two-layer one — with the three globals
`token_embedding` `[V, H]`, `final_norm` `[1, H]`, `lm_head` `[V, H]` and the
nine per-layer roles `input_norm` and `post_attention_norm` `[1, H]`, `query`
`[H, H]`, `key` and `value` `[Hkv*D, H]`, `attention_output` `[H, H]`,
`mlp_gate` and `mlp_up` `[I, H]`, and `mlp_down` `[H, I]`, each selected
`TensorSpec` being BF16/NONE with `2 * product(logical shape)` logical bytes.

For every checkpoint the case creates one destination per published index from
`tensor_spec(index)`, queries `upload_workspace_requirements`, provisions
reusable scratch only when the reported maximum is positive, uploads through
`upload_weights`, and reads every destination back with real `TensorView`
transfers whose separately queried download requirement is never assumed from
the upload one. The CPU binding reports `{0, 1}` and therefore realizes with the
default empty view, and no positive raw-workspace factory call is made on a
device that rejects one. Exact BF16 bits of every role are compared with the
independent fixture bytes; there is no numeric tolerance, no CPU-computation
oracle, and no persistent second whole-checkpoint host bank. The same source is
realized on the driver's CPU reference device and on the selected device, and
the realized destinations stay readable after the fixture artifacts are deleted
and the source is released.

The case also observes the failure policy of this section: valid extra tensors,
a second shard, and unrelated documents change no selected result; an absent
final required role, a wrong final role schema, a transposed final destination,
a foreign-`Device` destination, a null final destination, a count one short or
one long, and one owner bound to two equal-metadata roles all preserve the
seeded destination sentinels with no published source or upload.

# CUDA loading conformance and observable inventory

The CUDA driver adds no loader port and no second scenario:
`test/cuda/test_cuda_conformance.cpp` registers the `CUDA model loading*` case
of the existing `iom_cuda_conformance_tests` target, whose conditional
registration is unchanged, and calls
`iom_conformance::run_model_loading_conformance(devices.conformance())`
unchanged with its own devices. The checkpoints, selected inventory, ordered
destination binding, workspace ordering, byte comparison, and failure policy are
exactly the CPU contract above; this subsection records only what is
CUDA-specific.

**CUDA setup.** The case opens with `REQUIRE(cuInit(0) == CUDA_SUCCESS)` and
constructs the driver's `CudaDevices`: a CPU device owning the driver's checking
host allocator as the comparison reference, the selected CUDA device on ordinal
0 as the candidate, and a distinct second CUDA `Device` instance as the foreign
destination owner, each with the driver's synthetic `DeviceMemoryConfig` arena.
Both model-loading devices are real CUDA runtime, storage, and allocator
machinery, and the same published source realizes on the CPU reference and on
the CUDA candidate bit-for-bit. BF16 storage capability is required rather than
a skip reason: the shared scenario creates every destination from the published
BF16 specifications and its binding preflight rejects a device whose
`supported_data_types()` omits BF16, so a CUDA configuration without BF16 fails
the case.

**CUDA workspace.** A CUDA destination reports the positive staging requirement
of its own `host_transfer_workspace_requirements`, so the complete binding
reports the maximum serial per-owner requirement, the caller provisions real
`Device::create_workspace` scratch from that reported maximum, and the empty
default view is refused before the first copied role with every destination byte
left intact. The `{0, 1}` zero-workspace policy stays the behavior of branches
whose own requirement is zero; CUDA neither subdivides nor extends it.

**CUDA limitations.** No per-backend loader, CUDA-specific expectation,
backend-kind switch, or CUDA runtime header enters the common scenario, and the
expected bytes stay the fixture's own independent role bytes.
`CudaStorageOracle` remains a native-layout diagnostic of the existing storage
and transfer cases and is never a model-loading expected-byte generator. No
host-transfer fault injection is introduced: the existing CUDA
queued-operation injection seams prove queue submission behavior, not a
synchronous loader failure, so this case asserts no loader runtime failure
category of its own and the established CUDA and container categories of this
section stay unchanged.

```text
cmake --build build --target iom_cuda_conformance_tests
./build/test/iom_cuda_conformance_tests --test-case="CUDA model loading*"
```

# ROCm loading conformance and observable inventory

ROCm adds no loader and no second scenario. The driver registers the shared
CPU-first case of
[CPU loading conformance and observable inventory](#cpu-loading-conformance-and-observable-inventory)
unchanged on this backend's own devices, so every common rule — the published
inventory, destination creation from `tensor_spec`, the complete-binding
preflight, the upload and download workspace and ownership rules, the rejection
policy, and source lifetime — is the shared one and is not restated here.
`test/rocm/test_rocm_conformance.cpp` registers it as the
`iom_rocm_conformance_tests` case `ROCm model loading*`, and the driver's custom
doctest `main` is unchanged. The focused selections are:

```text
cmake --build build --target iom_rocm_conformance_tests
./build/test/iom_rocm_conformance_tests --test-case="ROCm model loading*"
ctest --test-dir build --output-on-failure -R '^iom_rocm_conformance_tests$'
```

Setup follows this driver's existing conformance convention. The CPU reference
runs over a caller `LinearAllocator`; the candidate and the foreign device are
two independent `make_rocm_device(0, …)` instances whose `DeviceMemoryConfig`
reserve is the caller-selected tensor-data arena. The foreign role is therefore
a distinct `Device` instance of the same ordinal, so exact `Device` identity
rejects a foreign destination — never an equal backend and ordinal.

ROCm host transfers need real positive scratch, so this instantiation exercises
the positive half of the workspace policy rather than the CPU `{0, 1}` half:
each destination view reports `{compute_staging_size(logical_nbytes), 32}`, the
complete binding reports that same maximum from
`upload_workspace_requirements`, the case provisions exactly one caller-owned
`create_workspace` range from that result, and that range is suballocated from
the reserved data arena with no new native backing. The empty default scratch is
refused as `std::invalid_argument` before the first copied role with every
seeded destination sentinel intact, and the readback requirement is queried
independently of the upload one.

Limitations. BF16 is a mandatory capability of the real device and never a skip
condition. This case uses no diagnostic seam: `HipStorageOracle` is diagnostic
only, and queued `inject_submission_fault_for_testing` faults cannot prove a
synchronous loader failure, so neither belongs to loading coverage. Real
accelerator build and execution are remote-only, and closure requires the
nonempty focused selection above on the actual enabled ROCm device together with
the full `iom_rocm_conformance_tests` suite.

# SYCL loading conformance and observable inventory

`test/sycl/test_sycl_conformance.cpp` registers the same shared scenario as
`iom_sycl_conformance_tests` case `SYCL model loading*`, which is the fourth
invocation of the CPU-first case above, after the ROCm case:

```text
cmake --build build --target iom_sycl_conformance_tests
./build/test/iom_sycl_conformance_tests --test-case="SYCL model loading*"
```

The case is the existing `SyclDevices` fixture, unchanged and not duplicated:
one independent CPU reference device, the selected eligible SYCL device as
candidate, and a second distinct SYCL device instance at the same ordinal as
the foreign device, whose owned context differs from the candidate's, so the
foreign-destination rejection is judged against the exact candidate instance
rather than the ordinal. Every destination is created on the candidate from
`ModelSource::tensor_spec(index)`, the binding preflight is compared with the
real maximum serial per-owner `copy_from_host` requirement reported by those
destination views, positive scratch is provisioned from that exact candidate
device through `create_workspace` and used as the upload workspace, and the
CPU reference realizes the same binding through its zero-byte `{0, 1}` path.
Both workspace policies are therefore observed by actual queries rather than a
backend-kind switch, and readback is a real `TensorView` transfer with a
separately queried download requirement.

SYCL setup and limitations:

- SYCL execution is remote-only. Configure with the actual `SYCL_ENABLED` flag
  and the existing SDK arguments under a profile override whose `REMOTE_SETUP`
  is empty; before every build and test run execute
  `set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u`,
  preserve that environment, and run `sycl-ls` to confirm a Level Zero GPU.
- BF16 storage is mandatory and the fixture requires an eligible device: an
  unavailable Level Zero GPU, or a device that cannot hold the required BF16
  roles, is a failure, never a skip.
- The loader stays backend-neutral: SYCL types remain backend-private, no
  accelerator header enters common code, `model.cpp` keeps its internal
  staging, and `src/sycl/device_tensor.cpp` and `src/sycl/copy.cpp` change only
  if a loading failure demonstrates a transfer defect.
