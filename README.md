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

Build with TTNN support:

```sh
cmake -S . -B build/ttnn \
  -DBUILD_TESTING=ON \
  -DTTNN_ENABLED=ON \
  -DCUDA_ENABLED=OFF \
  -DROCM_ENABLED=OFF
cmake --build build/ttnn --target iom_ttnn
```

The Tenstorrent SDK must be installed system-wide and expose its `tt-nn` and
`tt-metalium` CMake packages (`TT::Metalium`, `TTNN::TTNN`). Enabling TTNN
fails configuration when those packages are absent; it does not silently
disable the backend. TTNN owns native storage and uses internal staging or
emulation for non-native numeric leaves while preserving public logical shape
and ownership.
The supported storage leaves include `BOOL` and all 21 required numeric
`QuantizationFormat::NONE` leaves: `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`,
`U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`,
`F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`. TTNN need not store
`F8_E8M0`; grouped quantization remains invalid. ADD excludes `BOOL` and
`F8_E8M0`, and exposes no capability query: the exact three-view
`noexcept` `add` facade returns a positive token or negative `OidError`.
smoke test requires a usable Tenstorrent device/runtime and does not skip
when TTNN is enabled:

```sh
cmake --build build/ttnn --target iom_ttnn_smoke_tests
ctest --test-dir build/ttnn --output-on-failure \
  -R '^iom_ttnn_smoke_tests$'
```

The conformance test runs the shared storage, host transfer, asynchronous
copy, error, lifetime, and capability suite against the CPU reference and a
hardware-backed Tenstorrent device; it fails (never skips) when no device is
available:

```sh
cmake --build build/ttnn --target iom_ttnn_conformance_tests
ctest --test-dir build/ttnn --output-on-failure \
  -R '^iom_ttnn_conformance_tests$'
```

## Build with every backend together

`CUDA_ENABLED`, `ROCM_ENABLED`, and `TTNN_ENABLED` are independent: no
option disables another, and each controls only its own library,
dependencies, and tests. `libiom` always contains the common code and the
CPU backend, so any combination of accelerator options configures and
builds alongside it:

```sh
cmake -S . -B build/all \
  -DBUILD_TESTING=ON \
  -DCUDA_ENABLED=ON \
  -DROCM_ENABLED=ON \
  -DTTNN_ENABLED=ON \
  -DCUDA_PATH=/usr/local/cuda \
  -DROCM_PATH=/opt/rocm
cmake --build build/all --target iom_backend_coexistence_tests
```

The coexistence test links `libiom` and every enabled backend library into
one executable, constructs devices and queues from every backend in one
process, interleaves ADD (representative `I32`/`F32`) and asynchronous `BF16`
copy work across the queues, waits on each originating queue, and compares
bit-identical logical results. It fails (never skips) when an enabled backend
has no usable device:

```sh
ctest --test-dir build/all --output-on-failure \
  -R '^iom_backend_coexistence_tests$'
```

Run:

```sh
./build/iom
```

Test:

```
cmake --build build --target iom_tests
ctest --test-dir build --output-on-failure
```
