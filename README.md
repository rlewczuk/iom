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

Run:

```sh
./build/iom
```

Test:

```
cmake --build build --target iom_tests
ctest --test-dir build --output-on-failure
```
