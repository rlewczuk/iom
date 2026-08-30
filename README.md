# Inference of Oversized Models

Installing dependencies (Ubuntu):

```
sudo apt install build-essential cmake doctest-dev doxygen clangd lldb-20 nlohmann-json3-dev
```

Install ROCm 7.2 or newer separately when accelerator builds are needed. The
ROCm SDK must provide its HIP CMake package, normally under `/opt/rocm`.

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
