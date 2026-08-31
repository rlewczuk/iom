# Repository Guidelines

## Project Overview

IOM (Inference of Oversized Models) is an inference-only C++20 engine for sparse, oversized language models. It targets low host-RAM/VRAM usage through tiled tensors, memory-hierarchy-aware placement, speculation, and multi-precision sparse-expert weights. Computation is imperative; there is no autograd graph. The top-level design source is `docs/design/README.md`.

The repository is currently infrastructure and backend focused. `src/main.cpp` is the executable entry point and currently prints `Hello world!`; model execution is represented by the `iom::models` scaffolding.

## Architecture & Data Flow

- `include/iom/device.hpp` defines the backend-neutral `Device` contract. A device reports its backend/ordinal and creates `Tensor` owners and `DeviceOps` queues. Each backend exposes its own factory; there is no global active-backend registry. The creating `Device` must outlive its tensors and queues.
- `include/iom/tensor.hpp` defines `TensorShape`, `TensorSpec`, `TensorView`, and `Tensor`. Tensor metadata stays on the host; storage may be host or device resident. The final two dimensions use a fixed `16x16` tiled layout; leading dimensions are row-major and support views. `TensorView` is non-owning and supports leading-dimension `slice`, `select`, `permute`, and `reshape_leading`.
- `include/iom/iom.hpp` defines `DeviceOps`: an in-order asynchronous queue over caller-created views. Tokens pack a process-unique queue id and a monotonic 56-bit submission sequence. `wait()` is repeatable and rethrows retained asynchronous failures.
- Tensor operations do not allocate operand or output tensors. Callers create outputs and provide host buffers. Preserve this invariant when adding operations or backend copies.
- CPU, CUDA, and ROCm use the standard tiled layout and a caller-supplied `iom::Allocator`. TTNN owns native per-plane storage and has an explicit supported-leaf-type table (`include/iom/ttnn/device.hpp`). Common headers must not include CUDA, HIP, SYCL, or TTNN types.
- Model/data flow is `MappedFile` -> SafeTensors view/store -> caller-created tensors -> `DeviceOps` calls. See `include/iom/mmap.hpp`, `include/iom/safetensors.hpp`, and `include/iom/llama.hpp`. The current Llama implementation is scaffolding; compute methods are capability stubs until a backend implements them.

## Key Directories

- `include/iom/` — public backend-neutral APIs and backend factory headers.
- `src/` — core implementation (`iom.cpp`, allocation, mmap, SafeTensors, Llama) and the executable; `src/{cpu,cuda,rocm,ttnn}/` contains backend implementations and copy kernels.
- `test/` — doctest sources; `test/backend/backend_conformance.hpp` is the shared backend-neutral conformance harness; backend-specific tests live under `test/{cpu,cuda,rocm,ttnn}/`.
- `docs/design/` — architectural design notes. `docs/changes/0001-tensor-view/` contains the active tensor/view/queue contract and numbered implementation specs.
- `.agents/skills/remote-development/` — required workflow for GPU builds/tests; `.agents/skills/spec-tasks/` and `spec-critic/` describe the change-spec workflow.

## Development Commands

Install Ubuntu dependencies:

```sh
sudo apt install build-essential cmake doctest-dev doxygen clangd lldb-20 nlohmann-json3-dev
```

CPU/default build and run:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
./build/iom
ctest --test-dir build --output-on-failure
```

Optional backends are fail-fast: enabling one requires its SDK; CMake does not silently disable it.

```sh
# CUDA
cmake -S . -B build/cuda -DBUILD_TESTING=ON \
  -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda
cmake --build build/cuda --target iom_cuda
cmake --build build/cuda --target iom_cuda_smoke_tests
ctest --test-dir build/cuda --output-on-failure -R '^iom_cuda_smoke_tests$'

# ROCm/HIP (7.2+)
cmake -S . -B build/rocm -DBUILD_TESTING=ON \
  -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm
cmake --build build/rocm --target iom_rocm
cmake --build build/rocm --target iom_rocm_smoke_tests
ctest --test-dir build/rocm --output-on-failure -R '^iom_rocm_smoke_tests$'

# Tenstorrent TTNN
cmake -S . -B build/ttnn -DBUILD_TESTING=ON \
  -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF
cmake --build build/ttnn --target iom_ttnn
cmake --build build/ttnn --target iom_ttnn_smoke_tests
ctest --test-dir build/ttnn --output-on-failure -R '^iom_ttnn_smoke_tests$'
```

Use `CMAKE_EXPORT_COMPILE_COMMANDS=ON` (enabled by default) for tooling. No standalone lint or formatter target is defined in the repository; preserve the existing C++ style and rely on compiler/test diagnostics.

## Code Conventions & Common Patterns

- Use C++20, standard library facilities, and namespaces rooted at `iom`; CMake disables compiler extensions.
- Treat owning types as explicit lifetime boundaries. `Tensor` and `DeviceOps` are non-copyable/non-movable; use `std::unique_ptr` for device-created objects and keep `Device`/allocator lifetimes valid until all dependent objects are destroyed.
- Use `std::span` for borrowed host buffers and tensor-view inputs. Views may be copied but must never be retargeted; they reference an owner’s stable storage handle.
- Validate shape, dtype, quantization, span sizes, device identity, and overflow before doing work. Existing APIs report invalid inputs and resource failures with standard exceptions (`std::invalid_argument`, `std::runtime_error`, `std::overflow_error`, `std::bad_alloc`).
- Keep common code backend-neutral. Put runtime headers and kernels in the corresponding backend directory; implement backend behavior behind `Device`, `Tensor`, and `DeviceOps` interfaces rather than switching on backend kind in core code.
- Preserve the tiled-storage and host-encoding contracts. Standard-layout backends use the same `16x16` tile mapping, bit-packed sub-byte values, and byte-aligned little-endian multi-byte values. Transformations and transfers must not allocate or move owner storage.
- Use caller-provided output tensors for arithmetic. Backend queues are in-order; queue failures are completed and retained so later waits observe the same failure.
- Tests use sentence-style doctest case names such as `TensorSpec rejects ...`; shared helpers and independent encoding models belong in `test/backend/backend_conformance.hpp` or the relevant test driver, not production code.

## Important Files

- `CMakeLists.txt` — C++20 targets, dependency discovery, backend options, and library wiring.
- `test/CMakeLists.txt` — doctest executables and CTest registration.
- `include/iom/tensor.hpp`, `include/iom/device.hpp`, `include/iom/iom.hpp` — core tensor, device, and queue contracts.
- `src/iom.cpp` — tensor/view implementation and common queue-id/completion machinery.
- `include/iom/alloc.hpp`, `src/alloc.cpp` — caller-supplied allocator interface and linear/list/fixed-size allocators.
- `include/iom/safetensors.hpp`, `src/safetensors.cpp`, `include/iom/mmap.hpp`, `src/mmap.cpp` — model-weight mapping/loading.
- `include/iom/{cpu,cuda,rocm,ttnn}/device.hpp` — backend factories and TTNN capability table.
- `src/{cpu,cuda,rocm,ttnn}/` — backend devices, tensors, queues, and copy implementations.
- `test/backend/backend_conformance.hpp` — shared storage, transfer, async-copy, error, lifetime, and capability checks.
- `README.md` — authoritative install and per-backend command examples.
- `docs/design/README.md` — architectural intent; `docs/changes/0001-tensor-view/spec.md` — current API/behavior contract.

## Runtime/Tooling Preferences

- Use CMake with a C++20 compiler. The documented dependency set is Ubuntu-oriented and includes `nlohmann_json` and doctest; CUDA, ROCm/HIP, and TTNN SDKs are installed separately when enabled.
- CPU development and tests may run locally. GPU backend work must use the `remote-development` skill: edit the local workspace, sync it to a unique remote task directory, run build/test/execution through SSH, then clean the mirror.
- Configured remote profiles in `.remote-hosts.conf`: `rocm` on `apu1`, `cuda` on `beha`, and `ttnn` on `beha`. SSH details belong in `~/.ssh/config`; do not put credentials in the repository.
- Typical remote workflow:

```sh
.agents/skills/remote-development/scripts/remote-sync rocm task-123
.agents/skills/remote-development/scripts/remote-exec rocm task-123 'cmake -S . -B build && cmake --build build -j'
.agents/skills/remote-development/scripts/remote-exec rocm task-123 'ctest --test-dir build --output-on-failure'
.agents/skills/remote-development/scripts/remote-clean rocm task-123
```

- Use unique task ids and remote directories for concurrent work. Use `flock` for exclusive GPU tests and `tmux` for long-running remote jobs. Do not edit a shared remote checkout directly.

## Testing & QA

- The project uses doctest through CTest. Always-built targets are `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests`; backend smoke and conformance targets are added only when their CMake option is enabled.
- Select individual registered tests with `ctest --test-dir <build-dir> --output-on-failure -R '<regex>'`. Build a target first when needed, for example `cmake --build build --target iom_tests`.
- Smoke tests verify backend construction, device ordinal/context ownership, teardown, and basic capability rejection. They require live hardware and never skip when the backend is enabled; missing SDKs fail at configuration.
- Conformance tests instantiate `test/backend/backend_conformance.hpp` against a CPU reference, candidate, and foreign device. They cover storage/transfer, async copies, error paths, lifetime, and unsupported compute capability. Preserve independent host encodings, readback sentinels, allocator/traffic gating, and exact mismatch checks when extending the harness.
- Test names follow `test_<subject>.cpp` for core/CPU tests and `test_<backend>_<smoke|conformance>.cpp` for accelerator tests. Add backend behavior to the shared harness where the contract is common; keep runtime-specific setup in the backend driver.
- Treat numbered specs under `docs/changes/0001-tensor-view/` as acceptance criteria for the current tensor/view work. Use the documented `spec-tasks` and `spec-critic` workflows when changing those specs.
