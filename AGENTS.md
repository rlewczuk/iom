# Repository Guidance

IOM is an inference-only C++20 engine for sparse, oversized language models. Computation is imperative; there is no autograd graph.

## Components

- Core — `include/iom`, `src` excluding backend subdirectories — backend-neutral tensor, queue, allocation, model-loading, and model APIs.
- Backends — `include/iom/{cpu,cuda,rocm,ttnn}`, `src/{cpu,cuda,rocm,ttnn}` — runtime factories, storage, queues, and copy kernels.
- Tests — `test`, `test/{backend,cpu,cuda,rocm,ttnn}` — unit tests, shared conformance tests, and backend drivers.
- Design — `docs/design`, `docs/changes` — architecture and change specifications.
- Workflows — `.agents/skills` — task-specific procedures, including remote accelerator development.

## Architectural Invariants

- Common code stays backend-neutral: no runtime headers or types, backend-kind switches, or global active-backend registry. Devices expose factories and implement behavior through `Device`, `Tensor`, and `DeviceOps`.
- A creating `Device` and supplied allocator outlive their tensors and queues. Owners are non-copyable and non-movable; views are copyable, non-owning, and never retargeted.
- Tensor metadata remains on the host. The final two dimensions use fixed `16x16` tiles; leading dimensions are row-major and viewable.
- CPU, CUDA, and ROCm share the standard tiled encoding and caller allocator. TTNN may use native per-plane storage behind the same public contracts and must reject unsupported formats.
- Operations allocate neither operands nor outputs. Callers provide tensors and host buffers; views, transforms, and transfers do not relocate owner storage.
- Queues are asynchronous and in order. Waits are repeatable; completed failures remain observable and are rethrown.
- Weight flow remains mapped file → SafeTensors → caller-created tensors → queued operations.
- Validate shapes, dtypes, quantization, buffer sizes, device identity, capabilities, and overflow before work begins.

## Engineering Rules

- Use C++20, `iom` namespaces, standard-library types, `std::unique_ptr` for created owners, and `std::span` for borrowed buffers.
- Report invalid input, overflow, allocation, and runtime failures with the established standard exception categories.
- Optional backends fail configuration when their SDK is unavailable; enabled hardware tests fail rather than skip.
- Keep shared behavior in the backend conformance suite and runtime setup in backend-specific drivers.

## Workflow

- Build with CMake and test with CTest. There is no separate lint or formatter target; preserve existing style.
- CPU work may run locally. For accelerator build, test, or execution, follow the `remote-development` skill; do not duplicate its procedure here.
- Treat applicable change specifications as acceptance criteria.
