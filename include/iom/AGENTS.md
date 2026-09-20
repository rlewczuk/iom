# Public API Guidance

- `device.hpp` defines the backend-neutral device contract.
- `tensor.hpp` defines tensor metadata, views, ownership, layout, and encoding.
- `iom.hpp` defines queued operations, tokens, waits, and completion behavior.
- `alloc.hpp` defines caller-supplied allocators.
- `mmap.hpp` and `safetensors.hpp` define mapped weight access.
- Backend `device.hpp` files expose factories.

Keep common headers free of CUDA, HIP, and SYCL types. Public contract changes require updating every backend and conformance coverage.
