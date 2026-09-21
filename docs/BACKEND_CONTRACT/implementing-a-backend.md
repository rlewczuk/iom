# Part I — Implementing a backend

## What a backend contributes

A complete backend is a separate optional library and a public factory header.
It contributes all of the following:

1. a public `make_<backend>_device(...)` factory returning
   `std::unique_ptr<iom::Device>`;
2. a concrete, non-copyable/non-movable `Device` owning one backend runtime
   context and implementing capability reporting, tensor creation, and queue
   creation;
3. a concrete `Tensor` storage owner with synchronous logical host transfers;
4. one or more `DeviceOps` implementations that provide the required
   asynchronous `copy` operation and either implement or explicitly reject
   every compute capability;
5. storage/queue lifetime management that prevents storage reuse or release
   while the backend can still access it;
6. a smoke test, a conformance driver, CMake registration, and coexistence
   coverage when enabled with other backends.

The backend owns runtime state and storage mechanics. The common layer owns
logical tensor metadata, view transformations, queue-token encoding, and
wait/failure bookkeeping. Do not duplicate common validation or create a
backend-specific version of `TensorView`.

## Standard tiled storage

The retained CPU, CUDA, ROCm, and SYCL backends use the common 16x16
encoding directly. Each backend MUST allocate
`TensorSpec::tiled_storage_nbytes()`, report the immutable 23-type standard
capability table in its exact order, and preserve every standard physical tile
and padding byte.

## Recommended implementation order

1. **Add build and public surface.** Add the new `BackendKind` enumerator and
   `include/iom/<backend>/device.hpp` with only the factory/capability
   declarations required by users. Add an optional static library in the root
   `CMakeLists.txt`, link it privately to `libiom` and its SDK, and fail
   configuration if the enabled SDK is unavailable. Do not expose vendor types
   through `include/iom/device.hpp` or add a backend registry.
2. **Create and own one runtime context.** Resolve the backend-local ordinal,
   reject invalid ordinals before creating a usable device, and make that
   context/device identity stable for the `Device` lifetime. Borrowed
   allocator and `Device` lifetimes must outlast all tensors and queues they
   create.
3. **Implement capabilities and tensor creation.** Return an immutable,
   nonempty span from `supported_data_types()`. Validate `TensorSpec` and
   reject any unadvertised type before native allocation. Allocate one stable
   storage owner, construct the common `Tensor` base with the creating device,
   and ensure the allocation respects the 32-byte base-alignment contract for
   allocator-backed storage.
4. **Implement synchronous host transfers.** Implement `region_from_host` and
   `region_to_host` for arbitrary valid transformed `TensorView`s. Convert
   between contiguous logical host bytes and physical storage without numeric
   conversion. Preserve every untouched owner plane and required padding.
5. **Implement `DeviceOps::copy`.** Validate operands before reserving work,
   submit a sequence through the common `DeviceOps` helpers, retain or fence
   backend work, and complete it in order. Start with `copy`; leave compute
   methods inherited until their real semantics, validation, and tests exist.
6. **Make shutdown and failure paths real.** Account for work after native
   submission has begun. Release normal resources only after a completion
   proof; quarantine/retire resources if failure prevents proving completion.
   A synchronous pre-submit error must leave no work and no token behind.
7. **Add the backend conformance driver.** Create a CPU reference device, the
   candidate device, and an independent foreign device where the runtime
   permits it. Supply allocator/context setup and, for an accelerator or
   native layout, an independent `AcceleratorStorageOracle`.
8. **Register smoke, conformance, and coexistence tests.** Use
   `add_iom_backend_tests` in `test/CMakeLists.txt`; create separate smoke and
   conformance executables. Extend `test/backend/test_backend_coexistence.cpp`
   and its CMake wiring so all enabled factory headers and libraries coexist in
   one process. Enabled hardware tests MUST fail when hardware is unavailable;
   they do not skip.
9. **Run the complete backend gate.** Run the backend smoke and conformance
   tests plus coexistence with the intended option combinations. Accelerator
   build/test execution follows the repository `csw-remote` procedure.

## Minimal conformance-driver shape

The shared harness is backend-neutral. A backend driver SHOULD mirror the CPU,
CUDA, ROCm, or SYCL driver structure:

```cpp
struct BackendDevices {
    // Own allocators before the devices that borrow them.
    std::unique_ptr<iom::Device> reference; // CPU
    std::unique_ptr<iom::Device> candidate; // backend under test
    std::unique_ptr<iom::Device> foreign;   // independently created device

    iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};
```

The driver calls the shared scenarios with **every** datatype returned by the
candidate capability span. A `TrafficGate`-aware allocator is strongly
recommended: it makes an allocation or free during a transfer, view
transformation, or operation a test failure. The driver must not test an
internal copy helper against itself. Instead, an accelerator/native oracle
seeds and observes storage through an independent vendor path, using the
canonical standard tiled byte representation defined by
`backend_conformance_oracle.hpp`.

## Exit criteria

Before considering a backend complete, confirm all of the following:

- its factory/header/library are individually usable and coexist with every
  enabled backend without global dispatch;
- supported types, creation, transfers, transformed views, copies, errors,
  tokens, lifetime, and unsupported capabilities pass the shared suite;
- an independent oracle catches a deliberately perturbed physical mapping;
- no operation/transfer/view allocates or releases operand/output tensor
  storage after setup;
- every asynchronous path has a completion proof or a retained failure and
  cannot permit unsafe resource reuse;
- the backend does not skip enabled-hardware tests.
