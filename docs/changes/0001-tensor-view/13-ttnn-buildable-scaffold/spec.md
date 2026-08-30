# Add the buildable TTNN backend scaffold

**Order:** 13
**Priority:** P0 — TTNN dependency discovery and native device ownership must pass before TTNN tensor work.
**Blocked by:** `10-cuda-storage-copy`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

With `TTNN_ENABLED=ON`, TTNN is an independent optional library with a public factory that validates an ordinal, owns the native Tenstorrent device context, reports backend identity, and passes a hardware-backed construction/destruction smoke test.

## Scope

- Begin TTNN only after the SYCL implementation and conformance milestone is complete.
- Add the new build option, runtime discovery, isolated target, public factory, native context lifetime, and construction smoke test.

## Implementation references

- **Create (planned):** `include/iom/ttnn/device.hpp` — declare `make_ttnn_device(std::uint32_t)` without TTNN header types or native tensor details.
- **Create (planned):** `src/ttnn/device.cpp` — validate ordinal, own the TTNN device/runtime context, report identity, and tear down deterministically.
- **Create (planned):** `test/ttnn/test_ttnn_smoke.cpp` — hardware-backed valid/invalid ordinal and context lifetime tests.
- **Modify:** `CMakeLists.txt` — add independent `TTNN_ENABLED` and optional target `iom_ttnn` with target-local TTNN dependencies.
- **Modify:** `test/CMakeLists.txt` — add `iom_ttnn_smoke_tests` only when TTNN is enabled.
- **Read:** existing backend factory headers — preserve the same backend-neutral layering while omitting `Allocator&` only for TTNN as specified.

TTNN SDK will be installed system wide and is available to cmake as libraries, for example:

```
target_link_libraries(my-lib PRIVATE
    TT::Metalium
    TTNN::TTNN
)
```

Use this formula when modifying build scripts.

Current version of tt-metalium/ttnn source code is in `/home/rlew/tt/src/tt-metal`, you can look at it for reference.

## Requirements

- With TTNN disabled, common headers, `libiom`, and other backend libraries have no TTNN headers, runtime links, compile settings, or symbols.
- Enabling TTNN discovers the required runtime package and fails configuration clearly if it is absent; it must not silently disable a requested backend.
- `make_ttnn_device(ordinal)` rejects an unavailable or invalid backend-local ordinal before tensor or queue resources.
- Successful construction owns the native device/runtime context, reports `BackendKind::TTNN` and the configured ordinal, and does not receive or use `iom::Allocator` for native tensor storage.
- The context is destroyed exactly once after dependent tensors and queues. No borrowed context, global active device, singleton, registry, or runtime backend switch is allowed.
- During this explicit scaffold milestone, `create_tensor` and `create_ops` may throw `std::runtime_error` for not-yet-delivered TTNN capabilities without fake resources or sequence consumption. Task `14-ttnn-storage-copy` replaces those failures for supported storage/copy behavior.
- The smoke test uses real Tenstorrent hardware/runtime, covers the first invalid ordinal, and never skips when `TTNN_ENABLED=ON`.
- Existing CUDA, ROCm, and SYCL options remain independent and unmodified by TTNN enablement.

## Non-goals

- Native TTNN tensor allocation, host transfer, asynchronous copy, supported-type policy, or conformance.
- Exposing TTNN native storage internals through common headers.
- Numerical compute operations, borrowed contexts, or queue tuning.

## Acceptance criteria

- [ ] Core-plus-CPU and every existing backend build with TTNN disabled and no TTNN dependency.
- [ ] TTNN-only configuration builds `iom_ttnn` and `iom_ttnn_smoke_tests` with isolated runtime settings.
- [ ] Valid construction reports TTNN and the requested ordinal; invalid ordinal rejection precedes tensor/queue resources.
- [ ] Runtime instrumentation proves one owned device context and one matching teardown.
- [ ] Enabled smoke coverage runs on real hardware with no skip path.

## Verification

- `cmake -S . -B build/ttnn-smoke -DBUILD_TESTING=ON -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF`
- `cmake --build build/ttnn-smoke --target iom_ttnn_smoke_tests`
- `ctest --test-dir build/ttnn-smoke --output-on-failure -R '^iom_ttnn_smoke_tests$'`
