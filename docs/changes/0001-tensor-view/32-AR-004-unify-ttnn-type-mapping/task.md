**Status:** done

## Summary

Unified TTNN capability publication and native dtype conversion around the single constexpr `kSupportedToNative` mapping in `src/ttnn/device.cpp`. `kSupportedKeys` is mechanically derived at compile time with size and uniqueness assertions, and both capability membership and native dtype lookup use that mapping. Removed the redundant public `checked_plane_count` call while retaining the constructor-side validation gate. The public TTNN header and conformance tests remain unchanged.

## Verification

- `cmake --build build/ttnn --target iom_ttnn -j` via remote-development on `bv1` — built `iom_ttnn` successfully with TTNN enabled.
- `cmake --build build/ttnn --target iom_ttnn_conformance_tests -j` via remote-development on `bv1` — built the hardware conformance executable successfully.
- `./build/ttnn/test/iom_ttnn_conformance_tests -tc=*acceptance*` under `flock` via remote-development on `bv1` — 1 test case and 28 assertions passed.
- `./build/ttnn/test/iom_ttnn_conformance_tests -tc=*overflowing*` under `flock` via remote-development on `bv1` — 1 test case and 5 assertions passed; the maximal extent reached native allocation and was rejected there rather than by validation.
- `ctest --test-dir build/ttnn --output-on-failure -R ^iom_ttnn_conformance_tests$` under `flock` via remote-development on `bv1` — 1/1 TTNN conformance test passed.
- `cmake -S . -B build/cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build/cpu --target iom_backend_conformance_cpu_tests -j && ctest --test-dir build/cpu --output-on-failure -R '^iom_backend_conformance_cpu_tests$'` — 1/1 CPU backend conformance test passed.
- Source checks — all `DataType::` occurrences in `src/ttnn/device.cpp` are inside `kSupportedToNative`; `checked_plane_count` has only its helper definition and the constructor call; `include/iom/ttnn/device.hpp` is unchanged.
