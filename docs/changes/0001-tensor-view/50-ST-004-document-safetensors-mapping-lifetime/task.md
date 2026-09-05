**Status:** done

## Summary

Documented the non-owning `SafeTensorView` mapping lifetime contract and the concrete mapping owners of `SafeTensorsFile` and `SafeTensorsDir` in `include/iom/safetensors.hpp`. Added a regression case that copies an F32 view's bytes while its `SafeTensorsFile` is alive and compares only the caller-owned copy after store destruction.

## Verification

- `cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build --target iom_tests && ctest --test-dir build --output-on-failure -R '^iom_tests$'` — configured, built `iom_tests`, and passed 1/1 CTest target.
- `./build/test/iom_tests --test-case='SafeTensorView is non-owning: copy bytes before store destruction'` — 1 test case and 3 assertions passed; 96 other cases skipped.
- Source and diff checks — required lifetime phrases are present; the new test copies from `view.raw<std::uint8_t>()` before the store scope ends and compares only `copied` with `payload` afterward; `git diff --name-only` lists only `include/iom/safetensors.hpp` and `test/test_safetensors.cpp`, and `git diff --check` passed.
