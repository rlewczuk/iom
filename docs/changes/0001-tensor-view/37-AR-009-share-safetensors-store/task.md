**Status:** done

## Summary

Added the private `iom::detail::SafeTensorsStoreBase` composition layer in `include/iom/safetensors.hpp`. `SafeTensorsFile` and `SafeTensorsDir` now delegate lookup, size, and insertion-order key access to the shared base, while their constructors retain file and shard ownership and feed the base through `insert`.

## Verification

- `cmake -S . -B build/ar009 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/ar009 -j --target iom_tests` — configured and built `iom_tests` successfully.
- `./build/ar009/test/iom_tests --test-case='SafeTensors*' -d` — 8 test cases and 211 assertions passed with no failures.
- `ctest --test-dir build/ar009 --output-on-failure -R '^iom_tests$'` — 1/1 test passed.
- `cmake --build build/ar009 -j && ctest --test-dir build/ar009 --output-on-failure` — all 3 configured CPU test targets built and passed.
- Structural checks — `include/iom/safetensors.hpp` is 134 lines; `src/safetensors.cpp` has no override definitions or standalone `tensors_`/`keys_` members and contains exactly two `base_.insert` call sites.
