**Status:** done

## Summary

`SafeTensorsFile` now validates checked expected payload sizes and strictly ordered, non-overlapping tensor ranges before constructing views. The SafeTensors tests now derive dtype payload sizes correctly and cover truncated, oversized, overlapping, out-of-order, and overflowing entries.

## Verification

- `g++ -std=c++20 -Iinclude /tmp/cc001_reproduce.cpp cmake-build-debug/libiom.a -pthread -o /tmp/cc001_reproduce && /tmp/cc001_reproduce` — uncorrected baseline reproduced the defect by accepting the truncated tensor with `nbytes=4` and exiting 1.
- `cmake -S . -B build/cc001 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF` — configured successfully.
- `cmake --build build/cc001 -j` — all CPU targets built successfully.
- `./build/cc001/test/iom_tests --test-case='*mismatched*'` — the new rejection case passed 10 assertions with no failures.
- `ctest --test-dir build/cc001 --output-on-failure -R '^iom_tests$'` — `iom_tests` passed.
- `ctest --test-dir build/cc001 --output-on-failure` — all 3 CPU test targets passed.
