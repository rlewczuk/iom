**Status:** done

## Summary

Extracted backend-neutral workspace ownership and Device live-workspace registry definitions into src/workspace.cpp while retaining src/iom.cpp and preserving behavior.

## Verification

- cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF — configure completed successfully.
- cmake --build build/split-cpu --target libiom iom_tests — libiom and iom_tests built successfully.
- ctest --test-dir build/split-cpu --output-on-failure -R ^iom_tests$ — 1/1 test passed.
- wc -l src/workspace.cpp src/iom.cpp — workspace.cpp 111 lines (<=499); iom.cpp 1731 informational lines.
