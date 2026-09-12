**Status:** done

## Summary

Split tensor shape/spec/layout and tensor-view definitions into dedicated private translation units with explicit internal linkage while retaining iom.cpp behavior and build membership.

## Verification

- cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF — configure completed successfully.
- cmake --build build/split-cpu --target libiom iom_tests — libiom and iom_tests built successfully.
- ctest --test-dir build/split-cpu --output-on-failure -R ^iom_tests$ — 1/1 test passed.
- wc -l src/tensor.cpp src/tensor_view.cpp src/iom_internal.hpp src/iom.cpp — new files are 197, 337, and 39 lines; iom.cpp is 1207 intermediate lines.
- moved-symbol search in src/iom.cpp — no matches for representative TensorShape, TensorSpec, TensorView, and Tensor definitions.
