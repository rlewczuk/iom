**Status:** done

## Summary

Extended the existing combined backend coexistence executable with interleaved integer and floating ADD across every enabled participant, two queues, positive token uniqueness, queue ordering, out-of-order and repeated waits, independent-device rejection, owner and handle stability, short-lived derived views, exact aliasing, and retained-failure isolation; existing copy coverage remains unchanged.

## Verification

- CUDA+TTNN: remote-sync cuda 002-eltwise-add-17-cuda-ttnn; remote-exec cuda 002-eltwise-add-17-cuda-ttnn cmake --build build/coexist-cuda-ttnn --target iom_backend_coexistence_tests -j && ctest --test-dir build/coexist-cuda-ttnn --output-on-failure -R ^iom_backend_coexistence_tests$; 1/1 passed.
- ROCm+SYCL: remote-sync rocm 002-eltwise-add-17-rocm-sycl; remote-exec rocm 002-eltwise-add-17-rocm-sycl cmake --build build/coexist-rocm-sycl --target iom_backend_coexistence_tests -j && set +u; source /opt/intel/oneapi/setvars.sh; set -u; ctest --test-dir build/coexist-rocm-sycl --output-on-failure -R ^iom_backend_coexistence_tests$; 1/1 passed.
