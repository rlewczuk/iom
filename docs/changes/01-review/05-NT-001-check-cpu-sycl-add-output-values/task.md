**Status:** done

## Summary

Verified CPU and SYCL ADD values through real queues.

## Verification

- CPU: cmake --build build/review-cpu --target iom_backend_conformance_cpu_tests && ctest --test-dir build/review-cpu --output-on-failure -R ^iom_backend_conformance_cpu_tests$ (passed 1/1); SYCL: set +u; source /opt/intel/oneapi/setvars.sh; set -u; sycl-ls enumerated Level Zero GPU; cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/sycl --target iom_sycl_conformance_tests && ctest --test-dir build/sycl --output-on-failure -R ^iom_sycl_conformance_tests$ (passed 1/1)
