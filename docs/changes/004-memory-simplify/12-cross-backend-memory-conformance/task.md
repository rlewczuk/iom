**Status:** done

## Summary

Added backend-neutral rank, queue-admission, workspace, ownership, caller, coexistence, documentation, and evidence conformance for CPU, CUDA, ROCm, SYCL, and TTNN. Repaired TTNN queue teardown ordering and quarantine-fault setup exposed by the final enabled-backend gate, and made smoke targets link libiom explicitly.

## Verification

- CPU: cmake -S . -B build/final-cpu -DBUILD_TESTING=ON; cmake --build build/final-cpu -j2; ctest -R iom_scalar_add_tests|iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests: 4/4 passed. CUDA via remote-development: smoke, conformance, coexistence: 3/3 passed. ROCm via remote-development: smoke, conformance, coexistence: 3/3 passed. SYCL via remote-development with /tmp/iom-remote-sycl-no-setup.conf and sourced oneAPI: smoke, conformance, coexistence: 3/3 passed. TTNN via remote-development: smoke, conformance, coexistence: 3/3 passed. Factory/workspace caller audit completed; examples/ absent. Two/eight-GPU topology and matched pre-change performance evidence unavailable on the exercised hosts and remain non-universal risks.
