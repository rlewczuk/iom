**Status:** done

## Summary

TTNN binary mapping reuses bounded request-local arrays for result and operand coordinates, removing per-element coordinate-vector allocation while preserving plane mapping and output behavior.

## Verification

- cmake -S . -B /tmp/iom-01-review-06-cpu -DCMAKE_BUILD_TYPE=Debug && cmake --build /tmp/iom-01-review-06-cpu -j2 --target iom_backend_conformance_cpu_tests && ctest --test-dir /tmp/iom-01-review-06-cpu --output-on-failure -R '^iom_backend_conformance_cpu_tests$' — configured, built, and the CPU conformance control passed
- .omp/skills/remote-development/scripts/remote-sync ttnn 01-review-06-PF-002 followed by remote-exec TTNN build and ctest — synchronized the exact worktree, built TTNN conformance and CPU control targets, and both tests passed
- source mapping-storage check in src/ttnn/binary.cpp — bounded std::array coordinate buffers are outside the flat-element loop and no std::vector<std::size_t> mapping construction remains
