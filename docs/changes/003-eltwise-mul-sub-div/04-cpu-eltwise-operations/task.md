**Status:** done

## Summary

Migrated CPU elementwise execution to one operation-neutral standard-tiled traversal for ADD, MUL, SUB, and floating DIV with preserved queue and storage semantics.

## Verification

- cmake -S . -B build && cmake --build build --target iom_backend_conformance_cpu_tests && ctest --test-dir build -R ^iom_backend_conformance_cpu_tests$ --output-on-failure — target built and 1/1 test passed.
