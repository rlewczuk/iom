**Status:** done

## Summary

Removed the stale #include "iom/llama.hpp", the dead TEST_CASE "DeviceOps queue drives the llama models through owner views", and the op_names helper the deleted test exclusively used from test/test_iom.cpp; no llama/models reference remains in test/ and the iom_tests target registers unchanged.

## Verification

- cmake -S . -B build -DBUILD_TESTING=ON && cmake --build build --target iom_tests — configure and build exit 0, iom_tests links
- ctest --test-dir build -R ^iom_tests$ --output-on-failure — 100% tests passed (iom_tests Passed)
- cmake --build build --target iom_backend_conformance_cpu_tests && ctest --test-dir build -R ^iom_backend_conformance_cpu_tests$ — Passed in 31.16s (CPU conformance suite)
- grep -rn llama test/ and test/-wide grep for llama|Llama|iom::models — no matches
