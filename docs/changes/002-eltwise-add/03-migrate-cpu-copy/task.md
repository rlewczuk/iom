**Status:** done

## Summary

Migrated CPU copy tests to the signed 55-bit OID boundary, negative validation/unsupported results, and common protected-hook expectations.

## Verification

- cmake --build /tmp/iom-002-eltwise-add-03-build --target libiom && /usr/bin/c++ -std=c++20 -include ./include/iom/iom.hpp -I./include /tmp/iom-002-eltwise-add-03-smoke.cpp /tmp/iom-002-eltwise-add-03-build/libiom.a -pthread -o /tmp/iom-002-eltwise-add-03-smoke && /tmp/iom-002-eltwise-add-03-smoke — CPU copy positive token/repeat wait, data movement, invalid-input no-write, and unsupported ADD passed
- ctest --test-dir /tmp/iom-002-eltwise-add-wave-build --output-on-failure -R ^iom_cpu_tests — finalized local wave CPU tests passed
