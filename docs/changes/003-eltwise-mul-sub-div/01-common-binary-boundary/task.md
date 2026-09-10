**Status:** done

## Summary

Delivered the neutral four-operation DeviceOps boundary with validated snapshots, ownership registration, ordered OIDs, and retained completion failures.

## Verification

- Direct fake-boundary scenario: c++ -std=c++20 -Iinclude -I. -Itest -DDOCTEST_CONFIG_IMPLEMENT_WITH_MAIN test/test_iom.cpp src/iom.cpp src/mmap.cpp src/safetensors.cpp src/alloc.cpp -o /tmp/iom_tests_common && /tmp/iom_tests_common — 75 test cases and 10916 assertions passed.
