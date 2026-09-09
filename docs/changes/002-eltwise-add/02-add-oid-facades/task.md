**Status:** done

## Summary

Added non-throwing common OID facades with protected backend hooks, exact device/view validation, synchronous error mapping, and retained queue semantics.

## Verification

- cmake --build /tmp/iom-002-eltwise-add-02-build --target libiom — built libiom and migrated copy hooks successfully
- /usr/bin/c++ -std=c++20 -include ./include/iom/device.hpp -I./include /tmp/iom-002-eltwise-add-02-smoke.cpp /tmp/iom-002-eltwise-add-02-build/libiom.a -pthread -o /tmp/iom-002-eltwise-add-02-smoke && /tmp/iom-002-eltwise-add-02-smoke — noexcept facade signatures, accepted/repeat waits, six error mappings, exact-device rejection, unsupported defaults, and parameter validation passed
