**Status:** done

## Summary

Defined signed public OID errors, checked 55-bit queue-token encoding, bounded sequence allocation, and strict common wait validation.

## Verification

- cmake --build /tmp/iom-002-eltwise-add-01-build --target libiom — built libiom successfully
- /usr/bin/c++ -std=c++20 -I./include /tmp/iom-002-eltwise-add-01-smoke.cpp /tmp/iom-002-eltwise-add-01-build/libiom.a -pthread -o /tmp/iom-002-eltwise-add-01-smoke && /tmp/iom-002-eltwise-add-01-smoke — OID classification, all 255 queue IDs, INT64_MAX boundary, invalid waits, and sequence exhaustion passed
