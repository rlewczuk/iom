**Status:** done

## Summary

Documented the signed OID compatibility contract in public headers and architecture/backend guides, including exact errors, token layout, synchronous timing, wait semantics, protected hooks, and caller cutover.

## Verification

- Manual consistency review of include/iom/oid.hpp, include/iom/iom.hpp, docs/ARCHITECTURE.md, and docs/BACKEND_CONTRACT.md — no targeted stale unsigned/56-bit/throwing OID wording remained; copy rejection is documented as negative InvalidArgument, non-ADD hooks remain Unsupported, and all-255/skipped/repeat/retained-failure rules agree with migrated tests.
