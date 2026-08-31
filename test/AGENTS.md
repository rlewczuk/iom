# Test Guidance

- Root tests cover core metadata, queues, allocators, mapping, and SafeTensors.
- `backend` contains the shared backend-neutral conformance harness.
- Backend directories contain runtime setup, smoke tests, and conformance drivers.

Put common behavior in the shared harness and runtime-only setup in its driver. Preserve independent host encodings, sentinels, allocator/traffic checks, exact mismatch checks, and lifetime/error coverage. Use sentence-style doctest names and `test_<subject>.cpp` filenames.
