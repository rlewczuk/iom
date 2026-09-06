**Status:** done

## Summary

Wired the existing `SyclStorageOracle` into the SYCL asynchronous-copy and full shared conformance cases. Both cases now observe complete native destination and source owner storage after queued copies while remaining bound to `devices.candidate_allocator`.

## Verification

- `REMOTE_DEV_CONFIG=/tmp/iom-68-NT-002-20260906-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 68-NT-002-20260906 'flock /tmp/agent-gpu0.lock ./build/sycl/test/iom_sycl_conformance_tests -tc="*asynchronous copies against the CPU reference*"'` — passed; 12,931 assertions.
- `REMOTE_DEV_CONFIG=/tmp/iom-68-NT-002-20260906-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 68-NT-002-20260906 'flock /tmp/agent-gpu0.lock ./build/sycl/test/iom_sycl_conformance_tests -tc="*full shared suite*"'` — passed; 1,221 assertions.
- Temporary widened SYCL kernel demonstration — failed at `candidate oracle destination offset source to earlier destination: storage diverges at byte 768 of 1536`; the destination logical checks still passed, and the temporary production edit was reverted.
- `REMOTE_DEV_CONFIG=/tmp/iom-68-NT-002-20260906-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 68-NT-002-20260906 'flock /tmp/agent-gpu0.lock ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_conformance_tests$"'` — 1/1 passed after reverting the demonstration edit.
- Backend conformance targets — CPU, CUDA, ROCm, and TTNN each configured, built, and passed their respective `ctest` conformance test.
- Final source audit — `test/sycl/test_sycl_conformance.cpp` is the only changed project source; `src/sycl/` has no diff.
