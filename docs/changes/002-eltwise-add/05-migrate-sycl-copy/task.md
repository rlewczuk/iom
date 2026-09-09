**Status:** done

## Summary

Migrated SYCL copy submission to the signed OID facade, preserving queue state and updated conformance token expectations.

## Verification

- REMOTE_DEV_CONFIG=/tmp/iom-sycl-remote-hosts.conf remote-exec sycl 002-eltwise-add-05-sycl: set +u; source /opt/intel/oneapi/setvars.sh; set -u; sycl-ls — two Level Zero GPU devices were enumerated
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-remote-hosts.conf remote-exec sycl 002-eltwise-add-05-sycl: cmake configure/build iom_sycl_smoke_tests — SYCL library and smoke executable built successfully
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-remote-hosts.conf remote-exec sycl 002-eltwise-add-05-sycl: ctest --test-dir build/sycl --output-on-failure -R ^iom_sycl_smoke_tests$ — SYCL smoke test passed
