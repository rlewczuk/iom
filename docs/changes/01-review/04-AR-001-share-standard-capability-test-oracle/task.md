**Status:** done

## Summary

Centralized the independent standard capability oracle across backend conformance.

## Verification

- CPU, CUDA, ROCm, and SYCL backend conformance passed 1/1; SYCL sycl-ls enumerated Level Zero GPUs. Source search found exactly one shared 23-entry standard capability oracle and no local standard_supported_data_types arrays.
