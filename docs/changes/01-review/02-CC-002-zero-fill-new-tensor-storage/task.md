**Status:** done

## Summary

Zero-filled new tensor storage before exposure across standard tiled backends.

## Verification

- CPU conformance and CPU unit tests passed; CUDA, SYCL, and TTNN conformance passed 1/1; ROCm full conformance passed 1/1 after checked hipDeviceSynchronize was added after hipMemset to order default-stream zeroing against reused storage.
