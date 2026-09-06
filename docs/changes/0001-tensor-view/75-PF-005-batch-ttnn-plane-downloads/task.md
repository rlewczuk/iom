**Status:** done

## Summary

Reworked `iom::ttnn_detail::region_to_host` to enqueue all TTNN plane downloads into one uninitialized per-call staging allocation, wait once, and assemble the completed tile-major windows in native view-plane order. The old blocking per-plane host-tensor allocation path was replaced by anonymous submit and assembly helpers; checked staging-size arithmetic and drain-then-rethrow handling preserve synchronous errors and protect in-flight reads.

## Verification

- Remote pre-fix control build and `ctest --test-dir build/ttnn-pf005 --output-on-failure -R "^iom_ttnn_(conformance|smoke)_tests$"` — passed 2/2 TTNN tests.
- Remote post-fix build and the same TTNN CTest filter — passed 2/2 tests, including conformance and smoke coverage.
- Remote source audit — zero `allocate_tensor_on_host`, blocking `copy_to_host`, `std::fill`, and legacy `void download_plane` matches; exactly three `finish()` call sites.
- Throwaway public-API benchmark on TT hardware, paired control/post-fix rerun with 15 samples per shape — `{64,512,512}` download median 25.691 ms → 25.596 ms (−0.095 ms), upload 17.884 ms → 18.256 ms; `{1024,32,32}` download 70.140 ms → 69.973 ms (−0.167 ms), upload 25.938 ms → 26.019 ms. All sampled values matched; the remote mirror and local throwaway files were cleaned.
