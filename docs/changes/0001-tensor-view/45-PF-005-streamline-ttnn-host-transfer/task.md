**Status:** done

## Summary

Reworked `src/ttnn/copy.cpp` TTNN host transfers to fill one zero-initialized typed tile-major buffer per upload plane using face-major 16-column segments, construct `Layout::TILE` host tensors directly, and inverse-map tile-major download bytes into row-major destinations without layout conversion. Existing dtype dispatch, mutex ownership, synchronous barriers, and public interfaces remain unchanged.

## Verification

- `.agents/skills/remote-development/scripts/remote-sync ttnn pf005-ttnn-host-transfer` — synced the exact task worktree to `bv1:agent-work/iom/pf005-ttnn-host-transfer`.
- Remote CMake configure/build with `-DTTNN_ENABLED=ON -DBUILD_TESTING=ON` and targets `iom_ttnn_conformance_tests iom_ttnn_smoke_tests` — both targets built successfully on the TTNN host.
- `ctest --test-dir build/ttnn-pf005 --output-on-failure -R "^iom_ttnn_(conformance|smoke)_tests$"` — 2/2 tests passed on Tenstorrent hardware.
- Scoped remote staging-pattern check over `upload_plane`/`download_plane` — no `std::vector<std::byte>`, `to_layout`, or `make_host_buffer` matches; the required `typed_buffer` helper remains outside those bodies.
- Baseline/post-fix static TTNN archive public-symbol-name comparison — unchanged symbol names.
- Local `git diff --stat HEAD` over public headers, `src/ttnn/copy.hpp`, `src/ttnn/device.cpp`, and `test/ttnn/test_ttnn_conformance.cpp` — no changes.
