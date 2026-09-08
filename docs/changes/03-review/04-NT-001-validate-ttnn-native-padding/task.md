**Status:** done

## Summary

Extended the independent TTNN native-storage oracle over every padded cell and verified native-only U8 padding mutations at extra columns and rows are detected and cleared by logical upload without changing production transfer code.

## Verification

- focused TTNN verification: cmake --build build/ttnn --target iom_ttnn_conformance_tests && ctest --test-dir build/ttnn -R "^iom_ttnn_conformance_tests$" --output-on-failure — 1/1 conformance test passed on Blackhole TTNN hardware.
- final-train TTNN combined verification: cmake --build build/ttnn --target iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build/ttnn -R "iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests" --output-on-failure — 3/3 tests passed.
