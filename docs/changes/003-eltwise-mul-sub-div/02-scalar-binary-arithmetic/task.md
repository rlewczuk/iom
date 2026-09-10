**Status:** done

## Summary

Delivered the shared operation-specialized scalar codec and independent ADD/MUL/SUB/DIV oracle coverage, including compact finite-only saturation alignment and a one-adjacent-ULP finite floating conformance comparator with exact special and signed-zero checks.

## Verification

- Focused scalar gate before common-boundary stacking: cmake --build build --target iom_scalar_add_tests && ctest --test-dir build -R ^iom_scalar_add_tests$ --output-on-failure — target built and 1/1 test passed after correcting compact finite-only infinity saturation.
- Shared GPU conformance now accepts the specified adjacent finite ULP envelope.
