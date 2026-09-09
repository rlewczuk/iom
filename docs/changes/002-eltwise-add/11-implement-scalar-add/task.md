**Status:** done

## Summary

Implemented independent exact scalar ADD codecs and oracle for required integer and floating leaves, including OCP compact formats.

## Verification

- ctest --test-dir /tmp/iom-002-eltwise-add-11-build --output-on-failure -R ^iom_scalar_add_tests$ passed; combined finalized CPU train passed iom_scalar_add_tests, iom_tests, and iom_backend_conformance_cpu_tests; oracle independence review found no production-code inclusion.
