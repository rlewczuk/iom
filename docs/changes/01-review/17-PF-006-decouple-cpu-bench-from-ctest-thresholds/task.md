**Status:** done

## Summary

iom_cpu_bench is decoupled from ctest thresholds: the default ctest run is a reporting-only generic pass (runner/compiler/build identity, sample distributions, medians) with no CHECK_* floors; the calibrated floors remain in the binary behind the explicit IOM_CPU_BENCH_ENFORCE_FLOORS=1 controlled-run contract with a reference-host runner guard. The previously observed loaded-machine flakes (f32_large 2.95<4.0, i4_large 0.11<0.6, queued differential 0.4us>0.1us) are structurally unreachable in the default gate.

## Verification

- ctest -R "^iom_cpu_bench$" local loaded reference host — PASSED (generic run, reporting only)
- iom_cpu_bench on bv2 (Ryzen 7 9700X) — PASSED
- combined final train: 4/4 on CPU local + CUDA + ROCm + SYCL + TTNN incl. iom_cpu_bench
