# Tooling Guide

Use tools selectively to verify review hypotheses. Do not dump raw diagnostics into the review.

## General C++

- compiler warnings with project-supported configurations;
- clang-tidy (bugprone, concurrency, cppcoreguidelines, performance, project-selected checks);
- ASan/UBSan, and TSan where applicable;
- CodeQL or equivalent deeper static analysis;
- include-what-you-use when dependency/interface hygiene is relevant.

## CUDA

- Compute Sanitizer: memory access, race, initialization, synchronization checks;
- Nsight Systems: host/device timeline, copies, synchronization, stream overlap;
- Nsight Compute: material kernel analysis, memory workload, occupancy, roofline.

## HIP / ROCm

- ROCprofiler-SDK: runtime/activity tracing and system behavior;
- ROCm Compute Profiler: kernel counters, memory, roofline;
- project-supported sanitizers/debugging tools when available.

## Vulkan

- Vulkan Validation Layers;
- synchronization validation;
- GPU-assisted validation where useful;
- vendor/timeline profiling tools for performance.

## SYCL / oneAPI

- compiler/device sanitizers supported by the project's toolchain;
- implementation/vendor profilers for queue and kernel analysis.

## Cross-backend

- differential operator tests;
- build matrix with backends independently enabled/disabled;
- backend capability matrix;
- controlled benchmark suite with machine-readable baseline/current outputs.

## Tool-result rule

Record:

- exact command/tool/version where material;
- backend/device/workload;
- whether the result is direct evidence or only a lead;
- profiler instrumentation limitations.
