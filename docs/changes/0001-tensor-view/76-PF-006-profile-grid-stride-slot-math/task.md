**Status:** done

## Summary

Measurement-only investigation. No production, header, test, build, kernel, or CMake files changed.

The shared copy implementation contains both `grid_stride_copy_kernel` and `grid_stride_copy_inline_kernel` in `src/shared/standard_tiled_copy.inl`; CUDA `src/cuda/copy.cu` and ROCm `src/rocm/copy.hip` both launch through `detail::launch_grid_stride_copy`. The profiled target was the inline helper selected by the current tensor-copy path.

The throwaway public-API harness used rank-2 `{4096,4096}` and rank-4 `{4,4,1024,1024}`, `I4` and `F32`, dense cases for all shapes and rank-4 permuted-source cases. Each target run queued five warmup copies before the sixth matching helper launch. The destination was copied back and compared against the source's logical view bytes; every target run passed. Separate dense controls used native `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` and `hipMemcpy(..., hipMemcpyDeviceToDevice)`.

## Tooling and command evidence

CUDA host:

- `ncu`: `/usr/local/cuda/bin/ncu`, `NVIDIA Nsight Compute CLI Version 2026.1.1.0 (build 37634170) (public-release)`.
- `nsys`: `/usr/local/bin/nsys`, `NVIDIA Nsight Systems version 2026.4.1.191-264138605071v0`.
- `ncu --csv --page raw --target-processes application-only ./_local/pf006-profile/profile_cuda --mode kernel --shape rank2 --dtype i4 --case dense` reached the application, which passed correctness, but returned `ERR_NVGPUCTRPERM` because NVIDIA GPU performance-counter access was unavailable.
- `nsys profile --trace=cuda --sample=none --cuda-memory-usage=false --stats=false --export=sqlite --force-overwrite=true --output=<output> ./_local/pf006-profile/profile_cuda --mode kernel --shape <shape> --dtype <dtype> --case <case>` was used for each of the six target tuples.
- The separate CUDA control command was the same profiler command with `--mode control --shape <shape> --dtype <dtype> --case dense` and a distinct output path. CUDA D2D rows were identified in the generated SQLite as `CUPTI_ACTIVITY_KIND_MEMCPY.copyKind=8`.
- `ncu --help` documented `--kernel-name`, `--kernel-name-base`, `--launch-count`, `--launch-skip`, `--csv`, `--page`, and `--metrics`; `nsys profile --help` documented `--trace`, `--sample`, `--cuda-memory-usage`, `--stats`, `--export`, `--force-overwrite`, and `--output`. The commands above use only those documented options.

ROCm host:

- `rocprofv3`: `/usr/bin/rocprofv3`, version `1.3.5`, ROCm `10.0.0`, git revision `6b0e43f341195e203754e08f850e437ff2fc09f9`.
- The exact target command form was `rocprofv3 --kernel-trace --memory-copy-trace --output-format csv --output-directory _local/pf006-profile/rocm-<tuple>-kernel -- ./_local/pf006-profile/profile_rocm --mode kernel --shape <shape> --dtype <dtype> --case <case>` for `rank2-i4-dense`, `rank4-i4-dense`, `rank4-i4-permute`, `rank2-f32-dense`, `rank4-f32-dense`, and `rank4-f32-permute`.
- The exact control command form was `rocprofv3 --runtime-trace --memory-copy-trace --output-format csv --output-directory _local/pf006-profile/rocm-<tuple>-control -- ./_local/pf006-profile/profile_rocm --mode control --shape <shape> --dtype <dtype> --case dense` for the four dense tuples.
- `rocprofv3 --help` documented `--kernel-trace`, `--memory-copy-trace`, `--runtime-trace`, `--pmc`, `--kernel-include-regex`, `--kernel-iteration-range`, `--output-format`, and `--output-directory`; those options were used for the traces and counter attempt.
- The target ROCm symbol was recorded from the unfiltered trace as `iom::detail::(anonymous namespace)::grid_stride_copy_inline_kernel(unsigned char const*, unsigned char*, iom::detail::(anonymous namespace)::InlineCopyMetadata)`. The native D2D control symbol was `__amd_rocclr_copyBuffer`.
- The documented counter command `rocprofv3 --pmc VALUInsts MemUnitBusy WAVE_ISSUE_WAIT --kernel-include-regex grid_stride_copy_inline_kernel --kernel-iteration-range 6 --output-format csv --output-directory <output> -- ./_local/pf006-profile/profile_rocm --mode kernel --shape rank2 --dtype i4 --case dense` produced `rocprofiler_iterate_agent_supported_counters failed ... Agent HW architecture is not supported, no counter metrics found` for `gfx1036`; the emitted counter values were zero. Thus no ROCm ALU, memory-unit, or scheduler pressure data was treated as valid.

## Measurements

Durations are profiler-reported nanoseconds for the sixth matching target launch after five warmups. Decimal GB/s is derived as `bytes / duration_ns`; dense kernel/control ratios are derived as `kernel_duration_ns / control_duration_ns`. Permuted cases intentionally have no control invocation.

| Backend / case | Bytes | Target duration (ns) | D2D control (ns) | Target GB/s | Control GB/s | Kernel/control | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| CUDA rank2 I4 dense | 8,388,608 | 50,240 | 3,904 | 166.970701 | 2,148.721311 | 0.077707 | INCONCLUSIVE |
| CUDA rank4 I4 dense | 8,388,608 | 54,401 | 6,816 | 154.199518 | 1,230.723005 | 0.125292 | INCONCLUSIVE |
| CUDA rank4 I4 permute | 8,388,608 | 54,305 | — | 154.472111 | — | — | INCONCLUSIVE |
| CUDA rank2 F32 dense | 67,108,864 | 124,896 | 50,689 | 537.317961 | 1,323.933477 | 0.405850 | INCONCLUSIVE |
| CUDA rank4 F32 dense | 67,108,864 | 157,665 | 51,712 | 425.642115 | 1,297.742574 | 0.327987 | INCONCLUSIVE |
| CUDA rank4 F32 permute | 67,108,864 | 155,329 | — | 432.043366 | — | — | INCONCLUSIVE |
| ROCm rank2 I4 dense | 8,388,608 | 372,783 | 38,841 | 22.502657 | 215.973018 | 0.104192 | INCONCLUSIVE |
| ROCm rank4 I4 dense | 8,388,608 | 537,654 | 43,989 | 15.602242 | 190.697856 | 0.081817 | INCONCLUSIVE |
| ROCm rank4 I4 permute | 8,388,608 | 532,216 | — | 15.761661 | — | — | INCONCLUSIVE |
| ROCm rank2 F32 dense | 67,108,864 | 904,335 | 331,403 | 74.207969 | 202.499265 | 0.366460 | INCONCLUSIVE |
| ROCm rank4 F32 dense | 67,108,864 | 1,273,112 | 355,723 | 52.712459 | 188.654835 | 0.279412 | INCONCLUSIVE |
| ROCm rank4 F32 permute | 67,108,864 | 800,231 | — | 83.861865 | — | — | INCONCLUSIVE |

The dense ratios are materially below one, but the required ALU/scheduler-pressure evidence is unavailable: NVIDIA counters failed with `ERR_NVGPUCTRPERM`, and ROCm counters are unsupported for the selected `gfx1036` agent. The rubric therefore does not permit `CONFIRMED` or `FALSIFIED`; all tuples are `INCONCLUSIVE`. Recommendation: rerun on a CUDA host with performance-counter permission and a ROCm agent/driver combination exposing the documented counters, then extend shapes/dtypes only if needed to resolve the verdict.

## Verification

- Built remotely on the CUDA host with `CUDA_ENABLED=ON`, `ROCM_ENABLED=OFF`, `SYCL_ENABLED=OFF`, `TTNN_ENABLED=OFF`; `iom_cuda`, `iom_cuda_smoke_tests`, and `iom_cuda_conformance_tests` built successfully.
- Built remotely on the ROCm host with `ROCM_ENABLED=ON`, `CUDA_ENABLED=OFF`, `SYCL_ENABLED=OFF`, `TTNN_ENABLED=OFF`; `iom_rocm`, `iom_rocm_smoke_tests`, and `iom_rocm_conformance_tests` built successfully.
- Unprofiled CUDA and ROCm target/control smoke runs passed for every required tuple; target output reported `warmups=5 correctness=PASS` and controls reported `correctness=NOT_APPLICABLE`.
- Profiler-generated target traces contained exactly six matching `grid_stride_copy_inline_kernel` launches; the sixth was used for every target duration. CUDA Nsight Systems target symbol was `iom::detail::<unnamed>::grid_stride_copy_inline_kernel(const unsigned char *, unsigned char *, iom::detail::<unnamed>::InlineCopyMetadata)`.
- Remote `remote-clean` removed the CUDA and ROCm mirrors. The local `_local/pf006-profile` harness and all generated profiler artifacts were removed. No `_local` profiling artifact remains.
- Source audit found only the existing shared helper and CUDA/ROCm launch call sites; no implementation change was made. `git diff --check` passed before this annotation was added.
