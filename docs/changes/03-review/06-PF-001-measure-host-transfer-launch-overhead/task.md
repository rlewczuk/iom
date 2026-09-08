**Status:** done

## Summary

Release-only throwaway F32 {P,1,32,64} sweep at P=1,8,32,64,128 completed on CUDA RTX 5090, ROCm Radeon AI PRO R9700, and SYCL Arc Pro B60 with separate upload/download medians, exact logical=padded bytes, current per-plane counts P and prototype count 1, metadata preparation included, and exact output parity. CUDA and SYCL met the high-P speedup threshold but regressed at P=1 in one direction; ROCm met the no-P=1-regression and high-P speedup thresholds. Finalize measurement-only with no production batching; any follow-on is separately scoped to ROCm only.

## Verification

- CUDA Release: ./_local/pf001/measure_cuda full via remote-exec; P1/8/32/64/128 logical=padded 8192/65536/262144/524288/1048576, current upload/download medians 7033/11702, 26421/31821, 71116/89882, 132543/161438, 253735/301796 ns; prototype 13135/8756, 17022/13916, 34566/29737, 59093/55486, 96735/109669 ns; driver observed current launches P and prototype 1; parity passed.
- ROCm Release: ./_local/pf001/measure_rocm full via remote-exec; P1/8/32/64/128 logical=padded 8192/65536/262144/524288/1048576, current upload/download medians 63049/93878, 110459/121008, 279208/296160, 499274/517639, 913476/953171 ns; prototype 60835/88878, 71775/69822, 130928/139453, 151446/158409, 225787/245384 ns; driver observed current launches P and prototype 1; parity passed.
- SYCL Release: sycl-ls plus ./_local/pf001/measure_sycl full via setvars-initialized remote-exec; P1/8/32/64/128 logical=padded 8192/65536/262144/524288/1048576, current upload/download medians 9067/11121, 50235/50295, 183216/185250, 326497/338399, 642785/601617 ns; prototype 7003/14788, 9418/13616, 23174/28674, 38482/46768, 64983/76014 ns; existing launch seam observed current launches P and prototype 1; parity passed.
- CUDA profiler: nsys profile --trace=cuda,nvtx,osrt --sample=none --stats=true -o _local/pf001/cuda-profile-release -- ./_local/pf001/measure_cuda profile; Release trace reported 714 cudaLaunchKernel calls, 2.1748 us average, and kernel totals of 466 gather, 233 scatter, 10 prototype_download, 5 prototype_upload instances including the driver parity pass.
- ROCm profiler: rocprofv3 --hip-trace --kernel-trace --output-file _local/pf001/rocm-trace-release -- ./_local/pf001/measure_rocm profile; Release trace DB reported 466 gather, 233 scatter, 10 prototype_download, 5 prototype_upload dispatches including the driver parity pass; command succeeded with timestamp-swap and intercept-queue warnings.
- Fixed-condition evidence: CUDA nvidia-smi reported RTX 5090 P8, 600 W limit, 15.49 W draw; ROCm rocm-smi reported Radeon AI PRO R9700/gfx1201; SYCL sycl-ls enumerated two Arc Pro B60 Level Zero GPUs. Temporary local and remote measurement artifacts were removed after capture.
