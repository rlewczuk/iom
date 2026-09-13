# HIP / ROCm Review Checklist

- [ ] device selection and stream ownership are explicit
- [ ] async lifetime is preserved through stream completion/events
- [ ] host/device copies do not rely on CUDA-specific assumptions that differ under HIP
- [ ] capability checks account for target architecture/features where required
- [ ] launch and deferred execution errors are surfaced
- [ ] allocator/cache state is device-qualified
- [ ] global synchronization on hot paths is justified
- [ ] peer access/multi-GPU paths are capability checked
- [ ] ROCprofiler timeline evidence is used for overlap/copy/synchronization questions
- [ ] ROCm Compute Profiler is used only for material kernels, with awareness of profiling perturbation
