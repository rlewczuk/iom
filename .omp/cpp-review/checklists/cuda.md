# CUDA Review Checklist

- [ ] current device/context assumptions are explicit where multiple devices are possible
- [ ] stream used for production and consumption is correct
- [ ] event record/wait ordering is valid
- [ ] host memory used for async copies remains live and has appropriate transfer semantics
- [ ] device allocation is not freed/reused before in-flight work completes
- [ ] default-stream semantics are not relied on accidentally
- [ ] launch configuration/index width covers maximum supported shapes
- [ ] launch/enqueue errors and deferred execution errors are both observed
- [ ] graph capture does not include incompatible operations or hidden synchronization
- [ ] `cudaDeviceSynchronize`/equivalent global waits are justified, especially in per-token paths
- [ ] peer access is capability checked for multi-GPU transfers
- [ ] Compute Sanitizer can be used to verify memory/race/synchronization hypotheses
- [ ] Nsight Systems is used before Nsight Compute for unexplained end-to-end regressions
