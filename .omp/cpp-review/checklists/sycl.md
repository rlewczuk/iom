# SYCL / oneAPI Review Checklist

- [ ] queue/context/device association is explicit
- [ ] event dependency graph covers out-of-order execution where queues allow it
- [ ] USM/buffer lifetime extends through dependent work
- [ ] host accessors/waits do not introduce unintended synchronization
- [ ] device aspects/features are queried rather than assumed
- [ ] kernel/index/range calculations fit supported device limits
- [ ] cache keys include device/backend/compiler properties when generated binaries depend on them
- [ ] exceptions/asynchronous error handlers surface device failures correctly
- [ ] multi-device selection and memory migration behavior are intentional
- [ ] supported device sanitizers/profilers are used to verify material hypotheses
