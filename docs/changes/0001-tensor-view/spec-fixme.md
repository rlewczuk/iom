Remarks for further design refinement:
* numeric IDs from `backend_device()` should be consistent with underlying backend platform IDs, eg. numeric CUDA or ROCm device ordinal, this means that only combination (backend_kind, backend_device) is unique for each device
* adapt Allocator API to make aligned allocations possible (eg. for matrix instructions)
* add instructions that CPU and ROCm backends should be implemented as part of this change, consider having those implementations in backend-specific files instead of mixing all together 
* we have `DeviceOps.copy()` method and Tensor `copy_from*()` methods, so we have a bit of redundancy here, consider removing some redundancy
  * note that `DeviceOps` always queues all calls, it works as kind of synchronization at least across a single device

