#pragma once

#include <cuda.h>

namespace iom::cuda_detail {

struct DriverCalls {
    CUresult (*primary_ctx_retain)(CUcontext*, CUdevice)
            = &cuDevicePrimaryCtxRetain;
    CUresult (*ctx_set_current)(CUcontext) = &cuCtxSetCurrent;
    CUresult (*primary_ctx_release)(CUdevice) = &cuDevicePrimaryCtxRelease;
};

extern DriverCalls driver_calls;

}  // namespace iom::cuda_detail
