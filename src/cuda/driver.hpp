#pragma once

#include <cuda.h>
#include <stdexcept>
#include <string>

namespace iom::cuda_detail {

struct DriverCalls {
    CUresult (*primary_ctx_retain)(CUcontext*, CUdevice)
            = &cuDevicePrimaryCtxRetain;
    CUresult (*ctx_set_current)(CUcontext) = &cuCtxSetCurrent;
    CUresult (*primary_ctx_release)(CUdevice) = &cuDevicePrimaryCtxRelease;
    CUresult (*init)(unsigned int) = &cuInit;
    CUresult (*device_get_count)(int*) = &cuDeviceGetCount;
    CUresult (*device_get)(CUdevice*, int) = &cuDeviceGet;
};

extern DriverCalls driver_calls;

}  // namespace iom::cuda_detail

namespace iom {

[[nodiscard]] std::runtime_error cuda_error(
        const char* operation, CUresult status);
void check_cuda(const char* operation, CUresult status);

inline std::runtime_error cuda_error(
        const char* operation, CUresult status) {
    const char* name = nullptr;
    const char* description = nullptr;
    (void)cuGetErrorName(status, &name);
    (void)cuGetErrorString(status, &description);
    return std::runtime_error(
            std::string(operation) + " failed with "
            + (name != nullptr ? name : "unknown CUDA error") + ": "
            + (description != nullptr ? description : "unknown error"));
}

inline void check_cuda(const char* operation, CUresult status) {
    if (status != CUDA_SUCCESS) {
        throw cuda_error(operation, status);
    }
}

}  // namespace iom
