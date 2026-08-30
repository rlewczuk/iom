#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <memory>
#include <span>

#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom::rocm_detail {

void region_from_host(
        hipCtx_t context, const TensorView& destination,
        std::span<const std::byte> source);

void region_to_host(
        hipCtx_t context, const TensorView& source,
        std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, hipCtx_t context);

}  // namespace iom::rocm_detail
