#pragma once

#include <cuda.h>

#include <cstddef>
#include <memory>
#include <span>

#include "iom/iom.hpp"

namespace iom::cuda_detail {

void region_from_host(
        CUcontext context, const TensorView& destination,
        std::span<const std::byte> source);

void region_to_host(
        CUcontext context, const TensorView& source,
        std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, CUcontext context);

}  // namespace iom::cuda_detail
