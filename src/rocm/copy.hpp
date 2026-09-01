#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <memory>
#include <span>

#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom::rocm_detail {

void region_from_host(
        int device_ordinal, const TensorView& destination,
        std::span<const std::byte> source);

void region_to_host(
        int device_ordinal, const TensorView& source,
        std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, int device_ordinal);

}  // namespace iom::rocm_detail
