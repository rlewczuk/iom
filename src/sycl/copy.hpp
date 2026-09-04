#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <memory>
#include <span>

#include "iom/iom.hpp"

namespace iom::sycl_detail {

void region_from_host(
        const sycl::context& context, const sycl::device& device,
        const TensorView& destination, void* storage,
        std::span<const std::byte> source);

void region_to_host(
        const sycl::context& context, const sycl::device& device,
        const TensorView& source, const void* storage,
        std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, const sycl::context& context,
        const sycl::device& native_device);

}  // namespace iom::sycl_detail
