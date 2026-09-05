#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace iom::gpu_algorithm {

[[nodiscard]] inline std::size_t compute_staging_size(
        std::size_t logical_nbytes) {
    constexpr std::size_t kWordBytes = sizeof(std::uint32_t);
    const std::size_t padding =
            (kWordBytes - logical_nbytes % kWordBytes) % kWordBytes;
    if (padding
            > std::numeric_limits<std::size_t>::max() - logical_nbytes) {
        throw std::overflow_error("GPU transfer staging size overflows");
    }
    return logical_nbytes + padding;
}

}  // namespace iom::gpu_algorithm
