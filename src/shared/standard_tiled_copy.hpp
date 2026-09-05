#pragma once

#include <cstddef>
#include <span>

#include "iom/iom.hpp"

namespace iom::detail {

    template <typename Policy, typename StreamPool, typename StagingPool>
    void synchronous_transfer(
            StreamPool& transfer_pool, StagingPool& staging_pool,
            typename Policy::context_type context, const TensorView& view,
            std::span<const std::byte> source,
            std::span<std::byte> destination, bool from_host);

}  // namespace iom::detail
