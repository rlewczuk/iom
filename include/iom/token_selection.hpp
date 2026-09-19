#pragma once

#include <cstddef>
#include <span>

#include "oid.hpp"
#include "tensor.hpp"

namespace iom {

    struct TokenSelectorScratch {
        std::span<std::byte> host;
        RawWorkspaceView device;
    };

    struct TokenSelectorScratchRequirements {
        std::size_t host_bytes;
        WorkspaceRequirements device;
    };

    class TokenSelector {
    public:
        virtual ~TokenSelector() = default;
        virtual TokenSelectorScratchRequirements scratch_requirements(
                const TensorView&, std::size_t) const = 0;
        virtual std::size_t select(
                DeviceOps&, const TensorView&, std::size_t, oid,
                std::span<const std::size_t>, TokenSelectorScratch) = 0;
    };

}  // namespace iom
