#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "iom/cpu/device.hpp"
#include "iom/detail/outstanding_work_registry.hpp"

namespace iom {

class CpuDevice final : public Device {
public:
    CpuDevice(Allocator& allocator, QueueConfig queue_config);

    [[nodiscard]] BackendKind backend_kind() const noexcept override;
    [[nodiscard]] std::uint32_t backend_device() const noexcept override;
    [[nodiscard]] std::span<const DataType>
            supported_data_types() const noexcept override;
    [[nodiscard]] detail::RegistryState& registry_state() noexcept;
    [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
            const TensorSpec& spec) override;
    [[nodiscard]] std::unique_ptr<RawWorkspace> create_workspace(
            std::size_t bytes) override;
    [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

private:
    Allocator& allocator_;
    detail::RegistryState registry_state_;
};

}  // namespace iom
