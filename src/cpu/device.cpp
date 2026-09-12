#include "iom/cpu/device.hpp"
#include "device_internal.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>

#include "iom/iom.hpp"
#include "../shared/standard_tiled_copy.hpp"

namespace iom {

CpuDevice::CpuDevice(Allocator& allocator, QueueConfig queue_config)
        : Device(queue_config), allocator_(allocator) {}

BackendKind CpuDevice::backend_kind() const noexcept {
    return BackendKind::CPU;
}

std::uint32_t CpuDevice::backend_device() const noexcept {
    return 0;
}

std::span<const DataType> CpuDevice::supported_data_types() const noexcept {
    return detail::standard_supported_data_types();
}

detail::RegistryState& CpuDevice::registry_state() noexcept {
    return registry_state_;
}

namespace {

class CpuWorkspace final : public RawWorkspace {
public:
    CpuWorkspace(CpuDevice& device, std::size_t bytes)
            : RawWorkspace(device, bytes) {}
};

}  // namespace

std::unique_ptr<RawWorkspace> CpuDevice::create_workspace(
        std::size_t bytes) {
    if (bytes != 0) {
        throw std::invalid_argument(
                "CPU devices do not support positive raw workspace "
                "allocation");
    }
    return std::make_unique<CpuWorkspace>(*this, 0);
}

std::unique_ptr<Device> make_cpu_device(
        Allocator& allocator, QueueConfig queue_config) {
    return std::make_unique<CpuDevice>(allocator, queue_config);
}

}  // namespace iom
