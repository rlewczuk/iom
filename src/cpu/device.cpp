#include "iom/cpu/device.hpp"
#include "device_internal.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>

#include "iom/iom.hpp"
#include "iom/detail/aligned_storage.hpp"
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
    CpuWorkspace(
            CpuDevice& device, Allocator& allocator, std::size_t bytes)
            : RawWorkspace(device, bytes), allocator_(allocator) {
        if (bytes != 0) {
            address_ = detail::allocate_aligned_storage(
                    allocator_, bytes, [] {},
                    "CPU workspace is not 32-byte aligned");
        }
    }

    ~CpuWorkspace() noexcept override {
        detail::release_aligned_storage(allocator_, address_);
    }

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    Allocator& allocator_;
    void* address_ = nullptr;
};

}  // namespace

std::unique_ptr<RawWorkspace> CpuDevice::create_workspace(
        std::size_t bytes) {
    return std::make_unique<CpuWorkspace>(*this, allocator_, bytes);
}

std::unique_ptr<Device> make_cpu_device(
        Allocator& allocator, QueueConfig queue_config) {
    return std::make_unique<CpuDevice>(allocator, queue_config);
}

}  // namespace iom
