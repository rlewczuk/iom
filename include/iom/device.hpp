#pragma once

#include <cstdint>
#include <memory>

#include "tensor.hpp"

namespace iom {

    class DeviceOps;

    /**
     * Backend-neutral device contract. One instance owns one backend runtime
     * context and creates that context's tensors and operation queues. There
     * is no registry and no backend switch: each backend ships a separate
     * factory returning std::unique_ptr<Device>, and user code subclasses
     * Device is non-copyable and non-movable so its owner address stays stable
     * for the tensors and queues it created. The Device must outlive every
     * tensor and queue it created. This header stays free of any CUDA, HIP,
     * SYCL, or TTNN type.
     */
    class Device {
    public:
        virtual ~Device() = default;
        Device() = default;
        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;
        Device(Device&&) = delete;
        Device& operator=(Device&&) = delete;

        [[nodiscard]] virtual BackendKind backend_kind() const noexcept = 0;
        [[nodiscard]] virtual std::uint32_t backend_device() const noexcept = 0;

        [[nodiscard]] virtual std::unique_ptr<Tensor> create_tensor(
            const TensorSpec& spec) = 0;
        [[nodiscard]] virtual std::unique_ptr<DeviceOps> create_ops() = 0;
    };

}  // namespace iom
