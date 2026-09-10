#pragma once

#include <cstdint>
#include <memory>

#include "tensor.hpp"

namespace iom {

    class DeviceOps;

    /**
     * Backend-neutral device contract. One instance owns one backend runtime
     * context and creates that context's tensors and operation queues. Device
     * identity is exact: queues reject views from another instance even on the
     * same ordinal. The device owner must outlive every tensor and queue it
     * created; tensors and queues retain stable caller-owned identities. There
     * is no registry or backend switch, and this header stays free of runtime
     * types.
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
        /**
         * Returns an immutable storage table for `QuantizationFormat::NONE`.
         * ADD, MUL, and SUB accept the 21 required numeric leaves; DIV accepts
         * the nine floating leaves. BOOL, F8_E8M0, non-NONE quantization, and
         * integer DIV are unsupported by the operation facades after common
         * validation. Required leaves are not narrowed for SDK limitations:
         * staging or emulation is internal.
         */
        [[nodiscard]] virtual std::span<const iom::DataType> supported_data_types() const noexcept = 0;
        [[nodiscard]] virtual std::uint32_t backend_device() const noexcept = 0;

        [[nodiscard]] virtual std::unique_ptr<Tensor> create_tensor(
            const TensorSpec& spec) = 0;
        [[nodiscard]] virtual std::unique_ptr<DeviceOps> create_ops() = 0;
    };

}  // namespace iom
