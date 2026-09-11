#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "tensor.hpp"

namespace iom {

    class DeviceOps;

    /**
     * Immutable per-device queue configuration. `max_in_flight_per_queue`
     * bounds native in-flight work per queue and is selected once at Device
     * construction, applying to every queue the Device creates; zero is
     * invalid and is rejected with `std::invalid_argument` at construction.
     * The value is passed by value and is never mutated for the Device
     * lifetime.
     */
    struct QueueConfig {
        std::size_t max_in_flight_per_queue = 16;

        QueueConfig() = default;
        explicit QueueConfig(std::size_t max_in_flight)
                : max_in_flight_per_queue(max_in_flight) {
            if (max_in_flight_per_queue == 0) {
                throw std::invalid_argument(
                        "QueueConfig max_in_flight_per_queue must be nonzero");
            }
        }
    };

    /**
     * Explicit device memory configuration. `tensor_arena_bytes` reserves
     * the fixed tensor-data arena; standard-GPU factories require a nonzero
     * capacity divisible by 32 and reject other values with
     * `std::invalid_argument` before any native reservation.
     */
    struct DeviceMemoryConfig {
        std::size_t tensor_arena_bytes = 0;
    };

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
