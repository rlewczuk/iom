#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>

#include "tensor.hpp"

namespace iom {

    class DeviceOps;
    class RawWorkspace;

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
        /**
         * Creates one explicitly owned raw device workspace. Zero bytes
         * return a valid allocation-free empty owner on every backend;
         * positive bytes suballocate the already reserved 32-byte-aligned
         * data arena on CUDA, ROCm, and SYCL (never a new native backing;
         * exhaustion including fragmentation is std::bad_alloc) and are
         * rejected as unsupported device scratch with
         * `std::invalid_argument` on CPU and TTNN. The owner is stable,
         * non-copyable, non-movable, and must be destroyed before its
         * creating Device.
         */
        [[nodiscard]] virtual std::unique_ptr<RawWorkspace>
                create_workspace(std::size_t bytes) = 0;
        [[nodiscard]] virtual std::unique_ptr<DeviceOps> create_ops() = 0;

        /**
         * True exactly when `workspace` is a live owner created by this
         * Device. The pointer is used only as an opaque identity for the
         * live-owner set, so shared validation can reject foreign and dead
         * owners without dereferencing them.
         */
        [[nodiscard]] bool owns_workspace(
                const RawWorkspace* workspace) const noexcept;

    protected:
        friend class RawWorkspace;

        // RawWorkspace construction and destruction maintain the exact
        // live-owner identity set these hooks guard.
        void register_workspace(const RawWorkspace* workspace) const;
        void unregister_workspace(
                const RawWorkspace* workspace) const noexcept;

    private:
        mutable std::mutex workspace_registry_mutex_;
        mutable std::set<const RawWorkspace*> live_workspaces_;
    };

}  // namespace iom
