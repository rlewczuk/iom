#pragma once

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

#include "iom/alloc.hpp"
#include "iom/detail/outstanding_work_registry.hpp"
#include "iom/sycl/device.hpp"
#include "../shared/queue_resources.hpp"

namespace iom {

    class SyclTensor;

    class SyclDevice final : public Device,
                             public detail::QueueResourceProvider {
    public:
        SyclDevice(
                std::uint32_t ordinal, sycl::device device,
                sycl::context context,
                std::unique_ptr<ListAllocator> data_allocator,
                std::unique_ptr<FixedSizeAllocator> metadata_allocator,
                void* data_backing, std::size_t data_backing_bytes,
                void* metadata_backing);

        SyclDevice(const SyclDevice&) = delete;
        SyclDevice& operator=(const SyclDevice&) = delete;
        ~SyclDevice() override;

        [[nodiscard]] BackendKind backend_kind() const noexcept override;
        [[nodiscard]] std::uint32_t backend_device() const noexcept override;
        [[nodiscard]] std::span<const iom::DataType>
                supported_data_types() const noexcept override;

        [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                const TensorSpec& spec) override;
        [[nodiscard]] std::unique_ptr<RawWorkspace> create_workspace(
                std::size_t bytes) override;
        [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

        // -- detail::QueueResourceProvider: fixed queue-resource geometry.
        [[nodiscard]] std::size_t queue_slot_count() const noexcept override;
        [[nodiscard]] detail::QueueResourceLease
                reserve_queue_resources() override;
        void release_queue_resources(
                std::size_t first_slot,
                std::size_t slot_count) noexcept override;
        void retain_unknown_lease(
                std::function<bool()> reclaim,
                std::function<void()> discard) override;
        void reclaim_retained_leases() override;

        [[nodiscard]] const sycl::context& context() const noexcept;
        [[nodiscard]] const sycl::device& native_device() const noexcept;
        [[nodiscard]] sycl::queue& transfer_queue() noexcept;
        [[nodiscard]] detail::RegistryState& registry_state() noexcept;

        // Tensor and raw-workspace data ranges suballocate from the device
        // data arena under the device bookkeeping boundary.
        [[nodiscard]] void* allocate_data(std::size_t bytes);
        void release_data(void* address);
        void release_workspace(
                const RawWorkspace* owner, void* address,
                std::size_t bytes) noexcept;
        [[nodiscard]] iom::Allocator& data_allocator() noexcept;
        [[nodiscard]] void* data_backing() const noexcept;
        [[nodiscard]] std::size_t data_backing_bytes() const noexcept;

    private:
        struct RetainedLease {
            std::function<bool()> reclaim;
            std::function<void()> discard;
        };

        void release_queue_resources_locked(
                std::size_t first_slot,
                std::size_t slot_count) noexcept;
        void discard_all_retained_leases() noexcept;

        sycl::device device_;
        std::optional<sycl::context> context_;
        std::optional<sycl::queue> transfer_queue_;
        detail::RegistryState registry_state_;
        std::uint32_t ordinal_;
        mutable std::mutex transfer_mutex_;
        bool transfer_resource_poisoned_ = false;
        std::mutex bookkeeping_mutex_;
        std::unique_ptr<ListAllocator> data_allocator_;
        std::unique_ptr<FixedSizeAllocator> metadata_allocator_;
        void* data_backing_ = nullptr;
        std::size_t data_backing_bytes_ = 0;
        void* metadata_backing_ = nullptr;
        // Fixed queue-resource geometry: exactly C slots per partition,
        // four partitions, and at most one retained lease per credit.
        std::size_t queue_slot_count_ = 0;
        std::mutex retained_mutex_;
        std::array<RetainedLease, detail::kMaxLiveGpuQueues>
                retained_leases_{};
        std::size_t retained_count_ = 0;

        friend class SyclTensor;
    };

}  // namespace iom
