#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <memory>
#include <span>

#include "iom/iom.hpp"
#include "iom/detail/outstanding_work_registry.hpp"
#include "../shared/metadata_slot_pool.hpp"
#include "../shared/queue_resources.hpp"

namespace iom::sycl_detail {

enum class SubmissionFault {
    none,
    state_allocation,
    fence_construction,
    outcome_insertion,
    first_submit,
    post_launch,
    queue_stream_create,
};

void inject_submission_fault_for_testing(SubmissionFault fault) noexcept;
void reset_fence_wait_count_for_testing() noexcept;

[[nodiscard]] std::size_t fence_wait_count_for_testing() noexcept;

void region_from_host(
        sycl::queue& transfer_queue, const Device& device,
        detail::RegistryState& registry_state,
        const TensorView& destination, RawWorkspaceView workspace,
        std::span<const std::byte> source, bool& resource_poisoned);

void region_to_host(
        sycl::queue& transfer_queue, const Device& device,
        detail::RegistryState& registry_state, const TensorView& source,
        RawWorkspaceView workspace, std::span<std::byte> destination,
        bool& resource_poisoned);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, detail::QueueResourceProvider& resource_provider,
        const sycl::context& context, const sycl::device& native_device,
        detail::RegistryState& registry_state);

#ifdef IOM_ENABLE_TESTING
// Observed fixed resource geometry of one live queue: the reserved
// partition (device address of its first 512-byte slot inside the Device
// metadata backing), the immutable per-queue C, and the current fixed-slot
// bookkeeping state. SYCL completion resources are the vendor event objects
// returned by each enqueued kernel and the fixed C-slot partition that
// bounds concurrent native in-flight work.
struct QueueResourceSnapshot {
    std::size_t slot_count = 0;
    void* device_base = nullptr;
    std::size_t slot_stride = 0;
    std::size_t events_total = 0;
    std::size_t events_in_use = 0;
    std::size_t slots_in_use = 0;
    std::size_t slots_protected = 0;
};

void queue_resource_snapshot_for_testing(
        DeviceOps& queue, QueueResourceSnapshot& snapshot);

// Attempts a covering proof for every queue lease retained by unknown
// completion on this Device's boundary; reclaimed leases return their
// partition and queue-count reservation.
void reclaim_retained_queue_leases_for_testing(Device& device);
#endif  // IOM_ENABLE_TESTING

}  // namespace iom::sycl_detail
