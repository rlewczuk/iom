#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <memory>
#include <span>

#include "iom/iom.hpp"
#include "registry_state.hpp"
#include "staging_pool.hpp"

namespace iom::sycl_detail {

enum class SubmissionFault {
    none,
    state_allocation,
    fence_construction,
    outcome_insertion,
    first_submit,
    post_launch,
};

void inject_submission_fault_for_testing(SubmissionFault fault) noexcept;
void reset_fence_wait_count_for_testing() noexcept;

[[nodiscard]] std::size_t fence_wait_count_for_testing() noexcept;

void region_from_host(
        StagingSlotPool& staging_pool, sycl::queue& transfer_queue,
        const TensorView& destination, void* storage,
        std::span<const std::byte> source);

void region_to_host(
        StagingSlotPool& staging_pool, sycl::queue& transfer_queue,
        const TensorView& source, const void* storage,
        std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, const sycl::context& context,
        const sycl::device& native_device, SyclRegistryState& registry_state);

}  // namespace iom::sycl_detail
