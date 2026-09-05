#pragma once

#include <cuda.h>

#include <cstddef>
#include <memory>
#include <span>

#include "transfer_pool.hpp"
#include "registry_state.hpp"

#include "iom/iom.hpp"

namespace iom::cuda_detail {
enum class SubmissionFault {
    none,
    event_create,
    third_plane_launch,
    event_record,
};

void inject_submission_fault_for_testing(SubmissionFault fault) noexcept;


void region_from_host(
        TransferStreamPool& pool, CUcontext context,
        const TensorView& destination, std::span<const std::byte> source);

void region_to_host(
        TransferStreamPool& pool, CUcontext context,
        const TensorView& source, std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, CUcontext context,
        CudaRegistryState& registry_state);

}  // namespace iom::cuda_detail
