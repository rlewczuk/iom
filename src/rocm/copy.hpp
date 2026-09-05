#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <memory>
#include <span>

#include "registry_state.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"


namespace iom::rocm_detail {
enum class SubmissionFault {
    none,
    event_create,
    third_plane_launch,
    event_record,
};

void inject_submission_fault_for_testing(SubmissionFault fault) noexcept;


void region_from_host(
        int device_ordinal, const TensorView& destination,
        std::span<const std::byte> source);

void region_to_host(
        int device_ordinal, const TensorView& source,
        std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, int device_ordinal,
        RocmRegistryState& registry_state);

}  // namespace iom::rocm_detail
