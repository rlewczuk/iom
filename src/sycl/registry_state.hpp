#pragma once

#include <cstdint>

#include "iom/detail/outstanding_work_registry.hpp"

namespace iom::sycl_detail {

struct SyclRegistryState {
    detail::OutstandingWorkRegistry registry;
    detail::Quarantine quarantine;
    detail::EntryId next_entry_id = 1;
    detail::QueueId next_queue_id = 1;
};

inline detail::QueueId allocate_queue_id(SyclRegistryState& state) {
    return detail::allocate_registry_queue_id(state.next_queue_id);
}

inline detail::EntryRegistration register_copy_entries(
        SyclRegistryState& state, detail::QueueId queue_id,
        std::uint64_t sequence, void* source, void* destination,
        const detail::Fence& fence) {
    return detail::register_registry_entries(
            state.registry, state.next_entry_id, queue_id, sequence, source,
            destination, fence);
}

}  // namespace iom::sycl_detail
