#pragma once

#include <cstdint>

#include "iom/outstanding_work_registry.hpp"

namespace iom::rocm_detail {

struct RocmRegistryState {
    detail::OutstandingWorkRegistry registry;
    detail::Quarantine quarantine;
    detail::EntryId next_entry_id = 1;
    detail::QueueId next_queue_id = 1;
};

inline detail::QueueId allocate_queue_id(RocmRegistryState& state) {
    return detail::allocate_registry_queue_id(state.next_queue_id);
}

inline detail::EntryRegistration register_copy_entries(
        RocmRegistryState& state, detail::QueueId queue_id,
        std::uint64_t sequence, void* source, void* destination,
        const detail::Fence& fence) {
    return detail::register_registry_entries(
            state.registry, state.next_entry_id, queue_id, sequence, source,
            destination, fence);
}

}  // namespace iom::rocm_detail
