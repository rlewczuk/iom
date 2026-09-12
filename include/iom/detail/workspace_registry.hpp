#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

#include "fence.hpp"
#include "outstanding_work_cleanup.hpp"
#include "outstanding_work_registry_core.hpp"

namespace iom::detail {

// ---------------------------------------------------------------------------
// Raw-workspace range leases (leaf 05). A lease is the exclusive right to
// use one byte range of one raw workspace owner for the lifetime of one
// queued submission. Records are keyed by the owner's exact identity, and
// overlap detection is range arithmetic on the leased extents, so disjoint
// aligned subranges of one owner coexist while overlapping ranges are
// resource exhaustion. Lease state lives alongside the existing
// owner-registration state and shares its allocation mutex; it never holds
// an allocator lock across submission, waits, callbacks, or drains.
struct WorkspaceLease {
    // Opaque exact-owner identity; only compared, never dereferenced.
    const void* owner_identity = nullptr;
    void* address = nullptr;
    std::size_t bytes = 0;
    // Covering submission sequence of the accepted work using the range.
    std::uint64_t sequence = 0;
    // Outstanding-work entry retaining the owner/range lifetime until the
    // covering completion proof arrives.
    EntryId entry_id = 0;
};

// Per-device lease bookkeeping, guarded by
// RegistryState::allocation_mutex. Quarantined records (completion
// unknown) keep blocking reuse of their owner range until a later
// covering completion proof resolves them; they never return the range
// to any allocator.
struct WorkspaceLeaseState {
    std::vector<WorkspaceLease> leases;
};

[[nodiscard]] inline bool workspace_lease_ranges_overlap(
        const WorkspaceLease& lease, std::uintptr_t begin,
        std::uintptr_t end) noexcept {
    const std::uintptr_t lease_begin =
            reinterpret_cast<std::uintptr_t>(lease.address);
    return begin < lease_begin + lease.bytes && lease_begin < end;
}

struct RegistryState {
    OutstandingWorkRegistry registry;
    Quarantine quarantine;
    EntryId next_entry_id = 1;
    QueueId next_queue_id = 1;
    // Raw-workspace exclusive-range lease bookkeeping (leaf 05); guarded
    // by the same allocation mutex as the id counters.
    WorkspaceLeaseState workspace_leases;
    // Serializes mutable ID allocation for this device registry. The
    // registry map mutex guards only map operations; this separate lock
    // guards the shared ID counters so concurrent queue creation and
    // submission reserve unique queue IDs and operation entry sets. It is
    // held across each complete copy pair or deduplicated ADD owner set.
    mutable std::mutex allocation_mutex;
};


inline EntryId allocate_registry_id(EntryId& next_id) {
    if (next_id == 0) {
        throw std::overflow_error("outstanding-work entry id is exhausted");
    }
    const EntryId id = next_id;
    if (next_id == std::numeric_limits<EntryId>::max()) {
        next_id = 0;
    } else {
        ++next_id;
    }
    return id;
}

inline QueueId allocate_registry_queue_id(QueueId& next_id) {
    if (next_id == 0) {
        throw std::overflow_error("outstanding-work queue id is exhausted");
    }
    const QueueId id = next_id;
    if (next_id == std::numeric_limits<QueueId>::max()) {
        next_id = 0;
    } else {
        ++next_id;
    }
    return id;
}

inline EntryRegistration register_registry_entries(
        OutstandingWorkRegistry& registry, EntryId& next_entry_id,
        QueueId queue_id, std::uint64_t sequence, void* source_address,
        void* destination_address, const Fence& fence) {
    constexpr EntryId max_id = std::numeric_limits<EntryId>::max();
    if (next_entry_id == 0 || next_entry_id > max_id - 1) {
        throw std::overflow_error(
                "outstanding-work entry id pair is exhausted");
    }
    const EntryId source = next_entry_id;
    const EntryId destination = source + 1;
    next_entry_id = destination == max_id ? 0 : destination + 1;
    registry.register_entry(
            source, source_address, sequence, queue_id, Fence(fence));
    try {
        registry.register_entry(
                destination, destination_address, sequence, queue_id,
                Fence(fence));
    } catch (...) {
        const std::array<EntryId, 1> rollback{source};
        registry.remove_entries(rollback);
        throw;
    }
    return {source, destination};
}


inline QueueId allocate_queue_id(RegistryState& state) {
    std::lock_guard<std::mutex> lock(state.allocation_mutex);
    return allocate_registry_queue_id(state.next_queue_id);
}

inline EntryRegistration register_copy_entries(
        RegistryState& state, QueueId queue_id, std::uint64_t sequence,
        void* source, void* destination, const Fence& fence) {
    std::lock_guard<std::mutex> lock(state.allocation_mutex);
    return register_registry_entries(
            state.registry, state.next_entry_id, queue_id, sequence, source,
            destination, fence);
}

[[nodiscard]] inline BinaryEntryRegistration register_binary_entries(
        RegistryState& state, QueueId queue_id, std::uint64_t sequence,
        std::span<const BinaryOwnerRegistration> owners, const Fence& fence) {
    std::lock_guard<std::mutex> lock(state.allocation_mutex);
    BinaryEntryRegistration result;
    try {
        for (std::size_t index = 0; index < owners.size(); ++index) {
            const BinaryOwnerRegistration owner = owners[index];
            if (owner.identity == nullptr || owner.address == nullptr) {
                throw std::invalid_argument("binary owner identity or address is null");
            }
            bool duplicate = false;
            for (std::size_t prior = 0; prior < index; ++prior) {
                if (owners[prior].identity == owner.identity) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }
            if (result.count == result.entries.size()) {
                throw std::invalid_argument("too many binary owners");
            }
            const EntryId id = allocate_registry_id(state.next_entry_id);
            state.registry.register_entry(
                    id, owner.address, sequence, queue_id, Fence(fence));
            result.entries[result.count++] = id;
        }
    } catch (...) {
        if (result.count != 0) {
            state.registry.remove_entries(
                    std::span<const EntryId>(result.entries.data(), result.count));
        }
        throw;
    }
    return result;
}

// Atomically acquires an exclusive lease on the 32-byte-aligned range
// [address, address + bytes) of one raw-workspace owner, registering the
// covering outstanding-work entry through `state` alongside the existing
// owner-registration machinery. Overlap with any live or quarantined
// lease of the same owner is resource exhaustion (`std::bad_alloc`);
// malformed, empty, misaligned, or overflowed ranges, a zero sequence or
// queue id, and an empty fence are invalid input
// (`std::invalid_argument`); overflow reports `std::overflow_error`. The
// transaction is all-or-nothing: a failure leaves neither a lease record
// nor a registry entry, and no allocator is touched.
[[nodiscard]] inline WorkspaceLease acquire_workspace_lease(
        RegistryState& state, const void* owner_identity, void* address,
        std::size_t bytes, std::uint64_t sequence, QueueId queue_id,
        const Fence& fence) {
    if (owner_identity == nullptr) {
        throw std::invalid_argument(
                "workspace lease owner identity is null");
    }
    if (address == nullptr) {
        throw std::invalid_argument(
                "workspace lease range address is null");
    }
    if (bytes == 0) {
        throw std::invalid_argument("workspace lease range is empty");
    }
    const std::uintptr_t range_begin =
            reinterpret_cast<std::uintptr_t>(address);
    if (range_begin % 32 != 0) {
        throw std::invalid_argument(
                "workspace lease range is not 32-byte aligned");
    }
    if (bytes
            > std::numeric_limits<std::uintptr_t>::max() - range_begin) {
        throw std::overflow_error("workspace lease range end overflows");
    }
    if (sequence == 0) {
        throw std::invalid_argument("workspace lease sequence is zero");
    }
    if (queue_id == 0) {
        throw std::invalid_argument("workspace lease queue id is zero");
    }
    if (!fence) {
        throw std::invalid_argument("workspace lease fence is empty");
    }

    std::lock_guard<std::mutex> lock(state.allocation_mutex);
    const std::uintptr_t range_end = range_begin + bytes;
    for (const WorkspaceLease& lease :
         state.workspace_leases.leases) {
        if (lease.owner_identity == owner_identity
                && workspace_lease_ranges_overlap(
                        lease, range_begin, range_end)) {
            throw std::bad_alloc();
        }
    }

    WorkspaceLease result{owner_identity, address, bytes, sequence, 0};
    result.entry_id = allocate_registry_id(state.next_entry_id);
    try {
        state.registry.register_entry(
                result.entry_id, address, sequence, queue_id,
                Fence(fence));
    } catch (...) {
        // register_entry is atomic: after its own rollback the entry is
        // absent, so the no-throw removal keeps this rollback total.
        state.registry.remove_entry_if_present(result.entry_id, address);
        throw;
    }
    try {
        state.workspace_leases.leases.push_back(result);
    } catch (...) {
        state.registry.remove_entry_if_present(result.entry_id, address);
        throw;
    }
    return result;
}

// Completes one lease with the caller's completion outcome. A covering
// completion proof releases the lease and its registry entry, so the
// range can be leased again. Anything else — failure, invalidated or
// pending fence state — is not proof: the range is quarantined, its
// covering entry invalidated, and the range stays out of reuse until a
// later covering completion proof resolves it. The resolution never returns
// the range to any allocator; storage release stays at the owning device's
// bookkeeping boundary.
inline void complete_workspace_lease(
        RegistryState& state, const WorkspaceLease& lease,
        bool covering_proof) noexcept {
    std::lock_guard<std::mutex> lock(state.allocation_mutex);
    std::vector<WorkspaceLease>& leases =
            state.workspace_leases.leases;
    const auto matching = [&](const WorkspaceLease& candidate) {
        return candidate.owner_identity == lease.owner_identity
                && candidate.address == lease.address
                && candidate.bytes == lease.bytes;
    };
    const auto found =
            std::find_if(leases.begin(), leases.end(), matching);
    if (found == leases.end()) {
        return;
    }
    if (covering_proof) {
        leases.erase(found);
        state.registry.remove_entry_if_present(
                lease.entry_id, lease.address);
        return;
    }
    const std::array<EntryId, 1> quarantine{lease.entry_id};
    state.registry.invalidate_entries(quarantine);
}

// True when any live or quarantined lease of `owner_identity` overlaps
// [address, address + bytes). The owning device consults this at
// workspace destruction so a retained lease keeps the arena range out of
// the allocator until its completion proof arrives.
[[nodiscard]] inline bool workspace_range_retained(
        const WorkspaceLeaseState& state, const void* owner_identity,
        void* address, std::size_t bytes) noexcept {
    if (address == nullptr || bytes == 0) {
        return false;
    }
    const std::uintptr_t range_begin =
            reinterpret_cast<std::uintptr_t>(address);
    const std::uintptr_t range_end = range_begin + bytes;
    for (const WorkspaceLease& lease : state.leases) {
        if (lease.owner_identity == owner_identity
                && workspace_lease_ranges_overlap(
                        lease, range_begin, range_end)) {
            return true;
        }
    }
    return false;
}

}  // namespace iom::detail
