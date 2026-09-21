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
#include "storage_identity.hpp"

namespace iom::detail {

// Exclusive logical workspace slices share the existing outstanding-work
// registry. Each acquisition has its own EntryId, even when opaque slices
// share a canonical key bucket. Allocation, completion, and retention queries
// are serialized by RegistryState::allocation_mutex.
struct WorkspaceLease {
    // Opaque exact-owner identity; only compared, never dereferenced.
    const void* owner_identity = nullptr;
    StorageRange range;
    // Covering submission sequence of the accepted work using the range.
    std::uint64_t sequence = 0;
    // Outstanding-work entry retaining the owner/range lifetime until the
    // covering completion proof arrives.
    EntryId entry_id = 0;
    // Preserve the original covering fence even if its registry entry is
    // invalidated; invalidation is not a replacement completion proof.
    Fence covering_fence;
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
        const WorkspaceLease& lease, const void* owner_identity,
        const StorageRange& range) noexcept {
    // Retain the established exact-owner lease policy for addressable ranges.
    // Descriptor admission independently rejects cross-owner byte overlap.
    if (lease.range.backing.base != nullptr && range.backing.base != nullptr) {
        return lease.owner_identity == owner_identity
                && storage_ranges_overlap(lease.range, range);
    }
    if (lease.owner_identity == owner_identity
            && lease.range.backing.base == nullptr
            && range.backing == lease.range.backing) {
        return range.offset < lease.range.offset + lease.range.bytes
                && lease.range.offset < range.offset + range.bytes;
    }
    // Different wrappers need not share a logical offset origin. Equal opaque
    // backing keys therefore conflict conservatively across exact owners.
    return storage_ranges_overlap(lease.range, range);
}

struct RegistryState {
    OutstandingWorkRegistry registry;
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
    // Drain before the mutex and lease facts are destroyed.
    Quarantine quarantine;
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
[[nodiscard]] inline SdpaEntryRegistration register_sdpa_entries(
        RegistryState& state, QueueId queue_id, std::uint64_t sequence,
        std::span<const SdpaOwnerRegistration> owners,
        const Fence& fence) {
    std::lock_guard<std::mutex> lock(state.allocation_mutex);
    SdpaEntryRegistration result;
    try {
        for (std::size_t index = 0; index < owners.size(); ++index) {
            const SdpaOwnerRegistration owner = owners[index];
            if (owner.identity == nullptr || owner.address == nullptr) {
                throw std::invalid_argument(
                        "SDPA owner identity or address is null");
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
                throw std::invalid_argument("too many SDPA owners");
            }
            const EntryId id = allocate_registry_id(state.next_entry_id);
            state.registry.register_entry(
                    id, owner.address, sequence, queue_id, Fence(fence));
            result.entries[result.count++] = id;
        }
    } catch (...) {
        if (result.count != 0) {
            state.registry.remove_entries(
                    std::span<const EntryId>(
                            result.entries.data(), result.count));
        }
        throw;
    }
    return result;
}


// Acquires one checked nonempty 32-byte-aligned logical slice. Opaque keys
// are only compared; addressable ranges retain their real checked addresses.
// Failure leaves neither a lease nor a registry entry.
[[nodiscard]] inline WorkspaceLease acquire_workspace_lease(
        RegistryState& state, const void* owner_identity, StorageRange range,
        std::uint64_t sequence, QueueId queue_id, const Fence& fence) {
    if (owner_identity == nullptr) {
        throw std::invalid_argument(
                "workspace lease owner identity is null");
    }
    range = checked_storage_range(
            range.backing, range.offset, range.bytes,
            "workspace lease range end overflows");
    if (range.bytes == 0) {
        throw std::invalid_argument("workspace lease range is empty");
    }
    const std::uintptr_t range_begin =
            range.backing.base == nullptr ? range.offset
                    : reinterpret_cast<std::uintptr_t>(range.address());
    if (range_begin % 32 != 0) {
        throw std::invalid_argument(
                "workspace lease range is not 32-byte aligned");
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
    void* const key = range.registry_key();
    for (const WorkspaceLease& lease :
         state.workspace_leases.leases) {
        if (workspace_lease_ranges_overlap(lease, owner_identity, range)) {
            throw std::bad_alloc();
        }
    }

    WorkspaceLease result{owner_identity, range, sequence, 0, Fence(fence)};
    result.entry_id = allocate_registry_id(state.next_entry_id);
    try {
        state.registry.register_entry(
                result.entry_id, key, sequence, queue_id,
                Fence(fence));
    } catch (...) {
        // register_entry is atomic: after its own rollback the entry is
        // absent, so the no-throw removal keeps this rollback total.
        state.registry.remove_entry_if_present(result.entry_id, key);
        throw;
    }
    try {
        state.workspace_leases.leases.push_back(result);
    } catch (...) {
        state.registry.remove_entry_if_present(result.entry_id, key);
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
        RegistryState& state, EntryId entry_id, bool covering_proof) noexcept {
    std::lock_guard<std::mutex> lock(state.allocation_mutex);
    std::vector<WorkspaceLease>& leases =
            state.workspace_leases.leases;
    const auto matching = [&](const WorkspaceLease& candidate) {
        return candidate.entry_id == entry_id;
    };
    const auto found =
            std::find_if(leases.begin(), leases.end(), matching);
    if (found == leases.end()) {
        return;
    }
    if (covering_proof) {
        state.registry.remove_entry_if_present(
                entry_id, found->range.registry_key());
        leases.erase(found);
        return;
    }
    const std::array<EntryId, 1> quarantine{entry_id};
    state.registry.invalidate_entries(quarantine);
}

// Opaque backing held by other wrappers also retains the allocation; their
// offsets need not share an origin. Callers hold the allocation mutex.
[[nodiscard]] inline bool workspace_range_retained(
        const WorkspaceLeaseState& state, const void* owner_identity,
        const StorageRange& range) noexcept {
    if (range.bytes == 0) return false;
    for (const WorkspaceLease& lease : state.leases) {
        if (workspace_lease_ranges_overlap(lease, owner_identity, range)) {
            return true;
        }
    }
    return false;
}

// Retain the cleanup action itself (including any native identity it owns)
// until every relevant slice has a covering proof. Quarantine is not proof.
class WorkspaceCleanupAction final : public CleanupAction {
public:
    WorkspaceCleanupAction(
            RegistryState& state, const void* owner_identity, StorageRange range,
            std::unique_ptr<CleanupAction> cleanup)
            : state_(state), owner_identity_(owner_identity), range_(range),
              cleanup_(std::move(cleanup)) {}

    void run() noexcept override {
        {
            std::lock_guard<std::mutex> lock(state_.allocation_mutex);
            if (workspace_range_retained(
                        state_.workspace_leases, owner_identity_, range_)) return;
        }
        cleanup_->run();
        started_ = true;
    }
    [[nodiscard]] bool completed() const noexcept override {
        return started_ && cleanup_->completed();
    }
    [[nodiscard]] bool failed() const noexcept override {
        return cleanup_->failed();
    }
    [[nodiscard]] std::exception_ptr failure() const noexcept override {
        return cleanup_->failure();
    }

private:
    RegistryState& state_;
    const void* owner_identity_;
    StorageRange range_;
    std::unique_ptr<CleanupAction> cleanup_;
    bool started_ = false;
};

}  // namespace iom::detail
