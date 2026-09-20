#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fence.hpp"

namespace iom::detail {

struct EntryRegistration {
    EntryId source = 0;
    EntryId destination = 0;
};

class OutstandingWorkRegistry {
public:
    struct Entry {
        EntryId id = 0;
        void* address = nullptr;
        std::uint64_t sequence = 0;
        QueueId queue_id = 0;
        EntryState state = EntryState::Live;
        Fence fence;
    };

    using EntrySnapshot = Entry;

    OutstandingWorkRegistry() = default;
    ~OutstandingWorkRegistry() = default;

    OutstandingWorkRegistry(const OutstandingWorkRegistry&) = delete;
    OutstandingWorkRegistry& operator=(const OutstandingWorkRegistry&) = delete;
    OutstandingWorkRegistry(OutstandingWorkRegistry&&) = delete;
    OutstandingWorkRegistry& operator=(OutstandingWorkRegistry&&) = delete;

    void register_entry(
            EntryId id, void* address, std::uint64_t sequence,
            QueueId queue_id, Fence fence) {
        if (id == 0) {
            throw std::invalid_argument("outstanding-work entry id is zero");
        }
        if (address == nullptr) {
            throw std::invalid_argument(
                    "outstanding-work entry address is null");
        }
        if (sequence == 0) {
            throw std::invalid_argument(
                    "outstanding-work entry sequence is zero");
        }
        if (queue_id == 0) {
            throw std::invalid_argument(
                    "outstanding-work queue id is zero");
        }
        if (!fence) {
            throw std::invalid_argument(
                    "outstanding-work entry fence is empty");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (by_id_.find(id) != by_id_.end()) {
            throw std::invalid_argument(
                    "outstanding-work entry id is already registered");
        }
        auto [entry_it, inserted] = by_id_.emplace(
                id,
                Entry{id, address, sequence, queue_id, EntryState::Live,
                      std::move(fence)});
        if (!inserted) {
            throw std::invalid_argument(
                    "outstanding-work entry id is already registered");
        }
        try {
            by_address_.emplace(address, id);
        } catch (...) {
            by_id_.erase(entry_it);
            throw;
        }
    }

    [[nodiscard]] std::vector<Entry> snapshot_for(void* address) const {
        if (address == nullptr) {
            throw std::invalid_argument(
                    "outstanding-work snapshot address is null");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Entry> result;
        const auto range = by_address_.equal_range(address);
        for (auto it = range.first; it != range.second; ++it) {
            const auto entry = by_id_.find(it->second);
            if (entry == by_id_.end()) {
                throw std::logic_error(
                        "outstanding-work address index is inconsistent");
            }
            result.push_back(entry->second);
        }
        return result;
    }

    // Missing IDs are expected when a tensor destructor wins the race with
    // normal queue completion. Invalidated entries are retained for cleanup.

    [[nodiscard]] bool try_release_entry(EntryId id) noexcept {
        if (id == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto entry = by_id_.find(id);
        if (entry == by_id_.end() || entry->second.state != EntryState::Live) {
            return false;
        }
        erase_entry_locked(entry);
        return true;
    }


    void invalidate_entries_for_queue(QueueId queue_id) noexcept {
        if (queue_id == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& item : by_id_) {
            if (item.second.queue_id == queue_id) {
                invalidate_entry_locked(item.second);
            }
        }
    }

    void invalidate_entries(std::span<const EntryId> ids) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const EntryId id : ids) {
            const auto entry = by_id_.find(id);
            if (entry != by_id_.end()) {
                invalidate_entry_locked(entry->second);
            }
        }
    }

    void remove_entry_if_present(
            EntryId id, void* expected_address) noexcept {
        if (id == 0 || expected_address == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto entry = by_id_.find(id);
        if (entry != by_id_.end()
                && entry->second.address == expected_address) {
            erase_entry_locked(entry);
        }
    }

    void remove_entries(
            std::span<const EntryId> ids, void* expected_address) {
        if (expected_address == nullptr) {
            throw std::invalid_argument(
                    "outstanding-work removal address is null");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        validate_ids_locked(ids, expected_address);
        for (const EntryId id : ids) {
            erase_entry_locked(by_id_.find(id));
        }
    }

    void remove_entries(std::span<const EntryId> ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_ids_locked(ids, nullptr);
        for (const EntryId id : ids) {
            erase_entry_locked(by_id_.find(id));
        }
    }
private:

    void invalidate_entry_locked(Entry& entry) noexcept {
        entry.state = EntryState::Invalidated;
        entry.fence = make_invalidated_fence();
    }


    void validate_ids_locked(
            std::span<const EntryId> ids, void* expected_address) const {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == 0) {
                throw std::invalid_argument(
                        "outstanding-work removal entry id is zero");
            }
            for (std::size_t j = 0; j < i; ++j) {
                if (ids[i] == ids[j]) {
                    throw std::invalid_argument(
                            "outstanding-work removal contains duplicate ids");
                }
            }
            const auto entry = by_id_.find(ids[i]);
            if (entry == by_id_.end()) {
                throw std::invalid_argument(
                        "outstanding-work removal entry id is missing");
            }
            if (expected_address != nullptr
                    && entry->second.address != expected_address) {
                throw std::invalid_argument(
                        "outstanding-work removal address does not match");
            }
        }
    }

    void erase_entry_locked(
            std::map<EntryId, Entry>::const_iterator entry) noexcept {
        if (entry == by_id_.end()) {
            return;
        }
        const Entry& value = entry->second;
        const auto address_range = by_address_.equal_range(value.address);
        for (auto it = address_range.first; it != address_range.second; ++it) {
            if (it->second == value.id) {
                by_address_.erase(it);
                break;
            }
        }
        by_id_.erase(entry);
    }

    mutable std::mutex mutex_;
    std::map<EntryId, Entry> by_id_;
    std::multimap<void*, EntryId> by_address_;
};

template <typename MakeCleanup, typename Release>
    requires std::invocable<MakeCleanup> && std::invocable<Release>
inline void release_or_quarantine(
        OutstandingWorkRegistry& registry, void* address,
        MakeCleanup make_cleanup, Release release) noexcept {
    std::vector<OutstandingWorkRegistry::EntrySnapshot> snapshots;
    try {
        snapshots = registry.snapshot_for(address);
    } catch (...) {
        make_cleanup();
        return;
    }

    bool safe_to_release = true;
    for (const auto& snapshot : snapshots) {
        if (snapshot.state == EntryState::Invalidated || !snapshot.fence) {
            safe_to_release = false;
            continue;
        }
        try {
            const FenceResult result = snapshot.fence();
            safe_to_release =
                    safe_to_release && result.succeeded && !result.failure;
        } catch (...) {
            safe_to_release = false;
        }
    }

    if (safe_to_release) {
        for (const auto& snapshot : snapshots) {
            registry.remove_entry_if_present(snapshot.id, address);
        }
        release();
        return;
    }

    make_cleanup();
    for (const auto& snapshot : snapshots) {
        registry.remove_entry_if_present(snapshot.id, address);
    }
}


struct SequenceOutcome {
    EntryId source_entry_id = 0;
    EntryId destination_entry_id = 0;
    std::exception_ptr retained_failure;
    bool native_work_submitted = false;
};

[[nodiscard]] inline bool release_or_invalidate_entries(
        OutstandingWorkRegistry& registry, const SequenceOutcome& outcome,
        bool failure, bool fence_succeeded) noexcept {
    const std::array<EntryId, 2> entries{
            outcome.source_entry_id, outcome.destination_entry_id};
    if (failure || !fence_succeeded) {
        registry.invalidate_entries(entries);
        return false;
    }
    (void)registry.try_release_entry(entries[0]);
    (void)registry.try_release_entry(entries[1]);
    return true;
}

struct BinaryEntryRegistration {
    std::array<EntryId, 3> entries{};
    std::size_t count = 0;
};

struct BinaryOwnerRegistration {
    const void* identity = nullptr;
    void* address = nullptr;
};

struct SdpaEntryRegistration {
    std::array<EntryId, 4> entries{};
    std::size_t count = 0;
};

struct SdpaOwnerRegistration {
    const void* identity = nullptr;
    void* address = nullptr;
};


[[nodiscard]] inline bool release_or_invalidate_binary_entries(
        OutstandingWorkRegistry& registry, const BinaryEntryRegistration& outcome,
        bool failure, bool fence_succeeded) noexcept {
    const std::span<const EntryId> ids(outcome.entries.data(), outcome.count);
    if (failure || !fence_succeeded) {
        registry.invalidate_entries(ids);
        return false;
    }
    for (const EntryId id : ids) {
        (void)registry.try_release_entry(id);
    }
    return true;
}

[[nodiscard]] inline bool release_or_invalidate_sdpa_entries(
        OutstandingWorkRegistry& registry,
        const SdpaEntryRegistration& outcome, bool failure,
        bool fence_succeeded) noexcept {
    const std::span<const EntryId> ids(outcome.entries.data(), outcome.count);
    if (failure || !fence_succeeded) {
        registry.invalidate_entries(ids);
        return false;
    }
    for (const EntryId id : ids) {
        (void)registry.try_release_entry(id);
    }
    return true;
}

}  // namespace iom::detail
