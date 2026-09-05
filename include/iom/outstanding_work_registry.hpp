#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "alloc.hpp"

namespace iom::detail {

using EntryId = std::uint64_t;
using QueueId = std::uint64_t;
enum class EntryState { Live, Invalidated };

struct FenceResult {
    bool succeeded = true;
    std::exception_ptr failure;

    [[nodiscard]] static FenceResult success() noexcept {
        return {true, nullptr};
    }

    [[nodiscard]] static FenceResult failed(
            std::exception_ptr error) noexcept {
        return {false, std::move(error)};
    }
};

using Fence = std::function<FenceResult()>;

struct EntryRegistration {
    EntryId source = 0;
    EntryId destination = 0;
};

class CleanupAction {
public:
    virtual ~CleanupAction() noexcept = default;

    virtual void run() noexcept = 0;
    [[nodiscard]] virtual bool failed() const noexcept = 0;
    [[nodiscard]] virtual std::exception_ptr failure() const noexcept = 0;

private:
    friend class Quarantine;
    std::unique_ptr<CleanupAction> next_;
};

class AllocatorCleanupAction final : public CleanupAction {
public:
    AllocatorCleanupAction(
            Allocator& allocator, void* address, std::size_t bytes,
            std::function<void()> pre_release = {})
            : allocator_(allocator), address_(address), bytes_(bytes),
              pre_release_(std::move(pre_release)) {}

    void run() noexcept override {
        if (attempted_ || address_ == nullptr) {
            return;
        }
        attempted_ = true;
        try {
            if (pre_release_) {
                pre_release_();
            }
        } catch (...) {
            remember_failure(std::current_exception());
        }
        try {
            allocator_.free(address_);
        } catch (...) {
            remember_failure(std::current_exception());
        }
        address_ = nullptr;
    }

    [[nodiscard]] bool failed() const noexcept override {
        return static_cast<bool>(failure_);
    }

    [[nodiscard]] std::exception_ptr failure() const noexcept override {
        return failure_;
    }

    [[nodiscard]] void* address() const noexcept { return address_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
    void remember_failure(std::exception_ptr failure) noexcept {
        if (failure_ == nullptr) {
            failure_ = std::move(failure);
        }
    }

    Allocator& allocator_;
    void* address_ = nullptr;
    std::size_t bytes_ = 0;
    std::function<void()> pre_release_;
    std::exception_ptr failure_;
    bool attempted_ = false;
};

class Quarantine {
public:
    Quarantine() = default;
    ~Quarantine() noexcept { drain(); }

    Quarantine(const Quarantine&) = delete;
    Quarantine& operator=(const Quarantine&) = delete;
    Quarantine(Quarantine&&) = delete;
    Quarantine& operator=(Quarantine&&) = delete;

    void add(std::unique_ptr<CleanupAction> action) noexcept {
        if (!action) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        action->next_ = std::move(actions_);
        actions_ = std::move(action);
    }

    template <typename Action, typename... Args>
    void emplace(Args&&... args) {
        add(std::make_unique<Action>(std::forward<Args>(args)...));
    }

    void drain() noexcept {
        for (;;) {
            std::unique_ptr<CleanupAction> action;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!actions_) {
                    return;
                }
                action = std::move(actions_);
                actions_ = std::move(action->next_);
            }
            action->run();
        }
    }

    [[nodiscard]] std::size_t size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t count = 0;
        for (const CleanupAction* action = actions_.get(); action != nullptr;
             action = action->next_.get()) {
            ++count;
        }
        return count;
    }

private:
    mutable std::mutex mutex_;
    std::unique_ptr<CleanupAction> actions_;
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

    struct EntrySnapshot {
        EntryId id = 0;
        void* address = nullptr;
        std::uint64_t sequence = 0;
        QueueId queue_id = 0;
        EntryState state = EntryState::Live;
        Fence fence;
    };

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
            auto address_it = by_address_.emplace(address, id);
            try {
                auto sequence_it = by_sequence_.emplace(sequence, id);
                try {
                    auto queue_it = by_queue_.emplace(queue_id, id);
                    (void)address_it;
                    (void)sequence_it;
                    (void)queue_it;
                } catch (...) {
                    by_sequence_.erase(sequence_it);
                    by_address_.erase(address_it);
                    by_id_.erase(entry_it);
                    throw;
                }
            } catch (...) {
                by_address_.erase(address_it);
                by_id_.erase(entry_it);
                throw;
            }
        } catch (...) {
            throw;
        }
    }

    [[nodiscard]] std::vector<EntrySnapshot> snapshot_for(
            void* address) const {
        if (address == nullptr) {
            throw std::invalid_argument(
                    "outstanding-work snapshot address is null");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<EntrySnapshot> result;
        const auto range = by_address_.equal_range(address);
        for (auto it = range.first; it != range.second; ++it) {
            const auto entry = by_id_.find(it->second);
            if (entry == by_id_.end()) {
                throw std::logic_error(
                        "outstanding-work address index is inconsistent");
            }
            const Entry& value = entry->second;
            result.push_back(
                    EntrySnapshot{value.id, value.address, value.sequence,
                                  value.queue_id, value.state, value.fence});
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

    void invalidate_entries_for_sequence(
            std::uint64_t sequence, QueueId queue_id) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = by_sequence_.lower_bound(sequence);
        while (it != by_sequence_.end() && it->first == sequence) {
            const EntryId id = it->second;
            const auto entry = by_id_.find(id);
            if (entry != by_id_.end()
                    && entry->second.queue_id == queue_id) {
                invalidate_entry_locked(entry->second);
                it = by_sequence_.lower_bound(sequence);
            } else {
                ++it;
            }
        }
    }

    void invalidate_entries_for_queue(QueueId queue_id) noexcept {
        if (queue_id == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto range = by_queue_.equal_range(queue_id);
        for (auto it = range.first; it != range.second; ++it) {
            const auto entry = by_id_.find(it->second);
            if (entry != by_id_.end()) {
                invalidate_entry_locked(entry->second);
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

    static FenceResult failed_invalidated_fence() noexcept {
        try {
            return FenceResult::failed(std::make_exception_ptr(
                    std::runtime_error(
                            "outstanding-work entry was invalidated")));
        } catch (...) {
            return FenceResult::failed(std::current_exception());
        }
    }

    void invalidate_entry_locked(Entry& entry) noexcept {
        entry.state = EntryState::Invalidated;
        entry.fence = &failed_invalidated_fence;
        erase_sequence_index_locked(entry.sequence, entry.id);
    }

    void erase_sequence_index_locked(
            std::uint64_t sequence, EntryId id) noexcept {
        const auto range = by_sequence_.equal_range(sequence);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == id) {
                by_sequence_.erase(it);
                return;
            }
        }
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

    void erase_entry_locked(std::map<EntryId, Entry>::const_iterator entry) noexcept {
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
        const auto sequence_range = by_sequence_.equal_range(value.sequence);
        for (auto it = sequence_range.first; it != sequence_range.second; ++it) {
            if (it->second == value.id) {
                by_sequence_.erase(it);
                break;
            }
        }
        const auto queue_range = by_queue_.equal_range(value.queue_id);
        for (auto it = queue_range.first; it != queue_range.second; ++it) {
            if (it->second == value.id) {
                by_queue_.erase(it);
                break;
            }
        }
        by_id_.erase(entry);
    }

    mutable std::mutex mutex_;
    std::map<EntryId, Entry> by_id_;
    std::multimap<void*, EntryId> by_address_;
    std::multimap<std::uint64_t, EntryId> by_sequence_;
    std::multimap<QueueId, EntryId> by_queue_;
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

}  // namespace iom::detail
