#pragma once

#include <cstddef>
#include <concepts>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "iom/alloc.hpp"

namespace iom::detail {

inline constexpr std::size_t kFenceStorageBytes = 32;
inline constexpr std::size_t kFenceStorageAlign = alignof(std::max_align_t);

using EntryId = std::uint64_t;
using QueueId = std::uint64_t;
enum class EntryState { Live, Invalidated };

struct FenceResult {
    bool succeeded = true;
    std::exception_ptr failure;

    [[nodiscard]] static FenceResult success() noexcept {
        return {true, nullptr};
    }

    // A submission that has not reached a terminal state yet. Never a
    // success: callers that observe it must hold the storage back.
    [[nodiscard]] static FenceResult pending() noexcept {
        return {false, nullptr};
    }

    [[nodiscard]] static FenceResult failed(
            std::exception_ptr error) noexcept {
        return {false, std::move(error)};
    }
};

struct Fence {
    alignas(kFenceStorageAlign) unsigned char storage[kFenceStorageBytes]{};
    FenceResult (*invoke)(const Fence&) noexcept = nullptr;
    void (*copy_construct)(Fence* dst, const Fence& src) noexcept = nullptr;
    void (*move_construct)(Fence* dst, Fence* src) noexcept = nullptr;
    void (*destroy)(Fence*) noexcept = nullptr;

    Fence() noexcept = default;

    ~Fence() noexcept {
        if (destroy != nullptr) {
            destroy(this);
        }
    }

    Fence(const Fence& other) noexcept {
        if (other.copy_construct != nullptr) {
            other.copy_construct(this, other);
        } else {
            std::memcpy(storage, other.storage, kFenceStorageBytes);
        }
        invoke = other.invoke;
        copy_construct = other.copy_construct;
        move_construct = other.move_construct;
        destroy = other.destroy;
    }

    Fence(Fence&& other) noexcept {
        if (other.move_construct != nullptr) {
            other.move_construct(this, &other);
        } else {
            std::memcpy(storage, other.storage, kFenceStorageBytes);
        }
        invoke = other.invoke;
        copy_construct = other.copy_construct;
        move_construct = other.move_construct;
        destroy = other.destroy;
        other.invoke = nullptr;
        other.copy_construct = nullptr;
        other.move_construct = nullptr;
        other.destroy = nullptr;
    }

    Fence& operator=(const Fence& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (destroy != nullptr) {
            destroy(this);
        }
        if (other.copy_construct != nullptr) {
            other.copy_construct(this, other);
        } else {
            std::memcpy(storage, other.storage, kFenceStorageBytes);
        }
        invoke = other.invoke;
        copy_construct = other.copy_construct;
        move_construct = other.move_construct;
        destroy = other.destroy;
        return *this;
    }

    Fence& operator=(Fence&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (destroy != nullptr) {
            destroy(this);
        }
        if (other.move_construct != nullptr) {
            other.move_construct(this, &other);
        } else {
            std::memcpy(storage, other.storage, kFenceStorageBytes);
        }
        invoke = other.invoke;
        copy_construct = other.copy_construct;
        move_construct = other.move_construct;
        destroy = other.destroy;
        other.invoke = nullptr;
        other.copy_construct = nullptr;
        other.move_construct = nullptr;
        other.destroy = nullptr;
        return *this;
    }

    FenceResult operator()() const { return invoke(*this); }
    explicit operator bool() const noexcept { return invoke != nullptr; }
};

static_assert(std::is_nothrow_default_constructible_v<Fence>);
static_assert(std::is_nothrow_copy_constructible_v<Fence>);
static_assert(std::is_nothrow_move_constructible_v<Fence>);
static_assert(std::is_nothrow_destructible_v<Fence>);
static_assert(std::is_nothrow_copy_assignable_v<Fence>);
static_assert(std::is_nothrow_move_assignable_v<Fence>);

template <typename Capture>
struct FenceCaptureOps {
    static_assert(std::is_nothrow_copy_constructible_v<Capture>
            && std::is_nothrow_move_constructible_v<Capture>
            && std::is_nothrow_destructible_v<Capture>);
    static_assert(sizeof(Capture) <= kFenceStorageBytes);
    static_assert(alignof(Capture) <= kFenceStorageAlign);

    static void copy_construct(
            Fence* destination, const Fence& source) noexcept {
        ::new (destination->storage) Capture{
                *std::launder(reinterpret_cast<const Capture*>(
                        source.storage))};
    }

    static void move_construct(Fence* destination, Fence* source) noexcept {
        ::new (destination->storage) Capture{
                std::move(*std::launder(reinterpret_cast<Capture*>(
                        source->storage)))};
        std::destroy_at(std::launder(reinterpret_cast<Capture*>(
                source->storage)));
    }

    static void destroy(Fence* fence) noexcept {
        std::destroy_at(
                std::launder(reinterpret_cast<Capture*>(fence->storage)));
    }
};

inline FenceResult failed_invalidated_fence_invoke(
        const Fence&) noexcept {
    try {
        return FenceResult::failed(std::make_exception_ptr(
                std::runtime_error(
                        "outstanding-work entry was invalidated")));
    } catch (...) {
        return FenceResult::failed(std::current_exception());
    }
}

inline Fence make_invalidated_fence() noexcept {
    Fence f;
    f.invoke = &failed_invalidated_fence_invoke;
    return f;
}

struct EntryRegistration {
    EntryId source = 0;
    EntryId destination = 0;
};

class CleanupAction {
public:
    virtual ~CleanupAction() noexcept = default;

    virtual void run() noexcept = 0;
    [[nodiscard]] virtual bool completed() const noexcept { return true; }
    [[nodiscard]] virtual bool failed() const noexcept = 0;
    [[nodiscard]] virtual std::exception_ptr failure() const noexcept = 0;
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
 
    [[nodiscard]] bool completed() const noexcept override {
        return attempted_;
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
    ~Quarantine() noexcept {
        drain();
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& action : actions_) {
            (void)action.release();
        }
    }

    Quarantine(const Quarantine&) = delete;
    Quarantine& operator=(const Quarantine&) = delete;
    Quarantine(Quarantine&&) = delete;
    Quarantine& operator=(Quarantine&&) = delete;

    void add(std::unique_ptr<CleanupAction> action) {
        if (!action) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            actions_.push_back(std::move(action));
        } catch (...) {
            (void)action.release();
            throw;
        }
    }

    template <typename Action, typename... Args>
    void emplace(Args&&... args) {
        add(std::make_unique<Action>(std::forward<Args>(args)...));
    }

    void drain() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = actions_.size(); index != 0; --index) {
            actions_[index - 1]->run();
        }

        std::size_t retained = 0;
        for (std::size_t index = 0; index < actions_.size(); ++index) {
            if (!actions_[index]->completed()) {
                if (retained != index) {
                    actions_[retained] = std::move(actions_[index]);
                }
                ++retained;
            }
        }
        actions_.resize(retained);
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<CleanupAction>> actions_;
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

struct RegistryState {
    OutstandingWorkRegistry registry;
    Quarantine quarantine;
    EntryId next_entry_id = 1;
    QueueId next_queue_id = 1;
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

struct BinaryEntryRegistration {
    std::array<EntryId, 3> entries{};
    std::size_t count = 0;
};
struct BinaryOwnerRegistration {
    const void* identity = nullptr;
    void* address = nullptr;
};

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

}  // namespace iom::detail
