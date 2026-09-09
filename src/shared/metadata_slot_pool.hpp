#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>

#include <stdexcept>
namespace iom::detail {

template <typename Policy>
class MetadataSlotPool final {
public:
    static constexpr std::size_t kMetadataSlotCount = 16;

    explicit MetadataSlotPool(typename Policy::context_type context)
            : context_(context) {}

    ~MetadataSlotPool() noexcept {
        try {
            Policy::activate(context_);
        } catch (...) {
        }
        for (Slot& slot : slots_) {
            // A protected slot may still be referenced by work whose
            // completion could not be proven. Do not speculatively free it
            // during pool teardown; retain both device and host storage.
            if (!slot.protected_) {
                Policy::free_noexcept(slot.device);
            } else {
                (void)slot.host.release();
            }
            slot.device = nullptr;
        }
    }

    MetadataSlotPool(const MetadataSlotPool&) = delete;
    MetadataSlotPool& operator=(const MetadataSlotPool&) = delete;

    void protect(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < slots_.size()) {
            slots_[index].protected_ = true;
            slots_[index].in_use = true;
        }
    }

    void release_after_proof(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < slots_.size() && slots_[index].in_use) {
            slots_[index].protected_ = false;
            slots_[index].in_use = false;
            completion_.notify_one();
        }
    }

    [[nodiscard]] std::size_t acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        completion_.wait(lock, [this] {
            for (const Slot& slot : slots_) {
                if (!slot.in_use) {
                    return true;
                }
            }
            return false;
        });
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            if (!slots_[index].in_use) {
                slots_[index].in_use = true;
                return index;
            }
        }
        throw std::logic_error("metadata slot acquisition lost a free slot");
    }

    void release(std::size_t index) noexcept {
        release_after_proof(index);
    }

    // acquire() only returns a slot after its owning fence has retired, so
    // replacing its storage does not require draining unrelated stream work.
    void ensure_slot_capacity(
            std::size_t index, std::size_t required_bytes) {
        Slot& slot = slots_.at(index);
        if (slot.capacity >= required_bytes) {
            return;
        }
        std::size_t capacity = slot.capacity == 0 ? 256 : slot.capacity;
        while (capacity < required_bytes) {
            if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
                capacity = required_bytes;
                break;
            }
            capacity *= 2;
        }
        std::unique_ptr<std::byte[]> replacement(
                new std::byte[capacity]);
        void* replacement_device = Policy::allocate(capacity);
        void* old_device = slot.device;
        slot.host = std::move(replacement);
        slot.device = replacement_device;
        slot.capacity = capacity;
        Policy::free_noexcept(old_device);
    }

    [[nodiscard]] std::byte* host_data(std::size_t index) {
        return slots_.at(index).host.get();
    }

    [[nodiscard]] void* device_data(std::size_t index) {
        return slots_.at(index).device;
    }

private:
    struct Slot {
        std::unique_ptr<std::byte[]> host;
        void* device = nullptr;
        std::size_t capacity = 0;
        bool in_use = false;
        bool protected_ = false;
    };

    typename Policy::context_type context_;
    std::array<Slot, kMetadataSlotCount> slots_;
    std::mutex mutex_;
    std::condition_variable completion_;
};

}  // namespace iom::detail

