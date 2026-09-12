#pragma once

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "iom/alloc.hpp"

namespace iom::detail {

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

}  // namespace iom::detail
