#pragma once

// Backend-neutral conformance harness: lifetime, capability, and suite
// scenarios (change 0001-tensor-view / 06).
//
// Provides the deterministic deferred and instrumented queue fixtures, the
// lifetime scenario that exercises stable addresses and repeatable waits,
// the compute-capability scenario that proves unsupported methods fail
// before submission, and the full-suite dispatcher that runs every
// scenario in dependency order. Depends on backend_conformance_common.hpp
// for the shared types and on backend_conformance_copy_storage.hpp for
// the storage-and-copy scenarios referenced by the suite dispatcher.

#include "backend/backend_conformance_copy_storage.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

// ---------------------------------------------------------------------------
// Deterministic deferred fake. Submissions are recorded with the exact view,
// owner, and native-handle addresses; the test thread plays the in-order
// worker by completing sequences through the common DeviceOps machinery.
// ---------------------------------------------------------------------------

class DeferredCopyQueue final : public iom::DeviceOps {
public:
    using iom::DeviceOps::complete;

    struct Record {
        std::uint64_t sequence;
        const void* source_owner;
        const iom::TensorView* source;
        const void* source_handle;
        void* destination_owner;
        iom::TensorView* destination;
        void* destination_handle;
    };

    [[nodiscard]] const std::vector<Record>& records() const noexcept {
        return records_;
    }

    // Submission that also records the owner addresses, mirroring what a
    // real queue observes about its operands.
    iom::oid copy(
            const iom::Tensor& source_owner, const iom::TensorView& source,
            iom::Tensor& destination_owner, iom::TensorView& destination) {
        const iom::oid token = copy(source, destination);
        records_.back().source_owner = &source_owner;
        records_.back().destination_owner = &destination_owner;
        return token;
    }

    iom::oid copy(
            const iom::TensorView& source,
            iom::TensorView& destination) override {
        return submit([&](std::uint64_t sequence) {
            records_.push_back(
                    {sequence, nullptr, &source, source.native_handle(),
                     nullptr, &destination, destination.native_handle()});
        });
    }

    // View-less submission used to observe queue identity and sequence
    // allocation directly.
    iom::oid probe() {
        return submit([&](std::uint64_t sequence) {
            records_.push_back(
                    {sequence, nullptr, nullptr, nullptr, nullptr, nullptr,
                     nullptr});
        });
    }

    // Submission-time view, owner, and native-handle addresses must equal
    // the live operands at completion time.
    void expect_stable(
            std::uint64_t sequence,
            const iom::Tensor& source_owner, const iom::TensorView& source,
            iom::Tensor& destination_owner, iom::TensorView& destination) const {
        const Record* record = find(sequence);
        REQUIRE_MESSAGE(record != nullptr,
                        "no deferred record for sequence " << sequence);
        CHECK_EQ(record->source_owner, &source_owner);
        CHECK_EQ(record->source, &source);
        CHECK_EQ(record->source_handle, source.native_handle());
        CHECK_EQ(record->destination_owner, &destination_owner);
        CHECK_EQ(record->destination, &destination);
        CHECK_EQ(record->destination_handle, destination.native_handle());
    }

    iom::oid add(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement add");
    }
    iom::oid mul(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement mul");
    }
    iom::oid silu(const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement silu");
    }
    iom::oid linear(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement linear");
    }
    iom::oid rmsnorm(const iom::TensorView&, iom::TensorView&, const iom::TensorView&,
                     float, size_t) override {
        throw std::runtime_error("deferred fake does not implement rmsnorm");
    }
    iom::oid sdpa(const iom::TensorView&, const iom::TensorView&, const iom::TensorView&,
                  size_t, size_t, size_t, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement sdpa");
    }

private:
    [[nodiscard]] const Record* find(std::uint64_t sequence) const {
        for (const Record& record : records_) {
            if (record.sequence == sequence) {
                return &record;
            }
        }
        return nullptr;
    }

    std::vector<Record> records_;
};

// ---------------------------------------------------------------------------
// Instrumented queue: journals submissions, completions, and its own
// destruction so the test can prove destruction neither synchronizes on
// outstanding work nor cancels it.
// ---------------------------------------------------------------------------

class InstrumentedQueue final : public iom::DeviceOps {
public:
    struct Event {
        enum class Kind { submit, complete, destroy_begin, destroy_end };

        Kind kind;
        std::uint64_t sequence;
    };

    explicit InstrumentedQueue(
            std::shared_ptr<std::vector<Event>> journal)
            : journal_(std::move(journal)) {}

    ~InstrumentedQueue() override {
        journal_->push_back({Event::Kind::destroy_begin, 0});
        // Nothing here waits for, completes, or cancels the outstanding
        // sequences; the common base only releases the queue id.
        journal_->push_back({Event::Kind::destroy_end, 0});
    }

    iom::oid probe() {
        return submit([&](std::uint64_t sequence) {
            journal_->push_back({Event::Kind::submit, sequence});
        });
    }

    // Test-driven completion of one sequence.
    void finish(std::uint64_t sequence) {
        journal_->push_back({Event::Kind::complete, sequence});
        complete(sequence);
    }

    iom::oid copy(const iom::TensorView&, iom::TensorView&) override {
        return probe();
    }
    iom::oid add(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement add");
    }
    iom::oid mul(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement mul");
    }
    iom::oid silu(const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement silu");
    }
    iom::oid linear(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement linear");
    }
    iom::oid rmsnorm(const iom::TensorView&, iom::TensorView&, const iom::TensorView&,
                     float, size_t) override {
        throw std::runtime_error("instrumented queue does not implement rmsnorm");
    }
    iom::oid sdpa(const iom::TensorView&, const iom::TensorView&, const iom::TensorView&,
                  size_t, size_t, size_t, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement sdpa");
    }

private:
    std::shared_ptr<std::vector<Event>> journal_;
};

// ---------------------------------------------------------------------------
// Lifetime and capability scenarios.
// ---------------------------------------------------------------------------

// Deferred-queue lifetime: stable view, owner, and native-handle addresses
// between submission and wait, repeatable waits, in-order completion,
// injected asynchronous failures rethrown on repeated waits, and queue
// destruction that neither synchronizes nor cancels.
inline void run_lifetime_conformance(
        iom::Device& candidate,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    REQUIRE_FALSE(supported_types.empty());
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, supported_types.front()};

    // Stable addresses across a deferred window.
    {
        DeferredCopyQueue queue;
        auto source = candidate.create_tensor(spec);
        auto interior = candidate.create_tensor(spec);
        auto destination = candidate.create_tensor(spec);
        if (observer != nullptr) {
            observer->setup_complete();
        }

        iom::TensorView full_source = source->view();
        iom::TensorView sliced_source = interior->view().slice(0, 1, 1);
        iom::TensorView sliced_destination =
                destination->view().slice(0, 0, 1);
        iom::TensorView permuted_source =
                source->view().permute(span_of({1, 0}));
        iom::TensorView permuted_destination =
                destination->view().permute(span_of({1, 0}));

        const iom::oid one =
                queue.copy(*source, full_source, *interior, interior->view());
        const iom::oid two = queue.copy(
                *interior, sliced_source, *destination, sliced_destination);
        const iom::oid three = queue.copy(
                *destination, permuted_destination, *source, permuted_source);
        CHECK_EQ(token_sequence(one), 1);
        CHECK_EQ(token_sequence(two), 2);
        CHECK_EQ(token_sequence(three), 3);

        queue.complete(1);
        queue.complete(
                2, std::make_exception_ptr(
                           std::runtime_error("injected asynchronous failure")));
        queue.complete(3);

        CHECK_NOTHROW(queue.wait(one));
        bool rethrown = false;
        try {
            queue.wait(two);
        } catch (const std::runtime_error& error) {
            rethrown = std::string_view(error.what())
                       == "injected asynchronous failure";
        }
        CHECK(rethrown);
        CHECK_NOTHROW(queue.wait(three));

        // Repeated waits rethrow the stored failure; successful waits are
        // idempotent.
        CHECK_THROWS_AS(queue.wait(two), std::runtime_error);
        CHECK_THROWS_AS(queue.wait(two), std::runtime_error);
        CHECK_NOTHROW(queue.wait(one));
        CHECK_NOTHROW(queue.wait(three));

        // Submission-time view, owner, and native-handle addresses all
        // survived until wait.
        queue.expect_stable(
                1, *source, full_source, *interior, interior->view());
        queue.expect_stable(
                2, *interior, sliced_source, *destination, sliced_destination);
        queue.expect_stable(
                3, *destination, permuted_destination, *source,
                permuted_source);

        if (observer != nullptr) {
            observer->case_complete();
        }
    }

    // In-order completion: completing a later sequence implies every earlier
    // sequence.
    {
        DeferredCopyQueue queue;
        const iom::oid first = queue.probe();
        const iom::oid second = queue.probe();
        queue.complete(2);
        CHECK_NOTHROW(queue.wait(first));
        CHECK_NOTHROW(queue.wait(second));
    }

    // Tokens of other queues and zero are rejected.
    {
        DeferredCopyQueue queue;
        DeferredCopyQueue other;
        const iom::oid own = queue.probe();
        const iom::oid foreign_token = other.probe();
        CHECK_THROWS_AS(queue.wait(0), std::invalid_argument);
        CHECK_THROWS_AS(queue.wait(foreign_token), std::invalid_argument);
        CHECK_THROWS_AS(other.wait(own), std::invalid_argument);
        queue.complete(1);
        CHECK_NOTHROW(queue.wait(own));
    }

    // Queue destruction neither synchronizes on an outstanding sequence nor
    // cancels it, and the queue id returns to the pool.
    {
        std::vector<InstrumentedQueue::Event> drained_journal;
        {
            auto journal =
                    std::make_shared<std::vector<InstrumentedQueue::Event>>();
            std::uint8_t released_id = 0;
            const auto began = std::chrono::steady_clock::now();
            {
                InstrumentedQueue pending(journal);
                const iom::oid outstanding = pending.probe();
                released_id = token_queue(outstanding);
                CHECK_EQ(token_sequence(outstanding), 1);
            }
            const auto ended = std::chrono::steady_clock::now();
            drained_journal = std::move(*journal);

            // An implicit wait would never return: nothing completes the
            // outstanding sequence.
            const auto elapsed =
                    std::chrono::duration_cast<std::chrono::seconds>(
                            ended - began);
            CHECK_LT(elapsed.count(), 1);

            bool saw_outstanding_completion = false;
            bool saw_destroy_begin = false;
            bool saw_destroy_end = false;
            for (const InstrumentedQueue::Event& event : drained_journal) {
                if (event.kind
                            == InstrumentedQueue::Event::Kind::complete
                    && event.sequence == 1) {
                    saw_outstanding_completion = true;
                }
                saw_destroy_begin = saw_destroy_begin
                        || event.kind
                                == InstrumentedQueue::Event::Kind::destroy_begin;
                saw_destroy_end = saw_destroy_end
                        || event.kind
                                == InstrumentedQueue::Event::Kind::destroy_end;
            }
            CHECK_FALSE(saw_outstanding_completion);
            CHECK(saw_destroy_begin);
            CHECK(saw_destroy_end);

            InstrumentedQueue successor(journal);
            CHECK_EQ(token_queue(successor.probe()), released_id);
            successor.finish(1);
            CHECK_NOTHROW(successor.wait(
                    (static_cast<iom::oid>(released_id)
                     << kTokenSequenceBits)
                            | 1));
            // Every successful submission was waited before destruction.
        }
    }
}

// Unsupported compute capabilities: every compute method rejects with
// std::runtime_error before submitting, consuming a sequence, or changing an
// output, including transformed operands.
inline void run_compute_capability_conformance(
        iom::Device& candidate,
        const std::span<const iom::DataType>,  // capability, not type-specific
        ConformanceObserver* observer = nullptr) {
    const iom::TensorSpec spec{iom::TensorShape{{2, 16, 16}}, iom::DataType::F32};
    auto x = candidate.create_tensor(spec);
    auto y = candidate.create_tensor(spec);
    auto w = candidate.create_tensor(spec);
    auto attn = candidate.create_tensor(spec);
    auto scratch = candidate.create_tensor(spec);
    if (observer != nullptr) {
        observer->setup_complete();
    }

    const std::vector<std::byte> y_pattern = encode_logical(spec, 41);
    const std::vector<std::byte> attn_pattern = encode_logical(spec, 42);
    y->view().copy_from_host(y_pattern);
    attn->view().copy_from_host(attn_pattern);

    auto queue = candidate.create_ops();
    CHECK_THROWS_AS(queue->add(x->view(), x->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(queue->mul(x->view(), x->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(queue->silu(x->view(), y->view()), std::runtime_error);
    CHECK_THROWS_AS(queue->linear(x->view(), w->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(
            queue->rmsnorm(x->view(), y->view(), w->view(), 1e-6F, 1),
            std::runtime_error);
    CHECK_THROWS_AS(
            queue->sdpa(x->view(), x->view(), x->view(), 1, 1, 16,
                        attn->view()),
            std::runtime_error);

    // Transformed operands compile against the view signatures and still
    // fail capability validation.
    iom::TensorView stepped_x = x->view().slice(0, 0, 1);
    iom::TensorView stepped_y = y->view().slice(0, 0, 1);
    const iom::TensorView merged_x =
            x->view().reshape_leading(span_of({2}));
    CHECK_THROWS_AS(queue->silu(stepped_x, stepped_y), std::runtime_error);
    CHECK_THROWS_AS(
            queue->add(merged_x, x->view(), y->view()), std::runtime_error);

    // No output changed and no sequence was consumed.
    require_logical_bytes(y->view(), y_pattern, "y after capability failures");
    require_logical_bytes(
            attn->view(), attn_pattern, "attn after capability failures");
    const iom::oid probe = queue->copy(x->view(), scratch->view());
    CHECK_EQ(token_sequence(probe), 1);
    queue->wait(probe);
    queue.reset();

    if (observer != nullptr) {
        observer->case_complete();
    }
}

// Full suite: storage, transfers, copies, errors, lifetime, and capabilities
// in dependency order.
inline void run_backend_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    run_storage_and_transfer_conformance(devices, supported_types, observer);
    run_async_copy_conformance(devices, supported_types, observer);
    run_copy_error_conformance(devices, supported_types, observer);
    run_transfer_error_conformance(devices, supported_types, observer);
    run_lifetime_conformance(devices.candidate, supported_types, observer);
    run_compute_capability_conformance(
            devices.candidate, supported_types, observer);
}

}  // namespace iom_conformance
