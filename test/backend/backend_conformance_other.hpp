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

#include "backend_conformance_copy_storage.hpp"
#include "backend_conformance_memory.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <new>
#include <optional>
#include <span>
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
    DeferredCopyQueue() = default;
    explicit DeferredCopyQueue(const iom::Device& device)
            : iom::DeviceOps(device) {}
    using iom::DeviceOps::copy;
    using iom::DeviceOps::complete;
    using iom::DeviceOps::seek_next_sequence;
    enum class CopyFailure {
        none,
        pre_enqueue,
        pre_enqueue_bad_alloc,
        pre_enqueue_overflow,
        pre_enqueue_nonstandard,
        post_enqueue,
    };

    void inject_copy_failure(CopyFailure failure) noexcept {
        next_copy_failure_ = failure;
    }


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

    // Owned seam for tests to reset the journal between submissions.
    void clear_records() noexcept {
        records_.clear();
    }

    // Submission that also records the owner addresses, mirroring what a
    // real queue observes about its operands.
    iom::oid copy(
            const iom::Tensor& source_owner, const iom::TensorView& source,
            iom::Tensor& destination_owner, iom::TensorView& destination) {
        const iom::oid token = iom::DeviceOps::copy(source, destination);
        if (iom::oid_is_token(token)) {
            records_.back().source_owner = &source_owner;
            records_.back().destination_owner = &destination_owner;
        }
        return token;
    }

protected:
    iom::oid copy_impl(
            const iom::TensorView& source,
            iom::TensorView& destination) override {
        const CopyFailure failure =
                std::exchange(next_copy_failure_, CopyFailure::none);
        switch (failure) {
            case CopyFailure::pre_enqueue:
                throw std::invalid_argument("deferred pre-enqueue failure");
            case CopyFailure::pre_enqueue_bad_alloc:
                throw std::bad_alloc();
            case CopyFailure::pre_enqueue_overflow:
                throw std::overflow_error(
                        "deferred pre-enqueue overflow");
            case CopyFailure::pre_enqueue_nonstandard:
                throw 42;  // unclassifiable -> InternalError
            case CopyFailure::none:
            case CopyFailure::post_enqueue:
                break;
        }
        return submit([&, failure](std::uint64_t sequence) {
            records_.push_back(
                    {sequence, nullptr, &source, source.native_handle(),
                     nullptr, &destination, destination.native_handle()});
            if (failure == CopyFailure::post_enqueue) {
                commit_failure(
                        sequence,
                        std::make_exception_ptr(
                                std::runtime_error(
                                        "deferred post-enqueue failure")));
            }
        });
    }
public:

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

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "deferred";
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
    CopyFailure next_copy_failure_ = CopyFailure::none;
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

    InstrumentedQueue() = default;
    InstrumentedQueue(
            const iom::Device& device,
            std::shared_ptr<std::vector<Event>> journal)
            : iom::DeviceOps(device), journal_(std::move(journal)) {}

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

protected:
    iom::oid copy_impl(const iom::TensorView&, iom::TensorView&) override {
        return probe();
    }
public:
    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "instrumented";
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
        DeferredCopyQueue queue{candidate};
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

    // A post-enqueue failure retains the token while a pre-enqueue failure
    // leaves the sequence counter untouched.
    {
        DeferredCopyQueue queue{candidate};
        auto source = candidate.create_tensor(spec);
        auto destination = candidate.create_tensor(spec);

        queue.inject_copy_failure(DeferredCopyQueue::CopyFailure::pre_enqueue);
        CHECK_EQ(
                queue.copy(source->view(), destination->view()),
                iom::to_oid(iom::OidError::InvalidArgument));
        const iom::oid first =
                queue.copy(*source, source->view(), *destination,
                           destination->view());
        CHECK_EQ(token_sequence(first), 1);
        queue.complete(1);
        CHECK_NOTHROW(queue.wait(first));

        queue.inject_copy_failure(
                DeferredCopyQueue::CopyFailure::post_enqueue);
        const iom::oid failed =
                queue.copy(*source, source->view(), *destination,
                           destination->view());
        CHECK_EQ(token_sequence(failed), 2);
        REQUIRE(queue.records().size() == 2);
        queue.expect_stable(
                2, *source, source->view(), *destination, destination->view());
        queue.complete(2);

        std::string failure_message;
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool rethrown = false;
            try {
                queue.wait(failed);
            } catch (const std::runtime_error& error) {
                rethrown = true;
                if (failure_message.empty()) {
                    failure_message = error.what();
                } else {
                    CHECK_EQ(std::string_view(error.what()), failure_message);
                }
            }
            CHECK(rethrown);
        }
        CHECK_EQ(failure_message, "deferred post-enqueue failure");
    }

    // Every synchronous error category the fake seam can produce maps to
    // its exact signed OID error and consumes no sequence.
    {
        DeferredCopyQueue queue{candidate};
        auto source = candidate.create_tensor(spec);
        auto destination = candidate.create_tensor(spec);
        const struct {
            DeferredCopyQueue::CopyFailure failure;
            iom::OidError expected;
        } categories[] = {
                {DeferredCopyQueue::CopyFailure::pre_enqueue,
                 iom::OidError::InvalidArgument},
                {DeferredCopyQueue::CopyFailure::pre_enqueue_bad_alloc,
                 iom::OidError::ResourceExhausted},
                {DeferredCopyQueue::CopyFailure::pre_enqueue_overflow,
                 iom::OidError::Overflow},
                {DeferredCopyQueue::CopyFailure::pre_enqueue_nonstandard,
                 iom::OidError::InternalError},
        };
        std::uint64_t expected_sequence = 1;
        for (const auto& category : categories) {
            CAPTURE(static_cast<int>(category.expected));
            queue.inject_copy_failure(category.failure);
            CHECK_EQ(
                    queue.copy(source->view(), destination->view()),
                    iom::to_oid(category.expected));
            // The rejected call consumed no sequence and queued nothing.
            CHECK(queue.records().empty());
            const iom::oid token =
                    queue.copy(source->view(), destination->view());
            CHECK_EQ(token_sequence(token), expected_sequence);
            queue.complete(expected_sequence);
            CHECK_NOTHROW(queue.wait(token));
            ++expected_sequence;
            queue.clear_records();
        }
    }

    // In-order completion: completing a later sequence implies every earlier
    // sequence.
    {
        DeferredCopyQueue queue{candidate};
        const iom::oid first = queue.probe();
        const iom::oid second = queue.probe();
        queue.complete(2);
        CHECK_NOTHROW(queue.wait(first));
        CHECK_NOTHROW(queue.wait(second));
    }

    // Invalid waits are rejected immediately, on the owning queue and on
    // any other queue, without side effects. Skipping a sequence never
    // makes the skipped value waitable, even after later sequences
    // completed in order.
    {
        DeferredCopyQueue queue{candidate};
        DeferredCopyQueue other{candidate};
        const iom::oid own = queue.probe();
        const iom::oid other_token = other.probe();

        for (DeferredCopyQueue* target : {&queue, &other}) {
            const iom::oid target_token =
                    target == &queue ? own : other_token;
            const iom::oid foreign_token =
                    target == &queue ? other_token : own;
            const iom::oid future_token =
                    (static_cast<iom::oid>(token_queue(target_token))
                     << kTokenSequenceBits)
                    | (token_sequence(target_token) + 1);
            CHECK_THROWS_AS(target->wait(-1), std::invalid_argument);
            CHECK_THROWS_AS(target->wait(0), std::invalid_argument);
            CHECK_THROWS_AS(target->wait(foreign_token),
                            std::invalid_argument);
            CHECK_THROWS_AS(target->wait(future_token),
                            std::invalid_argument);
            // An otherwise unsubmitted sequence with the target queue id.
            CHECK_THROWS_AS(
                    target->wait(
                            (static_cast<iom::oid>(token_queue(target_token))
                             << kTokenSequenceBits)
                                    | 4),
                    std::invalid_argument);
            CHECK_EQ(target->records().size(), std::size_t{1});
            // Complete the target's accepted token before the next loop
            // iteration, so no invalid wait can block on an accepted value.
            target->complete(1);
            CHECK_NOTHROW(target->wait(target_token));
        }

        // Rejected submissions consumed no sequence; the owning queue's
        // next accepted submission is sequence two.
        CHECK_EQ(token_sequence(queue.probe()), 2);
        queue.complete(2);
        CHECK_NOTHROW(queue.wait(own));


        // Reserve a gap, submit and complete a later sequence, then prove
        // the skipped and unsubmitted values stay immediately invalid.
        queue.seek_next_sequence(4);
        const iom::oid later = queue.probe();
        CHECK_EQ(token_sequence(later), 4);
        CHECK_EQ(queue.records().size(), std::size_t{3});
        queue.complete(4);
        CHECK_NOTHROW(queue.wait(later));
        CHECK_THROWS_AS(
                queue.wait(
                        (static_cast<iom::oid>(token_queue(later))
                         << kTokenSequenceBits)
                                | 3),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue.wait(
                        (static_cast<iom::oid>(token_queue(later))
                         << kTokenSequenceBits)
                                | 5),
                std::invalid_argument);
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
                InstrumentedQueue pending(candidate, journal);
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

            InstrumentedQueue successor(candidate, journal);
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

// Compute capability probes may pass an explicit linear expectation for a
// focused backend case. The full backend suite leaves it unspecified so the
// same scenario accepts either an unsupported operation or a newly landed
// backend port and derives its token accounting from the observed result. The
// frozen RMS normalization scenarios live in
// `backend_conformance_rmsnorm.hpp` and are not duplicated here.
inline void run_compute_capability_conformance(
        iom::Device& candidate,
        const std::span<const iom::DataType>,
        ConformanceObserver* observer = nullptr,
        std::string_view backend_label = {},
        bool binary_supported = false,
        std::optional<bool> linear_supported = std::nullopt) {
    const iom::TensorSpec spec{iom::TensorShape{{2, 16, 16}}, iom::DataType::F32};
    auto x = candidate.create_tensor(spec);
    auto y = candidate.create_tensor(spec);
    auto w = candidate.create_tensor(spec);
    auto attn = candidate.create_tensor(spec);
    auto scratch = candidate.create_tensor(spec);
    const iom::TensorSpec sdpa_q_spec{
            iom::TensorShape{{2, 1, 16, 16}}, iom::DataType::BF16};
    auto sdpa_q = candidate.create_tensor(sdpa_q_spec);
    auto sdpa_kv = candidate.create_tensor(sdpa_q_spec);
    const iom::TensorSpec sdpa_out_spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::BF16};
    auto sdpa_out = candidate.create_tensor(sdpa_out_spec);
    if (observer != nullptr) {
        observer->setup_complete();
    }

    const std::vector<std::byte> y_pattern = encode_logical(spec, 41);
    const std::vector<std::byte> attn_pattern = encode_logical(spec, 42);
    iom_conformance::copy_from_host(y->view(), y_pattern);
    iom_conformance::copy_from_host(attn->view(), attn_pattern);
    auto queue = candidate.create_ops();
    (void)backend_label;
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const auto submit_supported_binary =
            [&](const BinaryOperation operation) {
                const auto requirements =
                        query_binary_workspace_requirements(
                                *queue, operation, x->view(), x->view(),
                                y->view());
                std::unique_ptr<iom::RawWorkspace> workspace_owner;
                if (requirements.bytes != 0) {
                    workspace_owner =
                            candidate.create_workspace(requirements.bytes);
                }
                if (workspace_owner) {
                    const iom::RawWorkspaceView workspace =
                            workspace_owner->view();
                    return submit_binary_operation(
                            *queue, operation, x->view(), x->view(),
                            y->view(), workspace);
                }
                return submit_binary_operation(
                        *queue, operation, x->view(), x->view(), y->view());
            };
    const iom::oid binary_token =
            submit_supported_binary(BinaryOperation::add);
    if (binary_supported) {
        REQUIRE(iom::oid_is_token(binary_token));
        CHECK_NOTHROW(queue->wait(binary_token));
        for (const auto operation : {BinaryOperation::mul,
                                     BinaryOperation::sub,
                                     BinaryOperation::div}) {
            const iom::oid token = submit_supported_binary(operation);
            REQUIRE(iom::oid_is_token(token));
            CHECK_NOTHROW(queue->wait(token));
        }
    } else {
        CHECK_EQ(binary_token, unsupported);
        for (const auto operation : {BinaryOperation::mul,
                                     BinaryOperation::sub,
                                     BinaryOperation::div}) {
            CHECK_EQ(submit_binary_operation(
                             *queue, operation, x->view(), x->view(), y->view()),
                     unsupported);
        }
    }
    CHECK_EQ(queue->silu(x->view(), y->view()), unsupported);
    // The frozen linear ABI selects row `s` for `R` rows of the rank-two
    // HF-oriented `[O,I]` weight: the rank-three probe fixture supplies that
    // weight as its first `[16,16]` plane, so the request stays valid-shaped
    // and exercises the operation's own capability rather than structural
    // validation without allocating another tensor. Focused callers can
    // provide an explicit expectation; the full backend suite observes either
    // the supported token or the established Unsupported result.
    const iom::oid linear_token =
            queue->linear(
                    x->view(), w->view().select(0, 0), y->view(), 0, 16,
                    iom::LinearOutputLayout::ordinary, 1, 16);
    if (!linear_supported.has_value()) {
        if (iom::oid_is_token(linear_token)) {
            CHECK_NOTHROW(queue->wait(linear_token));
        } else {
            CHECK_EQ(linear_token, unsupported);
        }
    } else if (*linear_supported) {
        REQUIRE(iom::oid_is_token(linear_token));
        CHECK_NOTHROW(queue->wait(linear_token));
    } else {
        CHECK_EQ(linear_token, unsupported);
    }
    const bool linear_submitted = iom::oid_is_token(linear_token);
    CHECK_EQ(
            queue->sdpa(
                    sdpa_q->view(), sdpa_kv->view(), sdpa_kv->view(),
                    sdpa_out->view(), 0, 16),
            unsupported);
    (void)backend_label;


    if (!binary_supported) {
        require_logical_bytes(y->view(), y_pattern,
                               "y after capability failures");
        require_logical_bytes(attn->view(), attn_pattern,
                               "attn after capability failures");
    }
    const iom::oid probe = queue->copy(x->view(), scratch->view());
    CHECK_EQ(
            token_sequence(probe),
            (binary_supported ? 4 : 0) + (linear_submitted ? 1 : 0) + 1);
    queue->wait(probe);
    queue.reset();

    if (observer != nullptr) {
        observer->case_complete();
    }
}

// Full suite: storage, transfers, copies, errors, lifetime, and capabilities
// in dependency order. The frozen RMS normalization scenarios are composed by
// `run_rmsnorm_conformance` from the same driver files instead of a capability
// observation.
inline void run_backend_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr,
        AcceleratorStorageOracle* oracle = nullptr,
        bool binary_supported = false,
        std::optional<bool> linear_supported = std::nullopt) {
    run_storage_and_transfer_conformance(devices, supported_types, observer);
    run_async_copy_conformance(devices, supported_types, observer, oracle);
    run_copy_error_conformance(devices, supported_types, observer);
    run_transfer_error_conformance(devices, supported_types, observer);
    run_lifetime_conformance(devices.candidate, supported_types, observer);
    run_binary_request_conformance(devices, observer);
    run_binary_rank_boundary_conformance(devices.candidate);
    run_memory_contract_conformance(devices);
    run_compute_capability_conformance(
            devices.candidate, supported_types, observer, {}, binary_supported,
            linear_supported);
}
}  // namespace iom_conformance
