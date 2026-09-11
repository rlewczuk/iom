#include <array>
#include "copy.hpp"

#include "iom/device.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "scalar_add.hpp"
#include "runtime.hpp"

#define IOM_GPU_DEVICE
#define IOM_GPU_GLOBAL
#define IOM_GPU_GLOBAL_INDEX 0
#define IOM_GPU_GLOBAL_STRIDE 1
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    ((void)((kernel), (blocks), (threads), (stream), __VA_ARGS__))
#include "../shared/standard_tiled_copy.inl"
#undef IOM_LAUNCH_KERNEL
#undef IOM_GPU_GLOBAL_STRIDE
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

namespace iom::sycl_detail {
namespace {

std::atomic<SubmissionFault> g_submission_fault{
        SubmissionFault::none};
std::atomic<std::size_t> g_fence_wait_count{0};

[[nodiscard]] bool consume_submission_fault(
        SubmissionFault point) noexcept {
    SubmissionFault expected = point;
    return g_submission_fault.compare_exchange_strong(
            expected, SubmissionFault::none, std::memory_order_acq_rel);
}

// The fixed completion generation pool mirrors the task-06 queue geometry.
// SYCL events are SDK objects, so each slot receives the event generation
// returned by one dispatch and is reused only after the generation's result
// has been proved. The slot array itself never grows after queue setup.
class SyclCompletionPool final {
public:
    explicit SyclCompletionPool(std::size_t count)
            : slots_(std::make_unique<Slot[]>(count)), count_(count) {
        if (count_ == 0) {
            throw std::invalid_argument(
                    "SYCL completion pool requires at least one slot");
        }
    }

    SyclCompletionPool(const SyclCompletionPool&) = delete;
    SyclCompletionPool& operator=(const SyclCompletionPool&) = delete;

    [[nodiscard]] std::size_t count() const noexcept { return count_; }

    [[nodiscard]] std::size_t acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        completion_.wait(lock, [this] {
            for (std::size_t index = 0; index < count_; ++index) {
                if (!slots_[index].in_use) {
                    return true;
                }
            }
            return false;
        });
        for (std::size_t index = 0; index < count_; ++index) {
            if (!slots_[index].in_use) {
                slots_[index].in_use = true;
                return index;
            }
        }
        throw std::logic_error(
                "SYCL completion pool lost a free slot");
    }

    void set_event(std::size_t index, sycl::event event) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < count_ && slots_[index].in_use) {
            slots_[index].event = std::move(event);
            slots_[index].has_event = true;
        }
    }

    void wait(std::size_t index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= count_ || !slots_[index].in_use
                || !slots_[index].has_event) {
            return;
        }
        slots_[index].event.wait_and_throw();
    }

    void protect(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < count_) {
            slots_[index].protected_ = true;
            slots_[index].in_use = true;
        }
    }

    void release_after_proof(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < count_ && slots_[index].in_use) {
            slots_[index].event = sycl::event{};
            slots_[index].has_event = false;
            slots_[index].protected_ = false;
            slots_[index].in_use = false;
            completion_.notify_one();
        }
    }

    void release_all_protected() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < count_; ++index) {
            if (slots_[index].protected_) {
                slots_[index].event = sycl::event{};
                slots_[index].has_event = false;
                slots_[index].protected_ = false;
                slots_[index].in_use = false;
                completion_.notify_one();
            }
        }
    }

    [[nodiscard]] bool has_unproven_leases() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < count_; ++index) {
            if (slots_[index].in_use) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::size_t in_use_count() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t result = 0;
        for (std::size_t index = 0; index < count_; ++index) {
            result += slots_[index].in_use ? 1u : 0u;
        }
        return result;
    }

    [[nodiscard]] std::size_t protected_count() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t result = 0;
        for (std::size_t index = 0; index < count_; ++index) {
            result += slots_[index].protected_ ? 1u : 0u;
        }
        return result;
    }

private:
    struct Slot {
        sycl::event event{};
        bool in_use = false;
        bool protected_ = false;
        bool has_event = false;
    };

    std::unique_ptr<Slot[]> slots_;
    std::size_t count_ = 0;
    std::mutex mutex_;
    std::condition_variable completion_;
};

class SyclFenceState final {
public:
    static_assert(
            std::is_nothrow_move_constructible_v<sycl::event>
            && std::is_nothrow_move_assignable_v<sycl::event>);

    ~SyclFenceState() noexcept {
        release_metadata_slot();
        release_completion_slot();
    }

    void set_event(sycl::event incoming) noexcept {
        if (completion_pool_ != nullptr) {
            completion_pool_->set_event(completion_slot_, std::move(incoming));
        } else {
            event_ = std::move(incoming);
        }
    }

    void set_failure(std::exception_ptr incoming) noexcept {
        retained_failure_ = std::move(incoming);
    }

    void set_cleanup(std::function<void()> cleanup) {
        std::lock_guard<std::mutex> lock(mu_);
        cleanup_ = std::move(cleanup);
    }

    void set_completion_action(std::function<void()> action) {
        std::lock_guard<std::mutex> lock(mu_);
        completion_action_ = std::move(action);
    }

    void cleanup_now() noexcept {
        std::function<void()> cleanup;
        {
            std::lock_guard<std::mutex> lock(mu_);
            cleanup = std::move(cleanup_);
        }
        if (cleanup) {
            try {
                cleanup();
            } catch (...) {
            }
        }
    }

    void mark_completion_proven() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        proved_ = true;
    }

    [[nodiscard]] bool completion_proven() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return proved_;
    }

    [[nodiscard]] detail::FenceResult result() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        if (cached_.has_value()) {
            return *cached_;
        }

        detail::FenceResult result = detail::FenceResult::success();
        if (completion_pool_ != nullptr) {
            ++g_fence_wait_count;
            try {
                completion_pool_->wait(completion_slot_);
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
            }
        } else if (event_.has_value()) {
            ++g_fence_wait_count;
            try {
                event_->wait_and_throw();
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
            }
        }
        if (!result.failure && retained_failure_) {
            result = detail::FenceResult::failed(retained_failure_);
        }
        if (!result.failure && completion_action_) {
            try {
                completion_action_();
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
            }
        }
        completion_action_ = {};
        if (cleanup_) {
            try {
                cleanup_();
            } catch (...) {
                if (!result.failure) {
                    result = detail::FenceResult::failed(
                            std::current_exception());
                }
            }
            cleanup_ = {};
        }
        // The fixed slot's host mirror stays immutable until this result is
        // known. An operation failure can be retained independently from the
        // completion proof established by the explicit queue drain.
        proved_ = proved_ || (result.succeeded && result.failure == nullptr);
        cached_ = result;
        return result;
    }

    void clear_event() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        event_ = std::nullopt;
    }

    void release_metadata_slot() noexcept {
        detail::MetadataSlotPool* pool = nullptr;
        std::size_t slot = 0;
        bool proved = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            pool = pool_;
            slot = slot_;
            proved = proved_;
            pool_ = nullptr;
            slot_ = 0;
        }
        if (pool != nullptr) {
            if (proved) {
                pool->release_after_proof(slot);
            } else {
                // Unknown completion: the whole lease stays reserved; this
                // slot is never reassigned until a covering proof.
                pool->protect(slot);
            }
        }
    }

    void release_completion_slot() noexcept {
        SyclCompletionPool* pool = nullptr;
        std::size_t slot = 0;
        bool proved = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            pool = completion_pool_;
            slot = completion_slot_;
            proved = proved_;
            completion_pool_ = nullptr;
            completion_slot_ = 0;
        }
        if (pool != nullptr) {
            if (proved) {
                pool->release_after_proof(slot);
            } else {
                pool->protect(slot);
            }
        }
    }

private:
    void set_metadata_slot(
            detail::MetadataSlotPool& pool, std::size_t slot) noexcept {
        pool_ = &pool;
        slot_ = slot;
    }

    void set_completion_slot(
            SyclCompletionPool& pool, std::size_t slot) noexcept {
        completion_pool_ = &pool;
        completion_slot_ = slot;
    }

    mutable std::mutex mu_;
    std::optional<sycl::event> event_;
    std::exception_ptr retained_failure_;
    std::optional<detail::FenceResult> cached_;
    std::function<void()> completion_action_;
    std::function<void()> cleanup_;
    bool proved_ = false;
    std::size_t slot_ = 0;
    detail::MetadataSlotPool* pool_ = nullptr;
    std::size_t completion_slot_ = 0;
    SyclCompletionPool* completion_pool_ = nullptr;

    friend class SyclQueue;
};

struct SyclFenceCapture {
    std::shared_ptr<SyclFenceState> state;
};

detail::FenceResult sycl_fence_invoke(
        const detail::Fence& fence) noexcept {
    const auto& capture =
            *std::launder(reinterpret_cast<const SyclFenceCapture*>(
                    fence.storage));
    if (!capture.state) {
        return detail::FenceResult::success();
    }
    return capture.state->result();
}

static_assert(noexcept(sycl_fence_invoke(
        std::declval<const detail::Fence&>())));

detail::Fence build_sycl_fence(
        const std::shared_ptr<SyclFenceState>& state) noexcept {
    detail::Fence fence;
    ::new (fence.storage) SyclFenceCapture{state};
    fence.invoke = &sycl_fence_invoke;
    fence.copy_construct =
            &detail::FenceCaptureOps<SyclFenceCapture>::copy_construct;
    fence.move_construct =
            &detail::FenceCaptureOps<SyclFenceCapture>::move_construct;
    fence.destroy = &detail::FenceCaptureOps<SyclFenceCapture>::destroy;
    return fence;
}

void launch_scatter_words(
        sycl::queue& queue, const void* source, void* destination,
        std::uint64_t destination_plane, std::uint64_t logical_base,
        std::size_t word_count, std::size_t rows, std::size_t columns,
        unsigned int bits) {
    (void)queue.parallel_for(
            sycl::range<1>(word_count),
            [=](sycl::id<1> item) {
                detail::copy_logical_to_tiled_word(
                        static_cast<const unsigned char*>(source),
                        static_cast<unsigned char*>(destination),
                        destination_plane, logical_base, item[0], rows, columns,
                        bits);
            });
    if (launch_calls.kernel_launched != nullptr) {
        launch_calls.kernel_launched();
    }
}

void launch_gather_words(
        sycl::queue& queue, const void* source, void* destination,
        std::uint64_t source_plane, std::uint64_t logical_base,
        std::size_t first_word, std::size_t word_count, std::size_t rows,
        std::size_t columns, unsigned int bits) {
    (void)queue.parallel_for(
            sycl::range<1>(word_count),
            [=](sycl::id<1> item) {
                detail::copy_tiled_to_logical_word(
                        static_cast<const unsigned char*>(source),
                        static_cast<unsigned char*>(destination), source_plane,
                        logical_base, first_word + item[0], rows, columns,
                        bits);
            });
    if (launch_calls.kernel_launched != nullptr) {
        launch_calls.kernel_launched();
    }
}

void launch_view_transfer(
        sycl::queue& queue, const TensorView& view,
        const void* source, void* destination, bool from_host) {
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::span<const std::size_t> plane_strides =
            view.plane_strides();
    const std::size_t elements = rows * columns;
    const std::size_t plane_count =
            view.spec().shape.element_count() / elements;
    const unsigned int bits = static_cast<unsigned int>(
            detail::leaf_bits(view.spec().data_type));
    const std::size_t padded_rows =
            (rows + TensorSpec::TILE - 1) / TensorSpec::TILE
            * TensorSpec::TILE;
    const std::size_t padded_columns =
            (columns + TensorSpec::TILE - 1) / TensorSpec::TILE
            * TensorSpec::TILE;
    const std::size_t words_per_plane =
            padded_rows * padded_columns * bits / 32;

    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        std::size_t rest = logical_plane;
        std::size_t plane = view.plane_offset();
        for (std::size_t axis = leading_rank; axis-- > 0;) {
            plane += (rest % dimensions[axis]) * plane_strides[axis];
            rest /= dimensions[axis];
        }
        const std::uint64_t logical_base =
                static_cast<std::uint64_t>(logical_plane * elements);
        std::size_t first_word = 0;
        std::size_t word_count = words_per_plane;
        if (!from_host) {
            first_word = logical_base * bits / 32;
            const std::size_t logical_end =
                    (logical_base + elements) * bits;
            const std::size_t last_word =
                    logical_end / 32 + (logical_end % 32 != 0);
            word_count = last_word - first_word;
        }
        if (from_host) {
            launch_scatter_words(
                    queue, source, destination,
                    static_cast<std::uint64_t>(plane), logical_base,
                    word_count, rows, columns, bits);
        } else {
            launch_gather_words(
                    queue, source, destination,
                    static_cast<std::uint64_t>(plane), logical_base,
                    first_word, word_count, rows, columns, bits);
        }
    }
}

detail::Fence transfer_fence() noexcept;

class SyclQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence = 0;
        bool no_op = false;
        std::optional<CopyRequest> copy_request;
        std::shared_ptr<SyclFenceState> state;
        void* fence = nullptr;
        detail::EntryRegistration copy_entries;
        std::optional<BinaryRequest> binary_request;
        detail::BinaryEntryRegistration binary_entries;
    };
    struct SyclSequenceOutcome {
        detail::SequenceOutcome common;
        std::shared_ptr<SyclFenceState> state;
        std::optional<detail::BinaryEntryRegistration> binary_entries;
        detail::WorkspaceLease workspace_lease;
    };


public:
#ifdef IOM_ENABLE_TESTING
    [[nodiscard]] detail::MetadataSlotPool&
            metadata_pool_for_testing() noexcept {
        return *metadata_pool_;
    }
    [[nodiscard]] SyclCompletionPool&
            completion_pool_for_testing() noexcept {
        return *completion_pool_;
    }
#endif

    // Reservation order is fixed by member declaration order: the queue
    // resource lease (credit plus disjoint C-slot partition) is reserved
    // first, then the native in-order queue is created, and only then does
    // the worker start. A fifth live queue therefore throws std::bad_alloc
    // before any stream or worker exists.
    SyclQueue(
            const Device& device,
            detail::QueueResourceProvider& resource_provider,
            const sycl::context& context,
            const sycl::device& native_device,
            detail::RegistryState& state)
            : DeviceOps(device),
              device_(&device),
              state_(&state),
              resource_provider_(&resource_provider),
              registry_queue_id_(detail::allocate_queue_id(state)),
              metadata_pool_(std::make_shared<detail::MetadataSlotPool>(
                      resource_provider.reserve_queue_resources())),
              completion_pool_(std::make_shared<SyclCompletionPool>(
                      resource_provider.queue_slot_count())),
              queue_(make_queue_with_fault_check(context, native_device)),
              worker_(
                      detail::StagedWorker<Task>::Callbacks{
                              [this](Task& task) {
                                  execute(task);
                              },
                              [](void* fence) noexcept {
                                  if (fence != nullptr) {
                                      (void)static_cast<SyclFenceState*>(
                                                     fence)
                                              ->result();
                                  }
                              },
                              [](void* fence) noexcept {
                                  if (fence == nullptr) {
                                      return;
                                  }
                                  auto* state =
                                          static_cast<SyclFenceState*>(fence);
                                  (void)state->result();
                                  state->clear_event();
                                  state->release_metadata_slot();
                                  state->release_completion_slot();
                              },
                              [this](
                                      std::uint64_t sequence,
                                      std::exception_ptr failure) {
                                  complete_task(
                                          sequence, std::move(failure));
                              }},
                      detail::StagedWorker<Task>::PublishPolicy::Splice) {
        worker_.start();
    }
    ~SyclQueue() override {
        // Covering drain attempt: a successful wait proves every enqueued
        // access and releases all protected slots.
        bool drained = false;
        try {
            queue_.wait_and_throw();
            drained = true;
        } catch (...) {
        }
        worker_.shutdown_and_drain();
        close_and_drain();
        state_->registry.invalidate_entries_for_queue(registry_queue_id_);
        if (drained) {
            metadata_pool_->release_all_protected();
            completion_pool_->release_all_protected();
            return;
        }
        if (!metadata_pool_->has_unproven_leases()
                && !completion_pool_->has_unproven_leases()) {
            // Nothing native is unproven; release the lease normally.
            return;
        }
        // Unknown completion: quarantine the entire unresolved lease at the
        // Device boundary. The partition and queue-count reservation stay
        // retained with the queue's own covering-proof handle until this
        // queue's own drain is proved; a drain of another queue is never
        // sufficient.
        sycl::queue retained_queue = std::move(queue_);
        std::shared_ptr<detail::MetadataSlotPool> retained_pool =
                std::move(metadata_pool_);
        std::shared_ptr<SyclCompletionPool> retained_completion =
                std::move(completion_pool_);
        resource_provider_->retain_unknown_lease(
                [retained_queue, retained_pool, retained_completion]() mutable
                        -> bool {
                    try {
                        retained_queue.wait_and_throw();
                    } catch (...) {
                        return false;
                    }
                    // Covering proof: release protected slots and drop the
                    // lease (which returns the partition and credit).
                    retained_pool->release_all_protected();
                    retained_completion->release_all_protected();
                    retained_completion.reset();
                    retained_pool.reset();
                    return true;
                },
                [retained_queue, retained_pool, retained_completion]() mutable {
                    // Best-effort teardown; unproven leases keep their
                    // original token outcomes untouched.
                    try {
                        retained_queue.wait_and_throw();
                    } catch (...) {
                    }
                    retained_completion.reset();
                    retained_pool.reset();
                });

    }
    oid copy_impl(
            const TensorView& source,
            TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        const bool no_op = identical_window(source, destination);
        std::shared_ptr<SyclFenceState> state;
        if (!no_op) {
            if (consume_submission_fault(SubmissionFault::state_allocation)) {
                throw std::bad_alloc();
            }
            state = std::make_shared<SyclFenceState>();
        }
        if (consume_submission_fault(SubmissionFault::fence_construction)) {
            throw std::bad_alloc();
        }
        detail::Fence fence = no_op ? transfer_fence() : build_sycl_fence(state);
        return submit_copy(
                source, destination, *state_, registry_queue_id_, fence,
                [this, state](
                        std::uint64_t sequence, const CopyRequest& captured,
                        detail::EntryRegistration entries) {
                    Task task;
                    task.sequence = sequence;
                    task.no_op = captured.no_op;
                    task.copy_request.emplace(captured);
                    task.state = state;
                    task.fence = state.get();
                    task.copy_entries = entries;
                    worker_.submit_copy(std::move(task));
                });
    }

    oid binary_impl(const BinaryRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        if (consume_submission_fault(SubmissionFault::state_allocation)) {
            throw std::bad_alloc();
        }
        auto state = std::make_shared<SyclFenceState>();
        if (consume_submission_fault(SubmissionFault::fence_construction)) {
            throw std::bad_alloc();
        }
        detail::Fence fence = build_sycl_fence(state);
        return submit_binary(
                request, *state_, registry_queue_id_, fence,
                [this, state](
                        std::uint64_t sequence, const BinaryRequest& captured,
                        detail::BinaryEntryRegistration entries) {
                    Task task;
                    task.sequence = sequence;
                    task.state = state;
                    task.fence = state.get();
                    task.binary_request.emplace(captured);
                    task.binary_entries = entries;
                    worker_.submit_copy(std::move(task));
                });
    }

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "SYCL";
    }

    // Pure deterministic raw-workspace requirement for the binary
    // operations: three whole-plane staging extents, each aligned up to
    // 32, summed with checked arithmetic at alignment 32. The three
    // slices start at offsets 0, align_up(lhs_bytes, 32), and
    // align_up(lhs_bytes, 32) + align_up(rhs_bytes, 32). Pure: no
    // allocation, registration, lease, token/queue resource, metadata
    // upload, or submission, and no dependence on free data-arena
    // capacity, queue occupancy, or completion state.
    [[nodiscard]] WorkspaceRequirements binary_workspace_requirements(
            const BinaryRequest& request) override {
        const std::size_t lhs_bytes =
                checked_binary_view_staging_bytes(request, request.lhs);
        const std::size_t rhs_bytes =
                checked_binary_view_staging_bytes(request, request.rhs);
        const std::size_t out_bytes =
                checked_binary_view_staging_bytes(request, request.out);
        const std::size_t lhs_aligned =
                align_up_checked(lhs_bytes, 32);
        const std::size_t rhs_aligned =
                align_up_checked(rhs_bytes, 32);
        const std::size_t total = checked_add_local(
                checked_add_local(
                        lhs_aligned, rhs_aligned,
                        "SYCL workspace staging sum overflows"),
                out_bytes,
                "SYCL workspace staging sum overflows");
        return {total, 32};
    }

private:
    // Device-USM-safe binary staging extent: whole plane blocks up to the
    // view's highest addressed plane, so untouched planes and tile padding
    // round-trip unchanged. validate_binary bounded this walk with checked
    // arithmetic and equal plane geometry, so the recompute cannot
    // overflow for an accepted request.
    static std::size_t binary_view_staging_bytes(
            const BinaryRequest& captured, const BinaryViewSnapshot& view) {
        const std::span<const std::size_t> dims =
                view.spec.shape.dimensions();
        const std::size_t leading_rank = dims.size() - 2;
        std::size_t max_plane = view.plane_offset;
        for (std::size_t axis = 0; axis < leading_rank; ++axis) {
            max_plane += (dims[axis] - 1) * view.plane_strides[axis];
        }
        const TensorShape padded_shape =
                view.spec.standard_padded_shape();
        const auto padded_dimensions = padded_shape.dimensions();
        const std::size_t padded_plane_elements =
                padded_dimensions[leading_rank]
                * padded_dimensions[leading_rank + 1];
        const std::size_t plane_bits =
                padded_plane_elements
                * detail::leaf_bits(captured.out.spec.data_type);
        const std::size_t plane_bytes =
                plane_bits / 8 + (plane_bits % 8 != 0);
        return (max_plane + 1) * plane_bytes;
    }

    // Checked arithmetic for the pure requirement queries: identical
    // whole-plane semantics to binary_view_staging_bytes with every step
    // overflow-checked, so overflow surfaces as std::overflow_error
    // instead of relying on the accepted-request bound.
    static std::size_t checked_add_local(
            std::size_t lhs, std::size_t rhs, const char* what) {
        if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
            throw std::overflow_error(what);
        }
        return lhs + rhs;
    }

    static std::size_t checked_mul_local(
            std::size_t lhs, std::size_t rhs, const char* what) {
        if (lhs != 0
                && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
            throw std::overflow_error(what);
        }
        return lhs * rhs;
    }

    static std::size_t align_up_checked(
            std::size_t value, std::size_t alignment) {
        const std::size_t remainder = value % alignment;
        if (remainder == 0) {
            return value;
        }
        return checked_add_local(
                value, alignment - remainder,
                "SYCL workspace staging alignment overflows");
    }

    static std::size_t checked_binary_view_staging_bytes(
            const BinaryRequest& captured, const BinaryViewSnapshot& view) {
        const std::span<const std::size_t> dims =
                view.spec.shape.dimensions();
        const std::size_t leading_rank = dims.size() - 2;
        std::size_t max_plane = view.plane_offset;
        for (std::size_t axis = 0; axis < leading_rank; ++axis) {
            max_plane = checked_add_local(
                    max_plane,
                    checked_mul_local(
                            dims[axis] - 1, view.plane_strides[axis],
                            "SYCL workspace staging plane walk overflows"),
                    "SYCL workspace staging plane walk overflows");
        }
        const TensorShape padded_shape =
                view.spec.standard_padded_shape();
        const auto padded_dimensions = padded_shape.dimensions();
        const std::size_t padded_plane_elements =
                checked_mul_local(
                        padded_dimensions[leading_rank],
                        padded_dimensions[leading_rank + 1],
                        "SYCL workspace staging plane size overflows");
        const std::size_t plane_bits =
                checked_mul_local(
                        padded_plane_elements,
                        detail::leaf_bits(captured.out.spec.data_type),
                        "SYCL workspace staging plane bits overflows");
        const std::size_t plane_bytes =
                plane_bits / 8 + (plane_bits % 8 != 0);
        return checked_mul_local(
                checked_add_local(
                        max_plane, 1,
                        "SYCL workspace staging plane count overflows"),
                plane_bytes,
                "SYCL workspace staging extent overflows");
    }

    // Exact shared scalar loop over host staging buffers.
    template <detail::scalar_add_detail::BinaryOp Op>
    static void binary_elements_impl(
            const BinaryRequest& captured, const unsigned char* lhs_storage,
            const unsigned char* rhs_storage, unsigned char* out_storage) {
        const auto dims = captured.result_shape.dimensions();
        const std::size_t rank = dims.size();
        const std::size_t bits = detail::leaf_bits(
                captured.out.spec.data_type);
        auto load = [bits](const void* ptr, std::size_t bit) {
            std::uint64_t value = 0;
            const auto* bytes =
                    static_cast<const unsigned char*>(ptr);
            for (std::size_t i = 0; i < bits; ++i)
                value |= static_cast<std::uint64_t>(
                                 (bytes[(bit + i) / 8]
                                  >> ((bit + i) % 8)) & 1u)
                        << i;
            return value;
        };
        auto store = [bits](void* ptr, std::size_t bit,
                            std::uint64_t value) {
            auto* bytes = static_cast<unsigned char*>(ptr);
            for (std::size_t i = 0; i < bits; ++i) {
                const unsigned char mask =
                        static_cast<unsigned char>(
                                1u << ((bit + i) % 8));
                if ((value >> i) & 1u)
                    bytes[(bit + i) / 8] |= mask;
                else
                    bytes[(bit + i) / 8] &= ~mask;
            }
        };
        std::vector<std::size_t> coord(rank);
        const std::size_t count =
                captured.result_shape.element_count();
        for (std::size_t linear = 0; linear < count; ++linear) {
            std::size_t rest = linear;
            for (std::size_t axis = rank; axis-- > 0;) {
                coord[axis] = rest % dims[axis];
                rest /= dims[axis];
            }
            auto plane = [&](const BinaryViewSnapshot& view) {
                std::size_t result = view.plane_offset;
                for (std::size_t axis = 0; axis < rank - 2;
                     ++axis) {
                    if (view.logical_plane_strides[axis] != 0)
                        result += coord[axis] *
                                  view.logical_plane_strides[axis];
                }
                return result;
            };
            const std::size_t row = coord[rank - 2];
            const std::size_t col = coord[rank - 1];
            const auto slot = [&](const BinaryViewSnapshot& view,
                                  std::size_t p,
                                  std::size_t r,
                                  std::size_t c) {
                return detail::standard_plane_slot(
                        view.spec, p,
                        view.broadcast_rows ? 0 : r,
                        view.broadcast_columns ? 0 : c);
            };
            const auto a = load(
                    lhs_storage,
                    slot(captured.lhs, plane(captured.lhs), row, col) * bits);
            const auto b = load(
                    rhs_storage,
                    slot(captured.rhs, plane(captured.rhs), row, col) * bits);
            const auto destination_slot = slot(
                    captured.out, plane(captured.out), row, col);
            store(out_storage,
                  destination_slot * bits,
                  detail::scalar_binary<Op>(
                          captured.out.spec.data_type, a, b));
        }
    }

    static void binary_elements(
            const BinaryRequest& captured, const unsigned char* lhs_storage,
            const unsigned char* rhs_storage, unsigned char* out_storage) {
        switch (captured.operation) {
            case BinaryOperation::Add:
                return binary_elements_impl<
                        detail::scalar_add_detail::BinaryOp::add>(
                        captured, lhs_storage, rhs_storage, out_storage);
            case BinaryOperation::Mul:
                return binary_elements_impl<
                        detail::scalar_add_detail::BinaryOp::mul>(
                        captured, lhs_storage, rhs_storage, out_storage);
            case BinaryOperation::Sub:
                return binary_elements_impl<
                        detail::scalar_add_detail::BinaryOp::sub>(
                        captured, lhs_storage, rhs_storage, out_storage);
            case BinaryOperation::Div:
                return binary_elements_impl<
                        detail::scalar_add_detail::BinaryOp::div>(
                        captured, lhs_storage, rhs_storage, out_storage);
        }
    }

    static void free_binary_host_staging(
            void* staging, const sycl::context& context) noexcept {
        if (staging == nullptr) {
            return;
        }
        try {
            sycl::free(staging, context);
        } catch (...) {
        }
    }


    [[nodiscard]] static detail::CopyMetadataLayout
            copy_metadata_layout_snapshot(const CopyRequest& request) {
        const auto dimensions = request.source.spec.shape.dimensions();
        const std::size_t leading_rank = dimensions.size() - 2;
        const auto padded = [](std::size_t value) {
            return checked_mul_local(
                    checked_add_local(
                            value, TensorSpec::TILE - 1,
                            "SYCL copy metadata padding overflows")
                            / TensorSpec::TILE,
                    TensorSpec::TILE,
                    "SYCL copy metadata padding overflows");
        };
        std::size_t plane_count = 1;
        for (std::size_t axis = 0; axis < leading_rank; ++axis) {
            plane_count = checked_mul_local(
                    plane_count, dimensions[axis],
                    "SYCL copy metadata plane count overflows");
        }
        const std::size_t plane_bits = checked_mul_local(
                checked_mul_local(
                        padded(dimensions[leading_rank]),
                        padded(dimensions[leading_rank + 1]),
                        "SYCL copy metadata plane size overflows"),
                detail::leaf_bits(request.source.spec.data_type),
                "SYCL copy metadata plane bits overflows");
        const std::size_t words_per_plane =
                checked_add_local(
                        plane_bits, 31,
                        "SYCL copy metadata word count overflows")
                / 32;
        const std::size_t total_words = checked_mul_local(
                plane_count, words_per_plane,
                "SYCL copy metadata total words overflows");
        const std::size_t array_bytes = checked_mul_local(
                checked_mul_local(
                        leading_rank, 3,
                        "SYCL copy metadata array count overflows"),
                sizeof(std::uint64_t),
                "SYCL copy metadata array bytes overflows");
        return {
                checked_add_local(
                        sizeof(detail::CopyMetadataHeader), array_bytes,
                        "SYCL copy metadata size overflows"),
                total_words};
    }

    static void write_copy_metadata_snapshot(
            std::byte* storage, const CopyRequest& request) {
        const auto dimensions = request.source.spec.shape.dimensions();
        const std::size_t leading_rank = dimensions.size() - 2;
        std::size_t plane_count = 1;
        for (std::size_t axis = 0; axis < leading_rank; ++axis) {
            plane_count = checked_mul_local(
                    plane_count, dimensions[axis],
                    "SYCL copy metadata plane count overflows");
        }
        auto* header =
                reinterpret_cast<detail::CopyMetadataHeader*>(storage);
        header->source_plane_offset =
                static_cast<std::uint64_t>(
                        request.source.plane_offset);
        header->destination_plane_offset =
                static_cast<std::uint64_t>(
                        request.destination.plane_offset);
        header->rows = static_cast<std::uint64_t>(
                dimensions[leading_rank]);
        header->columns = static_cast<std::uint64_t>(
                dimensions[leading_rank + 1]);
        header->plane_count = static_cast<std::uint64_t>(plane_count);
        header->bits = static_cast<std::uint32_t>(
                detail::leaf_bits(request.source.spec.data_type));
        header->leading_rank = static_cast<std::uint32_t>(leading_rank);
        auto* values = reinterpret_cast<std::uint64_t*>(header + 1);
        for (std::size_t axis = 0; axis < leading_rank; ++axis) {
            values[axis] = static_cast<std::uint64_t>(
                    request.source.plane_strides[axis]);
            values[leading_rank + axis] = static_cast<std::uint64_t>(
                    request.destination.plane_strides[axis]);
            values[2 * leading_rank + axis] = static_cast<std::uint64_t>(
                    dimensions[axis]);
        }
    }

    void execute(Task& task) {
        if (task.binary_request.has_value()) {
            if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
                throw std::bad_alloc();
            }
            {
                const auto [it, inserted] = outcomes_.emplace(
                        task.sequence,
                        SyclSequenceOutcome{
                                detail::SequenceOutcome{},
                                task.state, task.binary_entries,
                                task.binary_request->workspace_lease});
                if (!inserted) {
                    throw std::logic_error(
                            "duplicate SYCL outstanding-work sequence");
                }
            }
            const BinaryRequest captured = *task.binary_request;
            const std::size_t lhs_bytes =
                    binary_view_staging_bytes(captured, captured.lhs);
            const std::size_t rhs_bytes =
                    binary_view_staging_bytes(captured, captured.rhs);
            const std::size_t out_bytes =
                    binary_view_staging_bytes(captured, captured.out);
            const sycl::context context = queue_.get_context();
            void* lhs_stage = nullptr;
            void* rhs_stage = nullptr;
            void* out_stage = nullptr;
            void* lhs_device = nullptr;
            void* rhs_device = nullptr;
            void* out_device = nullptr;
            bool native_attempted = false;
            bool cleanup_installed = false;
            auto cleanup = [context, &lhs_stage, &rhs_stage, &out_stage]() {
                free_binary_host_staging(lhs_stage, context);
                free_binary_host_staging(rhs_stage, context);
                free_binary_host_staging(out_stage, context);
            };
            try {
                const std::size_t completion_slot =
                        completion_pool_->acquire();
                task.state->set_completion_slot(
                        *completion_pool_, completion_slot);
                lhs_stage = sycl::malloc_host(lhs_bytes, context);
                rhs_stage = sycl::malloc_host(rhs_bytes, context);
                out_stage = sycl::malloc_host(out_bytes, context);
                task.state->set_cleanup(
                        [context, lhs_stage, rhs_stage, out_stage]() {
                            free_binary_host_staging(lhs_stage, context);
                            free_binary_host_staging(rhs_stage, context);
                            free_binary_host_staging(out_stage, context);
                        });
                cleanup_installed = true;
                const std::size_t lhs_aligned =
                        align_up_checked(lhs_bytes, 32);
                const std::size_t rhs_aligned =
                        align_up_checked(rhs_bytes, 32);
                const std::size_t out_offset = checked_add_local(
                        lhs_aligned, rhs_aligned,
                        "SYCL workspace staging offset overflows");
                auto* workspace_base = static_cast<std::byte*>(
                        iom::detail::WorkspaceValidation::address(
                                captured.workspace));
                lhs_device = workspace_base;
                rhs_device = workspace_base + lhs_aligned;
                out_device = workspace_base + out_offset;
                if (lhs_stage == nullptr || rhs_stage == nullptr
                        || out_stage == nullptr || lhs_device == nullptr
                        || rhs_device == nullptr || out_device == nullptr) {
                    throw std::bad_alloc();
                }
                const auto* lhs_source =
                        static_cast<const unsigned char*>(
                                captured.lhs.native_handle);
                auto* lhs_target =
                        static_cast<unsigned char*>(lhs_device);
                // Once the first native enqueue is attempted, conservatively
                // retain the accepted outcome even if the enqueue throws:
                // the runtime may have submitted work before reporting the
                // error, and only a successful drain proves the workspace
                // range can be released.
                if (consume_submission_fault(SubmissionFault::first_submit)) {
                    throw std::runtime_error(
                            "injected SYCL first-submit failure");
                }
                native_attempted = true;
                queue_.parallel_for(
                        sycl::range<1>(lhs_bytes),
                        [=](sycl::id<1> item) {
                            lhs_target[item[0]] = lhs_source[item[0]];
                        });
                const auto* rhs_source =
                        static_cast<const unsigned char*>(
                                captured.rhs.native_handle);
                auto* rhs_target =
                        static_cast<unsigned char*>(rhs_device);
                queue_.parallel_for(
                        sycl::range<1>(rhs_bytes),
                        [=](sycl::id<1> item) {
                            rhs_target[item[0]] = rhs_source[item[0]];
                        });
                const auto* out_source =
                        static_cast<const unsigned char*>(
                                captured.out.native_handle);
                auto* out_target =
                        static_cast<unsigned char*>(out_device);
                queue_.parallel_for(
                        sycl::range<1>(out_bytes),
                        [=](sycl::id<1> item) {
                            out_target[item[0]] = out_source[item[0]];
                        });
                queue_.memcpy(lhs_stage, lhs_device, lhs_bytes);
                queue_.memcpy(rhs_stage, rhs_device, rhs_bytes);
                sycl::event input_event =
                        queue_.memcpy(out_stage, out_device, out_bytes);
                sycl::event binary_event = queue_.submit(
                        [&](sycl::handler& handler) {
                            handler.depends_on(input_event);
                            handler.host_task(
                                    [captured, lhs_stage, rhs_stage, out_stage] {
                                        binary_elements(
                                                captured,
                                                static_cast<const unsigned char*>(
                                                        lhs_stage),
                                                static_cast<const unsigned char*>(
                                                        rhs_stage),
                                                static_cast<unsigned char*>(
                                                        out_stage));
                                    });
                        });
                sycl::event output_copy_event = queue_.submit(
                        [&](sycl::handler& handler) {
                            handler.depends_on(binary_event);
                            handler.memcpy(out_device, out_stage, out_bytes);
                        });
                const auto* out_device_source =
                        static_cast<const unsigned char*>(out_device);
                auto* out_destination =
                        static_cast<unsigned char*>(
                                captured.out.native_handle);
                sycl::event output_event = queue_.submit(
                        [&](sycl::handler& handler) {
                            handler.depends_on(output_copy_event);
                            handler.parallel_for(
                                    sycl::range<1>(out_bytes),
                                    [=](sycl::id<1> item) {
                                        out_destination[item[0]] =
                                                out_device_source[item[0]];
                                    });
                        });
                task.state->set_event(std::move(output_event));
                if (consume_submission_fault(SubmissionFault::post_launch)) {
                    throw std::runtime_error(
                            "injected SYCL post-launch failure");
                }
            } catch (...) {
                if (!native_attempted) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(task.sequence);
                    }
                    task.state->mark_completion_proven();
                    if (cleanup_installed) {
                        task.state->cleanup_now();
                    } else {
                        cleanup();
                    }
                    task.state.reset();
                    task.fence = nullptr;
                    throw;
                }
                task.state->set_failure(std::current_exception());
            }
            return;
        }
        if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
            throw std::bad_alloc();
        }
        {
            const auto [it, inserted] = outcomes_.emplace(
                    task.sequence,
                    SyclSequenceOutcome{
                            detail::SequenceOutcome{
                                    task.copy_entries.source,
                                    task.copy_entries.destination},
                            task.state});
            if (!inserted) {
                throw std::logic_error(
                        "duplicate SYCL outstanding-work sequence");
            }
        }
        if (task.no_op) {
            task.state.reset();
            task.fence = nullptr;
            return;
        }

        bool native_attempted = false;
        bool metadata_enqueued = false;
        try {
            if (consume_submission_fault(SubmissionFault::first_submit)) {
                throw std::runtime_error(
                        "injected SYCL first-submit failure");
            }

            const std::size_t completion_slot =
                    completion_pool_->acquire();
            task.state->set_completion_slot(
                    *completion_pool_, completion_slot);
            // One fixed 512-byte slot from this queue's partition carries
            // the immutable pointer-copy descriptor; no growth, replacement,
            // or native allocation ever happens here.
            const std::size_t metadata_slot = metadata_pool_->acquire();
            task.state->set_metadata_slot(*metadata_pool_, metadata_slot);
            const detail::CopyMetadataLayout layout =
                    copy_metadata_layout_snapshot(*task.copy_request);
            write_copy_metadata_snapshot(
                    metadata_pool_->host_data(metadata_slot),
                    *task.copy_request);
            native_attempted = true;
            queue_.memcpy(
                    metadata_pool_->device_data(metadata_slot),
                    metadata_pool_->host_data(metadata_slot), layout.bytes);
            metadata_enqueued = true;

            const auto* source_handle = static_cast<const unsigned char*>(
                    task.copy_request->source.native_handle);
            auto* destination_handle = static_cast<unsigned char*>(
                    task.copy_request->destination.native_handle);
            const auto* metadata = static_cast<
                    const detail::CopyMetadataHeader*>(
                    metadata_pool_->device_data(metadata_slot));
            sycl::event event = queue_.parallel_for(
                    sycl::range<1>(layout.total_words),
                    [=](sycl::id<1> item) {
                        detail::copy_one_tiled_word(
                                source_handle, destination_handle, *metadata,
                                reinterpret_cast<const std::uint64_t*>(
                                        metadata + 1),
                                item[0]);
                    });
            if (launch_calls.kernel_launched != nullptr) {
                launch_calls.kernel_launched();
            }
            task.state->set_event(std::move(event));
            if (consume_submission_fault(SubmissionFault::post_launch)) {
                throw std::runtime_error(
                        "injected SYCL post-launch failure");
            }
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (!native_attempted) {
                task.state->mark_completion_proven();
                {
                    std::lock_guard<std::mutex> lock(outcome_mutex_);
                    outcomes_.erase(task.sequence);
                }
                task.fence = nullptr;
                task.state.reset();
                throw;
            }
            task.state->set_failure(failure);
        }
    }

    void complete_task(
            std::uint64_t sequence, std::exception_ptr callback_failure) {
        SyclSequenceOutcome outcome;
        bool has_outcome = false;
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto it = outcomes_.find(sequence);
            if (it != outcomes_.end()) {
                outcome = std::move(it->second);
                outcomes_.erase(it);
                has_outcome = true;
            }
        }

        std::exception_ptr combined_failure;
        if (has_outcome) {
            const detail::FenceResult fence_result =
                    outcome.state != nullptr
                    ? outcome.state->result()
                    : detail::FenceResult::success();
            combined_failure =
                    fence_result.failure ? fence_result.failure
                                          : callback_failure;
            const bool fence_succeeded =
                    fence_result.succeeded && !fence_result.failure;
            const bool failed = static_cast<bool>(combined_failure);
            if (outcome.binary_entries.has_value()) {
                (void)detail::release_or_invalidate_binary_entries(
                        state_->registry, *outcome.binary_entries, failed,
                        fence_succeeded);
                detail::complete_workspace_lease(
                        *state_, outcome.workspace_lease,
                        outcome.state != nullptr
                                && outcome.state->completion_proven());
            } else {
                (void)detail::release_or_invalidate_entries(
                        state_->registry, outcome.common, failed,
                        fence_succeeded);
            }
        } else {
            combined_failure = callback_failure;
        }
        complete(sequence, std::move(combined_failure));
    }

    // Constructs the native in-order queue behind the queue-stream fault
    // seam so construction rollback stays transactional.
    static sycl::queue make_queue_with_fault_check(
            const sycl::context& context, const sycl::device& native_device) {
        if (consume_submission_fault(SubmissionFault::queue_stream_create)) {
            throw std::runtime_error(
                    "injected SYCL queue creation failure");
        }
        return sycl::queue(
                context, native_device,
                sycl::property_list{sycl::property::queue::in_order{}});
    }

    const Device* device_;
    detail::RegistryState* state_;
    detail::QueueResourceProvider* resource_provider_;
    detail::QueueId registry_queue_id_;
    // The fixed partition lease is reserved before the native queue is
    // constructed (member declaration order).
    std::shared_ptr<detail::MetadataSlotPool> metadata_pool_;
    std::shared_ptr<SyclCompletionPool> completion_pool_;
    sycl::queue queue_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, SyclSequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
};


}  // namespace

#ifdef IOM_ENABLE_TESTING
void queue_resource_snapshot_for_testing(
        DeviceOps& queue, QueueResourceSnapshot& snapshot) {
    auto* sycl_queue = dynamic_cast<SyclQueue*>(&queue);
    if (sycl_queue == nullptr) {
        throw std::logic_error("queue is not a live SYCL queue");
    }
    detail::MetadataSlotPool& pool =
            sycl_queue->metadata_pool_for_testing();
    snapshot.slot_count = pool.slot_count();
    snapshot.device_base = pool.device_base();
    snapshot.slot_stride = pool.slot_stride();
    snapshot.slots_in_use = pool.in_use_count();
    snapshot.slots_protected = pool.protected_count();
    snapshot.events_total =
            sycl_queue->completion_pool_for_testing().count();
    snapshot.events_in_use =
            sycl_queue->completion_pool_for_testing().in_use_count();
}
#endif  // IOM_ENABLE_TESTING

void inject_submission_fault_for_testing(
        SubmissionFault fault) noexcept {
    g_submission_fault.store(fault, std::memory_order_release);
}

void reset_fence_wait_count_for_testing() noexcept {
    g_fence_wait_count.store(0, std::memory_order_release);
}

std::size_t fence_wait_count_for_testing() noexcept {
    return g_fence_wait_count.load(std::memory_order_acquire);
}

namespace {

detail::Fence transfer_fence() noexcept {
    detail::Fence fence;
    fence.invoke = [](const detail::Fence&) noexcept {
        return detail::FenceResult::success();
    };
    return fence;
}

struct WorkspaceAdmission {
    detail::WorkspaceLease lease;
    void* address = nullptr;
};

WorkspaceAdmission begin_workspace_transfer(
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& view, RawWorkspaceView workspace,
        bool& resource_poisoned) {
    const WorkspaceRequirements requirements =
            view.copy_from_host_workspace_requirements();
    const std::array<TensorView, 1> operands{view};
    const RawWorkspaceView checked =
            detail::WorkspaceValidation::validated(
                    device, workspace, requirements.bytes,
                    requirements.alignment, operands);
    if (resource_poisoned) {
        throw std::bad_alloc();
    }
    const detail::QueueId queue_id =
            detail::allocate_queue_id(registry_state);
    return {
            detail::acquire_workspace_lease(
                    registry_state, workspace.owner_identity(),
                    detail::WorkspaceValidation::address(checked),
                    checked.byte_size(), queue_id, queue_id, transfer_fence()),
            detail::WorkspaceValidation::address(checked)};
}

void finish_workspace_transfer(
        detail::RegistryState& registry_state,
        const detail::WorkspaceLease& lease, bool proof) noexcept {
    detail::complete_workspace_lease(registry_state, lease, proof);
}

}  // namespace

void region_from_host(
        sycl::queue& transfer_queue, const Device& device,
        detail::RegistryState& registry_state,
        const TensorView& destination, RawWorkspaceView workspace,
        std::span<const std::byte> source, bool& resource_poisoned) {
    WorkspaceAdmission admission = begin_workspace_transfer(
            device, registry_state, destination, workspace, resource_poisoned);
    try {
        const std::size_t logical_nbytes =
                destination.spec().logical_nbytes();
        transfer_queue.memcpy(
                admission.address, source.data(), logical_nbytes);
        launch_view_transfer(
                transfer_queue, destination, admission.address,
                const_cast<void*>(destination.native_handle()), true);
        transfer_queue.wait_and_throw();
        finish_workspace_transfer(registry_state, admission.lease, true);
    } catch (...) {
        resource_poisoned = true;
        try {
            transfer_queue.wait_and_throw();
        } catch (...) {
        }
        finish_workspace_transfer(registry_state, admission.lease, false);
        throw;
    }
}

void region_to_host(
        sycl::queue& transfer_queue, const Device& device,
        detail::RegistryState& registry_state, const TensorView& source,
        RawWorkspaceView workspace, std::span<std::byte> destination,
        bool& resource_poisoned) {
    WorkspaceAdmission admission = begin_workspace_transfer(
            device, registry_state, source, workspace, resource_poisoned);
    try {
        const std::size_t logical_nbytes = source.spec().logical_nbytes();
        const std::size_t logical_bits =
                source.spec().shape.element_count()
                * detail::leaf_bits(source.spec().data_type);
        const std::size_t staging_nbytes =
                gpu_algorithm::compute_staging_size(logical_nbytes);
        const std::size_t tail_word_start =
                (logical_bits / 32) * sizeof(std::uint32_t);
        const std::size_t tail_bytes = staging_nbytes - tail_word_start;
        if (tail_bytes != 0) {
            transfer_queue.memset(
                    static_cast<std::byte*>(admission.address)
                            + tail_word_start,
                    0, tail_bytes);
        }
        launch_view_transfer(
                transfer_queue, source, source.native_handle(),
                admission.address, false);
        transfer_queue.memcpy(
                destination.data(), admission.address, logical_nbytes);
        transfer_queue.wait_and_throw();
        finish_workspace_transfer(registry_state, admission.lease, true);
    } catch (...) {
        resource_poisoned = true;
        try {
            transfer_queue.wait_and_throw();
        } catch (...) {
        }
        finish_workspace_transfer(registry_state, admission.lease, false);
        throw;
    }
}

std::unique_ptr<DeviceOps> make_queue(
        const Device& device, detail::QueueResourceProvider& resource_provider,
        const sycl::context& context, const sycl::device& native_device,
        detail::RegistryState& registry_state) {
    return std::make_unique<SyclQueue>(
            device, resource_provider, context, native_device,
            registry_state);
}

#ifdef IOM_ENABLE_TESTING
void reclaim_retained_queue_leases_for_testing(Device& device) {
    auto* provider =
            dynamic_cast<detail::QueueResourceProvider*>(&device);
    if (provider == nullptr) {
        throw std::logic_error("device does not own fixed queue resources");
    }
    provider->reclaim_retained_leases();
}
#endif  // IOM_ENABLE_TESTING

}  // namespace iom::sycl_detail
