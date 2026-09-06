#include "copy.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
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

#include "iom/gpu_algorithm.hpp"
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

class SyclMetadataSlotPool final {
public:
    static constexpr std::size_t kMetadataSlotCount = 16;

    SyclMetadataSlotPool(
            const sycl::context& context, sycl::queue& queue)
            : context_(context), queue_(queue) {}

    ~SyclMetadataSlotPool() noexcept {
        for (Slot& slot : slots_) {
            free_slot_noexcept(slot);
        }
    }

    SyclMetadataSlotPool(const SyclMetadataSlotPool&) = delete;
    SyclMetadataSlotPool& operator=(const SyclMetadataSlotPool&) = delete;

    [[nodiscard]] std::size_t acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        completion_.wait(lock, [this] {
            for (std::size_t index = 0; index < slot_count_; ++index) {
                if (!slots_[index].in_use) {
                    return true;
                }
            }
            return slot_count_ < kMetadataSlotCount;
        });
        for (std::size_t index = 0; index < slot_count_; ++index) {
            if (!slots_[index].in_use) {
                slots_[index].in_use = true;
                return index;
            }
        }
        if (slot_count_ == kMetadataSlotCount) {
            throw std::logic_error(
                    "metadata slot acquisition lost a free slot");
        }
        const std::size_t index = slot_count_++;
        slots_[index].in_use = true;
        return index;
    }

    void release(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < slot_count_ && slots_[index].in_use) {
            slots_[index].in_use = false;
            completion_.notify_one();
        }
    }

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

        queue_.wait_and_throw();
        void* replacement_device =
                sycl::malloc_device(capacity, queue_.get_device(), context_);
        if (replacement_device == nullptr) {
            throw std::bad_alloc();
        }
        void* replacement_host = nullptr;
        try {
            replacement_host = sycl::malloc_host(capacity, context_);
            if (replacement_host == nullptr) {
                throw std::bad_alloc();
            }
        } catch (...) {
            try {
                sycl::free(replacement_device, context_);
            } catch (...) {
            }
            throw;
        }

        void* old_device = slot.device;
        void* old_host = slot.host;
        slot.device = replacement_device;
        slot.host = replacement_host;
        slot.capacity = capacity;
        free_pointer_noexcept(old_device);
        free_pointer_noexcept(old_host);
    }

    [[nodiscard]] std::byte* host_data(std::size_t index) {
        return static_cast<std::byte*>(slots_.at(index).host);
    }

    [[nodiscard]] void* device_data(std::size_t index) {
        return slots_.at(index).device;
    }

private:
    struct Slot {
        void* device = nullptr;
        void* host = nullptr;
        std::size_t capacity = 0;
        bool in_use = false;
    };

    void free_pointer_noexcept(void* pointer) noexcept {
        if (pointer == nullptr) {
            return;
        }
        try {
            sycl::free(pointer, context_);
        } catch (...) {
        }
    }

    void free_slot_noexcept(Slot& slot) noexcept {
        free_pointer_noexcept(slot.device);
        free_pointer_noexcept(slot.host);
        slot = Slot{};
    }

    sycl::context context_;
    sycl::queue& queue_;
    std::array<Slot, kMetadataSlotCount> slots_;
    std::size_t slot_count_ = 0;
    std::mutex mutex_;
    std::condition_variable completion_;
};

class SyclFenceState final {
public:
    static_assert(
            std::is_nothrow_move_constructible_v<sycl::event>
            && std::is_nothrow_move_assignable_v<sycl::event>);

    ~SyclFenceState() noexcept { release_metadata_slot(); }

    void set_event(sycl::event incoming) noexcept {
        event_ = std::move(incoming);
    }

    void set_failure(std::exception_ptr incoming) noexcept {
        retained_failure_ = std::move(incoming);
    }

    [[nodiscard]] detail::FenceResult result() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        if (cached_.has_value()) {
            return *cached_;
        }

        detail::FenceResult result = detail::FenceResult::success();
        if (event_.has_value()) {
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
        cached_ = result;
        return result;
    }

    void clear_event() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        event_ = std::nullopt;
    }

    void release_metadata_slot() noexcept {
        SyclMetadataSlotPool* pool = nullptr;
        std::size_t slot = 0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            pool = pool_;
            slot = slot_;
            pool_ = nullptr;
            slot_ = 0;
        }
        if (pool != nullptr) {
            pool->release(slot);
        }
    }

private:
    void set_metadata_slot(
            SyclMetadataSlotPool& pool, std::size_t slot) noexcept {
        pool_ = &pool;
        slot_ = slot;
    }

    std::mutex mu_;
    std::optional<sycl::event> event_;
    std::exception_ptr retained_failure_;
    std::optional<detail::FenceResult> cached_;
    std::size_t slot_ = 0;
    SyclMetadataSlotPool* pool_ = nullptr;

    friend class SyclQueue;
};

struct SyclFenceCapture {
    std::shared_ptr<SyclFenceState> state;
};

static_assert(
        sizeof(SyclFenceCapture) <= detail::kFenceStorageBytes);
static_assert(
        alignof(SyclFenceCapture) <= detail::kFenceStorageAlign);

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

void sycl_fence_copy_construct(
        detail::Fence* destination, const detail::Fence& source) noexcept {
    ::new (destination->storage) SyclFenceCapture{
            *std::launder(reinterpret_cast<const SyclFenceCapture*>(
                    source.storage))};
}

static_assert(noexcept(sycl_fence_copy_construct(
        std::declval<detail::Fence*>(),
        std::declval<const detail::Fence&>())));

void sycl_fence_move_construct(
        detail::Fence* destination, detail::Fence* source) noexcept {
    ::new (destination->storage) SyclFenceCapture{
            std::move(*std::launder(reinterpret_cast<SyclFenceCapture*>(
                    source->storage)))};
    std::destroy_at(std::launder(reinterpret_cast<SyclFenceCapture*>(
            source->storage)));
}

static_assert(noexcept(sycl_fence_move_construct(
        std::declval<detail::Fence*>(),
        std::declval<detail::Fence*>())));

void sycl_fence_storage_destroy(detail::Fence* fence) noexcept {
    std::destroy_at(std::launder(reinterpret_cast<SyclFenceCapture*>(
            fence->storage)));
}

static_assert(noexcept(sycl_fence_storage_destroy(
        std::declval<detail::Fence*>())));

detail::Fence build_sycl_fence(
        const std::shared_ptr<SyclFenceState>& state) noexcept {
    detail::Fence fence;
    ::new (fence.storage) SyclFenceCapture{state};
    fence.invoke = &sycl_fence_invoke;
    fence.copy_construct = &sycl_fence_copy_construct;
    fence.move_construct = &sycl_fence_move_construct;
    fence.destroy = &sycl_fence_storage_destroy;
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

class SyclQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        const TensorView* source;
        TensorView* destination;
        bool no_op;
        std::shared_ptr<SyclFenceState> state;
        void* fence = state.get();
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
    };
    struct SyclSequenceOutcome {
        detail::SequenceOutcome common;
        std::shared_ptr<SyclFenceState> state;
    };


public:
    SyclQueue(
            const Device& device, const sycl::context& context,
            const sycl::device& native_device,
            detail::RegistryState& state)
            : device_(&device),
              state_(&state),
              registry_queue_id_(detail::allocate_queue_id(*state_)),
              queue_(
                      context, native_device,
                      sycl::property_list{
                              sycl::property::queue::in_order{}}),
              metadata_pool_(context, queue_),
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
        state_->registry.invalidate_entries_for_queue(registry_queue_id_);
        worker_.shutdown_and_drain();
    }

    oid copy(
            const TensorView& source,
            TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        validate_copy(*device_, source, destination);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    worker_.submit_copy(Task{
                            sequence, &source, &destination, no_op});
                });
    }

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "SYCL";
    }

private:
    void execute(Task& task) {
        if (task.no_op) {
            task.state = nullptr;
            task.fence = nullptr;
            return;
        }

        if (consume_submission_fault(SubmissionFault::state_allocation)) {
            throw std::bad_alloc();
        }
        task.state = std::make_shared<SyclFenceState>();
        task.fence = task.state.get();

        detail::EntryRegistration entries;
        bool outcome_inserted = false;
        try {
            if (consume_submission_fault(
                        SubmissionFault::fence_construction)) {
                throw std::bad_alloc();
            }
            detail::Fence fence = build_sycl_fence(task.state);
            entries = detail::register_copy_entries(
                    *state_, registry_queue_id_, task.sequence,
                    const_cast<void*>(task.source->native_handle()),
                    task.destination->native_handle(), fence);
            task.source_entry_id = entries.source;
            task.destination_entry_id = entries.destination;

            if (consume_submission_fault(
                        SubmissionFault::outcome_insertion)) {
                throw std::bad_alloc();
            }
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto [it, inserted] = outcomes_.emplace(
                    task.sequence,
                    SyclSequenceOutcome{
                            detail::SequenceOutcome{
                                    entries.source, entries.destination},
                            task.state});
            if (!inserted) {
                throw std::logic_error(
                        "duplicate SYCL outstanding-work sequence");
            }
            (void)it;
            outcome_inserted = true;
        } catch (...) {
            if (outcome_inserted) {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                outcomes_.erase(task.sequence);
            }
            if (entries.source != 0) {
                const std::array<detail::EntryId, 2> ids{
                        entries.source, entries.destination};
                state_->registry.remove_entries(ids);
            }
            task.source_entry_id = 0;
            task.destination_entry_id = 0;
            task.fence = nullptr;
            task.state.reset();
            throw;
        }

        bool submitted_any = false;
        bool metadata_enqueued = false;
        try {
            if (consume_submission_fault(SubmissionFault::first_submit)) {
                throw std::runtime_error(
                        "injected SYCL first-submit failure");
            }

            const std::size_t metadata_slot = metadata_pool_.acquire();
            task.state->set_metadata_slot(metadata_pool_, metadata_slot);
            const detail::CopyMetadataLayout layout =
                    detail::copy_metadata_layout(
                            *task.source, *task.destination);
            metadata_pool_.ensure_slot_capacity(metadata_slot, layout.bytes);
            detail::write_copy_metadata(
                    metadata_pool_.host_data(metadata_slot),
                    *task.source, *task.destination);
            queue_.memcpy(
                    metadata_pool_.device_data(metadata_slot),
                    metadata_pool_.host_data(metadata_slot), layout.bytes);
            metadata_enqueued = true;

            const auto* source_handle = static_cast<const unsigned char*>(
                    task.source->native_handle());
            auto* destination_handle = static_cast<unsigned char*>(
                    task.destination->native_handle());
            const auto* metadata = static_cast<
                    const detail::CopyMetadataHeader*>(
                    metadata_pool_.device_data(metadata_slot));
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
            submitted_any = true;
            if (consume_submission_fault(SubmissionFault::post_launch)) {
                throw std::runtime_error(
                        "injected SYCL post-launch failure");
            }
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (!submitted_any) {
                if (metadata_enqueued) {
                    try {
                        queue_.wait_and_throw();
                    } catch (...) {
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(outcome_mutex_);
                    outcomes_.erase(task.sequence);
                }
                const std::array<detail::EntryId, 2> ids{
                        task.source_entry_id, task.destination_entry_id};
                state_->registry.remove_entries(ids);
                task.source_entry_id = 0;
                task.destination_entry_id = 0;
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
                    outcome.state->result();
            combined_failure =
                    fence_result.failure ? fence_result.failure
                                          : callback_failure;
            const bool fence_succeeded =
                    fence_result.succeeded && !fence_result.failure;
            (void)detail::release_or_invalidate_entries(
                    state_->registry, outcome.common,
                    static_cast<bool>(callback_failure), fence_succeeded);
        } else {
            combined_failure = callback_failure;
        }
        complete(sequence, std::move(combined_failure));
    }

    const Device* device_;
    detail::RegistryState* state_;
    detail::QueueId registry_queue_id_;
    sycl::queue queue_;
    SyclMetadataSlotPool metadata_pool_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, SyclSequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
};

}  // namespace

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

void region_from_host(
        StagingSlotPool& staging_pool, sycl::queue& transfer_queue,
        const TensorView& destination, void* storage,
        std::span<const std::byte> source) {
    const std::size_t logical_nbytes = destination.spec().logical_nbytes();
    const std::size_t staging_nbytes =
            gpu_algorithm::compute_staging_size(logical_nbytes);
    auto lease = staging_pool.acquire(staging_nbytes);
    try {
        std::memcpy(lease.host_mirror(), source.data(), logical_nbytes);
        transfer_queue.memcpy(
                lease.device_staging(), lease.host_mirror(), logical_nbytes);
        launch_view_transfer(
                transfer_queue, destination, lease.device_staging(),
                storage, true);
        transfer_queue.wait_and_throw();
    } catch (...) {
        lease.poison();
        throw;
    }
}

void region_to_host(
        StagingSlotPool& staging_pool, sycl::queue& transfer_queue,
        const TensorView& source, const void* storage,
        std::span<std::byte> destination) {
    const std::size_t logical_nbytes = source.spec().logical_nbytes();
    const std::size_t staging_nbytes =
            gpu_algorithm::compute_staging_size(logical_nbytes);
    const std::size_t logical_bits =
            source.spec().shape.element_count()
            * detail::leaf_bits(source.spec().data_type);
    auto lease = staging_pool.acquire(staging_nbytes);
    try {
        const std::size_t tail_word_start =
                (logical_bits / 32) * sizeof(std::uint32_t);
        const std::size_t tail_bytes = staging_nbytes - tail_word_start;
        if (tail_bytes != 0) {
            transfer_queue.memset(
                    static_cast<std::byte*>(lease.device_staging())
                            + tail_word_start,
                    0, tail_bytes);
        }
        launch_view_transfer(
                transfer_queue, source, storage, lease.device_staging(), false);
        transfer_queue.memcpy(
                lease.host_mirror(), lease.device_staging(), staging_nbytes);
        transfer_queue.wait_and_throw();
        std::memcpy(
                destination.data(), lease.host_mirror(), logical_nbytes);
    } catch (...) {
        lease.poison();
        throw;
    }
}

std::unique_ptr<DeviceOps> make_queue(
        const Device& device, const sycl::context& context,
        const sycl::device& native_device,
        detail::RegistryState& registry_state) {
    return std::make_unique<SyclQueue>(
            device, context, native_device, registry_state);
}

}  // namespace iom::sycl_detail
