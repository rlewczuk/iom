#include "copy.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
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

class SyclQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        const TensorView* source = nullptr;
        TensorView* destination = nullptr;
        bool no_op = false;
        std::shared_ptr<SyclFenceState> state;
        void* fence = state.get();
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
        std::optional<AddRequest> add_request;
        detail::AddEntryRegistration add_entries;
    };
    struct SyclSequenceOutcome {
        detail::SequenceOutcome common;
        std::shared_ptr<SyclFenceState> state;
        std::optional<detail::AddEntryRegistration> add_entries;
    };


public:
    SyclQueue(
            const Device& device, const sycl::context& context,
            const sycl::device& native_device,
            detail::RegistryState& state)
            : DeviceOps(device),
              device_(&device),
              state_(&state),
              registry_queue_id_(detail::allocate_queue_id(state)),
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
        // Drain queued device work before destroying task captures and
        // their owner registrations.
        try {
            queue_.wait_and_throw();
        } catch (...) {
        }
        worker_.shutdown_and_drain();
        state_->registry.invalidate_entries_for_queue(registry_queue_id_);
    }

    oid copy_impl(
            const TensorView& source,
            TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    worker_.submit_copy(Task{
                            sequence, &source, &destination, no_op});
                });
    }

    oid add_impl(const AddRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        if (consume_submission_fault(SubmissionFault::state_allocation)) {
            throw std::bad_alloc();
        }
        auto state = std::make_shared<SyclFenceState>();
        detail::Fence fence = build_sycl_fence(state);
        return submit_add(
                request, *state_, registry_queue_id_, fence,
                [this, state](std::uint64_t sequence,
                              const AddRequest& captured,
                              detail::AddEntryRegistration entries) {
                    Task task;
                    task.sequence = sequence;
                    task.state = state;
                    task.fence = state.get();
                    task.add_request.emplace(captured);
                    task.add_entries = entries;
                    worker_.submit_copy(std::move(task));
                });
    }

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "SYCL";
    }

private:
    // Device-USM-safe ADD staging extent: whole plane blocks up to the
    // view's highest addressed plane, so untouched planes and tile padding
    // round-trip unchanged. validate_add bounded this walk with checked
    // arithmetic and equal plane geometry, so the recompute cannot
    // overflow for an accepted request.
    static std::size_t add_view_staging_bytes(
            const AddRequest& captured, const AddViewSnapshot& view) {
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

    // Exact shared scalar loop over host staging buffers.
    static void add_elements(
            const AddRequest& captured, const unsigned char* lhs_storage,
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
                value |= ((bytes[(bit + i) / 8] >>
                           ((bit + i) % 8)) & 1u) << i;
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
            auto plane = [&](const AddViewSnapshot& view) {
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
            const auto slot = [&](const AddViewSnapshot& view,
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
                  detail::scalar_add(
                          captured.out.spec.data_type, a, b));
        }
    }

    static void free_add_staging(
            void* staging, const sycl::context& context) noexcept {
        if (staging == nullptr) {
            return;
        }
        try {
            sycl::free(staging, context);
        } catch (...) {
        }
    }

    void execute(Task& task) {
        if (task.add_request.has_value()) {
            if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
                throw std::bad_alloc();
            }
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                const auto [it, inserted] = outcomes_.emplace(
                        task.sequence,
                        SyclSequenceOutcome{
                                detail::SequenceOutcome{},
                                task.state, task.add_entries});
                if (!inserted) {
                    throw std::logic_error("duplicate SYCL outstanding-work sequence");
                }
            }
            const AddRequest& captured = *task.add_request;
            const std::size_t lhs_bytes =
                    add_view_staging_bytes(captured, captured.lhs);
            const std::size_t rhs_bytes =
                    add_view_staging_bytes(captured, captured.rhs);
            const std::size_t out_bytes =
                    add_view_staging_bytes(captured, captured.out);
            const sycl::context context = queue_.get_context();
            void* lhs_stage = nullptr;
            void* rhs_stage = nullptr;
            void* out_stage = nullptr;
            void* lhs_device = nullptr;
            void* rhs_device = nullptr;
            void* out_device = nullptr;
            bool submitted = false;
            bool staged_enqueued = false;
            try {
                if (consume_submission_fault(SubmissionFault::first_submit)) {
                    throw std::runtime_error("injected SYCL first-submit failure");
                }
                // Tensor storage is device USM and this runtime rejects
                // direct queue memcpy operations involving caller handles.
                // The existing copy kernels can access those handles, so
                // each extent first crosses through a temporary device-USM
                // buffer. Only the temporary buffers participate in memcpy;
                // the host scalar loop still uses the exact shared
                // long-double implementation, preserving cross-backend
                // bit-exactness. All transfers stay ordered on this
                // in-order queue. Tradeoff: the element loop executes on
                // the host between queue drains instead of a device kernel.
                lhs_stage = sycl::malloc_host(lhs_bytes, context);
                rhs_stage = sycl::malloc_host(rhs_bytes, context);
                out_stage = sycl::malloc_host(out_bytes, context);
                lhs_device = sycl::malloc_device(
                        lhs_bytes, queue_.get_device(), context);
                rhs_device = sycl::malloc_device(
                        rhs_bytes, queue_.get_device(), context);
                out_device = sycl::malloc_device(
                        out_bytes, queue_.get_device(), context);
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
                queue_.parallel_for(
                        sycl::range<1>(lhs_bytes),
                        [=](sycl::id<1> item) {
                            lhs_target[item[0]] = lhs_source[item[0]];
                        });
                staged_enqueued = true;
                submitted = true;
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
                queue_.memcpy(out_stage, out_device, out_bytes);
                queue_.wait_and_throw();
                add_elements(
                        captured,
                        static_cast<const unsigned char*>(lhs_stage),
                        static_cast<const unsigned char*>(rhs_stage),
                        static_cast<unsigned char*>(out_stage));
                queue_.memcpy(out_device, out_stage, out_bytes);
                const auto* out_device_source =
                        static_cast<const unsigned char*>(out_device);
                auto* out_destination =
                        static_cast<unsigned char*>(
                                captured.out.native_handle);
                queue_.parallel_for(
                        sycl::range<1>(out_bytes),
                        [=](sycl::id<1> item) {
                            out_destination[item[0]] =
                                    out_device_source[item[0]];
                        });
                if (consume_submission_fault(SubmissionFault::post_launch)) {
                    throw std::runtime_error("injected SYCL post-launch failure");
                }
                queue_.wait_and_throw();
            } catch (...) {
                if (staged_enqueued) {
                    try {
                        queue_.wait_and_throw();
                    } catch (...) {
                    }
                }
                free_add_staging(lhs_stage, context);
                free_add_staging(rhs_stage, context);
                free_add_staging(out_stage, context);
                free_add_staging(lhs_device, context);
                free_add_staging(rhs_device, context);
                free_add_staging(out_device, context);
                if (!submitted) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(task.sequence);
                    }
                    // Entry rollback belongs to submit_add: it registered
                    // the ADD entries, and a second remove here replaced
                    // the original failure with invalid_argument.
                    task.state.reset();
                    task.fence = nullptr;
                    throw;
                }
                task.state->set_failure(std::current_exception());
                return;
            }
            free_add_staging(lhs_stage, context);
            free_add_staging(rhs_stage, context);
            free_add_staging(out_stage, context);
            free_add_staging(lhs_device, context);
            free_add_staging(rhs_device, context);
            free_add_staging(out_device, context);
            return;
        }
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
            if (outcome.add_entries.has_value()) {
                (void)detail::release_or_invalidate_add_entries(
                        state_->registry, *outcome.add_entries,
                        static_cast<bool>(callback_failure),
                        fence_succeeded);
            } else {
                (void)detail::release_or_invalidate_entries(
                        state_->registry, outcome.common,
                        static_cast<bool>(callback_failure), fence_succeeded);
            }
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
        const std::exception_ptr enqueue_failure =
                std::current_exception();
        try {
            transfer_queue.wait_and_throw();
        } catch (...) {
        }
        lease.poison();
        std::rethrow_exception(enqueue_failure);
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
        const std::exception_ptr enqueue_failure =
                std::current_exception();
        try {
            transfer_queue.wait_and_throw();
        } catch (...) {
        }
        lease.poison();
        std::rethrow_exception(enqueue_failure);
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
