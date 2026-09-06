#include "copy.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

class SyclFenceState final {
public:
    static_assert(
            std::is_nothrow_move_constructible_v<sycl::event>
            && std::is_nothrow_move_assignable_v<sycl::event>);

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
        event_.reset();
    }

private:
    std::mutex mu_;
    std::optional<sycl::event> event_;
    std::exception_ptr retained_failure_;
    std::optional<detail::FenceResult> cached_;
};

constexpr std::size_t kTile = TensorSpec::TILE;
constexpr std::size_t kTileSlots = kTile * kTile;

struct PlanePair {
    std::size_t source;
    std::size_t destination;
};

std::vector<std::size_t> view_planes(const TensorView& view) {
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::size_t plane_elements = rows * columns;
    const std::size_t plane_count =
            view.spec().shape.element_count() / plane_elements;

    std::vector<std::size_t> planes;
    planes.reserve(plane_count);
    for (std::size_t logical_plane = 0; logical_plane < plane_count;
         ++logical_plane) {
        std::size_t rest = logical_plane;
        std::size_t plane = view.plane_offset();
        for (std::size_t axis = leading_rank; axis-- > 0;) {
            plane += (rest % dimensions[axis])
                    * view.plane_strides()[axis];
            rest /= dimensions[axis];
        }
        planes.push_back(plane);
    }
    return planes;
}

std::vector<PlanePair> plane_pairs(
        const TensorView& source, const TensorView& destination) {
    const std::vector<std::size_t> source_planes = view_planes(source);
    const std::vector<std::size_t> destination_planes =
            view_planes(destination);
    std::vector<PlanePair> pairs;
    pairs.reserve(source_planes.size());
    for (std::size_t i = 0; i < source_planes.size(); ++i) {
        pairs.push_back({source_planes[i], destination_planes[i]});
    }
    return pairs;
}


std::uint64_t device_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_rows =
            (rows + kTile - 1) / kTile;
    const std::uint64_t tile_columns =
            (columns + kTile - 1) / kTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kTile) * tile_columns + column / kTile;
    return tile_index * kTileSlots
            + (row % kTile) * kTile + column % kTile;
}

std::uint64_t device_read_bits(
        const unsigned char* base, std::uint64_t bit_offset,
        unsigned int nbits) {
    std::uint64_t value = 0;
    for (unsigned int i = 0; i < nbits; ++i) {
        const std::uint64_t bit = bit_offset + i;
        value |= static_cast<std::uint64_t>(
                         (base[bit / 8] >> (bit % 8)) & 1u)
                << i;
    }
    return value;
}

void device_write_bits(
        unsigned char* base, std::uint64_t bit_offset,
        unsigned int nbits, std::uint64_t value) {
    if (nbits % 8 == 0) {
        auto* destination = base + bit_offset / 8;
        for (unsigned int i = 0; i < nbits / 8; ++i) {
            destination[i] = static_cast<unsigned char>(value >> (i * 8));
        }
        return;
    }

    for (unsigned int i = 0; i < nbits; ++i) {
        const std::uint64_t bit = bit_offset + i;
        auto* word = reinterpret_cast<std::uint32_t*>(
                base + (bit / 32) * sizeof(std::uint32_t));
        const std::uint32_t mask =
                static_cast<std::uint32_t>(1u << (bit % 32));
        sycl::atomic_ref<
                std::uint32_t, sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                atomic_word(*word);
        if ((value >> i) & 1u) {
            atomic_word.fetch_or(mask);
        } else {
            atomic_word.fetch_and(~mask);
        }
    }
}

void launch_view_transfer(
        sycl::queue& queue, const TensorView& view,
        const void* source, void* destination, bool from_host) {
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::size_t rows = dimensions[dimensions.size() - 2];
    const std::size_t columns = dimensions[dimensions.size() - 1];
    const std::size_t elements = rows * columns;
    const unsigned int bits = static_cast<unsigned int>(
            detail::leaf_bits(view.spec().data_type));
    const std::vector<std::size_t> planes = view_planes(view);

    for (std::size_t logical_plane = 0; logical_plane < planes.size();
         ++logical_plane) {
        const std::uint64_t plane = planes[logical_plane];
        const std::uint64_t logical_base =
                static_cast<std::uint64_t>(logical_plane * elements);
        if (from_host) {
            queue.submit([=](sycl::handler& handler) {
                handler.parallel_for(
                        sycl::range<1>(elements),
                        [=](sycl::id<1> index) {
                            const std::uint64_t local = index[0];
                            const std::uint64_t row = local / columns;
                            const std::uint64_t column = local % columns;
                            const std::uint64_t destination_bit =
                                    device_plane_slot(
                                            plane, row, column, rows,
                                            columns)
                                    * bits;
                            device_write_bits(
                                    static_cast<unsigned char*>(destination),
                                    destination_bit, bits,
                                    device_read_bits(
                                            static_cast<const unsigned char*>(
                                                    source),
                                            (logical_base + local) * bits,
                                            bits));
                        });
            });
        } else {
            queue.submit([=](sycl::handler& handler) {
                handler.parallel_for(
                        sycl::range<1>(elements),
                        [=](sycl::id<1> index) {
                            const std::uint64_t local = index[0];
                            const std::uint64_t row = local / columns;
                            const std::uint64_t column = local % columns;
                            const std::uint64_t source_bit =
                                    device_plane_slot(
                                            plane, row, column, rows,
                                            columns)
                                    * bits;
                            device_write_bits(
                                    static_cast<unsigned char*>(destination),
                                    (logical_base + local) * bits, bits,
                                    device_read_bits(
                                            static_cast<const unsigned char*>(
                                                    source),
                                            source_bit, bits));
                        });
            });
        }
    }
}

void synchronous_transfer(
        const sycl::context& context, const sycl::device& device,
        const TensorView& view, const void* storage,
        std::span<const std::byte> source,
        std::span<std::byte> destination, bool from_host) {
    sycl::queue queue(
            context, device,
            sycl::property_list{sycl::property::queue::in_order{}});
    const std::size_t logical_nbytes = view.spec().logical_nbytes();
    constexpr std::size_t kWordBytes = sizeof(std::uint32_t);
    if (logical_nbytes
            > std::numeric_limits<std::size_t>::max() - (kWordBytes - 1)) {
        throw std::overflow_error("SYCL transfer staging size overflows");
    }
    const std::size_t staging_nbytes =
            (logical_nbytes + (kWordBytes - 1)) / kWordBytes * kWordBytes;
    void* staging = sycl::malloc_host(staging_nbytes, context);
    if (staging == nullptr) {
        throw std::bad_alloc();
    }

    try {
        if (from_host) {
            std::memcpy(staging, source.data(), logical_nbytes);
            launch_view_transfer(
                    queue, view, staging, const_cast<void*>(storage), true);
            queue.wait_and_throw();
        } else {
            std::memset(staging, 0, logical_nbytes);
            launch_view_transfer(
                    queue, view, storage, staging, false);
            queue.wait_and_throw();
            std::memcpy(destination.data(), staging, logical_nbytes);
        }
    } catch (...) {
        try {
            queue.wait_and_throw();
        } catch (...) {
        }
        try {
            sycl::free(staging, context);
        } catch (...) {
        }
        throw;
    }

    sycl::free(staging, context);
}

class SyclQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        const TensorView* source;
        TensorView* destination;
        std::vector<PlanePair> pairs;
        bool no_op;
        std::shared_ptr<SyclFenceState> state;
        void* fence = state.get();
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
    };

    struct SequenceOutcome {
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
        std::shared_ptr<SyclFenceState> state;
    };

public:
    SyclQueue(
            const Device& device, const sycl::context& context,
            const sycl::device& native_device, SyclRegistryState& state)
            : device_(&device),
              state_(&state),
              registry_queue_id_(allocate_queue_id(*state_)),
              queue_(
                      context, native_device,
                      sycl::property_list{
                              sycl::property::queue::in_order{}}),
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
        std::vector<PlanePair> pairs;
        if (!no_op) {
            pairs = plane_pairs(source, destination);
        }

        const std::span<const std::size_t> dimensions =
                source.spec().shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t columns = dimensions[dimensions.size() - 1];
        const unsigned int bits = static_cast<unsigned int>(
                detail::leaf_bits(source.spec().data_type));

        return submit(
                [this, &source, &destination, no_op,
                 pairs = std::move(pairs), rows, columns, bits](
                        std::uint64_t sequence) mutable {
                    worker_.submit_copy(Task{
                            sequence, &source, &destination, std::move(pairs),
                            no_op});
                    (void)rows;
                    (void)columns;
                    (void)bits;
                });
    }

    oid add(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("SYCL", "add");
    }

    oid mul(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("SYCL", "mul");
    }

    oid silu(const TensorView&, TensorView&) override {
        throw unsupported("SYCL", "silu");
    }

    oid linear(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("SYCL", "linear");
    }

    oid rmsnorm(
            const TensorView&, TensorView&, const TensorView&, float,
            size_t) override {
        throw unsupported("SYCL", "rmsnorm");
    }

    oid sdpa(
            const TensorView&, const TensorView&, const TensorView&, size_t,
            size_t, size_t, TensorView&) override {
        throw unsupported("SYCL", "sdpa");
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
            detail::Fence fence =
                    [state = task.state]() noexcept -> detail::FenceResult {
                return state->result();
            };
            entries = register_copy_entries(
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
                    SequenceOutcome{
                            entries.source, entries.destination, task.state});
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
        try {
            const void* source_handle = task.source->native_handle();
            void* destination_handle = task.destination->native_handle();
            const std::size_t rows =
                    task.source->spec().shape.dimensions()[
                            task.source->spec().shape.rank() - 2];
            const std::size_t columns =
                    task.source->spec().shape.dimensions()[
                            task.source->spec().shape.rank() - 1];
            const unsigned int bits = static_cast<unsigned int>(
                    detail::leaf_bits(task.source->spec().data_type));

            for (std::size_t index = 0; index < task.pairs.size(); ++index) {
                if (index == 0
                        && consume_submission_fault(
                                SubmissionFault::first_submit)) {
                    throw std::runtime_error(
                            "injected SYCL first-submit failure");
                }
                if (index == 1
                        && consume_submission_fault(
                                SubmissionFault::second_submit)) {
                    throw std::runtime_error(
                            "injected SYCL second-submit failure");
                }
                const PlanePair pair = task.pairs[index];
                sycl::event event = queue_.submit(
                        [=](sycl::handler& handler) {
                            handler.parallel_for(
                                    sycl::range<1>(rows * columns),
                                    [=](sycl::id<1> item) {
                                        const std::uint64_t local = item[0];
                                        const std::uint64_t row =
                                                local / columns;
                                        const std::uint64_t column =
                                                local % columns;
                                        const std::uint64_t source_bit =
                                                device_plane_slot(
                                                        pair.source, row,
                                                        column, rows, columns)
                                                * bits;
                                        const std::uint64_t destination_bit =
                                                device_plane_slot(
                                                        pair.destination, row,
                                                        column, rows, columns)
                                                * bits;
                                        device_write_bits(
                                                static_cast<unsigned char*>(
                                                        destination_handle),
                                                destination_bit, bits,
                                                device_read_bits(
                                                        static_cast<const unsigned char*>(
                                                                source_handle),
                                                        source_bit, bits));
                                    });
                        });
                task.state->set_event(std::move(event));
                submitted_any = true;
            }
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (!submitted_any) {
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
        SequenceOutcome outcome;
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

        std::exception_ptr combined_failure = std::move(callback_failure);
        if (has_outcome) {
            const detail::FenceResult result = outcome.state->result();
            if (!combined_failure) {
                combined_failure = result.failure;
            }
            const std::array<detail::EntryId, 2> ids{
                    outcome.source_entry_id,
                    outcome.destination_entry_id};
            if (combined_failure || !result.succeeded) {
                state_->registry.invalidate_entries(ids);
            } else {
                (void)state_->registry.try_release_entry(ids[0]);
                (void)state_->registry.try_release_entry(ids[1]);
            }
        }
        complete(sequence, std::move(combined_failure));
    }

    const Device* device_;
    SyclRegistryState* state_;
    detail::QueueId registry_queue_id_;
    sycl::queue queue_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, SequenceOutcome> outcomes_;
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
        const sycl::context& context, const sycl::device& device,
        const TensorView& destination, void* storage,
        std::span<const std::byte> source) {
    synchronous_transfer(
            context, device, destination, storage, source, {}, true);
}

void region_to_host(
        const sycl::context& context, const sycl::device& device,
        const TensorView& source, const void* storage,
        std::span<std::byte> destination) {
    synchronous_transfer(
            context, device, source, storage, {}, destination, false);
}

std::unique_ptr<DeviceOps> make_queue(
        const Device& device, const sycl::context& context,
        const sycl::device& native_device, SyclRegistryState& registry_state) {
    return std::make_unique<SyclQueue>(
            device, context, native_device, registry_state);
}

}  // namespace iom::sycl_detail
