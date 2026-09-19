#include "queue_internal.hpp"

#include <cstdint>
#include <cstddef>
#include <exception>
#include <new>
#include <stdexcept>
#include <utility>

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
template <typename Request>
[[nodiscard]] detail::CopyMetadataLayout
        copy_metadata_layout_snapshot(const Request& request) {
    const auto checked_add = [](std::size_t lhs, std::size_t rhs,
                                const char* what) {
        if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
            throw std::overflow_error(what);
        }
        return lhs + rhs;
    };
    const auto checked_mul = [](std::size_t lhs, std::size_t rhs,
                                const char* what) {
        if (lhs != 0
                && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
            throw std::overflow_error(what);
        }
        return lhs * rhs;
    };
    const auto padded = [&](std::size_t value) {
        return checked_mul(
                checked_add(
                        value, TensorSpec::TILE - 1,
                        "SYCL copy metadata padding overflows")
                        / TensorSpec::TILE,
                TensorSpec::TILE,
                "SYCL copy metadata padding overflows");
    };
    const auto dimensions = request.source.spec.shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = checked_mul(
                plane_count, dimensions[axis],
                "SYCL copy metadata plane count overflows");
    }
    const std::size_t plane_bits = checked_mul(
            checked_mul(
                    padded(dimensions[leading_rank]),
                    padded(dimensions[leading_rank + 1]),
                    "SYCL copy metadata plane size overflows"),
            detail::leaf_bits(request.source.spec.data_type),
            "SYCL copy metadata plane bits overflows");
    const std::size_t words_per_plane = checked_add(
            plane_bits, 31,
            "SYCL copy metadata word count overflows")
            / 32;
    const std::size_t total_words = checked_mul(
            plane_count, words_per_plane,
            "SYCL copy metadata total words overflows");
    const std::size_t array_bytes = checked_mul(
            checked_mul(
                    leading_rank, 3,
                    "SYCL copy metadata array count overflows"),
            sizeof(std::uint64_t),
            "SYCL copy metadata array bytes overflows");
    return {
            checked_add(
                    sizeof(detail::CopyMetadataHeader), array_bytes,
                    "SYCL copy metadata size overflows"),
            total_words};
}

template <typename Request>
void write_copy_metadata_snapshot(
        std::byte* storage, const Request& request) {
    const auto dimensions = request.source.spec.shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        if (dimensions[axis]
                > std::numeric_limits<std::size_t>::max() / plane_count) {
            throw std::overflow_error(
                    "SYCL copy metadata plane count overflows");
        }
        plane_count *= dimensions[axis];
    }
    auto* header =
            reinterpret_cast<detail::CopyMetadataHeader*>(storage);
    header->source_plane_offset =
            static_cast<std::uint64_t>(request.source.plane_offset);
    header->destination_plane_offset =
            static_cast<std::uint64_t>(request.destination.plane_offset);
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

}  // namespace
SyclQueue::SyclQueue(
        const Device& device,
        detail::QueueResourceProvider& resource_provider,
        const sycl::context& context, const sycl::device& native_device,
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
          fp64_supported_(native_device.has(sycl::aspect::fp64)),
          bf16_linear_supported_(
                  bf16_linear_device_capable(native_device)),
          queue_(make_queue_with_fault_check(context, native_device)),
          worker_(
                  detail::StagedWorker<Task>::Callbacks{
                          [this](Task& task) {
                              execute(task);
                          },
                          [](void* fence) noexcept {
                              if (fence != nullptr) {
                                  (void)static_cast<SyclFenceState*>(fence)
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
                              complete_task(sequence, std::move(failure));
                          }},
                  detail::StagedWorker<Task>::PublishPolicy::Splice) {
    worker_.start();
}
SyclQueue::~SyclQueue() {
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
oid SyclQueue::copy_impl(
        const TensorView& source, TensorView& destination) {
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
oid SyclQueue::binary_impl(const BinaryRequest& request) {
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
void SyclQueue::execute(Task& task) {
    if (task.embedding_request.has_value()) {
        execute_embedding(task);
        return;
    }
    if (task.binary_request.has_value()) {
        execute_binary(task);
        return;
    }
    if (task.rmsnorm_request.has_value()) {
        execute_rmsnorm(task);
        return;
    }
    if (task.linear_request.has_value()) {
        execute_linear(task);
        return;
    }
    if (task.rope_request.has_value()) {
        execute_rope(task);
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

        const auto completion_slot = completion_pool_->try_acquire();
        if (!completion_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        task.state->set_completion_slot(
                *completion_pool_, *completion_slot);
        // One fixed 512-byte slot from this queue's partition carries
        // the immutable pointer-copy descriptor; no growth, replacement,
        // or native allocation ever happens here.
        const auto metadata_slot = metadata_pool_->try_acquire();
        if (!metadata_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        const std::size_t metadata_index = *metadata_slot;
        task.state->set_metadata_slot(*metadata_pool_, metadata_index);
        const detail::CopyMetadataLayout layout =
                copy_metadata_layout_snapshot(*task.copy_request);
        write_copy_metadata_snapshot(
                metadata_pool_->host_data(metadata_index),
                *task.copy_request);
        native_attempted = true;
        queue_.memcpy(
                metadata_pool_->device_data(metadata_index),
                metadata_pool_->host_data(metadata_index), layout.bytes);
        metadata_enqueued = true;

        const auto* source_handle = static_cast<const unsigned char*>(
                task.copy_request->source.native_handle);
        auto* destination_handle = static_cast<unsigned char*>(
                task.copy_request->destination.native_handle);
        const auto* metadata = static_cast<
                const detail::CopyMetadataHeader*>(
                metadata_pool_->device_data(metadata_index));
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

void SyclQueue::complete_task(
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
                fence_result.failure ? fence_result.failure : callback_failure;
        const bool completion_proven =
                outcome.state != nullptr
                && outcome.state->completion_proven();
        const bool fence_succeeded = outcome.is_embedding
                ? completion_proven
                : fence_result.succeeded && !fence_result.failure;
        const bool failed = static_cast<bool>(combined_failure);
        if (outcome.is_embedding) {
            // Embedding observes native completion proof independently of a
            // retained semantic (OOV) failure: the kernel wrote the host USM
            // status cell through event-ordered `queue::memcpy`. Copy,
            // binary, and RMSNorm keep the strict semantic-failure-overrides
            // -proof rule below.
            const bool release_failed = !completion_proven;
            (void)detail::release_or_invalidate_binary_entries(
                    state_->registry, *outcome.binary_entries,
                    release_failed, fence_succeeded);
            detail::complete_workspace_lease(
                    *state_, outcome.workspace_lease, completion_proven);
        } else if (outcome.binary_entries.has_value()) {
            (void)detail::release_or_invalidate_binary_entries(
                    state_->registry, *outcome.binary_entries, failed,
                    fence_succeeded);
            detail::complete_workspace_lease(
                    *state_, outcome.workspace_lease, completion_proven);
        } else if (outcome.rmsnorm_entries.has_value()) {
            // RMS normalization registers the same read/read-deduplicated
            // owner set but consumes no `RawWorkspace`, so only the owner
            // entries are released or invalidated.
            (void)detail::release_or_invalidate_binary_entries(
                    state_->registry, *outcome.rmsnorm_entries, failed,
                    fence_succeeded);
        } else if (outcome.linear_entries.has_value()) {
            // Linear projections register the same read/read-deduplicated
            // owner set as RMS normalization. The twenty scalar leaves report
            // the `{0, 1}` requirement and carry no lease, while the native
            // `BF16` specialization leases its product scratch through proven
            // completion — an empty lease matches no record and retires
            // nothing.
            (void)detail::release_or_invalidate_binary_entries(
                    state_->registry, *outcome.linear_entries, failed,
                    fence_succeeded);
            detail::complete_workspace_lease(
                    *state_, outcome.workspace_lease, completion_proven);
        } else if (outcome.rope_entries.has_value()) {
            // RoPE uses the fixed zero-workspace requirement, so only its
            // deduplicated input/output owner registrations are released.
            (void)detail::release_or_invalidate_binary_entries(
                    state_->registry, *outcome.rope_entries, failed,
                    fence_succeeded);
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
sycl::queue SyclQueue::make_queue_with_fault_check(
        const sycl::context& context, const sycl::device& native_device) {
    if (consume_submission_fault(SubmissionFault::queue_stream_create)) {
        throw std::runtime_error(
                "injected SYCL queue creation failure");
    }
    return sycl::queue(
            context, native_device,
            sycl::property_list{sycl::property::queue::in_order{}});
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

}  // namespace iom::sycl_detail
