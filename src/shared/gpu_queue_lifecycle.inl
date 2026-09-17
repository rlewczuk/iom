namespace iom::detail {

template <typename Policy>
[[nodiscard]] detail::FenceResult GpuQueue<Policy>::fence_invoke(
        const detail::Fence& fence) noexcept {
    const auto& capture =
            *std::launder(reinterpret_cast<const typename GpuQueue<Policy>::GpuFenceCapture*>(
                    fence.storage));
    if (capture.completion == nullptr) {
        return detail::FenceResult::pending();
    }
    return capture.completion->result();
}

template <typename Policy>
[[nodiscard]] detail::Fence GpuQueue<Policy>::build_fence(
        std::shared_ptr<typename GpuQueue<Policy>::CompletionState>
                completion) noexcept {
    detail::Fence fence;
    ::new (fence.storage) typename GpuQueue<Policy>::GpuFenceCapture{
            std::move(completion)};
    fence.invoke = &GpuQueue<Policy>::fence_invoke;
    fence.copy_construct = &detail::FenceCaptureOps<
            typename GpuQueue<Policy>::GpuFenceCapture>::copy_construct;
    fence.move_construct = &detail::FenceCaptureOps<
            typename GpuQueue<Policy>::GpuFenceCapture>::move_construct;
    fence.destroy = &detail::FenceCaptureOps<
            typename GpuQueue<Policy>::GpuFenceCapture>::destroy;
    return fence;
}

template <typename Policy>
[[nodiscard]] typename detail::StagedWorker<
        typename GpuQueue<Policy>::Task>::Callbacks
GpuQueue<Policy>::make_worker_callbacks(
        GpuQueue<Policy>* self,
        std::shared_ptr<typename GpuQueue<Policy>::EventRing> state) {
    typename detail::StagedWorker<
            typename GpuQueue<Policy>::Task>::Callbacks callbacks{
            [self](typename GpuQueue<Policy>::Task& task) {
                self->execute(task);
            },
            [state](void* fence) {
                state->on_worker_complete(
                        *static_cast<typename GpuQueue<Policy>::EventRing::
                                Submission*>(fence));
            },
            [state](void* fence) {
                state->on_worker_destroy(
                        *static_cast<typename GpuQueue<Policy>::EventRing::
                                Submission*>(fence));
            },
            [self](
                    std::uint64_t sequence,
                    std::exception_ptr failure) {
                self->complete_task(sequence, std::move(failure));
            }};
    return callbacks;
}

template <typename Policy>
GpuQueue<Policy>::GpuQueue(
        const Device& device,
        QueueResourceProvider& resource_provider,
        typename Policy::context_type context,
        detail::RegistryState& registry_state)
        : DeviceOps(device),
          device_(&device),
          registry_state_(&registry_state),
          registry_queue_id_(
                  detail::allocate_queue_id(*registry_state_)),
          resource_provider_(&resource_provider),
          context_(context),
          metadata_pool_(std::make_shared<MetadataSlotPool>(
                  resource_provider.reserve_queue_resources())),
          state_(std::make_shared<EventRing>(
                  context_,
                  *metadata_pool_,
                  metadata_pool_->slot_count())),
          worker_(
                  make_worker_callbacks(this, state_),
                  detail::StagedWorker<Task>::PublishPolicy::Splice) {
    Policy::activate(context_);
    try {
        stream_ = Policy::create_queue_stream();
        worker_.start();
    } catch (...) {
        Policy::destroy_queue_stream_noexcept(stream_);
        stream_ = Policy::null_stream();
        throw;
    }
}

template <typename Policy>
GpuQueue<Policy>::~GpuQueue() {
    // Invalidate owner fences before draining the worker. Accepted work
    // that outlives this queue must quarantine its tensor payloads when
    // owners are destroyed after queue teardown; the device retains the
    // backing until a covering proof at its boundary.
    registry_state_->registry.invalidate_entries_for_queue(
            registry_queue_id_);
    // Close common admission before worker drain so no new operation
    // can race teardown; parked prepared leases are finished by the
    // second close after executing work retires.
    close_and_drain();
    worker_.shutdown_and_drain();
    // The first close leaves executing nodes to the backend drain. Once
    // the worker has retired them, finish the parked tail in FIFO order.
    close_and_drain();
    bool drained = false;
    try {
        Policy::activate(context_);
        drained = Policy::synchronize_stream_noexcept(stream_);
    } catch (...) {
        drained = false;
    }
    state_->on_queue_drain(drained);
    if (drained || !state_->has_unproven_completion()) {
        // Every lease is proved: release the native stream and return
        // the partition, queue-count reservation, and completion
        // resources for reuse.
        Policy::destroy_queue_stream_noexcept(stream_);
        stream_ = Policy::null_stream();
        state_.reset();
        metadata_pool_.reset();
        return;
    }
    // Unknown completion: quarantine the entire unresolved lease at the
    // Device boundary. The partition and queue-count reservation stay
    // retained with the completion resources, host mirrors, optional status
    // cells, and the queue's own covering-proof handle until this queue's
    // own drain is proved; a drain of another queue is never sufficient.
    // Capture order matters: the pool must be destroyed after the ring that
    // borrows it, so it is declared first (lambda captures are
    // destroyed in reverse declaration order).
    const typename Policy::context_type retained_context = context_;
    const typename Policy::stream_type retained_stream = stream_;
    stream_ = Policy::null_stream();
    std::shared_ptr<MetadataSlotPool> retained_pool =
            std::move(metadata_pool_);
    std::shared_ptr<EventRing> retained_state = std::move(state_);
    resource_provider_->retain_unknown_lease(
            [retained_context, retained_stream, retained_pool,
             retained_state]() mutable -> bool {
                try {
                    Policy::activate(retained_context);
                } catch (...) {
                    return false;
                }
                if (!Policy::synchronize_stream_noexcept(
                            retained_stream)) {
                    return false;
                }
                // Covering proof: release protected slots and drop the
                // lease. Dropping the captures destroys the completion
                // resources and returns the partition and credit.
                retained_state->on_queue_drain(true);
                Policy::destroy_queue_stream_noexcept(retained_stream);
                retained_state.reset();
                retained_pool.reset();
                return true;
            },
            [retained_context, retained_stream, retained_pool,
             retained_state]() mutable {
                // Best-effort teardown while the native context is still
                // valid; unproven leases keep their outcome untouched.
                try {
                    Policy::activate(retained_context);
                } catch (...) {
                }
                Policy::destroy_queue_stream_noexcept(retained_stream);
                retained_state.reset();
                retained_pool.reset();
            });
}

}  // namespace iom::detail
