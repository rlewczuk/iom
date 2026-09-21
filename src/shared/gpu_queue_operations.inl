namespace iom::detail {

template <typename Policy>
iom::oid GpuQueue<Policy>::copy_impl(
        const TensorView& source, TensorView& destination) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    // These injected failures model host registration/preparation faults,
    // so they are consumed before common acceptance reserves a sequence.
    if (Policy::consume_copy_registration_fault()
            || Policy::consume_copy_outcome_insertion_fault()) {
        throw std::bad_alloc();
    }
    auto completion = std::make_shared<CompletionState>();
    const detail::Fence fence = build_fence(completion);
    return submit_copy(
            source, destination, *registry_state_, registry_queue_id_,
            fence,
            [this, completion](
                    std::uint64_t sequence,
                    const CopyRequest& captured,
                    detail::EntryRegistration entries) {
                auto submission = state_->try_acquire();
                if (submission == nullptr) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                completion->bind(submission);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] =
                                outcomes_.try_emplace(sequence);
                        if (!inserted) {
                            throw std::logic_error(
                                    std::string("duplicate ")
                                    + Policy::backend_label()
                                    + " outstanding-work sequence");
                        }
                        it->second.common.source_entry_id =
                                entries.source;
                        it->second.common.destination_entry_id =
                                entries.destination;
                        it->second.completion = completion;
                    }
                    Task task;
                    task.sequence = sequence;
                    task.copy_request.emplace(captured);
                    task.entries = entries;
                    task.submission = submission.get();
                    task.completion = completion;
                    task.fence = task.submission;
                    worker_.submit_copy(std::move(task));
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(sequence);
                    }
                    completion->clear();
                    throw;
                }
            });
}

template <typename Policy>
oid GpuQueue<Policy>::binary_impl(const BinaryRequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    // Validate descriptor arithmetic before common acceptance. The
    // metadata slot itself is acquired only by the dispatching head.
    (void)detail::make_binary_metadata(request);
    auto completion = std::make_shared<CompletionState>();
    const detail::Fence fence = build_fence(completion);
    return submit_binary(
            request, *registry_state_, registry_queue_id_, fence,
            [this, completion](
                    std::uint64_t sequence,
                    const BinaryRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                auto submission = state_->try_acquire();
                if (submission == nullptr) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                const auto metadata_slot = metadata_pool_->try_acquire();
                if (!metadata_slot.has_value()) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                MetadataLease metadata_lease{
                        metadata_pool_.get(), *metadata_slot};
                completion->bind(submission);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] =
                                outcomes_.try_emplace(sequence);
                        if (!inserted) {
                            throw std::logic_error(
                                    "duplicate GPU binary sequence");
                        }
                        it->second.binary_entries = entries;
                        it->second.workspace_lease =
                                captured.workspace_lease;
                        it->second.completion = completion;
                        it->second.is_binary = true;
                    }
                    Task task;
                    task.sequence = sequence;
                    task.is_binary = true;
                    task.binary_request.emplace(captured);
                    task.binary_entries = entries;
                    task.submission = submission.get();
                    task.completion = completion;
                    task.fence = task.submission;
                    task.metadata_lease = std::move(metadata_lease);
                    worker_.submit_copy(std::move(task));
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(sequence);
                    }
                    completion->clear();
                    throw;
                }
            });
}
template <typename Policy>
WorkspaceRequirements
GpuQueue<Policy>::cache_append_workspace_requirements(
        const CacheAppendRequest&) {
    if constexpr (requires(
                          typename Policy::stream_type stream,
                          const CacheAppendRequest& request) {
                      Policy::launch_cache_append(stream, request);
                  }) {
        // CUDA and ROCm direct append paths use no common scratch. A policy
        // that has not yet added its native launcher remains explicitly
        // unsupported rather than acquiring resources in the shared queue.
        return {0, 1};
    } else {
        throw detail::UnsupportedOperation();
    }
}

template <typename Policy>
oid GpuQueue<Policy>::cache_append_impl(
        const CacheAppendRequest& request) {
    if constexpr (requires(
                          typename Policy::stream_type stream,
                          const CacheAppendRequest& candidate) {
                      Policy::launch_cache_append(stream, candidate);
                  }) {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        auto completion = std::make_shared<CompletionState>();
        const detail::Fence fence = build_fence(completion);
        return submit_cache_append(
                request, *registry_state_, registry_queue_id_, fence,
                [this, completion](
                        std::uint64_t sequence,
                        const CacheAppendRequest& captured,
                        detail::BinaryEntryRegistration entries) {
                    auto submission = state_->try_acquire();
                    if (submission == nullptr) {
                        throw detail::AdmissionResourceUnavailable{};
                    }
                    completion->bind(submission);
                    try {
                        {
                            std::lock_guard<std::mutex> lock(outcome_mutex_);
                            const auto [it, inserted] =
                                    outcomes_.try_emplace(sequence);
                            if (!inserted) {
                                throw std::logic_error(
                                        "duplicate GPU cache append sequence");
                            }
                            it->second.cache_append_entries = entries;
                            it->second.workspace_lease =
                                    captured.workspace_lease;
                            it->second.completion = completion;
                            it->second.is_cache_append = true;
                        }
                        Task task;
                        task.sequence = sequence;
                        task.is_cache_append = true;
                        task.cache_append_request.emplace(captured);
                        task.cache_append_entries = entries;
                        task.submission = submission.get();
                        task.completion = completion;
                        task.fence = task.submission;
                        worker_.submit_copy(std::move(task));
                    } catch (...) {
                        {
                            std::lock_guard<std::mutex> lock(outcome_mutex_);
                            outcomes_.erase(sequence);
                        }
                        completion->clear();
                        throw;
                    }
                });
    } else {
        static_cast<void>(request);
        throw detail::UnsupportedOperation();
    }
}

template <typename Policy>
oid GpuQueue<Policy>::rmsnorm_impl(const RmsnormRequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    // Descriptor arithmetic is validated before common acceptance. The
    // metadata slot itself is acquired only by the dispatching head, exactly
    // as the binary branch does; no shape, device, dtype, quantization,
    // alias, workspace, or epsilon rule is re-checked here.
    (void)detail::make_rmsnorm_metadata(request);
    auto completion = std::make_shared<CompletionState>();
    const detail::Fence fence = build_fence(completion);
    return submit_rmsnorm(
            request, *registry_state_, registry_queue_id_, fence,
            [this, completion](
                    std::uint64_t sequence,
                    const RmsnormRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                auto submission = state_->try_acquire();
                if (submission == nullptr) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                const auto metadata_slot = metadata_pool_->try_acquire();
                if (!metadata_slot.has_value()) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                MetadataLease metadata_lease{
                        metadata_pool_.get(), *metadata_slot};
                completion->bind(submission);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] =
                                outcomes_.try_emplace(sequence);
                        if (!inserted) {
                            throw std::logic_error(
                                    std::string("duplicate ")
                                    + Policy::backend_label()
                                    + " outstanding-work sequence");
                        }
                        it->second.rmsnorm_entries = entries;
                        it->second.completion = completion;
                        it->second.is_rmsnorm = true;
                    }
                    Task task;
                    task.sequence = sequence;
                    task.is_rmsnorm = true;
                    task.rmsnorm_request.emplace(captured);
                    task.rmsnorm_entries = entries;
                    task.submission = submission.get();
                    task.completion = completion;
                    task.fence = task.submission;
                    task.metadata_lease = std::move(metadata_lease);
                    worker_.submit_copy(std::move(task));
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(sequence);
                    }
                    completion->clear();
                    throw;
                }
            });
}

template <typename Policy>
WorkspaceRequirements GpuQueue<Policy>::silu_workspace_requirements_impl(
        const SiLURequest&) {
    // Capability is immutable per queue policy. The common facade has
    // already completed structural, device, alias, dtype, quantization, and
    // zero-workspace admission before reaching this pure hook. An unported
    // policy rejects before workspace inspection, owner registration,
    // sequence consumption, or queue-resource use.
    if (!Policy::silu_supported()) {
        throw detail::UnsupportedOperation();
    }
    return WorkspaceRequirements{0, 1};
}

template <typename Policy>
WorkspaceRequirements GpuQueue<Policy>::rope_workspace_requirements(
        const RopeRequest&) {
    // Capability is immutable per queue policy.  The common facade has
    // already completed structural/device/alias/parameter admission before
    // reaching this pure hook, and an unported policy must reject before
    // workspace inspection, registration, sequence consumption, or queue
    // resource use.
    if (!Policy::rope_supported()) {
        throw detail::UnsupportedOperation();
    }
    return WorkspaceRequirements{0, 1};
}

template <typename Policy>
oid GpuQueue<Policy>::silu_impl(const SiLURequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    // Descriptor representation is checked before common acceptance's queue
    // callback reserves a sequence. Shape, device, alias, dtype,
    // quantization, and workspace admission remain solely in the common
    // SiLU contract.
    if (!Policy::silu_supported()) {
        throw detail::UnsupportedOperation();
    }
    (void)detail::make_silu_metadata(request);
    auto completion = std::make_shared<CompletionState>();
    const detail::Fence fence = build_fence(completion);
    return submit_silu(
            request, *registry_state_, registry_queue_id_, fence,
            [this, completion](
                    std::uint64_t sequence,
                    const SiLURequest& captured,
                    detail::BinaryEntryRegistration entries) {
                auto submission = state_->try_acquire();
                if (submission == nullptr) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                const auto metadata_slot = metadata_pool_->try_acquire();
                if (!metadata_slot.has_value()) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                MetadataLease metadata_lease{
                        metadata_pool_.get(), *metadata_slot};
                completion->bind(submission);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] =
                                outcomes_.try_emplace(sequence);
                        if (!inserted) {
                            throw std::logic_error(
                                    std::string("duplicate ")
                                    + Policy::backend_label()
                                    + " outstanding-work sequence");
                        }
                        it->second.silu_entries = entries;
                        it->second.completion = completion;
                        it->second.is_silu = true;
                    }
                    Task task;
                    task.sequence = sequence;
                    task.is_silu = true;
                    task.silu_request.emplace(captured);
                    task.silu_entries = entries;
                    task.submission = submission.get();
                    task.completion = completion;
                    task.fence = task.submission;
                    task.metadata_lease = std::move(metadata_lease);
                    worker_.submit_copy(std::move(task));
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(sequence);
                    }
                    completion->clear();
                    throw;
                }
            });
}

template <typename Policy>
oid GpuQueue<Policy>::rope_impl(const RopeRequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    // Descriptor arithmetic is checked before common acceptance's queue
    // callback reserves a sequence.  Shape, device, alias, dtype,
    // quantization, position, theta, and workspace admission remain solely
    // in the common Rope contract.
    if (!Policy::rope_supported()) {
        throw detail::UnsupportedOperation();
    }
    (void)detail::make_rope_metadata(request);
    auto completion = std::make_shared<CompletionState>();
    const detail::Fence fence = build_fence(completion);
    return submit_rope(
            request, *registry_state_, registry_queue_id_, fence,
            [this, completion](
                    std::uint64_t sequence,
                    const RopeRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                auto submission = state_->try_acquire();
                if (submission == nullptr) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                const auto metadata_slot = metadata_pool_->try_acquire();
                if (!metadata_slot.has_value()) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                MetadataLease metadata_lease{
                        metadata_pool_.get(), *metadata_slot};
                completion->bind(submission);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] =
                                outcomes_.try_emplace(sequence);
                        if (!inserted) {
                            throw std::logic_error(
                                    std::string("duplicate ")
                                    + Policy::backend_label()
                                    + " outstanding-work sequence");
                        }
                        it->second.rope_entries = entries;
                        it->second.workspace_lease =
                                captured.workspace_lease;
                        it->second.completion = completion;
                        it->second.is_rope = true;
                    }
                    Task task;
                    task.sequence = sequence;
                    task.is_rope = true;
                    task.rope_request.emplace(captured);
                    task.rope_entries = entries;
                    task.submission = submission.get();
                    task.completion = completion;
                    task.fence = task.submission;
                    task.metadata_lease = std::move(metadata_lease);
                    worker_.submit_copy(std::move(task));
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(sequence);
                    }
                    completion->clear();
                    throw;
                }
            });
}


template <typename Policy>
bool GpuQueue<Policy>::rmsnorm_supported(DataType data_type) const {
    // The policy owns the immutable RMSNorm capability for the applicable
    // leaves; this port adds no leaf of its own and stays conservative until
    // the backend wrapper launches the shared tiled operation.
    static_cast<void>(data_type);
    return Policy::rmsnorm_supported();
}

template <typename Policy>
WorkspaceRequirements GpuQueue<Policy>::embedding_workspace_requirements_impl(
        const TensorView&, const TensorView&, const TensorView&) {
    return WorkspaceRequirements{32, 32};
}

template <typename Policy>
oid GpuQueue<Policy>::embedding_impl(const EmbeddingRequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    const std::size_t rank = request.indices.spec.shape.rank();
    (void)detail::embedding_metadata_storage_bytes(rank);
    auto completion = std::make_shared<CompletionState>();
    const detail::Fence fence = build_fence(completion);
    return submit_embedding(
            request, *registry_state_, registry_queue_id_, fence,
            [this, completion](
                    std::uint64_t sequence,
                    const EmbeddingRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                auto submission = state_->try_acquire();
                if (submission == nullptr) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                const auto metadata_slot = metadata_pool_->try_acquire();
                if (!metadata_slot.has_value()) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                MetadataLease metadata_lease{
                        metadata_pool_.get(), *metadata_slot};
                completion->bind(submission);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] =
                                outcomes_.try_emplace(sequence);
                        if (!inserted) {
                            throw std::logic_error(
                                    "duplicate GPU embedding sequence");
                        }
                        it->second.binary_entries = entries;
                        it->second.workspace_lease =
                                captured.workspace_lease;
                        it->second.completion = completion;
                        it->second.is_embedding = true;
                    }
                    Task task;
                    task.sequence = sequence;
                    task.is_embedding = true;
                    task.embedding_request.emplace(captured);
                    task.binary_entries = entries;
                    task.status_slot = *metadata_slot;
                    task.submission = submission.get();
                    task.completion = completion;
                    task.fence = task.submission;
                    task.metadata_lease = std::move(metadata_lease);
                    worker_.submit_copy(std::move(task));
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(sequence);
                    }
                    completion->clear();
                    throw;
                }
            });
}

template <typename Policy>
WorkspaceRequirements GpuQueue<Policy>::linear_workspace_requirements_impl(
        const TensorView& x, const TensorView& w, const TensorView&,
        std::size_t, std::size_t R, LinearOutputLayout, std::size_t,
        std::size_t) {
    const DataType leaf = x.spec().data_type;
    // The pure capability and scratch decision. A leaf this policy does not
    // carry is the established capability rejection and never inspects
    // scratch. Nothing here allocates, registers, leases, snapshots,
    // submits, or examines queue state.
    if (!Policy::linear_supported(leaf)) {
        throw detail::UnsupportedOperation();
    }
    // Optional native-BF16 seam. A policy that declares the two native
    // members owns its own packed-scratch formula and its own immutable
    // device gate; a policy without them keeps the frozen `{0, 1}`
    // zero-scratch decision for every leaf it carries, so this seam adds no
    // second capability convention to a scalar-only port. A device without
    // the proven facility is a capability rejection, never a queued failure.
    if constexpr (requires {
                      Policy::linear_native_bf16_available(context_);
                      Policy::linear_native_bf16_requirements(x, w, R);
                  }) {
        if (leaf == DataType::BF16) {
            if (!Policy::linear_native_bf16_available(context_)) {
                throw detail::UnsupportedOperation();
            }
            return Policy::linear_native_bf16_requirements(x, w, R);
        }
    }
    return WorkspaceRequirements{0, 1};
}

template <typename Policy>
oid GpuQueue<Policy>::linear_impl(const LinearRequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    // The capability decision is repeated here for defense in depth: the
    // common facade consults the pure query first, and this branch must never
    // dispatch a leaf the policy's scalar or native path does not carry.
    // Descriptor arithmetic is validated before common acceptance, and the
    // metadata slot itself is acquired only by the dispatching head.
    const DataType leaf = request.x.spec.data_type;
    if (!Policy::linear_supported(leaf)) {
        throw detail::UnsupportedOperation();
    }
    if constexpr (requires { Policy::linear_native_bf16_available(context_); }) {
        if (leaf == DataType::BF16
                && !Policy::linear_native_bf16_available(context_)) {
            throw detail::UnsupportedOperation();
        }
    }
    // The admitted requirement decides the workspace rule: a zero
    // requirement admits only the empty `RawWorkspaceView{}`, while a
    // positive one requires the caller's validated range, which the common
    // facade checked for liveness, exact-device identity, size, and
    // byte-range disjointness through common validation before dispatch.
    if (request.workspace_requirements.bytes == 0) {
        if (!request.workspace.empty()) {
            throw std::invalid_argument(
                    "LINEAR consumes no raw workspace, so the supplied "
                    "workspace view must be empty");
        }
    } else if (request.workspace.empty()) {
        throw std::invalid_argument(
                "LINEAR positive scratch requirement needs a non-empty "
                "workspace view");
    }
    (void)detail::make_linear_metadata(request);
    auto completion = std::make_shared<CompletionState>();
    const detail::Fence fence = build_fence(completion);
    return submit_linear(
            request, *registry_state_, registry_queue_id_, fence,
            [this, completion](
                    std::uint64_t sequence,
                    const LinearRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                auto submission = state_->try_acquire();
                if (submission == nullptr) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                const auto metadata_slot = metadata_pool_->try_acquire();
                if (!metadata_slot.has_value()) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                MetadataLease metadata_lease{
                        metadata_pool_.get(), *metadata_slot};
                completion->bind(submission);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] =
                                outcomes_.try_emplace(sequence);
                        if (!inserted) {
                            throw std::logic_error(
                                    std::string("duplicate ")
                                    + Policy::backend_label()
                                    + " outstanding-work sequence");
                        }
                        it->second.binary_entries = entries;
                        it->second.workspace_lease =
                                captured.workspace_lease;
                        it->second.completion = completion;
                        it->second.is_linear = true;
                    }
                    Task task;
                    task.sequence = sequence;
                    task.is_linear = true;
                    task.linear_request.emplace(captured);
                    task.binary_entries = entries;
                    task.submission = submission.get();
                    task.completion = completion;
                    task.fence = task.submission;
                    task.metadata_lease = std::move(metadata_lease);
                    worker_.submit_copy(std::move(task));
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(sequence);
                    }
                    completion->clear();
                    throw;
                }
            });
}

template <typename Policy>
WorkspaceRequirements GpuQueue<Policy>::sdpa_workspace_requirements_impl(
        const SdpaRequest& request) {
    if constexpr (requires { Policy::sdpa_supported(); }) {
        if (!Policy::sdpa_supported()) {
            throw detail::UnsupportedOperation();
        }
        // The exact scratch geometry is the backend policy's own fact: the
        // shared queue never computes or hardcodes a backend segment layout,
        // so a backend whose assessed range differs (for example a
        // conservative multi-segment pack/score/probability range) supplies
        // it through this one pure hook. It is consulted only after the
        // capability gate and before any workspace validation.
        return Policy::sdpa_workspace_requirements(request);
    } else {
        throw detail::UnsupportedOperation();
    }
}

template <typename Policy>
oid GpuQueue<Policy>::sdpa_impl(const SdpaRequest& request) {
    if constexpr (requires { Policy::sdpa_supported(); }) {
        if (!Policy::sdpa_supported()) {
            throw detail::UnsupportedOperation();
        }
    } else {
        throw detail::UnsupportedOperation();
    }
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    auto completion = std::make_shared<CompletionState>();
    const detail::Fence fence = build_fence(completion);
    return submit_sdpa(
            request, *registry_state_, registry_queue_id_, fence,
            [this, completion](
                    std::uint64_t sequence,
                    const SdpaRequest& captured,
                    detail::SdpaEntryRegistration entries) {
                auto submission = state_->try_acquire();
                if (submission == nullptr) {
                    throw detail::AdmissionResourceUnavailable{};
                }
                completion->bind(submission);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] =
                                outcomes_.try_emplace(sequence);
                        if (!inserted) {
                            throw std::logic_error(
                                    std::string("duplicate ")
                                    + Policy::backend_label()
                                    + " outstanding-work sequence");
                        }
                        it->second.sdpa_entries = entries;
                        it->second.workspace_lease =
                                captured.workspace_lease;
                        it->second.completion = completion;
                        it->second.is_sdpa = true;
                    }
                    Task task;
                    task.sequence = sequence;
                    task.is_sdpa = true;
                    task.sdpa_request.emplace(captured);
                    task.sdpa_entries = entries;
                    task.submission = submission.get();
                    task.completion = completion;
                    task.fence = task.submission;
                    worker_.submit_copy(std::move(task));
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcomes_.erase(sequence);
                    }
                    completion->clear();
                    throw;
                }
            });
}

template <typename Policy>
void GpuQueue<Policy>::execute(Task& task) {
    task.fence = task.submission;
    std::exception_ptr retained_failure;
    bool native_work_submitted = false;
    bool event_recorded = false;

    try {
        Policy::activate(context_);
        const auto execute_copy = [&] {
            const CopyRequest& request = *task.copy_request;
            if (request.no_op) {
                state_->mark_stream_drained(*task.submission);
                return;
            }
            const detail::CopyMetadataLayout layout =
                    detail::copy_metadata_layout(
                            SnapshotView{request.source},
                            SnapshotView{request.destination});
            detail::InlineCopyMetadata metadata{};
            detail::write_copy_metadata(
                    metadata, SnapshotView{request.source},
                    SnapshotView{request.destination});
            native_work_submitted = true;
            detail::launch_grid_stride_copy<Policy>(
                    stream_,
                    static_cast<const unsigned char*>(
                            request.source.native_handle),
                    static_cast<unsigned char*>(
                            request.destination.native_handle),
                    metadata, layout.total_words);
            Policy::check_kernel(Policy::copy_kernel_operation());
            Policy::after_grid_stride_launch();
            Policy::record_event(
                    state_->event_of(*task.submission), stream_);
            event_recorded = true;
            state_->mark_event_recorded(*task.submission);
        };

        if (task.is_binary) {
            const BinaryRequest& request = *task.binary_request;
            const std::size_t metadata_bytes =
                    detail::binary_metadata_storage_bytes(
                            request.result_shape.dimensions().size());
            const std::size_t metadata_slot =
                    task.metadata_lease.slot;
            task.submission->attach_metadata_slot(metadata_slot);
            task.metadata_lease.handoff();
            detail::write_binary_metadata(
                    metadata_pool_->host_data(metadata_slot),
                    metadata_pool_->device_data(metadata_slot), request);
            native_work_submitted = true;
            Policy::copy_from_host(
                    stream_, metadata_pool_->device_data(metadata_slot),
                    metadata_pool_->host_data(metadata_slot), metadata_bytes);
            const detail::BinaryMetadata metadata =
                    *reinterpret_cast<const detail::BinaryMetadata*>(
                            metadata_pool_->host_data(metadata_slot));
            const bool exact_alias =
                    request.lhs.owner_identity == request.out.owner_identity
                    || request.rhs.owner_identity
                            == request.out.owner_identity;
            const auto launch = [&]<DeviceBinaryOp Op>() {
                const auto lhs = static_cast<const unsigned char*>(
                        request.lhs.native_handle);
                const auto rhs = static_cast<const unsigned char*>(
                        request.rhs.native_handle);
                auto* out = static_cast<unsigned char*>(
                        request.out.native_handle);
                // Common validation rejects every same-owner non-exact
                // window, so owner identity is the established exact-alias
                // predicate here. Alias dispatch snapshots each tile before
                // any overlapping output word is written; disjoint requests
                // retain the word-grid fast path.
                if (exact_alias) {
                    detail::launch_alias_safe_binary<Policy, Op>(
                            stream_, lhs, rhs, out, metadata);
                } else {
                    detail::launch_grid_stride_binary<Policy, Op>(
                            stream_, lhs, rhs, out, metadata);
                }
            };
            switch (request.operation) {
                case DeviceOps::BinaryOperation::Add:
                    launch.template operator()<DeviceBinaryOp::add>();
                    break;
                case DeviceOps::BinaryOperation::Mul:
                    launch.template operator()<DeviceBinaryOp::mul>();
                    break;
                case DeviceOps::BinaryOperation::Sub:
                    launch.template operator()<DeviceBinaryOp::sub>();
                    break;
                case DeviceOps::BinaryOperation::Div:
                    launch.template operator()<DeviceBinaryOp::div>();
                    break;
            }
            Policy::check_kernel(Policy::copy_kernel_operation());
            Policy::after_grid_stride_launch();
            Policy::record_event(
                    state_->event_of(*task.submission), stream_);
            event_recorded = true;
            state_->mark_event_recorded(*task.submission);
        } else if (task.is_cache_append) {
            const CacheAppendRequest& request =
                    *task.cache_append_request;
            if constexpr (requires(
                                  typename Policy::stream_type stream,
                                  const CacheAppendRequest& candidate) {
                              Policy::launch_cache_append(stream, candidate);
                          }) {
                native_work_submitted = true;
                Policy::launch_cache_append(stream_, request);
                if constexpr (requires {
                                  Policy::after_cache_append_launch();
                              }) {
                    Policy::after_cache_append_launch();
                }
                Policy::record_event(
                        state_->event_of(*task.submission), stream_);
                event_recorded = true;
                state_->mark_event_recorded(*task.submission);
            } else {
                throw detail::UnsupportedOperation();
            }
        } else if (task.is_rmsnorm) {
            const RmsnormRequest& request = *task.rmsnorm_request;
            const std::size_t metadata_bytes =
                    detail::rmsnorm_metadata_storage_bytes(
                            request.x.spec.shape.dimensions().size());
            const std::size_t metadata_slot = task.metadata_lease.slot;
            task.submission->attach_metadata_slot(metadata_slot);
            task.metadata_lease.handoff();
            detail::write_rmsnorm_metadata(
                    metadata_pool_->host_data(metadata_slot),
                    metadata_pool_->device_data(metadata_slot), request);
            // The fixed metadata partition is uploaded on the queue's own
            // stream, so the device operation reads leading extents and
            // plane strides without a device-to-host round trip. From here
            // on the fixed event resource is only reusable through a proven
            // completion.
            native_work_submitted = true;
            Policy::copy_from_host(
                    stream_, metadata_pool_->device_data(metadata_slot),
                    metadata_pool_->host_data(metadata_slot), metadata_bytes);
            const detail::RmsnormMetadata metadata =
                    *reinterpret_cast<const detail::RmsnormMetadata*>(
                            metadata_pool_->host_data(metadata_slot));
            Policy::launch_rmsnorm(stream_, metadata);
            Policy::check_kernel(Policy::rmsnorm_kernel_operation());
            Policy::record_event(
                    state_->event_of(*task.submission), stream_);
            event_recorded = true;
            state_->mark_event_recorded(*task.submission);
        } else if (task.is_embedding) {
            const EmbeddingRequest& request = *task.embedding_request;
            const std::size_t metadata_slot =
                    task.metadata_lease.slot;
            const std::size_t metadata_bytes =
                    detail::embedding_metadata_storage_bytes(
                            request.indices.spec.shape.rank());
            task.submission->attach_metadata_slot(metadata_slot);
            task.metadata_lease.handoff();
            const auto metadata = detail::write_embedding_metadata(
                    metadata_pool_->host_data(metadata_slot),
                    metadata_pool_->device_data(metadata_slot), request);
            native_work_submitted = true;
            Policy::copy_from_host(
                    stream_, metadata_pool_->device_data(metadata_slot),
                    metadata_pool_->host_data(metadata_slot),
                    metadata_bytes);
            auto* status_cell =
                    metadata_pool_->status_data(task.status_slot);
            void* status_device =
                    detail::WorkspaceValidation::address(
                            request.workspace);
            if (status_cell == nullptr || status_device == nullptr) {
                throw std::runtime_error(
                        "CUDA embedding status resources are unavailable");
            }
            // The status word is device control state, not a caller
            // initialized value. It is reset in FIFO order immediately
            // before the gather and copied back only after that work.
            Policy::memset(
                    stream_, status_device, sizeof(std::uint32_t));
            detail::launch_grid_stride_embedding<Policy>(
                    stream_,
                    static_cast<const unsigned char*>(
                            request.table.native_handle),
                    static_cast<const unsigned char*>(
                            request.indices.native_handle),
                    static_cast<unsigned char*>(
                            request.out.native_handle),
                    static_cast<std::uint32_t*>(status_device), metadata);
            Policy::check_kernel(Policy::gather_kernel_operation());
            Policy::after_embedding_launch();
            Policy::copy_status_to_host(
                    stream_, status_cell, status_device,
                    sizeof(std::uint32_t));
            task.submission->attach_status_cell(status_cell);
            Policy::record_event(
                    state_->event_of(*task.submission), stream_);
            event_recorded = true;
            state_->mark_event_recorded(*task.submission);
        } else if (task.is_linear) {
            const LinearRequest& request = *task.linear_request;
            const std::size_t metadata_bytes =
                    detail::linear_metadata_storage_bytes(
                            request.x.spec.shape.rank());
            const std::size_t metadata_slot = task.metadata_lease.slot;
            task.submission->attach_metadata_slot(metadata_slot);
            task.metadata_lease.handoff();
            detail::write_linear_metadata(
                    metadata_pool_->host_data(metadata_slot),
                    metadata_pool_->device_data(metadata_slot), request);
            // The fixed metadata partition is uploaded on the queue's own
            // stream, so the device operation reads leading extents, plane
            // offsets, and plane strides without a device-to-host round trip.
            // From here on the fixed event resource is only reusable through
            // a proven completion.
            native_work_submitted = true;
            Policy::copy_from_host(
                    stream_, metadata_pool_->device_data(metadata_slot),
                    metadata_pool_->host_data(metadata_slot), metadata_bytes);
            const detail::LinearMetadata metadata =
                    *reinterpret_cast<const detail::LinearMetadata*>(
                            metadata_pool_->host_data(metadata_slot));
            Policy::launch_linear(stream_, metadata);
            Policy::check_kernel(Policy::linear_kernel_operation());
            Policy::record_event(
                    state_->event_of(*task.submission), stream_);
            event_recorded = true;
            state_->mark_event_recorded(*task.submission);
        } else if (task.is_silu) {
            const SiLURequest& request = *task.silu_request;
            const std::size_t metadata_bytes =
                    detail::silu_metadata_storage_bytes(request.x.rank);
            const std::size_t metadata_slot = task.metadata_lease.slot;
            task.submission->attach_metadata_slot(metadata_slot);
            task.metadata_lease.handoff();
            detail::write_silu_metadata(
                    metadata_pool_->host_data(metadata_slot),
                    metadata_pool_->device_data(metadata_slot), request);
            // The immutable descriptor is uploaded on the queue's existing
            // in-order stream. No caller workspace, hidden allocation, or
            // host payload round trip is introduced by the shared path.
            native_work_submitted = true;
            Policy::copy_from_host(
                    stream_, metadata_pool_->device_data(metadata_slot),
                    metadata_pool_->host_data(metadata_slot), metadata_bytes);
            const detail::SiluMetadata metadata =
                    *reinterpret_cast<const detail::SiluMetadata*>(
                            metadata_pool_->host_data(metadata_slot));
            Policy::launch_silu(stream_, metadata);
            Policy::check_kernel(Policy::silu_kernel_operation());
            Policy::record_event(
                    state_->event_of(*task.submission), stream_);
            event_recorded = true;
            state_->mark_event_recorded(*task.submission);
        } else if (task.is_rope) {
            const RopeRequest& request = *task.rope_request;
            const std::size_t metadata_bytes =
                    detail::rope_metadata_storage_bytes(request.x.rank);
            const std::size_t metadata_slot = task.metadata_lease.slot;
            task.submission->attach_metadata_slot(metadata_slot);
            task.metadata_lease.handoff();
            detail::write_rope_metadata(
                    metadata_pool_->host_data(metadata_slot),
                    metadata_pool_->device_data(metadata_slot), request);
            // The immutable descriptor is uploaded on the queue's existing
            // in-order stream. No caller workspace, hidden allocation, or
            // host payload round trip is introduced by the shared path.
            native_work_submitted = true;
            Policy::copy_from_host(
                    stream_, metadata_pool_->device_data(metadata_slot),
                    metadata_pool_->host_data(metadata_slot), metadata_bytes);
            const detail::RopeMetadata metadata =
                    *reinterpret_cast<const detail::RopeMetadata*>(
                            metadata_pool_->host_data(metadata_slot));
            Policy::launch_rope(stream_, metadata);
            Policy::check_kernel(Policy::rope_kernel_operation());
            Policy::record_event(
                    state_->event_of(*task.submission), stream_);
            event_recorded = true;
            state_->mark_event_recorded(*task.submission);
        } else if (task.is_sdpa) {
            const SdpaRequest& request = *task.sdpa_request;
            if constexpr (requires(
                                  typename Policy::stream_type stream,
                                  const SdpaRequest& candidate) {
                              Policy::launch_sdpa(stream, candidate);
                          }) {
                if constexpr (requires { Policy::sdpa_supported(); }) {
                    if (!Policy::sdpa_supported()) {
                        throw detail::UnsupportedOperation();
                    }
                } else {
                    throw detail::UnsupportedOperation();
                }
                // The private launcher owns only stage construction and
                // native calls. Completion remains the common queue's event
                // on this same in-order stream.
                native_work_submitted = true;
                Policy::launch_sdpa(stream_, request);
                Policy::record_event(
                        state_->event_of(*task.submission), stream_);
                event_recorded = true;
                state_->mark_event_recorded(*task.submission);
            } else {
                throw detail::UnsupportedOperation();
            }
        } else {
            execute_copy();
        }
    } catch (...) {
        retained_failure = std::current_exception();
        if (native_work_submitted && !event_recorded) {
            try {
                event_recorded = Policy::record_event_no_fault(
                        state_->event_of(*task.submission), stream_);
                if (event_recorded) {
                    state_->mark_event_recorded(*task.submission);
                } else if (Policy::synchronize_stream_noexcept(stream_)) {
                    state_->mark_stream_drained(*task.submission);
                } else {
                    state_->mark_retire_unknown(*task.submission);
                }
            } catch (...) {
                state_->mark_retire_unknown(*task.submission);
            }
        } else if (!native_work_submitted) {
            // No native operation was attempted, so the fixed event
            // resource is safe even though the accepted token fails.
            state_->mark_stream_drained(*task.submission);
        }
    }

    if (retained_failure != nullptr) {
        task.completion->set_failure(retained_failure);
    }
}

template <typename Policy>
void GpuQueue<Policy>::complete_task(
        std::uint64_t sequence, std::exception_ptr failure) {
    GpuOutcome outcome;
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
    if (!has_outcome) {
        complete(sequence, std::move(failure));
        return;
    }

    const bool completion_proven =
            outcome.completion == nullptr
            || outcome.completion->completion_proven();
    if (outcome.is_embedding && !failure && completion_proven) {
        const bool native_success =
                outcome.completion == nullptr
                || outcome.completion->result().failure == nullptr;
        if (native_success) {
            const auto status = outcome.completion != nullptr
                    ? outcome.completion->status_word() : std::nullopt;
            if (!status.has_value()) {
                failure = std::make_exception_ptr(std::runtime_error(
                        "CUDA embedding status cell is unavailable"));
            } else if (*status != 0) {
                failure = std::make_exception_ptr(std::invalid_argument(
                        "embedding index is out of range"));
            }
        }
    }
    if (outcome.completion != nullptr) {
        const detail::FenceResult result =
                outcome.completion->finalize(std::move(failure));
        failure = result.failure;
    }
    const bool failed = static_cast<bool>(failure);
    if (outcome.is_cache_append) {
        (void)detail::release_or_invalidate_binary_entries(
                registry_state_->registry, outcome.cache_append_entries,
                failed, completion_proven);
        detail::complete_workspace_lease(
                *registry_state_, outcome.workspace_lease.entry_id,
                completion_proven);
    } else if (outcome.is_sdpa) {
        (void)detail::release_or_invalidate_sdpa_entries(
                registry_state_->registry, outcome.sdpa_entries,
                failed, completion_proven);
        if (failed || !completion_proven
                || outcome.workspace_lease.entry_id == 0) {
            // A failed or unproven submission frees its range exactly as the
            // covering stream proof allows: a failed output has no consumer,
            // and an unproven range stays quarantined by the shared lease
            // protocol. A submission that leased no range has nothing to hold.
            detail::complete_workspace_lease(
                    *registry_state_, outcome.workspace_lease.entry_id,
                    completion_proven);
        } else {
            // Successful accepted work keeps its caller workspace live until
            // the caller observes the token: the range may be reused only
            // after that observation, so a second submission of the same live
            // range is bounded-resource exhaustion instead of a silent
            // overlap. `wait` releases it through `fence_through_sequence`,
            // and queue teardown releases whatever no caller observed.
            bool retained = false;
            try {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                retained = retained_leases_
                                   .emplace(sequence, outcome.workspace_lease)
                                   .second;
            } catch (...) {
                retained = false;
            }
            if (!retained) {
                // Retaining failed: fall back to the proof-based release
                // rather than dropping the lease record entirely.
                detail::complete_workspace_lease(
                        *registry_state_, outcome.workspace_lease.entry_id,
                        completion_proven);
            }
        }
    } else if (outcome.is_binary || outcome.is_embedding
            || outcome.is_linear) {
        (void)detail::release_or_invalidate_binary_entries(
                registry_state_->registry, outcome.binary_entries,
                failed, completion_proven);
        detail::complete_workspace_lease(
                *registry_state_, outcome.workspace_lease.entry_id,
                completion_proven);
    } else if (outcome.is_silu) {
        // SiLU consumes no raw workspace, so its admitted `{0, 1}`
        // requirement reserved no lease; the deduplicated input/output
        // registrations are finalized only after the same completion proof
        // that protects every other fixed-resource GPU operation.
        (void)detail::release_or_invalidate_binary_entries(
                registry_state_->registry, outcome.silu_entries,
                failed, completion_proven);
    } else if (outcome.is_rope) {
        // RoPE currently admits the fixed zero-workspace requirement, but
        // retains the same registered two-owner and lease state so a future
        // backend launch cannot bypass proven completion or quarantine.
        (void)detail::release_or_invalidate_binary_entries(
                registry_state_->registry, outcome.rope_entries,
                failed, completion_proven);
        detail::complete_workspace_lease(
                *registry_state_, outcome.workspace_lease.entry_id,
                completion_proven);
    } else if (outcome.is_rmsnorm) {
        // RMSNorm consumes no raw workspace, so its admitted `{0, 1}`
        // requirement reserved no lease; the deduplicated read/read
        // `x`/`scale` and disjoint-output registrations are finalized
        // under the same failure and completion-proof rules.
        (void)detail::release_or_invalidate_binary_entries(
                registry_state_->registry, outcome.rmsnorm_entries,
                failed, completion_proven);
    } else {
        (void)detail::release_or_invalidate_entries(
                registry_state_->registry, outcome.common, failed,
                completion_proven);
    }
    complete(sequence, std::move(failure));
}

template <typename Policy>
void GpuQueue<Policy>::fence_through_sequence(
        std::uint64_t sequence) noexcept {
    detail::WorkspaceLease lease{};
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        const auto found = retained_leases_.find(sequence);
        if (found == retained_leases_.end()) {
            return;
        }
        lease = found->second;
        retained_leases_.erase(found);
    }
    // The caller has now observed this token's terminal state, so its
    // retained workspace range returns to the shared lease registry and may
    // be leased again. Repeated waits find no record and change nothing.
    detail::complete_workspace_lease(*registry_state_, lease.entry_id, true);
}

template <typename Policy>
void GpuQueue<Policy>::release_retained_leases(
        bool covering_proof) noexcept {
    std::map<std::uint64_t, detail::WorkspaceLease> retained;
    try {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        retained.swap(retained_leases_);
    } catch (...) {
        // The device keeps a retained range out of its allocator while the
        // lease record survives, so an unavailable queue lock can only leave
        // the ranges reserved, never reuse them unsafely.
        return;
    }
    for (const auto& entry : retained) {
        detail::complete_workspace_lease(
                *registry_state_, entry.second.entry_id, covering_proof);
    }
}

}  // namespace iom::detail
