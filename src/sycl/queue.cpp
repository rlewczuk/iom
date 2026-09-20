#include "queue_internal.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "runtime.hpp"
#include "../iom_internal.hpp"

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

struct CacheAppendMetadata {
    std::uint64_t source_plane_offset = 0;
    std::uint64_t destination_plane_offset = 0;
    std::uint64_t source_rows = 0;
    std::uint64_t destination_rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t append_offset = 0;
    std::uint64_t plane_count = 0;
    std::uint64_t destination_words_per_plane = 0;
    std::uint64_t leading_rank = 0;
    std::uint64_t leading_dimensions[6]{};
    std::uint64_t source_strides[6]{};
    std::uint64_t destination_strides[6]{};
    std::uint32_t bits = 0;
};

static_assert(std::is_trivially_copyable_v<CacheAppendMetadata>);
static_assert(sizeof(CacheAppendMetadata) <= detail::kMetadataSlotBytes);

struct CacheAppendLayout {
    CacheAppendMetadata metadata{};
    std::size_t total_words = 0;
};

[[nodiscard]] std::size_t cache_append_checked_add(
        std::size_t lhs, std::size_t rhs, const char* message) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(message);
    }
    return lhs + rhs;
}

[[nodiscard]] std::size_t cache_append_checked_mul(
        std::size_t lhs, std::size_t rhs, const char* message) {
    if (lhs != 0
            && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

[[nodiscard]] std::uint64_t cache_append_metadata_u64(
        std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::size_t cache_append_padded(
        std::size_t value, const char* message) {
    const std::size_t tiles =
            value / TensorSpec::TILE + (value % TensorSpec::TILE != 0);
    return cache_append_checked_mul(
            tiles, TensorSpec::TILE, message);
}

[[nodiscard]] std::size_t cache_append_plane_slot(
        std::size_t plane, std::size_t row, std::size_t column,
        std::size_t rows, std::size_t columns) {
    const std::size_t tile_rows =
            cache_append_padded(rows, "SYCL cache append tile rows overflow")
            / TensorSpec::TILE;
    const std::size_t tile_columns =
            cache_append_padded(
                    columns, "SYCL cache append tile columns overflow")
            / TensorSpec::TILE;
    const std::size_t tiles_per_plane = cache_append_checked_mul(
            tile_rows, tile_columns,
            "SYCL cache append tile count overflows");
    const std::size_t tile_index = cache_append_checked_add(
            cache_append_checked_mul(
                    plane, tiles_per_plane,
                    "SYCL cache append tile index overflows"),
            cache_append_checked_add(
                    cache_append_checked_mul(
                            row / TensorSpec::TILE, tile_columns,
                            "SYCL cache append tile index overflows"),
                    column / TensorSpec::TILE,
                    "SYCL cache append tile index overflows"),
            "SYCL cache append tile index overflows");
    const std::size_t in_tile = cache_append_checked_add(
            cache_append_checked_mul(
                    row % TensorSpec::TILE, TensorSpec::TILE,
                    "SYCL cache append tile slot overflows"),
            column % TensorSpec::TILE,
            "SYCL cache append tile slot overflows");
    return cache_append_checked_add(
            cache_append_checked_mul(
                    tile_index,
                    TensorSpec::TILE * TensorSpec::TILE,
                    "SYCL cache append element slot overflows"),
            in_tile, "SYCL cache append element slot overflows");
}
template <typename View>
[[nodiscard]] std::size_t cache_append_max_plane(
        const View& view) {
    std::size_t result = view.plane_offset;
    for (std::size_t axis = 0; axis + 2 < view.rank; ++axis) {
        const std::size_t contribution = cache_append_checked_mul(
                view.dimensions[axis] - 1, view.plane_strides[axis],
                "SYCL cache append plane address overflows");
        result = cache_append_checked_add(
                result, contribution,
                "SYCL cache append plane address overflows");
    }
    return result;
}

template <typename Request>
[[nodiscard]] CacheAppendLayout build_cache_append_layout(
        const Request& request) {
    if (request.source.rank < 3 || request.source.rank > 8
            || request.destination.rank != request.source.rank) {
        throw std::invalid_argument(
                "SYCL cache append requires ranks three through eight");
    }
    const std::size_t rank = request.source.rank;
    const std::size_t leading_rank = rank - 2;
    const std::size_t source_rows = request.source.dimensions[rank - 2];
    const std::size_t destination_rows =
            request.destination.dimensions[rank - 2];
    const std::size_t columns = request.source.dimensions[rank - 1];
    if (source_rows == 0 || destination_rows == 0 || columns == 0
            || request.a > destination_rows
            || source_rows > destination_rows - request.a) {
        throw std::invalid_argument(
                "SYCL cache append row range is invalid");
    }
    for (std::size_t axis = 0; axis + 2 < rank; ++axis) {
        if (request.source.dimensions[axis]
                != request.destination.dimensions[axis]) {
            throw std::invalid_argument(
                    "SYCL cache append leading dimensions do not match");
        }
    }
    if (request.source.dimensions[rank - 1] != columns
            || request.destination.dimensions[rank - 1] != columns) {
        throw std::invalid_argument(
                "SYCL cache append feature dimensions do not match");
    }
    (void)cache_append_checked_add(
            request.a, source_rows,
            "SYCL cache append row position overflows");

    const std::size_t bits = detail::leaf_bits(request.source.data_type);
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = cache_append_checked_mul(
                plane_count, request.source.dimensions[axis],
                "SYCL cache append plane count overflows");
    }
    const std::size_t source_padded_rows = cache_append_padded(
            source_rows, "SYCL cache append source tile rows overflow");
    const std::size_t destination_padded_rows = cache_append_padded(
            destination_rows,
            "SYCL cache append destination tile rows overflow");
    const std::size_t padded_columns = cache_append_padded(
            columns, "SYCL cache append tile columns overflow");
    const std::size_t source_plane_bits = cache_append_checked_mul(
            cache_append_checked_mul(
                    source_padded_rows, padded_columns,
                    "SYCL cache append source plane size overflows"),
            bits, "SYCL cache append source plane bits overflow");
    const std::size_t destination_plane_bits = cache_append_checked_mul(
            cache_append_checked_mul(
                    destination_padded_rows, padded_columns,
                    "SYCL cache append destination plane size overflows"),
            bits, "SYCL cache append destination plane bits overflow");
    const std::size_t source_words_per_plane =
            cache_append_checked_add(
                    source_plane_bits, 31,
                    "SYCL cache append source word count overflows")
            / 32;
    const std::size_t destination_words_per_plane =
            cache_append_checked_add(
                    destination_plane_bits, 31,
                    "SYCL cache append destination word count overflows")
            / 32;
    const std::size_t total_words = cache_append_checked_mul(
            plane_count, destination_words_per_plane,
            "SYCL cache append total word count overflows");

    const std::size_t source_max_plane = cache_append_max_plane(
            request.source);
    const std::size_t destination_max_plane = cache_append_max_plane(
            request.destination);
    const std::size_t source_last_slot = cache_append_plane_slot(
            source_max_plane, source_rows - 1, columns - 1,
            source_rows, columns);
    const std::size_t destination_last_slot = cache_append_plane_slot(
            destination_max_plane, destination_rows - 1, columns - 1,
            destination_rows, columns);
    (void)cache_append_checked_add(
            cache_append_checked_mul(
                    source_last_slot, bits,
                    "SYCL cache append source address overflows"),
            bits, "SYCL cache append source address overflows");
    (void)cache_append_checked_add(
            cache_append_checked_mul(
                    destination_last_slot, bits,
                    "SYCL cache append destination address overflows"),
            bits, "SYCL cache append destination address overflows");
    (void)cache_append_metadata_u64(
            total_words, "SYCL cache append total words overflows");
    (void)cache_append_metadata_u64(
            source_words_per_plane,
            "SYCL cache append source word count overflows");

    CacheAppendLayout result;
    result.total_words = total_words;
    result.metadata.source_plane_offset = cache_append_metadata_u64(
            request.source.plane_offset,
            "SYCL cache append source plane offset overflows");
    result.metadata.destination_plane_offset = cache_append_metadata_u64(
            request.destination.plane_offset,
            "SYCL cache append destination plane offset overflows");
    result.metadata.source_rows = cache_append_metadata_u64(
            source_rows, "SYCL cache append source rows overflow");
    result.metadata.destination_rows = cache_append_metadata_u64(
            destination_rows, "SYCL cache append destination rows overflow");
    result.metadata.columns = cache_append_metadata_u64(
            columns, "SYCL cache append feature count overflows");
    result.metadata.append_offset = cache_append_metadata_u64(
            request.a, "SYCL cache append offset overflows");
    result.metadata.plane_count = cache_append_metadata_u64(
            plane_count, "SYCL cache append plane count overflows");
    result.metadata.destination_words_per_plane = cache_append_metadata_u64(
            destination_words_per_plane,
            "SYCL cache append destination word count overflows");
    result.metadata.leading_rank = cache_append_metadata_u64(
            leading_rank, "SYCL cache append leading rank overflows");
    result.metadata.bits = static_cast<std::uint32_t>(bits);
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        result.metadata.leading_dimensions[axis] =
                cache_append_metadata_u64(
                        request.source.dimensions[axis],
                        "SYCL cache append leading dimension overflows");
        result.metadata.source_strides[axis] =
                cache_append_metadata_u64(
                        request.source.plane_strides[axis],
                        "SYCL cache append source stride overflows");
        result.metadata.destination_strides[axis] =
                cache_append_metadata_u64(
                        request.destination.plane_strides[axis],
                        "SYCL cache append destination stride overflows");
    }
    return result;
}

inline void cache_append_one_word(
        const unsigned char* source, unsigned char* destination,
        const CacheAppendMetadata& metadata, std::uint64_t word) {
    const std::uint64_t logical_plane =
            word / metadata.destination_words_per_plane;
    const std::uint64_t word_in_plane =
            word % metadata.destination_words_per_plane;
    std::uint64_t source_plane = metadata.source_plane_offset;
    std::uint64_t destination_plane =
            metadata.destination_plane_offset;
    std::uint64_t rest = logical_plane;
    for (std::uint64_t axis = metadata.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % metadata.leading_dimensions[axis];
        rest /= metadata.leading_dimensions[axis];
        source_plane += coordinate * metadata.source_strides[axis];
        destination_plane += coordinate
                * metadata.destination_strides[axis];
    }
    const std::uint64_t destination_base_slot = detail::plane_slot(
            destination_plane, 0, 0, metadata.destination_rows,
            metadata.columns);
    const std::uint64_t destination_base_word =
            destination_base_slot * metadata.bits / 32;
    auto* destination_words =
            reinterpret_cast<std::uint32_t*>(destination)
            + destination_base_word + word_in_plane;
    std::uint32_t destination_word = *destination_words;

    const std::uint64_t padded_rows =
            (metadata.destination_rows / TensorSpec::TILE
             + (metadata.destination_rows % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t padded_columns =
            (metadata.columns / TensorSpec::TILE
             + (metadata.columns % TensorSpec::TILE != 0))
            * TensorSpec::TILE;
    const std::uint64_t plane_bits =
            padded_rows * padded_columns * metadata.bits;
    const std::uint64_t word_first_bit = word_in_plane * 32;
    const std::uint64_t word_end_bit =
            word_first_bit + 32 < plane_bits
            ? word_first_bit + 32
            : plane_bits;
    const std::uint64_t first_slot = word_first_bit / metadata.bits;
    const std::uint64_t last_slot =
            (word_end_bit + metadata.bits - 1) / metadata.bits;
    for (std::uint64_t slot = first_slot; slot < last_slot; ++slot) {
        const auto coordinate = detail::physical_coordinate(
                slot, metadata.destination_rows,
                metadata.columns);
        if (coordinate.row >= metadata.destination_rows
                || coordinate.column >= metadata.columns
                || coordinate.row < metadata.append_offset
                || coordinate.row
                        >= metadata.append_offset + metadata.source_rows) {
            continue;
        }
        const std::uint64_t source_bit =
                detail::plane_slot(
                        source_plane,
                        coordinate.row - metadata.append_offset,
                        coordinate.column, metadata.source_rows,
                        metadata.columns)
                * metadata.bits;
        const std::uint64_t destination_bit =
                detail::plane_slot(
                        destination_plane, coordinate.row,
                        coordinate.column, metadata.destination_rows,
                        metadata.columns)
                * metadata.bits;
        detail::merge_overlapping_field(
                destination_word, source, source_bit, destination_bit,
                word_first_bit + destination_base_word * 32,
                word_end_bit + destination_base_word * 32,
                static_cast<unsigned int>(metadata.bits));
    }
    detail::store_word(destination_words, destination_word);
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
          sdpa_supported_(sdpa_matrix_device_capable(native_device)),
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
WorkspaceRequirements SyclQueue::cache_append_workspace_requirements(
        const CacheAppendRequest& request) {
    (void)build_cache_append_layout(request);
    return {0, 1};
}

oid SyclQueue::cache_append_impl(const CacheAppendRequest& request) {
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
    return submit_cache_append(
            request, *state_, registry_queue_id_, fence,
            [this, state](
                    std::uint64_t sequence,
                    const CacheAppendRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                Task task;
                task.sequence = sequence;
                task.state = state;
                task.fence = state.get();
                task.cache_append_request.emplace(captured);
                task.cache_append_entries = entries;
                worker_.submit_copy(std::move(task));
            });
}

WorkspaceRequirements SyclQueue::silu_workspace_requirements_impl(
        const SiLURequest& request) {
    // Pure capability decision of the implemented SiLU leaf set. The common
    // facade has already completed every structural, device, owner, shape,
    // layout, dtype, quantization, alias, and checked-arithmetic admission
    // step, so this hook performs no allocation, registration, lease,
    // sequence, submission, queue or arena inspection, or data access. An
    // unported or semantically inapplicable leaf is `Unsupported` here; a
    // queued leaf consumes no raw workspace and reports the exact zero path.
    if (!silu_device_supported(request.x.data_type)) {
        throw detail::UnsupportedOperation();
    }
    return WorkspaceRequirements{0, 1};
}

oid SyclQueue::silu_impl(const SiLURequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    // Capability and descriptor representation are decided before the
    // submission sequence, the owner registration, the fixed queue resources,
    // and any output mutation, so an unported leaf and an unrepresentable
    // request are both rejected repeatably without side effects.
    if (!silu_device_supported(request.x.data_type)) {
        throw detail::UnsupportedOperation();
    }
    validate_silu_representation(request);
    if (consume_submission_fault(SubmissionFault::state_allocation)) {
        throw std::bad_alloc();
    }
    auto state = std::make_shared<SyclFenceState>();
    if (consume_submission_fault(SubmissionFault::fence_construction)) {
        throw std::bad_alloc();
    }
    detail::Fence fence = build_sycl_fence(state);
    return submit_silu(
            request, *state_, registry_queue_id_, fence,
            [this, state](
                    std::uint64_t sequence, const SiLURequest& captured,
                    detail::BinaryEntryRegistration entries) {
                Task task;
                task.sequence = sequence;
                task.state = state;
                task.fence = state.get();
                task.silu_request.emplace(captured);
                task.silu_entries = entries;
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
    if (task.cache_append_request.has_value()) {
        execute_cache_append(task);
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
    if (task.silu_request.has_value()) {
        execute_silu(task);
        return;
    }
    if (task.sdpa_plan.has_value()) {
        execute_sdpa(task);
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
void SyclQueue::execute_cache_append(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    {
        SyclSequenceOutcome outcome;
        outcome.state = task.state;
        outcome.workspace_lease =
                task.cache_append_request->workspace_lease;
        outcome.cache_append_entries = task.cache_append_entries;
        const auto [it, inserted] = outcomes_.emplace(
                task.sequence, std::move(outcome));
        if (!inserted) {
            throw std::logic_error(
                    "duplicate SYCL outstanding-work sequence");
        }
    }

    bool native_attempted = false;
    try {
        const auto completion_slot = completion_pool_->try_acquire();
        if (!completion_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        task.state->set_completion_slot(
                *completion_pool_, *completion_slot);
        const auto metadata_slot = metadata_pool_->try_acquire();
        if (!metadata_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        const std::size_t metadata_index = *metadata_slot;
        task.state->set_metadata_slot(*metadata_pool_, metadata_index);

        const CacheAppendLayout layout = build_cache_append_layout(
                *task.cache_append_request);
        std::memcpy(
                metadata_pool_->host_data(metadata_index),
                &layout.metadata, sizeof(layout.metadata));
        const auto* metadata = static_cast<const CacheAppendMetadata*>(
                metadata_pool_->device_data(metadata_index));
        if (metadata == nullptr) {
            throw std::bad_alloc();
        }
        if (consume_submission_fault(SubmissionFault::first_submit)) {
            throw std::runtime_error(
                    "injected SYCL first-submit failure");
        }

        native_attempted = true;
        // The accepted-failure seam is consumed after fixed resources and
        // the immutable request have been admitted but before payload work is
        // enqueued. This keeps the retained failure side-effect free while
        // still exercising the positive-OID fence and owner path.
        if (consume_submission_fault(SubmissionFault::post_launch)) {
            throw std::runtime_error(
                    "injected SYCL post-launch failure");
        }
        queue_.memcpy(
                metadata_pool_->device_data(metadata_index),
                metadata_pool_->host_data(metadata_index),
                sizeof(layout.metadata));
        const auto* source_handle = static_cast<const unsigned char*>(
                task.cache_append_request->source.native_handle);
        auto* destination_handle = static_cast<unsigned char*>(
                task.cache_append_request->destination.native_handle);
        sycl::event event = queue_.parallel_for(
                sycl::range<1>(layout.total_words),
                [=](sycl::id<1> item) {
                    cache_append_one_word(
                            source_handle, destination_handle, *metadata,
                            static_cast<std::uint64_t>(item[0]));
                });
        if (launch_calls.kernel_launched != nullptr) {
            launch_calls.kernel_launched();
        }
        task.state->set_event(std::move(event));
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
        } else if (outcome.cache_append_entries.has_value()) {
            // Cache append is direct and consumes no caller workspace; keep
            // the common owner registrations on the same proof path as the
            // other two-owner asynchronous operations.
            (void)detail::release_or_invalidate_binary_entries(
                    state_->registry, *outcome.cache_append_entries,
                    failed, fence_succeeded);
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
        } else if (outcome.silu_entries.has_value()) {
            // SiLU registers the same deduplicated read/read owner set and
            // consumes no `RawWorkspace` at all: there is no lease to retire,
            // and a supplied workspace range is never inspected or retained.
            (void)detail::release_or_invalidate_binary_entries(
                    state_->registry, *outcome.silu_entries, failed,
                    fence_succeeded);
        } else if (outcome.sdpa_entries.has_value()) {
            // SDPA registers four distinct tensor owners plus the caller
            // workspace lease, so an unknown completion retains or quarantines
            // the lease instead of releasing it early.
            (void)detail::release_or_invalidate_sdpa_entries(
                    state_->registry, *outcome.sdpa_entries, failed,
                    fence_succeeded);
            detail::complete_workspace_lease(
                    *state_, outcome.workspace_lease, completion_proven);
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
