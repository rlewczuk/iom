#include <array>
#include "queue_internal.hpp"

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
}

std::atomic<std::size_t> g_fence_wait_count{0};

[[nodiscard]] bool consume_submission_fault(
        SubmissionFault point) noexcept {
    SubmissionFault expected = point;
    return g_submission_fault.compare_exchange_strong(
            expected, SubmissionFault::none, std::memory_order_acq_rel);
}

namespace {
struct SyclFenceCapture {
    std::shared_ptr<SyclFenceState> state;
};
}  // namespace

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
namespace {

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

detail::Fence transfer_fence() noexcept {
    detail::Fence fence;
    fence.invoke = [](const detail::Fence&) noexcept {
        return detail::FenceResult::success();
    };
    return fence;
}

namespace {

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
