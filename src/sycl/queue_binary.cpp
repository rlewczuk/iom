#include "queue_internal.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "scalar_add.hpp"

namespace iom::sycl_detail {

WorkspaceRequirements SyclQueue::binary_workspace_requirements(
        const BinaryRequest& request) {
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

std::size_t SyclQueue::binary_view_staging_bytes(
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

std::size_t SyclQueue::checked_add_local(
        std::size_t lhs, std::size_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(what);
    }
    return lhs + rhs;
}

std::size_t SyclQueue::checked_mul_local(
        std::size_t lhs, std::size_t rhs, const char* what) {
    if (lhs != 0
            && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(what);
    }
    return lhs * rhs;
}

std::size_t SyclQueue::align_up_checked(
        std::size_t value, std::size_t alignment) {
    const std::size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return checked_add_local(
            value, alignment - remainder,
            "SYCL workspace staging alignment overflows");
}

std::size_t SyclQueue::checked_binary_view_staging_bytes(
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

template <detail::scalar_add_detail::BinaryOp Op>
void SyclQueue::binary_elements_impl(
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

void SyclQueue::binary_elements(
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

void SyclQueue::free_binary_host_staging(
        void* staging, const sycl::context& context) noexcept {
    if (staging == nullptr) {
        return;
    }
    try {
        sycl::free(staging, context);
    } catch (...) {
    }
}

void SyclQueue::execute_binary(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
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
        const auto completion_slot = completion_pool_->try_acquire();
        if (!completion_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        task.state->set_completion_slot(
                *completion_pool_, *completion_slot);
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
        const std::exception_ptr failure = std::current_exception();
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
        task.state->set_failure(failure);
    }
}

}  // namespace iom::sycl_detail
