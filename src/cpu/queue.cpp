#include "device_internal.hpp"
#include "transfer_helpers.hpp"

#include "../iom_internal.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "iom/iom.hpp"
#include "../shared/scalar_add.hpp"
#include "../shared/scalar_silu.hpp"

namespace iom {
namespace {


template <typename Carrier>
using CpuCarrierTraits = cpu_detail::CpuCarrierTraits<Carrier>;

template <typename Carrier>
using CpuRopeCodec =
        detail::scalar_binary_codec_detail::Codec<CpuCarrierTraits<Carrier>>;

}  // namespace

namespace cpu_detail {

namespace {

std::atomic<bool> silu_failure_armed{false};
std::atomic<std::uint64_t> linear_failure_plan{0};

}  // namespace

// CPU-local failure construction for the SiLU port. `libiom` is compiled
// without `IOM_ENABLE_TESTING` — only the accelerator libraries receive it —
// so no accelerator fault hook can reach this translation unit and the CPU
// SiLU driver owns this minimal seam instead. It is one process-wide one-shot
// latch: the next enqueued SiLU task consumes it after acceptance and before
// its element loop, so exactly that accepted sequence retains the failure while
// every later submission stays healthy. It carries no other state and is inert
// until a test arms it.
void arm_silu_failure() noexcept {
    silu_failure_armed.store(true, std::memory_order_release);
}

void clear_silu_failure() noexcept {
    silu_failure_armed.store(false, std::memory_order_release);
}

[[nodiscard]] bool consume_silu_failure() noexcept {
    return silu_failure_armed.exchange(false, std::memory_order_acq_rel);
}

// The matching seam for the linear port, which shares this driver's shape: it
// is one process-wide plan consumed by later enqueued linear tasks after
// acceptance and before their element loops, so an accepted projection can
// retain a post-acceptance failure while every later submission stays healthy.
// The plan lets `healthy_before` submissions pass first and then fails
// `failures` consecutive ones, which is what distinguishes an independent
// failure in either one of two sibling projection branches from a failure in
// both of them. One call sets the whole plan, and arming is meaningful while
// no linear task is executing. It carries no other state and is inert until a
// test arms it.
void arm_linear_failure(
        std::size_t healthy_before, std::size_t failures) noexcept {
    linear_failure_plan.store(
            (static_cast<std::uint64_t>(healthy_before) << 32)
                    | static_cast<std::uint64_t>(failures),
            std::memory_order_release);
}

void clear_linear_failure() noexcept {
    linear_failure_plan.store(0, std::memory_order_release);
}

[[nodiscard]] bool consume_linear_failure() noexcept {
    std::uint64_t plan = linear_failure_plan.load(std::memory_order_acquire);
    for (;;) {
        const std::size_t healthy = static_cast<std::size_t>(plan >> 32);
        const std::size_t failures =
                static_cast<std::size_t>(plan & 0xFFFFFFFFu);
        if (healthy == 0 && failures == 0) return false;
        const std::uint64_t next = healthy != 0
                ? plan - (std::uint64_t{1} << 32)
                : plan - 1;
        if (linear_failure_plan.compare_exchange_weak(
                    plan, next, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
            return healthy == 0;
        }
    }
}

}  // namespace cpu_detail


class CpuQueue final : public DeviceOps {
    struct HostTask {
        std::function<void()> work;
    };

public:
    explicit CpuQueue(CpuDevice& device)
            : DeviceOps(device), device_(&device),
              registry_queue_id_(detail::allocate_queue_id(
                      device.registry_state())) {
        worker_ = std::thread([this] { run_worker(); });
    }

    ~CpuQueue() override {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            worker_shutdown_ = true;
        }
        worker_cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        device_->registry_state().registry.invalidate_entries_for_queue(
                registry_queue_id_);
    }

    oid copy_impl(const TensorView& source, TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_copy(
                source, destination, device_->registry_state(),
                registry_queue_id_, fence,
                [this](std::uint64_t sequence, const CopyRequest& captured,
                       detail::EntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, sequence, captured, entries] {
                                    std::exception_ptr failure;
                                    try {
                                        if (!captured.no_op) {
                                            copy_elements(
                                                    captured.source,
                                                    captured.destination);
                                        }
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    const detail::SequenceOutcome outcome{
                                            entries.source,
                                            entries.destination,
                                            failure, false};
                                    (void)detail::release_or_invalidate_entries(
                                            device_->registry_state().registry,
                                            outcome, static_cast<bool>(failure),
                                            true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry
                                .remove_entry_if_present(
                                        entries.source,
                                        captured.source.native_handle);
                        device_->registry_state().registry
                                .remove_entry_if_present(
                                        entries.destination,
                                        captured.destination.native_handle);
                        throw;
                    }
                });
    }

    oid cache_append_impl(const CacheAppendRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_cache_append(
                request, device_->registry_state(), registry_queue_id_, fence,
                [this](
                        std::uint64_t sequence,
                        const CacheAppendRequest& captured,
                        detail::BinaryEntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, sequence, captured, entries] {
                                    std::exception_ptr failure;
                                    try {
                                        cache_append_elements(captured);
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    (void)
                                            detail::release_or_invalidate_binary_entries(
                                                    device_->registry_state()
                                                            .registry,
                                                    entries,
                                                    static_cast<bool>(failure),
                                                    true);
                                    detail::complete_workspace_lease(
                                            device_->registry_state(),
                                            captured.workspace_lease, true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        detail::complete_workspace_lease(
                                device_->registry_state(),
                                captured.workspace_lease, true);
                        throw;
                    }
                });
    }

    [[nodiscard]] WorkspaceRequirements
            cache_append_workspace_requirements(
                    const CacheAppendRequest&) override {
        return {0, 1};
    }

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "CPU";
    }

private:
    void enqueue(HostTask task) {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            if (worker_shutdown_) {
                throw std::logic_error("CPU queue worker is shut down");
            }
            worker_tasks_.push_back(std::move(task));
        }
        worker_cv_.notify_one();
    }

    void run_worker() noexcept {
        for (;;) {
            HostTask task;
            {
                std::unique_lock<std::mutex> lock(worker_mutex_);
                worker_cv_.wait(lock, [this] {
                    return worker_shutdown_ || !worker_tasks_.empty();
                });
                if (worker_tasks_.empty()) {
                    return;
                }
                task = std::move(worker_tasks_.front());
                worker_tasks_.pop_front();
            }
            try {
                task.work();
            } catch (...) {
            }
        }
    }

    static detail::FenceResult fence_pending(
            const detail::Fence&) noexcept {
        return detail::FenceResult::pending();
    }
    static detail::FenceResult fence_success(
            const detail::Fence&) noexcept {
        return detail::FenceResult::success();
    }

    using ScalarBinary = std::uint64_t (*) (
            DataType, std::uint64_t, std::uint64_t) noexcept;

    // Named-format decode/encode of the nine applicable floating leaves,
    // shared with the existing scalar codec; RMS normalization adds no
    // second codec and no host numeric format of its own.
    using RmsnormFormat = detail::scalar_add_detail::Format;

    static std::size_t source_plane(
            const DeviceOps::BinaryViewSnapshot& source,
            std::span<const std::size_t> result_dimensions,
            std::span<const std::size_t> coordinates) {
        const auto dimensions = source.spec.shape.dimensions();
        const std::size_t leading = result_dimensions.size() - 2;
        const std::size_t offset = leading - (dimensions.size() - 2);
        std::size_t plane = source.plane_offset;
        for (std::size_t axis = 0; axis < leading; ++axis) {
            if (axis >= offset && source.logical_plane_strides[axis] != 0) {
                plane += coordinates[axis]
                         * source.logical_plane_strides[axis];
            }
        }
        return plane;
    }

    static ScalarBinary select_binary(DeviceOps::BinaryOperation operation) {
        switch (operation) {
            case DeviceOps::BinaryOperation::Add:
                return &detail::scalar_binary<
                        detail::scalar_add_detail::BinaryOp::add>;
            case DeviceOps::BinaryOperation::Mul:
                return &detail::scalar_binary<
                        detail::scalar_add_detail::BinaryOp::mul>;
            case DeviceOps::BinaryOperation::Sub:
                return &detail::scalar_binary<
                        detail::scalar_add_detail::BinaryOp::sub>;
            case DeviceOps::BinaryOperation::Div:
                return &detail::scalar_binary<
                        detail::scalar_add_detail::BinaryOp::div>;
        }
        throw std::invalid_argument("unknown CPU binary operation");
    }

    static void binary_elements(
            const DeviceOps::BinaryRequest& request, ScalarBinary scalar) {
        const auto dimensions = request.result_shape.dimensions();
        const std::size_t rank = dimensions.size();
        const std::size_t rows = dimensions[rank - 2];
        const std::size_t columns = dimensions[rank - 1];
        const std::size_t bits = detail::leaf_bits(request.out.spec.data_type);
        auto* out_base = static_cast<unsigned char*>(request.out.native_handle);
        const auto* lhs_base =
                static_cast<const unsigned char*>(request.lhs.native_handle);
        const auto* rhs_base =
                static_cast<const unsigned char*>(request.rhs.native_handle);
        std::vector<std::size_t> coordinates(rank);
        auto visit = [&](auto&& self, std::size_t axis) -> void {
            if (axis + 2 < rank) {
                for (std::size_t index = 0; index < dimensions[axis]; ++index) {
                    coordinates[axis] = index;
                    self(self, axis + 1);
                }
                return;
            }
            const std::size_t lhs_plane =
                    source_plane(request.lhs, dimensions, coordinates);
            const std::size_t rhs_plane =
                    source_plane(request.rhs, dimensions, coordinates);
            const std::size_t out_plane =
                    source_plane(request.out, dimensions, coordinates);
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t column = 0; column < columns; ++column) {
                    const std::size_t lhs_row =
                            request.lhs.broadcast_rows ? 0 : row;
                    const std::size_t rhs_row =
                            request.rhs.broadcast_rows ? 0 : row;
                    const std::size_t lhs_column =
                            request.lhs.broadcast_columns ? 0 : column;
                    const std::size_t rhs_column =
                            request.rhs.broadcast_columns ? 0 : column;
                    const std::size_t lhs_slot = detail::standard_plane_slot(
                            request.lhs.spec, lhs_plane, lhs_row, lhs_column);
                    const std::size_t rhs_slot = detail::standard_plane_slot(
                            request.rhs.spec, rhs_plane, rhs_row, rhs_column);
                    const std::size_t out_slot = detail::standard_plane_slot(
                            request.out.spec, out_plane, row, column);
                    const std::uint64_t lhs = cpu_detail::load_bits(
                            lhs_base, lhs_slot * bits, bits);
                    const std::uint64_t rhs = cpu_detail::load_bits(
                            rhs_base, rhs_slot * bits, bits);
                    cpu_detail::store_bits(
                            out_base, out_slot * bits, bits,
                            scalar(request.out.spec.data_type, lhs, rhs));
                }
            }
        };
        visit(visit, 0);
    }

    oid binary_impl(const BinaryRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        const ScalarBinary scalar = select_binary(request.operation);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_binary(
                request, device_->registry_state(), registry_queue_id_, fence,
                [this, scalar](std::uint64_t sequence,
                               const BinaryRequest& captured,
                               detail::BinaryEntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, scalar, sequence, captured, entries] {
                                    std::exception_ptr failure;
                                    try {
                                        binary_elements(captured, scalar);
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    (void)detail::release_or_invalidate_binary_entries(
                                            device_->registry_state().registry,
                                            entries,
                                            static_cast<bool>(failure), true);
                                    detail::complete_workspace_lease(
                                            device_->registry_state(),
                                            captured.workspace_lease, true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        detail::complete_workspace_lease(
                                device_->registry_state(),
                                captured.workspace_lease, true);
                        throw;
                    }
                });
    }
    template <typename Operation>
    static void visit_embedding_planes(
            std::span<const std::size_t> dimensions,
            std::span<const std::size_t> index_strides,
            std::span<const std::size_t> output_strides,
            std::size_t leading_rank, std::size_t axis,
            std::size_t index_plane, std::size_t output_plane,
            Operation&& operation) {
        if (axis == leading_rank) {
            operation(index_plane, output_plane);
            return;
        }
        for (std::size_t index = 0; index < dimensions[axis]; ++index) {
            visit_embedding_planes(
                    dimensions, index_strides, output_strides, leading_rank,
                    axis + 1,
                    index_plane + index * index_strides[axis],
                    output_plane + index * output_strides[axis], operation);
        }
    }

    static bool embedding_index_is_signed(DataType type) noexcept {
        switch (type) {
            case DataType::I2:
            case DataType::I4:
            case DataType::I8:
            case DataType::I16:
            case DataType::I32:
            case DataType::I64:
                return true;
            case DataType::U2:
            case DataType::U4:
            case DataType::U8:
            case DataType::U16:
            case DataType::U32:
            case DataType::U64:
                return false;
            default:
                return false;
        }
    }

    static void embedding_elements(const EmbeddingRequest& request) {
        const auto table_dimensions = request.table.spec.shape.dimensions();
        const auto index_dimensions = request.indices.spec.shape.dimensions();
        const auto output_dimensions = request.out.spec.shape.dimensions();
        const std::size_t leading_rank = index_dimensions.size() - 2;
        const std::size_t run = index_dimensions.back();
        const std::size_t features = table_dimensions.back();
        const std::size_t vocabulary = table_dimensions.front();
        const std::size_t payload_bits =
                detail::leaf_bits(request.table.spec.data_type);
        const std::size_t index_bits =
                detail::leaf_bits(request.indices.spec.data_type);
        const std::uint64_t index_mask = index_bits == 64
                ? std::numeric_limits<std::uint64_t>::max()
                : (std::uint64_t{1} << index_bits) - 1;
        const bool index_is_signed =
                embedding_index_is_signed(request.indices.spec.data_type);
        const auto* table_base =
                static_cast<const unsigned char*>(request.table.native_handle);
        const auto* index_base =
                static_cast<const unsigned char*>(request.indices.native_handle);
        auto* output_base =
                static_cast<unsigned char*>(request.out.native_handle);

        const auto validate_indices = [&](std::size_t index_plane,
                                          std::size_t) {
            for (std::size_t row = 0; row < run; ++row) {
                const std::size_t index_slot = detail::standard_plane_slot(
                        request.indices.spec, index_plane, 0, row);
                const std::uint64_t raw = cpu_detail::load_bits(
                        index_base, index_slot * index_bits, index_bits)
                        & index_mask;
                if (index_is_signed
                        && (raw & (std::uint64_t{1} << (index_bits - 1)))
                                != 0) {
                    throw std::invalid_argument(
                            "CPU embedding index is negative");
                }
                if (raw >= static_cast<std::uint64_t>(vocabulary)) {
                    throw std::invalid_argument(
                            "CPU embedding index is out of range");
                }
            }
        };
        visit_embedding_planes(
                index_dimensions, request.indices.plane_strides,
                request.out.plane_strides, leading_rank, 0,
                request.indices.plane_offset, request.out.plane_offset,
                validate_indices);

        const auto gather = [&](std::size_t index_plane,
                                std::size_t output_plane) {
            for (std::size_t row = 0; row < run; ++row) {
                const std::size_t index_slot = detail::standard_plane_slot(
                        request.indices.spec, index_plane, 0, row);
                const std::uint64_t raw = cpu_detail::load_bits(
                        index_base, index_slot * index_bits, index_bits)
                        & index_mask;
                const std::size_t table_row = static_cast<std::size_t>(raw);
                for (std::size_t feature = 0; feature < features; ++feature) {
                    const std::size_t table_slot = detail::standard_plane_slot(
                            request.table.spec, request.table.plane_offset,
                            table_row, feature);
                    const std::size_t output_slot = detail::standard_plane_slot(
                            request.out.spec, output_plane, row, feature);
                    cpu_detail::copy_value(
                            output_base, output_slot * payload_bits, table_base,
                            table_slot * payload_bits, payload_bits);
                }
            }
        };
        visit_embedding_planes(
                output_dimensions, request.indices.plane_strides,
                request.out.plane_strides, leading_rank, 0,
                request.indices.plane_offset, request.out.plane_offset,
                gather);
    }

    oid embedding_impl(const EmbeddingRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_embedding(
                request, device_->registry_state(), registry_queue_id_, fence,
                [this](
                        std::uint64_t sequence,
                        const EmbeddingRequest& captured,
                        detail::BinaryEntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, sequence, captured, entries] {
                                    std::exception_ptr failure;
                                    try {
                                        embedding_elements(captured);
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    (void)detail::release_or_invalidate_binary_entries(
                                            device_->registry_state().registry,
                                            entries,
                                            static_cast<bool>(failure), true);
                                    detail::complete_workspace_lease(
                                            device_->registry_state(),
                                            captured.workspace_lease, true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        detail::complete_workspace_lease(
                                device_->registry_state(),
                                captured.workspace_lease, true);
                        throw;
                    }
                });
    }

    WorkspaceRequirements embedding_workspace_requirements_impl(
            const TensorView&, const TensorView&, const TensorView&) override {
        return {0, 1};
    }

    // The nine applicable ordinary signed floating leaves. Recognized
    // inapplicable leaves are rejected by common admission before this
    // predicate is consulted, and an unknown enumeration value never reaches
    // it either.
    [[nodiscard]] bool rmsnorm_supported(DataType data_type) const override {
        switch (data_type) {
            case DataType::F4_E2M1:
            case DataType::F6_E2M3:
            case DataType::F6_E3M2:
            case DataType::F8_E4M3FN:
            case DataType::F8_E5M2:
            case DataType::F16:
            case DataType::BF16:
            case DataType::F32:
            case DataType::F64:
                return true;
            default:
                return false;
        }
    }

    template <typename Carrier>
    [[nodiscard]] static Carrier decode_element(
            const unsigned char* base, const TensorSpec& spec,
            const RmsnormFormat& format, std::size_t plane, std::size_t row,
            std::size_t column) {
        return static_cast<Carrier>(detail::scalar_add_detail::decode_small(
                cpu_detail::load_logical_element(
                        base, spec, plane, row, column),
                format));
    }

    template <typename Carrier>
    static void encode_element(
            unsigned char* base, const TensorSpec& spec,
            const RmsnormFormat& format, std::size_t plane, std::size_t row,
            std::size_t column, Carrier value) {
        // The result of a row is encoded exactly once, and a NaN result is
        // stored in its canonical positive form by clearing the sign bit:
        // NaN payloads and NaN signs are outside the contract, and the
        // positive canonical form is the only one that also satisfies the
        // published comparison for the narrow finite-only leaves, where a NaN
        // saturates to a finite encoding instead of a NaN one.
        if (std::isnan(value)) {
            value = std::fabs(value);
        }
        cpu_detail::store_logical_element(
                base, spec, plane, row, column,
                detail::scalar_add_detail::encode_small(
                        static_cast<long double>(value), format));
    }

    // One independent leading plane. Every row reduces exactly its own
    // logical features in the accumulator domain of its leaf, and each
    // dependent value is rounded there before the single destination encode.
    template <typename Carrier>
    static void rmsnorm_plane(
            const RmsnormRequest& request, const RmsnormFormat& format,
            std::size_t x_plane, std::size_t out_plane, std::size_t rows,
            std::size_t features) {
        const TensorSpec& x_spec = request.x.spec;
        const TensorSpec& scale_spec = request.scale.spec;
        const TensorSpec& out_spec = request.out.spec;
        const auto* x_base =
                static_cast<const unsigned char*>(request.x.native_handle);
        const auto* scale_base =
                static_cast<const unsigned char*>(request.scale.native_handle);
        auto* out_base =
                static_cast<unsigned char*>(request.out.native_handle);
        const std::size_t scale_plane = request.scale.plane_offset;
        const Carrier divisor = static_cast<Carrier>(features);
        const Carrier epsilon = static_cast<Carrier>(request.epsilon);
        for (std::size_t row = 0; row < rows; ++row) {
            Carrier sum = static_cast<Carrier>(0);
            for (std::size_t feature = 0; feature < features; ++feature) {
                const Carrier value = decode_element<Carrier>(
                        x_base, x_spec, format, x_plane, row, feature);
                sum = sum + value * value;
            }
            const Carrier mean = sum / divisor;
            const Carrier shifted = mean + epsilon;
            const Carrier root = std::sqrt(shifted);
            const Carrier inverse = static_cast<Carrier>(1) / root;
            for (std::size_t feature = 0; feature < features; ++feature) {
                const Carrier value = decode_element<Carrier>(
                        x_base, x_spec, format, x_plane, row, feature);
                const Carrier scale = decode_element<Carrier>(
                        scale_base, scale_spec, format, scale_plane, 0,
                        feature);
                const Carrier normalized = value * inverse;
                encode_element<Carrier>(
                        out_base, out_spec, format, out_plane, row, feature,
                        normalized * scale);
            }
        }
    }

    // Walk every independent leading plane of `x` and `out` through their own
    // view offset and leading strides, then reduce that plane's rows. Only
    // the final two logical axes are addressed: rows are reduced over their
    // own features, and no padding, other row, or other plane contributes.
    template <typename Carrier>
    static void rmsnorm_elements(const RmsnormRequest& request) {
        const TensorSpec& spec = request.x.spec;
        const std::span<const std::size_t> dimensions = spec.shape.dimensions();
        const std::size_t rank = dimensions.size();
        const std::size_t rows = dimensions[rank - 2];
        const std::size_t features = dimensions[rank - 1];
        const RmsnormFormat format =
                detail::scalar_add_detail::format(spec.data_type);
        const std::span<const std::size_t> x_strides = request.x.plane_strides;
        const std::span<const std::size_t> out_strides =
                request.out.plane_strides;
        auto visit = [&](auto&& self, std::size_t axis, std::size_t x_plane,
                         std::size_t out_plane) -> void {
            if (axis + 2 == rank) {
                rmsnorm_plane<Carrier>(
                        request, format, x_plane, out_plane, rows, features);
                return;
            }
            for (std::size_t index = 0; index < dimensions[axis]; ++index) {
                self(self, axis + 1, x_plane + index * x_strides[axis],
                     out_plane + index * out_strides[axis]);
            }
        };
        visit(visit, 0, request.x.plane_offset, request.out.plane_offset);
    }

    // The value-captured immutable request is registered and retained by the
    // shared admission path; the kernel itself runs only on the existing
    // FIFO worker, with no workspace lease because RMS normalization
    // consumes no raw workspace.
    oid rmsnorm_impl(const RmsnormRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_rmsnorm(
                request, device_->registry_state(), registry_queue_id_, fence,
                [this](std::uint64_t sequence, const RmsnormRequest& captured,
                       detail::BinaryEntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, sequence, captured, entries] {
                                    std::exception_ptr failure;
                                    try {
                                        if (captured.x.spec.data_type
                                                == DataType::F64) {
                                            rmsnorm_elements<double>(captured);
                                        } else {
                                            rmsnorm_elements<float>(captured);
                                        }
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    (void)detail::release_or_invalidate_binary_entries(
                                            device_->registry_state().registry,
                                            entries, static_cast<bool>(failure),
                                            true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        throw;
                    }
                });
    }
    using RopeFormat = detail::scalar_binary_codec_detail::Format;

    template <typename Carrier>
    static Carrier rope_decode(
            const unsigned char* base,
            std::span<const std::size_t> dimensions,
            DataType data_type, const RopeFormat& format,
            std::size_t plane, std::size_t row, std::size_t column) {
        return CpuRopeCodec<Carrier>::decode(
                cpu_detail::load_logical_element(
                        base, dimensions, data_type, plane, row, column),
                format);
    }

    template <typename Carrier>
    static void rope_pair(
            const RopeRequest& request, const RopeFormat& format,
            const unsigned char* x_base, unsigned char* out_base,
            std::span<const std::size_t> x_dimensions,
            std::span<const std::size_t> out_dimensions,
            std::size_t x_plane, std::size_t out_plane, std::size_t row,
            std::size_t first_column, Carrier theta) {
        const std::size_t width = x_dimensions.back();
        const std::size_t half = width / 2;
        const std::size_t second_column = first_column + half;
        const Carrier first = rope_decode<Carrier>(
                x_base, x_dimensions, request.x.data_type, format, x_plane,
                row, first_column);
        const Carrier second = rope_decode<Carrier>(
                x_base, x_dimensions, request.x.data_type, format, x_plane,
                row, second_column);
        const Carrier exponent = static_cast<Carrier>(
                (-2.0 * static_cast<double>(first_column))
                / static_cast<double>(width));
        const Carrier frequency = std::pow(theta, exponent);
        const Carrier position = static_cast<Carrier>(request.a + row);
        const Carrier angle = position * frequency;
        const Carrier sine = std::sin(angle);
        const Carrier cosine = std::cos(angle);

        // Keep the two products and the following add/subtract as separate
        // operations. Volatile intermediates prevent contraction into FMA
        // while preserving the selected FP32/FP64 carrier domain.
        const volatile Carrier first_product = first * cosine;
        const volatile Carrier second_product = second * sine;
        const volatile Carrier first_output =
                first_product - second_product;
        const volatile Carrier second_cosine_product = second * cosine;
        const volatile Carrier first_sine_product = first * sine;
        const volatile Carrier second_output =
                second_cosine_product + first_sine_product;

        cpu_detail::store_logical_element(
                out_base, out_dimensions, request.out.data_type, out_plane,
                row, first_column,
                CpuRopeCodec<Carrier>::encode(first_output, format));
        cpu_detail::store_logical_element(
                out_base, out_dimensions, request.out.data_type, out_plane,
                row, second_column,
                CpuRopeCodec<Carrier>::encode(second_output, format));
    }

    template <typename Carrier>
    static void rope_plane(
            const RopeRequest& request, const RopeFormat& format,
            std::size_t x_plane, std::size_t out_plane,
            Carrier theta) {
        const std::span<const std::size_t> x_dimensions =
                request.x.shape_dimensions();
        const std::span<const std::size_t> out_dimensions =
                request.out.shape_dimensions();
        const std::size_t rank = x_dimensions.size();
        const std::size_t rows = x_dimensions[rank - 2];
        const std::size_t width = x_dimensions[rank - 1];
        const std::size_t half = width / 2;
        const std::size_t row_tiles =
                rows / TensorSpec::TILE
                + (rows % TensorSpec::TILE != 0);
        const std::size_t feature_tiles =
                width / TensorSpec::TILE
                + (width % TensorSpec::TILE != 0);
        const std::size_t pair_tiles =
                half / TensorSpec::TILE
                + (half % TensorSpec::TILE != 0);
        const auto* x_base =
                static_cast<const unsigned char*>(request.x.native_handle);
        auto* out_base =
                static_cast<unsigned char*>(request.out.native_handle);

        for (std::size_t tile_row = 0; tile_row < row_tiles; ++tile_row) {
            const std::size_t first_row = tile_row * TensorSpec::TILE;
            for (std::size_t row_in_tile = 0;
                 row_in_tile < TensorSpec::TILE
                         && first_row + row_in_tile < rows;
                 ++row_in_tile) {
                const std::size_t row = first_row + row_in_tile;
                const std::size_t position = request.a + row;
                if (position == 0) {
                    for (std::size_t tile_column = 0;
                         tile_column < feature_tiles; ++tile_column) {
                        const std::size_t first_column =
                                tile_column * TensorSpec::TILE;
                        for (std::size_t column_in_tile = 0;
                             column_in_tile < TensorSpec::TILE
                                     && first_column + column_in_tile < width;
                             ++column_in_tile) {
                            const std::size_t column =
                                    first_column + column_in_tile;
                            const std::uint64_t raw =
                                    cpu_detail::load_logical_element(
                                            x_base, x_dimensions,
                                            request.x.data_type, x_plane, row,
                                            column);
                            cpu_detail::store_logical_element(
                                    out_base, out_dimensions,
                                    request.out.data_type, out_plane, row,
                                    column, raw);
                        }
                    }
                    continue;
                }

                for (std::size_t tile_column = 0;
                     tile_column < pair_tiles; ++tile_column) {
                    const std::size_t first_column =
                            tile_column * TensorSpec::TILE;
                    for (std::size_t column_in_tile = 0;
                         column_in_tile < TensorSpec::TILE
                                 && first_column + column_in_tile < half;
                         ++column_in_tile) {
                        rope_pair<Carrier>(
                                request, format, x_base, out_base,
                                x_dimensions, out_dimensions, x_plane,
                                out_plane, row,
                                first_column + column_in_tile, theta);
                    }
                }
            }
        }
    }

    template <typename Carrier>
    static void rope_elements(const RopeRequest& request) {
        const std::span<const std::size_t> dimensions =
                request.x.shape_dimensions();
        const std::size_t leading_rank = dimensions.size() - 3;
        const std::size_t heads = dimensions[leading_rank];
        const Carrier theta = static_cast<Carrier>(request.theta);
        const RopeFormat format =
                CpuRopeCodec<Carrier>::format(request.x.data_type);
        const std::span<const std::size_t> x_strides =
                request.x.leading_plane_strides();
        const std::span<const std::size_t> out_strides =
                request.out.leading_plane_strides();
        auto visit = [&](auto&& self, std::size_t axis,
                         std::size_t x_plane, std::size_t out_plane) -> void {
            if (axis == leading_rank) {
                const std::size_t x_head_stride = x_strides[leading_rank];
                const std::size_t out_head_stride = out_strides[leading_rank];
                for (std::size_t head = 0; head < heads; ++head) {
                    rope_plane<Carrier>(
                            request, format,
                            x_plane + head * x_head_stride,
                            out_plane + head * out_head_stride, theta);
                }
                return;
            }
            for (std::size_t index = 0; index < dimensions[axis]; ++index) {
                self(self, axis + 1,
                     x_plane + index * x_strides[axis],
                     out_plane + index * out_strides[axis]);
            }
        };
        visit(visit, 0, request.x.plane_offset, request.out.plane_offset);
    }

    oid rope_impl(const RopeRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_rope(
                request, device_->registry_state(), registry_queue_id_, fence,
                [this](std::uint64_t sequence, const RopeRequest& captured,
                       detail::BinaryEntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, sequence, captured, entries] {
                                    std::exception_ptr failure;
                                    try {
                                        if (captured.x.data_type
                                                == DataType::F64) {
                                            rope_elements<double>(captured);
                                        } else {
                                            rope_elements<float>(captured);
                                        }
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    (void)detail::release_or_invalidate_binary_entries(
                                            device_->registry_state().registry,
                                            entries,
                                            static_cast<bool>(failure), true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        throw;
                    }
                });
    }

    [[nodiscard]] WorkspaceRequirements rope_workspace_requirements(
            const RopeRequest&) override {
        return {0, 1};
    }
    // ------------------------------------------------------------------
    // SDPA: the common facade has completed all structural, view, device,
    // dtype, alias, and workspace admission before this hook runs. The CPU
    // adapter below copies only fixed-size metadata into the worker request;
    // no borrowed TensorView reaches the FIFO task.
    static cpu_detail::SdpaView make_cpu_sdpa_view(
            const SdpaViewSnapshot& view) {
        cpu_detail::SdpaView result;
        result.rank = view.rank;
        for (std::size_t axis = 0; axis < view.rank; ++axis) {
            result.dimensions[axis] = view.dimensions[axis];
        }
        const std::size_t leading = view.rank - 2;
        for (std::size_t axis = 0; axis < leading; ++axis) {
            result.plane_strides[axis] = view.plane_strides[axis];
        }
        result.plane_offset = view.plane_offset;
        result.native_handle =
                static_cast<unsigned char*>(view.native_handle);
        return result;
    }

    static cpu_detail::SdpaRequest make_cpu_sdpa_request(
            const SdpaRequest& request) {
        return {
                make_cpu_sdpa_view(request.q),
                make_cpu_sdpa_view(request.k),
                make_cpu_sdpa_view(request.v),
                make_cpu_sdpa_view(request.out),
                request.a,
                request.L,
                request.Hq,
                request.Hkv,
                request.R,
                request.C,
                request.D,
                request.grouping,
                request.output_width,
                static_cast<unsigned char*>(
                        detail::WorkspaceValidation::address(
                                request.workspace))};
    }

    [[nodiscard]] static std::size_t sdpa_leading_planes(
            const SdpaRequest& request) {
        const std::size_t leading_rank = request.q.rank - 3;
        std::size_t planes = 1;
        for (std::size_t axis = 0; axis < leading_rank; ++axis) {
            planes = detail::checked_mul(
                    planes, request.q.dimensions[axis],
                    "CPU SDPA leading plane count overflows");
        }
        return planes;
    }

    oid sdpa_impl(const SdpaRequest& request) override {
        const cpu_detail::SdpaRequest cpu_request =
                make_cpu_sdpa_request(request);
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_sdpa(
                request, device_->registry_state(), registry_queue_id_, fence,
                [this, cpu_request](
                        std::uint64_t sequence,
                        const SdpaRequest& captured,
                        detail::SdpaEntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, sequence, captured, entries,
                                 cpu_request] {
                                    std::exception_ptr failure;
                                    try {
                                        if (cpu_detail::consume_sdpa_failure()) {
                                            throw std::runtime_error(
                                                    "CPU SDPA injected "
                                                    "post-acceptance failure");
                                        }
                                        cpu_detail::sdpa_elements(cpu_request);
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    (void)
                                            detail::release_or_invalidate_sdpa_entries(
                                                    device_->registry_state()
                                                            .registry,
                                                    entries,
                                                    static_cast<bool>(failure),
                                                    true);
                                    detail::complete_workspace_lease(
                                            device_->registry_state(),
                                            captured.workspace_lease, true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        detail::complete_workspace_lease(
                                device_->registry_state(),
                                captured.workspace_lease, true);
                        throw;
                    }
                });
    }

    [[nodiscard]] WorkspaceRequirements
            sdpa_workspace_requirements_impl(
                    const SdpaRequest& request) override {
        return cpu_detail::sdpa_workspace_requirements(
                sdpa_leading_planes(request), request.Hq, request.R,
                request.L);
    }



    // ------------------------------------------------------------------
    // SiLU: an asynchronous, in-order scalar activation over caller-owned tiled
    // storage. Nothing below allocates: the admitted request was already
    // captured by value, every element is addressed through the shared checked
    // layout helpers, and `src/shared/scalar_silu.hpp` is the only evaluator —
    // it decodes the named destination format, evaluates the stable expression,
    // and encodes each logical element exactly once with destination RNE. Only
    // the selected logical element of a plane is read and only its own output
    // element is written, so no padding, other plane, or other run
    // contributes.

    // Every independent leading plane of `x` and `y`, through its own selected
    // plane offset and own transformed leading strides. The carrier is the CPU
    // leaf's established evaluation split: FP32 through every leaf below F64,
    // and FP64 for F64 itself, exactly as the RMSNorm, RoPE, and linear paths
    // select it.
    template <typename Carrier>
    static void silu_elements(const SiLURequest& request) {
        const std::span<const std::size_t> dimensions =
                request.x.shape_dimensions();
        const std::size_t rank = dimensions.size();
        const std::size_t runs = dimensions[rank - 2];
        const std::size_t features = dimensions[rank - 1];
        const DataType data_type = request.x.data_type;
        const auto* x_base =
                static_cast<const unsigned char*>(request.x.native_handle);
        auto* y_base =
                static_cast<unsigned char*>(request.y.native_handle);
        const std::span<const std::size_t> x_strides =
                request.x.leading_plane_strides();
        const std::span<const std::size_t> y_strides =
                request.y.leading_plane_strides();
        auto visit = [&](auto&& self, std::size_t axis, std::size_t x_plane,
                         std::size_t y_plane) -> void {
            if (axis + 2 == rank) {
                for (std::size_t run = 0; run < runs; ++run) {
                    for (std::size_t feature = 0; feature < features;
                         ++feature) {
                        const std::uint64_t raw =
                                cpu_detail::load_logical_element(
                                        x_base, dimensions, data_type, x_plane,
                                        run, feature);
                        cpu_detail::store_logical_element(
                                y_base, dimensions, data_type, y_plane, run,
                                feature,
                                detail::scalar_silu_detail::scalar_silu<
                                        CpuCarrierTraits<Carrier>>(
                                        data_type, raw));
                    }
                }
                return;
            }
            for (std::size_t index = 0; index < dimensions[axis]; ++index) {
                self(self, axis + 1, x_plane + index * x_strides[axis],
                     y_plane + index * y_strides[axis]);
            }
        };
        visit(visit, 0, request.x.plane_offset, request.y.plane_offset);
    }

    // The value-captured request keeps every distinct owner registered until
    // this sequence's completion is proven, so temporary views and a released
    // owner stay safe; the kernel itself runs only on the existing FIFO worker,
    // never on the caller thread. SiLU consumes no raw workspace, so no lease
    // is acquired, and a failed completion invalidates its owners through the
    // shared retainer rather than promising any rollback.
    oid silu_impl(const SiLURequest& request) override {
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_silu(
                request, device_->registry_state(), registry_queue_id_, fence,
                [this](std::uint64_t sequence, const SiLURequest& captured,
                       detail::BinaryEntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, sequence, captured, entries] {
                                    std::exception_ptr failure;
                                    try {
                                        if (cpu_detail::consume_silu_failure()) {
                                            throw std::runtime_error(
                                                    "CPU SiLU injected "
                                                    "post-acceptance failure");
                                        }
                                        if (captured.x.data_type
                                                == DataType::F64) {
                                            silu_elements<double>(captured);
                                        } else {
                                            silu_elements<float>(captured);
                                        }
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    (void)detail::release_or_invalidate_binary_entries(
                                            device_->registry_state().registry,
                                            entries,
                                            static_cast<bool>(failure), true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        throw;
                    }
                });
    }

    [[nodiscard]] WorkspaceRequirements silu_workspace_requirements_impl(
            const SiLURequest&) override {
        return {0, 1};
    }


    // ------------------------------------------------------------------
    // Linear projection: an asynchronous, in-order scalar dot product over
    // caller-owned tiled storage. Nothing below allocates: the operation owns
    // no tensor, staging, or scratch range, and the admitted request was
    // already captured by value. Every element is addressed through the
    // shared checked layout helpers and only logical `I`/`O` coordinates are
    // touched, so no padding element is ever read and every output padding bit
    // keeps its caller-provided value.

    using LinearFormat = detail::scalar_add_detail::Format;

    // One projected element of an integer leaf. Every multiply and every add
    // reduces modulo `2^N` in unsigned arithmetic, which is the contract's
    // stepwise form of the exact sum: no step depends on signed overflow and
    // no value is ever cast to an out-of-range signed type. The stored code is
    // the two's-complement bit pattern of the accumulated value.
    static void linear_integer_element(
            const LinearRequest& request, std::size_t x_plane,
            std::size_t source_row, std::size_t weight_row,
            std::size_t out_plane, std::size_t out_row,
            std::size_t out_column) {
        const TensorSpec& x_spec = request.x.spec;
        const std::size_t bits = detail::leaf_bits(x_spec.data_type);
        const std::size_t features = x_spec.shape.dimensions().back();
        const std::uint64_t mask = bits == 64
                ? std::numeric_limits<std::uint64_t>::max()
                : (std::uint64_t{1} << bits) - 1;
        const auto* x_base =
                static_cast<const unsigned char*>(request.x.native_handle);
        const auto* w_base =
                static_cast<const unsigned char*>(request.w.native_handle);
        std::uint64_t accumulator = 0;
        for (std::size_t first = 0; first < features;
             first += TensorSpec::TILE) {
            const std::size_t remaining = features - first;
            const std::size_t extent = remaining < TensorSpec::TILE
                    ? remaining : TensorSpec::TILE;
            const std::size_t x_bit = cpu_detail::logical_element_bits(
                    x_spec, x_plane, source_row, first);
            const std::size_t w_bit = cpu_detail::logical_element_bits(
                    request.w.spec, request.w.plane_offset, weight_row, first);
            for (std::size_t index = 0; index < extent; ++index) {
                const std::uint64_t lhs = cpu_detail::load_bits(
                        x_base, x_bit + index * bits, bits) & mask;
                const std::uint64_t rhs = cpu_detail::load_bits(
                        w_base, w_bit + index * bits, bits) & mask;
                accumulator = (accumulator + lhs * rhs) & mask;
            }
        }
        cpu_detail::store_logical_element(
                static_cast<unsigned char*>(request.out.native_handle),
                request.out.spec, out_plane, out_row, out_column,
                accumulator);
    }

    // One projected element of a floating leaf. The recurrence starts at `+0`,
    // visits increasing `i`, performs one correctly rounded fused
    // multiply-add per step, and encodes the accumulated sum exactly once with
    // the existing named-format rules. Every floating leaf below `F64` decodes
    // exactly into FP32, and `F64` accumulates in FP64 with no narrowing.
    template <typename Carrier>
    static void linear_float_element(
            const LinearRequest& request, const LinearFormat& format,
            std::size_t x_plane, std::size_t source_row,
            std::size_t weight_row, std::size_t out_plane,
            std::size_t out_row, std::size_t out_column) {
        const TensorSpec& x_spec = request.x.spec;
        const std::size_t bits = detail::leaf_bits(x_spec.data_type);
        const std::size_t features = x_spec.shape.dimensions().back();
        const auto* x_base =
                static_cast<const unsigned char*>(request.x.native_handle);
        const auto* w_base =
                static_cast<const unsigned char*>(request.w.native_handle);
        Carrier accumulator = static_cast<Carrier>(0);
        for (std::size_t first = 0; first < features;
             first += TensorSpec::TILE) {
            const std::size_t remaining = features - first;
            const std::size_t extent = remaining < TensorSpec::TILE
                    ? remaining : TensorSpec::TILE;
            const std::size_t x_bit = cpu_detail::logical_element_bits(
                    x_spec, x_plane, source_row, first);
            const std::size_t w_bit = cpu_detail::logical_element_bits(
                    request.w.spec, request.w.plane_offset, weight_row, first);
            for (std::size_t index = 0; index < extent; ++index) {
                const Carrier lhs = static_cast<Carrier>(
                        detail::scalar_add_detail::decode_small(
                                cpu_detail::load_bits(
                                        x_base, x_bit + index * bits, bits),
                                format));
                const Carrier rhs = static_cast<Carrier>(
                        detail::scalar_add_detail::decode_small(
                                cpu_detail::load_bits(
                                        w_base, w_bit + index * bits, bits),
                                format));
                accumulator = std::fma(lhs, rhs, accumulator);
            }
        }
        cpu_detail::store_logical_element(
                static_cast<unsigned char*>(request.out.native_handle),
                request.out.spec, out_plane, out_row, out_column,
                detail::scalar_add_detail::encode_small(
                        static_cast<long double>(accumulator), format));
    }

    // Walk every independent leading plane of `x` and `out` through their own
    // selected plane offset and own transformed leading strides, then every
    // selected source row from `start_row` and every head. `ordinary` projects
    // one output column per weight row; `head_planar` projects the `D` columns
    // of each of the `H` head planes the output view inserts, selecting the
    // weight row `h*D+d` and addressing the head plane through the output
    // view's own head stride.
    template <typename ProjectElement>
    static void linear_planes(
            const LinearRequest& request, ProjectElement&& project_element) {
        const std::span<const std::size_t> dimensions =
                request.x.spec.shape.dimensions();
        const std::size_t leading_rank = dimensions.size() - 2;
        const std::size_t outer = request.w.spec.shape.dimensions()[0];
        const bool head_planar =
                request.layout == LinearOutputLayout::head_planar;
        const std::size_t heads = head_planar ? request.heads : 1;
        const std::size_t head_dim = head_planar ? request.head_dim : outer;
        // Only head-planar mode inserts the head axis between the leading
        // tuple and the row axis, so only that mode has a head stride.
        const std::size_t head_stride =
                head_planar ? request.out.plane_strides[leading_rank] : 0;
        const auto project = [&](std::size_t x_plane, std::size_t out_plane) {
            for (std::size_t row = 0; row < request.rows; ++row) {
                const std::size_t source_row = request.start_row + row;
                for (std::size_t head = 0; head < heads; ++head) {
                    const std::size_t head_plane =
                            out_plane + head * head_stride;
                    for (std::size_t column = 0; column < head_dim; ++column) {
                        project_element(
                                x_plane, source_row,
                                head * head_dim + column, head_plane, row,
                                column);
                    }
                }
            }
        };
        auto visit = [&](auto&& self, std::size_t axis, std::size_t x_plane,
                         std::size_t out_plane) -> void {
            if (axis == leading_rank) {
                project(x_plane, out_plane);
                return;
            }
            for (std::size_t index = 0; index < dimensions[axis]; ++index) {
                self(self, axis + 1,
                     x_plane + index * request.x.plane_strides[axis],
                     out_plane + index * request.out.plane_strides[axis]);
            }
        };
        visit(visit, 0, request.x.plane_offset, request.out.plane_offset);
    }

    template <typename Carrier>
    static void linear_floating_elements(const LinearRequest& request) {
        const LinearFormat format =
                detail::scalar_add_detail::format(request.x.spec.data_type);
        linear_planes(
                request,
                [&request, format](
                        std::size_t x_plane, std::size_t source_row,
                        std::size_t weight_row, std::size_t out_plane,
                        std::size_t out_row, std::size_t out_column) {
                    linear_float_element<Carrier>(
                            request, format, x_plane, source_row, weight_row,
                            out_plane, out_row, out_column);
                });
    }

    static void linear_elements(const LinearRequest& request) {
        const DataType data_type = request.x.spec.data_type;
        if (data_type == DataType::F64) {
            linear_floating_elements<double>(request);
            return;
        }
        if (detail::scalar_add_detail::format(data_type).bits != 0) {
            linear_floating_elements<float>(request);
            return;
        }
        linear_planes(
                request,
                [&request](
                        std::size_t x_plane, std::size_t source_row,
                        std::size_t weight_row, std::size_t out_plane,
                        std::size_t out_row, std::size_t out_column) {
                    linear_integer_element(
                            request, x_plane, source_row, weight_row, out_plane,
                            out_row, out_column);
                });
    }

    // The recorded contract reports exactly `{0, 1}`, so the empty default
    // view is the only admissible raw workspace and a supplied owner is
    // invalid input, here and before any owner registration or sequence
    // consumption. The kernel itself runs only on the existing FIFO worker,
    // and the captured value snapshot keeps every distinct owner registered
    // until that sequence's completion is proven.
    oid linear_impl(const LinearRequest& request) override {
        if (!request.workspace.empty()) {
            throw std::invalid_argument(
                    "CPU linear projection consumes no raw workspace");
        }
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        detail::Fence fence;
        fence.invoke = &fence_pending;
        return submit_linear(
                request, device_->registry_state(), registry_queue_id_, fence,
                [this](std::uint64_t sequence, const LinearRequest& captured,
                       detail::BinaryEntryRegistration entries) {
                    try {
                        enqueue(HostTask{
                                [this, sequence, captured, entries] {
                                    std::exception_ptr failure;
                                    try {
                                        if (cpu_detail::consume_linear_failure()) {
                                            throw std::runtime_error(
                                                    "CPU linear projection "
                                                    "injected post-acceptance "
                                                    "failure");
                                        }
                                        linear_elements(captured);
                                    } catch (...) {
                                        failure = std::current_exception();
                                    }
                                    (void)detail::release_or_invalidate_binary_entries(
                                            device_->registry_state().registry,
                                            entries,
                                            static_cast<bool>(failure), true);
                                    detail::complete_workspace_lease(
                                            device_->registry_state(),
                                            captured.workspace_lease, true);
                                    complete(sequence, std::move(failure));
                                }});
                    } catch (...) {
                        device_->registry_state().registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        detail::complete_workspace_lease(
                                device_->registry_state(),
                                captured.workspace_lease, true);
                        throw;
                    }
                });
    }

    WorkspaceRequirements linear_workspace_requirements_impl(
            const TensorView&, const TensorView&, const TensorView&,
            std::size_t, std::size_t, LinearOutputLayout, std::size_t,
            std::size_t) override {
        return {0, 1};
    }

    static void cache_append_elements(const CacheAppendRequest& request) {
        const std::span<const std::size_t> source_dimensions =
                request.source.shape_dimensions();
        const std::span<const std::size_t> destination_dimensions =
                request.destination.shape_dimensions();
        const std::size_t leading_rank = source_dimensions.size() - 2;
        const std::size_t rows = source_dimensions[leading_rank];
        const std::size_t features = source_dimensions[leading_rank + 1];
        const auto* source_base =
                static_cast<const unsigned char*>(
                        request.source.native_handle);
        auto* destination_base =
                static_cast<unsigned char*>(
                        request.destination.native_handle);

        auto visit = [&](auto&& self, std::size_t axis,
                         std::size_t source_plane,
                         std::size_t destination_plane) -> void {
            if (axis == leading_rank) {
                for (std::size_t row = 0; row < rows; ++row) {
                    cpu_detail::copy_row_span(
                            destination_base, destination_dimensions,
                            destination_plane, request.a + row, source_base,
                            source_dimensions, source_plane, row,
                            request.source.data_type, features);
                }
                return;
            }
            for (std::size_t index = 0;
                 index < source_dimensions[axis]; ++index) {
                self(
                        self, axis + 1,
                        source_plane
                                + index
                                        * request.source
                                                  .leading_plane_strides()[axis],
                        destination_plane
                                + index
                                        * request.destination
                                                  .leading_plane_strides()[axis]);
            }
        };
        visit(
                visit, 0, request.source.plane_offset,
                request.destination.plane_offset);
    }

    static void copy_elements(
            const CopyViewSnapshot& source,
            const CopyViewSnapshot& destination) {
        const auto* source_base =
                static_cast<const unsigned char*>(source.native_handle);
        auto* destination_base =
                static_cast<unsigned char*>(destination.native_handle);
        const std::size_t bits = detail::leaf_bits(source.spec.data_type);
        std::array<std::uint32_t, TensorSpec::TILE> shift_table{};
        std::array<std::uint32_t, TensorSpec::TILE> mask_table{};
        if (bits % 8 != 0) {
            for (std::size_t index = 0; index < TensorSpec::TILE; ++index) {
                shift_table[index] =
                        (index * bits) % (sizeof(std::uint32_t) * 8);
                mask_table[index] =
                        ((1u << bits) - 1u) << shift_table[index];
            }
        }
        cpu_detail::for_each_tile_lockstep(
                source.spec, source.plane_offset, source.plane_strides,
                destination.spec, destination.plane_offset,
                destination.plane_strides,
                [&](std::size_t, std::size_t, std::size_t source_byte,
                    std::size_t destination_byte, std::size_t elements) {
                    cpu_detail::copy_tile_row(
                            destination_base, destination_byte * 8,
                            source_base, source_byte * 8, elements, bits,
                            shift_table, mask_table);
                });
    }

    CpuDevice* device_;
    detail::QueueId registry_queue_id_;
    std::mutex submission_order_mutex_;
    std::mutex worker_mutex_;
    std::condition_variable worker_cv_;
    std::deque<HostTask> worker_tasks_;
    bool worker_shutdown_ = false;
    std::thread worker_;
};

std::unique_ptr<DeviceOps> CpuDevice::create_ops() {
    return std::make_unique<CpuQueue>(*this);
}

}  // namespace iom
