#include "device_internal.hpp"
#include "transfer_helpers.hpp"

#include <array>
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

namespace iom {

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
