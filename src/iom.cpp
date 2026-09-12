#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"
#include "iom/gpu_algorithm.hpp"

#include "iom_internal.hpp"
#include <algorithm>
#include <bitset>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace iom {

    using detail::bits_to_bytes;
    using detail::checked_add;
    using detail::checked_mul;
    using detail::kMaxTensorRank;
    using detail::UnsupportedOperation;



    void Device::reserve_queue_slot() const {
        std::lock_guard<std::mutex> lock(queue_registry_mutex_);
        if (live_queue_count_ == 4) {
            throw std::bad_alloc();
        }
        ++live_queue_count_;
    }

    void Device::release_queue_slot() const noexcept {
        std::lock_guard<std::mutex> lock(queue_registry_mutex_);
        if (live_queue_count_ != 0) {
            --live_queue_count_;
        }
    }

    namespace {

        // The only global queue state: the live eight-bit queue ids. Bit i
        // represents id i + 1. It never selects a backend, device, or
        // runtime context.
        std::mutex g_queue_ids_mutex;
        std::bitset<255> g_live_queue_ids;
        bool recognized_quantization(QuantizationFormat value) noexcept {
            switch (value) {
                case QuantizationFormat::NONE:
                case QuantizationFormat::INT8_SYMMETRIC:
                case QuantizationFormat::INT8_ASYMMETRIC:
                case QuantizationFormat::INT4_SYMMETRIC:
                case QuantizationFormat::INT4_ASYMMETRIC:
                case QuantizationFormat::OCP_MXFP4:
                case QuantizationFormat::OCP_MXFP8_E4M3:
                case QuantizationFormat::OCP_MXFP8_E5M2:
                case QuantizationFormat::NVIDIA_NVFP4:
                case QuantizationFormat::GGML_Q4_0:
                case QuantizationFormat::GGML_Q4_1:
                case QuantizationFormat::GGML_Q5_0:
                case QuantizationFormat::GGML_Q5_1:
                case QuantizationFormat::GGML_Q8_0:
                case QuantizationFormat::GGML_Q2_K:
                case QuantizationFormat::GGML_Q3_K:
                case QuantizationFormat::GGML_Q4_K:
                case QuantizationFormat::GGML_Q5_K:
                case QuantizationFormat::GGML_Q6_K:
                case QuantizationFormat::TT_BFP2:
                case QuantizationFormat::TT_BFP2A:
                case QuantizationFormat::TT_BFP4:
                case QuantizationFormat::TT_BFP4A:
                case QuantizationFormat::TT_BFP8:
                case QuantizationFormat::TT_BFP8A:
                    return true;
            }
            return false;
        }

        bool add_numeric_leaf(DataType value) noexcept {
            switch (value) {
                case DataType::I2: case DataType::U2:
                case DataType::I4: case DataType::U4:
                case DataType::I8: case DataType::U8:
                case DataType::I16: case DataType::U16:
                case DataType::I32: case DataType::U32:
                case DataType::I64: case DataType::U64:
                case DataType::F4_E2M1: case DataType::F6_E2M3:
                case DataType::F6_E3M2: case DataType::F8_E4M3FN:
                case DataType::F8_E5M2: case DataType::F16:
                case DataType::BF16: case DataType::F32:
                case DataType::F64:
                    return true;
                case DataType::BOOL: case DataType::F8_E8M0:
                    return false;
            }
            return false;
        }
        bool div_numeric_leaf(DataType value) noexcept {
            switch (value) {
                case DataType::F4_E2M1: case DataType::F6_E2M3:
                case DataType::F6_E3M2: case DataType::F8_E4M3FN:
                case DataType::F8_E5M2: case DataType::F16:
                case DataType::BF16: case DataType::F32:
                case DataType::F64:
                    return true;
                default:
                    return false;
            }
        }

        void validate_binary_spec(const TensorSpec& spec) {
            (void)detail::leaf_bits(spec.data_type);
            if (!recognized_quantization(spec.quantization)) {
                throw std::invalid_argument("unknown quantization format");
            }
            if (spec.shape.rank() < 2) {
                throw std::invalid_argument("ADD requires rank at least two");
            }
            if (spec.shape.rank() > kMaxTensorRank) {
                throw std::invalid_argument("ADD requires rank at most eight");
            }
            for (const std::size_t dimension : spec.shape.dimensions()) {
                if (dimension == 0) {
                    throw std::invalid_argument(
                            "ADD dimensions must be nonzero");
                }
            }
        }

        std::size_t checked_binary_plane_count(const TensorShape& shape) {
            std::size_t planes = 1;
            const std::span<const std::size_t> dimensions =
                    shape.dimensions();
            for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                planes = checked_mul(
                        planes, dimensions[i],
                        "ADD plane count overflows");
            }
            return planes;
        }

        void validate_binary_view(
                const Device& device, const TensorView& view) {
            const Tensor* owner = view.owner_identity();
            if (owner == nullptr) {
                throw std::invalid_argument("ADD view has no owner");
            }
            if (&view.device() != &device
                    || &owner->view().device() != &device
                    || owner->view().owner_identity() != owner) {
                throw std::invalid_argument(
                        "ADD view owner belongs to another device");
            }
            const void* handle = view.native_handle();
            if (handle == nullptr
                    || handle != owner->view().native_handle()) {
                throw std::invalid_argument(
                        "ADD view has no stable owner handle");
            }

            const TensorSpec& spec = view.spec();
            const TensorSpec& owner_spec = owner->view().spec();
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::span<const std::size_t> owner_dimensions =
                    owner_spec.shape.dimensions();
            if (spec.data_type != owner_spec.data_type
                    || spec.quantization != owner_spec.quantization
                    || dimensions[dimensions.size() - 2]
                            != owner_dimensions[owner_dimensions.size() - 2]
                    || dimensions.back() != owner_dimensions.back()) {
                throw std::invalid_argument(
                        "ADD view specification does not match its owner");
            }

            const std::size_t leading = dimensions.size() - 2;
            const std::span<const std::size_t> strides =
                    view.plane_strides();
            if (strides.size() != leading) {
                throw std::invalid_argument("invalid ADD plane stride count");
            }
            std::size_t max_plane = view.plane_offset();
            for (std::size_t i = 0; i < leading; ++i) {
                if (strides[i] == 0) {
                    throw std::invalid_argument(
                            "ADD does not permit zero strides");
                }
                max_plane = checked_add(
                        max_plane,
                        checked_mul(
                                dimensions[i] - 1, strides[i],
                                "ADD plane address overflows"),
                        "ADD plane address overflows");
            }
            if (max_plane >= checked_binary_plane_count(owner_spec.shape)) {
                throw std::invalid_argument(
                        "ADD view addresses outside its owner");
            }

            const std::size_t last_slot = detail::standard_plane_slot(
                    spec, max_plane, dimensions[leading] - 1,
                    dimensions[leading + 1] - 1);
            const std::size_t addressed_bits = checked_mul(
                    checked_add(last_slot, 1, "ADD slot count overflows"),
                    detail::leaf_bits(spec.data_type),
                    "ADD view size overflows");
            const std::size_t addressed_bytes =
                    bits_to_bytes(addressed_bits, "ADD view byte size overflows");
            const std::size_t owner_bits = checked_mul(
                    owner_spec.standard_padded_shape().element_count(),
                    detail::leaf_bits(owner_spec.data_type),
                    "ADD owner storage size overflows");
            if (addressed_bytes
                    > bits_to_bytes(
                            owner_bits, "ADD owner storage size overflows")) {
                throw std::invalid_argument(
                        "ADD view exceeds its owner storage");
            }
            const std::size_t logical_bits = checked_mul(
                    spec.shape.element_count(),
                    detail::leaf_bits(spec.data_type),
                    "ADD logical size overflows");
            (void)bits_to_bytes(logical_bits, "ADD logical size overflows");
        }
    }  // namespace


    DeviceOps::CopyViewSnapshot DeviceOps::snapshot_copy_view(
            const TensorView& view) {
        return DeviceOps::CopyViewSnapshot{
                view.spec(), &view.device(), view.owner_identity(),
                const_cast<void*>(view.native_handle()), view.plane_offset(),
                {view.plane_strides().begin(), view.plane_strides().end()}};
    }

    DeviceOps::BinaryViewSnapshot DeviceOps::snapshot_binary_view(
            const TensorView& view,
            std::span<const std::size_t> result_dimensions) {
        const std::span<const std::size_t> dimensions =
                view.spec().shape.dimensions();
        const std::size_t rank_offset =
                result_dimensions.size() - dimensions.size();
        const std::size_t result_leading =
                result_dimensions.size() - 2;
        std::vector<std::size_t> logical_plane_strides(
                result_leading, 0);
        bool broadcasts = dimensions.size() != result_dimensions.size();
        for (std::size_t axis = 0; axis < result_leading; ++axis) {
            if (axis < rank_offset) {
                broadcasts = broadcasts || result_dimensions[axis] != 1;
                continue;
            }
            const std::size_t source_axis = axis - rank_offset;
            if (dimensions[source_axis] == 1
                    && result_dimensions[axis] != 1) {
                broadcasts = true;
                continue;
            }
            logical_plane_strides[axis] =
                    view.plane_strides()[source_axis];
        }
        const bool broadcast_rows =
                dimensions[dimensions.size() - 2] == 1
                && result_dimensions[result_dimensions.size() - 2] != 1;
        const bool broadcast_columns =
                dimensions.back() == 1
                && result_dimensions.back() != 1;
        broadcasts = broadcasts || broadcast_rows || broadcast_columns;
        return DeviceOps::BinaryViewSnapshot{
                view.spec(), &view.device(), view.owner_identity(),
                const_cast<void*>(view.native_handle()), view.plane_offset(),
                {view.plane_strides().begin(), view.plane_strides().end()},
                std::move(logical_plane_strides), broadcast_rows,
                broadcast_columns, broadcasts};
    }

    DeviceOps::BinaryRequest DeviceOps::validate_binary(
            const Device& device, BinaryOperation operation,
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        validate_binary_spec(lhs.spec());
        validate_binary_spec(rhs.spec());
        validate_binary_spec(out.spec());
        if (lhs.spec().data_type != rhs.spec().data_type
                || lhs.spec().data_type != out.spec().data_type
                || lhs.spec().quantization != rhs.spec().quantization
                || lhs.spec().quantization != out.spec().quantization) {
            throw std::invalid_argument("binary specifications do not match");
        }
        validate_binary_view(device, lhs);
        validate_binary_view(device, rhs);
        validate_binary_view(device, out);
        const auto lhs_dims = lhs.spec().shape.dimensions();
        const auto rhs_dims = rhs.spec().shape.dimensions();
        const std::size_t rank = std::max(lhs_dims.size(), rhs_dims.size());
        std::vector<std::size_t> result(rank, 1);
        for (std::size_t i = 0; i < rank; ++i) {
            const std::size_t lhs_axis =
                    i < rank - lhs_dims.size() ? 1
                    : lhs_dims[i - (rank - lhs_dims.size())];
            const std::size_t rhs_axis =
                    i < rank - rhs_dims.size() ? 1
                    : rhs_dims[i - (rank - rhs_dims.size())];
            if (lhs_axis != rhs_axis && lhs_axis != 1 && rhs_axis != 1) {
                throw std::invalid_argument(
                        "binary shapes are not broadcast compatible");
            }
            result[i] = std::max(lhs_axis, rhs_axis);
        }
        if (out.spec().shape.dimensions().size() != result.size()
                || !std::equal(out.spec().shape.dimensions().begin(),
                               out.spec().shape.dimensions().end(),
                               result.begin())) {
            throw std::invalid_argument("binary output shape is incorrect");
        }
        // The computed broadcast result is itself a full tensor shape: it
        // must be validated before any snapshot, registration, sequence
        // reservation, token acceptance, metadata upload, or backend
        // dispatch exists.
        TensorShape result_shape{result};
        BinaryViewSnapshot lhs_snapshot = snapshot_binary_view(lhs, result);
        BinaryViewSnapshot rhs_snapshot = snapshot_binary_view(rhs, result);
        BinaryViewSnapshot out_snapshot = snapshot_binary_view(out, result);
        const auto exact_alias = [](const BinaryViewSnapshot& input,
                                    const BinaryViewSnapshot& output) {
            return !input.broadcasts
                    && input.owner_identity == output.owner_identity
                    && input.spec == output.spec
                    && input.plane_offset == output.plane_offset
                    && input.plane_strides == output.plane_strides
                    && input.logical_plane_strides
                            == output.logical_plane_strides
                    && input.broadcast_rows == output.broadcast_rows
                    && input.broadcast_columns == output.broadcast_columns;
        };
        if ((lhs_snapshot.owner_identity == out_snapshot.owner_identity
                    && !exact_alias(lhs_snapshot, out_snapshot))
                || (rhs_snapshot.owner_identity == out_snapshot.owner_identity
                    && !exact_alias(rhs_snapshot, out_snapshot))) {
            throw std::invalid_argument("binary input/output alias is forbidden");
        }
        if (lhs.spec().quantization != QuantizationFormat::NONE
                || lhs.spec().data_type == DataType::BOOL
                || lhs.spec().data_type == DataType::F8_E8M0
                || !add_numeric_leaf(lhs.spec().data_type)
                || (operation == BinaryOperation::Div
                    && !div_numeric_leaf(lhs.spec().data_type))) {
            throw UnsupportedOperation();
        }
        return BinaryRequest{operation, std::move(lhs_snapshot),
                             std::move(rhs_snapshot), std::move(out_snapshot),
                             std::move(result_shape), {}, {0, 1}, {}};
    }




    void DeviceOps::validate_copy(
            const Device& device, const TensorView& source,
            const TensorView& destination) {
        validate_views(device, {&source, &destination});
        if (!(source.spec() == destination.spec())) {
            throw std::invalid_argument(
                    "copy views must have identical shape, leaf type, "
                    "and quantization");
        }
    }

    bool DeviceOps::identical_window(
            const TensorView& source, const TensorView& destination) {
        return source.native_handle() == destination.native_handle()
                && source.plane_offset() == destination.plane_offset()
                && std::equal(
                        source.plane_strides().begin(),
                        source.plane_strides().end(),
                        destination.plane_strides().begin(),
                        destination.plane_strides().end());
    }

    std::runtime_error DeviceOps::unsupported(
            std::string_view backend, std::string_view operation) {
        std::string message(backend);
        message += " backend does not implement ";
        message += operation;
        return std::runtime_error(std::move(message));
    }
    DeviceOps::DeviceOps()
            : queue_id_(lease_queue_id()) {}

    DeviceOps::DeviceOps(const Device& device)
            : device_(&device), queue_id_(0) {
        device.reserve_queue_slot();
        queue_slot_reserved_ = true;
        try {
            queue_id_ = lease_queue_id();
            admission_capacity_ =
                    std::max<std::size_t>(
                            1, device.queue_config().max_in_flight_per_queue);
        } catch (...) {
            device.release_queue_slot();
            queue_slot_reserved_ = false;
            throw;
        }
    }

    const Device& DeviceOps::queue_device() const {
        if (device_ == nullptr) {
            throw std::logic_error("operation queue has no device");
        }
        return *device_;
    }
    void DeviceOps::validate_views(
            const Device& device,
            std::initializer_list<const TensorView*> views) {
        for (const TensorView* view : views) {
            if (view == nullptr || &view->device() != &device) {
                throw std::invalid_argument(
                        "operation views must belong to the queue's device");
            }
        }
    }
    oid DeviceOps::binary_impl(const BinaryRequest&) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::submit_prepared(
            std::function<void(std::uint64_t)> prepare,
            std::function<void(std::uint64_t)> dispatch,
            std::function<void()> rollback) {
        std::uint64_t sequence = 0;
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            if (admission_closing_) {
                throw std::logic_error("operation queue is closed");
            }
            if (next_sequence_ > kMaxSequence) {
                throw std::overflow_error(
                        "DeviceOps 55-bit submission sequence is exhausted");
            }
            sequence = next_sequence_++;
        }

        try {
            if (prepare) {
                prepare(sequence);
            }
            AdmissionNode node;
            node.dispatch = std::move(dispatch);
            // Keep a second callback handle locally so an insertion or FIFO
            // allocation failure can still roll back prepared ownership.
            // The node-owned copy is used if teardown drains it while parked.
            node.rollback = rollback;
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                if (admission_closing_) {
                    throw std::logic_error("operation queue is closed");
                }
                const auto [it, inserted] =
                        admission_nodes_.emplace(sequence, std::move(node));
                if (!inserted) {
                    throw std::logic_error(
                            "duplicate admission sequence");
                }
                try {
                    admission_fifo_.push_back(sequence);
                } catch (...) {
                    admission_nodes_.erase(it);
                    throw;
                }
            }
        } catch (...) {
            if (rollback) {
                try {
                    rollback();
                } catch (...) {
                }
            }
            abandon_sequence(sequence);
            throw;
        }
        std::exception_ptr dispatch_failure;
        pump_admission(&dispatch_failure, sequence);
        if (dispatch_failure) {
            std::rethrow_exception(dispatch_failure);
        }
        return encode_token(sequence);
    }

    void DeviceOps::abandon_sequence(std::uint64_t sequence) noexcept {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        pending_failures_.erase(sequence);
        if (next_sequence_ == sequence + 1 && completed_ < sequence) {
            --next_sequence_;
            return;
        }
        skipped_sequences_.emplace(sequence, sequence + 1);
    }

    void DeviceOps::pump_admission(
            std::exception_ptr* synchronous_failure,
            std::uint64_t synchronous_sequence) noexcept {
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            if (admission_pumping_) {
                return;
            }
            admission_pumping_ = true;
        }
        for (;;) {
            std::uint64_t sequence = 0;
            std::function<void(std::uint64_t)> dispatch;
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                if (admission_credits_ >= admission_capacity_
                        || admission_fifo_.empty()) {
                    admission_pumping_ = false;
                    return;
                }
                auto candidate = admission_fifo_.begin();
                while (candidate != admission_fifo_.end()) {
                    const auto it = admission_nodes_.find(*candidate);
                    if (it == admission_nodes_.end()) {
                        candidate = admission_fifo_.erase(candidate);
                        continue;
                    }
                    if (!it->second.executing) {
                        break;
                    }
                    ++candidate;
                }
                if (candidate == admission_fifo_.end()) {
                    admission_pumping_ = false;
                    return;
                }
                sequence = *candidate;
                const auto it = admission_nodes_.find(sequence);
                if (it == admission_nodes_.end()) {
                    admission_fifo_.erase(candidate);
                    continue;
                }
                it->second.executing = true;
                ++admission_credits_;
                dispatch = std::move(it->second.dispatch);
            }
            try {
                dispatch(sequence);
            } catch (const detail::AdmissionResourceUnavailable&) {
                // The accepted node remains the FIFO head. Restore its
                // callback and credit without rolling back prepared owners;
                // a later completion will retry admission after resources
                // become reusable.
                bool retry = false;
                {
                    std::lock_guard<std::mutex> lock(completion_mutex_);
                    const auto node = admission_nodes_.find(sequence);
                    if (node != admission_nodes_.end()) {
                        node->second.dispatch = std::move(dispatch);
                        node->second.executing = false;
                        if (admission_credits_ != 0) {
                            --admission_credits_;
                        }
                        retry = true;
                    }
                    admission_pumping_ = false;
                    completion_cv_.notify_all();
                }
                if (retry) {
                    pump_admission(synchronous_failure, synchronous_sequence);
                }
                return;
            } catch (...) {
                const std::exception_ptr failure =
                        std::current_exception();
                const bool preacceptance =
                        synchronous_failure != nullptr
                        && sequence == synchronous_sequence;
                std::function<void()> rollback;
                bool unaccepted = false;
                {
                    std::lock_guard<std::mutex> lock(completion_mutex_);
                    const auto node = admission_nodes_.find(sequence);
                    if (node != admission_nodes_.end()) {
                        if (preacceptance) {
                            // A dispatch failure while the submitting call
                            // is still on the stack is pre-acceptance. Drop
                            // its node and sequence instead of manufacturing
                            // a positive token for work that never linked.
                            rollback = std::move(node->second.rollback);
                            if (node->second.executing
                                    && admission_credits_ != 0) {
                                --admission_credits_;
                            }
                            admission_nodes_.erase(node);
                            const auto fifo = std::find(
                                    admission_fifo_.begin(),
                                    admission_fifo_.end(), sequence);
                            if (fifo != admission_fifo_.end()) {
                                admission_fifo_.erase(fifo);
                            }
                            completion_cv_.notify_all();
                            unaccepted = true;
                        } else {
                            // The token was already returned before a
                            // parked-node retirement failed. Keep it
                            // terminally waitable and only release prepared
                            // state that never reached the backend.
                            rollback = std::move(node->second.rollback);
                        }
                    }
                }
                if (rollback) {
                    try {
                        rollback();
                    } catch (...) {
                    }
                }
                if (preacceptance) {
                    if (unaccepted) {
                        abandon_sequence(sequence);
                    }
                    if (*synchronous_failure == nullptr) {
                        *synchronous_failure = failure;
                    }
                } else {
                    try {
                        complete(sequence, failure);
                    } catch (...) {
                    }
                }
            }
        }
    }

    oid DeviceOps::copy_impl(
            const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::silu_impl(const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::linear_impl(
            const TensorView&, const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::rmsnorm_impl(
            const TensorView&, TensorView&, const TensorView&, float, size_t) {
        throw UnsupportedOperation();
    }
    oid DeviceOps::sdpa_impl(
            const TensorView&, const TensorView&, const TensorView&,
            size_t, size_t, size_t, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::map_failure(std::exception_ptr failure) noexcept {
        try {
            if (failure != nullptr) {
                std::rethrow_exception(failure);
            }
        } catch (const UnsupportedOperation&) {
            return to_oid(OidError::Unsupported);
        } catch (const std::bad_alloc&) {
            return to_oid(OidError::ResourceExhausted);
        } catch (const std::overflow_error&) {
            return to_oid(OidError::Overflow);
        } catch (const std::invalid_argument&) {
            return to_oid(OidError::InvalidArgument);
        } catch (const std::exception&) {
            return to_oid(OidError::DeviceError);
        } catch (...) {
            return to_oid(OidError::InternalError);
        }
        return to_oid(OidError::InternalError);
    }

    oid DeviceOps::invoke_failure(std::exception_ptr failure) noexcept {
        return map_failure(std::move(failure));
    }

    oid DeviceOps::invoke(oid result) noexcept {
        return result;
    }

    oid DeviceOps::copy(
            const TensorView& source, TensorView& destination) noexcept {
        try {
            validate_copy(queue_device(), source, destination);
            return invoke(copy_impl(source, destination));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }
    oid DeviceOps::add(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), BinaryOperation::Add,
                                    lhs, rhs, out);
            const WorkspaceRequirements requirements =
                    binary_workspace_requirements(request);
            const std::array<TensorView, 3> operands{lhs, rhs, out};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            queue_device(), workspace,
                            requirements.bytes, requirements.alignment,
                            operands);
            return invoke(binary_impl(BinaryRequest{
                    request.operation, request.lhs, request.rhs, request.out,
                    request.result_shape, validated_workspace, requirements,
                    {}}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::mul(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), BinaryOperation::Mul,
                                    lhs, rhs, out);
            const WorkspaceRequirements requirements =
                    binary_workspace_requirements(request);
            const std::array<TensorView, 3> operands{lhs, rhs, out};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            queue_device(), workspace,
                            requirements.bytes, requirements.alignment,
                            operands);
            return invoke(binary_impl(BinaryRequest{
                    request.operation, request.lhs, request.rhs, request.out,
                    request.result_shape, validated_workspace, requirements,
                    {}}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::sub(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), BinaryOperation::Sub,
                                    lhs, rhs, out);
            const WorkspaceRequirements requirements =
                    binary_workspace_requirements(request);
            const std::array<TensorView, 3> operands{lhs, rhs, out};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            queue_device(), workspace,
                            requirements.bytes, requirements.alignment,
                            operands);
            return invoke(binary_impl(BinaryRequest{
                    request.operation, request.lhs, request.rhs, request.out,
                    request.result_shape, validated_workspace, requirements,
                    {}}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::div(
            const TensorView& lhs, const TensorView& rhs,
            TensorView& out, RawWorkspaceView workspace) noexcept {
        try {
            BinaryRequest request =
                    validate_binary(queue_device(), BinaryOperation::Div,
                                    lhs, rhs, out);
            const WorkspaceRequirements requirements =
                    binary_workspace_requirements(request);
            const std::array<TensorView, 3> operands{lhs, rhs, out};
            const RawWorkspaceView validated_workspace =
                    detail::WorkspaceValidation::validated(
                            queue_device(), workspace,
                            requirements.bytes, requirements.alignment,
                            operands);
            return invoke(binary_impl(BinaryRequest{
                    request.operation, request.lhs, request.rhs, request.out,
                    request.result_shape, validated_workspace, requirements,
                    {}}));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    WorkspaceRequirements
    DeviceOps::binary_workspace_requirements(const BinaryRequest&) {
        // CPU, TTNN, CUDA, and ROCm need no raw workspace for the binary
        // operations. SYCL overrides this hook with its checked
        // whole-plane staging sum.
        return {0, 1};
    }

    WorkspaceRequirements DeviceOps::add_workspace_requirements(
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        const BinaryRequest request = validate_binary(
                queue_device(), BinaryOperation::Add, lhs, rhs, out);
        return binary_workspace_requirements(request);
    }

    WorkspaceRequirements DeviceOps::mul_workspace_requirements(
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        const BinaryRequest request = validate_binary(
                queue_device(), BinaryOperation::Mul, lhs, rhs, out);
        return binary_workspace_requirements(request);
    }

    WorkspaceRequirements DeviceOps::sub_workspace_requirements(
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        const BinaryRequest request = validate_binary(
                queue_device(), BinaryOperation::Sub, lhs, rhs, out);
        return binary_workspace_requirements(request);
    }

    WorkspaceRequirements DeviceOps::div_workspace_requirements(
            const TensorView& lhs, const TensorView& rhs,
            const TensorView& out) {
        const BinaryRequest request = validate_binary(
                queue_device(), BinaryOperation::Div, lhs, rhs, out);
        return binary_workspace_requirements(request);
    }

    RawWorkspaceView detail::WorkspaceValidation::validated(
            const Device& device, const RawWorkspaceView& workspace,
            std::size_t required_capacity, std::size_t required_alignment,
            std::span<const TensorView> operands) {
        if (required_alignment == 0) {
            throw std::invalid_argument(
                    "workspace alignment requirement is zero");
        }
        if (required_capacity == 0) {
            // Zero-requirement operations touch no scratch: the facades
            // default to the empty view exactly for this case, and any
            // view value is acceptable.
            return workspace;
        }
        if (workspace.empty()) {
            throw std::invalid_argument(
                    "positive workspace requirement needs a non-empty "
                    "workspace view");
        }
        const RawWorkspace* owner = workspace.owner_identity();
        if (owner == nullptr || !device.owns_workspace(owner)) {
            throw std::invalid_argument(
                    "workspace owner is not live on this device");
        }
        if (&owner->device() != &device) {
            throw std::invalid_argument(
                    "workspace does not belong to this device");
        }
        if (workspace.byte_size() < required_capacity) {
            throw std::invalid_argument(
                    "workspace is smaller than the required capacity");
        }
        const void* raw_address = workspace.range_address();
        if (raw_address == nullptr) {
            throw std::invalid_argument("workspace range has no address");
        }
        const std::uintptr_t base =
                reinterpret_cast<std::uintptr_t>(raw_address);
        if (base % required_alignment != 0) {
            throw std::invalid_argument(
                    "workspace range alignment is insufficient");
        }
        if (workspace.byte_size()
                > std::numeric_limits<std::uintptr_t>::max() - base) {
            throw std::overflow_error("workspace range end overflows");
        }
        const std::uintptr_t range_end = base + workspace.byte_size();
        for (const TensorView& operand : operands) {
            if (&operand.device() != &device
                    || operand.owner_identity() == nullptr
                    || operand.native_handle() == nullptr) {
                throw std::invalid_argument(
                        "workspace operand belongs to another device");
            }
            const std::uintptr_t operand_base =
                    reinterpret_cast<std::uintptr_t>(
                            operand.native_handle());
            const std::size_t operand_bytes =
                    operand.owner_identity()
                            ->view()
                            .spec()
                            .tiled_storage_nbytes();
            if (operand_bytes
                    > std::numeric_limits<std::uintptr_t>::max()
                            - operand_base) {
                throw std::overflow_error(
                        "workspace operand storage range overflows");
            }
            const std::uintptr_t operand_end =
                    operand_base + operand_bytes;
            if (base < operand_end && operand_base < range_end) {
                throw std::invalid_argument(
                        "workspace range overlaps an operand or output "
                        "storage range");
            }
        }
        return workspace;
    }
    void* detail::WorkspaceValidation::address(
            const RawWorkspaceView& workspace) noexcept {
        return workspace.range_address();
    }


    oid DeviceOps::silu(
            const TensorView& x, TensorView& y) noexcept {
        try {
            validate_views(queue_device(), {&x, &y});
            return invoke(silu_impl(x, y));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::linear(
            const TensorView& x, const TensorView& w,
            TensorView& y) noexcept {
        try {
            validate_views(queue_device(), {&x, &w, &y});
            return invoke(linear_impl(x, w, y));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::rmsnorm(
            const TensorView& x, TensorView& y, const TensorView& w,
            float eps, size_t dim) noexcept {
        try {
            validate_views(queue_device(), {&x, &y, &w});
            if (!(eps >= 0.0F) || dim == 0) {
                throw std::invalid_argument("invalid rmsnorm parameters");
            }
            return invoke(rmsnorm_impl(x, y, w, eps, dim));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::sdpa(
            const TensorView& q, const TensorView& k, const TensorView& v,
            size_t n_heads, size_t n_kv_heads, size_t head_dim,
            TensorView& attn_out) noexcept {
        try {
            validate_views(queue_device(), {&q, &k, &v, &attn_out});
            if (n_heads == 0 || n_kv_heads == 0 || head_dim == 0
                    || n_heads % n_kv_heads != 0) {
                throw std::invalid_argument("invalid sdpa parameters");
            }
            return invoke(sdpa_impl(
                    q, k, v, n_heads, n_kv_heads, head_dim, attn_out));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }
    DeviceOps::~DeviceOps() {
        close_and_drain();
        release_queue_id(queue_id_);
        if (queue_slot_reserved_ && device_ != nullptr) {
            device_->release_queue_slot();
            queue_slot_reserved_ = false;
        }
    }

    std::uint8_t DeviceOps::lease_queue_id() {
        std::lock_guard<std::mutex> lock(g_queue_ids_mutex);
        for (std::size_t i = 0; i < g_live_queue_ids.size(); ++i) {
            if (!g_live_queue_ids.test(i)) {
                g_live_queue_ids.set(i);
                return static_cast<std::uint8_t>(i + 1);
            }
        }
        throw std::runtime_error("all 255 DeviceOps queue ids are live");
    }

    void DeviceOps::release_queue_id(std::uint8_t queue_id) noexcept {
        std::lock_guard<std::mutex> lock(g_queue_ids_mutex);
        g_live_queue_ids.reset(queue_id - 1);
    }

    oid DeviceOps::encode_token(std::uint64_t sequence) const noexcept {
        const std::uint64_t encoded =
                (std::uint64_t{queue_id_} << kSequenceBits) | sequence;
        return static_cast<oid>(encoded);
    }

    void DeviceOps::wait(oid token) {
        if (token <= 0) {
            throw std::invalid_argument("oid must be a positive token");
        }
        const std::uint64_t encoded = static_cast<std::uint64_t>(token);
        const std::uint64_t id = encoded >> kSequenceBits;
        const std::uint64_t sequence = encoded & kSequenceMask;
        if (id == 0) {
            throw std::invalid_argument("oid queue id is zero");
        }
        if (sequence == 0) {
            throw std::invalid_argument("oid sequence is zero");
        }
        if (id != queue_id_) {
            throw std::invalid_argument("oid belongs to another queue");
        }
        std::unique_lock<std::mutex> lock(completion_mutex_);

        if (sequence >= next_sequence_) {
            throw std::invalid_argument("oid sequence was never submitted");
        }
        const auto skipped = skipped_sequences_.upper_bound(sequence);
        if (skipped != skipped_sequences_.begin()) {
            auto previous = skipped;
            --previous;
            if (previous->second > sequence) {
                throw std::invalid_argument("oid sequence was never submitted");
            }
        }
        completion_cv_.wait(lock, [&] {
            return completed_ >= sequence || failures_.count(sequence) != 0;
        });
        if (const auto failure = failures_.find(sequence);
                failure != failures_.end()) {
            std::rethrow_exception(failure->second);
        }
        lock.unlock();
        fence_through_sequence(sequence);
        lock.lock();
        if (const auto failure = failures_.find(sequence);
                failure != failures_.end()) {
            std::rethrow_exception(failure->second);
        }
    }

    void DeviceOps::fence_through_sequence(
            std::uint64_t /*sequence*/) noexcept {}

    void DeviceOps::record_post_completion_failure(
            std::uint64_t sequence, std::exception_ptr failure) {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (failures_.find(sequence) == failures_.end()) {
            failures_.emplace(sequence, std::move(failure));
            completion_cv_.notify_all();
        }
    }

    void DeviceOps::complete(
            std::uint64_t sequence, std::exception_ptr failure) {
        if (sequence == 0) {
            throw std::invalid_argument("completion sequence is zero");
        }
        bool should_pump = false;
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            if (sequence >= next_sequence_) {
                throw std::invalid_argument(
                        "completion of a sequence that was never submitted");
            }
            const auto node = admission_nodes_.find(sequence);
            if (node == admission_nodes_.end()) {
                if (failures_.find(sequence) != failures_.end()) {
                    return;
                }
                throw std::invalid_argument(
                        "completion of an unknown admission sequence");
            }
            if (const auto pending = pending_failures_.find(sequence);
                    pending != pending_failures_.end()) {
                if (!failure) {
                    failure = pending->second;
                }
                pending_failures_.erase(pending);
            }
            if (failure && failures_.find(sequence) == failures_.end()) {
                failures_.emplace(sequence, failure);
            }
            if (node->second.executing && admission_credits_ != 0) {
                --admission_credits_;
            }
            admission_nodes_.erase(node);
            if (!admission_fifo_.empty()
                    && admission_fifo_.front() == sequence) {
                admission_fifo_.pop_front();
            } else {
                const auto fifo = std::find(
                        admission_fifo_.begin(), admission_fifo_.end(),
                        sequence);
                if (fifo != admission_fifo_.end()) {
                    admission_fifo_.erase(fifo);
                }
            }
            if (completed_ < sequence) {
                completed_ = sequence;
            }
            completion_cv_.notify_all();
            should_pump = !admission_closing_;
        }
        if (should_pump) {
            pump_admission();
        }
    }

    void DeviceOps::close_and_drain() noexcept {
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            admission_closing_ = true;
        }
        // Close marks admission immediately, but only parked nodes can be
        // completed here. Executing nodes still own backend resources and
        // must finish through their worker's drain before their terminal
        // result is recorded.
        std::vector<std::uint64_t> parked;
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            for (const std::uint64_t sequence : admission_fifo_) {
                const auto node = admission_nodes_.find(sequence);
                if (node != admission_nodes_.end()
                        && !node->second.executing) {
                    parked.push_back(sequence);
                }
            }
        }
        for (const std::uint64_t sequence : parked) {
            std::function<void()> rollback;
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                const auto node = admission_nodes_.find(sequence);
                if (node == admission_nodes_.end()
                        || node->second.executing) {
                    continue;
                }
                rollback = std::move(node->second.rollback);
            }
            if (rollback) {
                try {
                    rollback();
                } catch (...) {
                }
            }
            try {
                complete(
                        sequence,
                        std::make_exception_ptr(std::runtime_error(
                                "operation queue drained before completion")));
            } catch (...) {
            }
        }
    }

    /**
     * Retains a backend failure for a reserved sequence without completing
     * it. A normally allocated sequence is accepted when
     * sequence < next_sequence_ && sequence > completed_. A sequence at or
     * beyond next_sequence_ was never submitted, while one at or below
     * completed_ has already completed; ranges skipped by the test seam are
     * also never submitted.
     */
    void DeviceOps::commit_failure(
            std::uint64_t sequence, std::exception_ptr failure) {
        if (sequence == 0) {
            throw std::invalid_argument("retained failure sequence is zero");
        }
        if (!failure) {
            throw std::invalid_argument("retained failure is empty");
        }
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (sequence >= next_sequence_) {
            throw std::invalid_argument(
                    "retained failure is not for a reserved submission");
        }
        if (sequence <= completed_) {
            throw std::invalid_argument(
                    "retained failure is for a sequence that has already been completed");
        }
        const auto skipped = skipped_sequences_.upper_bound(sequence);
        if (skipped != skipped_sequences_.begin()) {
            auto previous = skipped;
            --previous;
            if (previous->second > sequence) {
                throw std::invalid_argument(
                        "retained failure is not for a reserved submission");
            }
        }
        pending_failures_[sequence] = std::move(failure);
    }

    void DeviceOps::seek_next_sequence(std::uint64_t next_sequence) {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (next_sequence == 0 || next_sequence < next_sequence_
                || next_sequence > kMaxSequence + 1) {
            throw std::invalid_argument("queue sequence numbers only move forward");
        }
        if (next_sequence > next_sequence_) {
            skipped_sequences_.emplace(next_sequence_, next_sequence);
        }
        next_sequence_ = next_sequence;
    }

}  // namespace iom
