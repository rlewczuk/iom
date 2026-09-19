#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"
#include "iom_internal.hpp"
#include <algorithm>
#include <bitset>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace iom {
    using detail::UnsupportedOperation;

    namespace detail {

        namespace {
            // Structurally rejected operand text, tagged with the operation
            // whose admission is being validated. Composed on the rejection
            // path only, so a successful admission allocates nothing.
            [[noreturn]] void reject_operand(
                    const char* operation, std::string_view what) {
                std::string message(operation);
                message += ' ';
                message += what;
                throw std::invalid_argument(std::move(message));
            }


            // Checked product of the leading plane-selecting extents.
            [[nodiscard]] std::size_t checked_plane_count(
                    const TensorShape& shape, const char* what) {
                std::size_t planes = 1;
                const std::span<const std::size_t> dimensions =
                        shape.dimensions();
                for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                    planes = checked_mul(planes, dimensions[i], what);
                }
                return planes;
            }

            // Checked padded element count of one shape, computed with scalar
            // rounding instead of a vector-backed padded shape. Precondition:
            // the shape has rank two through eight.
            [[nodiscard]] std::size_t checked_tiled_element_count(
                    const TensorShape& shape, const char* what) {
                const std::span<const std::size_t> dimensions =
                        shape.dimensions();
                const std::size_t last = dimensions.size() - 1;
                const std::size_t rows =
                        padded_extent(dimensions[last - 1], what);
                const std::size_t columns =
                        padded_extent(dimensions[last], what);
                std::size_t elements = 1;
                for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                    elements = checked_mul(elements, dimensions[i], what);
                }
                return checked_mul(
                        checked_mul(elements, rows, what), columns, what);
            }

            // Checked storage bytes a whole owner of `spec` reserves: the
            // allocation-free equivalent of
            // TensorSpec::tiled_storage_nbytes(). Precondition: the spec
            // carries a recognized leaf encoding and a valid shape.
            [[nodiscard]] std::size_t checked_tiled_storage_bytes(
                    const TensorSpec& spec, const char* what) {
                const std::size_t bits = leaf_bits(spec.data_type);
                const std::size_t elements =
                        checked_tiled_element_count(spec.shape, what);
                return bits_to_bytes(checked_mul(elements, bits, what), what);
            }
        }  // namespace

        bool recognized_data_type(DataType value) noexcept {
            // The declared leaves are contiguous, and leaf_bits covers exactly
            // this interval.
            return value >= DataType::BOOL && value <= DataType::F64;
        }

        bool recognized_quantization(QuantizationFormat value) noexcept {
            // TensorSpec::validate uses the same declared-enum interval.
            return value >= QuantizationFormat::NONE
                    && value <= QuantizationFormat::TT_BFP8A;
        }

        void validate_checked_spec(
                const TensorSpec& spec, const char* operation) {
            if (!recognized_data_type(spec.data_type)) {
                throw std::invalid_argument("unknown DataType value");
            }
            if (!recognized_quantization(spec.quantization)) {
                throw std::invalid_argument("unknown quantization format");
            }
            if (spec.shape.rank() < 2) {
                reject_operand(operation, "requires rank at least two");
            }
            if (spec.shape.rank() > kMaxTensorRank) {
                reject_operand(operation, "requires rank at most eight");
            }
            for (const std::size_t dimension : spec.shape.dimensions()) {
                if (dimension == 0) {
                    reject_operand(operation, "dimensions must be nonzero");
                }
            }
        }

        CheckedViewFacts validate_checked_view(
                const Device& device, const TensorView& view,
                const char* operation) {
            const Tensor* owner = view.owner_identity();
            if (owner == nullptr) {
                reject_operand(operation, "view has no owner");
            }
            if (&view.device() != &device
                    || &owner->view().device() != &device
                    || owner->view().owner_identity() != owner) {
                reject_operand(
                        operation, "view owner belongs to another device");
            }
            const void* handle = view.native_handle();
            if (handle == nullptr
                    || handle != owner->view().native_handle()) {
                reject_operand(operation, "view has no stable owner handle");
            }

            const TensorSpec& spec = view.spec();
            const TensorSpec& owner_spec = owner->view().spec();
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::span<const std::size_t> owner_dimensions =
                    owner_spec.shape.dimensions();
            // Callers validate the spec first, but the final matrix axes are
            // indexed below, so the rank minimum is re-checked here rather
            // than trusting a spec this path did not validate.
            if (dimensions.size() < 2) {
                reject_operand(operation, "requires rank at least two");
            }
            if (spec.data_type != owner_spec.data_type
                    || spec.quantization != owner_spec.quantization
                    || dimensions[dimensions.size() - 2]
                            != owner_dimensions[owner_dimensions.size() - 2]
                    || dimensions.back() != owner_dimensions.back()) {
                reject_operand(
                        operation,
                        "view specification does not match its owner");
            }

            const std::size_t leading = dimensions.size() - 2;
            const std::span<const std::size_t> strides =
                    view.plane_strides();
            if (strides.size() != leading) {
                reject_operand(
                        operation,
                        "plane stride count does not match the view rank");
            }
            std::size_t max_plane = view.plane_offset();
            for (std::size_t i = 0; i < leading; ++i) {
                if (strides[i] == 0) {
                    reject_operand(
                            operation,
                            "view does not permit zero plane strides");
                }
                max_plane = checked_add(
                        max_plane,
                        checked_mul(
                                dimensions[i] - 1, strides[i],
                                "plane address overflows"),
                        "plane address overflows");
            }
            if (max_plane
                    >= checked_plane_count(
                            owner_spec.shape, "plane count overflows")) {
                reject_operand(
                        operation, "view addresses outside its owner");
            }

            const std::size_t last_slot = standard_plane_slot(
                    spec, max_plane, dimensions[leading] - 1,
                    dimensions[leading + 1] - 1);
            const std::size_t addressed_bits = checked_mul(
                    checked_add(last_slot, 1, "slot count overflows"),
                    leaf_bits(spec.data_type), "view size overflows");
            const std::size_t addressed_bytes = bits_to_bytes(
                    addressed_bits, "view byte size overflows");
            const std::size_t storage_bytes = checked_tiled_storage_bytes(
                    owner_spec, "owner storage size overflows");
            if (addressed_bytes > storage_bytes) {
                reject_operand(
                        operation, "view exceeds its owner storage");
            }
            const std::size_t logical_bits = checked_mul(
                    spec.shape.element_count(), leaf_bits(spec.data_type),
                    "logical size overflows");
            const std::size_t logical_bytes = bits_to_bytes(
                    logical_bits, "logical byte size overflows");
            return CheckedViewFacts{
                    max_plane, addressed_bytes, storage_bytes,
                    logical_bytes};
        }

    }  // namespace detail

    void Device::reserve_queue_slot() const {
        std::lock_guard<std::mutex> lock(queue_registry_mutex_);
        if (live_queue_count_ == 4) throw std::bad_alloc();
        ++live_queue_count_;
    }
    void Device::release_queue_slot() const noexcept {
        std::lock_guard<std::mutex> lock(queue_registry_mutex_);
        if (live_queue_count_ != 0) --live_queue_count_;
    }
    namespace {
        std::mutex g_queue_ids_mutex;
        std::bitset<255> g_live_queue_ids;
    }
    std::runtime_error DeviceOps::unsupported(
            std::string_view backend, std::string_view operation) {
        std::string message(backend);
        message += " backend does not implement ";
        message += operation;
        return std::runtime_error(std::move(message));
    }
    DeviceOps::DeviceOps() : queue_id_(lease_queue_id()) {}
    DeviceOps::DeviceOps(const Device& device)
            : device_(&device), queue_id_(0) {
        device.reserve_queue_slot();
        queue_slot_reserved_ = true;
        try {
            queue_id_ = lease_queue_id();
            admission_capacity_ = std::max<std::size_t>(
                    1, device.queue_config().max_in_flight_per_queue);
        } catch (...) {
            device.release_queue_slot();
            queue_slot_reserved_ = false;
            throw;
        }
    }
    const Device& DeviceOps::device() const {
        return queue_device();
    }
    const Device& DeviceOps::queue_device() const {
        if (device_ == nullptr)
            throw std::logic_error("operation queue has no device");
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
    oid DeviceOps::submit_prepared(
            std::function<void(std::uint64_t)> prepare,
            std::function<void(std::uint64_t)> dispatch,
            std::function<void()> rollback) {
        std::uint64_t sequence = 0;
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            if (admission_closing_)
                throw std::logic_error("operation queue is closed");
            if (next_sequence_ > kMaxSequence) {
                throw std::overflow_error(
                        "DeviceOps 55-bit submission sequence is exhausted");
            }
            sequence = next_sequence_++;
        }
        try {
            if (prepare) prepare(sequence);
            AdmissionNode node;
            node.dispatch = std::move(dispatch);
            node.rollback = rollback;
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                if (admission_closing_)
                    throw std::logic_error("operation queue is closed");
                const auto [it, inserted] =
                        admission_nodes_.emplace(sequence, std::move(node));
                if (!inserted)
                    throw std::logic_error("duplicate admission sequence");
                try {
                    admission_fifo_.push_back(sequence);
                } catch (...) {
                    admission_nodes_.erase(it);
                    throw;
                }
            }
        } catch (...) {
            if (rollback) {
                try { rollback(); } catch (...) {}
            }
            abandon_sequence(sequence);
            throw;
        }
        std::exception_ptr dispatch_failure;
        pump_admission(&dispatch_failure, sequence);
        if (dispatch_failure) std::rethrow_exception(dispatch_failure);
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
            if (admission_pumping_ || admission_closing_) return;
            admission_pumping_ = true;
        }
        for (;;) {
            std::uint64_t sequence = 0;
            std::function<void(std::uint64_t)> dispatch;
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                if (admission_closing_
                        || admission_credits_ >= admission_capacity_
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
                    if (!it->second.executing) break;
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
                {
                    std::lock_guard<std::mutex> lock(completion_mutex_);
                    const auto node = admission_nodes_.find(sequence);
                    if (node != admission_nodes_.end()) {
                        node->second.dispatch = std::move(dispatch);
                        node->second.executing = false;
                        if (admission_credits_ != 0) --admission_credits_;
                    }
                    // Resource exhaustion is a nonterminal parked state. The
                    // completion/release path owns the next retry; never
                    // recurse from this dispatch path while no resource
                    // state changed.
                    admission_pumping_ = false;
                    completion_cv_.notify_all();
                }
                return;
            } catch (...) {
                const std::exception_ptr failure = std::current_exception();
                const bool preacceptance = synchronous_failure != nullptr
                        && sequence == synchronous_sequence;
                std::function<void()> rollback;
                bool unaccepted = false;
                {
                    std::lock_guard<std::mutex> lock(completion_mutex_);
                    const auto node = admission_nodes_.find(sequence);
                    if (node != admission_nodes_.end()) {
                        if (preacceptance) {
                            rollback = std::move(node->second.rollback);
                            if (node->second.executing
                                    && admission_credits_ != 0)
                                --admission_credits_;
                            admission_nodes_.erase(node);
                            const auto fifo = std::find(
                                    admission_fifo_.begin(),
                                    admission_fifo_.end(), sequence);
                            if (fifo != admission_fifo_.end())
                                admission_fifo_.erase(fifo);
                            completion_cv_.notify_all();
                            unaccepted = true;
                        } else {
                            rollback = std::move(node->second.rollback);
                        }
                    }
                }
                if (rollback) {
                    try { rollback(); } catch (...) {}
                }
                if (preacceptance) {
                    if (unaccepted) abandon_sequence(sequence);
                    if (*synchronous_failure == nullptr)
                        *synchronous_failure = failure;
                } else {
                    try { complete(sequence, failure); } catch (...) {}
                }
            }
        }
    }
    oid DeviceOps::map_failure(std::exception_ptr failure) noexcept {
        try {
            if (failure != nullptr) std::rethrow_exception(failure);
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
    oid DeviceOps::invoke(oid result) noexcept { return result; }
    RawWorkspaceView detail::WorkspaceValidation::validated(
            const Device& device, const RawWorkspaceView& workspace,
            std::size_t required_capacity, std::size_t required_alignment,
            std::span<const TensorView> operands) {
        if (required_alignment == 0)
            throw std::invalid_argument(
                    "workspace alignment requirement is zero");
        if (required_capacity == 0) return workspace;
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
        if (&owner->device() != &device)
            throw std::invalid_argument("workspace does not belong to this device");
        if (workspace.byte_size() < required_capacity)
            throw std::invalid_argument(
                    "workspace is smaller than the required capacity");
        const void* raw_address = workspace.range_address();
        if (raw_address == nullptr)
            throw std::invalid_argument("workspace range has no address");
        const std::uintptr_t base =
                reinterpret_cast<std::uintptr_t>(raw_address);
        if (base % required_alignment != 0)
            throw std::invalid_argument(
                    "workspace range alignment is insufficient");
        if (workspace.byte_size()
                > std::numeric_limits<std::uintptr_t>::max() - base)
            throw std::overflow_error("workspace range end overflows");
        const std::uintptr_t range_end = base + workspace.byte_size();
        for (const TensorView& operand : operands) {
            if (&operand.device() != &device
                    || operand.owner_identity() == nullptr
                    || operand.native_handle() == nullptr) {
                throw std::invalid_argument(
                        "workspace operand belongs to another device");
            }
            const std::uintptr_t operand_base =
                    reinterpret_cast<std::uintptr_t>(operand.native_handle());
            const std::size_t operand_bytes = operand.owner_identity()->view()
                    .spec().tiled_storage_nbytes();
            if (operand_bytes
                    > std::numeric_limits<std::uintptr_t>::max()
                            - operand_base)
                throw std::overflow_error(
                        "workspace operand storage range overflows");
            const std::uintptr_t operand_end = operand_base + operand_bytes;
            if (base < operand_end && operand_base < range_end)
                throw std::invalid_argument(
                        "workspace range overlaps an operand or output "
                        "storage range");
        }
        return workspace;
    }
    void* detail::WorkspaceValidation::address(
            const RawWorkspaceView& workspace) noexcept {
        return workspace.range_address();
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
        if (token <= 0)
            throw std::invalid_argument("oid must be a positive token");
        const std::uint64_t encoded = static_cast<std::uint64_t>(token);
        const std::uint64_t id = encoded >> kSequenceBits;
        const std::uint64_t sequence = encoded & kSequenceMask;
        if (id == 0) throw std::invalid_argument("oid queue id is zero");
        if (sequence == 0) throw std::invalid_argument("oid sequence is zero");
        if (id != queue_id_)
            throw std::invalid_argument("oid belongs to another queue");
        std::unique_lock<std::mutex> lock(completion_mutex_);
        if (sequence >= next_sequence_)
            throw std::invalid_argument("oid sequence was never submitted");
        const auto skipped = skipped_sequences_.upper_bound(sequence);
        if (skipped != skipped_sequences_.begin()) {
            auto previous = skipped;
            --previous;
            if (previous->second > sequence)
                throw std::invalid_argument("oid sequence was never submitted");
        }
        completion_cv_.wait(lock, [&] {
            return completed_ >= sequence || failures_.count(sequence) != 0;
        });
        if (const auto failure = failures_.find(sequence);
                failure != failures_.end())
            std::rethrow_exception(failure->second);
        lock.unlock();
        fence_through_sequence(sequence);
        lock.lock();
        if (const auto failure = failures_.find(sequence);
                failure != failures_.end())
            std::rethrow_exception(failure->second);
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
        if (sequence == 0)
            throw std::invalid_argument("completion sequence is zero");
        bool should_pump = false;
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            if (sequence >= next_sequence_)
                throw std::invalid_argument(
                        "completion of a sequence that was never submitted");
            const auto node = admission_nodes_.find(sequence);
            if (node == admission_nodes_.end()) {
                if (failures_.find(sequence) != failures_.end()) return;
                throw std::invalid_argument(
                        "completion of an unknown admission sequence");
            }
            if (const auto pending = pending_failures_.find(sequence);
                    pending != pending_failures_.end()) {
                if (!failure) failure = pending->second;
                pending_failures_.erase(pending);
            }
            if (failure && failures_.find(sequence) == failures_.end())
                failures_.emplace(sequence, failure);
            if (node->second.executing && admission_credits_ != 0)
                --admission_credits_;
            admission_nodes_.erase(node);
            if (!admission_fifo_.empty()
                    && admission_fifo_.front() == sequence) {
                admission_fifo_.pop_front();
            } else {
                const auto fifo = std::find(
                        admission_fifo_.begin(), admission_fifo_.end(),
                        sequence);
                if (fifo != admission_fifo_.end()) admission_fifo_.erase(fifo);
            }
            if (completed_ < sequence) completed_ = sequence;
            completion_cv_.notify_all();
            should_pump = !admission_closing_;
        }
        if (should_pump) pump_admission();
    }
    void DeviceOps::close_and_drain() noexcept {
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            admission_closing_ = true;
        }
        std::vector<std::uint64_t> parked;
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            for (const std::uint64_t sequence : admission_fifo_) {
                const auto node = admission_nodes_.find(sequence);
                if (node != admission_nodes_.end() && !node->second.executing)
                    parked.push_back(sequence);
            }
        }
        for (const std::uint64_t sequence : parked) {
            std::function<void()> rollback;
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                const auto node = admission_nodes_.find(sequence);
                if (node == admission_nodes_.end() || node->second.executing)
                    continue;
                rollback = std::move(node->second.rollback);
            }
            if (rollback) {
                try { rollback(); } catch (...) {}
            }
            try {
                complete(sequence, std::make_exception_ptr(
                        std::runtime_error(
                                "operation queue drained before completion")));
            } catch (...) {}
        }
    }
    void DeviceOps::commit_failure(
            std::uint64_t sequence, std::exception_ptr failure) {
        if (sequence == 0)
            throw std::invalid_argument("retained failure sequence is zero");
        if (!failure) throw std::invalid_argument("retained failure is empty");
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (sequence >= next_sequence_)
            throw std::invalid_argument(
                    "retained failure is not for a reserved submission");
        if (sequence <= completed_)
            throw std::invalid_argument(
                    "retained failure is for a sequence that has already been completed");
        const auto skipped = skipped_sequences_.upper_bound(sequence);
        if (skipped != skipped_sequences_.begin()) {
            auto previous = skipped;
            --previous;
            if (previous->second > sequence)
                throw std::invalid_argument(
                        "retained failure is not for a reserved submission");
        }
        pending_failures_[sequence] = std::move(failure);
    }
    void DeviceOps::seek_next_sequence(std::uint64_t next_sequence) {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (next_sequence == 0 || next_sequence < next_sequence_
                || next_sequence > kMaxSequence + 1)
            throw std::invalid_argument("queue sequence numbers only move forward");
        if (next_sequence > next_sequence_)
            skipped_sequences_.emplace(next_sequence_, next_sequence);
        next_sequence_ = next_sequence;
    }

}  // namespace iom
