// SYCL linear-projection queue path.
//
// One native in-order `parallel_for` launch over the logical output rows of the
// `docs/BACKEND_CONTRACT.md` **Linear projections** contract. One work item
// owns one complete output row: it resolves that row's leading plane and head
// through the operand's own transformed plane offsets and strides, walks the
// logical inner extent `I` of the selected source row and of the weight row the
// output column names, and encodes exactly one result per logical output
// element. The integer leaves accumulate an unsigned modulo-`2^N` dot that
// reduces after every multiply and every add; the floating leaves start at `+0`
// and accumulate a correctly rounded fused multiply-add in FP32 (FP64 for
// `F64`). Nothing is staged, relocated, or computed on the host: admission
// already validated the operands, so this translation unit only snapshots the
// request into a bounded trivially copyable descriptor, submits the device
// kernel on the existing in-order queue, and projects completion through the
// existing fence, owner-registration, and quarantine machinery.
#include "queue_internal.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "runtime.hpp"
#include "../iom_internal.hpp"

namespace iom::sycl_detail {
namespace {

// The common admission contract admits rank two through eight, so neither the
// input's leading rank nor the output's — which may carry the one inserted head
// axis inside that bound — can exceed six; the bound is a descriptor capacity
// guard, not a second admission check.
inline constexpr std::size_t kLinearMaxLeadingRank = 6;
inline constexpr std::uint64_t kLinearTile = TensorSpec::TILE;
inline constexpr std::uint64_t kLinearTileSlots =
        static_cast<std::uint64_t>(TensorSpec::TILE)
        * static_cast<std::uint64_t>(TensorSpec::TILE);

// Immutable bounded snapshot of one admitted request: the logical row window,
// the exact leaf encoding and integer width, the head scalars, the explicit
// output mode, every operand plane offset, the output's own leading extents and
// transformed strides, and the input's transformed leading strides. It is
// captured by value into the device kernel, so the task retains no borrowed
// view and the queue needs no extra slot or allocation.
struct LinearMetadata {
    // Output planes: the product of the output view's own leading extents,
    // which already carries the head axis in head-planar mode.
    std::uint64_t plane_count = 0;
    std::uint64_t rows = 0;
    std::uint64_t source_rows = 0;
    std::uint64_t inner = 0;
    std::uint64_t outer = 0;
    std::uint64_t out_columns = 0;
    std::uint64_t head_dim = 0;
    std::uint64_t start_row = 0;
    std::uint64_t leading_rank = 0;
    std::uint64_t out_leading_rank = 0;
    std::uint64_t x_plane_offset = 0;
    std::uint64_t w_plane_offset = 0;
    std::uint64_t out_plane_offset = 0;
    std::uint64_t out_dimensions[kLinearMaxLeadingRank]{};
    std::uint64_t x_strides[kLinearMaxLeadingRank]{};
    std::uint64_t out_strides[kLinearMaxLeadingRank]{};
    std::uint32_t bits = 0;
    std::uint32_t data_type = 0;
    std::uint32_t integer_width = 0;
    std::uint32_t head_planar = 0;
};

static_assert(std::is_trivially_copyable_v<LinearMetadata>);

// Exact unsigned width of one integer leaf; zero for every floating leaf and
// for any leaf this port does not implement.
[[nodiscard]] std::uint32_t linear_integer_width(DataType data_type) {
    switch (data_type) {
        case DataType::I2: case DataType::U2: return 2;
        case DataType::I4: case DataType::U4: return 4;
        case DataType::I8: case DataType::U8: return 8;
        case DataType::I16: case DataType::U16: return 16;
        case DataType::I32: case DataType::U32: return 32;
        case DataType::I64: case DataType::U64: return 64;
        default:
            return 0;
    }
}

[[nodiscard]] LinearMetadata build_linear_metadata(
        const DeviceOps::LinearRequest& request) {
    const std::span<const std::size_t> x_dimensions =
            request.x.spec.shape.dimensions();
    const std::span<const std::size_t> w_dimensions =
            request.w.spec.shape.dimensions();
    const std::span<const std::size_t> out_dimensions =
            request.out.spec.shape.dimensions();
    const std::size_t leading_rank = x_dimensions.size() - 2;
    const std::size_t out_leading_rank = out_dimensions.size() - 2;
    if (leading_rank > kLinearMaxLeadingRank
            || out_leading_rank > kLinearMaxLeadingRank) {
        throw std::overflow_error(
                "SYCL linear leading rank exceeds its descriptor");
    }
    LinearMetadata metadata;
    metadata.rows = out_dimensions[out_leading_rank];
    metadata.source_rows = x_dimensions[leading_rank];
    metadata.inner = x_dimensions[leading_rank + 1];
    metadata.outer = w_dimensions[0];
    metadata.out_columns = out_dimensions[out_leading_rank + 1];
    metadata.head_dim = request.layout == LinearOutputLayout::head_planar
            ? request.head_dim : metadata.outer;
    metadata.start_row = request.start_row;
    metadata.leading_rank = leading_rank;
    metadata.out_leading_rank = out_leading_rank;
    metadata.x_plane_offset = request.x.plane_offset;
    metadata.w_plane_offset = request.w.plane_offset;
    metadata.out_plane_offset = request.out.plane_offset;
    metadata.bits = static_cast<std::uint32_t>(
            detail::leaf_bits(request.out.spec.data_type));
    metadata.data_type = static_cast<std::uint32_t>(
            request.out.spec.data_type);
    metadata.integer_width =
            linear_integer_width(request.out.spec.data_type);
    metadata.head_planar =
            request.layout == LinearOutputLayout::head_planar ? 1u : 0u;
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < out_leading_rank; ++axis) {
        metadata.out_dimensions[axis] = out_dimensions[axis];
        metadata.out_strides[axis] = request.out.plane_strides[axis];
        plane_count *= out_dimensions[axis];
    }
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        metadata.x_strides[axis] = request.x.plane_strides[axis];
    }
    metadata.plane_count = plane_count;
    return metadata;
}

// Element slot of (row, column) inside one physical plane of the standard
// 16x16 tiled layout. Device code cannot call `detail::standard_plane_slot`, so
// the same tile-slot mapping is expressed here once for the two matrix axes.
// Precondition: row and column address the logical matrix.
[[nodiscard]] inline std::uint64_t linear_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) noexcept {
    const std::uint64_t tile_rows = (rows + kLinearTile - 1) / kLinearTile;
    const std::uint64_t tile_columns =
            (columns + kLinearTile - 1) / kLinearTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kLinearTile) * tile_columns + column / kLinearTile;
    return tile_index * kLinearTileSlots
            + (row % kLinearTile) * kLinearTile
            + column % kLinearTile;
}

// 32-bit word of the packed plane stream. Little-endian byte assembly keeps the
// access unaligned and byte-granular, exactly like the shared tiled copy/add
// word helpers.
[[nodiscard]] inline std::uint32_t linear_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value = 0;
    for (unsigned int index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(base[word * 4 + index])
                << (8 * index);
    }
    return value;
}

inline void linear_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned int index = 0; index < 4; ++index) {
        base[word * 4 + index] = static_cast<unsigned char>(
                value >> (8 * index));
    }
}

// One packed carrier of `bits` bits at bit offset `bit` of a plane stream.
[[nodiscard]] inline std::uint64_t linear_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned int bits) noexcept {
    const std::uint64_t word = bit / 32;
    std::uint64_t joined = linear_load_word(base, word);
    if (bit % 32 + bits > 32) {
        joined |= static_cast<std::uint64_t>(linear_load_word(base, word + 1))
                << 32;
    }
    return (joined >> (bit % 32))
            & (bits == 64
               ? ~std::uint64_t{}
               : ((std::uint64_t{1} << bits) - 1));
}

// Writes one packed carrier, touching only the bits it owns: padding bits,
// neighbouring carriers, and padding slots keep their observed value. Every
// physical tile row occupies a whole number of 32-bit words for all applicable
// leaf widths, and one work item owns one complete logical output row inside
// one plane, so its carriers never share a word with another writer's carriers.
inline void linear_store_bits(
        unsigned char* base, std::uint64_t bit, unsigned int bits,
        std::uint64_t value) noexcept {
    const std::uint64_t word = bit / 32;
    const unsigned int offset = static_cast<unsigned int>(bit % 32);
    if (offset + bits <= 32) {
        const std::uint32_t mask =
                (bits == 32 ? 0xffffffffu : ((std::uint32_t{1} << bits) - 1))
                << offset;
        const std::uint32_t observed = linear_load_word(base, word);
        linear_store_word(
                base, word,
                (observed & ~mask)
                        | ((static_cast<std::uint32_t>(value) << offset)
                           & mask));
        return;
    }
    const unsigned int low_bits = 32 - offset;
    const std::uint32_t low_mask = 0xffffffffu << offset;
    const std::uint32_t low_observed = linear_load_word(base, word);
    linear_store_word(
            base, word,
            (low_observed & ~low_mask)
                    | ((static_cast<std::uint32_t>(value) << offset)
                       & low_mask));
    const unsigned int high_bits = bits - low_bits;
    const std::uint32_t high_mask = high_bits == 32
            ? 0xffffffffu
            : ((std::uint32_t{1} << high_bits) - 1);
    const std::uint32_t high_observed = linear_load_word(base, word + 1);
    linear_store_word(
            base, word + 1,
            (high_observed & ~high_mask)
                    | (static_cast<std::uint32_t>(value >> low_bits)
                       & high_mask));
}

// Accumulator-domain device traits for the shared named-format codec. The
// `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, and `F32`
// leaves use the FP32 carrier; `F64` keeps the FP64 carrier.
template <typename Accumulator>
struct LinearTraits {
    using carrier_type = Accumulator;

    static carrier_type positive_infinity() noexcept {
        return std::numeric_limits<carrier_type>::infinity();
    }
    static carrier_type quiet_nan() noexcept {
        return std::numeric_limits<carrier_type>::quiet_NaN();
    }
    static carrier_type max_finite() noexcept {
        return std::numeric_limits<carrier_type>::max();
    }
    static bool isnan(carrier_type value) noexcept {
        return sycl::isnan(value);
    }
    static bool isinf(carrier_type value) noexcept {
        return sycl::isinf(value);
    }
    static bool signbit(carrier_type value) noexcept {
        return sycl::signbit(value);
    }
    static carrier_type fabs(carrier_type value) noexcept {
        return sycl::fabs(value);
    }
    static carrier_type floor(carrier_type value) noexcept {
        return sycl::floor(value);
    }
    static carrier_type ldexp(carrier_type value, int exponent) noexcept {
        return sycl::ldexp(value, exponent);
    }
    static carrier_type frexp(carrier_type value, int* exponent) noexcept {
        return sycl::frexp(value, exponent);
    }
};

// One logical output row's fixed mapping: the transformed input plane, the
// transformed output plane, and the weight row group the head coordinate
// selects. The head coordinate is a leading coordinate of the output view, so
// it is resolved here once per row instead of once per element.
struct LinearRowMapping {
    std::uint64_t x_plane = 0;
    std::uint64_t out_plane = 0;
    std::uint64_t weight_row_base = 0;
    std::uint64_t row = 0;
};

[[nodiscard]] inline LinearRowMapping linear_row_mapping(
        const LinearMetadata& metadata,
        std::uint64_t out_plane_row) noexcept {
    LinearRowMapping mapping{
            metadata.x_plane_offset, metadata.out_plane_offset, 0, 0};
    const std::uint64_t plane = out_plane_row / metadata.rows;
    mapping.row = out_plane_row % metadata.rows;
    std::uint64_t rest = plane;
    for (std::uint64_t axis = metadata.out_leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % metadata.out_dimensions[axis];
        rest /= metadata.out_dimensions[axis];
        mapping.out_plane += coordinate * metadata.out_strides[axis];
        if (axis < metadata.leading_rank) {
            mapping.x_plane += coordinate * metadata.x_strides[axis];
            continue;
        }
        // The inserted head-planar axis: its coordinate is already part of the
        // output plane index and selects the weight row group `h*D`. Ordinary
        // mode has no such axis, so the base stays zero.
        mapping.weight_row_base = coordinate * metadata.head_dim;
    }
    return mapping;
}

// One logical output row of an integer leaf: the unsigned modulo-`2^N` dot of
// the selected source row and the named weight row, reduced after every
// multiply and every add, stored as its two's-complement bit pattern.
inline void linear_integer_row(
        const unsigned char* x, const unsigned char* w, unsigned char* out,
        const LinearMetadata& metadata,
        std::uint64_t out_plane_row) noexcept {
    const unsigned int width =
            static_cast<unsigned int>(metadata.integer_width);
    const std::uint64_t mask = width >= 64
            ? ~std::uint64_t{}
            : ((std::uint64_t{1} << width) - 1);
    const LinearRowMapping mapping =
            linear_row_mapping(metadata, out_plane_row);
    const std::uint64_t source_row = metadata.start_row + mapping.row;
    for (std::uint64_t column = 0; column < metadata.out_columns; ++column) {
        const std::uint64_t weight_row = mapping.weight_row_base + column;
        std::uint64_t sum = 0;
        for (std::uint64_t inner = 0; inner < metadata.inner; ++inner) {
            const std::uint64_t x_offset =
                    linear_plane_slot(
                            mapping.x_plane, source_row, inner,
                            metadata.source_rows, metadata.inner)
                    * metadata.bits;
            const std::uint64_t w_offset =
                    linear_plane_slot(
                            metadata.w_plane_offset, weight_row, inner,
                            metadata.outer, metadata.inner)
                    * metadata.bits;
            const std::uint64_t lhs =
                    linear_load_bits(x, x_offset, metadata.bits) & mask;
            const std::uint64_t rhs =
                    linear_load_bits(w, w_offset, metadata.bits) & mask;
            sum = (sum + lhs * rhs) & mask;
        }
        const std::uint64_t out_offset =
                linear_plane_slot(
                        mapping.out_plane, mapping.row, column, metadata.rows,
                        metadata.out_columns)
                * metadata.bits;
        linear_store_bits(out, out_offset, metadata.bits, sum);
    }
}

// One logical output row of a floating leaf: decode every operand carrier on
// device, accumulate from `+0` over increasing `i` with one correctly rounded
// fused multiply-add per step, and encode the accumulated sum exactly once.
template <typename Accumulator>
inline void linear_scalar_row(
        const unsigned char* x, const unsigned char* w, unsigned char* out,
        const LinearMetadata& metadata,
        std::uint64_t out_plane_row) noexcept {
    using Codec =
            detail::scalar_binary_codec_detail::Codec<
                    LinearTraits<Accumulator>>;
    const auto format =
            Codec::format(static_cast<DataType>(metadata.data_type));
    const LinearRowMapping mapping =
            linear_row_mapping(metadata, out_plane_row);
    const std::uint64_t source_row = metadata.start_row + mapping.row;
    for (std::uint64_t column = 0; column < metadata.out_columns; ++column) {
        const std::uint64_t weight_row = mapping.weight_row_base + column;
        Accumulator sum = static_cast<Accumulator>(0);
        for (std::uint64_t inner = 0; inner < metadata.inner; ++inner) {
            const std::uint64_t x_offset =
                    linear_plane_slot(
                            mapping.x_plane, source_row, inner,
                            metadata.source_rows, metadata.inner)
                    * metadata.bits;
            const std::uint64_t w_offset =
                    linear_plane_slot(
                            metadata.w_plane_offset, weight_row, inner,
                            metadata.outer, metadata.inner)
                    * metadata.bits;
            const Accumulator lhs = Codec::decode(
                    linear_load_bits(x, x_offset, metadata.bits), format);
            const Accumulator rhs = Codec::decode(
                    linear_load_bits(w, w_offset, metadata.bits), format);
            sum = sycl::fma(lhs, rhs, sum);
        }
        const std::uint64_t out_offset =
                linear_plane_slot(
                        mapping.out_plane, mapping.row, column, metadata.rows,
                        metadata.out_columns)
                * metadata.bits;
        linear_store_bits(
                out, out_offset, metadata.bits, Codec::encode(sum, format));
    }
}

[[nodiscard]] inline std::size_t linear_row_items(
        const LinearMetadata& metadata) noexcept {
    return static_cast<std::size_t>(
            metadata.plane_count * metadata.rows);
}

template <typename Accumulator>
[[nodiscard]] sycl::event launch_linear_scalar_rows(
        sycl::queue& queue, const unsigned char* x, const unsigned char* w,
        unsigned char* out, const LinearMetadata& metadata) {
    return queue.parallel_for(
            sycl::range<1>(linear_row_items(metadata)),
            [=](sycl::id<1> item) {
                linear_scalar_row<Accumulator>(
                        x, w, out, metadata, item[0]);
            });
}

// The one native launch of the operation: one work item per logical output row
// on the existing in-order queue, with the integer, FP32, and FP64 recurrences
// instantiated as separate kernels so no device path ever widens `F64` or
// borrows the integer dot.
[[nodiscard]] sycl::event launch_linear(
        sycl::queue& queue, const DeviceOps::LinearRequest& request,
        const LinearMetadata& metadata) {
    const auto* x = static_cast<const unsigned char*>(
            request.x.native_handle);
    const auto* w = static_cast<const unsigned char*>(
            request.w.native_handle);
    auto* out = static_cast<unsigned char*>(request.out.native_handle);
    if (metadata.integer_width != 0) {
        return queue.parallel_for(
                sycl::range<1>(linear_row_items(metadata)),
                [=](sycl::id<1> item) {
                    linear_integer_row(x, w, out, metadata, item[0]);
                });
    }
    if (request.out.spec.data_type == DataType::F64) {
        return launch_linear_scalar_rows<double>(queue, x, w, out, metadata);
    }
    return launch_linear_scalar_rows<float>(queue, x, w, out, metadata);
}

}  // namespace

bool SyclQueue::linear_leaf_supported(DataType data_type) const noexcept {
    switch (data_type) {
        case DataType::I2: case DataType::U2:
        case DataType::I4: case DataType::U4:
        case DataType::I8: case DataType::U8:
        case DataType::I16: case DataType::U16:
        case DataType::I32: case DataType::U32:
        case DataType::I64: case DataType::U64:
        case DataType::F4_E2M1:
        case DataType::F6_E2M3: case DataType::F6_E3M2:
        case DataType::F8_E4M3FN: case DataType::F8_E5M2:
        case DataType::F16:
        case DataType::F32:
            return true;
        // The FP64 leaf keeps every product and every accumulation step in
        // FP64, so it is queueable exactly on a device that reports the FP64
        // aspect; it is never widened from FP32 or emulated here.
        case DataType::F64:
            return fp64_supported_;
        // Native BF16 is a separate specialization of its own leaf, and the
        // two recognized inapplicable leaves have no linear semantics on any
        // backend.
        case DataType::BF16:
        case DataType::BOOL:
        case DataType::F8_E8M0:
            return false;
    }
    return false;
}

WorkspaceRequirements SyclQueue::linear_workspace_requirements_impl(
        const TensorView& x, const TensorView& w, const TensorView& out,
        std::size_t s, std::size_t R, LinearOutputLayout layout, std::size_t H,
        std::size_t D) {
    (void)w;
    (void)out;
    (void)s;
    (void)R;
    (void)layout;
    (void)H;
    (void)D;
    // Pure capability decision: the query allocates nothing, constructs no
    // snapshot, registers or leases no owner, and mutates no queue state. A
    // leaf this port does not implement — including `F64` on a device without
    // the FP64 aspect — is a capability rejection before any queue effect.
    if (!linear_leaf_supported(x.spec().data_type)) {
        throw detail::UnsupportedOperation();
    }
    // The scalar leaves consume no raw workspace: the exact contract
    // requirement is the `{0, 1}` zero-scratch path.
    return WorkspaceRequirements{0, 1};
}

oid SyclQueue::linear_impl(const LinearRequest& request) {
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
    return submit_linear(
            request, *state_, registry_queue_id_, fence,
            [this, state](
                    std::uint64_t sequence,
                    const LinearRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                Task task;
                task.sequence = sequence;
                task.state = state;
                task.fence = state.get();
                task.linear_request.emplace(captured);
                task.linear_entries = entries;
                worker_.submit_copy(std::move(task));
            });
}

void SyclQueue::execute_linear(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    {
        const auto [it, inserted] = outcomes_.emplace(
                task.sequence,
                SyclSequenceOutcome{
                        detail::SequenceOutcome{}, task.state, std::nullopt,
                        {}, std::nullopt, false, task.linear_entries});
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
        const LinearMetadata metadata =
                build_linear_metadata(*task.linear_request);
        if (consume_submission_fault(SubmissionFault::first_submit)) {
            throw std::runtime_error(
                    "injected SYCL first-submit failure");
        }
        // Once the native enqueue is attempted, the accepted outcome is
        // retained even if the enqueue throws, because the runtime may have
        // submitted work before reporting the error and only a successful
        // drain proves completion.
        native_attempted = true;
        sycl::event event = launch_linear(
                queue_, *task.linear_request, metadata);
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

}  // namespace iom::sycl_detail
