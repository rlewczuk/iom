// SYCL RMS normalization queue path.
//
// One native in-order `parallel_for` launch over the logical planes and rows
// of the `docs/BACKEND_CONTRACT.md` **RMS normalization** contract. The
// kernel decodes the applicable encoded leaf on device, reduces squares over
// exactly the logical `F` features of its own row, normalizes and scales in
// the leaf's accumulator domain, and encodes one result per output element.
// Nothing is staged, relocated, or computed on the host: admission already
// validated the operands, so this translation unit only snapshots the request
// into a bounded trivially copyable descriptor, submits the device kernel on
// the existing in-order queue, and projects completion through the existing
// fence, owner-registration, and quarantine machinery.
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

namespace iom::sycl_detail {
namespace {

// The common admission contract admits rank two through eight, so a leading
// rank above six cannot reach this descriptor; the bound is a descriptor
// capacity guard, not a second admission check.
inline constexpr std::size_t kRmsnormMaxLeadingRank = 6;
inline constexpr std::uint64_t kRmsnormTile = TensorSpec::TILE;
inline constexpr std::uint64_t kRmsnormTileSlots =
        static_cast<std::uint64_t>(TensorSpec::TILE)
        * static_cast<std::uint64_t>(TensorSpec::TILE);

// Immutable bounded snapshot of one admitted request: R, F, the exact leaf
// encoding, every operand plane offset and leading stride, and the checked
// plane count. It is captured by value into the device kernel, so the task
// retains no borrowed view and the queue needs no extra slot or allocation.
struct RmsnormMetadata {
    std::uint64_t plane_count = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t leading_rank = 0;
    std::uint64_t x_plane_offset = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t out_plane_offset = 0;
    std::uint64_t leading_dimensions[kRmsnormMaxLeadingRank]{};
    std::uint64_t x_strides[kRmsnormMaxLeadingRank]{};
    std::uint64_t out_strides[kRmsnormMaxLeadingRank]{};
    std::uint32_t bits = 0;
    std::uint32_t data_type = 0;
};

static_assert(std::is_trivially_copyable_v<RmsnormMetadata>);

[[nodiscard]] RmsnormMetadata build_rmsnorm_metadata(
        const std::span<const std::size_t> dimensions,
        const std::span<const std::size_t> x_strides,
        const std::span<const std::size_t> out_strides,
        const std::size_t x_plane_offset,
        const std::size_t scale_plane_offset,
        const std::size_t out_plane_offset, const DataType data_type) {
    const std::size_t leading_rank = dimensions.size() - 2;
    if (leading_rank > kRmsnormMaxLeadingRank) {
        throw std::overflow_error(
                "SYCL RMSNorm leading rank exceeds its descriptor");
    }
    RmsnormMetadata metadata;
    metadata.rows = dimensions[leading_rank];
    metadata.columns = dimensions[leading_rank + 1];
    metadata.leading_rank = leading_rank;
    metadata.bits = static_cast<std::uint32_t>(
            detail::leaf_bits(data_type));
    metadata.data_type = static_cast<std::uint32_t>(data_type);
    metadata.x_plane_offset = x_plane_offset;
    metadata.scale_plane_offset = scale_plane_offset;
    metadata.out_plane_offset = out_plane_offset;
    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        metadata.leading_dimensions[axis] = dimensions[axis];
        metadata.x_strides[axis] = x_strides[axis];
        metadata.out_strides[axis] = out_strides[axis];
        plane_count *= dimensions[axis];
    }
    metadata.plane_count = plane_count;
    return metadata;
}

// Element slot of (row, column) inside one physical plane of the standard
// 16x16 tiled layout. Device code cannot call `detail::standard_plane_slot`,
// so the same tile-slot mapping is expressed here once for the two matrix
// axes. Precondition: row and column address the logical matrix.
[[nodiscard]] inline std::uint64_t rmsnorm_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) noexcept {
    const std::uint64_t tile_rows = (rows + kRmsnormTile - 1) / kRmsnormTile;
    const std::uint64_t tile_columns =
            (columns + kRmsnormTile - 1) / kRmsnormTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kRmsnormTile) * tile_columns + column / kRmsnormTile;
    return tile_index * kRmsnormTileSlots
            + (row % kRmsnormTile) * kRmsnormTile
            + column % kRmsnormTile;
}

// 32-bit word of the packed plane stream. Little-endian byte assembly keeps
// the access unaligned and byte-granular, exactly like the shared tiled
// copy/add word helpers.
[[nodiscard]] inline std::uint32_t rmsnorm_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value = 0;
    for (unsigned int index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(base[word * 4 + index])
                << (8 * index);
    }
    return value;
}

inline void rmsnorm_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned int index = 0; index < 4; ++index) {
        base[word * 4 + index] = static_cast<unsigned char>(
                value >> (8 * index));
    }
}

// One packed carrier of `bits` bits at bit offset `bit` of a plane stream.
[[nodiscard]] inline std::uint64_t rmsnorm_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned int bits) noexcept {
    const std::uint64_t word = bit / 32;
    std::uint64_t joined = rmsnorm_load_word(base, word);
    if (bit % 32 + bits > 32) {
        joined |= static_cast<std::uint64_t>(rmsnorm_load_word(base, word + 1))
                << 32;
    }
    return (joined >> (bit % 32))
            & (bits == 64
               ? ~std::uint64_t{}
               : ((std::uint64_t{1} << bits) - 1));
}

// Writes one packed carrier, touching only the bits it owns: padding bits,
// neighbouring carriers, and padding slots keep their observed value. Every
// physical tile row occupies a whole number of 32-bit words for all
// applicable even leaf widths, so a work item's row never shares a word with
// another work item's row.
inline void rmsnorm_store_bits(
        unsigned char* base, std::uint64_t bit, unsigned int bits,
        std::uint64_t value) noexcept {
    const std::uint64_t word = bit / 32;
    const unsigned int offset = static_cast<unsigned int>(bit % 32);
    if (offset + bits <= 32) {
        const std::uint32_t mask =
                (bits == 32 ? 0xffffffffu : ((std::uint32_t{1} << bits) - 1))
                << offset;
        const std::uint32_t observed = rmsnorm_load_word(base, word);
        rmsnorm_store_word(
                base, word,
                (observed & ~mask)
                        | ((static_cast<std::uint32_t>(value) << offset)
                           & mask));
        return;
    }
    const unsigned int low_bits = 32 - offset;
    const std::uint32_t low_mask = 0xffffffffu << offset;
    const std::uint32_t low_observed = rmsnorm_load_word(base, word);
    rmsnorm_store_word(
            base, word,
            (low_observed & ~low_mask)
                    | ((static_cast<std::uint32_t>(value) << offset)
                       & low_mask));
    const unsigned int high_bits = bits - low_bits;
    const std::uint32_t high_mask = high_bits == 32
            ? 0xffffffffu
            : ((std::uint32_t{1} << high_bits) - 1);
    const std::uint32_t high_observed = rmsnorm_load_word(base, word + 1);
    rmsnorm_store_word(
            base, word + 1,
            (high_observed & ~high_mask)
                    | (static_cast<std::uint32_t>(value >> low_bits)
                       & high_mask));
}

// Accumulator-domain device traits for the shared named-format codec. The
// `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
// and `F32` leaves use the FP32 carrier; `F64` keeps the FP64 carrier.
template <typename Accumulator>
struct RmsnormTraits {
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

[[nodiscard]] inline float rmsnorm_rsqrt(float value) noexcept {
    return sycl::rsqrt(value);
}

[[nodiscard]] inline double rmsnorm_rsqrt(double value) noexcept {
    return sycl::rsqrt(value);
}

// One logical row: decode exactly the logical features `[0, F)` of the
// selected plane, reduce their squares in the accumulator, then normalize
// and scale each feature and encode the result once.
template <typename Accumulator>
void rmsnorm_row(
        const unsigned char* x, const unsigned char* scale,
        unsigned char* out, const RmsnormMetadata& metadata,
        Accumulator epsilon, std::uint64_t plane_row) noexcept {
    using Codec =
            detail::scalar_binary_codec_detail::Codec<
                    RmsnormTraits<Accumulator>>;
    const auto format =
            Codec::format(static_cast<DataType>(metadata.data_type));
    const std::uint64_t plane = plane_row / metadata.rows;
    const std::uint64_t row = plane_row % metadata.rows;
    std::uint64_t x_plane = metadata.x_plane_offset;
    std::uint64_t out_plane = metadata.out_plane_offset;
    std::uint64_t rest = plane;
    for (std::uint64_t axis = metadata.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % metadata.leading_dimensions[axis];
        rest /= metadata.leading_dimensions[axis];
        x_plane += coordinate * metadata.x_strides[axis];
        out_plane += coordinate * metadata.out_strides[axis];
    }

    Accumulator sum = static_cast<Accumulator>(0);
    for (std::uint64_t column = 0; column < metadata.columns; ++column) {
        const std::uint64_t offset =
                rmsnorm_plane_slot(
                        x_plane, row, column, metadata.rows,
                        metadata.columns)
                * metadata.bits;
        const Accumulator feature = Codec::decode(
                rmsnorm_load_bits(x, offset, metadata.bits), format);
        sum = sum + feature * feature;
    }
    const Accumulator mean =
            sum / static_cast<Accumulator>(metadata.columns);
    const Accumulator inverse = rmsnorm_rsqrt(mean + epsilon);

    for (std::uint64_t column = 0; column < metadata.columns; ++column) {
        const std::uint64_t x_offset =
                rmsnorm_plane_slot(
                        x_plane, row, column, metadata.rows,
                        metadata.columns)
                * metadata.bits;
        // The `[1, F]` scale is one plane of a single row, shared by every
        // leading plane and row of this request.
        const std::uint64_t scale_offset =
                rmsnorm_plane_slot(
                        metadata.scale_plane_offset, 0, column, 1,
                        metadata.columns)
                * metadata.bits;
        const Accumulator feature = Codec::decode(
                rmsnorm_load_bits(x, x_offset, metadata.bits), format);
        const Accumulator factor = Codec::decode(
                rmsnorm_load_bits(scale, scale_offset, metadata.bits),
                format);
        const Accumulator normalized = feature * inverse;
        Accumulator scaled = normalized * factor;
        // Contract clause 3 and 9: a NaN result is stored in its canonical
        // positive form by clearing its sign before the single destination
        // encode. NaN payloads and NaN signs are outside the contract, and
        // only the positive canonical form also satisfies the published
        // comparison for the narrow finite-only leaves, where a NaN saturates
        // to a finite encoding instead of a NaN one. The CPU port applies the
        // identical rule, so every backend encodes the same result
        // independently of the device's invalid-operation NaN sign.
        if (RmsnormTraits<Accumulator>::isnan(scaled)) {
            scaled = RmsnormTraits<Accumulator>::fabs(scaled);
        }
        const std::uint64_t out_offset =
                rmsnorm_plane_slot(
                        out_plane, row, column, metadata.rows,
                        metadata.columns)
                * metadata.bits;
        rmsnorm_store_bits(
                out, out_offset, metadata.bits,
                Codec::encode(scaled, format));
    }
}

// The one native launch of the operation: one work item per independent
// (plane, row) pair on the existing in-order queue.
template <typename Accumulator>
[[nodiscard]] sycl::event launch_rmsnorm_rows(
        sycl::queue& queue, const unsigned char* x,
        const unsigned char* scale, unsigned char* out,
        const RmsnormMetadata& metadata, Accumulator epsilon) {
    const std::size_t rows = static_cast<std::size_t>(metadata.rows);
    const std::size_t plane_count =
            static_cast<std::size_t>(metadata.plane_count);
    return queue.parallel_for(
            sycl::range<1>(plane_count * rows),
            [=](sycl::id<1> item) {
                rmsnorm_row<Accumulator>(
                        x, scale, out, metadata, epsilon, item[0]);
            });
}

template <typename Request>
[[nodiscard]] RmsnormMetadata rmsnorm_metadata(const Request& request) {
    return build_rmsnorm_metadata(
            request.x.spec.shape.dimensions(), request.x.plane_strides,
            request.out.plane_strides, request.x.plane_offset,
            request.scale.plane_offset, request.out.plane_offset,
            request.x.spec.data_type);
}

template <typename Request>
[[nodiscard]] sycl::event launch_rmsnorm(
        sycl::queue& queue, const Request& request,
        const RmsnormMetadata& metadata) {
    const auto* x = static_cast<const unsigned char*>(
            request.x.native_handle);
    const auto* scale = static_cast<const unsigned char*>(
            request.scale.native_handle);
    auto* out = static_cast<unsigned char*>(request.out.native_handle);
    if (request.x.spec.data_type == DataType::F64) {
        return launch_rmsnorm_rows<double>(
                queue, x, scale, out, metadata,
                static_cast<double>(request.epsilon));
    }
    return launch_rmsnorm_rows<float>(
            queue, x, scale, out, metadata, request.epsilon);
}

}  // namespace

bool SyclQueue::rmsnorm_supported(DataType data_type) const {
    switch (data_type) {
        case DataType::F4_E2M1:
        case DataType::F6_E2M3:
        case DataType::F6_E3M2:
        case DataType::F8_E4M3FN:
        case DataType::F8_E5M2:
        case DataType::F16:
        case DataType::BF16:
        case DataType::F32:
            return true;
        // The FP64 leaf keeps every reduction and product step in FP64, so it
        // is queueable exactly on a device that reports the FP64 aspect; it
        // is never widened or emulated here.
        case DataType::F64:
            return fp64_supported_;
        case DataType::BOOL:
        case DataType::I2:
        case DataType::U2:
        case DataType::I4:
        case DataType::U4:
        case DataType::I8:
        case DataType::U8:
        case DataType::I16:
        case DataType::U16:
        case DataType::I32:
        case DataType::U32:
        case DataType::I64:
        case DataType::U64:
        case DataType::F8_E8M0:
            return false;
    }
    return false;
}

oid SyclQueue::rmsnorm_impl(const RmsnormRequest& request) {
    std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
    if (consume_submission_fault(SubmissionFault::state_allocation)) {
        throw std::bad_alloc();
    }
    auto state = std::make_shared<SyclFenceState>();
    if (consume_submission_fault(SubmissionFault::fence_construction)) {
        throw std::bad_alloc();
    }
    detail::Fence fence = build_sycl_fence(state);
    return submit_rmsnorm(
            request, *state_, registry_queue_id_, fence,
            [this, state](
                    std::uint64_t sequence,
                    const RmsnormRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                Task task;
                task.sequence = sequence;
                task.state = state;
                task.fence = state.get();
                task.rmsnorm_entries = entries;
                task.rmsnorm_request.emplace(captured);
                worker_.submit_copy(std::move(task));
            });
}

void SyclQueue::execute_rmsnorm(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    {
        const auto [it, inserted] = outcomes_.emplace(
                task.sequence,
                SyclSequenceOutcome{
                        detail::SequenceOutcome{}, task.state, std::nullopt,
                        {}, task.rmsnorm_entries});
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
        const RmsnormMetadata metadata = rmsnorm_metadata(
                *task.rmsnorm_request);
        if (consume_submission_fault(SubmissionFault::first_submit)) {
            throw std::runtime_error(
                    "injected SYCL first-submit failure");
        }
        // Once the native enqueue is attempted, the accepted outcome is
        // retained even if the enqueue throws, because the runtime may have
        // submitted work before reporting the error and only a successful
        // drain proves completion.
        native_attempted = true;
        sycl::event event = launch_rmsnorm(
                queue_, *task.rmsnorm_request, metadata);
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