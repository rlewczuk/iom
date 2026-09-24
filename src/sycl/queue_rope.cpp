// Native SYCL split-half rotary position encoding.
//
// Admission and immutable owner snapshots belong to DeviceOps.  This file
// keeps the accepted request value-copied, uploads only fixed control metadata,
// and performs every decode, trigonometric operation, pair arithmetic, and
// named-format encode in one native in-order device kernel.
#include "queue_internal.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "runtime.hpp"
#include "../iom_internal.hpp"

#include "scalar_binary_codec.hpp"

namespace iom::sycl_detail {
namespace {

inline constexpr std::size_t kRopeMaxLeadingRank = 6;
inline constexpr std::uint64_t kRopeTile = TensorSpec::TILE;
inline constexpr std::uint64_t kRopeTileSlots =
        static_cast<std::uint64_t>(TensorSpec::TILE)
        * static_cast<std::uint64_t>(TensorSpec::TILE);

// The final two TensorSpec axes are the tiled row/feature axes.  RoPE's H
// axis is therefore one of the leading plane axes and is retained in this
// descriptor along with all transformed leading strides.
struct RopeMetadata {
    std::uint64_t plane_count = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t leading_rank = 0;
    std::uint64_t x_plane_offset = 0;
    std::uint64_t out_plane_offset = 0;
    std::uint64_t leading_dimensions[kRopeMaxLeadingRank]{};
    std::uint64_t x_strides[kRopeMaxLeadingRank]{};
    std::uint64_t out_strides[kRopeMaxLeadingRank]{};
    std::uint64_t start_position = 0;
    float theta = 1.0F;
    std::uint32_t bits = 0;
    std::uint32_t data_type = 0;
};

static_assert(std::is_trivially_copyable_v<RopeMetadata>);
static_assert(sizeof(RopeMetadata) <= detail::kMetadataSlotBytes);

[[nodiscard]] std::size_t rope_checked_mul(
        std::size_t lhs, std::size_t rhs, const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

template <typename Request>
[[nodiscard]] RopeMetadata build_rope_metadata(const Request& request) {
    const std::span<const std::size_t> dimensions =
            request.x.shape_dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    if (leading_rank > kRopeMaxLeadingRank) {
        throw std::overflow_error("SYCL RoPE leading rank exceeds descriptor");
    }

    RopeMetadata metadata;
    metadata.rows = static_cast<std::uint64_t>(dimensions[leading_rank]);
    metadata.columns =
            static_cast<std::uint64_t>(dimensions[leading_rank + 1]);
    metadata.leading_rank = static_cast<std::uint64_t>(leading_rank);
    metadata.x_plane_offset =
            static_cast<std::uint64_t>(request.x.plane_offset);
    metadata.out_plane_offset =
            static_cast<std::uint64_t>(request.out.plane_offset);
    metadata.start_position = static_cast<std::uint64_t>(request.a);
    metadata.theta = static_cast<float>(request.theta);
    metadata.bits = static_cast<std::uint32_t>(
            detail::leaf_bits(request.x.data_type));
    metadata.data_type = static_cast<std::uint32_t>(request.x.data_type);

    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        metadata.leading_dimensions[axis] =
                static_cast<std::uint64_t>(dimensions[axis]);
        metadata.x_strides[axis] = static_cast<std::uint64_t>(
                request.x.plane_strides[axis]);
        metadata.out_strides[axis] = static_cast<std::uint64_t>(
                request.out.plane_strides[axis]);
        plane_count = rope_checked_mul(
                plane_count, dimensions[axis],
                "SYCL RoPE plane count overflows");
    }
    metadata.plane_count = static_cast<std::uint64_t>(plane_count);
    return metadata;
}

// Device-side packed-carrier access.  Every logical row starts on a whole
// 32-bit word boundary for the admitted even formats, so one work item owns
// all writes for one row and no two work items race while preserving padding
// and neighboring sub-byte carriers.
[[nodiscard]] inline std::uint32_t rope_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value = 0;
    for (unsigned int index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(base[word * 4 + index])
                << (8 * index);
    }
    return value;
}

inline void rope_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned int index = 0; index < 4; ++index) {
        base[word * 4 + index] = static_cast<unsigned char>(
                value >> (8 * index));
    }
}

[[nodiscard]] inline std::uint64_t rope_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned int bits) noexcept {
    const std::uint64_t word = bit / 32;
    const unsigned int offset = static_cast<unsigned int>(bit % 32);
    std::uint64_t joined = rope_load_word(base, word);
    if (offset + bits > 32) {
        joined |= static_cast<std::uint64_t>(
                          rope_load_word(base, word + 1))
                << 32;
    }
    const std::uint64_t mask = bits == 32
            ? ~std::uint64_t{}
            : (std::uint64_t{1} << bits) - 1;
    return (joined >> offset) & mask;
}

inline void rope_store_bits(
        unsigned char* base, std::uint64_t bit, unsigned int bits,
        std::uint64_t value) noexcept {
    const std::uint64_t word = bit / 32;
    const unsigned int offset = static_cast<unsigned int>(bit % 32);
    if (offset + bits <= 32) {
        const std::uint32_t field_mask = bits == 32
                ? 0xffffffffu
                : (std::uint32_t{1} << bits) - 1;
        const std::uint32_t mask = field_mask << offset;
        const std::uint32_t observed = rope_load_word(base, word);
        rope_store_word(
                base, word,
                (observed & ~mask)
                        | ((static_cast<std::uint32_t>(value)
                            << offset)
                           & mask));
        return;
    }

    const unsigned int low_bits = 32 - offset;
    const std::uint32_t low_mask = 0xffffffffu << offset;
    const std::uint32_t low_observed = rope_load_word(base, word);
    rope_store_word(
            base, word,
            (low_observed & ~low_mask)
                    | ((static_cast<std::uint32_t>(value) << offset)
                       & low_mask));

    const unsigned int high_bits = bits - low_bits;
    const std::uint32_t high_mask = high_bits == 32
            ? 0xffffffffu
            : (std::uint32_t{1} << high_bits) - 1;
    const std::uint32_t high_observed = rope_load_word(base, word + 1);
    rope_store_word(
            base, word + 1,
            (high_observed & ~high_mask)
                    | (static_cast<std::uint32_t>(value >> low_bits)
                       & high_mask));
}

[[nodiscard]] inline std::uint64_t rope_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) noexcept {
    const std::uint64_t tile_rows = (rows + kRopeTile - 1) / kRopeTile;
    const std::uint64_t tile_columns =
            (columns + kRopeTile - 1) / kRopeTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kRopeTile) * tile_columns + column / kRopeTile;
    return tile_index * kRopeTileSlots
            + (row % kRopeTile) * kRopeTile + column % kRopeTile;
}

struct RopeFloatTraits {
    using carrier_type = float;

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

using RopeCodec = detail::scalar_binary_codec_detail::Codec<RopeFloatTraits>;

// One device work item owns one independent (plane, head, row) and therefore
// all packed stores for that row.  This both preserves arbitrary output
// padding and makes sub-byte read/modify/write carriers race-free.
inline void rope_row(
        const unsigned char* x, unsigned char* out,
        const RopeMetadata& metadata, std::uint64_t plane_row) noexcept {
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

    const std::uint64_t half = metadata.columns / 2;
    const auto format = RopeCodec::format(
            static_cast<DataType>(metadata.data_type));
    const float position = static_cast<float>(
            metadata.start_position + row);
    if (metadata.start_position == 0 && row == 0) {
        for (std::uint64_t j = 0; j < half; ++j) {
            const std::uint64_t first_bit =
                    rope_plane_slot(
                            x_plane, row, j, metadata.rows,
                            metadata.columns)
                    * metadata.bits;
            const std::uint64_t second_bit =
                    rope_plane_slot(
                            x_plane, row, j + half, metadata.rows,
                            metadata.columns)
                    * metadata.bits;
            const std::uint64_t first_raw = rope_load_bits(
                    x, first_bit, static_cast<unsigned int>(metadata.bits));
            const std::uint64_t second_raw = rope_load_bits(
                    x, second_bit, static_cast<unsigned int>(metadata.bits));
            rope_store_bits(
                    out,
                    rope_plane_slot(
                            out_plane, row, j, metadata.rows,
                            metadata.columns)
                            * metadata.bits,
                    static_cast<unsigned int>(metadata.bits), first_raw);
            rope_store_bits(
                    out,
                    rope_plane_slot(
                            out_plane, row, j + half, metadata.rows,
                            metadata.columns)
                            * metadata.bits,
                    static_cast<unsigned int>(metadata.bits), second_raw);
        }
        return;
    }

    for (std::uint64_t j = 0; j < half; ++j) {
        const std::uint64_t first_bit =
                rope_plane_slot(
                        x_plane, row, j, metadata.rows, metadata.columns)
                * metadata.bits;
        const std::uint64_t second_bit =
                rope_plane_slot(
                        x_plane, row, j + half, metadata.rows,
                        metadata.columns)
                * metadata.bits;
        const std::uint64_t first_raw = rope_load_bits(
                x, first_bit, static_cast<unsigned int>(metadata.bits));
        const std::uint64_t second_raw = rope_load_bits(
                x, second_bit, static_cast<unsigned int>(metadata.bits));
        const float first = RopeCodec::decode(first_raw, format);
        const float second = RopeCodec::decode(second_raw, format);
        const float exponent =
                (-2.0F * static_cast<float>(j))
                / static_cast<float>(metadata.columns);
        const float frequency = sycl::pow(metadata.theta, exponent);
        const float angle = position * frequency;
        const float sine = sycl::sin(angle);
        const float cosine = sycl::cos(angle);

        // Volatile intermediates make the written noncontracted operation
        // boundaries explicit even on devices with fused multiply-add.
        const volatile float first_product = first * cosine;
        const volatile float second_product = second * sine;
        const volatile float first_output = first_product - second_product;
        const volatile float second_cosine_product = second * cosine;
        const volatile float first_sine_product = first * sine;
        const volatile float second_output =
                second_cosine_product + first_sine_product;
        const std::uint64_t first_encoded = RopeCodec::encode(
                first_output, format);
        const std::uint64_t second_encoded = RopeCodec::encode(
                second_output, format);

        rope_store_bits(
                out,
                rope_plane_slot(
                        out_plane, row, j, metadata.rows,
                        metadata.columns)
                        * metadata.bits,
                static_cast<unsigned int>(metadata.bits), first_encoded);
        rope_store_bits(
                out,
                rope_plane_slot(
                        out_plane, row, j + half, metadata.rows,
                        metadata.columns)
                        * metadata.bits,
                static_cast<unsigned int>(metadata.bits), second_encoded);
    }
}

template <typename Request>
[[nodiscard]] sycl::event launch_rope_kernel(
        sycl::queue& queue, const sycl::event& metadata_event,
        const Request& request, const RopeMetadata* metadata,
        std::size_t work_items) {
    const auto* x = static_cast<const unsigned char*>(
            request.x.native_handle);
    auto* out = static_cast<unsigned char*>(request.out.native_handle);
    return queue.submit([&](sycl::handler& handler) {
        handler.depends_on(metadata_event);
        handler.parallel_for(
                sycl::range<1>(work_items),
                [=](sycl::id<1> item) {
                    rope_row(x, out, *metadata,
                             static_cast<std::uint64_t>(item[0]));
                });
    });
}

}  // namespace

WorkspaceRequirements SyclQueue::rope_workspace_requirements(
        const RopeRequest& request) {
    switch (request.x.data_type) {
        case DataType::F4_E2M1:
        case DataType::F6_E2M3:
        case DataType::F6_E3M2:
        case DataType::F8_E4M3FN:
        case DataType::F8_E5M2:
        case DataType::F16:
        case DataType::BF16:
        case DataType::F32:
            return {0, 1};
        case DataType::F64:
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
            throw detail::UnsupportedOperation();
    }
    throw detail::UnsupportedOperation();
}

oid SyclQueue::rope_impl(const RopeRequest& request) {
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
    return submit_rope(
            request, *state_, registry_queue_id_, fence,
            [this, state](
                    std::uint64_t sequence, const RopeRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                Task task;
                task.sequence = sequence;
                task.state = state;
                task.fence = state.get();
                task.rope_request.emplace(captured);
                task.rope_entries = entries;
                worker_.submit_copy(std::move(task));
            });
}

void SyclQueue::execute_rope(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        const auto [it, inserted] = outcomes_.emplace(
                task.sequence,
                SyclSequenceOutcome{
                        detail::SequenceOutcome{}, task.state,
                        std::nullopt, {}, std::nullopt, false,
                        std::nullopt, task.rope_entries});
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
        task.state->set_metadata_slot(*metadata_pool_, *metadata_slot);
        const RopeMetadata metadata = build_rope_metadata(
                *task.rope_request);
        std::memcpy(
                metadata_pool_->host_data(*metadata_slot), &metadata,
                sizeof(metadata));
        const auto* metadata_device = static_cast<const RopeMetadata*>(
                metadata_pool_->device_data(*metadata_slot));
        if (metadata_device == nullptr) {
            throw std::bad_alloc();
        }
        if (consume_submission_fault(SubmissionFault::first_submit)) {
            throw std::runtime_error(
                    "injected SYCL first-submit failure");
        }
        // Once the metadata upload is attempted, retain the accepted outcome:
        // the runtime may have submitted work before reporting an enqueue
        // failure, and only the normal fence/drain path proves completion.
        native_attempted = true;
        const sycl::event metadata_event = queue_.memcpy(
                const_cast<RopeMetadata*>(metadata_device),
                metadata_pool_->host_data(*metadata_slot), sizeof(metadata));
        const sycl::event rope_event = launch_rope_kernel(
                queue_, metadata_event, *task.rope_request, metadata_device,
                rope_checked_mul(
                        static_cast<std::size_t>(metadata.plane_count),
                        static_cast<std::size_t>(metadata.rows),
                        "SYCL RoPE dispatch range overflows"));
        if (launch_calls.kernel_launched != nullptr) {
            launch_calls.kernel_launched();
        }
        task.state->set_event(std::move(rope_event));
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
