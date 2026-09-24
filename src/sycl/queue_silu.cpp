// Native SYCL SiLU activation for the eight non-F64 floating leaves.
//
// Admission and the immutable owner snapshots belong to `DeviceOps`; this file
// keeps the accepted request value-copied, uploads only a fixed control
// descriptor, and performs every decode, stable FP32 evaluation, and single
// target-format round-to-nearest-even store in one native in-order device
// `parallel_for`. Each work item exclusively owns one physical output word
// (or one aligned three-word packet for the six-bit leaves):
// it reads those words, replaces only the logical fields intersecting them,
// and stores each owned word once. There is no host task, no host arithmetic
// on operand data, no staging buffer, no data roundtrip, no hidden allocation,
// no fast-math or flush-to-zero, and no synchronous wait: the returned
// completion event is retained by the queue's fence until an explicit caller
// wait or drain.
//
// The ported leaf set is exactly `F4_E2M1`, `F6_E2M3`, `F6_E3M2`,
// `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, and `F32`. `F64` stays
// `Unsupported` because this path has no device-independent FP64 guarantee,
// regardless of `aspect::fp64`.
#include "queue_internal.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "runtime.hpp"
#include "../iom_internal.hpp"

#include "scalar_binary_codec.hpp"
#include "scalar_silu.hpp"

namespace iom::sycl_detail {
namespace {

inline constexpr std::size_t kSiluMaxLeadingRank = 6;
inline constexpr std::uint64_t kSiluTile = TensorSpec::TILE;
inline constexpr std::uint64_t kSiluTileSlots =
        static_cast<std::uint64_t>(TensorSpec::TILE)
        * static_cast<std::uint64_t>(TensorSpec::TILE);
inline constexpr std::uint64_t kSiluMaxPacketWords = 3;

// Immutable device-visible SiLU descriptor copied into one fixed metadata slot
// before submission. It owns every leading extent and stride the two views
// need, so no borrowed `TensorView` or `SiLUViewSnapshot` is read by the
// kernel and a temporary view may be destroyed immediately after the call.
struct SiluMetadata {
    std::uint64_t plane_count = 0;
    std::uint64_t plane_packets = 0;
    std::uint64_t packet_words = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t leading_rank = 0;
    std::uint64_t x_plane_offset = 0;
    std::uint64_t y_plane_offset = 0;
    std::uint64_t leading_dimensions[kSiluMaxLeadingRank]{};
    std::uint64_t x_strides[kSiluMaxLeadingRank]{};
    std::uint64_t y_strides[kSiluMaxLeadingRank]{};
    std::uint32_t bits = 0;
    std::uint32_t data_type = 0;
};

static_assert(std::is_trivially_copyable_v<SiluMetadata>);
static_assert(sizeof(SiluMetadata) <= detail::kMetadataSlotBytes);

[[nodiscard]] std::size_t silu_checked_add(
        std::size_t lhs, std::size_t rhs, const char* message) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(message);
    }
    return lhs + rhs;
}

[[nodiscard]] std::size_t silu_checked_mul(
        std::size_t lhs, std::size_t rhs, const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

[[nodiscard]] std::size_t silu_padded_dimension(
        std::size_t value, const char* message) {
    return silu_checked_mul(
            silu_checked_add(
                    value, static_cast<std::size_t>(kSiluTile - 1), message)
                    / static_cast<std::size_t>(kSiluTile),
            static_cast<std::size_t>(kSiluTile), message);
}

// Every supported leaf is packed into a named bit field in the standard
// tiled stream. The launcher derives no width for F64 or any inapplicable
// leaf, because `silu_device_supported` rejects those leaves first.
[[nodiscard]] std::uint32_t silu_leaf_bits(DataType data_type) {
    switch (data_type) {
        case DataType::F4_E2M1: return 4;
        case DataType::F6_E2M3: return 6;
        case DataType::F6_E3M2: return 6;
        case DataType::F8_E4M3FN: return 8;
        case DataType::F8_E5M2: return 8;
        case DataType::F16: return 16;
        case DataType::BF16: return 16;
        case DataType::F32: return 32;
        default: break;
    }
    throw detail::UnsupportedOperation();
}

// Element slot of (row, column) inside one physical plane of the standard
// 16x16 tiled layout. Device code cannot call `detail::standard_plane_slot`,
// so the same tile-slot mapping is expressed here once for the two matrix
// axes. Precondition: row and column address the logical matrix.
[[nodiscard]] inline std::uint64_t silu_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) noexcept {
    const std::uint64_t tile_rows = (rows + kSiluTile - 1) / kSiluTile;
    const std::uint64_t tile_columns =
            (columns + kSiluTile - 1) / kSiluTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kSiluTile) * tile_columns + column / kSiluTile;
    return tile_index * kSiluTileSlots
            + (row % kSiluTile) * kSiluTile + column % kSiluTile;
}

struct SiluPhysicalCoordinate {
    std::uint64_t row;
    std::uint64_t column;
};

[[nodiscard]] inline SiluPhysicalCoordinate silu_physical_coordinate(
        std::uint64_t slot, std::uint64_t rows,
        std::uint64_t columns) noexcept {
    const std::uint64_t tile_columns =
            (columns + kSiluTile - 1) / kSiluTile;
    const std::uint64_t tile_index = slot / kSiluTileSlots;
    const std::uint64_t in_tile = slot % kSiluTileSlots;
    return {
            (tile_index / tile_columns) * kSiluTile
                    + in_tile / kSiluTile,
            (tile_index % tile_columns) * kSiluTile
                    + in_tile % kSiluTile};
}

[[nodiscard]] inline std::uint32_t silu_field_mask(
        unsigned int bits) noexcept {
    return bits == 32
            ? 0xffffffffu
            : (std::uint32_t{1} << bits) - 1;
}

[[nodiscard]] inline std::uint32_t silu_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value = 0;
    for (unsigned int index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(base[word * 4 + index])
                << (8 * index);
    }
    return value;
}

inline void silu_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned int index = 0; index < 4; ++index) {
        base[word * 4 + index] = static_cast<unsigned char>(
                value >> (8 * index));
    }
}

[[nodiscard]] inline std::uint64_t silu_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned int bits) noexcept {
    const std::uint64_t word = bit / 32;
    const unsigned int offset = static_cast<unsigned int>(bit % 32);
    std::uint64_t joined = silu_load_word(base, word);
    if (offset + bits > 32) {
        joined |= static_cast<std::uint64_t>(
                          silu_load_word(base, word + 1))
                << 32;
    }
    const std::uint64_t mask = bits == 64
            ? ~std::uint64_t{}
            : (std::uint64_t{1} << bits) - 1;
    return (joined >> offset) & mask;
}

inline void silu_merge_field(
        std::uint32_t& destination_word, std::uint32_t value,
        unsigned int bit_offset, unsigned int bits) noexcept {
    const std::uint32_t mask = silu_field_mask(bits) << bit_offset;
    destination_word = (destination_word & ~mask)
            | ((value << bit_offset) & mask);
}

// FP32 evaluation traits of the shared SiLU evaluator and the shared named
// codec. `carrier_type` is the required FP32 domain, every predicate is the
// SYCL device math function, and `-ffp-model=precise` keeps subnormal and
// NaN/infinity behavior instead of the compiler's fast default model.
struct SiluFloatTraits {
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
    static carrier_type exp(carrier_type value) noexcept {
        return sycl::exp(value);
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

using SiluCodec = detail::scalar_binary_codec_detail::Codec<SiluFloatTraits>;

// One work item owns one physical output word, or one aligned packet of
// physical output words for six-bit leaves. It resolves both transformed
// leading-plane addresses, reads the existing destination words, evaluates
// each logical field intersecting the packet exactly once, merges only those
// field bits, and stores every owned word exactly once. Padding slots and
// padding bits are never decoded or changed.
inline void silu_packet(
        const unsigned char* x, unsigned char* y,
        const SiluMetadata& metadata, std::uint64_t plane,
        std::uint64_t packet_in_plane) noexcept {
    std::uint64_t x_plane = metadata.x_plane_offset;
    std::uint64_t y_plane = metadata.y_plane_offset;
    std::uint64_t rest = plane;
    for (std::uint64_t axis = metadata.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % metadata.leading_dimensions[axis];
        rest /= metadata.leading_dimensions[axis];
        x_plane += coordinate * metadata.x_strides[axis];
        y_plane += coordinate * metadata.y_strides[axis];
    }

    const auto format = SiluCodec::format(
            static_cast<DataType>(metadata.data_type));
    const std::uint64_t destination_base_slot = silu_plane_slot(
            y_plane, 0, 0, metadata.rows, metadata.columns);
    const std::uint64_t packet_first_word_in_plane =
            packet_in_plane * metadata.packet_words;
    const std::uint64_t packet_first_word =
            destination_base_slot * metadata.bits / 32
            + packet_first_word_in_plane;
    const std::uint64_t packet_first_bit = packet_first_word * 32;
    const std::uint64_t packet_last_bit = packet_first_bit
            + metadata.packet_words * 32;
    std::uint32_t merged[kSiluMaxPacketWords]{};
    for (std::uint64_t local_word = 0;
         local_word < metadata.packet_words; ++local_word) {
        merged[local_word] = silu_load_word(
                y, packet_first_word + local_word);
    }

    const std::uint64_t packet_local_first_bit =
            packet_first_word_in_plane * 32;
    const std::uint64_t packet_local_last_bit =
            packet_local_first_bit + metadata.packet_words * 32;
    const std::uint64_t first_slot =
            packet_local_first_bit / metadata.bits;
    const std::uint64_t last_slot =
            (packet_local_last_bit - 1) / metadata.bits;
    for (std::uint64_t slot = first_slot; slot <= last_slot; ++slot) {
        const std::uint64_t slot_bit = slot * metadata.bits;
        if (slot_bit + metadata.bits <= packet_local_first_bit
                || slot_bit >= packet_local_last_bit) {
            continue;
        }
        const SiluPhysicalCoordinate coordinate =
                silu_physical_coordinate(
                        slot, metadata.rows, metadata.columns);
        if (coordinate.row >= metadata.rows
                || coordinate.column >= metadata.columns) {
            continue;
        }
        const std::uint64_t input_bit = silu_plane_slot(
                x_plane, coordinate.row, coordinate.column,
                metadata.rows, metadata.columns) * metadata.bits;
        const std::uint64_t raw = silu_load_bits(
                x, input_bit, metadata.bits);
        const float value = SiluCodec::decode(raw, format);
        const std::uint64_t encoded = SiluCodec::encode(
                detail::scalar_silu_detail::evaluate<SiluFloatTraits>(value),
                format);
        const std::uint64_t output_bit = silu_plane_slot(
                y_plane, coordinate.row, coordinate.column,
                metadata.rows, metadata.columns) * metadata.bits;
        const std::uint64_t field_end = output_bit + metadata.bits;
        const std::uint64_t overlap_first =
                output_bit > packet_first_bit
                ? output_bit
                : packet_first_bit;
        const std::uint64_t overlap_end =
                field_end < packet_last_bit
                ? field_end
                : packet_last_bit;
        if (overlap_first >= overlap_end) continue;
        const std::uint64_t first_word = overlap_first / 32;
        const std::uint64_t last_word = (overlap_end - 1) / 32;
        for (std::uint64_t output_word = first_word;
             output_word <= last_word; ++output_word) {
            if (output_word < packet_first_word
                    || output_word >= packet_first_word
                            + metadata.packet_words) {
                continue;
            }
            const std::uint64_t word_first_bit = output_word * 32;
            const std::uint64_t word_overlap_first =
                    overlap_first > word_first_bit
                    ? overlap_first
                    : word_first_bit;
            const std::uint64_t word_overlap_end =
                    overlap_end < word_first_bit + 32
                    ? overlap_end
                    : word_first_bit + 32;
            if (word_overlap_first >= word_overlap_end) continue;
            const unsigned int destination_offset =
                    static_cast<unsigned int>(
                            word_overlap_first - word_first_bit);
            const unsigned int source_offset = static_cast<unsigned int>(
                    word_overlap_first - output_bit);
            const unsigned int count = static_cast<unsigned int>(
                    word_overlap_end - word_overlap_first);
            const std::uint32_t segment = static_cast<std::uint32_t>(
                    encoded >> source_offset) & silu_field_mask(count);
            silu_merge_field(
                    merged[output_word - packet_first_word], segment,
                    destination_offset, count);
        }
    }
    for (std::uint64_t local_word = 0;
         local_word < metadata.packet_words; ++local_word) {
        silu_store_word(
                y, packet_first_word + local_word, merged[local_word]);
    }
}

template <typename Request>
[[nodiscard]] sycl::event launch_silu_kernel(
        sycl::queue& queue, const sycl::event& metadata_event,
        const Request& request, const SiluMetadata* metadata,
        std::size_t work_items) {
    const auto* x = static_cast<const unsigned char*>(
            request.x.native_handle);
    auto* y = static_cast<unsigned char*>(request.y.native_handle);
    return queue.submit([&](sycl::handler& handler) {
        handler.depends_on(metadata_event);
        handler.parallel_for(
                sycl::range<1>(work_items),
                [=](sycl::id<1> item) {
                    const std::uint64_t index =
                            static_cast<std::uint64_t>(item[0]);
                    const std::uint64_t plane =
                            index / metadata->plane_packets;
                    if (plane >= metadata->plane_count) {
                        return;
                    }
                    const std::uint64_t packet_in_plane =
                            index - plane * metadata->plane_packets;
                    silu_packet(
                            x, y, *metadata, plane, packet_in_plane);
                });
    });
}

template <typename Request>
[[nodiscard]] SiluMetadata build_silu_metadata(const Request& request) {
    const std::span<const std::size_t> dimensions =
            request.x.shape_dimensions();
    const std::size_t rank = dimensions.size();
    if (rank < 2 || rank > 8 || request.y.rank != rank) {
        throw std::invalid_argument("SYCL SiLU rank is out of range");
    }
    const std::size_t leading_rank = rank - 2;
    if (leading_rank > kSiluMaxLeadingRank) {
        throw std::overflow_error(
                "SYCL SiLU leading rank exceeds descriptor");
    }

    SiluMetadata metadata;
    metadata.bits = silu_leaf_bits(request.x.data_type);
    metadata.data_type = static_cast<std::uint32_t>(request.x.data_type);
    metadata.leading_rank = static_cast<std::uint64_t>(leading_rank);
    metadata.rows = static_cast<std::uint64_t>(dimensions[leading_rank]);
    metadata.columns = static_cast<std::uint64_t>(
            dimensions[leading_rank + 1]);
    if (metadata.rows == 0 || metadata.columns == 0) {
        throw std::invalid_argument(
                "SYCL SiLU logical run and feature extents must be nonzero");
    }
    metadata.x_plane_offset =
            static_cast<std::uint64_t>(request.x.plane_offset);
    metadata.y_plane_offset =
            static_cast<std::uint64_t>(request.y.plane_offset);

    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        if (dimensions[axis] == 0) {
            throw std::invalid_argument(
                    "SYCL SiLU leading extents must be nonzero");
        }
        metadata.leading_dimensions[axis] =
                static_cast<std::uint64_t>(dimensions[axis]);
        metadata.x_strides[axis] = static_cast<std::uint64_t>(
                request.x.plane_strides[axis]);
        metadata.y_strides[axis] = static_cast<std::uint64_t>(
                request.y.plane_strides[axis]);
        plane_count = silu_checked_mul(
                plane_count, dimensions[axis],
                "SYCL SiLU plane count overflows");
    }
    const std::size_t padded_rows = silu_padded_dimension(
            dimensions[leading_rank], "SYCL SiLU padded row count overflows");
    const std::size_t padded_columns = silu_padded_dimension(
            dimensions[leading_rank + 1],
            "SYCL SiLU padded column count overflows");
    const std::size_t padded_elements = silu_checked_mul(
            padded_rows, padded_columns,
            "SYCL SiLU padded plane element count overflows");
    const std::size_t plane_bits = silu_checked_mul(
            padded_elements, metadata.bits,
            "SYCL SiLU padded plane bit count overflows");
    const std::size_t plane_words = silu_checked_add(
            plane_bits, 31, "SYCL SiLU plane word count overflows") / 32;
    const std::size_t packet_words = metadata.bits == 6 ? 3 : 1;
    if (plane_words % packet_words != 0) {
        throw std::overflow_error(
                "SYCL SiLU plane words are not packet aligned");
    }
    const std::size_t plane_packets = plane_words / packet_words;
    (void)silu_checked_mul(
            plane_count, plane_packets,
            "SYCL SiLU physical packet count overflows");
    metadata.plane_count = static_cast<std::uint64_t>(plane_count);
    metadata.plane_packets = static_cast<std::uint64_t>(plane_packets);
    metadata.packet_words = static_cast<std::uint64_t>(packet_words);
    return metadata;
}

}  // namespace

bool silu_device_supported(DataType data_type) noexcept {
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
        default:
            return false;
    }
}


void SyclQueue::validate_silu_representation(const SiLURequest& request) {
    (void)build_silu_metadata(request);
}

void SyclQueue::execute_silu(Task& task) {
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
                        std::nullopt, std::nullopt, std::nullopt,
                        task.silu_entries});
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
        const SiluMetadata metadata = build_silu_metadata(
                *task.silu_request);
        std::memcpy(
                metadata_pool_->host_data(*metadata_slot), &metadata,
                sizeof(metadata));
        const auto* metadata_device = static_cast<const SiluMetadata*>(
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
                const_cast<SiluMetadata*>(metadata_device),
                metadata_pool_->host_data(*metadata_slot),
                sizeof(metadata));
        const std::size_t work_items = silu_checked_mul(
                static_cast<std::size_t>(metadata.plane_count),
                static_cast<std::size_t>(metadata.plane_packets),
                "SYCL SiLU physical packet count overflows");
        const sycl::event silu_event = launch_silu_kernel(
                queue_, metadata_event, *task.silu_request, metadata_device,
                work_items);
        if (launch_calls.kernel_launched != nullptr) {
            launch_calls.kernel_launched();
        }
        task.state->set_event(std::move(silu_event));
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
