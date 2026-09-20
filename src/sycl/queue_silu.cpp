// Native SYCL BF16/F32 SiLU activation.
//
// Admission and the immutable owner snapshots belong to `DeviceOps`; this file
// keeps the accepted request value-copied, uploads only a fixed control
// descriptor, and performs every decode, stable FP32 evaluation, and single
// target-format round-to-nearest-even store in one native in-order device
// `parallel_for`. There is no host task, no host arithmetic on operand data,
// no staging buffer, no data roundtrip, no hidden allocation, no fast-math or
// flush-to-zero, and no synchronous wait: the returned completion event is
// retained by the queue's fence until an explicit caller wait or drain.
//
// The ported leaf set is exactly `BF16` and `F32`. `F64` stays `Unsupported`
// regardless of `aspect::fp64`, the six packed leaves are owned by
// `10-sycl-silu-packed-formats`, and every semantically inapplicable leaf
// (`BOOL`, the twelve integer leaves, and `F8_E8M0`) stays `Unsupported`. The
// descriptor, the FP32 traits, and the shared evaluator/codec helpers are the
// extension seam a packed-format leaf reuses with its own row- or word-owned
// work decomposition; nothing here needs to be redesigned for it.
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

// Immutable device-visible SiLU descriptor copied into one fixed metadata slot
// before submission. It owns every leading extent and stride the two views
// need, so no borrowed `TensorView` or `SiLUViewSnapshot` is read by the
// kernel and a temporary view may be destroyed immediately after the call.
struct SiluMetadata {
    std::uint64_t plane_count = 0;
    std::uint64_t plane_elements = 0;
    std::uint64_t element_count = 0;
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

[[nodiscard]] std::size_t silu_checked_mul(
        std::size_t lhs, std::size_t rhs, const char* message) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

// Whole-byte carrier width of the leaves this port queues. Every other leaf,
// including the packed formats of `10-sycl-silu-packed-formats`, is refused
// here as well as by `silu_device_supported`, so the launcher can never derive
// a width for a leaf it does not implement.
[[nodiscard]] std::uint32_t silu_leaf_bits(DataType data_type) {
    switch (data_type) {
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

// BF16 and F32 carriers are whole bytes, so one logically addressed element
// owns its complete physical slot: the load and the single store below never
// read or rewrite a neighbouring carrier or a padding byte.
[[nodiscard]] inline std::uint64_t silu_load_element(
        const unsigned char* base, std::uint64_t slot,
        std::uint32_t bits) noexcept {
    const std::uint64_t byte = slot * (bits / 8);
    std::uint64_t raw = 0;
    for (std::uint32_t index = 0; index < bits / 8; ++index) {
        raw |= static_cast<std::uint64_t>(base[byte + index])
                << (8 * index);
    }
    return raw;
}

inline void silu_store_element(
        unsigned char* base, std::uint64_t slot, std::uint32_t bits,
        std::uint64_t value) noexcept {
    const std::uint64_t byte = slot * (bits / 8);
    for (std::uint32_t index = 0; index < bits / 8; ++index) {
        base[byte + index] = static_cast<unsigned char>(
                value >> (8 * index));
    }
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

// One logical element: resolve both leading-plane addresses from the copied
// descriptor, decode once, evaluate the stable SiLU once with representable
// special values handled before finite arithmetic, and store exactly one
// target-format round-to-nearest-even encoding.
inline void silu_element(
        const unsigned char* x, unsigned char* y,
        const SiluMetadata& metadata, std::uint64_t plane,
        std::uint64_t row, std::uint64_t column) noexcept {
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
    const std::uint64_t raw = silu_load_element(
            x,
            silu_plane_slot(
                    x_plane, row, column, metadata.rows, metadata.columns),
            metadata.bits);
    const float value = SiluCodec::decode(raw, format);
    const std::uint64_t encoded = SiluCodec::encode(
            detail::scalar_silu_detail::evaluate<SiluFloatTraits>(value),
            format);
    silu_store_element(
            y,
            silu_plane_slot(
                    y_plane, row, column, metadata.rows, metadata.columns),
            metadata.bits, encoded);
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
                            index / metadata->plane_elements;
                    const std::uint64_t within_plane =
                            index - plane * metadata->plane_elements;
                    const std::uint64_t row =
                            within_plane / metadata->columns;
                    const std::uint64_t column =
                            within_plane - row * metadata->columns;
                    silu_element(
                            x, y, *metadata, plane, row, column);
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
    const std::size_t plane_elements = silu_checked_mul(
            dimensions[leading_rank], dimensions[leading_rank + 1],
            "SYCL SiLU plane element count overflows");
    metadata.plane_count = static_cast<std::uint64_t>(plane_count);
    metadata.plane_elements = static_cast<std::uint64_t>(plane_elements);
    metadata.element_count = static_cast<std::uint64_t>(
            silu_checked_mul(
                    plane_count, plane_elements,
                    "SYCL SiLU logical element count overflows"));
    return metadata;
}

}  // namespace

bool silu_device_supported(DataType data_type) noexcept {
    return data_type == DataType::BF16 || data_type == DataType::F32;
}

void SyclQueue::validate_silu_representation(const SiLURequest& request) {
    (void)build_silu_metadata(request);
}

void SyclQueue::execute_silu(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    {
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
        const sycl::event silu_event = launch_silu_kernel(
                queue_, metadata_event, *task.silu_request, metadata_device,
                static_cast<std::size_t>(metadata.element_count));
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
