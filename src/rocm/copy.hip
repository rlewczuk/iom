#include <array>
#include "copy.hpp"

#include "iom/device.hpp"
#include "../iom_internal.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <type_traits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace iom::rocm_detail {
namespace {

std::atomic<SubmissionFault> g_submission_fault{SubmissionFault::none};
std::atomic<int> g_event_record_failures{0};
std::atomic<int> g_stream_sync_failures{0};

}  // namespace

// External linkage: the policy's inline member functions in copy.hpp
// reference these from every including translation unit.
#ifdef IOM_ENABLE_TESTING
std::atomic<std::size_t> event_create_count_for_testing{0};
std::atomic<std::size_t> event_destroy_count_for_testing{0};
std::atomic<std::size_t> stream_create_count_for_testing{0};
std::atomic<std::size_t> stream_destroy_count_for_testing{0};
std::atomic<std::size_t> linear_native_bf16_launch_count_for_testing{0};
std::atomic<std::size_t> linear_scalar_launch_count_for_testing{0};
std::atomic<std::size_t> silu_launch_count_for_testing{0};
#endif  // IOM_ENABLE_TESTING

bool consume_submission_fault(
        SubmissionFault point) noexcept {
    if (point == SubmissionFault::event_record
            || point == SubmissionFault::stream_synchronize) {
        auto& failures = point == SubmissionFault::event_record
                ? g_event_record_failures : g_stream_sync_failures;
        int remaining = failures.load(std::memory_order_acquire);
        while (remaining > 0
                && !failures.compare_exchange_weak(
                        remaining, remaining - 1,
                        std::memory_order_acq_rel)) {
        }
        if (remaining > 0) {
            return true;
        }
    }
    SubmissionFault expected = point;
    return g_submission_fault.compare_exchange_strong(
            expected, SubmissionFault::none, std::memory_order_acq_rel);
}

}  // namespace iom::rocm_detail

#define IOM_GPU_DEVICE __device__
#define IOM_GPU_GLOBAL __global__
#define IOM_GPU_SHARED __shared__
#define IOM_GPU_BARRIER __syncthreads()
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    hipLaunchKernelGGL( \
            kernel, dim3(blocks), dim3(threads), 0, stream, __VA_ARGS__)
#include "../shared/standard_tiled_copy.inl"
#include "../shared/standard_tiled_embedding.inl"
#include "../shared/standard_tiled_add.inl"
#include "../shared/standard_tiled_rmsnorm.inl"
#include "../shared/standard_tiled_linear.inl"
#include "../shared/standard_tiled_rope.inl"

// The shared stable SiLU evaluator is instantiated by this translation unit for
// both execution spaces: its device instantiation is what the SiLU kernel
// calls, and its host entry point must stay valid beside it, so the shared
// qualifier macro is fixed here rather than derived from `IOM_GPU_DEVICE`.
// FMA contraction is off while the evaluator's body is parsed because the
// frozen comparison evaluates the identical expression without contraction: a
// contracted `1 + t*t` moves the stored bits of several `F32` inputs and of
// the narrow leaves, which would break the frozen 1/4/8-ULP ceilings and the
// exact `-104` / `-746` tail checks. The shared Rope traits apply the same
// policy through explicit round-to-nearest primitives. The contraction state
// is restored immediately to this target's own default (`fast`, measured for
// both passes of the checked toolchain), so every other kernel in this file
// keeps the exact arithmetic it was verified with.
#define IOM_SCALAR_SILU_QUALIFIER __host__ __device__ inline
#pragma clang fp contract(off)
#include "../shared/scalar_silu.hpp"
#pragma clang fp contract(fast)

namespace iom::detail {

// Cache append has at most five leading axes before the head axis at rank-3.
// This descriptor carries the independent source/destination plane mappings
// and the complete row-window geometry by value, so the HIP kernel never
// dereferences a borrowed TensorView or reads host metadata.
inline constexpr std::size_t kCacheAppendMaxLeadingRank = 5;

struct CacheAppendMetadata {
    std::uint64_t source_plane_offset = 0;
    std::uint64_t destination_plane_offset = 0;
    std::uint64_t source_head_stride = 0;
    std::uint64_t destination_head_stride = 0;
    std::uint64_t rows = 0;
    std::uint64_t capacity = 0;
    std::uint64_t offset = 0;
    std::uint64_t heads = 0;
    std::uint64_t features = 0;
    std::uint64_t plane_count = 0;
    std::uint32_t bits = 0;
    std::uint32_t leading_rank = 0;
    std::uint64_t source_strides[kCacheAppendMaxLeadingRank]{};
    std::uint64_t destination_strides[kCacheAppendMaxLeadingRank]{};
    std::uint64_t leading_dimensions[kCacheAppendMaxLeadingRank]{};
};

static_assert(std::is_trivially_copyable_v<CacheAppendMetadata>);
static_assert(sizeof(CacheAppendMetadata) <= kMetadataSlotBytes);
static_assert(alignof(CacheAppendMetadata) <= 32);

IOM_GPU_GLOBAL void cache_append_kernel(
        const unsigned char* source, unsigned char* destination,
        CacheAppendMetadata metadata) {
    const std::uint64_t padded_rows =
            (metadata.capacity + kTile - 1) / kTile * kTile;
    const std::uint64_t padded_columns =
            (metadata.features + kTile - 1) / kTile * kTile;
    const std::uint64_t padded_elements = padded_rows * padded_columns;
    const std::uint64_t plane_bits = padded_elements * metadata.bits;
    const std::uint64_t words_per_plane = (plane_bits + 31) / 32;
    const std::uint64_t total_planes = metadata.plane_count * metadata.heads;
    const std::uint64_t total_words = total_planes * words_per_plane;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    const std::uint64_t append_end = metadata.offset + metadata.rows;

    for (std::uint64_t word = IOM_GPU_GLOBAL_INDEX; word < total_words;
         word += stride) {
        const std::uint64_t logical_plane = word / words_per_plane;
        const std::uint64_t word_in_plane = word % words_per_plane;
        const std::uint64_t head = logical_plane % metadata.heads;
        std::uint64_t rest = logical_plane / metadata.heads;
        std::uint64_t source_plane = metadata.source_plane_offset
                + head * metadata.source_head_stride;
        std::uint64_t destination_plane = metadata.destination_plane_offset
                + head * metadata.destination_head_stride;
        for (std::uint32_t axis = metadata.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate =
                    rest % metadata.leading_dimensions[axis];
            rest /= metadata.leading_dimensions[axis];
            source_plane += coordinate * metadata.source_strides[axis];
            destination_plane += coordinate * metadata.destination_strides[axis];
        }

        const std::uint64_t word_first_bit = word_in_plane * 32;
        const std::uint64_t word_end_bit =
                word_first_bit + 32 < plane_bits
                ? word_first_bit + 32
                : plane_bits;
        const std::uint64_t first_slot = word_first_bit / metadata.bits;
        const std::uint64_t last_slot =
                (word_end_bit + metadata.bits - 1) / metadata.bits;
        const std::uint64_t destination_base_word =
                plane_slot(
                        destination_plane, 0, 0, metadata.capacity,
                        metadata.features)
                * metadata.bits / 32;
        auto* destination_word_ptr =
                reinterpret_cast<std::uint32_t*>(destination)
                + destination_base_word + word_in_plane;
        std::uint32_t destination_word = *destination_word_ptr;
        for (std::uint64_t slot = first_slot; slot < last_slot; ++slot) {
            const PhysicalCoordinate coordinate = physical_coordinate(
                    slot, metadata.capacity, metadata.features);
            if (coordinate.row < metadata.offset
                    || coordinate.row >= append_end
                    || coordinate.column >= metadata.features) {
                continue;
            }
            const std::uint64_t source_bit =
                    plane_slot(
                            source_plane,
                            coordinate.row - metadata.offset,
                            coordinate.column, metadata.rows,
                            metadata.features)
                    * metadata.bits;
            const std::uint64_t destination_bit =
                    plane_slot(
                            destination_plane, coordinate.row,
                            coordinate.column, metadata.capacity,
                            metadata.features)
                    * metadata.bits;
            merge_overlapping_field(
                    destination_word, source, source_bit, destination_bit,
                    word_first_bit + destination_base_word * 32,
                    word_end_bit + destination_base_word * 32,
                    static_cast<unsigned int>(metadata.bits));
        }
        store_word(destination_word_ptr, destination_word);
    }
}

template <typename Policy>
void launch_cache_append_kernel(
        typename Policy::stream_type stream,
        const unsigned char* source, unsigned char* destination,
        const CacheAppendMetadata& metadata, std::size_t total_words) {
    const std::size_t launch_words = checked_metadata_add(
            total_words, 255, "cache append launch count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::size_t>(launch_words / kThreads, kMaxBlocks));
    IOM_LAUNCH_KERNEL(
            cache_append_kernel, blocks, kThreads, stream,
            source, destination, metadata);
}

}  // namespace iom::detail


#include "../shared/gpu_queue.hpp"

// ---------------------------------------------------------------------------
// Native BF16 linear projection: direct GFX12 wave32 WMMA.
//
// Fragment slot mapping of `V_WMMA_F32_16X16X16_BF16` at wave32, established
// by execution on the checked gfx1201 device (one-hot A and B slot readout of
// the row and column membership plus a full-product comparison against the
// independent FP32 reference):
//   A slot (lane, element) holds A[m = lane % 16][k = element + 8*(lane/16)]
//   B slot (lane, element) holds B[k = element + 8*(lane/16)][n = lane % 16]
//   D slot (lane, element) holds D[m = element + 8*(lane/16)][n = lane % 16]
// The K labeling is one shared function of the slot identity for both inputs,
// so the reduction runs over the slots the facility pairs and never over a
// replicated copy. The facility is selected only for a device whose runtime
// gate proves it; a target without the instruction compiles the neutral
// fallback below and never reaches it through admission.
// ---------------------------------------------------------------------------
typedef short linear_bf16_short8 __attribute__((ext_vector_type(8)));
typedef float linear_bf16_float8 __attribute__((ext_vector_type(8)));

#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
#define IOM_ROCM_GFX12_BF16_WMMA 1
#else
#define IOM_ROCM_GFX12_BF16_WMMA 0
#endif

namespace iom::detail {
namespace {

constexpr unsigned int kLinearBf16WavesPerBlock = 4;
constexpr unsigned int kLinearBf16Threads = 256;

// One 16x16x16 BF16 WMMA step with FP32 accumulation. Every lane of the wave
// must reach this call in a converged execution mask; the callers below keep
// the enclosing loops wave-uniform for that reason.
IOM_GPU_DEVICE linear_bf16_float8 linear_bf16_wmma_step(
        linear_bf16_short8 multiplicand, linear_bf16_short8 multiplier,
        linear_bf16_float8 accumulator) noexcept {
#if IOM_ROCM_GFX12_BF16_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
            multiplicand, multiplier, accumulator);
#else
    (void)multiplicand;
    (void)multiplier;
    (void)accumulator;
    return linear_bf16_float8{};
#endif
}

// Gathers exactly the selected source rows into the caller's x_pack region in
// native A fragment order: one wave per 16x16 (row, inner) tile and one
// element per lane slot. Physical row and inner tails are written as neutral
// BF16 `+0` and no source padding cell is ever read.
IOM_GPU_GLOBAL void linear_bf16_pack_kernel(
        LinearMetadata m, unsigned short* x_pack) {
    const std::uint64_t lane = threadIdx.x & 31u;
    const std::uint64_t wave =
            static_cast<std::uint64_t>(blockIdx.x)
                    * (blockDim.x / 32u)
            + threadIdx.x / 32u;
    const std::uint64_t wave_count =
            static_cast<std::uint64_t>(gridDim.x) * (blockDim.x / 32u);
    const std::uint64_t row_tiles = (m.rows + kTile - 1) / kTile;
    const std::uint64_t inner_tiles = (m.inner + kTile - 1) / kTile;
    const std::uint64_t tiles_per_plane = row_tiles * inner_tiles;
    const std::uint64_t total = m.plane_count * tiles_per_plane;
    const std::uint64_t row_in_tile = lane % 16u;
    const std::uint64_t inner_in_tile = (lane / 16u) * 8u;
    for (std::uint64_t tile = wave; tile < total; tile += wave_count) {
        const std::uint64_t plane = tile / tiles_per_plane;
        const std::uint64_t in_plane = tile % tiles_per_plane;
        const std::uint64_t row = in_plane / inner_tiles * kTile
                + row_in_tile;
        const std::uint64_t first_inner = in_plane % inner_tiles * kTile
                + inner_in_tile;
        std::uint64_t x_plane = m.x_offset;
        std::uint64_t rest = plane;
        for (std::uint32_t axis = m.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate = rest % m.dims[axis];
            rest /= m.dims[axis];
            x_plane += coordinate * m.x_strides[axis];
        }
        unsigned short* destination =
                x_pack + tile * kTileSlots + lane * 8u;
        for (unsigned int index = 0; index < 8; ++index) {
            const std::uint64_t inner = first_inner + index;
            unsigned short code = 0;
            if (row < m.rows && inner < m.inner) {
                code = *reinterpret_cast<const unsigned short*>(
                        m.x
                        + plane_slot(
                                  x_plane, m.start_row + row, inner,
                                  m.source_rows, m.inner)
                                * 2u);
            }
            destination[index] = code;
        }
    }
}

// One full wave per 16x16 output tile: the K loop accumulates the whole inner
// extent in FP32 registers, starting at `+0`, and each element is rounded
// once with the shared named-format RNE encoding into the caller's y_pack
// region. A BF16 partial product is never stored or reconsumed, and the
// weight is read in place as the conceptual column-major B operand.
IOM_GPU_GLOBAL void linear_bf16_wmma_kernel(
        LinearMetadata m, const unsigned short* x_pack,
        unsigned short* y_pack) {
    const std::uint64_t lane = threadIdx.x & 31u;
    const std::uint64_t wave =
            static_cast<std::uint64_t>(blockIdx.x)
                    * (blockDim.x / 32u)
            + threadIdx.x / 32u;
    const std::uint64_t wave_count =
            static_cast<std::uint64_t>(gridDim.x) * (blockDim.x / 32u);
    const std::uint64_t row_tiles = (m.rows + kTile - 1) / kTile;
    const std::uint64_t outer_tiles = (m.outer + kTile - 1) / kTile;
    const std::uint64_t inner_tiles = (m.inner + kTile - 1) / kTile;
    const std::uint64_t tiles_per_plane = row_tiles * outer_tiles;
    const std::uint64_t total = m.plane_count * tiles_per_plane;
    const std::uint64_t column_in_tile = lane % 16u;
    const std::uint64_t inner_in_tile = (lane / 16u) * 8u;
    const scalar_binary_codec_detail::Format format =
            scalar_binary_codec_detail::Codec<
                    LinearFp32Traits>::format(DataType::BF16);
    for (std::uint64_t tile = wave; tile < total; tile += wave_count) {
        const std::uint64_t plane = tile / tiles_per_plane;
        const std::uint64_t in_plane = tile % tiles_per_plane;
        const std::uint64_t row_tile = in_plane / outer_tiles;
        const std::uint64_t outer_tile = in_plane % outer_tiles;
        const std::uint64_t column = outer_tile * kTile + column_in_tile;
        const bool column_live = column < m.outer;
        linear_bf16_float8 accumulator{};
        for (std::uint64_t k_tile = 0; k_tile < inner_tiles; ++k_tile) {
            const linear_bf16_short8 multiplicand =
                    *reinterpret_cast<const linear_bf16_short8*>(
                            x_pack
                            + ((plane * row_tiles + row_tile) * inner_tiles
                               + k_tile)
                                    * kTileSlots
                            + lane * 8u);
            linear_bf16_short8 multiplier{};
            for (unsigned int index = 0; index < 8; ++index) {
                const std::uint64_t inner = k_tile * kTile + inner_in_tile
                        + index;
                multiplier[index] = 0;
                if (column_live && inner < m.inner) {
                    multiplier[index] =
                            *reinterpret_cast<const unsigned short*>(
                                    m.w
                                    + plane_slot(
                                              m.w_offset, column, inner,
                                              m.outer, m.inner)
                                            * 2u);
                }
            }
            accumulator = linear_bf16_wmma_step(
                    multiplicand, multiplier, accumulator);
        }
        unsigned short* destination =
                y_pack
                + (plane * row_tiles + row_tile) * outer_tiles
                        * kTileSlots
                + outer_tile * kTileSlots;
        for (unsigned int index = 0; index < 8; ++index) {
            // The D slot row is `element + 8*(lane/16)`; y_pack stores the
            // packed product row-major inside each 16x16 tile.
            const std::uint64_t row = (lane / 16u) * 8u + index;
            destination[row * kTile + column_in_tile] =
                    static_cast<unsigned short>(
                            scalar_binary_codec_detail::Codec<
                                    LinearFp32Traits>::encode(
                                    accumulator[index], format));
        }
    }
}

// Copies the packed product into the declared ordinary or head-planar output.
// Only logical cells are written, so 16x16 tile padding keeps its value; the
// head-planar mapping is `o = h*D + d` from the weight orientation and never
// a head broadcast.
IOM_GPU_GLOBAL void linear_bf16_scatter_kernel(
        LinearMetadata m, const unsigned short* y_pack) {
    const std::uint64_t row_tiles = (m.rows + kTile - 1) / kTile;
    const std::uint64_t outer_tiles = (m.outer + kTile - 1) / kTile;
    const std::uint64_t chunks_per_row = (m.columns + kTile - 1) / kTile;
    const std::uint64_t per_plane =
            m.heads * m.rows * chunks_per_row;
    const std::uint64_t total = m.plane_count * per_plane;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    for (std::uint64_t unit = IOM_GPU_GLOBAL_INDEX; unit < total;
         unit += stride) {
        const std::uint64_t plane = unit / per_plane;
        const std::uint64_t in_plane = unit % per_plane;
        const std::uint64_t head = in_plane / (m.rows * chunks_per_row);
        const std::uint64_t in_head = in_plane % (m.rows * chunks_per_row);
        const std::uint64_t row = in_head / chunks_per_row;
        const std::uint64_t chunk = in_head % chunks_per_row;
        std::uint64_t out_plane = m.out_offset;
        std::uint64_t rest = plane;
        for (std::uint32_t axis = m.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate = rest % m.dims[axis];
            rest /= m.dims[axis];
            out_plane += coordinate * m.out_strides[axis];
        }
        if (m.head_planar != 0) {
            out_plane += head * m.out_strides[m.leading_rank];
        }
        const std::uint64_t first = chunk * kTile;
        const std::uint64_t length =
                m.columns - first < kTile ? m.columns - first : kTile;
        const std::uint64_t weight_first = head * m.columns + first;
        // The packed product is row-major inside each 16-wide outer tile, so a
        // logical row's packed columns are contiguous only within one tile. A
        // head stride that is not a multiple of the packed width — any
        // head-planar request with a nonzero head index whose `D` leaves a
        // remainder — crosses that boundary, so the copy walks the packed
        // slot of every element instead of one contiguous run. Ordinary mode
        // stays chunk-aligned and therefore degenerates to a single run.
        const unsigned short* packed_row =
                y_pack
                + (plane * row_tiles + row / kTile) * outer_tiles
                        * kTileSlots
                + (row % kTile) * kTile;
        std::uint64_t index = 0;
        while (index < length) {
            const std::uint64_t packed_column = weight_first + index;
            const std::uint64_t outer_tile = packed_column / kTile;
            const std::uint64_t in_tile = packed_column % kTile;
            const std::uint64_t remaining = length - index;
            const std::uint64_t run =
                    kTile - in_tile < remaining ? kTile - in_tile : remaining;
            const unsigned short* source =
                    packed_row + outer_tile * kTileSlots + in_tile;
            for (std::uint64_t step = 0; step < run; ++step) {
                const unsigned short code = source[step];
                *reinterpret_cast<unsigned short*>(
                        m.out
                        + plane_slot(
                                  out_plane, row, first + index + step, m.rows,
                                  m.columns)
                                * 2u) = code;
            }
            index += run;
        }
    }
}

// Checked `pad16`: round a logical extent up to a multiple of the 16x16 tile.
[[nodiscard]] std::size_t linear_bf16_pad16(
        std::size_t value, const char* message) {
    const std::size_t bumped = linear_checked_add(
            value, static_cast<std::size_t>(kTile) - 1, message);
    return bumped - bumped % static_cast<std::size_t>(kTile);
}

// Checked `A32`: round a byte count up to the 32-byte alignment the path
// reports.
[[nodiscard]] std::size_t linear_bf16_a32(
        std::size_t value, const char* message) {
    const std::size_t bumped =
            linear_checked_add(value, std::size_t{31}, message);
    return bumped - bumped % 32;
}

// Bounded launch extent: one block per `units_per_block` work units, capped
// at the fixed grid limit the shared launchers use.
[[nodiscard]] unsigned int linear_bf16_blocks(
        std::size_t units, unsigned int units_per_block,
        const char* message) {
    const std::size_t rounded = linear_checked_add(
            units, static_cast<std::size_t>(units_per_block) - 1, message);
    const std::size_t blocks = rounded / units_per_block;
    return static_cast<unsigned int>(
            blocks < kLinearMaxBlocks ? blocks : kLinearMaxBlocks);
}

}  // namespace

// Packed-region byte counts of one admitted native-BF16 request, in the
// frozen order `x_pack` then `y_pack`. This is the single computation the
// requirements seam and the launcher both consume, so the reported
// requirement and the dispatched split can never disagree.
struct LinearBf16PackLayout {
    std::size_t plane_count;
    std::size_t row_tiles;
    std::size_t inner_tiles;
    std::size_t outer_tiles;
    std::size_t x_pack_bytes;
    std::size_t y_pack_bytes;
};

[[nodiscard]] LinearBf16PackLayout linear_bf16_pack_layout_for(
        std::size_t plane_count, std::size_t rows, std::size_t inner,
        std::size_t outer) {
    LinearBf16PackLayout layout{
            plane_count,
            linear_bf16_pad16(rows, "LINEAR BF16 row count overflows")
                    / static_cast<std::size_t>(kTile),
            linear_bf16_pad16(inner, "LINEAR BF16 inner extent overflows")
                    / static_cast<std::size_t>(kTile),
            linear_bf16_pad16(outer, "LINEAR BF16 outer extent overflows")
                    / static_cast<std::size_t>(kTile),
            0,
            0};
    const auto pack_bytes = [&layout](std::size_t tiles) {
        return linear_bf16_a32(
                linear_checked_mul(
                        linear_checked_mul(
                                linear_checked_mul(
                                        layout.plane_count, layout.row_tiles,
                                        "LINEAR BF16 pack size overflows"),
                                tiles, "LINEAR BF16 pack size overflows"),
                        kTileSlots * 2, "LINEAR BF16 pack size overflows"),
                "LINEAR BF16 pack size overflows");
    };
    layout.x_pack_bytes = pack_bytes(layout.inner_tiles);
    layout.y_pack_bytes = pack_bytes(layout.outer_tiles);
    return layout;
}

[[nodiscard]] LinearBf16PackLayout linear_bf16_pack_layout(
        const LinearMetadata& m) {
    return linear_bf16_pack_layout_for(
            static_cast<std::size_t>(m.plane_count),
            static_cast<std::size_t>(m.rows),
            static_cast<std::size_t>(m.inner),
            static_cast<std::size_t>(m.outer));
}

// Launch the native BF16 linear specialization on the queue's existing
// nonblocking stream: pack, BF16 WMMA with FP32 accumulation, and the single
// final RNE scatter. No second stream, allocation, staging, host arithmetic,
// or synchronization is introduced here, and the caller workspace range was
// already validated and leased by common admission.
void launch_native_bf16_linear(
        hipStream_t stream, const LinearMetadata& m) {
    if (m.scratch == nullptr) {
        throw std::runtime_error(
                "LINEAR BF16 native path requires the caller workspace");
    }
    const LinearBf16PackLayout layout = linear_bf16_pack_layout(m);
    const std::size_t required = linear_checked_add(
            layout.x_pack_bytes, layout.y_pack_bytes,
            "LINEAR BF16 workspace size overflows");
    if (m.scratch_bytes < required) {
        throw std::runtime_error(
                "LINEAR BF16 native path received a short workspace");
    }
    auto* x_pack = reinterpret_cast<unsigned short*>(m.scratch);
    auto* y_pack = reinterpret_cast<unsigned short*>(
            m.scratch + layout.x_pack_bytes);
    const std::size_t pack_tiles = linear_checked_mul(
            linear_checked_mul(
                    layout.plane_count, layout.row_tiles,
                    "LINEAR BF16 pack tile count overflows"),
            layout.inner_tiles, "LINEAR BF16 pack tile count overflows");
    const std::size_t wmma_tiles = linear_checked_mul(
            linear_checked_mul(
                    layout.plane_count, layout.row_tiles,
                    "LINEAR BF16 WMMA tile count overflows"),
            layout.outer_tiles, "LINEAR BF16 WMMA tile count overflows");
    const std::size_t scatter_units = linear_checked_mul(
            linear_checked_mul(
                    static_cast<std::size_t>(m.heads),
                    static_cast<std::size_t>(m.rows),
                    "LINEAR BF16 scatter unit count overflows"),
            linear_checked_add(
                    static_cast<std::size_t>(m.columns), kTile - 1,
                    "LINEAR BF16 scatter unit count overflows")
                    / kTile,
            "LINEAR BF16 scatter unit count overflows");
    const std::size_t total_scatter = linear_checked_mul(
            layout.plane_count, scatter_units,
            "LINEAR BF16 scatter unit count overflows");
    const unsigned int pack_blocks = linear_bf16_blocks(
            pack_tiles, kLinearBf16WavesPerBlock,
            "LINEAR BF16 pack block count overflows");
    const unsigned int wmma_blocks = linear_bf16_blocks(
            wmma_tiles, kLinearBf16WavesPerBlock,
            "LINEAR BF16 WMMA block count overflows");
    const unsigned int scatter_blocks = linear_bf16_blocks(
            total_scatter, kLinearBf16Threads,
            "LINEAR BF16 scatter block count overflows");
    if (pack_blocks != 0) {
        IOM_LAUNCH_KERNEL(
                linear_bf16_pack_kernel, pack_blocks,
                kLinearBf16WavesPerBlock * 32u, stream, m, x_pack);
    }
    if (wmma_blocks != 0) {
        IOM_LAUNCH_KERNEL(
                linear_bf16_wmma_kernel, wmma_blocks,
                kLinearBf16WavesPerBlock * 32u, stream, m, x_pack, y_pack);
    }
    if (scatter_blocks != 0) {
        IOM_LAUNCH_KERNEL(
                linear_bf16_scatter_kernel, scatter_blocks,
                kLinearBf16Threads, stream, m, y_pack);
    }
}

}  // namespace iom::detail

// The device section of this translation unit ends here: the shared include
// macros are dropped again before the host-only queue implementation and the
// policy wrappers below, exactly as they were before this file grew a native
// device operation.
#undef IOM_GPU_BARRIER
#undef IOM_GPU_SHARED
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE
#undef IOM_ROCM_GFX12_BF16_WMMA

namespace iom::rocm_detail {

namespace {

[[nodiscard]] std::size_t cache_append_checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::size_t cache_append_checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0
            && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::uint64_t cache_append_u64(
        std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::size_t cache_append_padded_extent(
        std::size_t value, const char* message) {
    const std::size_t rounded = cache_append_checked_add(value, 15, message);
    return cache_append_checked_mul(rounded / 16, 16, message);
}

template <typename Request>
[[nodiscard]] detail::CacheAppendMetadata make_cache_append_metadata(
        const Request& request, std::size_t& total_words) {
    const auto& source = request.source;
    const auto& destination = request.destination;
    if (source.rank != destination.rank || source.rank < 3
            || source.rank > 8) {
        throw std::invalid_argument(
                "ROCm cache append metadata rank is invalid");
    }
    const std::size_t leading_rank = source.rank - 3;
    if (leading_rank > detail::kCacheAppendMaxLeadingRank) {
        throw std::overflow_error(
                "ROCm cache append metadata leading rank overflows");
    }
    const std::size_t head_axis = leading_rank;
    const std::size_t row_axis = leading_rank + 1;
    const std::size_t feature_axis = leading_rank + 2;
    if (source.dimensions[head_axis] != destination.dimensions[head_axis]
            || source.dimensions[feature_axis]
                    != destination.dimensions[feature_axis]) {
        throw std::invalid_argument(
                "ROCm cache append metadata shapes differ");
    }
    if (source.plane_strides[head_axis] == 0
            || destination.plane_strides[head_axis] == 0) {
        throw std::invalid_argument(
                "ROCm cache append metadata head stride is zero");
    }

    detail::CacheAppendMetadata metadata;
    metadata.source_plane_offset = cache_append_u64(
            source.plane_offset, "ROCm cache append source offset overflows");
    metadata.destination_plane_offset = cache_append_u64(
            destination.plane_offset,
            "ROCm cache append destination offset overflows");
    metadata.source_head_stride = cache_append_u64(
            source.plane_strides[head_axis],
            "ROCm cache append source head stride overflows");
    metadata.destination_head_stride = cache_append_u64(
            destination.plane_strides[head_axis],
            "ROCm cache append destination head stride overflows");
    metadata.rows = cache_append_u64(
            source.dimensions[row_axis],
            "ROCm cache append source rows overflows");
    metadata.capacity = cache_append_u64(
            destination.dimensions[row_axis],
            "ROCm cache append destination capacity overflows");
    metadata.offset = cache_append_u64(
            request.a, "ROCm cache append offset overflows");
    metadata.heads = cache_append_u64(
            source.dimensions[head_axis],
            "ROCm cache append head count overflows");
    metadata.features = cache_append_u64(
            source.dimensions[feature_axis],
            "ROCm cache append feature count overflows");
    metadata.leading_rank = static_cast<std::uint32_t>(leading_rank);

    std::size_t plane_count = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        plane_count = cache_append_checked_mul(
                plane_count, source.dimensions[axis],
                "ROCm cache append plane count overflows");
        metadata.source_strides[axis] = cache_append_u64(
                source.plane_strides[axis],
                "ROCm cache append source stride overflows");
        metadata.destination_strides[axis] = cache_append_u64(
                destination.plane_strides[axis],
                "ROCm cache append destination stride overflows");
        metadata.leading_dimensions[axis] = cache_append_u64(
                source.dimensions[axis],
                "ROCm cache append leading dimension overflows");
    }
    metadata.plane_count = cache_append_u64(
            plane_count, "ROCm cache append plane count overflows");

    const std::size_t bits = detail::leaf_bits(source.data_type);
    metadata.bits = static_cast<std::uint32_t>(bits);
    const std::size_t padded_rows = cache_append_padded_extent(
            destination.dimensions[row_axis],
            "ROCm cache append padded rows overflow");
    const std::size_t padded_features = cache_append_padded_extent(
            source.dimensions[feature_axis],
            "ROCm cache append padded features overflow");
    const std::size_t padded_elements = cache_append_checked_mul(
            padded_rows, padded_features,
            "ROCm cache append padded elements overflow");
    const std::size_t plane_bits = cache_append_checked_mul(
            padded_elements, bits,
            "ROCm cache append plane bits overflow");
    const std::size_t words_per_plane = cache_append_checked_add(
            plane_bits, 31, "ROCm cache append word count overflow") / 32;
    total_words = cache_append_checked_mul(
            cache_append_checked_mul(
                    plane_count, source.dimensions[head_axis],
                    "ROCm cache append total plane count overflow"),
            words_per_plane, "ROCm cache append total word count overflow");
    return metadata;
}

}  // namespace

// Launch a direct raw-word cache append on the queue's existing nonblocking
// stream.  All descriptor arithmetic is checked from the immutable request
// snapshot before the HIP launch; the kernel performs no conversion or
// allocation and writes each destination word exactly once.
template <typename Request>
void gpu_policy::launch_cache_append(
        hipStream_t stream, const Request& request) {
    // This accepted-failure seam is consumed before the direct launch so a
    // retained launch error cannot leave a partially updated destination.
    if (consume_submission_fault(SubmissionFault::third_plane_launch)) {
        check_hip(copy_kernel_operation(), hipErrorInvalidValue);
    }
    std::size_t total_words = 0;
    const detail::CacheAppendMetadata metadata =
            make_cache_append_metadata(request, total_words);
    detail::launch_cache_append_kernel<gpu_policy>(
            stream,
            static_cast<const unsigned char*>(
                    request.source.native_handle),
            static_cast<unsigned char*>(
                    request.destination.native_handle),
            metadata, total_words);
    check_kernel(copy_kernel_operation());
}

// Launch the completed shared standard-tiled logical-F RMSNorm operation on
// the queue's existing nonblocking stream. The wrapper performs no host
// arithmetic, staging, allocation, or synchronization.
void gpu_policy::launch_rmsnorm(
        hipStream_t stream, const detail::RmsnormMetadata& metadata) {
    detail::launch_standard_tiled_rmsnorm<gpu_policy>(stream, metadata);
}

// The SiLU activation kernel, its policy-parameterized launch wrapper, and
// this policy binding live in the SiLU section at the end of this file: the
// shared stable evaluator must be parsed inside a non-contracting region, and
// that region is kept terminal so no earlier kernel is affected by it.

// Launch the shared standard-tiled logical-F Rope operation on the queue's
// existing nonblocking stream. The wrapper adds no host arithmetic, staging,
// allocation, or synchronization.
void gpu_policy::launch_rope(
        hipStream_t stream, const detail::RopeMetadata& metadata) {
    detail::launch_standard_tiled_rope<gpu_policy>(stream, metadata);
}

// Launch the completed shared standard-tiled scalar linear projection, or the
// native BF16 specialization, on the queue's existing nonblocking stream. The
// wrapper performs no host arithmetic, staging, allocation, synchronization,
// or fallback: both kernels read and write device storage directly, and the
// BF16 path additionally uses the caller workspace range the common admission
// already validated and leased.
void gpu_policy::launch_linear(
        hipStream_t stream, const detail::LinearMetadata& metadata) {
    if (metadata.type == static_cast<std::uint32_t>(DataType::BF16)) {
#ifdef IOM_ENABLE_TESTING
        linear_native_bf16_launch_count_for_testing.fetch_add(
                1, std::memory_order_relaxed);
#endif
        detail::launch_native_bf16_linear(stream, metadata);
        return;
    }
#ifdef IOM_ENABLE_TESTING
    linear_scalar_launch_count_for_testing.fetch_add(
            1, std::memory_order_relaxed);
#endif
    detail::launch_standard_tiled_linear<gpu_policy>(stream, metadata);
}

// Native BF16 facility of one exact device ordinal. The checked route is the
// GFX12 wave32 BF16-to-FP32 WMMA instruction, so the gate accepts exactly the
// GFX12 ASIC ids whose ISA carries that 128-bit wave32 operand form (the same
// two ids rocWMMA gates the instruction to) on a wave32 device; every other
// ordinal, including the installed gfx1036, has no proven facility and is a
// capability rejection rather than a queued failure. This reads immutable
// device properties: it allocates, registers, leases, submits, and
// synchronizes nothing.
bool gpu_policy::linear_native_bf16_available(
        gpu_policy::context_type device_ordinal) noexcept {
    if (device_ordinal < 0) {
        return false;
    }
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device_ordinal) != hipSuccess) {
        return false;
    }
    if (properties.warpSize != 32) {
        return false;
    }
    const std::string_view architecture{properties.gcnArchName};
    return architecture.starts_with("gfx1200")
            || architecture.starts_with("gfx1201");
}

// Exact `{bytes, alignment}` of one admitted native-BF16 linear request: the
// frozen `A32(P*pad16(R)*pad16(I)*2) + A32(P*pad16(R)*pad16(O)*2)` sum with
// every product, `pad16`, byte conversion, alignment round-up, and addition
// checked, and alignment 32. `P` is the checked product of the logical
// leading extents of the input, `R` the selected row count, `I` the inner
// extent, and `O` the rank-two weight's output extent.
WorkspaceRequirements gpu_policy::linear_native_bf16_requirements(
        const TensorView& x, const TensorView& w, std::size_t rows) {
    const std::span<const std::size_t> dimensions =
            x.spec().shape.dimensions();
    const std::span<const std::size_t> weight_dimensions =
            w.spec().shape.dimensions();
    const std::size_t rank = dimensions.size();
    if (rank < 2 || weight_dimensions.size() != 2) {
        throw std::invalid_argument(
                "LINEAR BF16 requirement needs the admitted operand ranks");
    }
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis + 2 < rank; ++axis) {
        planes = detail::linear_checked_mul(
                planes, dimensions[axis],
                "LINEAR BF16 leading plane count overflows");
    }
    const detail::LinearBf16PackLayout layout =
            detail::linear_bf16_pack_layout_for(
                    planes, rows, dimensions[rank - 1],
                    weight_dimensions[0]);
    return WorkspaceRequirements{
            detail::linear_checked_add(
                    layout.x_pack_bytes, layout.y_pack_bytes,
                    "LINEAR BF16 workspace size overflows"),
            32};
}

void inject_submission_fault_for_testing(
        SubmissionFault fault) noexcept {
    g_event_record_failures.store(
            fault == SubmissionFault::event_record ? 2 : 0,
            std::memory_order_release);
    g_stream_sync_failures.store(
            fault == SubmissionFault::event_record ? 1 : 0,
            std::memory_order_release);
    g_submission_fault.store(fault, std::memory_order_release);
}

namespace {

detail::Fence transfer_fence() noexcept {
    detail::Fence fence;
    fence.invoke = [](const detail::Fence&) noexcept {
        return detail::FenceResult::success();
    };
    return fence;
}

struct WorkspaceAdmission {
    detail::WorkspaceLease lease;
    void* address = nullptr;
};

WorkspaceAdmission begin_workspace_transfer(
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& view, RawWorkspaceView workspace,
        bool& resource_poisoned) {
    const WorkspaceRequirements requirements =
            view.copy_from_host_workspace_requirements();
    const std::array<TensorView, 1> operands{view};
    const RawWorkspaceView checked =
            detail::WorkspaceValidation::validated(
                    device, workspace, requirements.bytes,
                    requirements.alignment, operands);
    if (resource_poisoned) {
        throw std::bad_alloc();
    }
    const detail::QueueId queue_id =
            detail::allocate_queue_id(registry_state);
    return {
            detail::acquire_workspace_lease(
                    registry_state, workspace.owner_identity(),
                    detail::WorkspaceValidation::address(checked),
                    checked.byte_size(), queue_id, queue_id, transfer_fence()),
            detail::WorkspaceValidation::address(checked)};
}

void finish_workspace_transfer(
        detail::RegistryState& registry_state,
        const detail::WorkspaceLease& lease, bool proof) noexcept {
    detail::complete_workspace_lease(registry_state, lease, proof);
}

}  // namespace

void region_from_host(
        hipStream_t transfer_stream, int device_ordinal,
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& destination, RawWorkspaceView workspace,
        std::span<const std::byte> source, bool& resource_poisoned) {
    WorkspaceAdmission admission = begin_workspace_transfer(
            device, registry_state, destination, workspace, resource_poisoned);
    try {
        detail::synchronous_transfer<gpu_policy>(
                transfer_stream, device_ordinal, destination, admission.address,
                source, {}, true);
        finish_workspace_transfer(registry_state, admission.lease, true);
    } catch (...) {
        resource_poisoned = true;
        finish_workspace_transfer(registry_state, admission.lease, false);
        throw;
    }
}

void region_to_host(
        hipStream_t transfer_stream, int device_ordinal,
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& source, RawWorkspaceView workspace,
        std::span<std::byte> destination, bool& resource_poisoned) {
    WorkspaceAdmission admission = begin_workspace_transfer(
            device, registry_state, source, workspace, resource_poisoned);
    try {
        detail::synchronous_transfer<gpu_policy>(
                transfer_stream, device_ordinal, source, admission.address,
                {}, destination, false);
        finish_workspace_transfer(registry_state, admission.lease, true);
    } catch (...) {
        resource_poisoned = true;
        finish_workspace_transfer(registry_state, admission.lease, false);
        throw;
    }
}

std::unique_ptr<DeviceOps> make_queue(
        const Device& device,
        detail::QueueResourceProvider& resource_provider, int device_ordinal,
        detail::RegistryState& registry_state) {
    return std::make_unique<detail::GpuQueue<gpu_policy>>(
            device, resource_provider, device_ordinal, registry_state);
}

#ifdef IOM_ENABLE_TESTING
void queue_resource_snapshot_for_testing(
        DeviceOps& queue, QueueResourceSnapshot& snapshot) {
    auto* gpu_queue = dynamic_cast<detail::GpuQueue<gpu_policy>*>(&queue);
    if (gpu_queue == nullptr) {
        throw std::logic_error("queue is not a live ROCm GpuQueue");
    }
    detail::MetadataSlotPool& pool = gpu_queue->metadata_pool_for_testing();
    EventRingState& ring = gpu_queue->event_ring_for_testing();
    snapshot.slot_count = pool.slot_count();
    snapshot.device_base = pool.device_base();
    snapshot.slot_stride = pool.slot_stride();
    snapshot.events_total = ring.event_count_for_testing();
    snapshot.events_in_use = ring.in_use_count_for_testing();
    snapshot.slots_in_use = pool.in_use_count();
    snapshot.slots_protected = pool.protected_count();
}

void reclaim_retained_queue_leases_for_testing(Device& device) {
    auto* provider =
            dynamic_cast<detail::QueueResourceProvider*>(&device);
    if (provider == nullptr) {
        throw std::logic_error("device does not own fixed queue resources");
    }
    provider->reclaim_retained_leases();
}
#endif  // IOM_ENABLE_TESTING

}  // namespace iom::rocm_detail

namespace iom::detail {

// Compile-time coverage of the otherwise deferred launch boundary: the shared
// launcher and its device operation are instantiated for this backend policy
// in every build of this translation unit, so both toolchains compile the
// complete shared implementation before a wrapper calls it. An explicit
// instantiation is only well-formed in an enclosing namespace of the
// template's own namespace, so this one lives here rather than in
// `iom::rocm_detail`.
template void launch_standard_tiled_rmsnorm<iom::rocm_detail::gpu_policy>(
        iom::rocm_detail::gpu_policy::stream_type stream,
        const RmsnormMetadata& metadata);

template void launch_standard_tiled_linear<iom::rocm_detail::gpu_policy>(
        iom::rocm_detail::gpu_policy::stream_type stream,
        const LinearMetadata& metadata);
template void launch_standard_tiled_rope<iom::rocm_detail::gpu_policy>(
        iom::rocm_detail::gpu_policy::stream_type stream,
        const RopeMetadata& metadata);

}  // namespace iom::detail
// ---------------------------------------------------------------------------
// SiLU activation over the standard 16x16 tiled encoding.
//
// The device operation is the one frozen scalar contract `x / (1 + exp(-x))`
// from `src/shared/scalar_silu.hpp` applied to every logical element of every
// leading plane, run, and feature of the admitted pair, through the shared
// named-format decode and exactly one round-to-nearest-even output encoding.
// `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
// and `F32` evaluate in the binary32 domain and `F64` keeps its own binary64
// domain, exactly as the normative SiLU contract fixes the intermediates. The
// exponential is evaluated in the binary64 domain and rounded once to the
// carrier: the contract's independent comparison uses a correctly rounded
// exponential, and the device library's `expf` is one ULP away from it at
// some arguments (measured on the configured ROCm device), which would exceed
// the one-ULP ceiling of the narrow leaves for those inputs. The `F64`
// carrier's exponential is the device binary64 exponential itself, which
// reproduces the reference over the whole contract range, including the
// `-746` subnormal tail.
//
// The kernel reads the input storage and writes the output storage directly on
// the queue's existing nonblocking stream: no host round trip, staging,
// allocation, second stream, fallback, or synchronization. A 16-feature tile-row
// packet owns its complete 32-bit word span exclusively, so each logical field
// is decoded, evaluated, RNE-encoded, and merged exactly once before each
// touched word is stored once. Padding rows, padding columns, and padding bits
// are never decoded or modified, and a packet with no logical field is not
// written.
//
// This section is placed after every other kernel in the translation unit and
// re-establishes the device qualifiers for its own region exactly as the
// earlier sections do. The shared evaluator it calls was parsed above with
// FMA contraction off (see the qualifier block near the top of the file).
// ---------------------------------------------------------------------------

#define IOM_GPU_DEVICE __device__
#define IOM_GPU_GLOBAL __global__
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_GPU_GLOBAL_STRIDE \
    (static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    hipLaunchKernelGGL( \
            kernel, dim3(blocks), dim3(threads), 0, stream, __VA_ARGS__)

namespace iom::detail {

// Both carriers drive the one shared named-format codec and the one shared
// stable SiLU evaluator in their own IEEE domain. The bit-pattern
// constructors stay in the double domain the established shared device codec
// already uses, and the finite maximum is the exact carrier maximum.
struct SiluFp32Traits {
    using carrier_type = float;

    IOM_GPU_DEVICE static carrier_type positive_infinity() noexcept {
        return static_cast<carrier_type>(__longlong_as_double(
                static_cast<long long>(0x7ff0000000000000ull)));
    }
    IOM_GPU_DEVICE static carrier_type quiet_nan() noexcept {
        return static_cast<carrier_type>(__longlong_as_double(
                static_cast<long long>(0x7ff8000000000000ull)));
    }
    IOM_GPU_DEVICE static carrier_type max_finite() noexcept {
        return 3.4028234663852886e+38F;
    }
    IOM_GPU_DEVICE static carrier_type exp(carrier_type value) noexcept {
        return static_cast<carrier_type>(::exp(static_cast<double>(value)));
    }
    IOM_GPU_DEVICE static bool isnan(carrier_type value) noexcept {
        return ::isnan(value);
    }
    IOM_GPU_DEVICE static bool isinf(carrier_type value) noexcept {
        return ::isinf(value);
    }
    IOM_GPU_DEVICE static bool signbit(carrier_type value) noexcept {
        return ::signbit(value);
    }
    IOM_GPU_DEVICE static carrier_type fabs(carrier_type value) noexcept {
        return ::fabs(value);
    }
    IOM_GPU_DEVICE static carrier_type floor(carrier_type value) noexcept {
        return ::floor(value);
    }
    IOM_GPU_DEVICE static carrier_type ldexp(
            carrier_type value, int exponent) noexcept {
        return ::ldexp(value, exponent);
    }
    IOM_GPU_DEVICE static carrier_type frexp(
            carrier_type value, int* exponent) noexcept {
        return ::frexp(value, exponent);
    }
};

struct SiluFp64Traits {
    using carrier_type = double;

    IOM_GPU_DEVICE static carrier_type positive_infinity() noexcept {
        return __longlong_as_double(
                static_cast<long long>(0x7ff0000000000000ull));
    }
    IOM_GPU_DEVICE static carrier_type quiet_nan() noexcept {
        return __longlong_as_double(
                static_cast<long long>(0x7ff8000000000000ull));
    }
    IOM_GPU_DEVICE static carrier_type max_finite() noexcept {
        return __longlong_as_double(
                static_cast<long long>(0x7fefffffffffffffull));
    }
    IOM_GPU_DEVICE static carrier_type exp(carrier_type value) noexcept {
        return ::exp(value);
    }
    IOM_GPU_DEVICE static bool isnan(carrier_type value) noexcept {
        return ::isnan(value);
    }
    IOM_GPU_DEVICE static bool isinf(carrier_type value) noexcept {
        return ::isinf(value);
    }
    IOM_GPU_DEVICE static bool signbit(carrier_type value) noexcept {
        return ::signbit(value);
    }
    IOM_GPU_DEVICE static carrier_type fabs(carrier_type value) noexcept {
        return ::fabs(value);
    }
    IOM_GPU_DEVICE static carrier_type floor(carrier_type value) noexcept {
        return ::floor(value);
    }
    IOM_GPU_DEVICE static carrier_type ldexp(
            carrier_type value, int exponent) noexcept {
        return ::ldexp(value, exponent);
    }
    IOM_GPU_DEVICE static carrier_type frexp(
            carrier_type value, int* exponent) noexcept {
        return ::frexp(value, exponent);
    }
};

namespace {

constexpr std::uint64_t kSiluTile = TensorSpec::TILE;
constexpr unsigned int kSiluThreads = 256;
constexpr unsigned int kSiluMaxBlocks = 65535;

[[nodiscard]] std::size_t silu_checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0
            && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t silu_checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

// Packed-field primitives, byte addressed so a view whose plane mapping does
// not start on a word boundary is still addressed correctly. Element slot `s`
// of a plane occupies bits `[s * bits, (s + 1) * bits)` of that plane's packed
// stream, and every plane's first slot starts on a 32-bit boundary for every
// applicable leaf width.
IOM_GPU_DEVICE std::uint32_t silu_load_word(
        const unsigned char* base, std::uint64_t word) noexcept {
    std::uint32_t value;
    for (unsigned i = 0; i < 4; ++i) {
        reinterpret_cast<unsigned char*>(&value)[i] = base[word * 4 + i];
    }
    return value;
}

IOM_GPU_DEVICE void silu_store_word(
        unsigned char* base, std::uint64_t word,
        std::uint32_t value) noexcept {
    for (unsigned i = 0; i < 4; ++i) {
        base[word * 4 + i] =
                reinterpret_cast<const unsigned char*>(&value)[i];
    }
}

IOM_GPU_DEVICE std::uint32_t silu_field_mask(unsigned bits) noexcept {
    return bits == 32 ? 0xffffffffu : ((std::uint32_t{1} << bits) - 1);
}

IOM_GPU_DEVICE std::uint64_t silu_load_bits(
        const unsigned char* base, std::uint64_t bit,
        unsigned bits) noexcept {
    const std::uint64_t word = bit / 32;
    std::uint64_t joined = silu_load_word(base, word);
    if (bit % 32 + bits > 32) {
        joined |= static_cast<std::uint64_t>(
                          silu_load_word(base, word + 1))
                << 32;
    }
    return (joined >> (bit % 32))
            & (bits == 64
               ? ~std::uint64_t{}
               : ((std::uint64_t{1} << bits) - 1));
}

// One thread owns one 16-feature packet of one logical row. The packet starts
// on a word boundary for every applicable leaf and its complete word range is
// exclusive, so fields that straddle words are still evaluated only once.
template <typename Traits>
IOM_GPU_DEVICE void silu_tiled_packets(const SiluMetadata& m) noexcept {
    const std::uint64_t bits = m.bits;
    const std::uint64_t chunks_per_row =
            (m.columns + kSiluTile - 1) / kSiluTile;
    const std::uint64_t tasks =
            m.plane_count * m.rows * chunks_per_row;
    const unsigned int packet_words =
            static_cast<unsigned int>(bits * kSiluTile / 32);
    const auto* x_bytes = m.x;
    auto* y_bytes = m.y;

    for (std::uint64_t task = IOM_GPU_GLOBAL_INDEX; task < tasks;
         task += IOM_GPU_GLOBAL_STRIDE) {
        const std::uint64_t chunk = task % chunks_per_row;
        const std::uint64_t row_index = task / chunks_per_row;
        const std::uint64_t row = row_index % m.rows;
        std::uint64_t x_plane = m.x_offset;
        std::uint64_t y_plane = m.y_offset;
        std::uint64_t rest = row_index / m.rows;
        for (std::uint32_t axis = m.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate = rest % m.dims[axis];
            rest /= m.dims[axis];
            x_plane += coordinate * m.x_strides[axis];
            y_plane += coordinate * m.y_strides[axis];
        }

        const std::uint64_t first_feature = chunk * kSiluTile;
        const std::uint64_t length =
                m.columns - first_feature < kSiluTile
                ? m.columns - first_feature
                : kSiluTile;
        const std::uint64_t x_bit = plane_slot(
                x_plane, row, first_feature, m.rows, m.columns) * bits;
        const std::uint64_t y_bit = plane_slot(
                y_plane, row, first_feature, m.rows, m.columns) * bits;
        const std::uint64_t first_word = y_bit / 32;
        std::uint32_t merged[32]{};
        bool owns_word[32]{};
        for (unsigned int local_word = 0;
             local_word < packet_words; ++local_word) {
            merged[local_word] = silu_load_word(
                    y_bytes, first_word + local_word);
        }
        for (std::uint64_t index = 0; index < length; ++index) {
            const std::uint64_t field_bit = y_bit + index * bits;
            const std::uint64_t encoded =
                    scalar_silu_detail::scalar_silu<Traits>(
                            static_cast<DataType>(m.type),
                            silu_load_bits(
                                    x_bytes, x_bit + index * bits,
                                    static_cast<unsigned>(bits)));
            const std::uint64_t field_end = field_bit + bits;
            const std::uint64_t first_output_word = field_bit / 32;
            const std::uint64_t last_output_word = (field_end - 1) / 32;
            for (std::uint64_t output_word = first_output_word;
                 output_word <= last_output_word; ++output_word) {
                if (output_word < first_word
                        || output_word >= first_word + packet_words) {
                    continue;
                }
                const std::uint64_t word_first_bit = output_word * 32;
                const std::uint64_t overlap_first =
                        field_bit > word_first_bit
                        ? field_bit
                        : word_first_bit;
                const std::uint64_t overlap_end =
                        field_end < word_first_bit + 32
                        ? field_end
                        : word_first_bit + 32;
                const unsigned int count = static_cast<unsigned int>(
                        overlap_end - overlap_first);
                const unsigned int position = static_cast<unsigned int>(
                        overlap_first - word_first_bit);
                const unsigned int shift = static_cast<unsigned int>(
                        overlap_first - field_bit);
                const std::uint32_t segment =
                        static_cast<std::uint32_t>(encoded >> shift)
                        & silu_field_mask(count);
                const unsigned int local_word = static_cast<unsigned int>(
                        output_word - first_word);
                merged[local_word] = (merged[local_word]
                                      & ~(silu_field_mask(count) << position))
                        | (segment << position);
                owns_word[local_word] = true;
            }
        }
        for (unsigned int local_word = 0;
             local_word < packet_words; ++local_word) {
            if (owns_word[local_word]) {
                silu_store_word(
                        y_bytes, first_word + local_word,
                        merged[local_word]);
            }
        }
    }
}

// The shared kernel is a plain function so this translation unit compiles the
// complete device operation for both carriers before the wrapper calls it.
IOM_GPU_GLOBAL void silu_kernel(SiluMetadata m) {
    if (m.type == static_cast<std::uint32_t>(DataType::F64)) {
        silu_tiled_packets<SiluFp64Traits>(m);
    } else {
        silu_tiled_packets<SiluFp32Traits>(m);
    }
}

}  // namespace

// Policy-parameterized launch of the SiLU device operation. The backend
// wrapper supplies the already-created nonblocking stream. Every product,
// packet count, and launch count is checked from the immutable descriptor
// before the HIP launch, exactly as the other ROCm launchers do; the launch
// introduces no allocation, staging, second stream, or synchronization.
template <typename Policy>
void launch_silu_kernel(
        typename Policy::stream_type stream,
        const SiluMetadata& metadata) {
    const std::size_t chunks_per_row = silu_checked_add(
            static_cast<std::size_t>(metadata.columns), kSiluTile - 1,
            "SiLU launch chunk count overflows") / kSiluTile;
    const std::size_t plane_rows = silu_checked_mul(
            static_cast<std::size_t>(metadata.plane_count),
            static_cast<std::size_t>(metadata.rows),
            "SiLU launch row count overflows");
    const std::size_t total_tasks = silu_checked_mul(
            plane_rows, chunks_per_row,
            "SiLU launch task count overflows");
    const std::size_t launch_tasks = silu_checked_add(
            total_tasks, kSiluThreads - 1,
            "SiLU launch block count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::size_t>(
                    launch_tasks / kSiluThreads, kSiluMaxBlocks));
    IOM_LAUNCH_KERNEL(
            silu_kernel, blocks, kSiluThreads, stream, metadata);
}

}  // namespace iom::detail

namespace iom::rocm_detail {

// Launch the completed shared SiLU activation on the queue's existing
// nonblocking stream. The wrapper performs no host arithmetic beyond the
// checked launch geometry and no staging, allocation, or synchronization.
void gpu_policy::launch_silu(
        hipStream_t stream, const detail::SiluMetadata& metadata) {
    // This accepted-failure seam is consumed before the native launch so an
    // armed failure never starts a partially activated plane.
    if (consume_submission_fault(SubmissionFault::silu_launch)) {
        check_hip(silu_kernel_operation(), hipErrorInvalidValue);
    }
    detail::launch_silu_kernel<gpu_policy>(stream, metadata);
#ifdef IOM_ENABLE_TESTING
    silu_launch_count_for_testing.fetch_add(1, std::memory_order_relaxed);
#endif
}

}  // namespace iom::rocm_detail

#undef IOM_LAUNCH_KERNEL
#undef IOM_GPU_GLOBAL_STRIDE
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE
