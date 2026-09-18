// CUDA native BF16 linear projection: the direct `<mma.h>` BF16-input /
// FP32-accumulate WMMA specialization of the shared linear descriptor.
//
// This translation unit is the only place where the vendor matrix intrinsics
// exist. It consumes the immutable `LinearMetadata` the queue already uploaded
// (src/shared/standard_tiled_linear.inl) and writes the caller's output
// directly in the shared standard 16x16 tiling: no stream, allocation, hidden
// scratch, staging buffer, relocation, persistent weight transpose, host round
// trip, cuBLAS call, or elementwise substitute is created here.
//
// The device operation is the frozen `Linear projections` contract of
// docs/BACKEND_CONTRACT.md. One warp owns one `(plane, head, row tile, column
// tile)` unit of the selected window; the logical rows are exactly
// `x[b,s+r,*]` for `r` in `0..R-1` and the logical output columns are
// `o = h*D+d`, so `w[o,*]` is the Hugging Face weight row and no plane,
// head, or row crosses another. A is the selected input tile in row-major
// order, B is the weight tile read without transposing checkpoint storage and
// interpreted as column-major (`B[i,o] = w[o,i]`), and the accumulator starts
// at FP32 `+0` and keeps FP32 precision across every K tile. Cells outside the
// logical inner, row, and output-column extents are neutral zero fragments, so
// no operand padding, tile tail, or uninitialized cell is ever read as a
// multiplicand, and the FP32 result is converted exactly once with the shared
// named-format round-to-nearest-ties-to-even encode.
//
// The bounded `alignas(32)` shared staging below is the contract's own
// kernel-local tile: it is neither an allocation nor caller workspace and it
// is discarded when the unit completes. Only logical BF16 cells are written,
// and every 32-bit destination word has exactly one owning lane, so packed
// tile padding, unrelated planes, and unselected rows keep their value.

#include "copy.hpp"

#include <cuda_runtime_api.h>

#include <cuda_bf16.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace iom::cuda_detail {
namespace {

// The architecture the loaded image of this translation unit was compiled for.
// `__CUDA_ARCH__` is undefined in the host pass, so the initializer is decided
// by the device pass that produced the loaded image: a `compute_75` JIT image
// carries 750, and only an image of an architecture at or above the BF16 WMMA
// floor carries the statements of the specialization below. The host reads it
// through the runtime symbol API, which is why it is deliberately not `const`.
__device__ unsigned int bf16_wmma_image_arch =
#if defined(__CUDA_ARCH__)
        __CUDA_ARCH__;
#else
        0;
#endif

}  // namespace

unsigned int linear_bf16_wmma_image_arch() noexcept {
    unsigned int arch = 0;
    if (cudaMemcpyFromSymbol(&arch, bf16_wmma_image_arch, sizeof(arch))
            != cudaSuccess) {
        // A device without a loadable image of this specialization has no
        // facility; the retained runtime error is cleared so the capability
        // query stays a side-effect-free fact.
        (void)cudaGetLastError();
        return 0;
    }
    return arch;
}

}  // namespace iom::cuda_detail

#define IOM_GPU_DEVICE __device__
#define IOM_GPU_GLOBAL __global__
#define IOM_GPU_SHARED __shared__
#define IOM_GPU_BARRIER __syncthreads()
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(__VA_ARGS__)
#include "../shared/standard_tiled_copy.inl"
#include "../shared/standard_tiled_linear.inl"

#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_BARRIER
#undef IOM_GPU_SHARED
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

namespace iom::cuda_detail {
namespace {

// The private launch policy of the native BF16 specialization: one warp owns
// one 16x16 logical output tile of one `(plane, head)` pair, one block hosts
// eight warps, and the unit index is a grid-stride so any launch width stays
// correct. The tile is the shared 16x16 tiling itself, which is why the
// fragments use `16x16x16` and a leading dimension of 16 throughout.
struct Bf16WmmaLaunchPolicy {
    static constexpr unsigned int tile = 16;
    static constexpr unsigned int lanes = 32;
    static constexpr unsigned int warps_per_block = 8;
    static constexpr unsigned int threads = lanes * warps_per_block;
    static constexpr std::size_t max_blocks = 65535;
    static constexpr std::size_t cells = tile * tile;
};

// Kernel-local staging of one warp's current tile: the row-major A tile of the
// selected input rows, the row-major-by-output-column B tile of the weight
// rows, and the FP32 accumulator store. `alignas(32)` satisfies the 256-bit
// base alignment `load_matrix_sync` and `store_matrix_sync` require; the fixed
// launch-time size is a kernel-local tile, never an allocation or caller
// workspace.
struct alignas(32) Bf16WmmaStaging {
    __nv_bfloat16 matrix_a[Bf16WmmaLaunchPolicy::warps_per_block]
                           [Bf16WmmaLaunchPolicy::cells];
    __nv_bfloat16 matrix_b[Bf16WmmaLaunchPolicy::warps_per_block]
                           [Bf16WmmaLaunchPolicy::cells];
    float accumulator[Bf16WmmaLaunchPolicy::warps_per_block]
                      [Bf16WmmaLaunchPolicy::cells];
};
static_assert(
        sizeof(Bf16WmmaStaging)
                == Bf16WmmaLaunchPolicy::warps_per_block
                        * (2 * Bf16WmmaLaunchPolicy::cells
                                   * sizeof(__nv_bfloat16)
                           + Bf16WmmaLaunchPolicy::cells * sizeof(float)),
        "BF16 WMMA staging geometry drifted");
static_assert(
        alignof(Bf16WmmaStaging) >= 32,
        "BF16 WMMA staging must satisfy the fragment base alignment");

// One logical BF16 operand cell of one plane through the shared opaque cell
// mapping and the shared named-format reader: the 16-bit field at the cell's
// own standard-tiled slot, exactly as the scalar recurrence reads it.
__device__ __nv_bfloat16 bf16_wmma_load(
        const unsigned char* base, std::uint64_t slot) {
    return __ushort_as_bfloat16(static_cast<unsigned short>(
            iom::detail::linear_load_bits(
                    base, slot * 16u, 16u)));
}

// The one final RNE encode of an accumulated FP32 result through the existing
// named-format codec: infinity, NaN, saturation, subnormal, and signed-zero
// rules are the shared ones, no intermediate result is ever re-encoded, and no
// NaN payload is promised.
__device__ unsigned int bf16_wmma_encode(float value) {
    using Codec =
            iom::detail::scalar_binary_codec_detail::Codec<
                    iom::detail::LinearFp32Traits>;
    return static_cast<unsigned int>(Codec::encode(
            value, Codec::format(DataType::BF16)));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

__device__ void bf16_wmma_linear_body(
        const iom::detail::LinearMetadata& m) {
    namespace wmma = nvcuda::wmma;
    constexpr std::uint64_t kTile = Bf16WmmaLaunchPolicy::tile;
    constexpr std::uint64_t kCells = Bf16WmmaLaunchPolicy::cells;
    constexpr std::uint64_t kLanes = Bf16WmmaLaunchPolicy::lanes;
    const __nv_bfloat16 kNeutral = __ushort_as_bfloat16(0u);

    const unsigned int warp =
            threadIdx.x / Bf16WmmaLaunchPolicy::lanes;
    const unsigned int lane =
            threadIdx.x % Bf16WmmaLaunchPolicy::lanes;
    __shared__ Bf16WmmaStaging staging;

    // The launch enumerates one logical chunk per output tile, exactly as the
    // shared descriptor validated: `plane_count` independent planes, `heads`
    // independent head planes, `ceil(R/16)` row tiles of the selected run, and
    // `ceil(columns/16)` column tiles of the whole output column extent.
    const std::uint64_t chunks_per_row = (m.columns + kTile - 1) / kTile;
    const std::uint64_t row_tiles = (m.rows + kTile - 1) / kTile;
    const std::uint64_t units_per_head = row_tiles * chunks_per_row;
    const std::uint64_t units_per_plane = m.heads * units_per_head;
    const std::uint64_t total = m.plane_count * units_per_plane;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(
                    blockDim.x / Bf16WmmaLaunchPolicy::lanes)
            * gridDim.x;
    const std::uint64_t inner_tiles = (m.inner + kTile - 1) / kTile;

    for (std::uint64_t unit =
                 static_cast<std::uint64_t>(blockIdx.x)
                         * (blockDim.x / Bf16WmmaLaunchPolicy::lanes)
                 + warp;
         unit < total; unit += stride) {
        const std::uint64_t plane = unit / units_per_plane;
        const std::uint64_t in_plane = unit % units_per_plane;
        const std::uint64_t head = in_plane / units_per_head;
        const std::uint64_t in_head = in_plane % units_per_head;
        const std::uint64_t row_tile = in_head / chunks_per_row;
        const std::uint64_t column_tile = in_head % chunks_per_row;

        // This plane's own transformed offset: the leading tuple is
        // decomposed with the operand's own leading extents and strides, so an
        // independent plane and a transformed view address exactly their own
        // logical cells.
        std::uint64_t x_plane = m.x_offset;
        std::uint64_t out_plane = m.out_offset;
        std::uint64_t rest = plane;
        for (std::uint32_t axis = m.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate = rest % m.dims[axis];
            rest /= m.dims[axis];
            x_plane += coordinate * m.x_strides[axis];
            out_plane += coordinate * m.out_strides[axis];
        }
        if (m.head_planar != 0) {
            out_plane += head * m.out_strides[m.leading_rank];
        }

        const std::uint64_t row_base = row_tile * kTile;
        const std::uint64_t column_base = column_tile * kTile;
        const std::uint64_t rows_remaining =
                m.rows - row_base < kTile ? m.rows - row_base : kTile;
        const std::uint64_t columns_remaining =
                m.columns - column_base < kTile
                ? m.columns - column_base
                : kTile;
        // The Hugging Face weight row of this unit's first output column:
        // `h*D + d` in head-planar mode and the plain output column `o` in
        // ordinary mode, where `heads` is one and `head` is always zero.
        const std::uint64_t weight_base = head * m.columns + column_base;
        const std::uint64_t source_base = m.start_row + row_base;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulated;
        wmma::fill_fragment(accumulated, 0.0f);
        for (std::uint64_t inner_tile = 0; inner_tile < inner_tiles;
             ++inner_tile) {
            // A[row, inner] is the selected row `start_row + row` of the
            // logical inner extent; B[column, inner] is the weight row of the
            // output coordinate. Cells outside any logical extent are neutral
            // zeros, so padding is never read and a tail can never contribute.
            for (std::uint64_t cell = lane; cell < kCells; cell += kLanes) {
                const std::uint64_t row = cell / kTile;
                const std::uint64_t inner = inner_tile * kTile + cell % kTile;
                staging.matrix_a[warp][cell] =
                        row < rows_remaining && inner < m.inner
                        ? bf16_wmma_load(
                                  m.x,
                                  iom::detail::plane_slot(
                                          x_plane, source_base + row, inner,
                                          m.source_rows, m.inner))
                        : kNeutral;
            }
            for (std::uint64_t cell = lane; cell < kCells; cell += kLanes) {
                const std::uint64_t column = cell / kTile;
                const std::uint64_t inner = inner_tile * kTile + cell % kTile;
                staging.matrix_b[warp][cell] =
                        column < columns_remaining && inner < m.inner
                        ? bf16_wmma_load(
                                  m.w,
                                  iom::detail::plane_slot(
                                          m.w_offset, weight_base + column,
                                          inner, m.outer, m.inner))
                        : kNeutral;
            }
            __syncwarp();
            // The whole warp loads identical fragment parameters and
            // participates unconditionally; the fragments never leave this
            // thread block.
            wmma::fragment<
                    wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                    wmma::row_major>
                    lhs;
            wmma::fragment<
                    wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                    wmma::col_major>
                    rhs;
            wmma::load_matrix_sync(lhs, staging.matrix_a[warp], 16);
            wmma::load_matrix_sync(rhs, staging.matrix_b[warp], 16);
            // `mma_sync` reads the fragments, not the staging tiles, so the
            // next K tile may overwrite them once this call returns.
            wmma::mma_sync(accumulated, lhs, rhs, accumulated);
            __syncwarp();
        }
        // The FP32 accumulators of every K tile are stored once, in row-major
        // order, into this warp's own result tile.
        wmma::store_matrix_sync(
                staging.accumulator[warp], accumulated, 16,
                wmma::mem_row_major);
        __syncwarp();

        // Packed writer ownership: the 16 BF16 columns of one logical row are
        // eight 32-bit words, so lane `l` owns word `l % 8` of row `l / 8` and
        // every fourth row after that. Each owning lane reads its word,
        // replaces only the logical cells it owns, and stores it once, so no
        // two lanes ever share a write unit and every padding field keeps its
        // value.
        for (unsigned int pass = 0; pass < 4; ++pass) {
            const std::uint64_t cell = lane + kLanes * pass;
            const std::uint64_t row = cell / (kTile / 2);
            const std::uint64_t column_pair = cell % (kTile / 2);
            const std::uint64_t column = column_base + column_pair * 2;
            if (row >= rows_remaining || column >= m.columns) {
                continue;
            }
            const std::uint64_t word =
                    iom::detail::plane_slot(
                            out_plane, row_base + row, column, m.rows,
                            m.columns)
                    / 2;
            unsigned int merged = iom::detail::linear_load_word(m.out, word);
            merged = (merged & 0xffff0000u)
                    | bf16_wmma_encode(
                              staging.accumulator[warp][row * kTile
                                                       + column_pair * 2]);
            if (column + 1 < m.columns) {
                merged = (merged & 0x0000ffffu)
                        | (bf16_wmma_encode(
                                   staging.accumulator
                                           [warp][row * kTile
                                                  + column_pair * 2 + 1])
                           << 16);
            }
            iom::detail::linear_store_word(m.out, word, merged);
        }
        __syncwarp();
    }
}

#endif  // __CUDA_ARCH__ >= 800

// The native BF16 kernel of this translation unit. A compiled image without
// the BF16 WMMA statements (the host pass, or any pre-Ampere target of the
// same translation unit) carries no work at all: the runtime facility fact
// never routes a request to such an image, so the empty body is unreachable on
// every device whose capability reports the specialization.
__global__ void standard_tiled_linear_bf16_kernel(
        iom::detail::LinearMetadata m) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    bf16_wmma_linear_body(m);
#else
    (void)m;
#endif
}

}  // namespace

// Policy-parameterized launch of the native BF16 device operation: the queue's
// already-created nonblocking stream, one uploaded fixed descriptor, and the
// fixed kernel-local geometry above. No second stream, allocation, staging, or
// synchronization is introduced.
void launch_linear_bf16(
        cudaStream_t stream, const iom::detail::LinearMetadata& metadata) {
    constexpr std::size_t kChunkSize = Bf16WmmaLaunchPolicy::tile;
    const std::size_t chunks_per_row = iom::detail::linear_checked_add(
            static_cast<std::size_t>(metadata.columns), kChunkSize - 1,
            "CUDA BF16 linear launch column count overflows")
            / kChunkSize;
    const std::size_t row_tiles = iom::detail::linear_checked_add(
            static_cast<std::size_t>(metadata.rows), kChunkSize - 1,
            "CUDA BF16 linear launch row count overflows")
            / kChunkSize;
    const std::size_t units_per_head = iom::detail::linear_checked_mul(
            row_tiles, chunks_per_row,
            "CUDA BF16 linear launch row count overflows");
    const std::size_t total = iom::detail::linear_checked_mul(
            iom::detail::linear_checked_mul(
                    static_cast<std::size_t>(metadata.plane_count),
                    static_cast<std::size_t>(metadata.heads),
                    "CUDA BF16 linear launch plane count overflows"),
            units_per_head,
            "CUDA BF16 linear launch chunk count overflows");
    const unsigned int blocks = static_cast<unsigned int>(
            total < Bf16WmmaLaunchPolicy::max_blocks
            ? total
            : Bf16WmmaLaunchPolicy::max_blocks);
    standard_tiled_linear_bf16_kernel<<<
            dim3(blocks), dim3(Bf16WmmaLaunchPolicy::threads), 0, stream>>>(
            metadata);
}

}  // namespace iom::cuda_detail
