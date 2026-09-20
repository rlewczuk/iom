#include "sdpa_matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <hip/hip_runtime.h>

#define IOM_GPU_DEVICE __device__
#define IOM_GPU_GLOBAL __global__
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_GPU_GLOBAL_STRIDE \
    (static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    hipLaunchKernelGGL( \
            kernel, dim3(blocks), dim3(threads), 0, stream, __VA_ARGS__)

namespace iom::rocm_detail {
namespace {
constexpr unsigned int kThreads = 256;
constexpr unsigned int kWavesPerBlock = kThreads / 32;
constexpr std::size_t kMaxBlocks = 65535;

[[nodiscard]] std::size_t checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0
            && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t pad16(
        std::size_t value, const char* message) {
    const std::size_t bumped = checked_add(value, 15, message);
    return bumped - bumped % 16;
}

[[nodiscard]] std::size_t align32(
        std::size_t value, const char* message) {
    const std::size_t bumped = checked_add(value, 31, message);
    return bumped - bumped % 32;
}

[[nodiscard]] std::size_t matrix_bytes(
        std::size_t planes, std::size_t rows, std::size_t columns,
        std::size_t element_bytes, const char* message) {
    const std::size_t padded_rows = pad16(rows, message);
    const std::size_t padded_columns = pad16(columns, message);
    return align32(
            checked_mul(
                    checked_mul(
                            checked_mul(planes, padded_rows, message),
                            padded_columns, message),
                    element_bytes, message),
            message);
}

[[nodiscard]] unsigned int launch_blocks(
        std::size_t units, unsigned int units_per_block, const char* message) {
    const std::size_t rounded = checked_add(
            units, static_cast<std::size_t>(units_per_block) - 1, message);
    const std::size_t blocks = rounded / units_per_block;
    return static_cast<unsigned int>(
            std::min<std::size_t>(blocks, kMaxBlocks));
}

[[nodiscard]] std::uint64_t to_u64(
        std::size_t value, const char* message) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::uint64_t>(value);
}

struct SdpaViewDevice {
    const unsigned char* data = nullptr;
    std::uint64_t plane_offset = 0;
    std::uint64_t plane_count = 0;
    std::uint64_t head_count = 0;
    std::uint64_t head_stride = 0;
    std::uint64_t leading_dimensions[kSdpaMatrixMaxLeadingRank]{};
    std::uint64_t leading_strides[kSdpaMatrixMaxLeadingRank]{};
    std::uint32_t leading_rank = 0;
};

static_assert(std::is_trivially_copyable_v<SdpaViewDevice>);

struct SdpaQkDeviceMetadata {
    SdpaViewDevice q;
    SdpaViewDevice k;
    unsigned char* q_pack = nullptr;
    unsigned char* kv_pack = nullptr;
    float* scores = nullptr;
    std::uint64_t plane_count = 0;
    std::uint64_t Hq = 0;
    std::uint64_t Hkv = 0;
    std::uint64_t R = 0;
    std::uint64_t C = 0;
    std::uint64_t D = 0;
    std::uint64_t L = 0;
    std::uint64_t Mp = 0;
    std::uint64_t Lp = 0;
    std::uint64_t Dp = 0;
    std::uint64_t grouping = 0;
};

struct SdpaPvDeviceMetadata {
    SdpaViewDevice v;
    const unsigned char* p_bf16 = nullptr;
    unsigned char* kv_pack = nullptr;
    unsigned char* pv_bf16 = nullptr;
    std::uint64_t plane_count = 0;
    std::uint64_t Hq = 0;
    std::uint64_t Hkv = 0;
    std::uint64_t R = 0;
    std::uint64_t C = 0;
    std::uint64_t D = 0;
    std::uint64_t L = 0;
    std::uint64_t Mp = 0;
    std::uint64_t Lp = 0;
    std::uint64_t Dp = 0;
    std::uint64_t grouping = 0;
};

static_assert(std::is_trivially_copyable_v<SdpaQkDeviceMetadata>);
static_assert(std::is_trivially_copyable_v<SdpaPvDeviceMetadata>);
static_assert(sizeof(SdpaQkDeviceMetadata) <= 512);
static_assert(sizeof(SdpaPvDeviceMetadata) <= 512);

[[nodiscard]] SdpaViewDevice make_device_view(const SdpaMatrixView& view) {
    SdpaViewDevice result;
    result.data = static_cast<const unsigned char*>(view.native_handle);
    result.plane_offset = to_u64(
            view.plane_offset, "SDPA matrix plane offset overflows");
    result.plane_count = to_u64(
            view.plane_count, "SDPA matrix plane count overflows");
    result.head_count = to_u64(
            view.head_count, "SDPA matrix head count overflows");
    result.head_stride = to_u64(
            view.head_stride, "SDPA matrix head stride overflows");
    result.leading_rank = static_cast<std::uint32_t>(view.leading_rank);
    for (std::size_t axis = 0; axis < view.leading_rank; ++axis) {
        result.leading_dimensions[axis] = to_u64(
                view.leading_dimensions[axis],
                "SDPA matrix leading dimension overflows");
        result.leading_strides[axis] = to_u64(
                view.leading_strides[axis],
                "SDPA matrix leading stride overflows");
    }
    return result;
}

[[nodiscard]] std::size_t expected_plane_count(
        const SdpaMatrixView& view, const char* message) {
    if (view.leading_rank > kSdpaMatrixMaxLeadingRank) {
        throw std::invalid_argument(
                "SDPA matrix leading rank exceeds the fixed snapshot");
    }
    std::size_t result = 1;
    for (std::size_t axis = 0; axis < view.leading_rank; ++axis) {
        if (view.leading_dimensions[axis] == 0
                || view.leading_strides[axis] == 0) {
            throw std::invalid_argument(
                    "SDPA matrix leading dimensions and strides must be "
                    "nonzero");
        }
        result = checked_mul(result, view.leading_dimensions[axis], message);
    }
    return result;
}

void validate_view(
        const SdpaMatrixView& view, std::size_t leading_rank,
        std::size_t planes, std::size_t heads, std::size_t rows,
        std::size_t columns, const char* name) {
    if (view.native_handle == nullptr) {
        throw std::invalid_argument(
                std::string(name) + " matrix handle is null");
    }
    if (view.leading_rank != leading_rank
            || view.head_count != heads
            || view.plane_count != planes
            || view.head_stride == 0) {
        throw std::invalid_argument(
                std::string(name) + " matrix snapshot shape mismatch");
    }
    if (reinterpret_cast<std::uintptr_t>(view.native_handle) % 2 != 0) {
        throw std::invalid_argument(
                std::string(name) + " BF16 handle is not two-byte aligned");
    }
    const std::size_t computed = expected_plane_count(
            view, "SDPA matrix leading plane count overflows");
    if (computed != planes) {
        throw std::invalid_argument(
                std::string(name) + " leading plane count mismatch");
    }
    const std::size_t plane_elements = checked_mul(
            pad16(rows, "SDPA source row padding overflows"),
            pad16(columns, "SDPA source column padding overflows"),
            "SDPA source plane size overflows");
    std::size_t maximum = view.plane_offset;
    maximum = checked_add(
            maximum,
            checked_mul(
                    heads - 1, view.head_stride,
                    "SDPA source head offset overflows"),
            "SDPA source head range overflows");
    for (std::size_t axis = 0; axis < view.leading_rank; ++axis) {
        maximum = checked_add(
                maximum,
                checked_mul(
                        view.leading_dimensions[axis] - 1,
                        view.leading_strides[axis],
                        "SDPA source leading offset overflows"),
                "SDPA source leading range overflows");
    }
    (void)checked_add(
            maximum,
            plane_elements - 1,
            "SDPA source plane range overflows");
}

void validate_common(
        std::size_t Hq, std::size_t Hkv, std::size_t R, std::size_t C,
        std::size_t D, std::size_t L, std::size_t grouping,
        std::size_t planes) {
    if (Hq == 0 || Hkv == 0 || R == 0 || C == 0 || D == 0 || L == 0) {
        throw std::invalid_argument(
                "SDPA matrix dimensions must be nonzero");
    }
    if (Hq % Hkv != 0 || grouping != Hq / Hkv) {
        throw std::invalid_argument(
                "SDPA matrix GQA grouping is inconsistent");
    }
    if (L > C) {
        throw std::invalid_argument(
                "SDPA matrix initialized length exceeds capacity");
    }
    if (planes == 0) {
        throw std::invalid_argument(
                "SDPA matrix leading plane count is zero");
    }
}

void validate_alignment(
        const void* address, const char* name) {
    if (address == nullptr
            || reinterpret_cast<std::uintptr_t>(address) % 32 != 0) {
        throw std::invalid_argument(
                std::string(name) + " scratch is not 32-byte aligned");
    }
}
struct ScratchRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

[[nodiscard]] ScratchRange make_scratch_range(
        const void* address, std::size_t bytes, const char* name) {
    validate_alignment(address, name);
    const std::uintptr_t begin =
            reinterpret_cast<std::uintptr_t>(address);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument(
                std::string(name) + " scratch range overflows");
    }
    return {begin, begin + bytes};
}

void reject_scratch_overlap(
        const ScratchRange& left, const ScratchRange& right,
        const char* message) {
    if (left.begin < right.end && right.begin < left.end) {
        throw std::invalid_argument(message);
    }
}


void validate_scratch(
        const SdpaQkRequest& request, std::size_t planes,
        std::size_t Mp, std::size_t Lp, std::size_t Dp) {
    const std::size_t q_bytes = matrix_bytes(
            checked_mul(planes, request.Hq,
                        "SDPA Q pack plane count overflows"),
            Mp, Dp, 2, "SDPA Q pack size overflows");
    const std::size_t kv_bytes = matrix_bytes(
            checked_mul(planes, request.Hkv,
                        "SDPA KV pack plane count overflows"),
            Lp, Dp, 2, "SDPA KV pack size overflows");
    const std::size_t scores_bytes = matrix_bytes(
            checked_mul(planes, request.Hq,
                        "SDPA score plane count overflows"),
            Mp, Lp, 4, "SDPA score size overflows");
    if (request.scratch.q_pack_bytes < q_bytes
            || request.scratch.kv_pack_bytes < kv_bytes
            || request.scratch.scores_bytes < scores_bytes) {
        throw std::invalid_argument(
                "SDPA QK scratch range is shorter than its checked segments");
    }
    const ScratchRange q_range = make_scratch_range(
            request.scratch.q_pack, q_bytes, "SDPA q_pack");
    const ScratchRange kv_range = make_scratch_range(
            request.scratch.kv_pack, kv_bytes, "SDPA kv_pack");
    const ScratchRange scores_range = make_scratch_range(
            request.scratch.scores, scores_bytes, "SDPA scores");
    reject_scratch_overlap(
            q_range, kv_range, "SDPA QK scratch segments overlap");
    reject_scratch_overlap(
            q_range, scores_range, "SDPA QK scratch segments overlap");
    reject_scratch_overlap(
            kv_range, scores_range, "SDPA QK scratch segments overlap");
}

void validate_scratch(
        const SdpaPvRequest& request, std::size_t planes,
        std::size_t Mp, std::size_t Lp, std::size_t Dp) {
    const std::size_t p_bytes = matrix_bytes(
            checked_mul(planes, request.Hq,
                        "SDPA probability plane count overflows"),
            Mp, Lp, 2, "SDPA probability size overflows");
    const std::size_t kv_bytes = matrix_bytes(
            checked_mul(planes, request.Hkv,
                        "SDPA KV pack plane count overflows"),
            Lp, Dp, 2, "SDPA KV pack size overflows");
    const std::size_t pv_bytes = matrix_bytes(
            checked_mul(planes, request.Hq,
                        "SDPA PV plane count overflows"),
            Mp, Dp, 2, "SDPA PV size overflows");
    if (request.scratch.p_bf16_bytes < p_bytes
            || request.scratch.kv_pack_bytes < kv_bytes
            || request.scratch.pv_bf16_bytes < pv_bytes) {
        throw std::invalid_argument(
                "SDPA PV scratch range is shorter than its checked segments");
    }
    const ScratchRange p_range = make_scratch_range(
            request.scratch.p_bf16, p_bytes, "SDPA p_bf16");
    const ScratchRange kv_range = make_scratch_range(
            request.scratch.kv_pack, kv_bytes, "SDPA kv_pack");
    const ScratchRange pv_range = make_scratch_range(
            request.scratch.pv_bf16, pv_bytes, "SDPA pv_bf16");
    reject_scratch_overlap(
            p_range, kv_range, "SDPA PV scratch segments overlap");
    reject_scratch_overlap(
            p_range, pv_range, "SDPA PV scratch segments overlap");
    reject_scratch_overlap(
            kv_range, pv_range, "SDPA PV scratch segments overlap");
}

void check_kernel(const char* operation) {
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        throw std::runtime_error(
                std::string(operation) + " failed with "
                + hipGetErrorName(status) + ": "
                + hipGetErrorString(status));
    }
}

#if defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
#define IOM_ROCM_SDPA_GFX12_BF16_WMMA 1
#endif
#endif
#ifndef IOM_ROCM_SDPA_GFX12_BF16_WMMA
#define IOM_ROCM_SDPA_GFX12_BF16_WMMA 0
#endif

typedef short sdpa_wmma_short8 __attribute__((ext_vector_type(8)));
typedef float sdpa_wmma_float8 __attribute__((ext_vector_type(8)));

IOM_GPU_DEVICE std::uint16_t sdpa_daz_bf16(std::uint16_t value) noexcept {
    // Preserve the sign bit while flushing only nonzero BF16 subnormals.
    return (value & 0x7f80u) == 0 ? value & 0x8000u : value;
}

IOM_GPU_DEVICE std::uint16_t sdpa_load_bf16(
        const unsigned char* base, std::uint64_t slot) noexcept {
    return reinterpret_cast<const std::uint16_t*>(base)[slot];
}

IOM_GPU_DEVICE void sdpa_store_bf16(
        unsigned char* base, std::uint64_t slot,
        std::uint16_t value) noexcept {
    reinterpret_cast<std::uint16_t*>(base)[slot] = value;
}

// Slot of one logical element inside a caller-owned tensor owner: the final
// two axes are the engine's standard 16x16 tile-major storage with row-major
// cells, and `rows`/`columns` are the logical extents that define the tile
// grid. Q, K, V and the merged destination are read or written through this
// mapping, never through the scratch mapping below.
IOM_GPU_DEVICE std::uint64_t sdpa_tile_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) noexcept {
    const std::uint64_t tile_columns = (columns + 15) / 16;
    const std::uint64_t tile_rows = (rows + 15) / 16;
    return (plane * tile_rows * tile_columns
            + (row / 16) * tile_columns + column / 16) * 256
            + (row % 16) * 16 + column % 16;
}

// Slot of one element inside a caller-owned packed scratch payload:
// `(plane * rows + row) * columns + column` over the checked 16-padded
// extents. q_pack, kv_pack, scores, p_bf16 and pv_bf16 are contiguous packed
// ranges so the private nonmatrix stages (src/rocm/sdpa_softmax.hpp) read and
// write the very same bytes through their plain element-strided layout, with
// one shared addressing convention for the joined pipeline and no conversion
// pass, extra segment, or hidden allocation between stages.
IOM_GPU_DEVICE std::uint64_t sdpa_packed_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) noexcept {
    return (plane * rows + row) * columns + column;
}

IOM_GPU_DEVICE std::uint64_t sdpa_source_plane(
        const SdpaViewDevice& view, std::uint64_t plane,
        std::uint64_t head) noexcept {
    std::uint64_t result = view.plane_offset + head * view.head_stride;
    std::uint64_t rest = plane;
    for (std::uint32_t axis = view.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % view.leading_dimensions[axis];
        rest /= view.leading_dimensions[axis];
        result += coordinate * view.leading_strides[axis];
    }
    return result;
}

IOM_GPU_DEVICE std::uint16_t sdpa_rne_bf16(float value) noexcept {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t rounded = bits + 0x7fffu
            + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

IOM_GPU_DEVICE sdpa_wmma_float8 sdpa_wmma_step(
        sdpa_wmma_short8 lhs, sdpa_wmma_short8 rhs,
        sdpa_wmma_float8 accumulator) noexcept {
#if IOM_ROCM_SDPA_GFX12_BF16_WMMA
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
            lhs, rhs, accumulator);
#else
    (void)lhs;
    (void)rhs;
    return accumulator;
#endif
}

IOM_GPU_GLOBAL void sdpa_q_pack_kernel(SdpaQkDeviceMetadata m) {
    const std::uint64_t total = m.plane_count * m.Hq * m.Mp * m.Dp;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    for (std::uint64_t index = IOM_GPU_GLOBAL_INDEX;
         index < total; index += stride) {
        const std::uint64_t plane_head = index / (m.Mp * m.Dp);
        const std::uint64_t in_plane = index % (m.Mp * m.Dp);
        const std::uint64_t row = in_plane / m.Dp;
        const std::uint64_t column = in_plane % m.Dp;
        const std::uint64_t plane = plane_head / m.Hq;
        const std::uint64_t head = plane_head % m.Hq;
        std::uint16_t value = 0;
        if (row < m.R && column < m.D) {
            const std::uint64_t source_plane =
                    sdpa_source_plane(m.q, plane, head);
            value = sdpa_daz_bf16(sdpa_load_bf16(
                    m.q.data,
                    sdpa_tile_slot(source_plane, row, column, m.R, m.D)));
        }
        sdpa_store_bf16(
                m.q_pack,
                sdpa_packed_slot(plane_head, row, column, m.Mp, m.Dp),
                value);
    }
}

IOM_GPU_GLOBAL void sdpa_k_pack_kernel(SdpaQkDeviceMetadata m) {
    const std::uint64_t total = m.plane_count * m.Hkv * m.Lp * m.Dp;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    for (std::uint64_t index = IOM_GPU_GLOBAL_INDEX;
         index < total; index += stride) {
        const std::uint64_t plane_head = index / (m.Lp * m.Dp);
        const std::uint64_t in_plane = index % (m.Lp * m.Dp);
        const std::uint64_t row = in_plane / m.Dp;
        const std::uint64_t column = in_plane % m.Dp;
        const std::uint64_t plane = plane_head / m.Hkv;
        const std::uint64_t head = plane_head % m.Hkv;
        std::uint16_t value = 0;
        if (row < m.L && column < m.D) {
            const std::uint64_t source_plane =
                    sdpa_source_plane(m.k, plane, head);
            value = sdpa_daz_bf16(sdpa_load_bf16(
                    m.k.data,
                    sdpa_tile_slot(source_plane, row, column, m.C, m.D)));
        }
        sdpa_store_bf16(
                m.kv_pack,
                sdpa_packed_slot(plane_head, row, column, m.Lp, m.Dp),
                value);
    }
}

IOM_GPU_GLOBAL void sdpa_qk_wmma_kernel(SdpaQkDeviceMetadata m) {
    const std::uint64_t lane = threadIdx.x & 31u;
    const std::uint64_t wave =
            static_cast<std::uint64_t>(blockIdx.x) * kWavesPerBlock
            + threadIdx.x / 32u;
    const std::uint64_t wave_count =
            static_cast<std::uint64_t>(gridDim.x) * kWavesPerBlock;
    const std::uint64_t row_tiles = m.Mp / 16;
    const std::uint64_t column_tiles = m.Lp / 16;
    const std::uint64_t inner_tiles = m.Dp / 16;
    const std::uint64_t tiles_per_plane = row_tiles * column_tiles;
    const std::uint64_t total = m.plane_count * m.Hq * tiles_per_plane;
    const std::uint64_t row_in_tile = lane % 16u;
    const std::uint64_t inner_in_tile = (lane / 16u) * 8u;
    const std::uint64_t column_in_tile = lane % 16u;
    for (std::uint64_t tile = wave; tile < total;
         tile += wave_count) {
        const std::uint64_t plane_head = tile / tiles_per_plane;
        const std::uint64_t in_plane = tile % tiles_per_plane;
        const std::uint64_t row_tile = in_plane / column_tiles;
        const std::uint64_t column_tile = in_plane % column_tiles;
        const std::uint64_t plane = plane_head / m.Hq;
        const std::uint64_t head = plane_head % m.Hq;
        const std::uint64_t kv_head = head / m.grouping;
        const std::uint64_t kv_plane_head = plane * m.Hkv + kv_head;
        sdpa_wmma_float8 accumulator{};
        for (std::uint64_t inner_tile = 0;
             inner_tile < inner_tiles; ++inner_tile) {
            sdpa_wmma_short8 lhs{};
            sdpa_wmma_short8 rhs{};
            for (unsigned int index = 0; index < 8; ++index) {
                const std::uint64_t row = row_tile * 16 + row_in_tile;
                const std::uint64_t column =
                        inner_tile * 16 + inner_in_tile + index;
                lhs[index] = static_cast<short>(sdpa_load_bf16(
                        m.q_pack,
                        sdpa_packed_slot(
                                plane_head, row, column, m.Mp, m.Dp)));
                const std::uint64_t key =
                        column_tile * 16 + column_in_tile;
                const std::uint64_t feature =
                        inner_tile * 16 + inner_in_tile + index;
                rhs[index] = static_cast<short>(sdpa_load_bf16(
                        m.kv_pack,
                        sdpa_packed_slot(
                                kv_plane_head, key, feature,
                                m.Lp, m.Dp)));
            }
            accumulator = sdpa_wmma_step(lhs, rhs, accumulator);
        }
        for (unsigned int index = 0; index < 8; ++index) {
            const std::uint64_t row = row_tile * 16
                    + (lane / 16u) * 8u + index;
            const std::uint64_t column = column_tile * 16
                    + column_in_tile;
            m.scores[sdpa_packed_slot(
                    plane_head, row, column, m.Mp, m.Lp)] = accumulator[index];
        }
    }
}

IOM_GPU_GLOBAL void sdpa_v_pack_kernel(SdpaPvDeviceMetadata m) {
    const std::uint64_t total = m.plane_count * m.Hkv * m.Lp * m.Dp;
    const std::uint64_t stride = IOM_GPU_GLOBAL_STRIDE;
    for (std::uint64_t index = IOM_GPU_GLOBAL_INDEX;
         index < total; index += stride) {
        const std::uint64_t plane_head = index / (m.Lp * m.Dp);
        const std::uint64_t in_plane = index % (m.Lp * m.Dp);
        const std::uint64_t row = in_plane / m.Dp;
        const std::uint64_t column = in_plane % m.Dp;
        const std::uint64_t plane = plane_head / m.Hkv;
        const std::uint64_t head = plane_head % m.Hkv;
        std::uint16_t value = 0;
        if (row < m.L && column < m.D) {
            const std::uint64_t source_plane =
                    sdpa_source_plane(m.v, plane, head);
            value = sdpa_daz_bf16(sdpa_load_bf16(
                    m.v.data,
                    sdpa_tile_slot(source_plane, row, column, m.C, m.D)));
        }
        sdpa_store_bf16(
                m.kv_pack,
                sdpa_packed_slot(plane_head, row, column, m.Lp, m.Dp),
                value);
    }
}

IOM_GPU_GLOBAL void sdpa_pv_wmma_kernel(SdpaPvDeviceMetadata m) {
    const std::uint64_t lane = threadIdx.x & 31u;
    const std::uint64_t wave =
            static_cast<std::uint64_t>(blockIdx.x) * kWavesPerBlock
            + threadIdx.x / 32u;
    const std::uint64_t wave_count =
            static_cast<std::uint64_t>(gridDim.x) * kWavesPerBlock;
    const std::uint64_t row_tiles = m.Mp / 16;
    const std::uint64_t column_tiles = m.Dp / 16;
    const std::uint64_t inner_tiles = m.Lp / 16;
    const std::uint64_t tiles_per_plane = row_tiles * column_tiles;
    const std::uint64_t total = m.plane_count * m.Hq * tiles_per_plane;
    const std::uint64_t row_in_tile = lane % 16u;
    const std::uint64_t inner_in_tile = (lane / 16u) * 8u;
    const std::uint64_t column_in_tile = lane % 16u;
    for (std::uint64_t tile = wave; tile < total;
         tile += wave_count) {
        const std::uint64_t plane_head = tile / tiles_per_plane;
        const std::uint64_t in_plane = tile % tiles_per_plane;
        const std::uint64_t row_tile = in_plane / column_tiles;
        const std::uint64_t column_tile = in_plane % column_tiles;
        const std::uint64_t plane = plane_head / m.Hq;
        const std::uint64_t head = plane_head % m.Hq;
        const std::uint64_t kv_head = head / m.grouping;
        const std::uint64_t kv_plane_head = plane * m.Hkv + kv_head;
        sdpa_wmma_float8 accumulator{};
        for (std::uint64_t inner_tile = 0;
             inner_tile < inner_tiles; ++inner_tile) {
            sdpa_wmma_short8 lhs{};
            sdpa_wmma_short8 rhs{};
            for (unsigned int index = 0; index < 8; ++index) {
                const std::uint64_t row = row_tile * 16 + row_in_tile;
                const std::uint64_t key =
                        inner_tile * 16 + inner_in_tile + index;
                std::uint16_t probability = 0;
                if (row < m.R && key < m.L) {
                    probability = sdpa_daz_bf16(sdpa_load_bf16(
                            m.p_bf16,
                            sdpa_packed_slot(
                                    plane_head, row, key, m.Mp, m.Lp)));
                }
                lhs[index] = static_cast<short>(probability);
                const std::uint64_t feature =
                        column_tile * 16 + column_in_tile;
                const std::uint64_t value_key = key;
                rhs[index] = static_cast<short>(sdpa_load_bf16(
                        m.kv_pack,
                        sdpa_packed_slot(
                                kv_plane_head, value_key, feature,
                                m.Lp, m.Dp)));
            }
            accumulator = sdpa_wmma_step(lhs, rhs, accumulator);
        }
        for (unsigned int index = 0; index < 8; ++index) {
            const std::uint64_t row = row_tile * 16
                    + (lane / 16u) * 8u + index;
            const std::uint64_t column = column_tile * 16
                    + column_in_tile;
            const float value = accumulator[index];
            const std::uint16_t encoded = value == 0.0F
                    ? static_cast<std::uint16_t>(0)
                    : sdpa_rne_bf16(value);
            sdpa_store_bf16(
                    m.pv_bf16,
                    sdpa_packed_slot(plane_head, row, column, m.Mp, m.Dp),
                    encoded);
        }
    }
}

#undef IOM_ROCM_SDPA_GFX12_BF16_WMMA

}  // namespace

}  // namespace iom::rocm_detail

namespace iom::rocm_detail {

SdpaMatrixWorkspaceRequirements sdpa_matrix_workspace_requirements(
        std::size_t planes, std::size_t Hq, std::size_t Hkv,
        std::size_t R, std::size_t L, std::size_t D) {
    if (planes == 0 || Hq == 0 || Hkv == 0 || R == 0 || L == 0 || D == 0) {
        throw std::invalid_argument(
                "SDPA matrix workspace dimensions must be nonzero");
    }
    const std::size_t q_planes = checked_mul(
            planes, Hq, "SDPA Q pack plane count overflows");
    const std::size_t kv_planes = checked_mul(
            planes, Hkv, "SDPA KV pack plane count overflows");
    const std::size_t q_bytes = matrix_bytes(
            q_planes, R, D, 2, "SDPA Q pack size overflows");
    const std::size_t kv_bytes = matrix_bytes(
            kv_planes, L, D, 2, "SDPA KV pack size overflows");
    const std::size_t score_bytes = matrix_bytes(
            q_planes, R, L, 4, "SDPA score size overflows");
    const std::size_t p_bytes = matrix_bytes(
            q_planes, R, L, 2, "SDPA probability size overflows");
    const std::size_t pv_bytes = matrix_bytes(
            q_planes, R, D, 2, "SDPA PV size overflows");
    SdpaMatrixWorkspaceRequirements result;
    result.q_pack_bytes = q_bytes;
    result.kv_pack_offset = align32(
            q_bytes, "SDPA KV pack offset overflows");
    result.kv_pack_bytes = kv_bytes;
    result.scores_offset = align32(
            checked_add(
                    result.kv_pack_offset, kv_bytes,
                    "SDPA score offset overflows"),
            "SDPA score offset alignment overflows");
    result.scores_bytes = score_bytes;
    result.p_bf16_offset = align32(
            checked_add(
                    result.scores_offset, score_bytes,
                    "SDPA probability offset overflows"),
            "SDPA probability offset alignment overflows");
    result.p_bf16_bytes = p_bytes;
    result.pv_bf16_offset = align32(
            checked_add(
                    result.p_bf16_offset, p_bytes,
                    "SDPA PV offset overflows"),
            "SDPA PV offset alignment overflows");
    result.pv_bf16_bytes = pv_bytes;
    result.bytes = align32(
            checked_add(
                    result.pv_bf16_offset, pv_bytes,
                    "SDPA matrix workspace size overflows"),
            "SDPA matrix workspace alignment overflows");
    return result;
}



SdpaMatrixCapability sdpa_matrix_capability(int device_ordinal) noexcept {
    if (device_ordinal < 0) {
        return SdpaMatrixCapability::unsupported;
    }
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device_ordinal) != hipSuccess
            || properties.warpSize != 32) {
        return SdpaMatrixCapability::unsupported;
    }
    const std::string_view architecture{properties.gcnArchName};
    if (architecture.starts_with("gfx1201")) {
        return SdpaMatrixCapability::gfx1201_wave32_wmma;
    }
    return SdpaMatrixCapability::unsupported;
}

namespace {

void validate_qk_request(
        const SdpaQkRequest& request, std::size_t& planes,
        std::size_t& Mp, std::size_t& Lp, std::size_t& Dp) {
    if (request.q.leading_rank != request.k.leading_rank) {
        throw std::invalid_argument(
                "SDPA QK snapshots have different leading ranks");
    }
    planes = request.q.plane_count;
    validate_common(
            request.Hq, request.Hkv, request.R, request.C, request.D,
            request.L, request.grouping, planes);
    validate_view(
            request.q, request.q.leading_rank, planes, request.Hq,
            request.R, request.D, "Q");
    validate_view(
            request.k, request.k.leading_rank, planes, request.Hkv,
            request.C, request.D, "K");
    if (request.a >= request.C
            || request.R > request.C - request.a) {
        throw std::invalid_argument(
                "SDPA QK causal window is outside the cache");
    }
    Mp = pad16(request.R, "SDPA QK row padding overflows");
    Lp = pad16(request.L, "SDPA QK length padding overflows");
    Dp = pad16(request.D, "SDPA QK feature padding overflows");
    validate_scratch(request, planes, Mp, Lp, Dp);
}

void validate_pv_request(
        const SdpaPvRequest& request, std::size_t& planes,
        std::size_t& Mp, std::size_t& Lp, std::size_t& Dp) {
    planes = request.v.plane_count;
    validate_common(
            request.Hq, request.Hkv, request.R, request.C, request.D,
            request.L, request.grouping, planes);
    validate_view(
            request.v, request.v.leading_rank, planes, request.Hkv,
            request.C, request.D, "V");
    if (request.a >= request.C
            || request.R > request.C - request.a) {
        throw std::invalid_argument(
                "SDPA PV causal window is outside the cache");
    }
    Mp = pad16(request.R, "SDPA PV row padding overflows");
    Lp = pad16(request.L, "SDPA PV length padding overflows");
    Dp = pad16(request.D, "SDPA PV feature padding overflows");
    validate_scratch(request, planes, Mp, Lp, Dp);
}

[[nodiscard]] SdpaQkDeviceMetadata make_qk_metadata(
        const SdpaQkRequest& request, std::size_t planes,
        std::size_t Mp, std::size_t Lp, std::size_t Dp) {
    SdpaQkDeviceMetadata metadata;
    metadata.q = make_device_view(request.q);
    metadata.k = make_device_view(request.k);
    metadata.q_pack = static_cast<unsigned char*>(request.scratch.q_pack);
    metadata.kv_pack = static_cast<unsigned char*>(request.scratch.kv_pack);
    metadata.scores = static_cast<float*>(request.scratch.scores);
    metadata.plane_count = to_u64(
            planes, "SDPA QK plane count overflows");
    metadata.Hq = to_u64(request.Hq, "SDPA QK Hq overflows");
    metadata.Hkv = to_u64(request.Hkv, "SDPA QK Hkv overflows");
    metadata.R = to_u64(request.R, "SDPA QK rows overflow");
    metadata.C = to_u64(request.C, "SDPA QK capacity overflows");
    metadata.D = to_u64(request.D, "SDPA QK features overflow");
    metadata.L = to_u64(request.L, "SDPA QK length overflows");
    metadata.Mp = to_u64(Mp, "SDPA QK padded rows overflow");
    metadata.Lp = to_u64(Lp, "SDPA QK padded length overflow");
    metadata.Dp = to_u64(Dp, "SDPA QK padded features overflow");
    metadata.grouping = to_u64(
            request.grouping, "SDPA QK grouping overflows");
    return metadata;
}

[[nodiscard]] SdpaPvDeviceMetadata make_pv_metadata(
        const SdpaPvRequest& request, std::size_t planes,
        std::size_t Mp, std::size_t Lp, std::size_t Dp) {
    SdpaPvDeviceMetadata metadata;
    metadata.v = make_device_view(request.v);
    metadata.p_bf16 = static_cast<const unsigned char*>(
            request.scratch.p_bf16);
    metadata.kv_pack = static_cast<unsigned char*>(request.scratch.kv_pack);
    metadata.pv_bf16 = static_cast<unsigned char*>(request.scratch.pv_bf16);
    metadata.plane_count = to_u64(
            planes, "SDPA PV plane count overflows");
    metadata.Hq = to_u64(request.Hq, "SDPA PV Hq overflows");
    metadata.Hkv = to_u64(request.Hkv, "SDPA PV Hkv overflows");
    metadata.R = to_u64(request.R, "SDPA PV rows overflow");
    metadata.C = to_u64(request.C, "SDPA PV capacity overflows");
    metadata.D = to_u64(request.D, "SDPA PV features overflow");
    metadata.L = to_u64(request.L, "SDPA PV length overflows");
    metadata.Mp = to_u64(Mp, "SDPA PV padded rows overflow");
    metadata.Lp = to_u64(Lp, "SDPA PV padded length overflow");
    metadata.Dp = to_u64(Dp, "SDPA PV padded features overflow");
    metadata.grouping = to_u64(
            request.grouping, "SDPA PV grouping overflows");
    return metadata;
}

}  // namespace

SdpaMatrixResult launch_sdpa_qk(
        hipStream_t stream, const SdpaQkRequest& request) {
    if (request.device_ordinal
            > static_cast<std::size_t>(
                      std::numeric_limits<int>::max())
            || sdpa_matrix_capability(
                       static_cast<int>(request.device_ordinal))
                    == SdpaMatrixCapability::unsupported) {
        return SdpaMatrixResult::unsupported;
    }
    if (stream == nullptr) {
        throw std::invalid_argument("SDPA QK stream is null");
    }
    std::size_t planes = 0;
    std::size_t Mp = 0;
    std::size_t Lp = 0;
    std::size_t Dp = 0;
    validate_qk_request(request, planes, Mp, Lp, Dp);
    const SdpaQkDeviceMetadata metadata =
            make_qk_metadata(request, planes, Mp, Lp, Dp);
    const std::size_t q_elements = checked_mul(
            checked_mul(planes, request.Hq,
                        "SDPA QK launch plane count overflows"),
            checked_mul(Mp, Dp, "SDPA QK launch tile size overflows"),
            "SDPA QK launch element count overflows");
    const std::size_t kv_elements = checked_mul(
            checked_mul(planes, request.Hkv,
                        "SDPA QK launch KV planes overflow"),
            checked_mul(Lp, Dp, "SDPA QK launch KV tile size overflows"),
            "SDPA QK launch KV element count overflows");
    const std::size_t qk_tiles = checked_mul(
            checked_mul(planes, request.Hq,
                        "SDPA QK launch query planes overflow"),
            checked_mul(Mp / 16, Lp / 16,
                        "SDPA QK launch tile count overflows"),
            "SDPA QK launch tile count overflows");
    const unsigned int pack_blocks = launch_blocks(
            q_elements, kThreads, "SDPA QK pack launch count overflows");
    const unsigned int kv_blocks = launch_blocks(
            kv_elements, kThreads, "SDPA QK KV launch count overflows");
    const unsigned int wmma_blocks = launch_blocks(
            qk_tiles, kWavesPerBlock,
            "SDPA QK WMMA launch count overflows");
    IOM_LAUNCH_KERNEL(
            sdpa_q_pack_kernel, pack_blocks, kThreads, stream, metadata);
    check_kernel("SDPA Q pack kernel launch");
    IOM_LAUNCH_KERNEL(
            sdpa_k_pack_kernel, kv_blocks, kThreads, stream, metadata);
    check_kernel("SDPA K pack kernel launch");
    IOM_LAUNCH_KERNEL(
            sdpa_qk_wmma_kernel, wmma_blocks,
            kWavesPerBlock * 32u, stream, metadata);
    check_kernel("SDPA QK WMMA kernel launch");
    if (request.completion != nullptr) {
        request.completion->qk_submitted = true;
    }
    return SdpaMatrixResult::submitted;
}

SdpaMatrixResult launch_sdpa_pv(
        hipStream_t stream, const SdpaPvRequest& request) {
    if (request.device_ordinal
            > static_cast<std::size_t>(
                      std::numeric_limits<int>::max())
            || sdpa_matrix_capability(
                       static_cast<int>(request.device_ordinal))
                    == SdpaMatrixCapability::unsupported) {
        return SdpaMatrixResult::unsupported;
    }
    if (stream == nullptr) {
        throw std::invalid_argument("SDPA PV stream is null");
    }
    std::size_t planes = 0;
    std::size_t Mp = 0;
    std::size_t Lp = 0;
    std::size_t Dp = 0;
    validate_pv_request(request, planes, Mp, Lp, Dp);
    const SdpaPvDeviceMetadata metadata =
            make_pv_metadata(request, planes, Mp, Lp, Dp);
    const std::size_t kv_elements = checked_mul(
            checked_mul(planes, request.Hkv,
                        "SDPA PV launch KV planes overflow"),
            checked_mul(Lp, Dp, "SDPA PV launch KV tile size overflows"),
            "SDPA PV launch KV element count overflows");
    const std::size_t pv_tiles = checked_mul(
            checked_mul(planes, request.Hq,
                        "SDPA PV launch query planes overflow"),
            checked_mul(Mp / 16, Dp / 16,
                        "SDPA PV launch tile count overflows"),
            "SDPA PV launch tile count overflows");
    const unsigned int pack_blocks = launch_blocks(
            kv_elements, kThreads, "SDPA PV V launch count overflows");
    const unsigned int wmma_blocks = launch_blocks(
            pv_tiles, kWavesPerBlock,
            "SDPA PV WMMA launch count overflows");
    IOM_LAUNCH_KERNEL(
            sdpa_v_pack_kernel, pack_blocks, kThreads, stream, metadata);
    check_kernel("SDPA V pack kernel launch");
    IOM_LAUNCH_KERNEL(
            sdpa_pv_wmma_kernel, wmma_blocks,
            kWavesPerBlock * 32u, stream, metadata);
    check_kernel("SDPA PV WMMA kernel launch");
    if (request.completion != nullptr) {
        request.completion->pv_submitted = true;
    }
    return SdpaMatrixResult::submitted;
}


}  // namespace iom::rocm_detail

#undef IOM_LAUNCH_KERNEL
#undef IOM_GPU_GLOBAL_STRIDE
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE
