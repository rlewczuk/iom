#include "sdpa_matrix.hpp"

#include "../iom_internal.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace iom::cuda_detail {
namespace {

constexpr std::uint64_t kTile = 16;
constexpr std::uint64_t kCells = kTile * kTile;
constexpr unsigned int kLanes = 32;
constexpr unsigned int kWarpsPerBlock = 8;
constexpr unsigned int kThreads = kLanes * kWarpsPerBlock;
constexpr std::uint64_t kMaxBlocks = 65535;

[[nodiscard]] std::uint64_t checked_add(
        std::uint64_t left, std::uint64_t right, const char* message) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t checked_mul(
        std::uint64_t left, std::uint64_t right, const char* message) {
    if (left != 0
            && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::uint64_t checked_pad16(
        std::uint64_t value, const char* message) {
    const std::uint64_t bumped = checked_add(value, kTile - 1, message);
    return checked_mul(bumped / kTile, kTile, message);
}

[[nodiscard]] std::uint64_t checked_plane_slots(
        std::uint64_t rows, std::uint64_t columns, const char* message) {
    return checked_mul(
            checked_pad16(rows, message), checked_pad16(columns, message),
            message);
}

[[nodiscard]] std::string cuda_error(
        const char* operation, cudaError_t status) {
    return std::string(operation) + " failed with "
            + cudaGetErrorName(status) + ": "
            + cudaGetErrorString(status);
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error(operation, status));
    }
}

[[nodiscard]] int resolve_device(std::int32_t requested) {
    if (requested >= 0) {
        int current = -1;
        require_cuda(cudaGetDevice(&current), "cudaGetDevice");
        if (current != requested) {
            // Pointer attributes are checked against the requested ordinal
            // below.  The launch itself still runs on the caller's current
            // device, so reject the mismatch before any native work.
            throw std::invalid_argument(
                    "CUDA SDPA matrix request device is not current");
        }
        return requested;
    }
    int current = -1;
    require_cuda(cudaGetDevice(&current), "cudaGetDevice");
    return current;
}

[[nodiscard]] bool device_supports_wmma(int device) noexcept {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return false;
    }
    return properties.major >= 8;
}

void require_device_pointer(
        const void* pointer, int expected_device, const char* name) {
    if (pointer == nullptr) {
        throw std::invalid_argument(
                std::string("CUDA SDPA matrix ") + name
                + " pointer is null");
    }
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        throw std::invalid_argument(
                std::string("CUDA SDPA matrix ") + name
                + " pointer is not a device allocation");
    }
    if (attributes.type != cudaMemoryTypeDevice
            || attributes.devicePointer == nullptr) {
        throw std::invalid_argument(
                std::string("CUDA SDPA matrix ") + name
                + " pointer is not device storage");
    }
    if (attributes.device != expected_device) {
        throw std::invalid_argument(
                std::string("CUDA SDPA matrix ") + name
                + " pointer belongs to another device");
    }
    if ((reinterpret_cast<std::uintptr_t>(pointer) & 31u) != 0) {
        throw std::invalid_argument(
                std::string("CUDA SDPA matrix ") + name
                + " pointer is not 32-byte aligned");
    }
}

void validate_leading_shape(const SdpaMatrixShape& shape) {
    if (shape.leading_rank > kSdpaMatrixMaxLeadingRank) {
        throw std::invalid_argument(
                "CUDA SDPA matrix leading rank exceeds the fixed ABI");
    }
    std::uint64_t planes = 1;
    for (std::uint32_t axis = 0; axis < shape.leading_rank; ++axis) {
        if (shape.leading_dimensions[axis] == 0) {
            throw std::invalid_argument(
                    "CUDA SDPA matrix leading dimensions must be nonzero");
        }
        planes = checked_mul(
                planes, shape.leading_dimensions[axis],
                "CUDA SDPA matrix leading plane count overflows");
    }
    if (planes != shape.planes || shape.planes == 0) {
        throw std::invalid_argument(
                "CUDA SDPA matrix leading plane count is inconsistent");
    }
}

void validate_shape(const SdpaMatrixShape& shape) {
    validate_leading_shape(shape);
    if (shape.Hq == 0 || shape.Hkv == 0 || shape.R == 0
            || shape.C == 0 || shape.D == 0 || shape.L == 0) {
        throw std::invalid_argument(
                "CUDA SDPA matrix dimensions must be nonzero");
    }
    if (shape.Hq % shape.Hkv != 0
            || shape.grouping != shape.Hq / shape.Hkv
            || shape.grouping == 0) {
        throw std::invalid_argument(
                "CUDA SDPA matrix GQA grouping is inconsistent");
    }
    if (shape.a >= shape.C || shape.L > shape.C
            || shape.R > shape.C - shape.a) {
        throw std::invalid_argument(
                "CUDA SDPA matrix causal range is invalid");
    }
    const std::uint64_t output_width = checked_mul(
            shape.Hq, shape.D,
            "CUDA SDPA matrix output width overflows");
    if (shape.output_width != output_width) {
        throw std::invalid_argument(
                "CUDA SDPA matrix output width is inconsistent");
    }
    if (shape.Rp != checked_pad16(
                          shape.R,
                          "CUDA SDPA matrix padded row extent overflows")
            || shape.Lp != checked_pad16(
                           shape.L,
                           "CUDA SDPA matrix padded key extent overflows")
            || shape.Dp != checked_pad16(
                           shape.D,
                           "CUDA SDPA matrix padded feature extent overflows")) {
        throw std::invalid_argument(
                "CUDA SDPA matrix padded extents are inconsistent");
    }
    (void)checked_mul(
            checked_mul(
                    checked_mul(shape.planes, shape.Hq,
                                "CUDA SDPA matrix score extent overflows"),
                    shape.Rp, "CUDA SDPA matrix score extent overflows"),
            shape.Lp, "CUDA SDPA matrix score extent overflows");
}

void validate_leading_address(
        const SdpaMatrixShape& shape, const SdpaMatrixOperand& operand,
        std::uint64_t heads, std::uint64_t rows, std::uint64_t columns,
        const char* name) {
    const std::uint64_t plane_slots = checked_plane_slots(
            rows, columns,
            "CUDA SDPA matrix tiled plane extent overflows");
    std::uint64_t last_plane = operand.plane_offset;
    for (std::uint32_t axis = 0; axis < shape.leading_rank; ++axis) {
        const std::uint64_t extent = shape.leading_dimensions[axis];
        const std::uint64_t contribution = checked_mul(
                extent - 1, operand.plane_strides[axis],
                "CUDA SDPA matrix plane address overflows");
        last_plane = checked_add(
                last_plane, contribution,
                "CUDA SDPA matrix plane address overflows");
    }
    if (heads > 1) {
        last_plane = checked_add(
                last_plane,
                checked_mul(
                        heads - 1, operand.head_stride,
                        "CUDA SDPA matrix head address overflows"),
                "CUDA SDPA matrix head address overflows");
    }
    (void)checked_add(
            checked_mul(
                    last_plane, plane_slots,
                    "CUDA SDPA matrix storage address overflows"),
            plane_slots - 1,
            "CUDA SDPA matrix storage address overflows");
    if (operand.data == nullptr) {
        throw std::invalid_argument(
                std::string("CUDA SDPA matrix ") + name
                + " pointer is null");
    }
}

void validate_output_address(
        const SdpaMatrixShape& shape, const SdpaMatrixOutput& output) {
    const std::uint64_t plane_slots = checked_plane_slots(
            shape.R, shape.output_width,
            "CUDA SDPA matrix output tiled extent overflows");
    std::uint64_t last_plane = output.plane_offset;
    for (std::uint32_t axis = 0; axis < shape.leading_rank; ++axis) {
        last_plane = checked_add(
                last_plane,
                checked_mul(
                        shape.leading_dimensions[axis] - 1,
                        output.plane_strides[axis],
                        "CUDA SDPA matrix output plane address overflows"),
                "CUDA SDPA matrix output plane address overflows");
    }
    (void)checked_add(
            checked_mul(
                    last_plane, plane_slots,
                    "CUDA SDPA matrix output address overflows"),
            plane_slots - 1,
            "CUDA SDPA matrix output address overflows");
    if (output.data == nullptr) {
        throw std::invalid_argument("CUDA SDPA matrix output pointer is null");
    }
}

void validate_qk(const SdpaQkRequest& request, int device) {
    validate_shape(request.shape);
    if (!device_supports_wmma(device)) {
        throw detail::UnsupportedOperation();
    }
    require_device_pointer(request.q.data, device, "Q");
    require_device_pointer(request.k.data, device, "K");
    require_device_pointer(request.scores, device, "scores");
    validate_leading_address(
            request.shape, request.q, request.shape.Hq,
            request.shape.R, request.shape.D, "Q");
    validate_leading_address(
            request.shape, request.k, request.shape.Hkv,
            request.shape.C, request.shape.D, "K");
}

void validate_pv(const SdpaPvRequest& request, int device) {
    validate_shape(request.shape);
    if (!device_supports_wmma(device)) {
        throw detail::UnsupportedOperation();
    }
    require_device_pointer(request.probability, device, "probability");
    require_device_pointer(request.v.data, device, "V");
    require_device_pointer(request.out.data, device, "output");
    validate_leading_address(
            request.shape, request.v, request.shape.Hkv,
            request.shape.C, request.shape.D, "V");
    validate_output_address(request.shape, request.out);
}

struct alignas(32) MatrixStaging {
    __nv_bfloat16 lhs[kWarpsPerBlock][kCells];
    __nv_bfloat16 rhs[kWarpsPerBlock][kCells];
    float accumulator[kWarpsPerBlock][kCells];
};

static_assert(std::is_trivially_copyable_v<MatrixStaging>);
static_assert(alignof(MatrixStaging) >= 32);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

__device__ __forceinline__ std::uint16_t matrix_bf16_bits(
        const unsigned char* base, std::uint64_t slot) {
    const auto* words = reinterpret_cast<const std::uint32_t*>(base);
    const std::uint32_t word = words[slot >> 1];
    return static_cast<std::uint16_t>(
            (word >> ((slot & 1u) * 16u)) & 0xffffu);
}

__device__ __forceinline__ std::uint16_t matrix_bf16_daz(
        std::uint16_t bits) {
    // BF16 exponent zero is exactly the operand range below 2^-126.  Keep
    // the sign bit for copysign(+0,x), while normals, infinities, and NaNs
    // pass through unchanged.
    if ((bits & 0x7f80u) == 0 && (bits & 0x007fu) != 0) {
        return static_cast<std::uint16_t>(bits & 0x8000u);
    }
    return bits;
}

__device__ __forceinline__ __nv_bfloat16 matrix_load_bf16(
        const unsigned char* base, std::uint64_t slot) {
    return __ushort_as_bfloat16(matrix_bf16_daz(matrix_bf16_bits(base, slot)));
}

__device__ __forceinline__ __nv_bfloat16 matrix_load_linear_bf16(
        const unsigned char* base, std::uint64_t element) {
    return __ushort_as_bfloat16(
            matrix_bf16_daz(matrix_bf16_bits(base, element)));
}

__device__ __forceinline__ std::uint64_t matrix_plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_rows = (rows + kTile - 1) / kTile;
    const std::uint64_t tile_columns = (columns + kTile - 1) / kTile;
    const std::uint64_t tile_index =
            plane * tile_rows * tile_columns
            + (row / kTile) * tile_columns + column / kTile;
    return tile_index * kCells
            + (row % kTile) * kTile + column % kTile;
}

__device__ __forceinline__ std::uint64_t matrix_plane_base(
        const SdpaMatrixShape& shape, const SdpaMatrixOperand& operand,
        std::uint64_t plane, std::uint64_t head) {
    std::uint64_t rest = plane;
    std::uint64_t result = operand.plane_offset;
    for (std::uint32_t axis = shape.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % shape.leading_dimensions[axis];
        rest /= shape.leading_dimensions[axis];
        result += coordinate * operand.plane_strides[axis];
    }
    return result + head * operand.head_stride;
}

__device__ __forceinline__ std::uint16_t matrix_bf16_encode(float value) {
    // The matrix result is rounded once at the destination.  Canonicalize
    // both signs of numerical zero before applying the exact BF16 RNE carry.
    if (value == 0.0f) {
        return 0;
    }
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    const std::uint32_t rounded = bits + 0x7fffu + lsb;
    return static_cast<std::uint16_t>(rounded >> 16);
}

__device__ __forceinline__ void matrix_store_bf16(
        unsigned char* base, std::uint64_t slot, std::uint16_t value) {
    auto* target =
            reinterpret_cast<std::uint32_t*>(base) + (slot >> 1);
    const unsigned int shift = static_cast<unsigned int>(slot & 1u) * 16u;
    const std::uint32_t mask = 0xffffu << shift;
    std::uint32_t observed = *target;
    while (true) {
        const std::uint32_t desired =
                (observed & ~mask)
                | (static_cast<std::uint32_t>(value) << shift);
        const std::uint32_t prior =
                atomicCAS(target, observed, desired);
        if (prior == observed) {
            return;
        }
        observed = prior;
    }
}

__device__ void sdpa_qk_body(const SdpaQkRequest& request) {
    namespace wmma = nvcuda::wmma;
    const SdpaMatrixShape& shape = request.shape;
    const unsigned int warp = threadIdx.x / kLanes;
    const unsigned int lane = threadIdx.x % kLanes;
    __shared__ MatrixStaging staging;

    const std::uint64_t row_tiles = (shape.R + kTile - 1) / kTile;
    const std::uint64_t key_tiles = (shape.L + kTile - 1) / kTile;
    const std::uint64_t units_per_head = row_tiles * key_tiles;
    const std::uint64_t units_per_plane = shape.Hq * units_per_head;
    const std::uint64_t total = shape.planes * units_per_plane;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(gridDim.x) * kWarpsPerBlock;
    const std::uint64_t depth_tiles = shape.Dp / kTile;

    for (std::uint64_t unit =
                 static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock
                         + warp;
         unit < total; unit += stride) {
        const std::uint64_t plane = unit / units_per_plane;
        const std::uint64_t in_plane = unit % units_per_plane;
        const std::uint64_t head = in_plane / units_per_head;
        const std::uint64_t in_head = in_plane % units_per_head;
        const std::uint64_t row_tile = in_head / key_tiles;
        const std::uint64_t key_tile = in_head % key_tiles;
        const std::uint64_t q_plane = matrix_plane_base(
                shape, request.q, plane, head);
        const std::uint64_t k_plane = matrix_plane_base(
                shape, request.k, plane, head / shape.grouping);
        const std::uint64_t row_base = row_tile * kTile;
        const std::uint64_t key_base = key_tile * kTile;
        const std::uint64_t rows_remaining =
                shape.R - row_base < kTile ? shape.R - row_base : kTile;
        const std::uint64_t keys_remaining =
                shape.L - key_base < kTile ? shape.L - key_base : kTile;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        for (std::uint64_t depth_tile = 0;
             depth_tile < depth_tiles; ++depth_tile) {
            for (std::uint64_t cell = lane; cell < kCells; cell += kLanes) {
                const std::uint64_t row = cell / kTile;
                const std::uint64_t depth =
                        depth_tile * kTile + cell % kTile;
                staging.lhs[warp][cell] =
                        row < rows_remaining && depth < shape.D
                        ? matrix_load_bf16(
                                  request.q.data,
                                  matrix_plane_slot(
                                          q_plane, row_base + row, depth,
                                          shape.R, shape.D))
                        : __ushort_as_bfloat16(0u);
            }
            // The K tile is staged as B[d,t] in column-major storage: memory
            // is indexed [token][depth], so WMMA sees K transposed without a
            // complete K pack or repeated KV-head materialization.
            for (std::uint64_t cell = lane; cell < kCells; cell += kLanes) {
                const std::uint64_t token = cell / kTile;
                const std::uint64_t depth =
                        depth_tile * kTile + cell % kTile;
                staging.rhs[warp][cell] =
                        token < keys_remaining && depth < shape.D
                        ? matrix_load_bf16(
                                  request.k.data,
                                  matrix_plane_slot(
                                          k_plane, key_base + token, depth,
                                          shape.C, shape.D))
                        : __ushort_as_bfloat16(0u);
            }
            __syncwarp();
            wmma::fragment<
                    wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                    wmma::row_major>
                    lhs;
            wmma::fragment<
                    wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                    wmma::col_major>
                    rhs;
            wmma::load_matrix_sync(lhs, staging.lhs[warp], 16);
            wmma::load_matrix_sync(rhs, staging.rhs[warp], 16);
            wmma::mma_sync(accumulator, lhs, rhs, accumulator);
            __syncwarp();
        }
        wmma::store_matrix_sync(
                staging.accumulator[warp], accumulator, 16,
                wmma::mem_row_major);
        __syncwarp();
        for (std::uint64_t cell = lane; cell < kCells; cell += kLanes) {
            const std::uint64_t row = cell / kTile;
            const std::uint64_t token = cell % kTile;
            if (row < rows_remaining && token < keys_remaining) {
                const std::uint64_t score_index =
                        (((plane * shape.Hq + head) * shape.Rp
                          + row_base + row)
                         * shape.Lp)
                        + key_base + token;
                request.scores[score_index] =
                        staging.accumulator[warp][cell];
            }
        }
        __syncwarp();
    }
}

__device__ void sdpa_pv_body(const SdpaPvRequest& request) {
    namespace wmma = nvcuda::wmma;
    const SdpaMatrixShape& shape = request.shape;
    const unsigned int warp = threadIdx.x / kLanes;
    const unsigned int lane = threadIdx.x % kLanes;
    __shared__ MatrixStaging staging;

    const std::uint64_t row_tiles = (shape.R + kTile - 1) / kTile;
    const std::uint64_t depth_tiles = (shape.D + kTile - 1) / kTile;
    const std::uint64_t units_per_head = row_tiles * depth_tiles;
    const std::uint64_t units_per_plane = shape.Hq * units_per_head;
    const std::uint64_t total = shape.planes * units_per_plane;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(gridDim.x) * kWarpsPerBlock;
    const std::uint64_t key_tiles = shape.Lp / kTile;

    for (std::uint64_t unit =
                 static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock
                         + warp;
         unit < total; unit += stride) {
        const std::uint64_t plane = unit / units_per_plane;
        const std::uint64_t in_plane = unit % units_per_plane;
        const std::uint64_t head = in_plane / units_per_head;
        const std::uint64_t in_head = in_plane % units_per_head;
        const std::uint64_t row_tile = in_head / depth_tiles;
        const std::uint64_t depth_tile = in_head % depth_tiles;
        const std::uint64_t v_plane = matrix_plane_base(
                shape, request.v, plane, head / shape.grouping);
        // The output has the common leading tuple only.  Keep its plane
        // mapping independent from the head-planar V mapping above.
        std::uint64_t output_plane = request.out.plane_offset;
        std::uint64_t rest = plane;
        for (std::uint32_t axis = shape.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate =
                    rest % shape.leading_dimensions[axis];
            rest /= shape.leading_dimensions[axis];
            output_plane += coordinate * request.out.plane_strides[axis];
        }
        const std::uint64_t row_base = row_tile * kTile;
        const std::uint64_t depth_base = depth_tile * kTile;
        const std::uint64_t rows_remaining =
                shape.R - row_base < kTile ? shape.R - row_base : kTile;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        for (std::uint64_t key_tile = 0; key_tile < key_tiles; ++key_tile) {
            for (std::uint64_t cell = lane; cell < kCells; cell += kLanes) {
                const std::uint64_t row = cell / kTile;
                const std::uint64_t token =
                        key_tile * kTile + cell % kTile;
                const std::uint64_t probability_index =
                        (((plane * shape.Hq + head) * shape.Rp
                          + row_base + row)
                         * shape.Lp)
                        + token;
                staging.lhs[warp][cell] =
                        row < rows_remaining && token < shape.L
                        ? matrix_load_linear_bf16(
                                  request.probability, probability_index)
                        : __ushort_as_bfloat16(0u);
            }
            for (std::uint64_t cell = lane; cell < kCells; cell += kLanes) {
                const std::uint64_t token =
                        key_tile * kTile + cell / kTile;
                const std::uint64_t depth =
                        depth_base + cell % kTile;
                staging.rhs[warp][cell] =
                        token < shape.L && depth < shape.D
                        ? matrix_load_bf16(
                                  request.v.data,
                                  matrix_plane_slot(
                                          v_plane, token, depth,
                                          shape.C, shape.D))
                        : __ushort_as_bfloat16(0u);
            }
            __syncwarp();
            wmma::fragment<
                    wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                    wmma::row_major>
                    lhs;
            wmma::fragment<
                    wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                    wmma::row_major>
                    rhs;
            wmma::load_matrix_sync(lhs, staging.lhs[warp], 16);
            wmma::load_matrix_sync(rhs, staging.rhs[warp], 16);
            wmma::mma_sync(accumulator, lhs, rhs, accumulator);
            __syncwarp();
        }
        wmma::store_matrix_sync(
                staging.accumulator[warp], accumulator, 16,
                wmma::mem_row_major);
        __syncwarp();

        // One warp owns one output tile.  A lane owns one output pair per
        // pass; every halfword update is atomic so transformed odd offsets
        // and adjacent head columns cannot lose each other.
        for (unsigned int pass = 0; pass < 4; ++pass) {
            const std::uint64_t pair = lane + kLanes * pass;
            const std::uint64_t row = pair / (kTile / 2);
            const std::uint64_t pair_in_row = pair % (kTile / 2);
            const std::uint64_t depth = depth_base + pair_in_row * 2;
            if (row >= rows_remaining || depth >= shape.D) {
                continue;
            }
            const std::uint64_t output_column = head * shape.D + depth;
            const std::uint16_t first =
                    matrix_bf16_encode(
                            staging.accumulator
                                    [warp][row * kTile + pair_in_row * 2]);
            matrix_store_bf16(
                    request.out.data,
                    matrix_plane_slot(
                            output_plane, row_base + row, output_column,
                            shape.R, shape.output_width),
                    first);
            if (depth + 1 < shape.D) {
                const std::uint16_t second =
                        matrix_bf16_encode(
                                staging.accumulator[warp][
                                        row * kTile + pair_in_row * 2 + 1]);
                matrix_store_bf16(
                        request.out.data,
                        matrix_plane_slot(
                                output_plane, row_base + row,
                                output_column + 1, shape.R,
                                shape.output_width),
                        second);
            }
        }
        __syncwarp();
    }
}

#endif  // __CUDA_ARCH__ >= 800

__global__ void sdpa_qk_kernel(SdpaQkRequest request) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    sdpa_qk_body(request);
#else
    (void)request;
#endif
}

__global__ void sdpa_pv_kernel(SdpaPvRequest request) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    sdpa_pv_body(request);
#else
    (void)request;
#endif
}

[[nodiscard]] std::size_t launch_blocks(
        const SdpaMatrixShape& shape, bool pv) {
    const std::uint64_t row_tiles = (shape.R + kTile - 1) / kTile;
    const std::uint64_t other_tiles =
            (pv ? shape.D : shape.L) + kTile - 1;
    const std::uint64_t tiles = other_tiles / kTile;
    const std::uint64_t units = checked_mul(
            checked_mul(shape.planes, shape.Hq,
                        "CUDA SDPA matrix launch units overflow"),
            checked_mul(row_tiles, tiles,
                        "CUDA SDPA matrix launch units overflow"),
            "CUDA SDPA matrix launch units overflow");
    const std::uint64_t blocks =
            (units + kWarpsPerBlock - 1) / kWarpsPerBlock;
    return static_cast<std::size_t>(
            blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

}  // namespace

bool sdpa_matrix_wmma_supported(std::int32_t device_ordinal) noexcept {
    int device = -1;
    if (device_ordinal >= 0) {
        device = device_ordinal;
    } else if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    return device_supports_wmma(device);
}

void launch_sdpa_qk(cudaStream_t stream, const SdpaQkRequest& request) {
    const int device = resolve_device(request.shape.device_ordinal);
    validate_qk(request, device);
    const std::size_t blocks = launch_blocks(request.shape, false);
    sdpa_qk_kernel<<<dim3(static_cast<unsigned int>(blocks)), dim3(kThreads),
                     0, stream>>>(request);
    require_cuda(cudaGetLastError(), "CUDA SDPA QK kernel launch");
}

void launch_sdpa_pv(cudaStream_t stream, const SdpaPvRequest& request) {
    const int device = resolve_device(request.shape.device_ordinal);
    validate_pv(request, device);
    const std::size_t blocks = launch_blocks(request.shape, true);
    sdpa_pv_kernel<<<dim3(static_cast<unsigned int>(blocks)), dim3(kThreads),
                     0, stream>>>(request);
    require_cuda(cudaGetLastError(), "CUDA SDPA PV kernel launch");
}

}  // namespace iom::cuda_detail
