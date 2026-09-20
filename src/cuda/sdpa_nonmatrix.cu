#include "sdpa_nonmatrix.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace iom::cuda_detail {
namespace {

constexpr std::uint64_t kTile = 16;
constexpr std::uint64_t kTileSlots = kTile * kTile;
constexpr unsigned int kThreads = 256;
constexpr unsigned int kMaxBlocks = 65535;

[[nodiscard]] std::size_t checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::size_t checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (left != 0
            && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::uint64_t checked_add_u64(
        std::uint64_t left, std::uint64_t right, const char* message) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t checked_mul_u64(
        std::uint64_t left, std::uint64_t right, const char* message) {
    if (left != 0
            && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t to_size(
        std::uint64_t value, const char* message) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(message);
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::uint64_t padded16(
        std::uint64_t value, const char* message) {
    return checked_mul_u64(
            (checked_add_u64(value, kTile - 1, message) / kTile), kTile,
            message);
}

[[nodiscard]] std::size_t aligned32(
        std::size_t bytes, const char* message) {
    return checked_mul(
            checked_add(bytes, 31, message) / 32, 32, message);
}

struct Geometry {
    std::size_t score_bytes = 0;
    std::size_t probability_bytes = 0;
    std::size_t output_width = 0;
};

[[nodiscard]] Geometry validate_geometry(
        const SdpaNonmatrixRequest& request) {
    if (request.scores == nullptr || request.probability == nullptr
            || request.output.data == nullptr) {
        throw std::invalid_argument(
                "CUDA SDPA nonmatrix pointers must be non-null");
    }
    if (request.data_type != DataType::BF16
            || request.quantization != QuantizationFormat::NONE) {
        throw std::invalid_argument(
                "CUDA SDPA nonmatrix stages require BF16 and NONE");
    }
    if (request.leading_rank > kSdpaNonmatrixMaxLeadingRank) {
        throw std::invalid_argument(
                "CUDA SDPA nonmatrix leading rank exceeds fixed capacity");
    }
    if (request.Hq == 0 || request.Hkv == 0 || request.R == 0
            || request.C == 0 || request.D == 0 || request.L == 0
            || request.planes == 0) {
        throw std::invalid_argument(
                "CUDA SDPA nonmatrix dimensions must be nonzero");
    }
    if (request.L > request.C || request.a >= request.C
            || request.R > request.C - request.a) {
        throw std::invalid_argument(
                "CUDA SDPA nonmatrix causal dimensions are invalid");
    }
    if (request.Hq % request.Hkv != 0
            || request.grouping != request.Hq / request.Hkv) {
        throw std::invalid_argument(
                "CUDA SDPA nonmatrix GQA dimensions are invalid");
    }

    std::uint64_t planes = 1;
    for (std::size_t axis = 0; axis < request.leading_rank; ++axis) {
        if (request.leading_dimensions[axis] == 0) {
            throw std::invalid_argument(
                    "CUDA SDPA nonmatrix leading dimensions must be nonzero");
        }
        planes = checked_mul_u64(
                planes, request.leading_dimensions[axis],
                "CUDA SDPA nonmatrix plane count overflows");
    }
    if (planes != request.planes) {
        throw std::invalid_argument(
                "CUDA SDPA nonmatrix plane count does not match leading shape");
    }

    const std::uint64_t expected_rp = padded16(
            request.R, "CUDA SDPA nonmatrix Rp overflows");
    const std::uint64_t expected_lp = padded16(
            request.L, "CUDA SDPA nonmatrix Lp overflows");
    if (request.Rp != expected_rp || request.Lp != expected_lp) {
        throw std::invalid_argument(
                "CUDA SDPA nonmatrix physical extents do not match pad16");
    }

    const std::uint64_t output_width_u64 = checked_mul_u64(
            request.Hq, request.D,
            "CUDA SDPA nonmatrix output width overflows");
    const std::size_t output_width = to_size(
            output_width_u64,
            "CUDA SDPA nonmatrix output width exceeds host size");

    const std::size_t planes_size = to_size(
            request.planes, "CUDA SDPA nonmatrix plane count exceeds host size");
    const std::size_t heads_size = to_size(
            request.Hq, "CUDA SDPA nonmatrix head count exceeds host size");
    const std::size_t rows_size = to_size(
            request.R, "CUDA SDPA nonmatrix row count exceeds host size");
    const std::size_t rp_size = to_size(
            request.Rp, "CUDA SDPA nonmatrix Rp exceeds host size");
    const std::size_t lp_size = to_size(
            request.Lp, "CUDA SDPA nonmatrix Lp exceeds host size");
    const std::size_t score_elements = checked_mul(
            checked_mul(
                    checked_mul(planes_size, heads_size,
                               "CUDA SDPA score extent overflows"),
                    rp_size, "CUDA SDPA score extent overflows"),
            lp_size, "CUDA SDPA score extent overflows");
    const std::size_t probability_elements = score_elements;
    const std::size_t score_bytes = aligned32(
            checked_mul(score_elements, sizeof(float),
                        "CUDA SDPA score bytes overflow"),
            "CUDA SDPA score alignment overflows");
    const std::size_t probability_bytes = aligned32(
            checked_mul(probability_elements, sizeof(std::uint16_t),
                        "CUDA SDPA probability bytes overflow"),
            "CUDA SDPA probability alignment overflows");

    const std::uint64_t row_tiles =
            (request.R + kTile - 1) / kTile;
    const std::uint64_t column_tiles =
            (output_width_u64 + kTile - 1) / kTile;
    const std::uint64_t plane_slots = checked_mul_u64(
            checked_mul_u64(row_tiles, column_tiles,
                            "CUDA SDPA output tile count overflows"),
            kTileSlots, "CUDA SDPA output plane slots overflow");

    std::uint64_t last_plane = request.output.plane_offset;
    for (std::size_t axis = 0; axis < request.leading_rank; ++axis) {
        if (request.leading_dimensions[axis] > 1
                && request.output.plane_strides[axis] == 0) {
            throw std::invalid_argument(
                    "CUDA SDPA output stride must advance each leading plane");
        }
        last_plane = checked_add_u64(
                last_plane,
                checked_mul_u64(
                        request.leading_dimensions[axis] - 1,
                        request.output.plane_strides[axis],
                        "CUDA SDPA output plane range overflows"),
                "CUDA SDPA output plane range overflows");
    }
    const std::uint64_t last_slot = checked_add_u64(
            checked_mul_u64(
                    last_plane, plane_slots,
                    "CUDA SDPA output storage range overflows"),
            checked_add_u64(
                    checked_mul_u64(
                            (request.R - 1) / kTile, column_tiles,
                            "CUDA SDPA output storage range overflows"),
                    checked_add_u64(
                            checked_mul_u64(
                                    (output_width_u64 - 1) / kTile,
                                    kTileSlots,
                                    "CUDA SDPA output storage range overflows"),
                            ((request.R - 1) % kTile) * kTile
                                    + ((output_width_u64 - 1) % kTile),
                            "CUDA SDPA output storage range overflows"),
                    "CUDA SDPA output storage range overflows"),
            "CUDA SDPA output storage range overflows");
    const std::uint64_t required_output_bytes = checked_mul_u64(
            checked_add_u64(
                    last_slot, 1,
                    "CUDA SDPA output storage range overflows"),
            sizeof(std::uint16_t), "CUDA SDPA output byte range overflows");
    if (request.output.storage_bytes < required_output_bytes) {
        throw std::invalid_argument(
                "CUDA SDPA output storage is smaller than its logical view");
    }

    return Geometry{score_bytes, probability_bytes, output_width};
}

[[nodiscard]] std::uintptr_t checked_end(
        const void* pointer, std::size_t bytes, const char* message) {
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error(message);
    }
    return begin + bytes;
}

void validate_pointer(
        const void* pointer, std::size_t bytes, std::uint32_t ordinal,
        const char* name) {
    if ((reinterpret_cast<std::uintptr_t>(pointer) & 31u) != 0) {
        throw std::invalid_argument(
                std::string(name) + " must be 32-byte aligned");
    }
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::invalid_argument(
                std::string(name) + " must point to CUDA device storage");
    }
#if CUDART_VERSION >= 10000
    if (attributes.type != cudaMemoryTypeDevice
            || attributes.device != static_cast<int>(ordinal)) {
#else
    if (attributes.memoryType != cudaMemoryTypeDevice
            || attributes.device != static_cast<int>(ordinal)) {
#endif
        throw std::invalid_argument(
                std::string(name) + " is not storage on the requested CUDA device");
    }
    (void)checked_end(pointer, bytes, "CUDA SDPA pointer range overflows");
}

void validate_nonoverlap(
        const void* first, std::size_t first_bytes,
        const void* second, std::size_t second_bytes,
        const char* message) {
    const std::uintptr_t first_begin =
            reinterpret_cast<std::uintptr_t>(first);
    const std::uintptr_t first_end =
            checked_end(first, first_bytes, message);
    const std::uintptr_t second_begin =
            reinterpret_cast<std::uintptr_t>(second);
    const std::uintptr_t second_end =
            checked_end(second, second_bytes, message);
    if (first_begin < second_end && second_begin < first_end) {
        throw std::invalid_argument(message);
    }
}

[[nodiscard]] unsigned int launch_blocks(std::uint64_t work_items) {
    const std::uint64_t blocks =
            (work_items + kThreads - 1) / kThreads;
    return static_cast<unsigned int>(
            blocks == 0 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks));
}

void check_launch(const char* operation) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
                std::string(operation) + " failed with "
                + cudaGetErrorName(status) + ": "
                + cudaGetErrorString(status));
    }
}

__device__ std::uint64_t output_plane(
        const SdpaNonmatrixRequest& request, std::uint64_t plane) {
    std::uint64_t rest = plane;
    std::uint64_t result = request.output.plane_offset;
    for (std::uint32_t axis = request.leading_rank; axis-- > 0;) {
        const std::uint64_t coordinate =
                rest % request.leading_dimensions[axis];
        rest /= request.leading_dimensions[axis];
        result += coordinate * request.output.plane_strides[axis];
    }
    return result;
}

__device__ std::uint64_t tiled_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_columns = (columns + kTile - 1) / kTile;
    const std::uint64_t tile_index =
            plane * ((rows + kTile - 1) / kTile) * tile_columns
            + (row / kTile) * tile_columns + column / kTile;
    return tile_index * kTileSlots
            + (row % kTile) * kTile + column % kTile;
}

__device__ std::uint16_t bf16_rne(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t exponent = bits & 0x7f800000u;
    const std::uint32_t fraction = bits & 0x007fffffu;
    if (exponent == 0x7f800000u) {
        if (fraction == 0) {
            return static_cast<std::uint16_t>(bits >> 16);
        }
        // Payload and sign are unspecified by the contract; this is a quiet
        // positive NaN and never rounds a narrow NaN into infinity.
        return static_cast<std::uint16_t>(0x7fc0u);
    }
    if ((bits & 0x7fffffffu) == 0) {
        return 0;
    }
    const std::uint32_t rounded =
            bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

__device__ std::uint64_t score_index(
        const SdpaNonmatrixRequest& request, std::uint64_t plane,
        std::uint64_t head, std::uint64_t row, std::uint64_t token) {
    return (((plane * request.Hq + head) * request.Rp + row)
            * request.Lp + token);
}

__global__ void scale_mask_kernel(SdpaNonmatrixRequest request) {
    const std::uint64_t total =
            request.planes * request.Hq * request.R * request.L;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    const float scale = 1.0f / sqrtf(static_cast<float>(request.D));
    for (std::uint64_t item =
                 static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
         item < total; item += stride) {
        std::uint64_t rest = item;
        const std::uint64_t token = rest % request.L;
        rest /= request.L;
        const std::uint64_t row = rest % request.R;
        rest /= request.R;
        const std::uint64_t head = rest % request.Hq;
        const std::uint64_t plane = rest / request.Hq;
        if (token <= request.a + row) {
            const std::uint64_t index =
                    score_index(request, plane, head, row, token);
            const float scaled = request.scores[index] * scale;
            request.scores[index] = scaled == 0.0f ? 0.0f : scaled;
        }
    }
}

__global__ void softmax_kernel(SdpaNonmatrixRequest request) {
    const std::uint64_t rows = request.planes * request.Hq * request.R;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    const std::uint64_t probability_width = request.Lp;
    auto* probability = reinterpret_cast<std::uint16_t*>(
            request.probability);
    for (std::uint64_t item =
                 static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
         item < rows; item += stride) {
        std::uint64_t rest = item;
        const std::uint64_t row = rest % request.R;
        rest /= request.R;
        const std::uint64_t head = rest % request.Hq;
        const std::uint64_t plane = rest / request.Hq;
        const std::uint64_t visible_end =
                request.L < request.a + row + 1
                ? request.L : request.a + row + 1;

        bool has_nan = false;
        bool all_negative_infinity = true;
        std::uint64_t positive_infinities = 0;
        float maximum = -__int_as_float(0x7f800000u);
        for (std::uint64_t token = 0; token < visible_end; ++token) {
            const float score = request.scores[
                    score_index(request, plane, head, row, token)];
            if (isnan(score)) {
                has_nan = true;
                continue;
            }
            if (isinf(score) && score > 0.0f) {
                ++positive_infinities;
                all_negative_infinity = false;
                continue;
            }
            if (isinf(score) && score < 0.0f) {
                continue;
            }
            all_negative_infinity = false;
            if (score > maximum) {
                maximum = score;
            }
        }

        if (has_nan || all_negative_infinity) {
            for (std::uint64_t token = 0; token < request.L; ++token) {
                const std::uint64_t index =
                        ((plane * request.Hq + head) * request.Rp + row)
                        * probability_width + token;
                probability[index] = token < visible_end
                        ? bf16_rne(__int_as_float(0x7fc00000u))
                        : static_cast<std::uint16_t>(0);
            }
            continue;
        }

        if (positive_infinities != 0) {
            const float equal_probability =
                    1.0f / static_cast<float>(positive_infinities);
            for (std::uint64_t token = 0; token < request.L; ++token) {
                const std::uint64_t index =
                        ((plane * request.Hq + head) * request.Rp + row)
                        * probability_width + token;
                bool positive_infinity = false;
                if (token < visible_end) {
                    const float score = request.scores[
                            score_index(request, plane, head, row, token)];
                    positive_infinity = isinf(score) && score > 0.0f;
                }
                probability[index] = positive_infinity
                        ? bf16_rne(equal_probability)
                        : static_cast<std::uint16_t>(0);
            }
            continue;
        }

        float sum = 0.0f;
        for (std::uint64_t token = 0; token < visible_end; ++token) {
            const float score = request.scores[
                    score_index(request, plane, head, row, token)];
            if (isinf(score) && score < 0.0f) {
                continue;
            }
            sum += expf(score - maximum);
        }
        for (std::uint64_t token = 0; token < request.L; ++token) {
            const std::uint64_t index =
                    ((plane * request.Hq + head) * request.Rp + row)
                    * probability_width + token;
            if (token >= visible_end) {
                probability[index] = 0;
                continue;
            }
            const float score = request.scores[
                    score_index(request, plane, head, row, token)];
            if (isinf(score) && score < 0.0f) {
                probability[index] = 0;
                continue;
            }
            probability[index] = bf16_rne(expf(score - maximum) / sum);
        }
    }
}

__global__ void canonicalize_output_kernel(SdpaNonmatrixRequest request) {
    const std::uint64_t width = request.Hq * request.D;
    const std::uint64_t total = request.planes * request.R * width;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    auto* output = reinterpret_cast<std::uint16_t*>(request.output.data);
    for (std::uint64_t item =
                 static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
         item < total; item += stride) {
        std::uint64_t rest = item;
        const std::uint64_t column = rest % width;
        rest /= width;
        const std::uint64_t row = rest % request.R;
        const std::uint64_t plane = rest / request.R;
        const std::uint64_t slot = tiled_slot(
                output_plane(request, plane), row, column,
                request.R, width);
        std::uint16_t bits = output[slot];
        const std::uint16_t exponent = bits & 0x7f80u;
        const std::uint16_t fraction = bits & 0x007fu;
        if ((bits & 0x7fffu) == 0) {
            bits = 0;
        } else if (exponent == 0x7f80u && fraction != 0) {
            bits = 0x7fc0u;
        }
        output[slot] = bits;
    }
}

}  // namespace

void validate_sdpa_nonmatrix_request(
        const SdpaNonmatrixRequest& request) {
    const Geometry geometry = validate_geometry(request);
    if (request.device_ordinal
            > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
                "CUDA SDPA device ordinal exceeds runtime range");
    }
    int current_device = -1;
    const cudaError_t device_status = cudaGetDevice(&current_device);
    if (device_status != cudaSuccess) {
        throw std::runtime_error(
                std::string("cudaGetDevice failed with ")
                + cudaGetErrorName(device_status) + ": "
                + cudaGetErrorString(device_status));
    }
    if (current_device != static_cast<int>(request.device_ordinal)) {
        throw std::invalid_argument(
                "CUDA SDPA request device differs from current CUDA device");
    }

    validate_pointer(
            request.scores, geometry.score_bytes, request.device_ordinal,
            "CUDA SDPA scores");
    validate_pointer(
            request.probability, geometry.probability_bytes,
            request.device_ordinal, "CUDA SDPA probability");
    validate_pointer(
            request.output.data,
            to_size(request.output.storage_bytes,
                    "CUDA SDPA output storage exceeds host size"),
            request.device_ordinal, "CUDA SDPA output");
    validate_nonoverlap(
            request.scores, geometry.score_bytes,
            request.probability, geometry.probability_bytes,
            "CUDA SDPA scores and probability ranges overlap");
    validate_nonoverlap(
            request.scores, geometry.score_bytes,
            request.output.data,
            to_size(request.output.storage_bytes,
                     "CUDA SDPA output storage exceeds host size"),
            "CUDA SDPA scores overlap output storage");
    validate_nonoverlap(
            request.probability, geometry.probability_bytes,
            request.output.data,
            to_size(request.output.storage_bytes,
                     "CUDA SDPA output storage exceeds host size"),
            "CUDA SDPA probability overlaps output storage");
}

void launch_sdpa_scale_mask(
        cudaStream_t stream, const SdpaNonmatrixRequest& request) {
    validate_sdpa_nonmatrix_request(request);
    const std::uint64_t total =
            request.planes * request.Hq * request.R * request.L;
    scale_mask_kernel<<<launch_blocks(total), kThreads, 0, stream>>>(request);
    check_launch("CUDA SDPA scale/mask kernel launch");
}

void launch_sdpa_softmax(
        cudaStream_t stream, const SdpaNonmatrixRequest& request) {
    validate_sdpa_nonmatrix_request(request);
    const std::uint64_t rows = request.planes * request.Hq * request.R;
    softmax_kernel<<<launch_blocks(rows), kThreads, 0, stream>>>(request);
    check_launch("CUDA SDPA softmax kernel launch");
}

void launch_sdpa_canonicalize_output(
        cudaStream_t stream, const SdpaNonmatrixRequest& request) {
    validate_sdpa_nonmatrix_request(request);
    const std::uint64_t total =
            request.planes * request.R * request.Hq * request.D;
    canonicalize_output_kernel<<<launch_blocks(total), kThreads, 0, stream>>>(
            request);
    check_launch("CUDA SDPA output canonicalization kernel launch");
}

}  // namespace iom::cuda_detail
