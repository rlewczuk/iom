#include <doctest/doctest.h>

#include <cuda_runtime.h>

#include <bit>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda/sdpa_matrix.hpp"

namespace {

constexpr std::uint64_t kTile = 16;

[[nodiscard]] std::uint64_t pad16(std::uint64_t value) {
    return (value + kTile - 1) / kTile * kTile;
}

[[nodiscard]] std::uint64_t plane_slot(
        std::uint64_t plane, std::uint64_t row, std::uint64_t column,
        std::uint64_t rows, std::uint64_t columns) {
    const std::uint64_t tile_rows = pad16(rows) / kTile;
    const std::uint64_t tile_columns = pad16(columns) / kTile;
    return (plane * tile_rows * tile_columns
                    + (row / kTile) * tile_columns + column / kTile)
            * kTile * kTile
            + (row % kTile) * kTile + column % kTile;
}

[[nodiscard]] std::uint16_t bf16_from_float(float value) {
    if (value == 0.0F) {
        return 0;
    }
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounded =
            bits + 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

[[nodiscard]] float bf16_to_float_daz(std::uint16_t bits) {
    if ((bits & 0x7f80u) == 0 && (bits & 0x007fu) != 0) {
        bits = static_cast<std::uint16_t>(bits & 0x8000u);
    }
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

void require_cuda(cudaError_t status, const char* operation) {
    REQUIRE_MESSAGE(
            status == cudaSuccess,
            operation << ": " << cudaGetErrorName(status) << ": "
                      << cudaGetErrorString(status));
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        require_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)cudaFree(pointer_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] void* get() const noexcept { return pointer_; }

private:
    void* pointer_ = nullptr;
};

void copy_to_device(
        void* destination, const void* source, std::size_t bytes) {
    require_cuda(
            cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy host to device");
}

void copy_from_device(
        void* destination, const void* source, std::size_t bytes) {
    require_cuda(
            cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host");
}

struct Case {
    std::size_t R;
    std::size_t a;
    std::size_t L;
};

}  // namespace

TEST_CASE("CUDA SDPA native matrix stages cover decode and prefill rows") {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    REQUIRE(iom::cuda_detail::sdpa_matrix_wmma_supported(device));

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags");

    constexpr std::uint64_t planes = 2;
    constexpr std::uint64_t Hq = 4;
    constexpr std::uint64_t Hkv = 2;
    constexpr std::uint64_t C = 24;
    constexpr std::uint64_t D = 7;
    constexpr std::uint64_t grouping = Hq / Hkv;
    constexpr std::uint64_t q_plane_stride = Hq * 2;
    constexpr std::uint64_t kv_plane_stride = Hkv * 2;
    constexpr std::uint64_t output_plane_stride = 2;
    constexpr std::uint64_t plane_offset = 1;

    for (const Case fixture : {
                 Case{1, 2, 19}, Case{15, 0, 15},
                 Case{16, 0, 16}, Case{17, 0, 19}}) {
        const std::uint64_t R = fixture.R;
        const std::uint64_t L = fixture.L;
        const std::uint64_t Rp = pad16(R);
        const std::uint64_t Lp = pad16(L);
        const std::uint64_t Dp = pad16(D);
        const std::uint64_t output_width = Hq * D;

        const std::uint64_t q_owner_planes =
                plane_offset + (planes - 1) * q_plane_stride
                + (Hq - 1) + 1;
        const std::uint64_t kv_owner_planes =
                plane_offset + (planes - 1) * kv_plane_stride
                + (Hkv - 1) + 1;
        const std::uint64_t out_owner_planes =
                plane_offset + (planes - 1) * output_plane_stride + 1;

        std::vector<std::uint16_t> q_host(
                q_owner_planes * pad16(R) * Dp, 0);
        std::vector<std::uint16_t> k_host(
                kv_owner_planes * pad16(C) * Dp, 0);
        std::vector<std::uint16_t> v_host(
                kv_owner_planes * pad16(C) * Dp, 0x7fc1u);
        std::vector<std::uint16_t> out_host(
                out_owner_planes * pad16(R) * pad16(output_width), 0x3555u);

        auto q_value = [&](std::uint64_t b, std::uint64_t h,
                           std::uint64_t r, std::uint64_t d) {
            const float value =
                    0.125F * static_cast<float>((b + 1) * (h + 1))
                    + 0.03125F * static_cast<float>(r + 1)
                    - 0.015625F * static_cast<float>(d + 1);
            return value;
        };
        auto k_value = [&](std::uint64_t b, std::uint64_t h,
                           std::uint64_t t, std::uint64_t d) {
            return 0.0625F * static_cast<float>((b + 1) * (h + 2))
                    - 0.0234375F * static_cast<float>(t + 1)
                    + 0.0078125F * static_cast<float>(d + 1);
        };
        auto v_value = [&](std::uint64_t b, std::uint64_t h,
                           std::uint64_t t, std::uint64_t d) {
            return 0.09375F * static_cast<float>((b + 1) * (h + 1))
                    + 0.015625F * static_cast<float>(t + 1)
                    - 0.01171875F * static_cast<float>(d + 1);
        };

        for (std::uint64_t b = 0; b < planes; ++b) {
            for (std::uint64_t h = 0; h < Hq; ++h) {
                for (std::uint64_t r = 0; r < R; ++r) {
                    for (std::uint64_t d = 0; d < D; ++d) {
                        q_host[plane_slot(
                                       plane_offset + b * q_plane_stride + h,
                                       r, d, R, D)] = bf16_from_float(
                                q_value(b, h, r, d));
                    }
                }
            }
            for (std::uint64_t h = 0; h < Hkv; ++h) {
                for (std::uint64_t t = 0; t < C; ++t) {
                    for (std::uint64_t d = 0; d < D; ++d) {
                        k_host[plane_slot(
                                       plane_offset + b * kv_plane_stride + h,
                                       t, d, C, D)] = bf16_from_float(
                                k_value(b, h, t, d));
                        if (t < L) {
                            v_host[plane_slot(
                                           plane_offset + b * kv_plane_stride
                                                   + h,
                                           t, d, C, D)] = bf16_from_float(
                                    v_value(b, h, t, d));
                        }
                    }
                }
            }
        }
        // An exact BF16 subnormal in a participating Q cell is device-local
        // DAZ evidence; it must contribute zero to every formed score.
        q_host[plane_slot(plane_offset, 0, 0, R, D)] = 0x0001u;

        const std::size_t score_elements =
                planes * Hq * Rp * Lp;
        std::vector<float> scores_host(score_elements, -12345.0F);
        const std::size_t probability_elements = score_elements;
        std::vector<std::uint16_t> probability_host(
                probability_elements, 0x0001u);
        for (std::uint64_t b = 0; b < planes; ++b) {
            for (std::uint64_t h = 0; h < Hq; ++h) {
                for (std::uint64_t r = 0; r < R; ++r) {
                    for (std::uint64_t t = 0; t < L; ++t) {
                        const std::size_t index =
                                (((b * Hq + h) * Rp + r) * Lp) + t;
                        probability_host[index] = bf16_from_float(
                                0.03125F * static_cast<float>((t % 5) + 1));
                    }
                }
            }
        }

        DeviceBuffer q_device(q_host.size() * sizeof(std::uint16_t));
        DeviceBuffer k_device(k_host.size() * sizeof(std::uint16_t));
        DeviceBuffer v_device(v_host.size() * sizeof(std::uint16_t));
        DeviceBuffer output_device(out_host.size() * sizeof(std::uint16_t));
        DeviceBuffer score_device(scores_host.size() * sizeof(float));
        DeviceBuffer probability_device(
                probability_host.size() * sizeof(std::uint16_t));
        copy_to_device(q_device.get(), q_host.data(),
                       q_host.size() * sizeof(std::uint16_t));
        copy_to_device(k_device.get(), k_host.data(),
                       k_host.size() * sizeof(std::uint16_t));
        copy_to_device(v_device.get(), v_host.data(),
                       v_host.size() * sizeof(std::uint16_t));
        copy_to_device(output_device.get(), out_host.data(),
                       out_host.size() * sizeof(std::uint16_t));
        copy_to_device(score_device.get(), scores_host.data(),
                       scores_host.size() * sizeof(float));
        copy_to_device(probability_device.get(), probability_host.data(),
                       probability_host.size() * sizeof(std::uint16_t));

        iom::cuda_detail::SdpaMatrixShape shape;
        shape.leading_rank = 1;
        shape.leading_dimensions[0] = planes;
        shape.planes = planes;
        shape.Hq = Hq;
        shape.Hkv = Hkv;
        shape.R = R;
        shape.C = C;
        shape.D = D;
        shape.a = fixture.a;
        shape.L = L;
        shape.grouping = grouping;
        shape.output_width = output_width;
        shape.Rp = Rp;
        shape.Lp = Lp;
        shape.Dp = Dp;
        shape.device_ordinal = device;

        iom::cuda_detail::SdpaMatrixOperand q_operand;
        q_operand.data = static_cast<const unsigned char*>(q_device.get());
        q_operand.plane_offset = plane_offset;
        q_operand.plane_strides[0] = q_plane_stride;
        q_operand.head_stride = 1;
        iom::cuda_detail::SdpaMatrixOperand k_operand;
        k_operand.data = static_cast<const unsigned char*>(k_device.get());
        k_operand.plane_offset = plane_offset;
        k_operand.plane_strides[0] = kv_plane_stride;
        k_operand.head_stride = 1;

        iom::cuda_detail::SdpaQkRequest qk;
        qk.shape = shape;
        qk.q = q_operand;
        qk.k = k_operand;
        qk.scores = static_cast<float*>(score_device.get());
        iom::cuda_detail::launch_sdpa_qk(stream, qk);
        require_cuda(cudaStreamSynchronize(stream), "QK synchronize");
        copy_from_device(scores_host.data(), score_device.get(),
                         scores_host.size() * sizeof(float));

        for (std::uint64_t b = 0; b < planes; ++b) {
            for (std::uint64_t h = 0; h < Hq; ++h) {
                const std::uint64_t grouped = h / grouping;
                for (std::uint64_t r = 0; r < R; ++r) {
                    for (std::uint64_t t = 0; t < L; ++t) {
                        float expected = 0.0F;
                        for (std::uint64_t d = 0; d < D; ++d) {
                            expected += bf16_to_float_daz(q_host[plane_slot(
                                    plane_offset + b * q_plane_stride + h,
                                    r, d, R, D)])
                                    * bf16_to_float_daz(k_host[plane_slot(
                                            plane_offset
                                                    + b * kv_plane_stride
                                                    + grouped,
                                            t, d, C, D)]);
                        }
                        const std::size_t index =
                                (((b * Hq + h) * Rp + r) * Lp) + t;
                        CHECK(scores_host[index]
                              == doctest::Approx(expected).epsilon(1e-3));
                    }
                }
            }
        }
        for (std::size_t index = 0; index < score_elements; ++index) {
            const std::uint64_t rest = index % Lp;
            const std::uint64_t row = (index / Lp) % Rp;
            if (rest >= L || row >= R) {
                CHECK(scores_host[index] == -12345.0F);
            }
        }

        iom::cuda_detail::SdpaPvRequest pv;
        pv.shape = shape;
        pv.probability = static_cast<const unsigned char*>(
                probability_device.get());
        pv.v = iom::cuda_detail::SdpaMatrixOperand{
                static_cast<const unsigned char*>(v_device.get()), plane_offset,
                {kv_plane_stride, 0, 0, 0, 0}, 1};
        pv.out = iom::cuda_detail::SdpaMatrixOutput{
                static_cast<unsigned char*>(output_device.get()), plane_offset,
                {output_plane_stride, 0, 0, 0, 0}};
        iom::cuda_detail::launch_sdpa_pv(stream, pv);
        require_cuda(cudaStreamSynchronize(stream), "PV synchronize");
        copy_from_device(out_host.data(), output_device.get(),
                         out_host.size() * sizeof(std::uint16_t));

        for (std::uint64_t b = 0; b < planes; ++b) {
            for (std::uint64_t h = 0; h < Hq; ++h) {
                const std::uint64_t grouped = h / grouping;
                for (std::uint64_t r = 0; r < R; ++r) {
                    for (std::uint64_t d = 0; d < D; ++d) {
                        float expected = 0.0F;
                        for (std::uint64_t t = 0; t < L; ++t) {
                            const std::size_t p_index =
                                    (((b * Hq + h) * Rp + r) * Lp) + t;
                            expected += bf16_to_float_daz(
                                    probability_host[p_index])
                                    * bf16_to_float_daz(v_host[plane_slot(
                                            plane_offset
                                                    + b * kv_plane_stride
                                                    + grouped,
                                            t, d, C, D)]);
                        }
                        const std::uint16_t expected_bits =
                                bf16_from_float(expected);
                        const std::uint64_t out_plane =
                                plane_offset + b * output_plane_stride;
                        const std::uint16_t observed = out_host[plane_slot(
                                out_plane, r, h * D + d, R, output_width)];
                        CHECK(observed == expected_bits);
                    }
                }
            }
        }
        std::printf(
                "cuda-sdpa-matrix-record R=%zu L=%zu D=%zu Hq=%zu Hkv=%zu "
                "kernel=sdpa_qk_kernel,sdpa_pv_kernel device=%d\n",
                fixture.R, fixture.L, static_cast<std::size_t>(D),
                static_cast<std::size_t>(Hq), static_cast<std::size_t>(Hkv),
                device);
    }
    require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
