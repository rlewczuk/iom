#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda/sdpa_nonmatrix.hpp"
#include "iom/tensor.hpp"

namespace {

struct DeviceBuffer {
    void* data = nullptr;
    std::size_t bytes = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t size) : bytes(size) {
        REQUIRE(cudaMalloc(&data, bytes) == cudaSuccess);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (data != nullptr) {
            (void)cudaFree(data);
        }
    }
};

struct DeviceStream {
    cudaStream_t value = nullptr;
    DeviceStream() { REQUIRE(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking) == cudaSuccess); }
    DeviceStream(const DeviceStream&) = delete;
    DeviceStream& operator=(const DeviceStream&) = delete;
    ~DeviceStream() {
        if (value != nullptr) {
            (void)cudaStreamDestroy(value);
        }
    }
};

[[nodiscard]] std::size_t aligned32(std::size_t bytes) {
    return ((bytes + 31) / 32) * 32;
}

[[nodiscard]] std::size_t score_index(
        std::size_t plane, std::size_t head, std::size_t row,
        std::size_t token, std::size_t Hq, std::size_t Rp,
        std::size_t Lp) {
    return (((plane * Hq + head) * Rp + row) * Lp + token);
}

[[nodiscard]] std::size_t probability_index(
        std::size_t plane, std::size_t head, std::size_t row,
        std::size_t token, std::size_t Hq, std::size_t Rp,
        std::size_t Lp) {
    return score_index(plane, head, row, token, Hq, Rp, Lp);
}


[[nodiscard]] float bf16_value(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float result = 0.0f;
    std::memcpy(&result, &wide, sizeof(result));
    return result;
}

[[nodiscard]] std::size_t output_slot(
        std::size_t planes, std::size_t rows, std::size_t width,
        std::size_t plane, std::size_t row, std::size_t column) {
    const iom::TensorSpec spec{
            iom::TensorShape{{planes, rows, width}}, iom::DataType::BF16};
    return iom::detail::standard_plane_slot(spec, plane, row, column);
}

[[nodiscard]] iom::cuda_detail::SdpaNonmatrixRequest make_request(
        float* scores, unsigned char* probability, unsigned char* output,
        std::size_t output_bytes) {
    iom::cuda_detail::SdpaNonmatrixRequest request;
    request.scores = scores;
    request.probability = probability;
    request.output.data = output;
    request.output.plane_strides[0] = 1;
    request.output.storage_bytes = output_bytes;
    request.leading_rank = 1;
    request.leading_dimensions[0] = 2;
    request.planes = 2;
    request.Hq = 4;
    request.Hkv = 2;
    request.R = 17;
    request.C = 24;
    request.D = 8;
    request.a = 3;
    request.L = 19;
    request.grouping = 2;
    request.Rp = 32;
    request.Lp = 32;
    request.device_ordinal = 0;
    return request;
}

void run_boundary_case(std::size_t rows) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t Hq = 2;
    constexpr std::size_t Hkv = 1;
    constexpr std::size_t C = 32;
    constexpr std::size_t D = 3;
    constexpr std::size_t a = 1;
    constexpr std::size_t L = 17;
    constexpr std::size_t width = Hq * D;
    const std::size_t Rp = ((rows + 15) / 16) * 16;
    constexpr std::size_t Lp = 32;
    const std::size_t score_elements = planes * Hq * Rp * Lp;
    const std::size_t score_bytes = aligned32(score_elements * sizeof(float));
    const std::size_t probability_bytes =
            aligned32(score_elements * sizeof(std::uint16_t));
    const std::size_t padded_width = ((width + 15) / 16) * 16;
    const std::size_t output_bytes =
            planes * Rp * padded_width * sizeof(std::uint16_t);
    std::vector<float> scores(score_bytes / sizeof(float), 777.0f);
    for (std::size_t head = 0; head < Hq; ++head) {
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t visible = std::min(L, a + row + 1);
            for (std::size_t token = 0; token < visible; ++token) {
                scores[score_index(0, head, row, token, Hq, Rp, Lp)] = 0.0f;
            }
        }
    }
    std::vector<std::uint16_t> probabilities(
            probability_bytes / sizeof(std::uint16_t), 0xa5a5);
    std::vector<std::uint16_t> output(
            output_bytes / sizeof(std::uint16_t), 0x3555);
    output[output_slot(planes, Rp, width, 0, 0, 0)] = 0x8000;

    DeviceBuffer score_device(score_bytes);
    DeviceBuffer probability_device(probability_bytes);
    DeviceBuffer output_device(output_bytes);
    REQUIRE(cudaMemcpy(
                    score_device.data, scores.data(), score_bytes,
                    cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    probability_device.data, probabilities.data(),
                    probability_bytes, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    output_device.data, output.data(), output_bytes,
                    cudaMemcpyHostToDevice)
            == cudaSuccess);

    iom::cuda_detail::SdpaNonmatrixRequest request;
    request.scores = static_cast<float*>(score_device.data);
    request.probability =
            static_cast<unsigned char*>(probability_device.data);
    request.output.data = static_cast<unsigned char*>(output_device.data);
    request.output.storage_bytes = output_bytes;
    request.planes = planes;
    request.Hq = Hq;
    request.Hkv = Hkv;
    request.R = rows;
    request.C = C;
    request.D = D;
    request.a = a;
    request.L = L;
    request.grouping = Hq / Hkv;
    request.Rp = Rp;
    request.Lp = Lp;
    request.device_ordinal = 0;

    DeviceStream stream;
    iom::cuda_detail::launch_sdpa_scale_mask(stream.value, request);
    iom::cuda_detail::launch_sdpa_softmax(stream.value, request);
    iom::cuda_detail::launch_sdpa_canonicalize_output(stream.value, request);
    REQUIRE(cudaStreamSynchronize(stream.value) == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    probabilities.data(), probability_device.data,
                    probability_bytes, cudaMemcpyDeviceToHost)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    output.data(), output_device.data, output_bytes,
                    cudaMemcpyDeviceToHost)
            == cudaSuccess);

    for (std::size_t head = 0; head < Hq; ++head) {
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t visible = std::min(L, a + row + 1);
            for (std::size_t token = 0; token < visible; ++token) {
                CHECK(probabilities[probability_index(
                              0, head, row, token, Hq, Rp, Lp)]
                      != 0);
            }
            for (std::size_t token = visible; token < L; ++token) {
                CHECK(probabilities[probability_index(
                              0, head, row, token, Hq, Rp, Lp)]
                      == 0);
            }
            CHECK(probabilities[probability_index(
                          0, head, row, L, Hq, Rp, Lp)]
                  == 0xa5a5);
        }
    }
    CHECK(output[output_slot(planes, Rp, width, 0, 0, 0)] == 0);
    if (rows < Rp) {
        CHECK(output[output_slot(planes, Rp, width, 0, rows, 0)] == 0x3555);
    }
}

void run_subnormal_case() {
    constexpr std::size_t planes = 2;
    constexpr std::size_t Hq = 1;
    constexpr std::size_t Hkv = 1;
    constexpr std::size_t R = 1;
    constexpr std::size_t C = 2;
    constexpr std::size_t D = 1;
    constexpr std::size_t a = 1;
    constexpr std::size_t L = 2;
    constexpr std::size_t Rp = 16;
    constexpr std::size_t Lp = 16;
    constexpr std::size_t score_elements = planes * Hq * Rp * Lp;
    const std::size_t score_bytes = aligned32(score_elements * sizeof(float));
    const std::size_t probability_bytes =
            aligned32(score_elements * sizeof(std::uint16_t));
    const std::size_t output_bytes =
            planes * Rp * 16 * sizeof(std::uint16_t);
    std::vector<float> scores(score_bytes / sizeof(float), 777.0f);
    scores[score_index(0, 0, 0, 0, Hq, Rp, Lp)] = 0.0f;
    scores[score_index(0, 0, 0, 1, Hq, Rp, Lp)] = -92.5f;
    scores[score_index(1, 0, 0, 0, Hq, Rp, Lp)] = 0.0f;
    scores[score_index(1, 0, 0, 1, Hq, Rp, Lp)] = -93.2f;
    std::vector<std::uint16_t> probabilities(
            probability_bytes / sizeof(std::uint16_t), 0xa5a5);
    std::vector<std::uint16_t> output(
            output_bytes / sizeof(std::uint16_t), 0x3555);
    DeviceBuffer score_device(score_bytes);
    DeviceBuffer probability_device(probability_bytes);
    DeviceBuffer output_device(output_bytes);
    REQUIRE(cudaMemcpy(
                    score_device.data, scores.data(), score_bytes,
                    cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    probability_device.data, probabilities.data(),
                    probability_bytes, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    output_device.data, output.data(), output_bytes,
                    cudaMemcpyHostToDevice)
            == cudaSuccess);
    iom::cuda_detail::SdpaNonmatrixRequest request;
    request.scores = static_cast<float*>(score_device.data);
    request.probability =
            static_cast<unsigned char*>(probability_device.data);
    request.output.data = static_cast<unsigned char*>(output_device.data);
    request.output.plane_strides[0] = 1;
    request.output.storage_bytes = output_bytes;
    request.leading_rank = 1;
    request.leading_dimensions[0] = planes;
    request.planes = planes;
    request.Hq = Hq;
    request.Hkv = Hkv;
    request.R = R;
    request.C = C;
    request.D = D;
    request.a = a;
    request.L = L;
    request.grouping = 1;
    request.Rp = Rp;
    request.Lp = Lp;
    request.device_ordinal = 0;
    DeviceStream stream;
    iom::cuda_detail::launch_sdpa_scale_mask(stream.value, request);
    iom::cuda_detail::launch_sdpa_softmax(stream.value, request);
    REQUIRE(cudaStreamSynchronize(stream.value) == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    probabilities.data(), probability_device.data,
                    probability_bytes, cudaMemcpyDeviceToHost)
            == cudaSuccess);
    CHECK(probabilities[probability_index(0, 0, 0, 1, Hq, Rp, Lp)] == 0x0001);
    CHECK(probabilities[probability_index(1, 0, 0, 1, Hq, Rp, Lp)] == 0x0000);
}

}  // namespace

TEST_CASE("CUDA SDPA nonmatrix stages honor logical rows and special policy") {
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    int device_count = 0;
    REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
    REQUIRE(device_count > 0);

    constexpr std::size_t P = 2;
    constexpr std::size_t Hq = 4;
    constexpr std::size_t R = 17;
    constexpr std::size_t D = 8;
    constexpr std::size_t width = Hq * D;
    constexpr std::size_t Rp = 32;
    constexpr std::size_t L = 19;
    constexpr std::size_t Lp = 32;
    constexpr std::size_t score_elements = P * Hq * Rp * Lp;
    const std::size_t score_bytes = aligned32(score_elements * sizeof(float));
    const std::size_t probability_bytes =
            aligned32(score_elements * sizeof(std::uint16_t));
    const std::size_t output_bytes = P * Rp * width * sizeof(std::uint16_t);

    std::vector<float> scores(score_bytes / sizeof(float), 12345.0f);
    auto set_score = [&](std::size_t plane, std::size_t head,
                         std::size_t row, std::size_t token, float value) {
        scores[score_index(plane, head, row, token, Hq, Rp, Lp)] = value;
    };
    for (std::size_t row = 0; row < R; ++row) {
        const std::size_t visible = std::min(L, std::size_t{3} + row + 1);
        for (std::size_t token = 0; token < visible; ++token) {
            set_score(0, 0, row, token, 0.0f);
        }
    }
    set_score(0, 0, 0, 0, 0.0f);
    set_score(0, 0, 0, 1, 1.0f);
    set_score(0, 0, 0, 2, 2.0f);
    set_score(0, 0, 0, 3, 3.0f);
    set_score(0, 0, 1, 0, std::numeric_limits<float>::infinity());
    set_score(0, 0, 1, 1, 0.0f);
    set_score(0, 0, 1, 2, std::numeric_limits<float>::infinity());
    for (std::size_t token = 0; token < 6; ++token) {
        set_score(0, 0, 2, token, -std::numeric_limits<float>::infinity());
    }
    set_score(0, 0, 3, 1, std::numeric_limits<float>::quiet_NaN());
    set_score(0, 0, 4, 0, -std::numeric_limits<float>::infinity());

    std::vector<std::uint16_t> probabilities(
            probability_bytes / sizeof(std::uint16_t), 0xa5a5);
    std::vector<std::uint16_t> output(output_bytes / sizeof(std::uint16_t), 0x3555);
    const auto slot = [&](std::size_t plane, std::size_t row,
                          std::size_t column) {
        return output_slot(P, Rp, width, plane, row, column);
    };
    output[slot(0, 0, 0)] = 0x8000;
    output[slot(0, 0, 1)] = 0x7fff;
    output[slot(0, 0, 2)] = 0x3f80;

    DeviceBuffer score_device(score_bytes);
    DeviceBuffer probability_device(probability_bytes);
    DeviceBuffer output_device(output_bytes);
    REQUIRE(cudaMemcpy(
                    score_device.data, scores.data(), score_bytes,
                    cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    probability_device.data, probabilities.data(),
                    probability_bytes, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    output_device.data, output.data(), output_bytes,
                    cudaMemcpyHostToDevice)
            == cudaSuccess);

    auto request = make_request(
            static_cast<float*>(score_device.data),
            static_cast<unsigned char*>(probability_device.data),
            static_cast<unsigned char*>(output_device.data), output_bytes);
    DeviceStream stream;
    iom::cuda_detail::launch_sdpa_scale_mask(stream.value, request);
    iom::cuda_detail::launch_sdpa_softmax(stream.value, request);
    iom::cuda_detail::launch_sdpa_canonicalize_output(stream.value, request);
    REQUIRE(cudaStreamSynchronize(stream.value) == cudaSuccess);

    REQUIRE(cudaMemcpy(
                    scores.data(), score_device.data, score_bytes,
                    cudaMemcpyDeviceToHost)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    probabilities.data(), probability_device.data,
                    probability_bytes, cudaMemcpyDeviceToHost)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(
                    output.data(), output_device.data, output_bytes,
                    cudaMemcpyDeviceToHost)
            == cudaSuccess);

    CHECK(std::fabs(
                  scores[score_index(0, 0, 0, 1, Hq, Rp, Lp)]
                  - 1.0f / std::sqrt(8.0f))
          < 1.0e-6f);
    CHECK(scores[score_index(0, 0, 0, 4, Hq, Rp, Lp)] == 12345.0f);
    CHECK(scores[score_index(0, 0, 0, L, Hq, Rp, Lp)] == 12345.0f);

    const auto probability_at = [&](std::size_t row, std::size_t token) {
        return probabilities[probability_index(0, 0, row, token, Hq, Rp, Lp)];
    };
    CHECK(probability_at(1, 0) == 0x3f00);
    CHECK(probability_at(1, 2) == 0x3f00);
    CHECK(probability_at(1, 1) == 0);
    CHECK(probability_at(1, 4) == 0);
    for (std::size_t token = 0; token < 6; ++token) {
        CHECK(probability_at(2, token) == 0x7fc0);
    }
    CHECK(probability_at(2, 6) == 0);
    for (std::size_t token = 0; token < 7; ++token) {
        CHECK(probability_at(3, token) == 0x7fc0);
    }
    CHECK(probability_at(3, 7) == 0);
    CHECK(probability_at(4, 0) == 0);
    CHECK(probability_at(0, L) == 0xa5a5);
    CHECK(probabilities[probability_index(0, 0, R, 0, Hq, Rp, Lp)] == 0xa5a5);

    const float ordinary_sum = [&] {
        float sum = 0.0f;
        for (std::size_t token = 0; token < 4; ++token) {
            sum += bf16_value(probability_at(0, token));
        }
        return sum;
    }();
    CHECK(std::fabs(ordinary_sum - 1.0f) < 0.01f);

    CHECK(output[slot(0, 0, 0)] == 0);
    CHECK(output[slot(0, 0, 1)] == 0x7fc0);
    CHECK(output[slot(0, 0, 2)] == 0x3f80);
    CHECK(output[slot(0, R, 0)] == 0x3555);
    CHECK(output[slot(1, 0, 0)] == 0x3555);
}

TEST_CASE("CUDA SDPA nonmatrix covers all physical row boundaries") {
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    int device_count = 0;
    REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
    REQUIRE(device_count > 0);
    for (const std::size_t rows : {std::size_t{1}, std::size_t{15},
                                   std::size_t{16}, std::size_t{17}}) {
        run_boundary_case(rows);
    }
    run_subnormal_case();
}

TEST_CASE("CUDA SDPA nonmatrix boundary rejects malformed fixed layouts") {
    iom::cuda_detail::SdpaNonmatrixRequest request;
    request.R = 1;
    request.Hq = 1;
    request.Hkv = 1;
    request.C = 1;
    request.D = 1;
    request.L = 1;
    request.Rp = 16;
    request.Lp = 16;
    request.planes = 1;
    request.grouping = 1;
    request.leading_rank = 0;
    request.output.storage_bytes = 32;
    CHECK_THROWS_AS(
            iom::cuda_detail::validate_sdpa_nonmatrix_request(request),
            std::invalid_argument);
}
