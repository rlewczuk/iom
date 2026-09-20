#include <doctest/doctest.h>

#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend_conformance_sdpa_reference.hpp"
#include "rocm/sdpa_softmax.hpp"

namespace {

using iom::rocm_detail::SdpaSoftmaxRequest;
using iom::rocm_detail::SdpaStageLayout;
using iom_conformance::SdpaReferenceCase;
using iom_conformance::SdpaReferenceValue;

constexpr std::uint64_t kTile = 16;

[[nodiscard]] std::uint64_t pad16(std::uint64_t value) {
    return (value + kTile - 1) / kTile * kTile;
}

void require_hip(hipError_t status, const char* operation) {
    REQUIRE_MESSAGE(status == hipSuccess, operation << ": "
            << hipGetErrorName(status) << ": " << hipGetErrorString(status));
}

template <typename T>
class DeviceBuffer final {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() {
        if (data_ != nullptr) (void)hipFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count) {
        REQUIRE(data_ == nullptr);
        count_ = count;
        require_hip(
                hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                "hipMalloc");
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

[[nodiscard]] std::uint16_t bf16_rne(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffU) == 0) return 0;
    std::uint32_t upper = bits >> 16;
    const std::uint32_t lower = bits & 0xffffU;
    if (lower > 0x8000U || (lower == 0x8000U && (upper & 1U) != 0)) {
        ++upper;
    }
    return static_cast<std::uint16_t>(upper);
}

[[nodiscard]] bool bf16_is_nan(std::uint16_t bits) {
    return (bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0;
}

[[nodiscard]] float materialized_add(float left, float right) {
    volatile float result = left + right;
    return result;
}

[[nodiscard]] float materialized_mul(float left, float right) {
    volatile float result = left * right;
    return result;
}

[[nodiscard]] float materialized_div(float left, float right) {
    volatile float result = left / right;
    return result;
}

[[nodiscard]] float materialized_exp(float value) {
    volatile float result = std::exp(value);
    return result;
}

[[nodiscard]] std::size_t score_offset(
        std::uint64_t plane, std::uint64_t head, std::uint64_t row,
        std::uint64_t token, std::uint64_t hq, std::uint64_t score_rows,
        std::uint64_t score_columns) {
    return static_cast<std::size_t>(
            (((plane * hq + head) * score_rows + row) * score_columns)
            + token);
}

[[nodiscard]] std::size_t probability_offset(
        std::uint64_t plane, std::uint64_t head, std::uint64_t row,
        std::uint64_t token, std::uint64_t hq, std::uint64_t probability_rows,
        std::uint64_t probability_columns) {
    return static_cast<std::size_t>(
            (((plane * hq + head) * probability_rows + row)
             * probability_columns) + token);
}

[[nodiscard]] SdpaStageLayout contiguous_layout(
        std::uint64_t planes, std::uint64_t heads, std::uint64_t rows,
        std::uint64_t columns) {
    return SdpaStageLayout{
            0,
            heads * rows * columns,
            rows * columns,
            columns,
            1,
            rows,
            columns};
}

struct SoftmaxFixture final {
    std::uint64_t planes;
    std::uint64_t hq;
    std::uint64_t hkv;
    std::uint64_t rows;
    std::uint64_t capacity;
    std::uint64_t head_dim;
    std::uint64_t a;
    std::uint64_t length;
    std::uint64_t physical_rows;
    std::uint64_t physical_columns;
    SdpaStageLayout scores_layout;
    SdpaStageLayout probability_layout;
    std::vector<float> scores;
    DeviceBuffer<float> device_scores;
    DeviceBuffer<std::uint16_t> device_probability;
    SdpaSoftmaxRequest request;

    SoftmaxFixture(
            std::uint64_t planes_, std::uint64_t hq_, std::uint64_t hkv_,
            std::uint64_t rows_, std::uint64_t capacity_,
            std::uint64_t head_dim_, std::uint64_t a_, std::uint64_t length_)
            : planes(planes_), hq(hq_), hkv(hkv_), rows(rows_),
              capacity(capacity_), head_dim(head_dim_), a(a_),
              length(length_), physical_rows(pad16(rows_)),
              physical_columns(pad16(length_)),
              scores_layout(contiguous_layout(
                      planes_, hq_, pad16(rows_), pad16(length_))),
              probability_layout(contiguous_layout(
                      planes_, hq_, pad16(rows_), pad16(length_))),
              scores(static_cast<std::size_t>(
                             planes_ * hq_ * physical_rows * physical_columns),
                     0.0F),
              device_scores(scores.size()),
              device_probability(static_cast<std::size_t>(
                      planes_ * hq_ * physical_rows * physical_columns)) {
        request.planes = planes;
        request.hq = hq;
        request.hkv = hkv;
        request.rows = rows;
        request.capacity = capacity;
        request.head_dim = head_dim;
        request.a = a;
        request.L = length;
        request.scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        request.mask = iom::rocm_detail::sdpa_required_mask;
        request.scores = device_scores.data();
        request.p_bf16 = device_probability.data();
        request.scores_layout = scores_layout;
        request.probability_layout = probability_layout;
    }

    void upload() {
        require_hip(
                hipMemcpy(
                        device_scores.data(), scores.data(),
                        scores.size() * sizeof(float), hipMemcpyHostToDevice),
                "copy SDPA scores");
        require_hip(
                hipMemset(
                        device_probability.data(), 0xa5,
                        device_probability.count() * sizeof(std::uint16_t)),
                "seed SDPA probabilities");
    }

    [[nodiscard]] std::vector<std::uint16_t> execute() {
        iom::rocm_detail::launch_sdpa_softmax(nullptr, request);
        require_hip(hipDeviceSynchronize(), "synchronize SDPA softmax");
        std::vector<std::uint16_t> result(device_probability.count());
        require_hip(
                hipMemcpy(
                        result.data(), device_probability.data(),
                        result.size() * sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost),
                "read SDPA probabilities");
        return result;
    }
};

[[nodiscard]] std::vector<std::uint16_t> reference_probabilities(
        const SoftmaxFixture& fixture) {
    std::vector<std::uint16_t> result(
            fixture.device_probability.count(), std::uint16_t{0});
    for (std::uint64_t plane = 0; plane < fixture.planes; ++plane) {
        for (std::uint64_t head = 0; head < fixture.hq; ++head) {
            for (std::uint64_t row = 0; row < fixture.rows; ++row) {
                bool has_nan = false;
                bool all_negative_infinity = true;
                std::uint64_t positive_infinities = 0;
                for (std::uint64_t token = 0; token < fixture.length; ++token) {
                    if (token > fixture.a + row) continue;
                    const float score = materialized_mul(
                            fixture.scores[score_offset(
                                    plane, head, row, token, fixture.hq,
                                    fixture.physical_rows,
                                    fixture.physical_columns)],
                            fixture.request.scale);
                    has_nan = has_nan || std::isnan(score);
                    positive_infinities +=
                            std::isinf(score) && !std::signbit(score) ? 1 : 0;
                    all_negative_infinity = all_negative_infinity
                            && std::isinf(score) && std::signbit(score);
                }

                float maximum = -std::numeric_limits<float>::infinity();
                float sum = 0.0F;
                if (!has_nan && !all_negative_infinity
                        && positive_infinities == 0) {
                    for (std::uint64_t token = 0; token < fixture.length;
                         ++token) {
                        if (token > fixture.a + row) continue;
                        const float score = materialized_mul(
                                fixture.scores[score_offset(
                                        plane, head, row, token, fixture.hq,
                                        fixture.physical_rows,
                                        fixture.physical_columns)],
                                fixture.request.scale);
                        if (score > maximum) maximum = score;
                    }
                    for (std::uint64_t token = 0; token < fixture.length;
                         ++token) {
                        if (token > fixture.a + row) continue;
                        const float score = materialized_mul(
                                fixture.scores[score_offset(
                                        plane, head, row, token, fixture.hq,
                                        fixture.physical_rows,
                                        fixture.physical_columns)],
                                fixture.request.scale);
                        if (!std::isinf(score) || !std::signbit(score)) {
                            sum = materialized_add(
                                    sum, materialized_exp(score - maximum));
                        }
                    }
                }

                for (std::uint64_t token = 0;
                     token < fixture.physical_columns; ++token) {
                    if (token >= fixture.length || token > fixture.a + row) {
                        continue;
                    }
                    float probability = 0.0F;
                    if (has_nan || all_negative_infinity) {
                        probability = std::numeric_limits<float>::quiet_NaN();
                    } else if (positive_infinities != 0) {
                        const float score = materialized_mul(
                                fixture.scores[score_offset(
                                        plane, head, row, token, fixture.hq,
                                        fixture.physical_rows,
                                        fixture.physical_columns)],
                                fixture.request.scale);
                        probability = std::isinf(score) && !std::signbit(score)
                                ? materialized_div(
                                          1.0F,
                                          static_cast<float>(positive_infinities))
                                : 0.0F;
                    } else {
                        const float score = materialized_mul(
                                fixture.scores[score_offset(
                                        plane, head, row, token, fixture.hq,
                                        fixture.physical_rows,
                                        fixture.physical_columns)],
                                fixture.request.scale);
                        probability = std::isinf(score) && std::signbit(score)
                                ? 0.0F
                                : materialized_div(
                                          materialized_exp(score - maximum),
                                          sum);
                    }
                    result[probability_offset(
                            plane, head, row, token, fixture.hq,
                            fixture.physical_rows, fixture.physical_columns)] =
                            bf16_rne(probability);
                }
            }
        }
    }
    return result;
}

void check_probability_result(
        const SoftmaxFixture& fixture,
        const std::vector<std::uint16_t>& actual,
        const std::vector<std::uint16_t>& expected,
        const char* label) {
    REQUIRE_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK_MESSAGE(
                actual[index] == expected[index],
                label << ": probability mismatch at physical index " << index);
    }
}

[[nodiscard]] SdpaReferenceCase make_shape_case(std::uint64_t rows) {
    constexpr std::uint64_t planes = 2;
    constexpr std::uint64_t hq = 4;
    constexpr std::uint64_t hkv = 2;
    constexpr std::uint64_t head_dim = 3;
    constexpr std::uint64_t a = 2;
    const std::uint64_t length = rows == 1 ? 2 : rows - 2;
    const std::uint64_t capacity = a + rows + 3;
    const std::vector<double> q = iom_conformance::sdpa_oracle::fill_q(
            planes, hq, rows, head_dim, 11 + rows);
    const std::vector<double> k = iom_conformance::sdpa_oracle::fill_kv(
            planes, hkv, capacity, head_dim, 17 + rows);
    const std::vector<double> v = iom_conformance::sdpa_oracle::fill_kv(
            planes, hkv, capacity, head_dim, 23 + rows);
    return iom_conformance::sdpa_oracle::make_case(
            iom_conformance::SdpaReferenceCaseKind::mixed_gqa,
            iom::DataType::BF16, {planes}, hq, hkv, rows, head_dim, capacity,
            a, length, q, k, v);
}

void fill_qk_scores(
        const SdpaReferenceCase& reference_case, SoftmaxFixture& fixture) {
    fixture.scores.assign(fixture.scores.size(), 0.0F);
    const std::size_t planes = static_cast<std::size_t>(fixture.planes);
    const std::size_t group = static_cast<std::size_t>(
            reference_case.hq / reference_case.hkv);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t head = 0; head < reference_case.hq; ++head) {
            const std::size_t kv_head = head / group;
            for (std::size_t row = 0; row < reference_case.rows; ++row) {
                for (std::size_t token = 0;
                     token < reference_case.length; ++token) {
                    float score = 0.0F;
                    for (std::size_t feature = 0;
                         feature < reference_case.head_dim; ++feature) {
                        const float q = iom_conformance::sdpa_oracle::matrix_daz(
                                iom_conformance::sdpa_oracle::decode_f32(
                                        reference_case.data_type,
                                        reference_case.q_bits[
                                                iom_conformance::sdpa_oracle::q_offset(
                                                        reference_case, plane,
                                                        head, row, feature)]));
                        const float k = iom_conformance::sdpa_oracle::matrix_daz(
                                iom_conformance::sdpa_oracle::decode_f32(
                                        reference_case.data_type,
                                        reference_case.k_bits[
                                                iom_conformance::sdpa_oracle::kv_offset(
                                                        reference_case, plane,
                                                        kv_head, token, feature)]));
                        score = materialized_add(score, materialized_mul(q, k));
                    }
                    fixture.scores[score_offset(
                            plane, head, row, token, fixture.hq,
                            fixture.physical_rows, fixture.physical_columns)] =
                            score;
                }
            }
        }
    }
}

struct MergeFixture final {
    std::uint64_t planes;
    std::uint64_t hq;
    std::uint64_t hkv;
    std::uint64_t rows;
    std::uint64_t capacity;
    std::uint64_t head_dim;
    std::uint64_t a;
    std::uint64_t length;
    std::uint64_t physical_rows;
    std::uint64_t physical_features;
    std::uint64_t physical_output_features;
    DeviceBuffer<std::uint16_t> device_pv;
    DeviceBuffer<std::uint16_t> device_merged;
    std::vector<std::uint16_t> pv;
    SdpaSoftmaxRequest request;

    MergeFixture(
            std::uint64_t planes_, std::uint64_t hq_, std::uint64_t hkv_,
            std::uint64_t rows_, std::uint64_t capacity_,
            std::uint64_t head_dim_, std::uint64_t a_, std::uint64_t length_)
            : planes(planes_), hq(hq_), hkv(hkv_), rows(rows_),
              capacity(capacity_), head_dim(head_dim_), a(a_),
              length(length_), physical_rows(pad16(rows_)),
              physical_features(pad16(head_dim_)),
              physical_output_features(pad16(hq_ * head_dim_)),
              device_pv(static_cast<std::size_t>(
                      planes_ * hq_ * physical_rows * physical_features)),
              device_merged(static_cast<std::size_t>(
                      planes_ * physical_rows * physical_output_features)),
              pv(device_pv.count(), std::uint16_t{0x5555}) {
        request.planes = planes;
        request.hq = hq;
        request.hkv = hkv;
        request.rows = rows;
        request.capacity = capacity;
        request.head_dim = head_dim;
        request.a = a;
        request.L = length;
        request.mask = iom::rocm_detail::sdpa_required_mask;
        request.scale = 1.0F;
        request.pv_bf16 = device_pv.data();
        request.merged = device_merged.data();
        request.pv_layout = contiguous_layout(
                planes, hq, physical_rows, physical_features);
        request.merged_layout = contiguous_layout(
                planes, 1, physical_rows, physical_output_features);
    }

    void upload() {
        require_hip(
                hipMemcpy(
                        device_pv.data(), pv.data(),
                        pv.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice),
                "copy SDPA PV");
        require_hip(
                hipMemset(
                        device_merged.data(), 0xa5,
                        device_merged.count() * sizeof(std::uint16_t)),
                "seed SDPA merged output");
    }

    [[nodiscard]] std::vector<std::uint16_t> execute() {
        iom::rocm_detail::launch_sdpa_merge(nullptr, request);
        require_hip(hipDeviceSynchronize(), "synchronize SDPA merge");
        std::vector<std::uint16_t> result(device_merged.count());
        require_hip(
                hipMemcpy(
                        result.data(), device_merged.data(),
                        result.size() * sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost),
                "read SDPA merged output");
        return result;
    }
};

[[nodiscard]] std::size_t pv_offset(
        const MergeFixture& fixture, std::uint64_t plane,
        std::uint64_t head, std::uint64_t row, std::uint64_t feature) {
    return static_cast<std::size_t>(
            (((plane * fixture.hq + head) * fixture.physical_rows + row)
             * fixture.physical_features) + feature);
}

[[nodiscard]] std::size_t merged_offset(
        const MergeFixture& fixture, std::uint64_t plane,
        std::uint64_t row, std::uint64_t feature) {
    return static_cast<std::size_t>(
            ((plane * fixture.physical_rows + row)
             * fixture.physical_output_features) + feature);
}

TEST_CASE("ROCm SDPA nonmatrix stages follow the shared GQA shape oracle") {
    for (const std::uint64_t rows : {
            std::uint64_t{1}, std::uint64_t{15}, std::uint64_t{16},
            std::uint64_t{17}}) {
        const SdpaReferenceCase reference_case = make_shape_case(rows);
        SoftmaxFixture softmax(
                reference_case.leading_dimensions.front(), reference_case.hq,
                reference_case.hkv, reference_case.rows, reference_case.capacity,
                reference_case.head_dim, reference_case.position,
                reference_case.length);
        fill_qk_scores(reference_case, softmax);
        softmax.upload();
        const std::vector<std::uint16_t> expected =
                reference_probabilities(softmax);
        const std::vector<std::uint16_t> baseline = softmax.execute();
        check_probability_result(
                softmax, baseline, expected, "shared GQA shape baseline");

        for (std::uint64_t plane = 0; plane < softmax.planes; ++plane) {
            for (std::uint64_t head = 0; head < softmax.hq; ++head) {
                for (std::uint64_t row = 0; row < softmax.physical_rows; ++row) {
                    for (std::uint64_t token = 0;
                         token < softmax.physical_columns; ++token) {
                        const bool included = row < softmax.rows
                                && token < softmax.length
                                && token <= softmax.a + row;
                        if (!included) {
                            softmax.scores[score_offset(
                                    plane, head, row, token, softmax.hq,
                                    softmax.physical_rows,
                                    softmax.physical_columns)] =
                                    12345.0F + static_cast<float>(token);
                        }
                    }
                }
            }
        }
        softmax.upload();
        const std::vector<std::uint16_t> perturbed = softmax.execute();
        CHECK(perturbed == baseline);

        MergeFixture merge(
                reference_case.leading_dimensions.front(), reference_case.hq,
                reference_case.hkv, reference_case.rows, reference_case.capacity,
                reference_case.head_dim, reference_case.position,
                reference_case.length);
        const std::vector<SdpaReferenceValue> oracle_output =
                iom_conformance::evaluate(reference_case);
        for (std::uint64_t plane = 0; plane < merge.planes; ++plane) {
            for (std::uint64_t row = 0; row < merge.rows; ++row) {
                for (std::uint64_t head = 0; head < merge.hq; ++head) {
                    for (std::uint64_t feature = 0;
                         feature < merge.head_dim; ++feature) {
                        const std::size_t source =
                                iom_conformance::sdpa_oracle::output_offset(
                                        reference_case, plane, row, head,
                                        feature);
                        merge.pv[pv_offset(
                                merge, plane, head, row, feature)] =
                                static_cast<std::uint16_t>(
                                        oracle_output[source].bits);
                    }
                }
            }
        }
        merge.pv[pv_offset(merge, 0, 0, 0, 0)] = 0x8000U;
        merge.upload();
        const std::vector<std::uint16_t> merged = merge.execute();
        for (std::uint64_t plane = 0; plane < merge.planes; ++plane) {
            for (std::uint64_t row = 0; row < merge.rows; ++row) {
                for (std::uint64_t head = 0; head < merge.hq; ++head) {
                    for (std::uint64_t feature = 0;
                         feature < merge.head_dim; ++feature) {
                        const std::size_t source =
                                iom_conformance::sdpa_oracle::output_offset(
                                        reference_case, plane, row, head,
                                        feature);
                        std::uint16_t expected_value = static_cast<std::uint16_t>(
                                oracle_output[source].bits);
                        if (plane == 0 && row == 0 && head == 0 && feature == 0) {
                            expected_value = 0;
                        }
                        CHECK_MESSAGE(
                                merged[merged_offset(
                                        merge, plane, row,
                                        head * merge.head_dim + feature)]
                                        == expected_value,
                                "merged GQA output mismatch");
                    }
                }
            }
        }
        for (std::uint64_t plane = 0; plane < merge.planes; ++plane) {
            for (std::uint64_t row = 0; row < merge.physical_rows; ++row) {
                for (std::uint64_t feature = 0;
                     feature < merge.physical_output_features; ++feature) {
                    const bool logical = row < merge.rows
                            && feature < merge.hq * merge.head_dim;
                    if (!logical) {
                        CHECK_EQ(
                                merged[merged_offset(merge, plane, row, feature)],
                                std::uint16_t{0xa5a5});
                    }
                }
            }
        }
    }
}

TEST_CASE("ROCm SDPA nonmatrix special values and BF16 boundary are exact") {
    SoftmaxFixture fixture(1, 1, 1, 1, 6, 3, 1, 4);
    const auto run = [&fixture](std::initializer_list<float> values) {
        fixture.scores.assign(fixture.scores.size(), 0.0F);
        std::size_t index = 0;
        for (const float value : values) {
            fixture.scores[score_offset(
                    0, 0, 0, index++, fixture.hq, fixture.physical_rows,
                    fixture.physical_columns)] = value;
        }
        fixture.upload();
        const std::vector<std::uint16_t> actual = fixture.execute();
        check_probability_result(
                fixture, actual, reference_probabilities(fixture),
                "special-value policy");
        return actual;
    };

    const std::vector<std::uint16_t> nan_row = run({
            std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F, 0.0F});
    CHECK(bf16_is_nan(nan_row[0]));
    CHECK(bf16_is_nan(nan_row[1]));
    const std::vector<std::uint16_t> positive_infinity_row = run({
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(), -3.0F, 99.0F});
    CHECK_EQ(positive_infinity_row[0], bf16_rne(0.5F));
    CHECK_EQ(positive_infinity_row[1], bf16_rne(0.5F));
    CHECK_EQ(positive_infinity_row[2], std::uint16_t{0});
    const std::vector<std::uint16_t> all_negative_infinity_row = run({
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(), 77.0F});
    CHECK(bf16_is_nan(all_negative_infinity_row[0]));
    CHECK(bf16_is_nan(all_negative_infinity_row[1]));
    CHECK_EQ(all_negative_infinity_row[2], std::uint16_t{0});

    fixture.length = 2;
    fixture.request.L = 2;
    fixture.scores.assign(fixture.scores.size(), 0.0F);
    fixture.scores[0] = 0.0F;
    fixture.scores[1] = -std::numeric_limits<float>::infinity();
    fixture.upload();
    const std::vector<std::uint16_t> ordinary_negative_infinity = fixture.execute();
    CHECK_EQ(ordinary_negative_infinity[0], bf16_rne(1.0F));
    CHECK_EQ(ordinary_negative_infinity[1], std::uint16_t{0});

    fixture.length = 2;
    fixture.request.L = 2;
    fixture.request.scale = 1.0F;
    fixture.scores.assign(fixture.scores.size(), 0.0F);
    fixture.scores[0] = 0.0F;
    fixture.scores[1] = -90.0F;
    fixture.upload();
    const std::vector<std::uint16_t> subnormal = fixture.execute();
    check_probability_result(
            fixture, subnormal, reference_probabilities(fixture),
            "BF16 gradual-underflow boundary");
    CHECK_NE(subnormal[1], std::uint16_t{0});
}

TEST_CASE("ROCm SDPA nonmatrix validators reject malformed checked requests") {
    SoftmaxFixture fixture(1, 1, 1, 2, 5, 3, 1, 2);
    fixture.upload();

    SdpaSoftmaxRequest request = fixture.request;
    request.scores = nullptr;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::invalid_argument);

    request = fixture.request;
    request.p_bf16 = nullptr;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::invalid_argument);

    request = fixture.request;
    request.mask = iom::rocm_detail::sdpa_mask_causal;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::invalid_argument);

    request = fixture.request;
    request.scale = 0.0F;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::invalid_argument);

    request = fixture.request;
    request.hq = 3;
    request.hkv = 2;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::invalid_argument);

    request = fixture.request;
    request.capacity = 1;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::invalid_argument);

    request = fixture.request;
    request.scores_layout.plane_offset = std::numeric_limits<std::uint64_t>::max();
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::overflow_error);

    request = fixture.request;
    request.probability_layout.row_stride = 0;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::invalid_argument);

    request = fixture.request;
    request.probability_layout.column_stride = 0;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_softmax_request(request),
            std::invalid_argument);

    request = fixture.request;
    request.pv_bf16 = nullptr;
    request.merged = fixture.device_probability.data();
    request.pv_layout = fixture.probability_layout;
    request.merged_layout = fixture.probability_layout;
    CHECK_THROWS_AS(
            iom::rocm_detail::validate_sdpa_merge_request(request),
            std::invalid_argument);
}

}  // namespace
