#include <doctest/doctest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <new>
#include <memory>
#include <optional>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/alloc.hpp"
#include "iom/detail/outstanding_work_registry.hpp"
#include "iom/tensor.hpp"

namespace iom_test {

std::atomic<bool> allocation_fault_armed{false};
std::atomic<std::size_t> allocation_fault_at{0};
std::atomic<std::size_t> allocation_count{0};

bool should_fail_allocation() noexcept {
    if (!allocation_fault_armed.load(std::memory_order_relaxed)) {
        return false;
    }
    const std::size_t ordinal =
            allocation_count.fetch_add(1, std::memory_order_relaxed) + 1;
    return allocation_fault_at.load(std::memory_order_relaxed) != 0
            && ordinal
                    == allocation_fault_at.load(std::memory_order_relaxed);
}

void arm_counting() noexcept {
    allocation_fault_armed.store(false, std::memory_order_relaxed);
    allocation_fault_at.store(0, std::memory_order_relaxed);
    allocation_count.store(0, std::memory_order_relaxed);
    allocation_fault_armed.store(true, std::memory_order_relaxed);
}

void arm_failure(std::size_t ordinal) noexcept {
    allocation_fault_armed.store(false, std::memory_order_relaxed);
    allocation_fault_at.store(ordinal, std::memory_order_relaxed);
    allocation_count.store(0, std::memory_order_relaxed);
    allocation_fault_armed.store(true, std::memory_order_relaxed);
}

std::size_t disarm() noexcept {
    allocation_fault_armed.store(false, std::memory_order_relaxed);
    return allocation_count.load(std::memory_order_relaxed);
}

void* allocate(std::size_t size, std::size_t alignment) {
    if (should_fail_allocation()) {
        throw std::bad_alloc();
    }
    size = std::max<std::size_t>(size, 1);
    void* allocation = nullptr;
    if (alignment <= alignof(std::max_align_t)) {
        allocation = std::malloc(size);
    } else {
        const std::size_t remainder = size % alignment;
        if (remainder != 0) {
            const std::size_t padding = alignment - remainder;
            if (size > std::numeric_limits<std::size_t>::max() - padding) {
                throw std::bad_alloc();
            }
            size += padding;
        }
        allocation = std::aligned_alloc(alignment, size);
    }
    if (allocation == nullptr) {
        throw std::bad_alloc();
    }
    return allocation;
}

}  // namespace iom_test

void* operator new(std::size_t size) {
    return iom_test::allocate(size, alignof(std::max_align_t));
}

void* operator new[](std::size_t size) {
    return iom_test::allocate(size, alignof(std::max_align_t));
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    return iom_test::allocate(size, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    return iom_test::allocate(size, static_cast<std::size_t>(alignment));
}

void operator delete(void* allocation) noexcept {
    std::free(allocation);
}

void operator delete[](void* allocation) noexcept {
    std::free(allocation);
}

void operator delete(void* allocation, std::size_t) noexcept {
    std::free(allocation);
}

void operator delete[](void* allocation, std::size_t) noexcept {
    std::free(allocation);
}

void operator delete(void* allocation, std::align_val_t) noexcept {
    std::free(allocation);
}

void operator delete[](void* allocation, std::align_val_t) noexcept {
    std::free(allocation);
}

void operator delete(
        void* allocation, std::size_t, std::align_val_t) noexcept {
    std::free(allocation);
}

void operator delete[](
        void* allocation, std::size_t, std::align_val_t) noexcept {
    std::free(allocation);
}


namespace {

iom::TensorShape make_shape(std::initializer_list<std::size_t> dimensions) {
    return iom::TensorShape{std::vector<std::size_t>{dimensions}};
}

std::vector<std::size_t> dimensions_of(const iom::TensorShape& shape) {
    return {shape.dimensions().begin(), shape.dimensions().end()};
}

iom::TensorSpec make_spec(
        std::vector<std::size_t> dimensions,
        iom::DataType data_type,
        iom::QuantizationFormat quantization = iom::QuantizationFormat::NONE) {
    return iom::TensorSpec{iom::TensorShape{std::move(dimensions)}, data_type, quantization};
}

std::size_t slot(
        const iom::TensorSpec& spec, std::initializer_list<std::size_t> coordinates) {
    return iom::detail::standard_layout_slot(spec, coordinates);
}

constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();

constexpr std::initializer_list<iom::DataType> kAllDataTypes = {
    iom::DataType::BOOL,
    iom::DataType::I2, iom::DataType::U2,
    iom::DataType::I4, iom::DataType::U4,
    iom::DataType::I8, iom::DataType::U8,
    iom::DataType::I16, iom::DataType::U16,
    iom::DataType::I32, iom::DataType::U32,
    iom::DataType::I64, iom::DataType::U64,
    iom::DataType::F4_E2M1,
    iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
    iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2, iom::DataType::F8_E8M0,
    iom::DataType::F16, iom::DataType::BF16,
    iom::DataType::F32, iom::DataType::F64,
};

enum class TestFenceMode { success, failed, throwing };

struct TestFenceCapture {
    int* calls;
    TestFenceMode mode;
};

static_assert(
        sizeof(TestFenceCapture)
        <= iom::detail::kFenceStorageBytes);
static_assert(
        alignof(TestFenceCapture)
        <= iom::detail::kFenceStorageAlign);

iom::detail::FenceResult test_fence_invoke(
        const iom::detail::Fence& fence) noexcept {
    const auto& capture =
            *std::launder(reinterpret_cast<const TestFenceCapture*>(
                    fence.storage));
    try {
        if (capture.calls != nullptr) {
            ++*capture.calls;
        }
        if (capture.mode == TestFenceMode::failed) {
            throw std::runtime_error("fence failed");
        }
        if (capture.mode == TestFenceMode::throwing) {
            throw std::runtime_error("fence threw");
        }
        return iom::detail::FenceResult::success();
    } catch (...) {
        return iom::detail::FenceResult::failed(
                std::current_exception());
    }
}

iom::detail::Fence make_test_fence(
        int* calls,
        TestFenceMode mode = TestFenceMode::success) noexcept {
    iom::detail::Fence fence;
    ::new (fence.storage) TestFenceCapture{calls, mode};
    fence.invoke = &test_fence_invoke;
    return fence;
}

struct FenceLeaseProbeResource {
    std::atomic<std::size_t> refcount{1};
    std::optional<iom::detail::FenceResult> cached_result;
    std::mutex cached_result_mu;
    std::atomic<std::size_t> synchronize_count{0};
    std::atomic<std::size_t> destroy_count{0};
};

struct FenceLeaseProbe {
    FenceLeaseProbeResource* resource = nullptr;

    FenceLeaseProbe() noexcept = default;

    static FenceLeaseProbe acquire_lease(
            FenceLeaseProbeResource* resource) noexcept {
        FenceLeaseProbe lease;
        lease.resource = resource;
        if (resource != nullptr) {
            resource->refcount.fetch_add(1, std::memory_order_relaxed);
        }
        return lease;
    }

    FenceLeaseProbe(const FenceLeaseProbe& other) noexcept
            : resource(other.resource) {
        if (resource != nullptr) {
            resource->refcount.fetch_add(1, std::memory_order_relaxed);
        }
    }

    FenceLeaseProbe(FenceLeaseProbe&& other) noexcept
            : resource(other.resource) {
        other.resource = nullptr;
    }

    FenceLeaseProbe& operator=(const FenceLeaseProbe&) = delete;
    FenceLeaseProbe& operator=(FenceLeaseProbe&&) = delete;

    ~FenceLeaseProbe() noexcept {
        if (resource != nullptr
                && resource->refcount.fetch_sub(
                           1, std::memory_order_acq_rel)
                        == 1) {
            resource->destroy_count.fetch_add(
                    1, std::memory_order_relaxed);
        }
    }
};

static_assert(sizeof(FenceLeaseProbe) <= iom::detail::kFenceStorageBytes);
static_assert(
        alignof(FenceLeaseProbe) <= iom::detail::kFenceStorageAlign);
static_assert(!std::is_copy_assignable_v<FenceLeaseProbe>);
static_assert(!std::is_move_assignable_v<FenceLeaseProbe>);

iom::detail::FenceResult probe_fence_result(
        FenceLeaseProbeResource& resource) noexcept {
    std::lock_guard<std::mutex> lock(resource.cached_result_mu);
    if (!resource.cached_result.has_value()) {
        resource.synchronize_count.fetch_add(
                1, std::memory_order_relaxed);
        resource.cached_result.emplace(
                iom::detail::FenceResult::success());
    }
    return *resource.cached_result;
}

void probe_fence_worker_release(
        FenceLeaseProbeResource& resource) noexcept {
    (void)probe_fence_result(resource);
    if (resource.refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        resource.destroy_count.fetch_add(1, std::memory_order_relaxed);
    }
}

iom::detail::FenceResult probe_fence_invoke(
        const iom::detail::Fence& fence) noexcept {
    const auto& lease =
            *std::launder(reinterpret_cast<const FenceLeaseProbe*>(
                    fence.storage));
    if (lease.resource == nullptr) {
        return iom::detail::FenceResult::success();
    }
    return probe_fence_result(*lease.resource);
}

void probe_fence_copy_construct(
        iom::detail::Fence* destination,
        const iom::detail::Fence& source) noexcept {
    ::new (destination->storage) FenceLeaseProbe{
            *std::launder(reinterpret_cast<const FenceLeaseProbe*>(
                    source.storage))};
}

void probe_fence_move_construct(
        iom::detail::Fence* destination,
        iom::detail::Fence* source) noexcept {
    ::new (destination->storage) FenceLeaseProbe{
            std::move(*std::launder(reinterpret_cast<FenceLeaseProbe*>(
                    source->storage)))};
    std::destroy_at(std::launder(reinterpret_cast<FenceLeaseProbe*>(
            source->storage)));
}

void probe_fence_destroy(iom::detail::Fence* fence) noexcept {
    std::destroy_at(std::launder(reinterpret_cast<FenceLeaseProbe*>(
            fence->storage)));
}

iom::detail::Fence make_probe_fence(
        FenceLeaseProbeResource& resource) noexcept {
    iom::detail::Fence fence;
    ::new (fence.storage) FenceLeaseProbe{
            FenceLeaseProbe::acquire_lease(&resource)};
    fence.invoke = &probe_fence_invoke;
    fence.copy_construct = &probe_fence_copy_construct;
    fence.move_construct = &probe_fence_move_construct;
    fence.destroy = &probe_fence_destroy;
    return fence;
}

static_assert(noexcept(probe_fence_result(
        std::declval<FenceLeaseProbeResource&>())));
static_assert(noexcept(probe_fence_worker_release(
        std::declval<FenceLeaseProbeResource&>())));
static_assert(noexcept(probe_fence_invoke(
        std::declval<const iom::detail::Fence&>())));
static_assert(noexcept(probe_fence_copy_construct(
        std::declval<iom::detail::Fence*>(),
        std::declval<const iom::detail::Fence&>())));
static_assert(noexcept(probe_fence_move_construct(
        std::declval<iom::detail::Fence*>(),
        std::declval<iom::detail::Fence*>())));
static_assert(noexcept(probe_fence_destroy(
        std::declval<iom::detail::Fence*>())));

static_assert(noexcept(test_fence_invoke(
        std::declval<const iom::detail::Fence&>())));

static_assert(iom::TensorSpec::TILE == 16);

}  // namespace

TEST_CASE("TensorShape accepts ranks two through eight and round-trips dimensions") {
    const std::vector<std::vector<std::size_t>> cases = {
        {1, 1},
        {2, 3},
        {2, 3, 4},
        {2, 3, 4, 5},
        {2, 3, 4, 5, 6},
        {9, 7, 5, 3, 2, 1},
        {2, 3, 4, 5, 6, 7, 16, 16},
        {2, 2, 2, 2, 2, 2, 17, 33},
    };
    for (const auto& dimensions : cases) {
        CAPTURE(dimensions);
        const iom::TensorShape shape{dimensions};

        CHECK_EQ(shape.rank(), dimensions.size());
        std::size_t expected_count = 1;
        for (std::size_t i = 0; i < dimensions.size(); ++i) {
            CHECK_EQ(shape.dimension(i), dimensions[i]);
            expected_count *= dimensions[i];
        }
        CHECK_EQ(dimensions_of(shape), dimensions);
        CHECK_EQ(shape.element_count(), expected_count);
        CHECK_EQ(shape, iom::TensorShape{dimensions});
    }

    const iom::TensorShape shape = make_shape({2, 3});
    CHECK_NE(shape, make_shape({2, 4}));
    CHECK_NE(shape, make_shape({3, 2}));
    CHECK_NE(shape, make_shape({2, 3, 4}));
}

TEST_CASE("TensorShape rejects ranks outside two through eight and zero dimensions") {
    const std::vector<std::size_t> empty;
    const std::vector<std::size_t> single{5};
    CHECK_THROWS_AS(iom::TensorShape{empty}, std::invalid_argument);
    CHECK_THROWS_AS(iom::TensorShape{single}, std::invalid_argument);

    // Rank nine and above are rejected with the same invalid_argument
    // category; no rank-nine full shape can ever be formed.
    for (std::size_t rank = 9; rank <= 17; ++rank) {
        std::vector<std::size_t> dimensions(rank, 1);
        dimensions[rank - 2] = 17;
        dimensions[rank - 1] = 33;
        CAPTURE(rank);
        CHECK_THROWS_AS(
                iom::TensorShape{dimensions}, std::invalid_argument);
    }

    for (std::size_t rank = 2; rank <= 8; ++rank) {
        for (std::size_t position = 0; position < rank; ++position) {
            std::vector<std::size_t> dimensions(rank, 1);
            dimensions[position] = 0;
            CAPTURE(rank);
            CAPTURE(position);
            CHECK_THROWS_AS(iom::TensorShape{dimensions}, std::invalid_argument);
        }
    }
}

TEST_CASE("TensorShape dimension throws out of range past the rank") {
    const iom::TensorShape shape = make_shape({2, 3});
    CHECK_THROWS_AS((void)shape.dimension(2), std::out_of_range);
    CHECK_THROWS_AS((void)shape.dimension(100), std::out_of_range);
}

TEST_CASE("TensorShape element count overflows instead of wrapping") {
    const iom::TensorShape shape = make_shape({kMax, 2});
    CHECK_THROWS_AS((void)shape.element_count(), std::overflow_error);
}

TEST_CASE("TensorSpec defaults quantization to NONE and compares by value") {
    const iom::TensorSpec spec = make_spec({2, 3}, iom::DataType::F16);
    CHECK(spec.quantization == iom::QuantizationFormat::NONE);
    CHECK(spec.shape == make_shape({2, 3}));

    iom::TensorSpec same = spec;
    CHECK(spec == same);
    same.quantization = iom::QuantizationFormat::GGML_Q4_0;
    CHECK(spec != same);
    same = spec;
    same.data_type = iom::DataType::F32;
    CHECK(spec != same);
    same = spec;
    same.shape = make_shape({2, 4});
    CHECK(spec != same);
}

TEST_CASE("TensorSpec validate accepts every declared DataType with NONE") {
    for (const iom::DataType data_type : kAllDataTypes) {
        CAPTURE(static_cast<int>(data_type));
        const iom::TensorSpec spec = make_spec({16, 16}, data_type);
        CHECK_NOTHROW(spec.validate());
        CHECK_NOTHROW((void)spec.logical_nbytes());
        CHECK_NOTHROW((void)spec.tiled_storage_nbytes());
    }
}

TEST_CASE("TensorSpec validate rejects every recognized grouped quantization format") {
    for (int value = 1; value <= 24; ++value) {
        CAPTURE(value);
        const auto quantization = static_cast<iom::QuantizationFormat>(value);
        const iom::TensorSpec spec = make_spec({16, 16}, iom::DataType::F32, quantization);
        CHECK_THROWS_AS(spec.validate(), std::runtime_error);
        CHECK_THROWS_AS((void)spec.logical_nbytes(), std::runtime_error);
        CHECK_THROWS_AS((void)spec.tiled_storage_nbytes(), std::runtime_error);
    }
}

TEST_CASE("TensorSpec validate rejects values outside either enum") {
    for (const int value : {25, 100, -1}) {
        CAPTURE(value);

        const auto quantization = static_cast<iom::QuantizationFormat>(value);
        const iom::TensorSpec grouped = make_spec({16, 16}, iom::DataType::F32, quantization);
        CHECK_THROWS_AS(grouped.validate(), std::invalid_argument);

        const auto data_type = static_cast<iom::DataType>(value);
        const iom::TensorSpec leaf = make_spec({16, 16}, data_type);
        CHECK_THROWS_AS(leaf.validate(), std::invalid_argument);
    }

    const auto unknown_data_type = static_cast<iom::DataType>(-1);
    const iom::TensorSpec both_invalid = make_spec(
            {16, 16}, unknown_data_type, iom::QuantizationFormat::GGML_Q4_0);
    CHECK_THROWS_AS(both_invalid.validate(), std::invalid_argument);
}

TEST_CASE("TensorSpec standard padded shape rounds only the final two dimensions") {
    struct PaddingCase {
        std::vector<std::size_t> logical;
        std::vector<std::size_t> padded;
    };
    const std::vector<PaddingCase> cases = {
        {{1, 17}, {16, 32}},
        {{2, 8, 31, 33}, {2, 8, 32, 48}},
        {{9, 7, 17, 1}, {9, 7, 32, 16}},
        {{3, 1, 4, 1, 5, 9}, {3, 1, 4, 1, 16, 16}},
        {{32, 32}, {32, 32}},
        {{16, 16}, {16, 16}},
    };
    for (const PaddingCase& padding_case : cases) {
        CAPTURE(padding_case.logical);
        const iom::TensorSpec spec{iom::TensorShape{padding_case.logical}, iom::DataType::I8};
        const iom::TensorShape padded = spec.standard_padded_shape();
        CHECK_EQ(dimensions_of(padded), padding_case.padded);
        CHECK(spec.shape == iom::TensorShape{padding_case.logical});
    }
}

TEST_CASE("TensorSpec padding overflow throws instead of wrapping") {
    const iom::TensorSpec spec = make_spec({2, kMax}, iom::DataType::I8);
    CHECK_THROWS_AS((void)spec.standard_padded_shape(), std::overflow_error);
}

TEST_CASE("TensorSpec reports exact leaf bit widths through byte counts") {
    const std::vector<std::pair<iom::DataType, std::size_t>> bits_per_element = {
        {iom::DataType::BOOL, 8},
        {iom::DataType::I2, 2}, {iom::DataType::U2, 2},
        {iom::DataType::I4, 4}, {iom::DataType::U4, 4}, {iom::DataType::F4_E2M1, 4},
        {iom::DataType::F6_E2M3, 6}, {iom::DataType::F6_E3M2, 6},
        {iom::DataType::I8, 8}, {iom::DataType::U8, 8},
        {iom::DataType::F8_E4M3FN, 8}, {iom::DataType::F8_E5M2, 8}, {iom::DataType::F8_E8M0, 8},
        {iom::DataType::I16, 16}, {iom::DataType::U16, 16},
        {iom::DataType::F16, 16}, {iom::DataType::BF16, 16},
        {iom::DataType::I32, 32}, {iom::DataType::U32, 32}, {iom::DataType::F32, 32},
        {iom::DataType::I64, 64}, {iom::DataType::U64, 64}, {iom::DataType::F64, 64},
    };
    for (const auto& [data_type, bits] : bits_per_element) {
        CAPTURE(static_cast<int>(data_type));

        const iom::TensorSpec eight_elements = make_spec({1, 8}, data_type);
        CHECK_EQ(eight_elements.logical_nbytes(), bits);

        const iom::TensorSpec one_tile = make_spec({16, 16}, data_type);
        const std::size_t tile_bytes = one_tile.tiled_storage_nbytes();
        CHECK_EQ(tile_bytes, 32 * bits);
        CHECK(tile_bytes % 32 == 0);
    }
}

TEST_CASE("TensorSpec logical byte counts round sub-byte elements up to bytes") {
    const std::vector<std::tuple<std::vector<std::size_t>, iom::DataType, std::size_t>> cases = {
        {{1, 3}, iom::DataType::I2, 1},
        {{1, 5}, iom::DataType::I2, 2},
        {{1, 1}, iom::DataType::I2, 1},
        {{1, 3}, iom::DataType::I4, 2},
        {{1, 3}, iom::DataType::F6_E2M3, 3},
        {{1, 1}, iom::DataType::BOOL, 1},
        {{1, 17}, iom::DataType::F32, 68},
        {{2, 8}, iom::DataType::BF16, 32},
    };
    for (const auto& [dimensions, data_type, expected_nbytes] : cases) {
        CAPTURE(dimensions);
        CAPTURE(static_cast<int>(data_type));
        const iom::TensorSpec spec = make_spec(dimensions, data_type);
        CHECK_EQ(spec.logical_nbytes(), expected_nbytes);
    }
}

TEST_CASE("TensorSpec tiled storage byte counts cover every padded slot") {
    const iom::TensorSpec rank_two = make_spec({1, 17}, iom::DataType::F32);
    CHECK_EQ(rank_two.logical_nbytes(), 68);
    CHECK_EQ(rank_two.tiled_storage_nbytes(), 16 * 32 * 4);

    const iom::TensorSpec rank_four = make_spec({2, 8, 31, 33}, iom::DataType::BF16);
    CHECK_EQ(rank_four.tiled_storage_nbytes(), 2 * 8 * 32 * 48 * 2);
}

TEST_CASE("TensorSpec logical byte count overflows instead of wrapping") {
    const iom::TensorSpec wide = make_spec({1, kMax}, iom::DataType::F32);
    CHECK_THROWS_AS((void)wide.logical_nbytes(), std::overflow_error);

    const std::size_t odd_bits = (std::size_t{1} << 63) - 1;
    const iom::TensorSpec round_up = make_spec({1, odd_bits}, iom::DataType::I2);
    CHECK(round_up.shape.element_count() == odd_bits);
    CHECK_THROWS_AS((void)round_up.logical_nbytes(), std::overflow_error);
}

TEST_CASE("TensorSpec tiled storage byte count overflows instead of wrapping") {
    const iom::TensorSpec spec = make_spec({1, std::size_t{1} << 56}, iom::DataType::F64);
    CHECK_NOTHROW((void)spec.shape.element_count());
    CHECK_NOTHROW((void)spec.standard_padded_shape().element_count());
    CHECK_THROWS_AS((void)spec.tiled_storage_nbytes(), std::overflow_error);
}

TEST_CASE("TensorSpec validates before calculating sizes") {
    const iom::TensorSpec grouped = make_spec(
            {1, std::size_t{1} << 62}, iom::DataType::F64, iom::QuantizationFormat::GGML_Q4_0);
    CHECK_THROWS_AS((void)grouped.logical_nbytes(), std::runtime_error);
    CHECK_THROWS_AS((void)grouped.tiled_storage_nbytes(), std::runtime_error);

    const auto unknown_data_type = static_cast<iom::DataType>(200);
    const iom::TensorSpec unknown = make_spec({1, std::size_t{1} << 62}, unknown_data_type);
    CHECK_THROWS_AS((void)unknown.logical_nbytes(), std::invalid_argument);
    CHECK_THROWS_AS((void)unknown.tiled_storage_nbytes(), std::invalid_argument);
}

TEST_CASE("Standard tiled layout addresses a 32x32 matrix in 16x16 tiles") {
    const iom::TensorSpec spec = make_spec({32, 32}, iom::DataType::F16);
    CHECK_EQ((slot(spec, {0, 0})), 0);
    CHECK_EQ((slot(spec, {0, 16})), 256);
    CHECK_EQ((slot(spec, {16, 0})), 512);
    CHECK_EQ((slot(spec, {16, 16})), 768);
    CHECK_EQ((slot(spec, {0, 15})), 15);
    CHECK_EQ((slot(spec, {15, 31})), 256 + 15 * 16 + 15);
    CHECK_EQ((slot(spec, {31, 31})), 3 * 256 + 15 * 16 + 15);
}

TEST_CASE("Standard tiled layout orders tiles within multiple leading planes") {
    const iom::TensorSpec spec = make_spec({2, 3, 16, 16}, iom::DataType::I8);
    CHECK_EQ((slot(spec, {0, 0, 0, 0})), 0);
    CHECK_EQ((slot(spec, {0, 0, 15, 15})), 255);
    CHECK_EQ((slot(spec, {0, 1, 0, 0})), 256);
    CHECK_EQ((slot(spec, {0, 2, 0, 0})), 512);
    CHECK_EQ((slot(spec, {1, 0, 0, 0})), 3 * 256);
    CHECK_EQ((slot(spec, {1, 2, 15, 15})), 5 * 256 + 255);
}

TEST_CASE("Standard tiled layout handles rank above four") {
    const iom::TensorSpec rank_five = make_spec({2, 2, 2, 16, 16}, iom::DataType::I8);
    CHECK_EQ((slot(rank_five, {1, 1, 1, 0, 0})), 7 * 256);
    CHECK_EQ((slot(rank_five, {0, 0, 1, 15, 15})), 256 + 255);

    const iom::TensorSpec rank_five_padded = make_spec({2, 3, 4, 17, 33}, iom::DataType::I8);
    CHECK_EQ((slot(rank_five_padded, {1, 2, 3, 16, 32})), 143 * 256);
}

TEST_CASE("Standard tiled layout handles non-square and padded matrices") {
    const iom::TensorSpec non_square = make_spec({20, 35}, iom::DataType::I8);
    CHECK_EQ((slot(non_square, {0, 0})), 0);
    CHECK_EQ((slot(non_square, {0, 16})), 256);
    CHECK_EQ((slot(non_square, {0, 34})), 2 * 256 + 2);
    CHECK_EQ((slot(non_square, {15, 34})), 2 * 256 + 15 * 16 + 2);
    CHECK_EQ((slot(non_square, {16, 0})), 3 * 256);

    const iom::TensorSpec padded = make_spec({2, 8, 31, 33}, iom::DataType::I8);
    CHECK_EQ((slot(padded, {0, 1, 0, 0})), 1 * 6 * 256);
    CHECK_EQ((slot(padded, {1, 0, 30, 32})), 53 * 256 + 14 * 16);
}

TEST_CASE("Standard tiled layout maps every logical element to a unique slot") {
    const std::vector<std::vector<std::size_t>> cases = {
        {20, 35},
        {2, 3, 17, 33},
    };
    for (const auto& logical : cases) {
        CAPTURE(logical);
        const iom::TensorSpec spec{iom::TensorShape{logical}, iom::DataType::I2};
        const iom::TensorShape padded = spec.standard_padded_shape();
        const std::size_t padded_count = padded.element_count();
        const std::size_t logical_count = spec.shape.element_count();
        REQUIRE(padded_count >= logical_count);

        std::vector<char> seen(padded_count, 0);
        std::size_t visited = 0;
        std::vector<std::size_t> coordinates(logical.size(), 0);
        for (;;) {
            const std::size_t address = iom::detail::standard_layout_slot(spec, coordinates);
            REQUIRE(address < padded_count);
            REQUIRE(seen[address] == 0);
            seen[address] = 1;
            ++visited;

            std::size_t axis = coordinates.size();
            for (; axis > 0; --axis) {
                ++coordinates[axis - 1];
                if (coordinates[axis - 1] < logical[axis - 1]) {
                    break;
                }
                coordinates[axis - 1] = 0;
            }
            if (axis == 0) {
                break;
            }
        }
        CHECK_EQ(visited, logical_count);
    }
}

TEST_CASE("Standard tiled layout validates coordinates") {
    const iom::TensorSpec spec = make_spec({2, 3, 16, 16}, iom::DataType::F32);

    CHECK_THROWS_AS((slot(spec, {0, 0, 0})), std::invalid_argument);
    CHECK_THROWS_AS((slot(spec, {0, 0, 0, 0, 0})), std::invalid_argument);

    CHECK_THROWS_AS((slot(spec, {2, 0, 0, 0})), std::out_of_range);
    CHECK_THROWS_AS((slot(spec, {0, 3, 0, 0})), std::out_of_range);
    CHECK_THROWS_AS((slot(spec, {0, 0, 16, 0})), std::out_of_range);
    CHECK_THROWS_AS((slot(spec, {0, 0, 0, 16})), std::out_of_range);
}

TEST_CASE("Standard tiled layout validates the spec before addressing") {
    const iom::TensorSpec grouped = make_spec(
            {2, 3, 16, 16}, iom::DataType::F32, iom::QuantizationFormat::OCP_MXFP4);
    CHECK_THROWS_AS((slot(grouped, {0, 0, 0, 0})), std::runtime_error);

    const auto unknown_data_type = static_cast<iom::DataType>(200);
    const iom::TensorSpec unknown = make_spec({2, 3, 16, 16}, unknown_data_type);
    CHECK_THROWS_AS((slot(unknown, {0, 0, 0, 0})), std::invalid_argument);
}

TEST_CASE("Standard tiled layout overflows instead of wrapping") {
    const iom::TensorSpec stride_overflow = make_spec({kMax / 2 + 1, 4, 16}, iom::DataType::I8);
    CHECK_THROWS_AS((slot(stride_overflow, {kMax / 2, 0, 0})), std::overflow_error);

    const iom::TensorSpec slot_overflow =
            make_spec({3, 16, std::size_t{1} << 60}, iom::DataType::I8);
    CHECK_THROWS_AS(
            (slot(slot_overflow, {2, 15, (std::size_t{1} << 60) - 1})), std::overflow_error);
}

// ---------------------------------------------------------------------------
// Tensor owner and TensorView fixtures: a fake device and fake owner proving
// the common contract without any backend header or concrete storage.
// ---------------------------------------------------------------------------

namespace {

class FakeTensor final : public iom::Tensor {
public:
    FakeTensor(iom::TensorSpec spec, iom::Device& device)
            : iom::Tensor(std::move(spec), device) {}

    [[nodiscard]] void* storage_handle() noexcept override {
        return handle_;
    }

    void region_from_host(
            const iom::TensorView& destination,
            std::span<const std::byte> source,
            iom::RawWorkspaceView workspace) override {
        ++from_host_calls;
        from_view_offset = destination.plane_offset();
        from_view_strides.assign(
                destination.plane_strides().begin(),
                destination.plane_strides().end());
        from_workspace_owner = workspace.owner_identity();
        from_workspace_offset = workspace.offset();
        from_workspace_bytes = workspace.byte_size();
        from_bytes.assign(source.begin(), source.end());
    }

    void region_to_host(
            const iom::TensorView& source,
            std::span<std::byte> destination,
            iom::RawWorkspaceView workspace) const override {
        ++to_host_calls;
        to_view_offset = source.plane_offset();
        to_view_strides.assign(
                source.plane_strides().begin(), source.plane_strides().end());
        to_workspace_owner = workspace.owner_identity();
        to_workspace_offset = workspace.offset();
        to_workspace_bytes = workspace.byte_size();
        to_destination_size = destination.size();
    }


    int storage_ = 0;
    void use_storage_handle(void* handle) noexcept {
        handle_ = handle;
    }

    void* handle_ = &storage_;


    std::size_t from_host_calls = 0;
    std::size_t from_view_offset = 0;
    std::vector<std::size_t> from_view_strides;
    const iom::RawWorkspace* from_workspace_owner = nullptr;
    std::size_t from_workspace_offset = 0;
    std::size_t from_workspace_bytes = 0;
    std::vector<std::byte> from_bytes;

    mutable std::size_t to_host_calls = 0;
    mutable std::size_t to_view_offset = 0;
    mutable std::vector<std::size_t> to_view_strides;
    mutable const iom::RawWorkspace* to_workspace_owner = nullptr;
    mutable std::size_t to_workspace_offset = 0;
    mutable std::size_t to_workspace_bytes = 0;
    mutable std::size_t to_destination_size = 0;
};

// Deterministic deferred operation queue: submissions are recorded and the
// test thread plays the in-order worker by completing sequences through the
// common DeviceOps machinery. No threads, backends, or real work involved.
class FakeQueue final : public iom::DeviceOps {
public:
    enum class AddFailure {
        none,
        bad_alloc,
        runtime,
        internal,
        post_acceptance,
    };

    struct Submission {
        std::uint64_t sequence;
        const char* op;
    };

    struct AddViewRecord {
        iom::TensorSpec spec;
        const iom::Device* device;
        const iom::Tensor* owner;
        void* handle;
        std::size_t plane_offset;
        std::vector<std::size_t> plane_strides;
        std::vector<std::size_t> logical_plane_strides;
        bool broadcast_rows;
        bool broadcast_columns;
        bool broadcasts;
    };

    struct BinaryRecord {
        std::uint64_t sequence;
        std::array<AddViewRecord, 3> views;
        std::vector<std::size_t> result_shape;
        iom::detail::BinaryEntryRegistration entries;
        iom::detail::WorkspaceLease workspace_lease;
        bool retained_failure;
    };

    FakeQueue() = default;
    explicit FakeQueue(const iom::Device& device)
            : iom::DeviceOps(device) {}

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "fake";
    }

    using iom::DeviceOps::complete;
    using iom::DeviceOps::seek_next_sequence;

    void inject_add_failure(AddFailure failure) noexcept {
        next_add_failure_ = failure;
    }

    [[nodiscard]] const std::vector<BinaryRecord>& add_records() const noexcept {
        return add_records_;
    }

    [[nodiscard]] std::size_t registered_at(void* address) const {
        return registry_state_.registry.snapshot_for(address).size();
    }
    void finish_add(std::uint64_t sequence) {
        for (const BinaryRecord& record : add_records_) {
            if (record.sequence != sequence) {
                continue;
            }
            (void)iom::detail::release_or_invalidate_binary_entries(
                    registry_state_.registry, record.entries,
                    record.retained_failure, !record.retained_failure);
            iom::detail::complete_workspace_lease(
                    registry_state_, record.workspace_lease, true);
            complete(sequence);
            return;
        }
        throw std::invalid_argument("unknown fake ADD sequence");
    }

    // Successfully queued operations in submission order.
    std::vector<Submission> submissions;
    bool silu_aliased = false;

    // A view-less submission used to observe queue identity and sequence
    // allocation directly.
    iom::oid probe() {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "probe"});
        });
    }

protected:
    iom::oid copy_impl(const iom::TensorView& source,
                       iom::TensorView& destination) override {
        if (source.spec() != destination.spec()) {
            throw std::invalid_argument("fake copy requires identical specs");
        }
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "copy"});
        });
    }

    iom::oid binary_impl(const BinaryRequest& request) override {
        const AddFailure failure =
                std::exchange(next_add_failure_, AddFailure::none);
        switch (failure) {
            case AddFailure::bad_alloc:
                throw std::bad_alloc();
            case AddFailure::runtime:
                throw std::runtime_error("fake ADD runtime failure");
            case AddFailure::internal:
                throw 42;
            case AddFailure::none:
            case AddFailure::post_acceptance:
                break;
        }

        iom::detail::Fence fence;
        fence.invoke = [](const iom::detail::Fence&) noexcept {
            return iom::detail::FenceResult::pending();
        };
        return submit_binary(
                request, registry_state_, registry_queue_id_, fence,
                [this, failure](
                        std::uint64_t sequence, const BinaryRequest& snapshot,
                        iom::detail::BinaryEntryRegistration entries) {
                    const auto record_view =
                            [](const BinaryViewSnapshot& view) {
                                return AddViewRecord{
                                        view.spec,
                                        view.device_identity,
                                        view.owner_identity,
                                        view.native_handle,
                                        view.plane_offset,
                                        view.plane_strides,
                                        view.logical_plane_strides,
                                        view.broadcast_rows,
                                        view.broadcast_columns,
                                        view.broadcasts};
                            };
                    BinaryRecord record{
                            sequence,
                            {record_view(snapshot.lhs),
                             record_view(snapshot.rhs),
                             record_view(snapshot.out)},
                            {snapshot.result_shape.dimensions().begin(),
                             snapshot.result_shape.dimensions().end()},
                            entries,
                            snapshot.workspace_lease,
                            failure == AddFailure::post_acceptance};
                    add_records_.push_back(std::move(record));
                    submissions.push_back({sequence, "add"});
                    if (failure == AddFailure::post_acceptance) {
                        commit_failure(
                                sequence,
                                std::make_exception_ptr(std::runtime_error(
                                        "fake ADD retained failure")));
                    }
                });
    }


    iom::oid silu_impl(const iom::TensorView& x, iom::TensorView& y) override {
        silu_aliased = &x == &y;
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "silu"});
        });
    }

    iom::oid linear_impl(const iom::TensorView&, const iom::TensorView&,
                         iom::TensorView&) override {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "linear"});
        });
    }

    iom::oid rmsnorm_impl(const iom::TensorView&, iom::TensorView&,
                          const iom::TensorView&, float, size_t) override {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "rmsnorm"});
        });
    }

    iom::oid sdpa_impl(const iom::TensorView&, const iom::TensorView&,
                       const iom::TensorView&, size_t, size_t, size_t,
                       iom::TensorView&) override {
        return submit([&](std::uint64_t sequence) {
            submissions.push_back({sequence, "sdpa"});
        });
    }

private:
    iom::detail::RegistryState registry_state_;
    iom::detail::QueueId registry_queue_id_ =
            iom::detail::allocate_queue_id(registry_state_);
    std::vector<BinaryRecord> add_records_;
    AddFailure next_add_failure_ = AddFailure::none;
};

class InlineQueue final : public iom::DeviceOps {
public:
    enum class Mode {
        complete,
        complete_with_failure,
        throw_before_complete,
        complete_then_throw,
        commit_failure_then_complete,
        commit_failure_then_throw,
    };

    explicit InlineQueue(Mode mode = Mode::complete)
            : mode(mode) {}
    InlineQueue(const iom::Device& device, Mode mode = Mode::complete)
            : iom::DeviceOps(device), mode(mode) {}

    using iom::DeviceOps::commit_failure;
    using iom::DeviceOps::complete;
    using iom::DeviceOps::seek_next_sequence;
    using iom::DeviceOps::submit;

protected:
    iom::oid copy_impl(
            const iom::TensorView& source,
            iom::TensorView& destination) override {
        if (source.spec() != destination.spec()) {
            throw std::invalid_argument("inline copy requires identical specs");
        }
        return submit([this](std::uint64_t sequence) {
            ++inline_calls;
            switch (mode) {
            case Mode::complete:
                complete(sequence);
                break;
            case Mode::complete_with_failure:
                complete(sequence, std::make_exception_ptr(
                        std::runtime_error("inline boom")));
                break;
            case Mode::throw_before_complete:
                throw std::runtime_error("event_create boom");
            case Mode::complete_then_throw:
                complete(sequence);
                throw std::runtime_error("post-complete boom");
            case Mode::commit_failure_then_complete:
                commit_failure(sequence, std::make_exception_ptr(
                        std::runtime_error("post-link boom")));
                complete(sequence);
                break;
            case Mode::commit_failure_then_throw:
                commit_failure(sequence, std::make_exception_ptr(
                        std::runtime_error("post-link boom")));
                throw std::runtime_error("post-commit boom");
            }
        });
    }

public:
    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "inline";
    }

    Mode mode;
    std::size_t inline_calls = 0;
};


constexpr std::uint64_t kTokenSequenceBits = 55;
constexpr std::uint64_t kTokenSequenceMask =
        (std::uint64_t{1} << kTokenSequenceBits) - 1;
constexpr std::uint64_t kMaxSequence = kTokenSequenceMask;

std::uint8_t token_queue(iom::oid token) {
    return static_cast<std::uint8_t>(
            static_cast<std::uint64_t>(token) >> kTokenSequenceBits);
}

std::uint64_t token_sequence(iom::oid token) {
    return static_cast<std::uint64_t>(token) & kTokenSequenceMask;
}

iom::oid make_token(std::uint64_t queue_id, std::uint64_t sequence) {
    return static_cast<iom::oid>(
            (queue_id << kTokenSequenceBits) | sequence);
}


constexpr std::array kFakeDeviceSupportedDataTypes = {
        iom::DataType::BOOL,
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
        iom::DataType::F8_E8M0,
        iom::DataType::F16, iom::DataType::BF16,
        iom::DataType::F32, iom::DataType::F64,
};

constexpr std::array kFakeShrunkDeviceSupportedDataTypes = {
        iom::DataType::BOOL,
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
        iom::DataType::F8_E8M0,
        iom::DataType::F16, iom::DataType::BF16,
        iom::DataType::F32,
};

/**
 * Test raw-workspace owner. The address is arbitrary: the fake registers
 * its exact identity through the RawWorkspace base like every real owner,
 * so live/foreign/dead validation is exercised without a native arena.
 */
class FakeWorkspace final : public iom::RawWorkspace {
public:
    FakeWorkspace(iom::Device& device, void* address, std::size_t bytes)
            : iom::RawWorkspace(device, bytes), address_(address) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    void* address_;
};

class FakeDevice final : public iom::Device {

public:
    [[nodiscard]] iom::BackendKind backend_kind() const noexcept override {
        return iom::BackendKind::CPU;
    }

    [[nodiscard]] std::uint32_t backend_device() const noexcept override {
        return 3;
    }
    [[nodiscard]] std::span<const iom::DataType>
            supported_data_types() const noexcept override {
        return {kFakeDeviceSupportedDataTypes.data(),
                kFakeDeviceSupportedDataTypes.size()};
    }

    [[nodiscard]] std::unique_ptr<iom::Tensor> create_tensor(
            const iom::TensorSpec& spec) override {
        return std::make_unique<FakeTensor>(spec, *this);
    }

    [[nodiscard]] std::unique_ptr<iom::RawWorkspace> create_workspace(
            std::size_t bytes) override {
        if (bytes != 0) {
            throw std::invalid_argument(
                    "fake CPU device does not support positive raw "
                    "workspace allocation");
        }
        return std::make_unique<FakeWorkspace>(*this, nullptr, 0);
    }

    // The deterministic deferred queue stands in for the concrete backends.
    [[nodiscard]] std::unique_ptr<iom::DeviceOps> create_ops() override {
        return std::make_unique<FakeQueue>(*this);
    }
};

class FakeShrunkDevice final : public iom::Device {
public:
    [[nodiscard]] iom::BackendKind backend_kind() const noexcept override {
        return iom::BackendKind::CPU;
    }

    [[nodiscard]] std::uint32_t backend_device() const noexcept override {
        return 3;
    }

    [[nodiscard]] std::span<const iom::DataType>
            supported_data_types() const noexcept override {
        return {kFakeShrunkDeviceSupportedDataTypes.data(),
                kFakeShrunkDeviceSupportedDataTypes.size()};
    }

    [[nodiscard]] std::unique_ptr<iom::Tensor> create_tensor(
            const iom::TensorSpec& spec) override {
        return std::make_unique<FakeTensor>(spec, *this);
    }

    [[nodiscard]] std::unique_ptr<iom::RawWorkspace> create_workspace(
            std::size_t bytes) override {
        if (bytes != 0) {
            throw std::invalid_argument(
                    "fake CPU device does not support positive raw "
                    "workspace allocation");
        }
        return std::make_unique<FakeWorkspace>(*this, nullptr, 0);
    }
    [[nodiscard]] std::unique_ptr<iom::DeviceOps> create_ops() override {
        return std::make_unique<FakeQueue>(*this);
    }
};

TEST_CASE("Device::supported_data_types mutation shrinks per-driver coverage") {
    FakeDevice fake;
    FakeShrunkDevice shrunk;
    const std::span<const iom::DataType> full = fake.supported_data_types();
    const std::span<const iom::DataType> reduced =
            shrunk.supported_data_types();

    REQUIRE_EQ(full.size(), kFakeDeviceSupportedDataTypes.size());
    for (std::size_t i = 0; i < full.size(); ++i) {
        CHECK_EQ(full[i], kFakeDeviceSupportedDataTypes[i]);
    }
    REQUIRE_EQ(reduced.size(), kFakeShrunkDeviceSupportedDataTypes.size());
    for (std::size_t i = 0; i < reduced.size(); ++i) {
        CHECK_EQ(reduced[i], kFakeShrunkDeviceSupportedDataTypes[i]);
    }
    CHECK_EQ(full.size() - reduced.size(), 1);
    CHECK(std::find(
                  reduced.begin(), reduced.end(), iom::DataType::F64)
          == reduced.end());

    const std::span<const iom::DataType> full_again =
            fake.supported_data_types();
    const std::span<const iom::DataType> reduced_again =
            shrunk.supported_data_types();
    CHECK_EQ(full_again.data(), full.data());
    CHECK_EQ(full_again.size(), full.size());
    CHECK_EQ(reduced_again.data(), reduced.data());
    CHECK_EQ(reduced_again.size(), reduced.size());
}

FakeTensor make_tensor(
        iom::Device& device,
        std::vector<std::size_t> dimensions,
        iom::DataType data_type = iom::DataType::F32) {
    return FakeTensor{make_spec(std::move(dimensions), data_type), device};
}

std::vector<std::size_t> dims_of(const iom::TensorView& view) {
    return {view.spec().shape.dimensions().begin(),
            view.spec().shape.dimensions().end()};
}

std::vector<std::size_t> leading_of(const iom::TensorView& view) {
    const std::span<const std::size_t> dims = view.spec().shape.dimensions();
    return {dims.begin(), dims.end() - 2};
}

std::vector<std::size_t> strides_of(const iom::TensorView& view) {
    return {view.plane_strides().begin(), view.plane_strides().end()};
}

std::size_t view_plane(
        const iom::TensorView& view, std::span<const std::size_t> coordinates) {
    REQUIRE(coordinates.size() == view.plane_strides().size());
    std::size_t plane = view.plane_offset();
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        plane += coordinates[i] * view.plane_strides()[i];
    }
    return plane;
}

std::vector<std::vector<std::size_t>> permutations_of(std::size_t rank) {
    std::vector<std::vector<std::size_t>> result;
    std::vector<std::size_t> current(rank);
    std::iota(current.begin(), current.end(), std::size_t{0});
    do {
        result.push_back(current);
    } while (std::next_permutation(current.begin(), current.end()));
    return result;
}

// A span over a temporary initializer-list array; valid for the full
// expression that consumes it.
std::span<const std::size_t> span_of(std::initializer_list<std::size_t> values) {
    return {values.begin(), values.size()};
}

// Independent reference model of nested transforms. Every derived axis
// tracks the original leading axes it covers; selected axes are pinned to
// their chosen original coordinates. No iom code participates in the
// model's arithmetic.
struct ModelAxis {
    std::vector<std::size_t> axes;
    std::vector<std::size_t> dims;
    std::size_t scale = 1;
    std::size_t base = 0;
};

struct Model {
    std::vector<ModelAxis> axes;
    std::vector<std::pair<std::size_t, std::size_t>> fixed;
};

Model dense_model(std::span<const std::size_t> leading_dims) {
    Model model;
    for (std::size_t i = 0; i < leading_dims.size(); ++i) {
        model.axes.push_back(ModelAxis{{i}, {leading_dims[i]}, 1, 0});
    }
    return model;
}

void slice_model(
        Model& model, std::size_t axis, std::size_t first, std::size_t step) {
    model.axes[axis].base += first;
    model.axes[axis].scale *= step;
}

void select_model(Model& model, std::size_t axis, std::size_t index) {
    const ModelAxis selected = model.axes[axis];
    std::size_t remainder = selected.base + selected.scale * index;
    for (std::size_t k = selected.axes.size(); k-- > 0;) {
        model.fixed.emplace_back(selected.axes[k], remainder % selected.dims[k]);
        remainder /= selected.dims[k];
    }
    model.axes.erase(model.axes.begin() + static_cast<std::ptrdiff_t>(axis));
}

void permute_model(Model& model, std::span<const std::size_t> order) {
    std::vector<ModelAxis> reordered;
    reordered.reserve(order.size());
    for (const std::size_t axis : order) {
        reordered.push_back(model.axes[axis]);
    }
    model.axes = std::move(reordered);
}

// Merge consecutive model axes into groups, mirroring reshape_leading on a
// contiguous source: every grouped run of original axes becomes one derived
// axis. Parts must still be unscaled and unshifted, as they are directly
// after the dense reshape under test.
void reshape_model(Model& model, std::span<const std::size_t> group_sizes) {
    std::vector<ModelAxis> grouped;
    grouped.reserve(group_sizes.size());
    std::size_t next = 0;
    for (const std::size_t count : group_sizes) {
        REQUIRE(count >= 1);
        REQUIRE(next + count <= model.axes.size());
        ModelAxis merged;
        for (std::size_t k = 0; k < count; ++k) {
            const ModelAxis& part = model.axes[next + k];
            REQUIRE(part.scale == 1);
            REQUIRE(part.base == 0);
            merged.axes.insert(
                    merged.axes.end(), part.axes.begin(), part.axes.end());
            merged.dims.insert(
                    merged.dims.end(), part.dims.begin(), part.dims.end());
        }
        next += count;
        grouped.push_back(std::move(merged));
    }
    REQUIRE(next == model.axes.size());
    model.axes = std::move(grouped);
}

std::size_t model_plane(
        const Model& model,
        const std::vector<std::size_t>& dense_strides,
        std::span<const std::size_t> coordinates) {
    REQUIRE(coordinates.size() == model.axes.size());
    std::vector<std::size_t> mapped(dense_strides.size(), 0);
    std::vector<bool> pinned(dense_strides.size(), false);
    for (std::size_t j = 0; j < model.axes.size(); ++j) {
        const ModelAxis& axis = model.axes[j];
        std::size_t remainder = axis.base + axis.scale * coordinates[j];
        for (std::size_t k = axis.axes.size(); k-- > 0;) {
            mapped[axis.axes[k]] = remainder % axis.dims[k];
            pinned[axis.axes[k]] = true;
            remainder /= axis.dims[k];
        }
    }
    for (const auto& [axis, coordinate] : model.fixed) {
        mapped[axis] = coordinate;
        pinned[axis] = true;
    }
    std::size_t plane = 0;
    for (std::size_t i = 0; i < dense_strides.size(); ++i) {
        REQUIRE(pinned[i]);
        plane += mapped[i] * dense_strides[i];
    }
    return plane;
}

}  // namespace

TEST_CASE("Tensor owner constructs through a device with a stable full view") {
    static_assert(!std::is_default_constructible_v<iom::Tensor>);
    static_assert(!std::is_copy_constructible_v<iom::Tensor>);
    static_assert(!std::is_copy_assignable_v<iom::Tensor>);
    static_assert(!std::is_move_constructible_v<iom::Tensor>);
    static_assert(!std::is_move_assignable_v<iom::Tensor>);
    static_assert(std::is_abstract_v<iom::Device>);
    static_assert(std::is_copy_constructible_v<iom::TensorView>);
    static_assert(std::is_move_constructible_v<iom::TensorView>);
    static_assert(!std::is_copy_assignable_v<iom::TensorView>);
    static_assert(!std::is_move_assignable_v<iom::TensorView>);

    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 16, 16}, iom::DataType::F16);

    iom::TensorView& view = tensor.view();
    CHECK(&tensor.view() == &view);
    const iom::Tensor& frozen = tensor;
    CHECK(&frozen.view() == &view);

    CHECK(view.spec() == make_spec({2, 3, 16, 16}, iom::DataType::F16));
    CHECK_EQ(view.plane_offset(), 0);
    CHECK_EQ(strides_of(view), std::vector<std::size_t>({3, 1}));

    CHECK(view.native_handle() == tensor.storage_handle());
    CHECK(std::as_const(view).native_handle() == tensor.storage_handle());

    // Rank two has exactly one plane and an empty stride vector.
    FakeTensor rank_two = make_tensor(device, {4, 9}, iom::DataType::I8);
    CHECK(rank_two.view().plane_strides().empty());
    CHECK_EQ(rank_two.view().plane_offset(), 0);

    // Dense row-major plane strides at higher ranks.
    FakeTensor rank_five = make_tensor(device, {2, 3, 4, 5, 16, 16});
    CHECK_EQ(
            strides_of(rank_five.view()),
            std::vector<std::size_t>({60, 20, 5, 1}));

    std::unique_ptr<iom::Tensor> owned =
            device.create_tensor(make_spec({2, 3, 16, 16}, iom::DataType::F16));
    REQUIRE(owned != nullptr);
    CHECK(&owned->view() == &owned->view());
    CHECK(owned->view().native_handle() != nullptr);
}

TEST_CASE("Device owners are non-copyable and non-movable") {
    static_assert(!std::is_copy_constructible_v<FakeDevice>);
    static_assert(!std::is_copy_assignable_v<FakeDevice>);
    static_assert(!std::is_move_constructible_v<FakeDevice>);
    static_assert(!std::is_move_assignable_v<FakeDevice>);
}

TEST_CASE("Tensor owner validates its specification") {
    FakeDevice device;

    const iom::TensorSpec grouped = make_spec(
            {16, 16}, iom::DataType::F32, iom::QuantizationFormat::GGML_Q4_0);
    CHECK_THROWS_AS((FakeTensor{grouped, device}), std::runtime_error);

    const auto unknown = static_cast<iom::DataType>(77);
    CHECK_THROWS_AS(
            (FakeTensor{make_spec({16, 16}, unknown), device}),
            std::invalid_argument);
}

TEST_CASE("Rank eight owners views and broadcasts preserve addressing and mapping") {
    FakeDevice device;
    FakeQueue queue(device);

    // Rank-eight owner: six leading axes plus the two tiled matrix axes.
    const FakeTensor tensor =
            make_tensor(device, {2, 3, 4, 5, 6, 7, 17, 33});
    const iom::TensorView& full = tensor.view();
    CHECK_EQ(full.spec().shape.rank(), 8);
    CHECK_EQ(
            strides_of(full),
            std::vector<std::size_t>({2520, 840, 210, 42, 7, 1}));
    CHECK_EQ(full.plane_offset(), 0);
    CHECK_EQ(full.plane_strides().size(), 6);

    // slice keeps rank eight and adjusts the first leading axis.
    const iom::TensorView sliced = full.slice(0, 1, 1);
    CHECK_EQ(sliced.spec().shape.rank(), 8);
    CHECK(
            leading_of(sliced)
            == std::vector<std::size_t>({1, 3, 4, 5, 6, 7}));
    CHECK_EQ(sliced.plane_offset(), 2520);

    // select drops exactly one leading axis and stays inside the interval.
    const iom::TensorView selected = sliced.select(3, 2);
    CHECK_EQ(selected.spec().shape.rank(), 7);
    CHECK(
            leading_of(selected)
            == std::vector<std::size_t>({1, 3, 4, 6, 7}));
    CHECK_EQ(selected.plane_offset(), 2520 + 2 * 42);

    // permute and identity reshape keep the rank and mapping exact.
    const iom::TensorView permuted = full.permute(span_of({2, 1, 0, 3, 4, 5}));
    CHECK_EQ(permuted.spec().shape.rank(), 8);
    CHECK(
            leading_of(permuted)
            == std::vector<std::size_t>({4, 3, 2, 5, 6, 7}));
    CHECK(
            strides_of(permuted)
            == std::vector<std::size_t>({210, 840, 2520, 42, 7, 1}));
    const iom::TensorView reshaped =
            full.reshape_leading(span_of({2, 3, 4, 5, 6, 7}));
    CHECK_EQ(reshaped.spec().shape.rank(), 8);
    CHECK_EQ(strides_of(reshaped), strides_of(full));

    // A rank-eight broadcast add: a size-one leading axis on rhs, a
    // size-one row axis on rhs, and a size-one column axis on lhs all
    // broadcast to the output extent.
    FakeTensor lhs = make_tensor(device, {2, 1, 1, 1, 1, 1, 17, 1});
    FakeTensor rhs = make_tensor(device, {1, 1, 1, 1, 1, 1, 1, 33});
    FakeTensor out = make_tensor(device, {2, 1, 1, 1, 1, 1, 17, 33});
    const iom::oid token = queue.add(lhs.view(), rhs.view(), out.view());
    REQUIRE(iom::oid_is_token(token));
    const FakeQueue::BinaryRecord& record = queue.add_records().back();
    CHECK_EQ(
            record.result_shape,
            std::vector<std::size_t>({2, 1, 1, 1, 1, 1, 17, 33}));
    // lhs broadcast its single column; all six leading axes match the
    // result, so every logical plane stride stays dense.
    CHECK_EQ(
            record.views[0].logical_plane_strides,
            std::vector<std::size_t>({1, 1, 1, 1, 1, 1}));
    CHECK(record.views[0].broadcast_columns);
    CHECK_FALSE(record.views[0].broadcast_rows);
    CHECK(record.views[0].broadcasts);
    // rhs broadcast its leading extent and its single row.
    CHECK_EQ(
            record.views[1].logical_plane_strides,
            std::vector<std::size_t>({0, 1, 1, 1, 1, 1}));
    CHECK(record.views[1].broadcast_rows);
    CHECK_FALSE(record.views[1].broadcast_columns);
    CHECK(record.views[1].broadcasts);
    CHECK_FALSE(record.views[2].broadcasts);
    queue.finish_add(token_sequence(token));
    CHECK_NOTHROW(queue.wait(token));
    CHECK_NOTHROW(queue.wait(token));
}

TEST_CASE("Rank nine is rejected before allocation registration or token acceptance") {
    FakeDevice device;

    // Full-shape formation rejects the rank outright, so no owner storage,
    // registration, sequence, or token can ever exist for it.
    std::vector<std::size_t> rank_nine(9, 1);
    rank_nine[7] = 17;
    rank_nine[8] = 33;
    CHECK_THROWS_AS(iom::TensorShape{rank_nine}, std::invalid_argument);
    CHECK_THROWS_AS(
            (FakeTensor{make_spec(rank_nine, iom::DataType::F32), device}),
            std::invalid_argument);

    // A rank-increasing reshape to rank nine is rejected even when the
    // source is contiguous and the leading plane count is unchanged, and
    // leaves the source view and owner untouched.
    const FakeTensor planar = make_tensor(device, {1, 1, 1, 1, 1, 1, 16, 16});
    const iom::TensorView& source = planar.view();
    const iom::TensorView snapshot = source;
    CHECK_THROWS_AS(
            (void)source.reshape_leading(span_of({1, 1, 1, 1, 1, 1, 1})),
            std::invalid_argument);
    CHECK(source.spec() == snapshot.spec());
    CHECK_EQ(source.plane_offset(), snapshot.plane_offset());
    CHECK_EQ(source.plane_strides().size(), snapshot.plane_strides().size());
    CHECK(std::equal(
            source.plane_strides().begin(), source.plane_strides().end(),
            snapshot.plane_strides().begin()));
    CHECK(source.native_handle() == snapshot.native_handle());

    // A rank-two contiguous source reshaped to rank nine is rejected the
    // same way; the short helper span itself is not a full shape.
    const FakeTensor rank_two = make_tensor(device, {16, 16});
    const iom::TensorView rank_two_snapshot = rank_two.view();
    CHECK_THROWS_AS(
            (void)rank_two.view().reshape_leading(
                    span_of({1, 1, 1, 1, 1, 1, 1})),
            std::invalid_argument);
    CHECK(rank_two.view().spec() == rank_two_snapshot.spec());
    CHECK_EQ(rank_two.view().plane_offset(), rank_two_snapshot.plane_offset());

    // Rejection stays std::invalid_argument (never Unsupported, Overflow,
    // ResourceExhausted, or an accepted token): doctest fails on any other
    // exception type, and a subsequent valid submission starts at
    // sequence one with no consumed token or queue record.
    FakeQueue queue(device);
    FakeTensor lhs = make_tensor(device, {2, 3, 17, 33});
    FakeTensor rhs = make_tensor(device, {2, 3, 17, 33});
    FakeTensor out = make_tensor(device, {2, 3, 17, 33});
    const iom::oid token = queue.add(lhs.view(), rhs.view(), out.view());
    REQUIRE(iom::oid_is_token(token));
    CHECK_EQ(token_sequence(token), 1);
    CHECK_EQ(queue.add_records().size(), std::size_t{1});
}

TEST_CASE("TensorView accessors report owner and view state") {
    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 17, 33}, iom::DataType::I8);

    const iom::TensorView& view = tensor.view();
    CHECK(&view.device() == &device);
    CHECK(view.backend_kind() == iom::BackendKind::CPU);
    CHECK_EQ(view.backend_device(), 3);
    CHECK(view.spec() == make_spec({2, 3, 17, 33}, iom::DataType::I8));

    // Copies stay pinned to the same owner and report its storage handle.
    iom::TensorView copy = view;
    CHECK(&copy.device() == &device);
    CHECK(copy.native_handle() == tensor.storage_handle());

    iom::TensorView derived = copy.slice(0, 1, 1).select(1, 2);
    CHECK(&derived.device() == &device);
    CHECK_EQ(derived.backend_device(), 3);
    CHECK(derived.native_handle() == tensor.storage_handle());
}

TEST_CASE("TensorView slice transforms one leading dimension") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {10, 20, 16, 16});
    const iom::TensorView full = tensor.view();

    const iom::TensorView middle = full.slice(0, 3, 4);
    CHECK_EQ(leading_of(middle), std::vector<std::size_t>({4, 20}));
    CHECK_EQ(strides_of(middle), std::vector<std::size_t>({20, 1}));
    CHECK_EQ(middle.plane_offset(), 60);

    const iom::TensorView stepped = full.slice(1, 2, 5, 3);
    CHECK_EQ(leading_of(stepped), std::vector<std::size_t>({10, 5}));
    CHECK_EQ(strides_of(stepped), std::vector<std::size_t>({20, 3}));
    CHECK_EQ(stepped.plane_offset(), 2);

    const iom::TensorView nested = full.slice(0, 2, 6).slice(1, 1, 3, 2);
    CHECK_EQ(leading_of(nested), std::vector<std::size_t>({6, 3}));
    CHECK_EQ(strides_of(nested), std::vector<std::size_t>({20, 2}));
    CHECK_EQ(nested.plane_offset(), 41);

    // A full-extent unstepped slice preserves the mapping exactly.
    const iom::TensorView identity = full.slice(1, 0, 20);
    CHECK_EQ(dims_of(identity), dims_of(full));
    CHECK_EQ(strides_of(identity), strides_of(full));
    CHECK_EQ(identity.plane_offset(), 0);

    // The last selected index may sit on the final coordinate.
    const iom::TensorView tail = full.slice(0, 9, 1, 1);
    CHECK_EQ(tail.plane_offset(), 180);
    CHECK_EQ(leading_of(tail), std::vector<std::size_t>({1, 20}));
}

TEST_CASE("TensorView select erases exactly one leading axis") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {2, 3, 4, 16, 16});
    const iom::TensorView full = tensor.view();

    const iom::TensorView picked = full.select(1, 2);
    CHECK_EQ(leading_of(picked), std::vector<std::size_t>({2, 4}));
    // The remaining strides are kept, not recomputed (dense would be {4, 1}).
    CHECK_EQ(strides_of(picked), std::vector<std::size_t>({12, 1}));
    CHECK_EQ(picked.plane_offset(), 8);

    const iom::TensorView reduced = picked.select(0, 1);
    CHECK_EQ(leading_of(reduced), std::vector<std::size_t>({4}));
    CHECK_EQ(strides_of(reduced), std::vector<std::size_t>({1}));
    CHECK_EQ(reduced.plane_offset(), 20);

    const iom::TensorView plane = reduced.select(0, 3);
    CHECK_EQ(plane.spec().shape.rank(), 2);
    CHECK(plane.plane_strides().empty());
    CHECK_EQ(plane.plane_offset(), 23);

    // Earlier strides survive a stepped slice untouched.
    const iom::TensorView stepped = full.slice(1, 0, 2, 1).select(2, 1);
    CHECK_EQ(leading_of(stepped), std::vector<std::size_t>({2, 2}));
    CHECK_EQ(strides_of(stepped), std::vector<std::size_t>({12, 4}));
    CHECK_EQ(stepped.plane_offset(), 1);
}

TEST_CASE("TensorView permute reorders leading axes exactly") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {2, 3, 4, 16, 16});
    const iom::TensorView full = tensor.view();

    const std::size_t dims[] = {2, 3, 4};
    const std::size_t strides[] = {12, 4, 1};
    for (const auto& order : permutations_of(3)) {
        CAPTURE(order);
        const iom::TensorView permuted = full.permute(order);
        CHECK_EQ(permuted.plane_offset(), 0);
        std::vector<std::size_t> want_dims(3);
        std::vector<std::size_t> want_strides(3);
        for (std::size_t i = 0; i < 3; ++i) {
            want_dims[i] = dims[order[i]];
            want_strides[i] = strides[order[i]];
        }
        CHECK_EQ(leading_of(permuted), want_dims);
        CHECK_EQ(strides_of(permuted), want_strides);
    }

    // The required rank-two empty permutation is a no-op.
    const FakeTensor rank_two = make_tensor(device, {4, 9});
    const iom::TensorView nothing = rank_two.view().permute(span_of({}));
    CHECK_EQ(dims_of(nothing), std::vector<std::size_t>({4, 9}));
    CHECK(nothing.plane_strides().empty());
    CHECK_EQ(nothing.plane_offset(), 0);

    // An offset survives permuting a sliced view.
    const iom::TensorView moved = full.slice(0, 1, 1).permute(span_of({2, 1, 0}));
    CHECK_EQ(moved.plane_offset(), 12);
    CHECK_EQ(leading_of(moved), std::vector<std::size_t>({4, 3, 1}));
    CHECK_EQ(strides_of(moved), std::vector<std::size_t>({1, 4, 12}));
}

TEST_CASE("TensorView reshape_leading splits merges and edits size-one axes") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {2, 3, 4, 5, 16, 16});
    const iom::TensorView full = tensor.view();

    struct ReshapeCase {
        std::vector<std::size_t> leading;
        std::vector<std::size_t> strides;
    };
    const std::vector<ReshapeCase> cases = {
        {{2, 3, 4, 5}, {60, 20, 5, 1}},          // identity
        {{6, 20}, {20, 1}},                      // merge
        {{120}, {1}},                            // full merge
        {{12, 10}, {10, 1}},                     // merge and split
        {{4, 6, 5}, {30, 5, 1}},                 // split
        {{2, 12, 5}, {60, 5, 1}},                // interior merge
        {{1, 2, 3, 4, 5}, {120, 60, 20, 5, 1}},  // size-one insert
    };
    for (const ReshapeCase& reshape_case : cases) {
        CAPTURE(reshape_case.leading);
        const iom::TensorView reshaped = full.reshape_leading(reshape_case.leading);
        CHECK_EQ(reshaped.plane_offset(), 0);
        CHECK_EQ(leading_of(reshaped), reshape_case.leading);
        CHECK_EQ(strides_of(reshaped), reshape_case.strides);
    }

    // The offset survives reshaping.
    const iom::TensorView moved =
            full.slice(0, 1, 1).reshape_leading(span_of({60}));
    CHECK_EQ(moved.plane_offset(), 60);
    CHECK_EQ(leading_of(moved), std::vector<std::size_t>({60}));
    CHECK_EQ(strides_of(moved), std::vector<std::size_t>({1}));

    // Size-one axes never constrain their stored stride.
    const iom::TensorView padded_axis =
            full.reshape_leading(span_of({1, 120})).slice(0, 0, 1, 7);
    CHECK_EQ(strides_of(padded_axis), std::vector<std::size_t>({840, 1}));
    const iom::TensorView merged = padded_axis.reshape_leading(span_of({120}));
    CHECK_EQ(leading_of(merged), std::vector<std::size_t>({120}));
    CHECK_EQ(strides_of(merged), std::vector<std::size_t>({1}));

    // Empty leading dimensions carry plane count one across the boundary.
    const FakeTensor rank_two = make_tensor(device, {4, 9});
    const iom::TensorView empty = rank_two.view().reshape_leading(span_of({}));
    CHECK_EQ(dims_of(empty), std::vector<std::size_t>({4, 9}));
    CHECK(empty.plane_strides().empty());

    const iom::TensorView single = rank_two.view().reshape_leading(span_of({1}));
    CHECK_EQ(dims_of(single), std::vector<std::size_t>({1, 4, 9}));
    CHECK_EQ(strides_of(single), std::vector<std::size_t>({1}));

    const iom::TensorView collapsed = single.reshape_leading(span_of({}));
    CHECK_EQ(dims_of(collapsed), std::vector<std::size_t>({4, 9}));
    CHECK(collapsed.plane_strides().empty());
}

TEST_CASE("TensorView reshape_leading maps every coordinate through the source layout") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {4, 3, 2, 16, 16});

    const iom::TensorView reshaped =
            tensor.view().reshape_leading(span_of({2, 12}));
    CHECK_EQ(leading_of(reshaped), std::vector<std::size_t>({2, 12}));
    CHECK_EQ(strides_of(reshaped), std::vector<std::size_t>({12, 1}));
    CHECK_EQ(reshaped.plane_offset(), 0);

    for (std::size_t c0 = 0; c0 < 2; ++c0) {
        for (std::size_t c1 = 0; c1 < 12; ++c1) {
            const std::size_t flat = c0 * 12 + c1;
            const std::size_t d0 = flat / 6;
            const std::size_t d1 = (flat / 2) % 3;
            const std::size_t d2 = flat % 2;
            CHECK_EQ(view_plane(reshaped, span_of({c0, c1})), d0 * 6 + d1 * 2 + d2);
        }
    }

    // A split with inserted and removed size-one axes keeps the same mapping.
    const iom::TensorView split =
            tensor.view().reshape_leading(span_of({1, 4, 1, 3, 2, 1}));
    CHECK_EQ(strides_of(split), std::vector<std::size_t>({24, 6, 6, 2, 1, 1}));
    for (std::size_t c1 = 0; c1 < 4; ++c1) {
        for (std::size_t c3 = 0; c3 < 3; ++c3) {
            for (std::size_t c4 = 0; c4 < 2; ++c4) {
                const std::size_t flat = (c1 * 3 + c3) * 2 + c4;
                CHECK_EQ(
                        view_plane(split, span_of({0, c1, 0, c3, c4, 0})),
                        flat);
            }
        }
    }
}

TEST_CASE("TensorView nested transforms match an independent reference plane map") {
    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 4, 5, 16, 16});
    const std::vector<std::size_t> dense_strides = {60, 20, 5, 1};

    Model model = dense_model(span_of({2, 3, 4, 5}));

    // Reshape groups consecutive original axes: {0}, {1}, {2, 3}.
    const iom::TensorView grouped =
            tensor.view().reshape_leading(span_of({2, 3, 20}));
    reshape_model(model, span_of({1, 1, 2}));

    const iom::TensorView sliced = grouped.slice(0, 1, 1, 1);
    slice_model(model, 0, 1, 1);

    const iom::TensorView picked = sliced.select(1, 1);
    select_model(model, 1, 1);

    const iom::TensorView stepped = picked.slice(1, 2, 6, 3);
    slice_model(model, 1, 2, 3);

    const iom::TensorView view = stepped.permute(span_of({1, 0}));
    permute_model(model, span_of({1, 0}));

    CHECK_EQ(leading_of(view), std::vector<std::size_t>({6, 1}));
    CHECK_EQ(strides_of(view), std::vector<std::size_t>({3, 60}));
    CHECK_EQ(view.plane_offset(), 82);

    std::vector<std::size_t> coordinates(view.spec().shape.rank() - 2, 0);
    std::size_t visited = 0;
    for (;;) {
        CAPTURE(coordinates);
        CHECK_EQ(
                view_plane(view, coordinates),
                model_plane(model, dense_strides, coordinates));
        ++visited;

        const std::vector<std::size_t> extents = dims_of(view);
        std::size_t axis = coordinates.size();
        for (; axis > 0; --axis) {
            ++coordinates[axis - 1];
            if (coordinates[axis - 1] < extents[axis - 1]) {
                break;
            }
            coordinates[axis - 1] = 0;
        }
        if (axis == 0) {
            break;
        }
    }
    CHECK_EQ(visited, 6);
}

TEST_CASE("TensorView rejects invalid transforms") {
    FakeDevice device;
    const FakeTensor tensor = make_tensor(device, {2, 3, 4, 5, 16, 16});
    const iom::TensorView full = tensor.view();
    const FakeTensor rank_two = make_tensor(device, {4, 9});

    // Leading dimensions only: tiled dimensions and past-the-rank rejected.
    CHECK_THROWS_AS((void)full.slice(4, 0, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.slice(5, 0, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.slice(6, 0, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.select(4, 0), std::out_of_range);
    CHECK_THROWS_AS((void)full.select(6, 0), std::out_of_range);
    CHECK_THROWS_AS((void)rank_two.view().slice(0, 0, 1), std::out_of_range);
    CHECK_THROWS_AS((void)rank_two.view().select(0, 0), std::out_of_range);

    // Zero count or step.
    CHECK_THROWS_AS((void)full.slice(0, 0, 0), std::invalid_argument);
    CHECK_THROWS_AS((void)full.slice(0, 0, 1, 0), std::invalid_argument);

    // Out-of-range coordinates, including the last selected index boundary.
    CHECK_THROWS_AS((void)full.slice(0, 2, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.slice(0, 1, 2, 2), std::out_of_range);
    CHECK_THROWS_AS((void)full.slice(2, 1, 3, 2), std::out_of_range);
    CHECK_THROWS_AS((void)full.select(1, 3), std::out_of_range);
    CHECK_THROWS_AS((void)full.select(2, 4), std::out_of_range);

    // Permutations must be exact.
    CHECK_THROWS_AS((void)full.permute(span_of({0, 1, 2})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.permute(span_of({0, 1, 2, 3, 0})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.permute(span_of({0, 0, 1, 2})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.permute(span_of({0, 1, 2, 4})), std::invalid_argument);

    // Reshape rejections: zero dimensions, plane-count mismatch.
    CHECK_THROWS_AS(
            (void)full.reshape_leading(span_of({2, 0, 20})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.reshape_leading(span_of({119})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.reshape_leading(span_of({7, 18})), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.reshape_leading(span_of({})), std::invalid_argument);

    // Reshape rejections: non-contiguous sources.
    const iom::TensorView stepped = full.slice(2, 0, 2, 2);
    CHECK_THROWS_AS(
            (void)stepped.reshape_leading(span_of({120})), std::invalid_argument);
    const iom::TensorView holed = full.select(1, 0);
    CHECK_THROWS_AS(
            (void)holed.reshape_leading(span_of({60})), std::invalid_argument);
    const iom::TensorView reordered = full.permute(span_of({0, 2, 1, 3}));
    CHECK_THROWS_AS(
            (void)reordered.reshape_leading(span_of({120})), std::invalid_argument);
}

TEST_CASE("TensorView transform arithmetic overflows instead of wrapping") {
    FakeDevice device;

    // The last-index arithmetic wraps before the bounds check can run.
    const FakeTensor small = make_tensor(device, {2, 16, 16});
    CHECK_THROWS_AS(
            (void)small.view().slice(0, kMax - 1, 2, 2), std::overflow_error);

    // Offset arithmetic overflows even for in-bounds coordinates.
    const FakeTensor wide = make_tensor(device, {kMax, 16, 16, 16});
    CHECK_THROWS_AS(
            (void)wide.view().slice(0, kMax - 1, 1), std::overflow_error);
    CHECK_THROWS_AS(
            (void)wide.view().select(0, kMax - 1), std::overflow_error);

    // Plane-count products overflow on both sides of the reshape.
    const FakeTensor huge_planes = make_tensor(device, {kMax, 2, 16, 16});
    CHECK_THROWS_AS(
            (void)huge_planes.view().reshape_leading(span_of({1})),
            std::overflow_error);
    CHECK_THROWS_AS(
            (void)huge_planes.view().reshape_leading(span_of({kMax, kMax})),
            std::overflow_error);
}

TEST_CASE("TensorView host transfers delegate the logical region to the owner") {
    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 16, 16}, iom::DataType::BOOL);
    const std::size_t nbytes = tensor.view().spec().logical_nbytes();
    REQUIRE(nbytes == 2 * 3 * 16 * 16);

    std::vector<std::byte> source(nbytes);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<std::byte>(i % 2);
    }

    tensor.view().copy_from_host(source);
    CHECK_EQ(tensor.from_host_calls, 1);
    CHECK(tensor.from_bytes == source);
    CHECK_EQ(tensor.from_view_offset, 0);
    CHECK_EQ(tensor.from_view_strides, std::vector<std::size_t>({3, 1}));

    // A derived view delegates itself, not the full view.
    iom::TensorView derived = tensor.view().slice(0, 1, 1).select(1, 2);
    CHECK_EQ(derived.plane_offset(), 5);
    CHECK_EQ(strides_of(derived), std::vector<std::size_t>({3}));
    const std::size_t derived_nbytes = derived.spec().logical_nbytes();
    std::vector<std::byte> small(derived_nbytes, std::byte{1});
    FakeWorkspace workspace(
            device, reinterpret_cast<void*>(0x1000), 64);
    const iom::RawWorkspaceView workspace_view =
            workspace.view().subrange(32, 32);
    derived.copy_from_host(small, workspace_view);
    CHECK_EQ(tensor.from_host_calls, 2);
    CHECK(tensor.from_bytes == small);
    CHECK_EQ(tensor.from_view_offset, 5);
    CHECK_EQ(tensor.from_view_strides, std::vector<std::size_t>({3}));
    CHECK(tensor.from_workspace_owner == &workspace);
    CHECK_EQ(tensor.from_workspace_offset, 32);
    CHECK_EQ(tensor.from_workspace_bytes, 32);

    // Wrong byte counts are rejected before the owner hook runs.
    CHECK_THROWS_AS(
            tensor.view().copy_from_host(
                    std::span<const std::byte>(source.data(), nbytes - 1)),
            std::invalid_argument);
    CHECK_THROWS_AS(
            tensor.view().copy_from_host(
                    std::span<const std::byte>(source.data(), nbytes + 1)),
            std::invalid_argument);
    CHECK_EQ(tensor.from_host_calls, 2);

    // Non-canonical BOOL bytes are rejected before any backend write.
    std::vector<std::byte> invalid = source;
    invalid[0] = std::byte{2};
    CHECK_THROWS_AS(tensor.view().copy_from_host(invalid), std::invalid_argument);
    invalid = source;
    invalid[invalid.size() - 1] = std::byte{255};
    CHECK_THROWS_AS(tensor.view().copy_from_host(invalid), std::invalid_argument);
    CHECK_EQ(tensor.from_host_calls, 2);
    // Const reads delegate the same view metadata and workspace.
    std::vector<std::byte> sink(derived_nbytes);
    derived.copy_to_host(sink, workspace_view);
    CHECK_EQ(tensor.to_host_calls, 1);
    CHECK_EQ(tensor.to_view_offset, 5);
    CHECK_EQ(tensor.to_view_strides, std::vector<std::size_t>({3}));
    CHECK(tensor.to_workspace_owner == &workspace);
    CHECK_EQ(tensor.to_workspace_offset, 32);
    CHECK_EQ(tensor.to_workspace_bytes, 32);
    CHECK_EQ(tensor.to_destination_size, derived_nbytes);

    CHECK_THROWS_AS(
            derived.copy_to_host(
                    std::span<std::byte>(sink.data(), derived_nbytes + 1)),
            std::invalid_argument);
    CHECK_EQ(tensor.to_host_calls, 1);
}

TEST_CASE("TensorView transforms leave owner storage untouched") {
    FakeDevice device;
    FakeTensor tensor = make_tensor(device, {2, 3, 4, 5, 16, 16});

    void* handle = tensor.view().native_handle();
    const iom::TensorView& full = tensor.view();

    iom::TensorView derived = full.slice(0, 1, 1)
                                      .select(1, 2)
                                      .reshape_leading(span_of({20}))
                                      .slice(0, 3, 4, 3)
                                      .permute(span_of({0}));
    CHECK_EQ(derived.plane_offset(), 103);
    CHECK(derived.native_handle() == handle);
    CHECK(derived.native_handle() == tensor.storage_handle());

    // The full view object kept its address; no hook touched storage.
    CHECK(&tensor.view() == &full);
    CHECK_EQ(tensor.from_host_calls, 0);
    CHECK_EQ(tensor.to_host_calls, 0);

    // Copies pin the same owner and handle.
    iom::TensorView copy = derived;
    CHECK(copy.native_handle() == handle);
    CHECK(&copy.device() == &device);
}

TEST_CASE("DeviceOps view signatures are exact and view-only") {
    using iom::DeviceOps;
    using iom::TensorView;

    static_assert(std::is_same_v<iom::oid, std::int64_t>);
    static_assert(iom::to_oid(iom::OidError::InvalidArgument) == -1);
    static_assert(iom::to_oid(iom::OidError::Unsupported) == -2);
    static_assert(iom::to_oid(iom::OidError::Overflow) == -3);
    static_assert(iom::to_oid(iom::OidError::ResourceExhausted) == -4);
    static_assert(iom::to_oid(iom::OidError::DeviceError) == -5);
    static_assert(iom::to_oid(iom::OidError::InternalError) == -6);
    static_assert(iom::oid_is_error(-1));
    static_assert(iom::oid_is_token(1));
    static_assert(!iom::oid_is_error(0));
    static_assert(!iom::oid_is_token(0));
    static_assert(std::is_same_v<
        decltype(&DeviceOps::copy),
        iom::oid (DeviceOps::*)(const TensorView&, TensorView&) noexcept>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::add),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&,
                                TensorView&, iom::RawWorkspaceView) noexcept>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::mul),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&,
                                TensorView&, iom::RawWorkspaceView) noexcept>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::silu),
        iom::oid (DeviceOps::*)(const TensorView&, TensorView&) noexcept>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::linear),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&,
                                TensorView&) noexcept>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::sub),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&,
                                TensorView&, iom::RawWorkspaceView) noexcept>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::div),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&,
                                TensorView&, iom::RawWorkspaceView) noexcept>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::rmsnorm),
        iom::oid (DeviceOps::*)(const TensorView&, TensorView&,
                                const TensorView&, float, size_t) noexcept>);
    static_assert(std::is_same_v<
        decltype(&DeviceOps::sdpa),
        iom::oid (DeviceOps::*)(const TensorView&, const TensorView&,
                                const TensorView&, size_t, size_t, size_t,
                                TensorView&) noexcept>);

    // silu deliberately accepts the same window as const input and mutable
    // output; a const view is refused as an output.
    static_assert(std::is_invocable_v<
        decltype(&DeviceOps::silu), DeviceOps*, const TensorView&, TensorView&>);
    static_assert(!std::is_invocable_v<
        decltype(&DeviceOps::silu), DeviceOps*, TensorView&, const TensorView&>);

    // No Tensor operand overload survives the cutover: Tensor does not
    // convert to TensorView.
    static_assert(!std::is_invocable_v<
        decltype(&DeviceOps::copy), DeviceOps*, const iom::Tensor&, iom::Tensor&>);
    static_assert(!std::is_invocable_v<
        decltype(&DeviceOps::add),
        DeviceOps*, const iom::Tensor&, const iom::Tensor&, iom::Tensor&>);
    static_assert(!std::is_invocable_v<
        decltype(&DeviceOps::sdpa), DeviceOps*, const iom::Tensor&, const iom::Tensor&,
        const iom::Tensor&, size_t, size_t, size_t, iom::Tensor&>);

    static_assert(!std::is_copy_constructible_v<DeviceOps>);
    static_assert(!std::is_move_constructible_v<DeviceOps>);
    static_assert(!std::is_copy_assignable_v<DeviceOps>);
    static_assert(!std::is_move_assignable_v<DeviceOps>);
}

TEST_CASE("DeviceOps view signatures accept stable owner views from callers") {
    FakeDevice device;
    FakeQueue queue(device);
    FakeTensor a = make_tensor(device, {2, 3, 16, 16});
    const FakeTensor& frozen = a;
    FakeTensor b = make_tensor(device, {2, 3, 16, 16});

    CHECK(iom::oid_is_token(queue.copy(a.view(), b.view())));
    CHECK(iom::oid_is_token(queue.add(frozen.view(), a.view(), b.view())));
    CHECK(iom::oid_is_token(queue.mul(a.view(), frozen.view(), b.view())));
    CHECK(iom::oid_is_token(queue.silu(b.view(), b.view())));
    CHECK(queue.silu_aliased);
    CHECK(iom::oid_is_token(queue.linear(a.view(), b.view(), b.view())));
    CHECK(iom::oid_is_token(queue.rmsnorm(a.view(), b.view(), b.view(), 1e-6F, 1)));
    CHECK(iom::oid_is_token(queue.sdpa(a.view(), b.view(), b.view(), 2, 1, 16, b.view())));

    // Derived views are equally acceptable operands.
    const iom::TensorView selected = a.view().select(1, 2);
    iom::TensorView selected_b = b.view().select(1, 2);
    CHECK(iom::oid_is_token(queue.copy(selected, selected_b)));

    REQUIRE(queue.submissions.size() == 8);
    CHECK_EQ(queue.submissions.back().sequence, 8);
}

TEST_CASE("ADD accepts every numeric NONE leaf through immutable fake snapshots") {
    FakeDevice device;
    FakeQueue queue(device);
    std::uint64_t expected_sequence = 1;

    for (const iom::DataType type : kFakeDeviceSupportedDataTypes) {
        if (type == iom::DataType::BOOL
                || type == iom::DataType::F8_E8M0) {
            continue;
        }
        CAPTURE(static_cast<int>(type));
        FakeTensor lhs = make_tensor(device, {2, 3, 17, 33}, type);
        FakeTensor rhs = make_tensor(device, {2, 3, 17, 33}, type);
        FakeTensor out = make_tensor(device, {2, 3, 17, 33}, type);
        lhs.storage_ = 11;
        rhs.storage_ = 12;
        out.storage_ = 13;
        const void* const lhs_handle = lhs.view().native_handle();
        const void* const rhs_handle = rhs.view().native_handle();
        void* const out_handle = out.view().native_handle();

        const iom::oid token =
                queue.add(lhs.view(), rhs.view(), out.view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_EQ(token_sequence(token), expected_sequence);
        REQUIRE_EQ(queue.add_records().size(), expected_sequence);
        const FakeQueue::BinaryRecord& record = queue.add_records().back();
        CHECK_EQ(record.sequence, expected_sequence);
        CHECK(record.views[0].spec == lhs.view().spec());
        CHECK(record.views[1].spec == rhs.view().spec());
        CHECK(record.views[2].spec == out.view().spec());
        CHECK_EQ(record.views[0].handle, lhs_handle);
        CHECK_EQ(record.views[1].handle, rhs_handle);
        CHECK_EQ(record.views[2].handle, out_handle);
        CHECK_EQ(record.entries.count, 3);
        CHECK_EQ(lhs.storage_, 11);
        CHECK_EQ(rhs.storage_, 12);
        CHECK_EQ(out.storage_, 13);

        queue.finish_add(expected_sequence);
        CHECK_NOTHROW(queue.wait(token));
        CHECK_NOTHROW(queue.wait(token));
        CHECK_EQ(queue.registered_at(const_cast<void*>(lhs_handle)), 0);
        CHECK_EQ(queue.registered_at(const_cast<void*>(rhs_handle)), 0);
        CHECK_EQ(queue.registered_at(out_handle), 0);
        ++expected_sequence;
    }
    CHECK_EQ(expected_sequence, 22);
}

TEST_CASE("ADD validation preserves error precedence and rejection effects") {
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    FakeDevice device;
    FakeDevice foreign;
    FakeQueue queue(device);

    {
        FakeTensor lhs = make_tensor(device, {2, 16, 16});
        FakeTensor rhs = make_tensor(device, {2, 16, 16});
        FakeTensor out = make_tensor(device, {2, 16, 16});
        const auto unknown = static_cast<iom::DataType>(127);
        const_cast<iom::TensorSpec&>(lhs.view().spec()).data_type = unknown;
        const_cast<iom::TensorSpec&>(rhs.view().spec()).data_type = unknown;
        const_cast<iom::TensorSpec&>(out.view().spec()).data_type = unknown;
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor lhs = make_tensor(device, {2, 16, 16});
        FakeTensor rhs = make_tensor(device, {2, 16, 16});
        FakeTensor out = make_tensor(device, {2, 16, 16});
        const auto unknown = static_cast<iom::QuantizationFormat>(127);
        const_cast<iom::TensorSpec&>(lhs.view().spec()).quantization = unknown;
        const_cast<iom::TensorSpec&>(rhs.view().spec()).quantization = unknown;
        const_cast<iom::TensorSpec&>(out.view().spec()).quantization = unknown;
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor lhs = make_tensor(device, {2, 16, 16});
        FakeTensor rhs = make_tensor(device, {2, 16, 16});
        FakeTensor out = make_tensor(device, {2, 16, 16});
        for (iom::TensorView* view :
             {&lhs.view(), &rhs.view(), &out.view()}) {
            const_cast<std::size_t*>(view->spec().shape.dimensions().data())[0] =
                    0;
        }
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor lhs = make_tensor(device, {2, 16, 16});
        FakeTensor rhs = make_tensor(device, {2, 16, 16});
        FakeTensor out = make_tensor(device, {2, 16, 16});
        lhs.use_storage_handle(nullptr);
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor lhs = make_tensor(
                device, {2, 16, 16}, iom::DataType::BOOL);
        FakeTensor rhs = make_tensor(device, {2, 16, 16});
        FakeTensor out = make_tensor(device, {2, 16, 16});
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor lhs = make_tensor(device, {2, 16, 16});
        FakeTensor rhs = make_tensor(device, {2, 16, 16});
        FakeTensor out = make_tensor(device, {2, 16, 16});
        const_cast<iom::TensorSpec&>(lhs.view().spec()).quantization =
                iom::QuantizationFormat::OCP_MXFP4;
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    for (const iom::DataType excluded :
         {iom::DataType::BOOL, iom::DataType::F8_E8M0}) {
        FakeTensor lhs = make_tensor(device, {2, 16, 16}, excluded);
        FakeTensor rhs = make_tensor(device, {2, 16, 16}, excluded);
        FakeTensor out = make_tensor(device, {2, 16, 16}, excluded);
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), unsupported);
    }
    {
        FakeTensor lhs = make_tensor(device, {2, 16, 16});
        FakeTensor rhs = make_tensor(device, {2, 16, 16});
        FakeTensor out = make_tensor(device, {2, 16, 16});
        for (iom::TensorView* view :
             {&lhs.view(), &rhs.view(), &out.view()}) {
            const_cast<iom::TensorSpec&>(view->spec()).quantization =
                    iom::QuantizationFormat::GGML_Q4_0;
        }
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), unsupported);
    }
    {
        FakeTensor lhs = make_tensor(foreign, {2, 16, 16});
        FakeTensor rhs = make_tensor(device, {2, 16, 16});
        FakeTensor out = make_tensor(device, {2, 16, 16});
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor lhs = make_tensor(device, {2, 17, 33});
        FakeTensor rhs = make_tensor(device, {3, 17, 33});
        FakeTensor out = make_tensor(device, {3, 17, 33});
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor lhs = make_tensor(device, {1, 17, 33});
        FakeTensor rhs = make_tensor(device, {3, 17, 33});
        FakeTensor out = make_tensor(device, {1, 17, 33});
        CHECK_EQ(queue.add(lhs.view(), rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor owner = make_tensor(device, {3, 17, 33});
        FakeTensor rhs = make_tensor(device, {3, 17, 33});
        FakeTensor out = make_tensor(device, {3, 17, 33});
        iom::TensorView malformed = owner.view();
        const_cast<std::size_t*>(malformed.plane_strides().data())[0] = 0;
        CHECK_EQ(queue.add(malformed, rhs.view(), out.view()), invalid);

        iom::TensorView out_of_range = owner.view();
        const_cast<std::size_t*>(out_of_range.plane_strides().data())[0] = 2;
        CHECK_EQ(queue.add(out_of_range, rhs.view(), out.view()), invalid);
    }
    {
        FakeTensor lhs = make_tensor(device, {3, 17, 33});
        FakeTensor rhs = make_tensor(device, {3, 17, 33});
        FakeTensor out = make_tensor(device, {3, 17, 33});
        iom::TensorView overflowed = lhs.view();
        const_cast<std::size_t*>(overflowed.plane_strides().data())[0] =
                std::numeric_limits<std::size_t>::max();
        CHECK_EQ(
                queue.add(overflowed, rhs.view(), out.view()),
                iom::to_oid(iom::OidError::Overflow));
    }
    {
        FakeTensor owner = make_tensor(device, {2, 2, 17, 33});
        FakeTensor rhs = make_tensor(device, {2, 17, 33});
        iom::TensorView lhs = owner.view().select(0, 0);
        iom::TensorView out = owner.view().select(0, 1);
        CHECK_EQ(queue.add(lhs, rhs.view(), out), invalid);
    }
    {
        FakeTensor owner = make_tensor(device, {3, 17, 33});
        FakeTensor rhs = make_tensor(device, {3, 17, 33});
        iom::TensorView lhs = owner.view().slice(0, 0, 1);
        CHECK_EQ(queue.add(lhs, rhs.view(), owner.view()), invalid);
    }

    CHECK(queue.add_records().empty());
    CHECK(queue.submissions.empty());
    const iom::oid first = queue.probe();
    CHECK_EQ(token_sequence(first), 1);
}

TEST_CASE("ADD snapshots broadcast and transformed mappings before temporaries die") {
    FakeDevice device;
    FakeQueue queue(device);
    FakeTensor lhs = make_tensor(device, {2, 1, 3, 1, 17, 1});
    FakeTensor rhs = make_tensor(device, {1, 4, 1, 5, 1, 33});
    FakeTensor out = make_tensor(device, {2, 4, 3, 5, 17, 33});

    const iom::oid broadcast =
            queue.add(lhs.view(), rhs.view(), out.view());
    REQUIRE(iom::oid_is_token(broadcast));
    const FakeQueue::BinaryRecord& broadcast_record = queue.add_records().back();
    CHECK_EQ(
            broadcast_record.result_shape,
            std::vector<std::size_t>({2, 4, 3, 5, 17, 33}));
    CHECK_EQ(
            broadcast_record.views[0].logical_plane_strides,
            std::vector<std::size_t>({3, 0, 1, 0}));
    CHECK_EQ(
            broadcast_record.views[1].logical_plane_strides,
            std::vector<std::size_t>({0, 5, 0, 1}));
    CHECK(broadcast_record.views[0].broadcast_columns);
    CHECK(broadcast_record.views[1].broadcast_rows);
    CHECK_FALSE(broadcast_record.views[2].broadcasts);
    queue.finish_add(token_sequence(broadcast));

    FakeTensor transformed_lhs =
            make_tensor(device, {4, 3, 17, 33});
    FakeTensor transformed_rhs =
            make_tensor(device, {4, 3, 17, 33});
    FakeTensor transformed_out =
            make_tensor(device, {3, 2, 17, 33});
    iom::TensorView out_view = transformed_out.view();
    const iom::oid transformed = queue.add(
            transformed_lhs.view().slice(0, 1, 2).permute(span_of({1, 0})),
            transformed_rhs.view().slice(0, 0, 2).permute(span_of({1, 0})),
            out_view);
    REQUIRE(iom::oid_is_token(transformed));
    const FakeQueue::BinaryRecord& transformed_record =
            queue.add_records().back();
    CHECK_EQ(transformed_record.views[0].plane_offset, 3);
    CHECK_EQ(
            transformed_record.views[0].plane_strides,
            std::vector<std::size_t>({1, 3}));
    CHECK_EQ(transformed_record.views[1].plane_offset, 0);
    CHECK_EQ(
            transformed_record.views[1].plane_strides,
            std::vector<std::size_t>({1, 3}));
    CHECK_EQ(transformed_record.views[0].owner, &transformed_lhs);
    CHECK_EQ(
            transformed_record.views[0].handle,
            transformed_lhs.view().native_handle());
    queue.finish_add(token_sequence(transformed));
    CHECK_NOTHROW(queue.wait(transformed));
}

TEST_CASE("ADD lifetime registration deduplicates owners and retains failures") {
    FakeDevice device;
    {
        FakeQueue queue(device);
        FakeTensor value = make_tensor(device, {2, 17, 33});
        const iom::oid token =
                queue.add(value.view(), value.view(), value.view());
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_EQ(queue.add_records().back().entries.count, 1);
        CHECK_EQ(queue.registered_at(value.view().native_handle()), 1);
        queue.finish_add(token_sequence(token));
        CHECK_EQ(queue.registered_at(value.view().native_handle()), 0);
        CHECK_NOTHROW(queue.wait(token));
        CHECK_NOTHROW(queue.wait(token));
    }
    {
        FakeQueue queue(device);
        FakeTensor lhs = make_tensor(device, {2, 17, 33});
        FakeTensor rhs = make_tensor(device, {2, 17, 33});
        FakeTensor out = make_tensor(device, {2, 17, 33});
        const iom::oid lhs_alias =
                queue.add(out.view(), rhs.view(), out.view());
        REQUIRE(iom::oid_is_token(lhs_alias));
        REQUIRE_EQ(queue.add_records().back().entries.count, 2);
        queue.finish_add(token_sequence(lhs_alias));
        const iom::oid rhs_alias =
                queue.add(lhs.view(), out.view(), out.view());
        REQUIRE(iom::oid_is_token(rhs_alias));
        REQUIRE_EQ(queue.add_records().back().entries.count, 2);
        queue.finish_add(token_sequence(rhs_alias));
    }
    {
        FakeQueue queue(device);
        FakeTensor lhs = make_tensor(device, {2, 17, 33});
        FakeTensor rhs = make_tensor(device, {2, 17, 33});
        FakeTensor out = make_tensor(device, {2, 17, 33});
        rhs.use_storage_handle(lhs.view().native_handle());
        out.use_storage_handle(lhs.view().native_handle());
        const iom::oid token =
                queue.add(lhs.view(), rhs.view(), out.view());
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_EQ(queue.add_records().back().entries.count, 3);
        CHECK_EQ(queue.registered_at(lhs.view().native_handle()), 3);
        queue.finish_add(token_sequence(token));
        CHECK_EQ(queue.registered_at(lhs.view().native_handle()), 0);
    }
    {
        FakeQueue queue(device);
        FakeTensor lhs = make_tensor(device, {2, 17, 33});
        FakeTensor rhs = make_tensor(device, {2, 17, 33});
        FakeTensor out = make_tensor(device, {2, 17, 33});
        queue.inject_add_failure(FakeQueue::AddFailure::post_acceptance);
        const iom::oid token =
                queue.add(lhs.view(), rhs.view(), out.view());
        REQUIRE(iom::oid_is_token(token));
        queue.finish_add(token_sequence(token));
        for (int attempt = 0; attempt < 2; ++attempt) {
            CHECK_THROWS_WITH_AS(
                    queue.wait(token), "fake ADD retained failure",
                    std::runtime_error);
        }
        CHECK_EQ(queue.registered_at(lhs.view().native_handle()), 1);
        CHECK_EQ(queue.registered_at(rhs.view().native_handle()), 1);
        CHECK_EQ(queue.registered_at(out.view().native_handle()), 1);
    }
}

TEST_CASE("ADD maps pre-acceptance failures without consuming a sequence") {
    FakeDevice device;
    FakeQueue queue(device);
    FakeTensor lhs = make_tensor(device, {2, 17, 33});
    FakeTensor rhs = make_tensor(device, {2, 17, 33});
    FakeTensor out = make_tensor(device, {2, 17, 33});
    out.storage_ = 71;

    const struct {
        FakeQueue::AddFailure failure;
        iom::OidError expected;
    } failures[] = {
            {FakeQueue::AddFailure::bad_alloc,
             iom::OidError::ResourceExhausted},
            {FakeQueue::AddFailure::runtime, iom::OidError::DeviceError},
            {FakeQueue::AddFailure::internal, iom::OidError::InternalError},
    };
    for (const auto& failure : failures) {
        queue.inject_add_failure(failure.failure);
        CHECK_EQ(
                queue.add(lhs.view(), rhs.view(), out.view()),
                iom::to_oid(failure.expected));
        CHECK(queue.add_records().empty());
        CHECK(queue.submissions.empty());
        CHECK_EQ(out.storage_, 71);
        CHECK_EQ(queue.registered_at(lhs.view().native_handle()), 0);
        CHECK_EQ(queue.registered_at(rhs.view().native_handle()), 0);
        CHECK_EQ(queue.registered_at(out.view().native_handle()), 0);
    }
    const iom::oid first = queue.add(lhs.view(), rhs.view(), out.view());
    REQUIRE(iom::oid_is_token(first));
    CHECK_EQ(token_sequence(first), 1);
    queue.finish_add(1);

    FakeQueue exhausted(device);
    exhausted.seek_next_sequence(kMaxSequence + 1);
    CHECK_EQ(
            exhausted.add(lhs.view(), rhs.view(), out.view()),
            iom::to_oid(iom::OidError::Overflow));
    CHECK(exhausted.add_records().empty());
    CHECK(exhausted.submissions.empty());
}

TEST_CASE("DeviceOps queue ids lease exclusively across threads and are reused after release") {
    std::mutex live_mutex;
    std::vector<std::unique_ptr<FakeQueue>> live;
    std::atomic<int> rejections = 0;
    {
        std::vector<std::thread> builders;
        for (int builder = 0; builder < 8; ++builder) {
            builders.emplace_back([&] {
                for (;;) {
                    std::unique_ptr<FakeQueue> queue;
                    try {
                        queue = std::make_unique<FakeQueue>();
                    } catch (const std::runtime_error&) {
                        ++rejections;
                        return;
                    }
                    const std::lock_guard<std::mutex> lock(live_mutex);
                    live.push_back(std::move(queue));
                }
            });
        }
        for (std::thread& builder : builders) {
            builder.join();
        }
    }
    REQUIRE(live.size() == 255);
    CHECK_EQ(rejections.load(), 8);

    std::set<std::uint8_t> ids;
    for (const std::unique_ptr<FakeQueue>& queue : live) {
        CHECK(ids.insert(token_queue(queue->probe())).second);
    }
    REQUIRE(ids.size() == 255);

    // Releasing one live queue makes exactly its id reusable.
    const std::uint8_t released = token_queue(live.back()->probe());
    live.pop_back();
    {
        FakeQueue successor;
        CHECK_EQ(token_queue(successor.probe()), released);
        CHECK_THROWS_AS(FakeQueue{}, std::runtime_error);
    }

    // Every release is observed; an empty pool starts from a low id again.
    live.clear();
    FakeQueue fresh;
    CHECK_EQ(token_queue(fresh.probe()), 1);
}

TEST_CASE("DeviceOps queue tokens encode queue id above monotonic sequences") {
    FakeDevice device;
    FakeQueue first(device);
    FakeQueue second(device);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid first_token = first.probe();
    const iom::oid second_token = first.copy(a.view(), b.view());
    const iom::oid other_token = second.probe();

    CHECK_EQ(token_sequence(first_token), 1);
    CHECK_EQ(token_sequence(second_token), 2);
    CHECK_EQ(token_sequence(other_token), 1);
    CHECK_NE(token_queue(first_token), 0);
    CHECK_NE(token_queue(first_token), token_queue(other_token));
    CHECK_EQ(first_token, make_token(token_queue(first_token), 1));

    // A validation failure consumes no sequence.
    FakeTensor wide = make_tensor(device, {5, 8});
    CHECK_EQ(first.copy(a.view(), wide.view()),
             iom::to_oid(iom::OidError::InvalidArgument));
    REQUIRE(first.submissions.size() == 2);
    const iom::oid next = first.probe();
    CHECK_EQ(token_sequence(next), 3);

    // An identical-window copy is still a queued operation.
    const iom::oid noop = first.copy(b.view(), b.view());
    CHECK_EQ(token_sequence(noop), 4);
    CHECK_EQ(first.submissions.back().sequence, 4);
    CHECK(std::string_view(first.submissions.back().op) == "copy");
}

TEST_CASE("DeviceOps queue sequences exhaust at the 55-bit boundary") {
    FakeDevice device;
    FakeQueue queue(device);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    // The seam only moves forward and never allocates a sequence.
    CHECK_THROWS_AS(queue.seek_next_sequence(0), std::invalid_argument);
    CHECK_EQ(token_sequence(queue.probe()), 1);
    CHECK_THROWS_AS(queue.seek_next_sequence(1), std::invalid_argument);

    queue.seek_next_sequence(kMaxSequence);
    REQUIRE(queue.submissions.size() == 1);
    const iom::oid last = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(last), kMaxSequence);
    REQUIRE(queue.submissions.size() == 2);

    // Submission past the last sequence throws before queuing.
    CHECK_EQ(queue.copy(a.view(), b.view()),
             iom::to_oid(iom::OidError::Overflow));
    CHECK_EQ(queue.submissions.size(), 2);

    // The boundary submission is still waitable.
    queue.complete(kMaxSequence);
    CHECK_NOTHROW(queue.wait(last));
}

TEST_CASE("DeviceOps submit commits the sequence before queue_work runs and survives inline complete") {
    FakeDevice device;
    InlineQueue queue(device);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid first = queue.copy(a.view(), b.view());
    CHECK_EQ(queue.inline_calls, 1);
    for (int i = 0; i < 3; ++i) {
        CHECK_NOTHROW(queue.wait(first));
    }

    const iom::oid second = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(second), token_sequence(first) + 1);
    CHECK_NOTHROW(queue.wait(second));

    queue.mode = InlineQueue::Mode::complete_with_failure;
    const iom::oid failed = queue.copy(a.view(), b.view());
    for (int i = 0; i < 3; ++i) {
        bool matched = false;
        try {
            queue.wait(failed);
        } catch (const std::runtime_error& error) {
            matched = std::string_view(error.what()) == "inline boom";
        }
        CHECK(matched);
    }

    InlineQueue exhausted(device);
    FakeTensor exhausted_a = make_tensor(device, {4, 8});
    FakeTensor exhausted_b = make_tensor(device, {4, 8});
    exhausted.seek_next_sequence(kMaxSequence + 1);
    const std::size_t calls_before = exhausted.inline_calls;
    CHECK_EQ(exhausted.copy(exhausted_a.view(), exhausted_b.view()),
             iom::to_oid(iom::OidError::Overflow));
    CHECK_EQ(exhausted.inline_calls, calls_before);
}

TEST_CASE("DeviceOps submit rolls back a synchronous queue_work failure") {
    FakeDevice device;
    InlineQueue queue(device, InlineQueue::Mode::throw_before_complete);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    CHECK_EQ(queue.copy(a.view(), b.view()),
             iom::to_oid(iom::OidError::DeviceError));
    queue.mode = InlineQueue::Mode::complete;
    const iom::oid next = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(next), 1);
    CHECK_NOTHROW(queue.wait(next));
}
 
TEST_CASE("submit rollback reclaims a pending failure committed before a synchronous throw") {
    FakeDevice device;
    InlineQueue queue(device, InlineQueue::Mode::commit_failure_then_throw);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    CHECK_EQ(queue.copy(a.view(), b.view()),
             iom::to_oid(iom::OidError::DeviceError));
    queue.mode = InlineQueue::Mode::complete;
    const iom::oid next = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(next), 1);
    for (int i = 0; i < 3; ++i) {
        CHECK_NOTHROW(queue.wait(next));
    }
}

TEST_CASE("submit suppresses rollback when queue_work completes inline and then throws") {
    FakeDevice device;
    InlineQueue queue(device);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});
    const iom::oid seed = queue.copy(a.view(), b.view());
    CHECK_NOTHROW(queue.wait(seed));
    queue.mode = InlineQueue::Mode::complete_then_throw;

    CHECK_EQ(queue.copy(a.view(), b.view()),
             iom::to_oid(iom::OidError::DeviceError));
    queue.mode = InlineQueue::Mode::complete;
    const iom::oid next = queue.copy(a.view(), b.view());
    CHECK_EQ(token_sequence(next), 3);
    CHECK_NOTHROW(queue.wait(next));
}


TEST_CASE("commit_failure accepts a reserved-but-not-completed sequence after submit reservation") {
    FakeDevice device;
    InlineQueue queue(device, InlineQueue::Mode::commit_failure_then_complete);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid token = queue.copy(a.view(), b.view());
    for (int i = 0; i < 3; ++i) {
        bool matched = false;
        try {
            queue.wait(token);
        } catch (const std::runtime_error& error) {
            matched = std::string_view(error.what()) == "post-link boom";
        }
        CHECK(matched);
    }
}

TEST_CASE("commit_failure accepts a reserved sequence under concurrent submit") {
    InlineQueue queue;
    std::promise<iom::oid> second_token_promise;
    std::future<iom::oid> second_token_future = second_token_promise.get_future();

    const iom::oid first = queue.submit([&](std::uint64_t sequence) {
        std::thread second([&] {
            try {
                second_token_promise.set_value(queue.submit(
                        [](std::uint64_t) {}));
            } catch (...) {
                second_token_promise.set_exception(std::current_exception());
            }
        });

        const iom::oid second_token = second_token_future.get();
        queue.commit_failure(
                sequence,
                std::make_exception_ptr(
                        std::runtime_error("concurrent boom")));
        queue.complete(sequence);
        queue.complete(token_sequence(second_token));
        second.join();
    });

    for (int i = 0; i < 3; ++i) {
        bool matched = false;
        try {
            queue.wait(first);
        } catch (const std::runtime_error& error) {
            matched = std::string_view(error.what()) == "concurrent boom";
        }
        CHECK(matched);
    }
}

TEST_CASE("commit_failure rejects a never-submitted sequence") {
    InlineQueue queue;
    const auto failure = std::make_exception_ptr(std::runtime_error("x"));
    queue.seek_next_sequence(3);

    for (const std::uint64_t sequence : {std::uint64_t{1}, std::uint64_t{2}}) {
        bool matched = false;
        try {
            queue.commit_failure(sequence, failure);
        } catch (const std::invalid_argument& error) {
            matched = std::string_view(error.what())
                    == "retained failure is not for a reserved submission";
        }
        CHECK(matched);
    }
}

TEST_CASE("commit_failure rejects an already-completed sequence") {
    FakeDevice device;
    InlineQueue queue(device);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid token = queue.copy(a.view(), b.view());
    CHECK_NOTHROW(queue.wait(token));

    bool matched = false;
    try {
        queue.commit_failure(
                token_sequence(token),
                std::make_exception_ptr(std::runtime_error("late failure")));
    } catch (const std::invalid_argument& error) {
        matched = std::string_view(error.what())
                == "retained failure is for a sequence that has already been completed";
    }
    CHECK(matched);
}

TEST_CASE("DeviceOps queue waits reject invalid live tokens") {
    FakeDevice device;
    FakeQueue queue(device);
    FakeQueue foreign(device);
    FakeTensor a = make_tensor(device, {4, 8});
    FakeTensor b = make_tensor(device, {4, 8});

    const iom::oid own = queue.probe();
    const iom::oid other = foreign.probe();

    CHECK_THROWS_AS(queue.wait(0), std::invalid_argument);
    CHECK_THROWS_AS(queue.wait(make_token(token_queue(own), 0)), std::invalid_argument);
    CHECK_THROWS_AS(queue.wait(other), std::invalid_argument);
    CHECK_THROWS_AS(queue.wait(make_token(200, 1)), std::invalid_argument);
    CHECK_THROWS_AS(queue.wait(make_token(token_queue(own), 2)), std::invalid_argument);

    // Rejections consumed nothing and completed nothing.
    queue.complete(1);
    CHECK_NOTHROW(queue.wait(own));
    CHECK_THROWS_AS(foreign.wait(own), std::invalid_argument);
}

TEST_CASE("DeviceOps queue waits are idempotent and retain per-sequence results") {
    FakeDevice device;
    FakeQueue queue(device);
    const iom::oid one = queue.probe();
    const iom::oid two = queue.probe();
    const iom::oid three = queue.probe();
    const iom::oid four = queue.probe();

    queue.complete(1);
    CHECK_NOTHROW(queue.wait(one));
    CHECK_NOTHROW(queue.wait(one));

    // In-order completion with a retained failure at two.
    queue.complete(2, std::make_exception_ptr(std::runtime_error("async boom")));
    queue.complete(3);
    CHECK_NOTHROW(queue.wait(one));
    bool rethrown = false;
    try {
        queue.wait(two);
    } catch (const std::runtime_error& error) {
        rethrown = std::string_view(error.what()) == "async boom";
    }
    CHECK(rethrown);
    CHECK_THROWS_AS(queue.wait(two), std::runtime_error);
    CHECK_NOTHROW(queue.wait(three));
    CHECK_THROWS_AS(queue.wait(two), std::runtime_error);

    queue.complete(4);
    CHECK_NOTHROW(queue.wait(four));
}

TEST_CASE("DeviceOps queue wait wakes a blocked waiter on completion") {
    FakeDevice device;
    FakeQueue queue(device);
    const iom::oid token = queue.probe();

    std::atomic<bool> returned = false;
    auto waiter = std::async(std::launch::async, [&] {
        queue.wait(token);
        returned.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK_FALSE(returned.load());
    queue.complete(1);
    REQUIRE(waiter.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(returned.load());
}

TEST_CASE("DeviceOps queue destruction neither waits nor cancels and releases the id") {
    FakeDevice device;
    std::uint8_t released = 0;
    {
        FakeQueue queue(device);
        FakeTensor a = make_tensor(device, {4, 8});
        FakeTensor b = make_tensor(device, {4, 8});
        released = token_queue(queue.copy(a.view(), b.view()));
        // Sequence one is left incomplete: destruction must neither block
        // on it nor cancel it, and the id must return to the pool.
    }
    FakeQueue successor;
    CHECK_EQ(token_queue(successor.probe()), released);
}

namespace {

struct WorkerTask {
    std::uint64_t sequence;
    void* fence = nullptr;
};
struct CounterAction final : iom::detail::CleanupAction {
    CounterAction(bool& destructor_ran, bool& ran) noexcept
            : destructor_ran_(destructor_ran), ran_(ran) {}

    ~CounterAction() noexcept override {
        destructor_ran_ = true;
    }

    void run() noexcept override {
        ran_ = true;
    }

    [[nodiscard]] bool failed() const noexcept override {
        return false;
    }

    [[nodiscard]] std::exception_ptr failure() const noexcept override {
        return nullptr;
    }
    [[nodiscard]] bool destructor_ran() const noexcept {
        return destructor_ran_;
    }

    [[nodiscard]] bool ran() const noexcept {
        return ran_;
    }

private:
    bool& destructor_ran_;
    bool& ran_;
};

}  // namespace
static_assert(std::is_same_v<
              iom::detail::OutstandingWorkRegistry::EntrySnapshot,
              iom::detail::OutstandingWorkRegistry::Entry>);

TEST_CASE("Inline Fence copies and moves captures without allocation") {
    static_assert(
            std::is_nothrow_default_constructible_v<iom::detail::Fence>);
    static_assert(
            std::is_nothrow_copy_constructible_v<iom::detail::Fence>);
    static_assert(
            std::is_nothrow_move_constructible_v<iom::detail::Fence>);
    static_assert(
            std::is_nothrow_copy_assignable_v<iom::detail::Fence>);
    static_assert(
            std::is_nothrow_move_assignable_v<iom::detail::Fence>);

    int calls = 0;
    iom::detail::Fence original = make_test_fence(&calls);
    iom::detail::Fence copy(original);
    iom::detail::Fence moved(std::move(copy));
    iom::detail::Fence assigned;
    assigned = original;
    iom::detail::Fence move_assigned;
    move_assigned = std::move(assigned);

    CHECK(static_cast<bool>(original));
    CHECK_FALSE(static_cast<bool>(copy));
    CHECK(static_cast<bool>(moved));
    CHECK(static_cast<bool>(move_assigned));
    CHECK(original().succeeded);
    CHECK(moved().succeeded);
    CHECK(move_assigned().succeeded);
    CHECK_EQ(calls, 3);
}

TEST_CASE(
        "Inline fence snapshot copy remains invocable after worker drops "
        "its Task refcount") {
    FenceLeaseProbeResource resource;
    iom::detail::Fence builder = make_probe_fence(resource);
    CHECK_EQ(resource.refcount.load(), 2);

    iom::detail::Fence source_entry(builder);
    CHECK_EQ(resource.refcount.load(), 3);
    iom::detail::Fence destination_entry(source_entry);
    CHECK_EQ(resource.refcount.load(), 4);

    builder = iom::detail::Fence{};
    CHECK_EQ(resource.refcount.load(), 3);

    probe_fence_worker_release(resource);
    CHECK_EQ(resource.refcount.load(), 2);
    CHECK_EQ(resource.synchronize_count.load(), 1);

    source_entry = iom::detail::Fence{};
    CHECK_EQ(resource.refcount.load(), 1);

    iom::detail::Fence snapshot(destination_entry);
    CHECK_EQ(resource.refcount.load(), 2);
    destination_entry = iom::detail::Fence{};
    CHECK_EQ(resource.refcount.load(), 1);

    CHECK(snapshot().succeeded);
    CHECK_EQ(resource.synchronize_count.load(), 1);

    std::atomic<bool> first_succeeded{false};
    std::atomic<bool> second_succeeded{false};
    std::thread first([&] { first_succeeded.store(snapshot().succeeded); });
    std::thread second(
            [&] { second_succeeded.store(snapshot().succeeded); });
    first.join();
    second.join();
    CHECK(first_succeeded.load());
    CHECK(second_succeeded.load());
    CHECK_EQ(resource.synchronize_count.load(), 1);

    snapshot = iom::detail::Fence{};
    CHECK_EQ(resource.refcount.load(), 0);
    CHECK_EQ(resource.destroy_count.load(), 1);
}

TEST_CASE("StagedWorker preserves fenced callback order") {
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::string> events;
    int completions = 0;
    int fence_value = 0;

    iom::detail::StagedWorker<WorkerTask> worker(
            {
                    [&](WorkerTask& task) {
                        std::lock_guard<std::mutex> lock(mutex);
                        events.push_back("execute");
                        task.fence = &fence_value;
                    },
                    [&](void* fence) {
                        std::lock_guard<std::mutex> lock(mutex);
                        CHECK_EQ(fence, &fence_value);
                        events.push_back("fence_complete");
                    },
                    [&](void* fence) {
                        std::lock_guard<std::mutex> lock(mutex);
                        CHECK_EQ(fence, &fence_value);
                        events.push_back("fence_destroy");
                    },
                    [&](std::uint64_t sequence, std::exception_ptr failure) {
                        std::lock_guard<std::mutex> lock(mutex);
                        CHECK_EQ(sequence, 1);
                        CHECK_FALSE(failure);
                        events.push_back("complete");
                        ++completions;
                        condition.notify_all();
                    }},
            iom::detail::StagedWorker<WorkerTask>::PublishPolicy::Splice);
    worker.start();
    worker.submit_copy(WorkerTask{1});

    std::unique_lock<std::mutex> lock(mutex);
    REQUIRE(condition.wait_for(
            lock, std::chrono::seconds(2),
            [&] { return completions == 1; }));
    lock.unlock();
    worker.shutdown_and_drain();

    CHECK(events == std::vector<std::string>{
                           "execute", "fence_complete", "fence_destroy",
                           "complete"});
}

TEST_CASE("StagedWorker removes pre-link failures before sequence reuse") {
    bool fail = true;
    int completions = 0;
    iom::detail::StagedWorker<WorkerTask> worker(
            {
                    [&](WorkerTask&) {
                        if (fail) {
                            throw std::runtime_error("pre-link failure");
                        }
                    },
                    [](void*) {},
                    [](void*) {},
                    [&](std::uint64_t sequence, std::exception_ptr failure) {
                        CHECK_EQ(sequence, 1);
                        CHECK_FALSE(failure);
                        ++completions;
                    }},
            iom::detail::StagedWorker<WorkerTask>::PublishPolicy::
                    CompleteOnThrow);
    worker.start();

    CHECK_THROWS_WITH(
            worker.submit_copy(WorkerTask{1}), "pre-link failure");
    fail = false;
    CHECK_NOTHROW(worker.submit_copy(WorkerTask{1}));
    worker.shutdown_and_drain();
    CHECK_EQ(completions, 1);
}

TEST_CASE("StagedWorker drains tasks without fence completion") {
    int fence_completions = 0;
    int fence_destroys = 0;
    int completions = 0;
    iom::detail::StagedWorker<WorkerTask> worker(
            {
                    [](WorkerTask& task) {
                        if (task.sequence != 3) {
                            task.fence = task.sequence == 1
                                    ? reinterpret_cast<void*>(1)
                                    : reinterpret_cast<void*>(2);
                        }
                    },
                    [&](void*) { ++fence_completions; },
                    [&](void*) { ++fence_destroys; },
                    [&](std::uint64_t, std::exception_ptr failure) {
                        CHECK_FALSE(failure);
                        ++completions;
                    }},
            iom::detail::StagedWorker<WorkerTask>::PublishPolicy::Splice);

    worker.submit_copy(WorkerTask{1});
    worker.submit_copy(WorkerTask{2});
    worker.submit_copy(WorkerTask{3});
    worker.shutdown_and_drain();

    CHECK_EQ(fence_completions, 0);
    CHECK_EQ(fence_destroys, 2);
    CHECK_EQ(completions, 3);
}

TEST_CASE("OutstandingWorkRegistry releases exact same-address entries") {
    iom::detail::OutstandingWorkRegistry registry;
    void* address = reinterpret_cast<void*>(0x1000);
    int fence_calls = 0;
    const iom::detail::Fence fence = make_test_fence(&fence_calls);

    registry.register_entry(1, address, 1, 1, fence);
    registry.register_entry(2, address, 1, 1, fence);
    registry.register_entry(3, address, 2, 1, fence);
    registry.register_entry(4, address, 2, 1, fence);

    const auto initial = registry.snapshot_for(address);
    REQUIRE_EQ(initial.size(), 4);
    CHECK(registry.try_release_entry(1));
    CHECK(registry.try_release_entry(2));
    CHECK_FALSE(registry.try_release_entry(1));

    const auto concurrent = registry.snapshot_for(address);
    REQUIRE_EQ(concurrent.size(), 2);
    CHECK_EQ(concurrent[0].id, 3);
    CHECK_EQ(concurrent[1].id, 4);
    CHECK_EQ(fence_calls, 0);

    const std::array<iom::detail::EntryId, 2> invalidation_ids{3, 4};
    registry.invalidate_entries(invalidation_ids);
    const auto invalidated = registry.snapshot_for(address);
    REQUIRE_EQ(invalidated.size(), 2);
    CHECK(invalidated[0].state == iom::detail::EntryState::Invalidated);
    CHECK(invalidated[1].state == iom::detail::EntryState::Invalidated);
    const iom::detail::FenceResult result = invalidated[0].fence();
    CHECK_FALSE(result.succeeded);
    CHECK(result.failure != nullptr);

    const std::array<iom::detail::EntryId, 2> remaining{3, 4};
    registry.remove_entries(remaining, address);
    CHECK(registry.snapshot_for(address).empty());
}

TEST_CASE("OutstandingWorkRegistry invalidates only matching queue entries") {
    iom::detail::OutstandingWorkRegistry registry;
    void* shared_address = reinterpret_cast<void*>(0x1000);
    void* second_address = reinterpret_cast<void*>(0x2000);
    const iom::detail::Fence fence = make_test_fence(nullptr);

    registry.register_entry(1, shared_address, 1, 11, fence);
    registry.register_entry(2, shared_address, 2, 22, fence);
    registry.register_entry(3, second_address, 3, 11, fence);

    registry.invalidate_entries_for_queue(11);

    const auto shared_entries = registry.snapshot_for(shared_address);
    REQUIRE_EQ(shared_entries.size(), 2);
    CHECK(shared_entries[0].state == iom::detail::EntryState::Invalidated);
    CHECK(shared_entries[1].state == iom::detail::EntryState::Live);
    CHECK_FALSE(registry.try_release_entry(1));
    CHECK(registry.try_release_entry(2));

    const auto second_entries = registry.snapshot_for(second_address);
    REQUIRE_EQ(second_entries.size(), 1);
    CHECK(second_entries[0].state == iom::detail::EntryState::Invalidated);

    registry.remove_entry_if_present(1, shared_address);
    registry.remove_entry_if_present(3, second_address);
    CHECK(registry.snapshot_for(shared_address).empty());
    CHECK(registry.snapshot_for(second_address).empty());
}

TEST_CASE("OutstandingWorkRegistry rolls back every registry index") {
    static int fence_calls = 0;
    fence_calls = 0;
    const iom::detail::Fence fence = make_test_fence(&fence_calls);

    iom::detail::OutstandingWorkRegistry registry;
    const auto probe_address = reinterpret_cast<void*>(0x1000);
    iom_test::arm_counting();
    registry.register_entry(1, probe_address, 1, 1, fence);
    const std::size_t allocation_count = iom_test::disarm();
    REQUIRE(allocation_count > 0);
    CHECK(registry.try_release_entry(1));

    for (std::size_t ordinal = 1; ordinal <= allocation_count; ++ordinal) {
        const iom::detail::EntryId id =
                static_cast<iom::detail::EntryId>(ordinal + 1);
        void* address = reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(0x2000 + ordinal * 0x100));
        const std::uint64_t sequence = ordinal + 1;
        const iom::detail::QueueId queue_id = ordinal + 1;

        iom_test::arm_failure(ordinal);
        bool threw_bad_alloc = false;
        bool threw_unexpected = false;
        try {
            registry.register_entry(id, address, sequence, queue_id, fence);
        } catch (const std::bad_alloc&) {
            threw_bad_alloc = true;
        } catch (...) {
            threw_unexpected = true;
        }
        const std::size_t allocations_seen = iom_test::disarm();

        REQUIRE_EQ(allocations_seen, ordinal);
        REQUIRE(threw_bad_alloc);
        CHECK_FALSE(threw_unexpected);
        CHECK(registry.snapshot_for(address).empty());
        CHECK_FALSE(registry.try_release_entry(id));
        CHECK_EQ(fence_calls, 0);

        CHECK_NOTHROW(
                registry.register_entry(id, address, sequence, queue_id, fence));
        const auto registered = registry.snapshot_for(address);
        REQUIRE_EQ(registered.size(), 1);
        CHECK_EQ(registered[0].id, id);
        CHECK(registered[0].state == iom::detail::EntryState::Live);

        const std::array<iom::detail::EntryId, 1> invalidation_id{id};
        registry.invalidate_entries(invalidation_id);
        const auto invalidated = registry.snapshot_for(address);
        REQUIRE_EQ(invalidated.size(), 1);
        CHECK(invalidated[0].state == iom::detail::EntryState::Invalidated);

        registry.remove_entry_if_present(id, address);
        CHECK(registry.snapshot_for(address).empty());
    }
}

TEST_CASE(
        "release_or_quarantine releases storage only when every snapshot "
        "fence succeeds") {
    class CountingCleanup final : public iom::detail::CleanupAction {
    public:
        explicit CountingCleanup(int& run_count) noexcept
                : run_count_(run_count) {}

        void run() noexcept override {
            ++run_count_;
        }

        [[nodiscard]] bool failed() const noexcept override {
            return false;
        }

        [[nodiscard]] std::exception_ptr failure() const noexcept override {
            return nullptr;
        }

    private:
        int& run_count_;
    };

    const auto evaluate = [&](
                                  iom::detail::OutstandingWorkRegistry& registry,
                                  iom::detail::Quarantine& quarantine,
                                  void* address, bool expect_quarantine) {
        int make_calls = 0;
        int release_calls = 0;
        int cleanup_runs = 0;
        iom::detail::release_or_quarantine(
                registry, address,
                [&] {
                    ++make_calls;
                    quarantine.emplace<CountingCleanup>(cleanup_runs);
                },
                [&] { ++release_calls; });

        CHECK(registry.snapshot_for(address).empty());
        if (expect_quarantine) {
            CHECK_EQ(make_calls, 1);
            CHECK_EQ(release_calls, 0);
            CHECK_EQ(cleanup_runs, 0);
            quarantine.drain();
            CHECK_EQ(cleanup_runs, 1);
            quarantine.drain();
            CHECK_EQ(cleanup_runs, 1);
        } else {
            CHECK_EQ(make_calls, 0);
            CHECK_EQ(release_calls, 1);
            CHECK_EQ(cleanup_runs, 0);
            quarantine.drain();
            quarantine.drain();
            CHECK_EQ(cleanup_runs, 0);
        }
    };

    {
        iom::detail::OutstandingWorkRegistry registry;
        iom::detail::Quarantine quarantine;
        evaluate(
                registry, quarantine, reinterpret_cast<void*>(0x3000), false);
    }

    {
        iom::detail::OutstandingWorkRegistry registry;
        iom::detail::Quarantine quarantine;
        void* address = reinterpret_cast<void*>(0x3100);
        int fence_calls = 0;
        const iom::detail::Fence fence = make_test_fence(&fence_calls);
        registry.register_entry(1, address, 1, 1, fence);
        evaluate(registry, quarantine, address, false);
        CHECK_EQ(fence_calls, 1);
    }

    {
        iom::detail::OutstandingWorkRegistry registry;
        iom::detail::Quarantine quarantine;
        void* address = reinterpret_cast<void*>(0x3200);
        int fence_calls = 0;
        const iom::detail::Fence fence = make_test_fence(&fence_calls);
        registry.register_entry(2, address, 1, 1, fence);
        registry.register_entry(3, address, 2, 1, fence);
        evaluate(registry, quarantine, address, false);
        CHECK_EQ(fence_calls, 2);
    }

    {
        iom::detail::OutstandingWorkRegistry registry;
        iom::detail::Quarantine quarantine;
        void* address = reinterpret_cast<void*>(0x3300);
        int fence_calls = 0;
        const iom::detail::Fence fence = make_test_fence(&fence_calls);
        registry.register_entry(4, address, 1, 44, fence);
        registry.invalidate_entries_for_queue(44);
        const auto entries = registry.snapshot_for(address);
        REQUIRE_EQ(entries.size(), 1);
        CHECK(entries[0].state == iom::detail::EntryState::Invalidated);
        evaluate(registry, quarantine, address, true);
        CHECK_EQ(fence_calls, 0);
    }

    {
        iom::detail::OutstandingWorkRegistry registry;
        iom::detail::Quarantine quarantine;
        void* address = reinterpret_cast<void*>(0x3400);
        int fence_calls = 0;
        const iom::detail::Fence fence = make_test_fence(
                &fence_calls, TestFenceMode::failed);
        registry.register_entry(5, address, 1, 1, fence);
        evaluate(registry, quarantine, address, true);
        CHECK_EQ(fence_calls, 1);
    }

    {
        iom::detail::OutstandingWorkRegistry registry;
        iom::detail::Quarantine quarantine;
        void* address = reinterpret_cast<void*>(0x3500);
        int fence_calls = 0;
        const iom::detail::Fence fence = make_test_fence(
                &fence_calls, TestFenceMode::throwing);
        registry.register_entry(6, address, 1, 1, fence);
        evaluate(registry, quarantine, address, true);
        CHECK_EQ(fence_calls, 1);
    }

    {
        iom::detail::OutstandingWorkRegistry registry;
        iom::detail::Quarantine quarantine;
        void* address = reinterpret_cast<void*>(0x3600);
        int live_fence_calls = 0;
        const iom::detail::Fence live_fence =
                make_test_fence(&live_fence_calls);
        registry.register_entry(7, address, 1, 70, live_fence);
        registry.register_entry(8, address, 2, 71, live_fence);
        registry.invalidate_entries_for_queue(71);
        const auto entries = registry.snapshot_for(address);
        REQUIRE_EQ(entries.size(), 2);
        CHECK(entries[0].state == iom::detail::EntryState::Live);
        CHECK(entries[1].state == iom::detail::EntryState::Invalidated);
        evaluate(registry, quarantine, address, true);
        CHECK_EQ(live_fence_calls, 1);
    }
}

TEST_CASE(
        "register_registry_entries rolls back the source entry when the "
        "destination registration throws") {
    static int fence_calls = 0;
    fence_calls = 0;
    const iom::detail::Fence fence = make_test_fence(&fence_calls);

    iom::detail::OutstandingWorkRegistry registry;
    void* source_address = reinterpret_cast<void*>(0x3700);
    void* destination_address = reinterpret_cast<void*>(0x3800);
    iom::detail::EntryId next_entry_id = 5;
    registry.register_entry(6, destination_address, 1, 1, fence);

    CHECK_THROWS_AS(
            iom::detail::register_registry_entries(
                    registry, next_entry_id, 1, 1, source_address,
                    destination_address, fence),
            std::invalid_argument);
    CHECK(registry.snapshot_for(source_address).empty());
    CHECK_FALSE(registry.try_release_entry(5));
    CHECK_EQ(fence_calls, 0);
    const auto destination_entries = registry.snapshot_for(destination_address);
    REQUIRE_EQ(destination_entries.size(), 1);
    CHECK_EQ(destination_entries[0].id, 6);
    CHECK(destination_entries[0].state == iom::detail::EntryState::Live);

    registry.remove_entry_if_present(6, destination_address);
    next_entry_id = 5;
    const iom::detail::EntryRegistration registration =
            iom::detail::register_registry_entries(
                    registry, next_entry_id, 1, 1, source_address,
                    destination_address, fence);
    CHECK_EQ(registration.source, 5);
    CHECK_EQ(registration.destination, 6);
    CHECK_EQ(next_entry_id, 7);
    CHECK_EQ(fence_calls, 0);
    CHECK_EQ(registry.snapshot_for(source_address).size(), 1);
    CHECK_EQ(registry.snapshot_for(destination_address).size(), 1);

    registry.remove_entry_if_present(5, source_address);
    registry.remove_entry_if_present(6, destination_address);
    CHECK(registry.snapshot_for(source_address).empty());
    CHECK(registry.snapshot_for(destination_address).empty());

    next_entry_id = std::numeric_limits<iom::detail::EntryId>::max();
    CHECK_THROWS_AS(
            iom::detail::register_registry_entries(
                    registry, next_entry_id, 1, 1, source_address,
                    destination_address, fence),
            std::overflow_error);
    CHECK_EQ(
            next_entry_id,
            std::numeric_limits<iom::detail::EntryId>::max());
    CHECK(registry.snapshot_for(source_address).empty());
    CHECK(registry.snapshot_for(destination_address).empty());
}

TEST_CASE(
        "RegistryState allocates unique queue ids and entry-id pairs "
        "under concurrent threads") {
    iom::detail::RegistryState state;
    constexpr int kThreads = 8;
    constexpr int kRegistrationsPerThread = 64;
    constexpr std::size_t kRegistrations =
            static_cast<std::size_t>(kThreads)
            * kRegistrationsPerThread;

    std::atomic<std::size_t> next_address{0};
    std::atomic<bool> failed{false};
    std::array<iom::detail::QueueId, kThreads> queue_ids{};
    std::vector<iom::detail::EntryId> source_ids(kRegistrations, 0);
    std::vector<iom::detail::EntryId> destination_ids(
            kRegistrations, 0);

    const auto run_one = [&](int thread) {
        queue_ids[thread] = iom::detail::allocate_queue_id(state);
        for (int iteration = 0;
             iteration < kRegistrationsPerThread; ++iteration) {
            const std::size_t slot =
                    static_cast<std::size_t>(thread)
                    * kRegistrationsPerThread + iteration;
            const std::size_t base =
                    next_address.fetch_add(2, std::memory_order_relaxed);
            void* source_address =
                    reinterpret_cast<void*>(0x4000 + 2 * base);
            void* destination_address =
                    reinterpret_cast<void*>(0x4000 + 2 * base + 1);
            const iom::detail::EntryRegistration registration =
                    iom::detail::register_copy_entries(
                            state, queue_ids[thread],
                            static_cast<std::uint64_t>(iteration + 1),
                            source_address, destination_address,
                            make_test_fence(nullptr));
            source_ids[slot] = registration.source;
            destination_ids[slot] = registration.destination;
        }
    };

    std::barrier gate(kThreads + 1);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int thread = 0; thread < kThreads; ++thread) {
        threads.emplace_back([&, thread] {
            try {
                gate.arrive_and_wait();
                run_one(thread);
            } catch (...) {
                failed.store(true, std::memory_order_release);
            }
        });
    }
    gate.arrive_and_wait();
    for (std::thread& thread : threads) {
        thread.join();
    }
    CHECK_FALSE(failed.load(std::memory_order_acquire));

    // Queue ids on one RegistryState are unique and never skipped: the
    // counter assigns exactly 1..kThreads across the racing callers.
    std::set<iom::detail::QueueId> unique_queue_ids(
            queue_ids.begin(), queue_ids.end());
    CHECK_EQ(unique_queue_ids.size(), kThreads);
    CHECK_EQ(*unique_queue_ids.begin(), 1u);
    CHECK_EQ(*unique_queue_ids.rbegin(),
             static_cast<iom::detail::QueueId>(kThreads));

    // Every reserved source/destination pair is one atomic reservation:
    // the 2*kRegistrations entry ids are exactly 1..2*kRegistrations, each
    // pair contiguous (destination == source + 1), with no duplicate or
    // skipped id observed under the barrier-synchronized submissions.
    std::set<iom::detail::EntryId> all_ids(
            source_ids.begin(), source_ids.end());
    for (const iom::detail::EntryId id : destination_ids) {
        all_ids.insert(id);
    }
    CHECK_EQ(all_ids.size(), 2 * kRegistrations);
    CHECK_EQ(*all_ids.begin(), 1u);
    CHECK_EQ(*all_ids.rbegin(),
             static_cast<iom::detail::EntryId>(2 * kRegistrations));
    for (std::size_t slot = 0; slot < kRegistrations; ++slot) {
        CHECK_NE(source_ids[slot], 0u);
        CHECK_EQ(destination_ids[slot], source_ids[slot] + 1);
    }
}

TEST_CASE(
        "registry release tolerates the destructor-versus-completion race") {
    iom::detail::OutstandingWorkRegistry registry;
    void* completion_address = reinterpret_cast<void*>(0x3900);
    void* destructor_address = reinterpret_cast<void*>(0x3A00);
    int fence_calls = 0;
    const iom::detail::Fence fence = make_test_fence(&fence_calls);

    registry.register_entry(9, completion_address, 1, 1, fence);
    const auto completion_snapshot = registry.snapshot_for(completion_address);
    REQUIRE_EQ(completion_snapshot.size(), 1);
    CHECK(registry.try_release_entry(9));
    CHECK_NOTHROW(completion_snapshot[0].fence());
    CHECK_EQ(fence_calls, 1);
    CHECK_NOTHROW(
            registry.remove_entry_if_present(9, completion_address));
    CHECK_FALSE(registry.try_release_entry(9));
    CHECK(registry.snapshot_for(completion_address).empty());

    registry.register_entry(10, destructor_address, 2, 1, fence);
    const auto destructor_snapshot = registry.snapshot_for(destructor_address);
    REQUIRE_EQ(destructor_snapshot.size(), 1);
    CHECK_NOTHROW(destructor_snapshot[0].fence());
    CHECK_EQ(fence_calls, 2);
    CHECK_NOTHROW(
            registry.remove_entry_if_present(10, destructor_address));
    CHECK_FALSE(registry.try_release_entry(10));
    CHECK(registry.snapshot_for(destructor_address).empty());
}

TEST_CASE("Quarantine runs allocator cleanup at most once") {
    class CountingAllocator final : public iom::Allocator {
    public:
        void* alloc(std::size_t size) override {
            return ::operator new(size, std::align_val_t(32));
        }

        void free(void* buffer) override {
            ++free_calls;
            ::operator delete(buffer, std::align_val_t(32));
        }

        void reset() override {}

        int free_calls = 0;
    };

    CountingAllocator allocator;
    void* first_address = allocator.alloc(64);
    void* second_address = allocator.alloc(64);
    iom::detail::Quarantine quarantine;
    quarantine.emplace<iom::detail::AllocatorCleanupAction>(
            allocator, first_address, 64);
    quarantine.emplace<iom::detail::AllocatorCleanupAction>(
            allocator, second_address, 64);
    quarantine.drain();
    quarantine.drain();
    CHECK_EQ(allocator.free_calls, 2);
}

TEST_CASE("Quarantine::add pins the action on growth allocation failure") {
    bool leaked_destructor_ran = false;
    bool leaked_ran = false;
    auto action = std::make_unique<CounterAction>(
            leaked_destructor_ran, leaked_ran);
    CounterAction* leaked = action.get();

    iom::detail::Quarantine quarantine;
    iom_test::arm_failure(1);
    try {
        quarantine.add(std::move(action));
        FAIL("add must throw");
    } catch (const std::bad_alloc&) {
        // The failed action node is deliberately leaked so its owned
        // resources cannot be destroyed while quarantine recording fails.
    }
    CHECK_EQ(iom_test::disarm(), 1);
    CHECK(action == nullptr);
    CHECK_FALSE(leaked->destructor_ran());
    CHECK_FALSE(leaked->ran());

    bool healthy_destructor_ran = false;
    bool healthy_ran = false;
    quarantine.emplace<CounterAction>(healthy_destructor_ran, healthy_ran);
    quarantine.drain();
    quarantine.drain();
    CHECK(healthy_destructor_ran);
    CHECK(healthy_ran);
    CHECK_FALSE(leaked->destructor_ran());
    CHECK_FALSE(leaked->ran());
}

// ---------------------------------------------------------------------------
// Raw workspace ownership, views, pure queries, validation, and leases
// (leaf 05).

TEST_CASE("RawWorkspace ownership and RawWorkspaceView type traits") {
    // The owner is explicitly created, stable, and can never be copied,
    // moved, or default-constructed.
    static_assert(!std::is_default_constructible_v<iom::RawWorkspace>);
    static_assert(!std::is_copy_constructible_v<iom::RawWorkspace>);
    static_assert(!std::is_copy_assignable_v<iom::RawWorkspace>);
    static_assert(!std::is_move_constructible_v<iom::RawWorkspace>);
    static_assert(!std::is_move_assignable_v<iom::RawWorkspace>);

    // Views are values: defaultable and copy-constructible, never
    // retargetable.
    static_assert(std::is_default_constructible_v<iom::RawWorkspaceView>);
    static_assert(std::is_copy_constructible_v<iom::RawWorkspaceView>);
    static_assert(!std::is_copy_assignable_v<iom::RawWorkspaceView>);
    static_assert(!std::is_move_constructible_v<iom::RawWorkspaceView>);
    static_assert(!std::is_move_assignable_v<iom::RawWorkspaceView>);

    // No view is constructible from an arbitrary pointer or handle.
    static_assert(!std::is_constructible_v<iom::RawWorkspaceView, void*>);
    static_assert(
            !std::is_constructible_v<iom::RawWorkspaceView, const void*>);
    static_assert(!std::is_constructible_v<
                  iom::RawWorkspaceView, void*, std::size_t>);
    static_assert(!std::is_constructible_v<
                  iom::RawWorkspaceView, void*, std::size_t, std::size_t>);
}

TEST_CASE("Default workspace view is empty and zero-byte owners stay valid") {
    const iom::RawWorkspaceView empty{};
    CHECK(empty.empty());
    CHECK_EQ(empty.byte_size(), 0);
    CHECK_EQ(empty.offset(), 0);
    CHECK_EQ(empty.range_begin(), 0);
    CHECK_EQ(empty.range_end(), 0);
    CHECK(empty.owner_identity() == nullptr);
    CHECK_THROWS_AS((void)empty.device(), std::logic_error);
    CHECK_THROWS_AS((void)empty.backend_kind(), std::logic_error);
    CHECK_THROWS_AS((void)empty.subrange(0, 0), std::invalid_argument);

    FakeDevice device;
    const std::unique_ptr<iom::RawWorkspace> workspace =
            device.create_workspace(0);
    REQUIRE(workspace != nullptr);
    CHECK(workspace->empty());
    CHECK_EQ(workspace->byte_size(), 0);
    CHECK(&workspace->device() == &device);
    CHECK(workspace->backend_kind() == iom::BackendKind::CPU);
    CHECK_EQ(workspace->backend_device(), 3);

    const iom::RawWorkspaceView view = workspace->view();
    CHECK_FALSE(view.empty());
    CHECK(view.owner_identity() == workspace.get());
    CHECK_EQ(view.byte_size(), 0);
    CHECK_EQ(view.offset(), 0);
    CHECK(&view.device() == &device);
    CHECK_EQ(view.backend_device(), 3);
    CHECK(view == workspace->view());

    // Positive creation on a CPU-kind device is unsupported scratch, and
    // it manufactures no dummy storage.
    CHECK_THROWS_AS((void)device.create_workspace(1), std::invalid_argument);
    CHECK_THROWS_AS((void)device.create_workspace(32), std::invalid_argument);
}

TEST_CASE("Workspace views copy exactly and subranges stay checked") {
    FakeDevice device;
    FakeWorkspace workspace(
            device, reinterpret_cast<void*>(0x4200), 64);

    const iom::RawWorkspaceView full = workspace.view();
    const iom::RawWorkspaceView second_half = full.subrange(32, 32);
    const iom::RawWorkspaceView copy = second_half;
    CHECK(copy == second_half);
    CHECK_FALSE(copy == full);
    CHECK(copy.owner_identity() == &workspace);
    CHECK(&copy.device() == &device);
    CHECK_EQ(copy.offset(), 32);
    CHECK_EQ(copy.byte_size(), 32);
    CHECK_EQ(copy.range_begin(), 32);
    CHECK_EQ(copy.range_end(), 64);

    // Owner-absolute bounds: offset and extent must fit the owner.
    CHECK_THROWS_AS((void)full.subrange(96, 0), std::out_of_range);
    CHECK_THROWS_AS((void)full.subrange(64, 1), std::out_of_range);
    CHECK_THROWS_AS((void)full.subrange(32, 33), std::out_of_range);
    CHECK_EQ(full.subrange(64, 0).byte_size(), 0);

    // Subranges are 32-byte aligned within the owner.
    CHECK_THROWS_AS((void)full.subrange(16, 16), std::invalid_argument);
    CHECK_THROWS_AS((void)full.subrange(33, 0), std::invalid_argument);
}

TEST_CASE("Binary workspace requirement queries are pure and deterministic") {
    FakeDevice device;
    FakeDevice foreign;
    FakeQueue queue(device);
    FakeTensor lhs = make_tensor(device, {2, 3, 17, 33});
    FakeTensor rhs = make_tensor(device, {2, 3, 17, 33});
    FakeTensor out = make_tensor(device, {2, 3, 17, 33});

    const iom::WorkspaceRequirements zero{0, 1};
    CHECK(queue.add_workspace_requirements(
                  lhs.view(), rhs.view(), out.view()) == zero);
    CHECK(queue.mul_workspace_requirements(
                  lhs.view(), rhs.view(), out.view()) == zero);
    CHECK(queue.sub_workspace_requirements(
                  lhs.view(), rhs.view(), out.view()) == zero);
    CHECK(queue.div_workspace_requirements(
                  lhs.view(), rhs.view(), out.view()) == zero);

    // Queries repeat identically and never depend on any queue state.
    CHECK(queue.add_workspace_requirements(
                  lhs.view(), rhs.view(), out.view()) == zero);

    // Validation runs exactly as for the corresponding binary operation.
    FakeTensor foreign_lhs = make_tensor(foreign, {2, 3, 17, 33});
    CHECK_THROWS_AS((void)
            queue.add_workspace_requirements(
                    foreign_lhs.view(), rhs.view(), out.view()),
            std::invalid_argument);
    FakeTensor bool_lhs =
            make_tensor(device, {2, 16, 16}, iom::DataType::BOOL);
    FakeTensor bool_rhs =
            make_tensor(device, {2, 16, 16}, iom::DataType::BOOL);
    FakeTensor bool_out =
            make_tensor(device, {2, 16, 16}, iom::DataType::BOOL);
    CHECK_THROWS_AS((void)
            queue.add_workspace_requirements(
                    bool_lhs.view(), bool_rhs.view(), bool_out.view()),
            std::runtime_error);
    FakeTensor mismatched = make_tensor(device, {2, 3, 16, 16});
    CHECK_THROWS_AS((void)
            queue.add_workspace_requirements(
                    lhs.view(), mismatched.view(), out.view()),
            std::invalid_argument);

    // No submission, no record, no registration, no sequence, no lease.
    CHECK(queue.submissions.empty());
    CHECK(queue.add_records().empty());
    CHECK_EQ(queue.registered_at(lhs.view().native_handle()), 0);
    CHECK_EQ(queue.registered_at(rhs.view().native_handle()), 0);
    CHECK_EQ(queue.registered_at(out.view().native_handle()), 0);
}

TEST_CASE(
        "Host transfer workspace requirement queries are pure and "
        "deterministic") {
    FakeDevice device;
    const FakeTensor tensor =
            make_tensor(device, {2, 3, 16, 16}, iom::DataType::F32);
    const iom::WorkspaceRequirements zero{0, 1};
    CHECK(tensor.view().copy_from_host_workspace_requirements() == zero);
    CHECK(tensor.view().copy_to_host_workspace_requirements() == zero);

    // Checked logical-byte arithmetic propagates overflow.
    const FakeTensor huge =
            make_tensor(device, {1, kMax}, iom::DataType::I8);
    CHECK_THROWS_AS(
            (void)huge.view().copy_from_host_workspace_requirements(),
            std::overflow_error);
    CHECK_THROWS_AS(
            (void)huge.view().copy_to_host_workspace_requirements(),
            std::overflow_error);

    // The queries never touch the owner or the host transfer path.
    const void* handle = tensor.view().native_handle();
    (void)tensor.view().copy_from_host_workspace_requirements();
    (void)tensor.view().copy_to_host_workspace_requirements();
    CHECK(tensor.view().native_handle() == handle);
}

TEST_CASE(
        "Shared workspace validation rejects foreign dead undersized "
        "misaligned and overlapping views") {
    FakeDevice device;
    FakeDevice foreign;
    FakeTensor operand = make_tensor(device, {2, 17, 33});
    const std::vector<iom::TensorView> operands{operand.view()};

    // A positive requirement rejects the empty default view; a zero
    // requirement accepts any view value.
    CHECK_THROWS_AS(
            (void)iom::detail::WorkspaceValidation::validated(
                    device, iom::RawWorkspaceView{}, 32, 32, {}),
            std::invalid_argument);
    CHECK_NOTHROW((void)iom::detail::WorkspaceValidation::validated(
            device, iom::RawWorkspaceView{}, 0, 1, {}));

    FakeWorkspace workspace(
            device, reinterpret_cast<void*>(0x4300), 64);
    const iom::RawWorkspaceView full = workspace.view();

    // Live, correctly sized, aligned, and disjoint from every operand.
    CHECK(iom::detail::WorkspaceValidation::validated(
                  device, full, 64, 32, operands) == full);
    CHECK(iom::detail::WorkspaceValidation::validated(
                  device, full, 64, 32, {}) == full);

    // Foreign device and dead owner.
    CHECK_THROWS_AS(
            (void)iom::detail::WorkspaceValidation::validated(
                    foreign, full, 32, 32, {}),
            std::invalid_argument);
    const auto make_dead_view = [](iom::Device& owner) {
        FakeWorkspace dead(owner, reinterpret_cast<void*>(0x4400), 32);
        return dead.view();
    };
    const iom::RawWorkspaceView dead_view = make_dead_view(device);
    CHECK_THROWS_AS(
            (void)iom::detail::WorkspaceValidation::validated(
                    device, dead_view, 16, 32, {}),
            std::invalid_argument);

    // Undersized and misaligned ranges.
    FakeWorkspace small(device, reinterpret_cast<void*>(0x4500), 32);
    CHECK_THROWS_AS(
            (void)iom::detail::WorkspaceValidation::validated(
                    device, small.view(), 64, 32, {}),
            std::invalid_argument);
    FakeWorkspace misaligned(
            device, reinterpret_cast<void*>(0x4501), 64);
    CHECK_THROWS_AS(
            (void)iom::detail::WorkspaceValidation::validated(
                    device, misaligned.view(), 64, 32, {}),
            std::invalid_argument);

    // Workspace range overlapping operand storage: the fake operand is
    // pointed at a properly aligned buffer so the overlap branch (not the
    // alignment gate) produces the rejection.
    FakeTensor overlapping_operand = make_tensor(device, {2, 17, 33});
    void* overlap_buffer = ::operator new(256, std::align_val_t(32));
    overlapping_operand.use_storage_handle(overlap_buffer);
    FakeWorkspace overlapping(device, overlap_buffer, 64);
    const std::vector<iom::TensorView> overlap_operands{
            overlapping_operand.view()};
    CHECK_THROWS_AS(
            (void)iom::detail::WorkspaceValidation::validated(
                    device, overlapping.view(), 64, 32, overlap_operands),
            std::invalid_argument);
    CHECK_NOTHROW((void)iom::detail::WorkspaceValidation::validated(
            device, overlapping.view(), 64, 32, {}));
    ::operator delete(overlap_buffer, std::align_val_t(32));
}

TEST_CASE(
        "Workspace leases are exclusive transactional and controlled by "
        "completion proofs") {
    iom::detail::RegistryState state;
    const auto* owner = reinterpret_cast<const void*>(0x3100);
    const auto* other_owner = reinterpret_cast<const void*>(0x3101);
    void* base = reinterpret_cast<void*>(0x3200);
    int fence_calls = 0;
    const iom::detail::Fence fence = make_test_fence(&fence_calls);

    const iom::detail::WorkspaceLease first = iom::detail::
            acquire_workspace_lease(state, owner, base, 64, 1, 1, fence);
    CHECK(first.entry_id != 0);
    CHECK_EQ(first.sequence, 1);
    REQUIRE_EQ(state.workspace_leases.leases.size(), 1);
    REQUIRE_EQ(state.registry.snapshot_for(base).size(), 1);

    // Overlapping leases on the same owner are resource exhaustion; the
    // rejected acquisitions leave no state behind.
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, base, 64, 2, 1, fence),
            std::bad_alloc);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, base, 32, 2, 1, fence),
            std::bad_alloc);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, static_cast<char*>(base) + 32, 32, 2, 1,
                    fence),
            std::bad_alloc);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, static_cast<char*>(base) + 16, 32, 2, 1,
                    fence),
            std::invalid_argument);
    CHECK_EQ(state.workspace_leases.leases.size(), 1);

    // Disjoint aligned subranges of one owner coexist, and another
    // owner's identical range is independent.
    CHECK_NOTHROW((void)iom::detail::acquire_workspace_lease(
            state, owner, static_cast<char*>(base) + 64, 32, 2, 1, fence));
    CHECK_NOTHROW((void)iom::detail::acquire_workspace_lease(
            state, other_owner, static_cast<char*>(base) + 256, 64, 3, 1,
            fence));
    CHECK_EQ(state.workspace_leases.leases.size(), 3);

    // Malformed requests are rejected before any state is touched.
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, nullptr, base, 64, 4, 1, fence),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, nullptr, 64, 4, 1, fence),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, base, 0, 4, 1, fence),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, static_cast<char*>(base) + 1, 64, 4, 1,
                    fence),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, base, 64, 0, 1, fence),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, base, 64, 4, 0, fence),
            std::invalid_argument);
    const iom::detail::Fence empty_fence{};
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, base, 64, 4, 1, empty_fence),
            std::invalid_argument);
    CHECK_EQ(state.workspace_leases.leases.size(), 3);

    // A covering completion proof releases the lease and its registry
    // entry, so the range can be leased again.
    iom::detail::complete_workspace_lease(state, first, true);
    CHECK_EQ(state.workspace_leases.leases.size(), 2);
    CHECK(state.registry.snapshot_for(base).empty());
    const iom::detail::WorkspaceLease reacquired = iom::detail::
            acquire_workspace_lease(state, owner, base, 64, 5, 1, fence);
    CHECK_EQ(state.workspace_leases.leases.size(), 3);

    // Runtime failure alone is not proof: the range is quarantined, its
    // covering entry invalidated, and no overlapping reuse is possible.
    iom::detail::complete_workspace_lease(state, reacquired, false);
    REQUIRE_EQ(state.registry.snapshot_for(base).size(), 1);
    CHECK(state.registry.snapshot_for(base)[0].state
          == iom::detail::EntryState::Invalidated);
    CHECK_THROWS_AS(
            (void)iom::detail::acquire_workspace_lease(
                    state, owner, base, 64, 6, 1, fence),
            std::bad_alloc);
    CHECK_EQ(state.workspace_leases.leases.size(), 3);

    // A later covering proof resolves the quarantine.
    iom::detail::complete_workspace_lease(state, reacquired, true);
    CHECK_EQ(state.workspace_leases.leases.size(), 2);
    CHECK(state.registry.snapshot_for(base).empty());
    CHECK_NOTHROW((void)iom::detail::acquire_workspace_lease(
            state, owner, base, 64, 7, 1, fence));

    // The device-side retention query follows the same lifecycle.
    const iom::detail::WorkspaceLease retained_lease = iom::detail::
            acquire_workspace_lease(
                    state, owner, static_cast<char*>(base) + 128, 64, 8, 1,
                    fence);
    CHECK(iom::detail::workspace_range_retained(
            state.workspace_leases, owner,
            static_cast<char*>(base) + 128, 64));
    CHECK_FALSE(iom::detail::workspace_range_retained(
            state.workspace_leases, owner,
            static_cast<char*>(base) + 192, 64));
    CHECK_FALSE(iom::detail::workspace_range_retained(
            state.workspace_leases, other_owner,
            static_cast<char*>(base) + 128, 64));
    iom::detail::complete_workspace_lease(state, retained_lease, true);
    CHECK_FALSE(iom::detail::workspace_range_retained(
            state.workspace_leases, owner,
            static_cast<char*>(base) + 128, 64));
}

TEST_CASE(
        "Workspace lease acquisition rolls back every trace on failure") {
    iom::detail::RegistryState state;
    const auto* owner = reinterpret_cast<const void*>(0x3300);
    void* base = reinterpret_cast<void*>(0x3400);
    int fence_calls = 0;
    const iom::detail::Fence fence = make_test_fence(&fence_calls);

    iom_test::arm_counting();
    const iom::detail::WorkspaceLease probe = iom::detail::
            acquire_workspace_lease(state, owner, base, 64, 1, 1, fence);
    const std::size_t allocation_count = iom_test::disarm();
    REQUIRE(allocation_count > 0);
    iom::detail::complete_workspace_lease(state, probe, true);

    for (std::size_t ordinal = 1; ordinal <= allocation_count; ++ordinal) {
        iom_test::arm_failure(ordinal);
        bool threw_bad_alloc = false;
        bool succeeded = false;
        iom::detail::WorkspaceLease attempt;
        try {
            attempt = iom::detail::acquire_workspace_lease(
                    state, owner, base, 64, 2, 1, fence);
            succeeded = true;
        } catch (const std::bad_alloc&) {
            threw_bad_alloc = true;
        }
        (void)iom_test::disarm();
        if (succeeded) {
            // The injected failure landed outside this transaction
            // (allocation counts shifted with registry growth); release
            // the lease so the loop keeps its precondition.
            iom::detail::complete_workspace_lease(state, attempt, true);
            continue;
        }
        CHECK(threw_bad_alloc);
        // Neither a lease record nor an owner registration survives.
        CHECK(state.workspace_leases.leases.empty());
        CHECK(state.registry.snapshot_for(base).empty());
    }

    // A clean acquisition still succeeds after every rolled-back attempt.
    CHECK_NOTHROW((void)iom::detail::acquire_workspace_lease(
            state, owner, base, 64, 3, 1, fence));
}
