#include <doctest/doctest.h>

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "iom/alloc.hpp"
#include "iom/cuda/device.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace {

class UnusedAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t) override {
        ++allocations;
        return nullptr;
    }

    void free(void*) override { ++frees; }

    void reset() override { ++resets; }

    std::size_t allocations = 0;
    std::size_t frees = 0;
    std::size_t resets = 0;
};

}  // namespace

TEST_CASE("CUDA factory reports a live hardware device and owns its context") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    UnusedAllocator allocator;
    {
        auto device = iom::make_cuda_device(0, allocator);
        REQUIRE(device != nullptr);
        CHECK(device->backend_kind() == iom::BackendKind::CUDA);
        CHECK(device->backend_device() == 0);

        CUcontext current_context = nullptr;
        REQUIRE(cuCtxGetCurrent(&current_context) == CUDA_SUCCESS);
        CHECK(current_context != nullptr);

        const iom::TensorSpec spec{
                iom::TensorShape{{16, 16}}, iom::DataType::F32};
        CHECK_THROWS_AS((void)device->create_tensor(spec), std::runtime_error);
        CHECK_THROWS_AS((void)device->create_ops(), std::runtime_error);
        CHECK(allocator.allocations == 0);
        CHECK(allocator.frees == 0);
    }

    auto recreated = iom::make_cuda_device(0, allocator);
    REQUIRE(recreated != nullptr);
    CHECK(recreated->backend_kind() == iom::BackendKind::CUDA);
    CHECK(recreated->backend_device() == 0);
    CHECK(allocator.allocations == 0);
    CHECK(allocator.frees == 0);
}

TEST_CASE("CUDA factory rejects the first unavailable ordinal") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    UnusedAllocator allocator;
    CHECK_THROWS_AS(
            (void)iom::make_cuda_device(
                    static_cast<std::uint32_t>(device_count), allocator),
            std::invalid_argument);
    CHECK(allocator.allocations == 0);
    CHECK(allocator.frees == 0);
}
