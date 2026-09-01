#include <doctest/doctest.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <new>

#include "iom/alloc.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/rocm/device.hpp"
#include "iom/tensor.hpp"

namespace {

class UnusedAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        ++allocations;
        void* block = nullptr;
        if (hipMalloc(&block, size) != hipSuccess) {
            throw std::bad_alloc();
        }
        return block;
    }

    void free(void* buffer) override {
        ++frees;
        CHECK(hipFree(buffer) == hipSuccess);
    }

    void reset() override { ++resets; }

    std::size_t allocations = 0;
    std::size_t frees = 0;
    std::size_t resets = 0;
};

}  // namespace

TEST_CASE("ROCm factory reports a live hardware device and selects it current") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    UnusedAllocator allocator;
    {
        auto device = iom::make_rocm_device(0, allocator);
        REQUIRE(device != nullptr);
        CHECK(device->backend_kind() == iom::BackendKind::ROCM);
        CHECK(device->backend_device() == 0);

        int current_device = -1;
        REQUIRE(hipGetDevice(&current_device) == hipSuccess);
        CHECK(current_device == 0);

        auto tensor = device->create_tensor(
                iom::TensorSpec{
                        iom::TensorShape{{16, 16}}, iom::DataType::F32});
        CHECK(tensor != nullptr);
        auto queue = device->create_ops();
        CHECK(queue != nullptr);
        CHECK(allocator.allocations == 1);
    }
    CHECK(allocator.frees == 1);

    // A second construction after the first owner leaves scope remains
    // deterministic: the factory re-selects the device on this thread.
    auto recreated = iom::make_rocm_device(0, allocator);
    REQUIRE(recreated != nullptr);
    CHECK(recreated->backend_kind() == iom::BackendKind::ROCM);
    CHECK(recreated->backend_device() == 0);
    CHECK(allocator.allocations == 1);
}

TEST_CASE("ROCm factory rejects the first unavailable ordinal") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    UnusedAllocator allocator;
    CHECK_THROWS_AS(
            (void)iom::make_rocm_device(
                    static_cast<std::uint32_t>(device_count), allocator),
            std::invalid_argument);
    CHECK(allocator.allocations == 0);
    CHECK(allocator.frees == 0);
}
