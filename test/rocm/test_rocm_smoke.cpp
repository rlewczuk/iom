#include <doctest/doctest.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <new>
#include <thread>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/rocm/device.hpp"
#include "copy.hpp"
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

TEST_CASE("ROCm staging pool preserves accounting across allocation failures") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    iom::rocm_detail::StagingSlotPool pool;

    CHECK_THROWS_AS(
            (void)pool.acquire(
                    iom::rocm_detail::StagingSlotPool::kMaxStagingBytes + 1),
            std::invalid_argument);
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    pool.fail_next_allocation_for_testing();
    CHECK_THROWS_AS((void)pool.acquire(4), std::bad_alloc);
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    {
        auto lease = pool.acquire(4);
        CHECK_EQ(lease.capacity(), 4);
        CHECK(lease.staging() != nullptr);
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 1);
    CHECK_EQ(pool.idle_count_for_testing(), 1);

    pool.fail_next_allocation_for_testing();
    CHECK_THROWS_AS((void)pool.acquire(8), std::bad_alloc);
    CHECK_EQ(pool.allocation_count_for_testing(), 1);
    CHECK_EQ(pool.idle_count_for_testing(), 1);

    {
        auto lease = pool.acquire(4);
        lease.poison();
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    for (std::size_t i = 0;
         i < 2 * iom::rocm_detail::StagingSlotPool::kMaxSlotCount; ++i) {
        auto lease = pool.acquire(4);
        lease.poison();
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    {
        auto lease = pool.acquire(4);
        CHECK_EQ(lease.capacity(), 4);
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 1);
    pool.destroy();
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);
}

TEST_CASE("ROCm transfer streams and odd-tail staging are reusable") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    iom::rocm_detail::TransferStreamPool streams;
    {
        auto scope = streams.acquire();
        scope.poison();
    }
    CHECK_EQ(streams.idle_count_for_testing(), 0);
    {
        auto scope = streams.acquire();
    }
    CHECK_EQ(streams.idle_count_for_testing(), 1);
    streams.destroy();

    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    const iom::TensorSpec spec{
            iom::TensorShape{{1, 17}}, iom::DataType::I2};
    auto tensor = device->create_tensor(spec);
    const std::vector<std::byte> input(
            spec.logical_nbytes(), static_cast<std::byte>(0xA5));
    std::vector<std::byte> expected = input;
    expected.back() = static_cast<std::byte>(0x01);
    std::vector<std::byte> output(spec.logical_nbytes());
    for (int iteration = 0; iteration < 3; ++iteration) {
        tensor->view().copy_from_host(input);
        tensor->view().copy_to_host(output);
        CHECK_EQ(output, expected);
        CHECK((static_cast<unsigned int>(output.back()) & 0xFCu) == 0);
    }
}

TEST_CASE("ROCm concurrent host transfers acquire independent resources") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    const iom::TensorSpec spec{
            iom::TensorShape{{512, 512}}, iom::DataType::F32};
    auto tensor_a = device->create_tensor(spec);
    auto tensor_b = device->create_tensor(spec);
    const std::vector<std::byte> input(
            spec.logical_nbytes(), static_cast<std::byte>(0x3C));
    std::vector<std::byte> output_a(spec.logical_nbytes());
    std::vector<std::byte> output_b(spec.logical_nbytes());
    std::barrier start_gate(3);
    std::atomic<bool> failed = false;

    std::thread thread_a([&] {
        try {
            start_gate.arrive_and_wait();
            tensor_a->view().copy_from_host(input);
            tensor_a->view().copy_to_host(output_a);
        } catch (...) {
            failed.store(true, std::memory_order_release);
        }
    });
    std::thread thread_b([&] {
        try {
            start_gate.arrive_and_wait();
            tensor_b->view().copy_from_host(input);
            tensor_b->view().copy_to_host(output_b);
        } catch (...) {
            failed.store(true, std::memory_order_release);
        }
    });
    start_gate.arrive_and_wait();
    thread_a.join();
    thread_b.join();

    CHECK_FALSE(failed.load(std::memory_order_acquire));
    CHECK_EQ(output_a, input);
    CHECK_EQ(output_b, input);
}
