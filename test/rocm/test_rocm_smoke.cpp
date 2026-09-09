#include <doctest/doctest.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <new>
#include <string_view>
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

TEST_CASE("ROCm event ring reuses events and enforces bounded capacity") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::rocm_detail::gpu_policy> metadata_pool(
            0);
    auto state = std::make_shared<iom::rocm_detail::EventRingState>(
            0, metadata_pool);

    for (int i = 0; i < 32; ++i) {
        auto record = state->acquire();
    }
    CHECK_EQ(state->created_event_count_for_testing(), 1);
    CHECK_EQ(state->in_use_count_for_testing(), 0);

    std::vector<
            std::shared_ptr<iom::rocm_detail::EventRingState::Submission>>
            held;
    held.reserve(iom::rocm_detail::EventRingState::kEventRingCount);
    for (std::size_t i = 0;
         i < iom::rocm_detail::EventRingState::kEventRingCount; ++i) {
        held.push_back(state->acquire());
    }
    CHECK_EQ(state->created_event_count_for_testing(), 16);
    CHECK_EQ(state->in_use_count_for_testing(), 16);

    std::atomic<bool> waiter_started = false;
    std::atomic<bool> waiter_acquired = false;
    std::thread waiter([&] {
        waiter_started.store(true, std::memory_order_release);
        auto record = state->acquire();
        waiter_acquired.store(true, std::memory_order_release);
    });
    while (!waiter_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    CHECK_EQ(state->in_use_count_for_testing(), 16);
    held.front().reset();
    waiter.join();
    CHECK(waiter_acquired.load(std::memory_order_acquire));
    for (auto& record : held) {
        record.reset();
    }
}

TEST_CASE("ROCm event ring consumes create faults before allocating") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::rocm_detail::gpu_policy> metadata_pool(
            0);
    auto state = std::make_shared<iom::rocm_detail::EventRingState>(
            0, metadata_pool);

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::event_create);
    CHECK_THROWS_AS((void)state->acquire(), std::runtime_error);
    CHECK_EQ(state->created_event_count_for_testing(), 0);
    CHECK_EQ(state->in_use_count_for_testing(), 0);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
}

TEST_CASE("ROCm event ring releases attached metadata when retired") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::rocm_detail::gpu_policy> metadata_pool(
            0);
    auto state = std::make_shared<iom::rocm_detail::EventRingState>(
            0, metadata_pool);
    auto submission = state->acquire();
    std::vector<std::size_t> metadata_slots;
    metadata_slots.reserve(
            iom::detail::MetadataSlotPool<
                    iom::rocm_detail::gpu_policy>::kMetadataSlotCount);
    for (std::size_t i = 0;
         i < iom::detail::MetadataSlotPool<
                     iom::rocm_detail::gpu_policy>::kMetadataSlotCount;
         ++i) {
        metadata_slots.push_back(metadata_pool.acquire());
    }
    submission->attach_metadata_slot(metadata_slots.front());
    // Retiring the submission releases the attached metadata; the pooled
    // event stays reserved until the last record reference is dropped.
    state->on_worker_destroy(*submission);
    CHECK_EQ(state->in_use_count_for_testing(), 1);
    submission.reset();
    CHECK_EQ(state->in_use_count_for_testing(), 0);
    for (std::size_t i = 1; i < metadata_slots.size(); ++i) {
        metadata_pool.release(metadata_slots[i]);
    }
    std::vector<std::size_t> reacquired_slots;
    reacquired_slots.reserve(metadata_slots.size());
    for (std::size_t i = 0; i < metadata_slots.size(); ++i) {
        reacquired_slots.push_back(metadata_pool.acquire());
    }
    CHECK_EQ(reacquired_slots.size(), metadata_slots.size());
    for (const std::size_t slot : reacquired_slots) {
        metadata_pool.release(slot);
    }
}

TEST_CASE("ROCm event ring fences are pending until own completion") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::rocm_detail::gpu_policy> metadata_pool(
            0);
    auto state = std::make_shared<iom::rocm_detail::EventRingState>(
            0, metadata_pool);
    auto first = state->acquire();
    const std::size_t first_index = state->slot_index(*first);

    // A pending submission never reports the cached default success: an
    // in-use slot must not claim success before its event is synchronized.
    CHECK_FALSE(first->invoke_result().succeeded);
    state->mark_event_recorded(*first);
    state->on_worker_complete(*first);
    state->on_worker_destroy(*first);
    const iom::detail::FenceResult recorded = first->invoke_result();
    CHECK(recorded.succeeded);

    // A fence reference (as held by registry entries and destructor
    // snapshots) keeps the pooled event reserved, so a later submission
    // cannot reuse the slot and cannot alter the earlier fence's result.
    auto earlier_fence = first;
    first.reset();
    CHECK_EQ(state->in_use_count_for_testing(), 1);
    auto reused = state->acquire();
    CHECK_NE(state->slot_index(*reused), first_index);
    CHECK_FALSE(reused->invoke_result().succeeded);
    CHECK(earlier_fence->invoke_result().succeeded);

    // Once the earlier fence reference is gone the slot is reusable; a
    // later submission may land on it and still starts pending.
    earlier_fence.reset();
    auto resettled = state->acquire();
    CHECK_EQ(state->slot_index(*resettled), first_index);
    CHECK_FALSE(resettled->invoke_result().succeeded);
    resettled.reset();
}

TEST_CASE("ROCm double-fault retirement protects metadata until covering drain") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::rocm_detail::gpu_policy> metadata_pool(
            0);
    auto state = std::make_shared<iom::rocm_detail::EventRingState>(
            0, metadata_pool);
    auto submission = state->acquire();
    const auto first = metadata_pool.acquire();
    submission->attach_metadata_slot(first);
    CHECK_THROWS_AS(state->on_worker_complete(*submission), std::runtime_error);
    state->on_worker_destroy(*submission);
    std::vector<std::size_t> held;
    for (std::size_t i = 1;
         i < iom::detail::MetadataSlotPool<
                     iom::rocm_detail::gpu_policy>::kMetadataSlotCount;
         ++i) {
        held.push_back(metadata_pool.acquire());
    }
    std::atomic<bool> acquired = false;
    std::thread waiter([&] {
        const auto slot = metadata_pool.acquire();
        acquired.store(true, std::memory_order_release);
        metadata_pool.release(slot);
    });
    std::this_thread::yield();
    CHECK_FALSE(acquired.load(std::memory_order_acquire));
    state->on_queue_drain(false);
    CHECK_FALSE(acquired.load(std::memory_order_acquire));
    state->on_queue_drain(true);
    waiter.join();
    CHECK(acquired.load(std::memory_order_acquire));
    for (const auto slot : held) {
        metadata_pool.release(slot);
    }
}

TEST_CASE("ROCm rejected create_tensor leaves the current device unchanged") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    UnusedAllocator allocator;
    auto device = iom::make_rocm_device(0, allocator);
    REQUIRE(device != nullptr);

    // Move the calling thread onto another device so a stray hipSetDevice
    // inside create_tensor (the pre-fix entry-point activation) would be
    // observable as a mutation of the thread's current device.
    const int other_device = device_count > 1 ? 1 : 0;
    REQUIRE(hipSetDevice(other_device) == hipSuccess);
    int current_before = -1;
    REQUIRE(hipGetDevice(&current_before) == hipSuccess);
    CHECK(current_before == other_device);

    const iom::TensorSpec invalid_spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32,
            iom::QuantizationFormat::INT8_SYMMETRIC};
    try {
        (void)device->create_tensor(invalid_spec);
        FAIL("create_tensor accepted a grouped-quantization spec");
    } catch (const std::runtime_error& error) {
        CHECK(std::string_view(error.what()).starts_with(
                "grouped quantization formats are not supported"));
    } catch (...) {
        FAIL("create_tensor threw an unexpected exception type");
    }

    int current_after = -1;
    REQUIRE(hipGetDevice(&current_after) == hipSuccess);
    CHECK(current_after == current_before);
    CHECK(allocator.allocations == 0);
}
