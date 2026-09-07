#include <doctest/doctest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/cuda/device.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"
#include "copy.hpp"
#include "driver.hpp"

namespace cuda_test {
bool fail_next_allocation = false;
}  // namespace cuda_test

void* operator new(std::size_t size) {
    if (cuda_test::fail_next_allocation) {
        cuda_test::fail_next_allocation = false;
        throw std::bad_alloc();
    }
    if (void* allocation = std::malloc(size); allocation != nullptr) {
        return allocation;
    }
    throw std::bad_alloc();
}

void operator delete(void* allocation) noexcept {
    std::free(allocation);
}

namespace {

struct DriverCallProbe {
    bool retained = false;
    CUdevice retained_device = 0;
    CUcontext retained_context = nullptr;
    std::size_t release_count = 0;
    CUdevice released_device = 0;
    std::size_t primary_ctx_retain_count = 0;
    std::size_t init_count = 0;
    std::size_t device_get_count_count = 0;
    int device_get_count_value = 0;
    CUdevice device_get_device = 0;
    int device_get_ordinal = -1;
    std::size_t ctx_set_current_count = 0;
};

DriverCallProbe* active_probe = nullptr;

CUresult counting_primary_ctx_retain(CUcontext* context, CUdevice device) {
    if (active_probe != nullptr) {
        ++active_probe->primary_ctx_retain_count;
    }
    const CUresult status = cuDevicePrimaryCtxRetain(context, device);
    if (status == CUDA_SUCCESS && active_probe != nullptr) {
        active_probe->retained = true;
        active_probe->retained_device = device;
        active_probe->retained_context = *context;
    }
    return status;
}

CUresult failing_ctx_set_current(CUcontext) {
    return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult pass_through_ctx_set_current(CUcontext context) {
    if (active_probe != nullptr) {
        ++active_probe->ctx_set_current_count;
    }
    return cuCtxSetCurrent(context);
}

CUresult counting_primary_ctx_release(CUdevice device) {
    if (active_probe != nullptr) {
        ++active_probe->release_count;
        active_probe->released_device = device;
    }
    return cuDevicePrimaryCtxRelease(device);
}

CUresult counting_init(unsigned int flags) {
    if (active_probe != nullptr) {
        ++active_probe->init_count;
    }
    return cuInit(flags);
}

CUresult counting_device_get_count(int* count) {
    const CUresult status = cuDeviceGetCount(count);
    if (active_probe != nullptr) {
        ++active_probe->device_get_count_count;
        if (status == CUDA_SUCCESS) {
            active_probe->device_get_count_value = *count;
        }
    }
    return status;
}

CUresult counting_device_get(CUdevice* device, int ordinal) {
    const CUresult status = cuDeviceGet(device, ordinal);
    if (active_probe != nullptr) {
        active_probe->device_get_ordinal = ordinal;
        if (status == CUDA_SUCCESS) {
            active_probe->device_get_device = *device;
        }
    }
    return status;
}

class DriverCallsRestore final {
public:
    DriverCallsRestore()
            : saved_(iom::cuda_detail::driver_calls),
              saved_probe_(active_probe) {}

    DriverCallsRestore(const DriverCallsRestore&) = delete;
    DriverCallsRestore& operator=(const DriverCallsRestore&) = delete;

    ~DriverCallsRestore() {
        iom::cuda_detail::driver_calls = saved_;
        active_probe = saved_probe_;
    }

private:
    iom::cuda_detail::DriverCalls saved_;
    DriverCallProbe* saved_probe_;
};

}  // namespace
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

class MisalignedAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        ++allocations;
        raw_ = ::operator new(size + 32, std::align_val_t(32));
        return static_cast<char*>(raw_) + 1;
    }

    void free(void* buffer) override {
        ++frees;
        CHECK(buffer == static_cast<char*>(raw_) + 1);
        ::operator delete(raw_, std::align_val_t(32));
        raw_ = nullptr;
    }

    void reset() override {}

    std::size_t allocations = 0;
    std::size_t frees = 0;

private:
    void* raw_ = nullptr;
};

class CudaAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        void* pointer = nullptr;
        if (cudaMalloc(&pointer, size) != cudaSuccess) {
            throw std::bad_alloc();
        }
        return pointer;
    }

    void free(void* pointer) override {
        if (pointer != nullptr && cudaFree(pointer) != cudaSuccess) {
            throw std::runtime_error("cudaFree failed in smoke allocator");
        }
    }

    void reset() override {}
};

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
        CHECK_THROWS_AS((void)device->create_tensor(spec), std::bad_alloc);
        auto queue = device->create_ops();
        REQUIRE(queue != nullptr);
        CHECK(allocator.allocations == 1);
        CHECK(allocator.frees == 0);
    }

    auto recreated = iom::make_cuda_device(0, allocator);
    REQUIRE(recreated != nullptr);
    CHECK(recreated->backend_kind() == iom::BackendKind::CUDA);
    CHECK(recreated->backend_device() == 0);
    CHECK(allocator.allocations == 1);
    CHECK(allocator.frees == 0);
}

TEST_CASE("CUDA tensor rejects misaligned allocator storage exactly once") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    MisalignedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    CHECK_THROWS_AS((void)device->create_tensor(spec), std::runtime_error);
    CHECK(allocator.allocations == 1);
    CHECK(allocator.frees == 1);
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

TEST_CASE("CUDA factory releases retained context when activation fails") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    DriverCallsRestore restore;
    DriverCallProbe probe;
    active_probe = &probe;
    auto calls = iom::cuda_detail::driver_calls;
    calls.primary_ctx_retain = &counting_primary_ctx_retain;
    calls.ctx_set_current = &failing_ctx_set_current;
    calls.primary_ctx_release = &counting_primary_ctx_release;
    calls.init = &counting_init;
    calls.device_get_count = &counting_device_get_count;
    calls.device_get = &counting_device_get;
    iom::cuda_detail::driver_calls = calls;

    UnusedAllocator allocator;
    bool threw = false;
    try {
        (void)iom::make_cuda_device(0, allocator);
    } catch (const std::runtime_error& error) {
        threw = true;
        CHECK(std::string_view(error.what()).starts_with(
                "cuCtxSetCurrent failed with"));
    } catch (...) {
        FAIL("CUDA factory threw an unexpected exception type");
    }

    CHECK(threw);
    CHECK(probe.retained);
    CHECK(probe.release_count == 1);
    CHECK(probe.released_device == probe.retained_device);
}

TEST_CASE("CUDA factory releases retained context when device allocation fails") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    DriverCallsRestore restore;
    DriverCallProbe probe;
    active_probe = &probe;
    auto calls = iom::cuda_detail::driver_calls;
    calls.primary_ctx_retain = &counting_primary_ctx_retain;
    calls.ctx_set_current = &pass_through_ctx_set_current;
    calls.primary_ctx_release = &counting_primary_ctx_release;
    calls.init = &counting_init;
    calls.device_get_count = &counting_device_get_count;
    calls.device_get = &counting_device_get;
    iom::cuda_detail::driver_calls = calls;

    UnusedAllocator allocator;
    bool threw = false;
    cuda_test::fail_next_allocation = true;
    try {
        (void)iom::make_cuda_device(0, allocator);
    } catch (const std::bad_alloc&) {
        threw = true;
    } catch (...) {
        FAIL("CUDA factory threw an unexpected exception type");
    }
    cuda_test::fail_next_allocation = false;

    CHECK(threw);
    CHECK(probe.retained);
    CHECK(probe.release_count == 1);
    CHECK(probe.released_device == probe.retained_device);
}

TEST_CASE("CUDA factory dismisses the primary-context guard on success") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    DriverCallsRestore restore;
    DriverCallProbe probe;
    active_probe = &probe;
    auto calls = iom::cuda_detail::driver_calls;
    calls.primary_ctx_retain = &counting_primary_ctx_retain;
    calls.ctx_set_current = &pass_through_ctx_set_current;
    calls.primary_ctx_release = &counting_primary_ctx_release;
    calls.init = &counting_init;
    calls.device_get_count = &counting_device_get_count;
    calls.device_get = &counting_device_get;
    iom::cuda_detail::driver_calls = calls;

    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    REQUIRE(device != nullptr);
    CHECK(probe.release_count == 0);

    device.reset();
    CHECK(probe.retained);
    CHECK(probe.release_count == 1);
    CHECK(probe.released_device == probe.retained_device);
}

namespace {

void require_cuda_hardware() {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);
}

}  // namespace

TEST_CASE("CUDA host transfers on independent threads do not share a stream") {
    require_cuda_hardware();
    CudaAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    const iom::TensorSpec spec{
            iom::TensorShape{{2048, 2048}}, iom::DataType::F32};
    auto tensor_a = device->create_tensor(spec);
    auto tensor_b = device->create_tensor(spec);
    const std::vector<std::byte> input(
            spec.logical_nbytes(), static_cast<std::byte>(0x3c));
    std::vector<std::byte> output_a(spec.logical_nbytes());
    std::vector<std::byte> output_b(spec.logical_nbytes());
    constexpr int transfers = 4;

    const auto serial_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < transfers; ++i) {
        tensor_a->view().copy_from_host(input);
        tensor_a->view().copy_to_host(output_a);
        tensor_b->view().copy_from_host(input);
        tensor_b->view().copy_to_host(output_b);
    }
    const auto serial_end = std::chrono::steady_clock::now();
    const auto serial_total = serial_end - serial_begin;

    std::barrier start_gate(3);
    std::atomic<bool> failed = false;
    const auto concurrent_begin = std::chrono::steady_clock::now();
    std::thread thread_a([&] {
        try {
            start_gate.arrive_and_wait();
            for (int i = 0; i < transfers; ++i) {
                tensor_a->view().copy_from_host(input);
                tensor_a->view().copy_to_host(output_a);
            }
        } catch (...) {
            failed.store(true, std::memory_order_release);
        }
    });
    std::thread thread_b([&] {
        try {
            start_gate.arrive_and_wait();
            for (int i = 0; i < transfers; ++i) {
                tensor_b->view().copy_from_host(input);
                tensor_b->view().copy_to_host(output_b);
            }
        } catch (...) {
            failed.store(true, std::memory_order_release);
        }
    });
    start_gate.arrive_and_wait();
    thread_a.join();
    thread_b.join();
    const auto concurrent_end = std::chrono::steady_clock::now();

    CHECK_FALSE(failed.load(std::memory_order_acquire));
    CHECK(concurrent_end - concurrent_begin < 2 * serial_total);
}

TEST_CASE(
        "CUDA host transfers from multiple queues on one device share the "
        "transfer-stream pool") {
    require_cuda_hardware();
    CudaAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    const iom::TensorSpec spec{
            iom::TensorShape{{1024, 1024}}, iom::DataType::F32};
    auto tensor_a = device->create_tensor(spec);
    auto tensor_b = device->create_tensor(spec);
    auto tensor_c = device->create_tensor(spec);
    auto queue_a = device->create_ops();
    auto queue_b = device->create_ops();
    const std::vector<std::byte> input(
            spec.logical_nbytes(), static_cast<std::byte>(0x5a));
    std::vector<std::byte> output(spec.logical_nbytes());
    constexpr int transfers = 4;

    const auto serial_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < transfers; ++i) {
        tensor_a->view().copy_from_host(input);
        tensor_a->view().copy_to_host(output);
    }
    const auto serial_end = std::chrono::steady_clock::now();
    const auto serial_total = serial_end - serial_begin;

    std::barrier start_gate(3);
    std::atomic<bool> failed = false;
    const auto concurrent_begin = std::chrono::steady_clock::now();
    std::thread host_thread([&] {
        try {
            start_gate.arrive_and_wait();
            for (int i = 0; i < transfers; ++i) {
                tensor_a->view().copy_from_host(input);
                tensor_a->view().copy_to_host(output);
            }
        } catch (...) {
            failed.store(true, std::memory_order_release);
        }
    });
    std::thread queue_thread([&] {
        try {
            start_gate.arrive_and_wait();
            for (int i = 0; i < transfers; ++i) {
                const iom::oid token =
                        queue_b->copy(tensor_b->view(), tensor_c->view());
                queue_b->wait(token);
            }
        } catch (...) {
            failed.store(true, std::memory_order_release);
        }
    });
    start_gate.arrive_and_wait();
    host_thread.join();
    queue_thread.join();
    const auto concurrent_end = std::chrono::steady_clock::now();

    CHECK_FALSE(failed.load(std::memory_order_acquire));
    CHECK(concurrent_end - concurrent_begin < 2 * serial_total);
    queue_a.reset();
}

TEST_CASE("CUDA errored host transfer drops its stream from the pool") {
    require_cuda_hardware();
    CudaAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto tensor = device->create_tensor(spec);
    std::vector<std::byte> input(spec.logical_nbytes());
    CUcontext context = nullptr;
    REQUIRE(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
    iom::cuda_detail::StagingSlotPool staging_pool;
    iom::cuda_detail::TransferStreamPool pool;

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::third_plane_launch);
    CHECK_THROWS_AS(
            iom::cuda_detail::region_from_host(
                    pool, staging_pool, context, tensor->view(), input),
            std::runtime_error);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

    CHECK_EQ(pool.idle_count_for_testing(), 0);
    CHECK_NOTHROW(
            iom::cuda_detail::region_from_host(
                    pool, staging_pool, context, tensor->view(), input));
    CHECK_EQ(pool.idle_count_for_testing(), 1);
    pool.destroy();
}

TEST_CASE("CUDA poisoned Scope drops its stream from the pool") {
    require_cuda_hardware();
    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    iom::cuda_detail::TransferStreamPool pool;
    {
        auto scope = pool.acquire();
        scope.poison();
    }
    CHECK_EQ(pool.idle_count_for_testing(), 0);
    {
        auto scope = pool.acquire();
    }
    CHECK_EQ(pool.idle_count_for_testing(), 1);
    pool.destroy();
}

TEST_CASE("CUDA staging pool preserves accounting across allocation failures") {
    require_cuda_hardware();
    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    iom::cuda_detail::StagingSlotPool pool;

    CHECK_THROWS_AS(
            pool.acquire(iom::cuda_detail::StagingSlotPool::kMaxStagingBytes + 1),
            std::invalid_argument);
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    pool.fail_next_allocation_for_testing();
    CHECK_THROWS_AS(pool.acquire(4), std::bad_alloc);
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    {
        auto lease = pool.acquire(4);
        CHECK_EQ(lease.capacity(), 4);
        CHECK(lease.staging() != 0);
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 1);
    CHECK_EQ(pool.idle_count_for_testing(), 1);

    pool.fail_next_allocation_for_testing();
    CHECK_THROWS_AS(pool.acquire(8), std::bad_alloc);
    CHECK_EQ(pool.allocation_count_for_testing(), 1);
    CHECK_EQ(pool.idle_count_for_testing(), 1);

    {
        auto lease = pool.acquire(4);
        lease.poison();
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    for (std::size_t i = 0;
         i < 2 * iom::cuda_detail::StagingSlotPool::kMaxSlotCount; ++i) {
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

TEST_CASE("CUDA driver-call seam intercepts every claimed driver call") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    DriverCallsRestore restore;
    DriverCallProbe probe;
    active_probe = &probe;
    auto calls = iom::cuda_detail::driver_calls;
    calls.primary_ctx_retain = &counting_primary_ctx_retain;
    calls.ctx_set_current = &pass_through_ctx_set_current;
    calls.primary_ctx_release = &counting_primary_ctx_release;
    calls.init = &counting_init;
    calls.device_get_count = &counting_device_get_count;
    calls.device_get = &counting_device_get;
    iom::cuda_detail::driver_calls = calls;

    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    REQUIRE(device != nullptr);
    CHECK(probe.init_count == 1);
    CHECK(probe.device_get_count_count == 1);
    CHECK(probe.device_get_count_value > 0);
    CHECK(probe.device_get_count_value == probe.device_get_ordinal + 1);
    CHECK(probe.primary_ctx_retain_count == 1);
    CHECK(probe.retained);
    CHECK(probe.ctx_set_current_count >= 1);
    CHECK(probe.release_count == 0);

    device.reset();
    CHECK(probe.release_count == 1);
}

TEST_CASE("CUDA event ring reuses events and enforces bounded capacity") {
    require_cuda_hardware();
    CUcontext context = nullptr;
    REQUIRE(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::cuda_detail::gpu_policy> metadata_pool(
            context);
    auto state = std::make_shared<iom::cuda_detail::EventRingState>(
            context, metadata_pool);

    for (int i = 0; i < 32; ++i) {
        auto record = state->acquire();
    }
    CHECK_EQ(state->created_event_count_for_testing(), 1);
    CHECK_EQ(state->in_use_count_for_testing(), 0);

    std::vector<
            std::shared_ptr<iom::cuda_detail::EventRingState::Submission>>
            held;
    held.reserve(iom::cuda_detail::EventRingState::kEventRingCount);
    for (std::size_t i = 0;
         i < iom::cuda_detail::EventRingState::kEventRingCount; ++i) {
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

TEST_CASE("CUDA event ring consumes create faults before allocating") {
    require_cuda_hardware();
    CUcontext context = nullptr;
    REQUIRE(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::cuda_detail::gpu_policy> metadata_pool(
            context);
    auto state = std::make_shared<iom::cuda_detail::EventRingState>(
            context, metadata_pool);

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::event_create);
    CHECK_THROWS_AS((void)state->acquire(), std::runtime_error);
    CHECK_EQ(state->created_event_count_for_testing(), 0);
    CHECK_EQ(state->in_use_count_for_testing(), 0);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
}

TEST_CASE("CUDA event ring releases attached metadata when retired") {
    require_cuda_hardware();
    CUcontext context = nullptr;
    REQUIRE(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::cuda_detail::gpu_policy> metadata_pool(
            context);
    auto state = std::make_shared<iom::cuda_detail::EventRingState>(
            context, metadata_pool);
    auto submission = state->acquire();
    std::vector<std::size_t> metadata_slots;
    metadata_slots.reserve(
            iom::detail::MetadataSlotPool<
                    iom::cuda_detail::gpu_policy>::kMetadataSlotCount);
    for (std::size_t i = 0;
         i < iom::detail::MetadataSlotPool<
                     iom::cuda_detail::gpu_policy>::kMetadataSlotCount;
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

TEST_CASE("CUDA event ring fences are pending until own completion") {
    require_cuda_hardware();
    CUcontext context = nullptr;
    REQUIRE(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    iom::detail::MetadataSlotPool<iom::cuda_detail::gpu_policy> metadata_pool(
            context);
    auto state = std::make_shared<iom::cuda_detail::EventRingState>(
            context, metadata_pool);
    auto first = state->acquire();
    const std::size_t first_index = state->slot_index(*first);

    // A pending submission never reports the cached default success: an
    // in-use slot must not claim success before its event is synchronized.
    CHECK_FALSE(first->invoke_result().succeeded);
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

TEST_CASE("CUDA create_tensor validates the spec before any context activation") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    DriverCallsRestore restore;
    DriverCallProbe probe;
    active_probe = &probe;
    auto calls = iom::cuda_detail::driver_calls;
    calls.primary_ctx_retain = &counting_primary_ctx_retain;
    calls.ctx_set_current = &pass_through_ctx_set_current;
    calls.primary_ctx_release = &counting_primary_ctx_release;
    calls.init = &counting_init;
    calls.device_get_count = &counting_device_get_count;
    calls.device_get = &counting_device_get;
    iom::cuda_detail::driver_calls = calls;

    UnusedAllocator allocator;
    auto device = iom::make_cuda_device(0, allocator);
    REQUIRE(device != nullptr);
    const std::size_t activation_count = probe.ctx_set_current_count;

    // Rank and dimension violations are rejected by TensorShape before
    // create_tensor is reached, with no driver interaction and no
    // allocator allocation.
    try {
        (void)iom::TensorSpec{
                iom::TensorShape{{16}}, iom::DataType::F32};
        FAIL("TensorSpec accepted a rank-one shape");
    } catch (const std::invalid_argument&) {
    }
    try {
        (void)iom::TensorSpec{
                iom::TensorShape{{16, 0}}, iom::DataType::F32};
        FAIL("TensorSpec accepted a zero dimension");
    } catch (const std::invalid_argument&) {
    }
    CHECK(probe.ctx_set_current_count == activation_count);
    CHECK(allocator.allocations == 0);

    // Grouped quantization is rejected by the base Tensor ctor's spec
    // validation with the same error the ctor produces today, while the
    // driver-call probe records zero context-set calls and the allocator
    // never runs.
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
    CHECK(probe.ctx_set_current_count == activation_count);
    CHECK(allocator.allocations == 0);

    // A valid spec activates the runtime exactly once, at allocation time,
    // through the CudaTensor ctor's pre_allocate callback; the allocator's
    // nullptr return then surfaces as bad_alloc before any storage exists.
    const iom::TensorSpec valid_spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    CHECK_THROWS_AS(
            (void)device->create_tensor(valid_spec), std::bad_alloc);
    CHECK(probe.ctx_set_current_count == activation_count + 1);
    CHECK(allocator.allocations == 1);
    CHECK(allocator.frees == 0);
}
