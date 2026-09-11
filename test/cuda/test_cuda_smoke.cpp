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
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

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

struct AllocationProbe {
    std::vector<iom::cuda_detail::AllocationRecord> records;
};

AllocationProbe* active_allocation_probe = nullptr;

void capture_allocation(const iom::cuda_detail::AllocationRecord& record) {
    if (active_allocation_probe != nullptr) {
        active_allocation_probe->records.push_back(record);
    }
}

CUresult failing_mem_alloc(CUdeviceptr*, std::size_t) {
    return CUDA_ERROR_OUT_OF_MEMORY;
}

class AllocationCallsRestore final {
public:
    AllocationCallsRestore()
            : saved_calls_(iom::cuda_detail::allocation_calls),
              saved_observer_(iom::cuda_detail::allocation_observer),
              saved_probe_(active_allocation_probe) {}

    AllocationCallsRestore(const AllocationCallsRestore&) = delete;
    AllocationCallsRestore& operator=(const AllocationCallsRestore&) = delete;

    ~AllocationCallsRestore() {
        iom::cuda_detail::allocation_calls = saved_calls_;
        iom::cuda_detail::allocation_observer = saved_observer_;
        active_allocation_probe = saved_probe_;
    }

private:
    iom::cuda_detail::AllocationCalls saved_calls_;
    iom::cuda_detail::AllocationObserver saved_observer_;
    AllocationProbe* saved_probe_;
};

}  // namespace
namespace {

// Standard arena capacity for smoke tests: 64 MiB covers every live tensor
// set below (the largest single tensor is 2048x2048 F32 = 16 MiB, with at
// most three such tensors live together).
constexpr std::size_t kArenaBytes = 64u * 1024 * 1024;

[[nodiscard]] iom::TensorSpec spec_16x16_f32() {
    return iom::TensorSpec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
}

}  // namespace

TEST_CASE("CUDA factory reports a live hardware device and owns its context") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    {
        auto device = iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
        REQUIRE(device != nullptr);
        CHECK(device->backend_kind() == iom::BackendKind::CUDA);
        CHECK(device->backend_device() == 0);

        CUcontext current_context = nullptr;
        REQUIRE(cuCtxGetCurrent(&current_context) == CUDA_SUCCESS);
        CHECK(current_context != nullptr);

        auto tensor = device->create_tensor(spec_16x16_f32());
        REQUIRE(tensor != nullptr);
        auto queue = device->create_ops();
        REQUIRE(queue != nullptr);
    }

    auto recreated = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(recreated != nullptr);
    CHECK(recreated->backend_kind() == iom::BackendKind::CUDA);
    CHECK(recreated->backend_device() == 0);
}

TEST_CASE("CUDA factory rejects the first unavailable ordinal") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;

    CHECK_THROWS_AS(
            (void)iom::make_cuda_device(
                    static_cast<std::uint32_t>(device_count),
                    iom::DeviceMemoryConfig{kArenaBytes}),
            std::invalid_argument);
    CHECK(probe.records.empty());
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

    bool threw = false;
    try {
        (void)iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
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

TEST_CASE(
        "CUDA factory rolls back the data backing and context when device "
        "allocation fails") {
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

    AllocationCallsRestore allocation_restore;
    AllocationProbe allocation_probe;
    active_allocation_probe = &allocation_probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;
    iom::cuda_detail::allocation_calls.mem_alloc = &failing_mem_alloc;

    bool threw = false;
    try {
        (void)iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
    } catch (const std::bad_alloc&) {
        threw = true;
    } catch (...) {
        FAIL("CUDA factory threw an unexpected exception type");
    }

    CHECK(threw);
    CHECK(probe.retained);
    CHECK(probe.release_count == 1);
    CHECK(probe.released_device == probe.retained_device);

    // The failed data-backing attempt stays observable and no backing was
    // freed because none was acquired.
    REQUIRE_EQ(allocation_probe.records.size(), 1u);
    const auto& record = allocation_probe.records.front();
    CHECK(record.phase == iom::cuda_detail::AllocationPhase::setup);
    CHECK(record.classification
          == iom::cuda_detail::AllocationClass::data_backing);
    CHECK(record.kind == iom::cuda_detail::AllocationKind::allocate);
    CHECK_FALSE(record.succeeded);
    CHECK(record.address == nullptr);
}

TEST_CASE(
        "CUDA factory rolls back both backings when the metadata backing "
        "allocation fails") {
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

    AllocationCallsRestore allocation_restore;
    AllocationProbe allocation_probe;
    active_allocation_probe = &allocation_probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;
    iom::cuda_detail::allocation_calls.mem_alloc =
            [](CUdeviceptr* address, std::size_t bytes) {
                static thread_local std::size_t calls = 0;
                (void)bytes;
                if (calls++ == 0) {
                    return cuMemAlloc(address, bytes);
                }
                return CUresult{CUDA_ERROR_OUT_OF_MEMORY};
            };

    bool threw = false;
    try {
        (void)iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
    } catch (const std::bad_alloc&) {
        threw = true;
    } catch (...) {
        FAIL("CUDA factory threw an unexpected exception type");
    }

    CHECK(threw);
    CHECK(probe.retained);
    CHECK(probe.release_count == 1);
    CHECK(probe.released_device == probe.retained_device);

    // Setup allocated the data backing, failed the metadata backing, then
    // rolled the data backing back through the same seam before the context
    // was released.
    REQUIRE_EQ(allocation_probe.records.size(), 3u);
    const auto& data_alloc = allocation_probe.records[0];
    const auto& metadata_alloc = allocation_probe.records[1];
    const auto& data_free = allocation_probe.records[2];
    CHECK(data_alloc.phase == iom::cuda_detail::AllocationPhase::setup);
    CHECK(data_alloc.classification
          == iom::cuda_detail::AllocationClass::data_backing);
    CHECK(data_alloc.kind == iom::cuda_detail::AllocationKind::allocate);
    CHECK(data_alloc.succeeded);
    CHECK(data_alloc.address != nullptr);
    CHECK_EQ(data_alloc.bytes, kArenaBytes);
    CHECK(metadata_alloc.classification
          == iom::cuda_detail::AllocationClass::metadata_backing);
    CHECK(metadata_alloc.kind == iom::cuda_detail::AllocationKind::allocate);
    CHECK_FALSE(metadata_alloc.succeeded);
    CHECK(data_free.phase == iom::cuda_detail::AllocationPhase::setup);
    CHECK(data_free.classification
          == iom::cuda_detail::AllocationClass::data_backing);
    CHECK(data_free.kind == iom::cuda_detail::AllocationKind::free);
    CHECK(data_free.address == data_alloc.address);
}

TEST_CASE(
        "CUDA factory rolls back both backings and context when host "
        "allocation fails") {
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

    AllocationCallsRestore allocation_restore;
    AllocationProbe allocation_probe;
    // Pre-reserve so the probe's own record push_backs never consume the
    // injected host-allocation failure below; the failure must land on the
    // factory's arena bookkeeping so the two backings are acquired, listed,
    // and rolled back in reverse order.
    allocation_probe.records.reserve(8);
    active_allocation_probe = &allocation_probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;

    bool threw = false;
    cuda_test::fail_next_allocation = true;
    try {
        (void)iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
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

    // Both setup backings were acquired and then freed in reverse order by
    // the rollback path.
    REQUIRE_EQ(allocation_probe.records.size(), 4u);
    CHECK(allocation_probe.records[0].classification
          == iom::cuda_detail::AllocationClass::data_backing);
    CHECK(allocation_probe.records[0].succeeded);
    CHECK(allocation_probe.records[1].classification
          == iom::cuda_detail::AllocationClass::metadata_backing);
    CHECK(allocation_probe.records[1].succeeded);
    CHECK(allocation_probe.records[2].classification
          == iom::cuda_detail::AllocationClass::metadata_backing);
    CHECK(allocation_probe.records[2].kind
          == iom::cuda_detail::AllocationKind::free);
    CHECK(allocation_probe.records[3].classification
          == iom::cuda_detail::AllocationClass::data_backing);
    CHECK(allocation_probe.records[3].kind
          == iom::cuda_detail::AllocationKind::free);
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

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    iom::detail::MetadataSlotPool<iom::cuda_detail::gpu_policy> metadata_pool(
            context);
    auto state = std::make_shared<iom::cuda_detail::EventRingState>(
            context, metadata_pool);
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

TEST_CASE("CUDA double-fault retirement protects metadata until covering drain") {
    require_cuda_hardware();
    CUcontext context = nullptr;
    REQUIRE(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    iom::detail::MetadataSlotPool<iom::cuda_detail::gpu_policy> metadata_pool(
            context);
    auto state = std::make_shared<iom::cuda_detail::EventRingState>(
            context, metadata_pool);
    auto submission = state->acquire();
    const auto first = metadata_pool.acquire();
    submission->attach_metadata_slot(first);
    CHECK_THROWS_AS(state->on_worker_complete(*submission), std::runtime_error);
    state->on_worker_destroy(*submission);
    std::vector<std::size_t> held;
    for (std::size_t i = 1;
         i < iom::detail::MetadataSlotPool<
                     iom::cuda_detail::gpu_policy>::kMetadataSlotCount;
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

    // Two 16x16 F32 tensors of 1024 bytes each fit a 2048-byte arena
    // exactly; a third request exposes arena exhaustion.
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{2048});
    REQUIRE(device != nullptr);
    const std::size_t activation_count = probe.ctx_set_current_count;

    // Rank and dimension violations are rejected by TensorShape before
    // create_tensor is reached, with no driver interaction and no arena
    // activity.
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

    // Grouped quantization is rejected by the base Tensor ctor's spec
    // validation with the same error the ctor produces today, while the
    // driver-call probe records zero context-set calls.
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

    // A valid spec activates the runtime exactly once for its setup work;
    // storage suballocates from the data arena.
    auto first = device->create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{16, 16}}, iom::DataType::F32});
    REQUIRE(first != nullptr);
    CHECK(probe.ctx_set_current_count == activation_count + 1);

    auto second = device->create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{16, 16}}, iom::DataType::F32});
    REQUIRE(second != nullptr);
    CHECK(probe.ctx_set_current_count == activation_count + 2);

    // Arena exhaustion is reported before any native interaction: the
    // failed request never reaches the activation callback.
    CHECK_THROWS_AS(
            (void)device->create_tensor(
                    iom::TensorSpec{
                            iom::TensorShape{{16, 16}}, iom::DataType::F32}),
            std::bad_alloc);
    CHECK(probe.ctx_set_current_count == activation_count + 2);
}

TEST_CASE(
        "CUDA native allocation seam observes exactly two setup backings and "
        "post-publication staging and metadata boundaries") {
    require_cuda_hardware();
    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;

    const iom::TensorSpec staging_spec{
            iom::TensorShape{{1024, 1024}}, iom::DataType::F32};
    // The queue binary path always backs its metadata in a device-side
    // operation-metadata slot (grown lazily from 256 bytes), so a plain
    // rank-four tensor reliably forces the hidden native allocation.
    const iom::TensorSpec binary_spec{
            iom::TensorShape{{2, 2, 16, 16}}, iom::DataType::F32};
    iom::cuda_detail::AllocationRecord data_backing;
    iom::cuda_detail::AllocationRecord metadata_backing;
    {
        auto device = iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
        REQUIRE(device != nullptr);
        auto staging_tensor = device->create_tensor(staging_spec);
        auto lhs = device->create_tensor(binary_spec);
        auto rhs = device->create_tensor(binary_spec);
        auto out = device->create_tensor(binary_spec);
        auto queue = device->create_ops();
        REQUIRE(queue != nullptr);

        // Factory setup produced exactly two instrumented backing
        // allocations; tensor and queue creation suballocate and never
        // route through the seam.
        std::vector<iom::cuda_detail::AllocationRecord> setup_allocations;
        for (const auto& record : probe.records) {
            if (record.phase == iom::cuda_detail::AllocationPhase::setup
                    && record.kind
                            == iom::cuda_detail::AllocationKind::allocate) {
                setup_allocations.push_back(record);
            }
        }
        REQUIRE_EQ(setup_allocations.size(), 2u);
        CHECK_EQ(probe.records.size(), 2u);
        std::size_t data_count = 0;
        std::size_t metadata_count = 0;
        for (const auto& record : setup_allocations) {
            CHECK(record.succeeded);
            CHECK(record.address != nullptr);
            CHECK(reinterpret_cast<std::uintptr_t>(record.address) % 32 == 0);
            if (record.classification
                == iom::cuda_detail::AllocationClass::data_backing) {
                ++data_count;
                data_backing = record;
            }
            if (record.classification
                == iom::cuda_detail::AllocationClass::metadata_backing) {
                ++metadata_count;
                metadata_backing = record;
            }
        }
        CHECK_EQ(data_count, 1u);
        CHECK_EQ(metadata_count, 1u);
        CHECK_EQ(data_backing.bytes, kArenaBytes);
        CHECK_EQ(metadata_backing.bytes, 4 * 16 * 512u);

        // Disjoint data and metadata domains.
        const std::uintptr_t data_begin =
                reinterpret_cast<std::uintptr_t>(data_backing.address);
        const std::uintptr_t data_end = data_begin + data_backing.bytes;
        const std::uintptr_t metadata_begin =
                reinterpret_cast<std::uintptr_t>(metadata_backing.address);
        const std::uintptr_t metadata_end =
                metadata_begin + metadata_backing.bytes;
        const bool domains_disjoint =
                data_end <= metadata_begin || metadata_end <= data_begin;
        CHECK(domains_disjoint);

        std::vector<std::byte> input(
                staging_spec.logical_nbytes(), std::byte{0x5a});
        staging_tensor->view().copy_from_host(input);

        const iom::oid token =
                queue->add(lhs->view(), rhs->view(), out->view());
        REQUIRE(iom::oid_is_token(token));
        queue->wait(token);
    }  // queue and device teardown free the pooled metadata, staging, and
       // the two arena backings

    std::size_t staging_allocations = 0;
    std::size_t metadata_allocations = 0;
    std::size_t failed = 0;
    std::map<void*, std::size_t> outstanding;
    // Seed with the two setup backings so their post-publication teardown
    // frees pair up with the reservation that created them.
    outstanding[data_backing.address] = data_backing.bytes;
    outstanding[metadata_backing.address] = metadata_backing.bytes;
    for (const auto& record : probe.records) {
        if (record.phase == iom::cuda_detail::AllocationPhase::setup) {
            continue;
        }
        if (record.kind == iom::cuda_detail::AllocationKind::allocate) {
            if (!record.succeeded) {
                ++failed;
                continue;
            }
            CHECK(record.address != nullptr);
            if (record.classification
                == iom::cuda_detail::AllocationClass::staging) {
                ++staging_allocations;
            }
            if (record.classification
                == iom::cuda_detail::AllocationClass::operation_metadata) {
                ++metadata_allocations;
            }
            outstanding[record.address] = record.bytes;
        } else {
            CHECK(record.succeeded);
            // A free must pair with an earlier allocation: transient churn
            // stays visible instead of vanishing into a net-byte counter.
            CHECK(outstanding.erase(record.address) == 1);
        }
    }
    CHECK_EQ(failed, 0u);
    CHECK_GE(staging_allocations, 1u);   // host-transfer staging boundary
    CHECK_GE(metadata_allocations, 1u);  // hidden lazy metadata boundary
    CHECK(outstanding.empty());          // every allocation freed by teardown
}

TEST_CASE(
        "CUDA tensor create and destroy after setup make no native "
        "allocation or free calls") {
    require_cuda_hardware();
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;

    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::U8};
    const void* first_address = nullptr;
    {
        auto first = device->create_tensor(spec);
        first_address = first->view().native_handle();
        CHECK(first_address != nullptr);
    }
    {
        // The destroyed tensor's range is legally reusable by the arena
        // (best-fit of the same size lands on the same block), so no
        // address inequality is asserted; the no-churn point is that the
        // create/destroy cycle performs zero native calls.
        auto second = device->create_tensor(spec);
        CHECK(second->view().native_handle() != nullptr);
        auto third = device->create_tensor(spec);
        CHECK(third->view().native_handle() != nullptr);
    }
    CHECK(probe.records.empty());
}

TEST_CASE(
        "CUDA native allocation seam retains failed attempts and cleanup "
        "frees") {
    require_cuda_hardware();
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);

    // Arm the seam after setup so the record covers only pool traffic: the
    // two setup backing reservations must not pollute this scenario.
    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;

    iom::cuda_detail::StagingSlotPool pool;

    // A failed native allocation stays observable even though no pointer is
    // returned; check_cuda turns the injected boundary error into the
    // runtime failure the pool propagates.
    iom::cuda_detail::allocation_calls.mem_alloc = &failing_mem_alloc;
    CHECK_THROWS_AS((void)pool.acquire(4096), std::runtime_error);
    iom::cuda_detail::allocation_calls =
            iom::cuda_detail::AllocationCalls{};

    {
        auto lease = pool.acquire(4096);
        REQUIRE(lease.staging() != 0);
        // Poisoning releases the slot immediately: the cleanup free of the
        // successful allocation is recorded as a paired staging free.
        lease.poison();
    }
    pool.destroy();

    std::size_t failed_attempts = 0;
    std::size_t successful_allocations = 0;
    std::size_t cleanup_frees = 0;
    std::map<void*, std::size_t> outstanding;
    for (const auto& record : probe.records) {
        CHECK(record.classification
              == iom::cuda_detail::AllocationClass::staging);
        CHECK(record.phase
              == iom::cuda_detail::AllocationPhase::post_publication);
        if (record.kind == iom::cuda_detail::AllocationKind::allocate) {
            if (!record.succeeded) {
                ++failed_attempts;
                CHECK(record.address == nullptr);
                continue;
            }
            ++successful_allocations;
            outstanding[record.address] = record.bytes;
        } else {
            CHECK(record.succeeded);
            CHECK(outstanding.erase(record.address) == 1);
            ++cleanup_frees;
        }
    }
    CHECK_EQ(failed_attempts, 1u);
    CHECK_EQ(successful_allocations, 1u);
    CHECK_EQ(cleanup_frees, 1u);
    CHECK(outstanding.empty());
}

namespace {

struct BackingProbe {
    iom::cuda_detail::AllocationRecord data;
    iom::cuda_detail::AllocationRecord metadata;
};

// Creates a device with the given memory/queue configuration and returns
// the two classified setup backing records.
[[nodiscard]] BackingProbe probe_setup_backings(
        std::size_t arena_bytes, iom::QueueConfig queue_config) {
    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{arena_bytes}, queue_config);
    REQUIRE(device != nullptr);

    BackingProbe result;
    std::size_t data_count = 0;
    std::size_t metadata_count = 0;
    for (const auto& record : probe.records) {
        if (record.phase != iom::cuda_detail::AllocationPhase::setup
                || record.kind
                        != iom::cuda_detail::AllocationKind::allocate) {
            continue;
        }
        REQUIRE(record.succeeded);
        if (record.classification
            == iom::cuda_detail::AllocationClass::data_backing) {
            result.data = record;
            ++data_count;
        }
        if (record.classification
            == iom::cuda_detail::AllocationClass::metadata_backing) {
            result.metadata = record;
            ++metadata_count;
        }
    }
    REQUIRE_EQ(data_count, 1u);
    REQUIRE_EQ(metadata_count, 1u);
    return result;
}

void require_disjoint_domains(
        const iom::cuda_detail::AllocationRecord& data,
        const iom::cuda_detail::AllocationRecord& metadata) {
    const std::uintptr_t data_begin =
            reinterpret_cast<std::uintptr_t>(data.address);
    const std::uintptr_t data_end = data_begin + data.bytes;
    const std::uintptr_t metadata_begin =
            reinterpret_cast<std::uintptr_t>(metadata.address);
    const std::uintptr_t metadata_end = metadata_begin + metadata.bytes;
    const bool domains_disjoint =
            data_end <= metadata_begin || metadata_end <= data_begin;
    CHECK(domains_disjoint);
    CHECK(reinterpret_cast<std::uintptr_t>(data.address) % 32 == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(metadata.address) % 32 == 0);
}

}  // namespace

TEST_CASE(
        "CUDA factory setups reserve exactly two backings for default and "
        "custom C") {
    require_cuda_hardware();

    {
        const BackingProbe backing = probe_setup_backings(
                kArenaBytes, iom::QueueConfig{});
        CHECK_EQ(backing.data.bytes, kArenaBytes);
        CHECK_EQ(backing.metadata.bytes, 4 * 16 * 512u);
        require_disjoint_domains(backing.data, backing.metadata);
    }
    {
        const BackingProbe backing = probe_setup_backings(
                kArenaBytes, iom::QueueConfig{1});
        CHECK_EQ(backing.data.bytes, kArenaBytes);
        CHECK_EQ(backing.metadata.bytes, 4 * 1 * 512u);
        require_disjoint_domains(backing.data, backing.metadata);
    }
    {
        const BackingProbe backing = probe_setup_backings(
                kArenaBytes, iom::QueueConfig{17});
        CHECK_EQ(backing.data.bytes, kArenaBytes);
        CHECK_EQ(backing.metadata.bytes, 4 * 17 * 512u);
        require_disjoint_domains(backing.data, backing.metadata);
    }
}

TEST_CASE("CUDA factory rejects invalid configurations before publication") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    int device_count = 0;
    REQUIRE(cuDeviceGetCount(&device_count) == CUDA_SUCCESS);
    REQUIRE(device_count > 0);

    CHECK_THROWS_AS(iom::QueueConfig{0}, std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{0}),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{16}),
            std::invalid_argument);

    // 4 * C * 512 overflows size_t for this capacity.
    CHECK_THROWS_AS(
            (void)iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{kArenaBytes},
                    iom::QueueConfig{std::numeric_limits<std::size_t>::max()
                                     / 2048 + 1}),
            std::overflow_error);
}

TEST_CASE(
        "CUDA tensor storage stays inside the disjoint data arena and is "
        "32-byte aligned") {
    require_cuda_hardware();

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::cuda_detail::allocation_observer.complete = &capture_allocation;

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);

    const std::uintptr_t data_base = [&] {
        for (const auto& record : probe.records) {
            if (record.phase == iom::cuda_detail::AllocationPhase::setup
                    && record.classification
                            == iom::cuda_detail::AllocationClass::data_backing) {
                return reinterpret_cast<std::uintptr_t>(record.address);
            }
        }
        FAIL("no data backing record");
        return std::uintptr_t{0};
    }();
    const std::uintptr_t data_end = data_base + kArenaBytes;

    const iom::TensorSpec sizes[] = {
            iom::TensorSpec{iom::TensorShape{{16, 16}}, iom::DataType::U8},
            iom::TensorSpec{iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::U8},
            iom::TensorSpec{iom::TensorShape{{16, 16}}, iom::DataType::F32},
    };
    for (const iom::TensorSpec& spec : sizes) {
        auto tensor = device->create_tensor(spec);
        REQUIRE(tensor != nullptr);
        const std::uintptr_t address =
                reinterpret_cast<std::uintptr_t>(
                        tensor->view().native_handle());
        CHECK(address % 32 == 0);
        CHECK(address >= data_base);
        CHECK(address < data_end);
        // The requested storage never crosses the arena boundary.
        CHECK(address + spec.tiled_storage_nbytes() <= data_end);
    }
}

TEST_CASE(
        "CUDA data arena fragments, reuses, and coalesces with stable live "
        "addresses") {
    require_cuda_hardware();
    // 2048 bytes: two 512-byte blocks and one 1024-byte block fill it.
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{2048});
    REQUIRE(device != nullptr);

    const iom::TensorSpec half{
            iom::TensorShape{{16, 16}}, iom::DataType::U16};
    const iom::TensorSpec full{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    const iom::TensorSpec one_and_half{
            iom::TensorShape{{3, 2, 16, 16}}, iom::DataType::U8};

    auto a = device->create_tensor(half);  // offset 0, 512 bytes
    auto b = device->create_tensor(half);  // offset 512, 512 bytes
    auto c = device->create_tensor(full);  // offset 1024, 1024 bytes
    const std::uintptr_t a_address =
            reinterpret_cast<std::uintptr_t>(a->view().native_handle());
    const std::uintptr_t c_address =
            reinterpret_cast<std::uintptr_t>(c->view().native_handle());

    // Fragmentation: freeing a and c leaves 1536 aggregate free bytes split
    // into 512 + 1024, so a 1536-byte contiguous request must fail even
    // though the total free space is sufficient.
    a.reset();
    c.reset();
    CHECK_THROWS_AS((void)device->create_tensor(one_and_half), std::bad_alloc);

    // The live 512-byte tensor keeps its exact address.
    CHECK(reinterpret_cast<std::uintptr_t>(
                  b->view().native_handle())
          == a_address + 512);

    // Best-fit reuse: a 1024-byte request lands exactly on the freed
    // 1024-byte block.
    auto reused = device->create_tensor(full);
    CHECK(reinterpret_cast<std::uintptr_t>(
                  reused->view().native_handle())
          == c_address);

    // Coalescing: freeing the surviving middle block merges all three
    // ranges back into one contiguous 2048-byte block.
    b.reset();
    reused.reset();
    auto whole = device->create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{4, 2, 16, 16}}, iom::DataType::U8});
    REQUIRE(whole != nullptr);
    CHECK(reinterpret_cast<std::uintptr_t>(
                  whole->view().native_handle())
          == a_address);
}

TEST_CASE("CUDA concurrent tensor bookkeeping is race-free and in-arena") {
    require_cuda_hardware();
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{256 * 1024});
    REQUIRE(device != nullptr);

    constexpr int kThreads = 8;
    constexpr int kIterations = 50;
    const iom::TensorSpec small{
            iom::TensorShape{{16, 16}}, iom::DataType::U8};
    const iom::TensorSpec large{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    std::atomic<bool> failed = false;
    std::barrier start_gate(kThreads + 1);

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int worker = 0; worker < kThreads; ++worker) {
        threads.emplace_back([&, worker] {
            try {
                start_gate.arrive_and_wait();
                for (int iteration = 0; iteration < kIterations; ++iteration) {
                    auto first = device->create_tensor(small);
                    auto second = device->create_tensor(large);
                    auto third = device->create_tensor(small);
                    if (reinterpret_cast<std::uintptr_t>(
                                first->view().native_handle())
                                    % 32
                            != 0
                            || reinterpret_cast<std::uintptr_t>(
                                       second->view().native_handle())
                                    % 32
                                    != 0) {
                        failed.store(true, std::memory_order_release);
                    }
                    third.reset();
                    auto fourth = device->create_tensor(large);
                    (void)fourth;
                }
            } catch (...) {
                failed.store(true, std::memory_order_release);
            }
        });
    }
    start_gate.arrive_and_wait();
    for (std::thread& thread : threads) {
        thread.join();
    }
    CHECK_FALSE(failed.load(std::memory_order_acquire));
}
