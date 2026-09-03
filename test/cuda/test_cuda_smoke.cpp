#include <doctest/doctest.h>

#include <cuda.h>

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>

#include "iom/alloc.hpp"
#include "iom/cuda/device.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"
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
};

DriverCallProbe* active_probe = nullptr;

CUresult counting_primary_ctx_retain(CUcontext* context, CUdevice device) {
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
    return cuCtxSetCurrent(context);
}

CUresult counting_primary_ctx_release(CUdevice device) {
    if (active_probe != nullptr) {
        ++active_probe->release_count;
        active_probe->released_device = device;
    }
    return cuDevicePrimaryCtxRelease(device);
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
