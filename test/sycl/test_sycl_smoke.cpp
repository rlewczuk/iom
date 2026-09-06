#include <doctest/doctest.h>

#include <sycl/sycl.hpp>
#include <algorithm>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>

#include "iom/alloc.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/sycl/device.hpp"
#include "staging_pool.hpp"
#include "iom/tensor.hpp"
#include "runtime.hpp"

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

struct ContextProbe {
    std::size_t created = 0;
    std::size_t destroyed = 0;
};

ContextProbe* active_probe = nullptr;

void count_context_created() {
    REQUIRE(active_probe != nullptr);
    ++active_probe->created;
}

void count_context_destroyed() {
    REQUIRE(active_probe != nullptr);
    ++active_probe->destroyed;
}

class ContextCallsRestore final {
public:
    ContextCallsRestore()
            : saved_(iom::sycl_detail::context_calls),
              saved_probe_(active_probe) {}

    ContextCallsRestore(const ContextCallsRestore&) = delete;
    ContextCallsRestore& operator=(const ContextCallsRestore&) = delete;

    ~ContextCallsRestore() {
        iom::sycl_detail::context_calls = saved_;
        active_probe = saved_probe_;
    }

private:
    iom::sycl_detail::ContextCalls saved_;
    ContextProbe* saved_probe_;
};

std::size_t eligible_device_count_from_runtime() {
    const auto devices = sycl::device::get_devices();
    return static_cast<std::size_t>(std::count_if(
            devices.begin(), devices.end(), [](const sycl::device& device) {
                return device.is_gpu() || device.is_accelerator();
            }));
}

sycl::device first_accelerator_device() {
    for (const sycl::device& device : sycl::device::get_devices()) {
        if (device.is_gpu() || device.is_accelerator()) {
            return device;
        }
    }
    throw std::runtime_error("SYCL accelerator device is unavailable");
}

}  // namespace

TEST_CASE("SYCL factory enumerates real accelerator devices") {
    const std::size_t runtime_count = eligible_device_count_from_runtime();
    REQUIRE(runtime_count > 0);
    CHECK(iom::sycl_detail::eligible_device_count() == runtime_count);

    UnusedAllocator allocator;
    auto device = iom::make_sycl_device(0, allocator);
    REQUIRE(device != nullptr);
    CHECK(device->backend_kind() == iom::BackendKind::SYCL);
    CHECK(device->backend_device() == 0);
    CHECK(allocator.allocations == 0);

    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    CHECK_THROWS_AS((void)device->create_tensor(spec), std::bad_alloc);
    auto queue = device->create_ops();
    CHECK(queue != nullptr);
}

TEST_CASE("SYCL factory owns and tears down one context") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

    ContextProbe probe;
    ContextCallsRestore restore;
    active_probe = &probe;
    iom::sycl_detail::context_calls = {
            &count_context_created, &count_context_destroyed};

    UnusedAllocator allocator;
    {
        auto device = iom::make_sycl_device(0, allocator);
        REQUIRE(device != nullptr);
        CHECK(probe.created == 1);
        CHECK(probe.destroyed == 0);
    }
    CHECK(probe.created == 1);
    CHECK(probe.destroyed == 1);
}

TEST_CASE("SYCL factory rejects the first unavailable ordinal") {
    const std::size_t device_count = iom::sycl_detail::eligible_device_count();
    REQUIRE(device_count > 0);

    ContextProbe probe;
    ContextCallsRestore restore;
    active_probe = &probe;
    iom::sycl_detail::context_calls = {
            &count_context_created, &count_context_destroyed};

    UnusedAllocator allocator;
    CHECK_THROWS_AS(
            (void)iom::make_sycl_device(
                    static_cast<std::uint32_t>(device_count), allocator),
            std::invalid_argument);
    CHECK(allocator.allocations == 0);
    CHECK(probe.created == 0);
    CHECK(probe.destroyed == 0);
}

TEST_CASE("SYCL staging pool preserves accounting across allocation failures") {
    const sycl::device device = first_accelerator_device();
    const sycl::context context(device);
    iom::sycl_detail::StagingSlotPool pool(context, device);

    CHECK_THROWS_AS(
            pool.acquire(
                    iom::sycl_detail::StagingSlotPool::kMaxStagingBytes + 1),
            std::invalid_argument);
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    pool.fail_next_allocation_for_testing();
    CHECK_THROWS_AS(pool.acquire(4096), std::bad_alloc);
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    {
        auto lease = pool.acquire(4096);
        REQUIRE(lease.staging() != nullptr);
        REQUIRE(lease.host_mirror() != nullptr);
        CHECK_EQ(pool.allocation_count_for_testing(), 1);
        CHECK_EQ(pool.idle_count_for_testing(), 0);
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 1);
    CHECK_EQ(pool.idle_count_for_testing(), 1);

    {
        auto lease = pool.acquire(8192);
        CHECK(lease.capacity() >= 8192);
        CHECK_EQ(pool.allocation_count_for_testing(), 1);
    }
    CHECK_EQ(pool.idle_count_for_testing(), 1);

    {
        auto lease = pool.acquire(4096);
        lease.poison();
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);

    pool.destroy();
    CHECK_EQ(pool.allocation_count_for_testing(), 0);
    CHECK_EQ(pool.idle_count_for_testing(), 0);
}
