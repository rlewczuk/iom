#include <doctest/doctest.h>

#include <sycl/sycl.hpp>
#include <algorithm>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

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
class SyclAllocator final : public iom::Allocator {
public:
    void bind_context(const sycl::context& context) {
        context_ = context;
        const std::vector<sycl::device> devices = context.get_devices();
        REQUIRE(!devices.empty());
        device_ = devices.front();
    }

    void* alloc(std::size_t size) override {
        REQUIRE(context_.has_value());
        void* pointer = sycl::malloc_shared(size, device_, *context_);
        if (pointer == nullptr) {
            throw std::bad_alloc();
        }
        return pointer;
    }

    void free(void* buffer) override {
        REQUIRE(context_.has_value());
        sycl::free(buffer, *context_);
    }

    void reset() override {}

private:
    std::optional<sycl::context> context_;
    sycl::device device_;
};

SyclAllocator* active_allocator = nullptr;

void capture_context(const sycl::context& context) {
    REQUIRE(active_allocator != nullptr);
    active_allocator->bind_context(context);
}

struct LaunchProbe {
    std::size_t launches = 0;
};

LaunchProbe* active_launch_probe = nullptr;

void count_kernel_launch() {
    REQUIRE(active_launch_probe != nullptr);
    ++active_launch_probe->launches;
}

class LaunchCallsRestore final {
public:
    LaunchCallsRestore()
            : saved_(iom::sycl_detail::launch_calls),
              saved_probe_(active_launch_probe) {}

    LaunchCallsRestore(const LaunchCallsRestore&) = delete;
    LaunchCallsRestore& operator=(const LaunchCallsRestore&) = delete;

    ~LaunchCallsRestore() {
        iom::sycl_detail::launch_calls = saved_;
        active_launch_probe = saved_probe_;
    }

private:
    iom::sycl_detail::LaunchCalls saved_;
    LaunchProbe* saved_probe_;
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
              saved_probe_(active_probe),
              saved_allocator_(active_allocator) {}

    ContextCallsRestore(const ContextCallsRestore&) = delete;
    ContextCallsRestore& operator=(const ContextCallsRestore&) = delete;

    ~ContextCallsRestore() {
        iom::sycl_detail::context_calls = saved_;
        active_probe = saved_probe_;
        active_allocator = saved_allocator_;
    }

private:
    iom::sycl_detail::ContextCalls saved_;
    ContextProbe* saved_probe_;
    SyclAllocator* saved_allocator_;
};

namespace {

struct AllocationProbe {
    std::mutex mutex;
    std::vector<iom::sycl_detail::AllocationRecord> records;
};

AllocationProbe* active_allocation_probe = nullptr;

// SYCL queue metadata and binary-fallback allocations happen on the worker
// thread while staging transfers run on the caller, so the probe guards its
// records; the seam itself adds no synchronization.
void capture_allocation(const iom::sycl_detail::AllocationRecord& record) {
    if (active_allocation_probe != nullptr) {
        std::lock_guard<std::mutex> lock(active_allocation_probe->mutex);
        active_allocation_probe->records.push_back(record);
    }
}

void* failing_device_alloc(
        std::size_t, const sycl::device&, const sycl::context&) {
    return nullptr;
}

class AllocationCallsRestore final {
public:
    AllocationCallsRestore()
            : saved_calls_(iom::sycl_detail::allocation_calls),
              saved_observer_(iom::sycl_detail::allocation_observer),
              saved_probe_(active_allocation_probe) {}

    AllocationCallsRestore(const AllocationCallsRestore&) = delete;
    AllocationCallsRestore& operator=(const AllocationCallsRestore&) = delete;

    ~AllocationCallsRestore() {
        iom::sycl_detail::allocation_calls = saved_calls_;
        iom::sycl_detail::allocation_observer = saved_observer_;
        active_allocation_probe = saved_probe_;
    }

private:
    iom::sycl_detail::AllocationCalls saved_calls_;
    iom::sycl_detail::AllocationObserver saved_observer_;
    AllocationProbe* saved_probe_;
};

}  // namespace

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

TEST_CASE("SYCL queued copy submits one kernel regardless of plane count") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

    ContextCallsRestore context_restore;
    SyclAllocator allocator;
    active_allocator = &allocator;
    iom::sycl_detail::context_calls.context_ready = &capture_context;
    auto device = iom::make_sycl_device(0, allocator);
    REQUIRE(device != nullptr);
    auto queue = device->create_ops();
    REQUIRE(queue != nullptr);

    LaunchProbe probe;
    LaunchCallsRestore launch_restore;
    active_launch_probe = &probe;
    iom::sycl_detail::launch_calls.kernel_launched = &count_kernel_launch;

    const auto exercise_copy = [&](const iom::TensorSpec& spec) {
        auto source = device->create_tensor(spec);
        auto destination = device->create_tensor(spec);
        REQUIRE(source != nullptr);
        REQUIRE(destination != nullptr);

        std::vector<std::byte> expected(spec.logical_nbytes());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            expected[index] = std::byte{
                    static_cast<unsigned char>((index * 13 + 7) % 251)};
        }
        source->view().copy_from_host(expected);

        probe.launches = 0;
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        queue->wait(token);
        const std::size_t launches = probe.launches;

        std::vector<std::byte> actual(expected.size());
        destination->view().copy_to_host(actual);
        CHECK(actual == expected);
        return launches;
    };

    const std::size_t single_plane_launches = exercise_copy(
            iom::TensorSpec{
                    iom::TensorShape{{16, 16}}, iom::DataType::F32});
    CHECK(single_plane_launches == 1);

    const std::size_t many_plane_launches = exercise_copy(
            iom::TensorSpec{
                    iom::TensorShape{{2, 32, 32, 64}}, iom::DataType::F32});
    CHECK(many_plane_launches == 1);
    CHECK(many_plane_launches == single_plane_launches);

    auto identical = device->create_tensor(iom::TensorSpec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32});
    REQUIRE(identical != nullptr);
    std::vector<std::byte> identical_data(
            identical->view().spec().logical_nbytes(), std::byte{0x3C});
    identical->view().copy_from_host(identical_data);

    probe.launches = 0;
    const iom::oid identical_token =
            queue->copy(identical->view(), identical->view());
    queue->wait(identical_token);
    CHECK(probe.launches == 0);

    std::vector<std::byte> identical_readback(identical_data.size());
    identical->view().copy_to_host(identical_readback);
    CHECK(identical_readback == identical_data);
}

TEST_CASE(
        "SYCL native allocation seam observes metadata, staging, and "
        "binary-fallback boundaries without caller tensor traffic") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

    ContextCallsRestore context_restore;
    SyclAllocator allocator;
    active_allocator = &allocator;
    iom::sycl_detail::context_calls.context_ready = &capture_context;

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::sycl_detail::allocation_observer.complete = &capture_allocation;

    const iom::TensorSpec staging_spec{
            iom::TensorShape{{1024, 1024}}, iom::DataType::F32};
    // Two leading planes: the binary fallback stages this tensor through
    // three temporary device-USM buffers (one per operand), each exactly
    // one padded plane pair (2 x 1 KiB for F32).
    const iom::TensorSpec binary_spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::F32};
    {
        auto device = iom::make_sycl_device(0, allocator);
        REQUIRE(device != nullptr);
        auto staging_source = device->create_tensor(staging_spec);
        auto staging_destination = device->create_tensor(staging_spec);
        auto lhs = device->create_tensor(binary_spec);
        auto rhs = device->create_tensor(binary_spec);
        auto out = device->create_tensor(binary_spec);
        auto queue = device->create_ops();
        REQUIRE(queue != nullptr);

        // Caller-Allocator tensor traffic (malloc_shared USM) plus
        // factory, tensor, and queue setup never route through the seam:
        // the IOM-native record starts empty.
        CHECK(probe.records.empty());

        std::vector<std::byte> input(
                staging_spec.logical_nbytes(), std::byte{0x3c});
        staging_source->view().copy_from_host(input);
        const iom::oid copy_token = queue->copy(
                staging_source->view(), staging_destination->view());
        REQUIRE(iom::oid_is_token(copy_token));
        queue->wait(copy_token);

        const iom::oid binary_token =
                queue->add(lhs->view(), rhs->view(), out->view());
        REQUIRE(iom::oid_is_token(binary_token));
        queue->wait(binary_token);
    }  // queue and device teardown free the pooled metadata and staging

    std::size_t metadata_allocations = 0;
    std::size_t staging_allocations = 0;
    std::size_t staging_2048 = 0;
    std::size_t failed = 0;
    std::map<void*, std::size_t> outstanding;
    for (const auto& record : probe.records) {
        CHECK(record.phase
              == iom::sycl_detail::AllocationPhase::post_publication);
        CHECK_FALSE(
                record.classification
                == iom::sycl_detail::AllocationClass::data_backing);
        CHECK_FALSE(
                record.classification
                == iom::sycl_detail::AllocationClass::metadata_backing);
        if (record.kind == iom::sycl_detail::AllocationKind::allocate) {
            if (!record.succeeded) {
                ++failed;
                continue;
            }
            CHECK(record.address != nullptr);
            if (record.classification
                == iom::sycl_detail::AllocationClass::staging) {
                ++staging_allocations;
                if (record.bytes == 2048) {
                    ++staging_2048;
                }
            }
            if (record.classification
                == iom::sycl_detail::AllocationClass::operation_metadata) {
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
    CHECK_GE(metadata_allocations, 1u);  // queue copy metadata slot
    CHECK_GE(staging_allocations, 4u);   // staging pool + binary fallback
    CHECK_EQ(staging_2048, 3u);          // the three binary temporary buffers
    CHECK(outstanding.empty());          // every allocation freed by teardown
}

TEST_CASE(
        "SYCL native allocation seam retains failed attempts and cleanup "
        "frees") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::sycl_detail::allocation_observer.complete = &capture_allocation;

    const sycl::device device = first_accelerator_device();
    const sycl::context context(device);
    iom::sycl_detail::StagingSlotPool pool(context, device);

    // A failed native allocation stays observable even though no pointer is
    // returned; the pool surfaces the null result as bad_alloc.
    iom::sycl_detail::allocation_calls.device_alloc = &failing_device_alloc;
    CHECK_THROWS_AS((void)pool.acquire(4096), std::bad_alloc);
    iom::sycl_detail::allocation_calls =
            iom::sycl_detail::AllocationCalls{};

    {
        auto lease = pool.acquire(4096);
        REQUIRE(lease.staging() != nullptr);
        REQUIRE(lease.host_mirror() != nullptr);
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
              == iom::sycl_detail::AllocationClass::staging);
        CHECK(record.phase
              == iom::sycl_detail::AllocationPhase::post_publication);
        if (record.kind == iom::sycl_detail::AllocationKind::allocate) {
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
