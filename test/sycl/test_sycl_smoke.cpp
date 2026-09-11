#include <doctest/doctest.h>

#include <sycl/sycl.hpp>
#include <algorithm>

#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/sycl/device.hpp"
#include "staging_pool.hpp"
#include "iom/tensor.hpp"
#include "copy.hpp"
#include "runtime.hpp"

namespace {

// Standard arena capacity for smoke tests: 64 MiB covers every live tensor
// set below (the largest tensor set is two 1024x1024 F32 staging tensors,
// two 2x32x32x64 F32 exercise tensors, and three binary operands).
constexpr std::size_t kArenaBytes = 64u * 1024 * 1024;

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

    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);
    CHECK(device->backend_kind() == iom::BackendKind::SYCL);
    CHECK(device->backend_device() == 0);

    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    auto tensor = device->create_tensor(spec);
    REQUIRE(tensor != nullptr);
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

    {
        auto device = iom::make_sycl_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
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

    CHECK_THROWS_AS(
            (void)iom::make_sycl_device(
                    static_cast<std::uint32_t>(device_count),
                    iom::DeviceMemoryConfig{kArenaBytes}),
            std::invalid_argument);
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

    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
        "SYCL native allocation seam observes exactly two setup backings and "
        "metadata, staging, and binary-fallback boundaries") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::sycl_detail::allocation_observer.complete = &capture_allocation;

    const iom::TensorSpec staging_spec{
            iom::TensorShape{{1024, 1024}}, iom::DataType::F32};
    // Three temporary host-USM buffers stage each operand through the
    // caller-supplied raw workspace; only the separate copy path uses the
    // device staging pool.
    const iom::TensorSpec binary_spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::F32};
    iom::sycl_detail::AllocationRecord data_backing;
    iom::sycl_detail::AllocationRecord metadata_backing;
    {
        auto device = iom::make_sycl_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
        REQUIRE(device != nullptr);
        auto staging_source = device->create_tensor(staging_spec);
        auto staging_destination = device->create_tensor(staging_spec);
        auto lhs = device->create_tensor(binary_spec);
        auto rhs = device->create_tensor(binary_spec);
        auto out = device->create_tensor(binary_spec);
        auto queue = device->create_ops();
        REQUIRE(queue != nullptr);

        // Factory setup produced exactly two instrumented backing
        // allocations; tensor and queue creation suballocate and never
        // route through the seam.
        std::vector<iom::sycl_detail::AllocationRecord> setup_allocations;
        for (const auto& record : probe.records) {
            if (record.phase == iom::sycl_detail::AllocationPhase::setup
                    && record.kind
                            == iom::sycl_detail::AllocationKind::allocate) {
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
                == iom::sycl_detail::AllocationClass::data_backing) {
                ++data_count;
                data_backing = record;
            }
            if (record.classification
                == iom::sycl_detail::AllocationClass::metadata_backing) {
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
                staging_spec.logical_nbytes(), std::byte{0x3c});
        staging_source->view().copy_from_host(input);
        const iom::oid copy_token = queue->copy(
                staging_source->view(), staging_destination->view());
        REQUIRE(iom::oid_is_token(copy_token));
        queue->wait(copy_token);

        const auto requirements = queue->add_workspace_requirements(
                lhs->view(), rhs->view(), out->view());
        REQUIRE_EQ(requirements.alignment, std::size_t{32});
        auto workspace_owner = device->create_workspace(requirements.bytes);
        const iom::RawWorkspaceView workspace = workspace_owner->view();
        const iom::oid binary_token = queue->add(
                lhs->view(), rhs->view(), out->view(), workspace);
        REQUIRE(iom::oid_is_token(binary_token));
        queue->wait(binary_token);
    }  // queue and device teardown free the pooled metadata, host staging,
       // workspace owner, and the two arena backings

    std::size_t metadata_allocations = 0;
    std::size_t staging_allocations = 0;
    std::size_t staging_2048 = 0;
    std::size_t failed = 0;
    std::map<void*, std::size_t> outstanding;
    // Seed with the two setup backings so their post-publication teardown
    // frees pair up with the reservation that created them.
    outstanding[data_backing.address] = data_backing.bytes;
    outstanding[metadata_backing.address] = metadata_backing.bytes;
    for (const auto& record : probe.records) {
        if (record.phase == iom::sycl_detail::AllocationPhase::setup) {
            continue;
        }
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
    CHECK_EQ(metadata_allocations, 0u);  // fixed slots; no metadata growth
    CHECK_GE(staging_allocations, 1u);   // copy staging pool only
    CHECK_EQ(staging_2048, 0u);          // binary staging uses workspace
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

namespace {

struct BackingProbe {
    iom::sycl_detail::AllocationRecord data;
    iom::sycl_detail::AllocationRecord metadata;
};

// Creates a device with the given memory/queue configuration and returns
// the two classified setup backing records.
[[nodiscard]] BackingProbe probe_setup_backings(
        std::size_t arena_bytes, iom::QueueConfig queue_config) {
    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::sycl_detail::allocation_observer.complete = &capture_allocation;

    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{arena_bytes}, queue_config);
    REQUIRE(device != nullptr);

    BackingProbe result;
    std::size_t data_count = 0;
    std::size_t metadata_count = 0;
    for (const auto& record : probe.records) {
        if (record.phase != iom::sycl_detail::AllocationPhase::setup
                || record.kind
                        != iom::sycl_detail::AllocationKind::allocate) {
            continue;
        }
        REQUIRE(record.succeeded);
        if (record.classification
            == iom::sycl_detail::AllocationClass::data_backing) {
            result.data = record;
            ++data_count;
        }
        if (record.classification
            == iom::sycl_detail::AllocationClass::metadata_backing) {
            result.metadata = record;
            ++metadata_count;
        }
    }
    REQUIRE_EQ(data_count, 1u);
    REQUIRE_EQ(metadata_count, 1u);
    return result;
}

void require_disjoint_domains(
        const iom::sycl_detail::AllocationRecord& data,
        const iom::sycl_detail::AllocationRecord& metadata) {
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
        "SYCL factory setups reserve exactly two backings for default and "
        "custom C") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

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

TEST_CASE("SYCL factory rejects invalid configurations before publication") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

    CHECK_THROWS_AS(iom::QueueConfig{0}, std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::make_sycl_device(
                    0, iom::DeviceMemoryConfig{0}),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::make_sycl_device(
                    0, iom::DeviceMemoryConfig{16}),
            std::invalid_argument);

    // 4 * C * 512 overflows size_t for this capacity.
    CHECK_THROWS_AS(
            (void)iom::make_sycl_device(
                    0, iom::DeviceMemoryConfig{kArenaBytes},
                    iom::QueueConfig{std::numeric_limits<std::size_t>::max()
                                     / 2048 + 1}),
            std::overflow_error);
}

TEST_CASE(
        "SYCL factory rolls back no backing when the data backing "
        "allocation fails") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::sycl_detail::allocation_observer.complete = &capture_allocation;
    iom::sycl_detail::allocation_calls.device_alloc = &failing_device_alloc;

    CHECK_THROWS_AS(
            (void)iom::make_sycl_device(
                    0, iom::DeviceMemoryConfig{kArenaBytes}),
            std::bad_alloc);

    // The failed data-backing attempt stays observable and no backing was
    // freed because none was acquired.
    REQUIRE_EQ(probe.records.size(), 1u);
    const auto& record = probe.records.front();
    CHECK(record.phase == iom::sycl_detail::AllocationPhase::setup);
    CHECK(record.classification
          == iom::sycl_detail::AllocationClass::data_backing);
    CHECK(record.kind == iom::sycl_detail::AllocationKind::allocate);
    CHECK_FALSE(record.succeeded);
    CHECK(record.address == nullptr);
}

TEST_CASE(
        "SYCL factory rolls back the data backing when the metadata backing "
        "allocation fails") {
    REQUIRE(eligible_device_count_from_runtime() > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::sycl_detail::allocation_observer.complete = &capture_allocation;
    iom::sycl_detail::allocation_calls.device_alloc =
            [](std::size_t bytes, const sycl::device& device,
               const sycl::context& context) -> void* {
                static thread_local std::size_t calls = 0;
                if (calls++ == 0) {
                    return sycl::malloc_device(bytes, device, context);
                }
                return nullptr;
            };

    CHECK_THROWS_AS(
            (void)iom::make_sycl_device(
                    0, iom::DeviceMemoryConfig{kArenaBytes}),
            std::bad_alloc);

    // Setup allocated the data backing, failed the metadata backing, then
    // rolled the data backing back through the same seam.
    REQUIRE_EQ(probe.records.size(), 3u);
    const auto& data_alloc = probe.records[0];
    const auto& metadata_alloc = probe.records[1];
    const auto& data_free = probe.records[2];
    CHECK(data_alloc.phase == iom::sycl_detail::AllocationPhase::setup);
    CHECK(data_alloc.classification
          == iom::sycl_detail::AllocationClass::data_backing);
    CHECK(data_alloc.kind == iom::sycl_detail::AllocationKind::allocate);
    CHECK(data_alloc.succeeded);
    CHECK(data_alloc.address != nullptr);
    CHECK_EQ(data_alloc.bytes, kArenaBytes);
    CHECK(metadata_alloc.classification
          == iom::sycl_detail::AllocationClass::metadata_backing);
    CHECK(metadata_alloc.kind == iom::sycl_detail::AllocationKind::allocate);
    CHECK_FALSE(metadata_alloc.succeeded);
    CHECK(data_free.phase == iom::sycl_detail::AllocationPhase::setup);
    CHECK(data_free.classification
          == iom::sycl_detail::AllocationClass::data_backing);
    CHECK(data_free.kind == iom::sycl_detail::AllocationKind::free);
    CHECK(data_free.address == data_alloc.address);
}

TEST_CASE(
        "SYCL tensor create and destroy after setup make no native "
        "allocation or free calls") {
    REQUIRE(eligible_device_count_from_runtime() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::sycl_detail::allocation_observer.complete = &capture_allocation;

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
        "SYCL data arena fragments, reuses, and coalesces with stable live "
        "addresses") {
    REQUIRE(eligible_device_count_from_runtime() > 0);
    // 2048 bytes: two 512-byte blocks and one 1024-byte block fill it.
    auto device = iom::make_sycl_device(
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

TEST_CASE("SYCL concurrent tensor bookkeeping is race-free and in-arena") {
    REQUIRE(eligible_device_count_from_runtime() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{256 * 1024});
    REQUIRE(device != nullptr);

    constexpr int kThreads = 4;
    constexpr int kIterations = 20;
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
                    auto fourth = device->create_tensor(small);
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

namespace {

// Extracts the metadata arena backing range recorded by the native
// allocation seam while the device was constructed.
[[nodiscard]] std::uintptr_t sycl_metadata_backing_begin(
        AllocationProbe& probe) {
    std::lock_guard<std::mutex> lock(probe.mutex);
    for (const auto& record : probe.records) {
        if (record.phase == iom::sycl_detail::AllocationPhase::setup
                && record.classification
                        == iom::sycl_detail::AllocationClass::metadata_backing
                && record.kind == iom::sycl_detail::AllocationKind::allocate
                && record.succeeded) {
            return reinterpret_cast<std::uintptr_t>(record.address);
        }
    }
    FAIL("no metadata backing allocation was observed");
    return 0;
}

}  // namespace

TEST_CASE(
        "SYCL four live queues own four disjoint fixed C-slot partitions "
        "for C = 1, 16, 17") {
    REQUIRE(eligible_device_count_from_runtime() > 0);
    for (const std::size_t slots : {std::size_t{1}, std::size_t{16},
             std::size_t{17}}) {
        AllocationCallsRestore restore;
        AllocationProbe probe;
        active_allocation_probe = &probe;
        iom::sycl_detail::allocation_observer.complete = &capture_allocation;

        auto device = iom::make_sycl_device(
                0, iom::DeviceMemoryConfig{kArenaBytes},
                iom::QueueConfig{slots});
        REQUIRE(device != nullptr);
        const std::uintptr_t metadata_begin =
                sycl_metadata_backing_begin(probe);
        const std::uintptr_t metadata_end =
                metadata_begin + 4 * slots * 512u;
        active_allocation_probe = nullptr;

        std::vector<std::unique_ptr<iom::DeviceOps>> queues;
        std::vector<std::uintptr_t> bases;
        for (std::size_t index = 0; index < 4; ++index) {
            queues.push_back(device->create_ops());
            REQUIRE(queues.back() != nullptr);
            iom::sycl_detail::QueueResourceSnapshot snapshot;
            iom::sycl_detail::queue_resource_snapshot_for_testing(
                    *queues.back(), snapshot);
            CHECK_EQ(snapshot.slot_count, slots);
            CHECK_EQ(snapshot.slot_stride, 512u);
            CHECK_EQ(snapshot.slots_protected, 0u);
            const auto base =
                    reinterpret_cast<std::uintptr_t>(snapshot.device_base);
            // Pointer-copy descriptors address the queue's own partition
            // inside the Device metadata backing at 32-byte alignment.
            CHECK(base % 32 == 0);
            CHECK(base >= metadata_begin);
            CHECK(base + slots * 512u <= metadata_end);
            bases.push_back(base);
        }
        for (std::size_t a = 0; a < bases.size(); ++a) {
            for (std::size_t b = a + 1; b < bases.size(); ++b) {
                const bool disjoint =
                        bases[a] + slots * 512u <= bases[b]
                        || bases[b] + slots * 512u <= bases[a];
                CHECK(disjoint);
            }
        }
    }
}

TEST_CASE("SYCL fifth live queue throws bad_alloc before queue resources") {
    REQUIRE(eligible_device_count_from_runtime() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);
    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);

    std::vector<std::unique_ptr<iom::DeviceOps>> queues;
    for (std::size_t index = 0; index < 4; ++index) {
        queues.push_back(device->create_ops());
        REQUIRE(queues.back() != nullptr);
    }
    CHECK_THROWS_AS((void)device->create_ops(), std::bad_alloc);

    for (const auto& queue : queues) {
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        CHECK(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
    }
}

TEST_CASE("SYCL failed queue construction rolls back every reservation") {
    REQUIRE(eligible_device_count_from_runtime() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);

    std::vector<std::unique_ptr<iom::DeviceOps>> queues;
    for (std::size_t index = 0; index < 3; ++index) {
        queues.push_back(device->create_ops());
        REQUIRE(queues.back() != nullptr);
    }

    // Native queue creation failure during construction.
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::queue_stream_create);
    CHECK_THROWS_AS((void)device->create_ops(), std::runtime_error);
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::none);

    // The rolled-back attempt returned its partition and credit: the fourth
    // live queue is still publishable.
    auto fourth = device->create_ops();
    REQUIRE(fourth != nullptr);
    iom::sycl_detail::QueueResourceSnapshot snapshot;
    iom::sycl_detail::queue_resource_snapshot_for_testing(*fourth, snapshot);
    CHECK_EQ(snapshot.slot_count, 16u);
    CHECK_THROWS_AS((void)device->create_ops(), std::bad_alloc);
}

TEST_CASE("SYCL safely drained queue partitions are reusable") {
    REQUIRE(eligible_device_count_from_runtime() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kArenaBytes}, iom::QueueConfig{1});
    REQUIRE(device != nullptr);

    auto first = device->create_ops();
    REQUIRE(first != nullptr);
    iom::sycl_detail::QueueResourceSnapshot first_snapshot;
    iom::sycl_detail::queue_resource_snapshot_for_testing(
            *first, first_snapshot);
    first.reset();

    auto recreated = device->create_ops();
    REQUIRE(recreated != nullptr);
    iom::sycl_detail::QueueResourceSnapshot recreated_snapshot;
    iom::sycl_detail::queue_resource_snapshot_for_testing(
            *recreated, recreated_snapshot);
    CHECK_EQ(recreated_snapshot.device_base, first_snapshot.device_base);
    CHECK_EQ(recreated_snapshot.slot_count, 1u);
}

TEST_CASE(
        "SYCL queued copies use one fixed slot, protect unproved slots, "
        "and make no post-setup native metadata calls") {
    REQUIRE(eligible_device_count_from_runtime() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);
    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    std::vector<std::byte> pattern(
            spec.logical_nbytes(), static_cast<std::byte>(0x5a));
    source->view().copy_from_host(pattern);

    auto queue = device->create_ops();
    REQUIRE(queue != nullptr);

    // Arm the seam after device setup: queue creation, submissions, waits,
    // and destruction must not touch native device allocation/free for
    // metadata at all.
    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::sycl_detail::allocation_observer.complete = &capture_allocation;

    const iom::oid copy_token =
            queue->copy(source->view(), destination->view());
    REQUIRE(iom::oid_is_token(copy_token));
    CHECK_NOTHROW(queue->wait(copy_token));

    iom::sycl_detail::QueueResourceSnapshot after_copy;
    iom::sycl_detail::queue_resource_snapshot_for_testing(*queue, after_copy);
    CHECK_EQ(after_copy.slots_in_use, 0u);

    // A retained post-launch failure leaves the copy's completion unproved:
    // its fixed slot is protected and never reassigned, and repeated waits
    // keep reporting the original failure.
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::post_launch);
    const iom::oid failed_token =
            queue->copy(source->view(), destination->view());
    REQUIRE(iom::oid_is_token(failed_token));
    CHECK_THROWS_AS(queue->wait(failed_token), std::runtime_error);
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::none);

    iom::sycl_detail::QueueResourceSnapshot protected_snapshot;
    iom::sycl_detail::queue_resource_snapshot_for_testing(
            *queue, protected_snapshot);
    CHECK_EQ(protected_snapshot.slots_protected, 1u);

    queue.reset();
    CHECK(probe.records.empty());
}
