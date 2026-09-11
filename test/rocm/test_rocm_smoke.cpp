#include <doctest/doctest.h>

#include <hip/hip_runtime_api.h>

#include <cstring>

#include <cstddef>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <new>
#include <string_view>
#include <thread>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/rocm/device.hpp"
#include "copy.hpp"
#include "iom/tensor.hpp"

namespace {

// Standard arena capacity for smoke tests: 64 MiB covers every live tensor
// set below (the largest single tensor is 2048x2048 F32 = 16 MiB, with at
// most two such tensors live together).
constexpr std::size_t kArenaBytes = 64u * 1024 * 1024;

}  // namespace

namespace {

struct AllocationProbe {
    std::vector<iom::rocm_detail::AllocationRecord> records;
};

AllocationProbe* active_allocation_probe = nullptr;

void capture_allocation(const iom::rocm_detail::AllocationRecord& record) {
    if (active_allocation_probe != nullptr) {
        active_allocation_probe->records.push_back(record);
    }
}

hipError_t failing_mem_alloc(void**, std::size_t) {
    return hipErrorOutOfMemory;
}

class AllocationCallsRestore final {
public:
    AllocationCallsRestore()
            : saved_calls_(iom::rocm_detail::allocation_calls),
              saved_observer_(iom::rocm_detail::allocation_observer),
              saved_probe_(active_allocation_probe) {}

    AllocationCallsRestore(const AllocationCallsRestore&) = delete;
    AllocationCallsRestore& operator=(const AllocationCallsRestore&) = delete;

    ~AllocationCallsRestore() {
        iom::rocm_detail::allocation_calls = saved_calls_;
        iom::rocm_detail::allocation_observer = saved_observer_;
        active_allocation_probe = saved_probe_;
    }

private:
    iom::rocm_detail::AllocationCalls saved_calls_;
    iom::rocm_detail::AllocationObserver saved_observer_;
    AllocationProbe* saved_probe_;
};

}  // namespace

TEST_CASE("ROCm factory reports a live hardware device and selects it current") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    {
        auto device = iom::make_rocm_device(
                0, iom::DeviceMemoryConfig{kArenaBytes});
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
    }

    // A second construction after the first owner leaves scope remains
    // deterministic: the factory re-selects the device on this thread.
    auto recreated = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(recreated != nullptr);
    CHECK(recreated->backend_kind() == iom::BackendKind::ROCM);
    CHECK(recreated->backend_device() == 0);
}

TEST_CASE("ROCm factory rejects the first unavailable ordinal") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::rocm_detail::allocation_observer.complete = &capture_allocation;

    CHECK_THROWS_AS(
            (void)iom::make_rocm_device(
                    static_cast<std::uint32_t>(device_count),
                    iom::DeviceMemoryConfig{kArenaBytes}),
            std::invalid_argument);
    CHECK(probe.records.empty());
}

TEST_CASE("ROCm staging pool preserves accounting across allocation failures") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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

namespace {

// Extracts the metadata arena backing range recorded by the native
// allocation seam while the device was constructed.
[[nodiscard]] std::uintptr_t metadata_backing_begin(
        const AllocationProbe& probe) {
    for (const auto& record : probe.records) {
        if (record.phase == iom::rocm_detail::AllocationPhase::setup
                && record.classification
                        == iom::rocm_detail::AllocationClass::metadata_backing
                && record.kind == iom::rocm_detail::AllocationKind::allocate
                && record.succeeded) {
            return reinterpret_cast<std::uintptr_t>(record.address);
        }
    }
    FAIL("no metadata backing allocation was observed");
    return 0;
}

[[nodiscard]] iom::TensorSpec rocm_spec_16x16_f32() {
    return iom::TensorSpec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
}

}  // namespace

TEST_CASE(
        "ROCm four live queues own four disjoint fixed C-slot partitions "
        "for C = 1, 16, 17") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    for (const std::size_t slots : {std::size_t{1}, std::size_t{16},
             std::size_t{17}}) {
        AllocationCallsRestore restore;
        AllocationProbe probe;
        active_allocation_probe = &probe;
        iom::rocm_detail::allocation_observer.complete = &capture_allocation;

        auto device = iom::make_rocm_device(
                0, iom::DeviceMemoryConfig{kArenaBytes},
                iom::QueueConfig{slots});
        REQUIRE(device != nullptr);
        const std::uintptr_t metadata_begin = metadata_backing_begin(probe);
        const std::uintptr_t metadata_end =
                metadata_begin + 4 * slots * 512u;
        active_allocation_probe = nullptr;

        std::vector<std::unique_ptr<iom::DeviceOps>> queues;
        std::vector<std::uintptr_t> bases;
        for (std::size_t index = 0; index < 4; ++index) {
            queues.push_back(device->create_ops());
            REQUIRE(queues.back() != nullptr);
            iom::rocm_detail::QueueResourceSnapshot snapshot;
            iom::rocm_detail::queue_resource_snapshot_for_testing(
                    *queues.back(), snapshot);
            CHECK_EQ(snapshot.slot_count, slots);
            CHECK_EQ(snapshot.slot_stride, 512u);
            CHECK_EQ(snapshot.events_total, slots);
            CHECK_EQ(snapshot.slots_protected, 0u);
            const auto base =
                    reinterpret_cast<std::uintptr_t>(snapshot.device_base);
            // Descriptor backing addresses lie inside the Device metadata
            // backing, keep the allocator's 32-byte slot alignment, and
            // never leave the partition.
            CHECK(base % 32 == 0);
            CHECK(base >= metadata_begin);
            CHECK(base + slots * 512u <= metadata_end);
            bases.push_back(base);
        }
        // The four partitions are pairwise disjoint.
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

TEST_CASE(
        "ROCm fifth live queue throws bad_alloc before native queue "
        "resources") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);
    const iom::TensorSpec spec = rocm_spec_16x16_f32();
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);

    std::vector<std::unique_ptr<iom::DeviceOps>> queues;
    for (std::size_t index = 0; index < 4; ++index) {
        queues.push_back(device->create_ops());
        REQUIRE(queues.back() != nullptr);
    }

    const auto events_before =
            iom::rocm_detail::event_create_count_for_testing.load();
    const auto streams_before =
            iom::rocm_detail::stream_create_count_for_testing.load();
    CHECK_THROWS_AS((void)device->create_ops(), std::bad_alloc);
    // The rejected fifth queue created no stream, no worker, and no
    // completion resource.
    CHECK_EQ(iom::rocm_detail::event_create_count_for_testing.load(),
            events_before);
    CHECK_EQ(iom::rocm_detail::stream_create_count_for_testing.load(),
            streams_before);

    // Every live queue still owns its lease and processes work.
    for (const auto& queue : queues) {
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        CHECK(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
    }
}

TEST_CASE("ROCm failed queue construction rolls back every reservation") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);

    std::vector<std::unique_ptr<iom::DeviceOps>> queues;
    for (std::size_t index = 0; index < 3; ++index) {
        queues.push_back(device->create_ops());
        REQUIRE(queues.back() != nullptr);
    }

    // Completion-resource failure during construction.
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::queue_event_create);
    CHECK_THROWS_AS((void)device->create_ops(), std::runtime_error);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
    // Native stream failure during construction.
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::queue_stream_create);
    CHECK_THROWS_AS((void)device->create_ops(), std::runtime_error);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);

    // Both rolled-back attempts returned their partition and credit: the
    // fourth live queue is still publishable.
    auto fourth = device->create_ops();
    REQUIRE(fourth != nullptr);
    iom::rocm_detail::QueueResourceSnapshot snapshot;
    iom::rocm_detail::queue_resource_snapshot_for_testing(*fourth, snapshot);
    CHECK_EQ(snapshot.slot_count, 16u);
    CHECK_EQ(snapshot.events_total, 16u);
    CHECK_THROWS_AS((void)device->create_ops(), std::bad_alloc);
}

TEST_CASE("ROCm safely drained queue partitions are reusable") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes}, iom::QueueConfig{1});
    REQUIRE(device != nullptr);

    auto first = device->create_ops();
    REQUIRE(first != nullptr);
    iom::rocm_detail::QueueResourceSnapshot first_snapshot;
    iom::rocm_detail::queue_resource_snapshot_for_testing(
            *first, first_snapshot);
    first.reset();

    // The safely drained queue returned its partition and queue-count
    // reservation, and a recreated queue receives the same geometry.
    auto recreated = device->create_ops();
    REQUIRE(recreated != nullptr);
    iom::rocm_detail::QueueResourceSnapshot recreated_snapshot;
    iom::rocm_detail::queue_resource_snapshot_for_testing(
            *recreated, recreated_snapshot);
    CHECK_EQ(recreated_snapshot.device_base, first_snapshot.device_base);
    CHECK_EQ(recreated_snapshot.slot_count, 1u);
    CHECK_EQ(recreated_snapshot.events_total, 1u);
}

TEST_CASE(
        "ROCm unknown completion quarantines the queue lease until a "
        "covering proof") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes}, iom::QueueConfig{1});
    REQUIRE(device != nullptr);
    const iom::TensorSpec spec = rocm_spec_16x16_f32();
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);

    auto unknown = device->create_ops();
    REQUIRE(unknown != nullptr);
    iom::rocm_detail::QueueResourceSnapshot unknown_snapshot;
    iom::rocm_detail::queue_resource_snapshot_for_testing(
            *unknown, unknown_snapshot);

    // Double fault: both event records fail and the stream drain fails, so
    // the submission retires with unknown completion.
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::event_record);
    const iom::oid unresolved =
            unknown->copy(source->view(), destination->view());
    CHECK(iom::oid_is_token(unresolved));
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
    // The covering drain at destruction fails too, so the whole lease is
    // quarantined instead of released.
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::stream_synchronize);
    unknown.reset();
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);

    // Three more queues fit; the quarantined lease keeps its own partition.
    std::vector<std::unique_ptr<iom::DeviceOps>> live;
    for (std::size_t index = 0; index < 3; ++index) {
        auto queue = device->create_ops();
        REQUIRE(queue != nullptr);
        iom::rocm_detail::QueueResourceSnapshot snapshot;
        iom::rocm_detail::queue_resource_snapshot_for_testing(
                *queue, snapshot);
        CHECK_NE(snapshot.device_base, unknown_snapshot.device_base);
        live.push_back(std::move(queue));
    }
    // Live plus quarantined leases exhaust all four credits: a fifth queue
    // cannot reuse the quarantined partition.
    CHECK_THROWS_AS((void)device->create_ops(), std::bad_alloc);

    // Even with one live queue safely drained, the recreated queue receives
    // the drained partition - never the quarantined one.
    iom::rocm_detail::QueueResourceSnapshot drained_snapshot;
    iom::rocm_detail::queue_resource_snapshot_for_testing(
            *live.back(), drained_snapshot);
    live.pop_back();
    auto fifth = device->create_ops();
    REQUIRE(fifth != nullptr);
    iom::rocm_detail::QueueResourceSnapshot fifth_snapshot;
    iom::rocm_detail::queue_resource_snapshot_for_testing(
            *fifth, fifth_snapshot);
    CHECK_EQ(fifth_snapshot.device_base, drained_snapshot.device_base);
    CHECK_NE(fifth_snapshot.device_base, unknown_snapshot.device_base);

    // A covering proof for the quarantined queue reclaims its partition and
    // reservation without changing the original token outcome.
    iom::rocm_detail::reclaim_retained_queue_leases_for_testing(*device);
    auto reclaimed = device->create_ops();
    REQUIRE(reclaimed != nullptr);
    iom::rocm_detail::QueueResourceSnapshot reclaimed_snapshot;
    iom::rocm_detail::queue_resource_snapshot_for_testing(
            *reclaimed, reclaimed_snapshot);
    CHECK_EQ(reclaimed_snapshot.device_base, unknown_snapshot.device_base);
    const iom::oid token =
            reclaimed->copy(source->view(), destination->view());
    CHECK(iom::oid_is_token(token));
    CHECK_NOTHROW(reclaimed->wait(token));
}

TEST_CASE(
        "ROCm inline copies, no-op copies, and binary operations run "
        "without post-setup native calls or dummy slots") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);
    const iom::TensorSpec copy_spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    const iom::TensorSpec binary_spec{
            iom::TensorShape{{2, 2, 16, 16}}, iom::DataType::F32};
    auto source = device->create_tensor(copy_spec);
    auto destination = device->create_tensor(copy_spec);
    auto lhs = device->create_tensor(binary_spec);
    auto rhs = device->create_tensor(binary_spec);
    auto out = device->create_tensor(binary_spec);

    auto queue = device->create_ops();
    REQUIRE(queue != nullptr);

    // Arm the seam after device setup: queue creation, submissions,
    // dispatch, retirement, waits, and queue destruction must not touch
    // native allocation/free at all.
    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::rocm_detail::allocation_observer.complete = &capture_allocation;

    iom::rocm_detail::QueueResourceSnapshot before;
    iom::rocm_detail::queue_resource_snapshot_for_testing(*queue, before);

    const iom::oid copy_token =
            queue->copy(source->view(), destination->view());
    CHECK(iom::oid_is_token(copy_token));
    // A no-op copy never consumes a metadata slot.
    const iom::oid no_op_token =
            queue->copy(source->view(), source->view());
    CHECK(iom::oid_is_token(no_op_token));
    const iom::oid binary_token =
            queue->add(lhs->view(), rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(binary_token));
    CHECK_NOTHROW(queue->wait(copy_token));
    CHECK_NOTHROW(queue->wait(no_op_token));
    CHECK_NOTHROW(queue->wait(binary_token));

    iom::rocm_detail::QueueResourceSnapshot after;
    iom::rocm_detail::queue_resource_snapshot_for_testing(*queue, after);
    CHECK_EQ(after.slots_in_use, 0u);
    CHECK_EQ(after.events_in_use, 0u);

    queue.reset();
    CHECK(probe.records.empty());
}

TEST_CASE(
        "ROCm simultaneous submissions keep independent snapshots and C "
        "bounds native in-flight work") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes}, iom::QueueConfig{17});
    REQUIRE(device != nullptr);
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 2, 16, 16}}, iom::DataType::F32};
    auto lhs = device->create_tensor(spec);
    auto rhs = device->create_tensor(spec);
    auto first_out = device->create_tensor(spec);
    auto second_out = device->create_tensor(spec);
    std::vector<std::byte> lhs_bytes(spec.logical_nbytes());
    std::vector<std::byte> rhs_bytes(spec.logical_nbytes());
    for (std::size_t index = 0; index < lhs_bytes.size(); ++index) {
        lhs_bytes[index] = std::byte{static_cast<unsigned char>(index % 251)};
        rhs_bytes[index] = std::byte{static_cast<unsigned char>((index * 7)
                                                               % 251)};
    }
    lhs->view().copy_from_host(lhs_bytes);
    rhs->view().copy_from_host(rhs_bytes);

    auto queue = device->create_ops();
    REQUIRE(queue != nullptr);
    // Simultaneous views of the same owners on one queue snapshot
    // independently: both in-flight descriptors keep their own fixed slot
    // and mapping, and both outputs are exact.
    const iom::oid first =
            queue->add(lhs->view(), rhs->view(), first_out->view());
    const iom::oid second =
            queue->add(rhs->view(), lhs->view(), second_out->view());
    REQUIRE(iom::oid_is_token(first));
    REQUIRE(iom::oid_is_token(second));
    CHECK_NOTHROW(queue->wait(first));
    CHECK_NOTHROW(queue->wait(second));
    // lhs + rhs and rhs + lhs round-trip bitwise identically for F32, and
    // each in-flight descriptor keeps its own fixed slot and mapping.
    std::vector<float> lhs_values(lhs_bytes.size() / sizeof(float));
    std::vector<float> rhs_values(rhs_bytes.size() / sizeof(float));
    std::memcpy(
            lhs_values.data(), lhs_bytes.data(), lhs_bytes.size());
    std::memcpy(
            rhs_values.data(), rhs_bytes.data(), rhs_bytes.size());
    std::vector<std::byte> expected_bytes(lhs_bytes.size());
    std::vector<float> expected_values(lhs_values.size());
    for (std::size_t index = 0; index < expected_values.size(); ++index) {
        expected_values[index] = lhs_values[index] + rhs_values[index];
    }
    std::memcpy(
            expected_bytes.data(), expected_values.data(),
            expected_bytes.size());
    std::vector<std::byte> first_actual(expected_bytes.size());
    std::vector<std::byte> second_actual(expected_bytes.size());
    first_out->view().copy_to_host(first_actual);
    second_out->view().copy_to_host(second_actual);
    CHECK(first_actual == expected_bytes);
    CHECK(second_actual == expected_bytes);

    // C = 17 rapid submissions on one queue all succeed; the fixed
    // completion-resource set bounds native in-flight work without growth.
    std::vector<iom::oid> tokens;
    for (int index = 0; index < 17; ++index) {
        const iom::oid token =
                queue->copy(lhs->view(), first_out->view());
        CHECK(iom::oid_is_token(token));
        tokens.push_back(token);
    }
    for (const iom::oid token : tokens) {
        CHECK_NOTHROW(queue->wait(token));
    }
    iom::rocm_detail::QueueResourceSnapshot snapshot;
    iom::rocm_detail::queue_resource_snapshot_for_testing(*queue, snapshot);
    CHECK_EQ(snapshot.slot_count, 17u);
    CHECK_EQ(snapshot.events_total, 17u);
    CHECK_EQ(snapshot.events_in_use, 0u);
    CHECK_EQ(snapshot.slots_in_use, 0u);
}

TEST_CASE("ROCm rejected create_tensor leaves the current device unchanged") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
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
}

TEST_CASE(
        "ROCm native allocation seam observes exactly two setup backings and "
        "post-publication staging and metadata boundaries") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::rocm_detail::allocation_observer.complete = &capture_allocation;

    const iom::TensorSpec staging_spec{
            iom::TensorShape{{1024, 1024}}, iom::DataType::F32};
    // The queue binary path backs its descriptor in one fixed 512-byte slot
    // of the queue's reserved partition, so a plain rank-four tensor
    // exercises the fixed-slot path without any native allocation.
    const iom::TensorSpec binary_spec{
            iom::TensorShape{{2, 2, 16, 16}}, iom::DataType::F32};
    iom::rocm_detail::AllocationRecord data_backing;
    iom::rocm_detail::AllocationRecord metadata_backing;
    {
        auto device = iom::make_rocm_device(
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
        std::vector<iom::rocm_detail::AllocationRecord> setup_allocations;
        for (const auto& record : probe.records) {
            if (record.phase == iom::rocm_detail::AllocationPhase::setup
                    && record.kind
                            == iom::rocm_detail::AllocationKind::allocate) {
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
                == iom::rocm_detail::AllocationClass::data_backing) {
                ++data_count;
                data_backing = record;
            }
            if (record.classification
                == iom::rocm_detail::AllocationClass::metadata_backing) {
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
        if (record.phase == iom::rocm_detail::AllocationPhase::setup) {
            continue;
        }
        if (record.kind == iom::rocm_detail::AllocationKind::allocate) {
            if (!record.succeeded) {
                ++failed;
                continue;
            }
            CHECK(record.address != nullptr);
            if (record.classification
                == iom::rocm_detail::AllocationClass::staging) {
                ++staging_allocations;
            }
            if (record.classification
                == iom::rocm_detail::AllocationClass::operation_metadata) {
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
    CHECK_EQ(metadata_allocations, 0u);  // fixed slots; no metadata growth
    CHECK(outstanding.empty());          // every allocation freed by teardown
}

TEST_CASE(
        "ROCm native allocation seam retains failed attempts and cleanup "
        "frees") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);

    // Arm the seam after setup so the record covers only pool traffic: the
    // two setup backing reservations must not pollute this scenario.
    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::rocm_detail::allocation_observer.complete = &capture_allocation;

    iom::rocm_detail::StagingSlotPool pool;

    // A failed native allocation stays observable even though no pointer is
    // returned; check_hip turns the injected boundary error into the
    // runtime failure the pool propagates.
    iom::rocm_detail::allocation_calls.mem_alloc = &failing_mem_alloc;
    CHECK_THROWS_AS((void)pool.acquire(4096), std::runtime_error);
    iom::rocm_detail::allocation_calls =
            iom::rocm_detail::AllocationCalls{};

    {
        auto lease = pool.acquire(4096);
        REQUIRE(lease.staging() != nullptr);
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
              == iom::rocm_detail::AllocationClass::staging);
        CHECK(record.phase
              == iom::rocm_detail::AllocationPhase::post_publication);
        if (record.kind == iom::rocm_detail::AllocationKind::allocate) {
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
    iom::rocm_detail::AllocationRecord data;
    iom::rocm_detail::AllocationRecord metadata;
};

// Creates a device with the given memory/queue configuration and returns
// the two classified setup backing records.
[[nodiscard]] BackingProbe probe_setup_backings(
        std::size_t arena_bytes, iom::QueueConfig queue_config) {
    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::rocm_detail::allocation_observer.complete = &capture_allocation;

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{arena_bytes}, queue_config);
    REQUIRE(device != nullptr);

    BackingProbe result;
    std::size_t data_count = 0;
    std::size_t metadata_count = 0;
    for (const auto& record : probe.records) {
        if (record.phase != iom::rocm_detail::AllocationPhase::setup
                || record.kind
                        != iom::rocm_detail::AllocationKind::allocate) {
            continue;
        }
        REQUIRE(record.succeeded);
        if (record.classification
            == iom::rocm_detail::AllocationClass::data_backing) {
            result.data = record;
            ++data_count;
        }
        if (record.classification
            == iom::rocm_detail::AllocationClass::metadata_backing) {
            result.metadata = record;
            ++metadata_count;
        }
    }
    REQUIRE_EQ(data_count, 1u);
    REQUIRE_EQ(metadata_count, 1u);
    return result;
}

void require_disjoint_domains(
        const iom::rocm_detail::AllocationRecord& data,
        const iom::rocm_detail::AllocationRecord& metadata) {
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
        "ROCm factory setups reserve exactly two backings for default and "
        "custom C") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

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

TEST_CASE("ROCm factory rejects invalid configurations before publication") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    CHECK_THROWS_AS(iom::QueueConfig{0}, std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::make_rocm_device(
                    0, iom::DeviceMemoryConfig{0}),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::make_rocm_device(
                    0, iom::DeviceMemoryConfig{16}),
            std::invalid_argument);

    // 4 * C * 512 overflows size_t for this capacity.
    CHECK_THROWS_AS(
            (void)iom::make_rocm_device(
                    0, iom::DeviceMemoryConfig{kArenaBytes},
                    iom::QueueConfig{std::numeric_limits<std::size_t>::max()
                                     / 2048 + 1}),
            std::overflow_error);
}

TEST_CASE(
        "ROCm factory rolls back no backing when the data backing "
        "allocation fails") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::rocm_detail::allocation_observer.complete = &capture_allocation;
    iom::rocm_detail::allocation_calls.mem_alloc = &failing_mem_alloc;

    CHECK_THROWS_AS(
            (void)iom::make_rocm_device(
                    0, iom::DeviceMemoryConfig{kArenaBytes}),
            std::bad_alloc);

    // The failed data-backing attempt stays observable and no backing was
    // freed because none was acquired.
    REQUIRE_EQ(probe.records.size(), 1u);
    const auto& record = probe.records.front();
    CHECK(record.phase == iom::rocm_detail::AllocationPhase::setup);
    CHECK(record.classification
          == iom::rocm_detail::AllocationClass::data_backing);
    CHECK(record.kind == iom::rocm_detail::AllocationKind::allocate);
    CHECK_FALSE(record.succeeded);
    CHECK(record.address == nullptr);
}

TEST_CASE(
        "ROCm factory rolls back the data backing when the metadata backing "
        "allocation fails") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::rocm_detail::allocation_observer.complete = &capture_allocation;
    iom::rocm_detail::allocation_calls.mem_alloc =
            [](void** address, std::size_t bytes) {
                static thread_local std::size_t calls = 0;
                (void)bytes;
                if (calls++ == 0) {
                    return hipMalloc(address, bytes);
                }
                return hipError_t{hipErrorOutOfMemory};
            };

    CHECK_THROWS_AS(
            (void)iom::make_rocm_device(
                    0, iom::DeviceMemoryConfig{kArenaBytes}),
            std::bad_alloc);

    // Setup allocated the data backing, failed the metadata backing, then
    // rolled the data backing back through the same seam.
    REQUIRE_EQ(probe.records.size(), 3u);
    const auto& data_alloc = probe.records[0];
    const auto& metadata_alloc = probe.records[1];
    const auto& data_free = probe.records[2];
    CHECK(data_alloc.phase == iom::rocm_detail::AllocationPhase::setup);
    CHECK(data_alloc.classification
          == iom::rocm_detail::AllocationClass::data_backing);
    CHECK(data_alloc.kind == iom::rocm_detail::AllocationKind::allocate);
    CHECK(data_alloc.succeeded);
    CHECK(data_alloc.address != nullptr);
    CHECK_EQ(data_alloc.bytes, kArenaBytes);
    CHECK(metadata_alloc.classification
          == iom::rocm_detail::AllocationClass::metadata_backing);
    CHECK(metadata_alloc.kind == iom::rocm_detail::AllocationKind::allocate);
    CHECK_FALSE(metadata_alloc.succeeded);
    CHECK(data_free.phase == iom::rocm_detail::AllocationPhase::setup);
    CHECK(data_free.classification
          == iom::rocm_detail::AllocationClass::data_backing);
    CHECK(data_free.kind == iom::rocm_detail::AllocationKind::free);
    CHECK(data_free.address == data_alloc.address);
}

TEST_CASE(
        "ROCm tensor storage stays inside the disjoint data arena and is "
        "32-byte aligned") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);

    AllocationCallsRestore restore;
    AllocationProbe probe;
    active_allocation_probe = &probe;
    iom::rocm_detail::allocation_observer.complete = &capture_allocation;

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kArenaBytes});
    REQUIRE(device != nullptr);

    const std::uintptr_t data_base = [&] {
        for (const auto& record : probe.records) {
            if (record.phase == iom::rocm_detail::AllocationPhase::setup
                    && record.classification
                            == iom::rocm_detail::AllocationClass::data_backing) {
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
        CHECK(address + spec.tiled_storage_nbytes() <= data_end);
    }
}

TEST_CASE(
        "ROCm data arena fragments, reuses, and coalesces with stable live "
        "addresses") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    // 2048 bytes: two 512-byte blocks and one 1024-byte block fill it.
    auto device = iom::make_rocm_device(
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

TEST_CASE("ROCm concurrent tensor bookkeeping is race-free and in-arena") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    auto device = iom::make_rocm_device(
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
