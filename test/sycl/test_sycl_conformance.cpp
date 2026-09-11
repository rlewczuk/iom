#include <doctest/doctest.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "iom/alloc.hpp"
#include "backend/backend_conformance_add.hpp"
#include "iom/cpu/device.hpp"
#include "iom/sycl/device.hpp"
#include "copy.hpp"
#include "staging_pool.hpp"
#include "runtime.hpp"
#include "iom/gpu_algorithm.hpp"
namespace {


class HostAllocator final : public iom::Allocator {
public:
    explicit HostAllocator(const iom_conformance::TrafficGate& gate) : gate_(gate) {}

    void* alloc(std::size_t size) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage allocated inside a transfer, transform, or operation");
        void* pointer = ::operator new(size, std::align_val_t(32));
        live_.insert(pointer);
        return pointer;
    }

    void free(void* buffer) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage freed inside a transfer, transform, or operation");
        const auto found = live_.find(buffer);
        REQUIRE_MESSAGE(
                found != live_.end(),
                "allocator freed an address it never handed out");
        live_.erase(found);
        ::operator delete(buffer, std::align_val_t(32));
    }

    void reset() override {}

private:
    const iom_conformance::TrafficGate& gate_;
    std::unordered_set<void*> live_;
};

// Every standard-GPU conformance fixture reserves this tensor-data arena:
// the largest concurrent set in the shared suite stays well below 512 MiB.
constexpr std::size_t kConformanceArenaBytes = 640u * 1024 * 1024;

// The factory's context_ready seam delivers each device's private context
// to the fixture so storage oracles and pooled-transfer helpers can build
// queues in the exact device context.
std::optional<sycl::context>* active_context_sink = nullptr;

void capture_context(const sycl::context& context) {
    REQUIRE(active_context_sink != nullptr);
    active_context_sink->emplace(context);
}

class ContextCallsRestore final {
public:
    ContextCallsRestore()
            : saved_(iom::sycl_detail::context_calls),
              saved_sink_(active_context_sink) {}

    ContextCallsRestore(const ContextCallsRestore&) = delete;
    ContextCallsRestore& operator=(const ContextCallsRestore&) = delete;

    ~ContextCallsRestore() {
        iom::sycl_detail::context_calls = saved_;
        active_context_sink = saved_sink_;
    }

private:
    iom::sycl_detail::ContextCalls saved_;
    std::optional<sycl::context>* saved_sink_;
};

struct SyclDevices {
    ContextCallsRestore context_restore;
    iom_conformance::TrafficGate gate;
    HostAllocator reference_allocator{gate};
    std::optional<sycl::context> candidate_context;
    std::optional<sycl::context> foreign_context;
    std::unique_ptr<iom::Device> reference;
    std::unique_ptr<iom::Device> candidate;
    std::unique_ptr<iom::Device> foreign;

    SyclDevices() {
        iom::sycl_detail::context_calls.context_ready = &capture_context;
        reference = iom::make_cpu_device(reference_allocator);

        active_context_sink = &candidate_context;
        candidate = iom::make_sycl_device(
                0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
        active_context_sink = &foreign_context;
        foreign = iom::make_sycl_device(
                0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
        active_context_sink = nullptr;
        REQUIRE(candidate_context.has_value());
        REQUIRE(foreign_context.has_value());
    }

    [[nodiscard]] iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};

class SyclStorageOracle final
        : public iom_conformance::AcceleratorStorageOracle {
public:
    explicit SyclStorageOracle(const sycl::context& context)
            : context_(context) {}

    void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        REQUIRE_EQ(encoded.size(), bytes);
        sycl::queue queue(
                context_, context_.get_devices().front(),
                sycl::property_list{sycl::property::queue::in_order{}});
        queue.memcpy(view.native_handle(), encoded.data(), bytes)
                .wait_and_throw();
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        std::vector<std::byte> result(bytes);
        sycl::queue queue(
                context_, context_.get_devices().front(),
                sycl::property_list{sycl::property::queue::in_order{}});
        queue.memcpy(result.data(), view.native_handle(), bytes)
                .wait_and_throw();
        return result;
    }

private:
    sycl::context context_;
};

}  // namespace
TEST_CASE("Device::supported_data_types returns the per-backend 23-entry span") {
    SyclDevices devices;
    iom_conformance::require_standard_capabilities(
            devices.candidate->supported_data_types());
}

TEST_CASE("SYCL tensor destruction fences queued work") {
    REQUIRE(iom::sycl_detail::eligible_device_count() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    const std::vector<std::byte> pattern(
            spec.logical_nbytes(), static_cast<std::byte>(0x3c));
    const std::vector<std::byte> zero(
            spec.logical_nbytes(), static_cast<std::byte>(0));
    source->view().copy_from_host(pattern);
    destination->view().copy_from_host(zero);

    auto queue = device->create_ops();
    const iom::oid token = queue->copy(
            source->view(), destination->view());
    source.reset();
    auto fresh_source = device->create_tensor(spec);
    fresh_source->view().copy_from_host(
            std::vector<std::byte>(
                    spec.logical_nbytes(), static_cast<std::byte>(0xa5)));

    CHECK_NOTHROW(queue->wait(token));
    std::vector<std::byte> observed(spec.logical_nbytes());
    destination->view().copy_to_host(observed);
    CHECK(observed == pattern);

    fresh_source.reset();
    destination.reset();
    queue.reset();
    device.reset();
}

TEST_CASE("SYCL pre-enqueue failures preserve submission sequences") {
    SyclDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    const iom::sycl_detail::SubmissionFault faults[] = {
            iom::sycl_detail::SubmissionFault::state_allocation,
            iom::sycl_detail::SubmissionFault::fence_construction,
            iom::sycl_detail::SubmissionFault::outcome_insertion,
            iom::sycl_detail::SubmissionFault::first_submit,
    };

    for (const auto fault : faults) {
        auto queue = devices.candidate->create_ops();
        iom::sycl_detail::inject_submission_fault_for_testing(fault);
        const iom::oid result = queue->copy(
                source->view(), destination->view());
        const iom::OidError expected =
                fault == iom::sycl_detail::SubmissionFault::first_submit
                ? iom::OidError::DeviceError
                : iom::OidError::ResourceExhausted;
        CHECK_EQ(result, iom::to_oid(expected));
        iom::sycl_detail::inject_submission_fault_for_testing(
                iom::sycl_detail::SubmissionFault::none);
        const iom::oid token = queue->copy(
                source->view(), destination->view());
        CHECK_EQ(iom_conformance::token_sequence(token), 1);
        CHECK_NOTHROW(queue->wait(token));
    }
}

TEST_CASE("SYCL submission remains transactional across post-enqueue failures") {
    REQUIRE(iom::sycl_detail::eligible_device_count() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    const void* source_address = source->view().native_handle();
    const void* destination_address =
            destination->view().native_handle();
    auto queue = device->create_ops();

    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::post_launch);
    const iom::oid token = queue->copy(
            source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(token), 1);
    iom_conformance::expect_repeated_runtime_failure(*queue, token);

    source.reset();
    destination.reset();
    auto fresh_source = device->create_tensor(spec);
    auto fresh_destination = device->create_tensor(spec);
    CHECK_NE(fresh_source->view().native_handle(), source_address);
    CHECK_NE(
            fresh_destination->view().native_handle(), destination_address);
    CHECK_NE(
            fresh_source->view().native_handle(),
            fresh_destination->view().native_handle());

    fresh_source.reset();
    fresh_destination.reset();
    queue.reset();
    device.reset();
}

TEST_CASE("SYCL queue destruction fences pending copies") {
    REQUIRE(iom::sycl_detail::eligible_device_count() > 0);
    auto device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    const void* source_address = source->view().native_handle();
    const void* destination_address =
            destination->view().native_handle();

    {
        auto queue = device->create_ops();
        CHECK(iom::oid_is_token(queue->copy(source->view(), destination->view())));
        CHECK(iom::oid_is_token(queue->copy(source->view(), destination->view())));
        iom::sycl_detail::inject_submission_fault_for_testing(
                iom::sycl_detail::SubmissionFault::post_launch);
        CHECK(iom::oid_is_token(queue->copy(source->view(), destination->view())));
        iom::sycl_detail::inject_submission_fault_for_testing(
                iom::sycl_detail::SubmissionFault::none);
    }

    source.reset();
    destination.reset();
    auto fresh_source = device->create_tensor(spec);
    auto fresh_destination = device->create_tensor(spec);
    CHECK_NE(fresh_source->view().native_handle(), source_address);
    CHECK_NE(
            fresh_destination->view().native_handle(), destination_address);
    fresh_source.reset();
    fresh_destination.reset();
    device.reset();
}

TEST_CASE("SYCL identical-window copy is a no-op") {
    SyclDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::U8};
    auto tensor = devices.candidate->create_tensor(spec);
    auto queue = devices.candidate->create_ops();
    iom::sycl_detail::reset_fence_wait_count_for_testing();
    const iom::oid token = queue->copy(tensor->view(), tensor->view());
    CHECK_NOTHROW(queue->wait(token));
    CHECK_EQ(iom::sycl_detail::fence_wait_count_for_testing(), 0);
}

TEST_CASE("SYCL host transfers reuse pooled staging") {
    SyclDevices devices;
    const iom::TensorSpec odd_spec{
            iom::TensorShape{{1, 17}}, iom::DataType::U4};
    auto odd_tensor = devices.candidate->create_tensor(odd_spec);
    sycl::queue transfer_queue(
            *devices.candidate_context,
            devices.candidate_context->get_devices().front(),
            sycl::property_list{sycl::property::queue::in_order{}});
    iom::sycl_detail::StagingSlotPool pool(
            *devices.candidate_context,
            devices.candidate_context->get_devices().front());

    std::vector<std::byte> odd_source(odd_spec.logical_nbytes());
    for (std::size_t index = 0; index < odd_source.size(); ++index) {
        odd_source[index] = static_cast<std::byte>(index * 13 + 7);
    }
    odd_source.back() &= std::byte{0x0f};
    std::vector<std::byte> odd_result(odd_source.size());
    for (int iteration = 0; iteration < 3; ++iteration) {
        iom::sycl_detail::region_from_host(
                pool, transfer_queue, odd_tensor->view(),
                odd_tensor->view().native_handle(), odd_source);
        iom::sycl_detail::region_to_host(
                pool, transfer_queue, odd_tensor->view(),
                odd_tensor->view().native_handle(), odd_result);
        CHECK(odd_result == odd_source);
    }
    CHECK_EQ(pool.allocation_count_for_testing(), 1);
    CHECK_EQ(pool.idle_count_for_testing(), 1);

    const iom::TensorSpec aligned_spec{
            iom::TensorShape{{1, 16}}, iom::DataType::U8};
    auto aligned_tensor = devices.candidate->create_tensor(aligned_spec);
    std::vector<std::byte> aligned_source(aligned_spec.logical_nbytes());
    for (std::size_t index = 0; index < aligned_source.size(); ++index) {
        aligned_source[index] = static_cast<std::byte>(0xa0 + index);
    }
    std::vector<std::byte> aligned_result(aligned_source.size());
    iom::sycl_detail::region_from_host(
            pool, transfer_queue, aligned_tensor->view(),
            aligned_tensor->view().native_handle(), aligned_source);
    iom::sycl_detail::region_to_host(
            pool, transfer_queue, aligned_tensor->view(),
            aligned_tensor->view().native_handle(), aligned_result);
    CHECK(aligned_result == aligned_source);
    CHECK_EQ(pool.allocation_count_for_testing(), 1);
    CHECK_EQ(pool.idle_count_for_testing(), 1);
}

TEST_CASE("SyclFenceState::result() idempotency and snapshot-after-clear") {
    SyclDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    auto queue = devices.candidate->create_ops();
    iom::sycl_detail::reset_fence_wait_count_for_testing();
    const iom::oid token = queue->copy(
            source->view(), destination->view());

    source.reset();
    CHECK_NOTHROW(queue->wait(token));
    destination.reset();
    CHECK_EQ(iom::sycl_detail::fence_wait_count_for_testing(), 1);
    CHECK_NOTHROW(queue->wait(token));
    CHECK_EQ(iom::sycl_detail::fence_wait_count_for_testing(), 1);
}


TEST_CASE("SYCL conformance: storage and host transfers for every leaf type") {
    SyclDevices devices;
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: storage oracle identifies perturbed transfer map") {
    SyclDevices devices;
    const std::span<const iom::DataType> one_type =
            devices.candidate->supported_data_types().subspan(0, 1);
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), one_type, &devices.gate);

    SyclStorageOracle direct(*devices.candidate_context);
    iom_conformance::PermutingStorageOracle perturbed(
            direct, iom_conformance::swap_first_adjacent_slots);
    CHECK_FALSE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, perturbed, &devices.gate,
            false, false));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: storage oracle covers every leaf width and padded shape") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            oracle, &devices.gate));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: asynchronous copies against the CPU reference") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate, &oracle);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: copy validation fails before writes and sequences") {
    SyclDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: transfer failures keep metadata and ownership") {
    SyclDevices devices;
    iom_conformance::run_transfer_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: deferred queue lifetime and stability") {
    SyclDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: compute methods reject capability without submitting") {
    SyclDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate, "SYCL", true);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: full shared suite") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, &oracle, true);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: binary requests use native queue and owner registry") {
    SyclDevices devices;
    iom_conformance::run_binary_request_conformance(devices.conformance());
    iom_conformance::run_binary_rank_boundary_conformance(
            *devices.candidate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: binary MUL SUB and floating DIV values through real queue") {
    SyclDevices devices;
    for (const auto operation : {
                 iom_conformance::BinaryOperation::add,
                 iom_conformance::BinaryOperation::mul,
                 iom_conformance::BinaryOperation::sub,
                 iom_conformance::BinaryOperation::div}) {
        iom_conformance::run_binary_value_conformance(
                *devices.candidate, operation);
    }
    CHECK_FALSE(devices.gate.armed());
}

namespace {

// File-scope sink for the leaf-01 allocation instrumentation, used to
// prove raw-workspace creation, exhaustion, and the pure requirement
// queries never reserve native device storage.
std::vector<iom::sycl_detail::AllocationRecord>* g_workspace_records =
        nullptr;

void capture_workspace_allocations(
        const iom::sycl_detail::AllocationRecord& record) {
    if (g_workspace_records != nullptr) {
        g_workspace_records->push_back(record);
    }
}

struct WorkspaceObserverRestore final {
    WorkspaceObserverRestore()
            : saved_(iom::sycl_detail::allocation_observer),
              saved_sink_(g_workspace_records) {}

    WorkspaceObserverRestore(const WorkspaceObserverRestore&) = delete;
    WorkspaceObserverRestore& operator=(const WorkspaceObserverRestore&) =
            delete;

    ~WorkspaceObserverRestore() {
        iom::sycl_detail::allocation_observer = saved_;
        g_workspace_records = saved_sink_;
    }

    iom::sycl_detail::AllocationObserver saved_;
    std::vector<iom::sycl_detail::AllocationRecord>* saved_sink_ = nullptr;
};

[[nodiscard]] std::size_t succeeded_data_backing_allocations(
        const std::vector<iom::sycl_detail::AllocationRecord>& records) {
    std::size_t count = 0;
    for (const auto& record : records) {
        if (record.classification
                == iom::sycl_detail::AllocationClass::data_backing
                && record.kind
                        == iom::sycl_detail::AllocationKind::allocate
                && record.succeeded) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST_CASE("SYCL conformance: raw workspace suballocates the reserved data arena") {
    SyclDevices devices;
    WorkspaceObserverRestore observer_restore;
    std::vector<iom::sycl_detail::AllocationRecord> records;
    g_workspace_records = &records;
    iom::sycl_detail::allocation_observer.complete =
            &capture_workspace_allocations;

    auto& device = *devices.candidate;
    const std::size_t backings_before =
            succeeded_data_backing_allocations(records);

    // Empty owner: valid and allocation-free.
    auto empty = device.create_workspace(0);
    REQUIRE(empty != nullptr);
    CHECK(empty->empty());
    CHECK_EQ(empty->byte_size(), 0);
    CHECK(empty->view().owner_identity() == empty.get());
    CHECK(&empty->view().device() == &device);
    CHECK_EQ(succeeded_data_backing_allocations(records), backings_before);

    // Positive creation suballocates the existing data arena: the range
    // is 32-byte aligned with the required capacity, and no new native
    // data backing appears.
    auto workspace = device.create_workspace(4096);
    REQUIRE(workspace != nullptr);
    const iom::RawWorkspaceView view = workspace->view();
    CHECK_EQ(view.byte_size(), 4096);
    CHECK(view.owner_identity() == workspace.get());
    CHECK_NOTHROW((void)iom::detail::WorkspaceValidation::validated(
            device, view, 4096, 32, {}));
    CHECK_EQ(succeeded_data_backing_allocations(records), backings_before);

    // Exhaustion beyond the remaining contiguous arena range is
    // std::bad_alloc with no fallback backing allocation.
    CHECK_THROWS_AS((void)
            device.create_workspace(640u * 1024 * 1024), std::bad_alloc);
    CHECK_EQ(succeeded_data_backing_allocations(records), backings_before);
}

TEST_CASE("SYCL conformance: workspace requirement queries are pure and exact") {
    SyclDevices devices;
    WorkspaceObserverRestore observer_restore;
    std::vector<iom::sycl_detail::AllocationRecord> records;
    g_workspace_records = &records;
    iom::sycl_detail::allocation_observer.complete =
            &capture_workspace_allocations;

    auto& device = *devices.candidate;
    const iom::TensorSpec rank_two{
            iom::TensorShape{{17, 33}}, iom::DataType::F32};
    const iom::TensorSpec rank_three{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::F32};
    auto two_lhs = device.create_tensor(rank_two);
    auto two_rhs = device.create_tensor(rank_two);
    auto two_out = device.create_tensor(rank_two);
    auto lhs = device.create_tensor(rank_three);
    auto rhs = device.create_tensor(rank_three);
    auto out = device.create_tensor(rank_three);
    auto queue = device.create_ops();

    // Whole-plane padded staging for one {17, 33} F32 plane is
    // 32 * 48 * 4 = 6144 bytes. A rank-two view touches one plane; a
    // rank-three {2, 3, 17, 33} view walks to the highest addressed
    // plane 6 ((2-1)*3 + (3-1)*1 + 1), so each slice stages six planes.
    const std::size_t plane_bytes = 32 * 48 * 4;
    const std::size_t rank_two_total = 3 * plane_bytes;
    const std::size_t rank_three_total =
            3 * (2 * 3 * plane_bytes);
    CHECK(queue->add_workspace_requirements(
                  two_lhs->view(), two_rhs->view(), two_out->view())
          == iom::WorkspaceRequirements{rank_two_total, 32});

    const iom::WorkspaceRequirements expected_rank_three{
            rank_three_total, 32};
    CHECK(queue->add_workspace_requirements(
                  lhs->view(), rhs->view(), out->view())
          == expected_rank_three);
    CHECK(queue->mul_workspace_requirements(
                  lhs->view(), rhs->view(), out->view())
          == expected_rank_three);
    CHECK(queue->sub_workspace_requirements(
                  lhs->view(), rhs->view(), out->view())
          == expected_rank_three);
    CHECK(queue->div_workspace_requirements(
                  lhs->view(), rhs->view(), out->view())
          == expected_rank_three);

    // Host transfers stage the word-padded logical bytes.
    const iom::WorkspaceRequirements transfer{
            iom::gpu_algorithm::compute_staging_size(
                    rank_three.logical_nbytes()),
            32};
    CHECK(lhs->view().copy_from_host_workspace_requirements() == transfer);
    CHECK(lhs->view().copy_to_host_workspace_requirements() == transfer);

    // The queries reserve no native device storage at all.
    CHECK_EQ(succeeded_data_backing_allocations(records), 0);

    // Validation matches the binary operation exactly.
    auto foreign_tensor = devices.reference->create_tensor(rank_three);
    CHECK_THROWS_AS((void)
            queue->add_workspace_requirements(
                    foreign_tensor->view(), rhs->view(), out->view()),
            std::invalid_argument);
    CHECK_FALSE(devices.gate.armed());
}
