#include <doctest/doctest.h>

#include <sycl/sycl.hpp>

#if defined(SYCL_EXT_ONEAPI_MATRIX) && SYCL_EXT_ONEAPI_MATRIX == 1
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#define IOM_SYCL_TEST_BF16_MATRIX 1
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include "backend/backend_conformance_cache_append.hpp"
#include "backend/backend_conformance_token_selection.hpp"
#include "backend/backend_conformance_embedding.hpp"
#include "backend/backend_conformance_linear.hpp"
#include "backend/backend_conformance_other.hpp"
#include "backend/backend_conformance_rmsnorm.hpp"
#include "backend/backend_conformance_silu.hpp"
#include "backend/backend_conformance_rope.hpp"
#include "backend/backend_conformance_rope_contract.hpp"
#include "iom/alloc.hpp"
#include "backend/backend_conformance_add.hpp"
#include "backend/backend_conformance_model_loading.hpp"
#include "backend/backend_conformance_sdpa.hpp"
#include "iom/cpu/device.hpp"
#include "iom/sycl/device.hpp"
#include "copy.hpp"
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
    // Cache append's independent storage oracle supplies expected bytes; use
    // a second queue on the exact candidate device for the shared reference
    // role because the CPU backend has no cache-append leaf in this target.
    [[nodiscard]] iom_conformance::ConformanceDevices
            cache_append_conformance() const {
        return {*candidate, *candidate, *foreign};
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
    iom_conformance::copy_from_host(source->view(), pattern);
    iom_conformance::copy_from_host(destination->view(), zero);

    auto queue = device->create_ops();
    const iom::oid token = queue->copy(
            source->view(), destination->view());
    source.reset();
    auto fresh_source = device->create_tensor(spec);
    iom_conformance::copy_from_host(fresh_source->view(), std::vector<std::byte>(
            spec.logical_nbytes(), static_cast<std::byte>(0xa5)));

    CHECK_NOTHROW(queue->wait(token));
    std::vector<std::byte> observed(spec.logical_nbytes());
    iom_conformance::copy_to_host(destination->view(), observed);
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

TEST_CASE("SYCL conformance: greedy token selection shared matrix and lifetime") {
    SyclDevices devices;
    SyclStorageOracle native_storage(*devices.candidate_context);
    const iom::WorkspaceRequirements expected_device_scratch{
            iom::gpu_algorithm::compute_staging_size(
                    17 * sizeof(std::uint16_t)),
            32};
    const iom_conformance::TokenSelectionNativeFailureSeam native_failure{
            {}, {}, "SYCL logical transfer failure",
            "SYCL transfer path has no accepted post-readiness failure seam"};
    const iom_conformance::TokenSelectionConformanceConfig config{
            devices.conformance(),
            devices.candidate->supported_data_types(),
            &devices.gate,
            &native_storage,
            expected_device_scratch,
            native_failure};
    iom_conformance::run_token_selection_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: compute methods reject unsupported capability without submitting") {
    SyclDevices devices;
    // RMS normalization, linear projections, SiLU, and SDPA all have landed
    // SYCL ports, so the shared probe queues their exact expectations through
    // the real path; the remaining unsupported leaves are rejected pure.
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate, "SYCL", true, true, true);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE(
        "SYCL conformance: cache append reference, admission, and lifetime") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    iom_conformance::CacheAppendConformanceConfig config{
            devices.cache_append_conformance(),
            devices.candidate->supported_data_types(),
            &oracle,
            iom_conformance::CacheAppendFaultSeam{
                    [](iom::DeviceOps& queue) {
                        (void)queue;
                        iom::sycl_detail::inject_submission_fault_for_testing(
                                iom::sycl_detail::SubmissionFault::post_launch);
                    }},
            {},
            &devices.gate};
    iom_conformance::run_cache_append_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}


// SYCL's declared embedding expectation: the complete 23-payload/12-index
// matrix the SYCL port must reach and the exact `{32, 32}` status-workspace
// contract, whose control-status read is the four-byte device-to-host copy
// into caller-owned USM memory. Leaf `08-sycl-embedding` implements the
// native parallel_for gather, so the implemented spans mirror the target
// matrix and the shared suite exercises both numerical success and the
// remaining rejection-only paths.
constexpr iom_conformance::EmbeddingDeclaration kSyclEmbeddingDeclaration{
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom::WorkspaceRequirements{32, 32}};

TEST_CASE("SYCL conformance: embedding lookup reference, admission, and lifetime") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    iom_conformance::run_embedding_conformance(
            devices.conformance(), kSyclEmbeddingDeclaration, &devices.gate,
            &oracle);
    CHECK_FALSE(devices.gate.armed());
}

// Regression for the round-2 defect: a data error (deferred OOV
// `std::invalid_argument`) must not erase native completion proof. Many
// repeated OOV embedding calls on one queue and one {32,32} workspace must
// (1) keep reporting `std::invalid_argument` on every wait, (2) leave the
// fixed metadata slot, completion slot, and workspace lease reusable, and
// (3) let a subsequent valid call still succeed and reach the native path.
TEST_CASE(
        "SYCL conformance: repeated OOV embedding does not consume capacity and "
        "lets a later valid call still succeed") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);

    const iom::TensorShape table_shape{{4, 16}};
    const iom::TensorShape index_shape{{1, 8}};
    const iom::TensorShape out_shape{{8, 16}};
    const iom::DataType payload = iom::DataType::U8;
    const iom::DataType index_type = iom::DataType::U8;

    auto table = devices.candidate->create_tensor(
            {table_shape, payload});
    auto indices = devices.candidate->create_tensor(
            {index_shape, index_type});
    auto out = devices.candidate->create_tensor(
            {out_shape, payload});
    auto workspace = devices.candidate->create_workspace(32);

    iom_conformance::copy_from_host(table->view(), std::vector<std::byte>(
            table->view().spec().logical_nbytes(),
            static_cast<std::byte>(0xA5)));
    iom_conformance::copy_from_host(out->view(), std::vector<std::byte>(
            out->view().spec().logical_nbytes(),
            static_cast<std::byte>(0x00)));

    auto queue = devices.candidate->create_ops();
    REQUIRE(queue != nullptr);

    // Pre-snapshot to detect ANY permanent quarantine across the loop.
    iom::sycl_detail::QueueResourceSnapshot before;
    iom::sycl_detail::queue_resource_snapshot_for_testing(
            *queue, before);

    constexpr std::size_t oov_count = 32;
    // 0xFF > V-1 (=3) every iteration: must always throw.
    const std::vector<std::byte> bad_index{
            std::byte{0xFF}, std::byte{0xFF}, std::byte{0xFF},
            std::byte{0xFF}, std::byte{0xFF}, std::byte{0xFF},
            std::byte{0xFF}, std::byte{0xFF}};
    for (std::size_t i = 0; i < oov_count; ++i) {
        iom_conformance::copy_from_host(indices->view(), bad_index);
        const iom::oid oov_token = queue->embedding(
                table->view(), indices->view(), out->view(),
                workspace->view());
        REQUIRE(iom::oid_is_token(oov_token));
        CHECK_THROWS_AS(queue->wait(oov_token), std::invalid_argument);
    }

    // Capacity untouched: no metadata slot was quarantined into a permanent
    // protected state and no completion slot was burned, because each OOV
    // path proved the native event after the kernel wrote the host USM
    // status cell through the event-ordered queue::memcpy.
    iom::sycl_detail::QueueResourceSnapshot after_oov;
    iom::sycl_detail::queue_resource_snapshot_for_testing(
            *queue, after_oov);
    CHECK_EQ(after_oov.slots_in_use, before.slots_in_use);
    CHECK_EQ(after_oov.slots_protected, before.slots_protected);

    // Reusing the SAME {32,32} workspace range after OOV: the next valid
    // embedding must accept the lease and succeed, not return
    // ResourceExhausted.
    const std::vector<std::byte> good_index{
            std::byte{0x00}, std::byte{0x01}, std::byte{0x02},
            std::byte{0x03}, std::byte{0x00}, std::byte{0x01},
            std::byte{0x02}, std::byte{0x03}};
    iom_conformance::copy_from_host(indices->view(), good_index);
    iom_conformance::copy_from_host(out->view(), std::vector<std::byte>(
            out->view().spec().logical_nbytes(),
            static_cast<std::byte>(0x00)));
    const iom::oid good_token = queue->embedding(
            table->view(), indices->view(), out->view(),
            workspace->view());
    REQUIRE(iom::oid_is_token(good_token));
    CHECK_NOTHROW(queue->wait(good_token));

    iom::sycl_detail::QueueResourceSnapshot after_good;
    iom::sycl_detail::queue_resource_snapshot_for_testing(
            *queue, after_good);
    CHECK_EQ(after_good.slots_in_use, 0u);
    CHECK_EQ(after_good.slots_protected, 0u);

    // And OOV still throws after all that.
    iom_conformance::copy_from_host(indices->view(), bad_index);
    const iom::oid oov_again = queue->embedding(
            table->view(), indices->view(), out->view(),
            workspace->view());
    REQUIRE(iom::oid_is_token(oov_again));
    CHECK_THROWS_AS(queue->wait(oov_again), std::invalid_argument);
}

// Regression for the round-3 defect: when an embedding submission has a
// post-launch fault injected after set_event, the deferred OOV completion
// action MUST NOT replace the native post_launch `std::runtime_error`
// with a stale status-cell `std::invalid_argument`. The native fault
// takes precedence; the cached result carries the runtime_error and the
// sticky invalid_argument never re-asserts.
TEST_CASE(
        "SYCL conformance: embedding post-launch native fault surfaces the "
        "native runtime_error, not a stale invalid_argument") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);

    const iom::TensorShape table_shape{{4, 16}};
    const iom::TensorShape index_shape{{1, 8}};
    const iom::TensorShape out_shape{{8, 16}};
    const iom::DataType payload = iom::DataType::U8;
    const iom::DataType index_type = iom::DataType::U8;

    auto table = devices.candidate->create_tensor(
            {table_shape, payload});
    auto indices = devices.candidate->create_tensor(
            {index_shape, index_type});
    auto out = devices.candidate->create_tensor(
            {out_shape, payload});
    auto workspace = devices.candidate->create_workspace(32);

    iom_conformance::copy_from_host(table->view(), std::vector<std::byte>(
            table->view().spec().logical_nbytes(),
            static_cast<std::byte>(0xA5)));
    // Bad index: kernel would set status_cell to 1 if it ever ran, but
    // the post_launch fault happens AFTER set_event so the native fault
    // must dominate; the status-cell read MUST NOT run.
    const std::vector<std::byte> bad_index{
            std::byte{0xFF}, std::byte{0xFF}, std::byte{0xFF},
            std::byte{0xFF}, std::byte{0xFF}, std::byte{0xFF},
            std::byte{0xFF}, std::byte{0xFF}};
    iom_conformance::copy_from_host(indices->view(), bad_index);

    auto queue = devices.candidate->create_ops();
    REQUIRE(queue != nullptr);

    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::post_launch);
    const iom::oid fault_token = queue->embedding(
            table->view(), indices->view(), out->view(),
            workspace->view());
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::none);
    REQUIRE(iom::oid_is_token(fault_token));
    // Native post_launch fault, NOT a stale invalid_argument from the
    // status cell. The status_cell read MUST be suppressed when native
    // completion failed.
    CHECK_THROWS_AS(queue->wait(fault_token), std::runtime_error);
}

// Regression for the F1 (round-5) race: a fence invoke landing in the
// pool-bound-but-event-unset window must NOT observe a phantom native
// completion and MUST NOT cache a success result. The fault seam cannot
// inject this race directly without a controlled window between
// set_completion_slot and set_event, so this case exercises the public
// contract through the same OOV+capacity path: even after the racing
// F1 closure change, repeated OOV embedding calls do not leak state
// (slots_in_use and slots_protected stay at zero) and pending valid
// embedding succeeds on the same workspace. If F1 closure were wrong,
// a fence invoke racing set_event would prove/cache success and the
// OOV read would not fire, the test would fail because OOV would no
// longer throw std::invalid_argument.
TEST_CASE(
        "SYCL conformance: F1 closure preserves OOV throw with capacity "
        "intact across many calls") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);

    const iom::TensorShape table_shape{{4, 16}};
    const iom::TensorShape index_shape{{1, 8}};
    const iom::TensorShape out_shape{{8, 16}};
    const iom::DataType payload = iom::DataType::U8;
    const iom::DataType index_type = iom::DataType::U8;

    auto table = devices.candidate->create_tensor(
            {table_shape, payload});
    auto indices = devices.candidate->create_tensor(
            {index_shape, index_type});
    auto out = devices.candidate->create_tensor(
            {out_shape, payload});
    auto workspace = devices.candidate->create_workspace(32);

    iom_conformance::copy_from_host(table->view(), std::vector<std::byte>(
            table->view().spec().logical_nbytes(),
            static_cast<std::byte>(0xA5)));
    iom_conformance::copy_from_host(out->view(), std::vector<std::byte>(
            out->view().spec().logical_nbytes(),
            static_cast<std::byte>(0x00)));

    auto queue = devices.candidate->create_ops();
    REQUIRE(queue != nullptr);

    const std::vector<std::byte> bad_index{
            std::byte{0xFF}, std::byte{0xFF}, std::byte{0xFF},
            std::byte{0xFF}, std::byte{0xFF}, std::byte{0xFF},
            std::byte{0xFF}, std::byte{0xFF}};
    for (std::size_t i = 0; i < 8; ++i) {
        iom_conformance::copy_from_host(indices->view(), bad_index);
        const iom::oid oov_token = queue->embedding(
                table->view(), indices->view(), out->view(),
                workspace->view());
        REQUIRE(iom::oid_is_token(oov_token));
        CHECK_THROWS_AS(queue->wait(oov_token), std::invalid_argument);
    }

    iom::sycl_detail::QueueResourceSnapshot after_oov;
    iom::sycl_detail::queue_resource_snapshot_for_testing(
            *queue, after_oov);
    CHECK_EQ(after_oov.slots_in_use, 0u);
    CHECK_EQ(after_oov.slots_protected, 0u);

    const std::vector<std::byte> good_index{
            std::byte{0x00}, std::byte{0x01}, std::byte{0x02},
            std::byte{0x03}, std::byte{0x00}, std::byte{0x01},
            std::byte{0x02}, std::byte{0x03}};
    iom_conformance::copy_from_host(indices->view(), good_index);
    iom_conformance::copy_from_host(out->view(), std::vector<std::byte>(
            out->view().spec().logical_nbytes(),
            static_cast<std::byte>(0x00)));
    const iom::oid good_token = queue->embedding(
            table->view(), indices->view(), out->view(),
            workspace->view());
    REQUIRE(iom::oid_is_token(good_token));
    CHECK_NOTHROW(queue->wait(good_token));
}

// Regression for the F2 (round-5) race: set_failure must publish the
// retained post-launch fault under mu_, so a fence invoke racing the
// dispatch path cannot cache a success result over the just-written
// retained failure. The seam injects post_launch in the SAME path that
// exercises F1/F2 — the result() call reads retained_failure_ under
// mu_, and set_failure now writes under the same mu_, so the post-launch
// runtime_error MUST surface as the wait() throw and not be masked by
// a cached success. This is the same observable as round-3 but
// strengthened here to depend on the F2 ordering fix.
TEST_CASE(
        "SYCL conformance: F2 set_failure/result() race surfaces the "
        "runtime_error, not a cached success") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);

    const iom::TensorShape table_shape{{4, 16}};
    const iom::TensorShape index_shape{{1, 8}};
    const iom::TensorShape out_shape{{8, 16}};
    const iom::DataType payload = iom::DataType::U8;
    const iom::DataType index_type = iom::DataType::U8;

    auto table = devices.candidate->create_tensor(
            {table_shape, payload});
    auto indices = devices.candidate->create_tensor(
            {index_shape, index_type});
    auto out = devices.candidate->create_tensor(
            {out_shape, payload});
    auto workspace = devices.candidate->create_workspace(32);

    iom_conformance::copy_from_host(table->view(), std::vector<std::byte>(
            table->view().spec().logical_nbytes(),
            static_cast<std::byte>(0xA5)));
    iom_conformance::copy_from_host(indices->view(), std::vector<std::byte>(
            8, static_cast<std::byte>(0x00)));
    iom_conformance::copy_from_host(out->view(), std::vector<std::byte>(
            out->view().spec().logical_nbytes(),
            static_cast<std::byte>(0x00)));

    auto queue = devices.candidate->create_ops();
    REQUIRE(queue != nullptr);

    // First embedding — valid, completes normally. Establishes that the
    // workspace + queue path is healthy before the post-launch injection.
    const iom::oid baseline = queue->embedding(
            table->view(), indices->view(), out->view(),
            workspace->view());
    REQUIRE(iom::oid_is_token(baseline));
    CHECK_NOTHROW(queue->wait(baseline));

    // Second embedding — post_launch injected between set_event and the
    // dispatch-path catch block. Without F2, a fence invoke could see a
    // cached success and the post_launch runtime_error would not surface.
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::post_launch);
    const iom::oid fault_token = queue->embedding(
            table->view(), indices->view(), out->view(),
            workspace->view());
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::none);
    REQUIRE(iom::oid_is_token(fault_token));
    CHECK_THROWS_AS(queue->wait(fault_token), std::runtime_error);

    // After the faulted embedding, a follow-up valid embedding still
    // succeeds — the post-launch path MUST NOT leak protected slots.
    const iom::oid follow_up = queue->embedding(
            table->view(), indices->view(), out->view(),
            workspace->view());
    REQUIRE(iom::oid_is_token(follow_up));
    CHECK_NOTHROW(queue->wait(follow_up));

    iom::sycl_detail::QueueResourceSnapshot after;
    iom::sycl_detail::queue_resource_snapshot_for_testing(*queue, after);
    CHECK_EQ(after.slots_in_use, 0u);
    CHECK_EQ(after.slots_protected, 0u);
}

// Regression for the LOW (round-5) finding: a non-embedding native
// failure is a TERMINAL observation and MUST be cached, so repeated
// waits do not re-wait on a permanent native failure. Without this, a
// copy/binary/rmsnorm that fails its native event wait would re-block
// on every subsequent wait() call.
TEST_CASE(
        "SYCL conformance: a non-embedding native failure is cached so "
        "repeated waits do not re-wait the native event") {
    SyclDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    auto queue = devices.candidate->create_ops();
    REQUIRE(queue != nullptr);

    iom::sycl_detail::reset_fence_wait_count_for_testing();
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::post_launch);
    const iom::oid failed_token =
            queue->copy(source->view(), destination->view());
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::none);
    REQUIRE(iom::oid_is_token(failed_token));
    CHECK_THROWS_AS(queue->wait(failed_token), std::runtime_error);
    const std::size_t first_waits =
            iom::sycl_detail::fence_wait_count_for_testing();

    // Repeated wait MUST throw the same terminal exception from cache
    // without re-waiting the native event. (This is the
    // retained-fault cache behavior, which round-4 already had.)
    CHECK_THROWS_AS(queue->wait(failed_token), std::runtime_error);
    CHECK_EQ(iom::sycl_detail::fence_wait_count_for_testing(), first_waits);
    CHECK_THROWS_AS(queue->wait(failed_token), std::runtime_error);
    CHECK_EQ(iom::sycl_detail::fence_wait_count_for_testing(), first_waits);
}

TEST_CASE("SYCL conformance: RMSNorm reference, admission, and lifetime") {
    // The declared span is the device's own immutable `aspect::fp64` fact:
    // the port queues the eight non-`F64` leaves on every device and `F64`
    // exactly where the aspect is reported.
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    const sycl::device native_device =
            devices.candidate_context->get_devices().front();
    const bool fp64_available = native_device.has(sycl::aspect::fp64);
    const std::span<const iom::DataType> supported =
            fp64_available
                    ? std::span<const iom::DataType>(
                              iom_conformance::kRmsNormAllLeafSpan)
                    : std::span<const iom::DataType>(
                              iom_conformance::kRmsNormNonF64LeafSpan);
    CHECK_EQ(
            supported.size(), fp64_available ? std::size_t{9} : std::size_t{8});
    iom_conformance::RmsNormConformanceConfig config{
            devices.conformance(), supported, &devices.gate, &oracle};
    // The SYCL testing seam reaches the real RMSNorm path: `execute_rmsnorm`
    // (src/sycl/queue_rmsnorm.cpp) consumes `SubmissionFault::post_launch`
    // after `native_attempted = true` and `launch_rmsnorm`, and its handler
    // stores that failure on the already-accepted token
    // (`task.state->set_failure`), which is exactly the retained
    // accepted-failure identity the shared scenario asserts. Only the
    // post-acceptance point is armed: `first_submit` and `outcome_insertion`
    // are earlier points of the same executor and are deliberately left
    // disarmed so the failure cannot be consumed before acceptance.
    config.native_failure = iom_conformance::RmsNormNativeFailureSeam{
            [] {
                iom::sycl_detail::inject_submission_fault_for_testing(
                        iom::sycl_detail::SubmissionFault::post_launch);
            },
            [] {
                iom::sycl_detail::inject_submission_fault_for_testing(
                        iom::sycl_detail::SubmissionFault::none);
            },
            "sycl_detail::SubmissionFault::post_launch",
            {}};
    iom_conformance::run_rmsnorm_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}
// SYCL queues the eight non-F64 floating leaves through the native device
// path. F64 remains an explicit capability rejection even on devices that
// expose `aspect::fp64`, because this operation has no device-independent
// FP64 guarantee.
inline constexpr auto kSyclSiluImplementedLeaves =
        iom_conformance::kSiluSyclExpectedSupported;

TEST_CASE("SYCL conformance: SiLU reference, admission, and lifetime") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    iom_conformance::SiluConformanceConfig config{
            devices.conformance(),
            kSyclSiluImplementedLeaves,
            &devices.gate,
            &oracle,
            iom_conformance::SiluNativeFailureSeam{
                    [] {
                        iom::sycl_detail::inject_submission_fault_for_testing(
                                iom::sycl_detail::SubmissionFault::post_launch);
                    },
                    [] {
                        iom::sycl_detail::inject_submission_fault_for_testing(
                                iom::sycl_detail::SubmissionFault::none);
                    },
                    "sycl_detail::SubmissionFault::post_launch",
                    {}}};
    iom_conformance::run_silu_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}

// Driver-local SiLU coverage the frozen shared suite does not own: the exact
// rejected-leaf categories against the device's own fp64 fact, pure
// zero-workspace handling, positive device admission through temporary views
// and a destroyed producer owner, retained event/lease/owner proof, and an
// accepted post-launch fault whose repeated waits are identical and which
// still leaves the queue usable.
TEST_CASE(
        "SYCL conformance: SiLU device admission, lifetime, and accepted "
        "faults") {
    SyclDevices devices;
    const sycl::device native_device =
            devices.candidate_context->get_devices().front();
    const bool fp64_available = native_device.has(sycl::aspect::fp64);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::TensorSpec spec =
            iom_conformance::silu_make_spec({2, 17, 17}, iom::DataType::F32);

    // F64 stays a capability rejection even where the device reports fp64.
    {
        CAPTURE(fp64_available);
        const iom::TensorSpec f64_spec = iom_conformance::silu_make_spec(
                {2, 17, 17}, iom::DataType::F64);
        auto input = devices.candidate->create_tensor(f64_spec);
        auto output = devices.candidate->create_tensor(f64_spec);
        const std::vector<std::byte> before =
                iom_conformance::read_logical(output->view());
        auto queue = devices.candidate->create_ops();
        CHECK_EQ(queue->silu(input->view(), output->view()), unsupported);
        CHECK_THROWS_AS(
                (void)queue->silu_workspace_requirements(
                        input->view(), output->view()),
                std::runtime_error);
        CHECK(iom_conformance::read_logical(output->view()) == before);
    }

    // Every semantically inapplicable leaf is a capability rejection on one
    // queue, none of them mutates the output, and none consumes a submission
    // sequence. The six packed additions are exercised by the shared positive
    // suite above rather than duplicated in this driver-local rejection probe.
    {
        std::vector<iom::DataType> rejected(
                iom_conformance::kSiluInapplicableDataTypes.begin(),
                iom_conformance::kSiluInapplicableDataTypes.end());
        auto queue = devices.candidate->create_ops();
        for (const iom::DataType leaf : rejected) {
            CAPTURE(static_cast<int>(leaf));
            const iom::TensorSpec leaf_spec =
                    iom_conformance::silu_make_spec({2, 17, 17}, leaf);
            auto input = devices.candidate->create_tensor(leaf_spec);
            auto output = devices.candidate->create_tensor(leaf_spec);
            const std::vector<std::byte> before =
                    iom_conformance::read_logical(output->view());
            CHECK_EQ(
                    queue->silu(input->view(), output->view()), unsupported);
            CHECK_THROWS_AS(
                    (void)queue->silu_workspace_requirements(
                            input->view(), output->view()),
                    std::runtime_error);
            CHECK(iom_conformance::read_logical(output->view()) == before);
        }

        // An unknown DataType enum is invalid input, not a capability result.
        auto unknown_input = devices.candidate->create_tensor(spec);
        auto unknown_output = devices.candidate->create_tensor(spec);
        iom::TensorSpec unknown = unknown_input->view().spec();
        unknown.data_type = static_cast<iom::DataType>(255);
        iom_conformance::SiluOwnerSpecRestore restore_input(
                *unknown_input, unknown);
        iom_conformance::SiluOwnerSpecRestore restore_output(
                *unknown_output, unknown);
        CHECK_EQ(
                queue->silu(unknown_input->view(), unknown_output->view()),
                invalid);
        CHECK_THROWS_AS(
                (void)queue->silu_workspace_requirements(
                        unknown_input->view(), unknown_output->view()),
                std::invalid_argument);

        // A recognized non-`NONE` quantization is a capability rejection.
        auto quantized_input = devices.candidate->create_tensor(spec);
        auto quantized_output = devices.candidate->create_tensor(spec);
        iom::TensorSpec grouped = quantized_input->view().spec();
        grouped.quantization = iom::QuantizationFormat::OCP_MXFP4;
        iom_conformance::SiluOwnerSpecRestore restore_quantized_input(
                *quantized_input, grouped);
        iom_conformance::SiluOwnerSpecRestore restore_quantized_output(
                *quantized_output, grouped);
        CHECK_EQ(
                queue->silu(
                        quantized_input->view(), quantized_output->view()),
                unsupported);

        // Every rejected probe stayed side-effect free: the first queued
        // submission on this queue is sequence one, and an empty workspace
        // plus the ignoring nonempty workspace both reach the device.
        auto accepted_input = devices.candidate->create_tensor(spec);
        auto accepted_output = devices.candidate->create_tensor(spec);
        std::array<std::byte, 64> scratch{};
        iom_conformance::SiluTestWorkspace nonempty_workspace(
                *devices.candidate, scratch.data(), scratch.size());
        CHECK(queue->silu_workspace_requirements(
                      accepted_input->view(), accepted_output->view())
              == iom::WorkspaceRequirements{0, 1});
        const iom::oid first = queue->silu(
                accepted_input->view(), accepted_output->view(),
                nonempty_workspace.view());
        REQUIRE(iom::oid_is_token(first));
        CHECK_EQ(iom_conformance::token_sequence(first), std::uint64_t{1});
        CHECK_NOTHROW(queue->wait(first));
        CHECK_NOTHROW(queue->wait(first));
    }

    // Positive BF16/F32 device admission: the immutable snapshot and both
    // owner identities outlive temporary views and a destroyed producer, the
    // completion event is waited exactly once across repeated waits, and the
    // released fixed slots prove proven completion under the retained lease.
    for (const iom::DataType type : {iom::DataType::BF16, iom::DataType::F32}) {
        CAPTURE(static_cast<int>(type));
        const iom_conformance::SiluReferenceCase fixture =
                iom_conformance::silu_make_pattern_case(
                        iom_conformance::SiluReferenceCaseKind::ordinary, type,
                        {2}, 2, 17,
                        "driver/ordinary/"
                                + iom_conformance::silu_oracle::leaf_name(
                                        type));
        const iom::TensorSpec case_spec =
                iom_conformance::silu_case_spec(fixture);
        auto input = devices.candidate->create_tensor(case_spec);
        auto output = devices.candidate->create_tensor(case_spec);
        iom_conformance::copy_from_host(
                input->view(),
                iom_conformance::silu_pack_bits(type, fixture.input_bits));
        iom_conformance::copy_from_host(
                output->view(),
                std::vector<std::byte>(
                        case_spec.logical_nbytes(),
                        iom_conformance::kReadbackSentinel));
        const std::vector<iom_conformance::SiluReferenceValue> expected =
                iom_conformance::silu_evaluate(fixture);
        auto queue = devices.candidate->create_ops();
        iom::sycl_detail::reset_fence_wait_count_for_testing();

        iom::oid token = 0;
        {
            iom_conformance::SiluCaseWindow window(&devices.gate);
            iom::TensorView temporary = output->view();
            token = queue->silu(input->view(), temporary);
        }
        REQUIRE(iom::oid_is_token(token));
        CHECK_EQ(iom_conformance::token_sequence(token), std::uint64_t{1});
        // Owner retention: the accepted work holds its own metadata, owner
        // identity, and native handles, so the producer owner may be destroyed
        // while the access is still retained and unproven.
        input.reset();
        CHECK_NOTHROW(queue->wait(token));
        CHECK_NOTHROW(queue->wait(token));
        CHECK_EQ(
                iom::sycl_detail::fence_wait_count_for_testing(),
                std::size_t{1});
        const std::string mismatch = iom_conformance::silu_compare(
                type, iom_conformance::read_logical(output->view()), expected,
                "driver device SiLU");
        CHECK_MESSAGE(mismatch.empty(), mismatch);
        iom::sycl_detail::QueueResourceSnapshot after;
        iom::sycl_detail::queue_resource_snapshot_for_testing(*queue, after);
        CHECK_EQ(after.slots_in_use, 0u);
        CHECK_EQ(after.slots_protected, 0u);
    }

    // Accepted post-launch device fault: the submission keeps its positive
    // OID, the output is unusable, and every repeated wait observes the
    // identical failure without re-waiting the native event. A later valid
    // submission on the same queue still succeeds and matches the reference.
    {
        const iom::DataType type = iom::DataType::F32;
        const iom_conformance::SiluReferenceCase fixture =
                iom_conformance::silu_make_pattern_case(
                        iom_conformance::SiluReferenceCaseKind::ordinary, type,
                        {2}, 2, 17, "driver/native-failure/F32");
        const iom::TensorSpec case_spec =
                iom_conformance::silu_case_spec(fixture);
        auto input = devices.candidate->create_tensor(case_spec);
        auto output = devices.candidate->create_tensor(case_spec);
        iom_conformance::copy_from_host(
                input->view(),
                iom_conformance::silu_pack_bits(type, fixture.input_bits));
        iom_conformance::copy_from_host(
                output->view(),
                std::vector<std::byte>(
                        case_spec.logical_nbytes(),
                        iom_conformance::kReadbackSentinel));
        const std::vector<iom_conformance::SiluReferenceValue> expected =
                iom_conformance::silu_evaluate(fixture);
        auto queue = devices.candidate->create_ops();

        const iom::oid healthy = queue->silu(
                input->view(), output->view());
        REQUIRE(iom::oid_is_token(healthy));
        CHECK_NOTHROW(queue->wait(healthy));

        iom::sycl_detail::inject_submission_fault_for_testing(
                iom::sycl_detail::SubmissionFault::post_launch);
        const iom::oid failed = queue->silu(
                input->view(), output->view());
        iom::sycl_detail::inject_submission_fault_for_testing(
                iom::sycl_detail::SubmissionFault::none);
        REQUIRE(iom::oid_is_token(failed));
        CHECK_EQ(
                iom_conformance::token_sequence(failed),
                iom_conformance::token_sequence(healthy) + 1);
        const std::size_t native_waits =
                iom::sycl_detail::fence_wait_count_for_testing();
        iom_conformance::expect_repeated_runtime_failure(*queue, failed);
        CHECK_EQ(
                iom::sycl_detail::fence_wait_count_for_testing(),
                native_waits + 1);

        const iom::oid recovered = queue->silu(
                input->view(), output->view());
        REQUIRE(iom::oid_is_token(recovered));
        CHECK_EQ(
                iom_conformance::token_sequence(recovered),
                iom_conformance::token_sequence(failed) + 1);
        CHECK_NOTHROW(queue->wait(recovered));
        CHECK_NOTHROW(queue->wait(recovered));
        const std::string mismatch = iom_conformance::silu_compare(
                type, iom_conformance::read_logical(output->view()), expected,
                "driver recovered SiLU");
        CHECK_MESSAGE(mismatch.empty(), mismatch);
    }
}

// The device fact behind SYCL's native BF16 linear specialization, queried here
// independently of the port and of its capability predicate: the pinned
// extension revision, the Intel matrix aspect, subgroup 16, and a
// BF16/BF16/FP32 combination covering both M shapes the row decomposition
// queues. A device that does not expose the fact keeps `BF16` a capability
// rejection rather than a conformance claim, exactly like the `F64` aspect
// gate.
[[nodiscard]] bool sycl_bf16_matrix_available(const sycl::device& device) {
#if defined(IOM_SYCL_TEST_BF16_MATRIX)
    if (!device.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }
    const std::vector<std::size_t> subgroup_sizes =
            device.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(
                subgroup_sizes.begin(), subgroup_sizes.end(), std::size_t{16})
            == subgroup_sizes.end()) {
        return false;
    }
    namespace matrix = sycl::ext::oneapi::experimental::matrix;
    const std::vector<matrix::combination> combinations = device.get_info<
            sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    const auto covers = [&combinations](std::size_t rows) {
        for (const matrix::combination& combination : combinations) {
            if (combination.atype != matrix::matrix_type::bf16
                    || combination.btype != matrix::matrix_type::bf16
                    || combination.ctype != matrix::matrix_type::fp32
                    || combination.dtype != matrix::matrix_type::fp32) {
                continue;
            }
            const bool rows_covered = combination.msize == 0
                    ? rows <= combination.max_msize
                    : combination.msize == rows;
            const bool columns_covered = combination.nsize == 0
                    ? std::size_t{16} <= combination.max_nsize
                    : combination.nsize == std::size_t{16};
            const bool inner_covered = combination.ksize == 0
                    ? std::size_t{16} <= combination.max_ksize
                    : combination.ksize == std::size_t{16};
            if (rows_covered && columns_covered && inner_covered) {
                return true;
            }
        }
        return false;
    };
    return covers(16) && covers(1);
#else
    (void)device;
    return false;
#endif  // defined(IOM_SYCL_TEST_BF16_MATRIX)
}

TEST_CASE("SYCL conformance: full shared suite") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    const iom_conformance::RopeContractConformanceConfig rope_contract{
            devices.conformance(),
            iom_conformance::rope_reference::kRopeSyclExpectedSupported,
            &devices.gate,
            iom_conformance::RopeNativeFailureSeam{
                    [] {
                        iom::sycl_detail::inject_submission_fault_for_testing(
                                iom::sycl_detail::SubmissionFault::post_launch);
                    },
                    [] {
                        iom::sycl_detail::inject_submission_fault_for_testing(
                                iom::sycl_detail::SubmissionFault::none);
                    },
                    "sycl_detail::SubmissionFault::post_launch",
                    {}}};
    const bool sdpa_available = sycl_bf16_matrix_available(
            devices.candidate_context->get_devices().front());
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, &oracle, true, true, &rope_contract,
            sdpa_available);
    CHECK_FALSE(devices.gate.armed());
}

// ---------------------------------------------------------------------------
// SYCL native BF16 linear projection.
// ---------------------------------------------------------------------------

// SYCL's declared linear expectation: the complete 21-leaf applicable matrix
// through the twenty-leaf scalar path plus the native BF16 specialization, the
// frozen `{0, 1}` scratch path for the scalar leaves, the documented aligned
// `A32(P*pad16(R)*pad16(O)*4)` BF16 path, and the two genuine runtime device
// facts: the `sycl::aspect::fp64` gate for `F64` and the queried subgroup-16
// BF16/BF16/FP32 `joint_matrix` facility the native BF16 specialization
// requires. Both gates are asserted in both directions regardless, so a port
// that queues a gated leaf on a device that does not expose the fact fails
// here instead of silently claiming support.
TEST_CASE("SYCL conformance: linear projection reference, admission, and lifetime") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    const sycl::device native_device =
            devices.candidate_context->get_devices().front();
    const bool fp64_available = native_device.has(sycl::aspect::fp64);
    const bool bf16_matrix_available =
            sycl_bf16_matrix_available(native_device);
    iom_conformance::LinearDeclaration declaration{
            iom_conformance::kLinearLeafSpan,
            iom_conformance::kLinearScalarLeafSpan,
            iom_conformance::kLinearNativeBf16Span,
            iom_conformance::kLinearLeafSpan,
            {},
            iom_conformance::LinearWorkspacePath::zero,
            iom_conformance::LinearWorkspacePath::sycl_bf16};
    declaration.device_available =
            [fp64_available, bf16_matrix_available](iom::DataType leaf) {
                if (leaf == iom::DataType::F64) {
                    return fp64_available;
                }
                if (leaf == iom::DataType::BF16) {
                    return bf16_matrix_available;
                }
                return true;
            };
    iom_conformance::run_linear_conformance(
            devices.conformance(), declaration, &devices.gate, &oracle);
    CHECK_FALSE(devices.gate.armed());
    // The gate is the device's own immutable fact, not a port statement.
    CHECK_EQ(declaration.device_available(iom::DataType::F64), fp64_available);
    CHECK_EQ(
            declaration.device_available(iom::DataType::BF16),
            bf16_matrix_available);
    CHECK(declaration.device_available(iom::DataType::F32));
}

TEST_CASE("SYCL conformance: SDPA capability follows the queried matrix facility") {
    // The native SDPA stages queue exactly the subgroup-16 BF16/BF16/FP32
    // `joint_matrix` family the native `BF16` linear specialization proves, so
    // the queue's immutable capability must agree with the device's own facts
    // in both directions: the exact four-segment alignment-32 requirement on a
    // capable device, and a pure capability rejection - no allocation,
    // registration, lease, sequence, or output mutation - otherwise.
    SyclDevices devices;
    const sycl::device native_device =
            devices.candidate_context->get_devices().front();
    const bool matrix_available = sycl_bf16_matrix_available(native_device);
    const auto queue = devices.candidate->create_ops();

    auto q = devices.candidate->create_tensor(
            iom::TensorSpec{iom::TensorShape{{1, 1, 1, 16}},
                            iom::DataType::BF16});
    auto kv = devices.candidate->create_tensor(
            iom::TensorSpec{iom::TensorShape{{1, 1, 1, 16}},
                            iom::DataType::BF16});
    auto out = devices.candidate->create_tensor(
            iom::TensorSpec{iom::TensorShape{{1, 1, 16}},
                            iom::DataType::BF16});
    const std::vector<std::byte> sentinel =
            iom_conformance::read_logical(out->view());
    if (matrix_available) {
        // One plane of `R=1`, `L=1`: padded scores 16x16 FP32 (1024 B),
        // probabilities 16x16 BF16 (512 B), PV 16x16 FP32 (1024 B), and head
        // staging 16x16 BF16 (512 B), every segment 32-byte aligned.
        const iom::WorkspaceRequirements requirements =
                queue->sdpa_workspace_requirements(
                        q->view(), kv->view(), kv->view(), out->view(), 0, 1);
        CHECK_EQ(requirements, (iom::WorkspaceRequirements{3072, 32}));
        const std::unique_ptr<iom::RawWorkspace> workspace =
                devices.candidate->create_workspace(requirements.bytes);
        REQUIRE(workspace != nullptr);
        const iom::oid token = queue->sdpa(
                q->view(), kv->view(), kv->view(), out->view(), 0, 1,
                workspace->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
    } else {
        CHECK_THROWS(
                (void)queue->sdpa_workspace_requirements(
                        q->view(), kv->view(), kv->view(), out->view(), 0, 1));
        CHECK_EQ(
                queue->sdpa(
                        q->view(), kv->view(), kv->view(), out->view(), 0, 1),
                iom::to_oid(iom::OidError::Unsupported));
        CHECK_EQ(iom_conformance::read_logical(out->view()), sentinel);
    }
    CHECK_FALSE(devices.gate.armed());
}
TEST_CASE("SYCL conformance: SDPA reference, admission, and lifetime") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    const bool matrix_available = sycl_bf16_matrix_available(
            devices.candidate_context->get_devices().front());
    // SYCL's declared SDPA expectation: the current BF16 leaf through the
    // shared independent oracle, the exact checked four-segment alignment-32
    // workspace, four distinct tensor owners plus the caller workspace lease,
    // and the accepted post-enqueue failure path. The positive matrix is
    // claimed only on a device that exposes the queried native matrix facility;
    // a device without it keeps BF16 a pure capability rejection, exactly like
    // the linear `BF16` gate.
    const iom_conformance::SdpaConformanceConfig config{
            devices.conformance(),
            matrix_available
                    ? std::span<const iom::DataType>(
                              iom_conformance::kSdpaCurrentSupportedDataTypes)
                    : std::span<const iom::DataType>{},
            matrix_available,
            &devices.gate,
            &oracle,
            {},
            {[] {
                 iom::sycl_detail::inject_submission_fault_for_testing(
                         iom::sycl_detail::SubmissionFault::
                                 sdpa_post_acceptance_failure);
             },
             [] {
                 iom::sycl_detail::inject_submission_fault_for_testing(
                         iom::sycl_detail::SubmissionFault::none);
             },
             "sycl_detail::SubmissionFault::sdpa_post_acceptance_failure",
             {}}};
    iom_conformance::run_sdpa_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}

// Native attribution of the two shapes the operation contract requires: one
// logical `R=1` decode row with a nonzero causal offset and one prefill request
// with the valid generic `L < a + R` case, a non-tile head dimension, and
// distinct grouped query heads. Both run through the public facade against the
// shared independent oracle, and the printed labels let a `SYCL_PI_TRACE=2`
// run attribute the direct QK and PV kernel launches to each shape.
TEST_CASE("SYCL conformance: SDPA native QK and PV stages for decode and prefill") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    const sycl::device native_device =
            devices.candidate_context->get_devices().front();
    REQUIRE_MESSAGE(
            sycl_bf16_matrix_available(native_device),
            "the native SDPA stages require the queried subgroup-16 "
            "BF16/BF16/FP32 joint_matrix facility");
    const iom_conformance::SdpaConformanceConfig config{
            devices.conformance(),
            iom_conformance::kSdpaCurrentSupportedDataTypes,
            true,
            &devices.gate,
            &oracle,
            {},
            {}};

    const iom_conformance::SdpaReferenceCase decode =
            iom_conformance::sdpa_incremental_slice(
                    iom_conformance::sdpa_oracle::make_cached_incremental_case(
                            iom::DataType::BF16),
                    3);
    std::printf(
            "sdpa-native-stage shape=decode rows=%zu position=%zu length=%zu "
            "capacity=%zu head_dim=%zu\n",
            decode.rows, decode.position, decode.length, decode.capacity,
            decode.head_dim);
    (void)iom_conformance::sdpa_detail::run_reference_case(
            config, decode, "native decode R=1");

    const iom_conformance::SdpaReferenceCase prefill =
            iom_conformance::sdpa_oracle::make_boundary_rows_case(
                    iom::DataType::BF16, 16);
    std::printf(
            "sdpa-native-stage shape=prefill rows=%zu position=%zu length=%zu "
            "capacity=%zu head_dim=%zu\n",
            prefill.rows, prefill.position, prefill.length, prefill.capacity,
            prefill.head_dim);
    (void)iom_conformance::sdpa_detail::run_reference_case(
            config, prefill, "native prefill R=16");
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: RMS normalization capability follows the device FP64 aspect") {
    // The `docs/BACKEND_CONTRACT.md` **RMS normalization** capability matrix
    // makes the SYCL port queue the eight non-`F64` applicable leaves and
    // queue `F64` exactly when the selected device reports
    // `sycl::aspect::fp64`. The pure query and the call share that one
    // immutable capability, so they must agree in either direction.
    SyclDevices devices;
    const auto queue = devices.candidate->create_ops();
    const iom::WorkspaceRequirements zero{0, 1};
    for (const iom::DataType data_type : {
                 iom::DataType::F4_E2M1,
                 iom::DataType::F6_E2M3,
                 iom::DataType::F6_E3M2,
                 iom::DataType::F8_E4M3FN,
                 iom::DataType::F8_E5M2,
                 iom::DataType::F16,
                 iom::DataType::BF16,
                 iom::DataType::F32}) {
        const iom::TensorSpec spec{iom::TensorShape{{1, 16}}, data_type};
        auto x = devices.candidate->create_tensor(spec);
        auto scale = devices.candidate->create_tensor(spec);
        auto out = devices.candidate->create_tensor(spec);
        CHECK(queue->rmsnorm_workspace_requirements(
                      x->view(), scale->view(), out->view(), 1e-6F)
              == zero);
    }

    const iom::TensorSpec double_spec{
            iom::TensorShape{{1, 16}}, iom::DataType::F64};
    auto x = devices.candidate->create_tensor(double_spec);
    auto scale = devices.candidate->create_tensor(double_spec);
    auto out = devices.candidate->create_tensor(double_spec);
    const sycl::device native_device =
            devices.candidate_context->get_devices().front();
    if (native_device.has(sycl::aspect::fp64)) {
        CHECK(queue->rmsnorm_workspace_requirements(
                      x->view(), scale->view(), out->view(), 1e-6F)
              == zero);
        const iom::oid token =
                queue->rmsnorm(x->view(), scale->view(), out->view(), 1e-6F);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
    } else {
        CHECK_THROWS_AS(
                (void)queue->rmsnorm_workspace_requirements(
                        x->view(), scale->view(), out->view(), 1e-6F),
                std::runtime_error);
        CHECK_EQ(
                queue->rmsnorm(
                        x->view(), scale->view(), out->view(), 1e-6F),
                iom::to_oid(iom::OidError::Unsupported));
    }
}
TEST_CASE("SYCL conformance: native RoPE reference and eight-leaf capability") {
    SyclDevices devices;
    SyclStorageOracle oracle(*devices.candidate_context);
    iom_conformance::rope_reference::run_rope_conformance(
            *devices.candidate,
            iom_conformance::rope_reference::kRopeSyclExpectedSupported,
            &oracle);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: RoPE rejects unsupported leaves and workspace") {
    SyclDevices devices;
    auto queue = devices.candidate->create_ops();
    const iom::TensorSpec f32_spec{
            iom::TensorShape{{1, 1, 2}}, iom::DataType::F32};
    auto f32_x = devices.candidate->create_tensor(f32_spec);
    auto f32_out = devices.candidate->create_tensor(f32_spec);
    auto workspace = devices.candidate->create_workspace(32);
    CHECK_EQ(
            queue->rope(
                    f32_x->view(), f32_out->view(), 0, 10000.0,
                    workspace->view()),
            iom::to_oid(iom::OidError::InvalidArgument));

    for (const iom::DataType data_type :
         iom_conformance::rope_reference::kRopeUnsupportedDataTypes) {
        const iom::TensorSpec spec{
                iom::TensorShape{{1, 1, 2}}, data_type};
        auto x = devices.candidate->create_tensor(spec);
        auto out = devices.candidate->create_tensor(spec);
        CHECK_THROWS_AS(
                (void)queue->rope_workspace_requirements(
                        x->view(), out->view(), 0, 10000.0),
                std::runtime_error);
        CHECK_EQ(
                queue->rope(x->view(), out->view(), 0, 10000.0),
                iom::to_oid(iom::OidError::Unsupported));
    }
    const iom::TensorSpec f64_spec{
            iom::TensorShape{{1, 1, 2}}, iom::DataType::F64};
    auto f64_x = devices.candidate->create_tensor(f64_spec);
    auto f64_out = devices.candidate->create_tensor(f64_spec);
    CHECK_THROWS_AS(
            (void)queue->rope_workspace_requirements(
                    f64_x->view(), f64_out->view(), 0, 10000.0),
            std::runtime_error);
    CHECK_EQ(
            queue->rope(f64_x->view(), f64_out->view(), 0, 10000.0),
            iom::to_oid(iom::OidError::Unsupported));
}

TEST_CASE("SYCL conformance: accepted RoPE failure repeats through the fence") {
    SyclDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{1, 1, 2}}, iom::DataType::F32};
    auto x = devices.candidate->create_tensor(spec);
    auto out = devices.candidate->create_tensor(spec);
    auto queue = devices.candidate->create_ops();
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::post_launch);
    const iom::oid token = queue->rope(x->view(), out->view(), 1, 10000.0);
    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::none);
    REQUIRE(iom::oid_is_token(token));
    CHECK_THROWS_AS(queue->wait(token), std::runtime_error);
    CHECK_THROWS_AS(queue->wait(token), std::runtime_error);
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

// The SYCL instantiation of the shared, backend-neutral model-loading
// scenario, and the sole integration point of this leaf. The existing
// SyclDevices fixture already provides the three roles it needs: the CPU
// reference device, the selected eligible SYCL device as candidate, and a
// second distinct SYCL device instance at the same ordinal as the foreign
// device, whose owned context differs from the candidate's, so the
// foreign-destination rejection is judged against the exact candidate
// instance rather than the ordinal. The candidate binding reports a positive
// word-padded staging requirement, so the shared case queries that
// requirement from the real destination views, provisions caller scratch from
// this exact device, and validates the empty default before the first copied
// role; the CPU reference realizes the same binding through its zero-byte
// `{0, 1}` path. Every destination is created from the source's own selected
// specification, readback is a real `TensorView` transfer, and BF16 and real
// Level Zero hardware are required: an ineligible device fails the fixture,
// it is never skipped.
TEST_CASE("SYCL model loading realizes every published weight role of each synthetic checkpoint") {
    SyclDevices devices;
    iom_conformance::run_model_loading_conformance(devices.conformance());
}

// Opt-in real-checkpoint loading, compiled only with
// `IOM_TEST_REAL_MODEL_LOADING=ON`. This case reserves the caller-selected real
// capacity on the eligible SYCL device instead of the synthetic conformance
// arena, which cannot hold the full official reference, so a missing, invalid,
// or insufficient `IOM_TEST_MODEL_ARENA_BYTES` fails it and an ineligible
// device fails the factory rather than skipping. No context sink or oracle is
// needed: the case observes the produced devices through public APIs only.
#ifdef IOM_TEST_REAL_MODEL_LOADING
TEST_CASE("SYCL real model loading") {
    const std::unique_ptr<iom::Device> candidate = iom::make_sycl_device(
            0, iom_conformance::real_model_memory_config());
    iom_conformance::run_real_model_loading(*candidate);
}
#endif
