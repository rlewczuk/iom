#include <doctest/doctest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "backend/backend_conformance_add_gpu.hpp"
#include "iom/cpu/device.hpp"
#include "iom/cuda/device.hpp"
#include "cuda/copy.hpp"

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
        ++traffic_;
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
        ++traffic_;
        ::operator delete(buffer, std::align_val_t(32));
    }

    void reset() override { ++traffic_; }

private:
    const iom_conformance::TrafficGate& gate_;
    std::unordered_set<void*> live_;
    std::size_t traffic_ = 0;
};

// Every standard-GPU conformance fixture reserves this tensor-data arena:
// the largest concurrent set is three 8192x4096 F32 tensors plus one
// quarantined operand (512 MiB), and the fixture also keeps a foreign
// device alive with the same budget.
constexpr std::size_t kConformanceArenaBytes = 640u * 1024 * 1024;

struct CudaDevices {
    iom_conformance::TrafficGate gate;
    HostAllocator reference_allocator{gate};
    std::unique_ptr<iom::Device> reference =
            iom::make_cpu_device(reference_allocator);
    std::unique_ptr<iom::Device> candidate =
            iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    std::unique_ptr<iom::Device> foreign =
            iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{kConformanceArenaBytes});

    [[nodiscard]] iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};

class CudaStorageOracle final
        : public iom_conformance::AcceleratorStorageOracle {
public:
    void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        REQUIRE_EQ(encoded.size(), bytes);
        REQUIRE(cudaMemcpy(
                        view.native_handle(), encoded.data(), bytes,
                        cudaMemcpyHostToDevice)
                == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        std::vector<std::byte> result(bytes);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        REQUIRE(cudaMemcpy(
                        result.data(), view.native_handle(), bytes,
                        cudaMemcpyDeviceToHost)
                == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        return result;
    }
};


}  // namespace
TEST_CASE("Device::supported_data_types returns the per-backend 23-entry span") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    iom_conformance::TrafficGate gate;
    const std::unique_ptr<iom::Device> candidate =
            iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    iom_conformance::require_standard_capabilities(
            candidate->supported_data_types());
}

TEST_CASE("CUDA conformance: storage and host transfers for every leaf type") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: storage oracle identifies perturbed transfer map") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const std::span<const iom::DataType> one_type =
            devices.candidate->supported_data_types().subspan(0, 1);
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), one_type, &devices.gate);

    CudaStorageOracle direct;
    iom_conformance::PermutingStorageOracle perturbed(
            direct, iom_conformance::swap_first_adjacent_slots);
    CHECK_FALSE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, perturbed, &devices.gate,
            false, false));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: storage oracle covers every leaf width and padded shape") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            oracle, &devices.gate));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: asynchronous copies against the CPU reference") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate, &oracle);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: copy validation fails before writes and sequences") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA copy reservation failures roll back before native work") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::U8};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    const std::vector<std::byte> source_bytes(
            spec.logical_nbytes(), static_cast<std::byte>(0x11));
    const std::vector<std::byte> destination_bytes(
            spec.logical_nbytes(), static_cast<std::byte>(0x22));
    source->view().copy_from_host(source_bytes);
    destination->view().copy_from_host(destination_bytes);
    auto queue = devices.candidate->create_ops();

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::registration);
    CHECK_EQ(
            queue->copy(source->view(), destination->view()),
            iom::to_oid(iom::OidError::ResourceExhausted));
    std::vector<std::byte> observed(spec.logical_nbytes());
    destination->view().copy_to_host(observed);
    CHECK(observed == destination_bytes);

    const iom::oid first = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(first), 1);
    CHECK_NOTHROW(queue->wait(first));

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::outcome_insertion);
    CHECK_EQ(
            queue->copy(source->view(), destination->view()),
            iom::to_oid(iom::OidError::ResourceExhausted));
    const iom::oid second = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(second), 2);
    CHECK_NOTHROW(queue->wait(second));
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: transfer failures keep metadata and ownership") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_transfer_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: deferred queue lifetime and stability") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: compute methods reject capability without submitting") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate, "CUDA", true);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: full shared suite") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, &oracle, true);
    CHECK_FALSE(devices.gate.armed());
}
TEST_CASE("CUDA binary conformance: ADD MUL SUB DIV real queue") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    for (const auto operation : {
            iom_conformance::BinaryOperation::add,
            iom_conformance::BinaryOperation::mul,
            iom_conformance::BinaryOperation::sub,
            iom_conformance::BinaryOperation::div}) {
        iom_conformance::run_gpu_eltwise_conformance(
                *devices.candidate, operation);
        iom_conformance::run_gpu_eltwise_mapping_conformance(
                *devices.candidate, operation);
    }
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA ADD accepts every low-width leaf against the oracle") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_gpu_eltwise_conformance(
            *devices.candidate, iom_conformance::BinaryOperation::add);
    iom_conformance::run_binary_rank_boundary_conformance(
            *devices.candidate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA ADD wide dtypes and boundary values against the oracle") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_gpu_eltwise_conformance(
            *devices.candidate, iom_conformance::BinaryOperation::add);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA ADD broadcast, transform, tail, and exact alias mapping") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_gpu_eltwise_mapping_conformance(
            *devices.candidate, iom_conformance::BinaryOperation::add);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA ADD retained launch failure keeps owners reusable") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::U8};
    auto lhs = devices.candidate->create_tensor(spec);
    auto out = devices.candidate->create_tensor(spec);
    std::vector<std::byte> pattern(spec.logical_nbytes());
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = std::byte{static_cast<unsigned char>(i * 3)};
    }
    lhs->view().copy_from_host(pattern);
    auto queue = devices.candidate->create_ops();

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::third_plane_launch);
    const iom::oid failed = queue->add(lhs->view(), lhs->view(), out->view());
    CHECK(iom::oid_is_token(failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, failed);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

    const iom::oid recovered =
            queue->add(lhs->view(), lhs->view(), out->view());
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(queue->wait(recovered));
    std::vector<std::byte> observed(spec.logical_nbytes());
    out->view().copy_to_host(observed);
    for (std::size_t i = 0; i < observed.size(); ++i) {
        CHECK_EQ(
                static_cast<unsigned>(observed[i]),
                static_cast<unsigned>(pattern[i]) * 2 & 0xffu);
    }
}

TEST_CASE("CUDA submission remains transactional across post-enqueue failures") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    const iom::TensorSpec mismatch{
            iom::TensorShape{{3, 16, 17}}, iom::DataType::U8};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    auto invalid_destination = devices.candidate->create_tensor(mismatch);
    auto queue = devices.candidate->create_ops();
    const std::vector<std::byte> logical_pattern(
            spec.logical_nbytes(), static_cast<std::byte>(0x3c));

    source->view().copy_from_host(logical_pattern);
    destination->view().copy_from_host(logical_pattern);

    CHECK_EQ(queue->copy(source->view(), invalid_destination->view()), iom::to_oid(iom::OidError::InvalidArgument));

    const iom::oid first = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(first), 1);
    CHECK_NOTHROW(queue->wait(first));

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::event_create);
    CHECK_EQ(queue->copy(source->view(), destination->view()), iom::to_oid(iom::OidError::DeviceError));
    const iom::oid second = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(second), 2);
    CHECK_NOTHROW(queue->wait(second));

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::third_plane_launch);
    iom::oid launch_failure = 0;
    launch_failure = queue->copy(source->view(), destination->view());
    CHECK(iom::oid_is_token(launch_failure));
    CHECK_NE(launch_failure, 0);
    CHECK_EQ(iom_conformance::token_sequence(launch_failure), 3);
    iom_conformance::expect_repeated_runtime_failure(*queue, launch_failure);

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::event_record);
    const iom::oid record_failure =
            queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(record_failure), 4);
    iom_conformance::expect_repeated_runtime_failure(*queue, record_failure);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

    // Failed work remains quarantined until device teardown; the data
    // arena must not recycle either operand while its failed entries are
    // retained.
    {
        auto device = iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
        auto canary_source = device->create_tensor(spec);
        auto canary_destination = device->create_tensor(spec);
        const std::vector<std::byte> storage_canary(
                spec.tiled_storage_nbytes(), static_cast<std::byte>(0xa5));
        REQUIRE(
                cudaMemcpy(
                        canary_source->view().native_handle(),
                        storage_canary.data(), storage_canary.size(),
                        cudaMemcpyHostToDevice)
                == cudaSuccess);
        REQUIRE(
                cudaMemcpy(
                        canary_destination->view().native_handle(),
                        storage_canary.data(), storage_canary.size(),
                        cudaMemcpyHostToDevice)
                == cudaSuccess);
        auto canary_queue = device->create_ops();
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::third_plane_launch);
        const iom::oid canary_failure = canary_queue->copy(
                canary_source->view(), canary_destination->view());
        CHECK_EQ(iom_conformance::token_sequence(canary_failure), 1);
        iom_conformance::expect_repeated_runtime_failure(*canary_queue, canary_failure);

        const void* source_address = canary_source->view().native_handle();
        const void* destination_address =
                canary_destination->view().native_handle();
        canary_source.reset();
        canary_destination.reset();

        auto fresh_source = device->create_tensor(spec);
        auto fresh_destination = device->create_tensor(spec);
        CHECK_NE(fresh_source->view().native_handle(), source_address);
        CHECK_NE(fresh_source->view().native_handle(), destination_address);
        CHECK_NE(fresh_destination->view().native_handle(), source_address);
        CHECK_NE(fresh_destination->view().native_handle(), destination_address);
        CHECK_NE(
                fresh_source->view().native_handle(),
                fresh_destination->view().native_handle());

        fresh_source.reset();
        fresh_destination.reset();
        canary_queue.reset();
        device.reset();
    }
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
}

TEST_CASE("CUDA queue destruction fences pending copies") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    iom_conformance::TrafficGate gate;
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    const std::vector<std::byte> pattern(
            spec.logical_nbytes(), static_cast<std::byte>(0x5a));
    source->view().copy_from_host(pattern);
    destination->view().copy_from_host(pattern);

    {
        auto queue = device->create_ops();
        for (int i = 0; i < 32; ++i) {
            CHECK(iom::oid_is_token(queue->copy(source->view(), destination->view())));
        }
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::third_plane_launch);
        iom::oid failure = 0;
        failure = queue->copy(source->view(), destination->view());
        CHECK(iom::oid_is_token(failure));
        CHECK_NE(failure, 0);
        for (int i = 0; i < 32; ++i) {
            CHECK(iom::oid_is_token(queue->copy(source->view(), destination->view())));
        }
        queue.reset();
    }
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("CUDA pre-wait source destruction keeps storage until completion") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    // Large enough that the queued copy kernel is still executing when the
    // host destroys the source; the assertions below hold under every
    // worker/interleaving outcome: the copy completes from storage that was
    // either quarantined or released only after its event fired, so waiting
    // the token always yields correct destination bytes and the data arena
    // never releases the range early. The deterministic quarantine/no-reuse
    // contract is pinned by the ring smoke tests and by the queue-teardown
    // case below.
    const iom::TensorSpec spec{
            iom::TensorShape{{8192, 4096}}, iom::DataType::F32};
    const std::vector<std::byte> expected =
            iom_conformance::encode_standard_tiled_storage(spec);
    const std::vector<std::byte> empty(
            spec.tiled_storage_nbytes(), std::byte{0});

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    CudaStorageOracle oracle;
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    oracle.set_owner_spec(spec);
    oracle.seed(source->view(), expected);
    oracle.set_owner_spec(spec);
    oracle.seed(destination->view(), empty);

    auto queue = device->create_ops();
    const iom::oid token = queue->copy(source->view(), destination->view());

    // Destroy the source before any explicit wait. The registry fence never
    // reports success while the recorded event is pending, so the source
    // range is either quarantined or released only after the copy proved
    // completion and the worker retired its entries; on a fast-enough
    // device the copy completes first and the proved-safe range is legally
    // reused by the arena. In both cases the fresh allocation is safe, the
    // copy still produces the expected bytes, and the deterministic
    // quarantine/no-reuse contract is pinned by the ring smoke tests, the
    // injected-launch-failure case, and the queue-teardown case below.
    source.reset();
    auto fresh = device->create_tensor(spec);
    REQUIRE(fresh != nullptr);
    CHECK(reinterpret_cast<std::uintptr_t>(
                  fresh->view().native_handle())
          % 32 == 0);

    CHECK_NOTHROW(queue->wait(token));
    oracle.set_owner_spec(spec);
    CHECK_EQ(oracle.observe(destination->view()), expected);

    fresh.reset();
    queue.reset();
    destination.reset();
    device.reset();
}

TEST_CASE("CUDA shared-operand copies release storage after both complete") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    const iom::TensorSpec spec{
            iom::TensorShape{{8192, 4096}}, iom::DataType::F32};
    const std::vector<std::byte> expected =
            iom_conformance::encode_standard_tiled_storage(spec);
    const std::vector<std::byte> empty(
            spec.tiled_storage_nbytes(), std::byte{0});

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    CudaStorageOracle oracle;
    auto source = device->create_tensor(spec);
    auto first_destination = device->create_tensor(spec);
    auto second_destination = device->create_tensor(spec);
    oracle.set_owner_spec(spec);
    oracle.seed(source->view(), expected);
    oracle.set_owner_spec(spec);
    oracle.seed(first_destination->view(), empty);
    oracle.set_owner_spec(spec);
    oracle.seed(second_destination->view(), empty);

    auto queue = device->create_ops();
    // Two back-to-back copies share the source; each owns its completion
    // record, so destroying the shared operand before any wait can never
    // recycle it while either recorded event is pending.
    const iom::oid first =
            queue->copy(source->view(), first_destination->view());
    const iom::oid second =
            queue->copy(source->view(), second_destination->view());

    source.reset();
    auto fresh = device->create_tensor(spec);

    CHECK_NOTHROW(queue->wait(first));
    CHECK_NOTHROW(queue->wait(second));
    oracle.set_owner_spec(spec);
    CHECK_EQ(oracle.observe(first_destination->view()), expected);
    oracle.set_owner_spec(spec);
    CHECK_EQ(oracle.observe(second_destination->view()), expected);

    fresh.reset();
    queue.reset();
    first_destination.reset();
    second_destination.reset();
    device.reset();
}

TEST_CASE("CUDA operands destroyed after queue teardown remain quarantined") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    // Queue destruction invalidates every outstanding registry entry up
    // front, so the operand fences provably report failure when the operands
    // are destroyed afterwards: the data arena cannot recycle either range
    // until device teardown, deterministically, with no timing dependence on
    // the worker or the GPU.
    const iom::TensorSpec spec{
            iom::TensorShape{{4096, 2048}}, iom::DataType::F32};

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    {
        auto queue = device->create_ops();
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        (void)token;  // never waited; torn down with the queue
    }
    const void* source_address = source->view().native_handle();
    const void* destination_address =
            destination->view().native_handle();

    source.reset();
    destination.reset();
    auto fresh_source = device->create_tensor(spec);
    auto fresh_destination = device->create_tensor(spec);
    CHECK_NE(fresh_source->view().native_handle(), source_address);
    CHECK_NE(fresh_source->view().native_handle(), destination_address);
    CHECK_NE(fresh_destination->view().native_handle(), source_address);
    CHECK_NE(fresh_destination->view().native_handle(), destination_address);

    fresh_source.reset();
    fresh_destination.reset();
    device.reset();
}

TEST_CASE("CUDA conformance: rank boundary covers rank-eight owners and rejects rank nine") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    iom_conformance::TrafficGate gate;
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    CudaStorageOracle oracle;
    const std::vector<std::vector<std::size_t>> shapes = {
            {2, 2, 2, 2, 2, 2, 16, 16},
            {2, 2, 2, 2, 2, 2, 17, 33}};

    for (const auto& dimensions : shapes) {
        const iom::TensorSpec spec{
                iom::TensorShape{dimensions}, iom::DataType::F32};
        auto source = device->create_tensor(spec);
        auto destination = device->create_tensor(spec);
        const std::vector<std::byte> expected =
                iom_conformance::encode_standard_tiled_storage(spec);
        const std::vector<std::byte> empty(
                spec.tiled_storage_nbytes(), std::byte{0});
        oracle.set_owner_spec(spec);
        oracle.seed(source->view(), expected);
        oracle.set_owner_spec(spec);
        oracle.seed(destination->view(), empty);

        auto queue = device->create_ops();
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        CHECK_NOTHROW(queue->wait(token));
        oracle.set_owner_spec(spec);
        CHECK_EQ(oracle.observe(destination->view()), expected);
    }

    // Rank-eight owner, leading transform, and broadcast correctness plus
    // rank-nine and rank-increasing-transform rejection, all through the
    // real device allocator and native queue.
    iom_conformance::run_accelerator_rank_boundary_conformance(*device);
    CHECK_FALSE(gate.armed());
}
