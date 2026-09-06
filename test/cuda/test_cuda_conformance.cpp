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
#include "iom/cpu/device.hpp"
#include "iom/cuda/device.hpp"
#include "cuda/copy.hpp"

namespace {


class TrafficGate final : public iom_conformance::ConformanceObserver {
public:
    void setup_complete() override { armed_ = true; }
    void case_complete() override { armed_ = false; }
    ~TrafficGate() override { armed_ = false; }

    [[nodiscard]] bool armed() const noexcept { return armed_; }

private:
    bool armed_ = false;
};

class HostAllocator final : public iom::Allocator {
public:
    explicit HostAllocator(const TrafficGate& gate) : gate_(gate) {}

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
    const TrafficGate& gate_;
    std::unordered_set<void*> live_;
    std::size_t traffic_ = 0;
};


class CudaAllocator final : public iom::Allocator {
public:
    explicit CudaAllocator(const TrafficGate& gate) : gate_(gate) {}

    void* alloc(std::size_t size) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage allocated inside a transfer, transform, or operation");
        void* pointer = nullptr;
        REQUIRE(cudaMalloc(&pointer, size) == cudaSuccess);
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
        REQUIRE(cudaFree(buffer) == cudaSuccess);
    }

    void reset() override { ++traffic_; }

private:
    const TrafficGate& gate_;
    std::unordered_set<void*> live_;
    std::size_t traffic_ = 0;
};
class ReusingCudaAllocator final : public iom::Allocator {
public:
    ~ReusingCudaAllocator() override {
        for (const Slot& slot : free_) {
            (void)cudaFree(slot.pointer);
        }
        for (const auto& [pointer, size] : live_) {
            (void)size;
            (void)cudaFree(pointer);
        }
    }

    void* alloc(std::size_t size) override {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->size != size) {
                continue;
            }
            void* pointer = it->pointer;
            live_.emplace(pointer, size);
            free_.erase(it);
            return pointer;
        }

        void* pointer = nullptr;
        if (cudaMalloc(&pointer, size) != cudaSuccess) {
            throw std::bad_alloc();
        }
        try {
            live_.emplace(pointer, size);
        } catch (...) {
            (void)cudaFree(pointer);
            throw;
        }
        return pointer;
    }

    void free(void* buffer) override {
        const auto found = live_.find(buffer);
        if (found == live_.end()) {
            throw std::runtime_error(
                    "CUDA reuse allocator received an unknown address");
        }
        free_.push_back({buffer, found->second});
        live_.erase(found);
    }

    void reset() override {}

    void release_free() noexcept {
        for (const Slot& slot : free_) {
            (void)cudaFree(slot.pointer);
        }
        free_.clear();
    }

    [[nodiscard]] std::size_t free_count() const noexcept {
        return free_.size();
    }

private:
    struct Slot {
        void* pointer;
        std::size_t size;
    };

    std::vector<Slot> free_;
    std::unordered_map<void*, std::size_t> live_;
};


struct CudaDevices {
    TrafficGate gate;
    HostAllocator reference_allocator{gate};
    CudaAllocator candidate_allocator{gate};
    CudaAllocator foreign_allocator{gate};
    std::unique_ptr<iom::Device> reference =
            iom::make_cpu_device(reference_allocator);
    std::unique_ptr<iom::Device> candidate =
            iom::make_cuda_device(0, candidate_allocator);
    std::unique_ptr<iom::Device> foreign =
            iom::make_cuda_device(0, foreign_allocator);

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


void expect_repeated_runtime_failure(
        iom::DeviceOps& queue, iom::oid token) {
    std::string message;
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool caught = false;
        try {
            queue.wait(token);
        } catch (const std::runtime_error& error) {
            caught = true;
            if (message.empty()) {
                message = error.what();
            } else {
                CHECK_EQ(std::string_view(error.what()), message);
            }
        }
        CHECK(caught);
    }
    CHECK_FALSE(message.empty());
}

}  // namespace
TEST_CASE("Device::supported_data_types returns the per-backend 23-entry span") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    TrafficGate gate;
    CudaAllocator allocator(gate);
    const std::unique_ptr<iom::Device> candidate =
            iom::make_cuda_device(0, allocator);
    const std::span<const iom::DataType> supported =
            candidate->supported_data_types();
    constexpr iom::DataType expected[] = {
            iom::DataType::BOOL,
            iom::DataType::I2, iom::DataType::U2,
            iom::DataType::I4, iom::DataType::U4,
            iom::DataType::I8, iom::DataType::U8,
            iom::DataType::I16, iom::DataType::U16,
            iom::DataType::I32, iom::DataType::U32,
            iom::DataType::I64, iom::DataType::U64,
            iom::DataType::F4_E2M1,
            iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
            iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
            iom::DataType::F8_E8M0,
            iom::DataType::F16, iom::DataType::BF16,
            iom::DataType::F32, iom::DataType::F64,
    };
    REQUIRE_EQ(supported.size(), sizeof(expected) / sizeof(expected[0]));
    for (std::size_t i = 0; i < supported.size(); ++i) {
        CHECK_EQ(supported[i], expected[i]);
    }
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
            &devices.gate, "CUDA");
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: full shared suite") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, &oracle);
    CHECK_FALSE(devices.gate.armed());
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

    CHECK_THROWS_AS(
            queue->copy(source->view(), invalid_destination->view()),
            std::invalid_argument);

    const iom::oid first = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(first), 1);
    CHECK_NOTHROW(queue->wait(first));

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::event_create);
    CHECK_THROWS_AS(
            queue->copy(source->view(), destination->view()),
            std::runtime_error);
    const iom::oid second = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(second), 2);
    CHECK_NOTHROW(queue->wait(second));

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::third_plane_launch);
    iom::oid launch_failure = 0;
    CHECK_NOTHROW(
            launch_failure = queue->copy(
                    source->view(), destination->view()));
    CHECK_NE(launch_failure, 0);
    CHECK_EQ(iom_conformance::token_sequence(launch_failure), 3);
    expect_repeated_runtime_failure(*queue, launch_failure);

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::event_record);
    const iom::oid record_failure =
            queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(record_failure), 4);
    expect_repeated_runtime_failure(*queue, record_failure);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

    // Failed work remains quarantined until device teardown; the allocator
    // must not recycle either operand while its failed entries are retained.
    {
        ReusingCudaAllocator allocator;
        auto device = iom::make_cuda_device(0, allocator);
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
        expect_repeated_runtime_failure(*canary_queue, canary_failure);

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
        allocator.release_free();
        CHECK_EQ(allocator.free_count(), 0);
        device.reset();
        CHECK_EQ(allocator.free_count(), 2);
    }
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
}

TEST_CASE("CUDA queue destruction fences pending copies") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    TrafficGate gate;
    CudaAllocator allocator(gate);
    auto device = iom::make_cuda_device(0, allocator);
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
            CHECK_NOTHROW(queue->copy(source->view(), destination->view()));
        }
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::third_plane_launch);
        iom::oid failure = 0;
        CHECK_NOTHROW(
                failure = queue->copy(source->view(), destination->view()));
        CHECK_NE(failure, 0);
        for (int i = 0; i < 32; ++i) {
            CHECK_NOTHROW(queue->copy(source->view(), destination->view()));
        }
        queue.reset();
    }
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
    CHECK_FALSE(gate.armed());
}
