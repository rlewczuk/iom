#include <doctest/doctest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <unordered_set>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "iom/cpu/device.hpp"
#include "iom/cuda/device.hpp"

namespace {

constexpr std::initializer_list<iom::DataType> kCudaLeafTypes = {
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

}  // namespace

TEST_CASE("CUDA conformance: storage and host transfers for every leaf type") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), kCudaLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: asynchronous copies against the CPU reference") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), kCudaLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: copy validation fails before writes and sequences") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), kCudaLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: transfer failures keep metadata and ownership") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_transfer_error_conformance(
            devices.conformance(), kCudaLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: deferred queue lifetime and stability") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, kCudaLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: compute methods reject capability without submitting") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, kCudaLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: full shared suite") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            std::span<const iom::DataType>{kCudaLeafTypes.begin(), 1},
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}
