#include <doctest/doctest.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <initializer_list>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/rocm/device.hpp"

namespace {

constexpr std::initializer_list<iom::DataType> kRocmLeafTypes = {
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

class HipAllocator final : public iom::Allocator {
public:
    explicit HipAllocator(const TrafficGate& gate) : gate_(gate) {}

    void* alloc(std::size_t size) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage allocated inside a transfer, transform, or operation");
        void* block = nullptr;
        const hipError_t status = hipMalloc(&block, size);
        if (status != hipSuccess) {
            throw std::runtime_error("hipMalloc failed in test allocator");
        }
        live_.insert(block);
        ++allocations;
        return block;
    }

    void free(void* buffer) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage freed inside a transfer, transform, or operation");
        const auto found = live_.find(buffer);
        REQUIRE_MESSAGE(
                found != live_.end(),
                "ROCm allocator freed an address it never handed out");
        live_.erase(found);
        CHECK(hipFree(buffer) == hipSuccess);
        ++frees;
    }

    void reset() override { ++resets; }

    std::size_t allocations = 0;
    std::size_t frees = 0;
    std::size_t resets = 0;

private:
    const TrafficGate& gate_;
    std::unordered_set<void*> live_;
};


}  // namespace

TEST_CASE("ROCm conformance: storage and host transfers for every leaf type") {
    // Keep the CPU allocator large enough for the largest shared case.
    TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    HipAllocator candidate_allocator(gate);
    HipAllocator foreign_allocator(gate);
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    auto foreign = iom::make_rocm_device(0, foreign_allocator);
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_storage_and_transfer_conformance(
            devices, kRocmLeafTypes, &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: asynchronous copies against the CPU reference") {
    TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    HipAllocator candidate_allocator(gate);
    HipAllocator foreign_allocator(gate);
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    auto foreign = iom::make_rocm_device(0, foreign_allocator);
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_async_copy_conformance(
            devices, kRocmLeafTypes, &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: copy validation fails before writes and sequences") {
    TrafficGate gate;
    std::vector<std::byte> storage(16 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    HipAllocator candidate_allocator(gate);
    HipAllocator foreign_allocator(gate);
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    auto foreign = iom::make_rocm_device(0, foreign_allocator);
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_copy_error_conformance(
            devices, kRocmLeafTypes, &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: transfer failures keep metadata and ownership") {
    TrafficGate gate;
    std::vector<std::byte> storage(16 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    HipAllocator candidate_allocator(gate);
    HipAllocator foreign_allocator(gate);
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    auto foreign = iom::make_rocm_device(0, foreign_allocator);
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_transfer_error_conformance(
            devices, kRocmLeafTypes, &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: deferred queue lifetime and stability") {
    TrafficGate gate;
    HipAllocator candidate_allocator(gate);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    iom_conformance::run_lifetime_conformance(
            *candidate, kRocmLeafTypes, &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: sub-byte odd-length host reads stay within the staged atomic word") {
    TrafficGate gate;
    HipAllocator candidate_allocator(gate);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    constexpr std::uint64_t salt = 0x180001ull;
    for (const iom::DataType type : {
                 iom::DataType::I2,
                 iom::DataType::F6_E2M3,
                 iom::DataType::F6_E3M2}) {
        const iom::TensorSpec spec{
                iom::TensorShape{std::vector<std::size_t>{1, 17}}, type};
        const std::vector<std::byte> expected =
                iom_conformance::encode_logical(spec, salt);
        CHECK_EQ(
                spec.logical_nbytes(),
                type == iom::DataType::I2 ? std::size_t{5} : std::size_t{13});
        auto tensor = candidate->create_tensor(spec);
        tensor->view().copy_from_host(expected);
        iom_conformance::require_logical_bytes(
                tensor->view(), expected, "odd-length sub-byte host read");
    }
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: compute methods reject capability without submitting") {
    TrafficGate gate;
    HipAllocator candidate_allocator(gate);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    iom_conformance::run_compute_capability_conformance(
            *candidate, kRocmLeafTypes, &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: full shared suite") {
    TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    HipAllocator candidate_allocator(gate);
    HipAllocator foreign_allocator(gate);
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    auto foreign = iom::make_rocm_device(0, foreign_allocator);
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_backend_conformance(
            devices, std::span<const iom::DataType>{kRocmLeafTypes.begin(), 1},
            &gate);
    CHECK_FALSE(gate.armed());
}
