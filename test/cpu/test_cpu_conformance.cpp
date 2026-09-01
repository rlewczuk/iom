#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"

namespace {

// Every declared leaf type; the CPU backend stores all of them with
// QuantizationFormat::NONE.
constexpr std::initializer_list<iom::DataType> kCpuLeafTypes = {
    iom::DataType::BOOL,
    iom::DataType::I2, iom::DataType::U2,
    iom::DataType::I4, iom::DataType::U4,
    iom::DataType::I8, iom::DataType::U8,
    iom::DataType::I16, iom::DataType::U16,
    iom::DataType::I32, iom::DataType::U32,
    iom::DataType::I64, iom::DataType::U64,
    iom::DataType::F4_E2M1,
    iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
    iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2, iom::DataType::F8_E8M0,
    iom::DataType::F16, iom::DataType::BF16,
    iom::DataType::F32, iom::DataType::F64,
};

// Observes harness phase boundaries and exposes the armed window in which
// tensor-storage allocators must stay silent.
class TrafficGate final : public iom_conformance::ConformanceObserver {
public:
    void setup_complete() override { armed_ = true; }
    void case_complete() override { armed_ = false; }
    ~TrafficGate() override { armed_ = false; }

    [[nodiscard]] bool armed() const noexcept { return armed_; }

private:
    bool armed_ = false;
};

// Recording allocator over 32-byte-aligned blocks. While the gate is armed —
// inside transfers, transforms, and queue operations — every alloc or free is
// a conformance failure: no case may allocate operands there.
class GatedAllocator final : public iom::Allocator {
public:
    explicit GatedAllocator(const TrafficGate& gate) : gate_(gate) {}

    void* alloc(std::size_t size) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage allocated inside a transfer, transform, or "
                "operation");
        void* block = ::operator new(size, std::align_val_t(32));
        live_.insert(block);
        ++traffic_;
        return block;
    }

    void free(void* buffer) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage freed inside a transfer, transform, or "
                "operation");
        const auto found = live_.find(buffer);
        REQUIRE_MESSAGE(
                found != live_.end(),
                "allocator freed an address it never handed out");
        live_.erase(found);
        ++traffic_;
        ::operator delete(buffer, std::align_val_t(32));
    }

    void reset() override {
        ++traffic_;
    }

private:
    const TrafficGate& gate_;
    std::unordered_set<void*> live_;
    std::size_t traffic_ = 0;
};

// One independent CPU reference device, one candidate device, and one foreign
// device instance of the same backend ordinal, each over its own allocator.
struct CpuDevices {
    TrafficGate gate;
    GatedAllocator reference_allocator{gate};
    GatedAllocator candidate_allocator{gate};
    GatedAllocator foreign_allocator{gate};
    std::unique_ptr<iom::Device> reference = iom::make_cpu_device(reference_allocator);
    std::unique_ptr<iom::Device> candidate = iom::make_cpu_device(candidate_allocator);
    std::unique_ptr<iom::Device> foreign = iom::make_cpu_device(foreign_allocator);

    [[nodiscard]] iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};

}  // namespace

// The CPU reference instantiation passes every shared case for every declared
// leaf type. Each entry point runs in its own test case with freshly and
// independently allocated devices, allocators, and host buffers.

TEST_CASE("CPU conformance: storage and host transfers for every leaf type") {
    CpuDevices devices;
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), kCpuLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: asynchronous copies against the CPU reference") {
    CpuDevices devices;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), kCpuLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: copy validation fails before writes and sequences") {
    CpuDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), kCpuLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: transfer failures keep metadata and ownership") {
    CpuDevices devices;
    iom_conformance::run_transfer_error_conformance(
            devices.conformance(), kCpuLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: deferred queue lifetime and stability") {
    CpuDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, kCpuLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: compute methods reject capability without submitting") {
    CpuDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, kCpuLeafTypes, &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: full suite composes every shared case") {
    CpuDevices devices;
    iom_conformance::run_backend_conformance(
            devices.conformance(), std::span<const iom::DataType>{kCpuLeafTypes.begin(), 1},
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

// A deliberately perturbed candidate layout or view map must surface as a
// bit-for-bit logical-byte mismatch; the independently generated encodings
// make round-trip cancellation impossible.
TEST_CASE("conformance harness detects perturbed candidate bytes") {
    TrafficGate gate;
    GatedAllocator reference_allocator(gate);
    GatedAllocator candidate_allocator(gate);
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_cpu_device(candidate_allocator);

    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::U8};
    auto reference_tensor = reference->create_tensor(spec);
    auto candidate_tensor = candidate->create_tensor(spec);

    const std::vector<std::byte> seeded = iom_conformance::encode_logical(spec, 7);
    reference_tensor->view().copy_from_host(seeded);
    candidate_tensor->view().copy_from_host(seeded);
    REQUIRE_FALSE(iom_conformance::first_logical_mismatch(
            candidate_tensor->view(), seeded)
                          .has_value());

    // Flip one byte of the candidate storage at the shared layout helper's
    // slot for logical element zero.
    const std::size_t element_zero_slot = iom::detail::standard_layout_slot(
            spec, iom_conformance::span_of({0, 0, 0, 0}));
    auto* storage =
            static_cast<std::byte*>(candidate_tensor->view().native_handle());
    storage[element_zero_slot] ^= std::byte{0xFF};

    const std::optional<std::size_t> layout_mismatch =
            iom_conformance::first_logical_mismatch(
                    candidate_tensor->view(), seeded);
    REQUIRE(layout_mismatch.has_value());
    CHECK_EQ(*layout_mismatch, element_zero_slot);

    iom::TensorView stepped_window = candidate_tensor->view().slice(0, 1, 1);
    const iom::TensorView earlier_window =
            candidate_tensor->view().slice(0, 0, 1);
    const std::vector<std::byte> stepped_pattern =
            iom_conformance::encode_logical(stepped_window.spec(), 8);
    const std::vector<std::byte> earlier_pattern =
            iom_conformance::encode_logical(earlier_window.spec(), 9);
    stepped_window.copy_from_host(stepped_pattern);
    CHECK_FALSE(iom_conformance::first_logical_mismatch(
            stepped_window, stepped_pattern)
                        .has_value());
    CHECK(iom_conformance::first_logical_mismatch(stepped_window, earlier_pattern)
                  .has_value());
}
