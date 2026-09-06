#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
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

    [[nodiscard]] std::size_t traffic() const noexcept {
        return traffic_;
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

void require_storage_outside_view_is_zero(
        const iom::TensorSpec& owner_spec, const iom::TensorView& view) {
    std::vector<unsigned char> touched(
            owner_spec.tiled_storage_nbytes(), 0);
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::span<const std::size_t> strides =
            view.plane_strides();
    REQUIRE_EQ(dimensions.size(), 4);
    for (std::size_t first = 0; first < dimensions[0]; ++first) {
        for (std::size_t second = 0; second < dimensions[1]; ++second) {
            const std::size_t plane =
                    view.plane_offset()
                    + first * strides[0] + second * strides[1];
            for (std::size_t row = 0; row < dimensions[2]; ++row) {
                for (std::size_t column = 0;
                     column < dimensions[3]; ++column) {
                    const std::size_t byte =
                            iom::detail::standard_plane_slot(
                                    owner_spec, plane, row, column);
                    REQUIRE_LT(byte, touched.size());
                    touched[byte] = 1;
                }
            }
        }
    }

    const auto* storage = static_cast<const unsigned char*>(
            view.native_handle());
    for (std::size_t byte = 0; byte < touched.size(); ++byte) {
        if (touched[byte] == 0) {
            CHECK_EQ(storage[byte], 0);
        }
    }
}

void require_subbyte_storage_scope_is_zero(
        const iom::TensorSpec& owner_spec, const iom::TensorView& view) {
    const std::size_t bits =
            iom::detail::leaf_bits(owner_spec.data_type);
    std::vector<unsigned char> touched(
            owner_spec.tiled_storage_nbytes(), 0);
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    REQUIRE_EQ(dimensions.size(), 2);
    for (std::size_t row = 0; row < dimensions[0]; ++row) {
        for (std::size_t column = 0; column < dimensions[1]; ++column) {
            const std::size_t bit =
                    iom::detail::standard_plane_slot(
                            owner_spec, view.plane_offset(), row, column)
                    * bits;
            for (std::size_t offset = 0; offset < bits; ++offset) {
                touched[(bit + offset) / 8] |=
                        static_cast<unsigned char>(
                                1u << ((bit + offset) % 8));
            }
        }
    }

    const auto* storage = static_cast<const unsigned char*>(
            view.native_handle());
    for (std::size_t byte = 0; byte < touched.size(); ++byte) {
        CHECK_EQ(
                static_cast<unsigned char>(storage[byte] & ~touched[byte]),
                0);
    }
}
TEST_CASE("Device::supported_data_types returns the per-backend 23-entry span") {
    std::vector<std::byte> storage(1024);
    iom::LinearAllocator allocator(storage.data(), storage.size());
    const std::unique_ptr<iom::Device> candidate =
            iom::make_cpu_device(allocator);
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


// The CPU reference instantiation passes every shared case for every declared
// leaf type. Each entry point runs in its own test case with freshly and
// independently allocated devices, allocators, and host buffers.

TEST_CASE("CPU conformance: storage and host transfers for every leaf type") {
    CpuDevices devices;
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: standard storage oracle covers every leaf width and padded shape") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            oracle, &devices.gate));
    CHECK_FALSE(devices.gate.armed());
}
TEST_CASE("CPU conformance: storage oracle identifies perturbed transfer map") {
    CpuDevices devices;
    const std::span<const iom::DataType> supported =
            devices.candidate->supported_data_types();
    REQUIRE(supported.size() >= 1);
    const std::span<const iom::DataType> one_type =
            supported.subspan(0, 1);

    iom_conformance::CpuStorageOracle direct;
    iom_conformance::PermutingStorageOracle perturbed(
            direct, iom_conformance::swap_first_adjacent_slots);
    CHECK_FALSE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, perturbed, &devices.gate,
            false, false));

    iom_conformance::PermutingStorageOracle identity(
            direct, [](std::size_t logical) { return logical; });
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, identity, &devices.gate));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: asynchronous copies against the CPU reference") {
    CpuDevices devices;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: copy validation fails before writes and sequences") {
    CpuDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: transfer failures keep metadata and ownership") {
    CpuDevices devices;
    iom_conformance::run_transfer_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: deferred queue lifetime and stability") {
    CpuDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: compute methods reject capability without submitting") {
    CpuDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate, "CPU");
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: full shared suite composes every shared case") {
    CpuDevices devices;
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
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

TEST_CASE("CPU conformance: tile-blocked copy preserves logical and physical window scope") {
    CpuDevices devices;
    const iom::TensorSpec owner_spec{
            iom::TensorShape{{4, 3, 17, 33}}, iom::DataType::U8};
    auto reference = devices.candidate->create_tensor(owner_spec);
    auto candidate = devices.candidate->create_tensor(owner_spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(owner_spec, 0x1234);
    reference->view().copy_from_host(pattern);
    std::memset(
            candidate->view().native_handle(), 0,
            owner_spec.tiled_storage_nbytes());

    const iom::TensorView source =
            reference->view().slice(0, 0, 2, 2);
    iom::TensorView destination =
            candidate->view().slice(0, 1, 2, 1);
    const std::vector<std::byte> expected =
            iom_conformance::read_logical(source);
    const std::size_t candidate_traffic =
            devices.candidate_allocator.traffic();
    devices.gate.setup_complete();
    auto queue = devices.candidate->create_ops();
    const iom::oid token = queue->copy(source, destination);
    queue->wait(token);
    CHECK_EQ(
            devices.candidate_allocator.traffic(), candidate_traffic);
    devices.gate.case_complete();

    iom_conformance::require_logical_bytes(
            destination, expected, "tile-blocked window copy");
    require_storage_outside_view_is_zero(owner_spec, destination);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: blocked byte-aligned copy matches `std::memcpy` byte-for-byte") {
    CpuDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{4096, 4096}}, iom::DataType::F32};
    auto reference = devices.candidate->create_tensor(spec);
    auto candidate = devices.candidate->create_tensor(spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0x5678);
    reference->view().copy_from_host(pattern);
    std::memset(
            candidate->view().native_handle(), 0,
            spec.tiled_storage_nbytes());

    const std::size_t candidate_traffic =
            devices.candidate_allocator.traffic();
    devices.gate.setup_complete();
    auto queue = devices.candidate->create_ops();
    const iom::oid token =
            queue->copy(reference->view(), candidate->view());
    queue->wait(token);
    CHECK_EQ(
            devices.candidate_allocator.traffic(), candidate_traffic);
    devices.gate.case_complete();

    CHECK_EQ(
            std::memcmp(
                    reference->view().native_handle(),
                    candidate->view().native_handle(),
                    spec.tiled_storage_nbytes()),
            0);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: blocked sub-byte copy preserves LSB-first packing and zero tail bits") {
    CpuDevices devices;
    for (const iom::DataType data_type : {
                 iom::DataType::I2,
                 iom::DataType::F6_E2M3,
                 iom::DataType::F6_E3M2}) {
        const iom::TensorSpec spec{
                iom::TensorShape{{1, 17}}, data_type};
        auto reference = devices.candidate->create_tensor(spec);
        auto candidate = devices.candidate->create_tensor(spec);
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(spec, 0x9ABC);
        reference->view().copy_from_host(pattern);
        std::memset(
                candidate->view().native_handle(), 0,
                spec.tiled_storage_nbytes());

        const std::size_t candidate_traffic =
                devices.candidate_allocator.traffic();
        devices.gate.setup_complete();
        auto queue = devices.candidate->create_ops();
        const iom::oid token =
                queue->copy(reference->view(), candidate->view());
        queue->wait(token);
        CHECK_EQ(
                devices.candidate_allocator.traffic(), candidate_traffic);
        devices.gate.case_complete();

        iom_conformance::require_logical_bytes(
                candidate->view(), pattern, "sub-byte blocked copy");
        require_subbyte_storage_scope_is_zero(
                spec, candidate->view());
        CHECK_FALSE(devices.gate.armed());
    }
}

TEST_CASE("CPU conformance: blocked copy walks lockstep planes without scratch allocation") {
    CpuDevices devices;
    const iom::TensorSpec owner_spec{
            iom::TensorShape{{4, 3, 17, 33}}, iom::DataType::U8};
    auto reference = devices.candidate->create_tensor(owner_spec);
    auto candidate = devices.candidate->create_tensor(owner_spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(owner_spec, 0xDEF0);
    reference->view().copy_from_host(pattern);
    std::memset(
            candidate->view().native_handle(), 0,
            owner_spec.tiled_storage_nbytes());

    const iom::TensorView source =
            reference->view().slice(0, 0, 2, 2);
    iom::TensorView destination =
            candidate->view().slice(0, 1, 2, 1);
    const std::vector<std::byte> expected =
            iom_conformance::read_logical(source);
    const std::size_t candidate_traffic =
            devices.candidate_allocator.traffic();
    devices.gate.setup_complete();
    auto queue = devices.candidate->create_ops();
    const iom::oid token = queue->copy(source, destination);
    queue->wait(token);
    CHECK_EQ(
            devices.candidate_allocator.traffic(), candidate_traffic);
    devices.gate.case_complete();

    iom_conformance::require_logical_bytes(
            destination, expected, "lockstep plane copy");
    CHECK_FALSE(devices.gate.armed());
}

namespace {
// Submits the copy from a frame that returns before the caller waits: the
// derived-view temporaries die when this function returns, so a queue that
// stored their addresses would leave the worker dereferencing dead stack
// storage. No named local binds either view; `copy`'s destination
// parameter is a non-const reference, so the rvalue destination view is
// bound through const_cast — the view object is never modified, the worker
// writes through the owner's storage.
iom::oid submit_temporary_copy(
        iom::DeviceOps& queue, iom::Tensor& t, iom::Tensor& u,
        std::size_t half) {
    return queue.copy(
            t.view().slice(0, 0, half),
            const_cast<iom::TensorView&>(
                    static_cast<const iom::TensorView&>(
                            u.view().slice(0, 0, half))));
}
}  // namespace

TEST_CASE("CPU copy survives derived-view temporaries") {
    CpuDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{4, 3, 17, 33}}, iom::DataType::U8};
    std::unique_ptr<iom::Tensor> t = devices.candidate->create_tensor(spec);
    std::unique_ptr<iom::Tensor> u = devices.candidate->create_tensor(spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0x5A7C);
    t->view().copy_from_host(pattern);
    std::memset(
            u->view().native_handle(), 0, spec.tiled_storage_nbytes());

    auto queue = devices.candidate->create_ops();
    const std::vector<std::size_t> dims = {4, 3, 17, 33};
    const iom::oid token =
            submit_temporary_copy(*queue, *t, *u, dims[0] / 2);
    REQUIRE_NOTHROW(queue->wait(token));

    const iom::TensorView expected_view = t->view().slice(0, 0, dims[0] / 2);
    const iom::TensorView actual_view = u->view().slice(0, 0, dims[0] / 2);
    const std::vector<std::byte> expected =
            iom_conformance::read_logical(expected_view);
    iom_conformance::require_logical_bytes(
            actual_view, expected, "temporary-view copy");
    CHECK_FALSE(devices.gate.armed());
}
