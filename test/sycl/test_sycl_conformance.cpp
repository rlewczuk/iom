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
#include <unordered_set>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/sycl/device.hpp"
#include "runtime.hpp"

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
    const TrafficGate& gate_;
    std::unordered_set<void*> live_;
};

class SyclAllocator final : public iom::Allocator {
public:
    explicit SyclAllocator(const TrafficGate& gate) : gate_(gate) {}

    void bind_context(const sycl::context& context) {
        context_ = context;
        const std::vector<sycl::device> devices = context.get_devices();
        REQUIRE(!devices.empty());
        device_ = devices.front();
    }

    void* alloc(std::size_t size) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage allocated inside a transfer, transform, or operation");
        REQUIRE(context_.has_value());
        void* pointer = sycl::malloc_shared(size, device_, *context_);
        if (pointer == nullptr) {
            throw std::bad_alloc();
        }
        live_.insert(pointer);
        ++allocations;
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
        REQUIRE(context_.has_value());
        sycl::free(buffer, *context_);
        ++frees;
    }

    void reset() override {}

    [[nodiscard]] const sycl::context& context() const {
        REQUIRE(context_.has_value());
        return *context_;
    }

    [[nodiscard]] const sycl::device& device() const {
        REQUIRE(context_.has_value());
        return device_;
    }

    std::size_t allocations = 0;
    std::size_t frees = 0;

private:
    const TrafficGate& gate_;
    std::optional<sycl::context> context_;
    sycl::device device_;
    std::unordered_set<void*> live_;
};

class HostPointerAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        ++allocations;
        raw_ = ::operator new(size, std::align_val_t(32));
        return raw_;
    }

    void free(void* buffer) override {
        CHECK_EQ(buffer, raw_);
        ++frees;
        ::operator delete(raw_, std::align_val_t(32));
        raw_ = nullptr;
    }

    void reset() override {}

    std::size_t allocations = 0;
    std::size_t frees = 0;

private:
    void* raw_ = nullptr;
};

SyclAllocator* active_context_allocator = nullptr;

void capture_context(const sycl::context& context) {
    REQUIRE(active_context_allocator != nullptr);
    active_context_allocator->bind_context(context);
}

class ContextCallsRestore final {
public:
    ContextCallsRestore()
            : saved_(iom::sycl_detail::context_calls),
              saved_allocator_(active_context_allocator) {}

    ContextCallsRestore(const ContextCallsRestore&) = delete;
    ContextCallsRestore& operator=(const ContextCallsRestore&) = delete;

    ~ContextCallsRestore() {
        iom::sycl_detail::context_calls = saved_;
        active_context_allocator = saved_allocator_;
    }

private:
    iom::sycl_detail::ContextCalls saved_;
    SyclAllocator* saved_allocator_;
};

struct SyclDevices {
    ContextCallsRestore context_restore;
    TrafficGate gate;
    HostAllocator reference_allocator{gate};
    SyclAllocator candidate_allocator{gate};
    SyclAllocator foreign_allocator{gate};
    std::unique_ptr<iom::Device> reference;
    std::unique_ptr<iom::Device> candidate;
    std::unique_ptr<iom::Device> foreign;

    SyclDevices() {
        iom::sycl_detail::context_calls.context_ready = &capture_context;
        reference = iom::make_cpu_device(reference_allocator);

        active_context_allocator = &candidate_allocator;
        candidate = iom::make_sycl_device(0, candidate_allocator);
        active_context_allocator = &foreign_allocator;
        foreign = iom::make_sycl_device(0, foreign_allocator);
        active_context_allocator = nullptr;
    }

    [[nodiscard]] iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};

class SyclStorageOracle final
        : public iom_conformance::AcceleratorStorageOracle {
public:
    explicit SyclStorageOracle(const SyclAllocator& allocator)
            : allocator_(allocator) {}

    void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        REQUIRE_EQ(encoded.size(), bytes);
        sycl::queue queue(
                allocator_.context(), allocator_.device(),
                sycl::property_list{sycl::property::queue::in_order{}});
        queue.memcpy(view.native_handle(), encoded.data(), bytes)
                .wait_and_throw();
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        std::vector<std::byte> result(bytes);
        sycl::queue queue(
                allocator_.context(), allocator_.device(),
                sycl::property_list{sycl::property::queue::in_order{}});
        queue.memcpy(result.data(), view.native_handle(), bytes)
                .wait_and_throw();
        return result;
    }

private:
    const SyclAllocator& allocator_;
};

}  // namespace
TEST_CASE("Device::supported_data_types returns the per-backend 23-entry span") {
    SyclDevices devices;
    const std::span<const iom::DataType> supported =
            devices.candidate->supported_data_types();
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


TEST_CASE("SYCL rejects aligned non-USM allocator storage") {
    REQUIRE(iom::sycl_detail::eligible_device_count() > 0);

    HostPointerAllocator allocator;
    auto device = iom::make_sycl_device(0, allocator);
    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::F32};

    CHECK_THROWS((void)device->create_tensor(spec));
    CHECK_EQ(allocator.allocations, 1);
    CHECK_EQ(allocator.frees, 1);
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

    SyclStorageOracle direct(devices.candidate_allocator);
    iom_conformance::PermutingStorageOracle perturbed(
            direct, iom_conformance::swap_first_adjacent_slots);
    CHECK_FALSE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, perturbed, &devices.gate,
            false, false));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: storage oracle covers every leaf width and padded shape") {
    SyclDevices devices;
    SyclStorageOracle oracle(devices.candidate_allocator);
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            oracle, &devices.gate));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: asynchronous copies against the CPU reference") {
    SyclDevices devices;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
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
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: full shared suite") {
    SyclDevices devices;
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}
