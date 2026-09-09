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
#include "iom/cpu/device.hpp"
#include "iom/sycl/device.hpp"
#include "copy.hpp"
#include "staging_pool.hpp"
#include "runtime.hpp"

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

class SyclAllocator final : public iom::Allocator {
public:
    explicit SyclAllocator(const iom_conformance::TrafficGate& gate) : gate_(gate) {}

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
    const iom_conformance::TrafficGate& gate_;
    std::optional<sycl::context> context_;
    sycl::device device_;
    std::unordered_set<void*> live_;
};

class ReusingSyclAllocator final : public iom::Allocator {
public:
    ~ReusingSyclAllocator() override {
        if (!context_.has_value()) {
            return;
        }
        for (const Slot& slot : free_) {
            try {
                sycl::free(slot.pointer, *context_);
            } catch (...) {
            }
        }
        for (const auto& [pointer, size] : live_) {
            (void)size;
            try {
                sycl::free(pointer, *context_);
            } catch (...) {
            }
        }
    }

    void bind_context(const sycl::context& context) {
        context_ = context;
        const std::vector<sycl::device> devices = context.get_devices();
        REQUIRE(!devices.empty());
        device_ = devices.front();
    }

    void* alloc(std::size_t size) override {
        REQUIRE(context_.has_value());
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->size != size) {
                continue;
            }
            void* pointer = it->pointer;
            live_.emplace(pointer, size);
            free_.erase(it);
            ++allocations;
            return pointer;
        }

        void* pointer = sycl::malloc_shared(size, device_, *context_);
        if (pointer == nullptr) {
            throw std::bad_alloc();
        }
        try {
            live_.emplace(pointer, size);
        } catch (...) {
            sycl::free(pointer, *context_);
            throw;
        }
        ++allocations;
        return pointer;
    }

    void free(void* buffer) override {
        const auto found = live_.find(buffer);
        REQUIRE_MESSAGE(
                found != live_.end(),
                "SYCL reuse allocator received an unknown address");
        free_.push_back({buffer, found->second});
        live_.erase(found);
        ++frees;
    }

    void reset() override {}

    void release_free() noexcept {
        if (!context_.has_value()) {
            return;
        }
        for (const Slot& slot : free_) {
            try {
                sycl::free(slot.pointer, *context_);
            } catch (...) {
            }
        }
        free_.clear();
    }

    [[nodiscard]] std::size_t free_count() const noexcept {
        return free_.size();
    }

    std::size_t allocations = 0;
    std::size_t frees = 0;

private:
    struct Slot {
        void* pointer;
        std::size_t size;
    };

    std::optional<sycl::context> context_;
    sycl::device device_;
    std::vector<Slot> free_;
    std::unordered_map<void*, std::size_t> live_;
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
ReusingSyclAllocator* active_reusing_allocator = nullptr;

void capture_context(const sycl::context& context) {
    REQUIRE(active_context_allocator != nullptr);
    active_context_allocator->bind_context(context);
}

void capture_reusing_context(const sycl::context& context) {
    REQUIRE(active_reusing_allocator != nullptr);
    active_reusing_allocator->bind_context(context);
}

class ContextCallsRestore final {
public:
    ContextCallsRestore()
            : saved_(iom::sycl_detail::context_calls),
              saved_allocator_(active_context_allocator),
              saved_reusing_allocator_(active_reusing_allocator) {}

    ContextCallsRestore(const ContextCallsRestore&) = delete;
    ContextCallsRestore& operator=(const ContextCallsRestore&) = delete;

    ~ContextCallsRestore() {
        iom::sycl_detail::context_calls = saved_;
        active_context_allocator = saved_allocator_;
        active_reusing_allocator = saved_reusing_allocator_;
    }

private:
    iom::sycl_detail::ContextCalls saved_;
    SyclAllocator* saved_allocator_;
    ReusingSyclAllocator* saved_reusing_allocator_;
};

struct SyclDevices {
    ContextCallsRestore context_restore;
    iom_conformance::TrafficGate gate;
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

struct ReusingSyclDevice {
    ContextCallsRestore context_restore;
    ReusingSyclAllocator allocator;
    std::unique_ptr<iom::Device> device;

    ReusingSyclDevice() {
        iom::sycl_detail::context_calls.context_ready =
                &capture_reusing_context;
        active_reusing_allocator = &allocator;
        device = iom::make_sycl_device(0, allocator);
        active_reusing_allocator = nullptr;
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

TEST_CASE("SYCL tensor destruction fences queued work") {
    REQUIRE(iom::sycl_detail::eligible_device_count() > 0);
    ReusingSyclDevice devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = devices.device->create_tensor(spec);
    auto destination = devices.device->create_tensor(spec);
    const std::vector<std::byte> pattern(
            spec.logical_nbytes(), static_cast<std::byte>(0x3c));
    const std::vector<std::byte> zero(
            spec.logical_nbytes(), static_cast<std::byte>(0));
    source->view().copy_from_host(pattern);
    destination->view().copy_from_host(zero);

    auto queue = devices.device->create_ops();
    const iom::oid token = queue->copy(
            source->view(), destination->view());
    source.reset();
    auto fresh_source = devices.device->create_tensor(spec);
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
    CHECK_EQ(devices.allocator.frees, 3);
    CHECK_EQ(devices.allocator.allocations, 3);
    devices.device.reset();
    devices.allocator.release_free();
    CHECK_EQ(devices.allocator.free_count(), 0);
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
    ReusingSyclDevice devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = devices.device->create_tensor(spec);
    auto destination = devices.device->create_tensor(spec);
    const void* source_address = source->view().native_handle();
    const void* destination_address =
            destination->view().native_handle();
    auto queue = devices.device->create_ops();

    iom::sycl_detail::inject_submission_fault_for_testing(
            iom::sycl_detail::SubmissionFault::post_launch);
    const iom::oid token = queue->copy(
            source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(token), 1);
    iom_conformance::expect_repeated_runtime_failure(*queue, token);

    source.reset();
    destination.reset();
    auto fresh_source = devices.device->create_tensor(spec);
    auto fresh_destination = devices.device->create_tensor(spec);
    CHECK_NE(fresh_source->view().native_handle(), source_address);
    CHECK_NE(
            fresh_destination->view().native_handle(), destination_address);
    CHECK_NE(
            fresh_source->view().native_handle(),
            fresh_destination->view().native_handle());

    fresh_source.reset();
    fresh_destination.reset();
    queue.reset();
    CHECK_EQ(devices.allocator.frees, 2);
    CHECK_EQ(devices.allocator.allocations, 4);
    devices.device.reset();
    CHECK_EQ(devices.allocator.frees, 4);
    devices.allocator.release_free();
    CHECK_EQ(devices.allocator.free_count(), 0);
}

TEST_CASE("SYCL queue destruction fences pending copies") {
    REQUIRE(iom::sycl_detail::eligible_device_count() > 0);
    ReusingSyclDevice devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = devices.device->create_tensor(spec);
    auto destination = devices.device->create_tensor(spec);
    const void* source_address = source->view().native_handle();
    const void* destination_address =
            destination->view().native_handle();

    {
        auto queue = devices.device->create_ops();
        CHECK_NOTHROW(queue->copy(source->view(), destination->view()));
        CHECK_NOTHROW(queue->copy(source->view(), destination->view()));
        iom::sycl_detail::inject_submission_fault_for_testing(
                iom::sycl_detail::SubmissionFault::post_launch);
        CHECK_NOTHROW(queue->copy(source->view(), destination->view()));
        iom::sycl_detail::inject_submission_fault_for_testing(
                iom::sycl_detail::SubmissionFault::none);
    }

    source.reset();
    destination.reset();
    CHECK_EQ(devices.allocator.frees, 0);
    auto fresh_source = devices.device->create_tensor(spec);
    auto fresh_destination = devices.device->create_tensor(spec);
    CHECK_NE(fresh_source->view().native_handle(), source_address);
    CHECK_NE(
            fresh_destination->view().native_handle(), destination_address);
    fresh_source.reset();
    fresh_destination.reset();
    CHECK_EQ(devices.allocator.frees, 2);
    devices.device.reset();
    CHECK_EQ(devices.allocator.frees, 4);
    devices.allocator.release_free();
    CHECK_EQ(devices.allocator.free_count(), 0);
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
            devices.candidate_allocator.context(),
            devices.candidate_allocator.device(),
            sycl::property_list{sycl::property::queue::in_order{}});
    iom::sycl_detail::StagingSlotPool pool(
            devices.candidate_allocator.context(),
            devices.candidate_allocator.device());

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
    SyclStorageOracle oracle(devices.candidate_allocator);
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
            &devices.gate, "SYCL");
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("SYCL conformance: full shared suite") {
    SyclDevices devices;
    SyclStorageOracle oracle(devices.candidate_allocator);
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, &oracle);
    CHECK_FALSE(devices.gate.armed());
}
