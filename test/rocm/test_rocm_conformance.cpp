#include <doctest/doctest.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <cstdint>
#include <memory>
#include <new>
#include <spawn.h>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include "backend/backend_conformance_common.hpp"

#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/rocm/device.hpp"
#include "rocm/copy.hpp"
extern char** environ;


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
class ReusingHipAllocator final : public iom::Allocator {
public:
    ~ReusingHipAllocator() override {
        for (const Slot& slot : free_) {
            (void)hipFree(slot.pointer);
        }
        for (const auto& [pointer, size] : live_) {
            (void)size;
            (void)hipFree(pointer);
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
        if (hipMalloc(&pointer, size) != hipSuccess) {
            throw std::bad_alloc();
        }
        try {
            live_.emplace(pointer, size);
        } catch (...) {
            (void)hipFree(pointer);
            throw;
        }
        return pointer;
    }

    void free(void* buffer) override {
        const auto found = live_.find(buffer);
        if (found == live_.end()) {
            throw std::runtime_error(
                    "ROCm reuse allocator received an unknown address");
        }
        free_.push_back({buffer, found->second});
        live_.erase(found);
    }

    void reset() override {}

    void release_free() noexcept {
        for (const Slot& slot : free_) {
            (void)hipFree(slot.pointer);
        }
        free_.clear();
    }

private:
    struct Slot {
        void* pointer;
        std::size_t size;
    };

    std::vector<Slot> free_;
    std::unordered_map<void*, std::size_t> live_;
};
const char* g_executable_path = nullptr;

[[nodiscard]] const char* watchdog_fault_name(
        iom::rocm_detail::SubmissionFault fault) {
    switch (fault) {
    case iom::rocm_detail::SubmissionFault::third_plane_launch:
        return "third_plane_launch";
    case iom::rocm_detail::SubmissionFault::event_record:
        return "event_record";
    default:
        return nullptr;
    }
}

[[nodiscard]] bool parse_watchdog_fault(
        int argc, char** argv,
        iom::rocm_detail::SubmissionFault& fault) {
    if (argc != 3
            || std::string_view(argv[1]) != "--iom-rocm-watchdog-fault") {
        return false;
    }
    const std::string_view name(argv[2]);
    if (name == "third_plane_launch") {
        fault = iom::rocm_detail::SubmissionFault::third_plane_launch;
    } else if (name == "event_record") {
        fault = iom::rocm_detail::SubmissionFault::event_record;
    } else {
        return false;
    }
    return true;
}


[[noreturn]] void run_watchdog_child(int argc, char** argv) {
    iom::rocm_detail::SubmissionFault fault =
            iom::rocm_detail::SubmissionFault::none;

    if (!parse_watchdog_fault(argc, argv, fault)) {
        _exit(1);
    }

    try {
        ReusingHipAllocator allocator;
        auto device = iom::make_rocm_device(0, allocator);
        const iom::TensorSpec spec{
                iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
        auto source = device->create_tensor(spec);
        auto destination = device->create_tensor(spec);
        auto queue = device->create_ops();
        const std::vector<std::byte> pattern(
                spec.logical_nbytes(), static_cast<std::byte>(0x3c));
        source->view().copy_from_host(pattern);
        destination->view().copy_from_host(pattern);

        iom::rocm_detail::inject_submission_fault_for_testing(fault);
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        queue->wait(token);
    } catch (const std::runtime_error&) {
        _exit(0);
    } catch (...) {
        _exit(1);
    }
    _exit(1);
}

void expect_bounded_wait_in_subprocess(
        iom::rocm_detail::SubmissionFault fault) {
    const char* fault_name = watchdog_fault_name(fault);
    REQUIRE(fault_name != nullptr);
    REQUIRE(g_executable_path != nullptr);

    char fault_option[] = "--iom-rocm-watchdog-fault";
    char* child_argv[] = {
            const_cast<char*>(g_executable_path), fault_option,
            const_cast<char*>(fault_name), nullptr};
    std::vector<char*> child_environment;
    for (char** entry = environ; *entry != nullptr; ++entry) {
        child_environment.push_back(*entry);
    }
    child_environment.push_back(
            const_cast<char*>("IOM_ROCM_WATCHDOG_CHILD=1"));
    child_environment.push_back(nullptr);

    pid_t child = 0;
    const int spawn_status = posix_spawn(
            &child, g_executable_path, nullptr, nullptr, child_argv,
            child_environment.data());
    REQUIRE_MESSAGE(spawn_status == 0, "posix_spawn failed");
    if (spawn_status != 0) {
        return;
    }

    int status = 0;
    const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        const pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) {
            break;
        }
        REQUIRE_MESSAGE(result != -1, "waitpid failed");
        if (result == -1) {
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            (void)kill(child, SIGKILL);
            (void)waitpid(child, &status, 0);
            std::exit(2);
        }
        const timespec pause{0, 10'000'000};
        (void)nanosleep(&pause, nullptr);
    }

    REQUIRE(WIFEXITED(status));
    CHECK_EQ(WEXITSTATUS(status), 0);
}

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

class HipStorageOracle final
        : public iom_conformance::AcceleratorStorageOracle {
public:
    void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        REQUIRE_EQ(encoded.size(), bytes);
        REQUIRE(hipMemcpy(
                        view.native_handle(), encoded.data(), bytes,
                        hipMemcpyHostToDevice)
                == hipSuccess);
        REQUIRE(hipDeviceSynchronize() == hipSuccess);
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        std::vector<std::byte> result(bytes);
        REQUIRE(hipDeviceSynchronize() == hipSuccess);
        REQUIRE(hipMemcpy(
                        result.data(), view.native_handle(), bytes,
                        hipMemcpyDeviceToHost)
                == hipSuccess);
        REQUIRE(hipDeviceSynchronize() == hipSuccess);
        return result;
    }
};
TEST_CASE("ROCm conformance: storage oracle identifies perturbed transfer map") {
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
    const std::span<const iom::DataType> one_type{
            kRocmLeafTypes.begin(), 1};
    iom_conformance::run_storage_and_transfer_conformance(
            devices, one_type, &gate);

    HipStorageOracle direct;
    iom_conformance::PermutingStorageOracle perturbed(
            direct, iom_conformance::swap_first_adjacent_slots);
    CHECK_FALSE(iom_conformance::run_storage_oracle_conformance(
            devices, one_type, perturbed, &gate, false, false));
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: storage oracle covers every leaf width and padded shape") {
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
    HipStorageOracle oracle;
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices, kRocmLeafTypes, oracle, &gate));
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

TEST_CASE("ROCm submission remains transactional across post-enqueue failures") {
    TrafficGate gate;
    HipAllocator candidate_allocator(gate);
    auto candidate = iom::make_rocm_device(0, candidate_allocator);
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    const iom::TensorSpec mismatch{
            iom::TensorShape{{3, 16, 17}}, iom::DataType::U8};
    auto source = candidate->create_tensor(spec);
    auto destination = candidate->create_tensor(spec);
    auto invalid_destination = candidate->create_tensor(mismatch);
    auto queue = candidate->create_ops();
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

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::event_create);
    CHECK_THROWS_AS(
            queue->copy(source->view(), destination->view()),
            std::runtime_error);
    const iom::oid second = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(second), 2);
    CHECK_NOTHROW(queue->wait(second));

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::third_plane_launch);
    iom::oid launch_failure = 0;
    CHECK_NOTHROW(
            launch_failure = queue->copy(
                    source->view(), destination->view()));
    CHECK_NE(launch_failure, 0);
    CHECK_EQ(iom_conformance::token_sequence(launch_failure), 3);
    expect_repeated_runtime_failure(*queue, launch_failure);
    expect_bounded_wait_in_subprocess(
            iom::rocm_detail::SubmissionFault::third_plane_launch);

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::event_record);
    const iom::oid record_failure =
            queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(record_failure), 4);
    expect_repeated_runtime_failure(*queue, record_failure);
    expect_bounded_wait_in_subprocess(
            iom::rocm_detail::SubmissionFault::event_record);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);

    {
        ReusingHipAllocator allocator;
        auto device = iom::make_rocm_device(0, allocator);
        auto canary_source = device->create_tensor(spec);
        auto canary_destination = device->create_tensor(spec);
        const std::vector<std::byte> storage_canary(
                spec.tiled_storage_nbytes(), static_cast<std::byte>(0xa5));
        REQUIRE(
                hipMemcpy(
                        canary_source->view().native_handle(),
                        storage_canary.data(), storage_canary.size(),
                        hipMemcpyHostToDevice)
                == hipSuccess);
        REQUIRE(
                hipMemcpy(
                        canary_destination->view().native_handle(),
                        storage_canary.data(), storage_canary.size(),
                        hipMemcpyHostToDevice)
                == hipSuccess);
        auto canary_queue = device->create_ops();
        iom::rocm_detail::inject_submission_fault_for_testing(
                iom::rocm_detail::SubmissionFault::third_plane_launch);
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
        CHECK(
                (fresh_source->view().native_handle() == source_address
                 || fresh_source->view().native_handle()
                         == destination_address));
        CHECK(
                (fresh_destination->view().native_handle() == source_address
                 || fresh_destination->view().native_handle()
                         == destination_address));
        CHECK_NE(
                fresh_source->view().native_handle(),
                fresh_destination->view().native_handle());
        std::vector<std::byte> fresh_source_bytes(spec.logical_nbytes());
        std::vector<std::byte> fresh_destination_bytes(spec.logical_nbytes());
        fresh_source->view().copy_to_host(fresh_source_bytes);
        fresh_destination->view().copy_to_host(fresh_destination_bytes);
        CHECK(std::all_of(
                fresh_source_bytes.begin(), fresh_source_bytes.end(),
                [](std::byte value) {
                    return value == static_cast<std::byte>(0xa5);
                }));
        CHECK(std::all_of(
                fresh_destination_bytes.begin(), fresh_destination_bytes.end(),
                [](std::byte value) {
                    return value == static_cast<std::byte>(0xa5);
                }));

        fresh_source.reset();
        fresh_destination.reset();
        canary_queue.reset();
        allocator.release_free();
        device.reset();
    }
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
}

int main(int argc, char** argv) {
    if (std::getenv("IOM_ROCM_WATCHDOG_CHILD") != nullptr) {
        run_watchdog_child(argc, argv);
    }
    g_executable_path = argv[0];
    doctest::Context context(argc, argv);
    return context.run();
}
