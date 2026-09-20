#include <doctest/doctest.h>

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
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
#include "backend/backend_conformance_rope.hpp"
#include "backend/backend_conformance_rope_contract.hpp"
#include "backend/backend_conformance_cache_append.hpp"
#include "backend/backend_conformance_embedding.hpp"
#include "backend/backend_conformance_linear.hpp"
#include "backend/backend_conformance_other.hpp"
#include "backend/backend_conformance_rmsnorm.hpp"
#include "backend/backend_conformance_add_gpu.hpp"
#include "backend/backend_conformance_model_loading.hpp"
#include "backend/backend_conformance_sdpa.hpp"
#include "backend/backend_conformance_token_selection.hpp"
#include "iom/rocm/device.hpp"
#include "rocm/copy.hpp"
#include "iom/gpu_algorithm.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
extern char** environ;


namespace {

// Every standard-GPU conformance fixture reserves this tensor-data arena:
// the largest concurrent set is three 8192x4096 F32 tensors plus one
// quarantined operand (512 MiB), and fixtures also keep a foreign device
// alive with the same budget.
constexpr std::size_t kConformanceArenaBytes = 640u * 1024 * 1024;
class HostAllocator final : public iom::Allocator {
public:
    explicit HostAllocator(const iom_conformance::TrafficGate& gate)
            : gate_(gate) {}

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
        auto device = iom::make_rocm_device(
                0, iom::DeviceMemoryConfig{1 * 1024 * 1024});
        const iom::TensorSpec spec{
                iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
        auto source = device->create_tensor(spec);
        auto destination = device->create_tensor(spec);
        auto queue = device->create_ops();
        const std::vector<std::byte> pattern(
                spec.logical_nbytes(), static_cast<std::byte>(0x3c));
        iom_conformance::copy_from_host(source->view(), pattern);
        iom_conformance::copy_from_host(destination->view(), pattern);

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

}  // namespace
TEST_CASE("Device::supported_data_types returns the per-backend 23-entry span") {
    iom_conformance::TrafficGate gate;
    const std::unique_ptr<iom::Device> candidate =
            iom::make_rocm_device(
                    0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    iom_conformance::require_standard_capabilities(
            candidate->supported_data_types());
}

TEST_CASE("ROCm conformance: storage and host transfers for every leaf type") {
    // Keep the CPU allocator large enough for the largest shared case.
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_storage_and_transfer_conformance(
            devices, candidate->supported_data_types(), &gate);
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
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    const std::span<const iom::DataType> one_type =
            candidate->supported_data_types().subspan(0, 1);
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
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    HipStorageOracle oracle;
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices, candidate->supported_data_types(), oracle, &gate));
    CHECK_FALSE(gate.armed());
}



TEST_CASE("ROCm conformance: asynchronous copies against the CPU reference") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    HipStorageOracle oracle;
    iom_conformance::run_async_copy_conformance(
            devices, candidate->supported_data_types(), &gate, &oracle);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: copy validation fails before writes and sequences") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(16 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_copy_error_conformance(
            devices, candidate->supported_data_types(), &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: cache append reference, admission, ordering, and lifetime") {
    iom_conformance::TrafficGate gate;
    HostAllocator reference_allocator{gate};
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    HipStorageOracle oracle;
    const iom_conformance::CacheAppendFaultSeam fault_seam{
            [](iom::DeviceOps&) {
                iom::rocm_detail::inject_submission_fault_for_testing(
                        iom::rocm_detail::SubmissionFault::third_plane_launch);
            }};
    const iom_conformance::CacheAppendConformanceConfig config{
            devices,
            candidate->supported_data_types(),
            &oracle,
            fault_seam,
            {},
            &gate};
    iom_conformance::run_cache_append_conformance(config);
    CHECK_FALSE(gate.armed());
}


TEST_CASE("ROCm cache append retained event failure recovers after reset") {
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec source_spec{
            iom::TensorShape{{1, 1, 1, 17}}, iom::DataType::F32};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{1, 1, 17, 17}}, iom::DataType::F32};
    auto source = device->create_tensor(source_spec);
    auto destination = device->create_tensor(destination_spec);
    const std::vector<std::byte> source_bytes =
            iom_conformance::encode_cache_append_logical(source_spec, 0x781);
    const std::vector<std::byte> before =
            iom_conformance::encode_logical(destination_spec, 0x782);
    iom_conformance::copy_from_host(source->view(), source_bytes);
    iom_conformance::copy_from_host(destination->view(), before);

    auto queue = device->create_ops();
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::event_record);
    const iom::oid failed =
            queue->cache_append(source->view(), destination->view(), 1);
    REQUIRE(iom::oid_is_token(failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, failed);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);

    auto recovered_destination = device->create_tensor(destination_spec);
    const std::vector<std::byte> recovery_before =
            iom_conformance::encode_logical(destination_spec, 0x783);
    iom_conformance::copy_from_host(
            recovered_destination->view(), recovery_before);
    auto recovery_queue = device->create_ops();
    const iom::oid recovered = recovery_queue->cache_append(
            source->view(), recovered_destination->view(), 1);
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(recovery_queue->wait(recovered));
    CHECK_NOTHROW(recovery_queue->wait(recovered));
    iom_conformance::require_logical_bytes(
            recovered_destination->view(),
            iom_conformance::cache_append_expected_logical(
                    source_spec, destination_spec, 1, source_bytes,
                    recovery_before),
            "ROCm cache append event-record recovery");
}

TEST_CASE("ROCm cache append retained launch failure recovers after reset") {
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec source_spec{
            iom::TensorShape{{2, 1, 1, 33}}, iom::DataType::F32};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{2, 1, 33, 33}}, iom::DataType::F32};
    auto source = device->create_tensor(source_spec);
    auto destination = device->create_tensor(destination_spec);
    const std::vector<std::byte> source_bytes =
            iom_conformance::encode_cache_append_logical(source_spec, 0x791);
    const std::vector<std::byte> before =
            iom_conformance::encode_logical(destination_spec, 0x792);
    iom_conformance::copy_from_host(source->view(), source_bytes);
    iom_conformance::copy_from_host(destination->view(), before);

    auto queue = device->create_ops();
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::third_plane_launch);
    const iom::oid failed =
            queue->cache_append(source->view(), destination->view(), 15);
    REQUIRE(iom::oid_is_token(failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, failed);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);

    auto recovered_destination = device->create_tensor(destination_spec);
    const std::vector<std::byte> recovery_before =
            iom_conformance::encode_logical(destination_spec, 0x793);
    iom_conformance::copy_from_host(
            recovered_destination->view(), recovery_before);
    auto recovery_queue = device->create_ops();
    const iom::oid recovered = recovery_queue->cache_append(
            source->view(), recovered_destination->view(), 15);
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(recovery_queue->wait(recovered));
    CHECK_NOTHROW(recovery_queue->wait(recovered));
    iom_conformance::require_logical_bytes(
            recovered_destination->view(),
            iom_conformance::cache_append_expected_logical(
                    source_spec, destination_spec, 15, source_bytes,
                    recovery_before),
            "ROCm cache append launch-failure recovery");
}

TEST_CASE("ROCm cache append queue teardown quarantines pending operands") {
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec source_spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{3, 64, 16}}, iom::DataType::U8};
    auto source = device->create_tensor(source_spec);
    auto destination = device->create_tensor(destination_spec);
    iom_conformance::copy_from_host(
            source->view(),
            iom_conformance::encode_cache_append_logical(source_spec, 0x7a1));
    iom_conformance::copy_from_host(
            destination->view(),
            iom_conformance::encode_logical(destination_spec, 0x7a2));
    const void* source_address = source->view().native_handle();
    const void* destination_address = destination->view().native_handle();
    {
        auto queue = device->create_ops();
        for (int i = 0; i < 4; ++i) {
            CHECK(iom::oid_is_token(
                    queue->cache_append(
                            source->view(), destination->view(), 1)));
        }
        iom::rocm_detail::inject_submission_fault_for_testing(
                iom::rocm_detail::SubmissionFault::third_plane_launch);
        const iom::oid failure =
                queue->cache_append(source->view(), destination->view(), 1);
        CHECK(iom::oid_is_token(failure));
        queue.reset();
    }
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
    source.reset();
    destination.reset();
    auto fresh_source = device->create_tensor(source_spec);
    auto fresh_destination = device->create_tensor(destination_spec);
    CHECK_NE(fresh_source->view().native_handle(), source_address);
    CHECK_NE(fresh_source->view().native_handle(), destination_address);
    CHECK_NE(fresh_destination->view().native_handle(), source_address);
    CHECK_NE(fresh_destination->view().native_handle(), destination_address);
}

TEST_CASE("ROCm cache append retains source through pre-wait destruction") {
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec source_spec{
            iom::TensorShape{{32, 1, 8192, 128}}, iom::DataType::U8};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{32, 1, 16384, 128}}, iom::DataType::U8};
    auto source = device->create_tensor(source_spec);
    auto destination = device->create_tensor(destination_spec);
    const std::vector<std::byte> source_bytes =
            iom_conformance::encode_cache_append_logical(source_spec, 0x7b1);
    const std::vector<std::byte> destination_before =
            iom_conformance::encode_logical(destination_spec, 0x7b2);
    iom_conformance::copy_from_host(source->view(), source_bytes);
    iom_conformance::copy_from_host(
            destination->view(), destination_before);

    auto queue = device->create_ops();
    const iom::oid token =
            queue->cache_append(source->view(), destination->view(), 17);
    REQUIRE(iom::oid_is_token(token));
    source.reset();
    auto fresh_source = device->create_tensor(source_spec);
    iom_conformance::copy_from_host(
            fresh_source->view(),
            iom_conformance::encode_cache_append_logical(
                    source_spec, 0x7b3));
    CHECK_NOTHROW(queue->wait(token));
    iom_conformance::require_logical_bytes(
            destination->view(),
            iom_conformance::cache_append_expected_logical(
                    source_spec, destination_spec, 17, source_bytes,
                    destination_before),
            "ROCm cache append source owner retention");
    fresh_source.reset();
}
TEST_CASE("ROCm copy reservation failures roll back before native work") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(16 * 1024 * 1024);
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::U8};
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    const std::vector<std::byte> source_bytes(
            spec.logical_nbytes(), static_cast<std::byte>(0x11));
    const std::vector<std::byte> destination_bytes(
            spec.logical_nbytes(), static_cast<std::byte>(0x22));
    iom_conformance::copy_from_host(source->view(), source_bytes);
    iom_conformance::copy_from_host(destination->view(), destination_bytes);
    auto queue = device->create_ops();

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::registration);
    CHECK_EQ(
            queue->copy(source->view(), destination->view()),
            iom::to_oid(iom::OidError::ResourceExhausted));
    std::vector<std::byte> observed(spec.logical_nbytes());
    iom_conformance::copy_to_host(destination->view(), observed);
    CHECK(observed == destination_bytes);

    const iom::oid first = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(first), 1);
    CHECK_NOTHROW(queue->wait(first));

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::outcome_insertion);
    CHECK_EQ(
            queue->copy(source->view(), destination->view()),
            iom::to_oid(iom::OidError::ResourceExhausted));
    const iom::oid second = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(second), 2);
    CHECK_NOTHROW(queue->wait(second));
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: transfer failures keep metadata and ownership") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(16 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_transfer_error_conformance(
            devices, candidate->supported_data_types(), &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: deferred queue lifetime and stability") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    iom_conformance::run_lifetime_conformance(
            *candidate, candidate->supported_data_types(), &gate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: sub-byte odd-length host reads stay within the staged atomic word") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
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
        iom_conformance::copy_from_host(tensor->view(), expected);
        iom_conformance::require_logical_bytes(
                tensor->view(), expected, "odd-length sub-byte host read");
    }
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: compute methods reject capability without submitting") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    iom_conformance::run_compute_capability_conformance(
            *candidate, candidate->supported_data_types(), &gate, "ROCm",
            true, true);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: shared SDPA admission and unsupported matrix") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::SdpaConformanceConfig config{
            {*reference, *candidate, *foreign}, {}, false, &gate};
    iom_conformance::run_sdpa_conformance(config);
    CHECK_FALSE(gate.armed());
}

// ROCm implements the complete embedding payload/index matrix through the
// shared raw-word gather and reports the accelerator control workspace.
constexpr iom_conformance::EmbeddingDeclaration kRocmEmbeddingDeclaration{
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom::WorkspaceRequirements{32, 32}};

TEST_CASE("ROCm conformance: embedding lookup reference, admission, and lifetime") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    HipStorageOracle oracle;
    iom_conformance::run_embedding_conformance(
            devices, kRocmEmbeddingDeclaration, &gate, &oracle);
    CHECK_FALSE(gate.armed());
}

// ROCm's declared linear expectation: the complete 21-leaf applicable matrix
// through the twenty-leaf scalar path plus the native BF16 specialization, the
// frozen `{0, 1}` scratch path for the scalar leaves, and the exact aligned
// `A32(P*pad16(R)*pad16(I)*2) + A32(P*pad16(R)*pad16(O)*2)` BF16 path. Both
// paths are implemented: the shared raw-word scalar kernel executes the
// integer modulo-2^N dot and the `+0`-started FP32/FP64 fused recurrence, and
// the native BF16 specialization executes direct GFX12 wave32 WMMA with FP32
// accumulation and one RNE BF16 scatter on a device whose checked facility
// permits it, so `implemented_leaves` is the complete 21-leaf span.
const iom_conformance::LinearDeclaration kRocmLinearDeclaration{
        iom_conformance::kLinearLeafSpan,
        iom_conformance::kLinearScalarLeafSpan,
        iom_conformance::kLinearNativeBf16Span,
        iom_conformance::kLinearLeafSpan,
        {},
        iom_conformance::LinearWorkspacePath::zero,
        iom_conformance::LinearWorkspacePath::rocm_bf16};

// Native BF16 availability of one ROCm ordinal, stated independently of the
// port: the checked route is the GFX12 wave32 BF16-to-FP32 WMMA facility, so
// only the GFX12 ASIC ids carrying that 128-bit wave32 operand form on a
// wave32 device are supported. Every other ordinal, including the installed
// gfx1036, must report `Unsupported` for the leaf rather than fail later.
[[nodiscard]] bool rocm_native_bf16_ordinal(int ordinal) {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, ordinal) != hipSuccess) {
        return false;
    }
    if (properties.warpSize != 32) {
        return false;
    }
    const std::string_view architecture{properties.gcnArchName};
    return architecture.starts_with("gfx1200")
            || architecture.starts_with("gfx1201");
}

TEST_CASE("ROCm conformance: native BF16 linear caller workspace and WMMA launch evidence") {
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    // The checked host exposes the GFX12 device at ordinal 0. An unavailable
    // facility is a configuration failure, never a silently skipped case.
    REQUIRE(rocm_native_bf16_ordinal(0));

    const iom::LinearOutputLayout ordinary = iom::LinearOutputLayout::ordinary;
    const iom::LinearOutputLayout head_planar =
            iom::LinearOutputLayout::head_planar;
    auto x = candidate->create_tensor(iom_conformance::linear_spec(
            {19, 3}, iom::DataType::BF16));
    auto w = candidate->create_tensor(iom_conformance::linear_spec(
            {10, 3}, iom::DataType::BF16));
    iom_conformance::copy_from_host(
            x->view(),
            iom_conformance::linear_logical_image(
                    iom::DataType::BF16, 19 * 3, 0x5A));
    iom_conformance::copy_from_host(
            w->view(),
            iom_conformance::linear_logical_image(
                    iom::DataType::BF16, 10 * 3, 0xC3));
    auto queue = candidate->create_ops();

    // The frozen four row runs through the real queue: the query reports the
    // exact aligned sum, the accepted submission leases the range, and the
    // execution reaches the native kernel and never the scalar recurrence.
    for (const std::size_t rows : {1u, 15u, 16u, 17u}) {
        CAPTURE(rows);
        auto out = candidate->create_tensor(iom_conformance::linear_spec(
                {rows, 10}, iom::DataType::BF16));
        const iom::WorkspaceRequirements requirement =
                queue->linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, rows, ordinary,
                        1, 10);
        const iom::WorkspaceRequirements expected =
                iom_conformance::linear_expected_workspace(
                        kRocmLinearDeclaration, iom::DataType::BF16,
                        iom_conformance::LinearShape{1, rows, 3, 10});
        CHECK_EQ(requirement.bytes, expected.bytes);
        CHECK_EQ(requirement.alignment, std::size_t{32});
        auto workspace = candidate->create_workspace(requirement.bytes);
        const std::size_t native_before =
                iom::rocm_detail::linear_native_bf16_launch_count_for_testing
                        .load();
        const std::size_t scalar_before =
                iom::rocm_detail::linear_scalar_launch_count_for_testing
                        .load();
        const iom::oid token = queue->linear(
                x->view(), w->view(), out->view(), 2, rows, ordinary, 1, 10,
                workspace->view().subrange(0, requirement.bytes));
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        CHECK_EQ(
                iom::rocm_detail::linear_native_bf16_launch_count_for_testing
                        .load(),
                native_before + 1);
        CHECK_EQ(
                iom::rocm_detail::linear_scalar_launch_count_for_testing
                        .load(),
                scalar_before);
        // Proven completion released the lease, so the same range is reusable.
        const iom::oid reused = queue->linear(
                x->view(), w->view(), out->view(), 2, rows, ordinary, 1, 10,
                workspace->view().subrange(0, requirement.bytes));
        REQUIRE(iom::oid_is_token(reused));
        CHECK_NOTHROW(queue->wait(reused));
    }
    // Head-planar shares the product and inserts the head axis, so it uses the
    // same native path with `O = H*D` as the packed column extent.
    {
        auto out = candidate->create_tensor(iom_conformance::linear_spec(
                {2, 16, 5}, iom::DataType::BF16));
        const iom::WorkspaceRequirements requirement =
                queue->linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 16, head_planar,
                        2, 5);
        const iom::WorkspaceRequirements expected =
                iom_conformance::linear_expected_workspace(
                        kRocmLinearDeclaration, iom::DataType::BF16,
                        iom_conformance::LinearShape{1, 16, 3, 10});
        CHECK_EQ(requirement.bytes, expected.bytes);
        CHECK_EQ(requirement.alignment, std::size_t{32});
        auto workspace = candidate->create_workspace(requirement.bytes);
        const std::size_t native_before =
                iom::rocm_detail::linear_native_bf16_launch_count_for_testing
                        .load();
        const iom::oid token = queue->linear(
                x->view(), w->view(), out->view(), 2, 16, head_planar, 2, 5,
                workspace->view().subrange(0, requirement.bytes));
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        CHECK_EQ(
                iom::rocm_detail::linear_native_bf16_launch_count_for_testing
                        .load(),
                native_before + 1);
    }
    // Every unusable caller range is rejected before any effect, and the
    // accepted range still works afterwards.
    {
        auto out = candidate->create_tensor(iom_conformance::linear_spec(
                {17, 10}, iom::DataType::BF16));
        const iom::WorkspaceRequirements requirement =
                queue->linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        10);
        REQUIRE(requirement.bytes > 32);
        const iom::oid invalid =
                iom::to_oid(iom::OidError::InvalidArgument);
        auto workspace = candidate->create_workspace(requirement.bytes);
        auto foreign_workspace =
                foreign->create_workspace(requirement.bytes);
        const auto submit = [&](iom::RawWorkspaceView scratch) {
            return queue->linear(
                    x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 10,
                    scratch);
        };
        CHECK_EQ(submit(iom::RawWorkspaceView{}), invalid);
        CHECK_EQ(
                submit(workspace->view().subrange(
                        0, requirement.bytes - 32)),
                invalid);
        CHECK_EQ(
                submit(foreign_workspace->view().subrange(
                        0, requirement.bytes)),
                invalid);
        {
            const iom_conformance::LinearConformanceWorkspace misaligned(
                    *candidate, reinterpret_cast<void*>(0x5401),
                    requirement.bytes + 64);
            CHECK_EQ(submit(misaligned.view()), invalid);
        }
        {
            const iom_conformance::LinearConformanceWorkspace overlapping(
                    *candidate, x->view().native_handle(),
                    requirement.bytes + 64);
            CHECK_EQ(submit(overlapping.view()), invalid);
        }
        const iom::RawWorkspaceView dead =
                iom_conformance::linear_dead_workspace_view(
                        *candidate, reinterpret_cast<void*>(0x5600),
                        requirement.bytes + 64);
        CHECK_EQ(submit(dead), invalid);
        const iom::oid accepted = submit(
                workspace->view().subrange(0, requirement.bytes));
        REQUIRE(iom::oid_is_token(accepted));
        CHECK_NOTHROW(queue->wait(accepted));
    }
}

TEST_CASE("ROCm conformance: native BF16 head-planar crossing factorization matches the reference") {
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    REQUIRE(rocm_native_bf16_ordinal(0));
    auto queue = candidate->create_ops();
    constexpr std::size_t kSourceRows = 19;
    constexpr std::size_t kInner = 3;
    constexpr std::size_t kStartRow = 2;
    constexpr std::size_t kRows = 17;
    // Both factorizations place a head's logical columns across a packed
    // 16-column boundary, so the boundary-crossing copy runs on the device and
    // its accepted OID must have reached the native kernel.
    for (const std::array<std::size_t, 2>& crossing :
         {std::array<std::size_t, 2>{2, 9},
          std::array<std::size_t, 2>{4, 5}}) {
        const std::size_t heads = crossing[0];
        const std::size_t head_dim = crossing[1];
        const std::size_t outer = heads * head_dim;
        CAPTURE(heads);
        CAPTURE(head_dim);
        auto x = candidate->create_tensor(iom_conformance::linear_spec(
                {kSourceRows, kInner}, iom::DataType::BF16));
        auto w = candidate->create_tensor(iom_conformance::linear_spec(
                {outer, kInner}, iom::DataType::BF16));
        auto out = candidate->create_tensor(iom_conformance::linear_spec(
                {heads, kRows, head_dim}, iom::DataType::BF16));
        const std::vector<std::byte> x_bytes =
                iom_conformance::linear_logical_image(
                        iom::DataType::BF16, kSourceRows * kInner, 0x71);
        const std::vector<std::byte> w_bytes =
                iom_conformance::linear_logical_image(
                        iom::DataType::BF16, outer * kInner, 0x93);
        const std::vector<std::byte> out_poison =
                iom_conformance::linear_logical_image(
                        iom::DataType::BF16, heads * kRows * head_dim, 0xB5);
        iom_conformance::copy_from_host(x->view(), x_bytes);
        iom_conformance::copy_from_host(w->view(), w_bytes);
        iom_conformance::copy_from_host(out->view(), out_poison);
        const iom_conformance::LinearRawImage x_image =
                iom_conformance::linear_padded_image(
                        iom::DataType::BF16, 1,
                        iom_conformance::linear_pad16(kSourceRows),
                        iom_conformance::linear_pad16(kInner), kSourceRows,
                        kInner, x_bytes, 0x0F0Full);
        const iom_conformance::LinearRawImage w_image =
                iom_conformance::linear_padded_image(
                        iom::DataType::BF16, 1,
                        iom_conformance::linear_pad16(outer),
                        iom_conformance::linear_pad16(kInner), outer, kInner,
                        w_bytes, 0xF0F0ull);
        const iom_conformance::LinearOracleRequest request{
                iom::DataType::BF16, 1, kSourceRows, kInner, outer, kStartRow,
                kRows, heads, head_dim, iom::LinearOutputLayout::head_planar};
        const iom_conformance::LinearReference reference =
                iom_conformance::linear_reference(request, x_image, w_image);
        const iom::WorkspaceRequirements requirement =
                queue->linear_workspace_requirements(
                        x->view(), w->view(), out->view(), kStartRow, kRows,
                        iom::LinearOutputLayout::head_planar, heads, head_dim);
        CHECK_EQ(
                requirement.bytes,
                iom_conformance::linear_expected_workspace(
                        kRocmLinearDeclaration, iom::DataType::BF16,
                        iom_conformance::LinearShape{1, kRows, kInner, outer})
                        .bytes);
        auto workspace = candidate->create_workspace(requirement.bytes);
        const std::size_t native_before =
                iom::rocm_detail::linear_native_bf16_launch_count_for_testing
                        .load();
        const iom::oid token = queue->linear(
                x->view(), w->view(), out->view(), kStartRow, kRows,
                iom::LinearOutputLayout::head_planar, heads, head_dim,
                workspace->view().subrange(0, requirement.bytes));
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        CHECK_EQ(
                iom::rocm_detail::linear_native_bf16_launch_count_for_testing
                        .load(),
                native_before + 1);
        const std::vector<std::byte> observed =
                iom_conformance::read_logical(out->view());
        const std::optional<std::string> mismatch =
                iom_conformance::linear_compare_image(
                        request, observed, reference,
                        "native BF16 crossing head stride");
        REQUIRE_MESSAGE(!mismatch.has_value(), mismatch.value_or(std::string{}));
    }
}

TEST_CASE("ROCm conformance: native BF16 linear capability follows the checked device") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count >= 1);
    const iom::LinearOutputLayout ordinary = iom::LinearOutputLayout::ordinary;
    std::size_t supported = 0;
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
        hipDeviceProp_t properties{};
        REQUIRE(hipGetDeviceProperties(&properties, ordinal) == hipSuccess);
        CAPTURE(ordinal);
        CAPTURE(properties.gcnArchName);
        const bool expected = rocm_native_bf16_ordinal(ordinal);
        auto device = iom::make_rocm_device(
                static_cast<std::uint32_t>(ordinal),
                iom::DeviceMemoryConfig{8u * 1024 * 1024});
        auto queue = device->create_ops();
        auto x = device->create_tensor(iom_conformance::linear_spec(
                {17, 3}, iom::DataType::BF16));
        auto w = device->create_tensor(iom_conformance::linear_spec(
                {10, 3}, iom::DataType::BF16));
        auto out = device->create_tensor(iom_conformance::linear_spec(
                {15, 10}, iom::DataType::BF16));
        if (expected) {
            ++supported;
            const iom::WorkspaceRequirements requirement =
                    queue->linear_workspace_requirements(
                            x->view(), w->view(), out->view(), 2, 15,
                            ordinary, 1, 10);
            CHECK(requirement.bytes > 0);
            CHECK_EQ(requirement.alignment, std::size_t{32});
            continue;
        }
        // A device without the proven facility rejects the leaf as a
        // capability, before any token, registration, or scratch inspection,
        // while the scalar leaves stay available on that same device.
        CHECK_THROWS_AS(
                (void)queue->linear_workspace_requirements(
                        x->view(), w->view(), out->view(), 2, 15, ordinary, 1,
                        10),
                std::runtime_error);
        CHECK_EQ(
                queue->linear(
                        x->view(), w->view(), out->view(), 2, 15, ordinary, 1,
                        10),
                iom::to_oid(iom::OidError::Unsupported));
        auto scalar_x = device->create_tensor(iom_conformance::linear_spec(
                {17, 3}, iom::DataType::F16));
        auto scalar_w = device->create_tensor(iom_conformance::linear_spec(
                {10, 3}, iom::DataType::F16));
        auto scalar_out = device->create_tensor(iom_conformance::linear_spec(
                {15, 10}, iom::DataType::F16));
        const iom::WorkspaceRequirements scalar =
                queue->linear_workspace_requirements(
                        scalar_x->view(), scalar_w->view(),
                        scalar_out->view(), 2, 15, ordinary, 1, 10);
        CHECK_EQ(scalar.bytes, std::size_t{0});
    }
    CHECK_GE(supported, std::size_t{1});
}

TEST_CASE("ROCm conformance: linear projection reference, admission, and lifetime") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    HipStorageOracle oracle;
    iom_conformance::run_linear_conformance(
            devices, kRocmLinearDeclaration, &gate, &oracle);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm embedding status transfer failure retires safely") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec table_spec{
            iom::TensorShape{{4, 8}}, iom::DataType::U8};
    const iom::TensorSpec index_spec{
            iom::TensorShape{{1, 8}}, iom::DataType::U32};
    const iom::TensorSpec output_spec{
            iom::TensorShape{{8, 8}}, iom::DataType::U8};
    auto table = candidate->create_tensor(table_spec);
    auto indices = candidate->create_tensor(index_spec);
    auto output = candidate->create_tensor(output_spec);
    iom_conformance::copy_from_host(
            table->view(),
            std::vector<std::byte>(table_spec.logical_nbytes()));
    iom_conformance::copy_from_host(
            indices->view(),
            std::vector<std::byte>(index_spec.logical_nbytes()));
    auto workspace = candidate->create_workspace(32);
    auto queue = candidate->create_ops();

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::embedding_status_copy);
    const iom::oid failed = queue->embedding(
            table->view(), indices->view(), output->view(), workspace->view());
    REQUIRE(iom::oid_is_token(failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, failed);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);

    const iom::oid recovered = queue->embedding(
            table->view(), indices->view(), output->view(), workspace->view());
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(queue->wait(recovered));

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::third_plane_launch);
    const iom::oid native_failed = queue->embedding(
            table->view(), indices->view(), output->view(), workspace->view());
    REQUIRE(iom::oid_is_token(native_failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, native_failed);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: RMSNorm reference, admission, and lifetime") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    HipStorageOracle oracle;
    iom_conformance::RmsNormConformanceConfig config{
            devices, iom_conformance::kRmsNormAllLeafSpan, &gate, &oracle};
    // The ROCm seam is the same shared-GPU-dispatch step as CUDA's: the HIP
    // RMSNorm launch is followed by `Policy::record_event` on the queue's
    // existing stream, which is exactly what `SubmissionFault::event_record`
    // fails, so the armed fault is consumed by the real RMSNorm submission
    // after acceptance. Consumption is proven behaviorally by the shared
    // scenario because the counted fault has no consumption accessor.
    config.native_failure = iom_conformance::RmsNormNativeFailureSeam{
            [] {
                iom::rocm_detail::inject_submission_fault_for_testing(
                        iom::rocm_detail::SubmissionFault::event_record);
            },
            [] {
                iom::rocm_detail::inject_submission_fault_for_testing(
                        iom::rocm_detail::SubmissionFault::none);
            },
            "rocm_detail::SubmissionFault::event_record",
            {}};
    iom_conformance::run_rmsnorm_conformance(config);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: RoPE reference, admission, and lifetime") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    HipStorageOracle oracle;
    iom_conformance::rope_reference::run_rope_conformance(
            *candidate,
            iom_conformance::rope_reference::kRopeRocmExpectedSupported,
            oracle);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: greedy token selection shared matrix and lifetime") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    HipStorageOracle oracle;
    // The selector's only device work after `wait(producer)` is
    // `TensorView::copy_to_host` for the logical BF16 bytes: the shared staged
    // transfer gathers device-to-staging through the HIP plane kernel and
    // `Policy::after_copy_plane_launch(2)` consumes this counted fault after
    // that launch, so the armed failure is the real logical-byte transfer
    // failure and not an unrelated later submission.
    const iom_conformance::TokenSelectionNativeFailureSeam native_failure{
            [] {
                iom::rocm_detail::inject_submission_fault_for_testing(
                        iom::rocm_detail::SubmissionFault::third_plane_launch);
            },
            [] {
                iom::rocm_detail::inject_submission_fault_for_testing(
                        iom::rocm_detail::SubmissionFault::none);
            },
            "rocm_detail::SubmissionFault::third_plane_launch",
            {}};
    const iom_conformance::TokenSelectionConformanceConfig config{
            devices,
            candidate->supported_data_types(),
            &gate,
            &oracle,
            iom::WorkspaceRequirements{
                    iom::gpu_algorithm::compute_staging_size(17 * sizeof(std::uint16_t)),
                    32},
            native_failure};
    iom_conformance::run_token_selection_conformance(config);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm conformance: full shared suite") {
    iom_conformance::TrafficGate gate;

    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    HipStorageOracle oracle;
    const iom_conformance::RopeContractConformanceConfig rope_contract{
            devices,
            iom_conformance::rope_reference::kRopeRocmExpectedSupported,
            &gate,
            iom_conformance::RopeNativeFailureSeam{
                    [] {
                        iom::rocm_detail::inject_submission_fault_for_testing(
                                iom::rocm_detail::SubmissionFault::event_record);
                    },
                    [] {
                        iom::rocm_detail::inject_submission_fault_for_testing(
                                iom::rocm_detail::SubmissionFault::none);
                    },
                    "rocm_detail::SubmissionFault::event_record",
                    {}}};
    iom_conformance::run_backend_conformance(
            devices, candidate->supported_data_types().subspan(0, 1),
            &gate, &oracle, true, true, &rope_contract);
    CHECK_FALSE(gate.armed());
}

// The ROCm instantiation of the shared, CPU-first model-loading scenario: the
// same backend-neutral case the CPU reference registers, on this driver's own
// devices. Each synthetic checkpoint is loaded through the production
// configuration and mapped-source API, realized on the CPU reference device and
// on the selected ROCm device, and read back bit-for-bit against the fixture's
// independent role bytes. There is no ROCm loader port and no per-backend
// expectation. BF16 is mandatory rather than a skip condition, and the real
// device's host transfers report positive staging scratch, so the case
// provisions one caller-owned workspace from the ROCm reserve and has the empty
// default scratch refused before the first copied role. The diagnostic
// `HipStorageOracle` and the queued `inject_submission_fault_for_testing` seams
// are deliberately absent: the production loader is synchronous, so neither a
// storage oracle nor a queued HIP fault proves a loader failure.
TEST_CASE("ROCm model loading realizes every published weight role of each synthetic checkpoint") {
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const std::span<const iom::DataType> supported =
            candidate->supported_data_types();
    REQUIRE(std::find(supported.begin(), supported.end(),
                      iom::DataType::BF16) != supported.end());
    const iom_conformance::ConformanceDevices devices{
            *reference, *candidate, *foreign};
    iom_conformance::run_model_loading_conformance(devices);
}

// Opt-in real-checkpoint loading, compiled only with
// `IOM_TEST_REAL_MODEL_LOADING=ON`. The caller-selected arena replaces the
// synthetic conformance reserve, which cannot hold the full official reference;
// a missing, invalid, or insufficient `IOM_TEST_MODEL_ARENA_BYTES` fails this
// case rather than skipping it or constraining the run silently.
#ifdef IOM_TEST_REAL_MODEL_LOADING
TEST_CASE("ROCm real model loading") {
    const std::unique_ptr<iom::Device> candidate = iom::make_rocm_device(
            0, iom_conformance::real_model_memory_config());
    iom_conformance::run_real_model_loading(*candidate);
}
#endif
TEST_CASE("ROCm binary conformance: ADD MUL SUB DIV real queue") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    for (const auto operation : {
            iom_conformance::BinaryOperation::add,
            iom_conformance::BinaryOperation::mul,
            iom_conformance::BinaryOperation::sub,
            iom_conformance::BinaryOperation::div}) {
        iom_conformance::run_gpu_eltwise_conformance(
                *candidate, operation);
        iom_conformance::run_gpu_eltwise_mapping_conformance(
                *candidate, operation);
        iom_conformance::run_gpu_exact_alias_conformance(
                *candidate, operation);
    }
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm ADD accepts every low-width leaf against the oracle") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    iom_conformance::run_gpu_eltwise_conformance(
            *candidate, iom_conformance::BinaryOperation::add);
    iom_conformance::run_binary_rank_boundary_conformance(*candidate);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm ADD wide dtypes and boundary values against the oracle") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    iom_conformance::run_gpu_eltwise_conformance(
            *candidate, iom_conformance::BinaryOperation::add);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm ADD broadcast, transform, tail, and exact alias mapping") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    iom_conformance::run_gpu_eltwise_mapping_conformance(
            *candidate, iom_conformance::BinaryOperation::add);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm ADD retained launch failure keeps owners reusable") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::U8};
    auto lhs = candidate->create_tensor(spec);
    auto out = candidate->create_tensor(spec);
    std::vector<std::byte> pattern(spec.logical_nbytes());
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = std::byte{static_cast<unsigned char>(i * 3)};
    }
    iom_conformance::copy_from_host(lhs->view(), pattern);
    auto queue = candidate->create_ops();

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::third_plane_launch);
    const iom::oid failed = queue->add(lhs->view(), lhs->view(), out->view());
    CHECK(iom::oid_is_token(failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, failed);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);

    const iom::oid recovered =
            queue->add(lhs->view(), lhs->view(), out->view());
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(queue->wait(recovered));
    std::vector<std::byte> observed(spec.logical_nbytes());
    iom_conformance::copy_to_host(out->view(), observed);
    for (std::size_t i = 0; i < observed.size(); ++i) {
        CHECK_EQ(
                static_cast<unsigned>(observed[i]),
                static_cast<unsigned>(pattern[i]) * 2 & 0xffu);
    }
}

TEST_CASE("ROCm submission remains transactional across post-enqueue failures") {
    iom_conformance::TrafficGate gate;
    auto candidate = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
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

    iom_conformance::copy_from_host(source->view(), logical_pattern);
    iom_conformance::copy_from_host(destination->view(), logical_pattern);

    CHECK_EQ(queue->copy(source->view(), invalid_destination->view()), iom::to_oid(iom::OidError::InvalidArgument));

    const iom::oid first = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(first), 1);
    CHECK_NOTHROW(queue->wait(first));

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::event_create);
    CHECK_EQ(queue->copy(source->view(), destination->view()), iom::to_oid(iom::OidError::DeviceError));
    const iom::oid second = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(second), 2);
    CHECK_NOTHROW(queue->wait(second));

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::third_plane_launch);
    iom::oid launch_failure = 0;
    launch_failure = queue->copy(source->view(), destination->view());
    CHECK(iom::oid_is_token(launch_failure));
    CHECK_NE(launch_failure, 0);
    CHECK_EQ(iom_conformance::token_sequence(launch_failure), 3);
    iom_conformance::expect_repeated_runtime_failure(*queue, launch_failure);
    expect_bounded_wait_in_subprocess(
            iom::rocm_detail::SubmissionFault::third_plane_launch);

    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::event_record);
    const iom::oid record_failure =
            queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(record_failure), 4);
    iom_conformance::expect_repeated_runtime_failure(*queue, record_failure);
    expect_bounded_wait_in_subprocess(
            iom::rocm_detail::SubmissionFault::event_record);
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);

    // Failed work remains quarantined until device teardown; the allocator
    // must not recycle either operand while its failed entries are retained.
    {
        auto device = iom::make_rocm_device(
                0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
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
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
}

TEST_CASE("ROCm queue destruction fences pending copies") {
    iom_conformance::TrafficGate gate;
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    const std::vector<std::byte> pattern(
            spec.logical_nbytes(), static_cast<std::byte>(0x5a));
    iom_conformance::copy_from_host(source->view(), pattern);
    iom_conformance::copy_from_host(destination->view(), pattern);

    {
        auto queue = device->create_ops();
        for (int i = 0; i < 32; ++i) {
            CHECK(iom::oid_is_token(queue->copy(source->view(), destination->view())));
        }
        iom::rocm_detail::inject_submission_fault_for_testing(
                iom::rocm_detail::SubmissionFault::third_plane_launch);
        iom::oid failure = 0;
        failure = queue->copy(source->view(), destination->view());
        CHECK(iom::oid_is_token(failure));
        CHECK_NE(failure, 0);
        for (int i = 0; i < 32; ++i) {
            CHECK(iom::oid_is_token(queue->copy(source->view(), destination->view())));
        }
        queue.reset();
    }
    iom::rocm_detail::inject_submission_fault_for_testing(
            iom::rocm_detail::SubmissionFault::none);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("ROCm pre-wait source destruction keeps storage until completion") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    // Large enough that the queued copy kernel is still executing when the
    // host destroys the source; the assertions below hold under every
    // worker/interleaving outcome: the copy completes from storage that was
    // either quarantined or released only after its event fired, so waiting
    // the token always yields correct destination bytes and the allocator
    // never sees a double free. The deterministic quarantine/no-reuse
    // contract is pinned by the ring smoke tests and by the
    // queue-teardown case below.
    const iom::TensorSpec spec{
            iom::TensorShape{{8192, 4096}}, iom::DataType::F32};
    const std::vector<std::byte> expected =
            iom_conformance::encode_standard_tiled_storage(spec);
    const std::vector<std::byte> empty(
            spec.tiled_storage_nbytes(), std::byte{0});

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    HipStorageOracle oracle;
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
    // block is either quarantined or released only after the copy finished
    // reading it; in both cases the fresh allocation is safe and the copy
    // still produces the expected bytes.
    source.reset();
    auto fresh = device->create_tensor(spec);

    CHECK_NOTHROW(queue->wait(token));
    oracle.set_owner_spec(spec);
    CHECK_EQ(oracle.observe(destination->view()), expected);

    fresh.reset();
    queue.reset();
    destination.reset();
    device.reset();
}

TEST_CASE("ROCm shared-operand copies release storage after both complete") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    const iom::TensorSpec spec{
            iom::TensorShape{{8192, 4096}}, iom::DataType::F32};
    const std::vector<std::byte> expected =
            iom_conformance::encode_standard_tiled_storage(spec);
    const std::vector<std::byte> empty(
            spec.tiled_storage_nbytes(), std::byte{0});

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    HipStorageOracle oracle;
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

TEST_CASE("ROCm operands destroyed after queue teardown remain quarantined") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    // Queue destruction invalidates every outstanding registry entry up
    // front, so the operand fences provably report failure when the operands
    // are destroyed afterwards: the data arena cannot recycle either range
    // until device teardown, deterministically, with no timing dependence on
    // the worker or the GPU.
    const iom::TensorSpec spec{
            iom::TensorShape{{4096, 2048}}, iom::DataType::F32};

    auto device = iom::make_rocm_device(
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

TEST_CASE("ROCm conformance: rank boundary covers rank-eight owners and rejects rank nine") {
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);
    REQUIRE(device_count > 0);
    iom_conformance::TrafficGate gate;
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    HipStorageOracle oracle;
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

namespace {

// File-scope sink for the leaf-01 allocation instrumentation, used to
// prove raw-workspace creation and exhaustion never reserve a new native
// data backing.
std::vector<iom::rocm_detail::AllocationRecord>* g_workspace_records =
        nullptr;

void capture_workspace_allocations(
        const iom::rocm_detail::AllocationRecord& record) {
    if (g_workspace_records != nullptr) {
        g_workspace_records->push_back(record);
    }
}

struct WorkspaceObserverRestore final {
    WorkspaceObserverRestore()
            : saved_(iom::rocm_detail::allocation_observer),
              saved_sink_(g_workspace_records) {}

    WorkspaceObserverRestore(const WorkspaceObserverRestore&) = delete;
    WorkspaceObserverRestore& operator=(const WorkspaceObserverRestore&) =
            delete;

    ~WorkspaceObserverRestore() {
        iom::rocm_detail::allocation_observer = saved_;
        g_workspace_records = saved_sink_;
    }

    iom::rocm_detail::AllocationObserver saved_;
    std::vector<iom::rocm_detail::AllocationRecord>* saved_sink_ = nullptr;
};

[[nodiscard]] std::size_t succeeded_data_backing_allocations(
        const std::vector<iom::rocm_detail::AllocationRecord>& records) {
    std::size_t count = 0;
    for (const auto& record : records) {
        if (record.classification
                == iom::rocm_detail::AllocationClass::data_backing
                && record.kind
                        == iom::rocm_detail::AllocationKind::allocate
                && record.succeeded) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST_CASE("ROCm conformance: raw workspace suballocates the reserved data arena") {
    WorkspaceObserverRestore observer_restore;
    std::vector<iom::rocm_detail::AllocationRecord> records;
    g_workspace_records = &records;
    iom::rocm_detail::allocation_observer.complete =
            &capture_workspace_allocations;

    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{1u * 1024 * 1024});
    const std::size_t backings_before =
            succeeded_data_backing_allocations(records);

    // Empty owner: valid and allocation-free.
    auto empty = device->create_workspace(0);
    REQUIRE(empty != nullptr);
    CHECK(empty->empty());
    CHECK_EQ(empty->byte_size(), 0);
    CHECK(empty->view().owner_identity() == empty.get());
    CHECK(&empty->view().device() == device.get());
    CHECK_EQ(succeeded_data_backing_allocations(records), backings_before);

    // Positive creation suballocates the existing data arena: the range
    // is 32-byte aligned with the required capacity, and no new native
    // data backing appears.
    auto workspace = device->create_workspace(4096);
    REQUIRE(workspace != nullptr);
    const iom::RawWorkspaceView view = workspace->view();
    CHECK_EQ(view.byte_size(), 4096);
    CHECK(view.owner_identity() == workspace.get());
    CHECK_NOTHROW((void)iom::detail::WorkspaceValidation::validated(
            *device, view, 4096, 32, {}));
    CHECK_EQ(succeeded_data_backing_allocations(records), backings_before);

    // Owner identity is stable across further device activity.
    const iom::RawWorkspaceView view_before = workspace->view();
    auto another = device->create_workspace(256);
    REQUIRE(another != nullptr);
    const iom::RawWorkspaceView view_after = workspace->view();
    CHECK(view_after == view_before);
    CHECK_EQ(succeeded_data_backing_allocations(records), backings_before);

    // Exhaustion beyond the remaining contiguous arena range is
    // std::bad_alloc with no fallback backing allocation.
    CHECK_THROWS_AS((void)
            device->create_workspace(2u * 1024 * 1024), std::bad_alloc);
    CHECK_EQ(succeeded_data_backing_allocations(records), backings_before);

    // Destroying a workspace returns its arena range for reuse.
    another.reset();
    auto reused = device->create_workspace(256);
    REQUIRE(reused != nullptr);
    CHECK_EQ(reused->byte_size(), 256);
    CHECK_EQ(succeeded_data_backing_allocations(records), backings_before);
}

TEST_CASE("ROCm conformance: workspace requirement queries are pure and exact") {
    iom_conformance::TrafficGate gate;
    std::vector<std::byte> storage(64 * 1024 * 1024);
    iom::LinearAllocator reference_allocator(storage.data(), storage.size());
    auto reference = iom::make_cpu_device(reference_allocator);
    auto device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::F32};
    auto lhs = device->create_tensor(spec);
    auto rhs = device->create_tensor(spec);
    auto out = device->create_tensor(spec);
    auto queue = device->create_ops();

    // ROCm binary operations need no raw workspace.
    const iom::WorkspaceRequirements zero{0, 1};
    CHECK(queue->add_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->mul_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->sub_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->div_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);

    // Host transfers stage the whole-plane word-padded logical bytes.
    const iom::WorkspaceRequirements transfer{
            iom::gpu_algorithm::compute_staging_size(
                    spec.logical_nbytes()),
            32};
    CHECK(lhs->view().copy_from_host_workspace_requirements() == transfer);
    CHECK(lhs->view().copy_to_host_workspace_requirements() == transfer);

    // Validation matches the binary operation exactly.
    auto foreign_tensor = reference->create_tensor(spec);
    CHECK_THROWS_AS((void)
            queue->add_workspace_requirements(
                    foreign_tensor->view(), rhs->view(), out->view()),
            std::invalid_argument);
    CHECK_FALSE(gate.armed());
}

int main(int argc, char** argv) {
    if (std::getenv("IOM_ROCM_WATCHDOG_CHILD") != nullptr) {
        run_watchdog_child(argc, argv);
    }
    g_executable_path = argv[0];
    doctest::Context context(argc, argv);
    return context.run();
}
