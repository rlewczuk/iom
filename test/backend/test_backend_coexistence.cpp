#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

// The coexistence executable is composed from exactly the enabled backend
// libraries: CMake defines one IOM_COEXIST_* macro per enabled option and
// links that backend's library, so this one translation unit compile-checks
// every enabled public factory and links them side by side. There is no
// registry, singleton, or BackendKind switch anywhere in this file.

#ifdef IOM_COEXIST_CUDA
// The driver header only: the CUDA and HIP runtime headers both declare
// the vector types (uchar1, dim3, ...) and cannot share one translation
// unit, so device memory comes from cuMemAlloc/cuMemFree below.
#include <cuda.h>

#include "iom/cuda/device.hpp"
#endif

#ifdef IOM_COEXIST_ROCM
#include "iom/rocm/device.hpp"

// Keep HIP's runtime header out of this translation unit. It declares CUDA
// vector types that conflict with the CUDA driver header when both backends
// are enabled. These three C ABI entry points are sufficient for the caller
// allocator and device-count guard below; the ROCm backend itself includes
// the full HIP API.
extern "C" int hipGetDeviceCount(int* count);
extern "C" int hipMalloc(void** pointer, std::size_t size);
extern "C" int hipFree(void* pointer);

constexpr int kHipSuccess = 0;
#endif

#ifdef IOM_COEXIST_TTNN
#include <tt-metalium/host_api.hpp>

#include "iom/ttnn/device.hpp"
#endif

namespace {

// One BF16 specification every enabled backend materializes, transfers,
// and copies: leading planes plus padding in both tiled dimensions.
const iom::TensorSpec& coexistence_spec() {
    static const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::BF16};
    return spec;
}

// Plain 32-byte-aligned heap allocator for CPU devices.
class HostAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        return ::operator new(size, std::align_val_t(32));
    }

    void free(void* buffer) override {
        ::operator delete(buffer, std::align_val_t(32));
    }

    void reset() override {}
};

#ifdef IOM_COEXIST_CUDA
class CudaMemoryAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        CUdeviceptr device_pointer = 0;
        REQUIRE(cuMemAlloc(&device_pointer, size) == CUDA_SUCCESS);
        return reinterpret_cast<void*>(device_pointer);
    }

    void free(void* buffer) override {
        REQUIRE(cuMemFree(reinterpret_cast<CUdeviceptr>(buffer))
                == CUDA_SUCCESS);
    }

    void reset() override {}
};

// Hardware is required, never skipped: an enabled backend fails the test
// when its runtime reports no usable device.
[[nodiscard]] int cuda_runtime_device_count() {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    int count = 0;
    REQUIRE(cuDeviceGetCount(&count) == CUDA_SUCCESS);
    REQUIRE(count > 0);
    return count;
}
#endif

#ifdef IOM_COEXIST_ROCM
class HipMemoryAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t size) override {
        void* block = nullptr;
        REQUIRE(hipMalloc(&block, size) == kHipSuccess);
        return block;
    }

    void free(void* buffer) override {
        REQUIRE(hipFree(buffer) == kHipSuccess);
    }

    void reset() override {}
};

[[nodiscard]] int hip_runtime_device_count() {
    int count = 0;
    REQUIRE(hipGetDeviceCount(&count) == kHipSuccess);
    REQUIRE(count > 0);
    return count;
}
#endif

#ifdef IOM_COEXIST_TTNN
[[nodiscard]] std::size_t ttnn_runtime_device_count() {
    const std::size_t count = tt::tt_metal::GetNumAvailableDevices();
    REQUIRE(count > 0);
    return count;
}
#endif

// Everything one backend contributes to the shared process: its device,
// at least two queues, and one source/destination tensor pair per queue.
struct BackendParticipant {
    std::string name;
    iom::BackendKind kind = iom::BackendKind::CPU;
    std::uint32_t ordinal = 0;
    std::unique_ptr<iom::Device> owned_device;
    iom::Device* device = nullptr;
    std::unique_ptr<iom::Tensor> source;
    std::vector<std::unique_ptr<iom::Tensor>> destinations;
    std::vector<std::unique_ptr<iom::DeviceOps>> queues;
    // Tokens each queue submitted, parallel to queues.
    std::vector<std::vector<iom::oid>> submitted;
};

BackendParticipant make_cpu_participant(HostAllocator& allocator) {
    BackendParticipant participant;
    participant.name = "cpu";
    participant.kind = iom::BackendKind::CPU;
    participant.owned_device = iom::make_cpu_device(allocator);
    participant.device = participant.owned_device.get();
    participant.source = participant.device->create_tensor(coexistence_spec());
    participant.queues.push_back(participant.device->create_ops());
    participant.queues.push_back(participant.device->create_ops());
    for (std::size_t i = 0; i < participant.queues.size(); ++i) {
        participant.destinations.push_back(
                participant.device->create_tensor(coexistence_spec()));
    }
    participant.submitted.resize(participant.queues.size());
    return participant;
}

#ifdef IOM_COEXIST_CUDA
BackendParticipant make_cuda_participant(CudaMemoryAllocator& allocator) {
    BackendParticipant participant;
    participant.name = "cuda";
    participant.kind = iom::BackendKind::CUDA;
    participant.owned_device = iom::make_cuda_device(0, allocator);
    participant.device = participant.owned_device.get();
    participant.source = participant.device->create_tensor(coexistence_spec());
    participant.queues.push_back(participant.device->create_ops());
    participant.queues.push_back(participant.device->create_ops());
    for (std::size_t i = 0; i < participant.queues.size(); ++i) {
        participant.destinations.push_back(
                participant.device->create_tensor(coexistence_spec()));
    }
    participant.submitted.resize(participant.queues.size());
    return participant;
}
#endif

#ifdef IOM_COEXIST_ROCM
BackendParticipant make_rocm_participant(HipMemoryAllocator& allocator) {
    BackendParticipant participant;
    participant.name = "rocm";
    participant.kind = iom::BackendKind::ROCM;
    participant.owned_device = iom::make_rocm_device(0, allocator);
    participant.device = participant.owned_device.get();
    participant.source = participant.device->create_tensor(coexistence_spec());
    participant.queues.push_back(participant.device->create_ops());
    participant.queues.push_back(participant.device->create_ops());
    for (std::size_t i = 0; i < participant.queues.size(); ++i) {
        participant.destinations.push_back(
                participant.device->create_tensor(coexistence_spec()));
    }
    participant.submitted.resize(participant.queues.size());
    return participant;
}
#endif

#ifdef IOM_COEXIST_TTNN
BackendParticipant make_ttnn_participant() {
    BackendParticipant participant;
    participant.name = "ttnn";
    participant.kind = iom::BackendKind::TTNN;
    participant.owned_device = iom::make_ttnn_device(0);
    participant.device = participant.owned_device.get();
    participant.source = participant.device->create_tensor(coexistence_spec());
    participant.queues.push_back(participant.device->create_ops());
    participant.queues.push_back(participant.device->create_ops());
    for (std::size_t i = 0; i < participant.queues.size(); ++i) {
        participant.destinations.push_back(
                participant.device->create_tensor(coexistence_spec()));
    }
    participant.submitted.resize(participant.queues.size());
    return participant;
}
#endif

}  // namespace

// The main coexistence proof: every enabled backend constructs devices and
// queues in one process, host transfers fill every source, same-device
// asynchronous copies are submitted interleaved across all backends and
// queues, each originating queue waits for its own tokens, and every
// destination matches the CPU reference bit for bit.
TEST_CASE("Backend coexistence: enabled backends interleave in one process") {
    const std::vector<std::byte> expected =
            iom_conformance::encode_logical(coexistence_spec(), 7);

    HostAllocator cpu_allocator;
    BackendParticipant cpu_participant = make_cpu_participant(cpu_allocator);
    std::vector<BackendParticipant*> participants{&cpu_participant};
#ifdef IOM_COEXIST_CUDA
    CudaMemoryAllocator cuda_allocator;
    BackendParticipant cuda_participant = make_cuda_participant(cuda_allocator);
    participants.push_back(&cuda_participant);
#endif
#ifdef IOM_COEXIST_ROCM
    HipMemoryAllocator rocm_allocator;
    BackendParticipant rocm_participant = make_rocm_participant(rocm_allocator);
    participants.push_back(&rocm_participant);
#endif
#ifdef IOM_COEXIST_TTNN
    BackendParticipant ttnn_participant = make_ttnn_participant();
    participants.push_back(&ttnn_participant);
#endif

    std::size_t queue_count = 0;
    for (const BackendParticipant* participant : participants) {
        CAPTURE(participant->name);
        CHECK(participant->device->backend_kind() == participant->kind);
        CHECK(participant->device->backend_device() == participant->ordinal);
        REQUIRE(participant->queues.size() >= 2);
        queue_count += participant->queues.size();
    }

    // Host transfer into every backend's source, then a readback of the
    // same logical bytes.
    for (BackendParticipant* participant : participants) {
        CAPTURE(participant->name);
        participant->source->view().copy_from_host(expected);
        iom_conformance::require_logical_bytes(
                participant->source->view(), expected, participant->name);
    }

    // Interleave submissions: two rounds of same-device asynchronous
    // copies round-robin across every queue of every backend, with no
    // waits between submissions.
    std::set<std::uint8_t> queue_ids;
    for (std::size_t round = 0; round < 2; ++round) {
        for (std::size_t burst = 0; burst < 2; ++burst) {
            for (BackendParticipant* participant : participants) {
                for (std::size_t index = 0;
                     index < participant->queues.size(); ++index) {
                    const iom::oid token = participant->queues[index]->copy(
                            participant->source->view(),
                            participant->destinations[index]->view());
                    queue_ids.insert(iom_conformance::token_queue(token));
                    participant->submitted[index].push_back(token);
                }
            }
        }
    }

    // Queue ids are process-unique across backend types: one distinct
    // live id per queue, with no backend-global selector.
    CHECK_EQ(queue_ids.size(), queue_count);

    // Wait on each originating queue -- tokens are waited only on the
    // queue that submitted them -- in reverse participant order, then
    // repeat one wait to prove waits stay repeatable under coexistence.
    for (auto participant_it = participants.rbegin();
         participant_it != participants.rend(); ++participant_it) {
        BackendParticipant* participant = *participant_it;
        for (std::size_t index = 0; index < participant->queues.size();
             ++index) {
            for (const iom::oid token : participant->submitted[index]) {
                participant->queues[index]->wait(token);
            }
        }
    }
    participants.front()->queues.front()->wait(
            participants.front()->submitted.front().front());

    // Every destination matches the CPU reference bit for bit.
    for (BackendParticipant* participant : participants) {
        CAPTURE(participant->name);
        for (const auto& destination : participant->destinations) {
            iom_conformance::require_logical_bytes(
                    destination->view(), expected, participant->name);
        }
    }

    // A second interleaved round after the first completed, waited in
    // forward order.
    for (BackendParticipant* participant : participants) {
        for (std::size_t index = 0; index < participant->queues.size();
             ++index) {
            const iom::oid token = participant->queues[index]->copy(
                    participant->source->view(),
                    participant->destinations[index]->view());
            participant->queues[index]->wait(token);
        }
    }
    for (BackendParticipant* participant : participants) {
        CAPTURE(participant->name);
        for (const auto& destination : participant->destinations) {
            iom_conformance::require_logical_bytes(
                    destination->view(), expected, participant->name);
        }
    }
}

// Each receiving queue rejects views created by another Device instance,
// including an independent instance of the same backend with the same
// ordinal.
TEST_CASE("Backend coexistence: queues reject views from another device") {
    HostAllocator allocator;
    HostAllocator foreign_allocator;

    const auto check_rejection = [&](const char* name, iom::Device& device,
                                     iom::Device& foreign) {
        CAPTURE(name);
        auto source = device.create_tensor(coexistence_spec());
        auto destination = device.create_tensor(coexistence_spec());
        auto foreign_source = foreign.create_tensor(coexistence_spec());
        auto foreign_destination = foreign.create_tensor(coexistence_spec());
        auto queue = device.create_ops();

        CHECK_THROWS_AS(
                (void)queue->copy(
                        foreign_source->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                (void)queue->copy(
                        source->view(), foreign_destination->view()),
                std::invalid_argument);

        // Validation happens before queue submission, so these rejected
        // pairs cannot alter device storage or queue completion state. The
        // successful same-device path is exercised by the interleaving case.
    };

    check_rejection(
            "cpu", *(iom::make_cpu_device(allocator)),
            *(iom::make_cpu_device(foreign_allocator)));
#ifdef IOM_COEXIST_CUDA
    {
        CudaMemoryAllocator device_allocator;
        CudaMemoryAllocator foreign_device_allocator;
        auto device = iom::make_cuda_device(0, device_allocator);
        auto foreign = iom::make_cuda_device(0, foreign_device_allocator);
        CHECK(device->backend_kind() == iom::BackendKind::CUDA);
        CHECK(foreign->backend_kind() == iom::BackendKind::CUDA);
        CHECK(device->backend_device() == foreign->backend_device());
        check_rejection("cuda", *device, *foreign);
    }
#endif
#ifdef IOM_COEXIST_ROCM
    {
        HipMemoryAllocator device_allocator;
        HipMemoryAllocator foreign_device_allocator;
        auto device = iom::make_rocm_device(0, device_allocator);
        auto foreign = iom::make_rocm_device(0, foreign_device_allocator);
        CHECK(device->backend_kind() == iom::BackendKind::ROCM);
        CHECK(foreign->backend_kind() == iom::BackendKind::ROCM);
        CHECK(device->backend_device() == foreign->backend_device());
        check_rejection("rocm", *device, *foreign);
    }
#endif
#ifdef IOM_COEXIST_TTNN
    {
        // tt-metal permits only one live context per physical device per
        // process, so an independently created CPU device fills the
        // foreign slot; queue validation rejects views by Device identity.
        auto device = iom::make_ttnn_device(0);
        auto foreign = iom::make_cpu_device(foreign_allocator);
        CHECK(device->backend_kind() == iom::BackendKind::TTNN);
        check_rejection("ttnn", *device, *foreign);
    }
#endif
}

// Where hardware provides more than one physical device, a second device
// of the same backend with the next ordinal constructs in the same
// process and reports its own kind and ordinal.
TEST_CASE("Backend coexistence: second devices report their own ordinal") {
    HostAllocator allocator;
    auto first_cpu = iom::make_cpu_device(allocator);
    auto second_cpu = iom::make_cpu_device(allocator);
    CHECK(first_cpu->backend_kind() == iom::BackendKind::CPU);
    CHECK(second_cpu->backend_kind() == iom::BackendKind::CPU);
    CHECK(first_cpu->backend_device() == second_cpu->backend_device());

#ifdef IOM_COEXIST_CUDA
    if (cuda_runtime_device_count() > 1) {
        CudaMemoryAllocator second_allocator;
        auto second = iom::make_cuda_device(1, second_allocator);
        REQUIRE(second != nullptr);
        CHECK(second->backend_kind() == iom::BackendKind::CUDA);
        CHECK(second->backend_device() == 1);
    }
#endif
#ifdef IOM_COEXIST_ROCM
    if (hip_runtime_device_count() > 1) {
        HipMemoryAllocator second_allocator;
        auto second = iom::make_rocm_device(1, second_allocator);
        REQUIRE(second != nullptr);
        CHECK(second->backend_kind() == iom::BackendKind::ROCM);
        CHECK(second->backend_device() == 1);
    }
#endif
#ifdef IOM_COEXIST_TTNN
    if (ttnn_runtime_device_count() > 1) {
        auto second = iom::make_ttnn_device(1);
        REQUIRE(second != nullptr);
        CHECK(second->backend_kind() == iom::BackendKind::TTNN);
        CHECK(second->backend_device() == 1);
    }
#endif
}

// Queue ids come from the one process-wide pool: live ids stay unique,
// a released id returns to the pool, and a stale token from a destroyed
// queue remains caller-invalid on whichever queue answers next, with no
// generation check.
TEST_CASE("Backend coexistence: queue ids release, reuse, and stay unique") {
    HostAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto source = device->create_tensor(coexistence_spec());
    auto destination = device->create_tensor(coexistence_spec());
    source->view().copy_from_host(
            iom_conformance::encode_logical(coexistence_spec(), 3));

    auto first = device->create_ops();
    auto second = device->create_ops();
    const iom::oid first_token =
            first->copy(source->view(), destination->view());
    const iom::oid repeat_token =
            first->copy(source->view(), destination->view());
    const iom::oid second_token =
            second->copy(source->view(), destination->view());
    const std::uint8_t first_id = iom_conformance::token_queue(first_token);
    const std::uint8_t second_id = iom_conformance::token_queue(second_token);
    CHECK_EQ(iom_conformance::token_queue(repeat_token), first_id);
    CHECK_NE(first_id, second_id);
    first->wait(first_token);
    first->wait(repeat_token);
    second->wait(second_token);

    first.reset();
    auto recreated = device->create_ops();
    const iom::oid recreated_token =
            recreated->copy(source->view(), destination->view());
    // The released id returns to the pool and is leased again.
    CHECK_EQ(iom_conformance::token_queue(recreated_token), first_id);
    CHECK_NE(iom_conformance::token_queue(recreated_token), second_id);
    recreated->wait(recreated_token);
    second->wait(second_token);

    // No generation check: the stale token collides with the recreated
    // queue's own sequence-1 submission and is silently accepted. The
    // collision is the caller's mistake, not a detected error.
    recreated->wait(first_token);

    // A stale sequence the recreated queue never submitted stays
    // caller-invalid even though it carries the recycled id.
    CHECK_THROWS_AS(recreated->wait(repeat_token), std::invalid_argument);
}
