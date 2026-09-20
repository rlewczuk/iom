#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_add.hpp"
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


#ifdef IOM_COEXIST_SYCL
#include <sycl/sycl.hpp>

#include "iom/sycl/device.hpp"
#include "runtime.hpp"
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
        ++allocations_;
        return ::operator new(size, std::align_val_t(32));
    }

    void free(void* buffer) override {
        ++frees_;
        ::operator delete(buffer, std::align_val_t(32));
    }

    void reset() override {}

    [[nodiscard]] std::size_t allocation_count() const noexcept {
        return allocations_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t free_count() const noexcept {
        return frees_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<std::size_t> allocations_{0};
    std::atomic<std::size_t> frees_{0};
};

// Coexistence traffic is tensors of at most a few KiB each, so a modest
// fixed data arena covers every live set on the standard-GPU backends.
constexpr std::size_t kCoexistenceArenaBytes = 16u * 1024 * 1024;

#ifdef IOM_COEXIST_SYCL
[[nodiscard]] std::size_t sycl_runtime_device_count() {
    const auto devices = sycl::device::get_devices();
    const std::size_t count = static_cast<std::size_t>(std::count_if(
            devices.begin(), devices.end(), [](const sycl::device& device) {
                return device.is_gpu() || device.is_accelerator();
            }));
    REQUIRE(count > 0);
    return count;
}
#endif


#ifdef IOM_COEXIST_CUDA
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
[[nodiscard]] int hip_runtime_device_count() {
    int count = 0;
    REQUIRE(hipGetDeviceCount(&count) == kHipSuccess);
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
BackendParticipant make_cuda_participant() {
    BackendParticipant participant;
    participant.name = "cuda";
    participant.kind = iom::BackendKind::CUDA;
    participant.owned_device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
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
BackendParticipant make_rocm_participant() {
    BackendParticipant participant;
    participant.name = "rocm";
    participant.kind = iom::BackendKind::ROCM;
    participant.owned_device = iom::make_rocm_device(
            0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
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

#ifdef IOM_COEXIST_SYCL
BackendParticipant make_sycl_participant() {
    BackendParticipant participant;
    participant.name = "sycl";
    participant.kind = iom::BackendKind::SYCL;
    participant.owned_device = iom::make_sycl_device(
            0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
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
    BackendParticipant cuda_participant = make_cuda_participant();
    participants.push_back(&cuda_participant);
#endif
#ifdef IOM_COEXIST_ROCM
    BackendParticipant rocm_participant = make_rocm_participant();
    participants.push_back(&rocm_participant);
#endif
#ifdef IOM_COEXIST_SYCL
    BackendParticipant sycl_participant = make_sycl_participant();
    participants.push_back(&sycl_participant);
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
        iom_conformance::copy_from_host(participant->source->view(), expected);
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
                    REQUIRE(iom::oid_is_token(token));
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

        CHECK_EQ(queue->copy(foreign_source->view(), destination->view()), iom::to_oid(iom::OidError::InvalidArgument));
        CHECK_EQ(queue->copy(source->view(), foreign_destination->view()), iom::to_oid(iom::OidError::InvalidArgument));

        // Validation happens before queue submission, so these rejected
        // pairs cannot alter device storage or queue completion state. The
        // successful same-device path is exercised by the interleaving case.
    };

    check_rejection(
            "cpu", *(iom::make_cpu_device(allocator)),
            *(iom::make_cpu_device(foreign_allocator)));
#ifdef IOM_COEXIST_CUDA
    {
        auto device = iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        auto foreign = iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        CHECK(device->backend_kind() == iom::BackendKind::CUDA);
        CHECK(foreign->backend_kind() == iom::BackendKind::CUDA);
        CHECK(device->backend_device() == foreign->backend_device());
        check_rejection("cuda", *device, *foreign);
    }
#endif
#ifdef IOM_COEXIST_ROCM
    {
        auto device = iom::make_rocm_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        auto foreign = iom::make_rocm_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        CHECK(device->backend_kind() == iom::BackendKind::ROCM);
        CHECK(foreign->backend_kind() == iom::BackendKind::ROCM);
        CHECK(device->backend_device() == foreign->backend_device());
        check_rejection("rocm", *device, *foreign);
    }
#endif
#ifdef IOM_COEXIST_SYCL
    {
        auto device = iom::make_sycl_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        auto foreign = iom::make_sycl_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        CHECK(device->backend_kind() == iom::BackendKind::SYCL);
        CHECK(foreign->backend_kind() == iom::BackendKind::SYCL);
        CHECK(device->backend_device() == foreign->backend_device());
        check_rejection("sycl", *device, *foreign);
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
        auto second = iom::make_cuda_device(
                1, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        REQUIRE(second != nullptr);
        CHECK(second->backend_kind() == iom::BackendKind::CUDA);
        CHECK(second->backend_device() == 1);
    }
#endif
#ifdef IOM_COEXIST_ROCM
    if (hip_runtime_device_count() > 1) {
        auto second = iom::make_rocm_device(
                1, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        REQUIRE(second != nullptr);
        CHECK(second->backend_kind() == iom::BackendKind::ROCM);
        CHECK(second->backend_device() == 1);
    }
#endif
#ifdef IOM_COEXIST_SYCL
    if (sycl_runtime_device_count() > 1) {
        auto second = iom::make_sycl_device(
                1, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        REQUIRE(second != nullptr);
        CHECK(second->backend_kind() == iom::BackendKind::SYCL);
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
    iom_conformance::copy_from_host(source->view(), iom_conformance::encode_logical(coexistence_spec(), 3));

    auto first = device->create_ops();
    auto second = device->create_ops();
    const iom::oid first_token =
            first->copy(source->view(), destination->view());
    const iom::oid repeat_token =
            first->copy(source->view(), destination->view());
    const iom::oid second_token =
            second->copy(source->view(), destination->view());
    REQUIRE(iom::oid_is_token(first_token));
    REQUIRE(iom::oid_is_token(repeat_token));
    REQUIRE(iom::oid_is_token(second_token));
    const std::uint8_t first_id = iom_conformance::token_queue(first_token);
    const std::uint8_t second_id = iom_conformance::token_queue(second_token);
    CHECK_EQ(iom_conformance::token_queue(repeat_token), first_id);
    CHECK_NE(first_id, second_id);
    CHECK_NOTHROW(first->wait(first_token));
    CHECK_NOTHROW(first->wait(repeat_token));
    CHECK_NOTHROW(second->wait(second_token));

    first.reset();
    auto recreated = device->create_ops();
    const iom::oid recreated_token =
            recreated->copy(source->view(), destination->view());
    // The released id returns to the pool and is leased again.
    REQUIRE(iom::oid_is_token(recreated_token));
    CHECK_EQ(iom_conformance::token_queue(recreated_token), first_id);
    CHECK_NE(iom_conformance::token_queue(recreated_token), second_id);
    CHECK_NOTHROW(recreated->wait(recreated_token));
    CHECK_NOTHROW(second->wait(second_token));

    // No generation check: the stale token collides with the recreated
    // queue's own sequence-1 submission and is silently accepted. The
    // collision is the caller's mistake, not a detected error.
    CHECK_NOTHROW(recreated->wait(first_token));

    // A stale sequence the recreated queue never submitted stays
    // caller-invalid even though it carries the recycled id.
    CHECK_THROWS_AS(recreated->wait(repeat_token), std::invalid_argument);
}

// Barrier-based concurrent queue creation and submission on one device.
// Independent threads race the per-device registry queue-id counter while
// creating queues and the per-device source/destination entry-id counter
// while submitting multi-plane copies; every queue must end up with a
// distinct registry identity, every operation with one unique entry pair,
// and destroying one queue must invalidate only that queue's own entries.
// The counting allocator makes the invalidation observable for CPU: a
// surviving queue's outstanding entry stays live, so its destination
// storage is released (freed) at tensor destruction -- never quarantined by
// another queue's teardown -- and every allocated tensor storage is freed
// exactly once. Standard-GPU backends suballocate device arenas, so their
// call sites pass a scalar placeholder (`int`, never dereferenced); for
// non-class placeholders the allocator-count code is discarded at compile
// time and the scenario runs without the free-count assertions.
template <typename CountingAllocator>
void run_concurrent_queue_scenario(
        const char* name, iom::Device& device,
        CountingAllocator* counting_allocator) {
    CAPTURE(name);
    // Deliberate four-live-queue cap per Device (change 004); a fifth
    // create_ops must be rejected, never raced here.
    constexpr int kQueues = 4;
    constexpr int kCopiesPerQueue = 12;
    const iom::TensorSpec spec = coexistence_spec();
    const std::vector<std::byte> expected =
            iom_conformance::encode_logical(spec, 13);

    [[maybe_unused]] std::size_t allocations_before = 0;
    [[maybe_unused]] std::size_t frees_before = 0;
    if constexpr (std::is_class_v<CountingAllocator>) {
        if (counting_allocator != nullptr) {
            allocations_before = counting_allocator->allocation_count();
            frees_before = counting_allocator->free_count();
        }
    } else {
        static_cast<void>(counting_allocator);
    }

    {
        auto source = device.create_tensor(spec);
        iom_conformance::copy_from_host(source->view(), expected);
        std::vector<std::unique_ptr<iom::Tensor>> destination;
        destination.reserve(kQueues);
        for (int queue = 0; queue < kQueues; ++queue) {
            destination.push_back(device.create_tensor(spec));
        }

        std::vector<std::unique_ptr<iom::DeviceOps>> queue(kQueues);
        std::vector<std::vector<iom::oid>> tokens(kQueues);
        std::atomic<bool> failed = false;
        std::barrier gate(kQueues + 1);
        std::vector<std::thread> threads;
        threads.reserve(kQueues);
        for (int worker = 0; worker < kQueues; ++worker) {
            threads.emplace_back([&, worker] {
                // A worker that fails must still contribute its remaining
                // barrier arrivals so a phase can never strand the main
                // thread; non-blocking arrive() drains without disturbing
                // the healthy workers' phase waits.
                int arrivals = 0;
                auto drain_arrivals = [&] {
                    while (arrivals < 3) {
                        (void)gate.arrive();
                        ++arrivals;
                    }
                };
                try {
                    // Queue creation races the per-device registry
                    // queue-id counter across these threads.
                    gate.arrive_and_wait();
                    ++arrivals;
                    queue[worker] = device.create_ops();
                    // Copy submission races the per-device entry-id
                    // counter across the queues' worker paths.
                    gate.arrive_and_wait();
                    ++arrivals;
                    for (int copy_index = 0;
                         copy_index < kCopiesPerQueue; ++copy_index) {
                        tokens[worker].push_back(queue[worker]->copy(
                                source->view(),
                                destination[worker]->view()));
                    }
                    gate.arrive_and_wait();
                    ++arrivals;
                    for (const iom::oid token : tokens[worker]) {
                        queue[worker]->wait(token);
                    }
                } catch (...) {
                    failed.store(true, std::memory_order_release);
                    drain_arrivals();
                }
            });
        }
        gate.arrive_and_wait();
        gate.arrive_and_wait();
        gate.arrive_and_wait();
        for (std::thread& thread : threads) {
            thread.join();
        }
        CHECK_FALSE(failed.load(std::memory_order_acquire));

        // Every operation completed; every destination matches the CPU
        // reference, proving each copy ran with its own unique entry pair.
        for (int worker = 0; worker < kQueues; ++worker) {
            iom_conformance::require_logical_bytes(
                    destination[worker]->view(), expected, name);
        }

        // Destroying one queue must invalidate only its own entries. A
        // surviving queue holds an outstanding copy; its entry must stay
        // live so its destination storage is released at destruction,
        // not quarantined by the destroyed queue's teardown.
        std::vector<std::unique_ptr<iom::Tensor>> late_destination(
                kQueues - 1);
        for (int survivor = 1; survivor < kQueues; ++survivor) {
            late_destination[survivor - 1] = device.create_tensor(spec);
        }
        std::vector<iom::oid> late_tokens(kQueues - 1);
        for (int survivor = 1; survivor < kQueues; ++survivor) {
            late_tokens[survivor - 1] = queue[survivor]->copy(
                    source->view(),
                    late_destination[survivor - 1]->view());
        }
        queue[0].reset();
        for (int survivor = 1; survivor < kQueues; ++survivor) {
            queue[survivor]->wait(late_tokens[survivor - 1]);
            if constexpr (std::is_class_v<CountingAllocator>) {
                if (counting_allocator != nullptr) {
                    const std::size_t frees_before_survivor =
                            counting_allocator->free_count();
                    late_destination[survivor - 1].reset();
                    CHECK_EQ(
                            counting_allocator->free_count(),
                            frees_before_survivor + 1);
                } else {
                    late_destination[survivor - 1].reset();
                }
            } else {
                late_destination[survivor - 1].reset();
            }
        }

        // Surviving queues keep submitting and completing after the other
        // queue's teardown.
        for (int survivor = 1; survivor < kQueues; ++survivor) {
            const iom::oid token = queue[survivor]->copy(
                    source->view(), destination[survivor]->view());
            queue[survivor]->wait(token);
            iom_conformance::require_logical_bytes(
                    destination[survivor]->view(), expected, name);
        }
    }

    // All 2 * kQueues tensor storages (one source + kQueues destinations +
    // kQueues - 1 late destinations) were freed exactly once at their
    // destruction: none was left quarantined by a cross-queue invalidation
    // and none was released twice.
    if constexpr (std::is_class_v<CountingAllocator>) {
        if (counting_allocator != nullptr) {
            CHECK_EQ(
                    counting_allocator->allocation_count(),
                    allocations_before + 2 * kQueues);
            CHECK_EQ(
                    counting_allocator->free_count(),
                    frees_before + 2 * kQueues);
        }
    }
}

// The registry stores per-device queue and entry ids; concurrent queue
// creation and submission on one device must stay serialized so every queue
// receives a distinct registry identity and every copy a unique entry pair.
TEST_CASE(
        "Backend coexistence: concurrent queue creation and submission "
        "on one device") {
    {
        HostAllocator allocator;
        auto device = iom::make_cpu_device(allocator);
        run_concurrent_queue_scenario("cpu", *device, &allocator);
    }
#ifdef IOM_COEXIST_CUDA
    {
        auto device = iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        // Tensor storage now suballocates the device data arena; the
        // registry invalidation scenario runs without caller-allocator
        // traffic on the standard-GPU backends.
        run_concurrent_queue_scenario<int>("cuda", *device, nullptr);
    }
#endif
#ifdef IOM_COEXIST_ROCM
    {
        auto device = iom::make_rocm_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        run_concurrent_queue_scenario<int>("rocm", *device, nullptr);
    }
#endif
#ifdef IOM_COEXIST_SYCL
    {
        auto device = iom::make_sycl_device(
                0, iom::DeviceMemoryConfig{kCoexistenceArenaBytes});
        run_concurrent_queue_scenario<int>("sycl", *device, nullptr);
    }
#endif
}
namespace {

std::vector<std::byte> coexistence_uniform(
        const iom::TensorSpec& spec, std::uint64_t value) {
    std::vector<std::byte> bytes(spec.logical_nbytes(), std::byte{0});
    const std::size_t bits = iom_conformance::bits_of(spec.data_type);
    auto* base = reinterpret_cast<unsigned char*>(bytes.data());
    for (std::size_t i = 0; i < spec.shape.element_count(); ++i) {
        iom_conformance::write_bits(base, i * bits, bits, value);
    }
    return bytes;
}

void run_interleaved_operations(
        const std::vector<BackendParticipant*>& participants,
        iom::DataType type, std::uint64_t lhs_value,
        std::uint64_t rhs_value) {
    using iom_conformance::BinaryOperation;
    const iom::TensorSpec spec{iom::TensorShape{{2, 17, 33}}, type};
    const std::vector<std::byte> lhs_bytes =
            coexistence_uniform(spec, lhs_value);
    const std::vector<std::byte> rhs_bytes =
            coexistence_uniform(spec, rhs_value);

    struct Work {
        std::unique_ptr<iom::Tensor> lhs, rhs, staged;
        std::unique_ptr<iom::Tensor> add_out, mul_out, sub_out, div_out,
                copied;
        const iom::Tensor* lhs_owner = nullptr;
        const iom::Tensor* rhs_owner = nullptr;
        void* lhs_handle = nullptr;
        void* rhs_handle = nullptr;
        iom::Device* device = nullptr;
        std::vector<std::unique_ptr<iom::DeviceOps>> queues;
        std::vector<std::unique_ptr<iom::RawWorkspace>> binary_workspaces;
        struct Expected {
            iom::oid token;
            iom::Tensor* output;
            std::vector<std::byte> bytes;
        };
        std::vector<Expected> expected;
    };

    std::vector<Work> work;
    work.reserve(participants.size());
    for (BackendParticipant* participant : participants) {
        Work item;
        item.device = participant->device;
        item.lhs = participant->device->create_tensor(spec);
        item.rhs = participant->device->create_tensor(spec);
        item.staged = participant->device->create_tensor(spec);
        item.add_out = participant->device->create_tensor(spec);
        item.mul_out = participant->device->create_tensor(spec);
        item.sub_out = participant->device->create_tensor(spec);
        item.div_out = participant->device->create_tensor(spec);
        item.copied = participant->device->create_tensor(spec);
        iom_conformance::copy_from_host(item.lhs->view(), lhs_bytes);
        iom_conformance::copy_from_host(item.rhs->view(), rhs_bytes);
        const std::vector<std::byte> sentinel(
                spec.logical_nbytes(), iom_conformance::kReadbackSentinel);
        for (iom::Tensor* output : {item.staged.get(), item.add_out.get(),
                                    item.mul_out.get(), item.sub_out.get(),
                                    item.div_out.get(), item.copied.get()}) {
            iom_conformance::copy_from_host(output->view(), sentinel);
        }
        item.queues.push_back(participant->device->create_ops());
        item.queues.push_back(participant->device->create_ops());
        item.lhs_owner = item.lhs->view().owner_identity();
        item.rhs_owner = item.rhs->view().owner_identity();
        item.lhs_handle = item.lhs->view().native_handle();
        item.rhs_handle = item.rhs->view().native_handle();
        work.push_back(std::move(item));
    }

    const auto expected_for = [&](iom_conformance::add_oracle::operation op,
                                  std::uint64_t lhs, std::uint64_t rhs) {
        return coexistence_uniform(
                spec, iom_conformance::add_oracle::binary(type, lhs, rhs, op));
    };
    const auto submit = [&](Work& item, std::size_t queue_index,
                            iom_conformance::add_oracle::operation operation,
                            iom::Tensor& lhs, iom::Tensor& rhs,
                            iom::Tensor& output, std::uint64_t a,
                            std::uint64_t b) {
        const BinaryOperation binary_operation = [&] {
            switch (operation) {
                case iom_conformance::add_oracle::operation::add:
                    return BinaryOperation::add;
                case iom_conformance::add_oracle::operation::mul:
                    return BinaryOperation::mul;
                case iom_conformance::add_oracle::operation::sub:
                    return BinaryOperation::sub;
                case iom_conformance::add_oracle::operation::div:
                    return BinaryOperation::div;
            }
            throw std::logic_error("unknown binary operation");
        }();
        auto& queue = *item.queues[queue_index];
        const auto requirements = query_binary_workspace_requirements(
                queue, binary_operation, lhs.view(), rhs.view(),
                output.view());
        if (requirements.bytes != 0) {
            item.binary_workspaces.push_back(
                    item.device->create_workspace(requirements.bytes));
        }
        iom::oid token = iom::to_oid(iom::OidError::InternalError);
        if (requirements.bytes != 0) {
            const iom::RawWorkspaceView workspace =
                    item.binary_workspaces.back()->view();
            token = submit_binary_operation(
                    queue, binary_operation, lhs.view(), rhs.view(),
                    output.view(), workspace);
        } else {
            token = submit_binary_operation(
                    queue, binary_operation, lhs.view(), rhs.view(),
                    output.view());
        }
        REQUIRE(iom::oid_is_token(token));
        item.expected.push_back(
                {token, &output, expected_for(operation, a, b)});
        return token;
    };

    HostAllocator foreign_allocator;
    auto foreign_device = iom::make_cpu_device(foreign_allocator);
    auto foreign_tensor = foreign_device->create_tensor(spec);
    iom_conformance::copy_from_host(foreign_tensor->view(), lhs_bytes);
    for (std::size_t i = 0; i < work.size(); ++i) {
        CAPTURE(participants[i]->name);
        const auto before = iom_conformance::read_logical(work[i].add_out->view());
        CHECK_EQ(work[i].queues[0]->add(foreign_tensor->view(),
                                        work[i].rhs->view(),
                                        work[i].add_out->view()),
                 iom::to_oid(iom::OidError::InvalidArgument));
        CHECK_EQ(iom_conformance::read_logical(work[i].add_out->view()), before);
    }

    std::set<iom::oid> tokens;
    std::set<std::uint8_t> queue_ids;
    for (std::size_t i = 0; i < work.size(); ++i) {
        Work& item = work[i];
        const iom::oid copy_token =
                item.queues[0]->copy(item.lhs->view(), item.staged->view());
        REQUIRE(iom::oid_is_token(copy_token));
        item.expected.push_back({copy_token, item.staged.get(), lhs_bytes});
        submit(item, 0, iom_conformance::add_oracle::operation::add,
               *item.staged, *item.rhs, *item.add_out, lhs_value, rhs_value);
        submit(item, 0, iom_conformance::add_oracle::operation::mul,
               *item.lhs, *item.rhs, *item.mul_out, lhs_value, rhs_value);
        submit(item, 1, iom_conformance::add_oracle::operation::sub,
               *item.lhs, *item.rhs, *item.sub_out, lhs_value, rhs_value);
        if (type != iom::DataType::I2 && type != iom::DataType::U2 &&
            type != iom::DataType::I4 && type != iom::DataType::U4 &&
            type != iom::DataType::I8 && type != iom::DataType::U8 &&
            type != iom::DataType::I16 && type != iom::DataType::U16 &&
            type != iom::DataType::I32 && type != iom::DataType::U32 &&
            type != iom::DataType::I64 && type != iom::DataType::U64) {
            submit(item, 1, iom_conformance::add_oracle::operation::div,
                   *item.lhs, *item.rhs, *item.div_out, lhs_value, rhs_value);
        }
        const iom::oid copy_back =
                item.queues[1]->copy(item.sub_out->view(), item.copied->view());
        REQUIRE(iom::oid_is_token(copy_back));
        item.expected.push_back({copy_back, item.copied.get(),
                                 expected_for(iom_conformance::add_oracle::operation::sub,
                                              lhs_value, rhs_value)});
        for (const Work::Expected& expected : item.expected) {
            CHECK(tokens.insert(expected.token).second);
            queue_ids.insert(iom_conformance::token_queue(expected.token));
        }
    }
    CHECK_EQ(queue_ids.size(), participants.size() * 2);
    for (std::size_t i = work.size(); i-- > 0;) {
        Work& item = work[i];
        for (std::size_t n = item.expected.size(); n-- > 0;) {
            const auto& expected = item.expected[n];
            const std::size_t queue = iom_conformance::token_queue(expected.token)
                                      == iom_conformance::token_queue(item.expected.front().token)
                                      ? 0 : 1;
            item.queues[queue]->wait(expected.token);
            iom_conformance::require_logical_bytes(
                    expected.output->view(), expected.bytes,
                    participants[i]->name);
        }
        item.queues[0]->wait(item.expected.front().token);
        CHECK(item.lhs_owner == item.lhs->view().owner_identity());
        CHECK(item.rhs_owner == item.rhs->view().owner_identity());
        CHECK(item.lhs_handle == item.lhs->view().native_handle());
        CHECK(item.rhs_handle == item.rhs->view().native_handle());
        auto lhs_derived = item.lhs->view().slice(0, 0, 2);
        auto rhs_derived = item.rhs->view().slice(0, 0, 2);
        auto out_derived = item.add_out->view().slice(0, 0, 2);
        const auto derived_requirements =
                query_binary_workspace_requirements(
                        *item.queues[0], BinaryOperation::add, lhs_derived,
                        rhs_derived, out_derived);
        if (derived_requirements.bytes != 0) {
            item.binary_workspaces.push_back(
                    item.device->create_workspace(
                            derived_requirements.bytes));
        }
        iom::oid derived = iom::to_oid(iom::OidError::InternalError);
        if (derived_requirements.bytes != 0) {
            const iom::RawWorkspaceView workspace =
                    item.binary_workspaces.back()->view();
            derived = item.queues[0]->add(
                    lhs_derived, rhs_derived, out_derived, workspace);
        } else {
            derived = item.queues[0]->add(
                    lhs_derived, rhs_derived, out_derived);
        }
        REQUIRE(iom::oid_is_token(derived));
        item.queues[0]->wait(derived);
        const std::uint64_t add_value =
                iom_conformance::add_oracle::binary(
                        type, lhs_value, rhs_value,
                        iom_conformance::add_oracle::operation::add);
        const auto alias_requirements =
                query_binary_workspace_requirements(
                        *item.queues[1], BinaryOperation::add,
                        item.add_out->view(), item.add_out->view(),
                        item.add_out->view());
        if (alias_requirements.bytes != 0) {
            item.binary_workspaces.push_back(
                    item.device->create_workspace(
                            alias_requirements.bytes));
        }
        iom::oid alias = iom::to_oid(iom::OidError::InternalError);
        if (alias_requirements.bytes != 0) {
            const iom::RawWorkspaceView workspace =
                    item.binary_workspaces.back()->view();
            alias = item.queues[1]->add(
                    item.add_out->view(), item.add_out->view(),
                    item.add_out->view(), workspace);
        } else {
            alias = item.queues[1]->add(
                    item.add_out->view(), item.add_out->view(),
                    item.add_out->view());
        }
        REQUIRE(iom::oid_is_token(alias));
        item.queues[1]->wait(alias);
        iom_conformance::require_logical_bytes(
                item.add_out->view(),
                coexistence_uniform(
                        spec, iom_conformance::add_oracle::binary(
                                      type, add_value, add_value,
                                      iom_conformance::add_oracle::operation::add)),
                participants[i]->name);
    }
    work.clear();

    // A retained failure must remain observable without poisoning another queue.
    auto fault_lhs = participants.front()->device->create_tensor(spec);
    auto fault_rhs = participants.front()->device->create_tensor(spec);
    auto fault_out = participants.front()->device->create_tensor(spec);
    iom_conformance::CommonBinaryQueue fault_queue(*participants.front()->device);
    fault_queue.fail_next_after_acceptance();
    const bool integer =
            type == iom::DataType::I2 || type == iom::DataType::U2 ||
            type == iom::DataType::I4 || type == iom::DataType::U4 ||
            type == iom::DataType::I8 || type == iom::DataType::U8 ||
            type == iom::DataType::I16 || type == iom::DataType::U16 ||
            type == iom::DataType::I32 || type == iom::DataType::U32 ||
            type == iom::DataType::I64 || type == iom::DataType::U64;
    const iom::oid failed = integer
                                    ? fault_queue.add(fault_lhs->view(),
                                                      fault_rhs->view(),
                                                      fault_out->view())
                                    : fault_queue.div(fault_lhs->view(),
                                                      fault_rhs->view(),
                                                      fault_out->view());
    REQUIRE(iom::oid_is_token(failed));
    fault_queue.finish(iom_conformance::token_sequence(failed));
    iom_conformance::expect_repeated_runtime_failure(fault_queue, failed);
}

TEST_CASE("Backend coexistence: ADD interleaves across enabled backends") {
    HostAllocator cpu_allocator;
    BackendParticipant cpu_participant = make_cpu_participant(cpu_allocator);
    std::vector<BackendParticipant*> participants{&cpu_participant};
#ifdef IOM_COEXIST_CUDA
    BackendParticipant cuda_participant = make_cuda_participant();
    participants.push_back(&cuda_participant);
#endif
#ifdef IOM_COEXIST_ROCM
    BackendParticipant rocm_participant = make_rocm_participant();
    participants.push_back(&rocm_participant);
#endif
#ifdef IOM_COEXIST_SYCL
    BackendParticipant sycl_participant = make_sycl_participant();
    participants.push_back(&sycl_participant);
#endif
    run_interleaved_operations(participants, iom::DataType::I32, 7, 5);
    run_interleaved_operations(participants, iom::DataType::U8, 7, 3);
    run_interleaved_operations(participants, iom::DataType::F32,
                               std::uint64_t{0x3fc00000},
                               std::uint64_t{0x40200000});
}

}  // namespace
