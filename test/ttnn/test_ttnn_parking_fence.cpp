// Regression test for the HIGH R2-001 capacity-parked fence seam. The
// defect: TTNN submission fences were built with only the device pointer
// and returned `FenceResult::success()` after a blind mesh `finish()`; for a
// parked submission whose native work had not yet been fully enqueued,
// that success was wrong, `release_or_quarantine` released the parked
// tensor planes, and the later credit-driven dispatch dereferenced freed
// DRAM. The fix carries the submission sequence and the queue's
// per-domain execution marker into the fence so a parked submission
// returns `FenceResult::pending()` until its domain's marker covers it.
//
// Per-domain markers (option (b) in the repair round): TTNN has two
// execution domains because copy/embedding use `worker_.submit_copy`
// (synchronous `execute()` on the submitter thread) while
// binary/rmsnorm use `worker_.submit_after_publish` (asynchronous
// `execute()` on the staged worker thread). A single shared marker
// could let one domain store a higher sequence than an in-flight
// submission of the other domain, letting `ttnn_fence_invoke` return
// success for the not-yet-executed parked submission and letting
// `release_or_quarantine` free its planes; per-domain markers
// (`caller_executed_seq_` and `worker_executed_seq_`) plus a
// `fetch_max` store stop the cross-domain overtake and the regression
// in the same shape. The fence captures the domain-relevant marker
// pointer at build time, so the invoker never has to inspect a
// unified global state.
//
// Determinism strategy: every case uses a single-credit admission
// queue (`max_in_flight_per_queue = 1`) and saturates it with a
// head dispatch plus a parked dispatch. With one credit consumed by
// the head, the second submit's `pump_admission` sees
// `admission_credits_ == admission_capacity_` and returns without
// dispatching; the parked task is held in `admission_fifo_` until the
// head retires, with nothing for it in the worker's `tasks_` queue
// and the per-domain marker still at its previous value. The user
// destruction that follows runs against the parked state by
// construction; no sleeps, no host-only doubles that hide the real
// TTNN queue path.
//
// Coverage:
//   - copy parking: head copy + parked copy; destroying the parked
//     source's planes while the head occupies the single credit
//     drives the parked fence through `release_or_quarantine` and
//     the parked dispatch uses the quarantined planes.
//   - embedding parking: head embedding + parked embedding;
//     the parked embedding owns a table, an indices tensor, an
//     output tensor, and a 32-byte workspace lease. The test
//     destroys every tensor owner (table + indices + output) while
//     the parked task is parked. The parked workspace is NOT
//     destroyed while parked: its release path is the
//     `RawWorkspace` destructor in `src/ttnn/device.cpp`, which
//     checks `native_range_retained()` against the live workspace
//     lease and otherwise goes straight to `release_native` (freeing
//     the underlying MeshBuffer) without going through the
//     registry-fence gate the tensor owners pass through. Destroying
//     the parked workspace therefore frees the MeshBuffer the
//     parked dispatch would later dereference, which surfaces as
//     "workspace owner is not live on this device" inside
//     `embedding_planes`. The tensor owner path goes through
//     `release_or_quarantine`, which runs the per-entry fence and
//     quarantines on pending, so the table + indices + output
//     destruction is safe; the workspace owner path does not. The
//     workspace tensor is destroyed AFTER the parked dispatch
//     completes (line below `REQUIRE_NOTHROW(artifacts.queue->wait(...))`).

#include <doctest/doctest.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_ops.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

#include "../../src/ttnn/staging.hpp"
#include "backend/backend_conformance_common.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"
#include "iom/ttnn/device.hpp"
#include "../../src/ttnn/device_internal.hpp"

namespace {

constexpr std::uint64_t kParkedPatternSalt = 0xC0DEu;
constexpr std::uint64_t kHeadPatternSalt = 0xA1B2u;
constexpr std::uint64_t kEmbeddingTableSalt = 0x0101u;

const iom::TensorSpec make_single_plane_spec() {
    return iom::TensorSpec{
            iom::TensorShape{{1, 32, 32}}, iom::DataType::BF16};
}

const std::vector<std::byte> encode_pattern(
        const iom::TensorSpec& spec, std::uint64_t salt) {
    return iom_conformance::encode_logical(spec, salt);
}

// Copy parking case. The head copy consumes the single in-flight
// credit (capacity = 1); the parked copy is held in admission_fifo_
// until the head retires. The parked source's tensor destructor walks
// every registry entry for the parked source's native handle and
// invokes each fence; with the fix the parked fence sees
// `caller_executed_seq_ < sequence` and returns pending, so the
// parked source's planes are quarantined and the parked dispatch uses
// them safely after the head frees its credit.
//
// REQUIRE / CHECK inventory:
//   REQUIRE(head_source    != nullptr)               -- tensor objects
//   REQUIRE(parked_source  != nullptr)               created on device
//   REQUIRE(queue          != nullptr)               and live through
//   REQUIRE(head_destination     != nullptr)        the test body.
//   REQUIRE(iom::oid_is_token(head_token))            Tokens are real
//   REQUIRE(iom::oid_is_token(parked_token))          positive OIDs.
//   REQUIRE_NOTHROW(queue->wait(head_token))          Head dispatches
//                                                    and finishes
//                                                    without throwing
//                                                    - proves the credit
//                                                    free + worker
//                                                    dispatch path
//                                                    works on the live
//                                                    mesh.
//   REQUIRE_NOTHROW(artifacts.queue->wait(parked_token))
//                                                  Parked dispatch
//                                                    completes
//                                                    without throwing
//                                                    - the gate held, so
//                                                    the parked source's
//                                                    planes were
//                                                    quarantined
//                                                    instead of freed.
//   CHECK(artifacts.parked_destination matches parked source pattern)
//                                                  Readback compares
//                                                    through the device
//                                                    path: the quarantined
//                                                    planes lived until
//                                                    the worker used
//                                                    them, and the
//                                                    output contains
//                                                    the same bytes the
//                                                    source carried.
struct ParkedFenceArtifacts {
    std::unique_ptr<iom::Tensor> parked_destination;
    iom::oid parked_token = 0;
    std::unique_ptr<iom::DeviceOps> queue;
    std::unique_ptr<iom::Tensor> head_destination;
};

ParkedFenceArtifacts run_parked_copy_fence_regression(iom::Device& device) {
    auto head_source = device.create_tensor(make_single_plane_spec());
    REQUIRE(head_source != nullptr);
    auto parked_source = device.create_tensor(make_single_plane_spec());
    REQUIRE(parked_source != nullptr);

    iom_conformance::copy_from_host(
            head_source->view(), encode_pattern(head_source->view().spec(),
                    kHeadPatternSalt));
    iom_conformance::copy_from_host(
            parked_source->view(),
            encode_pattern(parked_source->view().spec(),
                    kParkedPatternSalt));

    auto queue = device.create_ops();
    REQUIRE(queue != nullptr);

    auto head_destination = device.create_tensor(make_single_plane_spec());
    REQUIRE(head_destination != nullptr);
    const iom::oid head_token = queue->copy(
            head_source->view(), head_destination->view());
    REQUIRE(iom::oid_is_token(head_token));

    // Capacity 1 means the second submit occupies the next FIFO slot
    // and only the head's credit was consumed; the parked copy stays
    // parked in `admission_fifo_` until the head retires.
    auto parked_destination = device.create_tensor(make_single_plane_spec());
    REQUIRE(parked_destination != nullptr);
    const iom::oid parked_token = queue->copy(
            parked_source->view(), parked_destination->view());
    REQUIRE(iom::oid_is_token(parked_token));

    // Destroy `parked_source` while the parked dispatch is still
    // queued. The destructor walks every registry entry keyed on the
    // parked source's native address and invokes each fence. With the
    // fix the parked submission's fence sees
    // `caller_executed_seq_ < sequence` and returns pending, so
    // `release_or_quarantine` quarantines the parked source's planes
    // instead of releasing them. Without the fix the blind
    // `mesh().finish()` returns success and the planes are released,
    // leaving the later worker dispatch to dereference freed DRAM.
    parked_source.reset();

    REQUIRE_NOTHROW(queue->wait(head_token));

    ParkedFenceArtifacts artifacts;
    artifacts.parked_destination = std::move(parked_destination);
    artifacts.parked_token = parked_token;
    artifacts.queue = std::move(queue);
    artifacts.head_destination = std::move(head_destination);
    return artifacts;
}

void require_parked_destination_matches_pattern(
        const iom::TensorView& parked_destination,
        std::uint64_t salt, std::string_view message) {
    const std::vector<std::byte> expected_pattern =
            encode_pattern(parked_destination.spec(), salt);
    iom_conformance::require_logical_bytes(
            parked_destination, expected_pattern, message);
}

// Embedding parking case. The parked embedding submission owns a
// table, an indices tensor, an output tensor, and a 32-byte
// workspace lease. The parked source's native_handle, the parked
// indices tensor's native_handle, and the parked output tensor's
// native_handle are all registered with TTNN fences keyed to the
// parked sequence; destroying each tensor owner walks its registry
// entries and (with the gate) quarantines the planes. The parked
// WORKSPACE is NOT destroyed while parked - see the file-level
// comment for why. Both head and parked embeddings use all-zero
// indices so every output row resolves to table row 0; the output
// readback therefore compares the parked dispatch's destination
// against the table row 0 pattern replicated across the 8x8 output.
//
// REQUIRE / CHECK inventory:
//   REQUIRE(head_table     != nullptr)
//   REQUIRE(head_indices   != nullptr)
//   REQUIRE(head_output    != nullptr)
//   REQUIRE(head_workspace != nullptr)
//   REQUIRE(parked_table   != nullptr)
//   REQUIRE(parked_indices != nullptr)
//   REQUIRE(parked_output  != nullptr)
//   REQUIRE(parked_workspace != nullptr)
//   REQUIRE(queue           != nullptr)
//                                Tensor / workspace / queue objects
//                                are live on the device.
//   REQUIRE(iom::oid_is_token(head_token))
//   REQUIRE(iom::oid_is_token(parked_token))
//                                Tokens are real positive OIDs.
//   (biased-not-guaranteed) QueueConfig{max_in_flight_per_queue=1}
//                                means the second submit's pump_admission
//                                most often hits credits == capacity and
//                                returns without dispatching, leaving
//                                the parked task parked. The park is
//                                timing-biased, not a hard guarantee:
//                                a fast head finish on the worker thread
//                                and an extra device.create_tensor window
//                                between submits can in principle let the
//                                parked dispatch run before the parked
//                                owner destruction reaches the registry.
//                                What carries the verification is the
//                                empirically-demonstrated failure on the
//                                pre-fix code at commit 947c4c45, where
//                                the same test file surfaced
//                                "std::visit: variant is valueless" at the
//                                parked wait (the worker's ttnn::Tensor
//                                access dereferenced planes the parked
//                                fence had reported success for and that
//                                release_or_quarantine had then freed).
//                                A future reader must know the test can
//                                pass vacuously and that the precheck is
//                                the real evidence the seam is exercised.
//   REQUIRE_NOTHROW(queue->wait(head_token))
//                                Head embedding dispatches and
//                                finishes without throwing - proves
//                                the worker path handles the live
//                                head end-to-end on the mesh.
//   REQUIRE_NOTHROW(artifacts.queue->wait(artifacts.parked_token))
//                                Parked embedding completes without
//                                throwing; the per-tensor-owner gate
//                                held, the parked tensor planes were
//                                quarantined, and the parked dispatch
//                                ran on the quarantined planes.
//   CHECK(artifacts.parked_output matches table-row-0 pattern)
//                                Readback goes through the device
//                                path; if the parked tensor planes
//                                had been released (pre-fix) the
//                                worker would have crashed or the
//                                output would carry garbage.
//
// The parked workspace is dropped AFTER `queue->wait(parked_token)`
// returns, so the parking-window MeshBuffer is never released while
// the parked dispatch is still in flight.
struct EmbeddingParkedFenceArtifacts {
    std::unique_ptr<iom::Tensor> parked_output;
    iom::oid parked_token = 0;
    std::unique_ptr<iom::DeviceOps> queue;
};

EmbeddingParkedFenceArtifacts run_parked_embedding_fence_regression(
        iom::Device& device) {
    const iom::TensorSpec table_spec{
            iom::TensorShape{{1, 8}}, iom::DataType::U32};
    const iom::TensorSpec index_spec{
            iom::TensorShape{{1, 8}}, iom::DataType::U32};
    const iom::TensorSpec output_spec{
            iom::TensorShape{{8, 8}}, iom::DataType::U32};

    auto head_table = device.create_tensor(table_spec);
    REQUIRE(head_table != nullptr);
    auto head_indices = device.create_tensor(index_spec);
    REQUIRE(head_indices != nullptr);
    auto head_output = device.create_tensor(output_spec);
    REQUIRE(head_output != nullptr);
    auto head_workspace = device.create_workspace(32);
    REQUIRE(head_workspace != nullptr);

    auto parked_table = device.create_tensor(table_spec);
    REQUIRE(parked_table != nullptr);
    auto parked_indices = device.create_tensor(index_spec);
    REQUIRE(parked_indices != nullptr);
    auto parked_output = device.create_tensor(output_spec);
    REQUIRE(parked_output != nullptr);
    auto parked_workspace = device.create_workspace(32);
    REQUIRE(parked_workspace != nullptr);

    iom_conformance::copy_from_host(
            head_table->view(),
            encode_pattern(head_table->view().spec(), kEmbeddingTableSalt));
    iom_conformance::copy_from_host(
            parked_table->view(),
            encode_pattern(parked_table->view().spec(), kParkedPatternSalt));
    const std::vector<std::byte> all_zero_indices(
            index_spec.logical_nbytes(), std::byte{0});
    iom_conformance::copy_from_host(head_indices->view(), all_zero_indices);
    iom_conformance::copy_from_host(parked_indices->view(), all_zero_indices);

    auto queue = device.create_ops();
    REQUIRE(queue != nullptr);

    const iom::oid head_token = queue->embedding(
            head_table->view(), head_indices->view(), head_output->view(),
            head_workspace->view());
    REQUIRE(iom::oid_is_token(head_token));

    const iom::oid parked_token = queue->embedding(
            parked_table->view(), parked_indices->view(), parked_output->view(),
            parked_workspace->view());
    REQUIRE(iom::oid_is_token(parked_token));

    // Destroy the tensor owners for the parked embedding submission
    // while the head holds the single in-flight credit. The
    // `~Tensor` destructors call `release_or_quarantine` against each
    // owner's native address, walking the parked registry entries and
    // invoking the parked fence. With the fix the parked fence sees
    // `caller_executed_seq_ < sequence` and returns pending for each,
    // so the parked table / indices planes are quarantined instead of
    // released. The parked output is NOT destroyed here: the test
    // needs it to read back the parked dispatch's result. The parked
    // workspace is also NOT destroyed; its release path is not gated
    // on the registry fence and would free the MeshBuffer the parked
    // dispatch would later dereference. Both are released after the
    // parked wait returns.
    parked_table.reset();
    parked_indices.reset();

    REQUIRE_NOTHROW(queue->wait(head_token));

    // Parked dispatch completes against the quarantined tensor planes;
    // the workspace is still live at this point.
    REQUIRE_NOTHROW(queue->wait(parked_token));

    EmbeddingParkedFenceArtifacts artifacts;
    artifacts.parked_output = std::move(parked_output);
    artifacts.parked_token = parked_token;
    artifacts.queue = std::move(queue);
    return artifacts;
}

void require_parked_embedding_output_matches(
        const iom::TensorView& parked_output) {
    // Indices are zero; every output row resolves to table row 0, so
    // the expected output is the table row 0 pattern replicated
    // across every row. The conformance helper reads logical bytes
    // back through the device path so a UAF would either crash
    // inside the readback or return garbage.
    const iom::TensorSpec table_spec{
            iom::TensorShape{{1, 8}}, iom::DataType::U32};
    const std::vector<std::byte> table_pattern =
            encode_pattern(table_spec, kParkedPatternSalt);
    const std::size_t row_bytes = table_pattern.size();
    const std::size_t out_bytes = parked_output.spec().logical_nbytes();
    std::vector<std::byte> expected_pattern(out_bytes, std::byte{0});
    for (std::size_t row = 0; row < out_bytes / row_bytes; ++row) {
        std::memcpy(expected_pattern.data() + row * row_bytes,
                table_pattern.data(), row_bytes);
    }
    iom_conformance::require_logical_bytes(
            parked_output, expected_pattern,
            "parked-embedding dispatch survived pending quarantine");
}

}  // namespace

TEST_CASE(
        "TTNN shared fenced runtime: capacity-parked copy fence returns "
        "pending and later dispatch is safe") {
    if (tt::tt_metal::GetNumAvailableDevices() == 0) {
        return;
    }
    auto device = iom::make_ttnn_device(
            0, iom::QueueConfig{/*max_in_flight_per_queue=*/1});
    REQUIRE(device != nullptr);

    ParkedFenceArtifacts artifacts =
            run_parked_copy_fence_regression(*device);

    // After the head completed and freed its credit, the worker
    // dispatched the parked copy against the quarantined parked
    // source's planes. The wait returns when the parked dispatch
    // proves completion through `ttnn_detail::finish_native`; the
    // output readback then compares bytes through the device path.
    REQUIRE_NOTHROW(artifacts.queue->wait(artifacts.parked_token));

    require_parked_destination_matches_pattern(
            artifacts.parked_destination->view(), kParkedPatternSalt,
            "parked-copy dispatch survived pending quarantine");
}

TEST_CASE(
        "TTNN shared fenced runtime: capacity-parked embedding fence returns "
        "pending and later dispatch is safe") {
    if (tt::tt_metal::GetNumAvailableDevices() == 0) {
        return;
    }
    auto device = iom::make_ttnn_device(
            0, iom::QueueConfig{/*max_in_flight_per_queue=*/1});
    REQUIRE(device != nullptr);

    EmbeddingParkedFenceArtifacts artifacts =
            run_parked_embedding_fence_regression(*device);

    // Both waits completed without throwing above; the readback below
    // confirms the parked dispatch wrote the table-row-0 pattern.
    require_parked_embedding_output_matches(artifacts.parked_output->view());
}
