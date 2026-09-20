#include <doctest/doctest.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_embedding.hpp"
#include "backend/backend_conformance_linear.hpp"
#include "backend/backend_conformance_other.hpp"
#include "backend/backend_conformance_rmsnorm.hpp"
#include "backend/backend_conformance_rope.hpp"
#include "backend/backend_conformance_rope_contract.hpp"
#include "backend/backend_conformance_add_gpu.hpp"
#include "backend/backend_conformance_model_loading.hpp"
#include "backend/backend_conformance_cache_append.hpp"

#include "iom/cpu/device.hpp"
#include "iom/cuda/device.hpp"
#include "cuda/copy.hpp"
#include "cuda/driver.hpp"
#include "iom/gpu_algorithm.hpp"

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

// Every standard-GPU conformance fixture reserves this tensor-data arena:
// the largest concurrent set is three 8192x4096 F32 tensors plus one
// quarantined operand (512 MiB), and the fixture also keeps a foreign
// device alive with the same budget.
constexpr std::size_t kConformanceArenaBytes = 640u * 1024 * 1024;

// The device ordinal every CUDA conformance fixture selects. The declaration's
// runtime BF16 fact is a property of that exact ordinal.
constexpr std::uint32_t kCudaConformanceOrdinal = 0;

// The driver's own statement of one device's BF16 WMMA fact, read from the
// runtime API and never from a production helper: the native BF16 linear
// specialization requires the device's BF16 tensor-core facility, which is
// compute capability 8.0 or newer. The compiled-image half of the fact is a
// build property, not a device one, so it is not restated here.
[[nodiscard]] bool cuda_bf16_wmma_device_fact(std::uint32_t ordinal) {
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor,
                static_cast<int>(ordinal))
                != cudaSuccess
            || cudaDeviceGetAttribute(
                       &minor, cudaDevAttrComputeCapabilityMinor,
                       static_cast<int>(ordinal))
                    != cudaSuccess) {
        return false;
    }
    return major >= 8;
}

// One execution-connected native evidence record of a submitted BF16
// projection. The record carries the runtime device and toolchain facts, the
// fixture geometry and its padded extent, the accepted OID and its sequence,
// the kernel symbol, and the launch geometry the profiler run must observe for
// exactly this submission, so an observed native matrix instruction can be
// attributed to this execution and to no other.
struct CudaBf16NativeRecord {
    const char* backend = "CUDA";
    const char* layout = "ordinary";
    std::size_t planes = 0;
    std::size_t source_rows = 0;
    std::size_t inner = 0;
    std::size_t outer = 0;
    std::size_t heads = 0;
    std::size_t head_dim = 0;
    std::size_t start_row = 0;
    std::size_t rows = 0;
    std::size_t padded_rows = 0;
    std::size_t padded_columns = 0;
    std::size_t padded_inner = 0;
    std::size_t blocks = 0;
    iom::oid token = 0;
    bool supported = false;
};

// Emits one record as stable `key=value` lines. The suite's own assertion is
// the numeric comparison; this emission is the native evidence artifact the
// profiler command is recorded beside, and it names the exact submission the
// profiler must attribute its observed instruction to.
void emit_cuda_bf16_record(const CudaBf16NativeRecord& record) {
    std::printf(
            "cuda-linear-bf16-record backend=%s layout=%s P=%zu T=%zu I=%zu "
            "O=%zu H=%zu D=%zu s=%zu R=%zu Rp=%zu Op=%zu Ip=%zu blocks=%zu "
            "oid=%llu sequence=%llu kernel=%s facility=%s image_arch=%u\n",
            record.backend, record.layout, record.planes, record.source_rows,
            record.inner, record.outer, record.heads, record.head_dim,
            record.start_row, record.rows, record.padded_rows,
            record.padded_columns, record.padded_inner, record.blocks,
            static_cast<unsigned long long>(record.token),
            static_cast<unsigned long long>(
                    iom_conformance::token_sequence(record.token)),
            "standard_tiled_linear_bf16_kernel",
            record.supported ? "bf16-wmma-supported" : "unsupported",
            iom::cuda_detail::linear_bf16_wmma_image_arch());
}

struct CudaDevices {
    iom_conformance::TrafficGate gate;
    HostAllocator reference_allocator{gate};
    std::unique_ptr<iom::Device> reference =
            iom::make_cpu_device(reference_allocator);
    std::unique_ptr<iom::Device> candidate =
            iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    std::unique_ptr<iom::Device> foreign =
            iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{kConformanceArenaBytes});

    [[nodiscard]] iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};

class CudaStorageOracle final
        : public iom_conformance::AcceleratorStorageOracle {
public:
    void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        REQUIRE_EQ(encoded.size(), bytes);
        REQUIRE(cudaMemcpy(
                        view.native_handle(), encoded.data(), bytes,
                        cudaMemcpyHostToDevice)
                == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        const std::size_t bytes = owner_spec().tiled_storage_nbytes();
        std::vector<std::byte> result(bytes);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        REQUIRE(cudaMemcpy(
                        result.data(), view.native_handle(), bytes,
                        cudaMemcpyDeviceToHost)
                == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        return result;
    }
};


}  // namespace
TEST_CASE("Device::supported_data_types returns the per-backend 23-entry span") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    iom_conformance::TrafficGate gate;
    const std::unique_ptr<iom::Device> candidate =
            iom::make_cuda_device(
                    0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    iom_conformance::require_standard_capabilities(
            candidate->supported_data_types());
}

TEST_CASE("CUDA conformance: storage and host transfers for every leaf type") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: storage oracle identifies perturbed transfer map") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const std::span<const iom::DataType> one_type =
            devices.candidate->supported_data_types().subspan(0, 1);
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), one_type, &devices.gate);

    CudaStorageOracle direct;
    iom_conformance::PermutingStorageOracle perturbed(
            direct, iom_conformance::swap_first_adjacent_slots);
    CHECK_FALSE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, perturbed, &devices.gate,
            false, false));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: storage oracle covers every leaf width and padded shape") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            oracle, &devices.gate));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: asynchronous copies against the CPU reference") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate, &oracle);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: copy validation fails before writes and sequences") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA copy reservation failures roll back before native work") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::U8};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    const std::vector<std::byte> source_bytes(
            spec.logical_nbytes(), static_cast<std::byte>(0x11));
    const std::vector<std::byte> destination_bytes(
            spec.logical_nbytes(), static_cast<std::byte>(0x22));
    iom_conformance::copy_from_host(source->view(), source_bytes);
    iom_conformance::copy_from_host(destination->view(), destination_bytes);
    auto queue = devices.candidate->create_ops();

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::registration);
    CHECK_EQ(
            queue->copy(source->view(), destination->view()),
            iom::to_oid(iom::OidError::ResourceExhausted));
    std::vector<std::byte> observed(spec.logical_nbytes());
    iom_conformance::copy_to_host(destination->view(), observed);
    CHECK(observed == destination_bytes);

    const iom::oid first = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(first), 1);
    CHECK_NOTHROW(queue->wait(first));

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::outcome_insertion);
    CHECK_EQ(
            queue->copy(source->view(), destination->view()),
            iom::to_oid(iom::OidError::ResourceExhausted));
    const iom::oid second = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(second), 2);
    CHECK_NOTHROW(queue->wait(second));
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: transfer failures keep metadata and ownership") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_transfer_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: deferred queue lifetime and stability") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: compute methods reject capability without submitting") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate, "CUDA", true, true);
    CHECK_FALSE(devices.gate.armed());
}

// CUDA's declared embedding expectation: the complete 23-payload/12-index
// matrix and exact `{32, 32}` status-workspace contract. The native CUDA
// gather copies raw words and reports deferred queued-ID bounds failures.
constexpr iom_conformance::EmbeddingDeclaration kCudaEmbeddingDeclaration{
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom::WorkspaceRequirements{32, 32}};

TEST_CASE("CUDA conformance: embedding lookup reference, admission, and lifetime") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::run_embedding_conformance(
            devices.conformance(), kCudaEmbeddingDeclaration, &devices.gate,
            &oracle);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: cache append reference, admission, and lifetime") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::CacheAppendFaultSeam fault_seam{
            [](iom::DeviceOps&) {
                iom::cuda_detail::inject_submission_fault_for_testing(
                        iom::cuda_detail::SubmissionFault::third_plane_launch);
            }};
    iom_conformance::CacheAppendConformanceConfig config{
            devices.conformance(),
            devices.candidate->supported_data_types(),
            &oracle,
            fault_seam,
            {},
            nullptr};
    iom_conformance::run_cache_append_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA cache append retains launch failures") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    constexpr iom::DataType type = iom::DataType::F32;
    const iom::TensorSpec source_spec{
            iom::TensorShape{{1, 1, 1, 17}}, type};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{1, 1, 17, 17}}, type};
    auto source = devices.candidate->create_tensor(source_spec);
    auto destination = devices.candidate->create_tensor(destination_spec);
    const std::vector<std::byte> source_bytes =
            iom_conformance::encode_cache_append_logical(source_spec, 0x761);
    const std::vector<std::byte> destination_before =
            iom_conformance::encode_logical(destination_spec, 0x762);
    iom_conformance::copy_from_host(source->view(), source_bytes);
    iom_conformance::copy_from_host(
            destination->view(), destination_before);
    auto queue = devices.candidate->create_ops();

    try {
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::third_plane_launch);
        const iom::oid token =
                queue->cache_append(source->view(), destination->view(), 1);
        REQUIRE(iom::oid_is_token(token));
        iom_conformance::expect_repeated_runtime_failure(*queue, token);
    } catch (...) {
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::none);
        throw;
    }
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
    iom_conformance::require_logical_bytes(
            destination->view(), destination_before,
            "cache append launch failure leaves destination unchanged");

    auto recovery_destination =
            devices.candidate->create_tensor(destination_spec);
    const std::vector<std::byte> recovery_before =
            iom_conformance::encode_logical(destination_spec, 0x763);
    iom_conformance::copy_from_host(
            recovery_destination->view(), recovery_before);
    const iom::oid recovery = queue->cache_append(
            source->view(), recovery_destination->view(), 1);
    REQUIRE(iom::oid_is_token(recovery));
    CHECK_NOTHROW(queue->wait(recovery));
    CHECK_NOTHROW(queue->wait(recovery));
    iom_conformance::require_logical_bytes(
            recovery_destination->view(),
            iom_conformance::cache_append_expected_logical(
                    source_spec, destination_spec, 1, source_bytes,
                    recovery_before),
            "cache append recovered after launch failure");
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA cache append retains event-record failures") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    constexpr iom::DataType type = iom::DataType::F32;
    const iom::TensorSpec source_spec{
            iom::TensorShape{{1, 1, 1, 17}}, type};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{1, 1, 17, 17}}, type};
    auto source = devices.candidate->create_tensor(source_spec);
    auto destination = devices.candidate->create_tensor(destination_spec);
    const std::vector<std::byte> source_bytes =
            iom_conformance::encode_cache_append_logical(source_spec, 0x751);
    const std::vector<std::byte> destination_before =
            iom_conformance::encode_logical(destination_spec, 0x752);
    iom_conformance::copy_from_host(source->view(), source_bytes);
    iom_conformance::copy_from_host(
            destination->view(), destination_before);
    auto queue = devices.candidate->create_ops();

    try {
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::event_record);
        const iom::oid token =
                queue->cache_append(source->view(), destination->view(), 1);
        REQUIRE(iom::oid_is_token(token));
        iom_conformance::expect_repeated_runtime_failure(*queue, token);
    } catch (...) {
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::none);
        throw;
    }
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

    auto recovery_destination =
            devices.candidate->create_tensor(destination_spec);
    const std::vector<std::byte> recovery_before =
            iom_conformance::encode_logical(destination_spec, 0x753);
    iom_conformance::copy_from_host(
            recovery_destination->view(), recovery_before);
    const iom::oid recovery = queue->cache_append(
            source->view(), recovery_destination->view(), 1);
    REQUIRE(iom::oid_is_token(recovery));
    CHECK_NOTHROW(queue->wait(recovery));
    CHECK_NOTHROW(queue->wait(recovery));
    iom_conformance::require_logical_bytes(
            recovery_destination->view(),
            iom_conformance::cache_append_expected_logical(
                    source_spec, destination_spec, 1, source_bytes,
                    recovery_before),
            "cache append recovered after event-record failure");
    CHECK_FALSE(devices.gate.armed());
}


// CUDA's declared linear expectation: the complete 21-leaf applicable matrix
// through the twenty-leaf scalar path plus the native BF16 specialization, and
// the frozen `{0, 1}` scratch path for both. This revision's port queues all
// twenty-one leaves, so the implemented span is the complete target matrix: the
// suite compares the independent reference against real device results for
// every leaf, and the `BF16` comparison runs only where the selected device
// exposes the runtime BF16 WMMA facility. A device without that facility keeps
// `BF16` a capability rejection instead of a numerical claim, which is the
// genuine device fact and never a missing-implementation masquerade.
const iom_conformance::LinearDeclaration kCudaLinearDeclaration{
        iom_conformance::kLinearLeafSpan,
        iom_conformance::kLinearScalarLeafSpan,
        iom_conformance::kLinearNativeBf16Span,
        iom_conformance::kLinearLeafSpan,
        [](iom::DataType leaf) {
            return leaf != iom::DataType::BF16
                    || cuda_bf16_wmma_device_fact(kCudaConformanceOrdinal);
        },
        iom_conformance::LinearWorkspacePath::zero,
        iom_conformance::LinearWorkspacePath::zero};

TEST_CASE("CUDA conformance: linear projection reference, admission, and lifetime") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::run_linear_conformance(
            devices.conformance(), kCudaLinearDeclaration, &devices.gate,
            &oracle);
    CHECK_FALSE(devices.gate.armed());
}

// Focused CUDA observations of the scalar linear path that the shared suite
// cannot make through its backend-neutral declaration: an accepted submission
// runs on the queue's own in-order stream and is visible to the next
// submission without any host wait, the pure query reports the frozen `{0, 1}`
// requirement while a supplied owner range is invalid input before any
// effect, exactly the twenty non-BF16 applicable leaves are carried by this
// port, and a retained post-acceptance failure keeps its positive OID, its
// sequence, and the same error on every wait while its operands stay reusable.
TEST_CASE("CUDA linear keeps queue stream order, leaf capability, and retained failures") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::LinearOutputLayout ordinary = iom::LinearOutputLayout::ordinary;
    const iom::TensorSpec x_spec{
            iom::TensorShape{{2, 19, 3}}, iom::DataType::U8};
    const iom::TensorSpec w_spec{
            iom::TensorShape{{3, 3}}, iom::DataType::U8};
    const iom::TensorSpec out_spec{
            iom::TensorShape{{2, 17, 3}}, iom::DataType::U8};
    auto x = devices.candidate->create_tensor(x_spec);
    auto w = devices.candidate->create_tensor(w_spec);
    auto out = devices.candidate->create_tensor(out_spec);
    auto copied = devices.candidate->create_tensor(out_spec);
    auto queue = devices.candidate->create_ops();

    // The port needs no raw workspace, so the pure query reports exactly
    // `{0, 1}` and a supplied owner range is `InvalidArgument` before any
    // sequence, registration, or output effect.
    CHECK_EQ(
            queue->linear_workspace_requirements(
                    x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 3),
            (iom::WorkspaceRequirements{0, 1}));
    auto scratch = devices.candidate->create_workspace(64);
    const iom::oid supplied =
            queue->linear(
                    x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 3,
                    scratch->view());
    CHECK_EQ(supplied, iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_EQ(
            queue->linear_workspace_requirements(
                    x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 3),
            (iom::WorkspaceRequirements{0, 1}));

    // An exact identity weight makes the projection a pure row transport, so
    // a copy submitted behind it on the same queue observes the completed
    // selected rows without any oracle and without a host wait.
    std::vector<std::byte> x_bytes(x_spec.logical_nbytes());
    for (std::size_t index = 0; index < x_bytes.size(); ++index) {
        x_bytes[index] = std::byte{static_cast<unsigned char>(index * 7 + 1)};
    }
    std::vector<std::byte> w_identity(w_spec.logical_nbytes());
    w_identity[0] = std::byte{1};
    w_identity[4] = std::byte{1};
    w_identity[8] = std::byte{1};
    iom_conformance::copy_from_host(x->view(), x_bytes);
    iom_conformance::copy_from_host(w->view(), w_identity);
    std::vector<std::byte> expected(out_spec.logical_nbytes());
    for (std::size_t plane = 0; plane < 2; ++plane) {
        for (std::size_t row = 0; row < 17; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                expected[(plane * 17 + row) * 3 + column] =
                        x_bytes[(plane * 19 + (2 + row)) * 3 + column];
            }
        }
    }
    const iom::oid projection =
            queue->linear(
                    x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 3);
    REQUIRE(iom::oid_is_token(projection));
    const iom::oid transport = queue->copy(out->view(), copied->view());
    REQUIRE(iom::oid_is_token(transport));
    CHECK_EQ(
            iom_conformance::token_sequence(transport),
            iom_conformance::token_sequence(projection) + 1);
    CHECK_NOTHROW(queue->wait(transport));
    CHECK_NOTHROW(queue->wait(projection));
    iom_conformance::require_logical_bytes(
            copied->view(), expected, "FIFO projection transport");
    iom_conformance::require_logical_bytes(
            x->view(), x_bytes, "projection left its input unchanged");

    // Every applicable leaf of this revision is queued through the real queue:
    // the twenty leaves on the shared tiled scalar projection and `BF16` on the
    // native WMMA specialization when this device exposes the facility. The two
    // recognized inapplicable leaves stay `Unsupported` on every backend, and a
    // device without the facility keeps `BF16` the established explicit
    // rejection rather than any substitute.
    const bool bf16_facility =
            cuda_bf16_wmma_device_fact(kCudaConformanceOrdinal);
    for (const iom::DataType leaf : iom_conformance::kLinearScalarLeafSpan) {
        CAPTURE(static_cast<int>(leaf));
        auto leaf_x = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, leaf});
        auto leaf_w = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, leaf});
        auto leaf_out = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, leaf});
        CHECK_EQ(
                queue->linear_workspace_requirements(
                        leaf_x->view(), leaf_w->view(), leaf_out->view(), 0,
                        16, ordinary, 1, 16),
                (iom::WorkspaceRequirements{0, 1}));
        const iom::oid token = queue->linear(
                leaf_x->view(), leaf_w->view(), leaf_out->view(), 0, 16,
                ordinary, 1, 16);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
    }
    {
        auto leaf_x = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, iom::DataType::BF16});
        auto leaf_w = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, iom::DataType::BF16});
        auto leaf_out = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, iom::DataType::BF16});
        if (bf16_facility) {
            CHECK_EQ(
                    queue->linear_workspace_requirements(
                            leaf_x->view(), leaf_w->view(), leaf_out->view(),
                            0, 16, ordinary, 1, 16),
                    (iom::WorkspaceRequirements{0, 1}));
            const iom::oid token = queue->linear(
                    leaf_x->view(), leaf_w->view(), leaf_out->view(), 0, 16,
                    ordinary, 1, 16);
            REQUIRE(iom::oid_is_token(token));
            CHECK_NOTHROW(queue->wait(token));
        } else {
            CHECK_THROWS_AS(
                    (void)queue->linear_workspace_requirements(
                            leaf_x->view(), leaf_w->view(), leaf_out->view(),
                            0, 16, ordinary, 1, 16),
                    std::runtime_error);
            CHECK_EQ(
                    queue->linear(
                            leaf_x->view(), leaf_w->view(), leaf_out->view(),
                            0, 16, ordinary, 1, 16),
                    iom::to_oid(iom::OidError::Unsupported));
        }
    }
    for (const iom::DataType leaf :
         {iom::DataType::BOOL, iom::DataType::F8_E8M0}) {
        CAPTURE(static_cast<int>(leaf));
        auto leaf_x = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, leaf});
        auto leaf_w = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, leaf});
        auto leaf_out = devices.candidate->create_tensor(iom::TensorSpec{
                iom::TensorShape{{16, 16}}, leaf});
        CHECK_THROWS_AS(
                (void)queue->linear_workspace_requirements(
                        leaf_x->view(), leaf_w->view(), leaf_out->view(), 0,
                        16, ordinary, 1, 16),
                std::runtime_error);
        CHECK_EQ(
                queue->linear(
                        leaf_x->view(), leaf_w->view(), leaf_out->view(), 0, 16,
                        ordinary, 1, 16),
                iom::to_oid(iom::OidError::Unsupported));
    }
    {
        // A recognized non-`NONE` quantization format is a capability
        // rejection for the same request, after structural validation.
        const iom_conformance::LinearQuantizationQualification qualification(
                *x, *w, *out);
        CHECK_EQ(
                queue->linear(
                        x->view(), w->view(), out->view(), 2, 17, ordinary, 1,
                        3),
                iom::to_oid(iom::OidError::Unsupported));
    }

    // A retained post-acceptance failure keeps the accepted OID positive,
    // consumes its sequence, repeats the same error on every wait, and leaves
    // the operands reusable for a following submission.
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::event_record);
    const iom::oid failed = queue->linear(
            x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 3);
    REQUIRE(iom::oid_is_token(failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, failed);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
    const iom::oid recovered = queue->linear(
            x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 3);
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(queue->wait(recovered));
    iom_conformance::require_logical_bytes(
            out->view(), expected, "recovered projection transport");
    CHECK_FALSE(devices.gate.armed());
}

// The unsupported-device capability path of the native BF16 specialization,
// observed on a device that has the facility: the runtime fact is injected
// absent for exactly one queue creation, so that queue reports the established
// `Unsupported` for `BF16` in both directions — the pure query throws and the
// submission returns the negative OID with no output effect and no consumed
// sequence — while every other leaf on the same queue keeps the established
// scalar path and still computes its real result. The injection is a capability
// fact and not a hardware claim: `BF16` never substitutes a scalar, elementwise,
// cuBLAS, or host route, and a fresh queue resolves the device truth again.
TEST_CASE("CUDA BF16 linear is Unsupported on a device without the WMMA facility") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::LinearOutputLayout ordinary = iom::LinearOutputLayout::ordinary;
    const iom::TensorSpec x_spec{
            iom::TensorShape{{2, 19, 3}}, iom::DataType::BF16};
    const iom::TensorSpec w_spec{
            iom::TensorShape{{10, 3}}, iom::DataType::BF16};
    const iom::TensorSpec out_spec{
            iom::TensorShape{{2, 17, 10}}, iom::DataType::BF16};
    auto x = devices.candidate->create_tensor(x_spec);
    auto w = devices.candidate->create_tensor(w_spec);
    auto out = devices.candidate->create_tensor(out_spec);
    const std::vector<std::byte> out_poison = iom_conformance::linear_logical_image(
            iom::DataType::BF16, out_spec.shape.element_count(), 0x51A7ull,
            true);
    iom_conformance::copy_from_host(x->view(), std::vector<std::byte>(
            x_spec.logical_nbytes(), std::byte{0x3C}));
    iom_conformance::copy_from_host(w->view(), std::vector<std::byte>(
            w_spec.logical_nbytes(), std::byte{0x3C}));
    iom_conformance::copy_from_host(out->view(), out_poison);

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::bf16_wmma_facility);
    auto queue = devices.candidate->create_ops();
    CHECK_THROWS_AS(
            (void)queue->linear_workspace_requirements(
                    x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 10),
            std::runtime_error);
    CHECK_EQ(
            queue->linear(
                    x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 10),
            iom::to_oid(iom::OidError::Unsupported));
    iom_conformance::require_logical_bytes(
            out->view(), out_poison,
            "an unsupported BF16 submission changed the output");

    // The same queue and the same three operand shapes carry the scalar path:
    // an `F32` projection of the identical request on that queue is accepted
    // and computes its real result, so the rejection belongs to the `BF16`
    // leaf and never to a disabled queue.
    const iom::TensorSpec scalar_x{
            iom::TensorShape{{2, 19, 3}}, iom::DataType::F32};
    const iom::TensorSpec scalar_w{
            iom::TensorShape{{3, 3}}, iom::DataType::F32};
    const iom::TensorSpec scalar_out{
            iom::TensorShape{{2, 17, 3}}, iom::DataType::F32};
    auto scalar_x_owner = devices.candidate->create_tensor(scalar_x);
    auto scalar_w_owner = devices.candidate->create_tensor(scalar_w);
    auto scalar_out_owner = devices.candidate->create_tensor(scalar_out);
    const std::vector<std::byte> scalar_x_bytes =
            iom_conformance::linear_logical_image(
                    iom::DataType::F32, scalar_x.shape.element_count(),
                    0x2A40ull, false);
    std::vector<std::byte> scalar_w_bytes(
            scalar_w.logical_nbytes(), std::byte{0});
    // An exact identity weight turns the projection into a row transport, so
    // the scalar path of this queue is numerically observable without a second
    // reference implementation. The four bytes are the IEEE-754
    // single-precision encoding of `1.0`, the same carrier every standard
    // backend stores.
    const std::byte unity[4] = {
            std::byte{0x00}, std::byte{0x00}, std::byte{0x80},
            std::byte{0x3F}};
    const auto set_weight = [&scalar_w_bytes, &unity](
                                    std::size_t row, std::size_t column) {
        const std::size_t index = row * 3 + column;
        for (std::size_t byte = 0; byte < 4; ++byte) {
            scalar_w_bytes[index * 4 + byte] = unity[byte];
        }
    };
    for (std::size_t index = 0; index < 3; ++index) {
        set_weight(index, index);
    }
    std::vector<std::byte> scalar_expected(scalar_out.logical_nbytes());
    const std::size_t element_bytes = 4;
    for (std::size_t plane = 0; plane < 2; ++plane) {
        for (std::size_t row = 0; row < 17; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                const std::size_t source =
                        ((plane * 19) + 2 + row) * 3 + column;
                const std::size_t destination =
                        ((plane * 17) + row) * 3 + column;
                std::copy_n(
                        scalar_x_bytes.data() + source * element_bytes,
                        element_bytes,
                        scalar_expected.data()
                                + destination * element_bytes);
            }
        }
    }
    iom_conformance::copy_from_host(scalar_x_owner->view(), scalar_x_bytes);
    iom_conformance::copy_from_host(scalar_w_owner->view(), scalar_w_bytes);
    const iom::oid scalar_token = queue->linear(
            scalar_x_owner->view(), scalar_w_owner->view(),
            scalar_out_owner->view(), 2, 17, ordinary, 1, 3);
    REQUIRE(iom::oid_is_token(scalar_token));
    CHECK_NOTHROW(queue->wait(scalar_token));
    iom_conformance::require_logical_bytes(
            scalar_out_owner->view(), scalar_expected,
            "the scalar path of a facility-less queue still projects");
    CHECK_FALSE(devices.gate.armed());

    // A fresh queue resolves the device's own truth again: the injected
    // absence was consumed by exactly one creation.
    if (cuda_bf16_wmma_device_fact(kCudaConformanceOrdinal)) {
        auto recovered_queue = devices.candidate->create_ops();
        const iom::oid recovered = recovered_queue->linear(
                x->view(), w->view(), out->view(), 2, 17, ordinary, 1, 10);
        REQUIRE(iom::oid_is_token(recovered));
        CHECK_NOTHROW(recovered_queue->wait(recovered));
    }
    CHECK_FALSE(devices.gate.armed());
}

// The execution-connected native evidence runs of the BF16 specialization: the
// canonical non-square fixture through the real queue/OID path for `R=1`, `15`,
// `16`, and `17` in both layouts. Every run is compared against the shared
// independent reference and emits its own record — runtime device and toolchain
// facts, geometry and padded extent, the exercised submission with its accepted
// OID, the kernel symbol, and the launch geometry — so a profiler observation
// of the native matrix instruction is attributed to exactly these executions.
// A device without the facility reports the absence instead of claiming the
// evidence:
//
//   ncu --kernel-name regex:standard_tiled_linear_bf16_kernel --launch-count 8 \
//       --print-source sass <build>/test/iom_cuda_conformance_tests \
//       --test-case="CUDA BF16 native evidence*"
TEST_CASE("CUDA BF16 native evidence: WMMA projections for R=1,15,16,17") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    if (!cuda_bf16_wmma_device_fact(kCudaConformanceOrdinal)) {
        std::printf(
                "cuda-linear-bf16-record backend=CUDA facility=unsupported "
                "reason=device-without-bf16-wmma\n");
        std::printf(
                "cuda-linear-bf16-environment backend=CUDA device_facility="
                "unsupported: a device below compute capability 8.0 reports "
                "the native evidence unavailable instead of claiming it\n");
        return;
    }
    CudaDevices devices;
    int device_index = 0;
    REQUIRE(cudaGetDevice(&device_index) == cudaSuccess);
    cudaDeviceProp properties{};
    REQUIRE(cudaGetDeviceProperties(&properties, device_index) == cudaSuccess);
    int runtime_version = 0;
    int driver_version = 0;
    REQUIRE(cudaRuntimeGetVersion(&runtime_version) == cudaSuccess);
    REQUIRE(cudaDriverGetVersion(&driver_version) == cudaSuccess);
    std::printf(
            "cuda-linear-bf16-environment backend=CUDA device=%s "
            "compute_capability=%d.%d runtime=%d driver=%d image_arch=%u\n",
            properties.name, properties.major, properties.minor,
            runtime_version, driver_version,
            iom::cuda_detail::linear_bf16_wmma_image_arch());

    const iom::LinearOutputLayout layouts[2] = {
            iom::LinearOutputLayout::ordinary,
            iom::LinearOutputLayout::head_planar};
    const std::size_t run_rows[4] = {1, 15, 16, 17};
    for (const iom::LinearOutputLayout layout : layouts) {
        for (const std::size_t rows : run_rows) {
            CAPTURE(static_cast<int>(layout));
            CAPTURE(rows);
            const std::size_t planes = 2;
            const std::size_t source_rows = 19;
            const std::size_t inner = 3;
            const std::size_t outer = 10;
            const std::size_t start_row = 2;
            const std::size_t columns =
                    layout == iom::LinearOutputLayout::head_planar ? 5 : outer;
            // The submission's `H`/`D`: ordinary mode is exactly `H=1`,
            // `D=O`, and head-planar mode is the checked `O=H*D` with the
            // head axis inserted into the output.
            const std::size_t heads =
                    layout == iom::LinearOutputLayout::head_planar ? 2 : 1;
            const std::size_t head_dim = columns;
            std::vector<std::size_t> x_dimensions{planes, source_rows, inner};
            std::vector<std::size_t> out_dimensions{planes};
            if (layout == iom::LinearOutputLayout::head_planar) {
                out_dimensions.push_back(heads);
            }
            out_dimensions.push_back(rows);
            out_dimensions.push_back(columns);
            auto x_owner = devices.candidate->create_tensor(
                    iom::TensorSpec{
                            iom::TensorShape{std::move(x_dimensions)},
                            iom::DataType::BF16});
            auto w_owner = devices.candidate->create_tensor(iom::TensorSpec{
                    iom::TensorShape{{outer, inner}}, iom::DataType::BF16});
            auto out_owner = devices.candidate->create_tensor(
                    iom::TensorSpec{
                            iom::TensorShape{std::move(out_dimensions)},
                            iom::DataType::BF16});

            const std::uint64_t salt =
                    0x6F00ull + rows * 0x10ull + (heads == 2 ? 1ull : 0ull);
            const std::vector<std::byte> x_bytes =
                    iom_conformance::linear_logical_image(
                            iom::DataType::BF16,
                            x_owner->view().spec().shape.element_count(), salt,
                            true);
            const std::vector<std::byte> w_bytes =
                    iom_conformance::linear_logical_image(
                            iom::DataType::BF16,
                            w_owner->view().spec().shape.element_count(),
                            salt ^ 0x1234ull, true);
            const std::vector<std::byte> out_poison =
                    iom_conformance::linear_logical_image(
                            iom::DataType::BF16,
                            out_owner->view().spec().shape.element_count(),
                            salt ^ 0x4321ull, true);
            const std::vector<std::byte> x_view_bytes =
                    iom_conformance::linear_view_image(
                            x_owner->view(), x_bytes);
            const iom_conformance::LinearRawImage x_image =
                    iom_conformance::linear_padded_image(
                            iom::DataType::BF16, planes,
                            iom_conformance::linear_pad16(source_rows),
                            iom_conformance::linear_pad16(inner), source_rows,
                            inner, x_view_bytes, salt ^ 0x0F0Full);
            const iom_conformance::LinearRawImage w_image =
                    iom_conformance::linear_padded_image(
                            iom::DataType::BF16, 1,
                            iom_conformance::linear_pad16(outer),
                            iom_conformance::linear_pad16(inner), outer, inner,
                            w_bytes, salt ^ 0xF0F0ull);
            const iom_conformance::LinearOracleRequest request{
                    iom::DataType::BF16, planes, source_rows, inner, outer,
                    start_row, rows, heads, head_dim, layout};
            const iom_conformance::LinearReference reference =
                    iom_conformance::linear_reference(
                            request, x_image, w_image);

            iom_conformance::copy_from_host(x_owner->view(), x_bytes);
            iom_conformance::copy_from_host(w_owner->view(), w_bytes);
            iom_conformance::copy_from_host(out_owner->view(), out_poison);
            auto queue = devices.candidate->create_ops();
            const iom::oid token = queue->linear(
                    x_owner->view(), w_owner->view(), out_owner->view(),
                    start_row, rows, layout, heads, head_dim);
            REQUIRE(iom::oid_is_token(token));
            CHECK_NOTHROW(queue->wait(token));
            const std::optional<std::string> mismatch =
                    iom_conformance::linear_compare_image(
                            request, iom_conformance::read_logical(
                                             out_owner->view()),
                            reference,
                            std::string("native evidence ")
                                    + (layout
                                               == iom::LinearOutputLayout::
                                                          head_planar
                                       ? "head_planar"
                                       : "ordinary"));
            REQUIRE_MESSAGE(!mismatch.has_value(), mismatch.value_or(""));

            const std::size_t row_tiles = (rows + 15) / 16;
            const std::size_t column_tiles = (columns + 15) / 16;
            const std::size_t units = planes * heads * row_tiles * column_tiles;
            CudaBf16NativeRecord record;
            record.layout = layout == iom::LinearOutputLayout::head_planar
                    ? "head_planar"
                    : "ordinary";
            record.planes = planes;
            record.source_rows = source_rows;
            record.inner = inner;
            record.outer = outer;
            record.heads = heads;
            record.head_dim = head_dim;
            record.start_row = start_row;
            record.rows = rows;
            record.padded_rows = iom_conformance::linear_pad16(rows);
            record.padded_columns = iom_conformance::linear_pad16(columns);
            record.padded_inner = iom_conformance::linear_pad16(inner);
            record.blocks = units < 65535 ? units : 65535;
            record.token = token;
            record.supported = true;
            emit_cuda_bf16_record(record);
        }
    }
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA embedding status transfer failure retires safely") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::TensorSpec table_spec{
            iom::TensorShape{{4, 8}}, iom::DataType::U8};
    const iom::TensorSpec index_spec{
            iom::TensorShape{{1, 8}}, iom::DataType::U32};
    const iom::TensorSpec output_spec{
            iom::TensorShape{{8, 8}}, iom::DataType::U8};
    auto table = devices.candidate->create_tensor(table_spec);
    auto indices = devices.candidate->create_tensor(index_spec);
    auto output = devices.candidate->create_tensor(output_spec);
    iom_conformance::copy_from_host(
            table->view(),
            std::vector<std::byte>(table_spec.logical_nbytes()));
    iom_conformance::copy_from_host(
            indices->view(),
            std::vector<std::byte>(index_spec.logical_nbytes()));
    auto workspace = devices.candidate->create_workspace(32);
    auto queue = devices.candidate->create_ops();

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::embedding_status_copy);
    const iom::oid failed = queue->embedding(
            table->view(), indices->view(), output->view(), workspace->view());
    REQUIRE(iom::oid_is_token(failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, failed);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

    const iom::oid recovered = queue->embedding(
            table->view(), indices->view(), output->view(), workspace->view());
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(queue->wait(recovered));
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::third_plane_launch);
    const iom::oid native_failed = queue->embedding(
            table->view(), indices->view(), output->view(), workspace->view());
    REQUIRE(iom::oid_is_token(native_failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, native_failed);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

}

TEST_CASE("CUDA conformance: RMSNorm reference, admission, and lifetime") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::RmsNormConformanceConfig config{
            devices.conformance(),
            iom_conformance::kRmsNormAllLeafSpan,
            &devices.gate,
            &oracle};
    // The CUDA testing seam reaches the real RMSNorm path: the shared GPU
    // queue dispatch records the submission's completion event on the kernel's
    // own stream (`Policy::record_event` in src/shared/gpu_queue_operations.inl)
    // after the shared tiled RMSNorm launch, and that record step is exactly
    // what `SubmissionFault::event_record` fails, so the fault is consumed by
    // the real RMSNorm submission and the accepted token fails. The seam is a
    // counted fault with no consumption accessor, so the shared scenario
    // proves consumption behaviorally (disarmed control succeeds; armed
    // accepted token fails and keeps failing; cleared queue recovers).
    config.native_failure = iom_conformance::RmsNormNativeFailureSeam{
            [] {
                iom::cuda_detail::inject_submission_fault_for_testing(
                        iom::cuda_detail::SubmissionFault::event_record);
            },
            [] {
                iom::cuda_detail::inject_submission_fault_for_testing(
                        iom::cuda_detail::SubmissionFault::none);
            },
            "cuda_detail::SubmissionFault::event_record",
            {}};
    iom_conformance::run_rmsnorm_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: rotary position encoding reference and storage") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    iom_conformance::rope_reference::run_rope_conformance(
            *devices.candidate,
            iom_conformance::rope_reference::kRopeCudaExpectedSupported,
            oracle);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA conformance: full shared suite") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    CudaStorageOracle oracle;
    const iom_conformance::RopeContractConformanceConfig rope_contract{
            devices.conformance(),
            iom_conformance::rope_reference::kRopeCudaExpectedSupported,
            &devices.gate,
            iom_conformance::RopeNativeFailureSeam{
                    [] {
                        iom::cuda_detail::inject_submission_fault_for_testing(
                                iom::cuda_detail::SubmissionFault::event_record);
                    },
                    [] {
                        iom::cuda_detail::inject_submission_fault_for_testing(
                                iom::cuda_detail::SubmissionFault::none);
                    },
                    "cuda_detail::SubmissionFault::event_record",
                    {}}};
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, &oracle, true, true, &rope_contract);
    CHECK_FALSE(devices.gate.armed());
}

// The CUDA instantiation of the shared model-loading scenario. Each synthetic
// checkpoint is loaded through the production configuration and mapped-source
// API, realized on the CPU reference and the selected CUDA devices, and read
// back bit-for-bit against the fixture's independent role bytes. A CUDA
// binding reports a positive staging requirement, so it realizes with
// caller-provisioned scratch of the returned maximum and refuses the empty
// default before the first copied role.
TEST_CASE("CUDA model loading realizes every published weight role of each synthetic checkpoint") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_model_loading_conformance(devices.conformance());
}

// Opt-in real-checkpoint loading, compiled only with
// `IOM_TEST_REAL_MODEL_LOADING=ON`. The synthetic conformance arena is
// deliberately not reused: it cannot hold the full official reference, so the
// caller selects the real capacity through `IOM_TEST_MODEL_ARENA_BYTES` and a
// missing, invalid, or insufficient value fails this case instead of silently
// shrinking the run.
#ifdef IOM_TEST_REAL_MODEL_LOADING
TEST_CASE("CUDA real model loading") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    const std::unique_ptr<iom::Device> candidate = iom::make_cuda_device(
            0, iom_conformance::real_model_memory_config());
    iom_conformance::run_real_model_loading(*candidate);
}
#endif

TEST_CASE("CUDA binary conformance: ADD MUL SUB DIV real queue") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    for (const auto operation : {
            iom_conformance::BinaryOperation::add,
            iom_conformance::BinaryOperation::mul,
            iom_conformance::BinaryOperation::sub,
            iom_conformance::BinaryOperation::div}) {
        iom_conformance::run_gpu_eltwise_conformance(
                *devices.candidate, operation);
        iom_conformance::run_gpu_eltwise_mapping_conformance(
                *devices.candidate, operation);
        iom_conformance::run_gpu_exact_alias_conformance(
                *devices.candidate, operation);
    }
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA ADD accepts every low-width leaf against the oracle") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_gpu_eltwise_conformance(
            *devices.candidate, iom_conformance::BinaryOperation::add);
    iom_conformance::run_binary_rank_boundary_conformance(
            *devices.candidate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA ADD wide dtypes and boundary values against the oracle") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_gpu_eltwise_conformance(
            *devices.candidate, iom_conformance::BinaryOperation::add);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA ADD broadcast, transform, tail, and exact alias mapping") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    iom_conformance::run_gpu_eltwise_mapping_conformance(
            *devices.candidate, iom_conformance::BinaryOperation::add);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CUDA ADD retained launch failure keeps owners reusable") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 16, 16}}, iom::DataType::U8};
    auto lhs = devices.candidate->create_tensor(spec);
    auto out = devices.candidate->create_tensor(spec);
    std::vector<std::byte> pattern(spec.logical_nbytes());
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = std::byte{static_cast<unsigned char>(i * 3)};
    }
    iom_conformance::copy_from_host(lhs->view(), pattern);
    auto queue = devices.candidate->create_ops();

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::third_plane_launch);
    const iom::oid failed = queue->add(lhs->view(), lhs->view(), out->view());
    CHECK(iom::oid_is_token(failed));
    iom_conformance::expect_repeated_runtime_failure(*queue, failed);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

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

TEST_CASE("CUDA submission remains transactional across post-enqueue failures") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    CudaDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{3, 16, 16}}, iom::DataType::U8};
    const iom::TensorSpec mismatch{
            iom::TensorShape{{3, 16, 17}}, iom::DataType::U8};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    auto invalid_destination = devices.candidate->create_tensor(mismatch);
    auto queue = devices.candidate->create_ops();
    const std::vector<std::byte> logical_pattern(
            spec.logical_nbytes(), static_cast<std::byte>(0x3c));

    iom_conformance::copy_from_host(source->view(), logical_pattern);
    iom_conformance::copy_from_host(destination->view(), logical_pattern);

    CHECK_EQ(queue->copy(source->view(), invalid_destination->view()), iom::to_oid(iom::OidError::InvalidArgument));

    const iom::oid first = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(first), 1);
    CHECK_NOTHROW(queue->wait(first));

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::event_create);
    CHECK_EQ(queue->copy(source->view(), destination->view()), iom::to_oid(iom::OidError::DeviceError));
    const iom::oid second = queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(second), 2);
    CHECK_NOTHROW(queue->wait(second));

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::third_plane_launch);
    iom::oid launch_failure = 0;
    launch_failure = queue->copy(source->view(), destination->view());
    CHECK(iom::oid_is_token(launch_failure));
    CHECK_NE(launch_failure, 0);
    CHECK_EQ(iom_conformance::token_sequence(launch_failure), 3);
    iom_conformance::expect_repeated_runtime_failure(*queue, launch_failure);

    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::event_record);
    const iom::oid record_failure =
            queue->copy(source->view(), destination->view());
    CHECK_EQ(iom_conformance::token_sequence(record_failure), 4);
    iom_conformance::expect_repeated_runtime_failure(*queue, record_failure);
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);

    // Failed work remains quarantined until device teardown; the data
    // arena must not recycle either operand while its failed entries are
    // retained.
    {
        auto device = iom::make_cuda_device(
                0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
        auto canary_source = device->create_tensor(spec);
        auto canary_destination = device->create_tensor(spec);
        const std::vector<std::byte> storage_canary(
                spec.tiled_storage_nbytes(), static_cast<std::byte>(0xa5));
        REQUIRE(
                cudaMemcpy(
                        canary_source->view().native_handle(),
                        storage_canary.data(), storage_canary.size(),
                        cudaMemcpyHostToDevice)
                == cudaSuccess);
        REQUIRE(
                cudaMemcpy(
                        canary_destination->view().native_handle(),
                        storage_canary.data(), storage_canary.size(),
                        cudaMemcpyHostToDevice)
                == cudaSuccess);
        auto canary_queue = device->create_ops();
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::third_plane_launch);
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
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
}

TEST_CASE("CUDA queue destruction fences pending copies") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    iom_conformance::TrafficGate gate;
    auto device = iom::make_cuda_device(
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
        iom::cuda_detail::inject_submission_fault_for_testing(
                iom::cuda_detail::SubmissionFault::third_plane_launch);
        iom::oid failure = 0;
        failure = queue->copy(source->view(), destination->view());
        CHECK(iom::oid_is_token(failure));
        CHECK_NE(failure, 0);
        for (int i = 0; i < 32; ++i) {
            CHECK(iom::oid_is_token(queue->copy(source->view(), destination->view())));
        }
        queue.reset();
    }
    iom::cuda_detail::inject_submission_fault_for_testing(
            iom::cuda_detail::SubmissionFault::none);
    CHECK_FALSE(gate.armed());
}

TEST_CASE("CUDA pre-wait source destruction keeps storage until completion") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    // Large enough that the queued copy kernel is still executing when the
    // host destroys the source; the assertions below hold under every
    // worker/interleaving outcome: the copy completes from storage that was
    // either quarantined or released only after its event fired, so waiting
    // the token always yields correct destination bytes and the data arena
    // never releases the range early. The deterministic quarantine/no-reuse
    // contract is pinned by the ring smoke tests and by the queue-teardown
    // case below.
    const iom::TensorSpec spec{
            iom::TensorShape{{8192, 4096}}, iom::DataType::F32};
    const std::vector<std::byte> expected =
            iom_conformance::encode_standard_tiled_storage(spec);
    const std::vector<std::byte> empty(
            spec.tiled_storage_nbytes(), std::byte{0});

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    CudaStorageOracle oracle;
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
    // range is either quarantined or released only after the copy proved
    // completion and the worker retired its entries; on a fast-enough
    // device the copy completes first and the proved-safe range is legally
    // reused by the arena. In both cases the fresh allocation is safe, the
    // copy still produces the expected bytes, and the deterministic
    // quarantine/no-reuse contract is pinned by the ring smoke tests, the
    // injected-launch-failure case, and the queue-teardown case below.
    source.reset();
    auto fresh = device->create_tensor(spec);
    REQUIRE(fresh != nullptr);
    CHECK(reinterpret_cast<std::uintptr_t>(
                  fresh->view().native_handle())
          % 32 == 0);

    CHECK_NOTHROW(queue->wait(token));
    oracle.set_owner_spec(spec);
    CHECK_EQ(oracle.observe(destination->view()), expected);

    fresh.reset();
    queue.reset();
    destination.reset();
    device.reset();
}

TEST_CASE("CUDA shared-operand copies release storage after both complete") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    const iom::TensorSpec spec{
            iom::TensorShape{{8192, 4096}}, iom::DataType::F32};
    const std::vector<std::byte> expected =
            iom_conformance::encode_standard_tiled_storage(spec);
    const std::vector<std::byte> empty(
            spec.tiled_storage_nbytes(), std::byte{0});

    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    CudaStorageOracle oracle;
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

TEST_CASE("CUDA operands destroyed after queue teardown remain quarantined") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    // Queue destruction invalidates every outstanding registry entry up
    // front, so the operand fences provably report failure when the operands
    // are destroyed afterwards: the data arena cannot recycle either range
    // until device teardown, deterministically, with no timing dependence on
    // the worker or the GPU.
    const iom::TensorSpec spec{
            iom::TensorShape{{4096, 2048}}, iom::DataType::F32};

    auto device = iom::make_cuda_device(
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

TEST_CASE("CUDA conformance: rank boundary covers rank-eight owners and rejects rank nine") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    iom_conformance::TrafficGate gate;
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    CudaStorageOracle oracle;
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
std::vector<iom::cuda_detail::AllocationRecord>* g_workspace_records =
        nullptr;

void capture_workspace_allocations(
        const iom::cuda_detail::AllocationRecord& record) {
    if (g_workspace_records != nullptr) {
        g_workspace_records->push_back(record);
    }
}

struct WorkspaceObserverRestore final {
    WorkspaceObserverRestore()
            : saved_(iom::cuda_detail::allocation_observer),
              saved_sink_(g_workspace_records) {}

    WorkspaceObserverRestore(const WorkspaceObserverRestore&) = delete;
    WorkspaceObserverRestore& operator=(const WorkspaceObserverRestore&) =
            delete;

    ~WorkspaceObserverRestore() {
        iom::cuda_detail::allocation_observer = saved_;
        g_workspace_records = saved_sink_;
    }

    iom::cuda_detail::AllocationObserver saved_;
    std::vector<iom::cuda_detail::AllocationRecord>* saved_sink_ = nullptr;
};

[[nodiscard]] std::size_t succeeded_data_backing_allocations(
        const std::vector<iom::cuda_detail::AllocationRecord>& records) {
    std::size_t count = 0;
    for (const auto& record : records) {
        if (record.classification
                == iom::cuda_detail::AllocationClass::data_backing
                && record.kind
                        == iom::cuda_detail::AllocationKind::allocate
                && record.succeeded) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST_CASE("CUDA conformance: raw workspace suballocates the reserved data arena") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    WorkspaceObserverRestore observer_restore;
    std::vector<iom::cuda_detail::AllocationRecord> records;
    g_workspace_records = &records;
    iom::cuda_detail::allocation_observer.complete =
            &capture_workspace_allocations;

    auto device = iom::make_cuda_device(
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

TEST_CASE("CUDA conformance: workspace requirement queries are pure and exact") {
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    auto device = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::F32};
    auto lhs = device->create_tensor(spec);
    auto rhs = device->create_tensor(spec);
    auto out = device->create_tensor(spec);
    auto queue = device->create_ops();

    // CUDA binary operations need no raw workspace.
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
    auto foreign = iom::make_cuda_device(
            0, iom::DeviceMemoryConfig{kConformanceArenaBytes});
    auto foreign_tensor = foreign->create_tensor(spec);
    CHECK_THROWS_AS((void)
            queue->add_workspace_requirements(
                    foreign_tensor->view(), rhs->view(), out->view()),
            std::invalid_argument);
}


