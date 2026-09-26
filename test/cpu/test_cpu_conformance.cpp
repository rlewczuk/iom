#include <doctest/doctest.h>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>
#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_cache_append.hpp"

#include "backend/backend_conformance_embedding.hpp"
#include "backend/backend_conformance_linear.hpp"
#include "backend/backend_conformance_other.hpp"
#include "backend/backend_conformance_token_selection.hpp"
#include "backend/backend_conformance_rmsnorm.hpp"
#include "backend/backend_conformance_rope.hpp"
#include "backend/backend_conformance_silu.hpp"
#include "backend/backend_conformance_rope_contract.hpp"
#include "backend/backend_conformance_sdpa.hpp"

#include "backend/backend_conformance_add.hpp"
#include "backend/backend_conformance_model_loading.hpp"
#include "backend/backend_conformance_inference_metrics.hpp"
#ifdef IOM_TEST_REAL_MODEL_INFERENCE_CPU
#include "backend/backend_conformance_model_official.hpp"
#endif
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"

#include "../../src/cpu/avx512_bf16.hpp"

namespace {


// Recording allocator over 32-byte-aligned blocks. While the gate is armed —
// inside transfers, transforms, and queue operations — every alloc or free is
// a conformance failure: no case may allocate operands there.
class GatedAllocator final : public iom::Allocator {
public:
    explicit GatedAllocator(const iom_conformance::TrafficGate& gate) : gate_(gate) {}

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
    const iom_conformance::TrafficGate& gate_;
    std::unordered_set<void*> live_;
    std::size_t traffic_ = 0;
};

// One independent CPU reference device, one candidate device, and one foreign
// device instance of the same backend ordinal, each over its own allocator.
struct CpuDevices {
    iom_conformance::TrafficGate gate;
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
// Submit only derived temporary views so the worker can only succeed by using
// the value-captured snapshot. The source owner is released by the caller
// before the dependent copy is waited, exercising registry retention.
iom::oid submit_temporary_rope(
        iom::DeviceOps& queue, iom::Tensor& source,
        iom::Tensor& destination) {
    return queue.rope(
            source.view().slice(0, 0, 1),
            const_cast<iom::TensorView&>(
                    static_cast<const iom::TensorView&>(
                            destination.view().slice(0, 0, 1))),
            0, 10000.0);
}


}  // namespace

// Declared by the CPU ports in `src/cpu/queue.cpp` and `src/cpu/sdpa.cpp`:
// one-shot post-acceptance failure latches consumed inside the enqueued host
// task. The SDPA seam lets the shared harness verify owner retention,
// repeated failure observation, and healthy FIFO recovery without changing the
// production request path.
namespace iom::cpu_detail {

void arm_silu_failure() noexcept;
void clear_silu_failure() noexcept;
void arm_sdpa_failure() noexcept;
void clear_sdpa_failure() noexcept;

}  // namespace iom::cpu_detail

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
    iom_conformance::require_standard_capabilities(
            candidate->supported_data_types());
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
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate, &oracle);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: copy validation fails before writes and sequences") {
    CpuDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: common binary validation and lifetime policy") {
    CpuDevices devices;
    iom_conformance::run_binary_request_conformance(
            devices.conformance(), nullptr);
    iom_conformance::run_binary_rank_boundary_conformance(
            *devices.candidate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: all binary operations through the real queue") {
    CpuDevices devices;
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, nullptr, true, true, nullptr, true);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: binary MUL SUB and floating DIV values through real queue") {
    CpuDevices devices;
    for (const auto operation : {
                 iom_conformance::BinaryOperation::add,
                 iom_conformance::BinaryOperation::mul,
                 iom_conformance::BinaryOperation::sub,
                 iom_conformance::BinaryOperation::div}) {
        iom_conformance::run_binary_value_conformance(
                *devices.candidate, operation);
    }
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

TEST_CASE("CPU conformance: greedy token selection shared matrix and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle native_storage;
    const iom_conformance::TokenSelectionNativeFailureSeam native_failure{
            {}, {}, "CPU logical transfer failure",
            "CPU has no accepted post-readiness transfer-failure seam"};
    const iom_conformance::TokenSelectionConformanceConfig config{
            devices.conformance(),
            devices.candidate->supported_data_types(),
            &devices.gate,
            &native_storage,
            iom::WorkspaceRequirements{0, 1},
            native_failure};
    iom_conformance::run_token_selection_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: binary operations are supported") {
    CpuDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate, "CPU", true, true, true);
    CHECK_FALSE(devices.gate.armed());
}

// The AVX-512 BF16 detector must answer from this machine's real CPUID and
// OS-managed vector state, so this case reads both itself rather than reusing
// the library's query. Only baseline instructions are involved: CPUID leaves 1
// and 7 exist on any x86-64, and XGETBV runs only when CPUID reports OSXSAVE.
//
// The case does not observe the ordering of those two steps: a host whose
// OSXSAVE bit is set cannot show that the other branch skips XGETBV, so that
// half of the contract is checked against the compiled detector instead.
namespace {

struct HostVectorState {
    bool avx512f = false;
    bool avx512_bf16 = false;
    bool osxsave = false;
    std::uint64_t xcr0 = 0;

    // The same five XCR0 bits the workers' instructions need the OS to have
    // enabled: XMM, YMM, opmask, upper ZMM, and high ZMM.
    [[nodiscard]] bool supports_avx512_bf16() const {
        constexpr std::uint64_t required =
                (std::uint64_t{1} << 1) | (std::uint64_t{1} << 2) |
                (std::uint64_t{1} << 5) | (std::uint64_t{1} << 6) |
                (std::uint64_t{1} << 7);
        return avx512f && avx512_bf16 && osxsave &&
                (xcr0 & required) == required;
    }
};

HostVectorState read_host_vector_state() {
    HostVectorState state;
#if defined(__x86_64__)
    unsigned int registers[4] = {};
    __cpuid_count(7, 0, registers[0], registers[1], registers[2], registers[3]);
    state.avx512f = (registers[1] & (1u << 16)) != 0;
    const unsigned int highest_subleaf = registers[0];
    if (highest_subleaf >= 1) {
        __cpuid_count(
                7, 1, registers[0], registers[1], registers[2], registers[3]);
        state.avx512_bf16 = (registers[0] & (1u << 5)) != 0;
    }

    __cpuid_count(1, 0, registers[0], registers[1], registers[2], registers[3]);
    state.osxsave = (registers[2] & (1u << 27)) != 0;
    if (state.osxsave) {
        std::uint32_t low = 0;
        std::uint32_t high = 0;
        __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        state.xcr0 = (static_cast<std::uint64_t>(high) << 32) | low;
    }
#endif
    return state;
}

}  // namespace

TEST_CASE("CPU AVX-512 BF16 eligibility follows this host's vector state") {
    const HostVectorState host = read_host_vector_state();
    const bool eligible = iom::cpu_detail::avx512_bf16_available();

#ifdef IOM_TEST_AVX512_BF16_ENABLED
    // This configuration compiles the isolated AVX-512 BF16 sources, so the
    // detector reports exactly the features and vector-state bits this machine
    // offers.
    CHECK_EQ(eligible, host.supports_avx512_bf16());
#else
    // A configuration without those sources has no target code to enter, so it
    // answers false whatever the host reports.
    CHECK_FALSE(eligible);
#endif
}

TEST_CASE("CPU conformance: SDPA reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    const iom_conformance::SdpaConformanceConfig config{
            devices.conformance(),
            iom_conformance::kSdpaCurrentSupportedDataTypes,
            true,
            &devices.gate,
            &oracle,
            {},
            {&iom::cpu_detail::arm_sdpa_failure,
             &iom::cpu_detail::clear_sdpa_failure,
             "cpu_detail::sdpa",
             {}}};
    iom_conformance::run_sdpa_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}

// CPU's declared embedding expectation covers the complete 23-payload/12-index
// matrix and the fixed `{0, 1}` scratch policy.
constexpr iom_conformance::EmbeddingDeclaration kCpuEmbeddingDeclaration{
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom::WorkspaceRequirements{0, 1}};

TEST_CASE("CPU conformance: embedding lookup reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::run_embedding_conformance(
            devices.conformance(), kCpuEmbeddingDeclaration, &devices.gate,
            &oracle);
    CHECK_FALSE(devices.gate.armed());
    // The zero requirement still belongs to embedding admission, while the
    // device now exposes positive owners for CPU SDPA as well.
    const std::unique_ptr<iom::RawWorkspace> workspace =
            devices.candidate->create_workspace(1);
    REQUIRE(workspace != nullptr);
    CHECK_EQ(workspace->byte_size(), 1);
}

// CPU's declared linear expectation: the complete 21-leaf applicable matrix,
// one scalar recurrence covering every applicable leaf (including BF16, whose
// CPU path is that same recurrence), and the fixed `{0, 1}` scratch path. The
// CPU port implements that whole matrix, so the implemented span is the
// complete applicable leaf span and every declared leaf is compared
// numerically against the independent reference.
const iom_conformance::LinearDeclaration kCpuLinearDeclaration{
        iom_conformance::kLinearLeafSpan,
        iom_conformance::kLinearLeafSpan,
        iom_conformance::kNoLinearSpan,
        iom_conformance::kLinearLeafSpan,
        {},
        iom_conformance::LinearWorkspacePath::zero,
        iom_conformance::LinearWorkspacePath::zero};

TEST_CASE("CPU conformance: linear projection reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::run_linear_conformance(
            devices.conformance(), kCpuLinearDeclaration, &devices.gate,
            &oracle);
    CHECK_FALSE(devices.gate.armed());
}
// CPU's cache append port is the direct zero-workspace implementation. The
// shared runner owns the independent byte oracle, all 23 opaque leaves,
// transformed view offsets/strides, admission boundaries, FIFO pipeline,
// temporary-view lifetime, repeated waits, and untouched padding checks.
TEST_CASE("CPU conformance: cache row append reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::CacheAppendConformanceConfig config{
            devices.conformance(),
            iom_conformance::kStandardCapabilityOracle,
            &oracle,
            {},
            {},
            &devices.gate};
    // CPU has no post-acceptance failure seam by design; the shared
    // fault/repeat-wait section is intentionally unavailable here, while
    // successful repeated waits remain exercised by the common cases.
    iom_conformance::run_cache_append_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}
TEST_CASE("CPU cache append drains accepted work during queue destruction") {
    CpuDevices devices;
    const iom::TensorSpec source_spec{
            iom::TensorShape{{2, 3, 1, 17}}, iom::DataType::BF16};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{2, 3, 17, 17}}, iom::DataType::BF16};
    auto source = devices.candidate->create_tensor(source_spec);
    auto destination = devices.candidate->create_tensor(destination_spec);
    const std::vector<std::byte> source_bytes =
            iom_conformance::encode_cache_append_logical(source_spec, 0x711);
    const std::vector<std::byte> destination_before =
            iom_conformance::encode_logical(destination_spec, 0x712);
    const std::vector<std::byte> expected =
            iom_conformance::cache_append_expected_logical(
                    source_spec, destination_spec, 1, source_bytes,
                    destination_before);
    iom_conformance::copy_from_host(source->view(), source_bytes);
    iom_conformance::copy_from_host(
            destination->view(), destination_before);

    devices.gate.setup_complete();
    {
        auto queue = devices.candidate->create_ops();
        const iom::TensorView temporary_source = source->view();
        iom::TensorView temporary_destination = destination->view();
        const iom::oid token = queue->cache_append(
                temporary_source, temporary_destination, 1);
        REQUIRE(iom::oid_is_token(token));
        // Destruction must drain the FIFO worker before invalidating the
        // accepted owner registrations. No explicit wait is issued here.
    }
    devices.gate.case_complete();

    iom_conformance::require_logical_bytes(
            destination->view(), expected,
            "cache append queue destruction drain");
    CHECK_FALSE(devices.gate.armed());
}



// CPU's declared RMSNorm expectation: the nine applicable signed floating
// leaves, the frozen `{0, 1}` workspace requirement, and positive execution
// through the real queue, with host observation of the logical results and of
// deliberately poisoned tile padding.
TEST_CASE("CPU conformance: RMSNorm reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::RmsNormConformanceConfig config{
            devices.conformance(),
            iom_conformance::kRmsNormAllLeafSpan,
            &devices.gate,
            &oracle};
    // The CPU backend exposes no testing seam at all: `src/cpu` carries no
    // fault-injection hook, and a CPU RMSNorm request runs to completion on the
    // shared FIFO host worker, so no existing seam can produce an accepted
    // CPU RMSNorm failure. The harness records that gap explicitly. The minimal
    // test-only seam that would close it is a `cpu_detail` fault consumed
    // inside the enqueued host task (after acceptance, before the row loop)
    // under an `IOM_ENABLE_TESTING` guard.
    config.native_failure = iom_conformance::RmsNormNativeFailureSeam{
            {},
            {},
            "cpu_detail",
            "src/cpu exposes no fault-injection seam; an accepted CPU RMSNorm "
            "request cannot be made to fail after acceptance"};
    iom_conformance::run_rmsnorm_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}

// The reduction stage's own focused coverage, independent of the shared matrix.
//
// A request may be a transformed view of a larger owner, so its first pass has
// to reduce exactly the selected plane's logical features: the owner's
// unselected planes and the tile padding beside the feature tail are seeded with
// values that would both change a row inverse and trip the exception check if
// either were read. The submission is a real queue request, its expected values
// come from the shared independent oracle, and the case also runs as a baseline
// control in a configuration without the isolated AVX-512 BF16 sources.
TEST_CASE("CPU conformance: BF16 RMSNorm reduction honors a selected view, a feature tail, and padding") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    constexpr std::size_t kRows = 5;
    constexpr std::size_t kFeatures = 17;
    constexpr std::size_t kSelectedPlanes = 3;
    constexpr float kEpsilon = 1.0e-5F;

    // Two leading axes, so the sliced view has a nonzero plane offset *and*
    // multi-plane leading strides: it selects the owner's second group of three
    // planes, which the oracle evaluates as its planes zero through two.
    const iom::TensorSpec owner_spec{
            iom::TensorShape{{2, kSelectedPlanes, kRows, kFeatures}},
            iom::DataType::BF16};
    const iom::TensorSpec scale_spec{
            iom::TensorShape{{1, kFeatures}}, iom::DataType::BF16};
    const iom_conformance::RmsNormReferenceCase reference_case =
            iom_conformance::rmsnorm_oracle::make_case(
                    iom_conformance::RmsNormReferenceCaseKind::mixed,
                    iom::DataType::BF16, kSelectedPlanes, kRows, kFeatures,
                    kEpsilon,
                    iom_conformance::rmsnorm_oracle::pattern_content(
                            kSelectedPlanes, kRows, kFeatures),
                    iom_conformance::rmsnorm_oracle::scale_content(kFeatures));

    // The owner's unselected planes carry a magnitude no row wants, and the
    // output owner carries a distinct sentinel, so an operand-wide walk is
    // distinguishable from the selected view's own planes.
    const std::uint64_t unselected_code =
            iom_conformance::rmsnorm_oracle::value_bits(
                    iom::DataType::BF16, 4096.0);
    const std::uint64_t output_sentinel =
            iom_conformance::rmsnorm_oracle::value_bits(
                    iom::DataType::BF16, -123.5);
    const std::size_t owner_elements =
            2 * kSelectedPlanes * kRows * kFeatures;
    std::vector<std::uint64_t> input_codes(owner_elements, unselected_code);
    std::copy(
            reference_case.x_bits.begin(), reference_case.x_bits.end(),
            input_codes.begin()
                    + static_cast<std::ptrdiff_t>(kSelectedPlanes * kRows
                                                  * kFeatures));
    const std::vector<std::uint64_t> output_codes(
            owner_elements, output_sentinel);

    auto x_owner = devices.candidate->create_tensor(owner_spec);
    auto out_owner = devices.candidate->create_tensor(owner_spec);
    auto scale = devices.candidate->create_tensor(scale_spec);
    oracle.set_owner_spec(owner_spec);
    oracle.seed(
            x_owner->view(),
            iom_conformance::rmsnorm_padded_image(
                    owner_spec, input_codes,
                    iom_conformance::kRmsNormPaddingPoison));
    oracle.seed(
            out_owner->view(),
            iom_conformance::rmsnorm_padded_image(
                    owner_spec, output_codes,
                    iom_conformance::kRmsNormPaddingPoison));
    oracle.set_owner_spec(scale_spec);
    oracle.seed(
            scale->view(),
            iom_conformance::rmsnorm_padded_image(
                    scale_spec, reference_case.scale_bits,
                    iom_conformance::kRmsNormPaddingPoison));

    const iom::TensorView x_view = x_owner->view().slice(0, 1, 1);
    iom::TensorView out_view = out_owner->view().slice(0, 1, 1);
    REQUIRE_EQ(x_view.spec().shape.rank(), std::size_t{4});
    REQUIRE_EQ(x_view.plane_offset(), kSelectedPlanes);
    REQUIRE_EQ(out_view.plane_offset(), kSelectedPlanes);

    auto queue = devices.candidate->create_ops();
    devices.gate.setup_complete();
    const iom::oid token =
            queue->rmsnorm(x_view, scale->view(), out_view, kEpsilon);
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue->wait(token));
    devices.gate.case_complete();
    CHECK_FALSE(devices.gate.armed());

    const std::vector<iom_conformance::RmsNormReferenceValue> expected =
            iom_conformance::evaluate(reference_case);
    iom_conformance::check_rmsnorm_image(
            iom::DataType::BF16, iom_conformance::read_logical(out_view),
            expected,
            "cpu BF16 selected view, leading strides, feature tail 17");

    // Exactly the selected view's logical slots moved: the owner's other planes
    // and every padded bit keep the pattern they were seeded with, in both
    // operands.
    std::vector<std::uint64_t> expected_output_codes = output_codes;
    for (std::size_t plane = 0; plane < kSelectedPlanes; ++plane) {
        for (std::size_t row = 0; row < kRows; ++row) {
            for (std::size_t feature = 0; feature < kFeatures; ++feature) {
                const std::size_t element = row * kFeatures + feature;
                expected_output_codes[(kSelectedPlanes + plane) * kRows
                                              * kFeatures
                                      + element] =
                        expected[plane * kRows * kFeatures + element].bits;
            }
        }
    }
    oracle.set_owner_spec(owner_spec);
    CHECK(
            oracle.observe(x_owner->view())
            == iom_conformance::rmsnorm_padded_image(
                    owner_spec, input_codes,
                    iom_conformance::kRmsNormPaddingPoison));
    CHECK(
            oracle.observe(out_owner->view())
            == iom_conformance::rmsnorm_padded_image(
                    owner_spec, expected_output_codes,
                    iom_conformance::kRmsNormPaddingPoison));
}

#ifdef IOM_TEST_AVX512_BF16_ENABLED
// This configuration compiles the isolated AVX-512 BF16 sources, so a BF16
// request on a capable host must execute the vector row reduction, and a row
// whose features are not all finite must be visible as scalar fallback work
// instead. Both paths are asserted against the same independent oracle, so the
// executed-work observation is paired with the arithmetic it claims.
TEST_CASE("CPU conformance: AVX-512 BF16 RMSNorm reduction executes vector work") {
    CpuDevices devices;
    const iom_conformance::RmsNormConformanceConfig config{
            devices.conformance(), iom_conformance::kRmsNormWideLeafSpan};
    auto queue = devices.candidate->create_ops();

    // An ordinary finite geometry with a non-tile feature width: two full
    // blocks and a one-lane masked tail per row, over two independent planes.
    constexpr std::size_t kPlanes = 2;
    constexpr std::size_t kRows = 4;
    constexpr std::size_t kFeatures = 33;
    const iom_conformance::RmsNormReferenceCase ordinary =
            iom_conformance::rmsnorm_oracle::make_case(
                    iom_conformance::RmsNormReferenceCaseKind::mixed,
                    iom::DataType::BF16, kPlanes, kRows, kFeatures, 1.0e-5F,
                    iom_conformance::rmsnorm_oracle::pattern_content(
                            kPlanes, kRows, kFeatures),
                    iom_conformance::rmsnorm_oracle::scale_content(kFeatures));

    if (!iom::cpu_detail::avx512_bf16_available()) {
        MESSAGE(
                "host is not AVX-512 BF16 eligible: this build contains the "
                "isolated sources, but no vector work can execute here and no "
                "native observation is asserted");
        return;
    }

    iom::cpu_detail::avx512_bf16_test_reset_observations();
    iom_conformance::run_rmsnorm_reference_case(
            config, *queue, ordinary,
            std::vector<std::size_t>{kPlanes},
            "accelerated row reduction");
    const std::uint64_t native_blocks =
            iom::cpu_detail::avx512_bf16_test_observation(
                    iom::cpu_detail::Avx512Bf16Stage::RmsReduction,
                    iom::cpu_detail::Avx512Bf16Path::Native);
    const std::uint64_t native_fallbacks =
            iom::cpu_detail::avx512_bf16_test_observation(
                    iom::cpu_detail::Avx512Bf16Stage::RmsReduction,
                    iom::cpu_detail::Avx512Bf16Path::Fallback);
    CHECK_MESSAGE(
            native_blocks >= kPlanes * kRows,
            "the vector row reduction executed for every ordinary row: "
            "native blocks = " << native_blocks << ", rows = "
                               << kPlanes * kRows);
    CHECK_EQ(native_fallbacks, std::uint64_t{0});

    // A row holding a NaN or an infinity feature is reduced by the sequential
    // recurrence inside the same queued worker, so a request whose every row is
    // exceptional must show exactly fallback work, no vector reduction beside
    // it, and the contract's special-value classes from the oracle.
    constexpr std::size_t kExceptionalRows = 2;
    std::vector<double> nan_values =
            iom_conformance::rmsnorm_oracle::pattern_content(
                    1, kExceptionalRows, kFeatures);
    nan_values[2] = std::numeric_limits<double>::quiet_NaN();
    nan_values[kFeatures + 2] = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> infinity_values =
            iom_conformance::rmsnorm_oracle::pattern_content(
                    1, kExceptionalRows, kFeatures);
    infinity_values[2] = std::numeric_limits<double>::infinity();
    infinity_values[5] = -std::numeric_limits<double>::infinity();
    infinity_values[kFeatures + 2] = std::numeric_limits<double>::infinity();
    infinity_values[kFeatures + 5] = -std::numeric_limits<double>::infinity();
    const std::array<iom_conformance::RmsNormReferenceCase, 2> exceptional{
            iom_conformance::rmsnorm_oracle::make_case(
                    iom_conformance::RmsNormReferenceCaseKind::nan_row,
                    iom::DataType::BF16, 1, kExceptionalRows, kFeatures, 1.0e-5F,
                    nan_values,
                    iom_conformance::rmsnorm_oracle::scale_content(kFeatures)),
            iom_conformance::rmsnorm_oracle::make_case(
                    iom_conformance::RmsNormReferenceCaseKind::infinity_row,
                    iom::DataType::BF16, 1, kExceptionalRows, kFeatures, 1.0e-5F,
                    infinity_values,
                    iom_conformance::rmsnorm_oracle::scale_content(kFeatures))};
    iom::cpu_detail::avx512_bf16_test_reset_observations();
    for (const iom_conformance::RmsNormReferenceCase& reference_case :
         exceptional) {
        iom_conformance::run_rmsnorm_reference_case(
                config, *queue, reference_case, {}, "scalar fallback row");
    }
    CHECK_EQ(
            iom::cpu_detail::avx512_bf16_test_observation(
                    iom::cpu_detail::Avx512Bf16Stage::RmsReduction,
                    iom::cpu_detail::Avx512Bf16Path::Native),
            std::uint64_t{0});
    const std::uint64_t exceptional_fallbacks =
            iom::cpu_detail::avx512_bf16_test_observation(
                    iom::cpu_detail::Avx512Bf16Stage::RmsReduction,
                    iom::cpu_detail::Avx512Bf16Path::Fallback);
    CHECK_MESSAGE(
            exceptional_fallbacks >= kExceptionalRows,
            "every exceptional row stayed with the in-worker scalar "
            "recurrence: fallback rows = "
                    << exceptional_fallbacks);
}
#endif

// The optional AVX-512 BF16 store worker reports the work it really executed:
// an eligible ordinary BF16 request must store every full feature group from
// the SIMD normalization, scale, and encode result, a value the ISA conversion
// cannot encode under the contract must be stored by the compliant scalar codec
// inside the same worker, and a request that never enters the worker must leave
// the native count at zero. Every case is still compared against the
// independent oracle, and the two contract facts the comparison policy cannot
// see -- the canonical positive NaN encoding and a non-flushed subnormal
// result -- are asserted on the stored raw bits.
TEST_CASE("CPU AVX-512 BF16 RMSNorm store observes native groups and in-worker fallback") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::RmsNormConformanceConfig config{
            devices.conformance(),
            iom_conformance::kRmsNormAllLeafSpan,
            &devices.gate,
            &oracle};
    auto queue = devices.candidate->create_ops();

    constexpr iom::DataType kBf16 = iom::DataType::BF16;
    constexpr auto kStore = iom::cpu_detail::Avx512Bf16Stage::RmsStore;
    constexpr auto kNative = iom::cpu_detail::Avx512Bf16Path::Native;
    constexpr auto kFallback = iom::cpu_detail::Avx512Bf16Path::Fallback;
    // This build and this host decide which store path a BF16 request takes;
    // both outcomes are asserted, and the eligible one is the point of the
    // case, so the observed operand is reported with the verdict.
    const bool accelerated = iom::cpu_detail::avx512_bf16_available();
    INFO("AVX-512 BF16 store worker entered: " << accelerated);

    // One row-major `[1, rows, features]` BF16 fixture whose raw codes come
    // from the oracle's own encoder, so a case pins exactly the bits the
    // expectation is derived from.
    const auto bf16_case = [](iom_conformance::RmsNormReferenceCaseKind kind,
                              std::size_t rows, std::size_t features, float eps,
                              const std::vector<double>& x,
                              const std::vector<double>& scale) {
        iom_conformance::RmsNormReferenceCase reference_case;
        reference_case.kind = kind;
        reference_case.data_type = kBf16;
        reference_case.planes = 1;
        reference_case.rows = rows;
        reference_case.features = features;
        reference_case.eps = eps;
        for (const double value : x) {
            reference_case.x_bits.push_back(
                    iom_conformance::rmsnorm_oracle::value_bits(kBf16, value));
        }
        for (const double value : scale) {
            reference_case.scale_bits.push_back(
                    iom_conformance::rmsnorm_oracle::value_bits(kBf16, value));
        }
        return reference_case;
    };

    struct StoreRun {
        std::vector<std::byte> logical;
        std::uint64_t native = 0;
        std::uint64_t fallback = 0;
    };
    // One fixture through the real queue, with the stage counts of exactly that
    // request and the oracle comparison of its logical readback.
    const auto run_fixture = [&](const iom_conformance::RmsNormReferenceCase&
                                         reference_case,
                                 std::string_view label) {
        StoreRun run;
        iom::cpu_detail::avx512_bf16_test_reset_observations();
        iom_conformance::run_rmsnorm_fixture(
                config, *queue, reference_case, {}, run.logical);
        run.native = iom::cpu_detail::avx512_bf16_test_observation(
                kStore, kNative);
        run.fallback = iom::cpu_detail::avx512_bf16_test_observation(
                kStore, kFallback);
        iom_conformance::check_rmsnorm_image(
                kBf16, run.logical,
                iom_conformance::evaluate(reference_case), label);
        return run;
    };

    // Ordinary in-range BF16 values: every full group is stored from the SIMD
    // result, and no element needs the scalar codec. `pattern_content` and
    // `scale_content` are nonzero everywhere and exactly representable in
    // BF16, so no result is a zero, a subnormal, or a NaN.
    {
        constexpr std::size_t rows = 2;
        constexpr std::size_t features = 48;
        const StoreRun run = run_fixture(
                bf16_case(
                        iom_conformance::RmsNormReferenceCaseKind::scaled, rows,
                        features, 1.0e-5f,
                        iom_conformance::rmsnorm_oracle::pattern_content(
                                1, rows, features),
                        iom_conformance::rmsnorm_oracle::scale_content(
                                features)),
                "BF16 ordinary native store");
        CHECK_EQ(
                run.native,
                accelerated ? static_cast<std::uint64_t>(rows * features / 16)
                            : 0u);
        CHECK_EQ(
                run.fallback,
                accelerated ? 0u : static_cast<std::uint64_t>(rows * features));
    }

    // Directed in-worker fallback: a unit row with `eps == 0` has an exact
    // inverse of one, so the first group's result is the nonzero BF16 subnormal
    // `8 * 2^-133` -- below the smallest FP32 normal, where the ISA conversion
    // flushes instead of rounding -- and the second group's result is exactly
    // one. The first group must be stored by the scalar codec, the second by
    // the vector worker, and the subnormal must survive as its own encoding
    // rather than a flushed zero.
    {
        constexpr std::size_t features = 32;
        std::vector<double> scale(features, 1.0);
        for (std::size_t feature = 0; feature < 16; ++feature) {
            scale[feature] = std::ldexp(1.0, -130);
        }
        const StoreRun run = run_fixture(
                bf16_case(
                        iom_conformance::RmsNormReferenceCaseKind::
                                destination_underflow,
                        1, features, 0.0f,
                        std::vector<double>(features, 1.0), scale),
                "BF16 subnormal-result fallback store");
        CHECK_EQ(run.native, accelerated ? 1u : 0u);
        CHECK_EQ(run.fallback, accelerated ? 16u : features);
        for (std::size_t feature = 0; feature < 16; ++feature) {
            CHECK_EQ(
                    iom_conformance::rmsnorm_code_at(run.logical, kBf16,
                                                     feature),
                    0x8u);
        }
    }

    // A NaN feature poisons its whole row, so every stored result of that row
    // is a NaN and must take the compliant scalar encode; the sibling row is
    // ordinary and stays on the vector path. The stored NaN must be the
    // contract's canonical positive quiet NaN, which the comparison policy
    // deliberately does not distinguish from any other NaN payload.
    {
        constexpr std::size_t rows = 2;
        constexpr std::size_t features = 32;
        std::vector<double> x = iom_conformance::rmsnorm_oracle::pattern_content(
                1, rows, features);
        x[3] = std::numeric_limits<double>::quiet_NaN();
        const StoreRun run = run_fixture(
                bf16_case(
                        iom_conformance::RmsNormReferenceCaseKind::nan_row, rows,
                        features, 1.0e-5f, x,
                        iom_conformance::rmsnorm_oracle::scale_content(
                                features)),
                "BF16 NaN row fallback store");
        CHECK_EQ(run.native,
                 accelerated ? static_cast<std::uint64_t>(rows - 1) * features
                                     / 16
                             : 0u);
        CHECK_EQ(
                run.fallback,
                accelerated ? static_cast<std::uint64_t>(features)
                            : static_cast<std::uint64_t>(rows * features));
        for (std::size_t feature = 0; feature < features; ++feature) {
            CHECK_EQ(
                    iom_conformance::rmsnorm_code_at(run.logical, kBf16,
                                                     feature),
                    0x7FC0u);
        }
    }

    // A selected leading plane of a larger owner with a non-tile feature width:
    // only the view's logical features change, each full group comes from the
    // vector worker, the eight-feature tail is the scalar codec's, and the
    // other planes plus every padded element of the owner keep the caller's
    // sentinel. The input operand must be left exactly as it was.
    {
        constexpr std::size_t owner_planes = 3;
        constexpr std::size_t rows = 2;
        constexpr std::size_t features = 40;
        const iom::TensorSpec spec = iom_conformance::rmsnorm_spec(
                {owner_planes, rows, features}, kBf16);
        const iom::TensorSpec scale_spec =
                iom_conformance::rmsnorm_spec({1, features}, kBf16);
        auto x_owner = devices.candidate->create_tensor(spec);
        auto out_owner = devices.candidate->create_tensor(spec);
        auto scale = devices.candidate->create_tensor(scale_spec);
        iom::TensorView x = x_owner->view().slice(0, 1, 1);
        iom::TensorView out = out_owner->view().slice(0, 1, 1);
        const iom_conformance::RmsNormReferenceCase reference_case = bf16_case(
                iom_conformance::RmsNormReferenceCaseKind::boundary_sizes, rows,
                features, 1.0e-5f,
                iom_conformance::rmsnorm_oracle::pattern_content(
                        1, rows, features),
                iom_conformance::rmsnorm_oracle::scale_content(features));
        constexpr std::byte kSentinel{0x5A};
        std::vector<std::byte> sentinel(
                spec.tiled_storage_nbytes(), kSentinel);
        oracle.set_owner_spec(spec);
        iom_conformance::copy_from_host(
                x, iom_conformance::rmsnorm_pack_codes(
                           reference_case.x_bits, kBf16));
        iom_conformance::copy_from_host(
                scale->view(), iom_conformance::rmsnorm_pack_codes(
                                       reference_case.scale_bits, kBf16));
        oracle.seed(out, sentinel);
        const std::vector<std::byte> input_before = oracle.observe(x);
        const std::vector<std::byte> padding_evidence = oracle.observe(out);

        devices.gate.setup_complete();
        iom::cpu_detail::avx512_bf16_test_reset_observations();
        const iom::oid token =
                queue->rmsnorm(x, scale->view(), out, reference_case.eps);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        devices.gate.case_complete();

        const std::uint64_t native = iom::cpu_detail::avx512_bf16_test_observation(
                kStore, kNative);
        const std::uint64_t fallback =
                iom::cpu_detail::avx512_bf16_test_observation(
                        kStore, kFallback);
        // The independent expectation of the whole owner: the caller's
        // sentinel everywhere except the selected view's logical elements.
        const std::vector<iom_conformance::RmsNormReferenceValue> expected =
                iom_conformance::evaluate(reference_case);
        std::vector<std::uint64_t> expected_codes;
        expected_codes.reserve(expected.size());
        for (const iom_conformance::RmsNormReferenceValue& value : expected) {
            expected_codes.push_back(value.bits);
        }
        std::vector<std::byte> expected_image = sentinel;
        iom_conformance::apply_standard_tiled_view(
                out, spec,
                iom_conformance::rmsnorm_pack_codes(expected_codes, kBf16),
                expected_image);
        CHECK(oracle.observe(out) == expected_image);
        CHECK(oracle.observe(x) == input_before);
        // The image the operation must reproduce writes the selected view's
        // logical elements, so it cannot equal the untouched sentinel storage;
        // a comparison that could not tell them apart would be vacuous.
        CHECK(expected_image != padding_evidence);
        CHECK_EQ(
                native,
                accelerated ? static_cast<std::uint64_t>(rows * (features / 16))
                            : 0u);
        CHECK_EQ(
                fallback,
                accelerated ? static_cast<std::uint64_t>(rows * (features % 16))
                            : static_cast<std::uint64_t>(rows * features));
    }
}
// CPU's declared SiLU expectation: the nine applicable signed floating leaves,
// the frozen `{0, 1}` zero-workspace requirement, and positive execution through
// the real in-order CPU queue, observed by the shared independent oracle over
// rank-2..8 geometry, transformed views, non-tile dimensions, poisoned tile
// padding, the rejection matrix, queue ordering, and the CPU port's own
// post-acceptance failure latch.
TEST_CASE("CPU conformance: SiLU reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::SiluConformanceConfig config{
            devices.conformance(),
            iom_conformance::kSiluCpuExpectedSupported,
            &devices.gate,
            &oracle,
            iom_conformance::SiluNativeFailureSeam{
                    &iom::cpu_detail::arm_silu_failure,
                    &iom::cpu_detail::clear_silu_failure,
                    "cpu_detail::silu",
                    {}}};
    iom_conformance::run_silu_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}
// The shared native-failure scenario observes only the consumed sequence and the
// healthy recovery, so this CPU case owns the remaining accepted-failure
// evidence through the real queue: one submission whose operand owner is
// released before any wait, in-order FIFO completion of a dependent copy, one
// retained failure reported identically on every repeated wait with the OID
// still positive, and a conforming submission on the same queue and owners
// afterwards.
TEST_CASE("CPU SiLU retains owners and repeats accepted failures") {
    CpuDevices devices;
    constexpr iom::DataType type = iom::DataType::F32;
    const iom_conformance::SiluReferenceCase fixture =
            iom_conformance::silu_make_pattern_case(
                    iom_conformance::SiluReferenceCaseKind::boundary_sizes,
                    type, {2}, 2, 17, "cpu-silu-queue-order/F32");
    const iom::TensorSpec spec = iom_conformance::silu_case_spec(fixture);
    const std::vector<std::byte> input_bytes =
            iom_conformance::silu_pack_bits(type, fixture.input_bits);
    const std::vector<iom_conformance::SiluReferenceValue> expected =
            iom_conformance::silu_evaluate(fixture);
    auto input = devices.candidate->create_tensor(spec);
    auto activated = devices.candidate->create_tensor(spec);
    auto consumed = devices.candidate->create_tensor(spec);
    iom_conformance::copy_from_host(input->view(), input_bytes);
    iom_conformance::copy_from_host(
            activated->view(),
            std::vector<std::byte>(
                    spec.logical_nbytes(),
                    iom_conformance::kReadbackSentinel));
    iom_conformance::copy_from_host(
            consumed->view(),
            std::vector<std::byte>(
                    spec.logical_nbytes(),
                    iom_conformance::kReadbackSentinel));
    auto queue = devices.candidate->create_ops();
    iom::oid activation = 0;
    {
        iom_conformance::SiluCaseWindow window(&devices.gate);
        activation = queue->silu(input->view(), activated->view());
        REQUIRE(iom::oid_is_token(activation));
        // The value-captured submission must retain this owner: the caller
        // releases it here, before either wait, and the dependent copy still
        // observes the activated data.
        input.reset();
        iom::TensorView consumed_view = consumed->view();
        const iom::oid consumer =
                queue->copy(activated->view(), consumed_view);
        REQUIRE(iom::oid_is_token(consumer));
        CHECK_EQ(
                iom_conformance::token_sequence(consumer),
                iom_conformance::token_sequence(activation) + 1);
        CHECK_NOTHROW(queue->wait(consumer));
        CHECK_NOTHROW(queue->wait(activation));
        CHECK_NOTHROW(queue->wait(consumer));
    }
    const std::vector<std::byte> activated_bytes =
            iom_conformance::read_logical(activated->view());
    const std::string retained_mismatch = iom_conformance::silu_compare(
            type, activated_bytes, expected, "cpu-silu retained owner");
    CHECK_MESSAGE(retained_mismatch.empty(), retained_mismatch);
    CHECK(iom_conformance::read_logical(consumed->view()) == activated_bytes);

    const auto failure_text = [&](iom::oid token) {
        std::string observed = "<an accepted SiLU failure reported no error>";
        try {
            queue->wait(token);
        } catch (const std::runtime_error& error) {
            observed = error.what();
        } catch (const std::exception& error) {
            observed = std::string("unexpected failure category: ")
                    + error.what();
        }
        return observed;
    };
    iom::cpu_detail::arm_silu_failure();
    auto failure_input = devices.candidate->create_tensor(spec);
    iom_conformance::copy_from_host(failure_input->view(), input_bytes);
    const iom::oid failed =
            queue->silu(failure_input->view(), activated->view());
    REQUIRE(iom::oid_is_token(failed));
    CHECK_EQ(
            iom_conformance::token_sequence(failed),
            iom_conformance::token_sequence(activation) + 2);
    // The positive OID keeps its retained runtime-error context on every wait.
    const std::string first = failure_text(failed);
    const std::string second = failure_text(failed);
    CHECK_EQ(first, std::string("CPU SiLU injected post-acceptance failure"));
    CHECK_EQ(second, first);
    CHECK(iom::oid_is_token(failed));
    // The latch is consumed by exactly one SiLU task, so the same queue and the
    // same owners produce a conforming result again.
    const iom::oid recovered =
            queue->silu(failure_input->view(), activated->view());
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_EQ(
            iom_conformance::token_sequence(recovered),
            iom_conformance::token_sequence(failed) + 1);
    CHECK_NOTHROW(queue->wait(recovered));
    CHECK_NOTHROW(queue->wait(recovered));
    const std::string recovery_mismatch = iom_conformance::silu_compare(
            type, iom_conformance::read_logical(activated->view()), expected,
            "cpu-silu recovered submission");
    CHECK_MESSAGE(recovery_mismatch.empty(), recovery_mismatch);
    iom::cpu_detail::clear_silu_failure();
    CHECK_FALSE(devices.gate.armed());
}
// The eligible BF16 leaf is the only SiLU path that runs the isolated AVX-512
// BF16 worker, so its executed element work is observable at the `Silu` stage:
// one `Native` record per complete 16-feature tile row the vector kernel
// executes, and one `Fallback` record per element the shared scalar evaluator
// computes instead - every feature beyond the last complete tile row, plus
// every NaN or infinity inside one. Selecting the kernel is not work and is
// never recorded, so an ineligible host or a portable build records nothing
// here. The shared independent oracle keeps owning value parity, tile padding,
// transformed views, and the exceptional classes; this case pairs the
// executed-stage evidence with that oracle on the same submissions.
namespace {

// One `Native` record per complete tile row of each logical row, and one
// `Fallback` record per element the vector worker leaves to the scalar
// evaluator: the tail of a row that does not end on a tile boundary, and every
// contract-incompatible NaN or infinity inside its complete groups.
struct SiluVectorWorkExpectation {
    std::size_t native_groups = 0;
    std::size_t fallback_elements = 0;
};

[[nodiscard]] bool silu_nonfinite(iom::DataType type, std::uint64_t bits) {
    const iom_conformance::SiluReferenceClass value_class =
            iom_conformance::silu_oracle::class_of_bits(type, bits);
    return value_class == iom_conformance::SiluReferenceClass::quiet_nan
            || value_class
                    == iom_conformance::SiluReferenceClass::positive_infinity
            || value_class
                    == iom_conformance::SiluReferenceClass::negative_infinity;
}

[[nodiscard]] SiluVectorWorkExpectation silu_vector_work(
        const iom_conformance::SiluReferenceCase& reference_case) {
    const std::size_t rows =
            iom_conformance::silu_plane_count(
                    reference_case.leading_dimensions)
            * reference_case.runs;
    const std::size_t groups =
            reference_case.features / iom::TensorSpec::TILE;
    const std::size_t grouped_features = groups * iom::TensorSpec::TILE;
    SiluVectorWorkExpectation work;
    work.native_groups = rows * groups;
    work.fallback_elements =
            rows * (reference_case.features - grouped_features);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t feature = 0; feature < grouped_features; ++feature) {
            if (silu_nonfinite(
                        reference_case.data_type,
                        reference_case.input_bits[
                                row * reference_case.features + feature])) {
                ++work.fallback_elements;
            }
        }
    }
    return work;
}

}  // namespace

TEST_CASE("CPU AVX-512 BF16 SiLU records the vector element work it executes") {
    CpuDevices devices;
    const bool eligible = iom::cpu_detail::avx512_bf16_available();
    auto queue = devices.candidate->create_ops();
    const auto observed = [](iom::cpu_detail::Avx512Bf16Path path) {
        return iom::cpu_detail::avx512_bf16_test_observation(
                iom::cpu_detail::Avx512Bf16Stage::Silu, path);
    };
    const std::vector<iom_conformance::SiluReferenceCase> fixtures = {
            // Both stable branches and signed zeros with no feature tail:
            // every element of this fixture must be vector arithmetic, so a
            // nonzero fallback count is itself a defect.
            iom_conformance::silu_make_pattern_case(
                    iom_conformance::SiluReferenceCaseKind::boundary_sizes,
                    iom::DataType::BF16, {2}, 3, 32,
                    "cpu-avx512-bf16-silu/groups"),
            // A non-multiple feature tail over the shared pattern, which ends
            // rows on a partial tile.
            iom_conformance::silu_make_pattern_case(
                    iom_conformance::SiluReferenceCaseKind::boundary_sizes,
                    iom::DataType::BF16, {2}, 2, 17,
                    "cpu-avx512-bf16-silu/tail"),
            // The shared special-value pattern: NaN and both infinities inside
            // complete groups, plus its own feature tail.
            iom_conformance::silu_make_special_case(iom::DataType::BF16),
    };
    for (const iom_conformance::SiluReferenceCase& fixture : fixtures) {
        const iom::TensorSpec spec = iom_conformance::silu_case_spec(fixture);
        auto input = devices.candidate->create_tensor(spec);
        auto activated = devices.candidate->create_tensor(spec);
        iom_conformance::copy_from_host(
                input->view(),
                iom_conformance::silu_pack_bits(
                        fixture.data_type, fixture.input_bits));
        iom::cpu_detail::avx512_bf16_test_reset_observations();
        {
            iom_conformance::SiluCaseWindow window(&devices.gate);
            const iom::oid token =
                    queue->silu(input->view(), activated->view());
            REQUIRE(iom::oid_is_token(token));
            CHECK_NOTHROW(queue->wait(token));
        }
        const std::string mismatch = iom_conformance::silu_compare(
                fixture.data_type,
                iom_conformance::read_logical(activated->view()),
                iom_conformance::silu_evaluate(fixture), fixture.label);
        CHECK_MESSAGE(mismatch.empty(), mismatch);
        const SiluVectorWorkExpectation work = silu_vector_work(fixture);
        CHECK_EQ(
                observed(iom::cpu_detail::Avx512Bf16Path::Native),
                eligible ? std::uint64_t{work.native_groups} : 0);
        CHECK_EQ(
                observed(iom::cpu_detail::Avx512Bf16Path::Fallback),
                eligible ? std::uint64_t{work.fallback_elements} : 0);
    }
    // A leaf outside the BF16 vector path is not that path's fallback: an F32
    // request keeps its own scalar path and records no stage work at all.
    const iom_conformance::SiluReferenceCase scalar =
            iom_conformance::silu_make_pattern_case(
                    iom_conformance::SiluReferenceCaseKind::boundary_sizes,
                    iom::DataType::F32, {1}, 2, 17,
                    "cpu-avx512-bf16-silu/F32-control");
    const iom::TensorSpec scalar_spec = iom_conformance::silu_case_spec(scalar);
    auto scalar_input = devices.candidate->create_tensor(scalar_spec);
    auto scalar_output = devices.candidate->create_tensor(scalar_spec);
    iom_conformance::copy_from_host(
            scalar_input->view(),
            iom_conformance::silu_pack_bits(
                    scalar.data_type, scalar.input_bits));
    iom::cpu_detail::avx512_bf16_test_reset_observations();
    {
        iom_conformance::SiluCaseWindow window(&devices.gate);
        const iom::oid token =
                queue->silu(scalar_input->view(), scalar_output->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
    }
    const std::string scalar_mismatch = iom_conformance::silu_compare(
            scalar.data_type,
            iom_conformance::read_logical(scalar_output->view()),
            iom_conformance::silu_evaluate(scalar), scalar.label);
    CHECK_MESSAGE(scalar_mismatch.empty(), scalar_mismatch);
    CHECK_EQ(observed(iom::cpu_detail::Avx512Bf16Path::Native), 0);
    CHECK_EQ(observed(iom::cpu_detail::Avx512Bf16Path::Fallback), 0);
    CHECK_FALSE(devices.gate.armed());
}
// The CPU port covers the complete nine-leaf RoPE span. The runner owns the
// independent split-half oracle and exercises transformed views, tails,
// poisoned padding, absolute positions, and the fixed zero-workspace query
// through the real asynchronous CPU queue.
TEST_CASE("CPU conformance: RoPE reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::rope_reference::run_rope_conformance(
            *devices.candidate,
            iom_conformance::rope_reference::kRopeCpuExpectedSupported);
    CHECK_FALSE(devices.gate.armed());
}
TEST_CASE("CPU conformance: RoPE keeps inapplicable leaves explicitly unsupported") {
    CpuDevices devices;
    auto queue = devices.candidate->create_ops();
    for (const iom::DataType data_type
            : iom_conformance::rope_reference::kRopeUnsupportedDataTypes) {
        const iom::TensorSpec leaf_shape{
                iom::TensorShape{{1, 1, 1, 16}}, data_type};
        auto source = devices.candidate->create_tensor(leaf_shape);
        auto destination = devices.candidate->create_tensor(leaf_shape);
        CHECK_EQ(
                queue->rope(source->view(), destination->view(), 0, 1.0),
                iom::to_oid(iom::OidError::Unsupported));
        CHECK_THROWS_AS(
                (void)queue->rope_workspace_requirements(
                        source->view(), destination->view(), 0, 1.0),
                std::runtime_error);
    }
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU RoPE retains owners and preserves FIFO ordering for temporary views") {
    CpuDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 18}}, iom::DataType::F32};
    std::unique_ptr<iom::Tensor> source =
            devices.candidate->create_tensor(spec);
    std::unique_ptr<iom::Tensor> first =
            devices.candidate->create_tensor(spec);
    std::unique_ptr<iom::Tensor> second =
            devices.candidate->create_tensor(spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0xC0DE);
    const std::vector<std::byte> zero(
            spec.logical_nbytes(), std::byte{0});
    iom_conformance::copy_from_host(source->view(), pattern);
    iom_conformance::copy_from_host(first->view(), zero);
    iom_conformance::copy_from_host(second->view(), zero);

    const iom::TensorView first_slice = first->view().slice(0, 0, 1);
    const iom::TensorView second_slice = second->view().slice(0, 0, 1);
    const std::size_t slice_bytes = first_slice.spec().logical_nbytes();
    const std::vector<std::byte> expected_slice(
            pattern.begin(),
            pattern.begin() + static_cast<std::ptrdiff_t>(slice_bytes));

    auto queue = devices.candidate->create_ops();
    const iom::oid rope_token =
            submit_temporary_rope(*queue, *source, *first);
    REQUIRE(iom::oid_is_token(rope_token));
    // The worker must retain source's owner registration after this reset.
    source.reset();

    const iom::oid copy_token = queue->copy(
            first_slice,
            const_cast<iom::TensorView&>(
                    static_cast<const iom::TensorView&>(second_slice)));
    REQUIRE(iom::oid_is_token(copy_token));
    REQUIRE_NOTHROW(queue->wait(copy_token));
    CHECK_NOTHROW(queue->wait(rope_token));
    CHECK_NOTHROW(queue->wait(copy_token));

    const std::vector<std::byte> observed_first =
            iom_conformance::read_logical(first_slice);
    const std::vector<std::byte> observed_second =
            iom_conformance::read_logical(second_slice);
    CHECK(observed_second == observed_first);

    // Position zero is an encoded-bit identity. Checking each head's first
    // logical row also proves that the temporary-view slice addressed the
    // expected independent head planes rather than a contiguous owner base.
    const std::size_t rank = first_slice.spec().shape.rank();
    const std::size_t heads = first_slice.spec().shape.dimension(rank - 3);
    const std::size_t rows = first_slice.spec().shape.dimension(rank - 2);
    const std::size_t width = first_slice.spec().shape.dimension(rank - 1);
    const std::size_t row_bytes = width * sizeof(float);
    const std::size_t head_bytes = rows * row_bytes;
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_bytes;
        for (std::size_t byte = 0; byte < row_bytes; ++byte) {
            CHECK_EQ(observed_first[offset + byte], expected_slice[offset + byte]);
        }
    }
    CHECK_FALSE(devices.gate.armed());
    // CPU has no fault-injection seam for an accepted worker failure; repeat
    // waits above cover the successful retained-completion path.
}



TEST_CASE("CPU conformance: full shared suite composes every shared case") {
    CpuDevices devices;
    const iom_conformance::RopeContractConformanceConfig rope_contract{
            devices.conformance(),
            iom_conformance::rope_reference::kRopeCpuExpectedSupported,
            &devices.gate,
            iom_conformance::RopeNativeFailureSeam{
                    {},
                    {},
                    "cpu RoPE accepted-failure seam",
                    "the CPU port exposes no native fault-injection seam "
                    "for a post-acceptance RoPE worker failure"}};
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, nullptr, true, true, &rope_contract, true);
    CHECK_FALSE(devices.gate.armed());
}

// The CPU reference instantiation of the shared model-loading scenario. Each
// synthetic checkpoint is loaded through the production configuration and
// mapped-source API, realized on the CPU reference and candidate devices, and
// candidate binding reports `{0, 1}` for the model-loading operations, which
// do not use the positive workspace owner reserved for CPU SDPA.
TEST_CASE("CPU model loading realizes every published weight role of each synthetic checkpoint") {
    CpuDevices devices;
    iom_conformance::run_model_loading_conformance(devices.conformance());
}

TEST_CASE("Inference instrumentation parity preserves TinyLlama behavior and observations") {
    CpuDevices devices;
    iom_conformance::run_inference_instrumentation_conformance(*devices.candidate);
}

// Opt-in real-checkpoint loading, compiled only with
// `IOM_TEST_REAL_MODEL_LOADING=ON`: the ordinary CPU suite reads no model
// directory, assumes no default path, and downloads nothing. The case is this
// driver's selected CPU device over the caller's own allocator, which is the
// CPU ownership path; no second full-weight binding is created for a reference
// payload.
#ifdef IOM_TEST_REAL_MODEL_LOADING
TEST_CASE("CPU real model loading") {
    CpuDevices devices;
    iom_conformance::run_real_model_loading(*devices.candidate);
}
#endif

// Opt-in official TinyLlama inference, compiled only with
// `IOM_TEST_REAL_MODEL_INFERENCE_CPU=ON`. The common runner owns artifact
// verification, real prefill/cached decode, comparisons, and evidence; this
// driver contributes only CPU device 0 backed by its caller-owned allocator.
#ifdef IOM_TEST_REAL_MODEL_INFERENCE_CPU
TEST_CASE("CPU real model inference") {
    CpuDevices devices;
    iom_conformance::run_real_model_inference(*devices.candidate);
}
#endif

// A deliberately perturbed candidate layout or view map must surface as a
// bit-for-bit logical-byte mismatch; the independently generated encodings
// make round-trip cancellation impossible.
TEST_CASE("conformance harness detects perturbed candidate bytes") {
    iom_conformance::TrafficGate gate;
    GatedAllocator reference_allocator(gate);
    GatedAllocator candidate_allocator(gate);
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_cpu_device(candidate_allocator);

    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::U8};
    auto reference_tensor = reference->create_tensor(spec);
    auto candidate_tensor = candidate->create_tensor(spec);

    const std::vector<std::byte> seeded = iom_conformance::encode_logical(spec, 7);
    iom_conformance::copy_from_host(reference_tensor->view(), seeded);
    iom_conformance::copy_from_host(candidate_tensor->view(), seeded);
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
    iom_conformance::copy_from_host(stepped_window, stepped_pattern);
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
    iom_conformance::copy_from_host(reference->view(), pattern);
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
    iom_conformance::copy_from_host(reference->view(), pattern);
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
        iom_conformance::copy_from_host(reference->view(), pattern);
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
    iom_conformance::copy_from_host(reference->view(), pattern);
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
    iom_conformance::copy_from_host(t->view(), pattern);
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

TEST_CASE("CPU conformance: raw workspace contract supports empty and positive owners") {
    CpuDevices devices;

    // The empty workspace is valid on every backend and performs no
    // allocation: the traffic gate stays silent through creation and
    // destruction.
    devices.gate.setup_complete();
    {
        const std::unique_ptr<iom::RawWorkspace> workspace =
                devices.candidate->create_workspace(0);
        REQUIRE(workspace != nullptr);
        CHECK(workspace->empty());
        CHECK_EQ(workspace->byte_size(), 0);
        CHECK(&workspace->device() == devices.candidate.get());
        CHECK(workspace->backend_kind() == iom::BackendKind::CPU);

        const iom::RawWorkspaceView view = workspace->view();
        CHECK(view.owner_identity() == workspace.get());
        CHECK(&view.device() == devices.candidate.get());
        CHECK_EQ(view.byte_size(), 0);
        CHECK_FALSE(view.empty());
    }
    devices.gate.case_complete();
    CHECK_FALSE(devices.gate.armed());

    const std::unique_ptr<iom::RawWorkspace> candidate_workspace =
            devices.candidate->create_workspace(1);
    REQUIRE(candidate_workspace != nullptr);
    CHECK_FALSE(candidate_workspace->empty());
    CHECK_EQ(candidate_workspace->byte_size(), 1);
    CHECK(&candidate_workspace->device() == devices.candidate.get());
    const iom::RawWorkspaceView candidate_view = candidate_workspace->view();
    CHECK(candidate_view.owner_identity() == candidate_workspace.get());
    CHECK(&candidate_view.device() == devices.candidate.get());
    CHECK_EQ(candidate_view.byte_size(), 1);

    const std::unique_ptr<iom::RawWorkspace> reference_workspace =
            devices.reference->create_workspace(4096);
    REQUIRE(reference_workspace != nullptr);
    CHECK_EQ(reference_workspace->byte_size(), 4096);
    CHECK(
            &reference_workspace->device() == devices.reference.get());
}

TEST_CASE("CPU conformance: workspace requirement queries report exact zero") {
    CpuDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::F32};
    auto lhs = devices.candidate->create_tensor(spec);
    auto rhs = devices.candidate->create_tensor(spec);
    auto out = devices.candidate->create_tensor(spec);
    auto queue = devices.candidate->create_ops();

    const iom::WorkspaceRequirements zero{0, 1};
    CHECK(queue->add_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->mul_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->sub_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->div_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(lhs->view().copy_from_host_workspace_requirements() == zero);
    CHECK(lhs->view().copy_to_host_workspace_requirements() == zero);

    // Validation matches the binary operation: foreign-device views are
    // invalid input before any requirement is reported.
    auto foreign = devices.foreign->create_tensor(spec);
    CHECK_THROWS_AS((void)
            queue->add_workspace_requirements(
                    foreign->view(), rhs->view(), out->view()),
            std::invalid_argument);

    // Pure queries cause no allocator, registration, lease, token, or
    // queue traffic.
    devices.gate.setup_complete();
    (void)queue->add_workspace_requirements(
            lhs->view(), rhs->view(), out->view());
    (void)lhs->view().copy_from_host_workspace_requirements();
    (void)lhs->view().copy_to_host_workspace_requirements();
    devices.gate.case_complete();
    CHECK_FALSE(devices.gate.armed());
}
