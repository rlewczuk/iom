#include <doctest/doctest.h>

#include <tt-metalium/host_api.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "iom/cpu/device.hpp"
#include "iom/ttnn/device.hpp"

namespace {

// Plain 32-byte-aligned heap allocator for the CPU reference and foreign
// devices. TTNN tensors never touch an iom::Allocator: the TTNN device
// factory does not accept one and native storage is TTNN-owned.
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

struct TtnnDevices {
    HostAllocator reference_allocator;
    HostAllocator foreign_allocator;
    std::unique_ptr<iom::Device> reference =
            iom::make_cpu_device(reference_allocator);
    std::unique_ptr<iom::Device> candidate = iom::make_ttnn_device(0);
    // tt-metal permits only one live context per physical device per
    // process: a second make_ttnn_device(0) cannot coexist with the
    // candidate, so an independently created CPU device fills the foreign
    // slot. Queue validation rejects views by Device identity, which is
    // exactly what this exercises.
    std::unique_ptr<iom::Device> foreign =
            iom::make_cpu_device(foreign_allocator);

    [[nodiscard]] iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};

// Hardware presence is required, never skipped: an enabled TTNN conformance
// target fails when no Tenstorrent runtime is available.
void require_hardware() {
    REQUIRE(tt::tt_metal::GetNumAvailableDevices() > 0);
}

}  // namespace

// The supported-type table is explicit, nonempty, includes BF16, and is the
// single source for both creation validation and conformance
// parameterization. Every other declared leaf type and every grouped
// quantization format is rejected before native allocation.
TEST_CASE("TTNN supported-type table acceptance and rejection") {
    require_hardware();
    auto device = iom::make_ttnn_device(0);

    const std::span<const iom::DataType> supported =
            iom::ttnn_supported_data_types();
    REQUIRE_FALSE(supported.empty());
    CHECK(std::find(supported.begin(), supported.end(), iom::DataType::BF16)
          != supported.end());

    constexpr iom::DataType kAllLeafTypes[] = {
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

    for (const iom::DataType type : kAllLeafTypes) {
        CAPTURE(static_cast<int>(type));
        const iom::TensorSpec spec{iom::TensorShape{{16, 16}}, type};
        if (std::find(supported.begin(), supported.end(), type)
                != supported.end()) {
            CHECK_NOTHROW(device->create_tensor(spec));
        } else {
            CHECK_THROWS_AS(device->create_tensor(spec), std::runtime_error);
        }
    }

    for (const iom::QuantizationFormat format :
         {iom::QuantizationFormat::INT8_SYMMETRIC,
          iom::QuantizationFormat::GGML_Q4_0,
          iom::QuantizationFormat::TT_BFP8}) {
        CAPTURE(static_cast<int>(format));
        const iom::TensorSpec spec{
                iom::TensorShape{{16, 16}}, iom::DataType::BF16, format};
        CHECK_THROWS_AS(device->create_tensor(spec), std::runtime_error);
    }
}

// Every extent the TTNN native constructor cannot represent is rejected
// with std::overflow_error before any native object or allocation exists.
// The three review-cited shapes previously wrapped or narrowed silently;
// the boundary case proves the largest representable extent is retained.
TEST_CASE("TTNN rejects overflowing and narrowing extents before native allocation") {
    require_hardware();
    auto device = iom::make_ttnn_device(0);

    // Leading-plane product wraps modulo 2^64 to zero: the unchecked path
    // materialized zero native planes for a logical 2^63 * 2 * 16 * 16
    // plane count, and copies silently did no work.
    {
        const iom::TensorSpec spec{
                iom::TensorShape{{std::size_t{1} << 63, 2, 16, 16}},
                iom::DataType::BF16};
        CHECK_THROWS_AS(device->create_tensor(spec), std::overflow_error);
    }

    // Column count exceeds the uint32_t native extent ceiling: the
    // unchecked static_cast narrowed 2^32 + 1 columns to one, undersizing
    // the native plane and later host-buffer copies.
    {
        const iom::TensorSpec spec{
                iom::TensorShape{{1, (std::size_t{1} << 32) + 1}},
                iom::DataType::BF16};
        CHECK_THROWS_AS(device->create_tensor(spec), std::overflow_error);
    }

    // The leading-plane product overflows even though every individual
    // extent fits: 2 * (SIZE_MAX / 2 + 1) wraps modulo 2^64.
    {
        const iom::TensorSpec spec{
                iom::TensorShape{
                        {2,
                         std::numeric_limits<std::size_t>::max() / 2 + 1,
                         16, 16}},
                iom::DataType::BF16};
        CHECK_THROWS_AS(device->create_tensor(spec), std::overflow_error);
    }

    // The largest representable extent is retained: the maximal uint32_t
    // column count passes every pre-allocation check, so the call is never
    // rejected with std::overflow_error. On real hardware the only
    // rejection is the TTNN native allocator, which cannot back the
    // 256 GiB tiled plane of a {1, UINT32_MAX} BF16 tensor on any existing
    // device. A std::overflow_error here would mean the fix over-rejects
    // the largest extent the TTNN runtime can represent.
    {
        const iom::TensorSpec spec{
                iom::TensorShape{{1, std::numeric_limits<std::uint32_t>::max()}},
                iom::DataType::BF16};
        bool validation_rejected = false;
        try {
            auto tensor = device->create_tensor(spec);
            // A device with enough memory: creation succeeds and the
            // logical metadata stays unchanged.
            if (tensor != nullptr) {
                CHECK(tensor->view().spec().shape.dimension(0) == 1);
                CHECK(tensor->view().spec().shape.dimension(1)
                      == std::numeric_limits<std::uint32_t>::max());
            }
        } catch (const std::overflow_error&) {
            validation_rejected = true;
        } catch (const std::exception& e) {
            // Any exception here is the native allocation path rejecting a
            // validated extent; the pre-allocation checks accepted it. The
            // TT_FATAL payload embeds a backtrace; keep its first line.
            const std::string native_rejection = e.what();
            const std::size_t end = native_rejection.find('\n');
            MESSAGE("boundary extent passed validation; native rejection: "
                    << native_rejection.substr(0, end));
        }
        CHECK_FALSE(validation_rejected);
    }

    // Rejections leave the device healthy: a minimal supported tensor still
    // creates afterwards.
    CHECK_NOTHROW(device->create_tensor(
            iom::TensorSpec{iom::TensorShape{{16, 16}}, iom::DataType::BF16}));
}

TEST_CASE("TTNN conformance: storage and host transfers for every supported type") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), iom::ttnn_supported_data_types());
}

TEST_CASE("TTNN conformance: asynchronous copies against the CPU reference") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), iom::ttnn_supported_data_types());
}

TEST_CASE("TTNN conformance: copy validation fails before writes and sequences") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), iom::ttnn_supported_data_types());
}

TEST_CASE("TTNN conformance: transfer failures keep metadata and ownership") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_transfer_error_conformance(
            devices.conformance(), iom::ttnn_supported_data_types());
}

TEST_CASE("TTNN conformance: deferred queue lifetime and stability") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, iom::ttnn_supported_data_types());
}

TEST_CASE("TTNN conformance: compute methods reject capability without submitting") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, iom::ttnn_supported_data_types());
}

TEST_CASE("TTNN conformance: full shared suite") {
    require_hardware();
    TtnnDevices devices;
    const std::span<const iom::DataType> supported =
            iom::ttnn_supported_data_types();
    iom_conformance::run_backend_conformance(
            devices.conformance(), supported.subspan(0, 1));
}
