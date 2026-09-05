#include <doctest/doctest.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>
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

// The TTNN factory is the hardware gate. When no device is available it
// throws, so enabled conformance targets fail instead of skipping.
void require_hardware() {}

tt::tt_metal::DataType native_dtype(iom::DataType type) {
    switch (type) {
        case iom::DataType::BOOL:
        case iom::DataType::U8:
        case iom::DataType::I8:
            return tt::tt_metal::DataType::UINT8;
        case iom::DataType::U16:
        case iom::DataType::I16:
            return tt::tt_metal::DataType::UINT16;
        case iom::DataType::U32:
            return tt::tt_metal::DataType::UINT32;
        case iom::DataType::I32:
            return tt::tt_metal::DataType::INT32;
        case iom::DataType::BF16:
            return tt::tt_metal::DataType::BFLOAT16;
        case iom::DataType::F32:
            return tt::tt_metal::DataType::FLOAT32;
        default:
            throw std::invalid_argument(
                    "unsupported TTNN oracle data type");
    }
}

template <typename T>
tt::tt_metal::HostBuffer make_host_buffer(
        std::vector<std::byte>& bytes) {
    std::vector<T> typed(bytes.size() / sizeof(T));
    std::memcpy(typed.data(), bytes.data(), bytes.size());
    return tt::tt_metal::HostBuffer(std::move(typed));
}

tt::tt_metal::HostBuffer make_host_buffer(
        tt::tt_metal::DataType type, std::vector<std::byte>& bytes) {
    switch (type) {
        case tt::tt_metal::DataType::BFLOAT16:
            return make_host_buffer<bfloat16>(bytes);
        case tt::tt_metal::DataType::FLOAT32:
            return make_host_buffer<float>(bytes);
        case tt::tt_metal::DataType::UINT32:
            return make_host_buffer<std::uint32_t>(bytes);
        case tt::tt_metal::DataType::INT32:
            return make_host_buffer<std::int32_t>(bytes);
        case tt::tt_metal::DataType::UINT16:
            return make_host_buffer<std::uint16_t>(bytes);
        case tt::tt_metal::DataType::UINT8:
            return make_host_buffer<std::uint8_t>(bytes);
        default:
            throw std::invalid_argument(
                    "unsupported TTNN oracle host data type");
    }
}

std::size_t oracle_view_plane_at(
        const iom::TensorView& view, std::size_t index) {
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::span<const std::size_t> strides = view.plane_strides();
    std::size_t plane = view.plane_offset();
    for (std::size_t axis = leading_rank; axis-- > 0;) {
        plane += (index % dimensions[axis]) * strides[axis];
        index /= dimensions[axis];
    }
    return plane;
}

std::size_t owner_plane_count(const iom::TensorSpec& spec) {
    const std::span<const std::size_t> dimensions = spec.shape.dimensions();
    std::size_t count = 1;
    for (std::size_t axis = 0; axis + 2 < dimensions.size(); ++axis) {
        count *= dimensions[axis];
    }
    return count;
}
template <typename T>
void observe_plane_values(
        const ttnn::Tensor& plane, const iom::TensorSpec& owner,
        std::size_t plane_index, std::vector<std::byte>& storage) {
    const std::span<const std::size_t> dimensions = owner.shape.dimensions();
    const std::size_t rows = dimensions[dimensions.size() - 2];
    const std::size_t columns = dimensions[dimensions.size() - 1];
    const std::vector<T> values = plane.to_vector<T>();
    REQUIRE(values.size() >= rows * columns);
    const auto* bytes = reinterpret_cast<const std::byte*>(values.data());
    const std::size_t element_bytes = sizeof(T);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t slot = iom::detail::standard_plane_slot(
                    owner, plane_index, row, column);
            std::memcpy(
                    storage.data() + slot * element_bytes,
                    bytes + (row * columns + column) * element_bytes,
                    element_bytes);
        }
    }
}

class TtnnStorageOracle final
        : public iom_conformance::AcceleratorStorageOracle {
public:
    void seed(
            iom::TensorView& view,
            std::span<const std::byte> encoded) override {
        const iom::TensorSpec& owner = owner_spec();
        REQUIRE_EQ(encoded.size(), owner.tiled_storage_nbytes());
        const std::span<const std::size_t> dimensions =
                owner.shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t columns = dimensions[dimensions.size() - 1];
        const std::size_t planes_count = owner_plane_count(owner);
        const std::size_t bits = iom::detail::leaf_bits(owner.data_type);
        REQUIRE_EQ(bits % 8, std::size_t{0});
        const std::size_t element_bytes = bits / 8;
        auto* planes = static_cast<ttnn::Tensor*>(view.native_handle());

        // Exercise the view-coordinate plane map independently of the
        // production TTNN copy helper. The owner allocation itself is seeded
        // plane-by-plane so transformed views retain untouched planes.
        const std::size_t view_count =
                view.spec().shape.element_count() / (rows * columns);
        for (std::size_t i = 0; i < view_count; ++i) {
            REQUIRE(oracle_view_plane_at(view, i) < planes_count);
        }
        for (std::size_t plane_index = 0; plane_index < planes_count;
             ++plane_index) {
            ttnn::Tensor& plane = planes[plane_index];
            const std::size_t padded_rows =
                    static_cast<std::size_t>(plane.padded_shape()[-2]);
            const std::size_t padded_columns =
                    static_cast<std::size_t>(plane.padded_shape()[-1]);
            std::vector<std::byte> padded(
                    padded_rows * padded_columns * element_bytes,
                    std::byte{0});
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t column = 0; column < columns; ++column) {
                    const std::size_t slot = iom::detail::standard_plane_slot(
                            owner, plane_index, row, column);
                    std::memcpy(
                            padded.data()
                                    + (row * padded_columns + column)
                                            * element_bytes,
                            encoded.data() + slot * element_bytes,
                            element_bytes);
                }
            }
            ttnn::Tensor host_row_major(
                    make_host_buffer(plane.dtype(), padded),
                    plane.logical_shape(), plane.padded_shape(), plane.dtype(),
                    tt::tt_metal::Layout::ROW_MAJOR);
            const ttnn::Tensor host_tiled = tt::tt_metal::to_layout(
                    host_row_major, tt::tt_metal::Layout::TILE);
            ttnn::copy_to_device(host_tiled, plane);
        }
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        const iom::TensorSpec& owner = owner_spec();
        std::vector<std::byte> storage(
                owner.tiled_storage_nbytes(), std::byte{0});
        const std::span<const std::size_t> dimensions =
                owner.shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t columns = dimensions[dimensions.size() - 1];
        const std::size_t planes_count = owner_plane_count(owner);
        const std::size_t view_count =
                view.spec().shape.element_count() / (rows * columns);
        for (std::size_t i = 0; i < view_count; ++i) {
            REQUIRE(oracle_view_plane_at(view, i) < planes_count);
        }
        const auto* planes =
                static_cast<const ttnn::Tensor*>(view.native_handle());
        switch (native_dtype(owner.data_type)) {
            case tt::tt_metal::DataType::BFLOAT16:
                for (std::size_t i = 0; i < planes_count; ++i) {
                    observe_plane_values<bfloat16>(
                            planes[i], owner, i, storage);
                }
                break;
            case tt::tt_metal::DataType::FLOAT32:
                for (std::size_t i = 0; i < planes_count; ++i) {
                    observe_plane_values<float>(planes[i], owner, i, storage);
                }
                break;
            case tt::tt_metal::DataType::UINT32:
                for (std::size_t i = 0; i < planes_count; ++i) {
                    observe_plane_values<std::uint32_t>(
                            planes[i], owner, i, storage);
                }
                break;
            case tt::tt_metal::DataType::INT32:
                for (std::size_t i = 0; i < planes_count; ++i) {
                    observe_plane_values<std::int32_t>(
                            planes[i], owner, i, storage);
                }
                break;
            case tt::tt_metal::DataType::UINT16:
                for (std::size_t i = 0; i < planes_count; ++i) {
                    observe_plane_values<std::uint16_t>(
                            planes[i], owner, i, storage);
                }
                break;
            case tt::tt_metal::DataType::UINT8:
                for (std::size_t i = 0; i < planes_count; ++i) {
                    observe_plane_values<std::uint8_t>(
                            planes[i], owner, i, storage);
                }
                break;
            default:
                throw std::logic_error("unsupported TTNN oracle dtype");
        }
        return storage;
    }
};


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
TEST_CASE("Device::supported_data_types returns the per-backend 9-entry span") {
    require_hardware();
    const std::unique_ptr<iom::Device> candidate =
            iom::make_ttnn_device(0);
    const std::span<const iom::DataType> supported =
            candidate->supported_data_types();
    const std::span<const iom::DataType> canonical =
            iom::ttnn_supported_data_types();
    constexpr iom::DataType expected[] = {
            iom::DataType::BOOL, iom::DataType::U8, iom::DataType::I8,
            iom::DataType::U16, iom::DataType::I16, iom::DataType::U32,
            iom::DataType::I32, iom::DataType::BF16, iom::DataType::F32,
    };
    REQUIRE_EQ(supported.size(), sizeof(expected) / sizeof(expected[0]));
    for (std::size_t i = 0; i < supported.size(); ++i) {
        CHECK_EQ(supported[i], expected[i]);
    }
    CHECK_EQ(supported.data(), canonical.data());
    CHECK_EQ(supported.size(), canonical.size());
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

TEST_CASE("TTNN conformance: storage oracle identifies perturbed transfer map") {
    require_hardware();
    TtnnDevices devices;
    const std::span<const iom::DataType> supported =
            iom::ttnn_supported_data_types();
    const std::span<const iom::DataType> one_type = supported.subspan(0, 1);
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), one_type);

    TtnnStorageOracle direct;
    iom_conformance::PermutingStorageOracle perturbed(
            direct, iom_conformance::swap_first_adjacent_slots);
    CHECK_FALSE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, perturbed, nullptr, false,
            false));
}

TEST_CASE("TTNN conformance: storage oracle covers every leaf width and padded shape") {
    require_hardware();
    TtnnDevices devices;
    TtnnStorageOracle oracle;
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), iom::ttnn_supported_data_types(), oracle));
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
