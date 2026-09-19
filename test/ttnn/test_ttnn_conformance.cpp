#include <doctest/doctest.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <exception>
#include <iomanip>

#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <stdexcept>
#include <vector>

#include "../../src/ttnn/staging.hpp"
#include "backend/backend_conformance_oracle.hpp"
#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_embedding.hpp"
#include "backend/backend_conformance_linear.hpp"
#include "backend/backend_conformance_add.hpp"
#include "backend/backend_conformance_other.hpp"
#include "backend/backend_conformance_rmsnorm.hpp"
#include "backend/backend_conformance_model_loading.hpp"
#include "iom/cpu/device.hpp"
#include "iom/ttnn/device.hpp"
#include "../../src/ttnn/device_internal.hpp"
#include "../../src/ttnn/copy.hpp"
#include "../../src/ttnn/registry_state.hpp"

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
// Nonzero sentinel pre-seeded into the observer's independent byte
// representation. Any native cell the readback fails to observe — a padded
// slot in particular — stays nonzero and diverges from the zero-filled
// expected model instead of cancelling against it.
constexpr std::byte kObserverStorageSentinel{0xA5};

// One native TTNN TILE plane's padded geometry. The TTNN runtime pads every
// plane to 32x32-tile multiples and stores the complete allocation in
// physical tile-major order: tiles row-major over the padded grid, every
// 32x32 tile as four row-major 16x16 faces. This is the same tile geometry
// the TTNN transfer implementation assumes for its host staging.
struct TtnnPlaneLayout {
    std::size_t padded_rows = 0;
    std::size_t padded_columns = 0;
    std::size_t tile_columns = 0;

    explicit TtnnPlaneLayout(const ttnn::Tensor& plane) {
        padded_rows =
                static_cast<std::size_t>(plane.padded_shape()[-2]);
        padded_columns =
                static_cast<std::size_t>(plane.padded_shape()[-1]);
        tile_columns = padded_columns / 32;
    }

    // Physical tile-major element index of the padded coordinate (row,
    // column) inside this plane's native readback.
    [[nodiscard]] std::size_t element_index(
            std::size_t row, std::size_t column) const {
        const std::size_t tile_index =
                (row / 32) * tile_columns + column / 32;
        const std::size_t face_index =
                ((row % 32) / 16) * 2 + ((column % 32) / 16);
        return tile_index * 1024 + face_index * 256
                + (row % 16) * 16 + (column % 16);
    }
};

// Builds one native plane's padded row-major host image from the encoded
// canonical image: every canonical cell — logical and padded alike — is
// copied from its own canonical slot, and native cells beyond the canonical
// padded extents stay zero. Optionally plants nonzero sentinels at further
// padded coordinates. Uses only the standard slot arithmetic, independent of
// the production TTNN copy helper, so the seed path, the padding-mutation
// probe, and the shared padded-physical conformance case share one geometry.
std::vector<std::byte> padded_plane_image(
        const ttnn::Tensor& plane, const iom::TensorSpec& owner,
        std::size_t plane_index, std::span<const std::byte> encoded,
        std::span<const std::pair<std::size_t, std::size_t>> planted = {}) {
    const std::span<const std::size_t> dimensions = owner.shape.dimensions();
    const std::size_t rows = dimensions[dimensions.size() - 2];
    const std::size_t columns = dimensions[dimensions.size() - 1];
    const std::size_t padded_rows =
            static_cast<std::size_t>(plane.padded_shape()[-2]);
    const std::size_t padded_columns =
            static_cast<std::size_t>(plane.padded_shape()[-1]);
    const std::size_t bits = iom::detail::leaf_bits(owner.data_type);
    const std::size_t carrier_bytes =
            plane.dtype() == tt::tt_metal::DataType::UINT8
            ? 1
            : (plane.dtype() == tt::tt_metal::DataType::UINT16
                               || plane.dtype()
                                          == tt::tt_metal::DataType::BFLOAT16
                       ? 2
                       : 4);
    const std::size_t factor = bits > 32 ? 2 : 1;
    const std::size_t canonical_rows =
            iom_conformance::canonical_padded_extent(rows);
    const std::size_t canonical_columns =
            iom_conformance::canonical_padded_extent(columns);
    REQUIRE_LE(canonical_rows, padded_rows);
    REQUIRE_LE(canonical_columns * factor, padded_columns);
    std::vector<std::byte> padded(
            padded_rows * padded_columns * carrier_bytes, std::byte{0});
    for (std::size_t row = 0; row < canonical_rows; ++row) {
        for (std::size_t column = 0; column < canonical_columns; ++column) {
            const std::size_t slot = iom_conformance::canonical_plane_slot(
                    owner, plane_index, row, column);
            std::uint64_t value = 0;
            for (std::size_t bit = 0; bit < bits; ++bit) {
                const std::size_t source_bit = slot * bits + bit;
                if ((std::to_integer<unsigned char>(
                            encoded[source_bit / 8])
                     >> (source_bit % 8))
                    & 1u) {
                    value |= std::uint64_t{1} << bit;
                }
            }
            for (std::size_t part = 0; part < factor; ++part) {
                std::memcpy(
                        padded.data()
                                + (row * padded_columns + column * factor
                                   + part)
                                        * carrier_bytes,
                        reinterpret_cast<const std::byte*>(&value)
                                + part * carrier_bytes,
                        carrier_bytes);
            }
    }
    }
    for (const auto& [row, column] : planted) {
        const bool planted_inside_logical =
                row < rows && column < columns * factor;
        REQUIRE_FALSE(planted_inside_logical);
        REQUIRE(row < padded_rows);
        REQUIRE(column < padded_columns);
        std::byte* cell =
                padded.data()
                + (row * padded_columns + column) * carrier_bytes;
        std::fill(cell, cell + carrier_bytes, kObserverStorageSentinel);
    }
    return padded;
}

// Writes one padded row-major host image into a native plane through TTNN's
// own layout conversion, the same independent native path the oracle seed
// uses.
void write_padded_plane_image(
        ttnn::Tensor& plane, std::vector<std::byte>& padded) {
    ttnn::Tensor host_row_major(
            make_host_buffer(plane.dtype(), padded),
            plane.logical_shape(), plane.padded_shape(), plane.dtype(),
            tt::tt_metal::Layout::ROW_MAJOR);
    const ttnn::Tensor host_tiled = tt::tt_metal::to_layout(
            host_row_major, tt::tt_metal::Layout::TILE);
    ttnn::copy_to_device(host_tiled, plane);
}
// Reads one complete native padded TILE plane into row-major carrier order,
// independently of the production copy path.
std::vector<std::byte> read_padded_plane_image(
        const ttnn::Tensor& plane) {
    const TtnnPlaneLayout native(plane);
    const std::size_t carrier_size =
            iom::ttnn_detail::carrier_bytes(plane.dtype());
    const std::size_t bytes =
            native.padded_rows * native.padded_columns * carrier_size;
    std::vector<std::byte> tiled(bytes);
    auto* device = plane.device();
    REQUIRE(device != nullptr);
    auto& queue = device->mesh_command_queue(0);
    ttnn::copy_to_host(
            queue, plane, tiled.data(), std::nullopt, /*blocking=*/false);
    queue.finish();

    std::vector<std::byte> row_major(bytes);
    for (std::size_t row = 0; row < native.padded_rows; ++row) {
        for (std::size_t column = 0; column < native.padded_columns;
             ++column) {
            std::memcpy(
                    row_major.data()
                            + (row * native.padded_columns + column)
                                    * carrier_size,
                    tiled.data()
                            + native.element_index(row, column) * carrier_size,
                    carrier_size);
        }
    }
    return row_major;
}

// Observes one native plane's complete padded TILE readback into standard
// slot order, mapping every standard padded coordinate — padded rows and
// columns included — by its standard slot. The readback is raw physical
// tile-major bytes; coordinates are mapped through the native tile geometry,
// never by assuming the native and standard layouts agree, so native padding
// writes and physical tile-slot permutations surface instead of being
bool observe_plane_storage(
        std::span<const std::byte> readback,
        const ttnn::Tensor& plane, const iom::TensorSpec& owner,
        std::size_t plane_index, std::size_t carrier_bytes,
        std::vector<std::byte>& storage) {
    const TtnnPlaneLayout native(plane);
    REQUIRE_EQ(readback.size(),
               native.padded_rows * native.padded_columns * carrier_bytes);
    const std::span<const std::size_t> dimensions = owner.shape.dimensions();
    const std::size_t bits = iom::detail::leaf_bits(owner.data_type);
    const std::size_t rows = dimensions[dimensions.size() - 2];
    const std::size_t columns = dimensions[dimensions.size() - 1];
    const std::size_t factor = bits > 32 ? 2 : 1;
    const std::size_t padded_rows =
            iom_conformance::canonical_padded_extent(rows);
    const std::size_t padded_columns =
            iom_conformance::canonical_padded_extent(columns);
    bool native_padding_zero = true;
    for (std::size_t row = 0; row < native.padded_rows; ++row) {
        for (std::size_t column = 0; column < native.padded_columns;
             ++column) {
            const std::byte* cell = readback.data()
                    + native.element_index(row, column) * carrier_bytes;
            if (row >= padded_rows || column >= padded_columns * factor) {
                native_padding_zero = native_padding_zero
                        && std::all_of(
                                cell, cell + carrier_bytes,
                                [](std::byte value) {
                                    return value == std::byte{0};
                                });
            }
        }
        if (row >= padded_rows) continue;
        for (std::size_t logical_column = 0;
             logical_column < padded_columns
                    && logical_column * factor < native.padded_columns;
             ++logical_column) {
            std::uint64_t value = 0;
            for (std::size_t part = 0; part < factor; ++part) {
                const std::size_t column = logical_column * factor + part;
                if (column >= native.padded_columns) continue;
                const std::byte* cell = readback.data()
                        + native.element_index(row, column) * carrier_bytes;
                std::uint32_t carrier = 0;
                std::memcpy(&carrier, cell, carrier_bytes);
                value |= std::uint64_t{carrier} << (part * 32);
            }
            const std::size_t slot = iom_conformance::canonical_plane_slot(
                    owner, plane_index, row, logical_column);
            iom_conformance::write_storage_bits(
                    storage.data(), slot * bits, bits, value);
        }
    }
    return native_padding_zero;
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
            std::vector<std::byte> padded =
                    padded_plane_image(plane, owner, plane_index, encoded);
            write_padded_plane_image(plane, padded);
        }
    }

    [[nodiscard]] std::vector<std::byte> observe(
            const iom::TensorView& view) const override {
        const iom::TensorSpec& owner = owner_spec();
        // The independent byte representation is pre-seeded with a nonzero
        // sentinel: padded rows, padded columns, and untouched owner planes
        // are reported from the native readback, so a cell the observer fails
        // to fill cannot cancel against zero-initialized expected padding.
        std::vector<std::byte> storage(
                owner.tiled_storage_nbytes(), kObserverStorageSentinel);
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
        const std::size_t carrier_bytes =
                planes[0].dtype() == tt::tt_metal::DataType::UINT8
                ? 1
                : (planes[0].dtype() == tt::tt_metal::DataType::UINT16
                                   || planes[0].dtype()
                                              == tt::tt_metal::DataType::BFLOAT16
                           ? 2
                           : 4);

        // Raw physical readback of every owner plane. copy_to_host returns
        // the complete native padded allocation in tile-major order — the
        // same staging layout the production download assembly consumes —
        // including padded rows, padded columns, and untouched planes.
        auto* device = planes[0].device();
        REQUIRE(device != nullptr);
        auto& queue = device->mesh_command_queue(0);
        std::vector<std::size_t> plane_bytes(planes_count);
        std::size_t staging_bytes = 0;
        for (std::size_t i = 0; i < planes_count; ++i) {
            const TtnnPlaneLayout native(planes[i]);
            plane_bytes[i] =
                    native.padded_rows * native.padded_columns * carrier_bytes;
            staging_bytes += plane_bytes[i];
        }
        std::unique_ptr<std::byte[]> staging =
                std::make_unique_for_overwrite<std::byte[]>(staging_bytes);
        std::size_t offset = 0;
        for (std::size_t i = 0; i < planes_count; ++i) {
            ttnn::copy_to_host(
                    queue, planes[i], staging.get() + offset, std::nullopt,
                    /*blocking=*/false);
            offset += plane_bytes[i];
        }
        queue.finish();

        native_padding_zero_ = true;
        offset = 0;
        for (std::size_t i = 0; i < planes_count; ++i) {
            native_padding_zero_ =
                    observe_plane_storage(
                            std::span<const std::byte>(
                                    staging.get() + offset, plane_bytes[i]),
                            planes[i], owner, i, carrier_bytes, storage);
            offset += plane_bytes[i];
        }
        return storage;
    }

    [[nodiscard]] bool native_padding_zero() const {
        return native_padding_zero_;
    }

private:
    mutable bool native_padding_zero_ = true;
};


}  // namespace

TEST_CASE("TTNN download retirement preserves slot ownership") {
    using DownloadLease = iom::ttnn_detail::TtnnHostStaging::DownloadLease;
    static_assert(noexcept(std::declval<DownloadLease&>().retire()));
    static_assert(noexcept(std::declval<DownloadLease&>().~DownloadLease()));

    iom::ttnn_detail::TtnnHostStaging staging;
    auto lease = staging.acquire_download(8);
    std::byte* const retained = lease.data();
    const std::size_t allocations =
            iom::ttnn_test::host_transfer_staging_allocation_count_for_testing();
    lease.retire();

    CHECK(staging.download_retired());
    CHECK_EQ(
            iom::ttnn_test::host_transfer_staging_allocation_count_for_testing(),
            allocations);
    CHECK_THROWS_AS(
            staging.acquire_download(8), std::logic_error);

    staging.reclaim_download();
    auto reclaimed = staging.acquire_download(8);
    CHECK_EQ(reclaimed.data(), retained);
    reclaimed.release();
}

// The supported-type table is explicit and contains BOOL plus every required
// numeric NONE leaf. Every other declared leaf type and every grouped
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
TEST_CASE("Device::supported_data_types returns the canonical TTNN span") {
    require_hardware();
    const std::unique_ptr<iom::Device> candidate =
            iom::make_ttnn_device(0);
    const std::span<const iom::DataType> supported =
            candidate->supported_data_types();
    const std::span<const iom::DataType> canonical =
            iom::ttnn_supported_data_types();
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
            iom::DataType::F16, iom::DataType::BF16,
            iom::DataType::F32, iom::DataType::F64,
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
// Decisive padded-storage probe: the full-storage oracle observes the
// complete native padded allocation, so nonzero native-only writes fail it
// while the logical host projection stays byte-identical, and a logical host
// write re-establishes the documented zero policy.
TEST_CASE("TTNN conformance: full-storage oracle exposes native padding mutations") {
    require_hardware();
    TtnnDevices devices;
    TtnnStorageOracle oracle;
    const std::vector<std::pair<iom::TensorSpec, std::pair<std::size_t, std::size_t>>>
            cases = {
                    {iom::TensorSpec{
                             iom::TensorShape{{17, 33}}, iom::DataType::U8},
                     {0, 48}},
                    {iom::TensorSpec{
                             iom::TensorShape{{33, 17}}, iom::DataType::U8},
                     {48, 0}}};

    for (const auto& [spec, planted_cell] : cases) {
        CAPTURE(spec.shape.dimension(0));
        CAPTURE(spec.shape.dimension(1));
        CAPTURE(planted_cell.first);
        CAPTURE(planted_cell.second);
        auto tensor = devices.candidate->create_tensor(spec);
        oracle.set_owner_spec(spec);
        iom::TensorView& view = tensor->view();
        auto* planes = static_cast<ttnn::Tensor*>(view.native_handle());
        const TtnnPlaneLayout native(planes[0]);
        const iom::TensorShape standard_shape =
                spec.standard_padded_shape();
        const std::span<const std::size_t> standard =
                standard_shape.dimensions();
        CHECK_EQ(native.padded_rows, planted_cell.first == 48 ? 64 : 32);
        CHECK_EQ(native.padded_columns, planted_cell.second == 48 ? 64 : 32);
        CHECK_EQ(standard[standard.size() - 2],
                 planted_cell.first == 48 ? 48 : 32);
        CHECK_EQ(standard[standard.size() - 1],
                 planted_cell.second == 48 ? 48 : 32);
        const std::vector<std::byte> initial =
                iom_conformance::encode_standard_tiled_storage(spec);

        const std::pair<std::size_t, std::size_t> planted_cells[] = {
                planted_cell};
        std::vector<std::byte> mutated = padded_plane_image(
                planes[0], spec, 0, initial,
                std::span<const std::pair<std::size_t, std::size_t>>{
                        planted_cells});
        write_padded_plane_image(planes[0], mutated);

        const std::vector<std::byte> observed = oracle.observe(view);
        CHECK_FALSE(oracle.native_padding_zero());
        REQUIRE(iom_conformance::require_storage_oracle_bytes(
                observed, initial, "native-only padding mutation", true));
        iom_conformance::require_logical_bytes(
                view,
                iom_conformance::decode_standard_tiled_view(
                        view, spec, initial),
                "logical projection after native-only padding mutation");

        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(spec, 0x2A);
        iom_conformance::copy_from_host(view, pattern);
        std::vector<std::byte> expected = initial;
        iom_conformance::apply_standard_tiled_view(view, spec, pattern, expected);
        const std::vector<std::byte> restored = oracle.observe(view);
        REQUIRE(oracle.native_padding_zero());
        REQUIRE(iom_conformance::require_storage_oracle_bytes(
                restored, expected,
                "logical write re-establishes native zero padding", true));
        iom_conformance::require_logical_bytes(
                view, pattern, "logical write content");
    }
}

TEST_CASE("TTNN padded logical copy preserves destination padding") {
    require_hardware();
    TtnnDevices devices;
    const std::span<const iom::DataType> supported =
            iom::ttnn_supported_data_types();
    const iom::TensorShape geometries[] = {
            iom::TensorShape{{1, 1, 32, 48}},
            iom::TensorShape{{2, 3, 17, 33}}};
    std::uint64_t salt = 0x3500;

    for (const iom::DataType type : supported) {
        for (const iom::TensorShape& shape : geometries) {
            const iom::TensorSpec spec{shape, type};
            auto source = devices.candidate->create_tensor(spec);
            auto destination = devices.candidate->create_tensor(spec);
            iom::TensorView& source_owner = source->view();
            iom::TensorView& destination_owner = destination->view();
            auto* source_planes =
                    static_cast<ttnn::Tensor*>(source_owner.native_handle());
            auto* destination_planes = static_cast<ttnn::Tensor*>(
                    destination_owner.native_handle());
            const std::size_t plane_count = owner_plane_count(spec);
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            const std::size_t rows = dimensions[dimensions.size() - 2];
            const std::size_t columns = dimensions[dimensions.size() - 1];
            const std::size_t factor =
                    iom::ttnn_detail::carrier_factor(type);
            const std::size_t carrier_size =
                    iom::ttnn_detail::carrier_bytes(source_planes[0].dtype());
            REQUIRE(columns <= std::numeric_limits<std::size_t>::max() / factor);
            const std::size_t native_columns = columns * factor;

            std::vector<std::vector<std::byte>> source_images(plane_count);
            std::vector<std::vector<std::byte>> destination_images(plane_count);
            std::vector<std::byte> source_storage =
                    iom_conformance::encode_standard_tiled_storage(spec);
            std::vector<std::byte> destination_storage =
                    iom_conformance::encode_standard_tiled_storage(spec);
            const std::vector<std::byte> source_logical =
                    iom_conformance::encode_logical(spec, salt);
            const std::vector<std::byte> destination_logical =
                    iom_conformance::encode_logical(spec, salt + 1);
            ++salt;
            iom_conformance::apply_standard_tiled_view(
                    source_owner, spec, source_logical, source_storage);
            iom_conformance::apply_standard_tiled_view(
                    destination_owner, spec, destination_logical,
                    destination_storage);

            bool has_physical_padding = false;
            for (std::size_t plane_index = 0; plane_index < plane_count;
                 ++plane_index) {
                const TtnnPlaneLayout source_layout(source_planes[plane_index]);
                const TtnnPlaneLayout destination_layout(
                        destination_planes[plane_index]);
                source_images[plane_index] = padded_plane_image(
                        source_planes[plane_index], spec, plane_index,
                        source_storage);
                destination_images[plane_index] = padded_plane_image(
                        destination_planes[plane_index], spec, plane_index,
                        destination_storage);
                const bool row_padding = rows < source_layout.padded_rows;
                const bool column_padding =
                        native_columns < source_layout.padded_columns;
                has_physical_padding =
                        has_physical_padding || row_padding || column_padding;
                if (!row_padding && !column_padding) {
                    continue;
                }
                const std::size_t sentinel_row =
                        row_padding ? rows : 0;
                const std::size_t sentinel_column =
                        column_padding ? native_columns : 0;
                const bool sentinel_is_padding =
                        sentinel_row >= rows
                        || sentinel_column >= native_columns;
                REQUIRE(sentinel_is_padding);
                const auto plant = [&](std::vector<std::byte>& image,
                                       const TtnnPlaneLayout& layout,
                                       std::byte value) {
                    std::byte* cell =
                            image.data()
                            + (sentinel_row * layout.padded_columns
                               + sentinel_column)
                                    * carrier_size;
                    std::fill(cell, cell + carrier_size, value);
                };
                plant(source_images[plane_index], source_layout,
                      std::byte{0x35});
                plant(destination_images[plane_index], destination_layout,
                      std::byte{0xC7});
                write_padded_plane_image(
                        source_planes[plane_index],
                        source_images[plane_index]);
                write_padded_plane_image(
                        destination_planes[plane_index],
                        destination_images[plane_index]);
            }
            if (!has_physical_padding) {
                continue;
            }
            source_planes[0].device()->mesh_command_queue(0).finish();

            const std::size_t slice_start = dimensions[0] > 1 ? 1 : 0;
            iom::TensorView source_view =
                    source_owner.slice(0, slice_start, 1);
            iom::TensorView destination_view =
                    destination_owner.slice(0, slice_start, 1);
            auto queue = devices.candidate->create_ops();
            const iom::oid token =
                    queue->copy(source_view, destination_view);
            REQUIRE(iom::oid_is_token(token));
            REQUIRE_NOTHROW(queue->wait(token));

            for (std::size_t plane_index = 0; plane_index < plane_count;
                 ++plane_index) {
                std::vector<std::byte> expected =
                        destination_images[plane_index];
                for (std::size_t view_plane = 0;
                     view_plane
                             < iom::ttnn_detail::snapshot_plane_count(
                                     source_view.spec());
                     ++view_plane) {
                    const std::size_t source_plane_index =
                            oracle_view_plane_at(source_view, view_plane);
                    const std::size_t destination_plane_index =
                            oracle_view_plane_at(
                                    destination_view, view_plane);
                    if (destination_plane_index != plane_index) continue;
                    const TtnnPlaneLayout source_layout(
                            source_planes[source_plane_index]);
                    const TtnnPlaneLayout destination_layout(
                            destination_planes[destination_plane_index]);
                    for (std::size_t row = 0; row < rows; ++row) {
                        for (std::size_t column = 0; column < columns;
                             ++column) {
                            for (std::size_t part = 0; part < factor;
                                 ++part) {
                                const std::size_t native_column =
                                        column * factor + part;
                                std::memcpy(
                                        expected.data()
                                                + (row
                                                           * destination_layout
                                                                   .padded_columns
                                                   + native_column)
                                                        * carrier_size,
                                        source_images[source_plane_index].data()
                                                + (row
                                                           * source_layout
                                                                   .padded_columns
                                                   + native_column)
                                                        * carrier_size,
                                        carrier_size);
                            }
                        }
                    }
                }
                const std::vector<std::byte> observed =
                        read_padded_plane_image(
                                destination_planes[plane_index]);
                CHECK_EQ(observed, expected);
            }
        }
    }
}

TEST_CASE("TTNN conformance: asynchronous copies against the CPU reference") {
    require_hardware();
    TtnnDevices devices;
    TtnnStorageOracle oracle;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), iom::ttnn_supported_data_types(), nullptr,
            &oracle);
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

// The retained host-transfer staging facility allocates one byte slot per
// native upload dtype per plane plus one byte buffer per download. A
// six-plane region therefore warms exactly seven retained buffers; every
// later same-or-smaller transfer reuses them, and a larger shape grows
// every retained buffer once and then reuses it.
TEST_CASE("TTNN host transfers reuse retained staging after warm-up") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::BF16};
    auto tensor = devices.candidate->create_tensor(spec);

    // Warm-up: the first upload allocates the six BF16 plane slots and the
    // first download the byte staging buffer exactly once.
    const std::vector<std::byte> seed =
            iom_conformance::encode_logical(spec, 0x51);
    iom_conformance::copy_from_host(tensor->view(), seed);
    iom_conformance::require_logical_bytes(
            tensor->view(), seed, "warm-up readback");
    const std::size_t warmup_allocations =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();

    // Repeated same-or-smaller transfers allocate no fresh host staging.
    for (std::uint64_t salt = 0x60; salt < 0x64; ++salt) {
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(spec, salt);
        iom_conformance::copy_from_host(tensor->view(), pattern);
        iom_conformance::require_logical_bytes(
                tensor->view(), pattern, "reused staging round trip");
        CHECK_EQ(
                iom::ttnn_test::
                        host_transfer_staging_allocation_count_for_testing(),
                warmup_allocations);
    }

    // A larger padded shape grows every retained buffer once (six upload
    // slots plus the download buffer), then reuses it.
    const iom::TensorSpec larger{
            iom::TensorShape{{2, 3, 65, 33}}, iom::DataType::BF16};
    auto big = devices.candidate->create_tensor(larger);
    const std::vector<std::byte> large_seed =
            iom_conformance::encode_logical(larger, 0x71);
    iom_conformance::copy_from_host(big->view(), large_seed);
    iom_conformance::require_logical_bytes(
            big->view(), large_seed, "grown readback");
    CHECK_EQ(
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing(),
            warmup_allocations + 7);
    for (std::uint64_t salt = 0x72; salt < 0x75; ++salt) {
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(larger, salt);
        iom_conformance::copy_from_host(big->view(), pattern);
        iom_conformance::require_logical_bytes(
                big->view(), pattern, "grown staging round trip");
        CHECK_EQ(
                iom::ttnn_test::
                        host_transfer_staging_allocation_count_for_testing(),
                warmup_allocations + 7);
    }
}

// A failed host transfer discards the staging it held, never reuses it, and
// leaves the facility able to serve the next transfer from fresh, clean
// storage; the download failure additionally preserves the drain ordering.
TEST_CASE("TTNN host-transfer failures discard poisoned staging") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::U8};
    auto tensor = devices.candidate->create_tensor(spec);

    // Warm the facility so the faults below hit submitted staging, not an
    // allocation.
    const std::vector<std::byte> seed =
            iom_conformance::encode_logical(spec, 0x81);
    iom_conformance::copy_from_host(tensor->view(), seed);
    iom_conformance::require_logical_bytes(
            tensor->view(), seed, "warm-up");

    // An upload submission fault at plane 1: plane 0 reached the mesh, the
    // bounded recovery finish succeeds, and all leases become reusable only
    // after that completion proof. The original fault still propagates.
    iom::ttnn_test::fail_next_host_transfer_submission_for_testing(1);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0x82);
    REQUIRE_THROWS_AS(
            iom_conformance::copy_from_host(tensor->view(), pattern), std::runtime_error);
    CHECK(iom::ttnn_test::
                  host_transfer_submission_fault_consumed_for_testing());
    const std::size_t after_upload_failure =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();

    const std::vector<std::byte> pattern_2 =
            iom_conformance::encode_logical(spec, 0x83);
    iom_conformance::copy_from_host(tensor->view(), pattern_2);
    iom_conformance::require_logical_bytes(
            tensor->view(), pattern_2, "clean upload after failure");
    CHECK_EQ(
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing(),
            after_upload_failure);

    // A download submission fault at plane 1: plane 0 was enqueued, the
    // failure drain finishes it, and the byte staging buffer is discarded
    // instead of being reused; the next download allocates fresh storage
    // and returns the exact logical bytes.
    iom::ttnn_test::fail_next_host_transfer_submission_for_testing(1);
    std::vector<std::byte> readback(
            spec.logical_nbytes(), iom_conformance::kReadbackSentinel);
    REQUIRE_THROWS_AS(
            iom_conformance::copy_to_host(tensor->view(), readback), std::runtime_error);
    CHECK(iom::ttnn_test::
                  host_transfer_submission_fault_consumed_for_testing());
    const std::size_t after_download_failure =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();
    iom_conformance::require_logical_bytes(
            tensor->view(), pattern_2, "clean download after failure");
    CHECK_EQ(
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing(),
            after_download_failure + 1);

    // A staging-allocation failure leaves the retained slot untouched and
    // the facility able to serve the next (larger) transfer.
    const iom::TensorSpec larger{
            iom::TensorShape{{2, 3, 65, 33}}, iom::DataType::U8};
    auto big = devices.candidate->create_tensor(larger);
    const std::vector<std::byte> large_seed =
            iom_conformance::encode_logical(larger, 0x84);
    iom::ttnn_test::
            fail_next_host_transfer_staging_allocation_for_testing();
    REQUIRE_THROWS_AS(
            iom_conformance::copy_from_host(big->view(), large_seed), std::bad_alloc);
    CHECK(
            iom::ttnn_test::
                    host_transfer_staging_allocation_fault_consumed_for_testing());
    const std::size_t after_allocation_failure =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();

    iom_conformance::copy_from_host(big->view(), large_seed);
    iom_conformance::require_logical_bytes(
            big->view(), large_seed, "clean growth after allocation failure");
    CHECK_EQ(
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing(),
            after_allocation_failure + 7);
}

TEST_CASE("TTNN conformance: deferred queue lifetime and stability") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, iom::ttnn_supported_data_types());
}

TEST_CASE("TTNN conformance: compute capabilities") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, iom::ttnn_supported_data_types(), nullptr,
            "TTNN", true, false);
}
TEST_CASE("TTNN RMSNorm BF16 and F32 planes preserve output identity") {
    require_hardware();
    TtnnDevices devices;

    const auto run_identity = [&](iom::DataType type) {
        const iom::TensorSpec input_spec{
                iom::TensorShape{{2, 2, 17}}, type};
        const iom::TensorSpec scale_spec{
                iom::TensorShape{{1, 17}}, type};
        auto input = devices.candidate->create_tensor(input_spec);
        auto scale = devices.candidate->create_tensor(scale_spec);
        auto output = devices.candidate->create_tensor(input_spec);

        const std::size_t element_bytes = type == iom::DataType::BF16 ? 2 : 4;
        const std::size_t input_elements = 2 * 2 * 17;
        const std::size_t scale_elements = 17;
        std::vector<std::byte> input_bytes(input_spec.logical_nbytes());
        std::vector<std::byte> scale_bytes(scale_spec.logical_nbytes());
        const std::uint16_t bf16_one = 0x3F80;
        const std::uint32_t f32_one = 0x3F800000;
        for (std::size_t index = 0; index < input_elements; ++index) {
            const void* value = type == iom::DataType::BF16
                    ? static_cast<const void*>(&bf16_one)
                    : static_cast<const void*>(&f32_one);
            std::memcpy(
                    input_bytes.data() + index * element_bytes, value,
                    element_bytes);
        }
        for (std::size_t index = 0; index < scale_elements; ++index) {
            const void* value = type == iom::DataType::BF16
                    ? static_cast<const void*>(&bf16_one)
                    : static_cast<const void*>(&f32_one);
            std::memcpy(
                    scale_bytes.data() + index * element_bytes, value,
                    element_bytes);
        }
        iom_conformance::copy_from_host(input->view(), input_bytes);
        iom_conformance::copy_from_host(scale->view(), scale_bytes);
        std::vector<std::byte> zero_output(input_spec.logical_nbytes());
        iom_conformance::copy_from_host(output->view(), zero_output);

        iom::TensorView input_plane = input->view().slice(0, 1, 1);
        iom::TensorView output_plane = output->view().slice(0, 1, 1);
        void* const output_handle = output_plane.native_handle();
        auto queue = devices.candidate->create_ops();
        const iom::oid token = queue->rmsnorm(
                input_plane, scale->view(), output_plane, 0.0F);
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));
        REQUIRE_NOTHROW(queue->wait(token));
        CHECK_EQ(output_plane.native_handle(), output_handle);

        std::vector<std::byte> observed(output_plane.spec().logical_nbytes());
        iom_conformance::copy_to_host(output_plane, observed);
        std::vector<std::byte> expected(observed.size(), std::byte{0});
        for (std::size_t index = 0; index < observed.size();
             index += element_bytes) {
            const std::uint32_t value =
                    type == iom::DataType::BF16 ? bf16_one : f32_one;
            std::memcpy(expected.data() + index, &value, element_bytes);
        }
        CHECK(observed == expected);
    };

    run_identity(iom::DataType::BF16);
    run_identity(iom::DataType::F32);

    {
        const iom::TensorSpec input_spec{
                iom::TensorShape{{1, 1, 17}}, iom::DataType::F32};
        const iom::TensorSpec scale_owner_spec{
                iom::TensorShape{{2, 1, 17}}, iom::DataType::F32};
        auto input = devices.candidate->create_tensor(input_spec);
        auto scale_owner =
                devices.candidate->create_tensor(scale_owner_spec);
        auto output = devices.candidate->create_tensor(input_spec);

        std::vector<float> input_values(17, 1.0F);
        std::vector<float> scale_values(2 * 17, 2.0F);
        std::fill(scale_values.begin() + 17, scale_values.end(), 3.0F);
        std::vector<std::byte> input_bytes(input_spec.logical_nbytes());
        std::vector<std::byte> scale_bytes(
                scale_owner_spec.logical_nbytes());
        std::memcpy(input_bytes.data(), input_values.data(),
                    input_bytes.size());
        std::memcpy(scale_bytes.data(), scale_values.data(),
                    scale_bytes.size());
        iom_conformance::copy_from_host(input->view(), input_bytes);
        iom_conformance::copy_from_host(scale_owner->view(), scale_bytes);
        std::vector<std::byte> zero_output(input_spec.logical_nbytes());
        iom_conformance::copy_from_host(output->view(), zero_output);

        const iom::TensorView transformed_scale =
                scale_owner->view().select(0, 1);
        CHECK_EQ(transformed_scale.plane_offset(), std::size_t{1});
        auto queue = devices.candidate->create_ops();
        const iom::oid token = queue->rmsnorm(
                input->view(), transformed_scale, output->view(), 0.0F);
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));

        std::vector<std::byte> observed(input_spec.logical_nbytes());
        iom_conformance::copy_to_host(output->view(), observed);
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            float observed_value = 0.0F;
            std::memcpy(&observed_value, observed.data() +
                                             index * sizeof(float),
                        sizeof(float));
            CHECK_EQ(observed_value, 3.0F);
        }
    }

    // TTNN advertises only BF16/F32 for RMSNorm. Its six storable encoded
    // carriers are explicit queue-level capability limitations. F8_E8M0 is
    // rejected at tensor creation because TTNN has no storage carrier for it;
    // that storage-level limitation is covered by the supported-type test.
    const iom::DataType unsupported[] = {
            iom::DataType::BOOL, iom::DataType::I2, iom::DataType::U2,
            iom::DataType::I4, iom::DataType::U4, iom::DataType::I8,
            iom::DataType::U8, iom::DataType::I16, iom::DataType::U16,
            iom::DataType::I32, iom::DataType::U32, iom::DataType::I64,
            iom::DataType::U64, iom::DataType::F4_E2M1,
            iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
            iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
            iom::DataType::F16, iom::DataType::F64};
    for (const iom::DataType type : unsupported) {
        const iom::TensorSpec spec{
                iom::TensorShape{{2, 16, 16}}, type};
        const iom::TensorSpec scale_spec{
                iom::TensorShape{{1, 16}}, type};
        auto input = devices.candidate->create_tensor(spec);
        auto scale = devices.candidate->create_tensor(scale_spec);
        auto output = devices.candidate->create_tensor(spec);
        auto queue = devices.candidate->create_ops();
        CHECK_EQ(
                queue->rmsnorm(
                        input->view(), scale->view(), output->view(), 1e-6F),
                iom::to_oid(iom::OidError::Unsupported));
    }
}

// TTNN's native embedding path implements the final matrix: the 22 payload
// leaves its carrier table can store and all 12 integral index leaves, with
// `I64`/`U64`/`F64` payloads and `I64`/`U64` indices carried as two consecutive
// native UInt32 columns. The native path reports the exact `{32, 32}`
// status-workspace contract, whose status subrange alone is read back. Only
// `F8_E8M0` stays outside embedding support, and the device cannot store it.
constexpr iom_conformance::EmbeddingDeclaration kTtnnEmbeddingDeclaration{
        iom_conformance::kEmbeddingTtnnPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom_conformance::kEmbeddingTtnnPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom::WorkspaceRequirements{32, 32}};

TEST_CASE("TTNN conformance: embedding lookup reference, admission, and lifetime") {
    require_hardware();
    TtnnDevices devices;
    TtnnStorageOracle oracle;
    iom_conformance::run_embedding_conformance(
            devices.conformance(), kTtnnEmbeddingDeclaration, nullptr,
            &oracle);
}

// The double-carrier path on the real device: a 64-bit payload and a 64-bit
// index keep their low/high native column order across the 16-column face and
// 32-column tile boundaries of the doubled column space, an accepted
// high-word-only ID fails only after native proof, and the same status cell
// and workspace range stay reusable once that completion is proven.
TEST_CASE("TTNN embedding double carriers keep words and reuse proven scratch") {
    require_hardware();
    TtnnDevices devices;
    constexpr std::size_t vocabulary = 17;
    constexpr std::size_t features = 33;
    constexpr std::size_t run = 17;
    const iom::TensorSpec table_spec{
            iom::TensorShape{{vocabulary, features}}, iom::DataType::F64};
    const iom::TensorSpec index_spec{
            iom::TensorShape{{1, run}}, iom::DataType::I64};
    const iom::TensorSpec output_spec{
            iom::TensorShape{{run, features}}, iom::DataType::F64};
    auto table = devices.candidate->create_tensor(table_spec);
    auto indices = devices.candidate->create_tensor(index_spec);
    auto output = devices.candidate->create_tensor(output_spec);
    auto second_output = devices.candidate->create_tensor(output_spec);
    auto workspace = devices.candidate->create_workspace(32);
    auto queue = devices.candidate->create_ops();

    // Every element sets both carrier words, crosses the `2^53` integer
    // boundary, and carries a high-word sign bit, so a dropped or swapped word
    // and a numeric decode are all observable.
    const auto table_code = [](std::size_t linear) {
        return (static_cast<std::uint64_t>(linear) << 40)
                ^ 0x8000000000000001ull
                ^ (static_cast<std::uint64_t>(linear) * 0x0000000100000003ull);
    };
    const std::vector<std::byte> table_bytes =
            iom_conformance::encode_logical_codes(table_spec, table_code);
    std::vector<std::uint64_t> ids(run, 0);
    for (std::size_t row = 0; row < run; ++row) {
        ids[row] = (row * 5 + 1) % vocabulary;
    }
    const std::vector<std::byte> index_bytes =
            iom_conformance::encode_logical_codes(
                    index_spec,
                    [&ids](std::size_t linear) { return ids[linear % run]; });
    // Every code has a valid small low word and a nonzero high word, so a
    // narrowing implementation would gather row `2` instead of failing.
    const std::vector<std::byte> high_word_only_bytes =
            iom_conformance::encode_logical_codes(
                    index_spec,
                    [](std::size_t) { return 0x0000000100000002ull; });
    iom_conformance::copy_from_host(table->view(), table_bytes);
    iom_conformance::copy_from_host(indices->view(), index_bytes);
    const std::vector<std::byte> poison(
            output_spec.logical_nbytes(), std::byte{0x5A});
    iom_conformance::copy_from_host(output->view(), poison);
    iom_conformance::copy_from_host(second_output->view(), poison);
    const std::vector<std::byte> expected =
            iom_conformance::select_embedding_rows(
                    table_bytes, iom::DataType::F64, vocabulary, features, run,
                    std::span<const std::uint64_t>{ids});

    const iom::oid valid = queue->embedding(
            table->view(), indices->view(), output->view(),
            workspace->view());
    REQUIRE(iom::oid_is_token(valid));
    REQUIRE_NOTHROW(queue->wait(valid));
    iom_conformance::require_logical_bytes(
            output->view(), expected,
            "wide payload and wide index gather");
    iom_conformance::require_logical_bytes(
            table->view(), table_bytes, "wide gather changed the table");

    // An accepted high-word-only ID is a data failure: the low word alone is a
    // valid ID, so only the high-word check rejects it, and only after native
    // proof.
    iom_conformance::copy_from_host(indices->view(), high_word_only_bytes);
    const iom::oid rejected = queue->embedding(
            table->view(), indices->view(), second_output->view(),
            workspace->view());
    REQUIRE(iom::oid_is_token(rejected));
    iom_conformance::expect_repeated_invalid_argument(*queue, rejected);

    // Proven completion released the status cell and the range: the same
    // workspace and owners serve an independent wide gather again.
    iom_conformance::copy_from_host(indices->view(), index_bytes);
    const iom::oid reused = queue->embedding(
            table->view(), indices->view(), second_output->view(),
            workspace->view());
    REQUIRE(iom::oid_is_token(reused));
    REQUIRE_NOTHROW(queue->wait(reused));
    iom_conformance::require_logical_bytes(
            second_output->view(), expected,
            "wide gather after proven status and workspace reuse");
}

// TTNN's declared linear expectation: the mandatory BF16 leaf alone through the
// native path, the frozen `{0, 1}` direct-reader/direct-writer scratch path, and
// an explicit rejection for the other twenty applicable leaves, whose encoded
// carriers its native TILE compute cannot consume without the forbidden host
// staging. The direct Metalium route is implemented, so the BF16 leaf is the
// declared implemented span and the twenty other applicable leaves stay
// capability rejections. Two measured native-facility limitations are recorded
// instead of being asserted, exactly as the Linear projections section's
// `Observed TTNN nonfinite limitation`, `Facility class expressibility`, and
// `Facility subnormal handling` notes state: (1) the facility cannot express
// the contract's nonfinite result classes — `inf*inf`, `max_finite*inf`,
// `inf*max_finite`, and `1.7e38*1.7e38` return `+0` instead of an infinity,
// `inf*0`, `0*inf`, and `inf + (-inf)` return `+0` instead of `NaN`, every NaN
// operand returns `+inf`, and no tested operand encoding produces a NaN output
// at all; and (2) it flushes subnormal operands before the multiply — `1.0`
// times the largest BF16 subnormal (`0x007f`, `1.16631e-38`) returns `+0`
// where the FP32 rule gives `1.16631e-38`, and `max_finite` times it returns
// `+0` where the rule gives `3.95325`, while `0x0080` (`1.17549e-38`) and
// `0x0100` both agree. Every element those two limitations do not affect keeps
// the unchanged contract tolerances.
const iom_conformance::LinearDeclaration kTtnnLinearDeclaration{
        iom_conformance::kLinearTtnnLeafSpan,
        iom_conformance::kNoLinearSpan,
        iom_conformance::kLinearNativeBf16Span,
        iom_conformance::kLinearNativeBf16Span,
        {},
        iom_conformance::LinearWorkspacePath::zero,
        iom_conformance::LinearWorkspacePath::zero,
        false,
        false};

TEST_CASE("TTNN conformance: linear projection reference, admission, and lifetime") {
    require_hardware();
    TtnnDevices devices;
    TtnnStorageOracle oracle;
    iom_conformance::LinearComparisonRecord skips;
    iom_conformance::run_linear_conformance(
            devices.conformance(), kTtnnLinearDeclaration, nullptr, &oracle,
            &skips);
    // Both declared facility limitations must be exercised by the shared BF16
    // fixture rather than being dead declarations: the canonical fixture
    // carries nonfinite ladder operands and subnormal operands, so a run that
    // never observes one of them means the declaration no longer matches what
    // this port is measured to do.
    CHECK(skips.nonfinite_elements > 0);
    CHECK(skips.subnormal_elements > 0);
}

TEST_CASE("TTNN embedding owners survive accepted dispatch before wait") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec table_spec{
            iom::TensorShape{{1, 8}}, iom::DataType::U32};
    const iom::TensorSpec index_spec{
            iom::TensorShape{{1, 8}}, iom::DataType::U32};
    const iom::TensorSpec output_spec{
            iom::TensorShape{{8, 8}}, iom::DataType::U32};
    auto queue = devices.candidate->create_ops();

    for (int doomed = 0; doomed != 4; ++doomed) {
        auto table = devices.candidate->create_tensor(table_spec);
        auto indices = devices.candidate->create_tensor(index_spec);
        auto output = devices.candidate->create_tensor(output_spec);
        auto workspace = devices.candidate->create_workspace(32);
        const std::vector<std::byte> table_bytes(
                table_spec.logical_nbytes(), std::byte{7});
        const std::vector<std::byte> index_bytes(
                index_spec.logical_nbytes(), std::byte{0});
        const std::vector<std::byte> output_bytes(
                output_spec.logical_nbytes(), std::byte{0xA5});
        iom_conformance::copy_from_host(table->view(), table_bytes);
        iom_conformance::copy_from_host(indices->view(), index_bytes);
        iom_conformance::copy_from_host(output->view(), output_bytes);

        const iom::oid token = queue->embedding(
                table->view(), indices->view(), output->view(),
                workspace->view());
        REQUIRE(iom::oid_is_token(token));
        switch (doomed) {
            case 0: table.reset(); break;
            case 1: indices.reset(); break;
            case 2: output.reset(); break;
            case 3: workspace.reset(); break;
        }
        CHECK_NOTHROW(queue->wait(token));
        if (output != nullptr) {
            const std::vector<std::byte> expected(
                    output_spec.logical_nbytes(), std::byte{7});
            iom_conformance::require_logical_bytes(
                    output->view(), expected,
                    "embedding output after owner destruction");
        }
    }
}

// TTNN's native RMSNorm path is documented-nonconforming and is measured under
// its own published expectation policy: docs/BACKEND_CONTRACT.md's **RMS
// normalization** section records that the preallocated `ttnn::prim::LayerNorm`
// -backed adapter evaluates a row at BF16 compute fidelity and cannot produce
// the contract's special value classes. The declared deviations are named here
// — reduced accumulator fidelity and absent special values — and every element
// they observe is counted and printed; the frozen comparison policy still
// applies to the finite BF16 leaves, to every admission, workspace, alias,
// overflow, ownership, ordering, and retained-failure case, and to all four
// other backends, none of which declares any deviation.
TEST_CASE("TTNN conformance: RMSNorm documented BF16-fidelity policy") {
    require_hardware();
    TtnnDevices devices;
    TtnnStorageOracle oracle;
    iom_conformance::RmsNormComparisonRecord record;
    iom_conformance::RmsNormConformanceConfig config{
            devices.conformance(),
            iom_conformance::kRmsNormWideLeafSpan,
            nullptr,
            &oracle,
            iom_conformance::RmsNormComparisonMode{false, false},
            &record};
    // TTNN exposes no testing seam that reaches its RMSNorm path: every
    // `iom::ttnn_test` fault is copy-, binary-, or host-transfer-specific
    // (`fail_next_copy_finishes_for_testing`,
    // `fail_next_binary_outcome_insertion_for_testing`,
    // `fail_next_host_transfer_submission_for_testing`, ...) while
    // `TtnnQueue::execute_rmsnorm` carries no fault point, so no existing seam
    // can produce an accepted RMSNorm failure. The harness records the gap
    // explicitly. The minimal test-only seam that would close it is a
    // rmsnorm-scoped finish/outcome fault consumed in the rmsnorm outcome or
    // completion path under the existing `IOM_ENABLE_TESTING` guard.
    config.native_failure = iom_conformance::RmsNormNativeFailureSeam{
            {},
            {},
            "iom::ttnn_test",
            "no existing TTNN testing seam reaches execute_rmsnorm; the seams "
            "are copy/binary/host-transfer specific"};
    iom_conformance::run_rmsnorm_conformance(config);
    std::printf(
            "ttnn-rmsnorm-fidelity-record policy=documented-bf16-native "
            "fp32_finite=reduced_fidelity special_values=absent "
            "observed_precision_elements=%zu "
            "observed_special_value_elements=%zu\n",
            record.observed_precision_elements,
            record.observed_special_value_elements);
    CHECK_GT(record.observed_precision_elements, std::size_t{0});
    CHECK_GT(record.observed_special_value_elements, std::size_t{0});
}

TEST_CASE("TTNN conformance: full shared suite") {
    require_hardware();
    TtnnDevices devices;
    const std::span<const iom::DataType> supported =
            iom::ttnn_supported_data_types();
    TtnnStorageOracle oracle;
    iom_conformance::run_backend_conformance(
            devices.conformance(), supported.subspan(0, 1), nullptr, &oracle,
            true, false);
}

TEST_CASE("TTNN conformance: binary MUL SUB and floating DIV values through real queue") {
    require_hardware();
    TtnnDevices devices;
    for (const auto operation : {
                 iom_conformance::BinaryOperation::add,
                 iom_conformance::BinaryOperation::mul,
                 iom_conformance::BinaryOperation::sub,
                 iom_conformance::BinaryOperation::div}) {
        iom_conformance::run_binary_value_conformance(
                *devices.candidate, operation);
    }
}

TEST_CASE("TTNN quarantine action allocation failure leaks native storage") {
    require_hardware();
    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::BF16};
    auto device = iom::make_ttnn_device(0);
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    auto queue = device->create_ops();
    REQUIRE(iom::oid_is_token(queue->copy(source->view(), destination->view())));

    {
        iom::ttnn_test::fail_next_copy_finishes_for_testing(1);
        iom::ttnn_test::fail_next_quarantine_action_for_testing();
        queue.reset();
        CHECK_NOTHROW(destination.reset());
        CHECK(
                iom::ttnn_test::quarantine_action_fault_consumed_for_testing());
    }

    {
        const bool consumed =
                iom::ttnn_test::quarantine_action_fault_consumed_for_testing();
        CHECK_NOTHROW(source.reset());
        CHECK_EQ(
                iom::ttnn_test::quarantine_action_fault_consumed_for_testing(),
                consumed);
    }

    {
        // The healthy follow-up is the sole recorded quarantine action; its
        // device-teardown drain must complete without throwing.
        CHECK_NOTHROW(device.reset());
    }
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

TEST_CASE("TTNN copy survives derived-view temporaries") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{4, 3, 17, 33}}, iom::DataType::U8};
    std::unique_ptr<iom::Tensor> t = devices.candidate->create_tensor(spec);
    std::unique_ptr<iom::Tensor> u = devices.candidate->create_tensor(spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0x5A7C);
    iom_conformance::copy_from_host(t->view(), pattern);

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
}

namespace {
// Runs one healthy multi-plane copy through a fresh queue on the device,
// asserting neither the wait nor the readback observes anything left over
// from a previously failed operation.
void require_healthy_copy_after_failure(
        iom::Device& device, const iom::TensorSpec& spec,
        std::uint64_t salt, std::string_view context) {
    auto fresh_source = device.create_tensor(spec);
    auto fresh_destination = device.create_tensor(spec);
    auto fresh_queue = device.create_ops();
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, salt);
    iom_conformance::copy_from_host(fresh_source->view(), pattern);
    const iom::oid token =
            fresh_queue->copy(fresh_source->view(), fresh_destination->view());
    REQUIRE_NOTHROW(fresh_queue->wait(token));
    iom_conformance::require_logical_bytes(
            fresh_destination->view(), pattern, context);
}
}  // namespace

TEST_CASE("TTNN failed plane submissions drain before rethrow") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::F32};
    const std::size_t plane_count = 6;
    for (const std::size_t fail_plane :
         {std::size_t{0}, std::size_t{1}, plane_count - 1}) {
        CAPTURE(fail_plane);
        auto source = devices.candidate->create_tensor(spec);
        auto destination = devices.candidate->create_tensor(spec);
        iom_conformance::copy_from_host(source->view(), iom_conformance::encode_logical(spec, 21));

        auto queue = devices.candidate->create_ops();
        iom::ttnn_test::fail_next_copy_planes_submission_for_testing(
                fail_plane);
        CHECK_EQ(
                queue->copy(source->view(), destination->view()),
                iom::to_oid(iom::OidError::DeviceError));
        CHECK(iom::ttnn_test::
                      copy_planes_submission_fault_consumed_for_testing());

        // The owners can be destroyed and reused immediately: the thrown
        // call drained every submitted plane (or submitted none) before
        // ownership was removed, and the same queue stays healthy.
        source.reset();
        destination.reset();
        source = devices.candidate->create_tensor(spec);
        destination = devices.candidate->create_tensor(spec);
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(spec, 22);
        iom_conformance::copy_from_host(source->view(), pattern);
        const iom::oid token =
                queue->copy(source->view(), destination->view());
        REQUIRE_NOTHROW(queue->wait(token));
        iom_conformance::require_logical_bytes(
                destination->view(), pattern,
                "copy after partial-plane failure");
    }
}

TEST_CASE("TTNN registration and outcome insertion failures roll back ownership") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::F32};

    {
        // The registration phase fails before any entry or native plane
        // exists; the sequence reservation rolls back and the queue serves
        // the next copy normally.
        auto source = devices.candidate->create_tensor(spec);
        auto destination = devices.candidate->create_tensor(spec);
        auto queue = devices.candidate->create_ops();
        iom::ttnn_test::fail_next_copy_registration_for_testing();
        CHECK_EQ(
                queue->copy(source->view(), destination->view()),
                iom::to_oid(iom::OidError::ResourceExhausted));
        CHECK(iom::ttnn_test::
                      copy_registration_fault_consumed_for_testing());

        source.reset();
        destination.reset();
        require_healthy_copy_after_failure(
                *devices.candidate, spec, 23,
                "copy after registration failure");
    }

    {
        // The outcome insertion fails after both entries were registered:
        // the transaction must roll the entries back before rethrowing, and
        // a repeated failure reuses the rolled-back sequence each time.
        auto source = devices.candidate->create_tensor(spec);
        auto destination = devices.candidate->create_tensor(spec);
        auto queue = devices.candidate->create_ops();
        for (int failure = 0; failure < 2; ++failure) {
            iom::ttnn_test::fail_next_copy_outcome_insertion_for_testing();
            CHECK_EQ(
                    queue->copy(source->view(), destination->view()),
                    iom::to_oid(iom::OidError::ResourceExhausted));
            CHECK(iom::ttnn_test::
                          copy_outcome_insertion_fault_consumed_for_testing());
        }

        source.reset();
        destination.reset();
        require_healthy_copy_after_failure(
                *devices.candidate, spec, 24,
                "copy after outcome-insertion failure");
    }
}

TEST_CASE("TTNN binary pre-native failures roll back or retain correctly") {
    require_hardware();
    TtnnDevices devices;
    const std::size_t output_rows[] = {33, 65, 129, 257};
    const std::size_t output_columns[] = {17, 33, 65, 129};

    for (std::size_t operation = 0; operation < 4; ++operation) {
        const iom::TensorSpec spec{
                iom::TensorShape{{1, output_rows[operation],
                                  output_columns[operation]}},
                iom::DataType::BF16};
        CAPTURE(operation);
        auto lhs = devices.candidate->create_tensor(spec);
        auto rhs = devices.candidate->create_tensor(spec);
        auto out = devices.candidate->create_tensor(spec);

        // Seed all three owners through the native path so the public host
        // staging pool has no capacity before the first output upload.
        auto* lhs_planes =
                static_cast<ttnn::Tensor*>(lhs->view().native_handle());
        auto* rhs_planes =
                static_cast<ttnn::Tensor*>(rhs->view().native_handle());
        auto* out_planes =
                static_cast<ttnn::Tensor*>(out->view().native_handle());
        const ttnn::Tensor& out_plane = out_planes[0];
        const std::size_t carrier_bytes =
                out_plane.dtype() == tt::tt_metal::DataType::UINT8
                ? 1
                : (out_plane.dtype() == tt::tt_metal::DataType::UINT16
                                   || out_plane.dtype()
                                              == tt::tt_metal::DataType::BFLOAT16
                           ? 2
                           : 4);
        const std::size_t native_bytes =
                static_cast<std::size_t>(out_plane.padded_shape()[-2])
                * static_cast<std::size_t>(out_plane.padded_shape()[-1])
                * carrier_bytes;
        std::vector<std::byte> native_input(native_bytes, std::byte{0});
        std::vector<std::byte> native_sentinel(
                native_bytes, kObserverStorageSentinel);
        write_padded_plane_image(lhs_planes[0], native_input);
        write_padded_plane_image(rhs_planes[0], native_input);
        write_padded_plane_image(out_planes[0], native_sentinel);
        out_plane.device()->mesh_command_queue(0).finish();
        std::vector<std::byte> before(spec.logical_nbytes());
        iom_conformance::copy_to_host(out->view(), before);

        auto queue = devices.candidate->create_ops();
        const auto submit = [&](std::size_t index) {
            switch (index) {
                case 0:
                    return queue->add(
                            lhs->view(), rhs->view(), out->view());
                case 1:
                    return queue->mul(
                            lhs->view(), rhs->view(), out->view());
                case 2:
                    return queue->sub(
                            lhs->view(), rhs->view(), out->view());
                case 3:
                    return queue->div(
                            lhs->view(), rhs->view(), out->view());
            }
            return iom::to_oid(iom::OidError::InternalError);
        };
        iom::ttnn_test::fail_next_binary_outcome_insertion_for_testing();
        const iom::oid outcome_rejected = submit(operation);
        CHECK_EQ(
                outcome_rejected,
                iom::to_oid(iom::OidError::ResourceExhausted));
        CHECK(iom::ttnn_test::
                      binary_outcome_insertion_fault_consumed_for_testing());
        std::vector<std::byte> after_outcome_failure(
                spec.logical_nbytes());
        iom_conformance::copy_to_host(out->view(), after_outcome_failure);
        CHECK_EQ(after_outcome_failure, before);

        iom::ttnn_test::
                fail_next_host_transfer_staging_allocation_for_testing();
        const iom::oid retained = submit(operation);
        REQUIRE(iom::oid_is_token(retained));
        bool first_wait_threw = false;
        bool second_wait_threw = false;
        try {
            queue->wait(retained);
        } catch (const std::bad_alloc&) {
            first_wait_threw = true;
        }
        try {
            queue->wait(retained);
        } catch (const std::bad_alloc&) {
            second_wait_threw = true;
        }
        CHECK(first_wait_threw);
        CHECK(second_wait_threw);
        CHECK(iom::ttnn_test::
                      host_transfer_staging_allocation_fault_consumed_for_testing());
        std::vector<std::byte> after(spec.logical_nbytes());
        iom_conformance::copy_to_host(out->view(), after);
        CHECK_EQ(after, before);

        const iom::oid accepted = submit(operation);
        REQUIRE(iom::oid_is_token(accepted));
        CHECK_EQ(iom_conformance::token_sequence(accepted), 2);
        REQUIRE_NOTHROW(queue->wait(accepted));
    }
}

TEST_CASE("TTNN un-drainable failed submission reports a repeatable failed token") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::F32};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    iom_conformance::copy_from_host(source->view(), iom_conformance::encode_logical(spec, 31));

    auto queue = devices.candidate->create_ops();
    // The submission fails after one plane and the synchronous drain fails,
    // so the operation is published with the retained submission failure
    // instead of throwing: the caller receives a waitable token whose
    // repeated waits report the failure.
    iom::ttnn_test::fail_next_copy_planes_submission_for_testing(1);
    iom::ttnn_test::fail_next_copy_finishes_for_testing(2);
    const iom::oid token = queue->copy(source->view(), destination->view());

    std::string first_message;
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool caught = false;
        try {
            queue->wait(token);
        } catch (const std::runtime_error& error) {
            caught = true;
            const std::string message = error.what();
            CHECK(message.find("copy-plane submission failure")
                  != std::string::npos);
            if (first_message.empty()) {
                first_message = message;
            } else {
                CHECK_EQ(std::string_view(message), first_message);
            }
        }
        CHECK(caught);
    }
    CHECK_FALSE(iom::ttnn_test::copy_finish_fault_pending_for_testing());
    CHECK(iom::ttnn_test::
                  copy_planes_submission_fault_consumed_for_testing());

    // The retained completion left the entries invalidated; destroying and
    // reusing the owners never frees planes ahead of the pending mesh work,
    // and the device stays fully usable.
    queue.reset();
    source.reset();
    destination.reset();
    require_healthy_copy_after_failure(
            *devices.candidate, spec, 32,
            "copy after retained-failure drain");
}

TEST_CASE("TTNN native finish failure reports a repeatable failed token") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::F32};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    iom_conformance::copy_from_host(source->view(), iom_conformance::encode_logical(spec, 41));

    auto queue = devices.candidate->create_ops();
    // The submission itself completes; the worker's one native finish for
    // the operation fails, so the completion reports the finish failure.
    iom::ttnn_test::fail_next_copy_finishes_for_testing(1);
    const iom::oid token = queue->copy(source->view(), destination->view());

    std::string first_message;
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool caught = false;
        try {
            queue->wait(token);
        } catch (const std::runtime_error& error) {
            caught = true;
            const std::string message = error.what();
            CHECK(message.find("copy finish failure")
                  != std::string::npos);
            if (first_message.empty()) {
                first_message = message;
            } else {
                CHECK_EQ(std::string_view(message), first_message);
            }
        }
        CHECK(caught);
    }
    CHECK_FALSE(iom::ttnn_test::copy_finish_fault_pending_for_testing());

    // Queue destruction and owner reuse after repeated failed waits drain
    // without leaks or double release; the next operation is healthy.
    queue.reset();
    source.reset();
    destination.reset();
    require_healthy_copy_after_failure(
            *devices.candidate, spec, 42,
            "copy after native finish failure");
}

TEST_CASE("TTNN identical-window copies complete without native finish") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 2, 16, 16}}, iom::DataType::BF16};
    auto source = devices.candidate->create_tensor(spec);
    auto queue = devices.candidate->create_ops();

    iom::ttnn_test::reset_copy_finish_count_for_testing();
    const iom::oid token = queue->copy(source->view(), source->view());
    REQUIRE_NOTHROW(queue->wait(token));
    REQUIRE_NOTHROW(queue->wait(token));
    CHECK_EQ(iom::ttnn_test::copy_finish_count_for_testing(), 0);
}

TEST_CASE("TTNN mixed no-op and native batches finish native work once") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 2, 16, 16}}, iom::DataType::BF16};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    auto queue = devices.candidate->create_ops();

    iom::ttnn_test::reset_copy_finish_count_for_testing();
    const iom::oid no_op = queue->copy(source->view(), source->view());
    const iom::oid native =
            queue->copy(source->view(), destination->view());
    REQUIRE_NOTHROW(queue->wait(no_op));
    REQUIRE_NOTHROW(queue->wait(native));
    CHECK_EQ(iom::ttnn_test::copy_finish_count_for_testing(), 1);
    REQUIRE_NOTHROW(queue->wait(no_op));
    REQUIRE_NOTHROW(queue->wait(native));
    CHECK_EQ(iom::ttnn_test::copy_finish_count_for_testing(), 1);
}

TEST_CASE("TTNN no-wait bursts share one native finish per ready batch") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{4, 2, 16, 16}}, iom::DataType::BF16};

    // Serial submission with a wait after every copy: each completion is
    // its own one-task batch, so exactly one native mesh finish occurs per
    // copy regardless of the worker's timing. This pins the not-batched
    // path to one finish per task.
    {
        auto source = devices.candidate->create_tensor(spec);
        auto destination = devices.candidate->create_tensor(spec);
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(spec, 51);
        iom_conformance::copy_from_host(source->view(), pattern);
        auto queue = devices.candidate->create_ops();
        iom::ttnn_test::reset_copy_finish_count_for_testing();
        constexpr std::size_t kSerialCopies = 4;
        for (std::size_t i = 0; i < kSerialCopies; ++i) {
            const iom::oid token =
                    queue->copy(source->view(), destination->view());
            REQUIRE_NOTHROW(queue->wait(token));
        }
        CHECK_EQ(
                iom::ttnn_test::copy_finish_count_for_testing(),
                kSerialCopies);
        iom_conformance::require_logical_bytes(
                destination->view(), pattern, "serial copy content");
        queue.reset();
    }

    // A no-wait burst: the worker's single native finish covers every task
    // that was already executed when the batch was collected, so the whole
    // ready batch completes under one finish and no token is finished more
    // than once. The finish count is at most one per task, every token
    // settles in submission order with correct content, and waits stay
    // repeatable after the batched completion.
    {
        auto source = devices.candidate->create_tensor(spec);
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(spec, 52);
        iom_conformance::copy_from_host(source->view(), pattern);
        std::vector<std::unique_ptr<iom::Tensor>> destinations;
        for (std::size_t i = 0; i < 8; ++i) {
            destinations.push_back(
                    devices.candidate->create_tensor(spec));
        }
        auto queue = devices.candidate->create_ops();
        iom::ttnn_test::reset_copy_finish_count_for_testing();
        std::vector<iom::oid> tokens;
        for (const auto& destination : destinations) {
            tokens.push_back(
                    queue->copy(source->view(), destination->view()));
        }
        for (const iom::oid token : tokens) {
            REQUIRE_NOTHROW(queue->wait(token));
        }
        const std::uint64_t finishes =
                iom::ttnn_test::copy_finish_count_for_testing();
        CHECK_GT(finishes, 0);
        CHECK_LE(finishes, tokens.size());
        for (const auto& destination : destinations) {
            iom_conformance::require_logical_bytes(
                    destination->view(), pattern, "burst copy content");
        }
        REQUIRE_NOTHROW(queue->wait(tokens.front()));
        queue.reset();
    }
}

TEST_CASE("TTNN one failed batch finish fails every token of that batch") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::F32};
    auto source = devices.candidate->create_tensor(spec);
    std::vector<std::unique_ptr<iom::Tensor>> destinations;
    for (std::size_t i = 0; i < 4; ++i) {
        destinations.push_back(devices.candidate->create_tensor(spec));
    }
    auto queue = devices.candidate->create_ops();

    // Arm exactly one native-finish failure for the burst. The first batch
    // the worker collects always contains the first token, so it consumes
    // the seam: every token of that batch fails as a unit, every later
    // batch succeeds, and the failed tokens form a prefix of the
    // submission order.
    iom::ttnn_test::fail_next_copy_finishes_for_testing(1);
    std::vector<iom::oid> tokens;
    for (const auto& destination : destinations) {
        tokens.push_back(queue->copy(source->view(), destination->view()));
    }

    iom::oid first_failed_token = 0;
    bool failures_ended = false;
    bool saw_failure = false;
    for (const iom::oid token : tokens) {
        bool failed = false;
        try {
            queue->wait(token);
        } catch (const std::runtime_error& error) {
            failed = true;
            CHECK(std::string_view(error.what())
                          .find("copy finish failure")
                  != std::string_view::npos);
        }
        if (failed) {
            CHECK_FALSE(failures_ended);
            saw_failure = true;
            if (first_failed_token == 0) {
                first_failed_token = token;
            }
        } else {
            failures_ended = true;
        }
    }
    CHECK(saw_failure);
    REQUIRE_NE(first_failed_token, 0);
    CHECK_FALSE(iom::ttnn_test::copy_finish_fault_pending_for_testing());

    // The failed token's wait stays repeatable with the identical message.
    std::string first_message;
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool caught = false;
        try {
            queue->wait(first_failed_token);
        } catch (const std::runtime_error& error) {
            caught = true;
            CHECK(std::string_view(error.what())
                          .find("copy finish failure")
                  != std::string_view::npos);
            const std::string message = error.what();
            if (first_message.empty()) {
                first_message = message;
            } else {
                CHECK_EQ(std::string_view(message), first_message);
            }
        }
        CHECK(caught);
    }

    // The batch failure kept the ownership guarantees: destroying and
    // reusing the owners never frees planes ahead of the pending mesh
    // work, and the same device stays fully usable.
    queue.reset();
    source.reset();
    destinations.clear();
    require_healthy_copy_after_failure(
            *devices.candidate, spec, 61,
            "copy after failed batch finish");
}

namespace {

    void set_add_logical_value(
            const iom::TensorSpec& spec, std::vector<std::byte>& buffer,
            std::size_t element, std::uint64_t value) {
        const std::size_t bits = iom::detail::leaf_bits(spec.data_type);
        for (std::size_t bit = 0; bit < bits; ++bit) {
            if ((value >> bit) & 1u) {
                const std::size_t output_bit = element * bits + bit;
                buffer[output_bit / 8] |= std::byte{
                        static_cast<unsigned char>(1u << (output_bit % 8))};
            }
        }
    }

    std::uint64_t get_add_logical_value(
            const iom::TensorSpec& spec, std::span<const std::byte> buffer,
            std::size_t element) {
        const std::size_t bits = iom::detail::leaf_bits(spec.data_type);
        std::uint64_t value = 0;
        for (std::size_t bit = 0; bit < bits; ++bit) {
            const std::size_t input_bit = element * bits + bit;
            if ((std::to_integer<unsigned char>(buffer[input_bit / 8])
                 >> (input_bit % 8)) & 1u) {
                value |= std::uint64_t{1} << bit;
            }
        }
        return value;
    }

    iom::TensorSpec add_leaf_spec(iom::DataType type, std::size_t elements) {
        std::size_t columns = 32;
        while (columns * columns < elements) {
            columns *= 2;
        }
        return iom::TensorSpec{
                iom::TensorShape{{1, columns, columns}}, type};
    }

    enum class AddFloatClass {
        finite,
        zero,
        infinity,
        nan,
    };

    struct AddLeafComparison {
        bool matches = false;
        AddFloatClass comparison_class = AddFloatClass::finite;
    };

    struct AddFloatFormat {
        unsigned bits;
        unsigned exponent_bits;
        unsigned fraction_bits;
    };

    AddFloatFormat add_float_format(iom::DataType type) {
        switch (type) {
            case iom::DataType::F16: return {16, 5, 10};
            case iom::DataType::BF16: return {16, 8, 7};
            case iom::DataType::F32: return {32, 8, 23};
            case iom::DataType::F64: return {64, 11, 52};
            default: throw std::invalid_argument("not a supported floating ADD type");
        }
    }

    const char* add_type_name(iom::DataType type) {
        switch (type) {
            case iom::DataType::F16: return "F16";
            case iom::DataType::BF16: return "BF16";
            case iom::DataType::F32: return "F32";
            case iom::DataType::F64: return "F64";
            default: return "non-floating";
        }
    }
    bool add_supported_float_leaf(iom::DataType type) {
        return type == iom::DataType::F16
                || type == iom::DataType::BF16
                || type == iom::DataType::F32
                || type == iom::DataType::F64;
    }

    const char* add_float_class_name(AddFloatClass value) {
        switch (value) {
            case AddFloatClass::finite: return "finite";
            case AddFloatClass::zero: return "zero";
            case AddFloatClass::infinity: return "infinity";
            case AddFloatClass::nan: return "NaN";
        }
        return "unknown";
    }

    AddFloatClass classify_add_float(
            std::uint64_t raw, AddFloatFormat format) {
        const std::uint64_t fraction_mask =
                (std::uint64_t{1} << format.fraction_bits) - 1;
        const std::uint64_t exponent_mask =
                (std::uint64_t{1} << format.exponent_bits) - 1;
        const std::uint64_t exponent =
                (raw >> format.fraction_bits) & exponent_mask;
        const std::uint64_t fraction = raw & fraction_mask;
        if (exponent == exponent_mask) {
            return fraction == 0 ? AddFloatClass::infinity
                                 : AddFloatClass::nan;
        }
        return exponent == 0 && fraction == 0 ? AddFloatClass::zero
                                               : AddFloatClass::finite;
    }

    AddLeafComparison compare_add_leaf_encoding(
            iom::DataType type, std::uint64_t expected, std::uint64_t observed) {
        const AddFloatFormat format = add_float_format(type);
        const std::uint64_t sign_bit = std::uint64_t{1} << (format.bits - 1);
        const AddFloatClass expected_class =
                classify_add_float(expected, format);
        const AddFloatClass observed_class =
                classify_add_float(observed, format);
        if (expected_class != observed_class) {
            return {false, observed_class};
        }
        if (expected_class == AddFloatClass::nan) {
            return {true, expected_class};
        }
        if (expected_class == AddFloatClass::infinity
            || expected_class == AddFloatClass::zero) {
            return {expected == observed, expected_class};
        }
        const auto ordered = [sign_bit](std::uint64_t bits) {
            return (bits & sign_bit) != 0 ? ~bits : bits | sign_bit;
        };
        const std::uint64_t expected_order = ordered(expected);
        const std::uint64_t observed_order = ordered(observed);
        const std::uint64_t distance =
                expected_order > observed_order
                ? expected_order - observed_order
                : observed_order - expected_order;
        return {distance <= 1, expected_class};
    }

    bool add_integer_leaf(iom::DataType type) {
        switch (type) {
            case iom::DataType::I2:
            case iom::DataType::U2:
            case iom::DataType::I4:
            case iom::DataType::U4:
            case iom::DataType::I8:
            case iom::DataType::U8:
            case iom::DataType::I16:
            case iom::DataType::U16:
            case iom::DataType::I32:
            case iom::DataType::U32:
            case iom::DataType::I64:
            case iom::DataType::U64:
                return true;
            default:
                return false;
        }

    }
    bool add_result_matches_oracle(
            iom::DataType type, std::uint64_t lhs, std::uint64_t rhs,
            std::uint64_t observed) {
        const std::uint64_t expected =
                iom_conformance::add_oracle::add(type, lhs, rhs);
        if (expected == observed) {
            return true;
        }
        if (add_integer_leaf(type)) {
            return false;
        }
        if (!add_supported_float_leaf(type)) {
            return false;
        }
        return compare_add_leaf_encoding(type, expected, observed).matches;
    }

    void require_add_leaf_matches_oracle(
            iom::Device& device, iom::DataType type,
            std::span<const std::uint64_t> lhs_values,
            std::span<const std::uint64_t> rhs_values) {
        REQUIRE_EQ(lhs_values.size(), rhs_values.size());
        const iom::TensorSpec spec =
                add_leaf_spec(type, lhs_values.size());
        auto lhs = device.create_tensor(spec);
        auto rhs = device.create_tensor(spec);
        auto out = device.create_tensor(spec);
        REQUIRE(lhs != nullptr);
        REQUIRE(rhs != nullptr);
        REQUIRE(out != nullptr);
        std::vector<std::byte> lhs_bytes(
                spec.logical_nbytes(), std::byte{0});
        std::vector<std::byte> rhs_bytes(
                spec.logical_nbytes(), std::byte{0});
        for (std::size_t i = 0; i < lhs_values.size(); ++i) {
            set_add_logical_value(spec, lhs_bytes, i, lhs_values[i]);
            set_add_logical_value(spec, rhs_bytes, i, rhs_values[i]);
        }
        iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
        iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
        auto queue = device.create_ops();
        const iom::oid token =
                queue->add(lhs->view(), rhs->view(), out->view());
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));
        std::vector<std::byte> out_bytes(
                spec.logical_nbytes(), std::byte{0});
        iom_conformance::copy_to_host(out->view(), out_bytes);
        for (std::size_t i = 0; i < lhs_values.size(); ++i) {
            const std::uint64_t expected =
                    iom_conformance::add_oracle::add(
                            type, lhs_values[i], rhs_values[i]);
            const std::uint64_t observed =
                    get_add_logical_value(spec, out_bytes, i);
            const bool matches = add_result_matches_oracle(
                    type, lhs_values[i], rhs_values[i], observed);
            INFO("compact dtype enum=" << static_cast<int>(type)
                                       << " pair index=" << i
                                       << " lhs raw=" << lhs_values[i]
                                       << " rhs raw=" << rhs_values[i]
                                       << " expected raw=" << expected
                                       << " observed raw=" << observed);
            CHECK(matches);
        }
    }

    struct AddPairs {
        std::vector<std::uint64_t> lhs;
        std::vector<std::uint64_t> rhs;
    };

    AddPairs enumerate_add_pairs(unsigned width) {
        const std::size_t value_count = std::size_t{1} << width;
        const std::size_t pair_count = value_count * value_count;
        AddPairs pairs;
        pairs.lhs.resize(pair_count);
        pairs.rhs.resize(pair_count);
        for (std::size_t i = 0; i < pair_count; ++i) {
            pairs.lhs[i] = i / value_count;
            pairs.rhs[i] = i % value_count;
        }
        return pairs;
    }


    std::vector<std::uint64_t> representative_wide_values(
            iom::DataType type) {
        switch (type) {
            case iom::DataType::I16:
            case iom::DataType::U16:
                return {0, 1, 0x7FFF, 0x8000, 0xFFFF, 2, 0x8001};
            case iom::DataType::I32:
            case iom::DataType::U32:
                return {0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
                        0x12345678, 0x80000001};
            case iom::DataType::I64:
            case iom::DataType::U64:
                return {0, 1, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull,
                        0xFFFFFFFFFFFFFFFFull, 0x123456789ABCDEF0ull,
                        0x8000000000000001ull};
            case iom::DataType::F16:
                return {0, 0x8000, 0x3C00, 0x0400, 0x03FF, 0x7BFF, 0x7C00,
                        0xFC00, 0x7E00, 0x0001};
            case iom::DataType::BF16:
                return {0, 0x8000, 0x3F80, 0x0080, 0x007F, 0x7F7F, 0x7F80,
                        0xFF80, 0x7FC0, 0x0001};
            case iom::DataType::F32:
                return {0, 0x80000000, 0x3F800000, 0x00800000, 0x007FFFFF,
                        0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
                        0x00000001};
            case iom::DataType::F64:
                return {0, 0x8000000000000000ull, 0x3FF0000000000000ull,
                        0x0010000000000000ull, 0x000FFFFFFFFFFFFFull,
                        0x7FEFFFFFFFFFFFFFull, 0x7FF0000000000000ull,
                        0xFFF0000000000000ull, 0x7FF8000000000000ull,
                        0x0000000000000001ull};
            default:
                return {0, 1, 2, 3};
        }
    }
}

TEST_CASE("TTNN ADD floating comparison envelope rejects deterministic mutations") {
    const iom::DataType types[] = {
            iom::DataType::F16, iom::DataType::BF16,
            iom::DataType::F32, iom::DataType::F64};
    for (const iom::DataType type : types) {
        const AddFloatFormat format = add_float_format(type);
        const std::uint64_t exponent_mask =
                (std::uint64_t{1} << format.exponent_bits) - 1;
        const std::uint64_t sign_bit = std::uint64_t{1} << (format.bits - 1);
        const std::uint64_t one =
                std::uint64_t{1} << format.fraction_bits;
        const std::uint64_t infinity = exponent_mask << format.fraction_bits;
        const auto check = [type](std::uint64_t expected,
                                   std::uint64_t observed, bool wanted,
                                   const char* scenario) {
            const AddLeafComparison result =
                    compare_add_leaf_encoding(type, expected, observed);
            CHECK_MESSAGE(
                    result.matches == wanted,
                    add_type_name(type) << " " << scenario
                                        << " expected bits " << expected
                                        << " observed bits " << observed);
        };

        check(one, one, true, "exact finite");
        check(one, one + 1, true, "positive one-ULP finite");
        check(one, one + 2, false, "positive two-ULP finite");
        check(sign_bit | one, (sign_bit | one) - 1, true,
              "negative one-ULP finite");
        check(one, infinity, false, "finite versus infinity");
        check(one, infinity | 1, false, "finite versus NaN");
        check(infinity | 1, infinity | 2, true, "NaN payload");
        check(infinity, infinity | sign_bit, false, "infinity sign");
        check(0, sign_bit, false, "signed zero");
    }
}

TEST_CASE("TTNN ADD exhaustively covers every compact ordered pair") {
    require_hardware();
    TtnnDevices devices;
    const iom::DataType compact[] = {
            iom::DataType::I2, iom::DataType::U2,
            iom::DataType::I4, iom::DataType::U4,
            iom::DataType::F4_E2M1, iom::DataType::F6_E2M3,
            iom::DataType::F6_E3M2};
    for (const iom::DataType type : compact) {
        const unsigned width = type == iom::DataType::I2
                || type == iom::DataType::U2 ? 2
                : type == iom::DataType::I4
                || type == iom::DataType::U4
                || type == iom::DataType::F4_E2M1 ? 4 : 6;
        AddPairs pairs = enumerate_add_pairs(width);
        REQUIRE_EQ(pairs.lhs.size(), pairs.rhs.size());
        REQUIRE_EQ(
                pairs.lhs.size(),
                (std::size_t{1} << width) * (std::size_t{1} << width));
        if (type == iom::DataType::I2) {
            const std::uint64_t expected =
                    iom_conformance::add_oracle::add(
                            type, pairs.lhs[1], pairs.rhs[1]);
            CHECK_FALSE(add_result_matches_oracle(
                    type, pairs.lhs[1], pairs.rhs[1], expected ^ 1u));
        }
        require_add_leaf_matches_oracle(
                *devices.candidate, type, pairs.lhs, pairs.rhs);
    }
}

TEST_CASE("TTNN ADD representative wide leaves match the oracle") {
    require_hardware();
    TtnnDevices devices;
    const iom::DataType wide[] = {
            iom::DataType::I8, iom::DataType::U8,
            iom::DataType::I16, iom::DataType::U16,
            iom::DataType::I32, iom::DataType::U32,
            iom::DataType::I64, iom::DataType::U64,
            iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
            iom::DataType::F16, iom::DataType::BF16,
            iom::DataType::F32, iom::DataType::F64};
    for (const iom::DataType type : wide) {
        std::vector<std::uint64_t> values =
                representative_wide_values(type);
        require_add_leaf_matches_oracle(
                *devices.candidate, type, values, values);
    }
}

TEST_CASE("TTNN ADD accepts documented rank boundaries") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_binary_rank_boundary_conformance(
            *devices.candidate);
}

TEST_CASE("TTNN ADD broadcast, tail, transformed views, and aliases") {
    require_hardware();
    TtnnDevices devices;

    // [1,1] scalar broadcast through rank 3: a scalar operand combines
    // with a full matrix under the common exact-broadcast-output rule.
    {
        const iom::TensorSpec out_spec{
                iom::TensorShape{{1, 2, 2}}, iom::DataType::BF16};
        const iom::TensorSpec scalar_spec{
                iom::TensorShape{{1, 1, 1}}, iom::DataType::BF16};
        auto lhs = devices.candidate->create_tensor(scalar_spec);
        auto rhs = devices.candidate->create_tensor(out_spec);
        auto out = devices.candidate->create_tensor(out_spec);
        std::vector<std::byte> one(
                scalar_spec.logical_nbytes(), std::byte{0});
        std::vector<std::byte> two(
                out_spec.logical_nbytes(), std::byte{0});
        set_add_logical_value(scalar_spec, one, 0, 0x3F80);
        for (std::size_t i = 0; i < 4; ++i) {
            set_add_logical_value(
                    out_spec, two, i, 0x4000 + static_cast<std::uint64_t>(i));
        }
        iom_conformance::copy_from_host(lhs->view(), one);
        iom_conformance::copy_from_host(rhs->view(), two);
        auto queue = devices.candidate->create_ops();
        const iom::oid token =
                queue->add(lhs->view(), rhs->view(), out->view());
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));
        std::vector<std::byte> observed(
                out_spec.logical_nbytes(), std::byte{0});
        iom_conformance::copy_to_host(out->view(), observed);
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK_EQ(get_add_logical_value(out_spec, observed, i),
                     iom_conformance::add_oracle::add(
                             iom::DataType::BF16,
                             0x3F80,
                             0x4000 + static_cast<std::uint64_t>(i)));
        }
    }

    // Opposite-direction singleton broadcasts: lhs broadcasts columns,
    // rhs broadcasts rows, through rank-3 leading promotion.
    {
        const iom::TensorSpec out_spec{
                iom::TensorShape{{1, 2, 2}}, iom::DataType::I32};
        const iom::TensorSpec col_spec{
                iom::TensorShape{{1, 2, 1}}, iom::DataType::I32};
        const iom::TensorSpec row_spec{
                iom::TensorShape{{1, 1, 2}}, iom::DataType::I32};
        auto lhs = devices.candidate->create_tensor(col_spec);
        auto rhs = devices.candidate->create_tensor(row_spec);
        auto out = devices.candidate->create_tensor(out_spec);
        std::vector<std::byte> lhs_bytes(
                col_spec.logical_nbytes(), std::byte{0});
        std::vector<std::byte> rhs_bytes(
                row_spec.logical_nbytes(), std::byte{0});
        set_add_logical_value(col_spec, lhs_bytes, 0, 10);
        set_add_logical_value(col_spec, lhs_bytes, 1, 20);
        set_add_logical_value(row_spec, rhs_bytes, 0, 1);
        set_add_logical_value(row_spec, rhs_bytes, 1, 2);
        iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
        iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
        auto queue = devices.candidate->create_ops();
        const iom::oid token =
                queue->add(lhs->view(), rhs->view(), out->view());
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));
        std::vector<std::byte> observed(
                out_spec.logical_nbytes(), std::byte{0});
        iom_conformance::copy_to_host(out->view(), observed);
        const std::uint64_t expected[2][2] = {{11, 12}, {21, 22}};
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK_EQ(get_add_logical_value(out_spec, observed, i),
                     expected[i / 2][i % 2]);
        }
    }

    // Singleton final tiled axis with row and column tails.
    {
        const iom::TensorSpec spec{
                iom::TensorShape{{1, 33, 17}}, iom::DataType::BF16};
        auto lhs = devices.candidate->create_tensor(spec);
        auto rhs = devices.candidate->create_tensor(spec);
        auto out = devices.candidate->create_tensor(spec);
        const std::size_t elements = 33 * 17;
        std::vector<std::byte> lhs_bytes(
                spec.logical_nbytes(), std::byte{0});
        std::vector<std::byte> rhs_bytes(
                spec.logical_nbytes(), std::byte{0});
        for (std::size_t i = 0; i < elements; ++i) {
            set_add_logical_value(
                    spec, lhs_bytes, i, 0x3F80 + (i % 31));
            set_add_logical_value(
                    spec, rhs_bytes, i, 0x4000 + (i % 17));
        }
        iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
        iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
        auto queue = devices.candidate->create_ops();
        const iom::oid token =
                queue->add(lhs->view(), rhs->view(), out->view());
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));
        std::vector<std::byte> observed(
                spec.logical_nbytes(), std::byte{0});
        iom_conformance::copy_to_host(out->view(), observed);
        for (std::size_t i = 0; i < elements; ++i) {
            const std::uint64_t a = 0x3F80 + (i % 31);
            const std::uint64_t b = 0x4000 + (i % 17);
            CHECK_EQ(get_add_logical_value(spec, observed, i),
                     iom_conformance::add_oracle::add(
                             iom::DataType::BF16, a, b));
        }
    }

    // Transformed leading view through a plane slice.
    {
        const iom::TensorSpec spec{
                iom::TensorShape{{2, 2, 16, 16}}, iom::DataType::F32};
        auto lhs = devices.candidate->create_tensor(spec);
        auto rhs = devices.candidate->create_tensor(spec);
        auto out = devices.candidate->create_tensor(spec);
        std::vector<std::byte> pattern(
                spec.logical_nbytes(), std::byte{0});
        for (std::size_t i = 0; i < 512; ++i) {
            const std::uint32_t value =
                    static_cast<std::uint32_t>(0x3F800000u + i);
            std::memcpy(pattern.data() + i * 4, &value, 4);
        }
        iom_conformance::copy_from_host(lhs->view(), pattern);
        iom_conformance::copy_from_host(rhs->view(), pattern);
        auto queue = devices.candidate->create_ops();
        iom::TensorView lhs_sliced = lhs->view().slice(0, 0, 1);
        iom::TensorView rhs_sliced = rhs->view().slice(0, 0, 1);
        iom::TensorView out_sliced = out->view().slice(0, 0, 1);
        const iom::oid token =
                queue->add(lhs_sliced, rhs_sliced, out_sliced);
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));
        std::vector<std::byte> observed(
                out_sliced.spec().logical_nbytes(), std::byte{0});
        iom_conformance::copy_to_host(out_sliced, observed);
        for (std::size_t i = 0; i < 256; ++i) {
            std::uint32_t a = 0;
            std::uint32_t b = 0;
            std::memcpy(&a, pattern.data() + i * 4, 4);
            std::memcpy(&b, pattern.data() + i * 4, 4);
            std::uint32_t summed = 0;
            const std::uint64_t raw =
                    iom_conformance::add_oracle::add(
                            iom::DataType::F32, a, b);
            summed = static_cast<std::uint32_t>(raw);
            std::uint32_t got = 0;
            std::memcpy(&got, observed.data() + i * 4, 4);
            CHECK_EQ(got, summed);
        }
    }

    // Exact in-place alias: out and lhs share the owner; capture-before-
    // store must keep the result equal to the independent oracle.
    {
        const iom::TensorSpec spec{
                iom::TensorShape{{1, 2, 2}}, iom::DataType::U8};
        auto lhs = devices.candidate->create_tensor(spec);
        auto rhs = devices.candidate->create_tensor(spec);
        std::vector<std::byte> lhs_bytes(
                spec.logical_nbytes(), std::byte{0});
        std::vector<std::byte> rhs_bytes(
                spec.logical_nbytes(), std::byte{0});
        for (std::size_t i = 0; i < 4; ++i) {
            set_add_logical_value(spec, lhs_bytes, i, 200 + i);
            set_add_logical_value(spec, rhs_bytes, i, 100 + i);
        }
        iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
        iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
        auto queue = devices.candidate->create_ops();
        const iom::oid token =
                queue->add(lhs->view(), rhs->view(), lhs->view());
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));
        std::vector<std::byte> observed(
                spec.logical_nbytes(), std::byte{0});
        iom_conformance::copy_to_host(lhs->view(), observed);
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK_EQ(get_add_logical_value(spec, observed, i),
                     (200 + i + 100 + i) & 0xFF);
        }
    }

    // BOOL and F8_E8M0 ADD stay Unsupported while BOOL storage round-trips.
    {
        const iom::TensorSpec bool_spec{
                iom::TensorShape{{1, 2, 2}}, iom::DataType::BOOL};
        auto lhs = devices.candidate->create_tensor(bool_spec);
        auto rhs = devices.candidate->create_tensor(bool_spec);
        auto out = devices.candidate->create_tensor(bool_spec);
        std::vector<std::byte> pattern(
                bool_spec.logical_nbytes(), std::byte{0});
        for (std::size_t i = 0; i < 4; ++i) {
            set_add_logical_value(bool_spec, pattern, i, i % 2);
        }
        iom_conformance::copy_from_host(lhs->view(), pattern);
        iom_conformance::copy_from_host(rhs->view(), pattern);
        auto queue = devices.candidate->create_ops();
        CHECK_EQ(queue->add(lhs->view(), rhs->view(), out->view()),
                 iom::to_oid(iom::OidError::Unsupported));
        std::vector<std::byte> observed(
                bool_spec.logical_nbytes(), std::byte{0});
        iom_conformance::copy_to_host(out->view(), observed);
        CHECK(observed
              == std::vector<std::byte>(
                      bool_spec.logical_nbytes(), std::byte{0}));
    }
}

TEST_CASE("TTNN ADD retained failure repeats and staging stays reusable") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{1, 2, 2}}, iom::DataType::BF16};
    auto lhs = devices.candidate->create_tensor(spec);
    auto rhs = devices.candidate->create_tensor(spec);
    auto out = devices.candidate->create_tensor(spec);
    std::vector<std::byte> one(spec.logical_nbytes(), std::byte{0});
    std::vector<std::byte> two(spec.logical_nbytes(), std::byte{0});
    for (std::size_t i = 0; i < 4; ++i) {
        set_add_logical_value(spec, one, i, 0x3F80 + i);
        set_add_logical_value(spec, two, i, 0x4000 + i);
    }
    iom_conformance::copy_from_host(lhs->view(), one);
    iom_conformance::copy_from_host(rhs->view(), two);
    auto queue = devices.candidate->create_ops();

    // A retained post-acceptance finish failure: every repeated wait
    // rethrows the same retained failure and no retry consumes it.
    iom::ttnn_test::fail_next_copy_finishes_for_testing(1);
    const iom::oid failed =
            queue->add(lhs->view(), rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(failed));
    bool threw = false;
    try {
        queue->wait(failed);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    bool threw_again = false;
    try {
        queue->wait(failed);
    } catch (const std::runtime_error&) {
        threw_again = true;
    }
    CHECK(threw_again);
    iom::ttnn_test::fail_next_copy_finishes_for_testing(0);

    // The failed request's native work had already drained inside the
    // emulation path, so the observable output of the failed request is
    // the computed sum (diagnostic, not contractual).
    std::vector<std::byte> after_failed(
            spec.logical_nbytes(), std::byte{0});
    iom_conformance::copy_to_host(out->view(), after_failed);
    CHECK_EQ(get_add_logical_value(spec, after_failed, 0),
             iom_conformance::add_oracle::add(
                     iom::DataType::BF16, 0x3F80, 0x4000));

    // Reseed every owner so the recovered request is independent of the
    // failed request's effects, then prove the same owners, queue, and
    // device-owned staging slots still serve a correct ADD: the upload
    // lease returned cleanly and the emulation path repeats.
    std::vector<std::byte> zero(spec.logical_nbytes(), std::byte{0});
    iom_conformance::copy_from_host(out->view(), zero);
    iom_conformance::copy_from_host(lhs->view(), one);
    iom_conformance::copy_from_host(rhs->view(), two);
    const iom::oid recovered =
            queue->add(lhs->view(), rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(recovered));
    REQUIRE_NOTHROW(queue->wait(recovered));
    std::vector<std::byte> observed(
            spec.logical_nbytes(), std::byte{0});
    iom_conformance::copy_to_host(out->view(), observed);
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK_EQ(get_add_logical_value(spec, observed, i),
                 iom_conformance::add_oracle::add(
                         iom::DataType::BF16,
                         0x3F80 + i,
                         0x4000 + i));
    }
}

TEST_CASE("TTNN binary completion publishes one finish for every operation") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 32, 32}}, iom::DataType::F32};
    auto lhs = devices.candidate->create_tensor(spec);
    auto rhs = devices.candidate->create_tensor(spec);
    auto out = devices.candidate->create_tensor(spec);
    std::vector<std::byte> lhs_bytes(spec.logical_nbytes(), std::byte{0});
    std::vector<std::byte> rhs_bytes(spec.logical_nbytes(), std::byte{0});
    for (std::size_t index = 0; index < 2 * 32 * 32; ++index) {
        const float left = 2.0F;
        const float right = 4.0F;
        std::memcpy(lhs_bytes.data() + index * sizeof(float), &left,
                    sizeof(float));
        std::memcpy(rhs_bytes.data() + index * sizeof(float), &right,
                    sizeof(float));
    }
    iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
    iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
    auto queue = devices.candidate->create_ops();

    struct BinaryCase {
        enum class Kind { add, mul, sub, div };
        Kind kind;
        float expected;
    };
    const BinaryCase cases[] = {
            {BinaryCase::Kind::add, 6.0F},
            {BinaryCase::Kind::mul, 8.0F},
            {BinaryCase::Kind::sub, -2.0F},
            {BinaryCase::Kind::div, 0.5F}};
    for (const BinaryCase test : cases) {
        iom::ttnn_test::reset_copy_finish_count_for_testing();
        iom::oid token = 0;
        switch (test.kind) {
            case BinaryCase::Kind::add:
                token = queue->add(lhs->view(), rhs->view(), out->view());
                break;
            case BinaryCase::Kind::mul:
                token = queue->mul(lhs->view(), rhs->view(), out->view());
                break;
            case BinaryCase::Kind::sub:
                token = queue->sub(lhs->view(), rhs->view(), out->view());
                break;
            case BinaryCase::Kind::div:
                token = queue->div(lhs->view(), rhs->view(), out->view());
                break;
        }
        REQUIRE(iom::oid_is_token(token));
        REQUIRE_NOTHROW(queue->wait(token));
        REQUIRE_NOTHROW(queue->wait(token));
        CHECK_EQ(iom::ttnn_test::copy_finish_count_for_testing(), 1);

        std::vector<std::byte> observed(spec.logical_nbytes(), std::byte{0});
        iom_conformance::copy_to_host(out->view(), observed);
        for (std::size_t index = 0; index < 2 * 32 * 32; ++index) {
            float value = 0.0F;
            std::memcpy(&value,
                        observed.data() + index * sizeof(float),
                        sizeof(float));
            CHECK_EQ(value, test.expected);
        }
    }
}

TEST_CASE("TTNN binary publication precedes deferred worker execution") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{1, 2, 2}}, iom::DataType::F32};

    struct BinaryCase {
        enum class Kind { add, mul, sub, div };
        Kind kind;
        float expected;
    };
    const BinaryCase cases[] = {
            {BinaryCase::Kind::add, 6.0F},
            {BinaryCase::Kind::mul, 8.0F},
            {BinaryCase::Kind::sub, -2.0F},
            {BinaryCase::Kind::div, 0.5F}};

    for (const BinaryCase test : cases) {
        auto lhs = devices.candidate->create_tensor(spec);
        auto rhs = devices.candidate->create_tensor(spec);
        auto first_out = devices.candidate->create_tensor(spec);
        auto second_out = devices.candidate->create_tensor(spec);
        std::vector<std::byte> lhs_bytes(spec.logical_nbytes(), std::byte{0});
        std::vector<std::byte> rhs_bytes(spec.logical_nbytes(), std::byte{0});
        for (std::size_t index = 0; index < 4; ++index) {
            const float left = 2.0F;
            const float right = 4.0F;
            std::memcpy(
                    lhs_bytes.data() + index * sizeof(float), &left,
                    sizeof(float));
            std::memcpy(
                    rhs_bytes.data() + index * sizeof(float), &right,
                    sizeof(float));
        }
        iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
        iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
        auto queue = devices.candidate->create_ops();
        const auto submit = [&](iom::TensorView& out) {
            switch (test.kind) {
                case BinaryCase::Kind::add:
                    return queue->add(lhs->view(), rhs->view(), out);
                case BinaryCase::Kind::mul:
                    return queue->mul(lhs->view(), rhs->view(), out);
                case BinaryCase::Kind::sub:
                    return queue->sub(lhs->view(), rhs->view(), out);
                case BinaryCase::Kind::div:
                    return queue->div(lhs->view(), rhs->view(), out);
            }
            return iom::to_oid(iom::OidError::InternalError);
        };

        iom::ttnn_test::hold_binary_execution_barrier_for_testing();
        const iom::oid first = submit(first_out->view());
        iom::ttnn_test::wait_binary_execution_barrier_for_testing();
        const iom::oid second = submit(second_out->view());
        CHECK(iom::oid_is_token(first));
        CHECK(iom::oid_is_token(second));
        iom::ttnn_test::release_binary_execution_barrier_for_testing();
        if (iom::oid_is_token(first)) {
            CHECK_NOTHROW(queue->wait(first));
            CHECK_NOTHROW(queue->wait(first));
        }
        if (iom::oid_is_token(second)) {
            CHECK_NOTHROW(queue->wait(second));
            CHECK_NOTHROW(queue->wait(second));
        }

        for (iom::Tensor* output : {first_out.get(), second_out.get()}) {
            std::vector<std::byte> observed(
                    spec.logical_nbytes(), std::byte{0});
            iom_conformance::copy_to_host(output->view(), observed);
            for (std::size_t index = 0; index < 4; ++index) {
                float value = 0.0F;
                std::memcpy(
                        &value, observed.data() + index * sizeof(float),
                        sizeof(float));
                CHECK_EQ(value, test.expected);
            }
        }
    }
}

TEST_CASE("TTNN deferred binary failures remain repeatable after release") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{1, 2, 2}}, iom::DataType::BF16};
    auto lhs = devices.candidate->create_tensor(spec);
    auto rhs = devices.candidate->create_tensor(spec);
    auto out = devices.candidate->create_tensor(spec);
    std::vector<std::byte> lhs_bytes(spec.logical_nbytes(), std::byte{0});
    std::vector<std::byte> rhs_bytes(spec.logical_nbytes(), std::byte{0});
    for (std::size_t index = 0; index < 4; ++index) {
        set_add_logical_value(spec, lhs_bytes, index, 0x3F80 + index);
        set_add_logical_value(spec, rhs_bytes, index, 0x4000 + index);
    }
    iom_conformance::copy_from_host(lhs->view(), lhs_bytes);
    iom_conformance::copy_from_host(rhs->view(), rhs_bytes);
    auto queue = devices.candidate->create_ops();

    iom::ttnn_test::hold_binary_execution_barrier_for_testing();
    iom::ttnn_test::fail_next_copy_finishes_for_testing(1);
    const iom::oid token = queue->add(lhs->view(), rhs->view(), out->view());
    iom::ttnn_test::wait_binary_execution_barrier_for_testing();
    CHECK(iom::oid_is_token(token));
    iom::ttnn_test::release_binary_execution_barrier_for_testing();
    bool first_threw = false;
    bool second_threw = false;
    try {
        queue->wait(token);
    } catch (const std::runtime_error&) {
        first_threw = true;
    }
    try {
        queue->wait(token);
    } catch (const std::runtime_error&) {
        second_threw = true;
    }
    CHECK(first_threw);
    CHECK(second_threw);
    iom::ttnn_test::fail_next_copy_finishes_for_testing(0);
}


TEST_CASE("TTNN positive workspaces own exact aligned host ranges") {
    require_hardware();
    TtnnDevices devices;
    auto* const ttnn_device =
            dynamic_cast<iom::TtnnDevice*>(devices.candidate.get());
    REQUIRE(ttnn_device != nullptr);

    CHECK_THROWS_AS(
            (void)devices.candidate->create_workspace(
                    std::numeric_limits<std::size_t>::max()),
            std::overflow_error);

    auto workspace = devices.candidate->create_workspace(96);
    REQUIRE(workspace != nullptr);
    CHECK_FALSE(workspace->empty());
    CHECK_EQ(workspace->byte_size(), std::size_t{96});
    const iom::RawWorkspaceView full = workspace->view();
    const iom::ttnn_detail::HostWorkspace full_access =
            iom::ttnn_detail::checked_host_workspace(*ttnn_device, full);
    REQUIRE(full_access.data != nullptr);
    CHECK_EQ(full_access.byte_size, std::size_t{96});
    CHECK_EQ(
            reinterpret_cast<std::uintptr_t>(full_access.data) % 32,
            std::uintptr_t{0});
    CHECK(full_access.keepalive != nullptr);

    const iom::RawWorkspaceView first = full.subrange(0, 32);
    const iom::RawWorkspaceView second = full.subrange(32, 64);
    const iom::ttnn_detail::HostWorkspace first_access =
            iom::ttnn_detail::checked_host_workspace(*ttnn_device, first);
    const iom::ttnn_detail::HostWorkspace second_access =
            iom::ttnn_detail::checked_host_workspace(*ttnn_device, second);
    CHECK_NE(first_access.data, second_access.data);
    CHECK_EQ(second_access.data, first_access.data + 32);
    CHECK_THROWS_AS(
            (void)full.subrange(16, 32), std::invalid_argument);
    CHECK_THROWS_AS(
            (void)full.subrange(64, 64), std::out_of_range);

    auto empty = devices.candidate->create_workspace(0);
    REQUIRE(empty != nullptr);
    CHECK(empty->empty());
    CHECK_THROWS_AS(
            (void)iom::ttnn_detail::checked_host_workspace(
                    *ttnn_device, empty->view()),
            std::invalid_argument);
    auto foreign = devices.foreign->create_workspace(0);
    CHECK_THROWS_AS(
            (void)iom::ttnn_detail::checked_host_workspace(
                    *ttnn_device, foreign->view()),
            std::invalid_argument);
    const iom::RawWorkspaceView stale = workspace->view();
    workspace.reset();
    CHECK_THROWS_AS(
            (void)iom::ttnn_detail::checked_host_workspace(
                    *ttnn_device, stale),
            std::invalid_argument);
    CHECK_THROWS_AS(
            (void)iom::ttnn_detail::checked_host_workspace(
                    *ttnn_device, iom::RawWorkspaceView{}),
            std::invalid_argument);
}

TEST_CASE(
        "TTNN workspace owner destruction retains a live common lease") {
    require_hardware();
    TtnnDevices devices;
    auto* const ttnn_device =
            dynamic_cast<iom::TtnnDevice*>(devices.candidate.get());
    REQUIRE(ttnn_device != nullptr);

    auto table = devices.candidate->create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{1, 8}}, iom::DataType::U32});
    auto indices = devices.candidate->create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{1, 8}}, iom::DataType::U32});
    auto out = devices.candidate->create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{8, 8}}, iom::DataType::U32});
    REQUIRE(table != nullptr);
    REQUIRE(indices != nullptr);
    REQUIRE(out != nullptr);
    std::vector<std::byte> table_bytes(
            table->view().spec().logical_nbytes(), std::byte{0});
    std::vector<std::byte> index_bytes(
            indices->view().spec().logical_nbytes(), std::byte{0});
    iom_conformance::copy_from_host(table->view(), table_bytes);
    iom_conformance::copy_from_host(indices->view(), index_bytes);

    auto queue = devices.candidate->create_ops();
    auto workspace = devices.candidate->create_workspace(32);
    REQUIRE(workspace != nullptr);
    const iom::RawWorkspaceView view = workspace->view();
    std::byte* host_data = nullptr;
    {
        const iom::ttnn_detail::HostWorkspace access =
                iom::ttnn_detail::checked_host_workspace(
                        *ttnn_device, view);
        REQUIRE(access.data != nullptr);
        host_data = access.data;
        std::fill_n(host_data, 32, static_cast<std::byte>(0x5A));
    }

    iom::ttnn_test::reset_host_workspace_release_count_for_testing();
    const iom::oid token =
            queue->embedding(table->view(), indices->view(), out->view(), view);
    REQUIRE(iom::oid_is_token(token));

    // The accepted operation has a common workspace lease.  Destroying the
    // owner must transfer its host keepalive to quarantine rather than unmap
    // the bytes while that lease is still live.
    workspace.reset();
    CHECK_EQ(
            iom::ttnn_test::host_workspace_release_count_for_testing(),
            std::size_t{0});
    CHECK_EQ(*host_data, static_cast<std::byte>(0x5A));

    CHECK_NOTHROW(queue->wait(token));
    CHECK_NOTHROW(ttnn_device->registry_state().quarantine.drain());
    CHECK_EQ(
            iom::ttnn_test::host_workspace_release_count_for_testing(),
            std::size_t{1});
}

TEST_CASE("TTNN raw padded BF16 plane round trip uses host workspace") {
    require_hardware();
    TtnnDevices devices;
    auto* const ttnn_device =
            dynamic_cast<iom::TtnnDevice*>(devices.candidate.get());
    REQUIRE(ttnn_device != nullptr);
    auto tensor = devices.candidate->create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{17, 17}}, iom::DataType::BF16});
    REQUIRE(tensor != nullptr);
    auto* const plane =
            static_cast<ttnn::Tensor*>(tensor->view().native_handle());
    REQUIRE(plane != nullptr);
    const std::size_t bytes = iom::ttnn_detail::padded_plane_bytes(*plane);
    CHECK_EQ(bytes, std::size_t{2048});

    auto workspace = devices.candidate->create_workspace(bytes);
    REQUIRE(workspace != nullptr);
    auto access = iom::ttnn_detail::checked_host_workspace(
            *ttnn_device, workspace->view());
    REQUIRE(access.data != nullptr);
    std::vector<std::byte> expected(bytes);
    for (std::size_t index = 0; index < bytes; ++index) {
        expected[index] = static_cast<std::byte>(
                static_cast<unsigned char>((index * 37 + 11) & 0xff));
    }
    std::memcpy(access.data, expected.data(), bytes);
    {
        auto lease = iom::ttnn_detail::acquire_workspace_lease(
                *ttnn_device, workspace->view());
        iom::ttnn_detail::raw_upload_plane(*ttnn_device, *plane, lease);
        {
            std::lock_guard<std::mutex> lock(ttnn_device->api_mutex());
            ttnn_device->mesh().mesh_command_queue(0).finish();
        }
        lease.complete(true);
    }
    std::memset(access.data, 0, bytes);
    {
        auto lease = iom::ttnn_detail::acquire_workspace_lease(
                *ttnn_device, workspace->view());
        iom::ttnn_detail::raw_download_plane(*ttnn_device, *plane, lease);
        {
            std::lock_guard<std::mutex> lock(ttnn_device->api_mutex());
            ttnn_device->mesh().mesh_command_queue(0).finish();
        }
        lease.complete(true);
    }
    CHECK(std::memcmp(access.data, expected.data(), bytes) == 0);
}

TEST_CASE("TTNN raw transfer submission failure retains host lease") {
    require_hardware();
    TtnnDevices devices;
    auto* const ttnn_device =
            dynamic_cast<iom::TtnnDevice*>(devices.candidate.get());
    REQUIRE(ttnn_device != nullptr);
    auto tensor = devices.candidate->create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{17, 17}}, iom::DataType::BF16});
    REQUIRE(tensor != nullptr);
    auto* const plane =
            static_cast<ttnn::Tensor*>(tensor->view().native_handle());
    REQUIRE(plane != nullptr);
    const std::size_t bytes = iom::ttnn_detail::padded_plane_bytes(*plane);
    auto workspace = devices.candidate->create_workspace(bytes);
    REQUIRE(workspace != nullptr);
    std::byte* host_data = nullptr;
    {
        const iom::ttnn_detail::HostWorkspace access =
                iom::ttnn_detail::checked_host_workspace(
                        *ttnn_device, workspace->view());
        REQUIRE(access.data != nullptr);
        host_data = access.data;
        std::fill_n(host_data, bytes, static_cast<std::byte>(0xA5));
    }
    iom::ttnn_test::reset_host_workspace_release_count_for_testing();

    iom::ttnn_test::fail_next_host_transfer_submission_for_testing(0);
    iom::ttnn_test::fail_next_quarantine_action_for_testing();
    {
        auto lease = iom::ttnn_detail::acquire_workspace_lease(
                *ttnn_device, workspace->view());
        CHECK_THROWS_AS(
                iom::ttnn_detail::raw_download_plane(
                        *ttnn_device, *plane, lease),
                std::runtime_error);
        CHECK(
                iom::ttnn_test::
                        host_transfer_submission_fault_consumed_for_testing());
    }
    CHECK(
            iom::ttnn_test::quarantine_action_fault_consumed_for_testing());

    // The injected submission failure has no completion proof.  The lease
    // therefore deliberately leaks its payload when quarantine recording is
    // faulted, keeping the owner bytes addressable instead of releasing them.
    workspace.reset();
    CHECK_EQ(
            iom::ttnn_test::host_workspace_release_count_for_testing(),
            std::size_t{0});
    CHECK_EQ(*host_data, static_cast<std::byte>(0xA5));
    CHECK_NOTHROW(ttnn_device->registry_state().quarantine.drain());
    CHECK_EQ(
            iom::ttnn_test::host_workspace_release_count_for_testing(),
            std::size_t{0});
}

TEST_CASE("TTNN conformance: workspace requirement queries report exact zero") {
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::F32};
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
    CHECK_THROWS_AS(
            queue->add_workspace_requirements(
                    foreign->view(), rhs->view(), out->view()),
            std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Shared model loading and weight realization.
//
// TTNN consumes exactly the same backend-neutral scenario as every other
// driver: `iom_conformance::run_model_loading_conformance` owns the fixture
// directories, the mapped sources, the caller-owned destinations, the
// workspace preflight, and the readback expectations, and this driver only
// supplies the devices. `TtnnDevices` already implements the documented
// one-live-context policy - one TTNN context plus an independently created CPU
// reference device and the CPU `foreign` device that a second TTNN context
// would violate - so the scenario's foreign-destination rejection observes
// exact `Device` identity across two backend kinds instead of a second TTNN
// ordinal. Native storage stays TTNN-owned per plane behind the same public
// contracts, and the binding reports the `{0, 1}` host-transfer requirement
// that consumes no caller workspace.
// ---------------------------------------------------------------------------
TEST_CASE("TTNN model loading: complete checkpoint inventories realize exact BF16 weights") {
    require_hardware();
    TtnnDevices devices;

    // BF16 is mandatory for weight realization: a device that omits it cannot
    // host the published inventory, so this case fails rather than skips.
    const std::span<const iom::DataType> supported =
            devices.candidate->supported_data_types();
    REQUIRE(std::find(supported.begin(), supported.end(),
                      iom::DataType::BF16) != supported.end());

    iom_conformance::run_model_loading_conformance(devices.conformance());
}

// The realization of a complete ordered binding is synchronous and
// caller-owned: `ModelSource::upload_weights` validates the binding once and
// then copies one published role at a time into that role's destination full
// owner view. Every one of those copies runs through the device's retained
// host-transfer staging, so a failure after the binding was accepted is
// exactly the loading failure the contract must report without publishing a
// model, and the staging the interrupted transfer already acquired must not be
// freed while it could still be read.
//
// Every published role of these checkpoints is one 32x32-tile BF16 native
// plane, so the single-plane submission seam is reached by the transfer that
// carries the role. The staging-allocation seam is armed for the first upload
// of the binding on a device whose retained staging does not exist yet; the
// submission seam is armed for a later upload of the same binding on warm
// retained staging.
TEST_CASE("TTNN model loading: failed weight uploads publish nothing and retain staging") {
    require_hardware();
    TtnnDevices devices;

    const nlohmann::json config = iom_model_loading::two_layer_config();
    const std::vector<iom_model_loading::SafetensorsEntry> entries =
            iom_model_loading::required_weight_entries(config);
    const iom_model_loading::TempDir directory(
            iom_conformance::model_loading::fixture_tag(devices.conformance(),
                                                        "failed-upload"));
    const std::unique_ptr<iom::ModelSource> source =
            iom_conformance::model_loading::load_checkpoint(
                    directory.path(), config, entries);
    REQUIRE(source->weights().size() == entries.size());

    const std::size_t initial_allocations =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();

    // A failed retained staging allocation leaves no storage behind: the
    // failed attempt retains nothing, the documented `bad_alloc` reaches the
    // caller unchanged, and no destination of the aborted binding receives its
    // published role payload.
    {
        iom_conformance::model_loading::DestinationBinding binding =
                iom_conformance::model_loading::create_binding(
                        *devices.candidate, *source);
        iom::ttnn_test::
                fail_next_host_transfer_staging_allocation_for_testing();
        REQUIRE_THROWS_AS(
                (void)source->upload_weights(*devices.candidate,
                                             binding.span()),
                std::bad_alloc);
        CHECK(iom::ttnn_test::
                      host_transfer_staging_allocation_fault_consumed_for_testing());
        CHECK_EQ(
                iom::ttnn_test::
                        host_transfer_staging_allocation_count_for_testing(),
                initial_allocations);

        // Fresh TTNN storage content is unspecified by the contract, so the
        // negative observation of the aborted call is that no destination
        // carries its own published role payload and therefore no later role
        // of that call reached its destination.
        for (std::size_t index = 0; index < entries.size(); ++index) {
            CAPTURE(index);
            const std::string& payload = entries[index].payload;
            const std::span<const std::byte> published(
                    reinterpret_cast<const std::byte*>(payload.data()),
                    payload.size());
            CHECK(iom_conformance::first_logical_mismatch(
                          binding.owners[index]->view(), published)
                          .has_value());
        }

        // Only the next normal return publishes the weights. It reuses the one
        // retained upload slot every role of this binding fits, so the failed
        // allocation neither kept storage alive nor forced a replacement.
        const std::size_t before_realization =
                iom::ttnn_test::
                        host_transfer_staging_allocation_count_for_testing();
        source->upload_weights(*devices.candidate, binding.span());
        CHECK_EQ(
                iom::ttnn_test::
                        host_transfer_staging_allocation_count_for_testing(),
                before_realization + 1);
        iom_conformance::model_loading::require_realized_bytes(
                binding, entries, "publication after a failed allocation");
    }

    // Sentinel seeding uploads every destination of this binding and warms the
    // retained upload staging the later upload reuses, so the interrupted
    // transfer below holds acquired staging when it fails.
    iom_conformance::model_loading::DestinationBinding later =
            iom_conformance::model_loading::create_binding(*devices.candidate,
                                                           *source);
    const std::vector<std::vector<std::byte>> sentinels =
            iom_conformance::model_loading::seed_sentinels(later);
    const std::size_t warm_allocations =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();

    iom::ttnn_test::fail_next_host_transfer_submission_for_testing(0);
    REQUIRE_THROWS_AS(
            (void)source->upload_weights(*devices.candidate, later.span()),
            std::runtime_error);
    CHECK(iom::ttnn_test::
                  host_transfer_submission_fault_consumed_for_testing());

    // The complete ordered binding stopped at its first failed role: every
    // destination still holds the sentinel of the earlier successful upload,
    // so no later role of the aborted call was uploaded and that call
    // published nothing.
    iom_conformance::model_loading::require_sentinels(
            later, sentinels, "failed later weight upload");

    // The bounded recovery drain finished the work the failed transfer had
    // already submitted, so the staging it held returned to the facility
    // instead of being freed or replaced while it could still be read.
    CHECK_EQ(
            iom::ttnn_test::host_transfer_staging_allocation_count_for_testing(),
            warm_allocations);

    // A later normal return is the publication permission, and it reuses the
    // same retained staging the failed upload held.
    source->upload_weights(*devices.candidate, later.span());
    CHECK_EQ(
            iom::ttnn_test::host_transfer_staging_allocation_count_for_testing(),
            warm_allocations);
    iom_conformance::model_loading::require_realized_bytes(
            later, entries, "publication after a failed later upload");
}

// Opt-in real-checkpoint loading, compiled only with
// `IOM_TEST_REAL_MODEL_LOADING=ON`: the ordinary TTNN suite reads no model
// directory and downloads nothing. TTNN owns native per-plane storage and takes
// no caller arena, so this case is the existing `TtnnDevices` fixture - one
// live TTNN context on the selected device, with the CPU reference and foreign
// slots this driver already establishes - and the shared case loads the caller's
// official checkpoint through it. BF16 remains mandatory: a device that cannot
// host the published BF16 roles fails here rather than skipping.
#ifdef IOM_TEST_REAL_MODEL_LOADING
TEST_CASE("TTNN real model loading") {
    require_hardware();
    TtnnDevices devices;
    const std::span<const iom::DataType> supported =
            devices.candidate->supported_data_types();
    REQUIRE(std::find(supported.begin(), supported.end(),
                      iom::DataType::BF16) != supported.end());
    iom_conformance::run_real_model_loading(*devices.candidate);
}
#endif
