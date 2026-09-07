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
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <stdexcept>
#include <vector>

#include "../../src/ttnn/staging.hpp"
#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_other.hpp"
#include "iom/cpu/device.hpp"
#include "iom/ttnn/device.hpp"
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
// standard model: logical elements at their row-major padded positions and
// zero padding, optionally planting nonzero sentinels at padded coordinates.
// Uses only the standard slot arithmetic, independent of the production TTNN
// copy helper, so the seed path and the padding-mutation probe share one
// geometry.
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
    REQUIRE_EQ(bits % 8, std::size_t{0});
    const std::size_t element_bytes = bits / 8;
    std::vector<std::byte> padded(
            padded_rows * padded_columns * element_bytes, std::byte{0});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t slot = iom::detail::standard_plane_slot(
                    owner, plane_index, row, column);
            std::memcpy(
                    padded.data()
                            + (row * padded_columns + column) * element_bytes,
                    encoded.data() + slot * element_bytes, element_bytes);
        }
    }
    for (const auto& [row, column] : planted) {
        const bool planted_inside_logical =
                row < rows && column < columns;
        REQUIRE_FALSE(planted_inside_logical);
        REQUIRE(row < padded_rows);
        REQUIRE(column < padded_columns);
        std::byte* cell =
                padded.data() + (row * padded_columns + column) * element_bytes;
        std::fill(cell, cell + element_bytes, kObserverStorageSentinel);
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

// Observes one native plane's complete padded TILE readback into standard
// slot order, mapping every standard padded coordinate — padded rows and
// columns included — by its standard slot. The readback is raw physical
// tile-major bytes; coordinates are mapped through the native tile geometry,
// never by assuming the native and standard layouts agree, so native padding
// writes and physical tile-slot permutations surface instead of being
// normalized away.
void observe_plane_storage(
        std::span<const std::byte> readback,
        const ttnn::Tensor& plane, const iom::TensorSpec& owner,
        std::size_t plane_index, std::size_t element_bytes,
        std::vector<std::byte>& storage) {
    const TtnnPlaneLayout native(plane);
    REQUIRE_EQ(
            readback.size(),
            native.padded_rows * native.padded_columns * element_bytes);
    const iom::TensorShape padded_shape = owner.standard_padded_shape();
    const std::span<const std::size_t> padded = padded_shape.dimensions();
    const std::size_t padded_rows = padded[padded.size() - 2];
    const std::size_t padded_columns = padded[padded.size() - 1];
    for (std::size_t row = 0; row < padded_rows; ++row) {
        for (std::size_t column = 0; column < padded_columns; ++column) {
            const std::size_t slot = iom::detail::standard_plane_slot(
                    owner, plane_index, row, column);
            const std::size_t position = native.element_index(row, column);
            std::memcpy(
                    storage.data() + slot * element_bytes,
                    readback.data() + position * element_bytes,
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
        const std::size_t bits = iom::detail::leaf_bits(owner.data_type);
        REQUIRE_EQ(bits % 8, std::size_t{0});
        const std::size_t element_bytes = bits / 8;

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
                    native.padded_rows * native.padded_columns * element_bytes;
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

        offset = 0;
        for (std::size_t i = 0; i < planes_count; ++i) {
            observe_plane_storage(
                    std::span<const std::byte>(
                            staging.get() + offset, plane_bytes[i]),
                    planes[i], owner, i, element_bytes, storage);
            offset += plane_bytes[i];
        }
        return storage;
    }
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

// Decisive padded-storage probe: the full-storage oracle observes the
// complete native padded allocation, so a nonzero native padding write fails
// it while the logical host projection stays byte-identical, and a logical
// host write re-establishes the documented zero policy.
TEST_CASE("TTNN conformance: full-storage oracle exposes native padding mutations") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{17, 33}}, iom::DataType::U8};
    auto tensor = devices.candidate->create_tensor(spec);
    TtnnStorageOracle oracle;
    oracle.set_owner_spec(spec);
    iom::TensorView& view = tensor->view();
    auto* planes = static_cast<ttnn::Tensor*>(view.native_handle());
    const std::vector<std::byte> initial =
            iom_conformance::encode_standard_tiled_storage(spec);

    // Plant a nonzero sentinel in the first padded row's leading cell through
    // the independent padded native write path, preserving every logical
    // value from the encoded model.
    const std::size_t planted_row = spec.shape.dimension(0);
    const std::size_t planted_column = 0;
    const std::pair<std::size_t, std::size_t> planted_cells[] = {
            {planted_row, planted_column}};
    std::vector<std::byte> mutated = padded_plane_image(
            planes[0], spec, 0, initial,
            std::span<const std::pair<std::size_t, std::size_t>>{
                    planted_cells});
    write_padded_plane_image(planes[0], mutated);

    // A padded-cell mutation fails full-storage observation at exactly the
    // planted standard slot; the logical projection is unchanged.
    const std::vector<std::byte> observed = oracle.observe(view);
    CHECK_FALSE(iom_conformance::require_storage_oracle_bytes(
            observed, initial, "planted padding mutation", false));
    std::size_t first_mismatch = observed.size();
    std::size_t mismatch_count = 0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] != initial[i]) {
            if (mismatch_count == 0) {
                first_mismatch = i;
            }
            ++mismatch_count;
        }
    }
    const std::size_t planted_slot = iom::detail::standard_plane_slot(
            spec, 0, planted_row, planted_column);
    CHECK_EQ(mismatch_count, 1);
    CHECK_EQ(first_mismatch, planted_slot);
    iom_conformance::require_logical_bytes(
            view, iom_conformance::decode_standard_tiled_view(view, spec, initial),
            "logical projection after native padding mutation");

    // A logical host write goes through the production upload, which
    // zero-fills native padding: full-storage observation now matches the
    // expected untouched/zero policy exactly.
    const std::vector<std::byte> pattern = iom_conformance::encode_logical(spec, 0x2A);
    view.copy_from_host(pattern);
    std::vector<std::byte> expected = initial;
    iom_conformance::apply_standard_tiled_view(view, spec, pattern, expected);
    REQUIRE(iom_conformance::require_storage_oracle_bytes(
            oracle.observe(view), expected,
            "logical write re-establishes zero padding", true));
    iom_conformance::require_logical_bytes(view, pattern, "logical write content");
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
    tensor->view().copy_from_host(seed);
    iom_conformance::require_logical_bytes(
            tensor->view(), seed, "warm-up readback");
    const std::size_t warmup_allocations =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();

    // Repeated same-or-smaller transfers allocate no fresh host staging.
    for (std::uint64_t salt = 0x60; salt < 0x64; ++salt) {
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(spec, salt);
        tensor->view().copy_from_host(pattern);
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
    big->view().copy_from_host(large_seed);
    iom_conformance::require_logical_bytes(
            big->view(), large_seed, "grown readback");
    CHECK_EQ(
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing(),
            warmup_allocations + 7);
    for (std::uint64_t salt = 0x72; salt < 0x75; ++salt) {
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(larger, salt);
        big->view().copy_from_host(pattern);
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
    tensor->view().copy_from_host(seed);
    iom_conformance::require_logical_bytes(
            tensor->view(), seed, "warm-up");

    // An upload submission fault at plane 1: plane 0 reached the mesh, the
    // bounded recovery finish succeeds, and all leases become reusable only
    // after that completion proof. The original fault still propagates.
    iom::ttnn_test::fail_next_host_transfer_submission_for_testing(1);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0x82);
    REQUIRE_THROWS_AS(
            tensor->view().copy_from_host(pattern), std::runtime_error);
    CHECK(iom::ttnn_test::
                  host_transfer_submission_fault_consumed_for_testing());
    const std::size_t after_upload_failure =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();

    const std::vector<std::byte> pattern_2 =
            iom_conformance::encode_logical(spec, 0x83);
    tensor->view().copy_from_host(pattern_2);
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
            tensor->view().copy_to_host(readback), std::runtime_error);
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
            big->view().copy_from_host(large_seed), std::bad_alloc);
    CHECK(
            iom::ttnn_test::
                    host_transfer_staging_allocation_fault_consumed_for_testing());
    const std::size_t after_allocation_failure =
            iom::ttnn_test::
                    host_transfer_staging_allocation_count_for_testing();

    big->view().copy_from_host(large_seed);
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

TEST_CASE("TTNN conformance: compute methods reject capability without submitting") {
    require_hardware();
    TtnnDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, iom::ttnn_supported_data_types(), nullptr,
            "TTNN");
}

TEST_CASE("TTNN conformance: full shared suite") {
    require_hardware();
    TtnnDevices devices;
    const std::span<const iom::DataType> supported =
            iom::ttnn_supported_data_types();
    TtnnStorageOracle oracle;
    iom_conformance::run_backend_conformance(
            devices.conformance(), supported.subspan(0, 1), nullptr, &oracle);
}

TEST_CASE("TTNN quarantine action allocation failure leaks native storage") {
    require_hardware();
    const iom::TensorSpec spec{
            iom::TensorShape{{16, 16}}, iom::DataType::BF16};
    auto device = iom::make_ttnn_device(0);
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    auto queue = device->create_ops();
    REQUIRE_NOTHROW(queue->copy(source->view(), destination->view()));

    {
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
    t->view().copy_from_host(pattern);

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
    fresh_source->view().copy_from_host(pattern);
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
        source->view().copy_from_host(
                iom_conformance::encode_logical(spec, 21));

        auto queue = devices.candidate->create_ops();
        iom::ttnn_test::fail_next_copy_planes_submission_for_testing(
                fail_plane);
        REQUIRE_THROWS_AS(
                queue->copy(source->view(), destination->view()),
                std::runtime_error);
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
        source->view().copy_from_host(pattern);
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
        REQUIRE_THROWS_AS(
                queue->copy(source->view(), destination->view()),
                std::bad_alloc);
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
            REQUIRE_THROWS_AS(
                    queue->copy(source->view(), destination->view()),
                    std::bad_alloc);
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

TEST_CASE("TTNN un-drainable failed submission reports a repeatable failed token") {
    require_hardware();
    TtnnDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 16, 16}}, iom::DataType::F32};
    auto source = devices.candidate->create_tensor(spec);
    auto destination = devices.candidate->create_tensor(spec);
    source->view().copy_from_host(
            iom_conformance::encode_logical(spec, 31));

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
    source->view().copy_from_host(
            iom_conformance::encode_logical(spec, 41));

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
        source->view().copy_from_host(pattern);
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
        source->view().copy_from_host(pattern);
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
