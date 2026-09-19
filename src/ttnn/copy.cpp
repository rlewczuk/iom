#include "copy.hpp"
#include "device_internal.hpp"
#include "registry_state.hpp"
#include "staging.hpp"
#include "testing_internal.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>
#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
namespace iom::ttnn_detail {

    std::size_t carrier_bytes(tt::tt_metal::DataType type) {
        switch (type) {
            case tt::tt_metal::DataType::UINT8: return 1;
            case tt::tt_metal::DataType::UINT16:
            case tt::tt_metal::DataType::BFLOAT16: return 2;
            case tt::tt_metal::DataType::FLOAT32:
            case tt::tt_metal::DataType::INT32:
            case tt::tt_metal::DataType::UINT32: return 4;
            default:
                throw std::invalid_argument(
                        "TTNN native dtype has no byte carrier");
        }
    }

    namespace {
        template <typename Value>
        [[nodiscard]] std::size_t checked_plane_dimension(
                Value value, const char* what) {
            using Raw = std::remove_cv_t<Value>;
            if constexpr (std::is_signed_v<Raw>) {
                if (value <= 0) {
                    throw std::invalid_argument(
                            std::string("TTNN plane ") + what
                            + " is non-positive");
                }
            } else if (value == 0) {
                throw std::invalid_argument(
                        std::string("TTNN plane ") + what
                        + " is zero");
            }
            using Unsigned = std::make_unsigned_t<Raw>;
            const std::uintmax_t widened =
                    static_cast<std::uintmax_t>(
                            static_cast<Unsigned>(value));
            if (widened
                    > static_cast<std::uintmax_t>(
                              std::numeric_limits<std::size_t>::max())) {
                throw std::overflow_error(
                        std::string("TTNN plane ") + what
                        + " exceeds size_t");
            }
            return static_cast<std::size_t>(widened);
        }

        [[nodiscard]] std::size_t checked_product(
                std::size_t lhs, std::size_t rhs, const char* what) {
            if (lhs != 0
                    && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
                throw std::overflow_error(
                        std::string("TTNN ") + what + " overflows");
            }
            return lhs * rhs;
        }
    }  // namespace

    std::size_t padded_plane_bytes(const ttnn::Tensor& plane) {
        if (plane.layout() != tt::tt_metal::Layout::TILE) {
            throw std::invalid_argument(
                    "TTNN raw plane must use the TILE layout");
        }
        const auto padded = plane.padded_shape();
        const std::size_t rows = checked_plane_dimension(
                padded[-2], "padded row count");
        const std::size_t columns = checked_plane_dimension(
                padded[-1], "padded column count");
        if (rows % 32 != 0 || columns % 32 != 0) {
            throw std::invalid_argument(
                    "TTNN plane is not a complete TILE image");
        }
        const std::size_t elements =
                checked_product(rows, columns, "padded plane element count");
        return checked_product(
                elements, carrier_bytes(plane.dtype()),
                "padded plane byte count");
    }

    TtnnWorkspaceLease::TtnnWorkspaceLease(
            TtnnDevice& device, std::byte* data, std::size_t bytes,
            std::unique_ptr<HostWorkspaceLeasePayload> payload)
            : device_(&device), data_(data), bytes_(bytes),
              payload_(std::move(payload)) {}

    TtnnWorkspaceLease::~TtnnWorkspaceLease() noexcept {
        complete(false);
    }

    TtnnWorkspaceLease::TtnnWorkspaceLease(
            TtnnWorkspaceLease&& other) noexcept
            : device_(std::exchange(other.device_, nullptr)),
              data_(std::exchange(other.data_, nullptr)),
              bytes_(std::exchange(other.bytes_, 0)),
              payload_(std::move(other.payload_)) {}

    TtnnWorkspaceLease& TtnnWorkspaceLease::operator=(
            TtnnWorkspaceLease&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        complete(false);
        device_ = std::exchange(other.device_, nullptr);
        data_ = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        payload_ = std::move(other.payload_);
        return *this;
    }

    std::shared_ptr<void> TtnnWorkspaceLease::keepalive() const noexcept {
        return payload_ == nullptr ? std::shared_ptr<void>{}
                                    : payload_->keepalive;
    }

    void TtnnWorkspaceLease::complete(bool covering_proof) noexcept {
        if (payload_ == nullptr) {
            device_ = nullptr;
            data_ = nullptr;
            bytes_ = 0;
            return;
        }
        if (covering_proof) {
            payload_.reset();
        } else if (device_ != nullptr) {
            device_->retain_host_workspace(std::move(payload_));
        } else {
            // No device remains to own a quarantine record.  Retain the
            // payload permanently rather than releasing unproved bytes.
            (void)payload_.release();
        }
        device_ = nullptr;
        data_ = nullptr;
        bytes_ = 0;
    }

    TtnnWorkspaceLease acquire_workspace_lease(
            TtnnDevice& device, const RawWorkspaceView& workspace) {
        const HostWorkspace checked =
                checked_host_workspace(device, workspace);
        auto payload = std::make_unique<HostWorkspaceLeasePayload>();
        payload->keepalive = checked.keepalive;
        payload->pin.emplace(payload->keepalive);
        return TtnnWorkspaceLease{
                device, checked.data, checked.byte_size, std::move(payload)};
    }

    namespace {
        void validate_raw_range(
                const ttnn::Tensor& plane,
                const TtnnWorkspaceLease& workspace,
                std::size_t& bytes) {
            bytes = padded_plane_bytes(plane);
            if (workspace.empty() || workspace.data() == nullptr) {
                throw std::invalid_argument(
                        "TTNN raw transfer workspace is empty");
            }
            if (bytes > workspace.byte_size()) {
                throw std::invalid_argument(
                        "TTNN raw transfer workspace is smaller than plane");
            }
            const std::uintptr_t address =
                    reinterpret_cast<std::uintptr_t>(workspace.data());
            if (address == 0 || address % 32 != 0
                    || bytes
                            > std::numeric_limits<std::uintptr_t>::max()
                                    - address) {
                throw std::invalid_argument(
                        "TTNN raw transfer workspace range is invalid");
            }
        }

        template <typename T>
        void raw_upload_typed(
                ttnn::Tensor& plane, TtnnWorkspaceLease& workspace,
                std::size_t bytes) {
            if (bytes % sizeof(T) != 0) {
                throw std::logic_error(
                        "TTNN raw plane bytes do not match carrier width");
            }
            tt::tt_metal::HostBuffer host_buffer(
                    ttsl::Span<T>(
                            reinterpret_cast<T*>(workspace.data()),
                            bytes / sizeof(T)),
                    tt::tt_metal::MemoryPin(workspace.keepalive()));
            ttnn::Tensor host_tiled(
                    std::move(host_buffer), plane.logical_shape(),
                    plane.padded_shape(), plane.dtype(),
                    tt::tt_metal::Layout::TILE);
#ifdef IOM_ENABLE_TESTING
            ::iom::ttnn_detail::fail_host_transfer_submission_at(0);
#endif
            ttnn::copy_to_device(host_tiled, plane);
        }
    }  // namespace

    void raw_download_plane(
            TtnnDevice& device, const ttnn::Tensor& plane,
            TtnnWorkspaceLease& workspace) {
        std::size_t bytes = 0;
        validate_raw_range(plane, workspace, bytes);
        (void)bytes;
#ifdef IOM_ENABLE_TESTING
        ::iom::ttnn_detail::fail_host_transfer_submission_at(0);
#endif
        ttnn::copy_to_host(
                device.mesh().mesh_command_queue(0), plane, workspace.data(),
                std::nullopt, /*blocking=*/false);
    }

    void raw_upload_plane(
            TtnnDevice& device, ttnn::Tensor& plane,
            TtnnWorkspaceLease& workspace) {
        (void)device;
        std::size_t bytes = 0;
        validate_raw_range(plane, workspace, bytes);
        switch (plane.dtype()) {
            case tt::tt_metal::DataType::BFLOAT16:
                raw_upload_typed<bfloat16>(plane, workspace, bytes);
                break;
            case tt::tt_metal::DataType::FLOAT32:
                raw_upload_typed<float>(plane, workspace, bytes);
                break;
            case tt::tt_metal::DataType::UINT32:
                raw_upload_typed<std::uint32_t>(plane, workspace, bytes);
                break;
            case tt::tt_metal::DataType::INT32:
                raw_upload_typed<std::int32_t>(plane, workspace, bytes);
                break;
            case tt::tt_metal::DataType::UINT16:
                raw_upload_typed<std::uint16_t>(plane, workspace, bytes);
                break;
            case tt::tt_metal::DataType::UINT8:
                raw_upload_typed<std::uint8_t>(plane, workspace, bytes);
                break;
            default:
                throw std::invalid_argument(
                        "TTNN raw upload has no typed carrier");
        }
    }

    std::size_t upload_slot_index(tt::tt_metal::DataType type) {
        switch (type) {
            case tt::tt_metal::DataType::BFLOAT16: return 0;
            case tt::tt_metal::DataType::FLOAT32: return 1;
            case tt::tt_metal::DataType::UINT32: return 2;
            case tt::tt_metal::DataType::INT32: return 3;
            case tt::tt_metal::DataType::UINT16: return 4;
            case tt::tt_metal::DataType::UINT8: return 5;
            default:
                throw std::logic_error(
                        "TTNN native dtype has no upload staging slot");
        }
    }

    // Physical tile-major element index of the padded coordinate (row,
    // column): tiles row-major over the padded grid, every 32x32 tile as
    // four row-major 16x16 faces.
    std::size_t padded_cell_index(
            std::size_t row, std::size_t column,
            std::size_t num_tile_cols) {
        const std::size_t tile_index =
                (row / 32) * num_tile_cols + (column / 32);
        const std::size_t face_index =
                ((row % 32) / 16) * 2 + ((column % 32) / 16);
        return tile_index * 1024 + face_index * 256
                + (row % 16) * 16 + (column % 16);
    }
    namespace {

        // Number of logical planes a view addresses: the product of its
        // leading dimensions.
        std::size_t view_plane_count(const TensorView& view) {
            const std::span<const std::size_t> dimensions =
                    view.spec().shape.dimensions();
            std::size_t planes = 1;
            for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                planes *= dimensions[i];
            }
            return planes;
        }

        // Owner plane of the view's linear plane index, decomposed row-major
        // over the view's leading dimensions and mapped through the view's
        // plane offset and strides. View transforms keep every addressed
        // owner plane in bounds.
        std::size_t owner_plane_at(const TensorView& view, std::size_t index) {
            const std::span<const std::size_t> dimensions =
                    view.spec().shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;
            const std::span<const std::size_t> strides =
                    view.plane_strides();
            std::size_t plane = view.plane_offset();
            for (std::size_t k = leading_rank; k-- > 0;) {
                plane += (index % dimensions[k]) * strides[k];
                index /= dimensions[k];
            }
            return plane;
        }
        std::size_t snapshot_plane_count_impl(const CopySnapshot& view) {
            const std::span<const std::size_t> dimensions =
                    view.spec.shape.dimensions();
            std::size_t planes = 1;
            for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                planes *= dimensions[i];
            }
            return planes;
        }

        std::size_t snapshot_owner_plane_at_impl(
                const CopySnapshot& view, std::size_t index) {
            const std::span<const std::size_t> dimensions =
                    view.spec.shape.dimensions();
            const std::size_t leading_rank = dimensions.size() - 2;
            std::size_t plane = view.plane_offset;
            for (std::size_t k = leading_rank; k-- > 0;) {
                plane += (index % dimensions[k]) * view.plane_strides[k];
                index /= dimensions[k];
            }
            return plane;
        }



        // Native staging uses whole carrier cells; public logical encodings
        // may be packed and are converted by the transfer helpers below.
        std::size_t element_bytes(DataType type) {
            const std::size_t bits = detail::leaf_bits(type);
            return (bits + 7) / 8;
        }


        // Uploads logical standard bytes into a native or UINT32 carrier
        // plane. Sub-byte values are unpacked from the logical bitstream and
        // 64-bit values occupy two little-endian carrier cells.
        void upload_plane(
                ttnn::Tensor& plane,
                TtnnHostStaging::UploadLease& lease,
                const std::byte* source, std::size_t rows,
                std::size_t columns, std::size_t bits,
                std::size_t source_bit_base, std::size_t plane_index) {
            const std::size_t carrier_bytes =
                    plane.dtype() == tt::tt_metal::DataType::UINT8 ? 1
                    : plane.dtype() == tt::tt_metal::DataType::UINT16 ? 2
                    : plane.dtype() == tt::tt_metal::DataType::BFLOAT16 ? 2
                    : plane.dtype() == tt::tt_metal::DataType::FLOAT32 ? 4
                    : 4;
            const std::size_t factor = bits > 32 ? 2 : 1;
            const std::size_t padded_rows =
                    static_cast<std::size_t>(plane.padded_shape()[-2]);
            const std::size_t padded_columns =
                    static_cast<std::size_t>(plane.padded_shape()[-1]);
            const std::size_t num_tile_cols = padded_columns / 32;
            const std::size_t padded_elements = padded_rows * padded_columns;
            std::byte* buffer = lease.data();
            std::memset(buffer, 0, padded_elements * carrier_bytes);
            const auto value_at = [&](std::size_t index) {
                std::uint64_t value = 0;
                for (std::size_t bit = 0; bit < bits; ++bit) {
                    const std::size_t source_bit =
                            source_bit_base + index * bits + bit;
                    if ((std::to_integer<unsigned char>(
                                source[source_bit / 8])
                         >> (source_bit % 8))
                        & 1u) {
                        value |= std::uint64_t{1} << bit;
                    }
                }
                return value;
            };
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t column = 0; column < columns; ++column) {
                    const std::uint64_t value =
                            value_at(row * columns + column);
                    for (std::size_t part = 0; part < factor; ++part) {
                        const std::size_t native_column =
                                column * factor + part;
                        const std::size_t index = padded_cell_index(
                                row, native_column, num_tile_cols);
                        std::memcpy(
                                buffer + index * carrier_bytes,
                                reinterpret_cast<const std::byte*>(&value)
                                        + part * carrier_bytes,
                                carrier_bytes);
                    }
                }
            }

            auto upload_typed = [&]<typename T>() {
                tt::tt_metal::HostBuffer host_buffer(
                        ttsl::Span<T>(
                                reinterpret_cast<T*>(buffer),
                                padded_elements),
                        tt::tt_metal::MemoryPin(lease.keepalive()));
                ttnn::Tensor host_tiled(
                        std::move(host_buffer), plane.logical_shape(),
                        plane.padded_shape(), plane.dtype(),
                        tt::tt_metal::Layout::TILE);
#ifdef IOM_ENABLE_TESTING
                ::iom::ttnn_detail::fail_host_transfer_submission_at(plane_index);
#endif
                ttnn::copy_to_device(host_tiled, plane);
            };
            switch (plane.dtype()) {
                case tt::tt_metal::DataType::BFLOAT16:
                    upload_typed.template operator()<bfloat16>(); break;
                case tt::tt_metal::DataType::FLOAT32:
                    upload_typed.template operator()<float>(); break;
                case tt::tt_metal::DataType::UINT32:
                    upload_typed.template operator()<std::uint32_t>(); break;
                case tt::tt_metal::DataType::UINT16:
                    upload_typed.template operator()<std::uint16_t>(); break;
                case tt::tt_metal::DataType::UINT8:
                    upload_typed.template operator()<std::uint8_t>(); break;
                default:
                    throw std::logic_error(
                            "TTNN native dtype has no host element type");
            }
        }

        // Enqueues one plane's padded tile-major bytes into a caller-owned
        // staging window without waiting for the device read to complete.
        void submit_download_plane(
                tt::tt_metal::distributed::MeshCommandQueue& queue,
                const ttnn::Tensor& plane, std::byte* staging,
                std::size_t plane_index) {
#ifdef IOM_ENABLE_TESTING
            ::iom::ttnn_detail::fail_host_transfer_submission_at(plane_index);
#endif
            ttnn::copy_to_host(
                    queue, plane, staging, std::nullopt, /*blocking=*/false);
        }

        void assemble_download_plane(
                const std::byte* staging, std::byte* destination,
                std::size_t rows, std::size_t columns, std::size_t bits,
                std::size_t carrier_size, std::size_t factor,
                std::size_t destination_bit_base,
                std::size_t num_tile_cols) {
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t column = 0; column < columns; ++column) {
                    std::uint64_t value = 0;
                    for (std::size_t part = 0; part < factor; ++part) {
                        const std::size_t index = padded_cell_index(
                                row, column * factor + part, num_tile_cols);
                        std::uint32_t carrier = 0;
                        std::memcpy(&carrier,
                                    staging + index * carrier_size,
                                    carrier_size);
                        value |= std::uint64_t{carrier} << (part * 32);
                    }
                    const std::size_t element = row * columns + column;
                    for (std::size_t bit = 0; bit < bits; ++bit) {
                        if ((value >> bit) & 1u) {
                            const std::size_t output_bit =
                                    destination_bit_base + element * bits + bit;
                            destination[output_bit / 8] |=
                                    std::byte{static_cast<unsigned char>(
                                            1u << (output_bit % 8))};
                        }
                    }
                }
            }
        }

        std::size_t view_rows(const TensorView& view) {
            return view.spec().shape.dimension(
                    view.spec().shape.rank() - 2);
        }

        std::size_t view_columns(const TensorView& view) {
            return view.spec().shape.dimension(
                    view.spec().shape.rank() - 1);
        }
    }  // namespace
    std::size_t snapshot_plane_count(const CopySnapshot& view) {
        return snapshot_plane_count_impl(view);
    }

    std::size_t snapshot_owner_plane_at(
            const CopySnapshot& view, std::size_t index) {
        return snapshot_owner_plane_at_impl(view, index);
    }
    std::size_t snapshot_plane_count(const TensorSpec& spec) {
        const std::span<const std::size_t> dimensions =
                spec.shape.dimensions();
        std::size_t planes = 1;
        for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
            planes *= dimensions[i];
        }
        return planes;
    }

    std::size_t snapshot_owner_plane_at(
            const TensorSpec& spec, std::size_t plane_offset,
            std::span<const std::size_t> plane_strides,
            std::size_t index) {
        const std::span<const std::size_t> dimensions =
                spec.shape.dimensions();
        const std::size_t leading_rank = dimensions.size() - 2;
        std::size_t plane = plane_offset;
        for (std::size_t k = leading_rank; k-- > 0;) {
            plane += (index % dimensions[k]) * plane_strides[k];
            index /= dimensions[k];
        }
        return plane;
    }

    void region_from_host(
            tt::tt_metal::distributed::MeshDevice& device,
            TtnnHostStaging& staging, const TensorView& destination,
            ttnn::Tensor* planes, std::span<const std::byte> source) {
        const std::size_t rows = view_rows(destination);
        const std::size_t columns = view_columns(destination);
        const std::size_t bits =
                detail::leaf_bits(destination.spec().data_type);
        const std::size_t plane_bytes = (rows * columns * bits + 7) / 8;
        const std::size_t count = view_plane_count(destination);
        const ttnn::Tensor& first = planes[owner_plane_at(destination, 0)];
        const std::size_t carrier_size = carrier_bytes(first.dtype());
        const std::size_t padded_rows =
                static_cast<std::size_t>(first.padded_shape()[-2]);
        const std::size_t padded_columns =
                static_cast<std::size_t>(first.padded_shape()[-1]);
        const std::size_t staging_bytes =
                padded_rows * padded_columns * carrier_size;
        std::vector<TtnnHostStaging::UploadLease> leases;
        leases.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            leases.emplace_back(staging.acquire_upload(
                    upload_slot_index(first.dtype()), staging_bytes));
        }
        std::size_t submitted = 0;
        bool complete = false;
        try {
            for (std::size_t index = 0; index < count; ++index) {
                ++submitted;
                upload_plane(planes[owner_plane_at(destination, index)],
                             leases[index], source.data(), rows, columns, bits,
                             index * rows * columns * bits, index);
            }
            complete = true;
            device.mesh_command_queue(0).finish();
            for (auto& lease : leases) lease.release();
            staging.reclaim_retired_uploads();
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            bool drained = false;
            if (submitted != 0 && !complete) {
                try {
                    device.mesh_command_queue(0).finish();
                    drained = true;
                } catch (...) {
                }
            }
            for (auto& lease : leases) {
                if (drained) lease.release(); else lease.retire();
            }
            if (drained) staging.reclaim_retired_uploads();
            std::rethrow_exception(failure);
        }
    }

    void region_to_host(
            tt::tt_metal::distributed::MeshDevice& device,
            TtnnHostStaging& staging, const TensorView& source,
            const ttnn::Tensor* planes, std::span<std::byte> destination) {
        const std::size_t rows = view_rows(source);
        const std::size_t columns = view_columns(source);
        const std::size_t bits = detail::leaf_bits(source.spec().data_type);
        const std::size_t plane_bytes = (rows * columns * bits + 7) / 8;
        const std::size_t count = view_plane_count(source);
        const ttnn::Tensor& first = planes[owner_plane_at(source, 0)];
        const std::size_t carrier_size = carrier_bytes(first.dtype());
        const std::size_t padded_rows =
                static_cast<std::size_t>(first.padded_shape()[-2]);
        const std::size_t padded_columns =
                static_cast<std::size_t>(first.padded_shape()[-1]);
        const std::size_t padded_plane_bytes =
                padded_rows * padded_columns * carrier_size;
        auto& queue = device.mesh_command_queue(0);
        if (staging.download_retired()) {
            queue.finish();
            staging.reclaim_download();
        }
        std::memset(
                destination.data(), 0,
                (count * rows * columns * bits + 7) / 8);
        TtnnHostStaging::DownloadLease lease =
                staging.acquire_download(count * padded_plane_bytes);
        try {
            for (std::size_t index = 0; index < count; ++index) {
                submit_download_plane(
                        queue, planes[owner_plane_at(source, index)],
                        lease.data() + index * padded_plane_bytes, index);
            }
            queue.finish();
            const std::size_t factor = bits > 32 ? 2 : 1;
            const std::size_t num_tile_cols = padded_columns / 32;
            for (std::size_t index = 0; index < count; ++index) {
                assemble_download_plane(
                        lease.data() + index * padded_plane_bytes,
                        destination.data(), rows, columns, bits, carrier_size,
                        factor, index * rows * columns * bits, num_tile_cols);
            }
            lease.release();
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            bool drained = false;
            try {
                queue.finish();
                drained = true;
            } catch (...) {
            }
            if (drained) lease.discard(); else lease.retire();
            std::rethrow_exception(failure);
        }
    }
    void copy_planes(
            const CopySnapshot& source, const ttnn::Tensor* source_planes,
            const CopySnapshot& destination,
            ttnn::Tensor* destination_planes,
            bool& any_submitted) {
        any_submitted = false;
        const std::size_t count = snapshot_plane_count(source);
        for (std::size_t index = 0; index < count; ++index) {
#ifdef IOM_ENABLE_TESTING
            ::iom::ttnn_detail::fail_copy_planes_submission_at(index);
#endif
            ttnn::copy(
                    source_planes[snapshot_owner_plane_at(source, index)],
                    destination_planes[
                            snapshot_owner_plane_at(destination, index)]);
            any_submitted = true;
        }
    }


}  // namespace iom::ttnn_detail

