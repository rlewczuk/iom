#include "copy.hpp"
#include "registry_state.hpp"
#include "staging.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>
#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/tensor/tensor_ops.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

#ifdef IOM_ENABLE_TESTING
    // Test seam for submission-failure injection: the next copy_planes
    // call throws just before the chosen plane index is submitted, so
    // planes below it have already reached the mesh. A fault at index
    // zero fails before any submission. The fault fires exactly once at
    // the armed index; copy_planes runs under the device API mutex, so
    // arming and consumption never race with another plane loop.
    std::atomic<bool> g_copy_planes_fault_armed{false};
    std::atomic<std::size_t> g_copy_planes_fault_at{0};
    std::atomic<bool> g_copy_planes_fault_consumed{false};

    void fail_copy_planes_submission_at(std::size_t index) noexcept(false) {
        if (g_copy_planes_fault_armed.load(std::memory_order_acquire)
                && index
                        == g_copy_planes_fault_at.load(
                                std::memory_order_acquire)) {
            g_copy_planes_fault_armed.store(
                    false, std::memory_order_release);
            g_copy_planes_fault_consumed.store(
                    true, std::memory_order_release);
            throw std::runtime_error(
                    "injected TTNN copy-plane submission failure");
        }
    }

    // Test seam for host-transfer submission-failure injection: the next
    // plane submission of a region_from_host or region_to_host call throws
    // just before the chosen plane index reaches the mesh. Planes below it
    // have already been submitted (and, for downloads, enqueued). The fault
    // fires exactly once at the armed index; host transfers run under the
    // device API mutex, so arming and consumption never race.
    std::atomic<bool> g_host_transfer_fault_armed{false};
    std::atomic<std::size_t> g_host_transfer_fault_at{0};
    std::atomic<bool> g_host_transfer_fault_consumed{false};

    void fail_host_transfer_submission_at(std::size_t index) noexcept(false) {
        if (g_host_transfer_fault_armed.load(std::memory_order_acquire)
                && index
                        == g_host_transfer_fault_at.load(
                                std::memory_order_acquire)) {
            g_host_transfer_fault_armed.store(
                    false, std::memory_order_release);
            g_host_transfer_fault_consumed.store(
                    true, std::memory_order_release);
            throw std::runtime_error(
                    "injected TTNN host-transfer submission failure");
        }
    }
#endif

}  // namespace

namespace iom::ttnn_detail {

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

        // Every supported leaf type is byte-aligned, so one logical element
        // is one whole number of bytes.
        std::size_t element_bytes(DataType type) {
            const std::size_t bits = detail::leaf_bits(type);
            if (bits % 8 != 0) {
                throw std::logic_error(
                        "TTNN supported type is not byte-aligned");
            }
            return bits / 8;
        }

        // Maps a supported native dtype to its retained upload staging slot
        // list in TtnnHostStaging, mirroring the upload_typed dispatch below.
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

        // Uploads one plane's logical row-major bytes into the plane's
        // TTNN-native tiled tensor, zero-filling native padding. The buffer
        // is the caller-held retained staging slot: only the padding cells
        // the logical fill will not write are zero-initialized, and the
        // borrowed typed host tensor always reads the slot in place.
        void upload_plane(
                ttnn::Tensor& plane,
                TtnnHostStaging::UploadLease& lease,
                const std::byte* source, std::size_t rows,
                std::size_t columns, std::size_t element_size,
                std::size_t plane_index) {
            const std::size_t padded_rows =
                    static_cast<std::size_t>(plane.padded_shape()[-2]);
            const std::size_t padded_columns =
                    static_cast<std::size_t>(plane.padded_shape()[-1]);
            const std::size_t num_tile_cols = padded_columns / 32;
            const std::size_t padded_elements = padded_rows * padded_columns;
            std::byte* buffer = lease.data();

            // Initialize only the required padding: the padded cells the
            // logical fill below will not write. All logical cells are
            // overwritten element-for-element, so padding is the retained
            // buffer's only obligation to zero.
            for (std::size_t row = rows; row < padded_rows; ++row) {
                for (std::size_t column = 0; column < padded_columns;
                     ++column) {
                    std::memset(
                            buffer
                                    + padded_cell_index(
                                              row, column, num_tile_cols)
                                            * element_size,
                            0, element_size);
                }
            }
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t column = columns; column < padded_columns;
                     ++column) {
                    std::memset(
                            buffer
                                    + padded_cell_index(
                                              row, column, num_tile_cols)
                                            * element_size,
                            0, element_size);
                }
            }

            for (std::size_t row = 0; row < rows; ++row) {
                std::size_t col = 0;
                while (col < columns) {
                    const std::size_t seg_columns = std::min<std::size_t>(
                            16 - (col % 16), columns - col);
                    const std::size_t first_element_index =
                            padded_cell_index(row, col, num_tile_cols);
                    std::memcpy(
                            buffer + first_element_index * element_size,
                            source + (row * columns + col) * element_size,
                            seg_columns * element_size);
                    col += seg_columns;
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
                fail_host_transfer_submission_at(plane_index);
#endif
                ttnn::copy_to_device(host_tiled, plane);
            };

            switch (plane.dtype()) {
                case tt::tt_metal::DataType::BFLOAT16:
                    upload_typed.template operator()<bfloat16>();
                    break;
                case tt::tt_metal::DataType::FLOAT32:
                    upload_typed.template operator()<float>();
                    break;
                case tt::tt_metal::DataType::UINT32:
                    upload_typed.template operator()<std::uint32_t>();
                    break;
                case tt::tt_metal::DataType::INT32:
                    upload_typed.template operator()<std::int32_t>();
                    break;
                case tt::tt_metal::DataType::UINT16:
                    upload_typed.template operator()<std::uint16_t>();
                    break;
                case tt::tt_metal::DataType::UINT8:
                    upload_typed.template operator()<std::uint8_t>();
                    break;
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
            fail_host_transfer_submission_at(plane_index);
#endif
            ttnn::copy_to_host(
                    queue, plane, staging, std::nullopt, /*blocking=*/false);
        }

        // Assembles one completed padded tile-major staging window into the
        // logical row-major destination. Native padding never reaches it.
        void assemble_download_plane(
                const std::byte* staging, std::byte* destination,
                std::size_t rows, std::size_t columns,
                std::size_t element_size, std::size_t num_tile_cols) {
            for (std::size_t row = 0; row < rows; ++row) {
                std::size_t col = 0;
                while (col < columns) {
                    const std::size_t seg_columns = std::min<std::size_t>(
                            16 - (col % 16), columns - col);
                    const std::size_t first_element_index =
                            padded_cell_index(row, col, num_tile_cols);
                    std::memcpy(
                            destination + (row * columns + col) * element_size,
                            staging + first_element_index * element_size,
                            seg_columns * element_size);
                    col += seg_columns;
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

    void region_from_host(
            tt::tt_metal::distributed::MeshDevice& device,
            TtnnHostStaging& staging, const TensorView& destination,
            ttnn::Tensor* planes, std::span<const std::byte> source) {
        const std::size_t rows = view_rows(destination);
        const std::size_t columns = view_columns(destination);
        const std::size_t element_size =
                element_bytes(destination.spec().data_type);
        const std::size_t plane_bytes = rows * columns * element_size;
        const std::size_t count = view_plane_count(destination);

        // Each lease remains owned until the queue proves completion. A
        // submission exception is conservatively treated as possibly in
        // flight, even when the vendor call throws before returning.
        std::vector<TtnnHostStaging::UploadLease> leases;
        leases.reserve(count);
        if (count != 0) {
            const ttnn::Tensor& first_plane =
                    planes[owner_plane_at(destination, 0)];
            const std::size_t padded_rows = static_cast<std::size_t>(
                    first_plane.padded_shape()[-2]);
            const std::size_t padded_columns = static_cast<std::size_t>(
                    first_plane.padded_shape()[-1]);
            const std::size_t slot = upload_slot_index(first_plane.dtype());
            for (std::size_t index = 0; index < count; ++index) {
                leases.emplace_back(staging.acquire_upload(
                        slot, padded_rows * padded_columns * element_size));
            }
        }

        std::size_t submitted = 0;
        bool submissions_complete = false;
        try {
            for (std::size_t index = 0; index < count; ++index) {
                ++submitted;
                upload_plane(
                        planes[owner_plane_at(destination, index)],
                        leases[index], source.data() + index * plane_bytes,
                        rows, columns, element_size, index);
            }
            submissions_complete = true;
            device.mesh_command_queue(0).finish();
            for (TtnnHostStaging::UploadLease& lease : leases) {
                lease.release();
            }
            staging.reclaim_retired_uploads();
        } catch (...) {
            const std::exception_ptr original_failure =
                    std::current_exception();
            bool drained = false;
            if (submitted != 0 && !submissions_complete) {
                try {
                    device.mesh_command_queue(0).finish();
                    drained = true;
                } catch (...) {
                }
            }
            for (TtnnHostStaging::UploadLease& lease : leases) {
                if (drained) {
                    lease.release();
                } else {
                    lease.retire();
                }
            }
            if (drained) {
                staging.reclaim_retired_uploads();
            }
            std::rethrow_exception(original_failure);
        }
    }

    void region_to_host(
            tt::tt_metal::distributed::MeshDevice& device,
            TtnnHostStaging& staging, const TensorView& source,
            const ttnn::Tensor* planes, std::span<std::byte> destination) {
        const std::size_t rows = view_rows(source);
        const std::size_t columns = view_columns(source);
        const std::size_t element_size =
                element_bytes(source.spec().data_type);
        const std::size_t plane_bytes = rows * columns * element_size;
        const std::size_t count = view_plane_count(source);
        const ttnn::Tensor& first_plane =
                planes[owner_plane_at(source, 0)];
        const std::size_t padded_rows = static_cast<std::size_t>(
                first_plane.padded_shape()[-2]);
        const std::size_t padded_columns = static_cast<std::size_t>(
                first_plane.padded_shape()[-1]);
        if (padded_rows != 0
                && padded_columns
                        > std::numeric_limits<std::size_t>::max()
                                / padded_rows) {
            throw std::overflow_error(
                    "TTNN padded plane element count overflows");
        }
        const std::size_t padded_elements = padded_rows * padded_columns;
        if (padded_elements != 0
                && element_size
                        > std::numeric_limits<std::size_t>::max()
                                / padded_elements) {
            throw std::overflow_error("TTNN padded plane byte count overflows");
        }
        const std::size_t padded_plane_bytes =
                padded_elements * element_size;
        if (padded_plane_bytes != 0
                && count
                        > std::numeric_limits<std::size_t>::max()
                                / padded_plane_bytes) {
            throw std::overflow_error(
                    "TTNN download staging byte count overflows");
        }
        // The retained byte staging buffer covers every padded plane of the
        // region and is returned only after the single finish below.
        const std::size_t total_bytes = count * padded_plane_bytes;
        TtnnHostStaging::DownloadLease lease =
                staging.acquire_download(total_bytes);
        auto& queue = device.mesh_command_queue(0);
        std::exception_ptr original_failure;
        try {
            for (std::size_t index = 0; index < count; ++index) {
                submit_download_plane(
                        queue, planes[owner_plane_at(source, index)],
                        lease.data() + index * padded_plane_bytes, index);
            }
            queue.finish();

            const std::size_t num_tile_cols = padded_columns / 32;
            for (std::size_t index = 0; index < count; ++index) {
                assemble_download_plane(
                        lease.data() + index * padded_plane_bytes,
                        destination.data() + index * plane_bytes, rows,
                        columns, element_size, num_tile_cols);
            }
            lease.release();
        } catch (...) {
            if (!original_failure) {
                original_failure = std::current_exception();
            }
            bool drained = false;
            try {
                queue.finish();
                drained = true;
            } catch (...) {
            }
            if (drained) {
                // Every submitted plane reached the host staging; the
                // faulted plane may hold partial bytes, so discard the
                // storage instead of handing it out again.
                lease.discard();
            } else {
                // The mesh cannot be drained: an asynchronous reader may
                // still touch the staging, so it is retired — never reused
                // and freed only when the device is torn down.
                lease.retire();
            }
            std::rethrow_exception(original_failure);
        }
    }

    void copy_planes(
            const TensorView& source, const ttnn::Tensor* source_planes,
            const TensorView& destination, ttnn::Tensor* destination_planes,
            bool& any_submitted) {
        any_submitted = false;
        const std::size_t count = view_plane_count(source);
        for (std::size_t index = 0; index < count; ++index) {
#ifdef IOM_ENABLE_TESTING
            fail_copy_planes_submission_at(index);
#endif
            ttnn::copy(
                    source_planes[owner_plane_at(source, index)],
                    destination_planes[owner_plane_at(destination, index)]);
            any_submitted = true;
        }
    }

}  // namespace iom::ttnn_detail

#ifdef IOM_ENABLE_TESTING
namespace iom::ttnn_test {

    void fail_next_copy_planes_submission_for_testing(
            std::size_t plane_index) noexcept {
        g_copy_planes_fault_at.store(plane_index, std::memory_order_release);
        g_copy_planes_fault_consumed.store(
                false, std::memory_order_release);
        g_copy_planes_fault_armed.store(true, std::memory_order_release);
    }

    bool copy_planes_submission_fault_consumed_for_testing() noexcept {
        return g_copy_planes_fault_consumed.load(std::memory_order_acquire);
    }

    void fail_next_host_transfer_submission_for_testing(
            std::size_t plane_index) noexcept {
        g_host_transfer_fault_at.store(plane_index, std::memory_order_release);
        g_host_transfer_fault_consumed.store(
                false, std::memory_order_release);
        g_host_transfer_fault_armed.store(true, std::memory_order_release);
    }

    bool host_transfer_submission_fault_consumed_for_testing() noexcept {
        return g_host_transfer_fault_consumed.load(
                std::memory_order_acquire);
    }

    void fail_next_host_transfer_staging_allocation_for_testing() noexcept {
        iom::ttnn_detail::g_fail_next_host_staging_allocation.store(
                true, std::memory_order_release);
        iom::ttnn_detail::g_host_staging_allocation_fault_consumed.store(
                false, std::memory_order_release);
    }

    bool host_transfer_staging_allocation_fault_consumed_for_testing() noexcept {
        return iom::ttnn_detail::g_host_staging_allocation_fault_consumed.load(
                std::memory_order_acquire);
    }

    std::size_t host_transfer_staging_allocation_count_for_testing() noexcept {
        return iom::ttnn_detail::g_host_staging_allocations.load(
                std::memory_order_acquire);
    }

}  // namespace iom::ttnn_test
#endif
