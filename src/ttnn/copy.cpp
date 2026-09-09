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
#include <cstring>
#include <unordered_map>
#include "../shared/scalar_add.hpp"

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

        // Native staging uses whole carrier cells; public logical encodings
        // may be packed and are converted by the transfer helpers below.
        std::size_t element_bytes(DataType type) {
            const std::size_t bits = detail::leaf_bits(type);
            return (bits + 7) / 8;
        }

        std::size_t carrier_bytes(tt::tt_metal::DataType type) {
            switch (type) {
                case tt::tt_metal::DataType::UINT8: return 1;
                case tt::tt_metal::DataType::UINT16:
                case tt::tt_metal::DataType::BFLOAT16: return 2;
                default: return 4;
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
                fail_host_transfer_submission_at(plane_index);
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
            fail_host_transfer_submission_at(plane_index);
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
            const TensorView& source, const ttnn::Tensor* source_planes,
            const TensorView& destination, ttnn::Tensor* destination_planes,
            bool& any_submitted) {
        any_submitted = false;
        const std::size_t count = view_plane_count(source);
        for (std::size_t index = 0; index < count; ++index) {
#ifdef IOM_ENABLE_TESTING
            fail_copy_planes_submission_at(index);
#endif
            ttnn::copy(source_planes[owner_plane_at(source, index)],
                       destination_planes[owner_plane_at(destination, index)]);
            any_submitted = true;
        }
    }


    void add_planes(
            tt::tt_metal::distributed::MeshDevice& device,
            TtnnHostStaging& staging, const AddRequest& request,
            ttnn::Tensor* lhs_planes, ttnn::Tensor* rhs_planes,
            ttnn::Tensor* out_planes, bool& any_submitted) {
        any_submitted = false;
        const auto dims = request.result_shape.dimensions();
        const std::size_t rank = dims.size();
        const std::size_t leading_rank = rank - 2;
        const std::size_t rows = dims[rank - 2];
        const std::size_t columns = dims[rank - 1];
        const std::size_t bits = detail::leaf_bits(request.out.spec.data_type);
        const std::size_t factor = bits > 32 ? 2 : 1;
        auto& queue = device.mesh_command_queue(0);
        std::unordered_map<std::size_t, std::vector<std::byte>> lhs_cache;
        std::unordered_map<std::size_t, std::vector<std::byte>> rhs_cache;
        std::unordered_map<std::size_t, std::vector<std::byte>> out_cache;
        auto load = [&](ttnn::Tensor* planes, std::size_t plane,
                        auto& cache) -> std::vector<std::byte>& {
            auto [it, inserted] = cache.emplace(plane, std::vector<std::byte>{});
            if (inserted) {
                const std::size_t bytes =
                        static_cast<std::size_t>(planes[plane].padded_shape()[-2])
                        * static_cast<std::size_t>(planes[plane].padded_shape()[-1])
                        * carrier_bytes(planes[plane].dtype());
                it->second.resize(bytes);
                ttnn::copy_to_host(queue, planes[plane], it->second.data(),
                                   std::nullopt, /*blocking=*/true);
            }
            return it->second;
        };
        auto plane_at = [&](const AddSnapshot& view,
                            std::span<const std::size_t> coordinates) {
            std::size_t plane = view.plane_offset;
            for (std::size_t i = 0; i < leading_rank; ++i) {
                plane += coordinates[i] * view.logical_plane_strides[i];
            }
            return plane;
        };
        auto read = [&](const AddSnapshot& view,
                        ttnn::Tensor* planes,
                        auto& cache, std::span<const std::size_t> coordinates,
                        std::size_t row, std::size_t column) {
            const std::size_t plane = plane_at(view, coordinates);
            auto& raw = load(planes, plane, cache);
            const std::size_t padded_columns =
                    static_cast<std::size_t>(planes[plane].padded_shape()[-1]);
            const std::size_t native_column = column * factor;
            std::uint64_t value = 0;
            const std::size_t carrier = carrier_bytes(planes[plane].dtype());
            for (std::size_t part = 0; part < factor; ++part) {
                std::memcpy(reinterpret_cast<std::byte*>(&value)
                                    + part * carrier,
                            raw.data()
                                    + (padded_cell_index(
                                               row, native_column + part,
                                               padded_columns / 32)
                                       * carrier),
                            carrier);
            }
            return value;
        };
        // Iterate every leading-plane combination of the result, then the
        // matrix within each plane: a rank-4+ result addresses multiple
        // owner planes and must not collapse to the first combination.
        std::size_t plane_combos = 1;
        for (std::size_t i = 0; i < leading_rank; ++i) {
            plane_combos *= dims[i];
        }
        const std::size_t total = plane_combos * rows * columns;
        for (std::size_t flat = 0; flat < total; ++flat) {
            std::size_t rem = flat;
            std::vector<std::size_t> coordinates(leading_rank);
            for (std::size_t i = leading_rank; i-- > 0;) {
                coordinates[i] = rem % dims[i];
                rem /= dims[i];
            }
            const std::size_t matrix = rem;
            const std::size_t row = matrix / columns;
            const std::size_t column = matrix % columns;
            auto operand_coord = [&](const AddSnapshot& view) {
                std::vector<std::size_t> mapped(leading_rank, 0);
                const auto vdims = view.spec.shape.dimensions();
                const std::size_t leading_operand_rank = vdims.size() - 2;
                const std::size_t offset = leading_rank - leading_operand_rank;
                for (std::size_t i = 0; i < leading_operand_rank; ++i) {
                    mapped[offset + i] =
                            vdims[i] == 1 ? 0 : coordinates[offset + i];
                }
                return mapped;
            };
            const auto lc = operand_coord(request.lhs);
            const auto rc = operand_coord(request.rhs);
            const std::size_t lr = request.lhs.spec.shape.dimension(
                    request.lhs.spec.shape.rank() - 2) == 1 ? 0 : row;
            const std::size_t rr = request.rhs.spec.shape.dimension(
                    request.rhs.spec.shape.rank() - 2) == 1 ? 0 : row;
            const std::size_t lcol = request.lhs.spec.shape.dimension(
                    request.lhs.spec.shape.rank() - 1) == 1 ? 0 : column;
            const std::size_t rcol = request.rhs.spec.shape.dimension(
                    request.rhs.spec.shape.rank() - 1) == 1 ? 0 : column;
            const std::uint64_t value = detail::scalar_add(
                    request.out.spec.data_type,
                    read(request.lhs, lhs_planes, lhs_cache, lc, lr, lcol),
                    read(request.rhs, rhs_planes, rhs_cache, rc, rr, rcol));
            const std::size_t plane = plane_at(request.out, coordinates);
            auto& raw = load(out_planes, plane, out_cache);
            const std::size_t padded_columns =
                    static_cast<std::size_t>(out_planes[plane].padded_shape()[-1]);
            const std::size_t carrier = carrier_bytes(out_planes[plane].dtype());
            for (std::size_t part = 0; part < factor; ++part) {
                std::memcpy(raw.data()
                                    + padded_cell_index(
                                          row, column * factor + part,
                                          padded_columns / 32)
                                          * carrier,
                            reinterpret_cast<const std::byte*>(&value)
                                    + part * carrier,
                            carrier);
            }
        }
        std::vector<TtnnHostStaging::UploadLease> leases;
        leases.reserve(out_cache.size());
        for (auto& [plane, raw] : out_cache) {
            staging.reclaim_retired_uploads();
            const std::size_t padded_columns =
                    static_cast<std::size_t>(out_planes[plane].padded_shape()[-1]);
            const std::size_t carrier = carrier_bytes(out_planes[plane].dtype());
            leases.emplace_back(staging.acquire_upload(
                    upload_slot_index(out_planes[plane].dtype()),
                    padded_columns * static_cast<std::size_t>(
                            out_planes[plane].padded_shape()[-2]) * carrier));
            std::memcpy(leases.back().data(), raw.data(), raw.size());
            auto upload_typed = [&]<typename T>() {
                tt::tt_metal::HostBuffer host_buffer(
                        ttsl::Span<T>(
                                reinterpret_cast<T*>(leases.back().data()),
                                raw.size() / sizeof(T)),
                        tt::tt_metal::MemoryPin(leases.back().keepalive()));
                ttnn::Tensor host_tiled(
                        std::move(host_buffer), out_planes[plane].logical_shape(),
                        out_planes[plane].padded_shape(), out_planes[plane].dtype(),
                        tt::tt_metal::Layout::TILE);
                ttnn::copy_to_device(host_tiled, out_planes[plane]);
                any_submitted = true;
            };
            switch (out_planes[plane].dtype()) {
                case tt::tt_metal::DataType::BFLOAT16:
                    upload_typed.template operator()<bfloat16>(); break;
                case tt::tt_metal::DataType::FLOAT32:
                    upload_typed.template operator()<float>(); break;
                case tt::tt_metal::DataType::UINT16:
                    upload_typed.template operator()<std::uint16_t>(); break;
                case tt::tt_metal::DataType::UINT8:
                    upload_typed.template operator()<std::uint8_t>(); break;
                default:
                    upload_typed.template operator()<std::uint32_t>(); break;
            }
        }
        // One queue finish proves completion of every upload; only then
        // are the retained staging slots handed back for reuse.
        queue.finish();
        for (auto& lease : leases) {
            lease.release();
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
