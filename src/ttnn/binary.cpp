#include "copy.hpp"
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
#include <array>
#include <unordered_map>
#include "../shared/scalar_add.hpp"

namespace iom::ttnn_detail {

template <detail::scalar_add_detail::BinaryOp Op>
void binary_planes(
        tt::tt_metal::distributed::MeshDevice& device,
        TtnnHostStaging& staging, const BinaryRequest& request,
        ttnn::Tensor* lhs_planes, ttnn::Tensor* rhs_planes,
        ttnn::Tensor* out_planes, bool& any_submitted,
        bool& native_drained, BinaryFinish finish) {
    any_submitted = false;
    native_drained = false;
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
    auto plane_at = [&](const BinarySnapshot& view,
                        std::span<const std::size_t> coordinates) {
        std::size_t plane = view.plane_offset;
        for (std::size_t i = 0; i < leading_rank; ++i) {
            plane += coordinates[i] * view.logical_plane_strides[i];
        }
        return plane;
    };
    auto read = [&](const BinarySnapshot& view,
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

    // Tensor validation limits ranks to eight, so six leading coordinates
    // cover every binary request. Keep these buffers outside the hot loop:
    // coordinate and broadcast mapping state is request-local and reused for
    // every element instead of allocating vectors per element.
    constexpr std::size_t kMaxLeadingRank = 6;
    std::array<std::size_t, kMaxLeadingRank> coordinates{};
    std::array<std::size_t, kMaxLeadingRank> lhs_coordinates{};
    std::array<std::size_t, kMaxLeadingRank> rhs_coordinates{};
    const std::span<const std::size_t> coordinate_span(
            coordinates.data(), leading_rank);
    auto operand_coord = [&](const BinarySnapshot& view,
                             std::array<std::size_t, kMaxLeadingRank>& mapped) {
        mapped.fill(0);
        const auto vdims = view.spec.shape.dimensions();
        const std::size_t leading_operand_rank = vdims.size() - 2;
        const std::size_t offset = leading_rank - leading_operand_rank;
        for (std::size_t i = 0; i < leading_operand_rank; ++i) {
            mapped[offset + i] =
                    vdims[i] == 1 ? 0 : coordinates[offset + i];
        }
        return std::span<const std::size_t>(
                mapped.data(), leading_rank);
    };

    for (std::size_t flat = 0; flat < total; ++flat) {
        std::size_t rem = flat;
        for (std::size_t i = leading_rank; i-- > 0;) {
            coordinates[i] = rem % dims[i];
            rem /= dims[i];
        }
        const std::size_t matrix = rem;
        const std::size_t row = matrix / columns;
        const std::size_t column = matrix % columns;
        const auto lc = operand_coord(request.lhs, lhs_coordinates);
        const auto rc = operand_coord(request.rhs, rhs_coordinates);
        const std::size_t lr = request.lhs.spec.shape.dimension(
                request.lhs.spec.shape.rank() - 2) == 1 ? 0 : row;
        const std::size_t rr = request.rhs.spec.shape.dimension(
                request.rhs.spec.shape.rank() - 2) == 1 ? 0 : row;
        const std::size_t lcol = request.lhs.spec.shape.dimension(
                request.lhs.spec.shape.rank() - 1) == 1 ? 0 : column;
        const std::size_t rcol = request.rhs.spec.shape.dimension(
                request.rhs.spec.shape.rank() - 1) == 1 ? 0 : column;
        const auto left =
                read(request.lhs, lhs_planes, lhs_cache, lc, lr, lcol);
        const auto right =
                read(request.rhs, rhs_planes, rhs_cache, rc, rr, rcol);
        const std::uint64_t value = detail::scalar_binary<Op>(
                request.out.spec.data_type, left, right);
        const std::size_t plane = plane_at(request.out, coordinate_span);
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
    finish(device);
    native_drained = true;
    for (auto& lease : leases) {
        lease.release();
    }
}

template void binary_planes<detail::scalar_add_detail::BinaryOp::add>(
        tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&,
        const BinaryRequest&, ttnn::Tensor*, ttnn::Tensor*, ttnn::Tensor*,
        bool&, bool&, BinaryFinish);
template void binary_planes<detail::scalar_add_detail::BinaryOp::mul>(
        tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&,
        const BinaryRequest&, ttnn::Tensor*, ttnn::Tensor*, ttnn::Tensor*,
        bool&, bool&, BinaryFinish);
template void binary_planes<detail::scalar_add_detail::BinaryOp::sub>(
        tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&,
        const BinaryRequest&, ttnn::Tensor*, ttnn::Tensor*, ttnn::Tensor*,
        bool&, bool&, BinaryFinish);
template void binary_planes<detail::scalar_add_detail::BinaryOp::div>(
        tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&,
        const BinaryRequest&, ttnn::Tensor*, ttnn::Tensor*, ttnn::Tensor*,
        bool&, bool&, BinaryFinish);

}  // namespace iom::ttnn_detail
