#pragma once

#include <cstddef>
#include <vector>
#include <ttnn/tensor/tensor.hpp>
#include "iom/iom.hpp"
#include "../shared/scalar_add.hpp"

namespace tt::tt_metal::distributed { class MeshDevice; }

namespace iom::ttnn_detail {
class TtnnHostStaging;
void region_from_host(tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&, const TensorView&, ttnn::Tensor*, std::span<const std::byte>);
void region_to_host(tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&, const TensorView&, const ttnn::Tensor*, std::span<std::byte>);
struct CopySnapshot {
    TensorSpec spec;
    void* native_handle;
    std::size_t plane_offset;
    std::vector<std::size_t> plane_strides;
};
std::size_t snapshot_plane_count(const CopySnapshot&);
std::size_t snapshot_plane_count(const TensorSpec&);
std::size_t snapshot_owner_plane_at(
        const CopySnapshot&, std::size_t index);
std::size_t snapshot_owner_plane_at(
        const TensorSpec&, std::size_t plane_offset,
        std::span<const std::size_t> plane_strides, std::size_t index);
void copy_planes(const CopySnapshot&, const ttnn::Tensor*, const CopySnapshot&, ttnn::Tensor*, bool&);
std::size_t carrier_bytes(tt::tt_metal::DataType);
std::size_t upload_slot_index(tt::tt_metal::DataType);
std::size_t padded_cell_index(
        std::size_t row, std::size_t column, std::size_t num_tile_cols);
struct BinarySnapshot {
    TensorSpec spec;
    void* native_handle;
    std::size_t plane_offset;
    std::vector<std::size_t> logical_plane_strides;
};
struct BinaryRequest {
    BinarySnapshot lhs;
    BinarySnapshot rhs;
    BinarySnapshot out;
    TensorShape result_shape;
};
using BinaryFinish =
        void (*)(tt::tt_metal::distributed::MeshDevice&);
template <detail::scalar_add_detail::BinaryOp Op>
void binary_planes(
        tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&,
        const BinaryRequest&, ttnn::Tensor*, ttnn::Tensor*, ttnn::Tensor*,
        bool&, bool&, BinaryFinish);
}  // namespace iom::ttnn_detail
