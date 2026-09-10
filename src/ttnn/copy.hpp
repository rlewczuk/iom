#pragma once

#include <cstddef>
#include <vector>
#include <ttnn/tensor/tensor.hpp>
#include "iom/iom.hpp"

namespace tt::tt_metal::distributed { class MeshDevice; }

namespace iom::ttnn_detail {
class TtnnHostStaging;
void region_from_host(tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&, const TensorView&, ttnn::Tensor*, std::span<const std::byte>);
void region_to_host(tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&, const TensorView&, const ttnn::Tensor*, std::span<std::byte>);
void copy_planes(const TensorView&, const ttnn::Tensor*, const TensorView&, ttnn::Tensor*, bool&);

enum class BinaryOperation { add, mul, sub, div };
struct BinarySnapshot {
    TensorSpec spec;
    void* native_handle;
    std::size_t plane_offset;
    std::vector<std::size_t> logical_plane_strides;
};
struct BinaryRequest {
    BinaryOperation operation;
    BinarySnapshot lhs;
    BinarySnapshot rhs;
    BinarySnapshot out;
    TensorShape result_shape;
};
void binary_planes(tt::tt_metal::distributed::MeshDevice&, TtnnHostStaging&, const BinaryRequest&, ttnn::Tensor*, ttnn::Tensor*, ttnn::Tensor*, bool&);
}  // namespace iom::ttnn_detail
