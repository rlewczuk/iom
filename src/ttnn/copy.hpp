#pragma once

// TTNN-internal logical region transfers and copies. Never installed or
// included by public headers: TTNN types stay inside the iom_ttnn target.

#include <cstddef>
#include <span>

#include <ttnn/tensor/tensor.hpp>

#include "iom/tensor.hpp"

namespace tt::tt_metal::distributed {
    class MeshDevice;
}

namespace iom::ttnn_detail {

    // Every function below requires the caller to hold the owning device's
    // API mutex; TTNN runtime calls are serialized through it.

    // Uploads the view's logical region from the row-major host encoding.
    // planes is the owner tensor's native handle: one TTNN-native tiled
    // tensor per logical owner plane. Synchronous at return.
    void region_from_host(
            tt::tt_metal::distributed::MeshDevice& device,
            const TensorView& destination, ttnn::Tensor* planes,
            std::span<const std::byte> source);

    // Downloads the view's logical region into the row-major host encoding.
    // destination is pre-zeroed so unused tail bits read as zero. No padding
    // reaches the host buffer. Synchronous at return.
    void region_to_host(
            tt::tt_metal::distributed::MeshDevice& device,
            const TensorView& source, const ttnn::Tensor* planes,
            std::span<std::byte> destination);

    // Copies logical values plane by plane in view-coordinate order, mapping
    // each view's plane offset and strides onto the owner planes. Values only;
    // no host staging and no encoding conversion. Returns after submission;
    // the caller synchronizes the mesh command queue once per operation.
    // Sets any_submitted when at least one plane reached the mesh, so the
    // caller can distinguish a failure before the first submission (nothing
    // pending on the device) from a failure after one or more submissions
    // (mesh work that must be drained before ownership can be removed).
    void copy_planes(
            const TensorView& source, const ttnn::Tensor* source_planes,
            const TensorView& destination, ttnn::Tensor* destination_planes,
            bool& any_submitted);

}  // namespace iom::ttnn_detail
