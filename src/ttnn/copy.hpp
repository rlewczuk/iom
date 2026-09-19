#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include <ttnn/tensor/tensor.hpp>
#include "iom/iom.hpp"
#include "../shared/scalar_add.hpp"

namespace tt::tt_metal::distributed { class MeshDevice; }

namespace iom {
class TtnnDevice;
namespace ttnn_detail {
struct HostWorkspaceLeasePayload;

/**
 * Move-only access to one checked caller-owned TTNN workspace range.  A
 * payload keeps the host allocation and its TT-Metal pin alive until the
 * caller supplies a covering completion proof.  Destruction without proof
 * quarantines the payload through the owning `TtnnDevice`.
 */
class TtnnWorkspaceLease final {
public:
    ~TtnnWorkspaceLease() noexcept;
    TtnnWorkspaceLease(const TtnnWorkspaceLease&) = delete;
    TtnnWorkspaceLease& operator=(const TtnnWorkspaceLease&) = delete;
    TtnnWorkspaceLease(TtnnWorkspaceLease&&) noexcept;
    TtnnWorkspaceLease& operator=(TtnnWorkspaceLease&&) noexcept;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t byte_size() const noexcept { return bytes_; }
    [[nodiscard]] bool empty() const noexcept {
        return data_ == nullptr || bytes_ == 0;
    }
    [[nodiscard]] std::shared_ptr<void> keepalive() const noexcept;

    // A true disposition is a covering native completion proof.  A false
    // disposition keeps the payload quarantined and blocks reuse.
    void complete(bool covering_proof) noexcept;

private:
    friend TtnnWorkspaceLease acquire_workspace_lease(
            TtnnDevice&, const RawWorkspaceView&);
    TtnnWorkspaceLease(
            TtnnDevice&, std::byte*, std::size_t,
            std::unique_ptr<HostWorkspaceLeasePayload>);

    TtnnDevice* device_ = nullptr;
    std::byte* data_ = nullptr;
    std::size_t bytes_ = 0;
    std::unique_ptr<HostWorkspaceLeasePayload> payload_;
};

[[nodiscard]] TtnnWorkspaceLease acquire_workspace_lease(
        TtnnDevice& device, const RawWorkspaceView& workspace);

[[nodiscard]] std::size_t padded_plane_bytes(const ttnn::Tensor& plane);

void raw_download_plane(
        TtnnDevice& device, const ttnn::Tensor& plane,
        TtnnWorkspaceLease& workspace);
void raw_upload_plane(
        TtnnDevice& device, ttnn::Tensor& plane,
        TtnnWorkspaceLease& workspace);
}  // namespace ttnn_detail
}  // namespace iom

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
