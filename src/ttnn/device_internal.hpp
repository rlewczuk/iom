#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>

#include <tt-metalium/mesh_buffer.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "iom/ttnn/device.hpp"
#include "registry_state.hpp"
#include "staging.hpp"

namespace iom::ttnn_detail {

[[nodiscard]] bool is_supported(DataType type);
[[nodiscard]] bool rmsnorm_supported(DataType type) noexcept;
[[nodiscard]] tt::tt_metal::DataType native_dtype(DataType type);
[[nodiscard]] std::size_t carrier_factor(DataType type);
[[nodiscard]] std::invalid_argument invalid_ordinal(
        std::uint32_t ordinal, std::size_t device_count);
[[nodiscard]] std::uint32_t checked_to_uint32(
        std::size_t dimension, std::size_t index);
[[nodiscard]] std::size_t checked_plane_count(const TensorSpec& spec);

}  // namespace iom::ttnn_detail

namespace iom {

class TtnnDevice final : public Device {
public:
    TtnnDevice(
            std::uint32_t ordinal,
            std::shared_ptr<ttnn::MeshDevice> native_device,
            QueueConfig queue_config);

    TtnnDevice(const TtnnDevice&) = delete;
    TtnnDevice& operator=(const TtnnDevice&) = delete;

    ~TtnnDevice() override;

    [[nodiscard]] BackendKind backend_kind() const noexcept override;
    [[nodiscard]] std::uint32_t backend_device() const noexcept override;
    [[nodiscard]] std::span<const DataType> supported_data_types()
            const noexcept override;

    [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
            const TensorSpec& spec) override;
    [[nodiscard]] std::unique_ptr<RawWorkspace> create_workspace(
            std::size_t bytes) override;
    [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

    [[nodiscard]] tt::tt_metal::distributed::MeshDevice& mesh() noexcept;
    [[nodiscard]] std::mutex& api_mutex() noexcept;
    [[nodiscard]] ttnn_detail::TtnnHostStaging& host_staging() noexcept;
    [[nodiscard]] detail::RegistryState& registry_state() noexcept;

private:
    std::uint32_t ordinal_;
    // Declared before the native device so it is destroyed after
    // the mesh teardown: retired staging may still be referenced by
    // an undrainable asynchronous reader until the mesh is gone.
    ttnn_detail::TtnnHostStaging host_staging_;
    std::shared_ptr<ttnn::MeshDevice> native_device_;
    std::mutex api_mutex_;
    detail::RegistryState registry_state_;
};

namespace ttnn_detail {

/**
 * Checked, owner-absolute view of one live positive TTNN raw-workspace
 * owner. Queue code addresses the caller's native scratch only through this
 * record: the owning replicated DRAM allocation, its actual native page
 * size, the native base address that common range and lease identity use,
 * the owner's logical bytes, and the view's owner-absolute logical offset.
 * Neither a subrange nor an operation creates a second native buffer or a
 * new native view.
 */
struct NativeWorkspace {
    // The single owning replicated DRAM allocation of the workspace owner.
    tt::tt_metal::distributed::MeshBuffer* owner = nullptr;
    // Shared ownership keeps the real MeshBuffer alive across an accepted
    // asynchronous submission even if the caller destroys its RawWorkspace.
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> owner_handle;
    // Actual native page size; the owner is one contiguous page, so this is
    // the requested logical byte count rounded up to 32.
    std::uint64_t page_size = 0;
    // Native base address of the owner, shared by every subrange identity.
    std::uint64_t base = 0;
    // Logical bytes owned by the caller's request (`RawWorkspace::byte_size`).
    std::size_t logical_bytes = 0;
    // Owner-absolute logical offset of the checked view.
    std::size_t offset = 0;

    // Native address of this view's first byte.
    [[nodiscard]] std::uint64_t range_address() const noexcept {
        return base + offset;
    }
};

/**
 * Checked access to the native owner behind one live positive workspace view
 * created by `device`. An empty view, an owner that is not live on `device`
 * (foreign or already destroyed; never dereferenced), a zero-byte owner
 * without a native allocation, a non-32-byte-aligned owner-absolute offset,
 * an empty range, and a range that ends past the owner's logical bytes all
 * reject as `std::invalid_argument` before any native effect. A live native
 * owner without an address or without its checked page/alignment invariants
 * reports `std::runtime_error`.
 */
[[nodiscard]] NativeWorkspace checked_native_workspace(
        TtnnDevice& device, const RawWorkspaceView& workspace);

}  // namespace ttnn_detail

}  // namespace iom
