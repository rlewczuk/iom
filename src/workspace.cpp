#include "iom/device.hpp"
#include "iom/tensor.hpp"

#include <cstdint>
#include <mutex>

namespace iom {

    // ------------------------------------------------------------------
    // Raw workspace ownership and views (leaf 05).

    RawWorkspaceView::RawWorkspaceView(
            const RawWorkspace& owner, std::size_t offset,
            std::size_t bytes)
            : owner_(&owner),
              offset_(offset),
              bytes_(bytes) {
        const std::size_t owner_bytes = owner.byte_size();
        if (offset > owner_bytes) {
            throw std::out_of_range(
                    "workspace subrange offset exceeds the owner range");
        }
        if (bytes > owner_bytes - offset) {
            throw std::out_of_range(
                    "workspace subrange exceeds the owner range");
        }
    }

    const Device& RawWorkspaceView::device() const {
        if (owner_ == nullptr) {
            throw std::logic_error(
                    "empty workspace view has no device");
        }
        return owner_->device();
    }

    BackendKind RawWorkspaceView::backend_kind() const {
        return device().backend_kind();
    }

    std::uint32_t RawWorkspaceView::backend_device() const {
        return device().backend_device();
    }

    RawWorkspaceView RawWorkspaceView::subrange(
            std::size_t offset, std::size_t bytes) const {
        if (owner_ == nullptr) {
            throw std::invalid_argument(
                    "empty workspace view has no owner to subrange");
        }
        if (offset % 32 != 0) {
            throw std::invalid_argument(
                    "workspace subrange offset is not 32-byte aligned");
        }
        return RawWorkspaceView{*owner_, offset, bytes};
    }

    void* RawWorkspaceView::range_address() const noexcept {
        if (owner_ == nullptr) return nullptr;
        const void* const base = owner_->storage_identity().base;
        if (base == nullptr) return nullptr;
        return reinterpret_cast<void*>(
                reinterpret_cast<std::uintptr_t>(base) + offset_);
    }

    RawWorkspace::RawWorkspace(const Device& device, std::size_t bytes)
            : device_(&device),
              bytes_(bytes) {
        device.register_workspace(this);
    }

    RawWorkspace::~RawWorkspace() {
        device_->unregister_workspace(this);
    }

    BackendKind RawWorkspace::backend_kind() const noexcept {
        return device_->backend_kind();
    }

    std::uint32_t RawWorkspace::backend_device() const noexcept {
        return device_->backend_device();
    }

    RawWorkspaceView RawWorkspace::view() const {
        return RawWorkspaceView{*this, 0, bytes_};
    }

    void* RawWorkspace::workspace_address() const noexcept {
        return nullptr;
    }

    detail::StorageIdentity RawWorkspace::storage_identity() const noexcept {
        void* const base = workspace_address();
        return {base, base};
    }

    bool Device::owns_workspace(const RawWorkspace* workspace) const noexcept {
        if (workspace == nullptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(workspace_registry_mutex_);
        return live_workspaces_.find(workspace)
                != live_workspaces_.end();
    }

    void Device::register_workspace(const RawWorkspace* workspace) const {
        std::lock_guard<std::mutex> lock(workspace_registry_mutex_);
        live_workspaces_.insert(workspace);
    }

    void Device::unregister_workspace(
            const RawWorkspace* workspace) const noexcept {
        std::lock_guard<std::mutex> lock(workspace_registry_mutex_);
        live_workspaces_.erase(workspace);
    }

}  // namespace iom
