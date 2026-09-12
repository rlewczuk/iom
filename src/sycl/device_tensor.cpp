#include "device_internal.hpp"

#include <sycl/sycl.hpp>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "copy.hpp"

namespace iom {

    /**
     * Owner of one raw-workspace arena suballocation. The base
     * RawWorkspace registers the exact identity with the device
     * before the body can fail; destruction hands the range back
     * through the device boundary, which keeps a retained lease out
     * of the allocator.
     */
    class SyclWorkspace final : public RawWorkspace {
    public:
        SyclWorkspace(
                SyclDevice& device, void* address, std::size_t bytes)
                : RawWorkspace(device, bytes),
                  device_(device),
                  address_(address) {}

        ~SyclWorkspace() override {
            if (address_ != nullptr) {
                device_.release_workspace(this, address_, byte_size());
            }
        }

        SyclWorkspace(const SyclWorkspace&) = delete;
        SyclWorkspace& operator=(const SyclWorkspace&) = delete;

    private:
        [[nodiscard]] void* workspace_address() const noexcept override {
            return address_;
        }

        SyclDevice& device_;
        void* address_;
    };

    class SyclTensor final : public Tensor {
    public:
        SyclTensor(
                const TensorSpec& spec, SyclDevice& device,
                detail::RegistryState& state)
                : Tensor(spec, device),
                  device_(device),
                  state_(&state) {
            address_ = device_.allocate_data(
                    view().spec().tiled_storage_nbytes());

            try {
                if (sycl::get_pointer_type(address_, device_.context())
                            == sycl::usm::alloc::unknown
                        || sycl::get_pointer_device(
                                   address_, device_.context())
                                != device_.native_device()) {
                    throw std::runtime_error(
                            "SYCL tensor storage is incompatible with "
                            "the owning context");
                }
                auto& queue = device_.transfer_queue();
                queue.memset(
                        address_, 0,
                        view().spec().tiled_storage_nbytes());
                queue.wait_and_throw();
            } catch (...) {
                void* rejected = std::exchange(address_, nullptr);
                device_.release_data(rejected);
                throw;
            }
        }

        ~SyclTensor() noexcept override {
            if (address_ == nullptr) {
                return;
            }

            const std::size_t bytes =
                    view().spec().tiled_storage_nbytes();
            const auto quarantine_storage = [this, bytes]() noexcept {
                try {
                    state_->quarantine
                            .emplace<detail::AllocatorCleanupAction>(
                                    device_.data_allocator(), address_, bytes);
                } catch (...) {
                    // Leaking is safer than returning failed storage to
                    // the allocator when quarantine allocation fails.
                }
                address_ = nullptr;
            };
            const auto release_storage = [this]() noexcept {
                try {
                    device_.release_data(address_);
                } catch (...) {
                }
                address_ = nullptr;
            };
            iom::detail::release_or_quarantine(
                    state_->registry, address_, quarantine_storage,
                    release_storage);
        }

    private:
        [[nodiscard]] void* storage_handle() noexcept override {
            return address_;
        }

        void region_from_host(
                const TensorView& destination,
                std::span<const std::byte> source,
                RawWorkspaceView workspace) override {
            std::lock_guard<std::mutex> lock(device_.transfer_mutex_);
            sycl_detail::region_from_host(
                    device_.transfer_queue(), device_,
                    device_.registry_state_, destination, workspace,
                    source, device_.transfer_resource_poisoned_);
        }

        void region_to_host(
                const TensorView& source,
                std::span<std::byte> destination,
                RawWorkspaceView workspace) const override {
            std::lock_guard<std::mutex> lock(device_.transfer_mutex_);
            sycl_detail::region_to_host(
                    device_.transfer_queue(), device_,
                    device_.registry_state_, source, workspace, destination,
                    device_.transfer_resource_poisoned_);
        }

        SyclDevice& device_;
        detail::RegistryState* state_;
        void* address_ = nullptr;
    };

    std::unique_ptr<Tensor> SyclDevice::create_tensor(
            const TensorSpec& spec) {
        return std::make_unique<SyclTensor>(spec, *this, registry_state_);
    }

    std::unique_ptr<RawWorkspace> SyclDevice::create_workspace(
            std::size_t bytes) {
        if (bytes == 0) {
            // Valid empty owner: no arena suballocation, no native call.
            return std::make_unique<SyclWorkspace>(*this, nullptr, 0);
        }
        // Positive creation suballocates the already reserved data arena
        // under the allocator bookkeeping boundary; it never reserves a new
        // native backing. Exhaustion, including fragmentation with no fitting
        // contiguous range, is std::bad_alloc with no fallback.
        void* address = allocate_data(bytes);
        try {
            if (sycl::get_pointer_type(address, context())
                        == sycl::usm::alloc::unknown
                    || sycl::get_pointer_device(address, context())
                            != native_device()) {
                throw std::runtime_error(
                        "SYCL workspace storage is incompatible with "
                        "the owning context");
            }
            return std::make_unique<SyclWorkspace>(*this, address, bytes);
        } catch (...) {
            release_data(address);
            throw;
        }
    }

}  // namespace iom
