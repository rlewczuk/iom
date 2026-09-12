#include "device_internal.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace iom {

    namespace {

        // Validates that an arena suballocation is unmanaged device memory
        // on this exact ordinal and lies inside the owning arena backing.
        // Interior addresses are accepted: the identity checks pin the exact
        // ordinal, and membership in the device's own arena range binds the
        // address to this device's reservation without requiring it to be
        // the base of an independent native allocation (hipPointerGetAttributes
        // reports the queried address, not the containing allocation's base,
        // so base equality would reject every interior suballocation).
        void validate_native_storage(
                void* address, int ordinal, void* arena_base,
                std::size_t arena_bytes) {
            hipPointerAttribute_t attributes{};
            rocm_detail::check_hip(
                    "hipPointerGetAttributes",
                    hipPointerGetAttributes(&attributes, address));
            const std::uintptr_t raw =
                    reinterpret_cast<std::uintptr_t>(address);
            const std::uintptr_t base =
                    reinterpret_cast<std::uintptr_t>(arena_base);
            if (attributes.type != hipMemoryTypeDevice
                    || attributes.isManaged != 0
                    || attributes.device != ordinal || raw < base
                    || raw - base >= arena_bytes) {
                throw std::runtime_error(
                        "ROCm tensor storage is incompatible with the owning "
                        "device");
            }
        }

        /**
         * Owner of one raw-workspace arena suballocation. The base
         * RawWorkspace registers the exact identity with the device
         * before the body can fail; destruction hands the range back
         * through the device boundary, which keeps a retained lease out
         * of the allocator.
         */
        class RocmWorkspace final : public RawWorkspace {
        public:
            RocmWorkspace(
                    RocmDevice& device, void* address, std::size_t bytes)
                    : RawWorkspace(device, bytes),
                      device_(device),
                      address_(address) {}

            ~RocmWorkspace() override {
                if (address_ != nullptr) {
                    device_.release_workspace(this, address_, byte_size());
                }
            }

            RocmWorkspace(const RocmWorkspace&) = delete;
            RocmWorkspace& operator=(const RocmWorkspace&) = delete;

        private:
            [[nodiscard]] void* workspace_address() const noexcept override {
                return address_;
            }

            RocmDevice& device_;
            void* address_;
        };

    }  // namespace

    class RocmTensor final : public Tensor {
    public:
        RocmTensor(const TensorSpec& spec, RocmDevice& device)
                : Tensor(spec, device), device_(device),
                  state_(&device.registry_state()) {
            address_ = device_.allocate_data(
                    view().spec().tiled_storage_nbytes());
            try {
                device_.activate();
                validate_native_storage(
                        address_, static_cast<int>(device_.ordinal()),
                        device_.data_backing(), device_.data_backing_bytes());
                rocm_detail::check_hip(
                        "hipMemset",
                        hipMemset(
                                address_, 0,
                                view().spec().tiled_storage_nbytes()));
                rocm_detail::check_hip(
                        "hipDeviceSynchronize", hipDeviceSynchronize());
            } catch (...) {
                void* rejected = std::exchange(address_, nullptr);
                device_.release_data(rejected);
                throw;
            }
        }

        ~RocmTensor() noexcept override {
            if (address_ == nullptr) {
                return;
            }

            const std::size_t bytes = view().spec().tiled_storage_nbytes();
            const auto quarantine_storage = [this, bytes]() noexcept {
                try {
                    state_->quarantine.emplace<detail::AllocatorCleanupAction>(
                            device_.data_allocator(), address_, bytes);
                } catch (...) {
                    // Keep failed storage unavailable if quarantine allocation
                    // itself fails.
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
            rocm_detail::region_from_host(
                    device_.transfer_resource_->stream(),
                    static_cast<int>(device_.ordinal()), device_,
                    device_.registry_state_, destination, workspace, source,
                    device_.transfer_resource_poisoned_);
        }

        void region_to_host(
                const TensorView& source,
                std::span<std::byte> destination,
                RawWorkspaceView workspace) const override {
            std::lock_guard<std::mutex> lock(device_.transfer_mutex_);
            rocm_detail::region_to_host(
                    device_.transfer_resource_->stream(),
                    static_cast<int>(device_.ordinal()), device_,
                    device_.registry_state_, source, workspace, destination,
                    device_.transfer_resource_poisoned_);
        }

        RocmDevice& device_;
        detail::RegistryState* state_;
        void* address_ = nullptr;
    };

    std::unique_ptr<Tensor> RocmDevice::create_tensor(const TensorSpec& spec) {
        // The base Tensor ctor validates the spec before any device
        // interaction; the RocmTensor ctor then suballocates the data arena.
        return std::make_unique<RocmTensor>(spec, *this);
    }

    std::unique_ptr<RawWorkspace> RocmDevice::create_workspace(
            std::size_t bytes) {
        if (bytes == 0) {
            // Valid empty owner: no arena suballocation, no native call.
            return std::make_unique<RocmWorkspace>(*this, nullptr, 0);
        }
        // Positive creation suballocates the already reserved data arena
        // under the allocator bookkeeping boundary; it never reserves a new
        // native backing. Exhaustion, including fragmentation with no fitting
        // contiguous range, is std::bad_alloc with no fallback.
        void* address = allocate_data(bytes);
        try {
            activate();
            validate_native_storage(
                    address, ordinal_, data_backing_, data_backing_bytes_);
            return std::make_unique<RocmWorkspace>(*this, address, bytes);
        } catch (...) {
            release_data(address);
            throw;
        }
    }

}  // namespace iom
