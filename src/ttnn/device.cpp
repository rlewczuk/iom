#include "iom/ttnn/device.hpp"
#include "device_internal.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tile.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_ops.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <optional>
#include <new>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#include "copy.hpp"
#include "registry_state.hpp"
#include "staging.hpp"
#include "testing_internal.hpp"
#include "iom/iom.hpp"

namespace iom {

    namespace {
        using ttnn_detail::carrier_factor;
        using ttnn_detail::checked_plane_count;
        using ttnn_detail::checked_to_uint32;
        using ttnn_detail::is_supported;
        using ttnn_detail::native_dtype;



        // Owner of one TTNN-native tiled tensor per logical plane. Native
        // storage is created and destroyed through the TTNN runtime, never
        // through iom::Allocator; the native 32x32-tile byte count differs
        // from TensorSpec::tiled_storage_nbytes().
        class TtnnTensor final : public Tensor {
        public:
            TtnnTensor(const TensorSpec& spec, TtnnDevice& device)
                    : Tensor(spec, device), device_(device),
                      state_(&device.registry_state()),
                      planes_(
                              std::make_unique<
                                      std::vector<ttnn::Tensor>>()) {
                const std::span<const std::size_t> dimensions =
                        spec.shape.dimensions();
                const std::size_t plane_count = checked_plane_count(spec);
                std::size_t native_columns =
                        dimensions[dimensions.size() - 1];
                const std::size_t factor = carrier_factor(spec.data_type);
                if (native_columns != 0
                        && native_columns
                                > std::numeric_limits<std::size_t>::max()
                                        / factor) {
                    throw std::overflow_error(
                            "TTNN carrier column count overflows");
                }
                native_columns *= factor;
                static_cast<void>(checked_to_uint32(
                        native_columns, dimensions.size() - 1));
                const std::size_t native_rows =
                        (dimensions[dimensions.size() - 2] + 31) / 32 * 32;
                const std::size_t native_padded_columns =
                        (native_columns + 31) / 32 * 32;
                const std::size_t native_bytes =
                        native_dtype(spec.data_type)
                                == tt::tt_metal::DataType::UINT8
                        ? 1
                        : native_dtype(spec.data_type)
                                        == tt::tt_metal::DataType::UINT16
                                || native_dtype(spec.data_type)
                                           == tt::tt_metal::DataType::BFLOAT16
                            ? 2
                            : 4;
                constexpr std::size_t kMaxNativePlaneBytes =
                        std::size_t{1} << 30;
                if (native_rows != 0 && native_padded_columns != 0
                        && native_rows
                                > kMaxNativePlaneBytes
                                        / native_padded_columns
                        || native_rows * native_padded_columns
                                > kMaxNativePlaneBytes / native_bytes) {
                    throw std::bad_alloc();
                }
                const tt::tt_metal::TensorSpec plane_spec(
                        tt::tt_metal::Shape{
                                1u,
                                checked_to_uint32(
                                        dimensions[dimensions.size() - 2],
                                        dimensions.size() - 2),
                                checked_to_uint32(
                                        native_columns,
                                        dimensions.size() - 1)},
                        tt::tt_metal::TensorLayout(
                                native_dtype(spec.data_type),
                                tt::tt_metal::PageConfig(
                                        tt::tt_metal::Layout::TILE,
                                        tt::tt_metal::Tile()),
                                tt::tt_metal::MemoryConfig{}));
                planes_->reserve(plane_count);
                for (std::size_t plane = 0; plane < plane_count; ++plane) {
                    planes_->push_back(ttnn::create_device_tensor(
                            plane_spec, &device_.mesh()));
                }
            }

            ~TtnnTensor() noexcept override {
                const auto quarantine_native = [this]() noexcept {
                    std::unique_ptr<std::vector<ttnn::Tensor>> retained(
                            planes_.release());
                    if (!retained) {
                        return;
                    }
                    try {
#ifdef IOM_ENABLE_TESTING
                        ttnn_detail::consume_quarantine_action_fault_locked();
#endif
                        auto action = std::unique_ptr<
                                ttnn_detail::TtnnNativeCleanupAction>(
                                new ttnn_detail::TtnnNativeCleanupAction(
                                        std::move(*retained),
                                        [device = &device_] {
                                            std::lock_guard<std::mutex> lock(
                                                    device->api_mutex());
                                            device->mesh()
                                                    .mesh_command_queue(0)
                                                    .finish();
                                        }));
                        state_->quarantine.add(std::move(action));
                    } catch (...) {
                        // Keep native storage unreachable if quarantine action
                        // construction or recording fails.
                        (void)retained.release();
                    }
                };
                const auto release_native = [this]() noexcept {
                    std::lock_guard<std::mutex> lock(device_.api_mutex());
                    planes_->clear();
                };
                iom::detail::release_or_quarantine(
                        state_->registry, planes_->data(), quarantine_native,
                        release_native);
            }

        private:
            [[nodiscard]] void* storage_handle() noexcept override {
                return planes_->data();
            }
            [[nodiscard]] WorkspaceRequirements
                    host_transfer_workspace_requirements(
                            std::size_t) const override {
                return {0, 1};
            }

            void region_from_host(
                    const TensorView& destination,
                    std::span<const std::byte> source,
                    RawWorkspaceView) override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                ttnn_detail::region_from_host(
                        device_.mesh(), device_.host_staging(), destination,
                        planes_->data(), source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination,
                    RawWorkspaceView) const override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                ttnn_detail::region_to_host(
                        device_.mesh(), device_.host_staging(), source,
                        planes_->data(), destination);
            }

            TtnnDevice& device_;
            detail::RegistryState* state_;
            std::unique_ptr<std::vector<ttnn::Tensor>> planes_;
        };

        struct HostWorkspaceDeleter {
            std::size_t mapped_bytes = 0;

            void operator()(std::byte* data) const noexcept {
                if (data != nullptr && mapped_bytes != 0) {
                    (void)::munmap(data, mapped_bytes);
#ifdef IOM_ENABLE_TESTING
                    iom::ttnn_test::
                            record_host_workspace_release_for_testing();
#endif
                }
            }
        };

        using HostWorkspaceStorage = std::shared_ptr<std::byte>;

        [[nodiscard]] HostWorkspaceStorage make_host_workspace(
                std::size_t bytes) {
            if (bytes == 0) {
                return {};
            }
            if (bytes
                    > static_cast<std::size_t>(
                              std::numeric_limits<std::uintptr_t>::max())) {
                throw std::overflow_error(
                        "TTNN host workspace size exceeds address range");
            }
            const long system_page_size = ::sysconf(_SC_PAGESIZE);
            if (system_page_size <= 0) {
                throw std::runtime_error(
                        "TTNN host workspace page size is unavailable");
            }
            const std::size_t page_size =
                    static_cast<std::size_t>(system_page_size);
            const std::size_t remainder = bytes % page_size;
            const std::size_t padding = remainder == 0
                    ? 0
                    : page_size - remainder;
            if (padding > std::numeric_limits<std::size_t>::max() - bytes) {
                throw std::overflow_error(
                        "TTNN host workspace mapping size overflows");
            }
            const std::size_t mapped_bytes = bytes + padding;
            void* const mapped = ::mmap(
                    nullptr, mapped_bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mapped == MAP_FAILED) {
                throw std::bad_alloc();
            }
            std::byte* const data = static_cast<std::byte*>(mapped);
            const std::uintptr_t address =
                    reinterpret_cast<std::uintptr_t>(data);
            if (address == 0 || address % 32 != 0) {
                (void)::munmap(data, mapped_bytes);
                throw std::runtime_error(
                        "TTNN host workspace allocation is misaligned");
            }
            if (bytes
                    > std::numeric_limits<std::uintptr_t>::max()
                            - address) {
                (void)::munmap(data, mapped_bytes);
                throw std::overflow_error(
                        "TTNN host workspace address range overflows");
            }
            return HostWorkspaceStorage{
                    data, HostWorkspaceDeleter{mapped_bytes}};
        }
        [[nodiscard]] std::unique_ptr<
                ttnn_detail::HostWorkspaceLeasePayload>
        make_host_retention(const HostWorkspaceStorage& host) {
            if (!host) {
                return {};
            }
            auto payload = std::make_unique<
                    ttnn_detail::HostWorkspaceLeasePayload>();
            payload->keepalive = std::shared_ptr<void>(host);
            return payload;
        }

        [[nodiscard]] std::size_t checked_physical_workspace_size(
                std::size_t bytes) {
            if (bytes > std::numeric_limits<std::size_t>::max() - 31) {
                throw std::overflow_error(
                        "TTNN workspace size round-up overflows");
            }
            const std::size_t physical = (bytes + 31) / 32 * 32;
            if (physical > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                        "TTNN workspace size exceeds the native page size");
            }
            return physical;
        }


        [[nodiscard]] std::shared_ptr<
                tt::tt_metal::distributed::MeshBuffer>
        make_native_workspace(TtnnDevice& device, std::size_t bytes) {
            const std::size_t physical =
                    checked_physical_workspace_size(bytes);
            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer =
                    tt::tt_metal::distributed::MeshBuffer::create(
                            tt::tt_metal::distributed::ReplicatedBufferConfig{
                                    static_cast<tt::tt_metal::DeviceAddr>(
                                            physical)},
                            tt::tt_metal::distributed::DeviceLocalBufferConfig{
                                    .page_size =
                                            static_cast<tt::tt_metal::DeviceAddr>(
                                                    physical),
                                    .buffer_type =
                                            tt::tt_metal::BufferType::DRAM},
                            &device.mesh());
            const tt::tt_metal::Buffer* const reference =
                    buffer == nullptr ? nullptr
                                      : buffer->get_reference_buffer();
            if (buffer == nullptr || buffer->address() == 0
                    || reference == nullptr || reference->alignment() < 32
                    || buffer->device_local_config().page_size != physical
                    || buffer->size() != physical
                    || buffer->num_pages() != 1) {
                throw std::runtime_error(
                        "TTNN workspace native page invariants are not met");
            }
            return buffer;
        }

        // The old native embedding path still asks for a status MeshBuffer.
        // Keep that compatibility allocation lazy: creating a positive raw
        // workspace owns only its exact host range, while raw cache transfers
        // never manufacture a native backing buffer.
        class TtnnWorkspace final : public RawWorkspace {
            using RetentionCell =
                    std::shared_ptr<
                            tt::tt_metal::distributed::MeshBuffer>;

        public:
            TtnnWorkspace(TtnnDevice& device, std::size_t bytes)
                    : RawWorkspace(device, bytes), device_(device),
                      state_(&device.registry_state()),
                      host_(make_host_workspace(bytes)),
                      host_retention_(make_host_retention(host_)) {}

            ~TtnnWorkspace() noexcept override {
                const auto quarantine_host = [this]() noexcept {
                    if (host_retention_) {
                        device_.retain_host_workspace(
                                std::move(host_retention_));
                    }
                };
                const auto quarantine_native = [this]() noexcept {
                    if (!native_) {
                        return;
                    }
                    *retention_ = std::move(native_);
                    if (!*retention_) {
                        return;
                    }
                    try {
#ifdef IOM_ENABLE_TESTING
                        ttnn_detail::consume_quarantine_action_fault_locked();
#endif
                        auto action = std::unique_ptr<
                                ttnn_detail::TtnnNativeCleanupAction>(
                                new ttnn_detail::TtnnNativeCleanupAction(
                                        *retention_,
                                        [device = &device_] {
                                            std::lock_guard<std::mutex> lock(
                                                    device->api_mutex());
                                            device->mesh()
                                                    .mesh_command_queue(0)
                                                    .finish();
                                        }));
                        state_->quarantine.add(std::move(action));
                        retention_->reset();
                    } catch (...) {
                        (void)retention_.release();
                    }
                };
                const auto release_native = [this]() noexcept {
                    if (!native_) {
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> lock(
                                device_.api_mutex());
                        native_.reset();
                    }
                };
                if (native_range_retained()) {
                    quarantine_host();
                    quarantine_native();
                    return;
                }
                if (!native_) {
                    return;
                }
                iom::detail::release_or_quarantine(
                        state_->registry, workspace_address(),
                        quarantine_native, release_native);
            }

            [[nodiscard]] std::byte* host_data() const noexcept {
                return host_.get();
            }

            [[nodiscard]] std::shared_ptr<void> host_keepalive()
                    const noexcept {
                return std::shared_ptr<void>(host_);
            }

            // Compatibility access for the existing TTNN embedding kernel.
            // This path is intentionally lazy and is never used by the raw
            // host-plane transfer helpers.
            [[nodiscard]] tt::tt_metal::distributed::MeshBuffer* native_owner()
                    const {
                ensure_native();
                return native_.get();
            }

            [[nodiscard]] std::shared_ptr<
                    tt::tt_metal::distributed::MeshBuffer>
            native_owner_handle() const {
                ensure_native();
                return native_;
            }

            [[nodiscard]] std::uint64_t native_page_size() const {
                ensure_native();
                return native_ != nullptr
                        ? native_->device_local_config().page_size
                        : 0;
            }

        protected:
            [[nodiscard]] void* workspace_address() const noexcept override {
                return host_.get();
            }

        private:
            void ensure_native() const {
                if (native_ != nullptr || byte_size() == 0) {
                    return;
                }
                TtnnWorkspace* const self =
                        const_cast<TtnnWorkspace*>(this);
                self->retention_ = std::make_unique<RetentionCell>();
                self->native_ = make_native_workspace(
                        device_, byte_size());
            }

            [[nodiscard]] bool native_range_retained() noexcept {
                try {
                    std::lock_guard<std::mutex> lock(
                            state_->allocation_mutex);
                    return iom::detail::workspace_range_retained(
                            state_->workspace_leases, this,
                            workspace_address(), byte_size());
                } catch (...) {
                    return true;
                }
            }
            TtnnDevice& device_;
            detail::RegistryState* state_;
            HostWorkspaceStorage host_;
            std::unique_ptr<ttnn_detail::HostWorkspaceLeasePayload>
                    host_retention_;
            mutable std::shared_ptr<
                    tt::tt_metal::distributed::MeshBuffer>
                    native_;
            mutable std::unique_ptr<RetentionCell> retention_;
        };
    }  // namespace
    TtnnDevice::TtnnDevice(
            std::uint32_t ordinal,
            std::shared_ptr<ttnn::MeshDevice> native_device,
            QueueConfig queue_config)
        : Device(queue_config), ordinal_(ordinal),
          native_device_(std::move(native_device)) {}

    TtnnDevice::~TtnnDevice() {
        registry_state_.quarantine.drain();
    }

    BackendKind TtnnDevice::backend_kind() const noexcept {
        return BackendKind::TTNN;
    }

    std::uint32_t TtnnDevice::backend_device() const noexcept {
        return ordinal_;
    }

    std::span<const DataType> TtnnDevice::supported_data_types()
            const noexcept {
        return ttnn_supported_data_types();
    }

    tt::tt_metal::distributed::MeshDevice& TtnnDevice::mesh() noexcept {
        return *native_device_;
    }

    std::mutex& TtnnDevice::api_mutex() noexcept {
        return api_mutex_;
    }

    ttnn_detail::TtnnHostStaging& TtnnDevice::host_staging() noexcept {
        return host_staging_;
    }

    detail::RegistryState& TtnnDevice::registry_state() noexcept {
        return registry_state_;
    }

    void TtnnDevice::retain_host_workspace(
            std::unique_ptr<ttnn_detail::HostWorkspaceLeasePayload> payload)
            noexcept {
        if (!payload) {
            return;
        }
        try {
#ifdef IOM_ENABLE_TESTING
            ttnn_detail::consume_quarantine_action_fault_locked();
#endif
            auto action = std::make_unique<ttnn_detail::TtnnHostCleanupAction>(
                    std::move(payload),
                    [device = this] {
                        std::lock_guard<std::mutex> lock(
                                device->api_mutex());
                        device->mesh().mesh_command_queue(0).finish();
                    });
            registry_state_.quarantine.add(std::move(action));
        } catch (...) {
            // No safe cleanup record can be constructed.  Keep the pin and
            // allocation permanently retained rather than releasing bytes
            // that an in-flight queue may still access.
            (void)payload.release();
        }
    }



    std::unique_ptr<Tensor> TtnnDevice::create_tensor(const TensorSpec& spec) {
        spec.validate();
        if (!is_supported(spec.data_type)) {
            throw std::runtime_error(
                    "TTNN backend does not support the requested DataType");
        }
        std::lock_guard<std::mutex> lock(api_mutex_);
        return std::make_unique<TtnnTensor>(spec, *this);
    }

    std::unique_ptr<RawWorkspace> TtnnDevice::create_workspace(
            std::size_t bytes) {
        if (bytes == 0) {
            // The empty owner allocates nothing and touches no native
            // allocation.
            return std::make_unique<TtnnWorkspace>(*this, 0);
        }
        // Positive requests own one checked, aligned host range for the
        // owner lifetime.  The native DRAM page exists only for the lazy
        // compatibility embedding path and raw host-plane transfers never
        // create it.
        std::lock_guard<std::mutex> lock(api_mutex_);
        return std::make_unique<TtnnWorkspace>(*this, bytes);
    }

    ttnn_detail::HostWorkspace ttnn_detail::checked_host_workspace(
            TtnnDevice& device, const RawWorkspaceView& workspace) {
        if (workspace.empty()) {
            throw std::invalid_argument("workspace view is empty");
        }
        const RawWorkspace* const owner = workspace.owner_identity();
        if (owner == nullptr || !device.owns_workspace(owner)) {
            throw std::invalid_argument(
                    "workspace owner is not live on this device");
        }
        if (&workspace.device() != &device) {
            throw std::invalid_argument(
                    "workspace does not belong to this device");
        }
        const auto* const workspace_owner =
                dynamic_cast<const TtnnWorkspace*>(owner);
        if (workspace_owner == nullptr
                || workspace_owner->host_data() == nullptr) {
            throw std::invalid_argument(
                    "workspace owner has no positive host allocation");
        }
        const std::size_t logical_bytes = workspace_owner->byte_size();
        const std::size_t offset = workspace.offset();
        const std::size_t bytes = workspace.byte_size();
        if (offset % 32 != 0) {
            throw std::invalid_argument(
                    "workspace offset is not 32-byte aligned");
        }
        if (bytes == 0) {
            throw std::invalid_argument("workspace range is empty");
        }
        if (offset > logical_bytes || bytes > logical_bytes - offset) {
            throw std::invalid_argument(
                    "workspace range exceeds the owner's logical bytes");
        }
        const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(
                workspace_owner->host_data());
        if (base == 0 || base % 32 != 0) {
            throw std::runtime_error(
                    "TTNN host workspace base is null or misaligned");
        }
        if (offset > std::numeric_limits<std::uintptr_t>::max() - base) {
            throw std::overflow_error(
                    "TTNN host workspace offset overflows");
        }
        const std::uintptr_t range_begin = base + offset;
        if (bytes
                > std::numeric_limits<std::uintptr_t>::max()
                        - range_begin) {
            throw std::overflow_error(
                    "TTNN host workspace range end overflows");
        }
        return ttnn_detail::HostWorkspace{
                reinterpret_cast<std::byte*>(range_begin), bytes,
                workspace_owner->host_keepalive()};
    }

    ttnn_detail::NativeWorkspace ttnn_detail::checked_native_workspace(
            TtnnDevice& device, const RawWorkspaceView& workspace) {
        if (workspace.empty()) {
            throw std::invalid_argument(
                    "workspace view is empty");
        }
        const RawWorkspace* const owner = workspace.owner_identity();
        if (owner == nullptr || !device.owns_workspace(owner)) {
            // Foreign or already-destroyed owners are rejected by identity;
            // a dead owner is never dereferenced.
            throw std::invalid_argument(
                    "workspace owner is not live on this device");
        }
        if (&workspace.device() != &device) {
            throw std::invalid_argument(
                    "workspace does not belong to this device");
        }
        const auto* const workspace_owner =
                dynamic_cast<const TtnnWorkspace*>(owner);
        if (workspace_owner == nullptr
                || workspace_owner->native_owner() == nullptr) {
            throw std::invalid_argument(
                    "workspace owner has no positive native allocation");
        }
        const std::size_t logical_bytes = workspace_owner->byte_size();
        const std::size_t offset = workspace.offset();
        const std::size_t bytes = workspace.byte_size();
        if (offset % 32 != 0) {
            throw std::invalid_argument(
                    "workspace offset is not 32-byte aligned");
        }
        if (bytes == 0) {
            throw std::invalid_argument("workspace range is empty");
        }
        if (offset > logical_bytes || bytes > logical_bytes - offset) {
            throw std::invalid_argument(
                    "workspace range exceeds the owner's logical bytes");
        }
        if (offset > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                    "workspace offset exceeds the native address range");
        }
        tt::tt_metal::distributed::MeshBuffer* const native =
                workspace_owner->native_owner();
        if (native->address() == 0
                || workspace_owner->native_page_size() == 0) {
            throw std::runtime_error(
                    "TTNN workspace native owner has no address or page");
        }
        return ttnn_detail::NativeWorkspace{
                native,
                workspace_owner->native_owner_handle(),
                workspace_owner->native_page_size(),
                static_cast<std::uint64_t>(native->address()),
                logical_bytes,
                offset};


    }

    std::unique_ptr<Device> make_ttnn_device(
            std::uint32_t device_ordinal, QueueConfig queue_config) {
        const std::size_t device_count =
                tt::tt_metal::GetNumAvailableDevices();
        if (static_cast<std::size_t>(device_ordinal) >= device_count
                || device_ordinal
                        > static_cast<std::uint32_t>(
                                std::numeric_limits<int>::max())) {
            throw ttnn_detail::invalid_ordinal(
                    device_ordinal, device_count);
        }

        return std::make_unique<TtnnDevice>(
                device_ordinal,
                ttnn::open_mesh_device(static_cast<int>(device_ordinal)),
                queue_config);
    }

}  // namespace iom
