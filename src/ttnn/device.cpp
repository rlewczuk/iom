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

        // Checked round-up of one positive logical request to the native
        // 32-byte page granularity.
        [[nodiscard]] std::size_t checked_physical_workspace_size(
                std::size_t bytes) {
            if (bytes > std::numeric_limits<std::size_t>::max() - 31) {
                throw std::overflow_error(
                        "TTNN workspace size round-up overflows");
            }
            const std::size_t physical = (bytes + 31) / 32 * 32;
            if (physical > std::numeric_limits<std::uint32_t>::max()) {
                // The native page size is a 32-bit field, so a larger
                // request cannot be one contiguous native page.
                throw std::overflow_error(
                        "TTNN workspace size exceeds the native page size");
            }
            return physical;
        }

        // One owning replicated DRAM native page for a positive request:
        // the whole physically rounded request is a single contiguous page,
        // never an interleaved multi-page allocation and never the
        // non-owning explicit-address form.
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
            if (buffer == nullptr || buffer->address() == 0) {
                throw std::runtime_error(
                        "TTNN workspace has no native address");
            }
            if (reference == nullptr || reference->alignment() < 32) {
                throw std::runtime_error(
                        "TTNN workspace native alignment is below 32 bytes");
            }
            if (buffer->device_local_config().page_size != physical
                    || buffer->size() != physical
                    || buffer->num_pages() != 1) {
                throw std::runtime_error(
                        "TTNN workspace native page invariants are not met");
            }
            iom::ttnn_test::record_native_workspace_allocation_for_testing();
            return buffer;
        }

        /**
         * TTNN raw-workspace owner. A positive request owns exactly one
         * replicated DRAM `MeshBuffer` on the existing unit mesh: one
         * contiguous native page whose page size is the caller's logical
         * bytes rounded up to 32. `byte_size()` stays the caller-requested
         * logical bytes, and the native allocation is released only through
         * the workspace lease and the registry completion proof; unproved
         * use retains the real allocation in the device quarantine, or
         * deliberately holds on to it when no quarantine record can be
         * created. The zero-byte owner keeps the empty semantics and touches
         * no native allocation.
         */
        class TtnnWorkspace final : public RawWorkspace {
            // The storage an unproved native owner is handed to when no
            // quarantine record can be created. It is heap-allocated and
            // reserved while construction is still fail-safe, so the
            // `std::shared_ptr` handover is a noexcept move and the cell
            // itself can be deliberately leaked instead of returning the
            // real `MeshBuffer` to TTNN.
            using RetentionCell =
                    std::shared_ptr<
                            tt::tt_metal::distributed::MeshBuffer>;

        public:
            TtnnWorkspace(TtnnDevice& device, std::size_t bytes)
                    : RawWorkspace(device, bytes), device_(device),
                      state_(&device.registry_state()) {
                if (bytes != 0) {
                    // The retention cell is reserved before the native
                    // allocation, while construction can still fail
                    // cleanly: every owner that can hold native work then
                    // has allocation-free storage to be retained in.
                    retention_ = std::make_unique<RetentionCell>();
                    native_ = make_native_workspace(device, bytes);
                }
            }

            ~TtnnWorkspace() noexcept override {
                if (!native_) {
                    return;
                }
                const auto quarantine_native = [this]() noexcept {
                    // First step retains the owning reference in the cell
                    // reserved by the constructor: a `std::shared_ptr` move
                    // cannot allocate and cannot throw, so from here on the
                    // real `MeshBuffer` survives every failure below instead
                    // of being returned to TTNN without a completion proof.
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
                        // The recorded quarantine action is the owner now,
                        // so the cell's reference is dropped: the native
                        // allocation is released only once that action runs
                        // behind a completed proof.
                        retention_->reset();
                    } catch (...) {
                        // No quarantine record can be created: the cell is
                        // deliberately leaked, exactly like the tensor
                        // path's retained payload, so an unproved native
                        // owner is never returned to TTNN.
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
                    iom::ttnn_test::
                            record_native_workspace_release_for_testing();
                };
                if (native_range_retained()) {
                    // A live or quarantined lease of this owner is not a
                    // completion proof: the real native allocation stays
                    // retained instead of being released early.
                    quarantine_native();
                    return;
                }
                iom::detail::release_or_quarantine(
                        state_->registry, workspace_address(),
                        quarantine_native, release_native);
            }

            [[nodiscard]] tt::tt_metal::distributed::MeshBuffer* native_owner()
                    const noexcept {
                return native_.get();
            }

            [[nodiscard]] std::shared_ptr<
                    tt::tt_metal::distributed::MeshBuffer>
            native_owner_handle() const noexcept {
                return native_;
            }

            [[nodiscard]] std::uint64_t native_page_size() const noexcept {
                return native_ != nullptr
                        ? native_->device_local_config().page_size
                        : 0;
            }

        protected:
            [[nodiscard]] void* workspace_address() const noexcept override {
                return native_ != nullptr
                        ? reinterpret_cast<void*>(
                                  static_cast<std::uintptr_t>(
                                          native_->address()))
                        : nullptr;
            }

        private:
            // True while a live or quarantined lease of this owner retains
            // any range, whether or not a registry entry is still present.
            // Bookkeeping that cannot be read retains conservatively.
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
            std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> native_;
            // Reserved at construction; its heap cell is deliberately
            // leaked when `native_` cannot be recorded in the quarantine.
            std::unique_ptr<RetentionCell> retention_;
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
        // Positive requests own one replicated DRAM native page; creation
        // is serialized with every other TTNN runtime call of this device.
        std::lock_guard<std::mutex> lock(api_mutex_);
        return std::make_unique<TtnnWorkspace>(*this, bytes);
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
