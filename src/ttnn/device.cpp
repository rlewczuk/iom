#include "iom/ttnn/device.hpp"
#include "queue_internal.hpp"

#include <tt-metalium/host_api.hpp>
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

        /**
         * TTNN raw-workspace owner. Only the zero-byte empty workspace
         * exists: the base RawWorkspace registers the exact identity with
         * the device and no native allocation is ever touched.
         */
        class TtnnWorkspace final : public RawWorkspace {
        public:
            TtnnWorkspace(TtnnDevice& device, std::size_t bytes)
                    : RawWorkspace(device, bytes) {}
        };
    }  // namespace
    void TtnnQueue::execute_copy(Task& task) {
        const detail::EntryRegistration entries{
                task.source_entry_id, task.destination_entry_id};
        try {
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
#ifdef IOM_ENABLE_TESTING
                ttnn_detail::consume_copy_outcome_insertion_fault();
#endif
                const auto [it, inserted] = outcomes_.emplace(
                        task.sequence,
                        detail::SequenceOutcome{
                                entries.source, entries.destination,
                                nullptr});
                if (!inserted) {
                    throw std::logic_error(
                            "duplicate TTNN outstanding-work sequence");
                }
            }
        } catch (...) {
            state_->registry.remove_entry_if_present(
                    entries.source,
                    task.source->native_handle);
            state_->registry.remove_entry_if_present(
                    entries.destination,
                    task.destination->native_handle);
            throw;
        }

        if (task.no_op) {
            // An identical-window copy submits no native work. Its
            // registered entries are released by complete_task without a
            // device-wide finish.
            executed_seq_.store(task.sequence, std::memory_order_release);
            return;
        }

        std::lock_guard<std::mutex> api_lock(device_->api_mutex());
        bool any_submitted = false;
        std::exception_ptr submission_failure;
        try {
            const ttnn::Tensor* source_planes =
                    static_cast<const ttnn::Tensor*>(
                            task.source->native_handle);
            ttnn::Tensor* destination_planes =
                    static_cast<ttnn::Tensor*>(
                            task.destination->native_handle);
            ttnn_detail::copy_planes(
                    *task.source, source_planes,
                    *task.destination, destination_planes,
                    any_submitted);
        } catch (...) {
            submission_failure = std::current_exception();
        }
        if (!submission_failure) {
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                outcomes_.at(task.sequence).native_work_submitted = true;
            }
            // Every plane is submitted and owned; complete_task performs one
            // mesh finish per ready batch of contiguous executed tasks.
            executed_seq_.store(task.sequence, std::memory_order_release);
            return;
        }


        if (!any_submitted) {
            // The failure struck before the first plane reached the mesh:
            // nothing is pending, so ownership rolls back and the exception
            // propagates synchronously with no device work outstanding.
            state_->registry.remove_entry_if_present(
                    entries.source,
                    task.source->native_handle);
            state_->registry.remove_entry_if_present(
                    entries.destination,
                    task.destination->native_handle);
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                outcomes_.erase(task.sequence);
            }
            std::rethrow_exception(submission_failure);
        }
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            outcomes_.at(task.sequence).native_work_submitted = true;
        }


        // One or more planes reached the mesh. Drain them synchronously
        // under the API mutex before ownership is removed, so a thrown call
        // has established the terminal result of every submitted command
        // before the owners can be destroyed or reused.
        try {
            ttnn_detail::finish_locked(*device_);
        } catch (...) {
            // The mesh cannot be drained. Retain the failed fenced sequence:
            // the task is published normally, so the caller receives a
            // waitable token whose waits report the retained submission
            // failure, and the registered entries keep protecting the planes
            // until complete_task's finish retry (or the quarantine drain
            // when the owners are destroyed) establishes the terminal result.
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto it = outcomes_.find(task.sequence);
            if (it != outcomes_.end()) {
                it->second.retained_failure =
                        std::move(submission_failure);
            }
            executed_seq_.store(task.sequence, std::memory_order_release);
            return;
        }
        state_->registry.remove_entry_if_present(
                entries.source,
                task.source->native_handle);
        state_->registry.remove_entry_if_present(
                entries.destination,
                task.destination->native_handle);
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            outcomes_.erase(task.sequence);
        }
        std::rethrow_exception(submission_failure);
    }
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
        if (bytes != 0) {
            // Positive scratch is unsupported device storage here: no
            // dummy native storage is manufactured and no TTNN allocation
            // is touched.
            throw std::invalid_argument(
                    "TTNN devices do not support positive raw workspace "
                    "allocation");
        }
        return std::make_unique<TtnnWorkspace>(*this, 0);
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
