#include "iom/ttnn/device.hpp"

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
#include <memory>
#include <new>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "copy.hpp"
#include "registry_state.hpp"
#include "iom/iom.hpp"

namespace iom {

    namespace {
#ifdef IOM_ENABLE_TESTING
        std::atomic<bool> g_fail_next_quarantine_action{false};
        bool g_quarantine_action_fault_consumed = false;

        void consume_quarantine_action_fault_locked() noexcept(false) {
            if (g_fail_next_quarantine_action.exchange(
                        false, std::memory_order_acquire)
                    && !g_quarantine_action_fault_consumed) {
                g_quarantine_action_fault_consumed = true;
                throw std::bad_alloc();
            }
        }
#endif


        // The one explicit TTNN supported-type table: every leaf type whose
        // host encoding TTNN tiled storage reproduces bit-for-bit. Signed
        // 8/16-bit integers ride the unsigned tiles of the same width
        // (two's-complement fields are bit-identical); BOOL rides UINT8 with
        // canonical zero/one bytes validated by the common TensorView base.
        constexpr auto kSupportedToNative = std::array{
                std::pair{
                        DataType::BOOL, tt::tt_metal::DataType::UINT8},
                std::pair{DataType::U8, tt::tt_metal::DataType::UINT8},
                std::pair{DataType::I8, tt::tt_metal::DataType::UINT8},
                std::pair{
                        DataType::U16, tt::tt_metal::DataType::UINT16},
                std::pair{
                        DataType::I16, tt::tt_metal::DataType::UINT16},
                std::pair{DataType::U32, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::I32, tt::tt_metal::DataType::INT32},
                std::pair{
                        DataType::BF16,
                        tt::tt_metal::DataType::BFLOAT16},
                std::pair{DataType::F32, tt::tt_metal::DataType::FLOAT32},
        };

        constexpr auto kSupportedKeys = [] {
            std::array<DataType, kSupportedToNative.size()> keys{};
            for (std::size_t i = 0; i < kSupportedToNative.size(); ++i) {
                keys[i] = kSupportedToNative[i].first;
            }
            return keys;
        }();

        static_assert(kSupportedKeys.size() == kSupportedToNative.size());

        constexpr bool kSupportedKeysUnique = [] {
            for (std::size_t i = 0; i < kSupportedToNative.size(); ++i) {
                for (std::size_t j = i + 1; j < kSupportedToNative.size();
                     ++j) {
                    if (kSupportedToNative[i].first
                            == kSupportedToNative[j].first) {
                        return false;
                    }
                }
            }
            return true;
        }();
        static_assert(kSupportedKeysUnique);

        [[nodiscard]] bool is_supported(DataType type) {
            const std::span<const DataType> supported = kSupportedKeys;
            return std::find(supported.begin(), supported.end(), type)
                   != supported.end();
        }

        // Native tile dtype carrying the leaf encoding bit-for-bit.
        [[nodiscard]] tt::tt_metal::DataType native_dtype(DataType type) {
            for (const auto& [supported_type, native_type] :
                 kSupportedToNative) {
                if (supported_type == type) {
                    return native_type;
                }
            }
            throw std::invalid_argument(
                    "DataType has no TTNN native tile dtype");
        }

        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, std::size_t device_count) {
            return std::invalid_argument(
                    "TTNN device ordinal " + std::to_string(ordinal)
                    + " is unavailable; device count is "
                    + std::to_string(device_count));
        }

        // Checked narrowing of a TTNN native extent. tt::tt_metal::Shape
        // stores every dimension as uint32_t, so a logical dimension beyond
        // that ceiling is unreachable on the native path and must be
        // rejected before any native object is constructed.
        [[nodiscard]] std::uint32_t checked_to_uint32(
                std::size_t dimension, std::size_t index) {
            constexpr std::size_t kMaxNativeExtent =
                    std::numeric_limits<std::uint32_t>::max();
            if (dimension > kMaxNativeExtent) {
                throw std::overflow_error(
                        "TTNN native dimension " + std::to_string(index)
                        + " (" + std::to_string(dimension)
                        + ") exceeds the uint32_t native extent limit "
                        + std::to_string(kMaxNativeExtent));
            }
            return static_cast<std::uint32_t>(dimension);
        }

        // Checked leading-plane count for the TTNN creation path: validates
        // the final two dimensions against the native extent ceiling and
        // multiplies the leading dimensions with the checked-multiplication
        // pattern of iom::detail::checked_mul, throwing
        // std::overflow_error on the first wrap. Pure; runs before any
        // TTNN native object or allocation exists.
        [[nodiscard]] std::size_t checked_plane_count(const TensorSpec& spec) {
            const std::span<const std::size_t> dimensions =
                    spec.shape.dimensions();
            static_cast<void>(checked_to_uint32(
                    dimensions[dimensions.size() - 2],
                    dimensions.size() - 2));
            static_cast<void>(checked_to_uint32(
                    dimensions[dimensions.size() - 1],
                    dimensions.size() - 1));
            std::size_t plane_count = 1;
            for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                const std::size_t dimension = dimensions[i];
                if (dimension != 0
                        && plane_count
                                > std::numeric_limits<std::size_t>::max()
                                        / dimension) {
                    throw std::overflow_error(
                            "TTNN leading-plane count overflows at dimension "
                            + std::to_string(i));
                }
                plane_count *= dimension;
            }
            return plane_count;
        }

        class TtnnDevice final : public Device {
        public:
            TtnnDevice(
                    std::uint32_t ordinal,
                    std::shared_ptr<ttnn::MeshDevice> native_device)
                    : ordinal_(ordinal), native_device_(std::move(native_device)) {}

            TtnnDevice(const TtnnDevice&) = delete;
            TtnnDevice& operator=(const TtnnDevice&) = delete;

            ~TtnnDevice() override {
                registry_state_.quarantine.drain();
            }

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::TTNN;
            }

            [[nodiscard]] std::uint32_t backend_device() const noexcept override {
                return ordinal_;
            }
            [[nodiscard]] std::span<const iom::DataType>
                    supported_data_types() const noexcept override {
                return ttnn_supported_data_types();
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec& spec) override;

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

            [[nodiscard]] tt::tt_metal::distributed::MeshDevice& mesh()
                    noexcept {
                return *native_device_;
            }

            [[nodiscard]] std::mutex& api_mutex() noexcept {
                return api_mutex_;
            }

            [[nodiscard]] ttnn_detail::TtnnRegistryState&
                    registry_state() noexcept {
                return registry_state_;
            }

        private:
            std::uint32_t ordinal_;
            std::shared_ptr<ttnn::MeshDevice> native_device_;
            std::mutex api_mutex_;
            ttnn_detail::TtnnRegistryState registry_state_;
        };

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
                const tt::tt_metal::TensorSpec plane_spec(
                        tt::tt_metal::Shape{
                                1u,
                                checked_to_uint32(
                                        dimensions[dimensions.size() - 2],
                                        dimensions.size() - 2),
                                checked_to_uint32(
                                        dimensions[dimensions.size() - 1],
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
                const void* original_address = planes_->data();
                const auto quarantine_native = [this]() noexcept {
                    std::unique_ptr<std::vector<ttnn::Tensor>> retained(
                            planes_.release());
                    if (!retained) {
                        return;
                    }
                    try {
#ifdef IOM_ENABLE_TESTING
                        consume_quarantine_action_fault_locked();
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

                std::vector<
                        detail::OutstandingWorkRegistry::EntrySnapshot>
                        snapshots;
                try {
                    snapshots = state_->registry.snapshot_for(
                            const_cast<void*>(original_address));
                } catch (...) {
                    quarantine_native();
                    return;
                }

                bool safe_to_release = true;
                for (const auto& snapshot : snapshots) {
                    if (snapshot.state == detail::EntryState::Invalidated
                            || !snapshot.fence) {
                        safe_to_release = false;
                        continue;
                    }
                    try {
                        const detail::FenceResult result = snapshot.fence();
                        safe_to_release = safe_to_release
                                && result.succeeded && !result.failure;
                    } catch (...) {
                        safe_to_release = false;
                    }
                }

                if (safe_to_release) {
                    for (const auto& snapshot : snapshots) {
                        state_->registry.remove_entry_if_present(
                                snapshot.id,
                                const_cast<void*>(original_address));
                    }
                    std::lock_guard<std::mutex> lock(device_.api_mutex());
                    planes_->clear();
                    return;
                }

                quarantine_native();
                for (const auto& snapshot : snapshots) {
                    state_->registry.remove_entry_if_present(
                            snapshot.id,
                            const_cast<void*>(original_address));
                }
            }

        private:
            [[nodiscard]] void* storage_handle() noexcept override {
                return planes_->data();
            }

            void region_from_host(
                    const TensorView& destination,
                    std::span<const std::byte> source) override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                ttnn_detail::region_from_host(
                        device_.mesh(), destination, planes_->data(), source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                ttnn_detail::region_to_host(
                        device_.mesh(), source, planes_->data(), destination);
            }

            TtnnDevice& device_;
            ttnn_detail::TtnnRegistryState* state_;
            std::unique_ptr<std::vector<ttnn::Tensor>> planes_;
        };

        detail::Fence make_fence(TtnnDevice& device) {
            TtnnDevice* stable_device = &device;
            return [stable_device]() noexcept {
                try {
                    std::lock_guard<std::mutex> lock(
                            stable_device->api_mutex());
                    stable_device->mesh().mesh_command_queue(0).finish();
                    return detail::FenceResult::success();
                } catch (...) {
                    return detail::FenceResult::failed(
                            std::current_exception());
                }
            };
        }

        detail::FenceResult finish_native(TtnnDevice& device) noexcept {
            try {
                std::lock_guard<std::mutex> lock(device.api_mutex());
                device.mesh().mesh_command_queue(0).finish();
                return detail::FenceResult::success();
            } catch (...) {
                return detail::FenceResult::failed(
                        std::current_exception());
            }
        }

class TtnnQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        TensorView source;
        TensorView destination;
        bool no_op;
        void* fence = nullptr;
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;

        Task(std::uint64_t sequence_,
             const TensorView& source_,
             TensorView& destination_,
             bool no_op_)
            : sequence(sequence_),
              source(source_),
              destination(destination_),
              no_op(no_op_) {}
    };

    struct SequenceOutcome {
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
        bool fence_succeeded = false;
        std::exception_ptr retained_failure;
    };

public:
    explicit TtnnQueue(TtnnDevice& device)
            : device_(&device),
              state_(&device.registry_state()),
              registry_queue_id_(ttnn_detail::allocate_queue_id(*state_)),
              worker_(
                      detail::StagedWorker<Task>::Callbacks{
                              [this](Task& task) {
                                  execute(task);
                              },
                              [](void*) {},
                              [](void*) {},
                              [this](
                                      std::uint64_t sequence,
                                      std::exception_ptr failure) {
                                  complete_task(
                                          sequence, std::move(failure));
                              }},
                      detail::StagedWorker<Task>::PublishPolicy::
                              CompleteOnThrow) {
        worker_.start();
    }

    ~TtnnQueue() override {
        state_->registry.invalidate_entries_for_queue(registry_queue_id_);
        worker_.shutdown_and_drain();
    }

    oid copy(
            const TensorView& source,
            TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        validate_copy(*device_, source, destination);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    Task task(sequence, source, destination, no_op);
                    worker_.submit_copy(std::move(task));
                });
    }

    oid add(const TensorView&, const TensorView&, TensorView&)
            override {
        throw unsupported("TTNN", "add");
    }

    oid mul(const TensorView&, const TensorView&, TensorView&)
            override {
        throw unsupported("TTNN", "mul");
    }

    oid silu(const TensorView&, TensorView&) override {
        throw unsupported("TTNN", "silu");
    }

    oid linear(const TensorView&, const TensorView&, TensorView&)
            override {
        throw unsupported("TTNN", "linear");
    }

    oid rmsnorm(
            const TensorView&, TensorView&, const TensorView&, float,
            size_t) override {
        throw unsupported("TTNN", "rmsnorm");
    }

    oid sdpa(
            const TensorView&, const TensorView&, const TensorView&,
            size_t, size_t, size_t, TensorView&) override {
        throw unsupported("TTNN", "sdpa");
    }

private:
    void execute(Task& task) {
        if (!task.no_op) {
            std::lock_guard<std::mutex> api_lock(device_->api_mutex());
            const ttnn::Tensor* source_planes =
                    static_cast<const ttnn::Tensor*>(
                            task.source.native_handle());
            ttnn::Tensor* destination_planes =
                    static_cast<ttnn::Tensor*>(
                            task.destination.native_handle());
            ttnn_detail::copy_planes(
                    task.source, source_planes,
                    task.destination, destination_planes);
        }

        const detail::Fence fence = make_fence(*device_);
        detail::EntryRegistration entries;
        try {
            entries = ttnn_detail::register_copy_entries(
                    *state_, registry_queue_id_, task.sequence,
                    task.source.native_handle(),
                    task.destination.native_handle(), fence);
            task.source_entry_id = entries.source;
            task.destination_entry_id = entries.destination;
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto [it, inserted] = outcomes_.emplace(
                    task.sequence,
                    SequenceOutcome{
                            entries.source, entries.destination, true,
                            nullptr});
            if (!inserted) {
                throw std::logic_error(
                        "duplicate TTNN outstanding-work sequence");
            }
        } catch (...) {
            if (entries.source != 0) {
                state_->registry.remove_entry_if_present(
                        entries.source, task.source.native_handle());
            }
            if (entries.destination != 0) {
                state_->registry.remove_entry_if_present(
                        entries.destination,
                        task.destination.native_handle());
            }
            throw;
        }
    }

    void complete_task(
            std::uint64_t sequence, std::exception_ptr failure) {
        SequenceOutcome outcome;
        bool has_outcome = false;
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto it = outcomes_.find(sequence);
            if (it != outcomes_.end()) {
                outcome = std::move(it->second);
                outcomes_.erase(it);
                has_outcome = true;
            }
        }
        if (has_outcome) {
            detail::FenceResult fence_result =
                    failure ? detail::FenceResult::failed(failure)
                            : finish_native(*device_);
            outcome.fence_succeeded =
                    fence_result.succeeded && !fence_result.failure;
            const std::array<detail::EntryId, 2> entries{
                    outcome.source_entry_id,
                    outcome.destination_entry_id};
            if (!outcome.fence_succeeded) {
                state_->registry.invalidate_entries(entries);
                if (!failure) {
                    failure = fence_result.failure;
                }
            } else {
                (void)state_->registry.try_release_entry(entries[0]);
                (void)state_->registry.try_release_entry(entries[1]);
                last_finished_seq_.store(
                        sequence, std::memory_order_release);
            }
        }
        complete(sequence, std::move(failure));
    }

    TtnnDevice* device_;
    ttnn_detail::TtnnRegistryState* state_;
    detail::QueueId registry_queue_id_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, SequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
    std::mutex fence_mutex_;
    std::atomic<std::uint64_t> last_finished_seq_{0};
    void fence_through_sequence(
            std::uint64_t sequence) noexcept override;
};
        void TtnnQueue::fence_through_sequence(
                std::uint64_t sequence) noexcept {
            std::lock_guard<std::mutex> fence_lock(fence_mutex_);
            const std::uint64_t now =
                    last_finished_seq_.load(std::memory_order_acquire);
            if (sequence <= now) {
                return;
            }
            const detail::FenceResult result =
                    finish_native(*device_);
            if (result.succeeded && !result.failure) {
                last_finished_seq_.store(
                        sequence, std::memory_order_release);
            } else {
                record_post_completion_failure(
                        sequence, result.failure);
            }
        }


    }  // namespace

#ifdef IOM_ENABLE_TESTING
    namespace ttnn_test {
        void fail_next_quarantine_action_for_testing() noexcept {
            g_fail_next_quarantine_action.store(
                    true, std::memory_order_release);
        }

        bool quarantine_action_fault_consumed_for_testing() noexcept {
            return g_quarantine_action_fault_consumed;
        }
    }  // namespace ttnn_test
#endif

    std::span<const DataType> ttnn_supported_data_types() noexcept {
        return {kSupportedKeys.data(), kSupportedKeys.size()};
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

    std::unique_ptr<DeviceOps> TtnnDevice::create_ops() {
        return std::make_unique<TtnnQueue>(*this);
    }

    std::unique_ptr<Device> make_ttnn_device(
            std::uint32_t device_ordinal) {
        const std::size_t device_count =
                tt::tt_metal::GetNumAvailableDevices();
        if (static_cast<std::size_t>(device_ordinal) >= device_count
                || device_ordinal
                        > static_cast<std::uint32_t>(
                                std::numeric_limits<int>::max())) {
            throw invalid_ordinal(device_ordinal, device_count);
        }

        return std::make_unique<TtnnDevice>(
                device_ordinal,
                ttnn::open_mesh_device(static_cast<int>(device_ordinal)));
    }

}  // namespace iom
