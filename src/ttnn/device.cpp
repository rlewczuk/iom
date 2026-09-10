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
#include "staging.hpp"
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

        // Test seams for copy ownership transaction failure injection. Each
        // arming is consumed by exactly one operation; the atomic state keeps
        // arming and consumption race-free across the submit and worker
        // threads. The copy-plane seam itself lives with copy_planes in
        // copy.cpp; registration, outcome-insertion, and finish-failure seams
        // are consumed here where the transaction runs.
        std::atomic<bool> g_fail_next_copy_registration{false};
        std::atomic<bool> g_fail_next_copy_outcome_insertion{false};
        std::atomic<bool> g_fail_next_binary_outcome_insertion{false};
        std::atomic<bool> g_copy_registration_fault_consumed{false};
        std::atomic<bool> g_copy_outcome_insertion_fault_consumed{false};
        std::atomic<bool> g_binary_outcome_insertion_fault_consumed{false};

        // Remaining mesh-finish attempts that must fail. Count semantics let
        // a test fail both drain attempts of one failed submission (the
        // synchronous drain in execute and the completion retry) while every
        // later finish, including quarantine drains, succeeds.
        std::atomic<std::size_t> g_fail_copy_mesh_finishes{0};

        // Mesh-finish attempts routed through finish_locked for the
        // copy-drain paths, counted under IOM_ENABLE_TESTING.
        // Batch-completion tests assert one native finish per ready batch
        // through this counter.
        std::atomic<std::uint64_t> g_copy_mesh_finish_count{0};

        void consume_copy_registration_fault() noexcept(false) {
            if (g_fail_next_copy_registration.exchange(
                        false, std::memory_order_acquire)) {
                g_copy_registration_fault_consumed.store(
                        true, std::memory_order_release);
                throw std::bad_alloc();
            }
        }

        void consume_copy_outcome_insertion_fault() noexcept(false) {
            if (g_fail_next_copy_outcome_insertion.exchange(
                        false, std::memory_order_acquire)) {
                g_copy_outcome_insertion_fault_consumed.store(
                        true, std::memory_order_release);
                throw std::bad_alloc();
            }
        }
        void consume_binary_outcome_insertion_fault() noexcept(false) {
            if (g_fail_next_binary_outcome_insertion.exchange(
                        false, std::memory_order_acquire)) {
                g_binary_outcome_insertion_fault_consumed.store(
                        true, std::memory_order_release);
                throw std::bad_alloc();
            }
        }

        bool consume_copy_finish_fault() noexcept {
            std::size_t remaining = g_fail_copy_mesh_finishes.load(
                    std::memory_order_acquire);
            for (;;) {
                if (remaining == 0) {
                    return false;
                }
                if (g_fail_copy_mesh_finishes.compare_exchange_weak(
                            remaining, remaining - 1,
                            std::memory_order_acq_rel,
                            std::memory_order_acquire)) {
                    return true;
                }
            }
        }
#endif


        // Native TTNN dtypes are used where they preserve the leaf width.
        // Other NONE leaves use a UINT32 carrier; 64-bit leaves occupy two
        // adjacent carrier columns per logical element. This keeps one stable
        // TTNN-owned plane per leading allocation while conversion remains an
        // internal detail of the transfer path.
        constexpr auto kSupportedToNative = std::array{
                std::pair{DataType::BOOL, tt::tt_metal::DataType::UINT8},
                std::pair{DataType::I2, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::U2, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::I4, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::U4, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::I8, tt::tt_metal::DataType::UINT8},
                std::pair{DataType::U8, tt::tt_metal::DataType::UINT8},
                std::pair{DataType::I16, tt::tt_metal::DataType::UINT16},
                std::pair{DataType::U16, tt::tt_metal::DataType::UINT16},
                std::pair{DataType::I32, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::U32, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::I64, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::U64, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::F4_E2M1, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::F6_E2M3, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::F6_E3M2, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::F8_E4M3FN, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::F8_E5M2, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::F16, tt::tt_metal::DataType::UINT32},
                std::pair{DataType::BF16, tt::tt_metal::DataType::BFLOAT16},
                std::pair{DataType::F32, tt::tt_metal::DataType::FLOAT32},
                std::pair{DataType::F64, tt::tt_metal::DataType::UINT32},
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

        [[nodiscard]] std::size_t carrier_factor(DataType type) {
            return detail::leaf_bits(type) > 32 ? 2 : 1;
        }

        // Native tile dtype carrying the leaf encoding or its internal
        // carrier. UINT32 is intentionally used for all non-native widths.
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

            [[nodiscard]] ttnn_detail::TtnnHostStaging& host_staging()
                    noexcept {
                return host_staging_;
            }

            [[nodiscard]] detail::RegistryState&
                    registry_state() noexcept {
                return registry_state_;
            }

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
                    std::span<const std::byte> source) override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                ttnn_detail::region_from_host(
                        device_.mesh(), device_.host_staging(), destination,
                        planes_->data(), source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                ttnn_detail::region_to_host(
                        device_.mesh(), device_.host_staging(), source,
                        planes_->data(), destination);
            }

            TtnnDevice& device_;
            detail::RegistryState* state_;
            std::unique_ptr<std::vector<ttnn::Tensor>> planes_;
        };

        struct TtnnFenceCapture {
            TtnnDevice* device = nullptr;
        };

        static_assert(
                sizeof(TtnnFenceCapture)
                <= detail::kFenceStorageBytes);
        static_assert(
                alignof(TtnnFenceCapture)
                <= detail::kFenceStorageAlign);
        static_assert(std::is_trivially_copyable_v<TtnnFenceCapture>);
        static_assert(std::is_trivially_destructible_v<TtnnFenceCapture>);

        detail::FenceResult ttnn_fence_invoke(
                const detail::Fence& fence) noexcept {
            const auto& capture =
                    *std::launder(reinterpret_cast<const TtnnFenceCapture*>(
                            fence.storage));
            try {
                std::lock_guard<std::mutex> lock(
                        capture.device->api_mutex());
                capture.device->mesh().mesh_command_queue(0).finish();
                return detail::FenceResult::success();
            } catch (...) {
                return detail::FenceResult::failed(
                        std::current_exception());
            }
        }

        detail::Fence build_ttnn_fence(TtnnDevice& device) noexcept {
            detail::Fence fence;
            ::new (fence.storage) TtnnFenceCapture{&device};
            fence.invoke = &ttnn_fence_invoke;
            return fence;
        }

        // Finishes the mesh command queue. The caller must already hold the
        // device API mutex: the synchronous drain of a failed submission
        // runs under execute's api lock, and complete_task's retry reaches
        // the same drain through finish_native's own lock. Every copy-drain
        // path routes through here so an armed finish-failure seam faults
        // exactly those attempts; host transfers, fence invokes, and
        // quarantine drains keep their own finish calls unaffected.
        void finish_locked_mesh(
                tt::tt_metal::distributed::MeshDevice& device) {
#ifdef IOM_ENABLE_TESTING
            g_copy_mesh_finish_count.fetch_add(1, std::memory_order_relaxed);
            if (consume_copy_finish_fault()) {
                throw std::runtime_error(
                        "injected TTNN copy finish failure");
            }
#endif
            device.mesh_command_queue(0).finish();
        }

        void finish_locked(TtnnDevice& device) {
            finish_locked_mesh(device.mesh());
        }

        detail::FenceResult finish_native(TtnnDevice& device) noexcept {
            try {
                std::lock_guard<std::mutex> lock(device.api_mutex());
                finish_locked(device);
                return detail::FenceResult::success();
            } catch (...) {
                return detail::FenceResult::failed(
                        std::current_exception());
            }
        }

class TtnnQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        const TensorView* source = nullptr;
        TensorView* destination = nullptr;
        bool no_op = false;
        bool is_binary = false;
        DeviceOps::BinaryOperation operation =
                DeviceOps::BinaryOperation::Add;
        ttnn_detail::BinaryRequest binary_request{
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                TensorShape{{1, 1}}};
        void* fence = nullptr;
        detail::BinaryEntryRegistration binary_entries{};
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;

        Task(std::uint64_t sequence_, const TensorView& source_,
             TensorView& destination_, bool no_op_)
            : sequence(sequence_), source(&source_), destination(&destination_),
              no_op(no_op_) {}
        Task(
                std::uint64_t sequence_,
                DeviceOps::BinaryOperation operation_,
                const ttnn_detail::BinaryRequest& request,
                detail::BinaryEntryRegistration entries)
            : sequence(sequence_), is_binary(true), operation(operation_),
              binary_request(request), binary_entries(entries) {}
    };
    struct BinaryOutcome {
        detail::BinaryEntryRegistration entries;
        bool native_work_submitted = false;
        bool native_completion_proven = false;
        std::exception_ptr retained_failure;
    };
public:
    explicit TtnnQueue(TtnnDevice& device)
            : DeviceOps(device),
              device_(&device),
              state_(&device.registry_state()),
              registry_queue_id_(detail::allocate_queue_id(*state_)),
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
        // The drain completes every published task, including retained-failure
        // sequences left by an un-drainable failed submission: complete_task
        // retries the native finish and delivers the retained failure to the
        // wait tokens, under the device API mutex.
        worker_.shutdown_and_drain();
    }

    oid copy_impl(
            const TensorView& source,
            TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    Task task(sequence, source, destination, no_op);
                    worker_.submit_copy(std::move(task));
                });
    }
    oid binary_impl(const BinaryRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
        detail::Fence fence = build_ttnn_fence(*device_);
        return submit_binary(
                request, *state_, registry_queue_id_, fence,
                [this](std::uint64_t sequence, const BinaryRequest& captured,
                       detail::BinaryEntryRegistration entries) {
                    ttnn_detail::BinaryRequest internal{
                            {captured.lhs.spec, captured.lhs.native_handle,
                             captured.lhs.plane_offset,
                             captured.lhs.logical_plane_strides},
                            {captured.rhs.spec, captured.rhs.native_handle,
                             captured.rhs.plane_offset,
                             captured.rhs.logical_plane_strides},
                            {captured.out.spec, captured.out.native_handle,
                             captured.out.plane_offset,
                             captured.out.logical_plane_strides},
                            captured.result_shape};
                    worker_.submit_copy(
                            Task(sequence, captured.operation, internal, entries));
                });
    }
    void execute(Task& task) {
        if (task.is_binary) {
            bool submitted = false;
            bool completion_proven = false;
            try {
                {
                    std::lock_guard<std::mutex> lock(outcome_mutex_);
#ifdef IOM_ENABLE_TESTING
                    consume_binary_outcome_insertion_fault();
#endif
                    const auto [it, inserted] = binary_outcomes_.emplace(
                            task.sequence, BinaryOutcome{task.binary_entries});
                    if (!inserted) {
                        throw std::logic_error("duplicate TTNN binary sequence");
                    }
                }
                {
                    std::lock_guard<std::mutex> api_lock(
                            device_->api_mutex());
                    auto* lhs = static_cast<ttnn::Tensor*>(
                            task.binary_request.lhs.native_handle);
                    auto* rhs = static_cast<ttnn::Tensor*>(
                            task.binary_request.rhs.native_handle);
                    auto* out = static_cast<ttnn::Tensor*>(
                            task.binary_request.out.native_handle);
                    const auto binary = [&]<detail::scalar_add_detail::BinaryOp Op>() {
                        ttnn_detail::binary_planes<Op>(
                                device_->mesh(), device_->host_staging(),
                                task.binary_request, lhs, rhs, out, submitted,
                                completion_proven, finish_locked_mesh);
                    };
                    switch (task.operation) {
                        case DeviceOps::BinaryOperation::Add:
                            binary.template operator()<
                                    detail::scalar_add_detail::BinaryOp::add>();
                            break;
                        case DeviceOps::BinaryOperation::Mul:
                            binary.template operator()<
                                    detail::scalar_add_detail::BinaryOp::mul>();
                            break;
                        case DeviceOps::BinaryOperation::Sub:
                            binary.template operator()<
                                    detail::scalar_add_detail::BinaryOp::sub>();
                            break;
                        case DeviceOps::BinaryOperation::Div:
                            binary.template operator()<
                                    detail::scalar_add_detail::BinaryOp::div>();
                            break;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(outcome_mutex_);
                    binary_outcomes_.at(task.sequence).native_work_submitted =
                            submitted;
                    binary_outcomes_.at(task.sequence)
                            .native_completion_proven = completion_proven;
                }
                if (completion_proven) {
                    publish_native_completion(task.sequence);
                }
                executed_seq_.store(task.sequence, std::memory_order_release);
                return;
            } catch (...) {
                const std::exception_ptr submission_failure =
                        std::current_exception();
                if (!submitted) {
                    // No output upload reached the mesh. The common
                    // submit_binary transaction owns both registry rollback
                    // and sequence rollback; only discard this provisional
                    // backend outcome before propagating the failure.
                    std::lock_guard<std::mutex> lock(outcome_mutex_);
                    binary_outcomes_.erase(task.sequence);
                    std::rethrow_exception(submission_failure);
                }
                // Native work was accepted. Keep the outcome, owner
                // registrations, and staging leases under the normal
                // completion/quarantine path so the public token remains a
                // repeatably failed retained submission.
                {
                    std::lock_guard<std::mutex> lock(outcome_mutex_);
                    auto it = binary_outcomes_.find(task.sequence);
                    if (it != binary_outcomes_.end()) {
                        it->second.retained_failure = submission_failure;
                        it->second.native_work_submitted = submitted;
                        it->second.native_completion_proven =
                                completion_proven;
                    }
                }
                if (completion_proven) {
                    publish_native_completion(task.sequence);
                }
                executed_seq_.store(task.sequence, std::memory_order_release);
                return;
            }
        }
        execute_copy(task);
    }
    void publish_native_completion(std::uint64_t sequence) noexcept {
        std::lock_guard<std::mutex> fence_lock(fence_mutex_);
        const std::uint64_t now =
                last_finished_seq_.load(std::memory_order_acquire);
        if (sequence > now) {
            last_finished_seq_.store(sequence, std::memory_order_release);
        }
    }

    void execute_copy(Task& task) {
        // The transaction registers source and destination ownership before
        // any native plane reaches the mesh, so every submitted command is
        // associated with a completion record from the instant it is
        // enqueued. A failure during registration or outcome insertion
        // happens before any submission: roll back the partial entries and
        // rethrow, with nothing pending on the device.
        const detail::Fence fence = build_ttnn_fence(*device_);
        detail::EntryRegistration entries;
        try {
#ifdef IOM_ENABLE_TESTING
            consume_copy_registration_fault();
#endif
            entries = detail::register_copy_entries(
                    *state_, registry_queue_id_, task.sequence,
                    const_cast<void*>(task.source->native_handle()),
                    task.destination->native_handle(), fence);
            task.source_entry_id = entries.source;
            task.destination_entry_id = entries.destination;
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
#ifdef IOM_ENABLE_TESTING
                consume_copy_outcome_insertion_fault();
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
            if (entries.source != 0) {
                state_->registry.remove_entry_if_present(
                        entries.source,
                        const_cast<void*>(task.source->native_handle()));
            }
            if (entries.destination != 0) {
                state_->registry.remove_entry_if_present(
                        entries.destination,
                        task.destination->native_handle());
            }
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
                            task.source->native_handle());
            ttnn::Tensor* destination_planes =
                    static_cast<ttnn::Tensor*>(
                            task.destination->native_handle());
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
                    const_cast<void*>(task.source->native_handle()));
            state_->registry.remove_entry_if_present(
                    entries.destination,
                    task.destination->native_handle());
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
            finish_locked(*device_);
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
                const_cast<void*>(task.source->native_handle()));
        state_->registry.remove_entry_if_present(
                entries.destination,
                task.destination->native_handle());
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            outcomes_.erase(task.sequence);
        }
        std::rethrow_exception(submission_failure);
    }

    void complete_task(
            std::uint64_t sequence, std::exception_ptr failure) {
        BinaryOutcome binary_outcome;
        bool is_binary = false;
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto it = binary_outcomes_.find(sequence);
            if (it != binary_outcomes_.end()) {
                binary_outcome = std::move(it->second);
                binary_outcomes_.erase(it);
                is_binary = true;
            }
        }
        if (is_binary) {
            detail::FenceResult fence_result = detail::FenceResult::success();
            bool completion_proven =
                    binary_outcome.native_completion_proven;
            if (binary_outcome.native_work_submitted
                    && !completion_proven) {
                fence_result = finish_native(*device_);
                completion_proven =
                        fence_result.succeeded && !fence_result.failure;
                if (completion_proven) {
                    publish_native_completion(sequence);
                }
            }
            const std::exception_ptr operation_failure =
                    failure ? failure : binary_outcome.retained_failure;
            const bool fence_succeeded =
                    fence_result.succeeded && !fence_result.failure;
            const bool released = detail::release_or_invalidate_binary_entries(
                    state_->registry, binary_outcome.entries,
                    static_cast<bool>(operation_failure)
                            && !completion_proven,
                    fence_succeeded);
            (void)released;
            complete(sequence, operation_failure
                    ? operation_failure : fence_result.failure);
            return;
        }
        std::vector<std::pair<std::uint64_t, detail::SequenceOutcome>> batch;
        bool has_outcome = false;
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto first = outcomes_.find(sequence);
            if (first != outcomes_.end()) {
                batch.emplace_back(sequence, std::move(first->second));
                outcomes_.erase(first);
                has_outcome = true;
                const std::uint64_t executed =
                        executed_seq_.load(std::memory_order_acquire);
                std::uint64_t next = sequence + 1;
                for (auto it = outcomes_.upper_bound(sequence);
                     it != outcomes_.end() && it->first == next
                             && next <= executed;
                     it = outcomes_.erase(it), ++next) {
                    batch.emplace_back(next, std::move(it->second));
                }
            }
        }
        if (!has_outcome) {
            complete(sequence, std::move(failure));
            return;
        }

        bool has_native_work = false;
        for (const auto& member : batch) {
            has_native_work =
                    has_native_work
                    || member.second.native_work_submitted;
        }
        // A no-op-only batch has no mesh work to drain. Native work in any
        // member requires one finish for the whole contiguous batch.
        const detail::FenceResult fence_result =
                has_native_work
                        ? finish_native(*device_)
                        : detail::FenceResult::success();
        const bool fence_succeeded =
                fence_result.succeeded && !fence_result.failure;
        for (std::size_t index = 0; index < batch.size(); ++index) {
            const std::uint64_t seq = batch[index].first;
            const detail::SequenceOutcome& outcome = batch[index].second;
            // The worker-supplied failure belongs only to the head task;
            // later members carry at most their retained submission
            // failure.
            const std::exception_ptr operation_failure =
                    index == 0 && failure ? failure
                                          : outcome.retained_failure;
            const bool released =
                    detail::release_or_invalidate_entries(
                            state_->registry, outcome,
                            static_cast<bool>(operation_failure),
                            fence_succeeded);
            if (released) {
                last_finished_seq_.store(
                        seq, std::memory_order_release);
            }
            // The worker-supplied failure wins, then the retained
            // submission failure, then the native finish failure, in
            // submission order within the batch.
            std::exception_ptr completion_failure;
            if (index == 0 && failure) {
                completion_failure = failure;
            } else if (operation_failure) {
                completion_failure = operation_failure;
            } else {
                completion_failure = fence_result.failure;
            }
            complete(seq, std::move(completion_failure));
        }
    }

    TtnnDevice* device_;
    std::map<std::uint64_t, BinaryOutcome> binary_outcomes_;
    detail::RegistryState* state_;
    detail::QueueId registry_queue_id_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, detail::SequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
    std::mutex fence_mutex_;
    std::atomic<std::uint64_t> last_finished_seq_{0};
    // Highest sequence whose task finished executing with its outcome still
    // registered (every native plane enqueued, or a determined no-op).
    // Registration precedes native submission, so an outcome alone is not
    // proof its work reached the mesh; complete_task's batch collection
    // reads this marker, under outcome_mutex_, to stop the batch before
    // any not-yet-executed sequence.
    std::atomic<std::uint64_t> executed_seq_{0};
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
    const detail::FenceResult result = finish_native(*device_);
    if (result.succeeded && !result.failure) {
        last_finished_seq_.store(sequence, std::memory_order_release);
    } else {
        record_post_completion_failure(sequence, result.failure);
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

        void fail_next_binary_outcome_insertion_for_testing() noexcept {
            g_fail_next_binary_outcome_insertion.store(
                    true, std::memory_order_release);
        }

        bool binary_outcome_insertion_fault_consumed_for_testing() noexcept {
            return g_binary_outcome_insertion_fault_consumed.load(
                    std::memory_order_acquire);
        }
        void fail_next_copy_registration_for_testing() noexcept {
            g_fail_next_copy_registration.store(
                    true, std::memory_order_release);
        }

        bool copy_registration_fault_consumed_for_testing() noexcept {
            return g_copy_registration_fault_consumed.load(
                    std::memory_order_acquire);
        }

        void fail_next_copy_outcome_insertion_for_testing() noexcept {
            g_fail_next_copy_outcome_insertion.store(
                    true, std::memory_order_release);
        }

        bool copy_outcome_insertion_fault_consumed_for_testing() noexcept {
            return g_copy_outcome_insertion_fault_consumed.load(
                    std::memory_order_acquire);
        }

        void fail_next_copy_finishes_for_testing(
                std::size_t count) noexcept {
            g_fail_copy_mesh_finishes.store(count, std::memory_order_release);
        }

        bool copy_finish_fault_pending_for_testing() noexcept {
            return g_fail_copy_mesh_finishes.load(
                           std::memory_order_acquire)
                   != 0;
        }

        void reset_copy_finish_count_for_testing() noexcept {
            g_copy_mesh_finish_count.store(0, std::memory_order_release);
        }

        std::uint64_t copy_finish_count_for_testing() noexcept {
            return g_copy_mesh_finish_count.load(std::memory_order_acquire);
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
