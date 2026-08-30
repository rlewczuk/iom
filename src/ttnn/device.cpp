#include "iom/ttnn/device.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_ops.hpp>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "copy.hpp"
#include "iom/iom.hpp"

namespace iom {

    namespace {

        // The one explicit TTNN supported-type table: every leaf type whose
        // host encoding TTNN tiled storage reproduces bit-for-bit. Signed
        // 8/16-bit integers ride the unsigned tiles of the same width
        // (two's-complement fields are bit-identical); BOOL rides UINT8 with
        // canonical zero/one bytes validated by the common TensorView base.
        constexpr std::array kSupportedDataTypes = {
                DataType::BOOL,
                DataType::U8,
                DataType::I8,
                DataType::U16,
                DataType::I16,
                DataType::U32,
                DataType::I32,
                DataType::BF16,
                DataType::F32,
        };

        [[nodiscard]] bool is_supported(DataType type) {
            return std::find(
                           kSupportedDataTypes.begin(),
                           kSupportedDataTypes.end(), type)
                   != kSupportedDataTypes.end();
        }

        // Native tile dtype carrying the leaf encoding bit-for-bit.
        [[nodiscard]] tt::tt_metal::DataType native_dtype(DataType type) {
            switch (type) {
                case DataType::BOOL:
                case DataType::U8:
                case DataType::I8: return tt::tt_metal::DataType::UINT8;
                case DataType::U16:
                case DataType::I16: return tt::tt_metal::DataType::UINT16;
                case DataType::U32: return tt::tt_metal::DataType::UINT32;
                case DataType::I32: return tt::tt_metal::DataType::INT32;
                case DataType::BF16: return tt::tt_metal::DataType::BFLOAT16;
                case DataType::F32: return tt::tt_metal::DataType::FLOAT32;
                default:
                    throw std::invalid_argument(
                            "DataType has no TTNN native tile dtype");
            }
        }

        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, std::size_t device_count) {
            return std::invalid_argument(
                    "TTNN device ordinal " + std::to_string(ordinal)
                    + " is unavailable; device count is "
                    + std::to_string(device_count));
        }

        class TtnnDevice final : public Device {
        public:
            TtnnDevice(
                    std::uint32_t ordinal,
                    std::shared_ptr<ttnn::MeshDevice> native_device)
                    : ordinal_(ordinal), native_device_(std::move(native_device)) {}

            TtnnDevice(const TtnnDevice&) = delete;
            TtnnDevice& operator=(const TtnnDevice&) = delete;

            ~TtnnDevice() override = default;

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::TTNN;
            }

            [[nodiscard]] std::uint32_t backend_device() const noexcept override {
                return ordinal_;
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec& spec) override;

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

            [[nodiscard]] tt::tt_metal::distributed::MeshDevice& mesh()
                    noexcept {
                return *native_device_;
            }

            // Serializes every TTNN runtime interaction of this device's
            // tensors, queues, and host transfers.
            [[nodiscard]] std::mutex& api_mutex() noexcept {
                return api_mutex_;
            }

        private:
            std::uint32_t ordinal_;
            std::shared_ptr<ttnn::MeshDevice> native_device_;
            std::mutex api_mutex_;
        };

        // Owner of one TTNN-native tiled tensor per logical plane. Native
        // storage is created and destroyed through the TTNN runtime, never
        // through iom::Allocator; the native 32x32-tile byte count differs
        // from TensorSpec::tiled_storage_nbytes().
        class TtnnTensor final : public Tensor {
        public:
            // The caller holds the device's API mutex and has already
            // validated the specification against the supported-type table.
            TtnnTensor(const TensorSpec& spec, TtnnDevice& device)
                    : Tensor(spec, device), device_(device) {
                const std::span<const std::size_t> dimensions =
                        spec.shape.dimensions();
                const tt::tt_metal::TensorSpec plane_spec(
                        tt::tt_metal::Shape{
                                1u,
                                static_cast<std::uint32_t>(
                                        dimensions[dimensions.size() - 2]),
                                static_cast<std::uint32_t>(
                                        dimensions[dimensions.size() - 1])},
                        tt::tt_metal::TensorLayout(
                                native_dtype(spec.data_type),
                                tt::tt_metal::PageConfig(
                                        tt::tt_metal::Layout::TILE,
                                        tt::tt_metal::Tile()),
                                tt::tt_metal::MemoryConfig{}));
                std::size_t plane_count = 1;
                for (std::size_t i = 0; i + 2 < dimensions.size(); ++i) {
                    plane_count *= dimensions[i];
                }
                planes_.reserve(plane_count);
                for (std::size_t plane = 0; plane < plane_count; ++plane) {
                    planes_.push_back(ttnn::create_device_tensor(
                            plane_spec, &device_.mesh()));
                }
            }

            ~TtnnTensor() override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                planes_.clear();
            }

        private:
            [[nodiscard]] void* storage_handle() noexcept override {
                return planes_.data();
            }

            void region_from_host(
                    const TensorView& destination,
                    std::span<const std::byte> source) override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                ttnn_detail::region_from_host(
                        device_.mesh(), destination, planes_.data(), source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                std::lock_guard<std::mutex> lock(device_.api_mutex());
                ttnn_detail::region_to_host(
                        device_.mesh(), source, planes_.data(), destination);
            }

            TtnnDevice& device_;
            std::vector<ttnn::Tensor> planes_;
        };

        /**
         * One in-order asynchronous copy queue with one worker thread over
         * the device's default mesh command queue. Tasks are staged during
         * submit() and handed to the worker only after submit() returns, so
         * a sequence is always committed in the common base before the
         * worker can report its completion. The worker finishes the command
         * queue after each task, so an asynchronous TTNN failure is captured
         * for exactly the sequence that caused it.
         */
        class TtnnQueue final : public DeviceOps {
        public:
            explicit TtnnQueue(TtnnDevice& device)
                    : device_(&device),
                      worker_([this] { run(); }) {}

            ~TtnnQueue() override {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    shutdown_ = true;
                }
                completion_.notify_all();
                worker_.join();
            }

            oid copy(
                    const TensorView& source,
                    TensorView& destination) override {
                validate_copy(source, destination);

                // Identical windows share owner storage, logical shape,
                // plane offset, and plane strides; equal native handles
                // alone prove nothing, so every component participates in
                // the comparison.
                const bool identical_window =
                        source.native_handle() == destination.native_handle()
                        && source.plane_offset() == destination.plane_offset()
                        && std::equal(
                                source.plane_strides().begin(),
                                source.plane_strides().end(),
                                destination.plane_strides().begin(),
                                destination.plane_strides().end());

                const oid token = submit([&](std::uint64_t sequence) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    staged_.push_back(
                            Task{sequence, &source,
                                 static_cast<const ttnn::Tensor*>(
                                         source.native_handle()),
                                 &destination,
                                 static_cast<ttnn::Tensor*>(
                                         destination.native_handle()),
                                 identical_window});
                });
                publish_staged();
                return token;
            }

            oid add(const TensorView&, const TensorView&, TensorView&)
                    override {
                throw unsupported("add");
            }

            oid mul(const TensorView&, const TensorView&, TensorView&)
                    override {
                throw unsupported("mul");
            }

            oid silu(const TensorView&, TensorView&) override {
                throw unsupported("silu");
            }

            oid linear(const TensorView&, const TensorView&, TensorView&)
                    override {
                throw unsupported("linear");
            }

            oid rmsnorm(
                    const TensorView&, TensorView&, const TensorView&, float,
                    size_t) override {
                throw unsupported("rmsnorm");
            }

            oid sdpa(
                    const TensorView&, const TensorView&, const TensorView&,
                    size_t, size_t, size_t, TensorView&) override {
                throw unsupported("sdpa");
            }

        private:
            struct Task {
                std::uint64_t sequence;
                const TensorView* source;
                const ttnn::Tensor* source_planes;
                TensorView* destination;
                ttnn::Tensor* destination_planes;
                bool no_op;
            };

            static std::runtime_error unsupported(const char* operation) {
                return std::runtime_error(
                        std::string("TTNN backend does not implement ")
                        + operation);
            }

            void validate_copy(
                    const TensorView& source,
                    const TensorView& destination) const {
                if (&source.device() != device_
                        || &destination.device() != device_) {
                    throw std::invalid_argument(
                        "copy views must belong to the queue's own device");
                }
                if (!(source.spec() == destination.spec())) {
                    throw std::invalid_argument(
                        "copy views must have identical shape, leaf type, "
                        "and quantization");
                }
            }

            // Hands staged tasks to the worker. Called only after submit()
            // returned, which is what makes every staged sequence committed
            // before the worker can complete it.
            void publish_staged() {
                std::exception_ptr failure;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    while (!staged_.empty()) {
                        Task task = std::move(staged_.front());
                        staged_.pop_front();
                        try {
                            tasks_.push_back(std::move(task));
                        } catch (...) {
                            // The sequence is already consumed; report a
                            // failed completion so its wait cannot hang.
                            complete(task.sequence, std::current_exception());
                            if (!failure) {
                                failure = std::current_exception();
                            }
                        }
                    }
                    completion_.notify_one();
                }
                if (failure) {
                    std::rethrow_exception(failure);
                }
            }

            void run() {
                std::unique_lock<std::mutex> lock(mutex_);
                for (;;) {
                    completion_.wait(lock, [this] {
                        return shutdown_ || !tasks_.empty();
                    });
                    if (shutdown_) {
                        // Destruction neither waits for nor cancels
                        // submitted work: a contract-obeying caller has
                        // already waited for every task, so only unreachable
                        // leftovers could remain and their views may be
                        // gone.
                        return;
                    }
                    const Task task = std::move(tasks_.front());
                    tasks_.pop_front();
                    lock.unlock();

                    std::exception_ptr failure;
                    try {
                        if (!task.no_op) {
                            std::lock_guard<std::mutex> api_lock(
                                    device_->api_mutex());
                            ttnn_detail::copy_planes(
                                    *task.source, task.source_planes,
                                    *task.destination,
                                    task.destination_planes);
                            device_->mesh().mesh_command_queue(0).finish();
                        }
                    } catch (...) {
                        failure = std::current_exception();
                    }
                    complete(task.sequence, failure);

                    lock.lock();
                }
            }

            TtnnDevice* device_;
            std::mutex mutex_;
            std::condition_variable completion_;
            std::deque<Task> staged_;
            std::deque<Task> tasks_;
            bool shutdown_ = false;
            std::thread worker_;
        };

    }  // namespace

    std::span<const DataType> ttnn_supported_data_types() noexcept {
        return kSupportedDataTypes;
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
