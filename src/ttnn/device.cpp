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

            ~TtnnDevice() override = default;

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
            // validated the specification against the supported-type table
            // and the representable native extents; the checks are repeated
            // here so the native constructor can never be reached with a
            // non-representable extent from any entry path.
            TtnnTensor(const TensorSpec& spec, TtnnDevice& device)
                    : Tensor(spec, device), device_(device) {
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

        class TtnnQueue final : public DeviceOps {
            struct Task {
                std::uint64_t sequence;
                const TensorView* source;
                const ttnn::Tensor* source_planes;
                TensorView* destination;
                ttnn::Tensor* destination_planes;
                bool no_op;
                void* fence = nullptr;
            };

        public:
            explicit TtnnQueue(TtnnDevice& device)
                    : device_(&device),
                      worker_(
                              detail::StagedWorker<Task>::Callbacks{
                                      [this](Task& task) {
                                          if (!task.no_op) {
                                              std::lock_guard<std::mutex>
                                                      api_lock(
                                                              device_->api_mutex());
                                              ttnn_detail::copy_planes(
                                                      *task.source,
                                                      task.source_planes,
                                                      *task.destination,
                                                      task.destination_planes);
                                              device_->mesh()
                                                      .mesh_command_queue(0)
                                                      .finish();
                                          }
                                      },
                                      [](void*) {},
                                      [](void*) {},
                                      [this](
                                              std::uint64_t sequence,
                                              std::exception_ptr failure) {
                                          complete(
                                                  sequence,
                                                  std::move(failure));
                                      }},
                              detail::StagedWorker<Task>::PublishPolicy::
                                      CompleteOnThrow) {
                worker_.start();
            }

            ~TtnnQueue() override {
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
                            worker_.submit_copy(Task{
                                    sequence,
                                    &source,
                                    static_cast<const ttnn::Tensor*>(
                                            source.native_handle()),
                                    &destination,
                                    static_cast<ttnn::Tensor*>(
                                            destination.native_handle()),
                                    no_op});
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
            TtnnDevice* device_;
            std::mutex submission_order_mutex_;
            detail::StagedWorker<Task> worker_;
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
        // Reject every extent the TTNN native constructor cannot represent
        // before any native object or allocation exists: the final two
        // dimensions must fit the uint32_t native extent ceiling and the
        // leading-plane product must fit std::size_t. These are pure size
        // checks, so they run before the API lock and before the
        // supported-type table membership decides on native access.
        static_cast<void>(checked_plane_count(spec));
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
