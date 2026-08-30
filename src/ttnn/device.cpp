#include "iom/ttnn/device.hpp"

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace iom {

    namespace {

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
                    const TensorSpec&) override {
                throw std::runtime_error(
                        "TTNN tensor creation is not implemented");
            }

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                throw std::runtime_error(
                        "TTNN operation queues are not implemented");
            }

        private:
            std::uint32_t ordinal_;
            std::shared_ptr<ttnn::MeshDevice> native_device_;
        };

    }  // namespace

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
