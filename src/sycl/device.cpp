#include "iom/sycl/device.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime.hpp"

namespace iom::sycl_detail {

    ContextCalls context_calls{};

}  // namespace iom::sycl_detail

namespace iom {

    namespace {

        [[nodiscard]] std::vector<sycl::device> eligible_devices() {
            std::vector<sycl::device> devices = sycl::device::get_devices();
            devices.erase(
                    std::remove_if(
                            devices.begin(), devices.end(),
                            [](const sycl::device& device) {
                                return !device.is_gpu() && !device.is_accelerator();
                            }),
                    devices.end());
            return devices;
        }

        [[nodiscard]] std::invalid_argument invalid_ordinal(
                std::uint32_t ordinal, std::size_t device_count) {
            return std::invalid_argument(
                    "SYCL device ordinal " + std::to_string(ordinal)
                    + " is unavailable; eligible accelerator count is "
                    + std::to_string(device_count));
        }

        class SyclDevice final : public Device {
        public:
            SyclDevice(std::uint32_t ordinal, sycl::device device,
                       Allocator& allocator)
                    : device_(std::move(device)),
                      context_(device_),
                      ordinal_(ordinal),
                      allocator_(allocator) {
                if (sycl_detail::context_calls.context_created != nullptr) {
                    sycl_detail::context_calls.context_created();
                }
            }

            SyclDevice(const SyclDevice&) = delete;
            SyclDevice& operator=(const SyclDevice&) = delete;

            ~SyclDevice() override {
                context_.reset();
                if (sycl_detail::context_calls.context_destroyed != nullptr) {
                    sycl_detail::context_calls.context_destroyed();
                }
            }

            [[nodiscard]] BackendKind backend_kind() const noexcept override {
                return BackendKind::SYCL;
            }

            [[nodiscard]] std::uint32_t backend_device() const noexcept override {
                return ordinal_;
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec&) override {
                throw std::runtime_error(
                        "SYCL tensor storage is not implemented yet");
            }

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                throw std::runtime_error(
                        "SYCL operation queues are not implemented yet");
            }

        private:
            sycl::device device_;
            std::optional<sycl::context> context_;
            std::uint32_t ordinal_;
            Allocator& allocator_;
        };

    }  // namespace

    namespace sycl_detail {

        std::size_t eligible_device_count() {
            return eligible_devices().size();
        }

    }  // namespace sycl_detail

    std::unique_ptr<Device> make_sycl_device(
            std::uint32_t device_ordinal, Allocator& allocator) {
        std::vector<sycl::device> devices = eligible_devices();
        if (device_ordinal >= devices.size()) {
            throw invalid_ordinal(device_ordinal, devices.size());
        }
        return std::make_unique<SyclDevice>(
                device_ordinal, std::move(devices[device_ordinal]), allocator);
    }

}  // namespace iom
