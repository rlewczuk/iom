#include "iom/sycl/device.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "copy.hpp"
#include "runtime.hpp"


namespace iom::sycl_detail {

    ContextCalls context_calls{};

}  // namespace iom::sycl_detail

namespace iom {

    namespace {
        constexpr std::size_t kStorageAlignment = 32;


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
                if (sycl_detail::context_calls.context_ready != nullptr) {
                    sycl_detail::context_calls.context_ready(*context_);
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
                    const TensorSpec& spec) override;

            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                return sycl_detail::make_queue(*this, *context_, device_);
            }

            [[nodiscard]] const sycl::context& context() const noexcept {
                return *context_;
            }

            [[nodiscard]] const sycl::device& native_device() const noexcept {
                return device_;
            }

        private:
            sycl::device device_;
            std::optional<sycl::context> context_;
            std::uint32_t ordinal_;
            Allocator& allocator_;

            friend class SyclTensor;
        };
        class SyclTensor final : public Tensor {
        public:
            SyclTensor(const TensorSpec& spec, SyclDevice& device,
                       Allocator& allocator)
                    : Tensor(spec, device),
                      device_(device),
                      allocator_(allocator) {
                const std::size_t storage_nbytes =
                        view().spec().tiled_storage_nbytes();
                address_ = allocator_.alloc(storage_nbytes);
                if (address_ == nullptr) {
                    throw std::bad_alloc();
                }
                if (reinterpret_cast<std::uintptr_t>(address_)
                                % kStorageAlignment
                        != 0) {
                    void* rejected = std::exchange(address_, nullptr);
                    allocator_.free(rejected);
                    throw std::runtime_error(
                            "SYCL tensor storage is not 32-byte aligned");
                }

                try {
                    if (sycl::get_pointer_type(address_, device_.context())
                            == sycl::usm::alloc::unknown) {
                        throw std::runtime_error(
                                "SYCL tensor storage is incompatible with "
                                "the owned context");
                    }
                } catch (...) {
                    void* rejected = std::exchange(address_, nullptr);
                    try {
                        allocator_.free(rejected);
                    } catch (...) {
                    }
                    throw;
                }
            }

            ~SyclTensor() override {
                if (address_ == nullptr) {
                    return;
                }
                try {
                    allocator_.free(address_);
                } catch (...) {
                }
                address_ = nullptr;
            }

        private:
            [[nodiscard]] void* storage_handle() noexcept override {
                return address_;
            }

            void region_from_host(
                    const TensorView& destination,
                    std::span<const std::byte> source) override {
                sycl_detail::region_from_host(
                        device_.context(), device_.native_device(),
                        destination, address_, source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                sycl_detail::region_to_host(
                        device_.context(), device_.native_device(), source,
                        address_, destination);
            }

            SyclDevice& device_;
            Allocator& allocator_;
            void* address_ = nullptr;
        };

        std::unique_ptr<Tensor> SyclDevice::create_tensor(
                const TensorSpec& spec) {
            return std::make_unique<SyclTensor>(spec, *this, allocator_);
        }

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
