#include "iom/sycl/device.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "copy.hpp"
#include "runtime.hpp"
#include "iom/detail/aligned_storage.hpp"


namespace iom::sycl_detail {

    ContextCalls context_calls{};
    LaunchCalls launch_calls{};

}  // namespace iom::sycl_detail

namespace iom {

    namespace {
        constexpr std::array kSyclSupportedDataTypes = {
                iom::DataType::BOOL,
                iom::DataType::I2, iom::DataType::U2,
                iom::DataType::I4, iom::DataType::U4,
                iom::DataType::I8, iom::DataType::U8,
                iom::DataType::I16, iom::DataType::U16,
                iom::DataType::I32, iom::DataType::U32,
                iom::DataType::I64, iom::DataType::U64,
                iom::DataType::F4_E2M1,
                iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
                iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2,
                iom::DataType::F8_E8M0,
                iom::DataType::F16, iom::DataType::BF16,
                iom::DataType::F32, iom::DataType::F64,
        };


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
                      staging_pool_(*context_, device_),
                      transfer_queue_(
                              std::in_place, *context_, device_,
                              sycl::property_list{
                                      sycl::property::queue::in_order{}}),
                      registry_state_(),
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
                registry_state_.quarantine.drain();
                staging_pool_.destroy();
                try {
                    transfer_queue_->wait_and_throw();
                } catch (...) {
                }
                transfer_queue_.reset();
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
            [[nodiscard]] std::span<const iom::DataType>
                    supported_data_types() const noexcept override {
                return {kSyclSupportedDataTypes.data(),
                        kSyclSupportedDataTypes.size()};
            }

            [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
                    const TensorSpec& spec) override;
            [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override {
                return sycl_detail::make_queue(
                        *this, *context_, device_, registry_state_);
            }

            [[nodiscard]] const sycl::context& context() const noexcept {
                return *context_;
            }

            [[nodiscard]] const sycl::device& native_device() const noexcept {
                return device_;
            }
            [[nodiscard]] sycl_detail::StagingSlotPool&
                    staging_pool() noexcept {
                return staging_pool_;
            }

            [[nodiscard]] sycl::queue& transfer_queue() noexcept {
                return *transfer_queue_;
            }

            [[nodiscard]] detail::RegistryState&
                    registry_state() noexcept {
                return registry_state_;
            }

        private:
            sycl::device device_;
            std::optional<sycl::context> context_;
            sycl_detail::StagingSlotPool staging_pool_;
            std::optional<sycl::queue> transfer_queue_;
            detail::RegistryState registry_state_;
            std::uint32_t ordinal_;
            Allocator& allocator_;

            friend class SyclTensor;
        };
        class SyclTensor final : public Tensor {
        public:
            SyclTensor(
                    const TensorSpec& spec, SyclDevice& device,
                    detail::RegistryState& state,
                    Allocator& allocator)
                    : Tensor(spec, device),
                      device_(device),
                      state_(&state),
                      allocator_(allocator) {
                address_ = iom::detail::allocate_aligned_storage(
                        allocator_,
                        view().spec().tiled_storage_nbytes(),
                        [] {},
                        "SYCL tensor storage is not 32-byte aligned");

                try {
                    if (sycl::get_pointer_type(address_, device_.context())
                            == sycl::usm::alloc::unknown) {
                        throw std::runtime_error(
                                "SYCL tensor storage is incompatible with "
                                "the owned context");
                    }
                } catch (...) {
                    void* rejected = std::exchange(address_, nullptr);
                    iom::detail::release_aligned_storage(allocator_, rejected);
                    throw;
                }
            }

            ~SyclTensor() noexcept override {
                if (address_ == nullptr) {
                    return;
                }

                const std::size_t bytes =
                        view().spec().tiled_storage_nbytes();
                const auto quarantine_storage = [this, bytes]() noexcept {
                    try {
                        state_->quarantine
                                .emplace<detail::AllocatorCleanupAction>(
                                        allocator_, address_, bytes);
                    } catch (...) {
                        // Leaking is safer than returning failed storage to
                        // the allocator when quarantine allocation fails.
                    }
                    address_ = nullptr;
                };
                const auto release_storage = [this]() noexcept {
                    iom::detail::release_aligned_storage(
                            allocator_, address_);
                };
                iom::detail::release_or_quarantine(
                        state_->registry, address_, quarantine_storage,
                        release_storage);
            }

        private:
            [[nodiscard]] void* storage_handle() noexcept override {
                return address_;
            }

            void region_from_host(
                    const TensorView& destination,
                    std::span<const std::byte> source) override {
                sycl_detail::region_from_host(
                        device_.staging_pool(), device_.transfer_queue(),
                        destination, address_, source);
            }

            void region_to_host(
                    const TensorView& source,
                    std::span<std::byte> destination) const override {
                sycl_detail::region_to_host(
                        device_.staging_pool(), device_.transfer_queue(),
                        source, address_, destination);
            }

            SyclDevice& device_;
            detail::RegistryState* state_;
            Allocator& allocator_;
            void* address_ = nullptr;
        };

        std::unique_ptr<Tensor> SyclDevice::create_tensor(
                const TensorSpec& spec) {
            return std::make_unique<SyclTensor>(
                    spec, *this, registry_state_, allocator_);
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
