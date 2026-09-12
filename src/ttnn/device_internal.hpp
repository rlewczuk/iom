#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>

#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "iom/ttnn/device.hpp"
#include "registry_state.hpp"
#include "staging.hpp"

namespace iom::ttnn_detail {

[[nodiscard]] bool is_supported(DataType type);
[[nodiscard]] std::size_t carrier_factor(DataType type);
[[nodiscard]] tt::tt_metal::DataType native_dtype(DataType type);
[[nodiscard]] std::invalid_argument invalid_ordinal(
        std::uint32_t ordinal, std::size_t device_count);
[[nodiscard]] std::uint32_t checked_to_uint32(
        std::size_t dimension, std::size_t index);
[[nodiscard]] std::size_t checked_plane_count(const TensorSpec& spec);

}  // namespace iom::ttnn_detail

namespace iom {

class TtnnDevice final : public Device {
public:
    TtnnDevice(
            std::uint32_t ordinal,
            std::shared_ptr<ttnn::MeshDevice> native_device,
            QueueConfig queue_config);

    TtnnDevice(const TtnnDevice&) = delete;
    TtnnDevice& operator=(const TtnnDevice&) = delete;

    ~TtnnDevice() override;

    [[nodiscard]] BackendKind backend_kind() const noexcept override;
    [[nodiscard]] std::uint32_t backend_device() const noexcept override;
    [[nodiscard]] std::span<const DataType> supported_data_types()
            const noexcept override;

    [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
            const TensorSpec& spec) override;
    [[nodiscard]] std::unique_ptr<RawWorkspace> create_workspace(
            std::size_t bytes) override;
    [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

    [[nodiscard]] tt::tt_metal::distributed::MeshDevice& mesh() noexcept;
    [[nodiscard]] std::mutex& api_mutex() noexcept;
    [[nodiscard]] ttnn_detail::TtnnHostStaging& host_staging() noexcept;
    [[nodiscard]] detail::RegistryState& registry_state() noexcept;

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

}  // namespace iom
