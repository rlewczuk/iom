#include <doctest/doctest.h>

#include <tt-metalium/host_api.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"
#include "iom/ttnn/device.hpp"

TEST_CASE("TTNN factory reports a live hardware device and owns its context") {
    const std::size_t device_count =
            tt::tt_metal::GetNumAvailableDevices();
    REQUIRE(device_count > 0);

    {
        auto device = iom::make_ttnn_device(0);
        REQUIRE(device != nullptr);
        CHECK(device->backend_kind() == iom::BackendKind::TTNN);
        CHECK(device->backend_device() == 0);

        CHECK_THROWS_AS(
                device->create_tensor(
                        iom::TensorSpec{
                                iom::TensorShape{{16, 16}}, iom::DataType::F32}),
                std::runtime_error);
        CHECK_THROWS_AS(device->create_ops(), std::runtime_error);
    }

    // Reopening after the first owner leaves scope exercises deterministic
    // context teardown without relying on a process-global active device.
    auto recreated = iom::make_ttnn_device(0);
    REQUIRE(recreated != nullptr);
    CHECK(recreated->backend_kind() == iom::BackendKind::TTNN);
    CHECK(recreated->backend_device() == 0);
}

TEST_CASE("TTNN factory rejects the first unavailable ordinal") {
    const std::size_t device_count =
            tt::tt_metal::GetNumAvailableDevices();
    REQUIRE(device_count > 0);
    REQUIRE(device_count <= std::numeric_limits<std::uint32_t>::max());

    CHECK_THROWS_AS(
            iom::make_ttnn_device(static_cast<std::uint32_t>(device_count)),
            std::invalid_argument);
}
