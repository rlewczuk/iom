#include <doctest/doctest.h>

#include <tt-metalium/host_api.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
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

        // F64 uses the internal TTNN carrier and must be a usable storage
        // capability rather than an advertise-only table entry.
        auto f64 = device->create_tensor(
                iom::TensorSpec{
                        iom::TensorShape{{16, 16}}, iom::DataType::F64});
        REQUIRE(f64 != nullptr);
        std::vector<std::byte> f64_source(f64->view().spec().logical_nbytes());
        for (std::size_t i = 0; i < f64_source.size(); ++i) {
            f64_source[i] = std::byte{static_cast<unsigned char>(i * 17)};
        }
        f64->view().copy_from_host(f64_source);
        std::vector<std::byte> f64_round_trip(f64_source.size());
        f64->view().copy_to_host(f64_round_trip);
        CHECK(f64_round_trip == f64_source);

        // Supported storage and a queue are now live capabilities.
        {
            auto tensor = device->create_tensor(
                    iom::TensorSpec{
                            iom::TensorShape{{16, 16}}, iom::DataType::BF16});
            CHECK(tensor != nullptr);
        }
        auto ops = device->create_ops();
        CHECK(ops != nullptr);
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
