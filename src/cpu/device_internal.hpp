#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

#include "iom/cpu/device.hpp"
#include "iom/detail/outstanding_work_registry.hpp"

namespace iom {

namespace cpu_detail {
// Common CPU carrier behavior for the named scalar codecs. Keeping this
// backend-private trait in one header prevents each CPU operation from
// maintaining a separate copy of the IEEE helper surface.
template <typename Carrier>
struct CpuCarrierTraits {
    using carrier_type = Carrier;

    static Carrier exp(Carrier value) noexcept { return std::exp(value); }
    static Carrier positive_infinity() noexcept {
        return std::numeric_limits<Carrier>::infinity();
    }
    static Carrier quiet_nan() noexcept {
        return std::numeric_limits<Carrier>::quiet_NaN();
    }
    static Carrier max_finite() noexcept {
        return std::numeric_limits<Carrier>::max();
    }
    static bool isnan(Carrier value) noexcept { return std::isnan(value); }
    static bool isinf(Carrier value) noexcept { return std::isinf(value); }
    static bool signbit(Carrier value) noexcept {
        return std::signbit(value);
    }
    static Carrier fabs(Carrier value) noexcept { return std::fabs(value); }
    static Carrier floor(Carrier value) noexcept {
        return std::floor(value);
    }
    static Carrier ldexp(Carrier value, int exponent) noexcept {
        return std::ldexp(value, exponent);
    }
    static Carrier frexp(Carrier value, int* exponent) noexcept {
        return std::frexp(value, exponent);
    }
};

// The CPU SDPA worker receives only the fixed-size metadata that common
// admission already validated and snapshotted. Keeping this adapter
// backend-private avoids retaining a borrowed TensorView or constructing a
// TensorSpec/vector in the deferred worker.
struct SdpaView {
    std::size_t rank = 0;
    std::array<std::size_t, 8> dimensions{};
    std::array<std::size_t, 6> plane_strides{};
    std::size_t plane_offset = 0;
    unsigned char* native_handle = nullptr;
};

struct SdpaRequest {
    SdpaView q;
    SdpaView k;
    SdpaView v;
    SdpaView out;
    std::size_t a = 0;
    std::size_t L = 0;
    std::size_t Hq = 0;
    std::size_t Hkv = 0;
    std::size_t R = 0;
    std::size_t C = 0;
    std::size_t D = 0;
    std::size_t grouping = 0;
    std::size_t output_width = 0;
    unsigned char* workspace = nullptr;
};

[[nodiscard]] WorkspaceRequirements sdpa_workspace_requirements(
        std::size_t leading_planes, std::size_t heads, std::size_t rows,
        std::size_t length);
void sdpa_elements(const SdpaRequest& request);

void arm_sdpa_failure() noexcept;
void clear_sdpa_failure() noexcept;
[[nodiscard]] bool consume_sdpa_failure() noexcept;

// CPU-local failure construction for the cache append port, with the same
// one-shot semantics as the SDPA and SiLU latches: the next enqueued cache
// append task consumes it after acceptance and before its element loop.
void arm_cache_append_failure() noexcept;
void clear_cache_append_failure() noexcept;
[[nodiscard]] bool consume_cache_append_failure() noexcept;

// Narrow test observation seam for the cache append wait contract. While it is
// armed, the sequence of every accepted cache append is recorded and each
// caller wait that observes one of those sequences through a successful
// completion is counted once. A focused test can then prove that an accepted
// append OID was waited before the submitting stage returned, which row
// contents alone cannot show. The seam carries no other state and is inert
// until a test arms it.
void arm_cache_append_wait_observation() noexcept;
void clear_cache_append_wait_observation() noexcept;
void observe_cache_append_wait(std::uint64_t sequence) noexcept;
[[nodiscard]] std::size_t observed_cache_append_waits() noexcept;

}  // namespace cpu_detail

class CpuDevice final : public Device {
public:
    CpuDevice(Allocator& allocator, QueueConfig queue_config);

    [[nodiscard]] BackendKind backend_kind() const noexcept override;
    [[nodiscard]] std::uint32_t backend_device() const noexcept override;
    [[nodiscard]] std::span<const DataType>
            supported_data_types() const noexcept override;
    [[nodiscard]] detail::RegistryState& registry_state() noexcept;
    [[nodiscard]] std::unique_ptr<Tensor> create_tensor(
            const TensorSpec& spec) override;
    [[nodiscard]] std::unique_ptr<RawWorkspace> create_workspace(
            std::size_t bytes) override;
    [[nodiscard]] std::unique_ptr<DeviceOps> create_ops() override;

private:
    Allocator& allocator_;
    detail::RegistryState registry_state_;
};

}  // namespace iom

