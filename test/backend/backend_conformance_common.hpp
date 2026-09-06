#pragma once

// Backend-neutral conformance harness: shared definitions (change 0001-tensor-view / 06).
//
// Provides the independent host-encoding model, logical-byte observation
// helpers, case parameters (observer, devices), token decoding, span helper,
// and compile-time checks of every compute method's view signature. None of
// these depend on storage, copy, lifetime, or capability scenarios; the other
// parts of the harness build on top of them. The harness never switches on
// BackendKind, never constructs a device, and includes no accelerator header,
// so a backend-specific test can include it unchanged.

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

// ---------------------------------------------------------------------------
// Independent host-encoding model. The bit writer follows the host encoding
// directly (least-significant bit first at bit offset element_index * width)
// and shares no code with any backend.
// ---------------------------------------------------------------------------

inline std::size_t bits_of(iom::DataType type) {
    switch (type) {
        case iom::DataType::BOOL: return 8;
        case iom::DataType::I2:
        case iom::DataType::U2: return 2;
        case iom::DataType::I4:
        case iom::DataType::U4:
        case iom::DataType::F4_E2M1: return 4;
        case iom::DataType::F6_E2M3:
        case iom::DataType::F6_E3M2: return 6;
        case iom::DataType::I8:
        case iom::DataType::U8:
        case iom::DataType::F8_E4M3FN:
        case iom::DataType::F8_E5M2:
        case iom::DataType::F8_E8M0: return 8;
        case iom::DataType::I16:
        case iom::DataType::U16:
        case iom::DataType::F16:
        case iom::DataType::BF16: return 16;
        case iom::DataType::I32:
        case iom::DataType::U32:
        case iom::DataType::F32: return 32;
        case iom::DataType::I64:
        case iom::DataType::U64:
        case iom::DataType::F64: return 64;
    }
    REQUIRE_MESSAGE(false, "bits_of: unknown DataType");
    return 0;
}

inline void write_bits(
        unsigned char* base, std::size_t bit_offset, std::size_t nbits,
        std::uint64_t value) {
    for (std::size_t i = 0; i < nbits; ++i) {
        const std::size_t bit = bit_offset + i;
        unsigned char& byte = base[bit / 8];
        const unsigned char mask =
                static_cast<unsigned char>(1u << (bit % 8));
        if ((value >> i) & 1) {
            byte |= mask;
        } else {
            byte &= static_cast<unsigned char>(~mask);
        }
    }
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Deterministic element content as a function of the logical element index
// and a salt. BOOL is restricted to canonical zero and one.
inline std::uint64_t element_pattern(
        iom::DataType type, std::size_t linear, std::uint64_t salt) {
    const std::uint64_t raw = splitmix64(
            splitmix64(
                    static_cast<std::uint64_t>(linear) * 0x9E3779B97F4A7C15ull)
            ^ salt);
    if (type == iom::DataType::BOOL) {
        return raw & 1;
    }
    const std::size_t bits = bits_of(type);
    if (bits >= 64) {
        return raw;
    }
    return raw & ((std::uint64_t{1} << bits) - 1);
}

// Contiguous row-major host encoding of one specification's logical bytes.
inline std::vector<std::byte> encode_logical(
        const iom::TensorSpec& spec, std::uint64_t salt) {
    const std::size_t bits = bits_of(spec.data_type);
    const std::size_t count = spec.shape.element_count();
    std::vector<std::byte> buffer(spec.logical_nbytes(), std::byte{0});
    auto* base = reinterpret_cast<unsigned char*>(buffer.data());
    for (std::size_t linear = 0; linear < count; ++linear) {
        write_bits(
                base, linear * bits, bits,
                element_pattern(spec.data_type, linear, salt));
    }
    return buffer;
}

// ---------------------------------------------------------------------------
// Logical-byte observation. Readbacks are prefilled with a sentinel so a
// backend that leaks padding, skips bytes, or leaves output tail bits set
// fails the bit-for-bit comparison instead of cancelling a round trip.
// ---------------------------------------------------------------------------

constexpr std::byte kReadbackSentinel{0xAA};

inline std::vector<std::byte> read_logical(const iom::TensorView& view) {
    std::vector<std::byte> buffer(view.spec().logical_nbytes(), kReadbackSentinel);
    view.copy_to_host(buffer);
    return buffer;
}

// First byte position where the view's logical readback diverges from the
// expected encoding, or nullopt. Exposed so harness consumers can prove that
// a deliberately perturbed layout or view map is detected.
inline std::optional<std::size_t> first_logical_mismatch(
        const iom::TensorView& view, std::span<const std::byte> expected) {
    REQUIRE_EQ(expected.size(), view.spec().logical_nbytes());
    const std::vector<std::byte> actual = read_logical(view);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            return i;
        }
    }
    return std::nullopt;
}

inline void require_logical_bytes(
        const iom::TensorView& view, std::span<const std::byte> expected,
        std::string_view context) {
    const std::optional<std::size_t> mismatch =
            first_logical_mismatch(view, expected);
    REQUIRE_MESSAGE(!mismatch.has_value(),
                    context << ": logical bytes diverge at byte "
                            << mismatch.value_or(0) << " of " << expected.size());
}

// ---------------------------------------------------------------------------
// Case parameters.
// ---------------------------------------------------------------------------

class ConformanceObserver {
public:
    virtual ~ConformanceObserver() = default;

    // Fired after a case created its tensors and before its first transfer,
    // transform, or queue operation.
    virtual void setup_complete() = 0;

    // Fired when the case finished every transfer and operation, before the
    // case's tensors are destroyed.
    virtual void case_complete() = 0;
};

// Observes harness phase boundaries and exposes the armed window in which
// tensor-storage allocators must stay silent.
class TrafficGate final : public ConformanceObserver {
public:
    void setup_complete() override { armed_ = true; }
    void case_complete() override { armed_ = false; }
    ~TrafficGate() override { armed_ = false; }

    [[nodiscard]] bool armed() const noexcept { return armed_; }

private:
    bool armed_ = false;
};

struct ConformanceDevices {
    // CPU reference device every observable result is compared against.
    iom::Device& reference;
    // Device under test.
    iom::Device& candidate;
    // An independent instance of the candidate's own backend with the same
    // backend ordinal; its views must be rejected by candidate queues.
    iom::Device& foreign;
};

// Requires a deferred DeviceOps failure to rethrow the identical
// runtime_error message on every repeat wait.
inline void expect_repeated_runtime_failure(
        iom::DeviceOps& queue, iom::oid token) {
    std::string message;
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool caught = false;
        try {
            queue.wait(token);
        } catch (const std::runtime_error& error) {
            caught = true;
            if (message.empty()) {
                message = error.what();
            } else {
                CHECK_EQ(std::string_view(error.what()), message);
            }
        }
        CHECK(caught);
    }
    CHECK_FALSE(message.empty());
}

// ---------------------------------------------------------------------------
// Token decoding.
// ---------------------------------------------------------------------------

constexpr std::uint64_t kTokenSequenceBits = 56;
constexpr std::uint64_t kTokenSequenceMask =
        (std::uint64_t{1} << kTokenSequenceBits) - 1;

inline std::uint8_t token_queue(iom::oid token) {
    return static_cast<std::uint8_t>(token >> kTokenSequenceBits);
}

inline std::uint64_t token_sequence(iom::oid token) {
    return token & kTokenSequenceMask;
}

// A span over a temporary initializer-list array; valid for the full
// expression that consumes it.
inline std::span<const std::size_t> span_of(
        std::initializer_list<std::size_t> values) {
    return {values.begin(), values.size()};
}

// ---------------------------------------------------------------------------
// Compile-time checks of every compute method's view signature. No backend
// header participates: the signatures live on the common DeviceOps base.
// ---------------------------------------------------------------------------

static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::copy),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           iom::TensorView&)>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::add),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           const iom::TensorView&,
                                           iom::TensorView&)>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::mul),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           const iom::TensorView&,
                                           iom::TensorView&)>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::silu),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           iom::TensorView&)>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::linear),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           const iom::TensorView&,
                                           iom::TensorView&)>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::rmsnorm),
              iom::oid (iom::DeviceOps::*)(
                      const iom::TensorView&, iom::TensorView&,
                      const iom::TensorView&, float, size_t)>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::sdpa),
              iom::oid (iom::DeviceOps::*)(
                      const iom::TensorView&, const iom::TensorView&,
                      const iom::TensorView&, size_t, size_t, size_t,
                      iom::TensorView&)>);
static_assert(std::is_invocable_v<
              decltype(&iom::DeviceOps::silu), iom::DeviceOps*,
              const iom::TensorView&, iom::TensorView&>);
static_assert(!std::is_invocable_v<
              decltype(&iom::DeviceOps::silu), iom::DeviceOps*,
              iom::TensorView&, const iom::TensorView&>);
static_assert(!std::is_invocable_v<
              decltype(&iom::DeviceOps::copy), iom::DeviceOps*,
              const iom::Tensor&, iom::Tensor&>);

}  // namespace iom_conformance
