#pragma once

// Backend-neutral conformance harness: shared definitions (change 0001-tensor-view / 06).
//
// Provides the independent host-encoding model, logical-byte observation
// helpers, case parameters (observer, devices), token decoding, compile-time
// signature checks, and the backend-neutral ADD validation/lifetime fake. The
// remaining storage, copy, and capability scenarios build on these definitions.
// The harness never switches on
// BackendKind, never constructs a device, and includes no accelerator header,
// so a backend-specific test can include it unchanged.

#include <doctest/doctest.h>

#include <array>
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
// Independent test policy for the standard 23-leaf capability contract. Keep
// this literal oracle separate from every production capability declaration
// so narrowing, reordering, or other drift remains observable in conformance
// tests.
inline constexpr std::array<iom::DataType, 23> kStandardCapabilityOracle = {
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

inline void require_standard_capabilities(
        std::span<const iom::DataType> supported) {
    REQUIRE_EQ(supported.size(), kStandardCapabilityOracle.size());
    for (std::size_t i = 0;
         i < supported.size() && i < kStandardCapabilityOracle.size(); ++i) {
        CHECK_EQ(supported[i], kStandardCapabilityOracle[i]);
    }
}


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

constexpr std::uint64_t kTokenSequenceBits = 55;
constexpr std::uint64_t kTokenSequenceMask =
        (std::uint64_t{1} << kTokenSequenceBits) - 1;

constexpr std::uint8_t token_queue(iom::oid token) {
    return static_cast<std::uint8_t>(
            static_cast<std::uint64_t>(token) >> kTokenSequenceBits);
}

constexpr std::uint64_t token_sequence(iom::oid token) {
    return static_cast<std::uint64_t>(token) & kTokenSequenceMask;
}

// The public token layout: 55 sequence bits, queue ids in bits 55..62,
// so every id through 255 stays representable and no 56-bit decode exists.
static_assert(kTokenSequenceBits == 55);
static_assert(kTokenSequenceMask == (std::uint64_t{1} << 55) - 1);
static_assert(token_queue(0) == 0);
static_assert(token_sequence(0) == 0);
static_assert(token_queue((std::uint64_t{255} << 55) | 1) == 255);
static_assert(token_sequence((std::uint64_t{255} << 55) | 1) == 1);
static_assert(token_sequence(std::uint64_t{1} << 54)
              == (std::uint64_t{1} << 54));

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
                                           iom::TensorView&) noexcept>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::add),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           const iom::TensorView&,
                                           iom::TensorView&) noexcept>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::mul),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           const iom::TensorView&,
                                           iom::TensorView&) noexcept>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::silu),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           iom::TensorView&) noexcept>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::linear),
              iom::oid (iom::DeviceOps::*)(const iom::TensorView&,
                                           const iom::TensorView&,
                                           iom::TensorView&) noexcept>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::rmsnorm),
              iom::oid (iom::DeviceOps::*)(
                      const iom::TensorView&, iom::TensorView&,
                      const iom::TensorView&, float, size_t) noexcept>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::sdpa),
              iom::oid (iom::DeviceOps::*)(
                      const iom::TensorView&, const iom::TensorView&,
                      const iom::TensorView&, size_t, size_t, size_t,
                      iom::TensorView&) noexcept>);
static_assert(std::is_invocable_v<
              decltype(&iom::DeviceOps::silu), iom::DeviceOps*,
              const iom::TensorView&, iom::TensorView&>);
static_assert(!std::is_invocable_v<
              decltype(&iom::DeviceOps::silu), iom::DeviceOps*,
              iom::TensorView&, const iom::TensorView&>);
static_assert(!std::is_invocable_v<
              decltype(&iom::DeviceOps::copy), iom::DeviceOps*,
              const iom::Tensor&, iom::Tensor&>);


// ---------------------------------------------------------------------------
// Backend-neutral ADD policy fake. It consumes only the protected immutable
// request and the common registry seam; no arithmetic or backend runtime is
// involved.
// ---------------------------------------------------------------------------

class CommonAddQueue final : public iom::DeviceOps {
public:
    struct Record {
        std::uint64_t sequence;
        std::vector<std::size_t> result_shape;
        iom::detail::AddEntryRegistration entries;
        bool retained_failure;
    };

    explicit CommonAddQueue(const iom::Device& device)
            : iom::DeviceOps(device) {}

    void fail_next_after_acceptance() noexcept {
        fail_next_ = true;
    }

    [[nodiscard]] const std::vector<Record>& records() const noexcept {
        return records_;
    }

    [[nodiscard]] std::size_t registered_at(void* address) const {
        return state_.registry.snapshot_for(address).size();
    }

    void finish(std::uint64_t sequence) {
        for (const Record& record : records_) {
            if (record.sequence != sequence) {
                continue;
            }
            (void)iom::detail::release_or_invalidate_add_entries(
                    state_.registry, record.entries,
                    record.retained_failure, !record.retained_failure);
            complete(sequence);
            return;
        }
        throw std::invalid_argument("unknown common ADD sequence");
    }

protected:
    iom::oid add_impl(const AddRequest& request) override {
        iom::detail::Fence fence;
        fence.invoke = [](const iom::detail::Fence&) noexcept {
            return iom::detail::FenceResult::pending();
        };
        const bool retained_failure = std::exchange(fail_next_, false);
        return submit_add(
                request, state_, registry_queue_id_, fence,
                [this, retained_failure](
                        std::uint64_t sequence, const AddRequest& snapshot,
                        iom::detail::AddEntryRegistration entries) {
                    records_.push_back(
                            {sequence,
                             {snapshot.result_shape.dimensions().begin(),
                              snapshot.result_shape.dimensions().end()},
                             entries,
                             retained_failure});
                    if (retained_failure) {
                        commit_failure(
                                sequence,
                                std::make_exception_ptr(std::runtime_error(
                                        "common ADD retained failure")));
                    }
                });
    }

private:
    iom::detail::RegistryState state_;
    iom::detail::QueueId registry_queue_id_ =
            iom::detail::allocate_queue_id(state_);
    std::vector<Record> records_;
    bool fail_next_ = false;
};

inline void run_add_request_conformance(
        const ConformanceDevices& devices,
        ConformanceObserver* observer = nullptr) {
    const iom::TensorSpec matrix{
            iom::TensorShape{{2, 17, 33}}, iom::DataType::F32};
    const iom::TensorSpec scalar{
            iom::TensorShape{{1, 1}}, iom::DataType::F32};
    const iom::TensorSpec broadcast_result{
            iom::TensorShape{{2, 4, 3, 5, 17, 33}}, iom::DataType::F32};
    const iom::TensorSpec broadcast_lhs{
            iom::TensorShape{{2, 1, 3, 1, 17, 1}}, iom::DataType::F32};
    const iom::TensorSpec broadcast_rhs{
            iom::TensorShape{{1, 4, 1, 5, 1, 33}}, iom::DataType::F32};

    auto lhs = devices.candidate.create_tensor(matrix);
    auto rhs = devices.candidate.create_tensor(matrix);
    auto out = devices.candidate.create_tensor(matrix);
    auto foreign = devices.foreign.create_tensor(matrix);
    auto scalar_lhs = devices.candidate.create_tensor(scalar);
    auto scalar_rhs = devices.candidate.create_tensor(scalar);
    auto scalar_out = devices.candidate.create_tensor(scalar);
    auto broad_lhs = devices.candidate.create_tensor(broadcast_lhs);
    auto broad_rhs = devices.candidate.create_tensor(broadcast_rhs);
    auto broad_out = devices.candidate.create_tensor(broadcast_result);
    if (observer != nullptr) {
        observer->setup_complete();
    }

    CommonAddQueue queue(devices.candidate);
    const iom::oid first =
            queue.add(lhs->view(), rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(first));
    CHECK_EQ(token_sequence(first), 1);
    REQUIRE_EQ(queue.records().back().entries.count, 3);
    CHECK_EQ(queue.registered_at(lhs->view().native_handle()), 1);
    queue.finish(1);
    CHECK_NOTHROW(queue.wait(first));
    CHECK_NOTHROW(queue.wait(first));
    CHECK_EQ(queue.registered_at(lhs->view().native_handle()), 0);

    const iom::oid scalar_token =
            queue.add(scalar_lhs->view(), scalar_rhs->view(),
                      scalar_out->view());
    REQUIRE(iom::oid_is_token(scalar_token));
    CHECK_EQ(queue.records().back().result_shape,
             std::vector<std::size_t>({1, 1}));
    queue.finish(token_sequence(scalar_token));

    const iom::oid broadcast_token =
            queue.add(broad_lhs->view(), broad_rhs->view(),
                      broad_out->view());
    REQUIRE(iom::oid_is_token(broadcast_token));
    CHECK_EQ(
            queue.records().back().result_shape,
            std::vector<std::size_t>({2, 4, 3, 5, 17, 33}));
    queue.finish(token_sequence(broadcast_token));

    const iom::oid promoted =
            queue.add(scalar_lhs->view(), rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(promoted));
    CHECK_EQ(queue.records().back().result_shape,
             std::vector<std::size_t>({2, 17, 33}));
    queue.finish(token_sequence(promoted));

    const std::size_t accepted = queue.records().size();
    CHECK_EQ(
            queue.add(foreign->view(), rhs->view(), out->view()),
            iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_EQ(
            queue.add(lhs->view(), rhs->view(), scalar_out->view()),
            iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_EQ(queue.records().size(), accepted);

    const iom::oid aliased =
            queue.add(out->view(), out->view(), out->view());
    REQUIRE(iom::oid_is_token(aliased));
    REQUIRE_EQ(queue.records().back().entries.count, 1);
    queue.finish(token_sequence(aliased));

    queue.fail_next_after_acceptance();
    const iom::oid failed =
            queue.add(lhs->view(), rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(failed));
    queue.finish(token_sequence(failed));
    expect_repeated_runtime_failure(queue, failed);

    if (observer != nullptr) {
        observer->case_complete();
    }
}
inline void run_add_rank_boundary_conformance(iom::Device& candidate) {
    const iom::TensorSpec input_spec{
            iom::TensorShape{{17, 33}}, iom::DataType::U8};
    const std::vector<std::byte> lhs_bytes =
            encode_logical(input_spec, 0x13579BDF2468ACE0ull);
    const std::vector<std::byte> rhs_bytes =
            encode_logical(input_spec, 0x0ECA8642FDB97531ull);
    auto rhs = candidate.create_tensor(input_spec);
    rhs->view().copy_from_host(rhs_bytes);
    auto queue = candidate.create_ops();
    for (const std::size_t rank : {8u, 9u, 16u, 17u}) {
        std::vector<std::size_t> dimensions(rank, 1);
        dimensions[rank - 2] = 17;
        dimensions[rank - 1] = 33;
        const iom::TensorSpec lhs_spec{
                iom::TensorShape{dimensions}, iom::DataType::U8};
        const iom::TensorSpec output_spec{
                iom::TensorShape{dimensions}, iom::DataType::U8};
        auto lhs = candidate.create_tensor(lhs_spec);
        auto output = candidate.create_tensor(output_spec);
        lhs->view().copy_from_host(lhs_bytes);
        output->view().copy_from_host(
                std::vector<std::byte>(
                        output_spec.logical_nbytes(), std::byte{0}));
        const iom::oid token =
                queue->add(lhs->view(), rhs->view(), output->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        CHECK_NOTHROW(queue->wait(token));
        const std::vector<std::byte> actual =
                read_logical(output->view());
        REQUIRE_EQ(actual.size(), lhs_bytes.size());
        std::vector<std::byte> expected(actual.size());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            expected[index] = std::byte{
                    static_cast<unsigned char>(
                            static_cast<unsigned>(lhs_bytes[index])
                            + static_cast<unsigned>(rhs_bytes[index]))};
        }
        CHECK_EQ(actual, expected);
    }
}

}  // namespace iom_conformance
