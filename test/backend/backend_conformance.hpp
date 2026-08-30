#pragma once

// Backend-neutral conformance harness (change 0001-tensor-view / 06).
//
// Proves storage, host-transfer, transformed-view, asynchronous-copy, error,
// and lifetime behavior of a candidate device against a CPU reference device.
// The invoking backend test supplies every device instance and the explicit
// supported DataType table; the harness never switches on BackendKind, never
// constructs a device, and includes no accelerator header, so a
// backend-specific test can include it unchanged.

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

struct ConformanceDevices {
    // CPU reference device every observable result is compared against.
    iom::Device& reference;
    // Device under test.
    iom::Device& candidate;
    // An independent instance of the candidate's own backend with the same
    // backend ordinal; its views must be rejected by candidate queues.
    iom::Device& foreign;
};

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
// Transformed-view case matrix: final-dimension padding, leading ranks up to
// and above four, stepped slices, selects, every leading permutation when
// enumerable, contiguous reshapes, nested transforms, and rank-two
// boundaries. Builders only use public TensorView transforms.
// ---------------------------------------------------------------------------

struct ViewCase {
    std::string label;
    std::function<iom::TensorView(const iom::TensorView&)> build;
};

inline std::vector<ViewCase> view_cases_for(const iom::TensorSpec& owner_spec) {
    const std::span<const std::size_t> dims = owner_spec.shape.dimensions();
    const std::size_t leading = dims.size() - 2;

    std::vector<ViewCase> cases;
    cases.push_back({"full", [](const iom::TensorView& full) { return full; }});

    if (leading >= 1 && dims[0] >= 4) {
        const std::size_t d0 = dims[0];
        cases.push_back(
                {"stepped slice of the first leading axis",
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 0, d0 / 2, 2);
                 }});
    }
    if (leading >= 1 && dims[0] >= 2) {
        const std::size_t d0 = dims[0];
        cases.push_back(
                {"interior dense slice of the first leading axis",
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 1, d0 - 1);
                 }});
    }
    if (leading >= 1) {
        const std::size_t last_axis = leading - 1;
        const std::size_t last_dim = dims[last_axis];
        cases.push_back(
                {"select of the first leading axis",
                 [](const iom::TensorView& full) { return full.select(0, 0); }});
        cases.push_back(
                {"select of the last leading axis at its last index",
                 [last_axis, last_dim](const iom::TensorView& full) {
                     return full.select(last_axis, last_dim - 1);
                 }});
    }

    if (leading == 0) {
        cases.push_back(
                {"empty permute",
                 [](const iom::TensorView& full) {
                     const std::vector<std::size_t> empty;
                     return full.permute(std::span<const std::size_t>{empty});
                 }});
    } else if (leading <= 4) {
        std::vector<std::size_t> order(leading);
        std::iota(order.begin(), order.end(), std::size_t{0});
        do {
            const std::vector<std::size_t> captured = order;
            std::string label = "permute";
            for (const std::size_t axis : captured) {
                label += ' ';
                label += std::to_string(axis);
            }
            cases.push_back(
                    {std::move(label),
                     [captured](const iom::TensorView& full) {
                         return full.permute(
                                 std::span<const std::size_t>{captured});
                     }});
        } while (std::next_permutation(order.begin(), order.end()));
    }

    if (leading >= 1) {
        std::size_t product = 1;
        for (std::size_t i = 0; i < leading; ++i) {
            product *= dims[i];
        }
        cases.push_back(
                {"reshape to one leading axis",
                 [product](const iom::TensorView& full) {
                     return full.reshape_leading(span_of({product}));
                 }});
    }
    if (leading >= 2) {
        std::size_t head = 1;
        for (std::size_t i = 0; i < leading - 1; ++i) {
            head *= dims[i];
        }
        const std::size_t last_dim = dims[leading - 1];
        cases.push_back(
                {"merge the trailing pair of leading axes",
                 [head, last_dim](const iom::TensorView& full) {
                     return full.reshape_leading(span_of({head, last_dim}));
                 }});
    }
    if (leading >= 1 && dims[0] >= 4 && dims[0] % 2 == 0) {
        std::vector<std::size_t> split{2, dims[0] / 2};
        for (std::size_t i = 1; i < leading; ++i) {
            split.push_back(dims[i]);
        }
        cases.push_back(
                {"split the first leading axis",
                 [split](const iom::TensorView& full) {
                     return full.reshape_leading(
                             std::span<const std::size_t>{split});
                 }});
    }

    if (leading >= 3 && dims[0] >= 2 && dims[1] >= 3) {
        cases.push_back(
                {"select, stepped slice, and permute",
                 [](const iom::TensorView& full) {
                     return full.select(0, 1).slice(0, 0, 2, 2).permute(
                             span_of({1, 0}));
                 }});
    }
    if (leading >= 3 && dims[0] >= 2 && dims[2] >= 4) {
        const std::size_t d1 = dims[1];
        const std::size_t d2 = dims[2];
        cases.push_back(
                {"dense slice, reshape, select, and stepped slice",
                 [d1, d2](const iom::TensorView& full) {
                     return full.slice(0, 0, 2)
                             .reshape_leading(span_of({2 * d1, d2}))
                             .select(0, 2 * d1 - 1)
                             .slice(0, 0, d2 / 2, 2);
                 }});
    }

    return cases;
}

inline const std::vector<std::vector<std::size_t>>& transfer_owner_shapes() {
    static const std::vector<std::vector<std::size_t>> shapes = {
        {16, 16},                   // rank-two tile boundary
        {1, 17},                    // vector representation, row padding
        {17, 33},                   // both final dimensions padded
        {4, 3, 17, 33},             // leading rank two
        {2, 3, 4, 17, 33},          // leading rank three
        {2, 2, 2, 3, 17, 33},       // rank six: leading rank above three
    };
    return shapes;
}

// ---------------------------------------------------------------------------
// Copy case matrix: full windows, different source/destination offsets and
// strides, permuted windows, nested transforms, and the identical-window
// no-op. Source and destination windows never overlap, so destination bytes
// are deterministic after each copy.
// ---------------------------------------------------------------------------

struct CopyCase {
    std::string label;
    std::function<iom::TensorView(const iom::TensorView&)> build_source;
    std::function<iom::TensorView(const iom::TensorView&)> build_destination;
};

inline std::vector<CopyCase> copy_cases_for(const iom::TensorSpec& owner_spec) {
    const std::span<const std::size_t> dims = owner_spec.shape.dimensions();
    const std::size_t leading = dims.size() - 2;

    std::vector<CopyCase> cases;
    cases.push_back(
            {"full to full",
             [](const iom::TensorView& full) { return full; },
             [](const iom::TensorView& full) { return full; }});
    cases.push_back(
            {"identical window is a waitable no-op",
             [](const iom::TensorView& full) { return full; },
             [](const iom::TensorView& full) { return full; }});

    if (leading >= 1 && dims[0] >= 2) {
        const std::size_t d0 = dims[0];
        cases.push_back(
                {"offset source to earlier destination",
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 1, d0 - 1);
                 },
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 0, d0 - 1);
                 }});
    }
    if (leading >= 1 && dims[0] >= 4) {
        const std::size_t d0 = dims[0];
        cases.push_back(
                {"stepped source to stepped destination",
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 0, d0 / 2, 2);
                 },
                 [d0](const iom::TensorView& full) {
                     return full.slice(0, 1, d0 / 2, 2);
                 }});
    }
    if (leading >= 2) {
        std::vector<std::size_t> reversed(leading);
        std::iota(reversed.rbegin(), reversed.rend(), std::size_t{0});
        cases.push_back(
                {"permuted source to permuted destination",
                 [reversed](const iom::TensorView& full) {
                     return full.permute(std::span<const std::size_t>{reversed});
                 },
                 [reversed](const iom::TensorView& full) {
                     return full.permute(std::span<const std::size_t>{reversed});
                 }});
    }
    if (leading >= 2 && dims[1] >= 3) {
        cases.push_back(
                {"selected and stepped source to selected destination",
                 [](const iom::TensorView& full) {
                     return full.select(0, 1).slice(0, 0, 2, 2);
                 },
                 [](const iom::TensorView& full) {
                     return full.select(0, 0).slice(0, 0, 2, 2);
                 }});
    }

    return cases;
}

inline const std::vector<std::vector<std::size_t>>& copy_owner_shapes() {
    static const std::vector<std::vector<std::size_t>> shapes = {
        {17, 33},                   // rank-two padded
        {2, 3, 16, 16},             // exact tiles
        {2, 3, 4, 17, 33},          // leading rank three with padding
        {2, 2, 2, 3, 17, 33},       // rank six
    };
    return shapes;
}

// ---------------------------------------------------------------------------
// Deterministic deferred fake. Submissions are recorded with the exact view,
// owner, and native-handle addresses; the test thread plays the in-order
// worker by completing sequences through the common DeviceOps machinery.
// ---------------------------------------------------------------------------

class DeferredCopyQueue final : public iom::DeviceOps {
public:
    using iom::DeviceOps::complete;

    struct Record {
        std::uint64_t sequence;
        const void* source_owner;
        const iom::TensorView* source;
        const void* source_handle;
        void* destination_owner;
        iom::TensorView* destination;
        void* destination_handle;
    };

    [[nodiscard]] const std::vector<Record>& records() const noexcept {
        return records_;
    }

    // Submission that also records the owner addresses, mirroring what a
    // real queue observes about its operands.
    iom::oid copy(
            const iom::Tensor& source_owner, const iom::TensorView& source,
            iom::Tensor& destination_owner, iom::TensorView& destination) {
        const iom::oid token = copy(source, destination);
        records_.back().source_owner = &source_owner;
        records_.back().destination_owner = &destination_owner;
        return token;
    }

    iom::oid copy(
            const iom::TensorView& source,
            iom::TensorView& destination) override {
        return submit([&](std::uint64_t sequence) {
            records_.push_back(
                    {sequence, nullptr, &source, source.native_handle(),
                     nullptr, &destination, destination.native_handle()});
        });
    }

    // View-less submission used to observe queue identity and sequence
    // allocation directly.
    iom::oid probe() {
        return submit([&](std::uint64_t sequence) {
            records_.push_back(
                    {sequence, nullptr, nullptr, nullptr, nullptr, nullptr,
                     nullptr});
        });
    }

    // Submission-time view, owner, and native-handle addresses must equal
    // the live operands at completion time.
    void expect_stable(
            std::uint64_t sequence,
            const iom::Tensor& source_owner, const iom::TensorView& source,
            iom::Tensor& destination_owner, iom::TensorView& destination) const {
        const Record* record = find(sequence);
        REQUIRE_MESSAGE(record != nullptr,
                        "no deferred record for sequence " << sequence);
        CHECK_EQ(record->source_owner, &source_owner);
        CHECK_EQ(record->source, &source);
        CHECK_EQ(record->source_handle, source.native_handle());
        CHECK_EQ(record->destination_owner, &destination_owner);
        CHECK_EQ(record->destination, &destination);
        CHECK_EQ(record->destination_handle, destination.native_handle());
    }

    iom::oid add(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement add");
    }
    iom::oid mul(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement mul");
    }
    iom::oid silu(const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement silu");
    }
    iom::oid linear(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement linear");
    }
    iom::oid rmsnorm(const iom::TensorView&, iom::TensorView&, const iom::TensorView&,
                     float, size_t) override {
        throw std::runtime_error("deferred fake does not implement rmsnorm");
    }
    iom::oid sdpa(const iom::TensorView&, const iom::TensorView&, const iom::TensorView&,
                  size_t, size_t, size_t, iom::TensorView&) override {
        throw std::runtime_error("deferred fake does not implement sdpa");
    }

private:
    [[nodiscard]] const Record* find(std::uint64_t sequence) const {
        for (const Record& record : records_) {
            if (record.sequence == sequence) {
                return &record;
            }
        }
        return nullptr;
    }

    std::vector<Record> records_;
};

// ---------------------------------------------------------------------------
// Instrumented queue: journals submissions, completions, and its own
// destruction so the test can prove destruction neither synchronizes on
// outstanding work nor cancels it.
// ---------------------------------------------------------------------------

class InstrumentedQueue final : public iom::DeviceOps {
public:
    struct Event {
        enum class Kind { submit, complete, destroy_begin, destroy_end };

        Kind kind;
        std::uint64_t sequence;
    };

    explicit InstrumentedQueue(
            std::shared_ptr<std::vector<Event>> journal)
            : journal_(std::move(journal)) {}

    ~InstrumentedQueue() override {
        journal_->push_back({Event::Kind::destroy_begin, 0});
        // Nothing here waits for, completes, or cancels the outstanding
        // sequences; the common base only releases the queue id.
        journal_->push_back({Event::Kind::destroy_end, 0});
    }

    iom::oid probe() {
        return submit([&](std::uint64_t sequence) {
            journal_->push_back({Event::Kind::submit, sequence});
        });
    }

    // Test-driven completion of one sequence.
    void finish(std::uint64_t sequence) {
        journal_->push_back({Event::Kind::complete, sequence});
        complete(sequence);
    }

    iom::oid copy(const iom::TensorView&, iom::TensorView&) override {
        return probe();
    }
    iom::oid add(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement add");
    }
    iom::oid mul(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement mul");
    }
    iom::oid silu(const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement silu");
    }
    iom::oid linear(const iom::TensorView&, const iom::TensorView&, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement linear");
    }
    iom::oid rmsnorm(const iom::TensorView&, iom::TensorView&, const iom::TensorView&,
                     float, size_t) override {
        throw std::runtime_error("instrumented queue does not implement rmsnorm");
    }
    iom::oid sdpa(const iom::TensorView&, const iom::TensorView&, const iom::TensorView&,
                  size_t, size_t, size_t, iom::TensorView&) override {
        throw std::runtime_error("instrumented queue does not implement sdpa");
    }

private:
    std::shared_ptr<std::vector<Event>> journal_;
};

// ---------------------------------------------------------------------------
// Conformance scenarios.
// ---------------------------------------------------------------------------

// Storage and host transfers: matching reference and candidate tensors are
// seeded with identical logical host bytes, full and transformed views
// transfer independently generated encodings on both devices, and every
// readback is compared bit-for-bit against the expected encoding.
inline void run_storage_and_transfer_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    for (const iom::DataType type : supported_types) {
        CAPTURE(static_cast<int>(type));
        for (const std::vector<std::size_t>& dimensions :
             transfer_owner_shapes()) {
            CAPTURE(dimensions);
            const iom::TensorSpec spec{iom::TensorShape{dimensions}, type};

            auto reference = devices.reference.create_tensor(spec);
            auto candidate = devices.candidate.create_tensor(spec);
            if (observer != nullptr) {
                observer->setup_complete();
            }

            const std::vector<std::byte> seeded = encode_logical(spec, 0xABCD);
            reference->view().copy_from_host(seeded);
            candidate->view().copy_from_host(seeded);
            require_logical_bytes(
                    reference->view(), seeded, "reference full view");
            require_logical_bytes(
                    candidate->view(), seeded, "candidate full view");

            std::uint64_t salt = 1;
            for (const ViewCase& view_case : view_cases_for(spec)) {
                CAPTURE(view_case.label);
                iom::TensorView reference_view =
                        view_case.build(reference->view());
                iom::TensorView candidate_view =
                        view_case.build(candidate->view());
                REQUIRE(reference_view.spec() == candidate_view.spec());

                const std::vector<std::byte> pattern =
                        encode_logical(reference_view.spec(), salt);
                ++salt;
                reference_view.copy_from_host(pattern);
                candidate_view.copy_from_host(pattern);
                require_logical_bytes(
                        reference_view, pattern,
                        std::string("reference ") + view_case.label);
                require_logical_bytes(
                        candidate_view, pattern,
                        std::string("candidate ") + view_case.label);
            }

            if (observer != nullptr) {
                observer->case_complete();
            }
        }
    }
}

// Asynchronous copies: every supported type exercises the in-order
// submission chain and full, offset, stepped, permuted, nested, and
// identical-window copies on both devices; destination logical bytes are
// compared bit-for-bit against the independently generated source encoding.
inline void run_async_copy_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    for (const iom::DataType type : supported_types) {
        CAPTURE(static_cast<int>(type));

        // Same-queue submissions execute in call order without intervening
        // host waits, and waits are idempotent after success.
        {
            const iom::TensorSpec spec{
                    iom::TensorShape{{2, 3, 16, 16}}, type};
            auto first = devices.candidate.create_tensor(spec);
            auto second = devices.candidate.create_tensor(spec);
            auto staging = devices.candidate.create_tensor(spec);
            auto destination = devices.candidate.create_tensor(spec);
            if (observer != nullptr) {
                observer->setup_complete();
            }

            const std::vector<std::byte> first_pattern =
                    encode_logical(spec, 11);
            const std::vector<std::byte> second_pattern =
                    encode_logical(spec, 12);
            first->view().copy_from_host(first_pattern);
            second->view().copy_from_host(second_pattern);

            auto queue = devices.candidate.create_ops();
            const iom::oid one =
                    queue->copy(first->view(), staging->view());
            const iom::oid two =
                    queue->copy(staging->view(), destination->view());
            const iom::oid three =
                    queue->copy(second->view(), destination->view());
            CHECK_NE(token_queue(one), 0);
            CHECK_EQ(token_sequence(one), 1);
            CHECK_EQ(token_sequence(two), 2);
            CHECK_EQ(token_sequence(three), 3);
            CHECK_EQ(token_queue(one), token_queue(three));

            queue->wait(three);
            require_logical_bytes(
                    destination->view(), second_pattern,
                    "candidate chained destination");
            require_logical_bytes(
                    staging->view(), first_pattern, "candidate staging");
            queue->wait(one);
            queue->wait(three);
            queue.reset();

            if (observer != nullptr) {
                observer->case_complete();
            }
        }

        for (const std::vector<std::size_t>& dimensions : copy_owner_shapes()) {
            CAPTURE(dimensions);
            const iom::TensorSpec spec{iom::TensorShape{dimensions}, type};
            auto reference_source = devices.reference.create_tensor(spec);
            auto reference_destination =
                    devices.reference.create_tensor(spec);
            auto candidate_source = devices.candidate.create_tensor(spec);
            auto candidate_destination = devices.candidate.create_tensor(spec);
            if (observer != nullptr) {
                observer->setup_complete();
            }

            auto reference_queue = devices.reference.create_ops();
            auto candidate_queue = devices.candidate.create_ops();

            std::uint64_t salt = 100;
            for (const CopyCase& copy_case : copy_cases_for(spec)) {
                CAPTURE(copy_case.label);
                iom::TensorView reference_source_view =
                        copy_case.build_source(reference_source->view());
                iom::TensorView reference_destination_view =
                        copy_case.build_destination(reference_destination->view());
                iom::TensorView candidate_source_view =
                        copy_case.build_source(candidate_source->view());
                iom::TensorView candidate_destination_view =
                        copy_case.build_destination(candidate_destination->view());
                REQUIRE(candidate_source_view.spec()
                        == candidate_destination_view.spec());

                const std::vector<std::byte> pattern =
                        encode_logical(candidate_source_view.spec(), salt);
                ++salt;
                reference_source_view.copy_from_host(pattern);
                candidate_source_view.copy_from_host(pattern);

                const iom::oid reference_token = reference_queue->copy(
                        reference_source_view, reference_destination_view);
                const iom::oid candidate_token = candidate_queue->copy(
                        candidate_source_view, candidate_destination_view);
                reference_queue->wait(reference_token);
                candidate_queue->wait(candidate_token);

                require_logical_bytes(
                        reference_destination_view, pattern,
                        std::string("reference ") + copy_case.label);
                require_logical_bytes(
                        candidate_destination_view, pattern,
                        std::string("candidate ") + copy_case.label);
            }

            // Every successful submission was waited; queues die explicitly.
            reference_queue.reset();
            candidate_queue.reset();
            if (observer != nullptr) {
                observer->case_complete();
            }
        }
    }
}

// Copy validation: metadata mismatch and foreign-device views are rejected
// before any write and before consuming a sequence number.
inline void run_copy_error_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    for (const iom::DataType type : supported_types) {
        CAPTURE(static_cast<int>(type));
        const iom::DataType other_type =
                type == iom::DataType::F32 ? iom::DataType::I16
                                           : iom::DataType::F32;
        const iom::TensorSpec spec{iom::TensorShape{{2, 3, 17, 33}}, type};

        auto source = devices.candidate.create_tensor(spec);
        auto destination = devices.candidate.create_tensor(spec);
        auto other_shape = devices.candidate.create_tensor(
                iom::TensorSpec{iom::TensorShape{{3, 3, 17, 33}}, type});
        auto other_type_tensor = devices.candidate.create_tensor(
                iom::TensorSpec{iom::TensorShape{{2, 3, 17, 33}}, other_type});
        auto reference_view_tensor = devices.reference.create_tensor(spec);
        auto foreign_tensor = devices.foreign.create_tensor(spec);
        if (observer != nullptr) {
            observer->setup_complete();
        }

        auto queue = devices.candidate.create_ops();

        const std::vector<std::byte> source_pattern =
                encode_logical(spec, 21);
        const std::vector<std::byte> destination_pattern =
                encode_logical(spec, 22);
        source->view().copy_from_host(source_pattern);
        destination->view().copy_from_host(destination_pattern);

        CHECK_THROWS_AS(
                queue->copy(other_shape->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(destination->view(), other_shape->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(other_type_tensor->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(destination->view(), other_type_tensor->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(reference_view_tensor->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(destination->view(), reference_view_tensor->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(foreign_tensor->view(), destination->view()),
                std::invalid_argument);
        CHECK_THROWS_AS(
                queue->copy(destination->view(), foreign_tensor->view()),
                std::invalid_argument);

        // No rejected submission wrote or consumed a sequence.
        require_logical_bytes(
                destination->view(), destination_pattern,
                "destination after rejections");
        const iom::oid valid = queue->copy(source->view(), destination->view());
        CHECK_EQ(token_sequence(valid), 1);
        queue->wait(valid);
        require_logical_bytes(
                destination->view(), source_pattern,
                "destination after the valid copy");

        // An identical-window copy still submits a waitable no-op.
        const iom::oid no_op =
                queue->copy(destination->view(), destination->view());
        CHECK_EQ(token_sequence(no_op), 2);
        queue->wait(no_op);
        require_logical_bytes(
                destination->view(), source_pattern,
                "destination after the no-op copy");
        queue.reset();

        if (observer != nullptr) {
            observer->case_complete();
        }
    }
}

// Synchronous transfer failures: wrong span sizes and non-canonical BOOL
// bytes are rejected, and metadata, owner identity, and the native handle
// are unchanged afterwards even though destination values may be
// unspecified.
inline void run_transfer_error_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    for (const iom::DataType type : supported_types) {
        CAPTURE(static_cast<int>(type));
        const iom::TensorSpec spec{iom::TensorShape{{2, 3, 17, 33}}, type};
        for (iom::Device* device :
             {&devices.reference, &devices.candidate}) {
            auto tensor = device->create_tensor(spec);
            if (observer != nullptr) {
                observer->setup_complete();
            }
            iom::TensorView& view = tensor->view();

            const std::vector<std::byte> seeded =
                    encode_logical(spec, static_cast<std::uint64_t>(31));
            view.copy_from_host(seeded);

            const iom::TensorSpec spec_before = view.spec();
            const std::size_t offset_before = view.plane_offset();
            const std::vector<std::size_t> strides_before(
                    view.plane_strides().begin(), view.plane_strides().end());
            const iom::Device* device_before = &view.device();
            const void* handle_before = view.native_handle();

            std::vector<std::byte> short_write(seeded);
            short_write.pop_back();
            CHECK_THROWS_AS(view.copy_from_host(short_write),
                            std::invalid_argument);
            std::vector<std::byte> long_write(seeded);
            long_write.push_back(std::byte{0});
            CHECK_THROWS_AS(view.copy_from_host(long_write),
                            std::invalid_argument);

            std::vector<std::byte> short_read(
                    spec.logical_nbytes() - 1, kReadbackSentinel);
            CHECK_THROWS_AS(view.copy_to_host(short_read),
                            std::invalid_argument);
            CHECK(std::all_of(
                    short_read.begin(), short_read.end(),
                    [](std::byte value) { return value == kReadbackSentinel; }));
            std::vector<std::byte> long_read(
                    spec.logical_nbytes() + 1, kReadbackSentinel);
            CHECK_THROWS_AS(view.copy_to_host(long_read),
                            std::invalid_argument);
            CHECK(std::all_of(
                    long_read.begin(), long_read.end(),
                    [](std::byte value) { return value == kReadbackSentinel; }));

            if (type == iom::DataType::BOOL) {
                for (const std::size_t position :
                     {std::size_t{0}, seeded.size() / 2, seeded.size() - 1}) {
                    std::vector<std::byte> invalid = seeded;
                    invalid[position] = std::byte{2};
                    CAPTURE(position);
                    CHECK_THROWS_AS(view.copy_from_host(invalid),
                                    std::invalid_argument);
                }
                // Canonical zero and one bytes round-trip; non-canonical
                // bytes never wrote.
                require_logical_bytes(
                        view, seeded, "storage untouched after invalid BOOL");
            }

            // Metadata, owner identity, and native handle survive every
            // synchronous failure; values are deliberately not compared.
            CHECK(view.spec() == spec_before);
            CHECK_EQ(view.plane_offset(), offset_before);
            CHECK(std::equal(
                    strides_before.begin(), strides_before.end(),
                    view.plane_strides().begin(), view.plane_strides().end()));
            CHECK_EQ(device_before, &view.device());
            CHECK_EQ(handle_before, view.native_handle());

            if (observer != nullptr) {
                observer->case_complete();
            }
        }
    }
}

// Deferred-queue lifetime: stable view, owner, and native-handle addresses
// between submission and wait, repeatable waits, in-order completion,
// injected asynchronous failures rethrown on repeated waits, and queue
// destruction that neither synchronizes nor cancels.
inline void run_lifetime_conformance(
        iom::Device& candidate,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    REQUIRE_FALSE(supported_types.empty());
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, supported_types.front()};

    // Stable addresses across a deferred window.
    {
        DeferredCopyQueue queue;
        auto source = candidate.create_tensor(spec);
        auto interior = candidate.create_tensor(spec);
        auto destination = candidate.create_tensor(spec);
        if (observer != nullptr) {
            observer->setup_complete();
        }

        iom::TensorView full_source = source->view();
        iom::TensorView sliced_source = interior->view().slice(0, 1, 1);
        iom::TensorView sliced_destination =
                destination->view().slice(0, 0, 1);
        iom::TensorView permuted_source =
                source->view().permute(span_of({1, 0}));
        iom::TensorView permuted_destination =
                destination->view().permute(span_of({1, 0}));

        const iom::oid one =
                queue.copy(*source, full_source, *interior, interior->view());
        const iom::oid two = queue.copy(
                *interior, sliced_source, *destination, sliced_destination);
        const iom::oid three = queue.copy(
                *destination, permuted_destination, *source, permuted_source);
        CHECK_EQ(token_sequence(one), 1);
        CHECK_EQ(token_sequence(two), 2);
        CHECK_EQ(token_sequence(three), 3);

        queue.complete(1);
        queue.complete(
                2, std::make_exception_ptr(
                           std::runtime_error("injected asynchronous failure")));
        queue.complete(3);

        CHECK_NOTHROW(queue.wait(one));
        bool rethrown = false;
        try {
            queue.wait(two);
        } catch (const std::runtime_error& error) {
            rethrown = std::string_view(error.what())
                       == "injected asynchronous failure";
        }
        CHECK(rethrown);
        CHECK_NOTHROW(queue.wait(three));

        // Repeated waits rethrow the stored failure; successful waits are
        // idempotent.
        CHECK_THROWS_AS(queue.wait(two), std::runtime_error);
        CHECK_THROWS_AS(queue.wait(two), std::runtime_error);
        CHECK_NOTHROW(queue.wait(one));
        CHECK_NOTHROW(queue.wait(three));

        // Submission-time view, owner, and native-handle addresses all
        // survived until wait.
        queue.expect_stable(
                1, *source, full_source, *interior, interior->view());
        queue.expect_stable(
                2, *interior, sliced_source, *destination, sliced_destination);
        queue.expect_stable(
                3, *destination, permuted_destination, *source,
                permuted_source);

        if (observer != nullptr) {
            observer->case_complete();
        }
    }

    // In-order completion: completing a later sequence implies every earlier
    // sequence.
    {
        DeferredCopyQueue queue;
        const iom::oid first = queue.probe();
        const iom::oid second = queue.probe();
        queue.complete(2);
        CHECK_NOTHROW(queue.wait(first));
        CHECK_NOTHROW(queue.wait(second));
    }

    // Tokens of other queues and zero are rejected.
    {
        DeferredCopyQueue queue;
        DeferredCopyQueue other;
        const iom::oid own = queue.probe();
        const iom::oid foreign_token = other.probe();
        CHECK_THROWS_AS(queue.wait(0), std::invalid_argument);
        CHECK_THROWS_AS(queue.wait(foreign_token), std::invalid_argument);
        CHECK_THROWS_AS(other.wait(own), std::invalid_argument);
        queue.complete(1);
        CHECK_NOTHROW(queue.wait(own));
    }

    // Queue destruction neither synchronizes on an outstanding sequence nor
    // cancels it, and the queue id returns to the pool.
    {
        std::vector<InstrumentedQueue::Event> drained_journal;
        {
            auto journal =
                    std::make_shared<std::vector<InstrumentedQueue::Event>>();
            std::uint8_t released_id = 0;
            const auto began = std::chrono::steady_clock::now();
            {
                InstrumentedQueue pending(journal);
                const iom::oid outstanding = pending.probe();
                released_id = token_queue(outstanding);
                CHECK_EQ(token_sequence(outstanding), 1);
            }
            const auto ended = std::chrono::steady_clock::now();
            drained_journal = std::move(*journal);

            // An implicit wait would never return: nothing completes the
            // outstanding sequence.
            const auto elapsed =
                    std::chrono::duration_cast<std::chrono::seconds>(
                            ended - began);
            CHECK_LT(elapsed.count(), 1);

            bool saw_outstanding_completion = false;
            bool saw_destroy_begin = false;
            bool saw_destroy_end = false;
            for (const InstrumentedQueue::Event& event : drained_journal) {
                if (event.kind
                            == InstrumentedQueue::Event::Kind::complete
                    && event.sequence == 1) {
                    saw_outstanding_completion = true;
                }
                saw_destroy_begin = saw_destroy_begin
                        || event.kind
                                == InstrumentedQueue::Event::Kind::destroy_begin;
                saw_destroy_end = saw_destroy_end
                        || event.kind
                                == InstrumentedQueue::Event::Kind::destroy_end;
            }
            CHECK_FALSE(saw_outstanding_completion);
            CHECK(saw_destroy_begin);
            CHECK(saw_destroy_end);

            InstrumentedQueue successor(journal);
            CHECK_EQ(token_queue(successor.probe()), released_id);
            successor.finish(1);
            CHECK_NOTHROW(successor.wait(
                    (static_cast<iom::oid>(released_id)
                     << kTokenSequenceBits)
                            | 1));
            // Every successful submission was waited before destruction.
        }
    }
}

// Unsupported compute capabilities: every compute method rejects with
// std::runtime_error before submitting, consuming a sequence, or changing an
// output, including transformed operands.
inline void run_compute_capability_conformance(
        iom::Device& candidate,
        const std::span<const iom::DataType>,  // capability, not type-specific
        ConformanceObserver* observer = nullptr) {
    const iom::TensorSpec spec{iom::TensorShape{{2, 16, 16}}, iom::DataType::F32};
    auto x = candidate.create_tensor(spec);
    auto y = candidate.create_tensor(spec);
    auto w = candidate.create_tensor(spec);
    auto attn = candidate.create_tensor(spec);
    auto scratch = candidate.create_tensor(spec);
    if (observer != nullptr) {
        observer->setup_complete();
    }

    const std::vector<std::byte> y_pattern = encode_logical(spec, 41);
    const std::vector<std::byte> attn_pattern = encode_logical(spec, 42);
    y->view().copy_from_host(y_pattern);
    attn->view().copy_from_host(attn_pattern);

    auto queue = candidate.create_ops();
    CHECK_THROWS_AS(queue->add(x->view(), x->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(queue->mul(x->view(), x->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(queue->silu(x->view(), y->view()), std::runtime_error);
    CHECK_THROWS_AS(queue->linear(x->view(), w->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(
            queue->rmsnorm(x->view(), y->view(), w->view(), 1e-6F, 1),
            std::runtime_error);
    CHECK_THROWS_AS(
            queue->sdpa(x->view(), x->view(), x->view(), 1, 1, 16,
                        attn->view()),
            std::runtime_error);

    // Transformed operands compile against the view signatures and still
    // fail capability validation.
    iom::TensorView stepped_x = x->view().slice(0, 0, 1);
    iom::TensorView stepped_y = y->view().slice(0, 0, 1);
    const iom::TensorView merged_x =
            x->view().reshape_leading(span_of({2}));
    CHECK_THROWS_AS(queue->silu(stepped_x, stepped_y), std::runtime_error);
    CHECK_THROWS_AS(
            queue->add(merged_x, x->view(), y->view()), std::runtime_error);

    // No output changed and no sequence was consumed.
    require_logical_bytes(y->view(), y_pattern, "y after capability failures");
    require_logical_bytes(
            attn->view(), attn_pattern, "attn after capability failures");
    const iom::oid probe = queue->copy(x->view(), scratch->view());
    CHECK_EQ(token_sequence(probe), 1);
    queue->wait(probe);
    queue.reset();

    if (observer != nullptr) {
        observer->case_complete();
    }
}

// Full suite: storage, transfers, copies, errors, lifetime, and capabilities
// in dependency order.
inline void run_backend_conformance(
        const ConformanceDevices& devices,
        const std::span<const iom::DataType> supported_types,
        ConformanceObserver* observer = nullptr) {
    run_storage_and_transfer_conformance(devices, supported_types, observer);
    run_async_copy_conformance(devices, supported_types, observer);
    run_copy_error_conformance(devices, supported_types, observer);
    run_transfer_error_conformance(devices, supported_types, observer);
    run_lifetime_conformance(devices.candidate, supported_types, observer);
    run_compute_capability_conformance(
            devices.candidate, supported_types, observer);
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
