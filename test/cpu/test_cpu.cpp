#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "backend/backend_conformance_oracle.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace {

// ---------------------------------------------------------------------------
// Recording allocator: logs every call and can be driven into the failure
// modes the ownership contract distinguishes.
// ---------------------------------------------------------------------------

class RecordingAllocator final : public iom::Allocator {
public:
    struct Event {
        enum class Kind { alloc, free, reset };

        Kind kind;
        void* address;
        std::size_t bytes;
    };

    std::vector<Event> events;

    // Modes applying to the next alloc call only.
    bool null_next = false;
    bool misalign_next = false;
    bool throw_next = false;

    void* alloc(std::size_t size) override {
        if (throw_next) {
            throw_next = false;
            events.push_back({Event::Kind::alloc, nullptr, size});
            throw std::runtime_error("injected allocator failure");
        }
        if (null_next) {
            null_next = false;
            events.push_back({Event::Kind::alloc, nullptr, size});
            return nullptr;
        }
        // Headroom so the handed-out address can sit at a controlled
        // offset inside one aligned block.
        void* block = ::operator new(size + 32, std::align_val_t(32));
        auto* address = static_cast<std::byte*>(block)
                        + (misalign_next ? std::size_t{8} : std::size_t{32});
        misalign_next = false;
        live_[address] = block;
        events.push_back({Event::Kind::alloc, address, size});
        return address;
    }

    void free(void* buffer) override {
        const auto block = live_.find(buffer);
        REQUIRE_MESSAGE(block != live_.end(),
                        "allocator freed an address it never handed out");
        events.push_back({Event::Kind::free, buffer, 0});
        ::operator delete(block->second, std::align_val_t(32));
        live_.erase(block);
    }

    void reset() override {
        events.push_back({Event::Kind::reset, nullptr, 0});
    }

    [[nodiscard]] bool live_empty() const {
        return live_.empty();
    }

private:
    std::unordered_map<void*, void*> live_;
};

class RecyclingAllocator final : public iom::Allocator {
public:
    using Event = RecordingAllocator::Event;

    struct Slot {
        void* address;
        std::size_t bytes;
    };

    ~RecyclingAllocator() override {
        release_free();
        for (const auto& [address, bytes] : live_) {
            (void)bytes;
            ::operator delete(address, std::align_val_t(32));
        }
    }

    std::vector<Event> events;

    void* alloc(std::size_t size) override {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->bytes != size) {
                continue;
            }
            void* address = it->address;
            live_.emplace(address, size);
            free_.erase(it);
            events.push_back({Event::Kind::alloc, address, size});
            return address;
        }

        void* address = ::operator new(size, std::align_val_t(32));
        try {
            live_.emplace(address, size);
        } catch (...) {
            ::operator delete(address, std::align_val_t(32));
            throw;
        }
        events.push_back({Event::Kind::alloc, address, size});
        return address;
    }

    void free(void* buffer) override {
        const auto found = live_.find(buffer);
        REQUIRE_MESSAGE(found != live_.end(),
                        "allocator freed an address it never handed out");
        const std::size_t bytes = found->second;
        free_.push_back({buffer, bytes});
        events.push_back({Event::Kind::free, buffer, bytes});
        live_.erase(found);
    }

    void reset() override {
        events.push_back({Event::Kind::reset, nullptr, 0});
    }

    void release_free() noexcept {
        for (const Slot& slot : free_) {
            ::operator delete(slot.address, std::align_val_t(32));
        }
        free_.clear();
    }

    [[nodiscard]] std::size_t free_count() const noexcept {
        return free_.size();
    }

    [[nodiscard]] bool live_empty() const noexcept {
        return live_.empty();
    }

private:
    std::vector<Slot> free_;
    std::unordered_map<void*, std::size_t> live_;
};

// ---------------------------------------------------------------------------
// Independent test-side encoding model. The bit writer follows the section-3
// host encoding directly (least-significant bit first at bit offset
// element_index * width); it shares no code with the backend.
// ---------------------------------------------------------------------------

std::size_t test_bits(iom::DataType type) {
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
    REQUIRE_MESSAGE(false, "test_bits: unknown DataType");
    return 0;
}

void write_test_bits(
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

std::uint64_t read_test_bits(
        const unsigned char* base, std::size_t bit_offset,
        std::size_t nbits) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < nbits; ++i) {
        const std::size_t bit = bit_offset + i;
        value |= static_cast<std::uint64_t>((base[bit / 8] >> (bit % 8)) & 1)
                 << i;
    }
    return value;
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Deterministic pseudo-random element content as a function of the logical
// element index and a salt. BOOL is restricted to canonical zero and one.
std::uint64_t element_pattern(
        iom::DataType type, std::size_t linear, std::uint64_t salt) {
    const std::uint64_t raw = splitmix64(
            splitmix64(
                    static_cast<std::uint64_t>(linear) * 0x9E3779B97F4A7C15ull)
            ^ salt);
    if (type == iom::DataType::BOOL) {
        return raw & 1;
    }
    const std::size_t bits = test_bits(type);
    if (bits >= 64) {
        return raw;
    }
    return raw & ((std::uint64_t{1} << bits) - 1);
}

std::vector<std::byte> encoded_host(
        const iom::TensorView& view, std::uint64_t salt) {
    const iom::TensorSpec spec = view.spec();
    const std::size_t bits = test_bits(spec.data_type);
    const std::size_t count = spec.shape.element_count();
    std::vector<std::byte> buffer(spec.logical_nbytes(), std::byte{0});
    auto* base = reinterpret_cast<unsigned char*>(buffer.data());
    for (std::size_t linear = 0; linear < count; ++linear) {
        write_test_bits(
                base, linear * bits, bits,
                element_pattern(spec.data_type, linear, salt));
    }
    return buffer;
}

// ---------------------------------------------------------------------------
// Test-side helpers over tensors and the checked core layout helper.
// ---------------------------------------------------------------------------

iom::TensorSpec make_spec(
        std::vector<std::size_t> dimensions,
        iom::DataType data_type,
        iom::QuantizationFormat quantization = iom::QuantizationFormat::NONE) {
    return iom::TensorSpec{
            iom::TensorShape{std::move(dimensions)}, data_type, quantization};
}

bool is_aligned(const void* address, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(address) % alignment == 0;
}

void fill_storage(iom::Tensor& tensor, std::byte value) {
    const iom::TensorSpec spec = tensor.view().spec();
    auto* base = static_cast<std::byte*>(tensor.view().native_handle());
    std::fill(base, base + spec.tiled_storage_nbytes(), value);
}

std::vector<std::byte> snapshot_storage(const iom::Tensor& tensor) {
    const iom::TensorSpec spec = tensor.view().spec();
    const auto* base = static_cast<const std::byte*>(
            tensor.view().native_handle());
    return {base, base + spec.tiled_storage_nbytes()};
}

std::size_t first_mismatch(
        const void* expected, const void* actual, std::size_t bytes) {
    const auto* expected_bytes = static_cast<const unsigned char*>(expected);
    const auto* actual_bytes = static_cast<const unsigned char*>(actual);
    for (std::size_t i = 0; i < bytes; ++i) {
        if (expected_bytes[i] != actual_bytes[i]) {
            return i;
        }
    }
    return bytes;
}

void expect_storage_matches(
        const iom::Tensor& tensor, const std::vector<std::byte>& expected) {
    const std::size_t mismatch =
            first_mismatch(expected.data(), tensor.view().native_handle(),
                           expected.size());
    REQUIRE_MESSAGE(mismatch == expected.size(),
                    "storage diverges from the model at byte " << mismatch);
}

constexpr std::uint64_t kTokenSequenceBits = 55;
constexpr std::uint64_t kTokenSequenceMask =
        (std::uint64_t{1} << kTokenSequenceBits) - 1;

std::uint8_t token_queue(iom::oid token) {
    return static_cast<std::uint8_t>(
            static_cast<std::uint64_t>(token)
            >> kTokenSequenceBits);
}

std::uint64_t token_sequence(iom::oid token) {
    return static_cast<std::uint64_t>(token) & kTokenSequenceMask;
}

constexpr std::initializer_list<iom::DataType> kAllDataTypes = {
    iom::DataType::BOOL,
    iom::DataType::I2, iom::DataType::U2,
    iom::DataType::I4, iom::DataType::U4,
    iom::DataType::I8, iom::DataType::U8,
    iom::DataType::I16, iom::DataType::U16,
    iom::DataType::I32, iom::DataType::U32,
    iom::DataType::I64, iom::DataType::U64,
    iom::DataType::F4_E2M1,
    iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
    iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2, iom::DataType::F8_E8M0,
    iom::DataType::F16, iom::DataType::BF16,
    iom::DataType::F32, iom::DataType::F64,
};

// Full-storage oracle for one owner specification. Starts at the sentinel
// the tests write into the real allocation, then tracks logical writes.
// Physical slots come from the shared independent canonical encoder in
// backend_conformance_oracle.hpp (iom_conformance::standard_layout_view_slot)
// so the expected bytes never share tile arithmetic with production mapping.
class StorageModel {
public:
    StorageModel(const iom::TensorSpec& owner_spec, std::byte fill)
            : spec_(owner_spec),
              bytes_(owner_spec.tiled_storage_nbytes(), fill) {}

    [[nodiscard]] const std::vector<std::byte>& bytes() const {
        return bytes_;
    }

    // Every logical element of view takes pattern(type, linear, salt).
    void write_view(const iom::TensorView& view, std::uint64_t salt) {
        const std::size_t bits = test_bits(spec_.data_type);
        const std::size_t count = view.spec().shape.element_count();
        for (std::size_t linear = 0; linear < count; ++linear) {
            const std::size_t slot =
                    iom_conformance::standard_layout_view_slot(
                            view, spec_, linear);
            write_test_bits(
                    reinterpret_cast<unsigned char*>(bytes_.data()),
                    slot * bits, bits,
                    element_pattern(spec_.data_type, linear, salt));
        }
    }

    // Every logical element of destination takes the current storage
    // content of the source view's owner.
    void copy_view(
            const iom::TensorView& source_view,
            const iom::TensorSpec& source_spec, const void* source_storage,
            const iom::TensorView& destination_view) {
        const std::size_t bits = test_bits(spec_.data_type);
        const auto* storage =
                static_cast<const unsigned char*>(source_storage);
        const std::size_t count =
                destination_view.spec().shape.element_count();
        for (std::size_t linear = 0; linear < count; ++linear) {
            const std::size_t source_slot =
                    iom_conformance::standard_layout_view_slot(
                            source_view, source_spec, linear);
            const std::size_t destination_slot =
                    iom_conformance::standard_layout_view_slot(
                            destination_view, spec_, linear);
            write_test_bits(
                    reinterpret_cast<unsigned char*>(bytes_.data()),
                    destination_slot * bits, bits,
                    read_test_bits(storage, source_slot * bits, bits));
        }
    }

private:
    iom::TensorSpec spec_;
    std::vector<std::byte> bytes_;
};

constexpr std::byte kSentinel{0x5A};

// ---------------------------------------------------------------------------
// CPU RMSNorm expectations. 1.0, 0.5, and -1.0 are exactly representable in
// all nine applicable leaves, so every expected output below is a fixed raw
// leaf encoding derived from the named format fields (sign, exponent, and
// mantissa of `value = 2^(exponent - bias) * (1 + mantissa / 2^f)`) rather
// than a tolerance-compared approximation.
// ---------------------------------------------------------------------------

struct RmsnormEncodings {
    iom::DataType data_type;
    std::uint64_t one;
    std::uint64_t half;
    std::uint64_t minus_one;
};

constexpr std::array<RmsnormEncodings, 9> kRmsnormEncodings{{
        {iom::DataType::F4_E2M1, 0x2, 0x1, 0xA},
        {iom::DataType::F6_E2M3, 0x08, 0x04, 0x28},
        {iom::DataType::F6_E3M2, 0x0C, 0x08, 0x2C},
        {iom::DataType::F8_E4M3FN, 0x38, 0x30, 0xB8},
        {iom::DataType::F8_E5M2, 0x3C, 0x38, 0xBC},
        {iom::DataType::F16, 0x3C00u, 0x3800u, 0xBC00u},
        {iom::DataType::BF16, 0x3F80u, 0x3F00u, 0xBF80u},
        {iom::DataType::F32, 0x3F800000u, 0x3F000000u, 0xBF800000u},
        {iom::DataType::F64, 0x3FF0000000000000ull, 0x3FE0000000000000ull,
         0xBFF0000000000000ull},
}};

const RmsnormEncodings& rmsnorm_encodings(iom::DataType type) {
    for (const RmsnormEncodings& encodings : kRmsnormEncodings) {
        if (encodings.data_type == type) {
            return encodings;
        }
    }
    REQUIRE_MESSAGE(false, "rmsnorm_encodings: unclassified leaf");
    return kRmsnormEncodings[0];
}

std::vector<std::uint64_t> uniform_codes(
        std::size_t elements, std::uint64_t code) {
    return std::vector<std::uint64_t>(elements, code);
}

// Packed logical host payload holding one explicit raw leaf encoding per
// element in row-major order, in the section-3 host convention the production
// transfer path reads.
std::vector<std::byte> packed_logical(
        iom::DataType type, std::span<const std::uint64_t> values) {
    const std::size_t bits = test_bits(type);
    std::vector<std::byte> buffer(
            (values.size() * bits + 7) / 8, std::byte{0});
    auto* base = reinterpret_cast<unsigned char*>(buffer.data());
    for (std::size_t linear = 0; linear < values.size(); ++linear) {
        write_test_bits(base, linear * bits, bits, values[linear]);
    }
    return buffer;
}

// Braced call sites name one raw leaf encoding per logical element directly.
std::vector<std::byte> packed_logical(
        iom::DataType type, std::initializer_list<std::uint64_t> values) {
    return packed_logical(
            type, std::span<const std::uint64_t>(values.begin(), values.size()));
}

std::vector<std::byte> uniform_logical(
        iom::DataType type, std::size_t elements, std::uint64_t code) {
    return packed_logical(type, uniform_codes(elements, code));
}

std::uint64_t logical_code(
        std::span<const std::byte> buffer, iom::DataType type,
        std::size_t linear) {
    const std::size_t bits = test_bits(type);
    return read_test_bits(
            reinterpret_cast<const unsigned char*>(buffer.data()),
            linear * bits, bits);
}

// Expected full owner storage of an RMSNorm result: the caller's sentinel
// everywhere except the logical elements of `view`, which receive one raw
// encoding per element through the independent canonical tile-slot encoder.
// Padding therefore must stay exactly as the caller left it.
std::vector<std::byte> expected_rmsnorm_storage(
        const iom::TensorView& view, const iom::TensorSpec& owner_spec,
        std::span<const std::uint64_t> codes) {
    std::vector<std::byte> storage(
            owner_spec.tiled_storage_nbytes(), kSentinel);
    iom_conformance::apply_standard_tiled_view(
            view, owner_spec, packed_logical(owner_spec.data_type, codes),
            storage);
    return storage;
}

std::vector<std::byte> expected_rmsnorm_storage(
        const iom::TensorView& view, const iom::TensorSpec& owner_spec,
        std::initializer_list<std::uint64_t> codes) {
    return expected_rmsnorm_storage(
            view, owner_spec,
            std::span<const std::uint64_t>(codes.begin(), codes.size()));
}

std::vector<std::byte> expected_uniform_rmsnorm_storage(
        const iom::TensorView& view, const iom::TensorSpec& owner_spec,
        std::uint64_t code) {
    return expected_rmsnorm_storage(
            view, owner_spec,
            uniform_codes(view.spec().shape.element_count(), code));
}

}  // namespace

// ---------------------------------------------------------------------------
// Device identity and ownership
// ---------------------------------------------------------------------------

TEST_CASE("CPU device reports CPU backend with ordinal zero") {
    RecordingAllocator allocator;
    auto first = iom::make_cpu_device(allocator);
    auto second = iom::make_cpu_device(allocator);
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    CHECK(first.get() != second.get());

    CHECK(first->backend_kind() == iom::BackendKind::CPU);
    CHECK_EQ(first->backend_device(), 0);
    CHECK(second->backend_kind() == iom::BackendKind::CPU);
    CHECK_EQ(second->backend_device(), 0);

    // Devices and queues touch the allocator only through tensors.
    CHECK(allocator.events.empty());
    auto tensor = first->create_tensor(make_spec({16, 16}, iom::DataType::U8));
    CHECK(tensor->view().backend_kind() == iom::BackendKind::CPU);
    CHECK_EQ(tensor->view().backend_device(), 0);
    CHECK(&tensor->view().device() == first.get());
}

class DeviceLessQueue final : public iom::DeviceOps {
public:
    DeviceLessQueue() : iom::DeviceOps() {}
};

TEST_CASE("DeviceOps exposes exact queue device identity") {
    RecordingAllocator allocator;
    auto device_a = iom::make_cpu_device(allocator);
    auto device_b = iom::make_cpu_device(allocator);
    REQUIRE(device_a != nullptr);
    REQUIRE(device_b != nullptr);
    REQUIRE(device_a.get() != device_b.get());

    auto queue_a = device_a->create_ops();
    auto queue_b = device_a->create_ops();
    auto foreign_queue = device_b->create_ops();
    REQUIRE(queue_a != nullptr);
    REQUIRE(queue_b != nullptr);
    REQUIRE(foreign_queue != nullptr);

    CHECK(&queue_a->device() == device_a.get());
    CHECK(&queue_b->device() == device_a.get());
    CHECK(&queue_a->device() == &queue_b->device());
    CHECK(&foreign_queue->device() == device_b.get());
    CHECK(&queue_a->device() != &foreign_queue->device());

    DeviceLessQueue device_less_queue;
    CHECK_THROWS_AS((void)device_less_queue.device(), std::logic_error);
}

TEST_CASE("CPU tensors allocate exactly once per leaf type and free once") {
    for (const iom::DataType type : kAllDataTypes) {
        for (const std::vector<std::size_t> dimensions :
             {std::vector<std::size_t>{16, 16},
              std::vector<std::size_t>{2, 3, 17, 33}}) {
            CAPTURE(dimensions);
            CAPTURE(static_cast<int>(type));

            RecordingAllocator allocator;
            auto device = iom::make_cpu_device(allocator);
            const iom::TensorSpec spec = make_spec(dimensions, type);
            const std::size_t expected_bytes = spec.tiled_storage_nbytes();

            {
                auto tensor = device->create_tensor(spec);
                REQUIRE_EQ(allocator.events.size(), 1);
                CHECK(allocator.events[0].kind
                      == RecordingAllocator::Event::Kind::alloc);
                CHECK_EQ(allocator.events[0].bytes, expected_bytes);

                // The native handle is the allocation base address.
                CHECK(tensor->view().native_handle()
                      == allocator.events[0].address);
                CHECK(is_aligned(tensor->view().native_handle(), 32));
                CHECK_FALSE(allocator.live_empty());
            }

            REQUIRE_EQ(allocator.events.size(), 2);
            CHECK(allocator.events[1].kind
                  == RecordingAllocator::Event::Kind::free);
            CHECK(allocator.events[1].address
                  == allocator.events[0].address);
            CHECK(allocator.live_empty());
        }
    }
}

TEST_CASE("CPU create_tensor validates before allocating") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);

    const iom::TensorSpec grouped = make_spec(
            {16, 16}, iom::DataType::F32, iom::QuantizationFormat::GGML_Q4_0);
    CHECK_THROWS_AS(device->create_tensor(grouped), std::runtime_error);
    CHECK(allocator.events.empty());

    const auto unknown = static_cast<iom::DataType>(77);
    CHECK_THROWS_AS(
            device->create_tensor(make_spec({16, 16}, unknown)),
            std::invalid_argument);
    CHECK(allocator.events.empty());
}

TEST_CASE("CPU tensors reject null and misaligned allocations exactly once") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec =
            make_spec({2, 3, 17, 33}, iom::DataType::F32);

    allocator.null_next = true;
    CHECK_THROWS_AS(device->create_tensor(spec), std::bad_alloc);
    REQUIRE_EQ(allocator.events.size(), 1);
    CHECK(allocator.events[0].kind
          == RecordingAllocator::Event::Kind::alloc);
    CHECK(allocator.events[0].address == nullptr);

    allocator.misalign_next = true;
    CHECK_THROWS_AS(device->create_tensor(spec), std::runtime_error);
    REQUIRE_EQ(allocator.events.size(), 3);
    CHECK(allocator.events[1].kind
          == RecordingAllocator::Event::Kind::alloc);
    CHECK_FALSE(is_aligned(allocator.events[1].address, 32));
    CHECK(allocator.events[2].kind
          == RecordingAllocator::Event::Kind::free);
    CHECK(allocator.events[2].address == allocator.events[1].address);
    CHECK(allocator.live_empty());
}

TEST_CASE("CPU create_tensor propagates LinearAllocator exhaustion") {
    alignas(32) std::array<std::byte, 256> buffer{};
    iom::LinearAllocator allocator(buffer.data(), buffer.size());
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = make_spec({16, 16}, iom::DataType::U8);

    auto first = device->create_tensor(spec);
    REQUIRE(first != nullptr);
    fill_storage(*first, kSentinel);
    const auto expected_storage = snapshot_storage(*first);
    void* first_storage = first->view().native_handle();

    std::unique_ptr<iom::Tensor> second;
    CHECK_THROWS_AS(second = device->create_tensor(spec), std::bad_alloc);
    CHECK(second == nullptr);
    CHECK_EQ(first->view().native_handle(), first_storage);
    expect_storage_matches(*first, expected_storage);
}

TEST_CASE("CPU create_tensor propagates allocator failures without freeing") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);

    allocator.throw_next = true;
    CHECK_THROWS_AS(
            device->create_tensor(make_spec({16, 16}, iom::DataType::U8)),
            std::runtime_error);
    REQUIRE_EQ(allocator.events.size(), 1);
    CHECK(allocator.events[0].kind
          == RecordingAllocator::Event::Kind::alloc);
}

// ---------------------------------------------------------------------------
// Standard layout and host transfers
// ---------------------------------------------------------------------------
// A span over a temporary initializer-list array; valid for the full
// expression that consumes it.
std::span<const std::size_t> span_of(
        std::initializer_list<std::size_t> values) {
    return {values.begin(), values.size()};
}

TEST_CASE("CPU host writes land in exact tiled slots and reads hide padding") {
    struct LayoutCase {
        std::vector<std::size_t> dimensions;
        iom::DataType type;
    };
    const std::vector<LayoutCase> cases = {
        {{16, 16}, iom::DataType::U8},
        {{1, 17}, iom::DataType::I16},
        {{17, 1}, iom::DataType::U16},
        {{2, 3}, iom::DataType::I32},
        {{2, 3}, iom::DataType::F32},
        {{3, 5, 13, 11}, iom::DataType::I64},
        {{2, 8, 31, 33}, iom::DataType::BF16},
        {{2, 3, 2, 5, 17, 33}, iom::DataType::F64},
        {{1, 5}, iom::DataType::I2},
        {{2, 3, 9, 7}, iom::DataType::U2},
        {{3, 5, 13, 11}, iom::DataType::I4},
        {{2, 3, 9, 7}, iom::DataType::U4},
        {{2, 3, 9, 7}, iom::DataType::F4_E2M1},
        {{1, 7}, iom::DataType::F6_E2M3},
        {{2, 3, 9, 7}, iom::DataType::F6_E3M2},
        {{2, 3, 9, 7}, iom::DataType::BOOL},
        {{16, 16}, iom::DataType::F8_E4M3FN},
        {{2, 8, 31, 33}, iom::DataType::F8_E5M2},
        {{1, 17}, iom::DataType::F8_E8M0},
        {{2, 3}, iom::DataType::F16},
        {{17, 1}, iom::DataType::BF16},
        {{2, 3, 2, 5, 17, 33}, iom::DataType::U64},
        {{2, 3, 9, 7}, iom::DataType::I16},
        {{2, 3, 2, 5, 17, 33}, iom::DataType::I8},
        {{2, 3}, iom::DataType::U32},
    };

    for (const LayoutCase& layout_case : cases) {
        CAPTURE(layout_case.dimensions);
        CAPTURE(static_cast<int>(layout_case.type));

        RecordingAllocator allocator;
        auto device = iom::make_cpu_device(allocator);
        const iom::TensorSpec spec =
                make_spec(layout_case.dimensions, layout_case.type);
        auto tensor = device->create_tensor(spec);
        REQUIRE_EQ(allocator.events.size(), 1);

        fill_storage(*tensor, kSentinel);
        StorageModel model(spec, kSentinel);
        iom::TensorView& view = tensor->view();

        const std::vector<std::byte> host = encoded_host(view, 0xABCDEF);
        view.copy_from_host(host);
        model.write_view(view, 0xABCDEF);
        expect_storage_matches(*tensor, model.bytes());

        // The inverse read hides padding and zeroes unused tail bits.
        std::vector<std::byte> readback(spec.logical_nbytes(), std::byte{0xAA});
        view.copy_to_host(readback);
        CHECK(readback == host);

        // Transfers never call the tensor-storage allocator.
        CHECK_EQ(allocator.events.size(), 1);
    }
}

TEST_CASE("CPU tile-aligned transfers use canonical storage slots") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);

    {
        const iom::TensorSpec spec =
                make_spec({1, 32}, iom::DataType::F32);
        auto tensor = device->create_tensor(spec);
        fill_storage(*tensor, kSentinel);

        std::array<float, 32> values{};
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = 1000.0f + static_cast<float>(index);
        }
        std::vector<std::byte> host(sizeof(values));
        std::memcpy(host.data(), values.data(), host.size());
        tensor->view().copy_from_host(host);

        const auto* storage = static_cast<const std::byte*>(
                tensor->view().native_handle());
        float canonical = 0.0f;
        float broken = 0.0f;
        std::memcpy(&canonical, storage + 1024, sizeof(canonical));
        std::memcpy(&broken, storage + 64, sizeof(broken));
        CHECK_EQ(canonical, 1016.0f);
        CHECK_NE(broken, 1016.0f);
        for (std::size_t index = 0; index < sizeof(broken); ++index) {
            CHECK_EQ(storage[64 + index], kSentinel);
        }

        auto destination = device->create_tensor(spec);
        fill_storage(*destination, kSentinel);
        auto queue = device->create_ops();
        const iom::oid token =
                queue->copy(tensor->view(), destination->view());
        queue->wait(token);
        const auto* destination_storage = static_cast<const std::byte*>(
                destination->view().native_handle());
        CHECK(std::equal(
                storage, storage + spec.tiled_storage_nbytes(),
                destination_storage));
    }

    {
        const iom::TensorSpec spec =
                make_spec({1, 32}, iom::DataType::I4);
        auto tensor = device->create_tensor(spec);
        fill_storage(*tensor, kSentinel);
        iom::TensorView& view = tensor->view();
        const std::vector<std::byte> host = encoded_host(view, 43);
        view.copy_from_host(host);

        const auto* storage = reinterpret_cast<const unsigned char*>(
                tensor->view().native_handle());
        const std::size_t bits = iom::detail::leaf_bits(spec.data_type);
        const std::size_t canonical_bit =
                iom::detail::standard_plane_slot(spec, 0, 0, 16) * bits;
        CHECK_EQ(
                read_test_bits(storage, canonical_bit, bits),
                element_pattern(spec.data_type, 16, 43));
        CHECK_EQ(read_test_bits(storage, 16 * bits, bits), std::uint64_t{0xA});
    }

    {
        const iom::TensorSpec spec =
                make_spec({2, 3, 16, 48}, iom::DataType::U8);
        auto tensor = device->create_tensor(spec);
        fill_storage(*tensor, kSentinel);
        iom::TensorView& view = tensor->view();
        const std::vector<std::byte> host = encoded_host(view, 47);
        view.copy_from_host(host);

        const auto* storage = static_cast<const std::byte*>(
                tensor->view().native_handle());
        for (std::size_t plane = 0; plane < 6; ++plane) {
            for (std::size_t row = 0; row < 16; ++row) {
                for (std::size_t column = 0; column < 48; ++column) {
                    const std::size_t linear =
                            (plane * 16 + row) * 48 + column;
                    const std::size_t slot =
                            iom::detail::standard_plane_slot(
                                    spec, plane, row, column);
                    CHECK_EQ(storage[slot], host[linear]);
                }
            }
        }
    }
}


TEST_CASE("CPU storage places fields at exact little-endian bit offsets") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);

    // Multi-byte fields are little-endian whole bytes at their slots.
    {
        auto tensor =
                device->create_tensor(make_spec({1, 2}, iom::DataType::I16));
        fill_storage(*tensor, std::byte{0x00});
        const std::vector<std::byte> host = {
            std::byte{0x34}, std::byte{0x12},
            std::byte{0x78}, std::byte{0x56}};
        tensor->view().copy_from_host(host);
        const auto* storage =
                static_cast<const std::byte*>(tensor->view().native_handle());
        CHECK_EQ(storage[0], std::byte{0x34});
        CHECK_EQ(storage[1], std::byte{0x12});
        CHECK_EQ(storage[2], std::byte{0x78});
        CHECK_EQ(storage[3], std::byte{0x56});

        // Inverse: patched storage bytes read back as the same fields,
        // and the padding slots full of 0xFF never surface.
        fill_storage(*tensor, std::byte{0xFF});
        auto* writable =
                static_cast<std::byte*>(tensor->view().native_handle());
        writable[0] = std::byte{0x34};
        writable[1] = std::byte{0x12};
        writable[2] = std::byte{0x78};
        writable[3] = std::byte{0x56};
        std::vector<std::byte> readback(4, std::byte{0xAA});
        tensor->view().copy_to_host(readback);
        CHECK(readback == host);
    }

    // Sub-byte fields pack least-significant bits first at i * width.
    {
        auto tensor =
                device->create_tensor(make_spec({1, 3}, iom::DataType::U4));
        fill_storage(*tensor, std::byte{0x00});
        const std::vector<std::byte> host = {std::byte{0xA5}, std::byte{0x03}};
        tensor->view().copy_from_host(host);
        const auto* storage =
                static_cast<const std::byte*>(tensor->view().native_handle());
        CHECK_EQ(storage[0], std::byte{0xA5});
        CHECK_EQ(storage[1], std::byte{0x03});
    }
    {
        auto tensor =
                device->create_tensor(make_spec({1, 4}, iom::DataType::I2));
        fill_storage(*tensor, std::byte{0x00});
        // Elements 1, 2, 3, 0 pack into one byte 0b00111001.
        const std::vector<std::byte> host = {std::byte{0x39}};
        tensor->view().copy_from_host(host);
        const auto* storage =
                static_cast<const std::byte*>(tensor->view().native_handle());
        CHECK_EQ(storage[0], std::byte{0x39});
    }
    {
        // Six-bit fields cross byte boundaries: elements 0x2A, 0x15,
        // 0x3F, 0x00 encode to the host bytes below, where the second
        // element's low bits land in byte zero's top bits and the third
        // element fills byte one's high nibble.
        auto tensor = device->create_tensor(
                make_spec({1, 4}, iom::DataType::F6_E2M3));
        fill_storage(*tensor, std::byte{0x00});
        const std::vector<std::byte> host = {
            std::byte{0x6A}, std::byte{0xAF}, std::byte{0x03}};
        tensor->view().copy_from_host(host);
        const auto* storage =
                static_cast<const std::byte*>(tensor->view().native_handle());
        CHECK_EQ(storage[0], std::byte{0x6A});
        CHECK_EQ(storage[1], std::byte{0xAF});
        CHECK_EQ(storage[2], std::byte{0x03});
    }

    // Inverse read: storage bits outside the logical elements never reach
    // the host buffer and output tail bits read as zero.
    {
        auto tensor =
                device->create_tensor(make_spec({1, 3}, iom::DataType::U4));
        fill_storage(*tensor, std::byte{0xFF});
        auto* writable =
                static_cast<std::byte*>(tensor->view().native_handle());
        writable[0] = std::byte{0x2A};
        writable[1] = std::byte{0x50};
        std::vector<std::byte> readback(2, std::byte{0xAA});
        tensor->view().copy_to_host(readback);
        CHECK_EQ(readback[0], std::byte{0x2A});
        // Element two is the low nibble only; the tail bits of the host
        // byte are zeroed although storage holds 0x50 there.
        CHECK_EQ(readback[1], std::byte{0x00});
    }
}

TEST_CASE("CPU BOOL validates canonical host bytes before writes") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = make_spec({2, 3, 9, 7}, iom::DataType::BOOL);
    auto tensor = device->create_tensor(spec);

    fill_storage(*tensor, kSentinel);
    StorageModel model(spec, kSentinel);
    iom::TensorView& view = tensor->view();

    std::vector<std::byte> host = encoded_host(view, 7);
    REQUIRE_FALSE(host.empty());
    view.copy_from_host(host);
    model.write_view(view, 7);
    expect_storage_matches(*tensor, model.bytes());

    std::vector<std::byte> readback(spec.logical_nbytes(), std::byte{0xAA});
    view.copy_to_host(readback);
    CHECK(readback == host);

    // A non-canonical byte anywhere is rejected before any write.
    std::vector<std::byte> invalid = host;
    invalid[invalid.size() / 2] = std::byte{2};
    CHECK_THROWS_AS(view.copy_from_host(invalid), std::invalid_argument);
    expect_storage_matches(*tensor, model.bytes());

    // Wrong-size writes and reads fail before touching either side.
    std::vector<std::byte> short_host(host.begin(), host.end() - 1);
    CHECK_THROWS_AS(view.copy_from_host(short_host), std::invalid_argument);
    std::vector<std::byte> long_host(host);
    long_host.push_back(std::byte{0});
    CHECK_THROWS_AS(view.copy_from_host(long_host), std::invalid_argument);
    expect_storage_matches(*tensor, model.bytes());

    std::vector<std::byte> short_readback(
            spec.logical_nbytes() - 1, std::byte{0xAA});
    CHECK_THROWS_AS(view.copy_to_host(short_readback), std::invalid_argument);
    CHECK(std::all_of(
            short_readback.begin(), short_readback.end(),
            [](std::byte value) { return value == std::byte{0xAA}; }));
    expect_storage_matches(*tensor, model.bytes());

    CHECK_EQ(allocator.events.size(), 1);
}

TEST_CASE("CPU transformed views transfer exactly their logical planes") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec =
            make_spec({3, 8, 20, 18}, iom::DataType::I16);
    auto tensor = device->create_tensor(spec);

    fill_storage(*tensor, kSentinel);
    StorageModel model(spec, kSentinel);

    std::uint64_t salt = 100;
    auto write_read_and_check = [&](auto&& view) {
        const std::vector<std::byte> host = encoded_host(view, salt);
        view.copy_from_host(host);
        model.write_view(view, salt);

        std::vector<std::byte> readback(
                view.spec().logical_nbytes(), std::byte{0xAA});
        view.copy_to_host(readback);
        CHECK(readback == host);
        ++salt;
    };

    iom::TensorView& full = tensor->view();
    write_read_and_check(full);
    write_read_and_check(full.permute(span_of({1, 0})));
    write_read_and_check(full.reshape_leading(span_of({24})));
    write_read_and_check(full.reshape_leading(span_of({2, 12})));
    write_read_and_check(full.select(0, 1).slice(0, 0, 3, 2));
    write_read_and_check(
            full.select(0, 1)
                    .reshape_leading(span_of({2, 4}))
                    .slice(1, 0, 2, 2)
                    .permute(span_of({1, 0})));

    // Every write landed exactly in its addressed planes.
    expect_storage_matches(*tensor, model.bytes());
    CHECK_EQ(allocator.events.size(), 1);
}

TEST_CASE("CPU rank-two views transfer through empty transforms") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = make_spec({20, 18}, iom::DataType::I16);
    auto tensor = device->create_tensor(spec);

    fill_storage(*tensor, kSentinel);
    StorageModel model(spec, kSentinel);

    const std::vector<std::size_t> empty;
    iom::TensorView permuted =
            tensor->view().permute(std::span<const std::size_t>{empty});
    iom::TensorView reshaped =
            tensor->view().reshape_leading(std::span<const std::size_t>{empty});

    for (iom::TensorView* view : {&tensor->view(), &permuted, &reshaped}) {
        const std::vector<std::byte> host = encoded_host(*view, 55);
        view->copy_from_host(host);
        model.write_view(*view, 55);
        expect_storage_matches(*tensor, model.bytes());

        std::vector<std::byte> readback(
                view->spec().logical_nbytes(), std::byte{0xAA});
        view->copy_to_host(readback);
        CHECK(readback == host);
    }
    CHECK_EQ(allocator.events.size(), 1);
}

// ---------------------------------------------------------------------------
// Asynchronous copies
// ---------------------------------------------------------------------------


// CPU queue destruction drains accepted asynchronous work before tensors are
// destroyed, so their registrations are released and each storage block is
// freed exactly once. Unwaited tokens retain terminal history.
TEST_CASE(
        "CPU queue destruction with unwaited tokens keeps frees exactly once") {
    RecyclingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec =
            make_spec({2, 3, 16, 16}, iom::DataType::U8);
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    const std::vector<std::byte> source_host =
            encoded_host(source->view(), 202);
    source->view().copy_from_host(source_host);
    fill_storage(*destination, kSentinel);

    auto queue = device->create_ops();
    const iom::oid first = queue->copy(source->view(), destination->view());
    const iom::oid second = queue->copy(destination->view(), source->view());
    CHECK_EQ(token_sequence(first), 1);
    CHECK_EQ(token_sequence(second), 2);
    const void* source_address = source->view().native_handle();
    const void* destination_address =
            destination->view().native_handle();

    queue.reset();
    const std::size_t events_after_queue_reset = allocator.events.size();
    source.reset();
    destination.reset();
    const auto free_count_for = [&](const void* address) {
        return std::count_if(
                allocator.events.begin(), allocator.events.end(),
                [address](const RecyclingAllocator::Event& event) {
                    return event.kind
                                    == RecyclingAllocator::Event::Kind::free
                            && event.address == address;
                });
    };
    CHECK_EQ(free_count_for(source_address), 1);
    CHECK_EQ(free_count_for(destination_address), 1);
    CHECK_EQ(allocator.events.size(), events_after_queue_reset + 2);

    auto next_queue = device->create_ops();
    auto next_source = device->create_tensor(spec);
    auto next_destination = device->create_tensor(spec);
    REQUIRE_EQ(next_source->view().native_handle(), source_address);
    REQUIRE_EQ(next_destination->view().native_handle(), destination_address);
    next_source->view().copy_from_host(
            encoded_host(next_source->view(), 303));
    fill_storage(*next_destination, kSentinel);
    const iom::oid next = next_queue->copy(
            next_source->view(), next_destination->view());
    next_queue->wait(next);
    next_queue->wait(next);
    std::vector<std::byte> readback(spec.logical_nbytes());
    next_destination->view().copy_to_host(readback);
    CHECK(readback == encoded_host(next_source->view(), 303));

    next_source.reset();
    next_destination.reset();
    next_queue.reset();
    const std::size_t events_before_device_reset = allocator.events.size();
    device.reset();
    CHECK_EQ(allocator.events.size(), events_before_device_reset);
    allocator.release_free();
    CHECK_EQ(allocator.free_count(), 0);
    CHECK(allocator.live_empty());
}
TEST_CASE("CPU queue executes submissions in call order") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = make_spec({2, 3, 16, 16}, iom::DataType::U8);

    auto pattern_tensor = [&](std::uint64_t salt) {
        auto tensor = device->create_tensor(spec);
        tensor->view().copy_from_host(encoded_host(tensor->view(), salt));
        return tensor;
    };
    auto first_pattern = pattern_tensor(1);
    auto second_pattern = pattern_tensor(2);
    auto destination = device->create_tensor(spec);
    auto staging = device->create_tensor(spec);

    auto queue = device->create_ops();

    const iom::oid first =
            queue->copy(first_pattern->view(), destination->view());
    const iom::oid second =
            queue->copy(destination->view(), staging->view());
    const iom::oid third =
            queue->copy(second_pattern->view(), destination->view());

    CHECK_EQ(token_sequence(first), 1);
    CHECK_EQ(token_sequence(second), 2);
    CHECK_EQ(token_sequence(third), 3);
    CHECK_EQ(token_queue(first), token_queue(second));
    CHECK_EQ(token_queue(first), token_queue(third));
    CHECK_NE(token_queue(first), 0);

    queue->wait(first);
    queue->wait(second);
    queue->wait(third);
    // Waits are idempotent after success.
    queue->wait(first);
    queue->wait(third);

    std::vector<std::byte> staged(spec.logical_nbytes(), std::byte{0xAA});
    staging->view().copy_to_host(staged);
    CHECK(staged == encoded_host(first_pattern->view(), 1));

    std::vector<std::byte> final_destination(
            spec.logical_nbytes(), std::byte{0xAA});
    destination->view().copy_to_host(final_destination);
    CHECK(final_destination == encoded_host(second_pattern->view(), 2));
}

TEST_CASE("CPU queues from one device are independent") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = make_spec({2, 3, 16, 16}, iom::DataType::U8);
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    source->view().copy_from_host(encoded_host(source->view(), 9));

    auto first = device->create_ops();
    auto second = device->create_ops();

    const iom::oid from_first =
            first->copy(source->view(), destination->view());
    CHECK_EQ(token_sequence(from_first), 1);
    first->wait(from_first);

    // The producer's oid is waited before the cross-queue consumer runs.
    const iom::oid from_second =
            second->copy(destination->view(), source->view());
    CHECK_EQ(token_sequence(from_second), 1);
    CHECK_NE(token_queue(from_first), token_queue(from_second));

    CHECK_THROWS_AS(first->wait(from_second), std::invalid_argument);
    CHECK_THROWS_AS(second->wait(from_first), std::invalid_argument);

    second->wait(from_second);
    first->wait(from_first);
}

TEST_CASE("CPU copy validates before submission and writes") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    const iom::TensorSpec spec = make_spec({2, 3, 16, 16}, iom::DataType::U8);
    auto destination = device->create_tensor(spec);
    auto other_shape = device->create_tensor(
            make_spec({3, 3, 16, 16}, iom::DataType::U8));
    auto other_type = device->create_tensor(
            make_spec({2, 3, 16, 16}, iom::DataType::I16));

    fill_storage(*destination, kSentinel);
    const std::vector<std::byte> untouched = snapshot_storage(*destination);

    RecordingAllocator other_allocator;
    auto other_device = iom::make_cpu_device(other_allocator);
    auto foreign = other_device->create_tensor(spec);

    CHECK_EQ(
            queue->copy(other_shape->view(), destination->view()),
            iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_EQ(
            queue->copy(destination->view(), other_shape->view()),
            iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_EQ(
            queue->copy(other_type->view(), destination->view()),
            iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_EQ(
            queue->copy(destination->view(), other_type->view()),
            iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_EQ(
            queue->copy(foreign->view(), destination->view()),
            iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_EQ(
            queue->copy(destination->view(), foreign->view()),
            iom::to_oid(iom::OidError::InvalidArgument));

    // Nothing was written and no sequence was consumed.
    expect_storage_matches(*destination, untouched);
    const iom::oid ok = queue->copy(destination->view(), destination->view());
    CHECK_EQ(token_sequence(ok), 1);
    queue->wait(ok);
    expect_storage_matches(*destination, untouched);
}

TEST_CASE("CPU identical-window copy is a waitable no-op") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = make_spec({4, 16, 16}, iom::DataType::U32);
    auto tensor = device->create_tensor(spec);
    auto queue = device->create_ops();

    // Deterministic byte pattern so a stray write is detectable.
    {
        auto* base =
                static_cast<std::byte*>(tensor->view().native_handle());
        for (std::size_t i = 0; i < spec.tiled_storage_nbytes(); ++i) {
            base[i] = static_cast<std::byte>(i % 251);
        }
    }
    const std::vector<std::byte> snapshot = snapshot_storage(*tensor);

    const iom::oid same_object =
            queue->copy(tensor->view(), tensor->view());
    CHECK_EQ(token_sequence(same_object), 1);
    queue->wait(same_object);
    expect_storage_matches(*tensor, snapshot);

    iom::TensorView copied_view = tensor->view();
    const iom::oid same_window = queue->copy(copied_view, tensor->view());
    CHECK_EQ(token_sequence(same_window), 2);
    queue->wait(same_window);
    expect_storage_matches(*tensor, snapshot);

    // Equal native handles with different windows are a real copy.
    iom::TensorView first_planes = tensor->view().slice(0, 0, 2);
    iom::TensorView last_planes = tensor->view().slice(0, 2, 2);
    const std::vector<std::byte> pattern = encoded_host(first_planes, 21);
    first_planes.copy_from_host(pattern);
    const iom::oid moved = queue->copy(first_planes, last_planes);
    CHECK_EQ(token_sequence(moved), 3);
    queue->wait(moved);

    std::vector<std::byte> readback(
            last_planes.spec().logical_nbytes(), std::byte{0xAA});
    last_planes.copy_to_host(readback);
    CHECK(readback == pattern);

    // Overlapping but non-identical windows complete with unspecified
    // destination values. The views outlive the wait: the queue may hold
    // them until then.
    iom::TensorView overlap_source = tensor->view().slice(0, 0, 3);
    iom::TensorView overlap_destination = tensor->view().slice(0, 1, 3);
    const iom::oid overlapping =
            queue->copy(overlap_source, overlap_destination);
    queue->wait(overlapping);
    CHECK_EQ(token_sequence(overlapping), 4);
}

TEST_CASE("CPU copies move logical planes without touching padding") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    const iom::TensorSpec spec = make_spec({3, 20, 18}, iom::DataType::I16);
    auto source = device->create_tensor(spec);
    auto destination = device->create_tensor(spec);
    auto queue = device->create_ops();

    const std::vector<std::byte> source_host =
            encoded_host(source->view(), 31);
    source->view().copy_from_host(source_host);
    fill_storage(*destination, kSentinel);

    const iom::oid full = queue->copy(source->view(), destination->view());
    queue->wait(full);

    StorageModel destination_model(spec, kSentinel);
    destination_model.copy_view(
            source->view(), spec, source->view().native_handle(),
            destination->view());
    expect_storage_matches(*destination, destination_model.bytes());

    // A stepped source view copies exactly its selected planes.
    auto stepped_destination =
            device->create_tensor(make_spec({2, 20, 18}, iom::DataType::I16));
    fill_storage(*stepped_destination, kSentinel);
    const iom::TensorView stepped_source = source->view().slice(0, 0, 2, 2);
    const iom::oid stepped =
            queue->copy(stepped_source, stepped_destination->view());
    queue->wait(stepped);

    StorageModel stepped_model(
            stepped_destination->view().spec(), kSentinel);
    stepped_model.copy_view(
            stepped_source, spec, source->view().native_handle(),
            stepped_destination->view());
    expect_storage_matches(*stepped_destination, stepped_model.bytes());

    // Two allocations and a third: no allocator traffic during copies.
    CHECK_EQ(allocator.events.size(), 3);
}

// ---------------------------------------------------------------------------
// CPU supports all four binary operations and the layout-aware linear
// projection while retaining Unsupported for unrelated compute hooks.
// ---------------------------------------------------------------------------
TEST_CASE("CPU supports binary and linear operations and rejects other compute capabilities") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    const iom::TensorSpec spec = make_spec({16, 16}, iom::DataType::F32);
    auto x = device->create_tensor(spec);
    auto y = device->create_tensor(spec);
    auto w = device->create_tensor(spec);
    auto scale = device->create_tensor(make_spec({1, 16}, iom::DataType::F32));
    auto rmsnorm_out = device->create_tensor(spec);
    auto sdpa_q = device->create_tensor(
            make_spec({2, 1, 16, 16}, iom::DataType::BF16));
    auto sdpa_kv = device->create_tensor(
            make_spec({2, 1, 16, 16}, iom::DataType::BF16));
    auto sdpa_out = device->create_tensor(
            make_spec({2, 16, 16}, iom::DataType::BF16));

    fill_storage(*y, kSentinel);
    fill_storage(*sdpa_out, kSentinel);
    fill_storage(*rmsnorm_out, kSentinel);
    const std::vector<std::byte> sdpa_untouched =
            snapshot_storage(*sdpa_out);

    const iom::oid add_token = queue->add(x->view(), x->view(), y->view());
    const iom::oid mul_token = queue->mul(x->view(), x->view(), y->view());
    const iom::oid sub_token = queue->sub(x->view(), x->view(), y->view());
    const iom::oid div_token = queue->div(x->view(), x->view(), y->view());
    REQUIRE(iom::oid_is_token(add_token));
    REQUIRE(iom::oid_is_token(mul_token));
    REQUIRE(iom::oid_is_token(sub_token));
    REQUIRE(iom::oid_is_token(div_token));
    queue->wait(add_token);
    queue->wait(mul_token);
    queue->wait(sub_token);
    queue->wait(div_token);

    CHECK_EQ(
            queue->silu(x->view(), y->view()),
            iom::to_oid(iom::OidError::Unsupported));

    // Linear is implemented by the CPU port: the accepted request executes on
    // the existing FIFO worker without any allocator traffic, and a following
    // copy consumes the next sequence and observes the projected storage. A
    // row of ones times a weight row of ones over sixteen features is exactly
    // sixteen in F32, so the accepted request also pins the consumer-visible
    // result while every padding byte of both owners stays the caller's.
    constexpr std::uint64_t kF32Sixteen = 0x41800000u;
    const RmsnormEncodings& f32 = rmsnorm_encodings(iom::DataType::F32);
    auto linear_copy = device->create_tensor(spec);
    const std::size_t allocator_events = allocator.events.size();
    x->view().copy_from_host(
            uniform_logical(iom::DataType::F32, 16 * 16, f32.one));
    w->view().copy_from_host(
            uniform_logical(iom::DataType::F32, 16 * 16, f32.one));
    CHECK_EQ(
            queue->linear_workspace_requirements(
                    x->view(), w->view(), y->view(), 0, 16,
                    iom::LinearOutputLayout::ordinary, 1, 16),
            (iom::WorkspaceRequirements{0, 1}));
    const iom::oid linear_token = queue->linear(
            x->view(), w->view(), y->view(), 0, 16,
            iom::LinearOutputLayout::ordinary, 1, 16);
    REQUIRE(iom::oid_is_token(linear_token));
    const iom::oid linear_consumer =
            queue->copy(y->view(), linear_copy->view());
    REQUIRE(iom::oid_is_token(linear_consumer));
    CHECK_EQ(
            token_sequence(linear_consumer), token_sequence(linear_token) + 1);
    queue->wait(linear_consumer);
    CHECK_EQ(allocator.events.size(), allocator_events);
    expect_storage_matches(
            *y,
            expected_uniform_rmsnorm_storage(y->view(), spec, kF32Sixteen));
    expect_storage_matches(
            *linear_copy,
            expected_uniform_rmsnorm_storage(
                    linear_copy->view(), spec, kF32Sixteen));

    // The reported zero requirement admits only the empty default view: a
    // supplied owner is invalid input, and the CPU device still refuses to
    // create positive raw workspace at all.
    const std::unique_ptr<iom::RawWorkspace> empty_workspace =
            device->create_workspace(0);
    CHECK_EQ(
            queue->linear(
                    x->view(), w->view(), y->view(), 0, 16,
                    iom::LinearOutputLayout::ordinary, 1, 16,
                    empty_workspace->view()),
            iom::to_oid(iom::OidError::InvalidArgument));
    CHECK_THROWS_AS(device->create_workspace(1), std::invalid_argument);

    // RMSNorm is declared and admitted by common code, and the CPU port
    // implements it: the valid-shape request is accepted, executed on the
    // queue, and observed through the established wait path, while the pure
    // requirement query reports the exact zero-scratch requirement. A row of
    // ones with unit scale and `eps == 3` is exactly `1 / sqrt(1 + 3) = 0.5`
    // in F32, so the accepted request also pins the consumer-visible result.
    scale->view().copy_from_host(
            uniform_logical(iom::DataType::F32, 16, f32.one));
    const iom::oid rmsnorm_token =
            queue->rmsnorm(x->view(), scale->view(), rmsnorm_out->view(), 3.0f);
    REQUIRE(iom::oid_is_token(rmsnorm_token));
    CHECK_NOTHROW(queue->wait(rmsnorm_token));
    CHECK_EQ(
            queue->rmsnorm_workspace_requirements(
                    x->view(), scale->view(), rmsnorm_out->view(), 3.0f),
            (iom::WorkspaceRequirements{0, 1}));
    expect_storage_matches(
            *rmsnorm_out,
            expected_uniform_rmsnorm_storage(
                    rmsnorm_out->view(), spec, f32.half));
    CHECK_EQ(
            queue->sdpa(
                    sdpa_q->view(), sdpa_kv->view(), sdpa_kv->view(),
                    sdpa_out->view(), 0, 16),
            iom::to_oid(iom::OidError::Unsupported));
    expect_storage_matches(*sdpa_out, sdpa_untouched);

    // A recognized inapplicable leaf with a valid shape stays unsupported by
    // common admission on every backend, independent of any port, and leaves
    // its output untouched.
    auto integer_x =
            device->create_tensor(make_spec({16, 16}, iom::DataType::I16));
    auto integer_scale =
            device->create_tensor(make_spec({1, 16}, iom::DataType::I16));
    auto integer_out =
            device->create_tensor(make_spec({16, 16}, iom::DataType::I16));
    fill_storage(*integer_out, kSentinel);
    const std::vector<std::byte> integer_out_untouched =
            snapshot_storage(*integer_out);
    CHECK_EQ(
            queue->rmsnorm(
                    integer_x->view(), integer_scale->view(),
                    integer_out->view(), 1e-6f),
            iom::to_oid(iom::OidError::Unsupported));
    expect_storage_matches(*integer_out, integer_out_untouched);


    // The four accepted binary operations consume the first four sequences,
    // the accepted linear projection and its consumer the fifth and sixth, and
    // the accepted RMSNorm the seventh, so the probe copy is the eighth.
    const iom::oid probe = queue->copy(x->view(), y->view());
    CHECK_EQ(token_sequence(probe), 8);
    queue->wait(probe);
}

// ---------------------------------------------------------------------------
// CPU RMSNorm
// ---------------------------------------------------------------------------

TEST_CASE("CPU RMSNorm computes every applicable floating leaf in place") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    // Two independent leading planes, three rows, and a non-tile feature
    // width. A row of ones with unit scale and `eps == 3` is exactly
    // `1 / sqrt(1 + 3) = 0.5`, which all nine leaves represent, so each leaf's
    // expectation is one fixed raw encoding.
    constexpr std::size_t planes = 2;
    constexpr std::size_t rows = 3;
    constexpr std::size_t features = 17;
    constexpr std::size_t elements = planes * rows * features;
    for (const RmsnormEncodings& encodings : kRmsnormEncodings) {
        const iom::TensorSpec spec =
                make_spec({planes, rows, features}, encodings.data_type);
        auto x = device->create_tensor(spec);
        auto scale = device->create_tensor(
                make_spec({1, features}, encodings.data_type));
        auto out = device->create_tensor(spec);
        x->view().copy_from_host(
                uniform_logical(encodings.data_type, elements, encodings.one));
        scale->view().copy_from_host(uniform_logical(
                encodings.data_type, features, encodings.one));
        fill_storage(*out, kSentinel);

        const iom::oid token =
                queue->rmsnorm(x->view(), scale->view(), out->view(), 3.0f);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        expect_storage_matches(
                *out,
                expected_uniform_rmsnorm_storage(
                        out->view(), spec, encodings.half));
    }
}

TEST_CASE("CPU RMSNorm reads only logical features of transformed views") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    // A sub-byte leaf in a four-plane owner whose padding is poisoned: `x` is
    // filled with `0xFF` — every narrow-format padded element decodes to a
    // large negative value — before its logical ones are copied in, so any
    // padded contribution would be observable. Both requested windows are
    // transformed leading slices, so their own plane offsets and strides must
    // be honored.
    constexpr std::size_t rows = 17;
    constexpr std::size_t features = 17;
    constexpr std::size_t window_planes = 2;
    constexpr std::size_t elements = window_planes * rows * features;
    const iom::TensorSpec owner_spec =
            make_spec({4, rows, features}, iom::DataType::F4_E2M1);
    const RmsnormEncodings& encodings =
            rmsnorm_encodings(iom::DataType::F4_E2M1);
    auto x_owner = device->create_tensor(owner_spec);
    auto out_owner = device->create_tensor(owner_spec);
    auto scale = device->create_tensor(
            make_spec({1, features}, iom::DataType::F4_E2M1));
    fill_storage(*x_owner, std::byte{0xFF});
    fill_storage(*out_owner, kSentinel);

    iom::TensorView x = x_owner->view().slice(0, 1, window_planes);
    iom::TensorView out = out_owner->view().slice(0, 1, window_planes);
    x.copy_from_host(uniform_logical(
            iom::DataType::F4_E2M1, elements, encodings.one));
    scale->view().copy_from_host(uniform_logical(
            iom::DataType::F4_E2M1, features, encodings.one));

    const iom::oid token = queue->rmsnorm(x, scale->view(), out, 3.0f);
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue->wait(token));

    // Exactly the logical elements of the selected planes changed; the whole
    // of planes 0 and 3 and every padded element of planes 1 and 2 still hold
    // the sentinel the caller left there.
    expect_storage_matches(
            *out_owner,
            expected_uniform_rmsnorm_storage(out, owner_spec, encodings.half));
}

TEST_CASE("CPU RMSNorm applies a signed non-unit scale to mixed magnitudes") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    // Every `|x|` pair makes each row's sum exactly 6.25, so `eps = 2.4375`
    // gives the shared norm exactly `1 / sqrt(1.5625 + 2.4375) = 0.5` and each
    // output is the exactly representable product of three factors: 2, 1, 0.25
    // and 0.125 with both signs. The second row is the sign mirror of the
    // first, so a row swap or a shared reduction across rows is observable.
    const iom::TensorSpec spec = make_spec({2, 4}, iom::DataType::F32);
    auto x = device->create_tensor(spec);
    auto scale = device->create_tensor(make_spec({1, 4}, iom::DataType::F32));
    auto out = device->create_tensor(spec);
    x->view().copy_from_host(packed_logical(
            iom::DataType::F32,
            {0x40000000u, 0xBF000000u, 0x3F800000u, 0xBF800000u,
             0xC0000000u, 0x3F000000u, 0xBF800000u, 0x3F800000u}));
    scale->view().copy_from_host(packed_logical(
            iom::DataType::F32,
            {0x40000000u, 0xC0800000u, 0x3F000000u, 0xBE800000u}));
    fill_storage(*out, kSentinel);

    const iom::oid token =
            queue->rmsnorm(x->view(), scale->view(), out->view(), 2.4375f);
    REQUIRE(iom::oid_is_token(token));
    CHECK_NOTHROW(queue->wait(token));
    expect_storage_matches(
            *out,
            expected_rmsnorm_storage(
                    out->view(), spec,
                    {0x40000000u, 0x3F800000u, 0x3E800000u, 0x3E000000u,
                     0xC0000000u, 0xBF800000u, 0xBE800000u, 0xBE000000u}));
}

TEST_CASE("CPU RMSNorm preserves signed zeros and row-local special values") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    // Row-local signed zeros and halves. `F = 4` with `|x| == 1` in every row
    // makes the shared norm exactly `1 / sqrt(1 + 3) = 0.5`, so each output is
    // the product of three exactly representable values and the sign of the
    // zero produced by a zero scale is preserved.
    {
        const iom::TensorSpec spec = make_spec({2, 4}, iom::DataType::F32);
        auto x = device->create_tensor(spec);
        auto scale =
                device->create_tensor(make_spec({1, 4}, iom::DataType::F32));
        auto out = device->create_tensor(spec);
        x->view().copy_from_host(packed_logical(
                iom::DataType::F32,
                {0x3F800000u, 0xBF800000u, 0x3F800000u, 0xBF800000u,
                 0xBF800000u, 0x3F800000u, 0xBF800000u, 0x3F800000u}));
        scale->view().copy_from_host(packed_logical(
                iom::DataType::F32,
                {0x3F800000u, 0xBF800000u, 0x00000000u, 0x80000000u}));
        fill_storage(*out, kSentinel);

        const iom::oid token =
                queue->rmsnorm(x->view(), scale->view(), out->view(), 3.0f);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        expect_storage_matches(
                *out,
                expected_rmsnorm_storage(
                        out->view(), spec,
                        {0x3F000000u, 0x3F000000u, 0x00000000u, 0x00000000u,
                         0xBF000000u, 0xBF000000u, 0x80000000u, 0x80000000u}));
    }

    // An all-zero row with a finite epsilon keeps signed zeros: the zero sum
    // leaves a finite reciprocal square root, and each zero carries the
    // product of its input and scale signs.
    {
        const iom::TensorSpec spec = make_spec({1, 4}, iom::DataType::F32);
        auto x = device->create_tensor(spec);
        auto scale =
                device->create_tensor(make_spec({1, 4}, iom::DataType::F32));
        auto out = device->create_tensor(spec);
        x->view().copy_from_host(packed_logical(
                iom::DataType::F32,
                {0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u}));
        scale->view().copy_from_host(packed_logical(
                iom::DataType::F32,
                {0x3F800000u, 0xBF800000u, 0xBF800000u, 0x3F800000u}));
        fill_storage(*out, kSentinel);

        const iom::oid token =
                queue->rmsnorm(x->view(), scale->view(), out->view(), 3.0f);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        expect_storage_matches(
                *out,
                expected_rmsnorm_storage(
                        out->view(), spec,
                        {0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u}));
    }

    // `eps == 0` on an all-zero row: the reciprocal square root is infinite
    // and every output is a quiet NaN. F32 keeps the canonical quiet NaN
    // encoding, while F4_E2M1 has no NaN encoding and saturates to its
    // maximum finite value 6.
    {
        const RmsnormEncodings& f32 = rmsnorm_encodings(iom::DataType::F32);
        const iom::TensorSpec spec = make_spec({1, 4}, iom::DataType::F32);
        auto x = device->create_tensor(spec);
        auto scale =
                device->create_tensor(make_spec({1, 4}, iom::DataType::F32));
        auto out = device->create_tensor(spec);
        x->view().copy_from_host(
                uniform_logical(iom::DataType::F32, 4, 0x00000000u));
        scale->view().copy_from_host(uniform_logical(
                iom::DataType::F32, 4, f32.one));
        fill_storage(*out, kSentinel);

        const iom::oid token =
                queue->rmsnorm(x->view(), scale->view(), out->view(), 0.0f);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        expect_storage_matches(
                *out,
                expected_uniform_rmsnorm_storage(
                        out->view(), spec, 0x7FC00000u));

        const iom::TensorSpec narrow_spec =
                make_spec({1, 4}, iom::DataType::F4_E2M1);
        const RmsnormEncodings& narrow =
                rmsnorm_encodings(iom::DataType::F4_E2M1);
        auto narrow_x = device->create_tensor(narrow_spec);
        auto narrow_scale = device->create_tensor(
                make_spec({1, 4}, iom::DataType::F4_E2M1));
        auto narrow_out = device->create_tensor(narrow_spec);
        narrow_x->view().copy_from_host(
                uniform_logical(iom::DataType::F4_E2M1, 4, 0x0));
        narrow_scale->view().copy_from_host(uniform_logical(
                iom::DataType::F4_E2M1, 4, narrow.one));
        fill_storage(*narrow_out, kSentinel);

        const iom::oid narrow_token = queue->rmsnorm(
                narrow_x->view(), narrow_scale->view(), narrow_out->view(),
                0.0f);
        REQUIRE(iom::oid_is_token(narrow_token));
        CHECK_NOTHROW(queue->wait(narrow_token));
        // 0x7 is the maximum finite F4_E2M1 value, 6.
        expect_storage_matches(
                *narrow_out,
                expected_uniform_rmsnorm_storage(
                        narrow_out->view(), narrow_spec, 0x7u));
    }

    // Infinite features make their own row's sum infinite: the finite
    // features of that row normalize to signed zeros and the infinite
    // features become quiet NaNs before the scale multiply, while the
    // sibling zero row keeps its signed zeros.
    {
        const iom::TensorSpec spec = make_spec({2, 4}, iom::DataType::F16);
        auto x = device->create_tensor(spec);
        auto scale =
                device->create_tensor(make_spec({1, 4}, iom::DataType::F16));
        auto out = device->create_tensor(spec);
        x->view().copy_from_host(packed_logical(
                iom::DataType::F16,
                {0x7C00u, 0xFC00u, 0x3C00u, 0xBC00u,
                 0x0000u, 0x8000u, 0x0000u, 0x8000u}));
        scale->view().copy_from_host(
                uniform_logical(iom::DataType::F16, 4, 0x3C00u));
        fill_storage(*out, kSentinel);

        const iom::oid token =
                queue->rmsnorm(x->view(), scale->view(), out->view(), 1.0e-5f);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue->wait(token));
        expect_storage_matches(
                *out,
                expected_rmsnorm_storage(
                        out->view(), spec,
                        {0x7E00u, 0x7E00u, 0x0000u, 0x8000u,
                         0x0000u, 0x8000u, 0x0000u, 0x8000u}));
    }
}

TEST_CASE("CPU RMSNorm covers the tiled boundary extents with independent planes") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    // Rows and features immediately around the fixed 16x16 tile, with three
    // independent leading planes: the divisor is the logical `F` in every
    // case, so a row of ones with unit scale and `eps == 3` is exactly 0.5
    // even when the feature run is not tile-aligned.
    constexpr std::array<std::size_t, 4> kExtents = {1, 15, 16, 17};
    constexpr std::size_t planes = 3;
    for (const std::size_t rows : kExtents) {
        for (const std::size_t features : kExtents) {
            const iom::TensorSpec spec =
                    make_spec({planes, rows, features}, iom::DataType::F32);
            auto x = device->create_tensor(spec);
            auto scale = device->create_tensor(
                    make_spec({1, features}, iom::DataType::F32));
            auto out = device->create_tensor(spec);
            const std::size_t elements = planes * rows * features;
            x->view().copy_from_host(uniform_logical(
                    iom::DataType::F32, elements, 0x3F800000u));
            scale->view().copy_from_host(uniform_logical(
                    iom::DataType::F32, features, 0x3F800000u));
            fill_storage(*out, kSentinel);

            const iom::oid token =
                    queue->rmsnorm(x->view(), scale->view(), out->view(), 3.0f);
            REQUIRE(iom::oid_is_token(token));
            CHECK_NOTHROW(queue->wait(token));
            expect_storage_matches(
                    *out,
                    expected_uniform_rmsnorm_storage(
                            out->view(), spec, 0x3F000000u));
        }
    }
}

TEST_CASE("CPU RMSNorm admission rejects before effects") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    const iom::TensorSpec spec = make_spec({2, 17, 33}, iom::DataType::F32);
    auto x = device->create_tensor(spec);
    auto scale =
            device->create_tensor(make_spec({1, 33}, iom::DataType::F32));
    auto out = device->create_tensor(spec);
    x->view().copy_from_host(uniform_logical(
            iom::DataType::F32, spec.shape.element_count(), 0x3F800000u));
    scale->view().copy_from_host(
            uniform_logical(iom::DataType::F32, 33, 0x3F800000u));
    fill_storage(*out, kSentinel);
    const std::vector<std::byte> untouched = snapshot_storage(*out);

    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const iom::oid overflow = iom::to_oid(iom::OidError::Overflow);

    // `x` and `out` must agree on `[...,R,F]`, and `scale` must be exactly
    // `[1,F]`; a rank-one scale cannot even be materialized.
    auto mismatched =
            device->create_tensor(make_spec({2, 17, 17}, iom::DataType::F32));
    CHECK_EQ(
            queue->rmsnorm(
                    x->view(), scale->view(), mismatched->view(), 1e-6f),
            invalid);
    auto narrow_scale =
            device->create_tensor(make_spec({1, 17}, iom::DataType::F32));
    CHECK_EQ(
            queue->rmsnorm(
                    x->view(), narrow_scale->view(), out->view(), 1e-6f),
            invalid);
    auto plane_scale =
            device->create_tensor(make_spec({3, 33}, iom::DataType::F32));
    CHECK_EQ(
            queue->rmsnorm(x->view(), plane_scale->view(), out->view(), 1e-6f),
            invalid);

    // Leaf type and quantization must match and stay applicable.
    auto bf16_x =
            device->create_tensor(make_spec({2, 17, 33}, iom::DataType::BF16));
    CHECK_EQ(
            queue->rmsnorm(bf16_x->view(), scale->view(), out->view(), 1e-6f),
            invalid);
    // CPU storage cannot materialize a grouped-quantized owner, so the
    // non-`NONE` quantization probe mutates the specification copy that a view
    // shares with its own owner — keeping both consistent — and restores it
    // afterwards.
    {
        auto quantized_x = device->create_tensor(spec);
        auto quantized_scale =
                device->create_tensor(make_spec({1, 33}, iom::DataType::F32));
        auto quantized_out = device->create_tensor(spec);
        fill_storage(*quantized_out, kSentinel);
        const std::vector<std::byte> quantized_untouched =
                snapshot_storage(*quantized_out);
        const auto set_quantization = [](iom::TensorView& view,
                                         iom::QuantizationFormat quantization) {
            const_cast<iom::TensorSpec&>(view.spec()).quantization =
                    quantization;
        };
        const auto quantify_views = [&](iom::QuantizationFormat quantization) {
            for (iom::TensorView* view :
                 {&quantized_x->view(), &quantized_scale->view(),
                  &quantized_out->view()}) {
                set_quantization(*view, quantization);
            }
        };
        quantify_views(iom::QuantizationFormat::GGML_Q4_0);
        CHECK_EQ(
                queue->rmsnorm(
                        quantized_x->view(), quantized_scale->view(),
                        quantized_out->view(), 1e-6f),
                unsupported);
        CHECK_THROWS_AS(
                (void)queue->rmsnorm_workspace_requirements(
                        quantized_x->view(), quantized_scale->view(),
                        quantized_out->view(), 1e-6f),
                std::runtime_error);
        quantify_views(iom::QuantizationFormat::NONE);
        expect_storage_matches(*quantized_out, quantized_untouched);
    }
    for (const iom::DataType leaf :
         {iom::DataType::BOOL, iom::DataType::I16, iom::DataType::F8_E8M0}) {
        auto leaf_x = device->create_tensor(make_spec({2, 17, 33}, leaf));
        auto leaf_scale = device->create_tensor(make_spec({1, 33}, leaf));
        auto leaf_out = device->create_tensor(make_spec({2, 17, 33}, leaf));
        fill_storage(*leaf_out, kSentinel);
        const std::vector<std::byte> leaf_untouched = snapshot_storage(*leaf_out);
        CHECK_EQ(
                queue->rmsnorm(
                        leaf_x->view(), leaf_scale->view(), leaf_out->view(),
                        1e-6f),
                unsupported);
        CHECK_THROWS_AS(
                (void)queue->rmsnorm_workspace_requirements(
                        leaf_x->view(), leaf_scale->view(), leaf_out->view(),
                        1e-6f),
                std::runtime_error);
        expect_storage_matches(*leaf_out, leaf_untouched);
    }

    // A foreign device's storage never participates, even when it is another
    // CPU reference device.
    RecordingAllocator foreign_allocator;
    auto foreign_device = iom::make_cpu_device(foreign_allocator);
    auto foreign_x = foreign_device->create_tensor(spec);
    CHECK_EQ(
            queue->rmsnorm(foreign_x->view(), scale->view(), out->view(), 1e-6f),
            invalid);

    // Output storage must be disjoint from both inputs; RMSNorm consumes no
    // raw workspace, so any supplied workspace with an owner is invalid even
    // when its range is empty.
    CHECK_EQ(
            queue->rmsnorm(x->view(), scale->view(), x->view(), 1e-6f), invalid);
    const std::unique_ptr<iom::RawWorkspace> workspace =
            device->create_workspace(0);
    CHECK_EQ(
            queue->rmsnorm(
                    x->view(), scale->view(), out->view(), 1e-6f,
                    workspace->view()),
            invalid);

    // Epsilon must be finite and nonnegative.
    for (const float epsilon :
         {std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::infinity(), -1.0f}) {
        CHECK_EQ(
                queue->rmsnorm(
                        x->view(), scale->view(), out->view(), epsilon),
                invalid);
    }

    // Malformed view metadata and checked overflow are rejected before any
    // address arithmetic or output write. A second leading extent keeps the
    // stride product itself overflowing instead of merely addressing outside
    // the owner, which `validate_checked_view` rejects as invalid input.
    {
        const iom::TensorSpec wide_spec =
                make_spec({3, 2, 17, 33}, iom::DataType::F32);
        auto wide_x = device->create_tensor(wide_spec);
        auto wide_out = device->create_tensor(wide_spec);

        iom::TensorView malformed = wide_x->view();
        const_cast<std::size_t*>(malformed.plane_strides().data())[0] = 0;
        CHECK_EQ(
                queue->rmsnorm(malformed, scale->view(), wide_out->view(), 1e-6f),
                invalid);

        iom::TensorView overflowed = wide_x->view();
        const_cast<std::size_t*>(overflowed.plane_strides().data())[0] =
                std::numeric_limits<std::size_t>::max();
        CHECK_EQ(
                queue->rmsnorm(
                        overflowed, scale->view(), wide_out->view(), 1e-6f),
                overflow);
    }

    // Every rejection above wrote nothing and consumed no sequence: the first
    // accepted submission still takes sequence 1, its result is the exact
    // unit-scale one, and a completed token stays waitable.
    expect_storage_matches(*out, untouched);
    const iom::oid first =
            queue->rmsnorm(x->view(), scale->view(), out->view(), 3.0f);
    REQUIRE(iom::oid_is_token(first));
    CHECK_EQ(token_sequence(first), 1);
    CHECK_NOTHROW(queue->wait(first));
    CHECK_NOTHROW(queue->wait(first));
    expect_storage_matches(
            *out, expected_uniform_rmsnorm_storage(
                          out->view(), spec, 0x3F000000u));

    // Read/read overlap between `x` and `scale` stays valid and registers one
    // deduplicated owner.
    auto shared_operand =
            device->create_tensor(make_spec({1, 33}, iom::DataType::F32));
    auto shared_out =
            device->create_tensor(make_spec({1, 33}, iom::DataType::F32));
    shared_operand->view().copy_from_host(
            uniform_logical(iom::DataType::F32, 33, 0x3F800000u));
    fill_storage(*shared_out, kSentinel);
    const iom::oid aliased = queue->rmsnorm(
            shared_operand->view(), shared_operand->view(),
            shared_out->view(), 3.0f);
    REQUIRE(iom::oid_is_token(aliased));
    CHECK_EQ(token_sequence(aliased), 2);
    CHECK_NOTHROW(queue->wait(aliased));
    expect_storage_matches(
            *shared_out, expected_uniform_rmsnorm_storage(
                                 shared_out->view(), shared_out->view().spec(),
                                 0x3F000000u));
}

TEST_CASE("CPU RMSNorm keeps queue order and repeatable waits") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    // A non-tile BF16 plane: the queued RMSNorm is followed by a copy of its
    // own output, so the copy observes exactly the completed RMSNorm result
    // and the two tokens keep FIFO order. Owning tensors stay alive across
    // both waits, and a completed token is waitable repeatedly.
    const iom::TensorSpec spec = make_spec({2, 17, 33}, iom::DataType::BF16);
    auto x = device->create_tensor(spec);
    auto scale =
            device->create_tensor(make_spec({1, 33}, iom::DataType::BF16));
    auto out = device->create_tensor(spec);
    auto scratch = device->create_tensor(spec);
    x->view().copy_from_host(uniform_logical(
            iom::DataType::BF16, spec.shape.element_count(), 0x3F80u));
    scale->view().copy_from_host(
            uniform_logical(iom::DataType::BF16, 33, 0x3F80u));
    fill_storage(*out, kSentinel);
    fill_storage(*scratch, kSentinel);

    const iom::oid rmsnorm_token =
            queue->rmsnorm(x->view(), scale->view(), out->view(), 3.0f);
    const iom::oid copy_token = queue->copy(out->view(), scratch->view());
    REQUIRE(iom::oid_is_token(rmsnorm_token));
    REQUIRE(iom::oid_is_token(copy_token));
    CHECK_EQ(token_sequence(rmsnorm_token), 1);
    CHECK_EQ(token_sequence(copy_token), 2);

    queue->wait(copy_token);
    queue->wait(rmsnorm_token);
    CHECK_NOTHROW(queue->wait(rmsnorm_token));
    const std::vector<std::byte> expected =
            expected_uniform_rmsnorm_storage(out->view(), spec, 0x3F00u);
    expect_storage_matches(*out, expected);
    expect_storage_matches(*scratch, expected);
}
