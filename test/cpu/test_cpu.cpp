#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

constexpr std::uint64_t kTokenSequenceBits = 56;
constexpr std::uint64_t kTokenSequenceMask =
        (std::uint64_t{1} << kTokenSequenceBits) - 1;

std::uint8_t token_queue(iom::oid token) {
    return static_cast<std::uint8_t>(token >> kTokenSequenceBits);
}

std::uint64_t token_sequence(iom::oid token) {
    return token & kTokenSequenceMask;
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

// Owner element slot of the view's linear-th logical element, computed with
// the checked core layout helper: the view's offset and strides yield the
// owner plane, which is inverted into dense owner coordinates. Independent
// of the backend's traversal code.
std::size_t view_slot_at(
        const iom::TensorView& view, const iom::TensorSpec& owner_spec,
        std::size_t linear) {
    const std::span<const std::size_t> dims =
            view.spec().shape.dimensions();
    const std::size_t leading_rank = dims.size() - 2;
    const std::size_t rows = dims[leading_rank];
    const std::size_t columns = dims[leading_rank + 1];
    const std::span<const std::size_t> strides = view.plane_strides();
    const std::span<const std::size_t> owner_dims =
            owner_spec.shape.dimensions();
    const std::size_t owner_leading_rank = owner_dims.size() - 2;

    std::size_t rest = linear;
    const std::size_t column = rest % columns;
    rest /= columns;
    const std::size_t row = rest % rows;
    rest /= rows;
    std::size_t plane = view.plane_offset();
    for (std::size_t k = leading_rank; k-- > 0;) {
        plane += (rest % dims[k]) * strides[k];
        rest /= dims[k];
    }
    std::vector<std::size_t> coordinates(owner_dims.size());
    for (std::size_t k = owner_leading_rank; k-- > 0;) {
        coordinates[k] = plane % owner_dims[k];
        plane /= owner_dims[k];
    }
    coordinates[owner_leading_rank] = row;
    coordinates[owner_leading_rank + 1] = column;
    return iom::detail::standard_layout_slot(owner_spec, coordinates);
}

// Full-storage oracle for one owner specification. Starts at the sentinel
// the tests write into the real allocation, then tracks logical writes.
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
            const std::size_t slot = view_slot_at(view, spec_, linear);
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
                    view_slot_at(source_view, source_spec, linear);
            const std::size_t destination_slot =
                    view_slot_at(destination_view, spec_, linear);
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

    CHECK_THROWS_AS(
            queue->copy(other_shape->view(), destination->view()),
            std::invalid_argument);
    CHECK_THROWS_AS(
            queue->copy(destination->view(), other_shape->view()),
            std::invalid_argument);
    CHECK_THROWS_AS(
            queue->copy(other_type->view(), destination->view()),
            std::invalid_argument);
    CHECK_THROWS_AS(
            queue->copy(destination->view(), other_type->view()),
            std::invalid_argument);
    CHECK_THROWS_AS(
            queue->copy(foreign->view(), destination->view()),
            std::invalid_argument);
    CHECK_THROWS_AS(
            queue->copy(destination->view(), foreign->view()),
            std::invalid_argument);

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
// Unsupported compute capabilities
// ---------------------------------------------------------------------------

TEST_CASE("CPU compute operations reject capability without submitting") {
    RecordingAllocator allocator;
    auto device = iom::make_cpu_device(allocator);
    auto queue = device->create_ops();

    const iom::TensorSpec spec = make_spec({16, 16}, iom::DataType::F32);
    auto x = device->create_tensor(spec);
    auto y = device->create_tensor(spec);
    auto w = device->create_tensor(spec);
    auto attn = device->create_tensor(spec);

    fill_storage(*y, kSentinel);
    fill_storage(*attn, kSentinel);
    const std::vector<std::byte> y_untouched = snapshot_storage(*y);
    const std::vector<std::byte> attn_untouched = snapshot_storage(*attn);

    CHECK_THROWS_AS(queue->add(x->view(), x->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(queue->mul(x->view(), x->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(queue->silu(x->view(), y->view()), std::runtime_error);
    CHECK_THROWS_AS(queue->linear(x->view(), w->view(), y->view()),
                    std::runtime_error);
    CHECK_THROWS_AS(queue->rmsnorm(x->view(), y->view(), w->view(), 1e-6f, 1),
                    std::runtime_error);
    CHECK_THROWS_AS(
            queue->sdpa(x->view(), x->view(), x->view(), 1, 1, 16,
                        attn->view()),
            std::runtime_error);

    expect_storage_matches(*y, y_untouched);
    expect_storage_matches(*attn, attn_untouched);

    // Capability failures consumed no sequence numbers.
    const iom::oid probe = queue->copy(x->view(), y->view());
    CHECK_EQ(token_sequence(probe), 1);
    queue->wait(probe);
}
