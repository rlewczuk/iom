#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

#include "iom/alloc.hpp"

// The counting allocation-fault harness installed by test_iom.cpp's global
// operator new replace: it arms a failure at an exact allocation ordinal and
// counts observations. Declarations mirror test_iom.cpp exactly.
namespace iom_test {
bool should_fail_allocation() noexcept;
void arm_counting() noexcept;
void arm_failure(std::size_t ordinal) noexcept;
std::size_t disarm() noexcept;
}  // namespace iom_test

namespace {

bool is_aligned(void* ptr, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// Builds a deterministic ListAllocator geometry: after this call the free
// list is [0,64) [72,24) [120,8) (96 free bytes) with the 8-byte block at
// offset 64 and the 24-byte block at offset 96 still outstanding. Alignment
// 32 keeps every returned address aligned.
void prepare_split_list_state(
        iom::ListAllocator& allocator,
        void*& first,
        void*& second,
        void*& third,
        void*& fourth) {
    allocator.reset();
    first = allocator.alloc(32);
    second = allocator.alloc(32);
    third = allocator.alloc(8);
    fourth = allocator.alloc(24);
    allocator.free(first);
    allocator.free(second);
}

}  // namespace

TEST_CASE("LinearAllocator is non-copyable and non-movable") {
    CHECK(std::is_base_of_v<iom::Allocator, iom::LinearAllocator>);
    CHECK(std::is_base_of_v<iom::SingleBufferAllocatorBase, iom::LinearAllocator>);
    CHECK_FALSE(std::is_copy_constructible_v<iom::LinearAllocator>);
    CHECK_FALSE(std::is_copy_assignable_v<iom::LinearAllocator>);
    CHECK_FALSE(std::is_move_constructible_v<iom::LinearAllocator>);
    CHECK_FALSE(std::is_move_assignable_v<iom::LinearAllocator>);
}

TEST_CASE("LinearAllocator exposes normalized buffer properties") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::LinearAllocator allocator(buffer.data() + 1, buffer.size() - 1, 16);

    CHECK_EQ(allocator.align(), 16);
    CHECK_EQ(allocator.capacity(), 48);
    CHECK_EQ(allocator.alloc(1), buffer.data() + 16);
}

TEST_CASE("LinearAllocator allocates linearly with alignment") {
    alignas(32) std::array<std::byte, 128> buffer{};
    iom::LinearAllocator allocator(buffer.data(), buffer.size(), 16);

    void* first = allocator.alloc(5);
    void* second = allocator.alloc(5);

    CHECK(is_aligned(first, 16));
    CHECK(is_aligned(second, 16));
    CHECK_EQ(first, buffer.data());
    CHECK_EQ(second, buffer.data() + 16);
}

TEST_CASE("LinearAllocator uses default 32 byte alignment") {
    alignas(32) std::array<std::byte, 96> buffer{};
    iom::LinearAllocator allocator(buffer.data(), buffer.size());

    void* first = allocator.alloc(1);
    void* second = allocator.alloc(1);

    CHECK(is_aligned(first, 32));
    CHECK(is_aligned(second, 32));
}

TEST_CASE("LinearAllocator reset reuses the buffer") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::LinearAllocator allocator(buffer.data(), buffer.size());

    void* first = allocator.alloc(8);
    allocator.alloc(8);
    allocator.reset();
    void* after_reset = allocator.alloc(8);

    CHECK_EQ(after_reset, first);
}

TEST_CASE("LinearAllocator free is a no-op") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::LinearAllocator allocator(buffer.data(), buffer.size());

    void* first = allocator.alloc(8);
    allocator.free(first);
    void* second = allocator.alloc(8);

    CHECK_EQ(second, buffer.data() + 32);
}

TEST_CASE("LinearAllocator throws when out of memory") {
    alignas(32) std::array<std::byte, 32> buffer{};
    iom::LinearAllocator allocator(buffer.data(), buffer.size());

    allocator.alloc(32);

    CHECK_THROWS_AS(allocator.alloc(1), std::bad_alloc);
}

TEST_CASE("LinearAllocator rejects invalid construction arguments") {
    std::array<std::byte, 32> buffer{};

    CHECK_THROWS_AS(iom::LinearAllocator(nullptr, buffer.size()), std::invalid_argument);
    CHECK_THROWS_AS(iom::LinearAllocator(buffer.data(), buffer.size(), 0), std::invalid_argument);
    CHECK_THROWS_AS(iom::LinearAllocator(buffer.data(), buffer.size(), 3), std::invalid_argument);
}

TEST_CASE("SingleBufferAllocator rejects overflowing address ranges and alignment") {
    const auto near_limit = reinterpret_cast<void*>(
            std::numeric_limits<std::uintptr_t>::max() - static_cast<std::uintptr_t>(7));

    CHECK_THROWS_AS(iom::LinearAllocator(near_limit, 8, 8), std::overflow_error);
    CHECK_THROWS_AS(iom::LinearAllocator(near_limit, 0, 32), std::overflow_error);
}

TEST_CASE("ListAllocator is non-copyable and non-movable") {
    CHECK(std::is_base_of_v<iom::Allocator, iom::ListAllocator>);
    CHECK(std::is_base_of_v<iom::SingleBufferAllocatorBase, iom::ListAllocator>);
    CHECK_FALSE(std::is_copy_constructible_v<iom::ListAllocator>);
    CHECK_FALSE(std::is_copy_assignable_v<iom::ListAllocator>);
    CHECK_FALSE(std::is_move_constructible_v<iom::ListAllocator>);
    CHECK_FALSE(std::is_move_assignable_v<iom::ListAllocator>);
}

TEST_CASE("ListAllocator exposes normalized buffer properties") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::ListAllocator allocator(buffer.data() + 1, buffer.size() - 1, 16);

    CHECK_EQ(allocator.align(), 16);
    CHECK_EQ(allocator.capacity(), 48);
    CHECK_EQ(allocator.free_bytes(), 48);
    CHECK_EQ(allocator.alloc(1), buffer.data() + 16);
}

TEST_CASE("ListAllocator allocates aligned blocks") {
    alignas(32) std::array<std::byte, 128> buffer{};
    iom::ListAllocator allocator(buffer.data(), buffer.size(), 16);

    void* first = allocator.alloc(5);
    void* second = allocator.alloc(5);

    CHECK(is_aligned(first, 16));
    CHECK(is_aligned(second, 16));
    CHECK_EQ(first, buffer.data());
    CHECK_EQ(second, buffer.data() + 16);
}

TEST_CASE("ListAllocator handles zero allocation and null free") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::ListAllocator allocator(buffer.data(), buffer.size());

    CHECK_EQ(allocator.alloc(0), nullptr);
    allocator.free(nullptr);

    CHECK_EQ(allocator.free_bytes(), allocator.capacity());
}

TEST_CASE("ListAllocator uses best-fit free block") {
    alignas(32) std::array<std::byte, 128> buffer{};
    iom::ListAllocator allocator(buffer.data(), buffer.size(), 8);

    void* first = allocator.alloc(8);
    void* second = allocator.alloc(16);
    void* third = allocator.alloc(8);
    void* fourth = allocator.alloc(32);

    allocator.free(second);
    allocator.free(fourth);

    void* best_fit = allocator.alloc(12);

    CHECK_EQ(first, buffer.data());
    CHECK_EQ(third, buffer.data() + 24);
    CHECK_EQ(best_fit, second);
}

TEST_CASE("ListAllocator coalesces freed blocks") {
    alignas(32) std::array<std::byte, 128> buffer{};
    iom::ListAllocator allocator(buffer.data(), buffer.size(), 8);

    void* first = allocator.alloc(32);
    void* second = allocator.alloc(32);
    void* third = allocator.alloc(32);

    allocator.free(first);
    allocator.free(second);

    void* merged = allocator.alloc(64);

    CHECK_EQ(merged, buffer.data());

    allocator.free(merged);
    allocator.free(third);
    CHECK_EQ(allocator.free_bytes(), allocator.capacity());
}

TEST_CASE("ListAllocator reset reuses the buffer") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::ListAllocator allocator(buffer.data(), buffer.size());

    void* first = allocator.alloc(8);
    allocator.alloc(8);
    allocator.reset();
    void* after_reset = allocator.alloc(8);

    CHECK_EQ(after_reset, first);
    CHECK_EQ(allocator.free_bytes(), allocator.capacity() - 8);
}

TEST_CASE("ListAllocator rejects invalid frees and out-of-memory allocation") {
    alignas(32) std::array<std::byte, 32> buffer{};
    alignas(32) std::array<std::byte, 32> other{};
    iom::ListAllocator allocator(buffer.data(), buffer.size());

    void* allocated = allocator.alloc(32);

    CHECK_THROWS_AS(allocator.alloc(1), std::bad_alloc);
    CHECK_THROWS_AS(allocator.free(other.data()), std::invalid_argument);

    allocator.free(allocated);
    CHECK_THROWS_AS(allocator.free(allocated), std::invalid_argument);
}

TEST_CASE("ListAllocator rejects invalid construction arguments") {
    std::array<std::byte, 32> buffer{};

    CHECK_THROWS_AS(iom::ListAllocator(nullptr, buffer.size()), std::invalid_argument);
    CHECK_THROWS_AS(iom::ListAllocator(buffer.data(), buffer.size(), 0), std::invalid_argument);
    CHECK_THROWS_AS(iom::ListAllocator(buffer.data(), buffer.size(), 3), std::invalid_argument);
}

TEST_CASE("FixedSizeAllocator is non-copyable and non-movable") {
    CHECK(std::is_base_of_v<iom::Allocator, iom::FixedSizeAllocator>);
    CHECK(std::is_base_of_v<iom::SingleBufferAllocatorBase, iom::FixedSizeAllocator>);
    CHECK_FALSE(std::is_copy_constructible_v<iom::FixedSizeAllocator>);
    CHECK_FALSE(std::is_copy_assignable_v<iom::FixedSizeAllocator>);
    CHECK_FALSE(std::is_move_constructible_v<iom::FixedSizeAllocator>);
    CHECK_FALSE(std::is_move_assignable_v<iom::FixedSizeAllocator>);
}

TEST_CASE("FixedSizeAllocator exposes normalized buffer properties") {
    alignas(32) std::array<std::byte, 96> buffer{};
    iom::FixedSizeAllocator allocator(buffer.data() + 1, buffer.size() - 1, 16, 8);

    CHECK_EQ(allocator.align(), 16);
    CHECK_EQ(allocator.capacity(), 80);
    CHECK_EQ(allocator.payload_size(), 8);
    CHECK_EQ(allocator.stride(), 16);
    CHECK_EQ(allocator.block_count(), 5);
    CHECK_EQ(allocator.free_count(), 5);
    CHECK_EQ(allocator.alloc(1), buffer.data() + 16);
}

TEST_CASE("FixedSizeAllocator allocates fixed-size aligned blocks") {
    alignas(32) std::array<std::byte, 96> buffer{};
    iom::FixedSizeAllocator allocator(buffer.data(), buffer.size(), 16, 12);

    void* first = allocator.alloc(12);
    void* second = allocator.alloc(1);
    void* third = allocator.alloc(8);

    CHECK(is_aligned(first, 16));
    CHECK(is_aligned(second, 16));
    CHECK(is_aligned(third, 16));
    CHECK_EQ(first, buffer.data());
    CHECK_EQ(second, buffer.data() + 16);
    CHECK_EQ(third, buffer.data() + 32);
    CHECK_EQ(allocator.free_count(), 3);
}

TEST_CASE("FixedSizeAllocator handles zero allocation and null free") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::FixedSizeAllocator allocator(buffer.data(), buffer.size(), 8);

    CHECK_EQ(allocator.alloc(0), nullptr);
    allocator.free(nullptr);

    CHECK_EQ(allocator.free_count(), allocator.block_count());
}

TEST_CASE("FixedSizeAllocator reuses freed blocks") {
    alignas(32) std::array<std::byte, 96> buffer{};
    iom::FixedSizeAllocator allocator(buffer.data(), buffer.size(), 16, 8);

    void* first = allocator.alloc(8);
    void* second = allocator.alloc(8);

    allocator.free(first);
    void* reused = allocator.alloc(8);

    CHECK_EQ(second, buffer.data() + 16);
    CHECK_EQ(reused, first);
    CHECK_EQ(allocator.free_count(), allocator.block_count() - 2);
}

TEST_CASE("FixedSizeAllocator reset reuses all blocks") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::FixedSizeAllocator allocator(buffer.data(), buffer.size(), 16, 8);

    void* first = allocator.alloc(8);
    allocator.alloc(8);
    allocator.reset();
    void* after_reset = allocator.alloc(8);

    CHECK_EQ(after_reset, first);
    CHECK_EQ(allocator.free_count(), allocator.block_count() - 1);
}

TEST_CASE("FixedSizeAllocator maps between pointers and block indices") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::FixedSizeAllocator allocator(buffer.data(), buffer.size(), 16, 8);

    void* second = allocator.ptr_from_index(1);

    CHECK_EQ(second, buffer.data() + 16);
    CHECK_EQ(allocator.index_of(second), 1);
    CHECK_THROWS_AS(static_cast<void>(allocator.ptr_from_index(allocator.block_count())), std::out_of_range);
    CHECK_THROWS_AS(static_cast<void>(allocator.index_of(buffer.data() + 1)), std::invalid_argument);
}

TEST_CASE("FixedSizeAllocator rejects invalid frees and exhausted allocations") {
    alignas(32) std::array<std::byte, 32> buffer{};
    alignas(32) std::array<std::byte, 32> other{};
    iom::FixedSizeAllocator allocator(buffer.data(), buffer.size(), 16, 8);

    void* first = allocator.alloc(8);
    void* second = allocator.alloc(8);

    CHECK_THROWS_AS(allocator.alloc(9), std::bad_alloc);
    CHECK_THROWS_AS(allocator.alloc(8), std::bad_alloc);
    CHECK_THROWS_AS(allocator.free(other.data()), std::invalid_argument);

    allocator.free(first);
    CHECK_THROWS_AS(allocator.free(first), std::invalid_argument);
    allocator.free(second);
}

TEST_CASE("FixedSizeAllocator rejects invalid construction arguments") {
    std::array<std::byte, 32> buffer{};

    CHECK_THROWS_AS(iom::FixedSizeAllocator(nullptr, buffer.size(), 8), std::invalid_argument);
    CHECK_THROWS_AS(iom::FixedSizeAllocator(buffer.data(), buffer.size(), 0), std::invalid_argument);
    CHECK_THROWS_AS(iom::FixedSizeAllocator(buffer.data(), buffer.size(), 0, 8), std::invalid_argument);
    CHECK_THROWS_AS(iom::FixedSizeAllocator(buffer.data(), buffer.size(), 3, 8), std::invalid_argument);
}

TEST_CASE("FixedSizeAllocator rejects overflowing payload stride") {
    std::array<std::byte, 32> buffer{};

    CHECK_THROWS_AS(
            iom::FixedSizeAllocator(buffer.data(), buffer.size(), 32, std::numeric_limits<std::size_t>::max()),
            std::overflow_error);
    CHECK_THROWS_AS(iom::FixedSizeAllocator(buffer.data(), buffer.size(), 32, 0), std::invalid_argument);
}

TEST_CASE("ListAllocator allocation bookkeeping failure rolls back atomically") {
    // Prepared geometry: free list [0,64) [72,24) [120,8), 96 free bytes.
    // A 16-byte request best-fits [0,64) and splits it into suffix [16,48)
    // at the head; the next aligned 16-byte request then serves at +32.
    alignas(32) std::array<std::byte, 128> probe_buffer{};
    iom::ListAllocator probe(probe_buffer.data(), probe_buffer.size(), 32);
    void* first = nullptr;
    void* second = nullptr;
    void* third = nullptr;
    void* fourth = nullptr;
    prepare_split_list_state(probe, first, second, third, fourth);
    CHECK_EQ(probe.free_bytes(), 96);

    // Measure how many host bookkeeping allocations one allocation performs.
    iom_test::arm_counting();
    void* probed = probe.alloc(16);
    const std::size_t fault_points = iom_test::disarm();
    REQUIRE_EQ(probed, probe_buffer.data());
    REQUIRE(fault_points > 0);

    for (std::size_t ordinal = 1; ordinal <= fault_points; ++ordinal) {
        alignas(32) std::array<std::byte, 128> buffer{};
        iom::ListAllocator allocator(buffer.data(), buffer.size(), 32);
        prepare_split_list_state(allocator, first, second, third, fourth);
        CHECK_EQ(allocator.free_bytes(), 96);

        iom_test::arm_failure(ordinal);
        bool threw_bad_alloc = false;
        bool threw_unexpected = false;
        try {
            allocator.alloc(16);
        } catch (const std::bad_alloc&) {
            threw_bad_alloc = true;
        } catch (...) {
            threw_unexpected = true;
        }
        const std::size_t allocations_seen = iom_test::disarm();

        REQUIRE_EQ(allocations_seen, ordinal);
        REQUIRE(threw_bad_alloc);
        CHECK_FALSE(threw_unexpected);
        // No range was removed, split, or duplicated: free bytes and the
        // rest of the geometry are exactly as before the call.
        CHECK_EQ(allocator.free_bytes(), 96);

        // Recovery reuses the exact address the failed call would have
        // returned, exactly once. The split leaves the suffix [16,48) free;
        // offset 16 is not 32-aligned, so the next 16-byte request lands at
        // +32 (16 bytes padding) and the following one would take [48,64).
        void* recovered = allocator.alloc(16);
        REQUIRE_EQ(recovered, buffer.data());
        CHECK_EQ(allocator.free_bytes(), 80);
        void* following = allocator.alloc(16);
        REQUIRE_EQ(following, buffer.data() + 32);
        CHECK_EQ(allocator.free_bytes(), 64);
    }

    // One allocation past every internal fault point succeeds.
    alignas(32) std::array<std::byte, 128> last_buffer{};
    iom::ListAllocator last(last_buffer.data(), last_buffer.size(), 32);
    prepare_split_list_state(last, first, second, third, fourth);
    iom_test::arm_failure(fault_points + 1);
    bool threw = false;
    void* result = nullptr;
    try {
        result = last.alloc(16);
    } catch (const std::bad_alloc&) {
        threw = true;
    } catch (...) {
        threw = true;
    }
    iom_test::disarm();
    REQUIRE_FALSE(threw);
    REQUIRE_EQ(result, last_buffer.data());
}

TEST_CASE("ListAllocator release bookkeeping failure rolls back atomically") {
    // Two free blocks [0,64) [80,48) (112 free bytes) with the 16-byte block
    // at offset 64 outstanding. Freeing it inserts [64,16) and coalesces the
    // whole buffer back to [0,128), so recovery is observable as one 128-byte
    // reusable range at the original base address.
    const auto make_state = [](iom::ListAllocator& allocator) {
        void* a = allocator.alloc(64);
        void* b = allocator.alloc(16);
        allocator.free(a);
        return b;
    };

    alignas(32) std::array<std::byte, 128> probe_buffer{};
    iom::ListAllocator probe(probe_buffer.data(), probe_buffer.size(), 32);
    void* probe_owned = make_state(probe);
    CHECK_EQ(probe_owned, probe_buffer.data() + 64);
    CHECK_EQ(probe.free_bytes(), 112);

    iom_test::arm_counting();
    probe.free(probe_owned);
    const std::size_t fault_points = iom_test::disarm();
    REQUIRE(fault_points > 0);

    for (std::size_t ordinal = 1; ordinal <= fault_points; ++ordinal) {
        alignas(32) std::array<std::byte, 128> buffer{};
        iom::ListAllocator allocator(buffer.data(), buffer.size(), 32);
        void* owned = make_state(allocator);
        CHECK_EQ(allocator.free_bytes(), 112);

        iom_test::arm_failure(ordinal);
        bool threw_bad_alloc = false;
        bool threw_unexpected = false;
        try {
            allocator.free(owned);
        } catch (const std::bad_alloc&) {
            threw_bad_alloc = true;
        } catch (...) {
            threw_unexpected = true;
        }
        const std::size_t allocations_seen = iom_test::disarm();

        REQUIRE_EQ(allocations_seen, ordinal);
        REQUIRE(threw_bad_alloc);
        CHECK_FALSE(threw_unexpected);
        // The release did not happen: free bytes are unchanged and the
        // caller still owns the block.
        CHECK_EQ(allocator.free_bytes(), 112);

        // Recovery releases the block exactly once and coalesces the
        // original range exactly once into the full 128-byte range.
        CHECK_NOTHROW(allocator.free(owned));
        CHECK_EQ(allocator.free_bytes(), 128);
        CHECK_THROWS_AS(allocator.free(owned), std::invalid_argument);
        CHECK_EQ(allocator.free_bytes(), 128);

        // The coalesced range is reusable at its original base address and
        // no capacity was lost.
        void* merged = allocator.alloc(128);
        REQUIRE_EQ(merged, buffer.data());
        CHECK_EQ(allocator.free_bytes(), 0);
        CHECK_THROWS_AS(static_cast<void>(allocator.alloc(1)), std::bad_alloc);
    }

    // One allocation past every internal fault point succeeds.
    alignas(32) std::array<std::byte, 128> last_buffer{};
    iom::ListAllocator last(last_buffer.data(), last_buffer.size(), 32);
    void* owned = make_state(last);
    iom_test::arm_failure(fault_points + 1);
    bool threw = false;
    try {
        last.free(owned);
    } catch (const std::bad_alloc&) {
        threw = true;
    } catch (...) {
        threw = true;
    }
    iom_test::disarm();
    REQUIRE_FALSE(threw);
    CHECK_EQ(last.free_bytes(), 128);
}

TEST_CASE("FixedSizeAllocator release is transactional and allocation-free") {
    alignas(32) std::array<std::byte, 64> buffer{};
    iom::FixedSizeAllocator allocator(buffer.data(), buffer.size(), 16, 8);
    CHECK_EQ(allocator.block_count(), 4);
    CHECK_EQ(allocator.free_count(), 4);

    void* slot0 = allocator.alloc(8);
    void* slot1 = allocator.alloc(8);
    void* slot2 = allocator.alloc(8);
    void* slot3 = allocator.alloc(8);
    CHECK_EQ(slot0, buffer.data());
    CHECK_EQ(slot1, buffer.data() + 16);
    CHECK_EQ(slot2, buffer.data() + 32);
    CHECK_EQ(slot3, buffer.data() + 48);
    CHECK_EQ(allocator.free_count(), 0);

    // The release path appends the index before clearing the slot, so a
    // failed append would leave the slot in use. The free-slot vector is
    // pre-sized to exactly block_count, so the append performs no host
    // bookkeeping allocation at all: a host allocation failure during
    // release cannot lose or duplicate a slot.
    iom_test::arm_counting();
    allocator.free(slot1);
    const std::size_t release_allocations = iom_test::disarm();
    REQUIRE_EQ(release_allocations, 0);
    CHECK_EQ(allocator.free_count(), 1);
    CHECK(allocator.free_count() <= allocator.block_count());

    // A failed release (double free) leaves the slot and the free-slot
    // count unchanged.
    CHECK_THROWS_AS(allocator.free(slot1), std::invalid_argument);
    CHECK_EQ(allocator.free_count(), 1);

    // Reuse returns the released slot exactly once.
    void* reused = allocator.alloc(8);
    REQUIRE_EQ(reused, slot1);
    CHECK_EQ(allocator.free_count(), 0);

    // Invalid releases (foreign or unaligned pointer) leave the free-slot
    // count unchanged.
    alignas(32) std::array<std::byte, 32> foreign{};
    CHECK_THROWS_AS(allocator.free(foreign.data()), std::invalid_argument);
    CHECK_EQ(allocator.free_count(), 0);
    CHECK_THROWS_AS(allocator.free(buffer.data() + 1), std::invalid_argument);
    CHECK_EQ(allocator.free_count(), 0);

    // Release every slot exactly once: no duplicate index, no lost slot.
    allocator.free(slot0);
    allocator.free(reused);
    allocator.free(slot2);
    allocator.free(slot3);
    CHECK_EQ(allocator.free_count(), allocator.block_count());

    // Recycling the full set returns each block address exactly once.
    std::array<void*, 4> recycled{
            allocator.alloc(8), allocator.alloc(8), allocator.alloc(8), allocator.alloc(8)};
    const std::array<void*, 4> addresses{
            buffer.data(), buffer.data() + 16, buffer.data() + 32, buffer.data() + 48};
    for (void* pointer : recycled) {
        bool found = false;
        for (void* address : addresses) {
            found = found || pointer == address;
        }
        REQUIRE(found);
    }
    for (std::size_t i = 0; i < recycled.size(); ++i) {
        for (std::size_t j = i + 1; j < recycled.size(); ++j) {
            REQUIRE_NE(recycled[i], recycled[j]);
        }
    }
    CHECK_EQ(allocator.free_count(), 0);
}
