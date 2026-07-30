#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>

#include "iom/alloc.hpp"

namespace {

bool is_aligned(void* ptr, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
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

    CHECK_THROWS_AS(allocator.alloc(1), std::runtime_error);
}

TEST_CASE("LinearAllocator rejects invalid construction arguments") {
    std::array<std::byte, 32> buffer{};

    CHECK_THROWS_AS(iom::LinearAllocator(nullptr, buffer.size()), std::invalid_argument);
    CHECK_THROWS_AS(iom::LinearAllocator(buffer.data(), buffer.size(), 0), std::invalid_argument);
    CHECK_THROWS_AS(iom::LinearAllocator(buffer.data(), buffer.size(), 3), std::invalid_argument);
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
