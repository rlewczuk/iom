#pragma once
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace iom {
    class Allocator {
    public:
        virtual ~Allocator() = default;
        // Every backend's create_tensor requires the returned address to satisfy the engine's 32-byte base-address alignment. Backends must inject an allocator configured to honor this requirement (LinearAllocator(..., alignment = 32) by default). The contract is enforced by iom::detail::allocate_aligned_storage.
        virtual void* alloc(std::size_t sz) = 0;
        virtual void free(void* buffer) = 0;
        virtual void reset() = 0;
    };

    class SingleBufferAllocatorBase : public Allocator {
    public:
        SingleBufferAllocatorBase(void* buffer, std::size_t size, std::size_t alignment);

        [[nodiscard]] std::size_t capacity() const;
        [[nodiscard]] std::size_t align() const;

    protected:
        [[nodiscard]] void* ptr_from_addr(std::uintptr_t address) const;
        [[nodiscard]] std::uintptr_t addr_from_ptr(void* ptr) const;
        [[nodiscard]] bool owns(void* ptr) const;
        [[nodiscard]] std::size_t offset_of(void* ptr) const;
        [[nodiscard]] std::uintptr_t align_up_addr(std::uintptr_t address) const;

        std::uintptr_t raw_begin_ = 0;
        std::uintptr_t raw_end_ = 0;
        std::uintptr_t begin_ = 0;
        std::size_t align_ = 1;
    };

    class LinearAllocator : public SingleBufferAllocatorBase {
    public:
        LinearAllocator(void* buffer, std::size_t size, std::size_t alignment = 32);

        LinearAllocator(const LinearAllocator&) = delete;
        LinearAllocator& operator=(const LinearAllocator&) = delete;
        LinearAllocator(LinearAllocator&&) = delete;
        LinearAllocator& operator=(LinearAllocator&&) = delete;

        void* alloc(std::size_t sz) override;
        void free(void* buffer) override;
        void reset() override;

    private:
        std::uintptr_t current_ = 0;
    };

    class ListAllocator : public SingleBufferAllocatorBase {
    public:
        ListAllocator(void* buffer, std::size_t size, std::size_t alignment = 32);

        ListAllocator(const ListAllocator&) = delete;
        ListAllocator& operator=(const ListAllocator&) = delete;
        ListAllocator(ListAllocator&&) = delete;
        ListAllocator& operator=(ListAllocator&&) = delete;

        void* alloc(std::size_t sz) override;
        void free(void* buffer) override;
        void reset() override;

        [[nodiscard]] std::size_t free_bytes() const;

    private:
        struct Block {
            std::size_t offset;
            std::size_t size;
        };

        void insert_free_block(Block block);
        void coalesce();

        std::vector<Block> free_;
        std::unordered_map<std::uintptr_t, std::size_t> allocated_;
    };

    class FixedSizeAllocator final : public SingleBufferAllocatorBase {
    public:
        FixedSizeAllocator(void* buffer, std::size_t size, std::size_t payload_size);
        FixedSizeAllocator(void* buffer, std::size_t size, std::size_t alignment, std::size_t payload_size);

        FixedSizeAllocator(const FixedSizeAllocator&) = delete;
        FixedSizeAllocator& operator=(const FixedSizeAllocator&) = delete;
        FixedSizeAllocator(FixedSizeAllocator&&) = delete;
        FixedSizeAllocator& operator=(FixedSizeAllocator&&) = delete;

        void* alloc(std::size_t sz) override;
        void free(void* buffer) override;
        void reset() override;

        [[nodiscard]] std::size_t index_of(void* ptr) const;
        [[nodiscard]] void* ptr_from_index(std::size_t index) const;
        [[nodiscard]] std::size_t block_count() const;
        [[nodiscard]] std::size_t free_count() const;
        [[nodiscard]] std::size_t payload_size() const;
        [[nodiscard]] std::size_t stride() const;

    private:
        [[nodiscard]] static std::size_t align_up_payload(std::size_t size, std::size_t alignment);

        std::size_t payload_size_ = 0;
        std::size_t stride_ = 0;
        std::size_t block_count_ = 0;
        std::vector<std::size_t> free_indices_;
        std::vector<bool> in_use_;
    };
}
