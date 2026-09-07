#include "iom/alloc.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace iom {

    namespace {
        bool is_power_of_two(std::size_t value) {
            return value != 0 && (value & (value - 1)) == 0;
        }

        std::uintptr_t checked_raw_begin(void* buffer, std::size_t size, std::size_t alignment) {
            if (buffer == nullptr && size != 0) {
                throw std::invalid_argument("buffer is null");
            }
            if (!is_power_of_two(alignment)) {
                throw std::invalid_argument("alignment must be a non-zero power of two");
            }
            return reinterpret_cast<std::uintptr_t>(buffer);
        }

        std::uintptr_t checked_raw_end(std::uintptr_t raw_begin, std::size_t size) {
            if (size > std::numeric_limits<std::uintptr_t>::max() - raw_begin) {
                throw std::overflow_error("allocator buffer range overflows address space");
            }
            return raw_begin + size;
        }

    }  // namespace

    SingleBufferAllocatorBase::SingleBufferAllocatorBase(void* buffer, std::size_t size, std::size_t alignment)
            : raw_begin_(checked_raw_begin(buffer, size, alignment)),
              raw_end_(checked_raw_end(raw_begin_, size)),
              align_(alignment) {
        begin_ = align_up_addr(raw_begin_);
        if (begin_ > raw_end_) {
            begin_ = raw_end_;
        }
    }

    std::size_t SingleBufferAllocatorBase::capacity() const {
        return static_cast<std::size_t>(raw_end_ - begin_);
    }

    std::size_t SingleBufferAllocatorBase::align() const {
        return align_;
    }

    void* SingleBufferAllocatorBase::ptr_from_addr(std::uintptr_t address) const {
        return reinterpret_cast<void*>(address);
    }

    std::uintptr_t SingleBufferAllocatorBase::addr_from_ptr(void* ptr) const {
        return reinterpret_cast<std::uintptr_t>(ptr);
    }

    std::size_t SingleBufferAllocatorBase::offset_of(void* ptr) const {
        const auto address = addr_from_ptr(ptr);
        if (address < begin_ || address > raw_end_) {
            throw std::invalid_argument("pointer is outside allocator buffer");
        }
        return static_cast<std::size_t>(address - begin_);
    }

    std::uintptr_t SingleBufferAllocatorBase::align_up_addr(std::uintptr_t address) const {
        const auto mask = static_cast<std::uintptr_t>(align_ - 1);
        if (address > std::numeric_limits<std::uintptr_t>::max() - mask) {
            throw std::overflow_error("allocator address alignment overflows address space");
        }
        return (address + mask) & ~mask;
    }


    LinearAllocator::LinearAllocator(void* buffer, std::size_t size, std::size_t alignment)
            : SingleBufferAllocatorBase(buffer, size, alignment),
              current_(begin_) {
    }

    void* LinearAllocator::alloc(std::size_t sz) {
        const auto aligned = align_up_addr(current_);
        if (aligned > raw_end_ || raw_end_ - aligned < sz) {
            throw std::bad_alloc();
        }

        current_ = sz == 0 ? aligned : aligned + sz;
        return ptr_from_addr(aligned);
    }

    void LinearAllocator::free(void*) {
    }

    void LinearAllocator::reset() {
        current_ = begin_;
    }

    ListAllocator::ListAllocator(void* buffer, std::size_t size, std::size_t alignment)
            : SingleBufferAllocatorBase(buffer, size, alignment) {
        reset();
    }

    void* ListAllocator::alloc(std::size_t sz) {
        if (sz == 0) {
            return nullptr;
        }

        std::size_t best_index = std::numeric_limits<std::size_t>::max();
        std::size_t best_waste = std::numeric_limits<std::size_t>::max();
        std::uintptr_t best_addr = 0;
        std::size_t best_total = 0;

        for (std::size_t i = 0; i < free_.size(); ++i) {
            const auto& block = free_[i];
            const auto block_addr = begin_ + block.offset;
            const auto aligned = align_up_addr(block_addr);
            const auto padding = static_cast<std::size_t>(aligned - block_addr);

            if (padding > block.size) {
                continue;
            }

            const auto available = block.size - padding;
            if (available < sz) {
                continue;
            }

            const auto total = padding + sz;
            const auto waste = block.size - total;
            if (waste < best_waste) {
                best_index = i;
                best_waste = waste;
                best_addr = aligned;
                best_total = total;
            }
        }

        if (best_index == std::numeric_limits<std::size_t>::max()) {
            throw std::bad_alloc();
        }

        const auto chosen = free_[best_index];
        free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(best_index));

        const auto block_addr = begin_ + chosen.offset;
        const auto padding = static_cast<std::size_t>(best_addr - block_addr);
        if (padding != 0) {
            insert_free_block(Block{chosen.offset, padding});
        }

        const auto suffix_offset = chosen.offset + best_total;
        const auto suffix_size = chosen.size - best_total;
        if (suffix_size != 0) {
            insert_free_block(Block{suffix_offset, suffix_size});
        }

        allocated_[best_addr] = sz;
        return ptr_from_addr(best_addr);
    }

    void ListAllocator::free(void* buffer) {
        if (buffer == nullptr) {
            return;
        }

        const auto address = addr_from_ptr(buffer);
        const auto allocated = allocated_.find(address);
        if (allocated == allocated_.end()) {
            throw std::invalid_argument("free of unknown pointer");
        }

        const auto sz = allocated->second;
        allocated_.erase(allocated);

        insert_free_block(Block{static_cast<std::size_t>(address - begin_), sz});
        coalesce();
    }

    void ListAllocator::reset() {
        free_.clear();
        allocated_.clear();

        const auto size = capacity();
        if (size != 0) {
            free_.push_back(Block{0, size});
        }
    }

    std::size_t ListAllocator::free_bytes() const {
        std::size_t result = 0;
        for (const auto& block : free_) {
            result += block.size;
        }
        return result;
    }

    void ListAllocator::insert_free_block(Block block) {
        const auto pos = std::lower_bound(
                free_.begin(),
                free_.end(),
                block.offset,
                [](const Block& lhs, std::size_t offset) {
                    return lhs.offset < offset;
                });

        free_.insert(pos, block);
    }

    void ListAllocator::coalesce() {
        if (free_.empty()) {
            return;
        }

        std::vector<Block> merged;
        merged.reserve(free_.size());
        merged.push_back(free_[0]);

        for (std::size_t i = 1; i < free_.size(); ++i) {
            auto& last = merged.back();
            const auto& current = free_[i];
            if (last.offset + last.size == current.offset) {
                last.size += current.size;
            } else {
                merged.push_back(current);
            }
        }

        free_ = std::move(merged);
    }

    FixedSizeAllocator::FixedSizeAllocator(void* buffer, std::size_t size, std::size_t payload_size)
            : FixedSizeAllocator(buffer, size, 32, payload_size) {
    }

    FixedSizeAllocator::FixedSizeAllocator(
            void* buffer,
            std::size_t size,
            std::size_t alignment,
            std::size_t payload_size)
            : SingleBufferAllocatorBase(buffer, size, alignment),
              payload_size_(payload_size),
              stride_(align_up_payload(payload_size, alignment)) {
        block_count_ = capacity() / stride_;
        free_indices_.reserve(block_count_);
        in_use_.resize(block_count_, false);
        reset();
    }

    void* FixedSizeAllocator::alloc(std::size_t sz) {
        if (sz == 0) {
            return nullptr;
        }
        if (sz > payload_size_ || free_indices_.empty()) {
            throw std::bad_alloc();
        }

        const auto index = free_indices_.back();
        free_indices_.pop_back();

        if (in_use_[index]) {
            throw std::runtime_error("allocator metadata corruption");
        }

        in_use_[index] = true;
        return ptr_from_addr(begin_ + index * stride_);
    }

    void FixedSizeAllocator::free(void* buffer) {
        if (buffer == nullptr) {
            return;
        }

        const auto index = index_of(buffer);
        if (!in_use_[index]) {
            throw std::invalid_argument("double free");
        }

        in_use_[index] = false;
        free_indices_.push_back(index);
    }

    void FixedSizeAllocator::reset() {
        free_indices_.clear();
        for (std::size_t i = 0; i < block_count_; ++i) {
            free_indices_.push_back(block_count_ - 1 - i);
        }
        std::fill(in_use_.begin(), in_use_.end(), false);
    }

    std::size_t FixedSizeAllocator::index_of(void* ptr) const {
        const auto address = addr_from_ptr(ptr);
        const auto end = begin_ + block_count_ * stride_;
        if (address < begin_ || address >= end) {
            throw std::invalid_argument("pointer outside fixed-size allocator");
        }

        const auto delta = static_cast<std::size_t>(address - begin_);
        if (delta % stride_ != 0) {
            throw std::invalid_argument("pointer is not block-aligned");
        }

        return delta / stride_;
    }

    void* FixedSizeAllocator::ptr_from_index(std::size_t index) const {
        if (index >= block_count_) {
            throw std::out_of_range("block index out of range");
        }

        return ptr_from_addr(begin_ + index * stride_);
    }

    std::size_t FixedSizeAllocator::block_count() const {
        return block_count_;
    }

    std::size_t FixedSizeAllocator::free_count() const {
        return free_indices_.size();
    }

    std::size_t FixedSizeAllocator::payload_size() const {
        return payload_size_;
    }

    std::size_t FixedSizeAllocator::stride() const {
        return stride_;
    }

    std::size_t FixedSizeAllocator::align_up_payload(std::size_t size, std::size_t alignment) {
        if (size == 0) {
            throw std::invalid_argument("payload size must be non-zero");
        }
        if (!is_power_of_two(alignment)) {
            throw std::invalid_argument("alignment must be a non-zero power of two");
        }
        const auto mask = alignment - 1;
        if (size > std::numeric_limits<std::size_t>::max() - mask) {
            throw std::overflow_error("payload alignment overflows size range");
        }
        return (size + mask) & ~mask;
    }

}
