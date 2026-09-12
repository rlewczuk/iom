#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

#include "iom/iom.hpp"
#include "../shared/standard_tiled_copy.hpp"

namespace iom::cpu_detail {

inline std::uint64_t load_bits(
        const unsigned char* base, std::size_t bit_offset,
        std::size_t nbits) noexcept {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < nbits; ++i) {
        const std::size_t bit = bit_offset + i;
        value |= static_cast<std::uint64_t>(
                         (base[bit / 8] >> (bit % 8)) & 1)
                 << i;
    }
    return value;
}

inline void store_bits(
        unsigned char* base, std::size_t bit_offset,
        std::size_t nbits, std::uint64_t value) noexcept {
    for (std::size_t i = 0; i < nbits; ++i) {
        const std::size_t bit = bit_offset + i;
        unsigned char& byte = base[bit / 8];
        const unsigned char mask = static_cast<unsigned char>(1u << (bit % 8));
        if ((value >> i) & 1) {
            byte |= mask;
        } else {
            byte &= static_cast<unsigned char>(~mask);
        }
    }
}

inline void copy_value(
        unsigned char* destination, std::size_t destination_bit,
        const unsigned char* source, std::size_t source_bit,
        std::size_t nbits) {
    if (nbits % 8 == 0) {
        auto* destination_bytes = destination + destination_bit / 8;
        const auto* source_bytes = source + source_bit / 8;
        if (destination_bytes != source_bytes) {
            std::memcpy(destination_bytes, source_bytes, nbits / 8);
        }
        return;
    }
    store_bits(destination, destination_bit, nbits,
               load_bits(source, source_bit, nbits));
}

template <std::size_t kElementBytes>
inline void copy_tile_row_byte_aligned(
        unsigned char* destination, const unsigned char* source,
        std::size_t elements) {
    assert(elements <= TensorSpec::TILE);
    const std::size_t bytes = elements * kElementBytes;
    const std::uintptr_t destination_begin =
            reinterpret_cast<std::uintptr_t>(destination);
    const std::uintptr_t source_begin =
            reinterpret_cast<std::uintptr_t>(source);
    if (destination_begin + bytes <= source_begin
            || source_begin + bytes <= destination_begin) {
        std::memcpy(destination, source, bytes);
        return;
    }
    for (std::size_t index = 0; index < elements; ++index) {
        if (destination + index * kElementBytes
                != source + index * kElementBytes) {
            std::memcpy(
                    destination + index * kElementBytes,
                    source + index * kElementBytes, kElementBytes);
        }
    }
}

inline void copy_tile_row_subbyte(
        unsigned char* destination, std::size_t destination_bit,
        const unsigned char* source, std::size_t source_bit,
        std::size_t elements, std::size_t leaf_bits,
        std::array<std::uint32_t, TensorSpec::TILE>& shift_table,
        std::array<std::uint32_t, TensorSpec::TILE>& mask_table) {
    assert(elements <= TensorSpec::TILE);
    const std::size_t total_bits = elements * leaf_bits;
    const bool word_aligned =
            source_bit % (sizeof(std::uint32_t) * 8) == 0
            && destination_bit % (sizeof(std::uint32_t) * 8) == 0;
    if (word_aligned) {
        const std::size_t whole_word_bits =
                total_bits / (sizeof(std::uint32_t) * 8)
                * (sizeof(std::uint32_t) * 8);
        const std::size_t word_elements =
                whole_word_bits % leaf_bits == 0
                ? whole_word_bits / leaf_bits : 0;
        const std::size_t word_count =
                word_elements * leaf_bits / (sizeof(std::uint32_t) * 8);
        std::uint32_t covered_mask = 0;
        for (std::size_t element_index = 0;
             element_index < word_elements; ++element_index) {
            covered_mask |= mask_table[element_index];
            covered_mask |= (1u << shift_table[element_index])
                            & mask_table[element_index];
        }
        for (std::size_t word_index = 0;
             word_index < word_count; ++word_index) {
            std::uint32_t source_word = 0;
            std::memcpy(
                    &source_word,
                    source + source_bit / 8
                            + word_index * sizeof(std::uint32_t),
                    sizeof(std::uint32_t));
            const std::uint32_t destination_word = source_word & covered_mask;
            std::memcpy(
                    destination + destination_bit / 8
                            + word_index * sizeof(std::uint32_t),
                    &destination_word, sizeof(std::uint32_t));
        }
        for (std::size_t index = word_elements;
             index < elements; ++index) {
            copy_value(
                    destination, destination_bit + index * leaf_bits,
                    source, source_bit + index * leaf_bits, leaf_bits);
        }
        return;
    }
    for (std::size_t index = 0; index < elements; ++index) {
        copy_value(
                destination, destination_bit + index * leaf_bits,
                source, source_bit + index * leaf_bits, leaf_bits);
    }
}

inline void copy_tile_row(
        unsigned char* destination, std::size_t destination_bit,
        const unsigned char* source, std::size_t source_bit,
        std::size_t elements, std::size_t bits,
        std::array<std::uint32_t, TensorSpec::TILE>& shift_table,
        std::array<std::uint32_t, TensorSpec::TILE>& mask_table) {
    if (bits % 8 != 0) {
        copy_tile_row_subbyte(
                destination, destination_bit, source, source_bit,
                elements, bits, shift_table, mask_table);
        return;
    }
    switch (bits) {
        case 8:
            copy_tile_row_byte_aligned<1>(
                    destination + destination_bit / 8,
                    source + source_bit / 8, elements);
            return;
        case 16:
            copy_tile_row_byte_aligned<2>(
                    destination + destination_bit / 8,
                    source + source_bit / 8, elements);
            return;
        case 32:
            copy_tile_row_byte_aligned<4>(
                    destination + destination_bit / 8,
                    source + source_bit / 8, elements);
            return;
        case 64:
            copy_tile_row_byte_aligned<8>(
                    destination + destination_bit / 8,
                    source + source_bit / 8, elements);
            return;
        default:
            throw std::invalid_argument(
                    "unsupported byte-aligned CPU leaf width");
    }
}

template <typename Op>
inline void for_each_tile(const TensorView& view, Op&& op) {
    const TensorSpec& spec = view.spec();
    const auto dimensions = spec.shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    const std::size_t tile_rows =
            rows / TensorSpec::TILE + (rows % TensorSpec::TILE != 0);
    const std::size_t tile_columns =
            columns / TensorSpec::TILE + (columns % TensorSpec::TILE != 0);
    const std::size_t bits = detail::leaf_bits(spec.data_type);
    const auto strides = view.plane_strides();
    auto visit = [&](auto&& self, std::size_t leading_index,
                     std::size_t plane) -> void {
        if (leading_index != leading_rank) {
            for (std::size_t index = 0;
                 index < dimensions[leading_index]; ++index) {
                self(self, leading_index + 1,
                     plane + index * strides[leading_index]);
            }
            return;
        }
        const std::size_t tile_bytes =
                TensorSpec::TILE * TensorSpec::TILE * bits / 8;
        for (std::size_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
            const std::size_t first_row = tile_row * TensorSpec::TILE;
            const std::size_t tile_row_byte =
                    detail::standard_plane_slot(spec, plane, first_row, 0)
                    * bits / 8;
            for (std::size_t row_in_tile = 0;
                 row_in_tile < TensorSpec::TILE; ++row_in_tile) {
                const std::size_t row = first_row + row_in_tile;
                if (row >= rows) {
                    break;
                }
                const std::size_t row_byte =
                        tile_row_byte + row_in_tile * TensorSpec::TILE
                                * bits / 8;
                for (std::size_t tile_column = 0;
                     tile_column < tile_columns; ++tile_column) {
                    const std::size_t column = tile_column * TensorSpec::TILE;
                    const std::size_t remaining = columns - column;
                    const std::size_t elements =
                            remaining < TensorSpec::TILE
                            ? remaining : TensorSpec::TILE;
                    op(row, column,
                       row_byte + tile_column * tile_bytes, elements);
                }
            }
        }
    };
    visit(visit, 0, view.plane_offset());
}

template <typename Op>
inline void for_each_tile_lockstep(
        const TensorSpec& source_spec, std::size_t source_offset,
        std::span<const std::size_t> source_strides,
        const TensorSpec& destination_spec, std::size_t destination_offset,
        std::span<const std::size_t> destination_strides, Op&& op) {
    const auto source_dimensions = source_spec.shape.dimensions();
    const std::size_t leading_rank = source_dimensions.size() - 2;
    const std::size_t rows = source_dimensions[leading_rank];
    const std::size_t columns = source_dimensions[leading_rank + 1];
    const std::size_t tile_rows =
            rows / TensorSpec::TILE + (rows % TensorSpec::TILE != 0);
    const std::size_t tile_columns =
            columns / TensorSpec::TILE + (columns % TensorSpec::TILE != 0);
    const std::size_t source_bits = detail::leaf_bits(source_spec.data_type);
    const std::size_t destination_bits =
            detail::leaf_bits(destination_spec.data_type);
    bool identical_layout =
            source_offset == destination_offset
            && source_strides.size() == destination_strides.size();
    for (std::size_t index = 0;
         identical_layout && index < source_strides.size(); ++index) {
        identical_layout = source_strides[index] == destination_strides[index];
    }
    auto visit = [&](auto&& self, std::size_t leading_index,
                     std::size_t source_plane,
                     std::size_t destination_plane) -> void {
        if (leading_index != leading_rank) {
            for (std::size_t index = 0;
                 index < source_dimensions[leading_index]; ++index) {
                self(self, leading_index + 1,
                     source_plane + index * source_strides[leading_index],
                     destination_plane
                             + index * destination_strides[leading_index]);
            }
            return;
        }
        const std::size_t source_tile_bytes =
                TensorSpec::TILE * TensorSpec::TILE * source_bits / 8;
        const std::size_t destination_tile_bytes =
                TensorSpec::TILE * TensorSpec::TILE * destination_bits / 8;
        for (std::size_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
            const std::size_t first_row = tile_row * TensorSpec::TILE;
            const std::size_t source_tile_row_byte =
                    detail::standard_plane_slot(
                            source_spec, source_plane, first_row, 0)
                    * source_bits / 8;
            const std::size_t destination_tile_row_byte =
                    identical_layout
                    ? source_tile_row_byte
                    : detail::standard_plane_slot(
                              destination_spec, destination_plane,
                              first_row, 0)
                              * destination_bits / 8;
            for (std::size_t row_in_tile = 0;
                 row_in_tile < TensorSpec::TILE; ++row_in_tile) {
                const std::size_t row = first_row + row_in_tile;
                if (row >= rows) {
                    break;
                }
                const std::size_t source_row_byte =
                        source_tile_row_byte
                        + row_in_tile * TensorSpec::TILE * source_bits / 8;
                const std::size_t destination_row_byte =
                        destination_tile_row_byte
                        + row_in_tile * TensorSpec::TILE
                                * destination_bits / 8;
                for (std::size_t tile_column = 0;
                     tile_column < tile_columns; ++tile_column) {
                    const std::size_t column = tile_column * TensorSpec::TILE;
                    const std::size_t remaining = columns - column;
                    const std::size_t elements =
                            remaining < TensorSpec::TILE
                            ? remaining : TensorSpec::TILE;
                    op(row, column,
                       source_row_byte + tile_column * source_tile_bytes,
                       destination_row_byte
                               + tile_column * destination_tile_bytes,
                       elements);
                }
            }
        }
    };
    visit(visit, 0, source_offset, destination_offset);
}

}  // namespace iom::cpu_detail
