// SPDX-FileCopyrightText: © 2025 IOM contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Direct native readers of the TTNN BF16 linear projection. The kernel reads
// the selected source row window `x[..., s:s+R, I]` and the Hugging Face
// oriented `w[O, I]` weight rows straight out of the caller's native DRAM
// planes, re-laying each 32x32 BF16 tile into the native face layout the
// matrix engine consumes: `in0` holds the selected rows by inner columns and
// `in1` holds the weight rows transposed into inner by output columns.
//
// Nothing here allocates, transposes a tensor, or stages through host memory:
// whole native source tiles are read into a scratch circular buffer and the
// requested window rows are gathered into the operand tile. Rows and inner
// positions outside the request are written as +0 so every inner tile
// contributes only real products, and the checkpoint weight keeps its own
// orientation.
//
// Compile-time arguments: the four circular buffer identifiers followed by the
// tensor accessor arguments of `x` and then of `w`.
//
// Runtime arguments:
//   0  x plane base address
//   1  x plane native page size
//   2  x plane padded tile columns
//   3  selected row start `s`
//   4  w plane base address
//   5  w plane native page size
//   6  w plane padded tile columns
//   7  weight row start (`h*D` head-planar, `0` ordinary)
//   8  projected row count `R`
//   9  inner extent `I`
//   10 outer extent `N` (per-plane output features)
//   11 row tile count `ceil(R/32)`
//   12 column tile count `ceil(N/32)`
//   13 inner tile count `ceil(I/32)`

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

namespace {

constexpr std::uint32_t kIn0Cb = 0;
constexpr std::uint32_t kIn1Cb = 1;
constexpr std::uint32_t kScratchCb = 2;
constexpr std::uint32_t kTileCells = 32 * 32;
constexpr std::uint32_t kTileBytes = kTileCells * 2;

// Native cell offset of one logical (row, column) inside one 32x32 BF16 tile:
// tiles are four row-major 16x16 faces.
inline std::uint32_t tile_cell(std::uint32_t row, std::uint32_t column) {
    const std::uint32_t face = ((row % 32) / 16) * 2 + ((column % 32) / 16);
    return face * 256 + (row % 16) * 16 + (column % 16);
}

inline std::uint16_t load_cell(std::uint32_t base, std::uint32_t cell) {
    return *reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(
            base + cell * 2);
}

inline void store_cell(
        std::uint32_t base, std::uint32_t cell, std::uint16_t value) {
    *reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(base + cell * 2) =
            value;
}

// Stages the one or two native source tiles that can hold the contiguous
// source rows `[first_row, first_row + rows)`. A 32-row operand tile spans at
// most two native row blocks, so the scratch buffer needs exactly two pages,
// and the staged page of a source row is `(row / 32) - first_block`.
template <typename Accessor>
inline std::uint32_t stage_source_tiles(
        const Accessor& accessor, std::uint32_t tile_columns,
        std::uint32_t tile_block, std::uint32_t first_row,
        std::uint32_t rows) {
    cb_reserve_back(kScratchCb, 2);
    const std::uint32_t stage = get_write_ptr(kScratchCb);
    const std::uint32_t first_block = first_row / 32;
    const std::uint32_t last_block = (first_row + rows - 1) / 32;
    noc_async_read_page(
            first_block * tile_columns + tile_block, accessor, stage);
    if (last_block != first_block) {
        noc_async_read_page(
                last_block * tile_columns + tile_block, accessor,
                stage + kTileBytes);
    }
    noc_async_read_barrier();
    return stage;
}

inline void release_source_tiles() {
    cb_push_back(kScratchCb, 2);
    cb_wait_front(kScratchCb, 2);
    cb_pop_front(kScratchCb, 2);
}

// Builds one `in0` operand tile: destination row `r` is source row
// `row_start + r` of the selected window and destination column `c` is inner
// position `inner_tile * 32 + c`. Rows past the window and inner positions
// past `I` stay +0.
template <typename Accessor>
inline void build_row_window_tile(
        const Accessor& accessor, std::uint32_t tile_columns,
        std::uint32_t row_start, std::uint32_t rows, std::uint32_t inner,
        std::uint32_t inner_tile) {
    const std::uint32_t inner_position = inner_tile * 32;
    const std::uint32_t rows_in_tile = rows < 32 ? rows : 32;
    const std::uint32_t columns_in_tile =
            inner - inner_position < 32 ? inner - inner_position : 32;
    const std::uint32_t stage = stage_source_tiles(
            accessor, tile_columns, inner_tile, row_start, rows_in_tile);
    const std::uint32_t first_block = row_start / 32;
    cb_reserve_back(kIn0Cb, 1);
    const std::uint32_t destination = get_write_ptr(kIn0Cb);
    for (std::uint32_t row = 0; row < 32; ++row) {
        const bool row_valid = row < rows_in_tile;
        const std::uint32_t source_row = row_start + row;
        const std::uint32_t source =
                stage + ((source_row / 32) - first_block) * kTileBytes;
        for (std::uint32_t column = 0; column < 32; ++column) {
            std::uint16_t value = 0;
            if (row_valid && column < columns_in_tile) {
                value = load_cell(
                        source, tile_cell(source_row % 32, column));
            }
            store_cell(destination, tile_cell(row, column), value);
        }
    }
    cb_push_back(kIn0Cb, 1);
    release_source_tiles();
}

// Builds one `in1` operand tile: destination row `k` is inner position
// `inner_tile * 32 + k` and destination column `n` is weight row
// `weight_row_start + n` of the Hugging Face `[O, I]` orientation. Weight rows
// past this output window and inner positions past `I` stay +0.
template <typename Accessor>
inline void build_weight_tile(
        const Accessor& accessor, std::uint32_t tile_columns,
        std::uint32_t weight_row_start, std::uint32_t rows,
        std::uint32_t inner, std::uint32_t inner_tile) {
    const std::uint32_t inner_position = inner_tile * 32;
    const std::uint32_t rows_in_tile = rows < 32 ? rows : 32;
    const std::uint32_t inner_in_tile =
            inner - inner_position < 32 ? inner - inner_position : 32;
    const std::uint32_t stage = stage_source_tiles(
            accessor, tile_columns, inner_tile, weight_row_start,
            rows_in_tile);
    const std::uint32_t first_block = weight_row_start / 32;
    cb_reserve_back(kIn1Cb, 1);
    const std::uint32_t destination = get_write_ptr(kIn1Cb);
    for (std::uint32_t row = 0; row < 32; ++row) {
        const bool row_valid = row < inner_in_tile;
        for (std::uint32_t column = 0; column < 32; ++column) {
            std::uint16_t value = 0;
            if (row_valid && column < rows_in_tile) {
                const std::uint32_t source_row = weight_row_start + column;
                const std::uint32_t source =
                        stage + ((source_row / 32) - first_block) * kTileBytes;
                value = load_cell(source, tile_cell(source_row % 32, row));
            }
            store_cell(destination, tile_cell(row, column), value);
        }
    }
    cb_push_back(kIn1Cb, 1);
    release_source_tiles();
}

}  // namespace

void kernel_main() {
    const std::uint32_t x_base = get_arg_val<std::uint32_t>(0);
    const std::uint32_t x_page = get_arg_val<std::uint32_t>(1);
    const std::uint32_t x_tile_columns = get_arg_val<std::uint32_t>(2);
    const std::uint32_t row_start = get_arg_val<std::uint32_t>(3);
    const std::uint32_t w_base = get_arg_val<std::uint32_t>(4);
    const std::uint32_t w_page = get_arg_val<std::uint32_t>(5);
    const std::uint32_t w_tile_columns = get_arg_val<std::uint32_t>(6);
    const std::uint32_t weight_row_start = get_arg_val<std::uint32_t>(7);
    const std::uint32_t rows = get_arg_val<std::uint32_t>(8);
    const std::uint32_t inner = get_arg_val<std::uint32_t>(9);
    const std::uint32_t outer = get_arg_val<std::uint32_t>(10);
    const std::uint32_t row_tiles = get_arg_val<std::uint32_t>(11);
    const std::uint32_t column_tiles = get_arg_val<std::uint32_t>(12);
    const std::uint32_t inner_tiles = get_arg_val<std::uint32_t>(13);

    constexpr std::uint32_t kCompileArgs = 4;
    constexpr auto x_args = TensorAccessorArgs<kCompileArgs>();
    constexpr auto w_args =
            TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto x_accessor = TensorAccessor(x_args, x_base, x_page);
    const auto w_accessor = TensorAccessor(w_args, w_base, w_page);

    for (std::uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const std::uint32_t window_start = row_tile * 32;
        const std::uint32_t window_rows =
                rows - window_start < 32 ? rows - window_start : 32;
        for (std::uint32_t column_tile = 0; column_tile < column_tiles;
             ++column_tile) {
            const std::uint32_t weight_window_start = column_tile * 32;
            const std::uint32_t weight_rows =
                    outer - weight_window_start < 32
                    ? outer - weight_window_start
                    : 32;
            for (std::uint32_t inner_tile = 0; inner_tile < inner_tiles;
                 ++inner_tile) {
                build_row_window_tile(
                        x_accessor, x_tile_columns, row_start + window_start,
                        window_rows, inner, inner_tile);
                build_weight_tile(
                        w_accessor, w_tile_columns,
                        weight_row_start + weight_window_start, weight_rows,
                        inner, inner_tile);
            }
        }
    }
}
