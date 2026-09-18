// SPDX-FileCopyrightText: © 2025 IOM contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Native BF16 matrix accumulation for the TTNN linear projection. The matrix
// engine consumes the operand tiles produced by the reader: `in0` holds the 32
// selected rows by 32 inner positions and `in1` holds 32 inner positions by 32
// weight rows, and each output tile accumulates every inner tile in the 32-bit
// destination register before one BF16 store.
//
// Precision: `compute_kernel_hw_startup` and `pack_tile` are instantiated with
// the backend's `fp32_dest_acc_en` build mode, so the destination register
// starts at FP32 `+0`, every `matmul_tiles` accumulates in FP32, and the single
// `pack_tile` performs the only BF16 rounding of the whole projection. No
// intermediate product is stored or re-consumed in BF16, and the kernel never
// enables flush-to-zero or fast-math.
//
// Compile-time arguments: the `in0`, `in1`, and output circular buffer
// identifiers.
//
// Runtime arguments: row tile count, column tile count, inner tile count.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const std::uint32_t in0_cb = get_compile_time_arg_val(0);
    const std::uint32_t in1_cb = get_compile_time_arg_val(1);
    const std::uint32_t out_cb = get_compile_time_arg_val(2);

    const std::uint32_t row_tiles = get_arg_val<std::uint32_t>(0);
    const std::uint32_t column_tiles = get_arg_val<std::uint32_t>(1);
    const std::uint32_t inner_tiles = get_arg_val<std::uint32_t>(2);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, out_cb);
    matmul_init(in0_cb, in1_cb);

    for (std::uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (std::uint32_t column_tile = 0; column_tile < column_tiles;
             ++column_tile) {
            // Acquiring the destination register also clears it, so the
            // accumulation starts at FP32 +0 for every output tile.
            tile_regs_acquire();
            for (std::uint32_t inner_tile = 0; inner_tile < inner_tiles;
                 ++inner_tile) {
                cb_wait_front(in0_cb, 1);
                cb_wait_front(in1_cb, 1);
                matmul_tiles(in0_cb, in1_cb, 0, 0, 0);
                cb_pop_front(in0_cb, 1);
                cb_pop_front(in1_cb, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(out_cb, 1);
            // The one and only BF16 store of this projection.
            pack_tile(0, out_cb);
            cb_push_back(out_cb, 1);
            tile_regs_release();
        }
    }
}
