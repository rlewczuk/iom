// SPDX-FileCopyrightText: © 2025 IOM contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Direct native output writer for the TTNN linear projection. Every packed
// BF16 tile leaves the destination register exactly once and is written to the
// caller's own output plane at its native tile position: ordinary mode writes
// `[..., R, O]` and head-planar mode writes the `[R, D]` plane of one head, so
// no host rearrangement, output relocation, or intermediate tensor exists.
//
// Compile-time arguments: the output circular buffer identifier followed by the
// output tensor accessor arguments.
//
// Runtime arguments:
//   0 output plane base address
//   1 output plane native page size
//   2 output plane padded tile columns
//   3 row tile count
//   4 column tile count

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

namespace {

constexpr std::uint32_t kTileCells = 32 * 32;
constexpr std::uint32_t kTileBytes = kTileCells * 2;

}  // namespace

void kernel_main() {
    const std::uint32_t out_base = get_arg_val<std::uint32_t>(0);
    const std::uint32_t out_page = get_arg_val<std::uint32_t>(1);
    const std::uint32_t out_tile_columns = get_arg_val<std::uint32_t>(2);
    const std::uint32_t row_tiles = get_arg_val<std::uint32_t>(3);
    const std::uint32_t column_tiles = get_arg_val<std::uint32_t>(4);

    constexpr std::uint32_t kOutCb = 16;
    constexpr std::uint32_t kCompileArgs = 1;
    constexpr auto out_args = TensorAccessorArgs<kCompileArgs>();
    const auto out_accessor = TensorAccessor(out_args, out_base, out_page);

    for (std::uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (std::uint32_t column_tile = 0; column_tile < column_tiles;
             ++column_tile) {
            cb_wait_front(kOutCb, 1);
            const std::uint32_t source = get_read_ptr(kOutCb);
            noc_async_write_page(
                    row_tile * out_tile_columns + column_tile, out_accessor,
                    source);
            noc_async_write_barrier();
            cb_pop_front(kOutCb, 1);
        }
    }
}
