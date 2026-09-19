#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

#include <cstdint>

namespace {

// Intra-tile byte offset of one native carrier cell. Every logical column is
// a native carrier column; a 64-bit element owns two consecutive columns, and
// those columns are ordinary columns of the same four row-major 16x16 faces.
inline uint32_t native_cell_offset(
        uint32_t row, uint32_t column, uint32_t carrier_bytes) {
    const uint32_t face = ((row % 32) / 16) * 2 + (column % 32) / 16;
    const uint32_t cell = face * 256 + (row % 16) * 16 + (column % 16);
    return cell * carrier_bytes;
}

inline uint32_t native_tile_id(
        uint32_t row, uint32_t column, uint32_t tile_columns) {
    return (row / 32) * tile_columns + column / 32;
}

inline uint32_t matching_l1_address(
        uint32_t scratch, uint32_t noc_address, uint32_t alignment) {
    const uint32_t mask = alignment - 1;
    return (scratch & ~mask) | (noc_address & mask);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_index = 0;
    constexpr auto accessor_args = TensorAccessorArgs<1>();
    const uint32_t vocabulary = get_arg_val<uint32_t>(0);
    const uint32_t run = get_arg_val<uint32_t>(1);
    const uint32_t features = get_arg_val<uint32_t>(2);
    const uint32_t table_base = get_arg_val<uint32_t>(3);
    const uint32_t table_page = get_arg_val<uint32_t>(4);
    const uint32_t index_base = get_arg_val<uint32_t>(5);
    const uint32_t index_page = get_arg_val<uint32_t>(6);
    const uint32_t output_base = get_arg_val<uint32_t>(7);
    const uint32_t output_page = get_arg_val<uint32_t>(8);
    const uint32_t table_carrier = get_arg_val<uint32_t>(9);
    const uint32_t index_carrier = get_arg_val<uint32_t>(10);
    const uint32_t output_carrier = get_arg_val<uint32_t>(11);
    const uint32_t payload_bits = get_arg_val<uint32_t>(12);
    const uint32_t id_bits = get_arg_val<uint32_t>(13);
    const uint32_t id_signed = get_arg_val<uint32_t>(14);
    const uint32_t table_padded_columns = get_arg_val<uint32_t>(15);
    const uint32_t index_padded_columns = get_arg_val<uint32_t>(16);
    const uint32_t output_padded_columns = get_arg_val<uint32_t>(17);
    const uint32_t status_base = get_arg_val<uint32_t>(18);
    const uint32_t status_page = get_arg_val<uint32_t>(19);
    const uint32_t status_offset = get_arg_val<uint32_t>(20);
    // Native carrier columns per logical element: one for every 32-bit-or-narrower
    // encoding, and two consecutive columns (low word first) for a 64-bit payload
    // or index. The padded column words already describe native columns, so the
    // factor is only needed to place a logical element inside them.
    const uint32_t table_factor = get_arg_val<uint32_t>(21);
    const uint32_t index_factor = get_arg_val<uint32_t>(22);

    const auto table_accessor = TensorAccessor(
            accessor_args, table_base, table_page);
    const auto index_accessor = TensorAccessor(
            accessor_args, index_base, index_page);
    const auto output_accessor = TensorAccessor(
            accessor_args, output_base, output_page);
    const auto status_accessor = TensorAccessor(
            accessor_args, status_base, status_page);

    cb_reserve_back(cb_index, 1);
    const uint32_t l1_base = get_write_ptr(cb_index);
    // Keep independently aligned read/write staging windows. The page is
    // native-tile sized, so these windows leave room for every 16-cell
    // fragment and for one complete index face.
    const uint32_t l1_index_scratch = l1_base + 128;
    const uint32_t l1_read_scratch = l1_base + 256;
    const uint32_t l1_write_scratch = l1_base + 512;

    // Preserve an invalid flag set by an earlier leading plane. Match the
    // local address to the actual status DRAM address for both architectures.
    const uint32_t status_noc_address = status_base + status_offset;
    const uint32_t l1_status = matching_l1_address(
            l1_base, status_noc_address, NOC_DRAM_READ_ALIGNMENT_BYTES);
    noc_async_read(
            status_accessor.get_noc_addr(0, status_offset), l1_status,
            sizeof(uint32_t));
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* const status_word =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_status);
    uint32_t invalid = status_word[0] != 0 ? 1u : 0u;

    const uint32_t id_mask = id_bits >= 32
            ? 0xffffffffu
            : ((1u << id_bits) - 1u);
    const uint32_t table_tile_columns = table_padded_columns / 32;
    const uint32_t index_tile_columns = index_padded_columns / 32;
    const uint32_t output_tile_columns = output_padded_columns / 32;
    for (uint32_t output_row = 0; output_row < run; ++output_row) {
        // Index rows are the final logical axis, hence they occupy columns in
        // the native [1, 1, run] plane. A 64-bit ID owns two consecutive
        // native columns, low word first, so the lane is addressed in native
        // column space. The face read starts at the 16-column group holding
        // the lane, which is why a carrier pair never straddles it. Read a
        // complete face so every source address is aligned on both Wormhole
        // and Blackhole.
        const uint32_t index_column = output_row * index_factor;
        const uint32_t index_face_column = (index_column / 16) * 16;
        const uint32_t index_offset = native_cell_offset(
                0, index_face_column, index_carrier);
        const uint32_t index_page_id = native_tile_id(
                0, index_face_column, index_tile_columns);
        const uint32_t index_noc_address = index_base
                + index_page_id * index_page + index_offset;
        const uint32_t index_destination = matching_l1_address(
                l1_index_scratch, index_noc_address,
                NOC_DRAM_READ_ALIGNMENT_BYTES);
        noc_async_read(
                index_accessor.get_noc_addr(index_page_id, index_offset),
                index_destination, 16 * index_carrier);
        noc_async_read_barrier();

        const uint32_t index_lane =
                index_destination + (index_column % 16) * index_carrier;
        uint32_t raw_id = 0;
        if (index_carrier == 1) {
            raw_id = *reinterpret_cast<volatile tt_l1_ptr uint8_t*>(
                    index_lane);
        } else if (index_carrier == 2) {
            raw_id = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                    index_lane);
        } else {
            raw_id = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    index_lane);
        }
        uint32_t id_code = raw_id;
        bool id_invalid = false;
        if (index_factor == 2) {
            // A 64-bit ID is a pair of little-endian carrier words. The low
            // word is the row a valid ID addresses, and the native vocabulary
            // extent is a uint32, so any nonzero high word is out of range:
            // that covers an unsigned value `>= V` and a negative signed
            // value, whose sign bit lives in the high word. Both words are
            // inspected before any row or page arithmetic.
            const uint32_t id_high =
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            index_lane + index_carrier);
            id_invalid = id_high != 0;
        } else {
            id_code &= id_mask;
            id_invalid = id_signed != 0
                    && ((id_code >> (id_bits - 1)) & 1u) != 0;
        }
        if (id_invalid || id_code >= vocabulary) {
            invalid = 1;
            continue;
        }

        uint32_t feature = 0;
        while (feature < features) {
            // Native carrier columns of the current feature: a 64-bit payload
            // element owns two consecutive columns (low word at column 2*f,
            // high word at 2*f+1) inside the same 16x16 face. The run stops at
            // the face boundary, so one read and one write move whole carrier
            // cells without splitting a pair, and a cell run stays contiguous
            // inside its face.
            const uint32_t native_feature = feature * table_factor;
            uint32_t cells = 16 - (native_feature % 16);
            const uint32_t remaining_cells =
                    (features - feature) * table_factor;
            if (cells > remaining_cells) {
                cells = remaining_cells;
            }
            const uint32_t table_offset = native_cell_offset(
                    id_code, native_feature, table_carrier);
            const uint32_t output_offset = native_cell_offset(
                    output_row, native_feature, output_carrier);
            const uint32_t bytes = cells * table_carrier;
            const uint32_t table_page_id = native_tile_id(
                    id_code, native_feature, table_tile_columns);
            const uint32_t output_page_id = native_tile_id(
                    output_row, native_feature, output_tile_columns);
            const uint32_t table_noc_address = table_base
                    + table_page_id * table_page + table_offset;
            const uint32_t output_noc_address = output_base
                    + output_page_id * output_page + output_offset;
            const uint32_t l1_read = matching_l1_address(
                    l1_read_scratch, table_noc_address,
                    NOC_DRAM_READ_ALIGNMENT_BYTES);
            const uint32_t l1_write = matching_l1_address(
                    l1_write_scratch, output_noc_address,
                    NOC_DRAM_WRITE_ALIGNMENT_BYTES);
            noc_async_read(
                    table_accessor.get_noc_addr(table_page_id, table_offset),
                    l1_read, bytes);
            noc_async_read_barrier();

            // Native carrier cells hold one logical value. Clear only the
            // non-logical high bits for sub-32-bit payload carriers; a wide
            // carrier keeps both raw words exactly as stored.
            if (table_carrier == 4 && payload_bits < 32) {
                const uint32_t payload_mask = payload_bits == 32
                        ? 0xffffffffu
                        : ((1u << payload_bits) - 1u);
                volatile tt_l1_ptr uint32_t* const values =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                l1_read);
                for (uint32_t cell = 0; cell < cells; ++cell) {
                    values[cell] &= payload_mask;
                }
            }
            volatile tt_l1_ptr uint8_t* const read_bytes =
                    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_read);
            volatile tt_l1_ptr uint8_t* const write_bytes =
                    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write);
            for (uint32_t byte = 0; byte < bytes; ++byte) {
                write_bytes[byte] = read_bytes[byte];
            }
            noc_async_write(
                    l1_write,
                    output_accessor.get_noc_addr(output_page_id, output_offset),
                    bytes);
            noc_async_write_barrier();
            feature += cells / table_factor;
        }
    }

    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_status) = invalid;
    noc_async_write(
            l1_status, status_accessor.get_noc_addr(0, status_offset),
            sizeof(uint32_t));
    noc_async_write_barrier();
    cb_push_back(cb_index, 1);
    cb_wait_front(cb_index, 1);
    cb_pop_front(cb_index, 1);
}
