# 3. Logical representation and physical layout

1. The public shape has a minimum rank of two. All leading dimensions are
   row-major logical planes. The last two dimensions are a matrix stored in
   fixed 16x16 tiles for the standard representation.
2. Standard physical slot order is: row-major leading plane, tile row, tile
   column, row within tile, column within tile. The final axes are padded up to
   multiples of 16; all leading axes retain their logical extents. Standard
   storage size includes that padding. A newly created standard allocation, and
   every region not targeted by a later view write, MUST remain zero-filled;
   writing a view MUST leave padding and all other owner planes unchanged.
3. The logical host representation is contiguous row-major elements packed
   least-significant-bit first at each `DataType` width. Multi-byte fields are
   little-endian. Transfers MUST move bit patterns, not numerically convert
   values. Boolean host input bytes MUST be canonical `0` or `1`.
4. Upload scatters logical bytes into the backend's physical layout; download
   gathers them back to exactly the logical byte stream. Padding is not part of
   the host payload. A standard-layout backend MUST preserve zero padding; a
   native backend MUST ensure native-only padding cannot leak into logical or
   canonical observed storage.
5. The shared physical oracle re-derives the canonical slot mapping without
   calling production layout code. An accelerator/native conformance driver
   MUST implement `AcceleratorStorageOracle::seed` and `observe` with an
   independent native access path. `observe` returns the complete canonical
   padded owner allocation, not only logical view bytes, so it detects wrong
   tile maps, nonzero padding, and writes to untouched owner planes.
