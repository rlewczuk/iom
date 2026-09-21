# 4. View and transfer contract

1. `TensorView` is a non-owning, copyable but non-assignable window into one
   tensor owner. Its plane offset and plane strides count whole logical planes,
   never bytes or elements. `native_handle()` remains the owner handle and
   does not retain ownership.
2. Only leading dimensions may be transformed. `slice`, `select`, `permute`,
   and `reshape_leading` are common metadata operations that preserve the
   owner. Their output must be honored by every backend transfer and copy.
   The final two matrix dimensions MUST NOT be transformed.
3. A transformed view maps its leading logical indices through its plane offset
   and strides. A backend MUST support full, offset, stepped, selected,
   permuted, nested, and contiguous-reshaped views; it MUST leave all planes
   outside the destination view untouched.
4. `copy_from_host` and `copy_to_host` are synchronous and take exactly
   `view.spec().logical_nbytes()`. A short or long source/destination span
   MUST throw `std::invalid_argument` before altering host read buffers. A
   noncanonical BOOL upload MUST throw `std::invalid_argument`.
5. A failed host transfer MUST leave the view's spec, plane offset, plane
   strides, device identity, and native handle unchanged. Destination value
   bytes after a failed transfer are otherwise unspecified.
6. Host transfers do not implicitly wait for queues. The caller must wait for
   outstanding writes before a host read, and for every outstanding read/write
   before a host write or owner destruction. A backend MUST document/implement
   no hidden relaxation of this ordering rule.
