# 7. Copy contract

1. `copy(source, destination)` is required on every backend. It MUST call the
   equivalent of `DeviceOps::validate_copy` before reserving a sequence or
   causing writes: both views must belong to the queue's exact `Device`, and
   their specs must compare equal (shape, leaf type, quantization).
2. Shape/type/quantization mismatch and reference/foreign-device views MUST
   return the negative `InvalidArgument` OID result. A rejected copy MUST not
   write the destination and MUST not consume a token; the next valid copy
   receives the next unused sequence.
3. A valid copy maps every logical element from the source view to the matching
   logical element of the destination view. It MUST honor both independent
   plane offsets and strides; source and destination may be full, offset,
   stepped, selected, permuted, or nested views.
4. Same-queue copies execute in call order without an intervening host wait.
