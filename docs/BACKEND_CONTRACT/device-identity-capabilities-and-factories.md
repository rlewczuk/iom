# 1. Device identity, capabilities, and factories

1. A `Device` instance owns exactly one backend runtime context and creates
   tensors and queues for that context. It is non-copyable and non-movable.
   The `Device` owner address is therefore stable for all objects it creates.
2. `backend_kind()` MUST return the backend's fixed `BackendKind` value and
   `backend_device()` MUST return the resolved backend-local ordinal.
3. There is no active-backend global, backend switch, or global runtime
   selection service. Backends expose their own factory and objects retain the
   identity of their creating `Device`.
4. `supported_data_types()` MUST return an immutable, nonempty storage span.
   Standard backends retain the shared 23-entry sequence. ADD, MUL, and SUB
   accept with `NONE` exactly the 21 numeric leaves `I2`, `U2`, `I4`, `U4`,
   `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`,
   `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, `F64`.
   DIV accepts only the nine floating leaves
   `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
   `F32`, and `F64`. Matching `BOOL`, `F8_E8M0`, non-`NONE` quantization, and
   integer DIV are `Unsupported` after earlier validation. Required numeric
   leaves MUST use internal staging or emulation rather than SDK narrowing.
5. A capability table describes storage, not an operation query. It MUST reject
   unadvertised storage types before native allocation; operation support is
   reported only by the operation facade.
6. `create_tensor` MUST call/observe `TensorSpec::validate()`. Today that
   means rank at least two, nonzero dimensions, checked size arithmetic, a
   declared leaf type, and `QuantizationFormat::NONE`; every other declared
   quantization format is invalid.

The public `QuantizationFormat::TT_BFP2`, `TT_BFP2A`, `TT_BFP4`,
`TT_BFP4A`, `TT_BFP8`, and `TT_BFP8A` enumerators remain stable public names.
The retained backends reject each recognized format as `Unsupported` after
structural validation; this contract does not rename, remove, renumber, or
reinterpret those values.
7. `create_ops` MUST return an independent queue associated with that exact
   device. A queue created by one device MUST reject foreign views.
8. A backend factory MUST leave no partially usable device/context behind when
   initialization fails. Enabled hardware is never an implicit CPU fallback.
