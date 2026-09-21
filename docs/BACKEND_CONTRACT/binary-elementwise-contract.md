# 8. Binary elementwise contract

1. `add`, `mul`, `sub`, and `div` each have exactly the common non-virtual
   signature `oid op(const TensorView&, const TensorView&, TensorView&) noexcept`.
   Positive OIDs accept work; synchronous rejection returns one of the six
   negative `OidError` values. There are no options, promotion, public query,
   fallback selector, or signature knobs.
2. Validate in this order before effects, owner registration, sequence
   consumption, or token acceptance: recognized specs/rank/dim/device/owner/
   handle/view/storage and checked arithmetic; matching leaf/quantization;
   right-aligned broadcasting/output shape; mapping snapshot; exact alias rule;
   operation support. Mismatched recognized leaves or quantization are
   `InvalidArgument`; malformed view/device/shape/alias errors are
   `InvalidArgument`; checked arithmetic is `Overflow`; bounded resources,
   pre-acceptance runtime, and other failures map to `ResourceExhausted`,
   `DeviceError`, and `InternalError`.
3. Ranks below two are invalid. `[1,1]` broadcasts over all output axes (two
   scalars yield `[1,1]`); otherwise ranks right-align with conceptual leading
   ones, each axis equal or one, and output exactly the maximum shape.
   Singleton coordinates, including tiled tails, map to zero before tile-slot
   mapping; padding is never read. Transformed leading views preserve offsets
   and strides; broadcasting is internal, not a public zero-stride view.
4. ADD, MUL, and SUB support the 21 NONE numeric leaves listed in section 1;
   DIV supports the nine NONE floating leaves listed there. Matching BOOL,
   F8_E8M0, non-NONE quantization, and integer DIV return `Unsupported` only
   after earlier checks. Operand order is observable: SUB is lhs-rhs and DIV
   is lhs/rhs.
5. Integer MUL and SUB return low `w` bits modulo `2^w` with two's-complement
   signed or ordinary unsigned interpretation, without signed-overflow UB.
   Floating operands decode in their named format, compute in the extended
   mathematical/IEEE domain, and encode once with RNE. Gradual underflow and
   no FTZ/DAZ are required. MUL zero×infinity and NaN are NaN; SUB same-sign
   infinities and NaN are NaN; DIV NaN, 0/0, and infinity/infinity are NaN.
   Format-specific saturation and infinity/NaN classes follow the scalar
   reference; finite outputs are within one ULP.
6. Read/read overlap is valid. Same-owner input/output is allowed only for an
   exact unbroadcasted alias with identical spec, offset, strides, and mapping;
   all other relationships are rejected. Capture both inputs before each store,
   snapshot metadata rather than views, and register all three distinct owners
   with exact aliases deduplicated.
7. Binary work is in-order asynchronous (CPU may complete inline), repeat
   waits are valid, and accepted failures are retained and rethrown on every
   wait. Invalid waits throw `std::invalid_argument`. Pre-submit errors do not
   mutate output, consume a sequence, or register an owner. No caller operand/
   output storage is allocated, replaced, or relocated; staging, conversion,
   workspace, and emulation are internal. Accepted failures are not retried.
