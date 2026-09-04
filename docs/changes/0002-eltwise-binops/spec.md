# Elementwise Binary Operations

Implement the elementwise binary-operation contract on CPU, CUDA, ROCm, and
TTNN. CPU is the executable numerical reference. SYCL is outside this change:
the repository has only `BackendKind::SYCL`, not a SYCL factory, storage
implementation, operation queue, build target, or test driver. SYCL binary
operations remain blocked on delivery of
`docs/changes/0001-tensor-view/11-sycl-buildable-scaffold` and
`12-sycl-storage-copy`.

Operation semantics are modeled after the TTNN tensor/tensor operations
selected below, but the public API keeps caller-provided output storage:
`oid operation(const TensorView& a, const TensorView& b, TensorView& out)`.
Methods return a waitable queue token, never a tensor. `out` already owns its
storage. When `out` is the same logical window as `a`, the operation is
in-place. Every operation must honor offsets and strides of safe views over
the leading dimensions; the final two tiled dimensions are not transformable.

This change implements only naive element-by-element evaluation. `a`, `b`,
and `out` must have identical logical shapes. There is no broadcasting,
including singleton expansion, rank promotion, or scalar-tensor form.
Shape mismatch is rejected before submission or output writes.

All four backends expose one identical, explicit per-operation dtype matrix.
That matrix is limited to combinations the pinned TTNN implementation can
execute without host fallback. A backend must not advertise extra
combinations in this change. A tensor may be
storable by its backend but unsupported by a particular operation; that case
throws `std::runtime_error` during capability validation, before submission
or output writes.

`a` and `b` must have the same `DataType` and
`QuantizationFormat::NONE`; mixed-type promotion and format conversion are
outside this change. `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `logical_and`,
`logical_or`, `logical_xor`, and `isclose` require a `BOOL` `out` and write
canonical zero or one. Those operations may alias `out` with `a` only when
the inputs are themselves `BOOL`. Every other operation requires `out` to
have the input dtype. Dtype, quantization, and illegal-alias mismatches are
rejected before submission or output writes.

The common dtype matrix is deliberately minimal:

- `add`, `sub`, `mul`, `div`, `rsub`, `addalpha`, `subalpha`, `remainder`,
  `fmod`, `div_no_nan`, `pow`, `squared_difference`, `ldexp`, `logaddexp`,
  `logaddexp2`, `xlogy`, `hypot`, `atan2`, `maximum`, `minimum`,
  `bias_gelu`, all six comparisons, all three logical operations, and
  `isclose` accept only equal-type `BF16` or `F32` inputs;
- `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_left_shift`, and
  `bitwise_right_shift` accept only equal-type `I32`, `U16`, or `U32` inputs;
- `logical_right_shift` accepts only equal-type `I32` or `U32` inputs;
- `gcd` and `lcm` accept only equal-type `I32` inputs.

Supporting integer arithmetic, integer division/modulo, narrow signed
arithmetic, or integer relational operations is outside this change even
where a backend can execute it.

`out` may alias an input only when it is the identical logical window as
`a`: same owner, shape, plane offset, and plane strides. Otherwise `out` must
have a different owner from both inputs. `a` and `b` may alias each other.
`out == b`, partial overlap, and a distinct output view into either input
owner are rejected before submission, even when the windows are disjoint.

At the pinned TTNN revision, `fmod`, `atan2`, `div_no_nan`, and `isclose`
have no public preallocated-output overload. As a
temporary, explicit exception to the normal no-output-allocation rule, the
TTNN backend may call those public functions, retain their device-resident
result tensor, and copy or convert that result into `out` without host
staging. The returned IOM `oid` completes only after the final output write,
and every temporary remains alive until then. No other operation or backend
may allocate an output. Custom direct-output TTNN kernels are a future change.

The operation inventory derives from the full public TTNN tensor/tensor
elementwise catalog in the installed TTNN checkout at revision
`06994d4afdaa61e89753d73a59d7fd241187f37b`. The IOM API uses short
arithmetic names: TTNN `subtract` becomes `sub`, `multiply` becomes `mul`, and
TTNN `divide` plus parameterized `div` become one `div` family with an
explicit rounding argument. There are no long-name aliases.

`mul` and `div` expose only TTNN's precise mode:
`fast_and_approximate_mode=false`. The portable API has no approximate-mode
parameter. `div` takes a strongly typed rounding argument with the three TTNN
behaviors: true division, truncation toward zero, and floor toward negative
infinity.

- arithmetic and ordered variants: `add`, `sub`, `mul`, `div`, `rsub`,
  `addalpha`, `subalpha`, `remainder`, `fmod`, `div_no_nan`, `pow`,
  `squared_difference`, `ldexp`, `logaddexp`, `logaddexp2`, `xlogy`, `hypot`,
  and `atan2`;
- comparisons and selection: `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `maximum`,
  `minimum`, and `isclose`;
- logical and bitwise: `logical_and`, `logical_or`, `logical_xor`,
  `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_left_shift`,
  `bitwise_right_shift`, and `logical_right_shift`;
- integer-number-theory: `gcd` and `lcm`;
- fused elementwise function: `bias_gelu`.

Scalar-only overloads and separate underscore-suffixed in-place aliases are
outside this change. So are ternary `where`, internal quantization operations,
`outer` is not elementwise, `polyval` is not a tensor/tensor binary operation,
and `prelu` requires a scalar or channel-weight operand rather than the
identical-shape operands required here; all three are outside this change.
`nextafter` is also excluded: the pinned TTNN implementation adds or subtracts
a fixed hardware epsilon and does not implement adjacent-representable
IEEE/PyTorch semantics.
TTNN `floor_div` is covered by `div` with floor rounding, and TTNN
`logical_left_shift` is the same operation as `bitwise_left_shift`; neither
gets a duplicate IOM method.
* for each operation implement comprehensive set of backend conformance tests
  * keep tests properly split, so that single file does not exceed 500 lines
  * make sure tests execution won't take too long, aovid situations where test performs operation on a big tensor and then manually checks every element of this tensor
