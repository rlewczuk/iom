# TinyLlama forward layout — Dtype and numerical policy

This subsection defines semantic applicability and shared numerical boundaries
for embedding lookup, linear, RMSNorm, RoPE, cache-row append, SiLU, and SDPA.
It records the current implementation status of those contracts: SiLU is
implemented on the four retained backends, while the other operation statuses
remain as stated in their operation-owned sections. The binary rules in section
8 remain unchanged and are not restated here.

## Semantic applicability

In this table, “payload” means embedding-table and embedding-output data, and
cache means cache-row copy data. “ID” means an embedding index. “Linear” is a
general signed numeric matrix product. “Real arithmetic” comprises RMSNorm,
RoPE, SiLU, and SDPA. `applicable` is a semantic classification, not a claim
that every backend currently stores or computes that leaf.

| Leaf | Embedding payload / cache append | Embedding ID | Linear | RMSNorm / RoPE / SiLU / SDPA | Semantic reason |
| --- | --- | --- | --- | --- | --- |
| `BOOL` | applicable | inapplicable | inapplicable | inapplicable | Payload bits can be copied; boolean is neither an integer token ID contract nor a real-valued arithmetic result. |
| `I2` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic; real-valued normalization, trigonometry, activation, and attention are not defined. |
| `U2` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic; there is no implicit conversion to real arithmetic. |
| `I4` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U4` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `I8` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U8` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `I16` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U16` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `I32` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U32` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `I64` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U64` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `F4_E2M1` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed low-precision real value for numeric operations, never an ID. |
| `F6_E2M3` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed low-precision real value, never an ID. |
| `F6_E3M2` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed low-precision real value, never an ID. |
| `F8_E4M3FN` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed finite real format, never an ID. |
| `F8_E5M2` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed real format, never an ID. |
| `F8_E8M0` | applicable | inapplicable | inapplicable | inapplicable | Payload bits can be preserved, but an unsigned exponent-only encoding is not a general signed numeric result. |
| `F16` | applicable | inapplicable | applicable | applicable | Bit-preserving payload and ordinary signed real arithmetic. |
| `BF16` | applicable | inapplicable | applicable | applicable | Bit-preserving payload and the mandatory TinyLlama real-arithmetic/storage path. |
| `F32` | applicable | inapplicable | applicable | applicable | Bit-preserving payload and ordinary signed real arithmetic. |
| `F64` | applicable | inapplicable | applicable | applicable | Bit-preserving payload and ordinary signed real arithmetic; it is not silently narrowed to FP32. |

Embedding payload and cache append therefore have all 23 applicable leaves.
Embedding IDs have exactly the 12 integer leaves (`I2`, `U2`, `I4`, `U4`,
`I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, and `U64`): negative signed
IDs and every ID greater than or equal to vocabulary size `V` are invalid,
while `BOOL` and every floating leaf are inapplicable. Linear has those 12
integer leaves plus exactly the nine ordinary signed floating leaves
(`F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
`F32`, and `F64`). RMSNorm, RoPE, SiLU, and SDPA have exactly those nine
floating leaves. `F8_E8M0` MUST NOT be promoted to a general signed numeric
result.

Semantic applicability is independent of native storage or arithmetic
capability. For cache append, all 23 leaves are opaque storage payloads; a
backend capability row may reject a recognized leaf only after the common
structural and arithmetic admission checks. The 23 stored leaves on the
standard CPU, CUDA, ROCm, and SYCL paths are capability evidence rather than
permission to narrow this common classification. Each operation contract MUST
enumerate a backend implementation or a justified limitation for every
applicable dtype. BF16 weights, activations, and caches are mandatory for
TinyLlama on CPU, CUDA, ROCm, and SYCL. Integer linear accumulation, overflow
behavior, and any conversion before a kernel belong to the [Linear
projections](linear-projections.md#linear-projections) contract; this table does not infer integer
normalization or silently mean “all floats.” Only `QuantizationFormat::NONE` is
applicable; every other quantization format is rejected. For cache append, only
`QuantizationFormat::NONE` is applicable to opaque payload storage.
## SiLU-specific semantic, scalar, and numerical policy

SiLU has exactly 23 classified input/output leaves. Its semantic partition is
independent of whether a backend can store or compute a particular applicable
format:

| Leaf group | Leaves | SiLU classification | Reason |
| --- | --- | --- | --- |
| Boolean | `BOOL` | unsupported/inapplicable | A boolean is not a signed real-valued transcendental operand or result. |
| Integer | `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64` | unsupported/inapplicable | SiLU does not invent integer arithmetic or implicitly convert an integer operand to real arithmetic. |
| Low-precision floating | `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2` | applicable | These are signed real-valued formats; their named codec controls finite saturation and representable special classes. |
| Standard floating | `F16`, `BF16`, `F32`, `F64` | applicable | These are signed real-valued formats; `F64` remains an FP64 contract and is not silently narrowed. |
| Exponent-only | `F8_E8M0` | unsupported/inapplicable | An unsigned exponent-only encoding is not a general signed SiLU value. |

The operation's independent scalar/raw reference decodes and encodes through
the named-format rules in `src/shared/scalar_binary_codec.hpp`: finite-only
`F4`/`F6` encodings do not acquire invented infinities or NaNs,
`F8_E4M3FN` has NaN but no infinity, and the other applicable leaves retain
their declared IEEE classes, subnormals, signed zero, saturation, and
round-to-nearest, ties-to-even (RNE) encoding behavior. The reference computes
SiLU in the required wide domain and performs exactly one target-format encode
at the output boundary; production code is never its sole oracle.

For finite inputs the reference uses `x / (1 + exp(-x))` with the stable
negative branch `t = exp(x/2); y = ((x*t)*t)/(1+t*t)` (the written
left-associated numerator is normative), and a nonnegative branch
`x/(1+exp(-x))`. The alternative `x*exp(x)/(1+exp(x))` is permitted only
when it preserves every required representable tail. `+infinity` maps to
`+infinity`, `-infinity` to negative zero, signed zeros retain their signs,
and NaN maps to NaN without payload or sign equality. Finite inputs never
produce NaN; a negative finite result that rounds to zero retains its negative
sign. Intermediates are FP32 through `F32` and FP64 for `F64`, with no FTZ or
fast-math tail erasure. The `F32` input `-104` and `F64` input `-746` retain
nonzero negative subnormal tails.

The independent comparison ceilings are one target-format ULP for
`F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, and `BF16`;
four ULP for `F32`; and eight ULP for `F64`. Special-value classes,
signed-zero signs, exact zero behavior, and the two underflow-tail fixtures
are exact checks in addition to those ceilings. The shared SiLU reference and
all named behavior cases are owned by the implemented
`test/backend/backend_conformance_silu.hpp` suite described by the
[SiLU activation](silu-activation.md#silu-activation) contract below.


## Storage, rounding, masking, and nonfinite values

BF16 storage boundaries MUST use round-to-nearest, ties-to-even (RNE) at each
operation boundary. Embedding and append are bit-preserving copies and MUST
NOT re-encode their payloads. For BF16 operands:

- linear and normalization reductions MUST use FP32 or a demonstrably
  sufficiently wide equivalent;
- RMSNorm MUST multiply by its scale in the wide domain before rounding the
  stored result once;
- SiLU MUST evaluate stably in the wide domain and store BF16 once;
- RoPE MUST evaluate its angle, trigonometric values, and rotation
  intermediates in the wide domain and store BF16 once; and
- SDPA MUST compute QK/scale and stable masked softmax in FP32, round
  probability `P` to BF16 with RNE before a native BF16 `P V`, accumulate PV
  in FP32, and round its output to BF16.

The SDPA rule is not a guarantee that attention uses exact-FP32 `P`, and it
MUST NOT impose a BF16 probability-storage boundary on other
operation-supported dtypes. Precision rules for those dtypes belong to their
operation contracts; in particular, `F64` MUST NOT be narrowed to FP32.

The following cases are normative checks on the operation boundaries:

1. Around one, adjacent BF16 values are `1.0` and `1.0 + 2^-7`. Their midpoint
   is `1.0 + 2^-8`; RNE chooses `1.0` because the lower significand is even.
2. The midpoint between `1.0 + 2^-7` and `1.0 + 2*2^-7` is
   `1.0 + 3*2^-8`; RNE chooses the upper value because its significand is
   even.
3. The softmax probability `0.501953125` (`0.5 + 2^-9`) is exactly halfway
   between BF16 `0.5` and `0.50390625`; RNE stores `0.5`.
4. The probability `0.505859375` (`0.50390625 + 2^-9`) is exactly halfway
   between BF16 `0.50390625` and `0.5078125`; RNE stores `0.5078125` because
   the upper significand is even.

A candidate that first rounds through a BF16 intermediate where this policy
requires FP32 is non-conforming. In SDPA, omitting the explicit rounded-BF16
`P` boundary before a BF16-native PV is also non-conforming. Conversely, the
BF16 destination does not permit rounding QK, the softmax sum, or every FP32
accumulation.

Ordinary floating values follow their mathematically defined IEEE classes.
Masking MUST exclude inaccessible values entirely; admission MUST NOT scan
The owning operation contracts fix nonfinite tensor behavior, exact SiLU
infinity and signed-zero rules, finite-format saturation/NaN handling, the
supported RoPE position/trigonometric domain, and per-dtype tolerances. The
[SiLU activation](silu-activation.md#silu-activation) section below fixes the SiLU values and
reference ceilings; TinyLlama model parameters require positive finite
epsilon and finite positive theta.
RMSNorm with `eps=0` may yield NaN for a
zero row, and the final selector rejects every otherwise-valid nonfinite logit.
No NaN payload equality is promised.

Format-specific finite overflow, saturation, signed zero, and representable
special-value classes remain owned by each operation's numerical contract.
Implementations MUST NOT invent NaN or infinity encodings for finite-only
formats and MUST reuse the existing named-format encoding rules. `F64` and
other dtypes MUST NOT be forced through FP32.

## Reference ownership and coverage

Every future detailed numerical reference, fixture provenance, tolerance, and
snapshot/hook rule has exactly one owner:

- 03 embedding owns bit-preservation and index cases;
- 04 linear owns numeric, integer/orientation, and native-GEMM cases;
- 05 RMSNorm owns zero, scaled, reduction, and epsilon cases;
- 06 RoPE owns split-half positions and trigonometric cases;
- 07 append owns preservation and bounds;
- 08 SDPA owns GQA, causal masking, and probability rounding;
- 09 SiLU owns extremes; and
- 14 integration-reference-validation owns independent model and
  intermediate-logit fixtures.

Each owner MUST select fixed, justified tolerances and pin the reference
software or artifact identity before measuring a candidate. A production
implementation MUST NOT serve as its own oracle. This subsection adds no
duplicate fixture or kernel.

Future shared coverage MUST exercise logical `R=1,15,16,17`, non-tile feature
sizes, rank and planes, independent logical/tiled encodings, padding and tail
perturbations, causal future-token exclusion, cached/full equivalence, exact
capacity, and accepted/rejected failures. Exact token goldens are appropriate
only when the independent reference states its margin. Four-backend gates and
existing supported-operation coverage remain required. Unsupported neural
probes migrate only when a backend is actually ported and do not constitute
numerical conformance.
