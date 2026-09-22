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

The integration-reference-validation owner supplies the independent TinyLlama
model oracle.  Its frozen, offline corpus is
[`test/model/synthetic_reference.json`](../../test/model/synthetic_reference.json),
generated only by the opt-in
[`test/reference/generate_model_oracles.py`](../../test/reference/generate_model_oracles.py)
tool in CPython 3.11.16 with the pinned package manifest recorded in the
corpus.  The reference is the eager
`transformers.models.llama.modeling_llama.LlamaForCausalLM` path on CPU BF16
tensors; it is not IOM execution, runtime weight readback, or a copied C++
recurrence.  The generator records the sensitivity recipe, full per-position
decoder-layer/final-normalization/logit snapshots, full versus
`past_key_values` self-consistency, and the artifact/payload digests before
any candidate measurement.

The corpus uses two fixed configurations (`H18/I22/Hq3/Hkv1/D6` and
`H8/I12/Hq4/Hkv2/D2`), positions `R=1,15,16,17`, both teacher-forced
continuations, and the future-token perturbation.  BF16 values are encoded
with RNE; reductions and softmax are FP32, and the softmax probability `P` is
rounded to BF16 before BF16-native `PV`.  Candidate checkpoint and logit
values use the one frozen per-value bound
`abs(actual-ref) <= 0.05 + 0.02*abs(ref)`.  Greedy IDs are certified only
when the independent reference intervals are strictly separated; equal
values use the lowest vocabulary ID for selection but are not margin-stable.


Corpus `expected_result.stop_reason` uses the production
`max_new_tokens` outcome for every captured prefill, including the
zero-token full-prompt and perturbation runs; `initialized_kv_length` is the
captured prefix length.  This records a real session-compatible outcome rather
than inventing a non-production `not_decoded` sentinel.
For the future perturbation, `prefix_ids` is the unperturbed base prefix
through index 15, while the prompt changes the next token; for
teacher-forced cases, `expected_result.token_ids` is exactly the declared
`decode_ids`, and `production_greedy_ids` remains the observed reference
selection trace.


The ordinary reader/materializer and focused corruption/comparator coverage
live in
[`test/model/reference_fixture.hpp`](../../test/model/reference_fixture.hpp)
and
[`test/test_model_reference_fixture.cpp`](../../test/test_model_reference_fixture.cpp).
The reader uses the existing SafeTensors and tokenizer fixtures, so CTest
does not import Python model packages, access a network, or generate the
corpus.  These files own only independent model/intermediate/logit evidence;
operation conformance, tokenizer/chat behavior, and synchronous selector
coverage remain with their existing owners.
The complete-model integration checkpoint is owned separately by
[`test/backend/backend_conformance_model_reference.hpp`](../../test/backend/backend_conformance_model_reference.hpp)
and its CPU consumer
[`test/test_model_integration.cpp`](../../test/test_model_integration.cpp).
It materializes both pinned configurations through the same fixture, waits for
the positive full-prefill producer, then compares the completed final
`residual_after_mlp` (retaining every logical row), final normalization, and
last-position vocabulary logits at `R=1,15,16,17` using the frozen per-value
bound.  The shared function takes only `iom::Device&`; runtime setup remains
owned by each independent backend driver, and the private `RunBanks` seam is
used without publishing a snapshot API.


The focused test consumes all 14 corpus cases on CPU, comparing full
prefill and one-token cached rows for decoder-layer output, final norm, and
logits with the frozen tolerances; it asserts greedy IDs only at certified
reference-margin positions and checks teacher-forced IDs independently.
The committed stdlib-only
[`test/reference/verify_model_oracles_negative.py`](../../test/reference/verify_model_oracles_negative.py)
fixture keeps the verifier's policy-mutation rejection coverage reproducible.

### Official reference export

Task `14-integration-reference-validation` owns the independent official
TinyLlama model and intermediate-logit reference pack. The pinned offline
exporter is
[`test/reference/export_official_model_reference.py`](../../test/reference/export_official_model_reference.py);
its fixed geometry, tokenizer/artifact provenance, BF16 reference arithmetic,
and comparison policy are frozen in
[`test/model/official_reference_policy.json`](../../test/model/official_reference_policy.json).
The policy binds repository model ID
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` to revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, including the fixed config and
SafeTensors identities. That revision SHA is both the caller-supplied artifact
identity and the pack's `artifact_id`; `model_id` remains the repository ID.
Generation uses CPython 3.12.3 with `transformers==4.41.2`,
`tokenizers==0.19.1`, `sentencepiece==0.2.0`, `jinja2==3.1.6`,
`torch==2.3.1+cpu`, `numpy==1.26.4`, and `safetensors==0.4.3`. The shared
tokenizer runtime/package subset is owned by
[`test/reference/tokenizer_reference.py`](../../test/reference/tokenizer_reference.py);
the exporter adds only the neural-generation dependencies.
The exporter takes an explicit caller-supplied distribution, revision identity,
policy, and output path. It never discovers or downloads weights, consults an
IOM output, or runs as ordinary CTest. Its `--verify` path is stdlib-only and
rechecks the explicit model directory, artifact sizes and SHA-256 values,
recorded runtime/package policy, authenticated generation argv, tokenizer IDs
and rendered bytes, finite snapshots, positions, and the case-payload digest
without requiring the verifier host to run the generation interpreter.

For each nonzero raw or chat prompt, the pack records two distinct policies:
`production-greedy` is the production-style EOS/max-new-token/context path and
may enforce an exact token only when the independent reference margin is
certified; `fixed-reference-continuation` teacher-forces the frozen
`[3, 4, 5, 6]` IDs to retain comparable cached prefixes even when greedy
generation stops early. The forced continuation is never a greedy-token
golden. A zero-new-token case records no forward snapshot. Comparison uses
`abs(actual - ref) <= 0.53125 + 0.02*abs(ref)`, with lowest-ID ties and no
backend-specific relaxation. The binary-exact absolute term is the smallest
clean value above the complete cross-backend measured requirement
(`~0.517813`); it was fixed only after cache/session parity and identical-operand
backend diagnostics ruled out a contract violation. Stop bookkeeping considers
EOS before the token limit, the token limit before context exhaustion, and
keeps a committed terminal token in the result without appending it to KV
state.

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
