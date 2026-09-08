I'm about to implement elementwise add, mul and cmp operations on tensors. 
Please search existing solutions and provide me detailed analysis on operation semantics:
* general operation semantics and its various corner cases (eg. broadcasts, handling views, other transforms fused with elementwise operation etc.)
  * state use cases for individual corner cases (eg. broadcasting), focus on LLM inference as operations will be implemented in an inference engine
* datatypes supported, datatype conversion rules etc.
  * make a matrix of supported datatypes
* ?? do elementwise operations need to work directly on quantized data or can we assume that only matmul/attention needs ??
  * if elementwise operations work on quantized data, provide requirements, sample implementations and usage of such operations
* take into account support for following architectures: CUDA (nvidia ampere and newer), ROCm (RDNA3, RDNA4), SYCL (Intel B60), TTNN
  * generate support matrix for all datatypes from enum `DataType` (see below)
  * those are datatypes meant to be directly supported by underlying hardware
  * if there are datatypes not present in this enum but supported by one of above architectures, please add them to the matrix
* look for any other specifics and corner cases, please analyze and report them as well
* provide references to projects, documentation, materials, articles, papers, blog posts and other resources relevant to this analysis

Here is enum listing datatypes we currently plan to support:

```
    enum class DataType {
        BOOL,

        I2, U2,
        I4, U4,
        I8, U8,
        I16, U16,
        I32, U32,
        I64, U64,

        F4_E2M1,
        F6_E2M3,
        F6_E3M2,

        F8_E4M3FN,
        F8_E5M2,
        F8_E8M0,

        F16,
        BF16,
        F32,
        F64,
    };
```


---

I would implement these operations with a deliberately **boring logical contract** and put most of the complexity into broadcasting/layout/type-lowering and fusion. That is the pattern that scales best across CUDA, ROCm, SYCL, and TTNN.

The most important conclusion from the survey is this:

> **A datatype being accelerated by a matrix engine does not imply that ordinary elementwise add/mul/cmp should operate in that datatype.**

That applies especially to `I2/I4`, FP4, FP6, FP8, and E8M0. Ampere has 4-bit integer MMA, Xe2 XMX has INT2/INT4, RDNA4 has IU4 WMMA and FP8 WMMA, and Blackwell has FP4/FP6 Tensor Cores—but these are not general-purpose scalar/vector ALUs for those formats. ([NVIDIA Docs][1])

## 1. Recommended operation contract

I would make your canonical IR approximately:

```text
broadcast(lhs) -> lhs'
broadcast(rhs) -> rhs'
convert(lhs') -> compute_dtype
convert(rhs') -> compute_dtype

result = add|mul(lhs', rhs')
predicate = cmp(lhs', rhs', comparison_kind)
```

rather than making each backend kernel responsible for arbitrary broadcasting + arbitrary type promotion + quantization + arbitrary layout transformation.

This is quite close to the split between ONNX and compiler IRs:

* ONNX gives users convenient NumPy multidirectional broadcasting. ([onnx.ai][2])
* StableHLO's primitive elementwise operations require matching shapes/types, so broadcasts/conversions are explicit IR operations. ([OpenXLA Project][3])
* PyTorch's `TensorIterator` resolves the higher-level problem into a computational shape and byte strides, then reorders and coalesces dimensions for the kernel. It also performs overlap checking. ([GitHub][4])

For a portable inference runtime, I would adopt essentially this architecture:

```text
User/model op
   ↓
Broadcast inference
   ↓
Dtype / quantization normalization
   ↓
View/indexing normalization
   ↓
Canonical pointwise expression
   ↓
Fusion
   ↓
Backend-specific iterator / tiled kernel
```

That makes backend-specific restrictions such as TensorRT's "same rank before broadcasting" or TTNN's tile-oriented execution fairly easy to accommodate. TensorRT's current `ElementWise` requires equal ranks and then permits each dimension to either match or be `1`; TTNN supports broadcast through dimension 5 and requires higher dimensions to agree. ([NVIDIA Docs][5])

---

# 2. Semantics of `add`, `mul`, and `cmp`

I'm interpreting `cmp` as the six comparison operations `EQ`, `NE`, `LT`, `LE`, `GT`, `GE`, rather than a three-way `<=>` operation.

## Add

For normal numeric types:

```text
out[i] = lhs[i] + rhs[i]
```

after broadcasting.

I recommend:

* same logical compute datatype for both operands;
* output same datatype as compute datatype;
* integer overflow defined as **two's-complement/modulo 2^N wraparound**;
* no implicit saturation;
* floating point following the backend's IEEE-compatible arithmetic;
* no `BOOL + BOOL`.

StableHLO defines Boolean add as logical OR, but ONNX deliberately excludes Boolean from `Add`; TensorRT likewise has separate `AND/OR/XOR` operations. I think the latter API is less surprising for an inference engine. ([OpenXLA Project][3])

So:

```text
add(bool, bool)       -> invalid
logical_or(bool,bool) -> valid
```

is preferable.

If an importer consumes StableHLO, simply lower StableHLO Boolean `add` to `logical_or`.

## Mul

Same structure:

```text
out[i] = lhs[i] * rhs[i]
```

with integer modulo arithmetic.

Again, I would reject Boolean `mul`, despite StableHLO defining it as Boolean AND; import it as `logical_and`. StableHLO's definition is useful as an interoperability reference but need not dictate your public API. ([OpenXLA Project][3])

## Cmp

Recommended:

```cpp
enum class Compare {
    EQ, NE,
    LT, LE,
    GT, GE,
};
```

Result is logically `BOOL`.

For integers:

* signed types use signed comparison;
* unsigned types use unsigned comparison.

For floating point I would use ordinary IEEE quiet comparisons, matching StableHLO:

| operation   | NaN behavior                |
| ----------- | --------------------------- |
| `EQ`        | false if either operand NaN |
| `NE`        | true if either operand NaN  |
| `< <= > >=` | false if either operand NaN |

`+0 == -0` is true.

StableHLO explicitly specifies IEEE `compareQuietEqual`, `compareQuietNotEqual`, and the corresponding relational operations. It has a separate `TOTALORDER` comparison mode when deterministic ordering of NaNs/signed zero is needed. ([OpenXLA Project][3])

I would therefore **not** overload normal `cmp` with total-order behavior. If you later need sorting/top-k primitives, introduce something like:

```cpp
total_order_lt()
```

separately.

---

# 3. Broadcasting semantics

Use full NumPy/ONNX multidirectional broadcasting.

Given two shapes, align dimensions from the right by prepending `1`s:

```text
A: [B, S, H]
B:       [H]
=> [B, S, H]
```

Each aligned dimension must satisfy:

```text
a == b || a == 1 || b == 1
```

and output dimension is the non-1 extent.

ONNX specifies exactly these rules, including rank-0 scalars. ([onnx.ai][2])

### LLM-specific broadcasting cases

| Pattern           | Example                          | Use in inference                                        |
| ----------------- | -------------------------------- | ------------------------------------------------------- |
| scalar            | `[B,S,H] * []`                   | attention scaling, logits temperature, residual scaling |
| hidden vector     | `[B,S,H] * [H]`                  | RMSNorm/LayerNorm gamma, bias                           |
| per-head          | `[B,NH,S,D] * [NH,1,1]`          | head-dependent scales/biases                            |
| batch independent | `[B,NH,S,D] + [1,NH,S,D]`        | shared positional/head data                             |
| sequence mask     | `[B,NH,S,S] + [S,S]`             | attention masks                                         |
| singleton head    | `[B,NH,S,D]` against `[B,1,S,D]` | some MQA/GQA transforms                                 |
| scalar compare    | `logits > threshold`             | filtering/mask creation                                 |

The critical implementation detail is that **broadcast should normally be stride-0 indexing, not data replication**.

For a logical broadcasted dimension:

```text
stride = 0
```

so the same source address is reused.

TTNN's documentation describes broadcast as expansion/duplication at the semantic level, but you should not make physical duplication part of your abstract semantics. ([docs.tenstorrent.com][6])

### Zero-size dimensions

Support them.

For example:

```text
[0, H] + [1, H] -> [0, H]
```

The result has zero elements and the backend should early-exit without issuing an invalid zero-sized kernel launch.

This is valuable for dynamic batching/prefill pipelines, where empty sequence/batch fragments can naturally occur.

---

# 4. Views and non-contiguous tensors

This part is at least as important as broadcasting.

A byte-addressable tensor view should conceptually contain:

```cpp
struct TensorView {
    void* storage;
    int64_t storage_offset;
    Shape shape;
    Strides strides;     // preferably byte strides
    DataType dtype;
};
```

and logical element address is:

```text
base + storage_offset + Σ(index[d] * stride[d])
```

Broadcast dimensions use `stride = 0`.

PyTorch's `TensorIterator` does essentially this internally: broadcasted operands acquire computational strides, dimensions are reordered for efficient traversal, and adjacent dimensions are coalesced. ([GitHub][4])

## Views you should support well

At minimum:

* contiguous tensor;
* transposed/permuted tensor;
* sliced tensor;
* singleton/broadcasted dimensions;
* reshape-compatible views;
* positive arbitrary affine strides.

LLM cases include:

* Q/K/V reshaping and `[B,S,H,D] ↔ [B,H,S,D]` transposition;
* split views into combined QKV projections;
* KV-cache slices;
* sliding-window cache regions;
* views of head subsets;
* RoPE's even/odd or half-vector decomposition.

### Negative strides

I would **not require native negative-stride kernels initially**.

They are common in general NumPy semantics but have very little importance to normal transformer inference. You can reject/materialize them into positive-stride storage.

That reduces alias analysis and vectorization complexity substantially.

---

# 5. Output aliasing and in-place operations

An input broadcasted view can legitimately overlap itself:

```text
shape  = [32, 4096]
stride = [0, 2]
```

A writable output should not.

I recommend this rule:

> Outputs must be non-overlapping writable tensors.

For in-place operations, permit exact input/output aliasing only when every output element has exactly one corresponding writable address.

Reject cases such as:

```text
broadcasted_a += b
```

where multiple logical elements would write to the same physical address.

Also reject arbitrary partial overlap such as:

```text
dst = storage[1:]
src = storage[:-1]
dst += src
```

unless you explicitly implement well-defined overlap handling.

PyTorch's TensorIterator has these exact classes of checks: outputs are checked for internal overlap such as broadcasted views and inputs for unacceptable partial overlap with outputs. ([GitHub][4])

For an inference engine, conservative rejection/materialization is better than clever alias semantics.

---

# 6. Sub-byte tensors make ordinary "stride" semantics insufficient

This becomes important for your enum because of:

```text
I2/U2
I4/U4
F4
F6
```

You cannot adequately describe arbitrary views of those tensors with byte strides alone.

You need something closer to:

```cpp
struct PackedLayout {
    uint32_t bits_per_element;
    uint32_t packing_unit_bits;
    uint32_t values_per_unit;
    BitOrder bit_order;
};

int64_t bit_offset;
```

F6 is especially awkward because two 6-bit values do not align naturally with bytes. NVIDIA's PTX, for example, represents E2M3/E3M2 in padded packed forms (`e2m3x2`/`e3m2x2`) rather than treating six-bit floats as fundamental register types. NVIDIA explicitly says these alternate floating formats are **not fundamental PTX types**. ([NVIDIA Docs][7])

This has several consequences:

1. A slice may begin in the middle of a byte/word.
2. A transposed view may become extremely expensive to address.
3. Vectorized loads require alignment constraints.
4. General strided elementwise operations directly on packed FP4/I4/F6 can be much slower than unpacking.
5. Some transformations are better represented as logical index transforms rather than actual strides.

I recommend distinguishing:

```text
logical tensor layout
storage packing
quantization/block layout
```

rather than putting all of them into `strides`.

---

# 7. Dtype policy

## Strong recommendation: no implicit tensor-to-tensor promotion in the canonical IR

ONNX `Add`/`Mul` require both operands and the output to share one type. StableHLO likewise requires matching baseline types. TensorRT's elementwise operator requires its two inputs to use the same datatype. ([onnx.ai][8])

I would follow that model.

Canonical:

```text
add(F16, BF16)       INVALID
add(cast(F16,F32),
    cast(BF16,F32))  VALID
```

Benefits:

* completely deterministic;
* portable across backends;
* quantization interactions are obvious;
* no enormous promotion table;
* easier kernel dispatch;
* graph optimizer sees conversions and can fuse/remove them.

A high-level API can still provide convenience promotion, but lower it immediately.

## If you expose convenience promotion

My suggested policy would be:

| operands                         | result/compute type                         |
| -------------------------------- | ------------------------------------------- |
| same dtype                       | same dtype                                  |
| integer same signedness          | wider width                                 |
| signed + unsigned                | explicit conversion preferred               |
| `F16 + BF16`                     | `F32`                                       |
| float + narrower float           | wider/selected compute float                |
| integer + floating               | explicit conversion preferred               |
| low-precision FP4/6/8 + anything | explicit `F16/BF16/F32` compute type        |
| tensor + scalar literal          | literal adopts tensor type if representable |

The last rule matters surprisingly often. You do not want:

```text
F16_tensor * 0.5
```

to accidentally become an F64 operation because the host language literal happens to be a C++ `double`.

Treat literals as weakly typed constants.

---

# 8. Logical datatype support for the three operations

This is what I would expose independently of whether a backend executes the datatype directly.

| `DataType`  |  add  |  mul  |     cmp     | Recommended implementation               |
| ----------- | :---: | :---: | :---------: | ---------------------------------------- |
| `BOOL`      |   ✗   |   ✗   |    EQ/NE    | separate logical ops                     |
| `I2`        |   ✓   |   ✓   |      ✓      | widen                                    |
| `U2`        |   ✓   |   ✓   |      ✓      | widen                                    |
| `I4`        |   ✓   |   ✓   |      ✓      | widen                                    |
| `U4`        |   ✓   |   ✓   |      ✓      | widen                                    |
| `I8`        |   ✓   |   ✓   |      ✓      | native/widen backend-dependent           |
| `U8`        |   ✓   |   ✓   |      ✓      | native/widen                             |
| `I16`       |   ✓   |   ✓   |      ✓      | ordinary integer ALU                     |
| `U16`       |   ✓   |   ✓   |      ✓      | ordinary integer ALU                     |
| `I32`       |   ✓   |   ✓   |      ✓      | ordinary integer ALU                     |
| `U32`       |   ✓   |   ✓   |      ✓      | ordinary integer ALU                     |
| `I64`       |   ✓   |   ✓   |      ✓      | ordinary integer ALU / backend sequence  |
| `U64`       |   ✓   |   ✓   |      ✓      | ordinary integer ALU / backend sequence  |
| `F4_E2M1`   |   ✓*  |   ✓*  |      ✓*     | promote → op → round/repack              |
| `F6_E2M3`   |   ✓*  |   ✓*  |      ✓*     | promote → op → round/repack              |
| `F6_E3M2`   |   ✓*  |   ✓*  |      ✓*     | promote → op → round/repack              |
| `F8_E4M3FN` |   ✓*  |   ✓*  |      ✓*     | usually promote to ≥F16                  |
| `F8_E5M2`   |   ✓*  |   ✓*  |      ✓*     | usually promote to ≥F16                  |
| `F8_E8M0`   | **✗** | **✗** | generally ✗ | scale encoding, not ordinary value dtype |
| `F16`       |   ✓   |   ✓   |      ✓      | direct                                   |
| `BF16`      |   ✓   |   ✓   |      ✓      | direct or promote backend-dependent      |
| `F32`       |   ✓   |   ✓   |      ✓      | direct                                   |
| `F64`       |   ✓   |   ✓   |      ✓      | direct where supported                   |

`*` means **logical semantic support**, not that a dedicated packed arithmetic kernel is worthwhile.

I would specifically reconsider the name `F8_E8M0`. NVIDIA calls the format `UE8M0`: it is unsigned, with no mantissa, and is commonly used as a scale representation for microscaled formats such as MXFP8. It is not a normal FP8 activation datatype. ([NVIDIA Docs][7])

Something like:

```cpp
UE8M0_SCALE
```

would be much harder to misuse.

---

# 9. Do elementwise operations need to work on quantized tensors?

## Semantically: yes.

## As dedicated low-bit packed kernels: usually no.

This distinction is important.

StableHLO already defines quantized `add` and `multiply` as:

```text
dequantize -> operation -> quantize
```

and quantized compare as comparison of the dequantized values. ([OpenXLA Project][3])

TFLite has real INT8 kernels/specification entries for `ADD`, `MUL`, `LESS`, `GREATER`, `GREATER_EQUAL`, `LESS_EQUAL`, and `EQUAL`, so quantized elementwise computation is not merely theoretical. ([TensorFlow][9])

And llama.cpp provides a particularly relevant inference-engine example: its quantized-add path dequantizes a quantized source row into float scratch storage, adds an F32 operand, and optionally requantizes the destination. It explicitly restricts the layouts that this implementation accepts. ([GitHub][10])

### What matters in LLM inference

For common weight-only quantization:

```text
W4A16
W8A16
```

the low-bit tensor is normally a matrix weight.

The flow is:

```text
quantized weights
      ↓
GEMM
      ↓
F16/BF16 activation
      ↓
elementwise residual/norm/gate/etc.
```

In that case, **there is essentially no need for I4/FP4 elementwise arithmetic**.

TensorRT's INT4 design illustrates this clearly: it describes INT4 as a weight-compression format and says it is dequantized before computation. ([NVIDIA Docs][11])

For activation quantization:

```text
W8A8
FP8 activations
MXFP8 activations
NVFP4 activations
```

quantized elementwise support becomes more useful, especially for:

```text
quantized GEMM
  -> residual add
  -> quantized GEMM
```

because otherwise you introduce:

```text
DQ -> add -> Q
```

as separate memory-bandwidth-consuming operations.

But your optimizer should normally fuse that sequence rather than rely on a standalone packed arithmetic kernel.

---

# 10. Quantized add/mul/cmp semantics

For affine quantization:

```text
real(q) = scale * (q - zero_point)
```

Let:

```text
a = sA * (qA - zA)
b = sB * (qB - zB)
```

Then quantized add into output quantization `(sO,zO)` is:

```text
qO =
 clamp(
   round_nearest_even(
       (sA*(qA-zA) + sB*(qB-zB)) / sO
   ) + zO
 )
```

For multiplication:

```text
qO =
 clamp(
   round_nearest_even(
       (sA*sB / sO) *
       (qA-zA)*(qB-zB)
   ) + zO
 )
```

Implement that either with:

* widened integer fixed-point arithmetic, or
* FP32/F16/BF16 temporary arithmetic.

For comparison:

```text
cmp(
  sA*(qA-zA),
  sB*(qB-zB)
)
```

### Optimization when quantization is identical

If:

```text
sA == sB
zA == zB
sA > 0
```

then comparison can operate directly on integer codes.

Likewise addition can sometimes simplify substantially.

But do not compare raw codes when quantization metadata differs:

```text
qA == qB
```

does **not** imply equal represented values if their scale/zero point differ.

---

# 11. Quantization metadata must not be encoded solely in `DataType`

You will want something like:

```cpp
struct Quantization {
    DataType storage_type;
    DataType expressed_type;  // usually F16/BF16/F32

    QuantizationKind kind;

    Tensor scale;
    optional<Tensor> zero_point;

    QuantizationGranularity granularity;
    int axis;
    int block_size;

    RoundingMode rounding;
};
```

because these are fundamentally different objects:

```text
I8 tensor used as integer indices
I8 tensor quantizing real activations
I4 weight-only tensor
FP8 + per-tensor scale
MXFP8 + E8M0 block scales
NVFP4 + two-level scales
```

They cannot safely be distinguished by `DataType::I8` or `F4_E2M1` alone.

TensorRT, for example, distinguishes INT8, FP8, FP4 and their Q/DQ semantics and also supports E8M0 as part of its low-precision data-format ecosystem. ([NVIDIA Docs][12])

---

# 12. Quantized views have an additional corner case

Suppose a tensor uses per-block quantization:

```text
block = 32 values
```

and you slice:

```text
tensor[7:...]
```

The data view now starts **inside a quantization block**.

You cannot simply update:

```text
data_ptr += ...
```

and leave scale indexing untouched.

Similarly, transpose can change which logical dimension corresponds to a per-axis quantization axis.

Therefore views over quantized tensors should carry either:

* quantization-coordinate metadata relative to the base storage, or
* restrictions that only block-aligned views remain quantized views;
* otherwise materialize/requantize.

This will matter more as MXFP8/NVFP4-style formats enter LLM inference. NVIDIA describes MXFP8 as block-scaled FP8 with E8M0 scaling, and Blackwell's block-scaled MMA explicitly consumes separate scale factors. ([NVIDIA Developer][13])

---

# 13. Hardware support matrix

The following matrix deliberately distinguishes **ordinary elementwise support** from **matrix support**.

Legend:

* **E** — suitable native scalar/vector elementwise arithmetic path.
* **W** — unpack/promote/widen is required for normal elementwise arithmetic.
* **M** — matrix/dot-product hardware support; **not general elementwise support**.
* **C** — storage/conversion/scale-format support, but not arithmetic ALU.
* **A** — supported by current TTNN public elementwise API.
* `—` — no relevant documented direct support.
* combinations such as `W+M` mean "elementwise by widening, matrix engine has native support."

The CUDA cells are intended as an **implementation guide**, not a claim that every entry maps one-to-one to a single SASS instruction. PTX and CUDA Core capabilities differ from Tensor Core capabilities.

| datatype    | CUDA Ampere | CUDA Hopper | CUDA Blackwell | ROCm RDNA3 | ROCm RDNA4 | SYCL / Xe2 B60 | TTNN current       |
| ----------- | ----------- | ----------- | -------------- | ---------- | ---------- | -------------- | ------------------ |
| `BOOL`      | E           | E           | E              | E          | E          | E              | numeric-mask model |
| `I2`        | W           | W           | W              | W          | W          | **M+W**        | —                  |
| `U2`        | W           | W           | W              | W          | W          | **M+W**        | —                  |
| `I4`        | **M+W**     | W¹          | W¹             | **M+W**    | **M+W**    | **M+W**        | —                  |
| `U4`        | **M+W**     | W¹          | W¹             | **M+W**    | **M+W**    | **M+W**        | —                  |
| `I8`        | M+W         | M+W         | M+W            | E+M        | E+M        | E+M            | —                  |
| `U8`        | M+W         | M+W         | M+W            | E+M        | E+M        | E+M            | limited/cast       |
| `I16`       | E           | E           | E              | E          | E          | E              | —                  |
| `U16`       | E           | E           | E              | E          | E          | E              | **A**              |
| `I32`       | E           | E           | E              | E          | E          | E              | **A**              |
| `U32`       | E           | E           | E              | E          | E          | E              | **A**              |
| `I64`       | E           | E           | E              | E          | E          | E²             | —                  |
| `U64`       | E           | E           | E              | E          | E          | E²             | —                  |
| `F4_E2M1`   | —           | —           | **M+C**        | —          | —          | —              | —³                 |
| `F6_E2M3`   | —           | —           | **M+C**        | —          | —          | —              | —                  |
| `F6_E3M2`   | —           | —           | **M+C**        | —          | —          | —              | —                  |
| `F8_E4M3FN` | —           | **M+C**     | **M+C**        | —          | **M+C**    | —⁴             | —³                 |
| `F8_E5M2`   | —           | **M+C**     | **M+C**        | —          | **M+C**    | —⁴             | —                  |
| `F8_E8M0`   | —           | —           | **C / scale**  | —          | —          | —              | —                  |
| `F16`       | **E+M**     | **E+M**     | **E+M**        | **E+M**    | **E+M**    | **E+M**        | —                  |
| `BF16`      | **W+M**⁵    | **E+M**     | **E+M**        | **E+M**    | **E+M**    | **E+M**        | **A**              |
| `F32`       | E           | E           | E              | E          | E          | E              | **A**              |
| `F64`       | E+M⁶        | E+M         | E+M            | E          | E          | E²             | —                  |

### CUDA notes

**¹ INT4 on newer NVIDIA generations:** Ampere explicitly documents native 4-bit signed/unsigned IMMA. The modern Blackwell `tcgen05` interface instead lists legacy Tensor Core integer input as I8/U8 and new narrow FP4/6/8 inputs. Legacy `.s4/.u4` PTX exists, but reports and generated implementations on Hopper/Blackwell may use emulation rather than a corresponding modern native Tensor Core operation. I therefore would not build any runtime design assumption around native INT4 arithmetic after Ampere. ([NVIDIA Docs][1])

Ampere's matrix engines explicitly support:

* FP16,
* BF16,
* TF32,
* INT8/U8,
* INT4/U4,
* binary 1-bit,
* and FP64. ([NVIDIA Docs][1])

NVIDIA's current hardware table makes the CUDA-Core/Tensor-Core distinction especially explicit: Hopper/Blackwell Tensor Cores support lower precisions that ordinary CUDA cores do not; Blackwell adds FP4/FP6 Tensor Core precision. ([NVIDIA][14])

Blackwell CUTLASS further exposes native narrow `f4/f6/f8` MMA and block-scaled MX/NV formats. ([GitHub][15])

**⁵ BF16 on Ampere:** Ampere has BF16 Tensor Core support, but ordinary PTX BF16 elementwise arithmetic evolved later. For portable Ampere elementwise kernels I would compute in FP32 and convert back rather than making native BF16 scalar ALU semantics part of the contract. TensorRT nevertheless exposes BF16 as a supported precision on Ampere and later. ([NVIDIA Docs][1])

**⁶ FP64:** whether it is fast is obviously product-dependent; "supported" should not be interpreted as economical for LLM inference.

### AMD notes

ROCm's current precision table is unusually useful because it separately reports **HIP type availability, compute-unit support, and matrix-core support**.

For RDNA3/RDNA4:

* standard signed/unsigned 8/16/32/64-bit integers are compute-unit-supported;
* FP16/BF16/F32/F64 are compute-unit-supported;
* FP4/FP6/FP8 are **not** ordinary compute-unit arithmetic formats;
* RDNA4 matrix cores add E4M3/E5M2 FP8;
* FP4/FP6 remain absent from RDNA4 matrix support. ([AMD ROCm][16])

RDNA3 WMMA exposes FP16/BF16, IU8 and IU4 matrix formats. ([AMD GPUOpen][17])

RDNA4's documented dense WMMA rates add FP8/BF8 while retaining IU8/IU4. ([AMD GPUOpen][18])

That is another very strong reason not to equate "FP8 datatype exists in HIP" with "FP8 add instruction exists."

### Intel B60 / Xe2 notes

The Arc Pro B60 is Xe2 and has 160 XMX engines plus 160 Xe vector engines. ([Intel][19])

Xe2 documentation gives the useful split:

* ordinary vector engine: FP32, FP16 and integer/vector processing;
* XMX: FP16/BF16, INT8, INT4, INT2. ([Intel CDRD][20])

Intel's public AI datatype documentation likewise lists FP16, BF16, INT8, INT4 and INT2 for XMX. ([Intel][21])

So **INT2 support on B60 should be interpreted as dot/matrix support, not as reason to write `i2_add` kernels.**

`²` For I64/F64, support exists in the broader Xe vector ISA family, but performance is not remotely comparable to normal ML datatypes. Don't optimize these first.

`⁴` Current oneDNN/software paths can consume additional compressed formats on Xe2, but that does not turn those formats into native Xe2 XMX datatypes. Keep "library accepts it" separate from "underlying arithmetic unit executes it directly."

---

# 14. TTNN deserves a separate interpretation

TTNN is not itself a hardware architecture, so I would avoid interpreting its datatype names as exact equivalents of CUDA/IEEE formats.

Current TTNN `add` and `multiply` publicly support:

```text
BFLOAT16
BFLOAT8_B
BFLOAT4_B
FLOAT32
INT32
UINT32
UINT16
```

with TILE and ROW_MAJOR inputs; row-major inputs are internally converted to tile layout. TTNN also allows float-family mixing and automatically casts I32/U32 to the float operand in some mixed cases. ([docs.tenstorrent.com][22])

Current comparison APIs have broadly similar support; `ne_` includes:

```text
BFLOAT16
BFLOAT8_B
BFLOAT4_B
FLOAT32
INT32
UINT32
UINT16
UINT8 -> converted to UINT16
```

and documents broadcasting through dimension 5. ([docs.tenstorrent.com][6])

Two additional TTNN datatypes should therefore appear in your backend capability model:

```cpp
BFLOAT8_B
BFLOAT4_B
```

but **do not alias them to your `F8_E4M3FN` or `F4_E2M1`.**

`BFLOAT8_B` is explicitly block floating point: groups of 16 values share an exponent, and TTNN documents that small values can consequently flush to zero. ([docs.tenstorrent.com][23])

So:

```text
F4_E2M1        != TTNN BFLOAT4_B
F8_E4M3FN      != TTNN BFLOAT8_B
```

They need separate format identities.

That is `³` in the hardware matrix.

---

# 15. Additional datatype/precision concepts I would add

Not necessarily all to `DataType`.

## `TF32`

Both NVIDIA Ampere Tensor Cores and Intel's matrix ecosystem use TF32-like compute modes.

But I would **not make TF32 a normal tensor storage datatype**.

It is better modeled as:

```cpp
enum class MatmulComputeMode {
    IEEE_F32,
    TF32,
    ...
};
```

On Ampere, TF32 is explicitly a Tensor Core format for FP32 computation. ([NVIDIA Docs][1])

## Binary / 1-bit MMA

Ampere also supports binary MMA. ([NVIDIA Docs][1])

Again, I would not add this merely because matrix hardware has it unless you have a model format that actually stores binary tensors.

## TTNN block floating formats

These should probably be backend-specific/extensible storage formats:

```text
BFLOAT8_B
BFLOAT4_B
```

rather than squeezed into the generic IEEE-like float enum.

## Quantization scale formats

I would distinguish:

```text
UE8M0
```

from ordinary floating storage.

---

# 16. Fusions you should design for from day one

Elementwise kernels in LLMs are so memory-bound that a good fusion interface matters more than a highly tuned standalone add.

## Residual add + normalization

Transformer:

```text
x = x + attention(x)
x = RMSNorm(x)
```

or:

```text
x = x + mlp(x)
```

Ideal:

```text
residual_add_rmsnorm
```

One read/write of the activation rather than:

```text
read x
read residual
write temp
read temp
write normed
```

## Gated MLP

SwiGLU:

```text
out = SiLU(gate) * up
```

Elementwise `mul` should be fusible with the activation producer.

## Bias / scaling

```text
projection + bias
attention_scores * scale
logits / temperature
```

Broadcasted scalar/vector loads are natural fusion inputs.

## Compare + select

Avoid:

```text
mask = x < threshold
y = where(mask, a, b)
```

with a materialized Boolean tensor.

Prefer:

```text
where(x < threshold, a, b)
```

as one fused expression.

For attention masking this can save a large intermediate.

## Cast + operation

```text
cast(fp8 -> bf16)
add
cast(bf16 -> fp8)
```

should generally collapse into one fused implementation where useful.

## Dequantize + operation + requantize

Same principle:

```text
DQ -> residual add -> Q
```

should be a fusion pattern, not necessarily three kernels.

NVIDIA TensorRT's fused-attention machinery explicitly permits pointwise elementwise, scale, unary and activation operations in attention fusion regions, which is a useful precedent for treating pointwise expressions as fusion material. ([NVIDIA Docs][24])

## Transform + elementwise

Good candidates include:

```text
transpose-view -> add
slice-view     -> mul
RoPE indexing  -> mul/add
```

Do not materialize a transpose simply because the elementwise kernel expects contiguous memory.

---

# 17. Kernel implementation architecture

I would implement three paths per conventional GPU backend.

## A. Contiguous/vectorized fast path

Condition:

```text
all tensors dense
inner dimension contiguous
alignment sufficient
no complicated broadcast
```

Then vectorize:

```text
F32 -> float4/etc.
F16 -> half2 / wider vector load
BF16 -> bfloat162 / native vector where applicable
I8  -> packed load, widened vector arithmetic
```

This will handle the majority of transformer activation operations.

## B. Specialized broadcast path

Especially:

```text
tensor + scalar
tensor + [H]
tensor * [H]
```

These deserve dedicated low-overhead kernels because they are extremely common.

For `[B,S,H] + [H]`, map the innermost contiguous `H` dimension directly so the broadcast operand remains cache-friendly.

## C. Generic strided iterator

Use:

```text
logical linear index
 -> multidimensional coordinates
 -> address using stride arrays
```

but first apply dimension:

1. reordering;
2. coalescing;
3. 32-bit indexing optimization when safe.

This is essentially the successful TensorIterator model. PyTorch distinguishes user/logical shape from its reordered/coalesced computational shape for exactly this purpose. ([GitHub][4])

Do not force every fast kernel through general N-dimensional address arithmetic.

---

# 18. 32-bit versus 64-bit indexing

Tensor sizes and byte offsets in the API should be 64-bit.

But where you can prove:

```text
max address offset < 2^31/2^32
```

use 32-bit loop/index arithmetic in the device kernel.

The important part is proving it based on **strides and storage offsets**, not just `numel`.

A view can have:

```text
small numel
large stride
```

and still require 64-bit address arithmetic.

For very large LLM tensors this can become relevant, particularly KV cache storage.

---

# 19. Output layout semantics

Do not make physical output layout part of the mathematical meaning of `add`.

For:

```text
C = A + B
```

and:

```text
C = B + A
```

the logical result must of course be identical, and ideally layout selection should not accidentally depend on operand ordering.

This is not just theoretical: a recent PyTorch TensorIterator issue demonstrates that incompatible operand layouts can make output stride selection depend on which operand comes first despite addition being commutative. ([GitHub][25])

I would define:

* logical op semantics: independent of layout;
* if `out=` is given: preserve its compatible layout;
* otherwise: backend chooses a canonical/preferred output layout;
* layout optimizer may change the physical choice before lowering.

For TTNN that likely means favoring tiled/sharded output because elementwise execution is native in tile layout. TTNN explicitly recommends tile format and matching L1-sharded specs for performance. ([docs.tenstorrent.com][6])

---

# 20. Accuracy / determinism corner cases

You should specify these instead of inheriting arbitrary backend behavior.

## Floating arithmetic

Decide whether you promise:

1. exact correctly rounded storage-type arithmetic, or
2. numerically equivalent arithmetic through a wider compute type.

I recommend #2 for low-precision formats.

For example:

```text
BF16 input
F32 compute
RNE BF16 output
```

is a perfectly reasonable BF16 elementwise implementation.

TTNN itself exposes an interesting example: its BF16 `add` has a default fast approximate FPU path with max error documented as 1 ULP, versus an RNE SFPU path. ([docs.tenstorrent.com][22])

That means bit-identical cross-backend results are unrealistic unless you explicitly introduce a "strict" numerical mode.

## FP contraction

For a fused expression:

```text
a * b + c
```

your fusion optimizer may turn separate `mul` + `add` into an FMA.

That changes rounding.

Decide whether fusion is allowed to do this.

For an inference engine, I would allow it in normal mode.

Potential API:

```cpp
MathMode::FAST
MathMode::STRICT
```

with strict mode primarily for tests/reference comparisons.

## Denormals

FP16/BF16/low-float denormal handling may differ between architectures.

Do not promise identical subnormal treatment unless needed.

## NaN payloads

Promise comparison behavior, but not NaN payload propagation/bit patterns.

---

# 21. Integer overflow

Do **not** inherit host C++ signed-overflow semantics.

Explicitly define:

```text
I8/I16/I32/I64 add/mul:
result = low N bits of two's-complement arithmetic
```

That makes the result reproducible and maps cleanly to hardware.

Saturation, if needed, should be separate:

```text
saturating_add
saturating_mul
```

It is especially important not to make quantized integers use saturating arithmetic implicitly just because requantization ultimately clamps. Integer arithmetic and quantized-real arithmetic are different semantics.

---

# 22. Comparison result representation

Logically:

```text
cmp(...) -> BOOL tensor
```

Physically, do not over-specify it.

Possibilities:

* byte-per-bool external representation;
* predicate register while fused;
* integer mask on a backend that lacks convenient Boolean tensors;
* packed bitmask for a specialized downstream op.

TensorRT, for example, defines its BOOL as an 8-bit representation where `0=false`, `1=true`; StableHLO models comparison results as `i1`. ([NVIDIA Docs][26])

TTNN comparison interfaces make it especially useful to keep logical Boolean separate from physical device representation.

My choice would be:

```text
IR result = BOOL
storage lowering = backend-specific
```

and strongly encourage fusing masks into `where`/attention/etc. rather than materializing them.

---

# 23. What I would actually implement in priority order

For an LLM inference engine, I would not spend equal effort on every cell of your enum.

### Tier 1

Highly optimized:

```text
F16
BF16
F32

add
mul
cmp

same-shape
scalar broadcast
[H] broadcast
generic NumPy broadcast
common transposed/strided input views
```

These cover almost all ordinary transformer pointwise traffic.

### Tier 2

Good functional support:

```text
I8/U8
I16/U16
I32/U32
I64/U64
```

with integer widening/vectorized kernels where appropriate.

`I32/I64` are especially useful for shape/index/token-related tensors, although those often don't dominate GPU pointwise time.

### Tier 3

Functional fallback:

```text
I2/U2
I4/U4
FP4
FP6
FP8
```

using:

```text
unpack/convert
   ↓
F16/BF16/F32 or I32 arithmetic
   ↓
pack/convert
```

and rely on fusion to make this efficient when actually encountered.

### Special

```text
E8M0
```

should **not** be exposed to generic add/mul.

---

# 24. A useful backend capability abstraction

Instead of a single Boolean:

```cpp
supports(DataType)
```

I recommend something richer:

```cpp
struct TypeCapabilities {
    bool storage;
    bool load_store;

    bool elementwise_add;
    bool elementwise_mul;
    bool elementwise_cmp;

    bool convert_from;
    bool convert_to;

    bool matrix_input;
    bool matrix_accumulator;

    bool atomic;

    bool packed;
    bool block_scaled;
};
```

and ideally:

```cpp
enum class ExecutionSupport {
    Unsupported,

    Emulated,
    Widened,

    NativeVector,
    NativeMatrix,

    LibraryOnly,
};
```

Because the question:

```text
"Does RDNA4 support FP8?"
```

otherwise has three simultaneously valid answers:

```text
HIP datatype:          yes
ordinary CU FP8 ALU:   no
matrix-core FP8:       yes
```

ROCm's own documentation effectively makes this same distinction. ([AMD ROCm][16])

---

# 25. A suggested semantic specification

If I were writing the contract today, it would look approximately like this:

```text
ADD(lhs, rhs)
MUL(lhs, rhs)

Preconditions:
- lhs/rhs are numeric, non-E8M0 tensors.
- canonical IR operands have identical element datatype.
- shapes are broadcast-compatible.
- destination shape is the multidirectional broadcast shape.

Semantics:
- evaluate elementwise over the broadcast shape.
- signed/unsigned integer arithmetic is modulo 2^bit_width.
- floating-point arithmetic follows the selected math mode.
- output element type equals operand compute type.

CMP(lhs, rhs, predicate)

Preconditions:
- operands have identical canonical element datatype.
- shapes are broadcast-compatible.
- predicate ∈ {EQ, NE, LT, LE, GT, GE}.

Semantics:
- integer comparison follows datatype signedness.
- floating comparison follows IEEE quiet comparison.
- output logical datatype is BOOL.
```

Then separately:

```text
QUANTIZED ADD/MUL/CMP
```

is defined on **expressed values**, with Q/DQ semantics, while an optimizer is free to implement mathematically equivalent widened integer arithmetic.

That mirrors StableHLO's clean semantic model. ([OpenXLA Project][3])

---

# 26. Test cases I would consider mandatory

A serious pointwise test suite should include, beyond ordinary random arrays:

| category                      | examples                              |
| ----------------------------- | ------------------------------------- |
| scalar                        | `[] + [4,8]`                          |
| rank mismatch                 | `[3,1,5] + [5]`                       |
| bidirectional broadcast       | `[2,1,8] + [1,4,1]`                   |
| zero extent                   | `[0,4] + [1,4]`                       |
| transposed input              | `transpose(A) + B`                    |
| sliced input                  | `A[:, ::2] * B`                       |
| broadcast stride 0            | `[1,H] -> [B,S,H]`                    |
| output alias                  | exact in-place allowed                |
| invalid alias                 | broadcasted output rejected           |
| partial overlap               | rejected/materialized                 |
| signed integer extremes       | min/max overflow                      |
| unsigned overflow             | max + 1                               |
| float signed zero             | `+0 == -0`                            |
| NaN EQ                        | false                                 |
| NaN NE                        | true                                  |
| NaN LT/GT                     | false                                 |
| infinities                    | all comparisons                       |
| scalar literal                | ensure no accidental F64 promotion    |
| packed odd offset             | I4 subview beginning on second nibble |
| quantization differing scales | cmp must compare represented values   |
| quantized saturation          | output clamp                          |
| per-axis quantization         | broadcast involving quantization axis |
| block quantization            | unaligned slice                       |
| enormous stride               | force 64-bit indexing                 |
| empty tensor                  | no invalid kernel launch              |

For FP4/FP6/FP8 also test every bit pattern, because the state space is tiny and exhaustive encoding/conversion tests are cheap.

---

# 27. Existing systems worth using as design references

The most useful references I found are these:

* [StableHLO operation specification](https://openxla.org/stablehlo/spec?utm_source=chatgpt.com) — probably the best clean reference for mathematical add/mul/cmp and quantized semantics. ([OpenXLA Project][3])
* [ONNX multidirectional broadcasting specification](https://onnx.ai/onnx/repo-docs/Broadcasting.html?utm_source=chatgpt.com) — use this almost verbatim for public broadcasting semantics. ([onnx.ai][2])
* [ONNX Add operator](https://onnx.ai/onnx/operators/onnx__Add.html?utm_source=chatgpt.com) and [ONNX Mul operator](https://onnx.ai/onnx/operators/onnx__Mul.html?utm_source=chatgpt.com) — useful same-type numeric contract. ([onnx.ai][27])
* [PyTorch TensorIterator source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.h?utm_source=chatgpt.com) — the best practical reference I found for generic broadcasting/striding/overlap/coalescing infrastructure. ([GitHub][4])
* [TensorRT ElementWise operator](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/ElementWise.html?utm_source=chatgpt.com) — useful reality check for what a production inference stack actually exposes. ([NVIDIA Docs][5])
* [TensorRT quantization capabilities](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html?utm_source=chatgpt.com) — useful for keeping storage type and Q/DQ semantics separate. ([NVIDIA Docs][12])
* [llama.cpp / ggml CPU elementwise implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp?utm_source=chatgpt.com) — particularly relevant example of quantized tensor + F32 add implemented through temporary dequantization. ([GitHub][10])
* [TFLite INT8 quantization specification](https://www.tensorflow.org/lite/performance/quantization_spec?utm_source=chatgpt.com) — concrete quantized ADD/MUL/comparison requirements. ([TensorFlow][9])
* [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/?utm_source=chatgpt.com) — essential for distinguishing fundamental datatypes from alternate FP4/6/8/E8M0 formats. ([NVIDIA Docs][7])
* [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/?utm_source=chatgpt.com) — precise matrix-engine datatype table including I4 and B1. ([NVIDIA Docs][1])
* [CUTLASS Blackwell functionality](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md?utm_source=chatgpt.com) — best concrete description of Blackwell FP4/FP6/FP8 and block-scaled MMA. ([GitHub][15])
* [ROCm datatype and precision support](https://rocm.docs.amd.com/en/docs-7.14.0/reference/precision-support.html?utm_source=chatgpt.com) — excellent architecture table distinguishing compute units and matrix cores. ([AMD ROCm][16])
* [AMD RDNA3 WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/?utm_source=chatgpt.com) and [AMD RDNA4 AI overview](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/?utm_source=chatgpt.com) — concrete IU4/IU8 and FP8 capabilities. ([AMD GPUOpen][17])
* [Intel Arc Pro B60 specifications](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html?utm_source=chatgpt.com) and Intel's Xe2 architecture material — confirms the B60/Xe2/XMX target and the INT2/INT4/INT8/FP16/BF16 split. ([Intel][19])
* [TTNN add documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.add.html?utm_source=chatgpt.com), [TTNN multiply documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.multiply.html?utm_source=chatgpt.com), and [TTNN Tensor datatype documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html?utm_source=chatgpt.com) — especially important because TTNN's `BFLOAT*_B` formats are not your ordinary FP4/FP8 formats. ([docs.tenstorrent.com][22])

---

# 28. Bottom-line implementation choices

For your engine I would make the following choices:

1. **Use ONNX/NumPy multidirectional broadcasting at the model/API layer.**
2. **Canonicalize broadcasting explicitly before backend lowering.**
3. **Do not implicitly promote two tensor operands in canonical IR.** Insert explicit casts.
4. **Support arbitrary positive-strided input views**, including stride-0 broadcast.
5. **Require non-overlapping outputs** and conservatively handle aliasing.
6. **Define integer overflow as modulo 2^N.**
7. **Use IEEE quiet comparisons and Boolean results.**
8. **Keep BOOL out of numeric add/mul.**
9. **Treat FP4/FP6/FP8 elementwise operations as logical operations implemented by promotion unless a backend provides a compelling optimized fused path.**
10. **Do not treat E8M0 as an ordinary arithmetic tensor datatype.**
11. **Make quantization metadata orthogonal to `DataType`.**
12. **Support quantized elementwise semantics**, but prioritize `DQ → pointwise → Q` fusion rather than standalone packed kernels.
13. **Optimize F16/BF16/F32 first.**
14. **Build scalar and `[H]` broadcasting fast paths**, because these occur continually in transformers.
15. **Make compare+select, residual+norm, activation+mul, casts, and Q/DQ first-class fusion patterns.**
16. **Separate backend capability into storage / elementwise / matrix / conversion support.** A single `supports(FP8)` bit is not expressive enough.
17. Add backend-specific identities for **TTNN `BFLOAT8_B`/`BFLOAT4_B`**, and model **TF32 as a compute mode rather than a normal tensor datatype**.

The architectural point I would bake into the design now is that your `DataType` enum describes a **logical/storage representation**, while the actual execution type is a separate property. In practice, an operation such as

```text
F4_E2M1 tensor
   -> elementwise add
   -> F4_E2M1 tensor
```

may execute as

```text
packed F4 load
   ↓
F16
   ↓
F16 add
   ↓
RNE / scale / clamp
   ↓
packed F4 store
```

while the exact same `F4_E2M1` datatype participates directly in a Blackwell Tensor Core GEMM. Designing those as two different notions of "datatype support" will prevent a large amount of backend-specific special-casing later.

[1]: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html?source=post_page-----23cd1a0d4e39--------------------------------&utm_source=chatgpt.com "1. NVIDIA Ampere GPU Architecture Tuning Guide — Ampere Tuning Guide 13.3 documentation"
[2]: https://onnx.ai/onnx/repo-docs/Broadcasting.html?utm_source=chatgpt.com "Broadcasting in ONNX - ONNX 1.24.0 documentation"
[3]: https://openxla.org/stablehlo/spec?authuser=002&hl=en&utm_source=chatgpt.com "StableHLO Specification  |  OpenXLA Project"
[4]: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.h?utm_source=chatgpt.com "pytorch/aten/src/ATen/TensorIterator.h at main · pytorch/pytorch · GitHub"
[5]: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/ElementWise.html?utm_source=chatgpt.com "ElementWise - NVIDIA TensorRT Operators Documentation 11.2.1"
[6]: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.ne_.html?utm_source=chatgpt.com "ttnn.ne_ — TT-NN&trade; documentation"
[7]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma&utm_source=chatgpt.com "1. Introduction — PTX ISA 9.3 documentation"
[8]: https://onnx.ai/onnx/operators/onnx__Mul.html "https://onnx.ai/onnx/operators/onnx__Mul.html"
[9]: https://www.tensorflow.org/lite/performance/quantization_spec?hl=zh-cn&utm_source=chatgpt.com "TensorFlow Lite 8 位量化规范"
[10]: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp?utm_source=chatgpt.com "llama.cpp/ggml/src/ggml-cpu/ops.cpp at master · ggml-org/llama.cpp · GitHub"
[11]: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/architecture/capabilities.html?utm_source=chatgpt.com "TensorRT’s Capabilities — NVIDIA TensorRT"
[12]: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html?utm_source=chatgpt.com "TensorRT’s Capabilities — NVIDIA TensorRT"
[13]: https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/?utm_source=chatgpt.com "Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training | NVIDIA Technical Blog"
[14]: https://www.nvidia.com/en-au/data-center/tensor-cores/?utm_source=chatgpt.com "Tensor Cores: Versatility for HPC & AI | NVIDIA"
[15]: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md?utm_source=chatgpt.com "cutlass/media/docs/cpp/blackwell_functionality.md at main · NVIDIA/cutlass · GitHub"
[16]: https://rocm.docs.amd.com/en/docs-7.14.0/reference/precision-support.html "https://rocm.docs.amd.com/en/docs-7.14.0/reference/precision-support.html"
[17]: https://gpuopen.com/learn/wmma_on_rdna3/?utm_source=chatgpt.com "How to accelerate AI applications on RDNA 3 using WMMA - AMD GPUOpen"
[18]: https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/?utm_source=chatgpt.com "Accelerating Generative AI on AMD Radeon™ GPUs - AMD GPUOpen"
[19]: https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html "https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html"
[20]: https://cdrdv2-public.intel.com/824434/2024_Intel_Tech%20Tour%20TW_Xe2%20and%20Lunar%20Lakes%20GPU.pdf?utm_source=chatgpt.com "MTL Tech Day GFX Deep Dive"
[21]: https://www.intel.com/content/www/us/en/support/articles/000098346/graphics.html "https://www.intel.com/content/www/us/en/support/articles/000098346/graphics.html"
[22]: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.add.html?utm_source=chatgpt.com "ttnn.add — TT-NN™ documentation"
[23]: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html?utm_source=chatgpt.com "Tensor — TT-NN&trade; documentation"
[24]: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-fused-attention.html?utm_source=chatgpt.com "Fused Attention — NVIDIA TensorRT"
[25]: https://github.com/pytorch/pytorch/issues/191186?utm_source=chatgpt.com "disagreement-gated row-major preference in `TensorIterator::reorder_dimensions` · Issue #191186 · pytorch/pytorch · GitHub"
[26]: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/FoundationalTypes/DataType.html?utm_source=chatgpt.com "DataType - NVIDIA TensorRT Standard Python API Documentation 11.2.1"
[27]: https://onnx.ai/onnx/operators/onnx__Add.html "https://onnx.ai/onnx/operators/onnx__Add.html"
