#pragma once

#include <cstddef>
#include <cstdint>

#include "iom/tensor.hpp"

// Backend-private surface of the optional AVX-512 BF16 CPU path. Nothing here
// is public API: eligibility is an internal CPU dispatch decision, and the
// observation seam exists only in test builds, where it lets operation tests
// distinguish executed SIMD work from the portable fallback.
namespace iom {
namespace cpu_detail {

// Reports whether this process may enter AVX-512 BF16 target code. The answer
// is the conjunction of three independent facts: this build contains the
// isolated AVX-512 BF16 sources, CPUID advertises AVX512F (leaf 7 subleaf 0,
// EBX bit 16), AVX512_BF16 (leaf 7 subleaf 1, EAX bit 5), and OSXSAVE (leaf 1,
// ECX bit 27), and XGETBV(0) confirms the OS has enabled the XMM, YMM, opmask,
// upper-ZMM, and high-ZMM vector state in XCR0 (bits 1, 2, 5, 6, and 7).
//
// The function is deliberately baseline code with a scalar ABI: any caller may
// run it on any machine. Without the isolated sources it answers false and
// performs no target-specific operation at all, and XGETBV is reached only
// after CPUID reports OSXSAVE. The check is a per-dispatch decision, not a
// per-element one; callers choose once per request between their AVX-512 BF16
// worker and its scalar fallback.
[[nodiscard]] bool avx512_bf16_available() noexcept;

// One BF16 RMSNorm row's first pass: the vector FP32 square and reduction of
// exactly the logical `features` of `(plane, row)`, followed by the contract's
// mean, epsilon, square root, and reciprocal in FP32. The declaration is the
// baseline-callable boundary of an isolated AVX-512 BF16 source, so the
// definition exists only in a build that registers that source; callers reach
// it after `avx512_bf16_available()` has answered true, and the target code
// never runs on a host whose ISA or OS vector state cannot execute it.
//
// The return value reports which arithmetic reduced the row. `true` means the
// vector squares and their running sum executed and `inverse` received the
// row's reciprocal square root of `sum / features + epsilon`. `false` means
// the entry declined the row, which it does for any row holding a non-finite
// logical feature: the contract's special-value classes are then produced by
// the caller's sequential recurrence rather than by a lane-reassociated sum,
// and `inverse` is left untouched. Non-BF16 leaves never reach this entry.
[[nodiscard]] bool avx512_bf16_rmsnorm_reduce(
        const unsigned char* x_base, const TensorSpec& spec,
        std::size_t plane, std::size_t row, std::size_t features,
        float epsilon, float& inverse);

// Settled execution stages of the AVX-512 BF16 workers. A stage reports its own
// work, so SDPA's three stages stay individually observable.
enum class Avx512Bf16Stage : std::uint8_t {
    Linear = 0,
    SdpaQk,
    SdpaSoftmax,
    SdpaPv,
    RmsReduction,
    RmsStore,
    Rope,
    Silu,
    BinaryAdd,
    BinarySub,
    BinaryMul,
    BinaryDiv,
};

// Which path actually executed work for a stage.
enum class Avx512Bf16Path : std::uint8_t {
    Native = 0,
    Fallback,
};

#if defined(IOM_AVX512_BF16_TESTING)
// Test-build-only observation seam. A caller inside an AVX-512 BF16 worker
// records `Native` only inside the executed SIMD arithmetic loop, after the
// vector work has run, and records `Fallback` only where the corresponding
// fallback path actually executes; selecting a kernel is not work and is never
// recorded. The counters are process-wide and thread-safe, and production
// builds contain neither these functions nor their state.
void avx512_bf16_test_record(
        Avx512Bf16Stage stage, Avx512Bf16Path path) noexcept;
void avx512_bf16_test_reset_observations() noexcept;
[[nodiscard]] std::uint64_t avx512_bf16_test_observation(
        Avx512Bf16Stage stage, Avx512Bf16Path path) noexcept;
#endif

// Multiplies one standard-tile row of BF16 leaves in the caller's standard
// 16x16 tiled owner storage. The entry is baseline code: every argument is a
// scalar, no target type appears in its signature, its own translation unit is
// the only place that contains AVX-512 BF16 code, and the caller reaches it
// only from an accepted queued worker after avx512_bf16_available() reported
// true.
//
// The three bases are owner storage and the three bit offsets name the run's
// first leaf; the run is exactly the 16 logical features of one tile row, which
// the shared layout keeps in 16 contiguous slots. A column-broadcast operand
// therefore passes the bit offset of its single lane together with its
// broadcast flag instead of a contiguous run. Both operands are read before the
// result is stored, so an exact in-place alias stays correct.
//
// The vector arithmetic commits only when every lane's exact product is an FP32
// normal below 2^127: such a lane is exact in FP32 (two BF16 significands carry
// at most 16 significant bits), needs no second rounding step, and never meets
// the denormal handling of the FP32 to BF16 conversion. When any lane is
// exceptional -- `0 * infinity`, NaN, infinity, a signed zero, a product that
// underflows to subnormal or zero, or a product that could round to infinity --
// nothing is stored, the function answers false, and the caller recomputes the
// whole row with the scalar codec, which remains the semantic authority for
// BF16 MUL. The answer never reports work that was not committed.
[[nodiscard]] bool avx512_bf16_binary_mul_row(
        const unsigned char* lhs, std::size_t lhs_bit,
        const unsigned char* rhs, std::size_t rhs_bit,
        unsigned char* out, std::size_t out_bit,
        bool lhs_broadcast, bool rhs_broadcast) noexcept;

// Second RMSNorm pass for one logical BF16 row of one plane. The worker decodes
// the row's logical BF16 features, applies the row inverse the scalar first
// pass produced and the shared per-feature scale in FP32, and stores every
// logical output with exactly one BF16 round-to-nearest-even encode. Feature
// tails, selected plane offsets and view strides are honored through the same
// standard tiled addressing the portable pass uses: only logical features are
// addressed, so tile padding, other rows, and other planes keep their bits.
//
// Lanes whose value the ISA conversion cannot encode under this operation's
// contract -- a zero exponent field, i.e. a signed zero or a BF16-subnormal
// result, and NaN results, which the contract stores in canonical positive
// form -- are encoded by the compliant scalar codec inside the same worker.
// The function records `RmsStore`/`Native` once per executed 16-feature vector
// group that stored at least one lane from its SIMD result and
// `RmsStore`/`Fallback` once per logical feature whose encoding came from the
// compliant scalar path (a corrected lane, the scalar feature tail, or the
// whole row when the caller did not enter this worker).
//
// Contract precondition, checked by admission before the request is accepted:
// `x`, `scale`, and `out` are BF16 standard tiled views whose logical features
// are within their owners, so every group offset is inside the operand's
// storage.
void avx512_bf16_rmsnorm_store(
        unsigned char* out_base, const TensorSpec& out_spec,
        std::size_t out_plane, const unsigned char* x_base,
        const TensorSpec& x_spec, std::size_t x_plane,
        const unsigned char* scale_base, const TensorSpec& scale_spec,
        std::size_t scale_plane, std::size_t row, std::size_t features,
        float inverse);

}  // namespace cpu_detail
}  // namespace iom