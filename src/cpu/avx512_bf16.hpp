#pragma once

#include <cstdint>

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

}  // namespace cpu_detail
}  // namespace iom