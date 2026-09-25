#include "avx512_bf16.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#if defined(IOM_AVX512_BF16_COMPILED)
#include <cpuid.h>
#endif

namespace iom {
namespace cpu_detail {

namespace {

#if defined(IOM_AVX512_BF16_COMPILED)
// CPUID inputs of the eligibility check. Leaf 7 advertises the AVX-512
// features: subleaf 0 carries AVX512F in EBX bit 16 and reports how many
// subleaves exist in EAX, and subleaf 1 carries AVX512_BF16 in EAX bit 5.
// Leaf 1 carries OSXSAVE in ECX bit 27.
constexpr std::uint32_t kFeatureLeaf = 0x00000007;
constexpr std::uint32_t kFeatureSubLeaf = 0x00000000;
constexpr std::uint32_t kBf16FeatureSubLeaf = 0x00000001;
constexpr std::uint32_t kBaselineLeaf = 0x00000001;
constexpr std::uint32_t kLeaf7EbxAvx512F = 1u << 16;
constexpr std::uint32_t kLeaf71EaxAvx512Bf16 = 1u << 5;
constexpr std::uint32_t kLeaf1EcxOsxsave = 1u << 27;

// XCR0 bits the AVX-512 BF16 workers need the OS to have enabled: XMM (1),
// YMM (2), opmask (5), upper ZMM (6), and high ZMM (7).
constexpr std::uint64_t kRequiredXcr0State =
        (std::uint64_t{1} << 1) | (std::uint64_t{1} << 2) |
        (std::uint64_t{1} << 5) | (std::uint64_t{1} << 6) |
        (std::uint64_t{1} << 7);

// Registers are ordered EAX, EBX, ECX, EDX.
void read_cpuid(
        std::uint32_t leaf, std::uint32_t sub_leaf,
        std::uint32_t (&registers)[4]) noexcept {
    __cpuid_count(
            leaf, sub_leaf, registers[0], registers[1], registers[2],
            registers[3]);
}

// Reads XCR0. The _xgetbv intrinsic requires the xsave target feature, which
// the baseline command line does not enable, so the instruction is encoded
// directly. It is reached only after CPUID has reported OSXSAVE.
[[nodiscard]] std::uint64_t read_xcr0() noexcept {
    std::uint32_t low = 0;
    std::uint32_t high = 0;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (static_cast<std::uint64_t>(high) << 32) | low;
}
#endif  // IOM_AVX512_BF16_COMPILED

#if defined(IOM_AVX512_BF16_TESTING)
// One counter per (stage, path) pair, sized from the last enumerators so the
// table follows the enums without a second list to keep in step.
constexpr std::size_t kObservationPathCount =
        static_cast<std::size_t>(Avx512Bf16Path::Fallback) + 1;
constexpr std::size_t kObservationSlotCount =
        (static_cast<std::size_t>(Avx512Bf16Stage::BinaryDiv) + 1) *
        kObservationPathCount;

std::array<std::atomic<std::uint64_t>, kObservationSlotCount> observations{};

[[nodiscard]] std::size_t observation_slot(
        Avx512Bf16Stage stage, Avx512Bf16Path path) noexcept {
    return static_cast<std::size_t>(stage) * kObservationPathCount +
            static_cast<std::size_t>(path);
}
#endif  // IOM_AVX512_BF16_TESTING

}  // namespace

bool avx512_bf16_available() noexcept {
#if !defined(IOM_AVX512_BF16_COMPILED)
    // This build has no isolated AVX-512 BF16 source, so no target code can be
    // entered and no CPUID/XGETBV query is needed.
    return false;
#else
    std::uint32_t features[4] = {};
    read_cpuid(kFeatureLeaf, kFeatureSubLeaf, features);
    // EAX reports the highest leaf 7 subleaf; the BF16 feature lives in
    // subleaf 1, so an implementation without it is ineligible. Reading an
    // absent subleaf is not neutral: some CPUs answer with the subleaf 0
    // values, whose EAX alone would look like a feature bit.
    if ((features[1] & kLeaf7EbxAvx512F) == 0 ||
            features[0] < kBf16FeatureSubLeaf) {
        return false;
    }

    std::uint32_t bf16[4] = {};
    read_cpuid(kFeatureLeaf, kBf16FeatureSubLeaf, bf16);
    if ((bf16[0] & kLeaf71EaxAvx512Bf16) == 0) {
        return false;
    }

    std::uint32_t baseline[4] = {};
    read_cpuid(kBaselineLeaf, 0, baseline);
    if ((baseline[2] & kLeaf1EcxOsxsave) == 0) {
        return false;
    }

    return (read_xcr0() & kRequiredXcr0State) == kRequiredXcr0State;
#endif
}

#if defined(IOM_AVX512_BF16_TESTING)
void avx512_bf16_test_record(
        Avx512Bf16Stage stage, Avx512Bf16Path path) noexcept {
    const std::size_t slot = observation_slot(stage, path);
    if (slot < kObservationSlotCount) {
        observations[slot].fetch_add(1, std::memory_order_relaxed);
    }
}

void avx512_bf16_test_reset_observations() noexcept {
    for (auto& count : observations) {
        count.store(0, std::memory_order_relaxed);
    }
}

std::uint64_t avx512_bf16_test_observation(
        Avx512Bf16Stage stage, Avx512Bf16Path path) noexcept {
    const std::size_t slot = observation_slot(stage, path);
    if (slot >= kObservationSlotCount) {
        return 0;
    }
    return observations[slot].load(std::memory_order_relaxed);
}
#endif

}  // namespace cpu_detail
}  // namespace iom