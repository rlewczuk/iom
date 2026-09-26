#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// Private interface of the isolated AVX-512 BF16 RoPE pair kernel. Nothing
// here is public API or a backend contract: the CPU worker calls it only after
// the runtime detector has accepted this host, and the baseline translation
// unit never mentions the target instructions behind it.
namespace iom {
namespace cpu_detail {

// How many split-half feature pairs one invocation vectorizes. One 512-bit
// FP32 vector holds sixteen decoded BF16 pair sources, and one BF16 packing
// conversion returns the sixteen encoded results of each half.
inline constexpr std::size_t kAvx512Bf16RopePairs = 16;

// Vector split-half pair arithmetic for one nonzero-position logical row of a
// BF16 RoPE request: the target counterpart of the worker's scalar pair.
//
// The caller owns everything the RoPE contract fixes outside the four
// products. It selects the pairs, evaluates each pair's sine and cosine with
// the same expressions the scalar path uses, and passes those values in;
// `sine` and `cosine` each hold `count` values for pairs
// `[first_pair, first_pair + count)` of the row's first half, and a pair's
// second-half column is `half = dimensions.back() / 2` features further along
// the same row. `count` is between one and `kAvx512Bf16RopePairs`.
//
// The kernel decodes both source elements of every pair, forms the four
// products and the following subtraction/addition as separate FP32 vector
// operations, and encodes each destination element once with BF16
// round-to-nearest-ties-to-even. A pair whose sources or FP32 results leave the
// finite normal range is not encoded or stored: it is reported in the returned
// bit mask, bit `k` marking pair `first_pair + k`, and the caller must
// evaluate that pair with the compliant scalar codec instead. Only logical
// source elements are read and only logical destination elements are written,
// so tile padding and every other plane stay untouched.
[[nodiscard]] std::uint32_t avx512_bf16_rope_pairs(
        const unsigned char* x_base, unsigned char* out_base,
        std::span<const std::size_t> x_dimensions,
        std::span<const std::size_t> out_dimensions, std::size_t x_plane,
        std::size_t out_plane, std::size_t row, std::size_t first_pair,
        std::size_t count, const float* sine, const float* cosine) noexcept;

}  // namespace cpu_detail
}  // namespace iom