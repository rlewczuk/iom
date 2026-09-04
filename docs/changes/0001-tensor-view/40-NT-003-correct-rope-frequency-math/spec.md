# Correct RoPE frequency math in the scratchpad table test and implementation

**Order:** 40
**Priority:** P1 — required defect remediation: the suite's only numerical-table case pins a mathematically wrong table that blocks a future correct implementation, and the scratchpad itself computes the same wrong table.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `NT-003`
**Review severity:** low
**Review verification:** verified, confidence 95

## Outcome

The RoPE table test in `test/test_iom.cpp` and the `LlamaRoPE` constructor in `src/llama.cpp` compute `inv_freq[j] = 1 / θ^(2j/head_dim)` using floating-point division so the exponent math, the exponential base (`θ`), and the assertion values all match the mathematical contract. With `head_dim = 8` and `θ = 10000.0`, the asserted `inv_freq` table is `[1, 0.1, 0.01, 0.001]` for `j = 0..3` and the test continues to assert bit-exact float tables against the scratchpad's output buffers. The test is annotated as tracking the scratchpad model layer (which the in-file instruction excludes from review) and adds focused checks that distinguish `j = 0` from later `j` so a collapsing implementation cannot silently regress.

## Current failure

The `inv_freq` loop at `test/test_iom.cpp:1531-1534` evaluates `(2 * j) / kHeadDim` in integer division before `static_cast<double>`. For every `j < kHeadDim/2` the quotient is 0 (with `kHeadDim = 4` in the test, the loop only runs `j ∈ {0, 1}`, but `j = 1` still yields `(2*1)/4 = 0`), so the certified `inv_freq` table is constant `[1.0, 1.0, …]` — wrong for `j ≥ 1`. The same expression is in `src/llama.cpp:9`, so a future correct implementation that fixes the exponent division would fail the current test. The constructor's `theta` argument is also ignored for the exponential base (hardcoded `10000.0` literal in both `src/llama.cpp:9` and `test/test_iom.cpp:1533`), so `LlamaRoPE(dev, ctx, dim, /*theta=*/7.0, …)` and `LlamaRoPE(dev, ctx, dim, 10000.0, …)` would produce bit-identical `inv_freq` tables — the `theta` parameter is silently inert.

The contract: a numerical-table test must encode `inv_freq[j] = 1 / θ^(2j / head_dim)`, with the exponent division performed in floating point. The remedy is to compute `2.0 * j / static_cast<double>(head_dim)` in both the test and the scratchpad, and to substitute `this->theta` (the constructor argument) for the `10000.0` literal in both sites.

The review's suggested test shape (`head_dim = 8` columns must distinguish `1, 0.1, 0.01, …`) is met by tightening the existing assertion: the test must compare each `(i, j)` slot against the closed-form `std::sin(θ * i * inv_freq[j])` / `std::cos(…)` using the same corrected `inv_freq` formula, and it must add a sanity check that `j ≥ 1` columns differ from column 0 (a collapsing implementation would otherwise pass bit-exact because every entry is `1.0`). The existing `kHeadDim = 4` allows the existing double loop to remain in place — only `kHeadDim = 8` introduces a third column with a non-degenerate exponent `0.5`, so the `head_dim = 8` case is the focused extra fixture that demonstrates the fix.

## Scope

- Rewrite the `inv_freq` computation in the existing test case to use `2.0 * j / static_cast<double>(kHeadDim)` and to use `kTheta` instead of the `10000.0` literal.
- Rewrite the `inv_freq` computation in `src/llama.cpp` to use `2.0 * j / static_cast<double>(head_dim)` and to use the `theta` constructor member instead of the `10000.0` literal.
- Update the existing assertion loop to compare against `std::sin(theta * i * inv_freq[j])` / `std::cos(…)` using the same corrected `inv_freq` formula and the constructor's `theta`.
- Add a second `TEST_CASE` that uses `kHeadDim = 8` and asserts every `j ∈ [0, 4)` produces a distinct `inv_freq[j]` (specifically: `inv_freq[0] = 1`, `inv_freq[1] = 1/√10000 = 0.1`, `inv_freq[2] = 1/100 = 0.01`, `inv_freq[3] = 1/1000 = 0.001`) and that column 0 differs from every later column, so a collapsing implementation cannot pass.
- Annotate both test cases with a comment noting that they track the scratchpad model layer (which is excluded from review by `include/iom/llama.hpp:8-10`) and that their purpose is to certify the corrected math, not the model layer's overall correctness.

## Implementation references

- **Modify:** `src/llama.cpp` — `iom::models::LlamaRoPE::LlamaRoPE` constructor body, the `inv_freq` loop at `src/llama.cpp:8-10`. Replace `1.0 / std::pow(10000.0, 2*j/head_dim)` with `1.0 / std::pow(theta, 2.0 * static_cast<double>(j) / static_cast<double>(head_dim))` so the exponent division is floating-point and the constructor argument is honored.
- **Modify:** `test/test_iom.cpp` — `TEST_CASE("DeviceOps queue drives the llama models through owner views")`, the `inv_freq` loop at `test/test_iom.cpp:1531-1534` and the assertion loop at `:1544-1554`. Replace `(2 * j) / kHeadDim` with `2.0 * static_cast<double>(j) / static_cast<double>(kHeadDim)` and the `10000.0` literal with `kTheta`. Update the per-slot assertions to use the same `inv_freq[j]` formula and `kTheta` so the test mirrors the corrected implementation.
- **Modify:** `test/test_iom.cpp` — add a new `TEST_CASE` immediately after the existing llama-models test, named `"LlamaRoPE inv_freq table distinguishes every column for head_dim=8"`. Construct `iom::models::LlamaRoPE` with `kCtxLen = 8`, `kHeadDim = 8`, `kTheta = 10000.0`; build the expected `inv_freq` vector using the corrected formula; assert each `inv_freq[j]` matches the closed-form value `std::pow(10000.0, -static_cast<double>(j) * 2.0 / 8.0)` to a tight tolerance (bit-exact via the same `std::pow` call) and that `inv_freq[j] != inv_freq[0]` for every `j ∈ [1, 4)`. Use a separate `FakeTensor` pair sized `{kCtxLen, kHeadDim / 2}` for `sin` / `cos` so the test owns its lifetime and does not perturb other cases. Mirror the existing pattern (`sin.view().copy_from_host`, `cos.view().copy_from_host`) for the constructor invocation; only the `inv_freq` math is under test.
- **Read:** `include/iom/llama.hpp` — confirm the `LlamaRoPE` constructor signature (`DeviceOps&, size_t ctx_len, size_t head_dim, double theta, Tensor& sin, Tensor& cos`) and the `theta` member are unchanged; the fix is purely in the constructor body and the test mirror. Do not modify the header.
- **Read:** `test/test_iom.cpp:1510-1612` — preserve the existing `FakeQueue`, `FakeTensor`, and `FakeDevice` setup; the new `TEST_CASE` reuses the same fake fixtures and follows the existing `make_tensor(device, {kCtxLen, kHeadDim / 2})` shape pattern. Do not alter the `decoder.forward` / `model.forward` op-name assertions in the existing test; only the `inv_freq` and per-slot sin/cos comparison change.

## Requirements

- The corrected `inv_freq` formula is `1.0 / std::pow(theta, 2.0 * static_cast<double>(j) / static_cast<double>(head_dim))` in both `src/llama.cpp:9` and `test/test_iom.cpp:1533`. The exponent division must be floating-point and the base must be the constructor's `theta` argument (or the test's `kTheta` constant), never the `10000.0` literal.
- The per-slot assertions in `test/test_iom.cpp:1549-1552` must compare against `std::sin(kTheta * static_cast<double>(i) * inv_freq[j])` and `std::cos(kTheta * static_cast<double>(i) * inv_freq[j])` where `inv_freq` is built with the corrected formula. No bit-exact change in tolerance policy; `CHECK_EQ` against `static_cast<float>(…)` is preserved.
- The new `head_dim = 8` case must assert four distinct `inv_freq` values (`1.0`, `1.0 / std::sqrt(10000.0)`, `1.0 / 100.0`, `1.0 / 1000.0`) using `CHECK` against the same corrected formula, and must include an explicit `CHECK(inv_freq[j] != inv_freq[0])` for every `j ∈ [1, 4)` so a collapsing implementation (one that returns all `1.0`) is rejected.
- The new case must construct and tear down its own `FakeDevice`, `FakeQueue`, and `FakeTensor` fixtures; it must not share mutable state with the existing llama-models case.
- Both the existing test and the new case must carry a short comment explaining they track the `include/iom/llama.hpp:8-10` scratchpad model layer (excluded from review) and that their role is to certify the corrected RoPE math, not the model layer's overall behavior.
- The `Decoder::forward`, `Model::forward`, and `op_names` assertions in the existing test must remain bit-identical: the snake test observes op submission ordering through `FakeQueue`, not RoPE math, and must not regress.

## Non-goals

- Modifying `include/iom/llama.hpp`. The header, the `LlamaRoPE` constructor signature, the `rope()` no-op body, the `LlamaAttention`/`LlamaMlp`/`LlamaDecoder`/`Llama2Model` bodies, and the `Llama2` model wrapper remain untouched. The scratchpad exclusion (`include/iom/llama.hpp:8-10`) covers the broader model layer; this task fixes the test and the one-line `inv_freq` math only.
- Implementing `LlamaRoPE::rope` or any other mathematical compute kernel. The existing `rope()` no-op body remains unchanged.
- Changing `DeviceOps::linear`, `DeviceOps::sdpa`, or any other op signature, behavior, or capability rejection.
- Asserting `inv_freq` values for `head_dim` other than 8 in the new case. The `kHeadDim = 4` value in the existing test remains; the focused extra fixture uses `kHeadDim = 8` per the review's recommended discriminator.
- Adding GPU, ROCm, TTNN, or SYCL conformance cases. NT-003 is local and CPU-only; the test runs inside `iom_tests` and needs no accelerator backend.
- Touching any other test file (`test/test_safetensors.cpp`, `test/test_alloc.cpp`, `test/test_mmap.cpp`, `test/cpu/test_cpu.cpp`, etc.).

## Acceptance criteria

- [ ] `src/llama.cpp:9` reads `inv_freq[j] = 1.0 / std::pow(theta, 2.0 * static_cast<double>(j) / static_cast<double>(head_dim));` — no `10000.0` literal, no integer division in the exponent.
- [ ] `test/test_iom.cpp:1533` mirrors the corrected formula with `kTheta` substituted for the `10000.0` literal and `2.0 * static_cast<double>(j) / static_cast<double>(kHeadDim)` substituted for `(2 * j) / kHeadDim`.
- [ ] The existing per-slot sin/cos assertions in `test/test_iom.cpp:1549-1552` use the same corrected `inv_freq` and `kTheta`; the test continues to bit-exactly compare against `static_cast<float>(std::sin(kTheta * static_cast<double>(i) * inv_freq[j]))`.
- [ ] A new `TEST_CASE` named `"LlamaRoPE inv_freq table distinguishes every column for head_dim=8"` exists in `test/test_iom.cpp`, immediately after the existing `"DeviceOps queue drives the llama models through owner views"` case, and asserts four distinct `inv_freq` values matching `std::pow(10000.0, -j * 2.0 / 8.0)` for `j ∈ [0, 4)`, plus an explicit `CHECK(inv_freq[j] != inv_freq[0])` for every `j ∈ [1, 4)`.
- [ ] Both test cases carry a short comment marking them as tracking the `include/iom/llama.hpp` scratchpad and certifying the corrected RoPE math, not the broader model layer.
- [ ] Manually verified: collapsing the corrected formula to `1.0 / std::pow(theta, 0.0)` (i.e. always `1.0`) makes the new `head_dim = 8` case fail on `CHECK(inv_freq[j] != inv_freq[0])` and on the per-`j` expected-value checks; the corrected formula makes all assertions pass.
- [ ] The existing `decoder.forward` / `model.forward` op-name assertions in `test/test_iom.cpp:1585-1611` are unchanged: `dev.submissions.size() == decoder_ops.size()` and the appended `rmsnorm` + `linear` tail still hold.

## Verification

- Local CPU build + targeted case run:
  - `cmake -S . -B build/cpu -DBUILD_TESTING=ON`
  - `cmake --build build/cpu --target iom_tests`
  - `./build/cpu/test/iom_tests --test-case="DeviceOps queue drives the llama models through owner views"`
  - `./build/cpu/test/iom_tests --test-case="LlamaRoPE inv_freq table distinguishes every column for head_dim=8"`
  - Expect both cases to pass with the corrected formula.
- Full suite unchanged-green:
  - `ctest --test-dir build/cpu --output-on-failure`
  - Expect `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests` to pass (3/3, ~31k assertions) per the review's recorded baseline.
- Pre-fix regression check (run after temporarily reverting `test/test_iom.cpp:1533` and `src/llama.cpp:9` to the integer-division form):
  - Rebuild and rerun both targeted cases.
  - Expect `LlamaRoPE inv_freq table distinguishes every column for head_dim=8` to fail with a reported `inv_freq[j]` mismatch for `j ≥ 1`, and the existing llama-models case to fail on the first `(i, j)` slot where `sin` / `cos` diverge from the expected `static_cast<float>(std::sin(kTheta * i * inv_freq[j]))`.
  - Restore the corrected formula after this check; the diff is the only change in the working tree.
- No GPU, sanitizer, or remote-development runs are required for this finding. NT-003 is purely a CPU-only mathematical-table fix per the review's "Local, cheap" classification (`docs/changes/0001-tensor-view/review.md:490`).
