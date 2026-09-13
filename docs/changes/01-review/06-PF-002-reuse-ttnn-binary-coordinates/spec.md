# Reuse TTNN binary coordinate storage

**Order:** 06
**Priority:** P1 — per-element heap churn is mechanically present
**Blocked by:** None
**Review source:** `cpp-inference-performance` — whole-codebase review, branch `main`, HEAD `cb88619865ff9bcda0a4e3bac4bae4bd4bf510a8` (`Move cpp-inference-review skill to .omp`), baseline `origin/main`; clean at scope capture; upstream `origin/main` ahead 4/behind 0
**Finding:** PF-002
**Review area:** Performance
**Review severity:** low
**Review verification:** strongly-supported, confidence 90
**Review scope:** whole-codebase
**Backend scope:** TTNN
**Location:** `src/ttnn/binary.cpp:91-114` inside `iom::ttnn_detail::binary_planes`

## Outcome

TTNN binary coordinate and broadcast mapping storage is reused within each request or replaced by direct bounded index computation, eliminating per-element coordinate-vector heap allocation while preserving every binary output, view transform, broadcast, padding, and failure/lifetime behavior.

## Current problem

`binary_planes` computes `leading_rank = rank - 2` and iterates every result element (`src/ttnn/binary.cpp:26-32,91-114`). Inside that loop it constructs `std::vector<std::size_t> coordinates(leading_rank)` and, in `operand_coord`, constructs a second `mapped(leading_rank, 0)` vector for each operand; the lambda is called for `lhs` and `rhs`. For every non-empty rank 3–8 result (`leading_rank >= 1`), these are three non-empty vector-storage constructions per output element. Their sizes and mapping metadata are request-invariant, so this adds heap churn to the host fallback loop. The source establishes the mechanism; allocator call counts, wall-time materiality, and production-model impact are unmeasured. The existing SYCL loop's reused coordinate pattern (`src/sycl/queue_binary.cpp:72-78`) is a useful analogue, not a requirement for identical code.

## Scope

- Hoist/reuse coordinate and mapped storage outside the result-element loop, use bounded fixed/request-local storage for the public maximum of six leading axes (ranks 2–8), or compute plane indices directly from flat coordinates without vectors.
- Preserve right-aligned singleton broadcasting, transformed-view offsets/strides and logical-plane mapping, all four scalar operations, per-plane caches and output writes, untouched padding, and all queue/failure/lifetime semantics.
- Keep this independent from TTNN deferred execution: moving `binary_planes` to a worker would not remove these per-element constructions.

## Implementation references

- **Modify:** `src/ttnn/binary.cpp:53-145` — `binary_planes` `plane_at`, flat-coordinate decomposition, `operand_coord`, and `read` call path; reuse or eliminate request-invariant vectors.
- **Read:** `src/sycl/queue_binary.cpp:72-78` — one-coordinate reuse analogue; preserve TTNN's own native plane and broadcast differences.
- **Read:** `src/ttnn/queue.cpp:177-220` — `TtnnQueue::execute` calls all four operations through `binary_planes`; do not change its ownership or finish semantics.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:1897-2010,2161-2210` — rank-3/rank-4 binaries, broadcasts, transformed views, aliases, and padding; independent scalar oracle and retained-failure cases.

## Requirements

- Ensure rank 3 through rank 8 non-empty output requests perform no coordinate or mapped-vector heap allocation inside the result-element loop. Use one request-local reusable buffer, bounded fixed storage for at most six leading axes, or direct index computation; do not add a general allocator or unbounded scratch structure.
- Keep coordinate decomposition and `plane_at` arithmetic equivalent, including right-aligned singleton broadcasting for both operands, transformed views/strides/offsets, rank-2 behavior, and all plane combinations.
- Preserve `load`, `read`, per-plane cache reuse, factor-two carriers, scalar codec/operation selection, output cache and padded bytes, upload leases, finish and exception behavior. Do not change TTNN representation or public APIs.
- Treat wall-time materiality as unverified and do not use a percentage speedup as acceptance. Structural elimination of per-element coordinate storage is the acceptance condition.

## Non-goals

- Do not fix TTNN synchronous submission, blocking downloads/uploads, mesh finish, host emulation, or any CC-003 contract root.
- Do not alter native TTNN storage, scalar precision, capabilities, public APIs, rank/broadcast rules, workspace policy, or model-level behavior.
- Do not claim prefill/decode/KV-cache, attention, logits, generation, or production inference impact; no model executor exists in this tree.

## Acceptance criteria

- [ ] For non-empty rank-3 through rank-8 outputs, coordinate/mapped storage makes no heap allocation inside the result-element loop, and allocation count for this mapping state is independent of `plane_combos * rows * columns`; rank-2 remains correct.
- [ ] ADD/MUL/SUB/DIV outputs for broadcasts, transformed views, aliases, all supported TTNN dtypes, untouched padding, repeat waits, and retained failures remain byte-identical to the independent oracle, with no timing threshold or production-performance claim required.

## Verification

- `ctest --test-dir <build> --output-on-failure -R '^(iom_ttnn_conformance_tests|iom_backend_conformance_cpu_tests)$'` plus the focused TTNN binary conformance filters on the configured remote TTNN host.
- Use a source-bound/counting-allocation scenario for rank-2 control and rank-3/rank-4 `{1,1024,1024}`/`{8,1024,1024}` requests to show mapping storage allocation is request/constant-bounded rather than per element, then compare all outputs and padding to the oracle. No wall-time materiality threshold is required; supplied CPU-only validation passed 3/3 and TTNN remote sync attempts timed out before build, so no TTNN runtime result is claimed.
