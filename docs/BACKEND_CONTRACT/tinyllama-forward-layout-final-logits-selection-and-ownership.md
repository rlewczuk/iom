# TinyLlama forward layout — Final-logits selection and ownership

This subsection freezes and publishes the backend-neutral synchronous selection
boundary for the final untied LM-head result described by
[Embedding and projection boundaries](tinyllama-forward-layout-embedding-and-projection-boundaries.md#tinyllama-forward-layout--embedding-and-projection-boundaries).
The public seam is:

```cpp
struct TokenSelectorScratch {
    std::span<std::byte> host;
    RawWorkspaceView device;
};

struct TokenSelectorScratchRequirements {
    std::size_t host_bytes;
    WorkspaceRequirements device;
};

class TokenSelector {
public:
    virtual ~TokenSelector() = default;
    virtual TokenSelectorScratchRequirements scratch_requirements(
            const TensorView&, std::size_t) const = 0;
    virtual std::size_t select(
            DeviceOps&, const TensorView&, std::size_t, oid,
            std::span<const std::size_t>, TokenSelectorScratch) = 0;
};
```

`scratch_requirements(logits, valid_vocabulary)` is a pure query over the
supplied view and extent. It MUST perform no allocation, transfer, queue
reservation, wait, submission, owner registration, or data mutation. A
concrete implementation defines its own host byte count, device
`WorkspaceRequirements`, alignment, placement, and admission details; this
seam mandates no host transfer, staging direction, vendor type, scratch
capacity, or backend capability.

`TokenSelectorScratch` is a per-call descriptor. Its `host` span and `device`
`RawWorkspaceView` refer only to caller-provisioned reusable storage. They
MUST remain live through the synchronous `select` return and MAY be reused
only after that call completes. The selector MAY write its scratch during the
call but MUST NOT retain either span, the `RawWorkspaceView`, or any hidden
scratch allocation after return. No persistent selector ownership is implied.

`select` is synchronous and deliberately not `noexcept`. Its arguments are,
in order, the queue, logits, valid vocabulary, producing OID, complete
history span, and caller-owned scratch descriptor. Invalid input, readiness,
runtime, and device-data failures are delivered with standard exceptions. The
method returns one token ID only after the producing work and all selector
work have completed successfully; it exposes no asynchronous selection result
or selector OID.

`logits` MUST be a borrowed const BF16 view with exact logical rank-two shape
`[1,V]`, where both dimensions are nonzero. The view and its live storage
owner MUST belong to the exact device served by the queue. `valid_vocabulary`
MUST be nonzero and exactly equal the logical `V`; the only selectable IDs are
`[0,V)`. Physical 16x16 tile padding, and any physical row or feature outside
that logical extent, MUST NOT be read as a candidate or included in
validation of logit values.

The producing `oid` MUST be positive and MUST identify work actually
submitted by this same live `DeviceOps` queue. An OID from a different queue
is invalid even when both queues serve the same backend device; zero,
negative, foreign, future, skipped, reserved-but-never-submitted, and
otherwise unsubmitted values remain invalid under the existing queue
contract. The session MUST successfully wait for every direct prerequisite
before submitting a dependent final projection. Selection itself MUST then
successfully observe `queue.wait(producer)` before using the logits. Positive
admission alone is not readiness. An already successful wait does not remove
this requirement because successful waits are repeatable. If the producer has
a retained completion failure, this call rethrows it, and every later wait
for that OID rethrows the same failure.

The queue, logits view, logits storage owner, and the storage owner's exact
device identity MUST remain live and unchanged throughout the call.
`history` is likewise borrowed only for the call, and the caller-provisioned
scratch storage remains live until return. The selector MUST NOT retain the
queue, either borrowed view, either span, the scratch descriptor, or any
referenced storage after return. Selection MUST NOT mutate logits, history,
KV storage, or any owner identity. The selector MAY mutate only the supplied
scratch storage. Scratch placement, capacity, alignment, and lifetime are
caller/implementation boundaries described by the requirements query, not
hidden ownership in this abstract seam.

No full-vocabulary host transfer is required by this seam. Backend-neutral
`DeviceOps` and `TensorView` access is sufficient, and the concrete selector
chooses a permitted device-local or host-transfer strategy. This boundary
mandates neither a hidden transfer nor a vendor API or vendor type.

`history` MUST contain the entire prompt/input sequence, including every
policy-added special token, followed in order by every successfully committed
generated ID. It therefore includes the latest committed token even when that
token has not yet been appended to KV for a later decode step. The selector
neither appends nor removes history and owns no session state.

After a successful selector return, the session MUST independently validate
the returned ID against `[0,V)` before committing it. Only then does the
session append it to history, increment the committed generated-token count,
apply its EOS/new-token/context precedence, and decide whether another model
step and KV append are needed. EOS inclusion, simultaneous-stop precedence,
capacity, absolute positions, initialized cache length, cache mutation,
failure/poisoning, draining, and reset remain exclusively session-owned. The
selector MUST NOT interpret an ID as EOS or mutate those policies.

The following prompt and decode transitions are normative:

1. After successful prefill of prompt IDs `[p0,p1,p2]`, the session has
   `history=[p0,p1,p2]`, committed generated-token count zero, and
   `initializedL=3`. The first selector call borrows that complete history and
   the final `[1,V]` logits plus their producing queue/OID.
2. If it returns valid `t0`, the session commits
   `history=[p0,p1,p2,t0]` and generated-token count one. If EOS or an
   applicable new-token/context limit ends the request, `t0` remains included
   in committed output while `initializedL` remains three: no subsequent KV
   append is needed merely to record an already committed terminal token.
3. If generation continues instead, the session processes committed `t0` at
   absolute position three. Only after both K and V append OIDs succeed may it
   publish `initializedL=4`. The following selector call then receives
   `history=[p0,p1,p2,t0]`; history is not reconstructed from cache length.

Committed generated-token count and initialized cache length are therefore
distinct state. Selection cannot publish a cache prefix, and committing a
terminal token cannot imply a physical KV write.

The production selector sibling owns exactly greedy behavior. It MUST examine
every logical logit in `[0,V)` and establish that each is finite, including
values that cannot win. It MUST throw on any NaN, positive infinity, or
negative infinity; a nonwinning NaN is still a data failure. For an all-finite
input it returns the ID of the maximum value and resolves an exact tie by the
lowest ID. There is no stochastic path, RNG, temperature, top-k, top-p, beam
search, or hidden fallback.

Failure ownership and no-mutation behavior are normative:

- An invalid logits shape, extent, owner/device relationship, or producer OID
  is rejected by the selector boundary with a standard exception and no
  returned ID.
- A producer readiness/completion failure, selector runtime failure, or
  nonfinite logical logit is a selector-call failure. It returns no ID; the
  session commits no token and does not change history or initialized cache
  state. Supplied scratch may contain partial private work after a failure,
  but the selector retains no scratch ownership. Session poisoning and
  draining of already accepted OIDs remain the session's responsibility.
- A deterministic injected selector MAY replace greedy behavior only in tests.
  If it returns an ID outside `[0,V)`, the session rejects that result before
  history, generated count, or cache mutation. Injection cannot bypass
  producing-OID readiness, ID-range validation, history ownership, or cache
  rules.

Thus a failed or out-of-range selection consumes no invalid token. Selector
scratch may contain partial private work after a failure, but logits, history,
KV contents, owner identities, committed-token count, and initialized cache
length remain unmodified by selection.

The landed concrete implementation is `iom::GreedyTokenSelector final`
([`include/iom/token_selection.hpp`](../../include/iom/token_selection.hpp),
[`src/token_selection.cpp`](../../src/token_selection.cpp)). It declares and
defines both overrides together, is stateless between calls, allocates nothing
on the call path, and retains no view, span, queue, owner, or scratch reference
after return. Its requirements query returns the checked logical byte count
`host_bytes = 2 * V` with no host alignment requirement and delegates `device`
exactly to `copy_to_host_workspace_requirements()`: the CPU owner reports zero
workspace `{0, 1}`, while the standard accelerator owners report a
32-byte-aligned `compute_staging_size(2 * V)` staging range that must be the
queue device's exact-device, live, sufficiently sized, nonoverlapping
caller-owned storage. `select` validates every host-checkable fact before any
queue effect — unquantized BF16 `[1,V]`, nonzero extent exactly equal to `V`,
queue and owner device identity, positive same-queue producing OID, history
span, host-scratch capacity and logits nonoverlap, device-scratch admission,
and checked byte arithmetic — then successfully waits `producer`, copies
exactly the logical `2 * V` BF16 bytes through the existing
`TensorView::copy_to_host` path, and scans IDs in ascending order with a
strict `>` update so equal values, signed zero included, keep the lowest ID.
Invalid input is `std::invalid_argument`, checked byte arithmetic overflow is
`std::overflow_error`, a nonfinite logical logit is `std::runtime_error`, and
producer wait or transfer failures propagate unchanged. The full-logical-extent
host transfer and its O(V) host traffic are this policy's own strategy and
known cost, not a requirement of the abstract seam; no backend-specific
selector kernel, reduction, switch, hidden allocation, or fallback exists.

**Same-revision retained-backend evidence.** All four retained backends execute
the one shared dispatcher in
[`test/backend/backend_conformance_token_selection.hpp`](../../test/backend/backend_conformance_token_selection.hpp)
through their existing conformance targets, linked from
[`test/cpu/test_cpu_conformance.cpp`](../../test/cpu/test_cpu_conformance.cpp),
[`test/cuda/test_cuda_conformance.cpp`](../../test/cuda/test_cuda_conformance.cpp),
[`test/rocm/test_rocm_conformance.cpp`](../../test/rocm/test_rocm_conformance.cpp),
and [`test/sycl/test_sycl_conformance.cpp`](../../test/sycl/test_sycl_conformance.cpp).
The shared case set covers `V = 1` and practical `V = 32000` extents, poisoned
physical padding with native observation, unique interior/last/first maxima,
all-negative logits, signed-zero and equal-maxima lowest-ID ties, finite
subnormal and extreme leaves, NaN/`+inf`/`-inf` in a losing position, pure and
repeatable requirements, producer readiness, retained producer failure, the
complete invalid/range/identity/scratch rejection matrix, no queue or
input/history/owner mutation, and each backend's native transfer failure where
one exists. The gate ran from the prepared closure worktree on branch
`run-task/006-tinyllama--10-greedy-token-selection--10-five-backend-selection-gate`,
which is based at revision `2ef3dc950ba16ed23a2c8acdff58dc3faf5c8eb0`; the only
source change in this revision is the explicit unavailable-seam record added to
the shared harness. Every accelerator command used its configured remote
profile with a fresh sync immediately before execution, a bounded `flock -w`
GPU lock, and remote-side `timeout --kill-after=30s`; the transcripts are
retained in the task evidence
`.cswd/tasks/006-tinyllama/10-greedy-token-selection/10-five-backend-selection-gate/remote.log`,
and all three remote mirrors were removed after the run.

| Backend and runtime identity | Executed gate evidence |
| --- | --- |
| CPU: local `x86_64`, AMD Ryzen AI 9 HX 370 w/ Radeon 890M | `cmake --build build --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests -j8`; three anchored `ctest --test-dir build --output-on-failure --timeout 300 -R '^<target>$'` runs — `iom_tests` 1/1 (9.95 s), `iom_cpu_tests` 1/1 (0.15 s, the target hosting the `DeviceOps` queue-device identity fixture), `iom_backend_conformance_cpu_tests` 1/1 (83.51 s); `./build/test/iom_backend_conformance_cpu_tests --test-case='CPU conformance: greedy token selection shared matrix and lifetime'` — 1 case, 64859/64859 assertions passed. |
| CUDA profile `bv1`: NVIDIA GeForce RTX 5090, driver `595.71.05`, CUDA `13.2`, `nvcc` `V13.2.78`, mirror `greedy-selection-10-cuda` | `cmake --build build --target iom_cuda_conformance_tests`; `./build/test/iom_cuda_conformance_tests --test-case='CUDA conformance: greedy token selection shared matrix and lifetime'` — 1 case, 64942/64942 assertions passed; `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_cuda_conformance_tests$'` — 1/1 passed (78.42 s). The driver's counted native submission fault is consumed by the real staged logical transfer. |
| ROCm profile `bv2`: AMD Radeon AI PRO R9700 (`gfx1036` also enumerated), HIP `7.15.26333`, mirror `greedy-selection-10-rocm` | `cmake --build build --target iom_rocm_conformance_tests`; `./build/test/iom_rocm_conformance_tests --test-case='ROCm conformance: greedy token selection shared matrix and lifetime'` — 1 case, 64942/64942 assertions passed; `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_rocm_conformance_tests$'` — 1/1 passed (95.47 s). The driver's counted native submission fault is consumed by the real staged logical transfer. |
| SYCL profile `bv2`: two Intel Arc Pro B60 Level Zero GPUs (`sycl-ls` enumerates `[level_zero:gpu][level_zero:0]` first), oneAPI/`icpx` `2026.1.0`, mirror `greedy-selection-10-sycl` | `cmake --build build --target iom_sycl_conformance_tests`; `./build/test/iom_sycl_conformance_tests --test-case='SYCL conformance: greedy token selection shared matrix and lifetime'` — 1 case, 64811/64811 assertions passed; `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_sycl_conformance_tests$'` — 1/1 passed (21.71 s). |

The CPU and SYCL transfer paths expose no accepted post-readiness
transfer-failure seam, so their shared case prints a
`token-selection-native-failure-record ... coverage=unavailable` line with the
exact reason instead of passing silently; CUDA and ROCm exercise the real
transfer failure listed above. That record is a stated coverage limitation of
those two test seams, not a selection-result or support regression.

This section still adds no session implementation, facade, or kernel, and it
claims no backend support beyond the four executed conformance targets above.
Session integration, EOS/limit policy, and KV mutation remain owned by their
later session siblings, and existing neural hooks remain `Unsupported` until
their real operation ports land.
