# Opt-in real-model inference harness

The official inference harness is a default-off, caller-supplied validation path. It
is not part of ordinary backend conformance and it never downloads a checkpoint,
selects a default model, falls back to another backend, or changes production
session or metrics APIs. Backend drivers opt in explicitly; the shared runner owns
only validation, execution, comparison, and evidence publication.

## Inputs and fixed policy

The caller must set `IOM_TEST_MODEL_DIR`, `IOM_TEST_MODEL_ID`,
`IOM_TEST_MODEL_REFERENCE`, and `IOM_TEST_MODEL_EVIDENCE`. For the frozen
TinyLlama distribution, `IOM_TEST_MODEL_ID` is the pinned revision SHA
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, not the repository model name.
An optional `IOM_TEST_MODEL_ARENA_BYTES` is a positive decimal byte count
divisible by 32. The directory and reference pack are verified before model
execution. The pack binds that asserted revision identity to actual relative
files, sizes, and SHA-256 values, and records the case-payload digest,
generator provenance, software pins, precision policy, and model
configuration. The caller must keep
the model directory immutable for the complete run. Because the production
loader accepts paths rather than retained file handles, the shared runner
recomputes the complete size/SHA-256 inventory immediately before every
independent config/session load and again before publishing evidence; any
observed replacement or in-place mutation fails the run.

The fixed official geometry is untied BF16 Llama with
`N22,H2048,I5632,Hq32,Hkv4,D64,V32000,C2048`. The five cases are, in order:

1. raw production greedy;
2. raw fixed-reference continuation;
3. structured-chat production greedy;
4. structured-chat fixed-reference continuation; and
5. zero new tokens.

The raw prompt is `The capital of France is`; special-token addition and BOS
handling are recorded and checked. Chat uses the distribution template for
`[{"role":"user","content":"Hello."}]`, including the assistant prefix,
without adding a BOS token. Nonzero cases use a four-token limit. The fixed
continuation is explicitly scripted as token IDs `3,4,5,6`; those IDs are never
reported as ordinary greedy success.

## Reference and comparison contract

Each case uses the frozen `id` field only (no `name` alias) and records prompt
IDs, rendered UTF-8 bytes, logical positions, policy, input decode IDs, and the
output limit. Integer/boolean fields are type-exact rather than relying on JSON
numeric equality. Every prefill or cached-decode snapshot
records its phase, prefix IDs, position start and run length, absolute positions,
full finite BF16 logits, lowest-ID greedy argmax, selected ID, and the frozen
margin-certification bit. The expected result records token IDs, stop reason, and
initialized KV length. Official logits use
`abs(actual-reference) <= 0.25 + 0.02*abs(reference)`; exact greedy-ID agreement
requires the frozen margin inequality for every competing vocabulary entry.
After a non-certified greedy divergence, logits from different histories are not
compared. Forced continuation always compares the full vocabulary on identical
prefixes.

The shared runner uses the real formatter, tokenizer, session prefill, cached
decode, synchronous recording selector, scratch, and ownership rules. It runs
disabled, metrics-only, and explicitly prepared trace modes sequentially, releasing
large session/model owners between runs. The three modes must agree on finite
logits, selected and generated IDs, decoded text, stop/count state, history, KV
lengths, and poisoning state. Trace rows must have unique positive operation IDs,
valid prefill/decode windows, range-valid optional layer attribution, nonzero run
lengths, and no dropped rows. The current session records its embedding,
normalization, and LM-head submissions as unlayered session-level rows; it does
not fabricate decoder-layer rows for internal direct `DeviceOps` submissions.
Execution across every configured decoder layer is instead checked by requiring
one correctly advanced cache state per layer. EOS remains in generated output
and history without another KV append; stop precedence is EOS, new-token limit,
then context capacity. A zero
limit performs no forward, selection, or trace work and reports an honest
zero/unavailable observation rather than invented timing.

## Wrapper and evidence

`test/model/run_official_inference.py` is standard-library only. It validates all
caller files, artifact identities, the canonical case-payload digest, model
geometry, provenance pins, and the requested backend/executable. It then invokes
exactly one already-built backend test through a subprocess argument vector, never
through a shell. The evidence destination must be outside the model directory
and must not lexically, canonically, or by file identity alias the reference,
backend executable, or any model artifact; the direct C++ entry independently
enforces its reference/model separation before success or failure publication.
Official execution has a 1800-second deadline, followed by a 30-second
process-group `SIGTERM` grace interval and, if needed, `SIGKILL` plus one shared
30-second monotonic reap deadline across collection and wait. Launch, collection,
timeout, cleanup, validation, and unexpected failures all use the same
best-effort atomic failure-evidence path with their stage and available
argv/environment/supervision details. Successful stdout is a compact status,
backend, reference-digest, and evidence-path summary; full measurements remain
in the evidence file.

The C++ runner emits machine-readable evidence containing the verified reference
and artifact identities, backend/device/precision and exact environment/argv
context, logical case outputs and positions, logits, token and stop results,
counts, cache lengths, mode parity, diagnostics, and the initially unclaimed
native-capability field. Evidence publication is atomic and a failed execution is
recorded before its exception is rethrown.

No backend target, option, registration, or production observer is owned by this
contract. Backend-specific leaves provide those registrations and independently
supply successful official-checkpoint evidence; ordinary configuration remains
artifact-free.
