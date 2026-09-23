# Backend contract

This document is the normative contract for an IOM backend. It complements
[ARCHITECTURE.md](ARCHITECTURE.md): the architecture document explains how the
existing implementations are arranged; this document defines what a new
implementation must preserve. “MUST”, “MUST NOT”, and “SHOULD” are normative.

The contract is intentionally expressed in terms of observable `Device`,
`Tensor`, `TensorView`, and `DeviceOps` behavior. A backend may use a vendor
runtime, native storage layout, streams, events, staging buffers, or a
synchronous CPU implementation internally, provided that it meets this
contract. It MUST NOT add a global active-backend selector, make common code
include a vendor runtime, or silently change the device/context that owns an
object.

## Contents

The contract is split by section under `docs/BACKEND_CONTRACT/`. Each file is a
verbatim extract of the former single document: its title is the former section
heading, nested headings keep their relative depth, and every reference to
another section links directly to the file that owns it. Filenames are
descriptive rather than ordered; the list below keeps the order of the former
document. Read the file that owns a topic; a section never restates what an
earlier one already fixes.

### Part I — Implementing a backend

- [Implementing a backend](BACKEND_CONTRACT/implementing-a-backend.md) — what a backend contributes, standard tiled
  storage, the recommended implementation order, the minimal conformance-driver
  shape, and the exit criteria.

### Part II — Normative behavioral specification

- [1. Device identity, capabilities, and factories](BACKEND_CONTRACT/device-identity-capabilities-and-factories.md) — one
  runtime context per device, a stable owner address, backend kind and ordinal
  reporting, the 23-type capability table, tensor and queue creation, and
  factory failure cleanup.
- [2. Tensor owner and allocation contract](BACKEND_CONTRACT/tensor-owner-and-allocation-contract.md) — tensor
  materialization, stable native handles, device and allocator outliving, exact
  tiled allocation size and 32-byte alignment, native-layout conversion checks,
  and the no-allocation rule for operations, views, and transfers.
- [3. Logical representation and physical layout](BACKEND_CONTRACT/logical-representation-and-physical-layout.md) — rank
  and plane semantics, 16x16 tile slot order, padding and zero-fill rules,
  little-endian logical packing, scatter/gather transfer semantics, and the
  independent physical oracle.
- [4. View and transfer contract](BACKEND_CONTRACT/view-and-transfer-contract.md) — `TensorView` semantics,
  leading-dimension transforms, synchronous host-transfer sizes and validation,
  post-failure state, and the explicit no-implicit-wait ordering rule.
- [5. Queue and token contract](BACKEND_CONTRACT/queue-and-token-contract.md) — in-order queue semantics, OID
  encoding and sequence ranges, the six synchronous error values, wait validity,
  sequence exhaustion, and cleanup of outstanding work.
- [6. Memory, admission, and workspace contract](BACKEND_CONTRACT/memory-admission-and-workspace-contract.md) — the two
  reserved accelerator arenas, the per-device live-queue cap, host parking,
  metadata-slot and workspace leases, reuse after proven completion, and the
  IOM-native versus vendor allocation boundary.
- [7. Copy contract](BACKEND_CONTRACT/copy-contract.md) — the required `copy` operation: validate before
  effects, rejection results, logical element mapping across transformed views,
  and same-queue ordering.
- [8. Binary elementwise contract](BACKEND_CONTRACT/binary-elementwise-contract.md) — `add`, `mul`, `sub`, and
  `div`: signatures, validation precedence, broadcasting and shape rules,
  datatype and quantization support, arithmetic and nonfinite behavior, alias
  rules, and failure retention.
- [9. Other compute capabilities](BACKEND_CONTRACT/other-compute-capabilities.md) — the status of `silu`, GQA
  `sdpa`, `linear`, and RMSNorm across CPU, CUDA, ROCm, and SYCL; the entry point
  to the TinyLlama forward-layout and operation subsections below.
  - [TinyLlama forward layout — Embedding and projection boundaries](BACKEND_CONTRACT/tinyllama-forward-layout-embedding-and-projection-boundaries.md) — the
    frozen embedding and `linear` ABI, table and weight sharing across leading
    planes, ordinary versus head-planar output layout, the non-square reference
    case, and the shared checked-view admission path.
  - [TinyLlama forward layout — Dtype and numerical policy](BACKEND_CONTRACT/tinyllama-forward-layout-dtype-and-numerical-policy.md) — semantic
    applicability per leaf, SiLU-specific scalar and numerical policy, storage,
    rounding, masking, nonfinite handling, and reference ownership and coverage,
    including the pinned independent model corpus,
    `test/test_model_reference_fixture.cpp`, and the complete-model checkpoint
    harness in `test/backend/backend_conformance_model_reference.hpp` consumed by
    `test/test_model_integration.cpp`.
  - [SYCL synthetic model integration evidence](BACKEND_CONTRACT/model-inference-sycl-synthetic-evidence.md) — the
    real-device checkpoint and causal-cache results for the pinned offline
    corpus, with exact runtime identity, commands, and finite comparison results.
  - [CUDA pinned synthetic model-reference evidence](BACKEND_CONTRACT/model-inference-cuda-synthetic-evidence.md) — the CUDA device, corpus identity, and named
    complete-model/cache gate record.
  - [TinyLlama forward layout — Normalization and MLP boundaries](BACKEND_CONTRACT/tinyllama-forward-layout-normalization-and-mlp-boundaries.md) — the
    frozen RMSNorm, SiLU, and MLP interfaces with their parameters and storage
    rules, and the operation contracts they defer to.
  - [TinyLlama forward layout — Positions and cache boundaries](BACKEND_CONTRACT/tinyllama-forward-layout-positions-and-cache-boundaries.md) — the RoPE
    facade and admission, cache geometry, cache append rules and feasibility, the
    CUDA cache-row append boundary, and append queue, lifetime, admission, error,
    and coverage rules.
  - [TinyLlama forward layout — Causal grouped-query attention](BACKEND_CONTRACT/tinyllama-forward-layout-causal-grouped-query-attention.md) — the
    TinyLlama caller boundary for the operation-owned attention section: shapes,
    layouts, and the BF16 leaf the retained backends accept.
  - [TinyLlama forward layout — Workspace and execution](BACKEND_CONTRACT/tinyllama-forward-layout-workspace-and-execution.md) — the shared
    facade boundary for embedding, linear, RMSNorm, RoPE, and SiLU, plus the
    workspace and execution requirements of the forward path.
  - [TinyLlama forward layout — Final-logits selection and ownership](BACKEND_CONTRACT/tinyllama-forward-layout-final-logits-selection-and-ownership.md) — the
    backend-neutral synchronous selection boundary for the final untied LM-head
    result, its ownership seam, and its conformance obligations.
  - [TinyLlama forward layout — Session sizing and lifetime](BACKEND_CONTRACT/tinyllama-forward-layout-session-sizing-and-lifetime.md) — the
    bounded-storage plan: parameters and two-stage setup, checked cache and
    activation storage, forward stores and live ranges, requirement maxima,
    producer schedule, and abort behavior.
  - [TinyLlama forward layout — CUDA matrix feasibility](BACKEND_CONTRACT/tinyllama-forward-layout-cuda-matrix-feasibility.md) — the bounded CUDA
    record: linear, SDPA matrix, and SDPA integration native evidence, installed
    capability, fixed ABI, checked data flow, and the seven-operation fit.
  - [TinyLlama forward layout — ROCm matrix feasibility](BACKEND_CONTRACT/tinyllama-forward-layout-rocm-matrix-feasibility.md) — the bounded ROCm
    record: evidence boundary and installed target, native instruction route,
    logical products and tail mapping, caller-owned flow, and the seven-operation
    decision.
  - [ROCm pinned synthetic model-reference and causal-cache evidence](BACKEND_CONTRACT/model-inference-rocm-synthetic-evidence.md) — the exact named ROCm cases, shared frozen checkpoint/cache consumers, offline corpus identity, and remote gate receipt.
  - [ROCm official TinyLlama inference evidence](BACKEND_CONTRACT/model-inference-rocm-official-evidence.md) — the
    artifact-bound ROCm device-0 prefill, cached-decode, numerical, state,
    instrumentation, ownership, native-attribution, and profiler-limit record.
  - [TinyLlama forward layout — SYCL matrix feasibility](BACKEND_CONTRACT/tinyllama-forward-layout-sycl-matrix-feasibility.md) — the bounded SYCL
    record: the evidence boundary and the per-operation feasibility and support
    status of the seven-operation ABI.
  - [RMS normalization](BACKEND_CONTRACT/rms-normalization.md) — the operation-owned contract for
    `DeviceOps::rmsnorm`: facade, admission, requirement query, per-backend
    implementations, and gate evidence.
  - [SiLU activation](BACKEND_CONTRACT/silu-activation.md) — the operation-owned contract for
    `DeviceOps::silu`: facade, semantics, capability matrix, per-backend
    implementation boundaries, and the five-backend gate receipt.
- [10. Model loading and weight layout](BACKEND_CONTRACT/model-loading-and-weight-layout.md) — the section
  introduction with the supported TinyLlama configuration, configuration
  failures, and the mapped weight source and its logical inventory.
  - [Destination preflight and reusable transfer workspace](BACKEND_CONTRACT/destination-preflight-and-reusable-transfer-workspace.md) — the
    complete destination binding preflighted before scratch or transfers, the
    reusable transfer workspace, and synchronous realization and publication.
  - [Backend loading conformance and observable inventory](BACKEND_CONTRACT/backend-loading-conformance-and-observable-inventory.md) — the CPU,
    CUDA, ROCm, and SYCL loading conformance drivers and the observable weight
    inventory each of them checks.
  - [Opt-in real-checkpoint loading verification](BACKEND_CONTRACT/opt-in-real-checkpoint-loading-verification.md) — the default-off load of
    a caller-supplied official checkpoint on each retained backend.
- [Opt-in real-model inference harness](BACKEND_CONTRACT/opt-in-real-model-inference.md) — the default-off, caller-supplied
  official TinyLlama checkpoint/reference-pack harness and its backend-neutral evidence contract.
  - [CPU official TinyLlama inference evidence](BACKEND_CONTRACT/model-inference-cpu-official-evidence.md) — the
    artifact-bound CPU device-0 prefill, cached-decode, numerical, state,
    instrumentation, ownership, and host-only timing record.
  - [CUDA official TinyLlama inference evidence](BACKEND_CONTRACT/model-inference-cuda-official-evidence.md) — the
    artifact-bound CUDA prefill, cached-decode, numerical, state,
    instrumentation, ownership, native BF16 matrix, and profiler-capability
    record.
- [11. Tokenizer and prompt boundary](BACKEND_CONTRACT/tokenizer-and-prompt-boundary.md) — public owners and APIs,
  authoritative artifacts and tokenizer semantics, formatter metadata and fixed
  grammar, rendering and composition, and the error and reference rules of the
  host-side tokenizer and chat formatting contract.
- [12. Backend integration and conformance obligations](BACKEND_CONTRACT/backend-integration-and-conformance-obligations.md) — the executable
  contract coverage map and the integration and conformance obligations of the
  shared scalar, common, and backend scenario files.
- [13. Contract source map](BACKEND_CONTRACT/contract-source-map.md) — where to look when changing or
  extending the contract, and the rule that implementation, conformance,
  `ARCHITECTURE.md`, and this contract move together.
- [14. Inference timing and observation](BACKEND_CONTRACT/inference-timing-and-observation.md) — the caller-owned, backend-neutral `InferenceMetrics` recorder, its optional
  session attachment with load, prefill completion, and generation
  observation, prompt/generated counts, time-to-first-token and stop-aware
  decode throughput, explicit trace preparation, its scalar and bounded
  trace schemas, the supplied monotonic host clock, and the honest CPU, CUDA,
  ROCm, and SYCL limitations on genuine device duration.
  - The [CLI operation trace presentation](BACKEND_CONTRACT/inference-timing-and-observation.md#cli-operation-trace-presentation)
    subsection fixes the opt-in stderr fields, stream and setup boundaries,
    retained-wait semantics, and unavailable device timing.
  - [CLI metric presentation](BACKEND_CONTRACT/inference-timing-and-observation.md#cli-metric-presentation) —
    the stderr-only scalar report, unavailable-value rules, lifetime and
    timing-boundary obligations, and explicit absence of native device timing.

### Part III — Operation contracts

- [Operation contracts](BACKEND_CONTRACT/operation-contracts.md) — how operation sections are organized, what
  each one fixes, and how they relate to the shared contracts of Part II.
  - [Embedding lookup](BACKEND_CONTRACT/embedding-lookup.md) — the bit-preserving row gather: ABI and view
    semantics, payload and index classification, capability matrix, admission and
    error precedence, workspace and control-status protocol, queued index data,
    implementation recipes, and independent-reference obligations.
  - [Linear projections](BACKEND_CONTRACT/linear-projections.md) — `DeviceOps::linear`: ABI, validation and
    failure precedence, ownership, alias, and workspace rules, dtype and
    arithmetic behavior, capability matrix, packed-writer ownership, fixtures and
    tolerances, and evidence obligations.
  - [Scaled dot-product attention](BACKEND_CONTRACT/scaled-dot-product-attention.md) — causal grouped-query
    attention: ABI and admission, normative equations, the nine-leaf semantic
    table, capability matrix, BF16 arithmetic rules, formed-score special values,
    workspace and queue boundaries, reference policy, fixtures, and gate
    evidence.
