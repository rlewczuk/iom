# Destination preflight and reusable transfer workspace

Before it provisions scratch or transfers a weight, a caller preflights the
complete destination binding of a published source:

```cpp
WorkspaceRequirements ModelSource::upload_workspace_requirements(
    const Device& device, std::span<Tensor* const> destinations) const;
```

The caller first creates exactly one independent `Tensor` owner per `weights()`
entry from that entry's `tensor_spec(index)`, then supplies those owners as
pointers in exactly the published inventory order. The binding is one-to-one
and ordered: the list holds exactly `weights().size()` entries, and one owner
may not serve two roles, not even two roles whose selected logical metadata is
equal, such as the `[1, H]` normalization scales.

**Fixed ordering.** Create destination tensors on the chosen device, then query
this requirement, then provision the reusable scratch it reports, and only then
transfer. The query provisions, leases, submits, waits, allocates, and destroys
nothing, so a caller neither guesses a workspace size nor allocates device
scratch before the binding is known to be complete. The query deliberately takes
no workspace: positive workspace liveness, ownership, alignment, and overlap
rules belong to the synchronous upload, which enforces them with the shared
`detail::WorkspaceValidation::validated` rules of
[section 4](view-and-transfer-contract.md#4-view-and-transfer-contract).

**Complete validation before the first owner query.** Every destination is
validated against the published inventory before any per-owner requirement
query runs, so a rejected binding reports no partial requirement and performs
no query, allocation, or transfer. Validation covers, per destination in order:

- a non-null `Tensor*`;
- the owner's own full view, taken from `Tensor::view()`: the ABI binds owners
  rather than views, and a view whose owner identity is another object is
  rejected, so no transformed view, plane subrange, or retargeted owner can
  enter the binding;
- a `Tensor` created by the exact `device` argument instance; equal backend
  kind and ordinal on another instance are still a foreign binding;
- the destination device's own BF16 storage capability from
  `supported_data_types()`, checked once for the binding before any destination
  metadata is compared;
- the supplied destination's checked logical and standard 16x16 tiled sizing,
  and the checked aggregate logical and tiled byte totals of the whole binding;
- a full `TensorSpec` match with `tensor_spec(index)`: rank, dimensions, BF16
  leaf type, and `QuantizationFormat::NONE`.

The supplied destination is sized before its schema is compared, so an
unsizeable declared shape reports the checked arithmetic failure rather than
the schema mismatch. Live `Device`, allocator, and `Tensor` lifetimes and
distinct owners' nonoverlapping valid storage remain caller preconditions: this
query claims no dangling-reference or invalid-allocation detection beyond exact
owner and device identity.

**Reported requirement.** The result is the maximum `bytes` and the maximum
`alignment` over the binding's per-owner pure
`copy_from_host_workspace_requirements()` results, so one serial scratch range
satisfies every weight. It is never a sum of mutually exclusive scratch
requirements, a guessed native storage size, or an aggregate logical byte
count. A binding whose requirements are all zero returns exactly `{0, 1}` and
never requests a positive allocation. The per-owner requirement hook is pure
and deterministic, so the same complete destination list reports the same
maxima regardless of arena capacity, queue occupancy, leases, or completion
state.

**Destination failures.** These categories are normative:

| Condition | Exception |
| --- | --- |
| Missing or excess destinations, a null `Tensor*`, or one owner bound twice | `std::invalid_argument` naming the position, and both positions for a repeat |
| A destination of another `Device` instance, or not the owner's own full view | `std::invalid_argument` naming the position |
| A device whose `supported_data_types()` omits BF16 | `std::invalid_argument` |
| A wrong rank, dimensions, leaf type, or quantization for the published index | `std::invalid_argument` naming the position, the required rank/shape/dtype/quantization, and the actual one |
| Impossible checked destination or aggregate sizing | `std::overflow_error` |
| Backend requirement-query failure | that backend's established category |

`ModelSource` keeps exactly one private complete-binding validator, and the
preflight query and the synchronous upload share it: there is no optional
bypass, duplicate public binding container, or generic planning API. The query
stays backend-neutral — it contains no backend-kind switch and no native
runtime type — and it never claims readiness: successful validation of a
binding is not a completed upload and publishes no usable device weights. Only a
normal return of the synchronous realization below is a publication permission.

# Synchronous realization and publication

A caller realizes the same complete ordered binding it preflighted by calling

```cpp
void ModelSource::upload_weights(
    Device& device, std::span<Tensor* const> destinations,
    RawWorkspaceView workspace = {}) const;
```

The normal `void` return is the only successful completion and the only
publication permission: there is no ready wrapper, returned readiness object,
per-weight status, or partial success, so a caller exposes its usable model only
after this call returned. The call runs the shared complete-binding validator of
the previous subsection first, so the count, order, owner identity, exact
`Device` instance, BF16 capability, full selected specification, and checked
sizing of the whole binding are settled before the first copy, and it then copies
each published entry's exact private mapped row-major BF16 span directly into
that entry's destination full owner view, in exactly the published inventory
order. HF `[out, in]` orientation and row-major element order are preserved
unchanged, a rank-one normalization source stays a `2*H` byte payload realized
into its `[1, H]` logical destination, and no transpose, intermediate CPU copy,
per-weight checkpoint buffer, `DeviceOps::copy`, queued copy, OID, or
asynchronous work exists on this path.

**Workspace.** The binding's aggregate requirement is the maximum serial
per-owner `copy_from_host` requirement that the same validator reports. When
that requirement is positive, the supplied `workspace` is validated against
every destination's full owner view with the shared
`detail::WorkspaceValidation::validated` rules of
[section 4](view-and-transfer-contract.md#4-view-and-transfer-contract), even when the caller supplied the
empty view, so a missing, insufficient, misaligned, foreign, dead, or
destination-overlapping scratch range fails before the first copied role. A
zero-byte requirement of `{0, 1}` consumes no workspace at all, preserving
the CPU zero-workspace behavior. Retained accelerator backends provision
caller-owned scratch only when their queried requirement is positive.

**Ownership.** The caller creates and owns every destination and any scratch it
provisions from the preflight result. This call allocates no tensor, no
workspace, and no persistent host payload, retains no second checkpoint image,
and neither provisions nor destroys caller resources. Destinations and scratch
must be quiescent and alive for the whole call, and the source must outlive it:
every supplied source mapping stays alive through all synchronous copies. After
a successful call the source may be destroyed once no borrowed source metadata
is used, and the copied device tensors remain independently valid. Any future
aliased or direct-DMA realization must retain its backing independently through
use; it is not implemented here, so today's path adds no DMA handle,
registration, or zero-copy claim.

**Realization failures.** These categories are normative:

| Condition | Exception |
| --- | --- |
| Any complete-binding violation listed by the previous subsection | `std::invalid_argument` or `std::overflow_error` as listed there, before the first copied role |
| Missing, insufficient, misaligned, foreign, dead, or destination-overlapping positive workspace | `std::invalid_argument`, before the first copied role |
| A failed synchronous weight upload | that backend's established category, unchanged |

A validation failure uploads nothing. A later synchronous upload failure
propagates its original category and stops immediately: earlier destinations may
already hold copied bytes, later destinations stay untouched, and every
destination, workspace, and view keeps its caller ownership and its identity.
The call promises neither rollback nor retry and destroys or retargets no caller
resource; only setup-owned RAII resources follow the existing backend safe
release/quarantine rules. No exception category is wrapped at this boundary:
the fixed model-boundary container-JSON translation of the mapped-source
subsection remains the only translation, and `std::bad_alloc`,
`std::overflow_error`, and backend failures pass through unchanged.
