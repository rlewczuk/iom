---
name: cpp-inference-gpu-stability
description: Review C++ inference-engine code for ownership/lifetime, asynchronous GPU execution, synchronization, memory visibility, concurrency, error propagation, cleanup, integer-size safety, and multi-device stability across GPU backends. Use independently or as Area 2 of cpp-inference-code-review.
argument-hint: "[scope] [backend(s) optional]"
---

# C++ / GPU Stability Review

Treat asynchronous accelerator execution as a first-class correctness model. C++ lexical scope does **not** prove device work has completed.

Read `.agents/cpp-review/references/cpp-gpu-stability.md`, `.agents/cpp-review/references/finding-rubric.md`, `.agents/cpp-review/checklists/common.md`, and the checklist for every affected backend.

## Review order

### 1. Resource ownership

For every changed or central resource, establish:

- resource type: buffer/allocation, stream/queue, event/fence, module/kernel, graph, descriptor, command object, mapped memory, cache entry, context/device handle;
- owner vs borrower;
- move/copy behavior;
- partial-construction cleanup;
- destruction thread/context/device assumptions;
- whether RAII correctly matches the backend acquire/release protocol.

Raw handles are acceptable only when ownership and lifetime are still explicit.

### 2. Asynchronous lifetime

For each async enqueue/copy/kernel/graph operation ask:

- Which host/device resources remain referenced after the API call returns?
- What event/fence/queue completion establishes safe reuse or destruction?
- Can allocator reuse race with in-flight work?
- Can pageable/pinned host memory disappear before an async transfer completes?
- Can callbacks capture destroyed state?
- Does a temporary command/descriptor object outlive submission requirements?

### 3. Ordering and synchronization

Trace producer → dependency → consumer. Hunt for:

- missing event/fence/dependency;
- event recorded/waited on the wrong queue;
- default/global stream assumptions;
- whole-device synchronization where a narrower dependency suffices;
- synchronization inserted in destructors or hot-path accessors;
- missing memory visibility/barrier semantics;
- graph-capture incompatibilities;
- cross-device events/handles used with invalid semantics.

Treat **missing synchronization** as correctness risk and **excess synchronization** as potential performance risk. Cross-reference performance review rather than duplicating symptoms.

### 4. Concurrency and state

Inspect writable shared state such as:

- allocator pools;
- kernel/module caches;
- compiled graph caches;
- per-device capability caches;
- initialization flags;
- mutable singleton/backend registries;
- queue pools and scratch arenas.

Check whether state is correctly keyed by device/context/backend and whether supported concurrent calls can race.

### 5. Error propagation

Accelerator enqueue APIs may return before device execution fails. Verify:

- launch/enqueue errors are observed;
- deferred execution errors are eventually surfaced;
- synchronization/reporting does not attribute an old error to an unrelated later operation;
- exceptions/status values cannot bypass required cleanup;
- error paths do not continue with partially valid device state.

### 6. Size and arithmetic safety

Check shape/stride/byte computations for:

- signed/unsigned conversion;
- multiplication/addition overflow;
- truncation to 32-bit kernel/index parameters;
- alignment round-up overflow;
- negative dimensions converted to huge sizes;
- host/device type-width mismatches.

### 7. Multi-device state

When multiple devices are supported, verify:

- explicit current-device/context assumptions;
- handles/events/allocations belong to the correct device;
- caches are device-qualified;
- peer-to-peer support is capability checked;
- fallback for unavailable peer access is correct;
- per-thread device selection is not leaked unexpectedly;
- synchronization does not accidentally serialize all devices.

## Evidence

Prefer concrete call-path reasoning plus backend documentation/tool evidence. Static analysis or sanitizers are corroboration, not substitutes for understanding the lifetime/order model.

Use IDs `ST-###`. If the same root cause also creates a performance problem, keep one stability finding and cross-reference the performance impact in synthesis.
