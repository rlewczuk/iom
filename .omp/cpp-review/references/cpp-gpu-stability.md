# C++ / GPU Stability Reference

## Ownership model

For each backend resource determine:

- who creates it;
- who owns destruction;
- whether ownership transfers;
- whether handles are device/context qualified;
- whether cleanup is valid on every exit path;
- whether destruction may synchronize or fail.

Prefer RAII for C++ resource ownership, but remember that RAII only solves **host ownership**. It does not by itself prove an asynchronously referenced allocation is no longer in use.

## Async-lifetime model

Represent resource validity as:

```text
host ownership lifetime
        ∩
device in-flight use lifetime
        ∩
backend/context validity lifetime
```

A free/reuse is safe only after all relevant lifetimes allow it.

## Synchronization review

For each producer-consumer edge:

1. identify producer queue/stream;
2. identify consumer queue/stream/host;
3. identify the primitive that establishes ordering;
4. verify visibility semantics, not only chronological API calls;
5. ensure the dependency is no broader than necessary.

Global/device synchronization in a destructor/accessor should receive extra scrutiny because it can hide lifetime defects and serialize performance.

## Deferred errors

GPU runtimes often enqueue work asynchronously. A successful launch/enqueue return does not necessarily mean device execution succeeded. Verify where asynchronous errors become observable and that error state cannot be misattributed to a later unrelated call.

## Concurrency

Treat libraries as potentially called concurrently unless project contracts forbid it. Minimize writable process-global state. Device/backend caches should be keyed by all context necessary for correctness (for example device ordinal/context/architecture/driver-dependent compilation parameters as appropriate).

## Integer/shape safety

Inference engines frequently multiply dimensions, strides, element sizes, batch counts, and allocation alignment. Review operations before casts and before narrowing into kernel arguments.
