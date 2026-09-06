# Common Multi-Backend Checklist

Use this as a lead generator, not a quota. Every accepted item needs the candidate rubric's evidence and remediation seed.

## Contracts

- [ ] capability predicate matches implementation constraints
- [ ] fallback/rejection behavior is explicit
- [ ] shape/dtype/stride/layout/alignment assumptions are validated at the owning layer
- [ ] common code does not accidentally depend on one backend
- [ ] builds with unrelated backends enabled/disabled remain coherent
- [ ] one source owns each contract; validation/fallback policy is not duplicated

## Ownership and execution

- [ ] resource owner and borrower are explicit
- [ ] async work cannot outlive referenced memory/object state
- [ ] producer/consumer ordering and visibility are explicit
- [ ] synchronization scope is no broader than required
- [ ] deferred errors remain observable and correctly attributed
- [ ] cleanup handles partial failure
- [ ] state/caches are keyed by required backend/device/context facts
- [ ] one lifetime/completion mechanism exists per responsibility

## Numerical behavior and tests

- [ ] independent reference/differential coverage protects changed semantics
- [ ] tolerance is operator/dtype appropriate
- [ ] intended backend path is actually exercised
- [ ] boundary layouts/shapes/capability transitions are covered
- [ ] shared conformance behavior is not copied into backend-specific tests
- [ ] tests defend observable behavior rather than wiring or source structure

## Performance

- [ ] no accidental fallback, transfer, or conversion
- [ ] no hot-path global synchronization
- [ ] no avoidable hot-path allocation, compilation, validation, or capability query
- [ ] launch count, overlap, fusion, and memory footprint are not unintentionally degraded
- [ ] measured claims use controlled representative baselines
- [ ] remediation removes work before adding caches, schedulers, or configuration

## Simplicity and deletion

- [ ] one source of truth exists for capability, state, dispatch, and fallback
- [ ] no semantically duplicated backend implementation or cleanup protocol
- [ ] no unnecessary representation, conversion, state, wrapper, registry, cache, or extension point
- [ ] abstractions own a current invariant rather than hypothetical reuse
- [ ] caller special cases are not compensating for a missing owner-level contract
- [ ] invalid state combinations are removed or made unrepresentable
- [ ] dead compatibility/configuration/test machinery is deleted after dynamic use is checked
- [ ] simplification preserves backend-specific correctness, diagnostics, lifetime, and performance
- [ ] the proposed change reduces net concepts rather than moving code