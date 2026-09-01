# Common Multi-Backend Checklist

Use this as a lead generator, not as a requirement to comment on every item.

## Contracts

- [ ] capability predicate matches actual implementation constraints
- [ ] fallback/rejection behavior is explicit
- [ ] shape/dtype/stride/layout/alignment assumptions are validated at the correct layer
- [ ] common code does not accidentally depend on one backend
- [ ] builds with unrelated backends enabled/disabled remain coherent

## Ownership / execution

- [ ] resource owner is clear
- [ ] async work cannot outlive referenced memory/object state
- [ ] producer/consumer ordering is explicit
- [ ] synchronization scope is no broader than required
- [ ] deferred errors are eventually observed
- [ ] cleanup handles partial failure
- [ ] state/caches are correctly keyed for backend/device/context

## Numerical / testing

- [ ] reference/differential test exists for changed semantic path
- [ ] tolerance is dtype/operator appropriate
- [ ] intended backend path is actually exercised
- [ ] boundary shapes/layouts/capability cases are covered

## Performance

- [ ] no accidental host/device fallback or extra transfer
- [ ] no new hot-path global synchronization
- [ ] no avoidable hot-path allocation/compilation
- [ ] launch count/overlap/fusion not unintentionally degraded
- [ ] measured claims use controlled baseline and representative workload

## Simplicity

- [ ] one source of truth for capability/state
- [ ] no duplicated backend dispatch policy
- [ ] no unnecessary representation/conversion/state
- [ ] abstractions own a semantic responsibility
- [ ] simplification does not hide backend-specific correctness/performance semantics
