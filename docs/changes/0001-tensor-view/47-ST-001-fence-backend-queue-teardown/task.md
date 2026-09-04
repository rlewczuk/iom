**Status:** blocked

## Summary

Execution is blocked by the unresolved prerequisite tasks `27-ST-006-bound-rocm-failure-wait` and `30-AR-002-share-backend-queue-scaffold`. Neither prerequisite has a `task.md` annotation with `Status: done` on the integration branch, and the current source still lacks the shared `StagedWorker` queue scaffold required by this task.

## Errors

- `27-ST-006-bound-rocm-failure-wait` — no `task.md` with `Status: done`; ROCm source still contains the pre-scaffold retained-failure event-record path.
- `30-AR-002-share-backend-queue-scaffold` — no `task.md` with `Status: done`; `StagedWorker`/`shutdown_and_drain` is absent from the current integration source.

The exact worktree and feature branch are retained for a later run after both blockers are completed and integrated.
