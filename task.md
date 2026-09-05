**Status:** blocked

## Summary

No implementation was attempted because the requested task has three unfinished prerequisites.

## Errors

- `29-AR-001-share-cuda-rocm-copy-queue` — no `task.md` with `Status: done`; its branch is reachable from `main`, but the required shared queue implementation is not present in the integration tree.
- `28-ST-007-commit-sequence-before-queue-work` — no `task.md` with `Status: done`; its deterministic feature branch is absent.
- `27-ST-006-bound-rocm-failure-wait` — no `task.md` with `Status: done`; its deterministic feature branch is absent.

The deterministic worktree and feature branch are retained for a later run after all blockers are completed.
