**Status:** blocked

## Summary

No implementation was attempted because the task's required queue teardown and CPU/TTNN task-view prerequisites are not complete on the integration branch.

## Errors

- `47-ST-001-fence-backend-queue-teardown` — no `task.md` with `Status: done` is present on the integration branch; its retained worktree is not integrated.
- `48-ST-002-own-cpu-ttnn-task-views` — no `task.md` with `Status: done` is present on the integration branch; the prerequisite is not integrated.
