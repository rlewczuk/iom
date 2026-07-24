---
name: specflow-tdd
description: Use during SpecFlow implementation; enforces test-first RED-GREEN-REFACTOR, narrow evidence, regression verification, and no completion claims without fresh test output.
---

# SpecFlow test-driven development

## Rule

Write the behavior-defining test first whenever the repository and task permit
automated testing.

```text
RED → confirm the intended failure → GREEN → confirm passing → REFACTOR → confirm passing
```

## RED

- Write the smallest test that expresses the next required behavior.
- Run the narrowest relevant command.
- Confirm it fails for the expected reason, not from setup or syntax errors.
- Preserve the command and meaningful failure evidence.

If the test passes before implementation, determine whether behavior already
exists or the test is inadequate.

## GREEN

- Implement the minimum production change that satisfies the test.
- Avoid unrelated cleanup or speculative abstractions.
- Run the narrow test and confirm it passes.

## REFACTOR

- Improve clarity and remove duplication without extending scope.
- Run the narrow test again.
- Run the task's relevant regression suite.

## Exceptions

If a task cannot reasonably be test-first—for example, a documentation-only or
pure formatting task—record the reason and use deterministic verification.

Do not treat manual inspection as equivalent to executable tests when automated
tests are feasible.

## Evidence

Implementation results must include:

- commands run;
- RED evidence;
- GREEN evidence;
- regression result;
- untested behavior or limitations.

A stale test result from before the final edit is not completion evidence.
