# Enumerate TTNN compact ADD pairs

**Order:** 07
**Priority:** P1 — compact codec path untested
**Blocked by:** None
**Review source:** `cpp-inference-numerical-testing` — `whole-codebase checked-out main/HEAD ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb ("Update remote hosts"), clean working tree at review start`
**Finding:** NT-003
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** verified, confidence 99
**Review scope:** whole-codebase
**Backend scope:** TTNN
**Location:** `test/ttnn/test_ttnn_conformance.cpp:1489-1500,1539-1556` — `exhaustive_pairs` and `TTNN ADD exhaustively covers every compact leaf pair`

## Outcome

The TTNN compact ADD test supplies distinct left- and right-hand vectors containing every ordered raw-value pair for each 2-, 4-, and 6-bit compact leaf. The test checks all 16, 256, or 4096 pairs against the independent oracle using the intended `[1,32,32]` or `[1,64,64]` logical shape, so the compact codec and ADD path are no longer represented by only `(0,0)`.

## Current problem

`exhaustive_pairs(width, plane_elements)` reserves and appends two values per element, but the test calls it with `plane_elements=1` and passes the same resulting vector as both `lhs_values` and `rhs_values` at `test/ttnn/test_ttnn_conformance.cpp:1553-1555`. Every compact dtype therefore runs one logical element with both operands equal to zero. The test name and claim say exhaustive coverage while the actual input omits every nonzero and non-identical ordered pair, leaving compact encode/decode and arithmetic failures undetected.

## Scope

- Replace the misleading pair generator and its call site with distinct `lhs_values` and `rhs_values` vectors containing a true Cartesian product in deterministic order.
- Cover I2/U2 with 16 ordered pairs, I4/U4/F4_E2M1 with 256 ordered pairs, and F6_E2M3/F6_E3M2 with 4096 ordered pairs.
- Preserve the existing logical-shape selection: 2- and 4-bit cases use `[1,32,32]`; 6-bit cases use `[1,64,64]`; every pair is checked against the independent oracle.

## Implementation references

- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — the compact-pair generator near `exhaustive_pairs` and the `TTNN ADD exhaustively covers every compact leaf pair` test; these own compact input construction and invocation.
- **Read:** `test/ttnn/test_ttnn_conformance.cpp` — `add_leaf_spec`, `require_add_leaf_matches_oracle`, `set_add_logical_value`, and `get_add_logical_value`; reuse their shape, logical encoding, token, wait, and per-element observation conventions.
- **Read:** `test/backend/backend_conformance_add.hpp` — `add_oracle::add`; expected compact results must remain independent from TTNN production arithmetic.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — the compact exhaustive test on the configured TTNN hardware; retain the separate representative-wide and mapping tests without folding them into this generator.

## Requirements

- Remove the `exhaustive_pairs(width, plane_elements)` name and implementation because it does not generate pairs. Introduce one clearly named helper that returns or fills two distinct vectors, with `lhs_values.size() == rhs_values.size() == (1u << width) * (1u << width)`.
- For each flat index `i`, set `lhs_values[i] = i / (1u << width)` and `rhs_values[i] = i % (1u << width)`. This ordering must enumerate every ordered pair exactly once, including `(0,max)`, `(max,0)`, and all equal pairs, without relying on identical vector objects.
- Pass the two distinct vectors to `require_add_leaf_matches_oracle`; do not pass one vector for both operands and do not reduce the vectors to a sample or representative subset.
- Keep the compact type list and width mapping exact: I2/U2 width 2; I4/U4/F4_E2M1 width 4; F6_E2M3/F6_E3M2 width 6. The resulting shape must remain `[1,32,32]` for widths 2 and 4 and `[1,64,64]` for width 6, as selected by `add_leaf_spec`.
- Retain the independent per-element oracle comparison and corrected floating envelope from NT-002. A failure for any pair must identify the dtype, flat pair index, raw operands, expected result, and observed result.
- Update the test wording and helper comments so they claim exhaustive ordered-pair coverage, not the removed generator's misleading plane-element behavior.

## Non-goals

- Do not alter TTNN production ADD, compact encoding definitions, dtype support, broadcast mapping, transformed views, aliases, or representative wide-value coverage.
- Do not duplicate the GPU low-width helper or move this TTNN-specific Cartesian matrix into common backend tests.
- Do not replace exhaustive coverage with randomized, sampled, commutative, or same-operand-only cases.

## Acceptance criteria

- [ ] I2 and U2 execute and check exactly 16 ordered pairs each; I4, U4, and F4_E2M1 execute and check exactly 256 each; F6_E2M3 and F6_E3M2 execute and check exactly 4096 each.
- [ ] LHS and RHS are distinct vectors, and the observed output for every flat index is compared with `add_oracle::add` using the corresponding ordered raw operands.
- [ ] The compact tests use `[1,32,32]` for widths 2/4 and `[1,64,64]` for width 6, and include unequal, reversed, and maximum-value pairs rather than only `(0,0)`.
- [ ] A deliberate compact output mutation or wrong ordered operand fails the test, proving that the Cartesian matrix is value-sensitive and not merely a token/shape smoke check.
- [ ] The helper name, test name, and diagnostics no longer claim exhaustive coverage for a one-element generator or imply that one vector supplies both operands.

## Verification

- `cmake --build <configured TTNN build> --target iom_ttnn_conformance_tests` — proposed focused build; not run for this specification-writing task.
- **Remote TTNN hardware:** use `.agents/skills/remote-development/scripts/remote-sync ttnn <unique-task-id>`, then `.agents/skills/remote-development/scripts/remote-exec ttnn <unique-task-id> 'cmake --build <configured-build> --target iom_ttnn_conformance_tests && ctest --test-dir <configured-build> --output-on-failure -R "^iom_ttnn_conformance_tests$"'`; expected observation is a passing run that executes all 16/256/4096 ordered pairs and the stated shapes.
- Exercise the focused doctest case `TTNN ADD exhaustively covers every compact leaf pair` remotely and inspect its per-type element/pair counts; expected observations include nonzero unequal operands and no `(0,0)`-only pass path. No local accelerator result substitutes for this remote check.
