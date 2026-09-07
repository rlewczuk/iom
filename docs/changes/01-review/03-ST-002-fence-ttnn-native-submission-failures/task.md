**Status:** done

## Summary

TTNN native submission failures are fenced: ownership (fence + registry entries + outcome) is registered before any native plane reaches the mesh, pre-submission throws roll the entries back, the mesh drains synchronously under the API mutex before a post-launch failure rethrows, and an undrainable finish publishes a waitable token that repeatedly reports the retained submission failure; copy_planes reports any-submitted and complete_task retries finish_native composing worker/retained/finish failures. Four failure-injection conformance cases cover zero/one/many-plane and registration/outcome/finish failure paths.

## Verification

- ctest -R "iom_ttnn_(smoke|conformance)_tests" on bv1 — 2/2 passed, conformance 18.11s
- combined train on bv1: iom_ttnn smoke+conformance+coexistence 3/3 passed, conformance 17.20s
