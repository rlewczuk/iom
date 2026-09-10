**Status:** done

## Summary

TTNN binary completion now has one drain owner: the required native drain publishes the common completion marker, first and repeated waits add no drain, and pre-proof failures retain quarantine protection.

## Verification

- TTNN focused completion-count and native finish-failure/un-drainable-failure remote tests — all three selected tests passed.
- Combined final train TTNN conformance and backend coexistence — both targets passed.
