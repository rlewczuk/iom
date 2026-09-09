**Status:** done

## Summary

Type-aware TTNN floating ADD comparison envelope now distinguishes finite, signed zero, infinity, and NaN classes, accepts one ULP finite drift, and rejects deterministic mutations.

## Verification

- TTNN focused mutation case on bv1:agent-work/iom/01-review-nt002-ttnn: 1 passed, 36 assertions, 31 skipped.
- TTNN full conformance on bv1:agent-work/iom/01-review-nt002-ttnn-full: 1/1 CTest passed in 50.78 s.
