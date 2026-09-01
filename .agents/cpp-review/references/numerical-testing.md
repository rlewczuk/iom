# Numerical Correctness & Testing Reference

## Cross-backend oracle

Prefer differential testing of the same semantic operation against a trusted/reference implementation and across all registered compatible backends.

Do not assume bitwise identity for floating-point parallel implementations. Operation ordering, fused instructions, reduced precision, fast math, and hardware libraries can legitimately change low-order bits.

## Tolerance policy

Use tolerance derived from the operation and dtype. Consider absolute and relative tolerance, output scale, reduction length, quantization, accumulation precision, and special-value behavior.

A tolerance is part of the test contract and should be reviewable—not an unexplained constant copied between tests.

## Test should prove path selection

When backend dispatch/fallback is relevant, numerical equality alone can pass while accidentally executing the reference/CPU path. Tests should verify that the intended backend implementation was selected when feasible.

## Useful edge categories

- tile/workgroup boundaries;
- odd/small/zero dimensions;
- non-contiguous strides;
- broadcasts;
- alignment offsets;
- large indexing ranges;
- mixed/low precision;
- quantized boundary values;
- backend capability boundaries;
- nondeterministic reductions where supported.
