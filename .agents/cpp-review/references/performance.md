# Performance Review Reference

## Primary rule

First determine **where time or memory is lost**, then optimize the responsible kernel/path.

## System-level before kernel-level

Timeline/system profiling answers:

- did synchronization increase?
- did transfers/fallback appear?
- was overlap lost?
- are launches fragmented?
- are allocations/compilations occurring on hot paths?
- did work move to another device/backend?

Kernel profiling answers:

- is the hot kernel bandwidth/compute/latency limited?
- are memory accesses/coalescing/cache behavior poor?
- is register/shared/local-memory pressure limiting useful throughput?
- is divergence or launch geometry material?

Do not infer normal concurrency from instrumentation that serializes dispatches.

## Inference-specific workload separation

Measure separately when applicable:

- model load / compilation / graph capture;
- prompt processing / prefill;
- token generation / decode;
- batch throughput;
- memory footprint / peak scratch;
- multi-device scaling.

A cost paid once at startup is not equivalent to a cost paid once per token.

## Performance regression evidence

Strong review evidence includes:

- baseline vs target benchmark under controlled conditions;
- profiler timeline showing added synchronization/copy/fallback;
- code proving an unconditional critical-path round trip or global synchronization;
- allocation/launch count growth tied to representative workload.

For uncertain issues, define the exact benchmark/profile experiment that would confirm or reject the hypothesis.
