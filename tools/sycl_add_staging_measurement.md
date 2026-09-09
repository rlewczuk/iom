# SYCL ADD staging measurement

## Execution

Measured on the configured Level Zero device, not locally:

- backend: Level Zero
- device: Intel(R) Arc(TM) Pro B60 Graphics
- driver: 1.15.38646+7
- compiler: Intel(R) oneAPI DPC++/C++ Compiler 2026.1.0
- declared benchmark memory budget: 1,073,741,824 bytes
- warm-up: 10 iterations per workload
- measured iterations: 100 per workload
- command: `sycl-ls; build/sycl/sycl_add_staging_measure --iterations 100`

The configured repository contains no non-test `.add(` caller under `src/`; the harness therefore measures the representative backend path but has no production consumer.

## Absolute measurements

The ADD implementation has six USM allocations and six corresponding frees per request, three staging-copy kernel launches, one output-copy kernel launch, three staging memcpys, one output memcpy, and two `wait_and_throw` calls. Those operation counts are source-attributed from `SyclQueue::execute`; allocation/free durations are measured by repeating the same six allocations/frees with the same per-buffer extent. The public token wait interval is reported as the complete ADD/wait wall interval; the implementation does not expose separate wait timestamps. `host_loop_proxy_time_s` is a separately measured byte-loop proxy, not a claim that the private `add_elements` interval is directly observable.

| dtype | shape | logical bytes | staging bytes/buffer | native planes | iterations | min s | median s | p95 s | max s | total s | logical throughput B/s | alloc count | alloc time s | free count | free time s | kernel launches | memcpy count | memcpy bytes | wait calls | complete wait wall s | host-loop proxy s | staging peak bytes | independent baseline median s | host timeline intersection median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | `{8,1024,1024}` | 16,777,216 | 16,777,216 | 8 | 100 | 0.991603526 | 0.994009046 | 1.006803600 | 1.008638110 | 99.6212662 | 16,878,333 | 600 | 0.000883308 | 600 | 0.000049743 | 400 | 400 | 6,710,886,400 | 200 | 99.6212662 | 0.104506758 | 100,663,296 | 0.000014227 | 0.994009046 |
| F32 | `{8,1024,1024}` | 33,554,432 | 33,554,432 | 8 | 100 | 1.338845340 | 1.341118160 | 1.347803060 | 1.350299000 | 134.2152610 | 25,019,743 | 600 | 0.054088014 | 600 | 0.030821802 | 400 | 400 | 13,421,772,800 | 200 | 134.2152610 | 0.258980515 | 201,326,592 | 0.000011842 | 1.341118160 |
| BF16 | `{1,1024,1024}` | 2,097,152 | 2,097,152 | 1 | 100 | 0.122580798 | 0.122732004 | 0.124233283 | 0.124758076 | 12.2995612 | 17,087,247 | 600 | 0.000040589 | 600 | 0.000049542 | 400 | 400 | 838,860,800 | 200 | 12.2995612 | 0.004250090 | 12,582,912 | 0.000004840 | 0.122732004 |
| F32 | `{1,1024,1024}` | 4,194,304 | 4,194,304 | 1 | 100 | 0.167316050 | 0.168010584 | 0.168966001 | 0.171123374 | 16.8132722 | 24,964,523 | 600 | 0.051634447 | 600 | 0.031933850 | 400 | 400 | 1,677,721,600 | 200 | 16.8132722 | 0.010768143 | 25,165,824 | 0.000006713 | 0.168010584 |
| I64 | `{2,33,65}` | 34,320 | 61,440 | 2 | 100 | 0.000731203 | 0.000741793 | 0.000756250 | 0.000769906 | 0.074423275 | 46,266,276 | 600 | 0.012502306 | 600 | 0.018192845 | 400 | 400 | 24,576,000 | 200 | 0.074423275 | 0.000054313 | 368,640 | 0.000005350 | 0.000741793 |

The six-allocation and four-launch counts are per ADD request; the table totals multiply them by 100 measured iterations. `memcpy_bytes` is four source-attributed transfers per request, three staging transfers plus one output transfer. `staging_peak_bytes` is six simultaneous USM buffers at the measured extents. The independent operation is a separate Level Zero `parallel_for`; the baseline is its warmed no-ADD wall time. The host-timeline intersection is reported as an absolute duration, but private queue event timestamps are not part of the public IOM API, so it is not converted into a claimed device-overlap percentage.

## Gate and conclusion

The predeclared gate is unchanged: a phase share of at least 10% of ADD wall time, overlap loss of at least 10 percentage points versus the independent baseline, or staging peak above 5% of the declared 1,073,741,824-byte budget would trigger a separate optimization decision. The measured staging peaks are 0.94%, 1.88%, 0.07%, 0.14%, and 0.03% of the budget respectively. The host-timeline data does not provide a defensible device-overlap percentage without private Level Zero event timestamps, so no overlap gate claim is made.

**Conclusion: no representative production caller (hypothesis falsified).** The configured source tree has no production ADD caller under `src/`; these results are synthetic representative-path measurements only. No production implementation or scheduling change is prescribed, and no optimization task is opened automatically.
