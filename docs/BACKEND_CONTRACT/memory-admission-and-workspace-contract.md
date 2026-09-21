# 6. Memory, admission, and workspace contract

For CUDA, ROCm, and SYCL, setup MUST reserve exactly two IOM-native backing
allocations: the caller-selected tensor-data arena and a separate metadata
arena with checked capacity `4 * C * 512` bytes. `C` is the immutable,
nonzero `QueueConfig::max_in_flight_per_queue`; the metadata arena is
device-wide and one `FixedSizeAllocator` spans exactly `4 * C` 512-byte,
32-byte-aligned slots. The data arena uses one device-owned coalescing
allocator. Tensor and `RawWorkspace` addresses MUST stay in the data domain;
descriptor and metadata-slot addresses MUST stay in the metadata domain.

An arena suballocation changes allocator bookkeeping only; it is not a native
allocation. A metadata-slot lease, completion resource, host allocation,
caller workspace range, and accepted token are separate ownership records.
After setup, operations, transfers, views, waits, retirement, queue
recreation, parking, and failure recovery MUST make no additional IOM-native
device allocation/free calls and MUST NOT grow or resize fixed resource
arrays. Vendor/SDK-internal allocations are outside that IOM boundary and
MUST be reported separately rather than treated as IOM evidence.

Every accepted request retains its immutable host snapshot, token outcome,
owner registrations, and workspace lease. At most `C` requests per queue hold
native credits; later requests are host parked with no native effect and are
dispatched strictly FIFO when a completion proof returns a credit. No-op,
inline, and no-metadata requests use the same admission order and MUST NOT
bypass a parked head. Completion resources, metadata leases, and workspace
ranges are reusable only after proven completion. Unknown native use MUST
quarantine the complete unresolved lease, including the queue partition and
queue-count reservation, until that lease's own covering proof.

Positive workspace requirements MUST be queried through the pure operation or
transfer query before submission. A missing, undersized, misaligned, stale,
foreign, overlapping, or already-leased range is invalid or resource
exhausted as specified by the public facade; the caller retains ownership
through proven completion. CPU retains borrowed host/reference storage
without a device metadata arena or fabricated native slots. Retained
accelerator backends report vendor-internal runtime allocation separately from
IOM evidence whenever the runtime does not expose a comparable observation
boundary.

The four-live-queue cap and quota are local to the exact `Device`; there is no
global ordinal cap, active-backend registry, or queue selector. Two-/eight-GPU
isolation and matched-baseline first-use/warmed performance are
environment-dependent evidence only: record them when matching hardware and
baseline exist, otherwise report the unavailable result as a non-universal
risk. Report serialized host-transfer throughput separately from compute
throughput. This repository has no `examples/` directory, so caller audits
must record that fact rather than adding an example.
