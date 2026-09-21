# 5. Queue and token contract

1. `DeviceOps` is non-copyable/non-movable and represents one in-order queue.
   The caller serializes calls on a queue. The backend may synchronize or use
   asynchronous runtime work internally, but observable completion and
   visibility order is submission order.
2. Every successful submission uses the common facade and returns a positive
   `oid = static_cast<oid>((std::uint64_t{q} << 55) | sequence)`. Queue ID `q`
   is process-unique in the complete range `[1,255]`, represented in bits
   `55..62`; `sequence` is exactly 55 bits, begins at one, and is in
   `[1,2^55-1]`. Sequence zero is never submitted.
3. OID facades are common `noexcept` entry points. They validate, map errors,
   encode tokens, and register lifetimes before backend effects or acceptance;
   protected backend hooks cannot bypass those obligations. Synchronous
   failures return exactly `InvalidArgument=-1`, `Unsupported=-2`,
   `Overflow=-3`, `ResourceExhausted=-4`, `DeviceError=-5`, or
   `InternalError=-6`. Invalid input, unavailable operation/specification,
   checked arithmetic or sequence exhaustion, bounded-resource failure,
   pre-acceptance runtime failure, and otherwise unclassified failure map to
   those values respectively. No synchronous exception crosses an OID facade.
4. Sequence exhaustion returns `Overflow` before backend work, effects, or
   token acceptance. Every accepted submission returns one positive token.
   Failures after acceptance remain attached to that token and `wait` rethrows
   them repeatably; successful completion and visibility are repeatable.
5. `wait(token)` blocks for accepted work and immediately throws
   `std::invalid_argument` for a negative value, zero, a foreign queue token,
   a future value, a skipped/reserved-but-never-submitted value, or any other
   unsubmitted value. A skipped value remains invalid after later completion.
   Completing sequence *n* makes earlier submitted work observable as complete.
6. The common base releases a queue ID at destruction; it does not define an
   implicit wait or cancellation of outstanding work. A concrete backend MUST
   nevertheless perform runtime cleanup preventing use-after-free of streams,
   events, storage, and lifetime registrations. Callers must not use a token
   or queue after its owner is destroyed.
7. Backend queues MUST retain enough state to complete and fail submitted work
   correctly even if their worker observes a runtime error after launch. A
   completion proof can be a runtime event, successful stream drain, or
   equivalent native fence. If no proof is available, affected storage and
   metadata MUST be quarantined/retired rather than reused.
