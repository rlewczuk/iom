# Retire SYCL host-transfer staging only after queue completion

**Order:** 04
**Priority:** P0 — prevent USM staging use-after-free after partial SYCL transfer failure
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase review of main at fb88959fb884b9bb495e055b9fdd0756e430dbde (clean working tree)`
**Finding:** `ST-004`
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 88
**Review scope:** whole-codebase
**Backend scope:** sycl
**Location:** `src/sycl/copy.cpp:667-689` (`region_from_host`), `src/sycl/copy.cpp:692-723` (`region_to_host` failure handling), `src/sycl/staging_pool.cpp:163-186`, `src/sycl/device.cpp:73-85`

## Outcome

SYCL host-transfer staging remains owned until submitted transfer commands are terminal. After any partial enqueue, the transfer queue's completion wait runs before poisoned staging release; `wait_and_throw` returning or reporting an asynchronous error still means submitted commands are terminal, so the secondary async error is captured/discarded and the original enqueue error is rethrown. `SyclDevice::~SyclDevice` waits the transfer queue before destroying the staging pool, capturing any async error before destruction.

## Current problem

The invariant is that staging referenced by an enqueued SYCL host transfer must not be freed or reused until the transfer queue proves command completion. `src/sycl/copy.cpp:667-689` builds host regions, and `src/sycl/copy.cpp:692-723` catches a later enqueue failure and calls `lease.poison()`. The current path can release poisoned USM while an earlier plane remains queued, because `src/sycl/staging_pool.cpp:163-186` frees poisoned storage immediately.

Teardown compounds the defect: `src/sycl/device.cpp:73-85` destroys staging before waiting on `transfer_queue_->wait_and_throw()`. The repair uses the SYCL queue contract directly: wait before poisoned release and before pool destruction, capture/discard only the secondary asynchronous error, and preserve the original enqueue error.

## Scope

- After any partial enqueue, call the transfer queue's completion wait before poisoned staging release; `wait_and_throw` returning or reporting an asynchronous error is the terminality proof for submitted commands.
- Capture and discard a secondary asynchronous error from that completion wait, then rethrow the original enqueue error.
- In `SyclDevice::~SyclDevice`, run `transfer_queue_->wait_and_throw()` before `staging_pool_.destroy()`, capture the async error, then destroy the pool.
- Do not add a global synchronization to successful transfers; preserve original enqueue exceptions, host/device transfer semantics, and staging pool API shape.

## Implementation references

- **Modify:** `src/sycl/copy.cpp:667-723` — `region_from_host`, `region_to_host`, and catch paths; wait for transfer completion before poisoned staging release, capture the secondary async error, and rethrow the original enqueue error.
- **Modify:** `src/sycl/device.cpp:73-85` — `SyclDevice::~SyclDevice`; wait the transfer queue before `staging_pool_.destroy()`, capture the async error, then destroy.
- **Read:** `src/sycl/staging_pool.cpp:163-186` — poisoned release behavior; confirm the completion wait precedes its existing free without adding a new retirement mechanism.
- **Read:** SYCL queue/fence completion and existing no-throw cleanup conventions.
- **Tests:** SYCL host-transfer failure/conformance coverage; add partial enqueue failure and teardown ordering scenarios.

## Requirements

- Treat every host transfer enqueue that may have succeeded as in flight, including an enqueue that throws after prior planes were submitted.
- Call the transfer queue's completion wait before poisoned staging release; an asynchronous error reported by `wait_and_throw` is secondary because submitted commands are terminal.
- Capture/discard the secondary wait error and rethrow the original enqueue exception.
- Teardown must call `transfer_queue_->wait_and_throw()` before `staging_pool_.destroy()`, capture the async error, then destroy the pool.
- Do not synchronize successful transfers globally; preserve transfer representation, tiled layout, logical byte results, queue API, and successful-path performance.

## Non-goals

- Do not change SYCL host-transfer layout, conversion kernels, queue API, or general USM allocator policy.
- Do not change unrelated device teardown or add a global synchronization to successful transfers.
- Do not make a failed transfer silently succeed or replace the original enqueue error.

## Acceptance criteria

- [ ] A partial host enqueue failure waits for transfer completion before poisoned staging is freed, captures/discards any secondary async error, and rethrows the original enqueue error.
- [ ] A later transfer cannot observe staging freed before the earlier submitted commands are terminal, and no new retired-storage mechanism or global successful-path synchronization is introduced.
- [ ] Device teardown waits the transfer queue before staging destruction, captures any async error, then destroys the pool without freeing in-flight USM.
- [ ] A clean follow-up host round trip preserves exact logical bytes and existing SYCL smoke/conformance behavior.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — local common/CPU gates remain green.
- `.agents/skills/remote-development/scripts/remote-sync sycl 04-ST-004-sycl-host-transfer-staging-retirement`
- `.agents/skills/remote-development/scripts/remote-exec sycl 04-ST-004-sycl-host-transfer-staging-retirement 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake -S . -B build-sycl -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-sycl --target iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-sycl --output-on-failure -R "iom_sycl_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — SYCL devices enumerate and smoke, conformance, and coexistence pass.
- On SYCL hardware, inject a partial enqueue failure and an asynchronous queue error; expect `wait_and_throw` to establish terminality, the secondary error to be captured/discarded, the original enqueue error to be rethrown, and exact follow-up bytes. Verify teardown waits before staging destruction and run supported SYCL runtime memory diagnostics if available; expect no access after USM release.
