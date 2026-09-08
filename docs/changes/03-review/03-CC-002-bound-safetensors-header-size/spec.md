# Bound the SafeTensors header before conversion and parsing

**Order:** 03
**Priority:** P0 — enforce the format's parser DoS boundary at the untrusted length read
**Blocked by:** None
**Review source:** `cpp-inference-contract-correctness` — `whole-codebase reviewed state: branch main, clean HEAD 86533aef347935405cb86d465cc5489f5c3d530a (Remove review files.)`
**Finding:** `CC-002`
**Review area:** Contract & correctness
**Review severity:** high
**Review verification:** verified, confidence 95
**Review scope:** whole-codebase
**Backend scope:** common SafeTensors parser (`libiom`, `SafeTensorsDir`, and all enabled consumers)
**Location:** `src/safetensors.cpp` — `SafeTensorsFile::SafeTensorsFile`, untrusted `header_len_u64` handling

## Outcome

`SafeTensorsFile::SafeTensorsFile` rejects any untrusted header length greater than 100,000,000 bytes before converting it to `size_t`, computing ranges, or invoking JSON parsing, while retaining portability and file-bound guards. The inclusive 100,000,000-byte boundary remains permitted by the cap, valid ordinary headers continue to load, and cap-plus-one rejection uses the existing stable `std::runtime_error` style without changing ranges, metadata, or mmap behavior.

## Current problem

The constructor reads a u64 header length, checks only whether it fits `size_t` and whether it fits within the mapped file, then passes the entire declared range to `nlohmann::json::parse`. A file whose declared header is over the SafeTensors reference limit but fits the platform and file bounds therefore reaches JSON parsing, enabling unnecessarily large attacker-controlled JSON work and violating the format's 100 MB DoS-protection contract. The authoritative SafeTensors reference defines `MAX_HEADER_SIZE = 100,000,000` and rejects larger values before slicing or parsing. Existing tests cover malformed and bounded headers but do not exercise the cap without allocating a permanent 100 MB fixture.

## Scope

- Add one anonymous-namespace 100,000,000-byte format cap and compare `header_len_u64` against it before `size_t` conversion, range arithmetic, and JSON parse.
- Preserve the existing size_t representability and `header_len <= file_.size() - 8` guards and stable `std::runtime_error` behavior for oversized/truncated input.
- Add a focused sparse cap-plus-one test and a normal valid-header regression. Use sparse file sizing and bounded writes; do not modify range validation, metadata typing, or mmap.

## Implementation references

- **Modify:** `src/safetensors.cpp` — anonymous-namespace constants and `SafeTensorsFile::SafeTensorsFile` lines around `header_len_u64`; this constructor owns the common untrusted length boundary.
- **Read:** `src/mmap.cpp` — mapped file size/data behavior; retain file-bound protection and mapping ownership.
- **Read:** SafeTensors reference `MAX_HEADER_SIZE` and `read_metadata`; reuse the authoritative inclusive cap semantics (`>` 100,000,000).
- **Tests:** `test/test_safetensors.cpp` — SafeTensors parser error fixtures and temporary-file helpers; add only bounded sparse cap coverage and a normal valid-header regression.

## Requirements

- Define the cap as an anonymous-namespace constant with value `100'000'000` and reject `header_len_u64 > cap` while it is still u64. This check MUST precede `size_t` conversion, `file_.size()` range calculations, header pointer arithmetic, and `nlohmann::json::parse`.
- Keep the existing portability guard for values that exceed `size_t` and the file-bound guard for truncated files. Preserve stable `std::runtime_error` classification and the existing filename-bearing error style.
- The test MUST create a file with only a fixed-size header/zero buffer and sparse resize to satisfy the declared cap-plus-one length; it MUST prove cap-plus-one is rejected before JSON parsing without constructing a 100 MB heap string or permanent large fixture.
- Include a normal small valid SafeTensors header regression proving ordinary parsing and tensor payload access remain unchanged. Do not alter range/metadata/mmap behavior.

## Non-goals

- Do not change data-offset ordering/coverage, metadata typing, dtype mappings, tensor payload validation, JSON parser configuration, or mmap implementation.
- Do not add a public constant, generic parser abstraction, large checked-in fixture, profiler, allocator experiment, or 100 MB parser stress benchmark.
- Do not duplicate the cap in `SafeTensorsDir` or backend callers; the constructor remains the single source of truth.

## Acceptance criteria

- [ ] A file whose declared header length is 100,000,001 bytes and whose sparse file bounds are satisfied throws the oversized-header `std::runtime_error` before JSON parsing; the test allocates only bounded memory and leaves no large permanent fixture.
- [ ] A declared length of exactly 100,000,000 is not rejected by the cap (`>` rather than `>=`), and a small ordinary valid header still parses and exposes its expected tensor payload.
- [ ] Values exceeding `size_t` and truncated file lengths continue to take their existing portability/file-bound error paths, with no changes to ranges, metadata, or mmap semantics.

## Verification

- `cmake --build build --target iom_tests && ctest --test-dir build -R '^iom_tests$' --output-on-failure`
- Run the focused sparse cap boundary and normal valid-header regression. Expected: cap-plus-one fails before JSON parsing with stable `std::runtime_error`, exact cap is not rejected by the cap itself, no 100 MB heap fixture is allocated, and existing SafeTensors tests remain green. The root ledger reports no cap-specific stress run; this future focused gate has not been run.
- Baseline validation ledger only (not CC-002 coverage): local GNU 15.2 CPU build/tests/bench/conformance passed 4/4; remote CUDA 13.2.78, ROCm HIP Clang 23, SYCL IntelLLVM 2026.1 (on two enumerated Arc Pro B60 Level Zero GPUs), and TTNN smoke+conformance each passed 2/2. No cap-specific sparse test, 100 MB parse stress, sanitizer, or static-analysis gate has been run.
