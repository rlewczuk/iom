# 12. Backend integration and conformance obligations

The following source map is executable contract coverage. Shared scalar,
common validation, rank, queue, workspace, and memory-boundary scenarios live
in `test/backend/backend_conformance_common.hpp`,
`test/backend/backend_conformance_memory.hpp`, and
`test/backend/backend_conformance_add.hpp`; storage, transforms, tails, padding,
and copies live in `test/backend/backend_conformance_copy_storage.hpp` and the
independent `AcceleratorStorageOracle`; model loading and weight realization
live in `test/backend/backend_conformance_model_loading.hpp` with its
`test/model_loading_fixture.hpp` checkpoint fixture, registered for CPU as the
`CPU model loading*`. Backend-local targets are
`iom_backend_conformance_cpu_tests`, `iom_cuda_conformance_tests`,
`iom_rocm_conformance_tests`, and `iom_sycl_conformance_tests`. The accelerator
targets are registered by
`add_iom_backend_tests`; the CPU conformance executable is created directly
in `test/CMakeLists.txt` and is not created through that helper.
`test/backend/test_backend_coexistence.cpp` and target
`iom_backend_coexistence_tests` provide the combined coexistence gate.

The implemented SiLU operation has one shared executable source map,
`test/backend/backend_conformance_silu.hpp`, whose named cases cover its ABI,
pure zero-workspace query, shape and padding mapping, all 23 leaves and the
final backend matrix, admission and overflow, stable arithmetic and special
values, accepted failures and repeat waits, and the stored-result
`SiLU -> mul -> add` composition. Existing backend drivers consume that shared
header; no second SiLU test project is permitted.

Planned shared cache append coverage belongs in
`test/backend/backend_conformance_copy_storage.hpp`: its planned shared
byte-level reference (independent of production mapping) will exercise the
source/destination mapping, transformed leading planes, padding and cache-tail
isolation, aliases, checked overflow, no-side-effect admission failures,
accepted failures and repeat waits, and the exact offsets and row lengths
`1/15/16/17`. The independent physical oracle remains
`test/backend/backend_conformance_oracle.hpp`; these are future cases, not a
claim that the current backend drivers already contain or have run them.

Each backend driver exercises ADD, MUL, SUB, and DIV through its real queue for
every required leaf, as well as unsupported domains, validation precedence,
broadcasting, transformed mappings, exact aliases, owner deduplication, repeat
waits, and retained failures. CPU may complete inline; accelerator queues
preserve the same contract without SDK dtype narrowing.

CMake registration uses `add_iom_backend_tests` for the accelerator smoke and
conformance targets; `iom_backend_conformance_cpu_tests` is created and
registered directly in `test/CMakeLists.txt`. Drivers provide allocator/context
setup, CPU reference, foreign-device identity checks, hardware gating, and
native storage oracles; enabled hardware runs and never skips.
