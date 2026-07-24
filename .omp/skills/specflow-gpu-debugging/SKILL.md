---
name: specflow-gpu-debugging
description: Configure and drive serial remote C++/GPU debugging through OMP DAP adapters over SSH, with ROCgdb support and a documented CUDA-GDB server fallback.
---
# Remote GPU debugging

Load `specflow-cost-routing` and resolve role `debugger`. Pass its explicit model selector when spawning `specflow-gpu-debugger`. Debugging is always non-prewalk, even when the implementation task used prewalk.

Run `.omp/gpu-lab/bin/gpu-lab-configure` after editing hosts. It merges generated `gpu-lab-*` adapters into `.omp/dap.json` while preserving unrelated adapters. The SSH process transports DAP over stdio to remote GDB/ROCgdb/CUDA-GDB.

Before debugging:

1. reproduce twice on one host;
2. build the target's debug profile;
3. run `.omp/gpu-lab/bin/gpu-labctl debug-plan HOST PROGRAM --run-id RUN`;
4. verify local and remote source paths or compiler debug-prefix mapping;
5. ensure no concurrent job owns that GPU lock.

Use the explicit generated adapter. Bind a source breakpoint before trusting stack or variable data. Only one OMP root DAP session may be active, so terminate before changing hosts.

ROCm: remote ROCgdb has documented DAP support. Use debug `evaluate` in REPL context for ROCgdb CLI commands when needed.

NVIDIA: generic host-side GDB is supported by the generated host adapter. Direct CUDA-GDB DAP is opt-in and experimental because NVIDIA's documented remote CUDA path is `cuda-gdbserver` plus a CUDA-GDB client. If CUDA kernel stepping/focus is unreliable, stop and use the fallback described in `docs/DEBUGGING.md`; never fabricate CUDA-specific debugger observations.
