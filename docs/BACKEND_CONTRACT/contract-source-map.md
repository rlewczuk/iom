# 13. Contract source map

Use these sources when changing or extending the contract:

- public device/tensor/queue API: `include/iom/device.hpp`,
  `include/iom/tensor.hpp`, `include/iom/iom.hpp`;
- tokenizer and prompt boundary public API and focused ownership:
  `include/iom/tokenizer.hpp`, `include/iom/chat_format.hpp`,
  `test/test_tokenizer.cpp`, and `test/test_chat_format.cpp`;
- independent tokenizer reference and pinned oracle artifacts:
  `test/reference/tokenizer_reference.py`,
  `test/reference/tokenizer_reference_manifest.json`,
  `test/reference/generate_tokenizer_oracles.py`, and
  `test/reference/tokenizer_oracles.json`;
- standard capability and transfer interface:
  `src/shared/standard_tiled_copy.hpp`;
- common binary validation, operation dispatch, and independent scalar oracle:
  `test/backend/backend_conformance_common.hpp`,
  `test/backend/backend_conformance_add.hpp`;
- common RMS normalization layout, admission, capability, request snapshot,
  and pure requirement query: `src/device_ops_rmsnorm.cpp`, with its API,
  purity, precedence, rejection-effect, snapshot, and ownership tests in
  `test/test_iom.cpp` and the shared conformance scenarios in
  `test/backend/backend_conformance_rmsnorm.hpp`; CPU-local result probes
  remain in `test/cpu/test_cpu.cpp`;
- implemented SiLU ABI, scalar/raw reference, admission, zero-workspace,
  capability, numerical, queue/lifetime, and stored-result composition
  contract: [SiLU activation](silu-activation.md#silu-activation), `src/shared/scalar_binary_codec.hpp`,
  and the shared `test/backend/backend_conformance_silu.hpp` cases;
- cache append public facade, common admission, and current API/lifetime tests:
  `include/iom/iom.hpp` (`DeviceOps::cache_append` and
  `DeviceOps::cache_append_workspace_requirements`),
  `src/device_ops_cache_append.cpp`
  (`snapshot_cache_append_view`, `validate_cache_append`, and the queue
  handoff), and `test/test_iom.cpp` (ABI, query purity, precedence,
  no-side-effect, snapshot, workspace-retention, and retained-failure cases);
- cache append's common workspace, queue, and neural-hook precedent:
  `include/iom/iom.hpp` (`detail::WorkspaceValidation` and the current neural
  `DeviceOps` hooks), `src/device_ops.cpp` (`WorkspaceValidation::validated`,
  `encode_token`, `wait`, completion/failure retention, drain, and sequence
  helpers);
- transformed leading offsets and strides:
  `src/tensor_view.cpp` (`TensorView::slice`, `select`, `permute`, and
  `reshape_leading`);
- standard direct tiled mapping:
  `src/shared/standard_tiled_copy.inl` (`plane_slot`,
  `physical_coordinate`, and `copy_tiled_to_tiled_word`) and
  `src/shared/standard_tiled_copy_metadata.inl`;
- retained accelerator workspace and storage rules:
  this contract's standard tiled mapping and caller-owned workspace boundaries;
- storage, transfer, copy, and physical oracle:
  `test/backend/backend_conformance_copy_storage.hpp`,
  `test/backend/backend_conformance_oracle.hpp`;
- native setup/allocation seams: `test/cuda`, `test/rocm`, and `test/sycl`
  smoke/conformance drivers;
- model configuration intake and its isolated fixture: `include/iom/model.hpp`,
  `src/model.cpp`, `test/test_model_loading.cpp`, and
  `test/model_loading_fixture.hpp`;
- shared model loading and realization scenario:
  `test/backend/backend_conformance_model_loading.hpp`;
- shared official-model inference validation and launch:
  `test/backend/backend_conformance_model_official.hpp`,
  `test/model/run_official_inference.py`, and
  `test/model/test_run_official_inference.py`; the CPU registration and retained
  backend evidence are `test/cpu/test_cpu_conformance.cpp`,
  `test/CMakeLists.txt`, and
  [CPU official TinyLlama inference evidence](model-inference-cpu-official-evidence.md);
  the CUDA registration and retained backend evidence are
  `test/cuda/test_cuda_conformance.cpp`, `test/CMakeLists.txt`, and
  [CUDA official TinyLlama inference evidence](model-inference-cuda-official-evidence.md);
- ROCm official-model inference registration and retained evidence:
  `test/rocm/test_rocm_conformance.cpp`, `test/CMakeLists.txt`, and
  [ROCm official TinyLlama inference evidence](model-inference-rocm-official-evidence.md);
- SYCL official-model registration and retained backend evidence:
  `test/sycl/test_sycl_conformance.cpp`, `test/CMakeLists.txt`, and
  [SYCL official TinyLlama inference evidence](model-inference-sycl-official-evidence.md);
- backend-local full suites: `test/cpu/test_cpu_conformance.cpp`,
  `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`,
  `test/sycl/test_sycl_conformance.cpp`, and
- coexistence integration: `test/backend/test_backend_coexistence.cpp` and
  `test/CMakeLists.txt`.

If implementation and this document differ, update implementation, conformance,
`ARCHITECTURE.md`, and this contract together. Do not weaken shared tests.
