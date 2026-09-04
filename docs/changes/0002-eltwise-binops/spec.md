# Elementwise Binary Operations

Implement full set of elementwise binary operations for all supported platforms.
Please keep in mind following requirements:
* operations semantics should be modeled after TTNN semantics, so that TTNN implementation will be relatively thin
  * but you still need to keep APIs like `oid operation(TensorView& a, TensorView& b, TensorView& out)` (i.e. we don't return tensors)
  * out is output tensor with allocated buffer
  * if `out` is the same as `a`, then this will be inplace operation
  * but make sure operations work properly on our views (i.e. affecting all dimensions except those two tiled ones)
* look at ttnn to extract list of elementwise binary operations, we need to implement all of them - as thin wrappers in ttnn, but also for ROCm, CUDA, and SYCL
  * you can use `remote-development` skill to access ttnn machine and look through header files to find supported binary ops
* for each operation implement comprehensive set of backend conformance tests
  * keep tests properly split, so that single file does not exceed 500 lines
  * make sure tests execution won't take too long, aovid situations where test performs operation on a big tensor and then manually checks every element of this tensor
