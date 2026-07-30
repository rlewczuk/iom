# FIXME: remarks and requests to refine design

This design is way too complicated. Please shrink it and adapt to existing APIs.

Please revise design and fix following aspects:
* all tensor related classes and definitions should be in `iom` namespace
* remove explicit management of scratch memory used by kernels, assume that kernels will receive explicitly allocated tensors that will be used as scratch
* remove `TileShape` class as we only have 16x16 and 32x32 shapes; use trivial flag (eg. int constants or enum class) indicating which size is being used
* assume that max number of dimensions is 4;
* make sure tiled layout is friendly to WMMA instructions and tensor units in modern consumer GPUs
* merge all classes used inside `TensorSpec` into `TensorSpec` itself, remove `Shape` class and replace `TileShape` with enum mentioned above
* remove `PaddedShape` struct completely, padding is now implicit as it can be easily derived from the shape dimensions and tile size flag
  * you can just provide method in `TensorSpec` that returns either vector with padded shape or at last two dimensions, padded
* remove `DType` enum class and adapt existing `iom::DataType` from `include/iom/iom.hpp`
* place most of APIs directlyin `Tensor` class:
  * old version of class `Tensor` is already defined in `include/iom/iom.hpp` but possibly wrong, please read it definition but you are not bound to adapt your design to it, 
  * there is no point in hiding public APIs from kernels facing APIs, you can implement both surfaces in a single class
* remove allocator definition (`struct Allocation`, `class Allocator`) from design document, please adapt your design to use `iom::Allocator` interface from `include/iom/alloc.hpp`
  * existing allocator API does not have alignment parameter, please assume that allocator will always return properly aligned pointers
* remove notion of backend, execution context and all associated classes from design document as it is not in scope of this change
  * instead, `backend_kind()` and `backend_device()` methods which will return backend kind (eg. CPU, CUDA, ROCm, TTNN) and device ID (0, 1, 2, ...)
* remove `ConstHostTensorView` and `TensorView` classes and associated APIs from design document, instead propose easy to use accessor methods directly in `Tensor` class
  * all functions supposed to copy data between host and device along with layout translations should be removed and implemented as methods of `Tensor` class
  * also, there is no need to maintain non-materialized views, so `Tensor` only represents real tensors (it is also limitation of TTNN, so we cannot have tensor views anyway)
* tensor to tensor copies should also be implemented as methods of `Tensor` class
* operations are implemented as methods of `DeviceOps` class from `include/iom/iom.hpp`, please use those in your design instead of defining new ones
  * but take into account that those methods run asynchronously;
  * remove status returns from your design, use methods of `DeviceOps` as is
* remove all design decisions regarding operations (`Backend` class, all `validate_*` and `enqueue_*` methods), it is out of scope of this change, adapt design to reuse existing `DeviceOps` instead
 
