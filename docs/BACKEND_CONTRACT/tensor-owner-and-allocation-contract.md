# 2. Tensor owner and allocation contract

1. A backend `Tensor` is materialized only through its creating `Device`. It
   MUST initialize the common `Tensor` base with the requested spec and that
   device, expose one stable full view through `view()`, and remain
   non-copyable/non-movable.
2. The tensor's opaque `native_handle()` MUST remain stable for the full owner
   lifetime. A backend MUST NOT relocate storage because a view is made,
   transformed, transferred, or submitted to a queue.
3. The creating `Device` and its borrowed allocator MUST outlive every tensor
   and queue they create. A caller must retain tensor owners until all work
   accessing their storage is complete. A backend MUST snapshot every view
   metadata field it needs before `copy` returns: a derived `TensorView`
   temporary may be destroyed before the caller waits. A backend MUST make its
   own destruction safe when it owns outstanding runtime state; it MUST NOT
   free or reuse a resource the runtime can still access.
4. Allocator-backed standard storage MUST allocate exactly the checked
   `spec.tiled_storage_nbytes()` amount and require a 32-byte-aligned base
   address. A null allocation is `std::bad_alloc`; a returned misaligned
   address MUST be released and reported as an error.
5. Accelerator allocators may have stricter validity rules. CUDA allocations
   must be unmanaged device allocations from the exact owning context/ordinal;
   ROCm allocations must be unmanaged device allocations from the exact
   ordinal; SYCL allocation pointers must be known to the owned context.
   Validate these rules before accepting storage.
6. Native storage may be runtime-owned and have a different physical byte
   count, but it MUST remain associated with the same public spec/device/view
   contract. It MUST validate native extent conversion and leading-plane count
   overflow before creating native resources.
7. Operations, view transformations, and host transfers MUST NOT allocate or
   free operand or output tensor storage. Resource pools may allocate internal
   staging/metadata capacity, but they must not replace, retarget, or move a
   user tensor.
