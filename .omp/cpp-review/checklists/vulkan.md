# Vulkan Review Checklist

- [ ] resource state/layout transitions are valid
- [ ] access masks/stage masks/barriers establish required dependencies
- [ ] queue-family ownership transfers are correct when applicable
- [ ] descriptor/resource lifetime extends through submitted command execution
- [ ] command pool/buffer reset/reuse does not race in-flight submissions
- [ ] fences/semaphores/timeline semaphores are used with correct ordering/value semantics
- [ ] host-visible mapped memory obeys coherence/flush/invalidate requirements
- [ ] pipeline/shader specialization and cache keys match device/features/layout
- [ ] workgroup limits/subgroup assumptions are capability checked
- [ ] validation layers and synchronization validation are clean for relevant tests
- [ ] performance review distinguishes explicit Vulkan synchronization from unnecessary engine-level serialization
