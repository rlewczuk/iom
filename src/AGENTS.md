# Implementation Guidance

- `iom.cpp` implements tensor/view rules and common queue completion machinery.
- `alloc.cpp` implements allocators.
- `mmap.cpp` and `safetensors.cpp` implement weight mapping and parsing.
- Backend directories implement devices, storage, queues, and copies; keep runtime details there.

Do not allocate operation outputs, move owner storage, or add backend dispatch to common code. Preserve validation-before-effects and retained asynchronous failures.
