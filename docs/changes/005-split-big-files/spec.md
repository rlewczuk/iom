Currently `src/iom.cpp` contains implementations of methods for many classes: `TensorShape`, `TensorSpec`, `TensorView`, `RawWorkspaceView`, `Device`, `DeviceOps`. As a result, this file is very long:
* please split it into smaller files: tensor et al, device, device ops
* analyze DeviceOps methods and factor out individual operations or groups of operations
  * note that this class will grow by new operations implemented in it, design split in a way that will ensure future extensibility and ensure that common code (eg. general methods, oid handling, wait, queues handling etc). is separate from concrete operations
* analyze backend-specific `device.cpp` files and factor out logically consistent blocks of code (groups of methods, classes etc.)
* strive to make all source files shorter than 500 lines
