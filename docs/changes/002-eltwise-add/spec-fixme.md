Remarks to current spec:
* remove dedicated `add_support` method, return `Unsupported` error code intead (from `oid` errors enum), assume that for particular backend device op will automatically choose between native and emulated execution;
* align compatibility matrix so that all backends including TTNN will support all storage leaves (except BOOL and F8_E8M0), assume that for TTNN storage leaves that are not supported will be emulated to the best of abilities (even if it will be very inefficient, require allocating RAM/VRAM, require implicit conversions etc.);
