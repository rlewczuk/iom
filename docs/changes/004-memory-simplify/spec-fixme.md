Remarks to specification:
* as device, queue is now limited to 16 requests, make sure that excess requests will be parked host-side instead of DeviceOps API returning an error; excess requests should be resumed in order or submission as soon as in-flight requests from queue finish; no need to allocate metadataslots for parked requests until they are resumed and scheduled into a real queue
* make queue limit configurable, with 16 being the default, but make sure it won't cause reallocations after construction
* as we now have upper limit for metadata slots, their size can be fixed (constant) and code handling device-side metadata can be simplified (i.e. remove metadata slot resizing etc.)

Most (of not all) of above remarks may be already addressed, in such case just check them and report it.
