Remarks for further design refinement:
* rework DataPrecision and DataFormat into DataType and QuantizationFormat:
  * it is already redefined in `tensor.hpp`, update spec to match this definition
  * `DataType` will represent leaf data types (eg. ieee float32)
  * `QuantizationFormat` will represent grouped data (eg. various quantization formats or block floats).
* initial step of implementation (buildable scaffolds for backends) should be done one by one (i.e. single backend at a time), so that developer has possibility to install needed dependencies and test builds thoroughly
