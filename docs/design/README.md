# Top Level Design

IOM is inference engine designed to run sparse models leveraging memory hierarchy of underlying hardware
and advanced drafting techniques to serve very large models on high-end consumer hardware 
with minimal memory footprint on both VRAM and host RAM

Top level design decisions:
* inference-only architecture, no explicit computation graph, no need to implement autograd engine 
* computation implemented directly in-code (imperative), 
* pre-allocated tensor buffers where possible (including temporary tensors)
* tiled memory layout, WMMA friendly (for AMD, NVidia, Tenstorrent, Intel)
  * we compensate low WMMA utilization during decode by aggresive speculation, thus filling all rows of activation tensor tiles
* high precision arithmetic of model dense part (attention, embeddings, shared experts): bf16, eventually fp8/int8
* pluggable algorithms for sparse experts streaming/placement during prefill/decode
  * double buffering, predicting and loading experts for next layer 
  * each sparse expert i s available in various quantizations: from 8-bit, 4-bit, 1.6 bit to sub 1-bit
  * each experts on disk is represented in several precisions at once, inference engine chooses most fitting one for each inference run
  * depending on needs, specific experts in specific precisions may reside on different memory tiers (SSDk host RAM, VRAM)

# Tensors

Tensors are stored in following way:
* last two dimensions have tiled layout, changing view of last two dimensions is not allowed
* all other dimensions are stored in row-major (strided) layout, changing view on those dimensions is possible
* all tensor metadata is stored on host side, tensor data is stored on device or host depending on tensor allocation

# Tensor operations

* tensor operations do NOT allocate new tensors, output tensors must be provided by caller


# Weight Tensors

Layouts and Composition:
* represent quantized weights, thus in certain cases we can have more than one tensor (eg. LoRA)

Storage:
* weights can exist on disk in several variants (formats/precisions) at once
* each variant can be loaded and used by inference engine without modification (eg. it can be loaded via DMA directly onto device, if device supports it), this is friendly for dynamic weights loading for example



