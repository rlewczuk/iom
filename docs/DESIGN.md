# Top Level Design

IOM is inference engine designed to run sparse models leveraging memory hierarchy of underlying hardware
and advanced drafting techniques to serve very large models with minimal memory footprint on both VRAM and host RAM.

Top level design decisions:
* inference-only architecture, no explicit computation graph 
* computation implemented directly in-code
* pre-allocated tensor buffers where possible (including temporary tensors)
* tiled memory layout, WMMA friendly (for AMD, NVidia, Tenstorrent)
* high precision arithmetic of model dense part (attention, embeddings, shared experts): bf16, eventually fp8/int8
* pluggable algorithms for sparse experts streaming/placement
  * each sparse expert i s available in various quantizations: from 8-bit, 4-bit, 1.6 bit to sub 1-bit
  * each experts on disk is represented in several precisions at once, inference engine chooses most fitting one for each inference run
  * depending on needs, specific experts in specific precisions may reside on different memory tiers (SSDk host RAM, VRAM)

