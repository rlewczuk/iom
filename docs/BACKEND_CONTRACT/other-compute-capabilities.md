# 9. Other compute capabilities

`silu` is implemented on CPU, CUDA, ROCm, and SYCL; GQA `sdpa` is implemented
on the same four retained backends for its one current BF16 leaf, closed by
the retained-backend gate evidence in
[Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention), while the
eight non-BF16 SDPA leaves stay explicitly `Unsupported` everywhere. The
`linear` hooks are owned by
CPU, CUDA, ROCm, and SYCL implement all twenty-one applicable leaves — the
twenty non-BF16 leaves on the shared scalar projection path plus `BF16` on the
scalar recurrence on CPU and on a separate native specialization on CUDA, ROCm,
and SYCL, whose availability is a runtime device and loaded-image fact. A
backend without its own linear port keeps reporting `Unsupported`. CUDA and
ROCm provide source-inspected RMSNorm launch wrappers over the shared core, and
SYCL provides its native row kernel with the `aspect::fp64` guard. The
four-backend RMSNorm closure evidence and supported/limited matrix are recorded
in [RMS normalization](rms-normalization.md#rms-normalization).
