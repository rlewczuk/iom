#pragma once

#include <cmath>
#include <vector>

#include "iom/iom.hpp"

namespace iom::models {
    class LlamaRoPE {
        DeviceOps& dev;
        size_t ctx_len;
        size_t head_dim;
        double theta;
        Tensor& sin;
        Tensor& cos;
    public:
        LlamaRoPE(DeviceOps& dev, size_t ctx_len, size_t head_dim, double theta, Tensor& sin, Tensor& cos);

        void rope(const Tensor& x, size_t dim);

    };

    class LlamaAttention : public Block {
        DeviceOps& dev;
        Tensor& wq,  wk, wv, wo;  /// Weights: Wq, Wk, Wv, Wo
        Tensor& q, k, v, a;       /// Internal Activations: Q, K, V, A
        size_t h;                 /// Hidden size
        size_t n_head, n_kv;      /// Number of heads (overall, KV)
        LlamaRoPE& rope;
    public:
        LlamaAttention(DeviceOps& dev,
            size_t hidden_size, size_t num_heads, size_t num_hv_heads,
            Tensor& wq, Tensor& wk, Tensor& wv, Tensor& wo, Tensor& q, Tensor& k, Tensor& v, LlamaRoPE& rope);

        void forward(const Tensor& x, Tensor& y) override;
    };

    class LlamaMlp : public Block {
        DeviceOps& dev;
        Tensor& wu, wd, wg;
        Tensor& u, g;
    public:
        LlamaMlp(DeviceOps& dev, Tensor& wu, Tensor& wd, Tensor& wg, Tensor& w, Tensor& g);

        void forward(const Tensor& x, Tensor& y) override;
    };

    class LlamaDecoder : public Block {
        DeviceOps& dev;
        LlamaAttention& attn;
        LlamaMlp& mlp;
        const Tensor& wi, wm;    /// Input/output normalization weights: input norm, mlp norm
        Tensor& t, r;            /// Internal Activations: temporary, residual
    public:
        LlamaDecoder(DeviceOps& dev, LlamaAttention& attn, LlamaMlp& mlp, const Tensor& wi, const Tensor& wm, Tensor& t, Tensor& r);

        void forward(const Tensor& x, Tensor& y) override;
    };

    class Llama2Model : public Block {
        DeviceOps& dev;
        std::vector<std::reference_wrapper<LlamaDecoder>>& decoders;
        Tensor& wlm;   /// LM Head
        Tensor& wn;    /// Output norm
        Tensor& t;     /// Internal Activations: temporary
    public:
        Llama2Model(DeviceOps &dev, std::vector<std::reference_wrapper<LlamaDecoder>>& decoders, Tensor& wlm, Tensor& wn, Tensor& t);

        void forward(const Tensor& x, Tensor& y) override;
    };

    class Llama2 {
    public:
        Llama2(const std::string& modelPath, DeviceOps& dev);
    };

}
