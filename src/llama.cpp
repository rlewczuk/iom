#include "iom/llama.hpp"

#include <span>

namespace iom::models {
    LlamaRoPE::LlamaRoPE(DeviceOps& dev, size_t ctx_len, size_t head_dim, double theta, Tensor& sin, Tensor& cos) : dev(dev), ctx_len(ctx_len), head_dim(head_dim), theta(theta), sin(sin), cos(cos) {
        std::vector<double> inv_freq(head_dim/2, 0.0);
        for (int j = 0 ; j < head_dim/2 ; j++) {
            inv_freq[j] = 1.0 / std::pow(10000.0, 2*j/head_dim);
        }
        std::vector<float> buf(head_dim/2 * ctx_len, 0.0);
        for (int i = 0 ; i < ctx_len ; i++) {
            for (int j = 0 ; j < head_dim/2 ; j++) {
                buf[i*head_dim/2 + j] = static_cast<float>(std::sin(theta * i * inv_freq[j]));
            }
        }
        sin.view().copy_from_host(std::as_bytes(std::span{buf}));

        for (int i = 0 ; i < ctx_len ; i++) {
            for (int j = 0 ; j < head_dim/2 ; j++) {
                buf[i*head_dim/2 + j] = static_cast<float>(std::cos(theta * i * inv_freq[j]));
            }
        }
        cos.view().copy_from_host(std::as_bytes(std::span{buf}));
    }

    void LlamaRoPE::rope(const Tensor& x, size_t dim) {
        // TODO
    }

    LlamaAttention::LlamaAttention(DeviceOps& dev,
        size_t hidden_size, size_t num_heads, size_t num_hv_heads,
        Tensor& wq, Tensor& wk, Tensor& wv, Tensor& wo,
        Tensor& q, Tensor& k, Tensor& v, Tensor& a, LlamaRoPE& rope)
        : dev(dev), wq(wq), wk(wk), wv(wv), wo(wo), q(q), k(k), v(v), a(a),
          h(hidden_size), n_head(num_heads), n_kv(num_hv_heads), rope(rope) {
    }

    void LlamaAttention::forward(const Tensor& x, Tensor& y) {
        dev.linear(x.view(), wq.view(), q.view());
        dev.linear(x.view(), wk.view(), k.view());
        dev.linear(x.view(), wv.view(), v.view());

        rope.rope(q, 1);
        rope.rope(k, 1);
        dev.sdpa(q.view(), k.view(), v.view(), n_head, n_kv, h, a.view());
        dev.linear(a.view(), wo.view(), y.view());
    }

    LlamaMlp::LlamaMlp(DeviceOps& dev, Tensor& wu, Tensor& wd, Tensor& wg, Tensor& w, Tensor& g) : dev(dev), wu(wu), wd(wd), wg(wg), u(w), g(g) {
    }

    void LlamaMlp::forward(const Tensor& x, Tensor& y) {
        dev.linear(x.view(), wu.view(), u.view());
        dev.linear(x.view(), wg.view(), g.view());
        dev.silu(g.view(), g.view());
        dev.mul(g.view(), u.view(), u.view());
        dev.linear(g.view(), wd.view(), y.view());
    }

    LlamaDecoder::LlamaDecoder(DeviceOps& dev, LlamaAttention& attn, LlamaMlp& mlp, const Tensor& wi, const Tensor& wm, Tensor& t, Tensor& r) : dev(dev), attn(attn), mlp(mlp), wi(wi), wm(wm), t(t), r(r) {}

    void LlamaDecoder::forward(const Tensor& x, Tensor& y) {
        dev.rmsnorm(x.view(), t.view(), wi.view(), 1e-6, 1);
        attn.forward(t, t); // TODO where and how do we manage KV cache here ?
        dev.add(t.view(), x.view(), t.view());

        dev.copy(x.view(), r.view());
        dev.rmsnorm(t.view(), t.view(), wm.view(), 1e-6, 1);
        mlp.forward(t, t);
        dev.add(t.view(), r.view(), y.view());
    }

    Llama2Model::Llama2Model(DeviceOps &dev, std::vector<std::reference_wrapper<LlamaDecoder>>& decoders, Tensor& wlm, Tensor& wn, Tensor& t) : dev(dev), decoders(decoders), wlm(wlm), wn(wn), t(t) {}

    void Llama2Model::forward(const Tensor& x, Tensor& y) {
        for (auto& decoder : decoders) {
            decoder.get().forward(t, t);
        }
        dev.rmsnorm(t.view(), t.view(), wn.view(), 1e-6, 1);
        dev.linear(t.view(), wlm.view(), y.view());
    }
}
