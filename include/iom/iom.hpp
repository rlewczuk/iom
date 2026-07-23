#pragma once

#include <cstdint>
#include <string>

namespace iom {
    enum class DataType {
        BOOL,
        U8,
        I8,
        U16,
        I16,
        U32,
        I32,
        U64,
        I64,
        F16,
        BF16,
        F32,
        F64,
        F8_E5M2,
        F8_E4M3,
        F8_E8M0,
        F6_E2M3,
        F6_E3M2,
        F4
    };

    enum class DataOrder {
        ROW_MAJOR, /// Row-major layout
        TILED  /// Tiled layout
    };

    class Tensor {
        // tbd
    public:
        [[nodiscard]] std::vector<size_t> shape() const;
        [[nodiscard]] DataType dtype() const;

        void set_float(float *data, size_t sz);
    };

    typedef uint64_t oid;

    class DeviceOps {
    public:
        virtual ~DeviceOps() = default;

        virtual void wait(oid oid) = 0;

        virtual oid copy(const Tensor& i, Tensor& o) = 0;

        /** Addition: c = a + b **/
        virtual oid add(const Tensor& a, const Tensor& b, Tensor& c) = 0;

        /** Multiplication: y = x * y **/
        virtual oid mul(const Tensor& a, const Tensor& b, Tensor& c) = 0;

        /** SILU: t = silu(t) **/
        virtual oid silu(Tensor& x, Tensor& y) = 0;

        /** Linear: y = x * w **/
        virtual oid linear(const Tensor& x, const Tensor& w, Tensor& y) = 0;

        /** RMSNorm: y = x * (1 / sqrt(mean(x^2, dim) + eps)) **/
        virtual oid rmsnorm(const Tensor& x, Tensor& y, const Tensor& w, float eps, size_t dim) = 0;

        /**
         * SDPA variant for GQA. We avoid repeating across k and v, kernel takes it into account automatically.
         * It always uses implicit casual mask.
         * Result is already rearranged so that it can be passed directly into linear projection o_proj.
         */
        virtual oid sdpa(const Tensor& q, const Tensor& k, const Tensor& v, size_t n_heads, size_t n_kv_heads, size_t head_dim, Tensor& attn_out) = 0;

    };

    class Block {
    public:
        virtual ~Block() = default;
        virtual void forward(const Tensor& x, Tensor& y) = 0;
    };
}
