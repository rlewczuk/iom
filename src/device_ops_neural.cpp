#include "iom/iom.hpp"

#include "iom_internal.hpp"

namespace iom {

    using detail::UnsupportedOperation;

    oid DeviceOps::silu_impl(const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::linear_impl(
            const TensorView&, const TensorView&, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::rmsnorm_impl(
            const TensorView&, TensorView&, const TensorView&, float, size_t) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::sdpa_impl(
            const TensorView&, const TensorView&, const TensorView&,
            size_t, size_t, size_t, TensorView&) {
        throw UnsupportedOperation();
    }

    oid DeviceOps::silu(
            const TensorView& x, TensorView& y) noexcept {
        try {
            validate_views(queue_device(), {&x, &y});
            return invoke(silu_impl(x, y));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::linear(
            const TensorView& x, const TensorView& w,
            TensorView& y) noexcept {
        try {
            validate_views(queue_device(), {&x, &w, &y});
            return invoke(linear_impl(x, w, y));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::rmsnorm(
            const TensorView& x, TensorView& y, const TensorView& w,
            float eps, size_t dim) noexcept {
        try {
            validate_views(queue_device(), {&x, &y, &w});
            if (!(eps >= 0.0F) || dim == 0) {
                throw std::invalid_argument("invalid rmsnorm parameters");
            }
            return invoke(rmsnorm_impl(x, y, w, eps, dim));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

    oid DeviceOps::sdpa(
            const TensorView& q, const TensorView& k, const TensorView& v,
            size_t n_heads, size_t n_kv_heads, size_t head_dim,
            TensorView& attn_out) noexcept {
        try {
            validate_views(queue_device(), {&q, &k, &v, &attn_out});
            if (n_heads == 0 || n_kv_heads == 0 || head_dim == 0
                    || n_heads % n_kv_heads != 0) {
                throw std::invalid_argument("invalid sdpa parameters");
            }
            return invoke(sdpa_impl(
                    q, k, v, n_heads, n_kv_heads, head_dim, attn_out));
        } catch (...) {
            return invoke_failure(std::current_exception());
        }
    }

}  // namespace iom
