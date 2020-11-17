#include <migraphx/cpu/migemm.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/shape_for_each.hpp>
#include <blaze/math/CustomMatrix.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template <class T, class F>
void migemm_impl(
    tensor_view<T> cmat, tensor_view<T> amat, tensor_view<T> bmat, F alpha, F beta, std::false_type)
{
    std::size_t n_dims = cmat.get_shape().lens().size();
    std::size_t dim_0  = n_dims - 2;
    std::size_t dim_1  = n_dims - 1;
    auto k             = amat.get_shape().lens()[dim_1];

    assert(amat.get_shape().lens()[dim_1] == bmat.get_shape().lens()[dim_0]);
    assert(cmat.get_shape().lens()[dim_0] == amat.get_shape().lens()[dim_0]);
    assert(cmat.get_shape().lens()[dim_1] == bmat.get_shape().lens()[dim_1]);

    shape_for_each(cmat.get_shape(), [&](const auto& c_idx) {
        auto a_idx = c_idx;
        auto b_idx = c_idx;
        double s   = 0.0;
        dfor(k)([&](auto kk) {
            a_idx[dim_1] = b_idx[dim_0] = kk;
            s += amat(a_idx.begin(), a_idx.end()) * bmat(b_idx.begin(), b_idx.end());
        });
        cmat(c_idx.begin(), c_idx.end()) = alpha * s + cmat(c_idx.begin(), c_idx.end()) * beta;
    });
}

template <class T, class F>
void migemm_impl(tensor_view<T> cmat, tensor_view<T> amat, tensor_view<T> bmat, F alpha, F beta)
{
    migemm_impl(cmat, amat, bmat, alpha, beta, std::false_type{});
}

template <class F>
void migemm_tpl(
    const argument& c_arg, const argument& a_arg, const argument& b_arg, F alpha, F beta)
{
    visit_all(c_arg, a_arg, b_arg)(
        [&](auto cmat, auto amat, auto bmat) { migemm_impl(cmat, amat, bmat, alpha, beta); });
}

void migemm(
    const argument& c_arg, const argument& a_arg, const argument& b_arg, float alpha, float beta)
{
    migemm_tpl(c_arg, a_arg, b_arg, alpha, beta);
}

void migemm(const argument& c_arg,
            const argument& a_arg,
            const argument& b_arg,
            int32_t alpha,
            int32_t beta)
{
    migemm_tpl(c_arg, a_arg, b_arg, alpha, beta);
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
