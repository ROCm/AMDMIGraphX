#include <migraphx/cpu/gemm.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/requires.hpp>
#include <blaze/math/CustomMatrix.h>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

template <class T>
using matrix = blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded>; // NOLINT

template <class T>
static auto make_mat(tensor_view<T> x)
{
    const auto& s = x.get_shape();
    assert(s.lens().size() == 2);
    if(s.transposed())
        return matrix<T>{x.data(), s.lens()[1], s.lens()[0], s.strides()[1]};
    return matrix<T>{x.data(), s.lens()[0], s.lens()[1], s.strides()[0]};
}

template <class T, class F>
static void visit_mat(tensor_view<T> x, F f)
{
    auto mat = make_mat(x);
    if(x.get_shape().transposed())
        f(blaze::trans(mat));
    else
        f(mat);
}

template <class T>
struct is_fast_gemm_type : std::false_type
{
};

template <>
struct is_fast_gemm_type<float> : std::true_type
{
};

template <class T>
void migemm_impl(tensor_view<T> cmat,
                 tensor_view<T> amat,
                 tensor_view<T> bmat,
                 float alpha,
                 float beta,
                 std::true_type)
{
    visit_mat(amat, [&](const auto& a) {
        visit_mat(bmat, [&](const auto& b) {
            auto c = make_mat(cmat);
            c      = (a * b) * alpha + beta * c;
        });
    });
}

template <class T>
void migemm_impl(tensor_view<T> cmat,
                 tensor_view<T> amat,
                 tensor_view<T> bmat,
                 float alpha,
                 float beta,
                 std::false_type)
{
    auto m = cmat.get_shape().lens()[0];
    auto n = cmat.get_shape().lens()[1];
    auto k = amat.get_shape().lens()[1];

    assert(amat.get_shape().lens()[1] == bmat.get_shape().lens()[0]);
    assert(m == amat.get_shape().lens()[0]);
    assert(n == bmat.get_shape().lens()[1]);

    dfor(m, n)([&](auto ii, auto jj) {
        double s = cmat(ii, jj) * beta;
        dfor(k)([&](auto kk) { s += amat(ii, kk) * bmat(kk, jj); });
        cmat(ii, jj) = alpha * s;
    });
}

template <class T>
void migemm_impl(
    tensor_view<T> cmat, tensor_view<T> amat, tensor_view<T> bmat, float alpha, float beta)
{
    migemm_impl(cmat, amat, bmat, alpha, beta, is_fast_gemm_type<T>{});
}

void migemm(
    const argument& c_arg, const argument& a_arg, const argument& b_arg, float alpha, float beta)
{
    visit_all(c_arg, a_arg, b_arg)(
        [&](auto cmat, auto amat, auto bmat) { migemm_impl(cmat, amat, bmat, alpha, beta); });
}

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
