#include <migraphx/cpu/gemm.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/shape_for_each.hpp>
#include <blaze/math/CustomMatrix.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template <class T>
using matrix = blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded>; // NOLINT

template <class T>
static auto make_mat(tensor_view<T> x)
{
    const auto& s = x.get_shape();
    // assert(s.lens().size() == 2);
    std::size_t n_dims = s.lens().size();
    std::size_t dim_0  = n_dims - 2;
    std::size_t dim_1  = n_dims - 1;
    if(s.transposed())
        return matrix<T>{x.data(), s.lens()[dim_1], s.lens()[dim_0], s.strides()[dim_1]};
    return matrix<T>{x.data(), s.lens()[dim_0], s.lens()[dim_1], s.strides()[dim_0]};
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
            c      = beta * c;

            if(alpha != 0.0)
            {
                c = c + alpha * a * b;
            }
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
    auto a_lens = amat.get_shape().lens();
    auto b_lens = bmat.get_shape().lens();
    auto c_lens = cmat.get_shape().lens();

    std::size_t n_dims = c_lens.size();
    std::size_t dim_0  = n_dims - 2;
    std::size_t dim_1  = n_dims - 1;
    auto k             = a_lens[dim_1];

    assert(a_lens[dim_1] == b_lens[dim_0]);
    assert(c_lens[dim_0] == a_lens[dim_0]);
    assert(c_lens[dim_1] == b_lens[dim_1]);

    std::size_t a_len_diff = c_lens.size() - a_lens.size();
    std::size_t b_len_diff = c_lens.size() - b_lens.size();
    std::vector<std::size_t> a_idx(a_lens.size());
    std::vector<std::size_t> b_idx(b_lens.size());

    shape_for_each(cmat.get_shape(), [&](const auto& c_idx) {
        std::transform(c_lens.begin() + a_len_diff, c_lens.end(), a_lens.begin(), a_idx.begin(), [&](auto i, auto j) {
            return (j == 1) ? 0 : i;
        });
        std::transform(c_lens.begin() + b_len_diff, c_lens.end(), b_lens.begin(), b_idx.begin(), [&](auto i, auto j) {
            return (j == 1) ? 0 : i;
        });

        double s   = 0.0;
        dfor(k)([&](auto kk) {
            a_idx[dim_1] = b_idx[dim_0] = kk;
            s += amat(a_idx.begin(), a_idx.end()) * bmat(b_idx.begin(), b_idx.end());
        });
        cmat(c_idx.begin(), c_idx.end()) = alpha * s + cmat(c_idx.begin(), c_idx.end()) * beta;
    });
}

template <class T>
void migemm_impl(
    tensor_view<T> cmat, tensor_view<T> amat, tensor_view<T> bmat, float alpha, float beta)
{
    auto lens = cmat.get_shape().lens();
    std::size_t num_matrices =
        std::accumulate(lens.rbegin() + 2, lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
    if(num_matrices == 1)
    {
        migemm_impl(cmat, amat, bmat, alpha, beta, is_fast_gemm_type<T>{});
    }
    else
    {
        migemm_impl(cmat, amat, bmat, alpha, beta, std::false_type{});
    }
}

void migemm(
    const argument& c_arg, const argument& a_arg, const argument& b_arg, float alpha, float beta)
{
    visit_all(c_arg, a_arg, b_arg)(
        [&](auto cmat, auto amat, auto bmat) { migemm_impl(cmat, amat, bmat, alpha, beta); });
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
