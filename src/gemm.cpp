/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gemm.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/par_for.hpp>
#include <numeric>
#include <vector>


#if MIGRAPHX_USE_EIGEN
#include <Eigen/Core>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

template <class T, class U>
[[maybe_unused]] void gemm_naive(tensor_view<T> cmat, tensor_view<U> amat, tensor_view<U> bmat)
{
    std::size_t n_dims = cmat.get_shape().ndim();
    std::size_t dim_0  = n_dims - 2;
    std::size_t dim_1  = n_dims - 1;
    auto k             = amat.get_shape().lens()[dim_1];

    auto cs = cmat.get_shape();
    par_for(cs.elements(), [&](auto i) {
        auto c_idx = cs.multi(i);
        auto a_idx = c_idx;
        auto b_idx = c_idx;
        double s   = 0.0;
        dfor(k)([&](auto kk) {
            a_idx[dim_1] = b_idx[dim_0] = kk;
            s += static_cast<double>(amat(a_idx.begin(), a_idx.end())) *
                 static_cast<double>(bmat(b_idx.begin(), b_idx.end()));
        });
        cmat(c_idx.begin(), c_idx.end()) = static_cast<T>(s);
    });
}

#if MIGRAPHX_USE_EIGEN

using eigen_row_major = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using eigen_col_major = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using eigen_stride    = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

template <class T>
auto make_eigen_map(T* ptr, const shape& s)
{
    auto dim_0     = s.ndim() - 2;
    auto dim_1     = s.ndim() - 1;
    auto rows      = static_cast<Eigen::Index>(s.lens()[dim_0]);
    auto cols      = static_cast<Eigen::Index>(s.lens()[dim_1]);
    auto rowstride = static_cast<Eigen::Index>(s.strides()[dim_0]);
    auto colstride = static_cast<Eigen::Index>(s.strides()[dim_1]);
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned,
                      eigen_stride>{ptr, rows, cols, eigen_stride{rowstride, colstride}};
}

bool is_mat_layout_supported(const shape& s)
{
    if(s.ndim() < 2)
        return false;
    auto dim_0 = s.ndim() - 2;
    auto dim_1 = s.ndim() - 1;
    return s.strides()[dim_1] == 1 or s.strides()[dim_0] == 1;
}

shape make_batch_shape(const shape& s, std::size_t n_batch_dims)
{
    return {s.type(),
            {s.lens().begin(), s.lens().begin() + n_batch_dims},
            {s.strides().begin(), s.strides().begin() + n_batch_dims}};
}

template <class T, class U>
void copy_2d(T* dst,
             std::size_t dst_row_stride,
             std::size_t dst_col_stride,
             const U* src,
             std::size_t src_row_stride,
             std::size_t src_col_stride,
             std::size_t rows,
             std::size_t cols)
{
    shape mat_shape{shape::float_type, {rows, cols}};
    shape_for_each(mat_shape, [&](const auto& idx) {
        auto i = idx[0];
        auto j = idx[1];
        dst[i * dst_row_stride + j * dst_col_stride] =
            static_cast<T>(src[i * src_row_stride + j * src_col_stride]);
    });
}

template <class T>
constexpr bool is_eigen_native_type()
{
    return std::is_same<T, float>{} or std::is_same<T, double>{};
}

template <class T, class U>
void gemm_eigen(tensor_view<T> cmat, tensor_view<U> amat, tensor_view<U> bmat)
{
    std::size_t n_dims = cmat.get_shape().ndim();
    std::size_t dim_0  = n_dims - 2;
    std::size_t dim_1  = n_dims - 1;

    auto m_size = amat.get_shape().lens()[dim_0];
    auto k_size = amat.get_shape().lens()[dim_1];
    auto n_size = bmat.get_shape().lens()[dim_1];

    const auto& clens = cmat.get_shape().lens();
    auto num_batches =
        std::accumulate(clens.begin(), clens.begin() + dim_0, std::size_t{1}, std::multiplies<>{});

    auto a_batch = make_batch_shape(amat.get_shape(), dim_0);
    auto b_batch = make_batch_shape(bmat.get_shape(), dim_0);
    auto c_batch = make_batch_shape(cmat.get_shape(), dim_0);

    if constexpr(is_eigen_native_type<U>() and std::is_same<T, U>{})
    {
        if(is_mat_layout_supported(amat.get_shape()) and
           is_mat_layout_supported(bmat.get_shape()) and is_mat_layout_supported(cmat.get_shape()))
        {
            for(std::size_t batch = 0; batch < num_batches; batch++)
            {
                auto a = make_eigen_map(amat.data() + a_batch.index(batch), amat.get_shape());
                auto b = make_eigen_map(bmat.data() + b_batch.index(batch), bmat.get_shape());
                auto c = make_eigen_map(cmat.data() + c_batch.index(batch), cmat.get_shape());
                c.noalias() = a * b;
            }
            return;
        }
    }

    const auto row_col_strides = [&](const auto& mat) {
        const auto& strides = mat.get_shape().strides();
        return std::pair{strides[dim_0], strides[dim_1]};
    };

    const auto [a_row_stride, a_col_stride] = row_col_strides(amat);
    const auto [b_row_stride, b_col_stride] = row_col_strides(bmat);
    const auto [c_row_stride, c_col_stride] = row_col_strides(cmat);

    std::vector<float> a_buf(m_size * k_size);
    std::vector<float> b_buf(k_size * n_size);
    std::vector<float> c_buf(m_size * n_size);

    auto mi = static_cast<Eigen::Index>(m_size);
    auto ki = static_cast<Eigen::Index>(k_size);
    auto ni = static_cast<Eigen::Index>(n_size);

    for(std::size_t batch = 0; batch < num_batches; batch++)
    {
        copy_2d(a_buf.data(),
                k_size,
                std::size_t{1},
                amat.data() + a_batch.index(batch),
                a_row_stride,
                a_col_stride,
                m_size,
                k_size);
        copy_2d(b_buf.data(),
                n_size,
                std::size_t{1},
                bmat.data() + b_batch.index(batch),
                b_row_stride,
                b_col_stride,
                k_size,
                n_size);

        Eigen::Map<eigen_row_major> a(a_buf.data(), mi, ki);
        Eigen::Map<eigen_row_major> b(b_buf.data(), ki, ni);
        Eigen::Map<eigen_row_major> c(c_buf.data(), mi, ni);
        c.noalias() = a * b;

        copy_2d(cmat.data() + c_batch.index(batch),
                c_row_stride,
                c_col_stride,
                c_buf.data(),
                n_size,
                std::size_t{1},
                m_size,
                n_size);
    }
}

#endif

template <class Visitor>
void gemm_ref(const argument& c_arg, const argument& a_arg, const argument& b_arg, Visitor v)
{
    if(c_arg.get_shape().type() == a_arg.get_shape().type())
    {
        visit_all(c_arg, a_arg, b_arg)(
            [&](auto cmat, auto amat, auto bmat) { v(cmat, amat, bmat); });
    }
    else
    {
        c_arg.visit([&](auto cmat) {
            visit_all(a_arg, b_arg)([&](auto amat, auto bmat) { v(cmat, amat, bmat); });
        });
    }
}

} // namespace

void gemm(const argument& c_arg, const argument& a_arg, const argument& b_arg)
{
#if MIGRAPHX_USE_EIGEN
    gemm_ref(
        c_arg, a_arg, b_arg, [](auto cmat, auto amat, auto bmat) { gemm_eigen(cmat, amat, bmat); });
#else
    gemm_ref(
        c_arg, a_arg, b_arg, [](auto cmat, auto amat, auto bmat) { gemm_naive(cmat, amat, bmat); });
#endif

}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
