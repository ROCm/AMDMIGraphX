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
#include <migraphx/env.hpp>
#include <iostream>
#include <numeric>
#include <vector>

#if MIGRAPHX_USE_BLAZE
#include <blaze/Blaze.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_BLAZE_DEBUG)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_BLAZE)

namespace {

template <class T, class U>
void gemm_naive(tensor_view<T> cmat, tensor_view<U> amat, tensor_view<U> bmat)
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

#if MIGRAPHX_USE_BLAZE

template <class T>
using blaze_row_major = blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded, blaze::rowMajor>;
template <class T>
using blaze_col_major =
    blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded, blaze::columnMajor>;

enum class mat_order
{
    row_major,
    col_major,
    unsupported
};

struct blaze_layout
{
    mat_order order     = mat_order::unsupported;
    std::size_t spacing = 0;
    bool packed         = false;
};

blaze_layout get_blaze_layout(const shape& s)
{
    if(s.ndim() < 2)
        return {};
    auto dim_0 = s.ndim() - 2;
    auto dim_1 = s.ndim() - 1;
    auto s0    = s.strides()[dim_0];
    auto s1    = s.strides()[dim_1];
    auto rows  = s.lens()[dim_0];
    auto cols  = s.lens()[dim_1];
    if(s1 == 1 and s0 >= cols)
        return {mat_order::row_major, s0, s0 == cols};
    if(s0 == 1 and s1 >= rows)
        return {mat_order::col_major, s1, s1 == rows};
    return {};
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
constexpr bool is_blaze_native_type()
{
    return std::is_same<T, float>{} or std::is_same<T, double>{};
}

template <class T, class Func>
void with_blaze_mat(T* ptr, std::size_t rows, std::size_t cols, mat_order order, Func&& func)
{
    if(order == mat_order::col_major)
    {
        blaze_col_major<T> mat(ptr, rows, cols);
        func(mat);
    }
    else
    {
        blaze_row_major<T> mat(ptr, rows, cols);
        func(mat);
    }
}

template <class T>
void blaze_gemm_native(T* c, mat_order c_order,
                       const T* a, mat_order a_order,
                       const T* b, mat_order b_order,
                       std::size_t m, std::size_t k, std::size_t n)
{
    with_blaze_mat(const_cast<T*>(a), m, k, a_order, [&](const auto& a_mat) {
        with_blaze_mat(const_cast<T*>(b), k, n, b_order, [&](const auto& b_mat) {
            with_blaze_mat(c, m, n, c_order, [&](auto& c_mat) { c_mat = a_mat * b_mat; });
        });
    });
}

template <class T, class U>
void gemm_blaze(tensor_view<T> cmat, tensor_view<U> amat, tensor_view<U> bmat)
{
    std::size_t n_dims = cmat.get_shape().ndim();
    std::size_t dim_0  = n_dims - 2;
    std::size_t dim_1  = n_dims - 1;

    auto m_size = amat.get_shape().lens()[dim_0];
    auto k_size = amat.get_shape().lens()[dim_1];
    auto n_size = bmat.get_shape().lens()[dim_1];

    const auto& clens   = cmat.get_shape().lens();
    auto num_batches = std::accumulate(
        clens.begin(), clens.begin() + dim_0, std::size_t{1}, std::multiplies<>{});

    auto a_batch = make_batch_shape(amat.get_shape(), dim_0);
    auto b_batch = make_batch_shape(bmat.get_shape(), dim_0);
    auto c_batch = make_batch_shape(cmat.get_shape(), dim_0);

    // Native zero-copy path for float/double with packed layouts
    if constexpr(is_blaze_native_type<U>() and std::is_same<T, U>{})
    {
        auto a_layout = get_blaze_layout(amat.get_shape());
        auto b_layout = get_blaze_layout(bmat.get_shape());
        auto c_layout = get_blaze_layout(cmat.get_shape());

        if(a_layout.packed and b_layout.packed and c_layout.packed)
        {
            if(enabled(MIGRAPHX_BLAZE_DEBUG{}))
            {
                std::cerr << "[blaze gemm] " << m_size << "x" << k_size << "x" << n_size
                          << " batches=" << num_batches << " (native)" << std::endl;
            }

            for(std::size_t batch = 0; batch < num_batches; batch++)
            {
                blaze_gemm_native(
                    cmat.data() + c_batch.index(batch), c_layout.order,
                    amat.data() + a_batch.index(batch), a_layout.order,
                    bmat.data() + b_batch.index(batch), b_layout.order,
                    m_size, k_size, n_size);
            }
            return;
        }
    }

    // Copy path: convert to contiguous row-major float, run Blaze, copy back
    auto a_row_stride = amat.get_shape().strides()[dim_0];
    auto a_col_stride = amat.get_shape().strides()[dim_1];
    auto b_row_stride = bmat.get_shape().strides()[dim_0];
    auto b_col_stride = bmat.get_shape().strides()[dim_1];
    auto c_row_stride = cmat.get_shape().strides()[dim_0];
    auto c_col_stride = cmat.get_shape().strides()[dim_1];

    if(enabled(MIGRAPHX_BLAZE_DEBUG{}))
    {
        std::cerr << "[blaze gemm] " << m_size << "x" << k_size << "x" << n_size
                  << " batches=" << num_batches << " (copy to float)" << std::endl;
    }

    std::vector<float> a_buf(m_size * k_size);
    std::vector<float> b_buf(k_size * n_size);
    std::vector<float> c_buf(m_size * n_size);

    for(std::size_t batch = 0; batch < num_batches; batch++)
    {
        copy_2d(a_buf.data(), k_size, std::size_t{1},
                amat.data() + a_batch.index(batch), a_row_stride, a_col_stride,
                m_size, k_size);
        copy_2d(b_buf.data(), n_size, std::size_t{1},
                bmat.data() + b_batch.index(batch), b_row_stride, b_col_stride,
                k_size, n_size);

        blaze_row_major<float> a(a_buf.data(), m_size, k_size);
        blaze_row_major<float> b(b_buf.data(), k_size, n_size);
        blaze_row_major<float> c(c_buf.data(), m_size, n_size);
        c = a * b;

        copy_2d(cmat.data() + c_batch.index(batch), c_row_stride, c_col_stride,
                c_buf.data(), n_size, std::size_t{1},
                m_size, n_size);
    }
}

#endif

template <class Visitor>
void dispatch_gemm(const argument& c_arg, const argument& a_arg, const argument& b_arg, Visitor v)
{
    if(c_arg.get_shape().type() == a_arg.get_shape().type())
    {
        visit_all(c_arg, a_arg, b_arg)(
            [&](auto cmat, auto amat, auto bmat) { v(cmat, amat, bmat); });
    }
    else
    {
        c_arg.visit([&](auto cmat) {
            visit_all(a_arg, b_arg)(
                [&](auto amat, auto bmat) { v(cmat, amat, bmat); });
        });
    }
}

} 

void gemm(const argument& c_arg, const argument& a_arg, const argument& b_arg)
{
#if MIGRAPHX_USE_BLAZE
    if(not enabled(MIGRAPHX_DISABLE_BLAZE{}))
    {
        dispatch_gemm(c_arg, a_arg, b_arg,
                      [](auto cmat, auto amat, auto bmat) { gemm_blaze(cmat, amat, bmat); });
        return;
    }
#endif
    dispatch_gemm(c_arg, a_arg, b_arg,
                  [](auto cmat, auto amat, auto bmat) { gemm_naive(cmat, amat, bmat); });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
