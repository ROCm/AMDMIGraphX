/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_GEMM_HPP
#define MIGRAPHX_GUARD_RTGLIB_GEMM_HPP

#include <migraphx/config.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/float_equal.hpp>
#include <iostream>

#if MIGRAPHX_USE_BLAZE
#include <blaze/Blaze.h>
#include <vector>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#if MIGRAPHX_USE_BLAZE
namespace detail {

template <class T>
using blaze_matrix = blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded>;

template <class T>
constexpr bool is_blaze_native_type()
{
    return std::is_same<T, float>{} or std::is_same<T, double>{};
}

template <class T, class F>
void blaze_gemm_native(T* c_ptr,
                       T* a_ptr,
                       T* b_ptr,
                       std::size_t m,
                       std::size_t n,
                       std::size_t k,
                       F alpha,
                       F beta)
{
    blaze_matrix<T> a(a_ptr, m, k);
    blaze_matrix<T> b(b_ptr, k, n);
    blaze_matrix<T> c(c_ptr, m, n);
    if(float_equal(alpha, F{1}) and float_equal(beta, F{0}))
        c = a * b;
    else
        c = alpha * (a * b) + beta * c;
}

template <class T, class U, class F>
void blaze_gemm_upcast(T* c_ptr,
                       U* a_ptr,
                       U* b_ptr,
                       std::size_t m,
                       std::size_t n,
                       std::size_t k,
                       F alpha,
                       F beta)
{
    std::vector<float> a_buf(m * k);
    std::vector<float> b_buf(k * n);
    std::vector<float> c_buf(m * n);

    for(std::size_t i = 0; i < m * k; i++)
        a_buf[i] = static_cast<float>(a_ptr[i]);
    for(std::size_t i = 0; i < k * n; i++)
        b_buf[i] = static_cast<float>(b_ptr[i]);
    if(not float_equal(beta, F{0}))
    {
        for(std::size_t i = 0; i < m * n; i++)
            c_buf[i] = static_cast<float>(c_ptr[i]);
    }

    blaze_matrix<float> a(a_buf.data(), m, k);
    blaze_matrix<float> b(b_buf.data(), k, n);
    blaze_matrix<float> c(c_buf.data(), m, n);

    if(float_equal(alpha, F{1}) and float_equal(beta, F{0}))
        c = a * b;
    else
        c = alpha * (a * b) + beta * c;

    for(std::size_t i = 0; i < m * n; i++)
        c_ptr[i] = static_cast<T>(c_buf[i]);
}

} // namespace detail
#endif

template <class T, class U, class F>
void gemm(tensor_view<T> cmat, tensor_view<U> amat, tensor_view<U> bmat, F alpha, F beta)
{
    std::size_t n_dims = cmat.get_shape().lens().size();
    std::size_t dim_0  = n_dims - 2;
    std::size_t dim_1  = n_dims - 1;
    auto k             = amat.get_shape().lens()[dim_1];

    assert(amat.get_shape().lens()[dim_1] == bmat.get_shape().lens()[dim_0]);
    assert(cmat.get_shape().lens()[dim_0] == amat.get_shape().lens()[dim_0]);
    assert(cmat.get_shape().lens()[dim_1] == bmat.get_shape().lens()[dim_1]);

#if MIGRAPHX_USE_BLAZE
    // Use Blaze for packed standard-layout shapes (much faster than naive triple loop)
    if(cmat.get_shape().standard() and amat.get_shape().standard() and
       bmat.get_shape().standard())
    {
        auto m_size = amat.get_shape().lens()[dim_0];
        auto n_size = bmat.get_shape().lens()[dim_1];
        auto k_size = k;

        std::size_t num_batches = 1;
        for(std::size_t i = 0; i < dim_0; i++)
            num_batches *= cmat.get_shape().lens()[i];

        auto a_batch_stride = m_size * k_size;
        auto b_batch_stride = k_size * n_size;
        auto c_batch_stride = m_size * n_size;

        std::cerr << "[blaze gemm] " << m_size << "x" << k_size << "x" << n_size
                  << " batches=" << num_batches
                  << (detail::is_blaze_native_type<U>() and std::is_same<T, U>{}
                          ? " (native)"
                          : " (upcast to float)")
                  << std::endl;

        for(std::size_t batch = 0; batch < num_batches; batch++)
        {
            if constexpr(detail::is_blaze_native_type<U>() and std::is_same<T, U>{})
            {
                detail::blaze_gemm_native(cmat.data() + batch * c_batch_stride,
                                          amat.data() + batch * a_batch_stride,
                                          bmat.data() + batch * b_batch_stride,
                                          m_size,
                                          n_size,
                                          k_size,
                                          alpha,
                                          beta);
            }
            else
            {
                detail::blaze_gemm_upcast(cmat.data() + batch * c_batch_stride,
                                          amat.data() + batch * a_batch_stride,
                                          bmat.data() + batch * b_batch_stride,
                                          m_size,
                                          n_size,
                                          k_size,
                                          alpha,
                                          beta);
            }
        }
        return;
    }
#endif

    // Fallback: naive element-wise GEMM for non-standard shapes
    std::cerr << "[gemm fallback] non-standard shape, using naive loop" << std::endl;
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
        cmat(c_idx.begin(), c_idx.end()) = alpha * s + cmat(c_idx.begin(), c_idx.end()) * beta;
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
