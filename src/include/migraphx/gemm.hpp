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
using blaze_row_major = blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded, blaze::rowMajor>;
template <class T>
using blaze_col_major =
    blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded, blaze::columnMajor>;

template <class T>
constexpr bool is_blaze_native_type()
{
    return std::is_same<T, float>{} or std::is_same<T, double>{};
}

enum class mat_order
{
    row_major,
    col_major,
    unsupported
};

struct blaze_layout
{
    mat_order order   = mat_order::unsupported;
    std::size_t spacing = 0;
    bool packed         = false;
};

inline blaze_layout get_blaze_layout(const shape& s)
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

inline std::size_t batch_offset(const shape& s, std::size_t batch, std::size_t n_batch_dims)
{
    std::size_t offset    = 0;
    std::size_t remaining = batch;
    for(std::size_t d = n_batch_dims; d > 0; d--)
    {
        offset += (remaining % s.lens()[d - 1]) * s.strides()[d - 1];
        remaining /= s.lens()[d - 1];
    }
    return offset;
}

// Dispatch to Blaze matrix with correct storage order and call continuation
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

// Copy from arbitrary layout to contiguous row-major float buffer
template <class U>
void copy_to_rm_float(float* dst,
                      const U* src,
                      std::size_t rows,
                      std::size_t cols,
                      mat_order order,
                      std::size_t spacing)
{
    if(order == mat_order::row_major and spacing == cols)
    {
        for(std::size_t i = 0; i < rows * cols; i++)
            dst[i] = static_cast<float>(src[i]);
    }
    else
    {
        for(std::size_t i = 0; i < rows; i++)
            for(std::size_t j = 0; j < cols; j++)
            {
                auto src_idx = (order == mat_order::row_major) ? (i * spacing + j)
                                                               : (j * spacing + i);
                dst[i * cols + j] = static_cast<float>(src[src_idx]);
            }
    }
}

// Copy from contiguous row-major float buffer to arbitrary layout
template <class T>
void copy_from_rm_float(T* dst,
                        const float* src,
                        std::size_t rows,
                        std::size_t cols,
                        mat_order order,
                        std::size_t spacing)
{
    if(order == mat_order::row_major and spacing == cols)
    {
        for(std::size_t i = 0; i < rows * cols; i++)
            dst[i] = static_cast<T>(src[i]);
    }
    else
    {
        for(std::size_t i = 0; i < rows; i++)
            for(std::size_t j = 0; j < cols; j++)
            {
                auto dst_idx = (order == mat_order::row_major) ? (i * spacing + j)
                                                               : (j * spacing + i);
                dst[dst_idx] = static_cast<T>(src[i * cols + j]);
            }
    }
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
    auto a_layout = detail::get_blaze_layout(amat.get_shape());
    auto b_layout = detail::get_blaze_layout(bmat.get_shape());
    auto c_layout = detail::get_blaze_layout(cmat.get_shape());

    if(a_layout.order != detail::mat_order::unsupported and
       b_layout.order != detail::mat_order::unsupported and
       c_layout.order != detail::mat_order::unsupported)
    {
        auto m_size = amat.get_shape().lens()[dim_0];
        auto n_size = bmat.get_shape().lens()[dim_1];
        auto k_size = k;

        std::size_t num_batches = 1;
        for(std::size_t i = 0; i < dim_0; i++)
            num_batches *= cmat.get_shape().lens()[i];

        bool all_packed = a_layout.packed and b_layout.packed and c_layout.packed;
        bool native     = detail::is_blaze_native_type<U>() and std::is_same<T, U>{};

        auto order_char = [](detail::mat_order o) {
            return o == detail::mat_order::col_major ? "T" : "N";
        };
        std::cerr << "[blaze gemm] " << m_size << "x" << k_size << "x" << n_size
                  << " batches=" << num_batches << " layout="
                  << order_char(a_layout.order) << order_char(b_layout.order)
                  << order_char(c_layout.order)
                  << (native and all_packed ? " (native)" : " (upcast to float)") << std::endl;

        // Native path: direct Blaze GEMM with correct storage orders (no copy)
        if constexpr(detail::is_blaze_native_type<U>() and std::is_same<T, U>{})
        {
            if(all_packed)
            {
                for(std::size_t batch = 0; batch < num_batches; batch++)
                {
                    auto a_off = detail::batch_offset(amat.get_shape(), batch, dim_0);
                    auto b_off = detail::batch_offset(bmat.get_shape(), batch, dim_0);
                    auto c_off = detail::batch_offset(cmat.get_shape(), batch, dim_0);

                    detail::with_blaze_mat(
                        amat.data() + a_off, m_size, k_size, a_layout.order, [&](auto& a_mat) {
                            detail::with_blaze_mat(
                                bmat.data() + b_off,
                                k_size,
                                n_size,
                                b_layout.order,
                                [&](auto& b_mat) {
                                    detail::with_blaze_mat(
                                        cmat.data() + c_off,
                                        m_size,
                                        n_size,
                                        c_layout.order,
                                        [&](auto& c_mat) {
                                            if(float_equal(alpha, F{1}) and
                                               float_equal(beta, F{0}))
                                                c_mat = a_mat * b_mat;
                                            else
                                                c_mat = alpha * (a_mat * b_mat) + beta * c_mat;
                                        });
                                });
                        });
                }
                return;
            }
        }

        // Copy path: handles non-native types, padded layouts, or both.
        // Copies each batch slice to contiguous row-major float buffers,
        // runs Blaze GEMM, and copies back with layout conversion.
        std::vector<float> a_buf(m_size * k_size);
        std::vector<float> b_buf(k_size * n_size);
        std::vector<float> c_buf(m_size * n_size);

        for(std::size_t batch = 0; batch < num_batches; batch++)
        {
            auto a_off = detail::batch_offset(amat.get_shape(), batch, dim_0);
            auto b_off = detail::batch_offset(bmat.get_shape(), batch, dim_0);
            auto c_off = detail::batch_offset(cmat.get_shape(), batch, dim_0);

            detail::copy_to_rm_float(
                a_buf.data(), amat.data() + a_off, m_size, k_size,
                a_layout.order, a_layout.spacing);
            detail::copy_to_rm_float(
                b_buf.data(), bmat.data() + b_off, k_size, n_size,
                b_layout.order, b_layout.spacing);
            if(not float_equal(beta, F{0}))
                detail::copy_to_rm_float(
                    c_buf.data(), cmat.data() + c_off, m_size, n_size,
                    c_layout.order, c_layout.spacing);

            detail::blaze_row_major<float> a(a_buf.data(), m_size, k_size);
            detail::blaze_row_major<float> b(b_buf.data(), k_size, n_size);
            detail::blaze_row_major<float> c(c_buf.data(), m_size, n_size);

            if(float_equal(alpha, F{1}) and float_equal(beta, F{0}))
                c = a * b;
            else
                c = alpha * (a * b) + beta * c;

            detail::copy_from_rm_float(
                cmat.data() + c_off, c_buf.data(), m_size, n_size,
                c_layout.order, c_layout.spacing);
        }
        return;
    }
#endif

    // Fallback: naive element-wise GEMM for unsupported layouts
    std::cerr << "[gemm fallback] unsupported layout, using naive loop" << std::endl;
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
