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
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

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

// Strided 2D GEMM
template <class T, class U, class F>
void gemm(std::size_t M,
          std::size_t N,
          std::size_t K,
          std::size_t lda,
          std::size_t ldb,
          std::size_t ldc,
          T cmat,
          U amat,
          U bmat,
          F alpha,
          F beta,
          shape::type_t dtype,
          const bool b_transpose = false)
{
    auto cs     = shape{dtype, {M, N}};
    auto a_idx  = [&](auto i, auto k) { return k + (i * lda); };
    auto b_idx  = [&](auto k, auto j) { return j + (k * ldb); };
    auto bt_idx = [&](auto k, auto j) { return j + (k * ldb); };
    auto c_idx  = [&](auto i, auto j) { return j + (i * ldc); };

    par_for(cs.elements(), [&](auto i) {
        auto c_midx = cs.multi(i);
        auto ii     = c_midx[0];
        auto jj     = c_midx[1];
        double s    = 0.0;
        dfor(K)([&](auto kk) {
            auto a_i = a_idx(ii, kk);
            auto b_i = b_transpose ? bt_idx(jj, kk) : b_idx(kk, jj);
            s += static_cast<double>(amat[a_i]) * static_cast<double>(bmat[b_i]);
        });
        auto c_i  = c_idx(ii, jj);
        cmat[c_i] = static_cast<double>(alpha) * s + cmat[c_i] * static_cast<double>(beta);
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
