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
#include <algorithm>
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
using eigen_stride    = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

struct batch_slicer
{
    batch_slicer(const shape& mat_shape)
    {
        auto n_batch_dims = mat_shape.ndim() - 2;
        inner_shape       = shape{mat_shape.type(),
                                  {mat_shape.lens().end() - 2, mat_shape.lens().end()},
                                  {mat_shape.strides().end() - 2, mat_shape.strides().end()}};
        if(n_batch_dims > 0)
        {
            outer_shape =
                shape{mat_shape.type(),
                      {mat_shape.lens().begin(), mat_shape.lens().begin() + n_batch_dims},
                      {mat_shape.strides().begin(), mat_shape.strides().begin() + n_batch_dims}};
        }
    }

    template <class T>
    tensor_view<T> extract(tensor_view<T> view, std::size_t batch) const
    {
        std::size_t offset = 0;
        if(not outer_shape.lens().empty())
            offset = outer_shape.index(batch);
        return make_view(inner_shape, view.data() + offset);
    }

    std::size_t num_batches() const
    {
        if(outer_shape.lens().empty())
            return 1;
        return outer_shape.elements();
    }

    shape inner_shape;
    shape outer_shape;
};

template <class T>
auto make_eigen_map(tensor_view<T> view)
{
    const auto& s          = view.get_shape();
    auto dim_0             = s.ndim() - 2;
    auto dim_1             = s.ndim() - 1;
    Eigen::Index rows      = s.lens()[dim_0];
    Eigen::Index cols      = s.lens()[dim_1];
    Eigen::Index rowstride = s.strides()[dim_0];
    Eigen::Index colstride = s.strides()[dim_1];
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
                      Eigen::Unaligned,
                      eigen_stride>{view.data(), rows, cols, eigen_stride{rowstride, colstride}};
}

template <class T>
void eigen_multiply(tensor_view<T> cmat, tensor_view<T> amat, tensor_view<T> bmat)
{
    batch_slicer slicer(cmat.get_shape());
    batch_slicer a_slicer(amat.get_shape());
    batch_slicer b_slicer(bmat.get_shape());

    par_for(slicer.num_batches(), [&](auto batch) {
        auto a_slice = a_slicer.extract(amat, batch);
        auto b_slice = b_slicer.extract(bmat, batch);
        auto c_slice = slicer.extract(cmat, batch);

        auto a      = make_eigen_map(a_slice);
        auto b      = make_eigen_map(b_slice);
        auto c      = make_eigen_map(c_slice);
        c.noalias() = a * b;
    });
}

template <class AccType, class T, class U>
void gemm_eigen_with_copy(tensor_view<T> cmat, tensor_view<U> amat, tensor_view<U> bmat)
{
    std::vector<AccType> a_buf(amat.get_shape().elements());
    std::copy(amat.begin(), amat.end(), a_buf.begin());
    auto amat_flat =
        make_view(amat.get_shape().as_standard().with_type(shape::get_type<AccType>{}), a_buf.data());

    std::vector<AccType> b_buf(bmat.get_shape().elements());
    std::copy(bmat.begin(), bmat.end(), b_buf.begin());
    auto bmat_flat =
        make_view(bmat.get_shape().as_standard().with_type(shape::get_type<AccType>{}), b_buf.data());

    std::vector<AccType> c_buf(cmat.get_shape().elements(), AccType{0});
    auto c_shape_std = cmat.get_shape().as_standard().with_type(shape::get_type<AccType>{});
    auto cmat_flat   = make_view(c_shape_std, c_buf.data());

    eigen_multiply(cmat_flat, amat_flat, bmat_flat);

    std::copy(c_buf.begin(), c_buf.end(), cmat.begin());
}

template <class T, class U>
void gemm_eigen(tensor_view<T> cmat, tensor_view<U> amat, tensor_view<U> bmat)
{
    if constexpr(std::is_same<T, U>{} and (std::is_same<T, float>{} or std::is_same<T, double>{}))
    {
        eigen_multiply(cmat, amat, bmat);
    }
    else if constexpr(std::is_integral<U>{})
    {
        gemm_eigen_with_copy<int32_t>(cmat, amat, bmat);
    }
    else
    {
        gemm_eigen_with_copy<float>(cmat, amat, bmat);
    }
}

#endif

template <class Visitor>
void gemm_ref_visit(const argument& c_arg, const argument& a_arg, const argument& b_arg, Visitor v)
{
    c_arg.visit([&](auto cmat) {
        visit_all(a_arg, b_arg)([&](auto amat, auto bmat) { v(cmat, amat, bmat); });
    });
}

} // namespace

void gemm(const argument& c_arg, const argument& a_arg, const argument& b_arg)
{
#if MIGRAPHX_USE_EIGEN
    gemm_ref_visit(
        c_arg, a_arg, b_arg, [](auto cmat, auto amat, auto bmat) { gemm_eigen(cmat, amat, bmat); });
#else
    gemm_ref_visit(
        c_arg, a_arg, b_arg, [](auto cmat, auto amat, auto bmat) { gemm_naive(cmat, amat, bmat); });
#endif
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
