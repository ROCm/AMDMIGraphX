/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_ELEMENTWISE_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_ELEMENTWISE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>

#include "ck/device_utility/device_prop.hpp"
#include "ck/device_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_binary_elementwise_1d.hpp"

namespace migraphx {

using ADataType          = float;
using BDataType          = float;
using CDataType          = float;
using ElementwiseFunctor = float;

static constexpr auto I0 = ck::Number<0>{};
using index_t            = index_int;

template <class L, class S>
__host__ __device__ constexpr auto MakeDescriptor_M(const L& lengths, const S& strides)
{
    auto idx          = make_index();
    auto tupleOfShape = generate_tuple([&](auto I) { return static_cast<ck::index_t>(lengths[I]); },
                                       ck::Number<1>{});
    auto tupleOfStride = generate_tuple(
        [&](auto I) { return static_cast<ck::index_t>(strides[I]); }, ck::Number<1>{});
    const auto desc_m = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

    const auto M            = desc_m.GetLength(I0);
    const index_t loop_step = idx.nglobal(); // gridSize * blockSize * MPerThread;
    const auto pad          = ck::math::integer_least_multiple(M, loop_step) - M;
    const auto desc_m_pad =
        transform_tensor_descriptor(desc_m,
                                    make_tuple(ck::make_right_pad_transform(M, pad)),
                                    make_tuple(ck::Sequence<0>{}),
                                    make_tuple(ck::Sequence<0>{}));
    return desc_m_pad;
}

struct Add
{
    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 + x1;
    };
};

template <class T, class U, class V>
__device__ void ck_elementwise(const T& a_t, const U& b_t, const V& c_t)
{
    // auto add = [](auto a, auto b) { return a + b; };
    auto lengths = a_t.get_shape().lens;
    auto strides = a_t.get_shape().strides;
    auto a_desc  = MakeDescriptor_M(lengths, strides);

    using AGridDesc_M = decltype(a_desc);
    // using Add = ck::tensor_operation::element_wise::Add;
    using GridwiseBinEltwise = ck::GridwiseBinaryElementwise_1D<ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                CDataType,
                                                                AGridDesc_M,
                                                                AGridDesc_M,
                                                                AGridDesc_M,
                                                                Add,
                                                                8,
                                                                8,
                                                                8,
                                                                8>;
    auto op                  = Add{};
    GridwiseBinEltwise::Run(a_t.data(), b_t.data(), c_t.data(), a_desc, a_desc, a_desc, op);
    // auto kernel = ck::kernel_binary_elementwise_1d<GridwiseBinEltwise,
    //                                     ADataType,
    //                                     BDataType,
    //                                     CDataType,
    //                                     AGridDesc_M,
    //                                     AGridDesc_M,
    //                                     AGridDesc_M,
    //                                     Add>;
    // kernel(a_t.data(), b_t.data(), c_t.data(), a_desc, a_desc, a_desc, Add);

    // Argument arg{a_t.data(), b_t.data(), c_t.data(), c_t.get_shape().lens,
    // a_t.get_shape().strides, b_t.get_shape().strides, c_t.get_shape().strides,
    //     add};
    // auto lengths = a_t.get_shape().lens;
    // auto strides = a_t.get_shape().strides;
    // auto idx     = make_index();
    // b_t.get_shape();
    // c_t.get_shape();
    // auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, ck::Number<1>{});
    // auto tupleOfStride = generate_tuple([&](auto I) { return strides[I]; }, ck::Number<1>{});
    // const auto desc_m = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

    // const auto M            = desc_m.GetLength(I0);
    // const ck::index_t loop_step = idx.nglobal();//gridSize * blockSize * MPerThread;
    // const auto pad          = ck::math::integer_least_multiple(M, loop_step) - M;
    // const auto desc_m_pad =
    //     transform_tensor_descriptor(desc_m,
    //                                 make_tuple(ck::make_right_pad_transform(M, pad)),
    //                                 make_tuple(ck::Sequence<0>{}),
    //                                 make_tuple(ck::Sequence<0>{}));
}

} // namespace migraphx
#endif
