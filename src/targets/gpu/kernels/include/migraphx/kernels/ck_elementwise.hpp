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
#include <migraphx/kernels/tensor_view.hpp>

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

template <class L, class S, class N>
constexpr auto MakeDescriptor_M(const L& lengths, const S& strides, const N& /* ndim */)
{
    auto gridSize = 72;
    auto blockSize = 1024; 
    constexpr auto ndim = 1;
    //auto idx          = make_index();
    auto tupleOfShape = generate_tuple([&](auto I) { return static_cast<ck::index_t>(lengths[I]); },
                                       ck::Number<ndim>{});
    auto tupleOfStride = generate_tuple(
        [&](auto I) { return static_cast<ck::index_t>(strides[I]); }, ck::Number<1>{});
    const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);
    auto desc_m = desc;
    // merge nd to 1d desc - [s0 * s1 * ...]
    if constexpr(ndim > 1)
    {
        desc_m = transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform(tupleOfShape)),
            make_tuple(generate_sequence_v2([&](auto I) { return I; }, ck::Number<ndim>{})),
            make_tuple(ck::Sequence<0>{}));
    }

    const auto M            = desc_m.GetLength(I0);
    const ck::index_t loop_step = /* idx.nglobal(); // */ gridSize * blockSize/*  * MPerThread */;
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
    __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        y = x0 + x1;
    };
};

template <class T, class U, class V>
__device__ void ck_elementwise(const T& a_t, const U& b_t, const V& c_t)
{
    auto idx = make_index();
    if (idx.global == 0)
    {
        constexpr auto lengths = get_shape_c<T>{}.lens;
        constexpr auto strides = get_shape_c<T>{}.strides;
        constexpr auto a_desc  = MakeDescriptor_M(lengths, strides, 1);

        using AGridDesc_M = decltype(a_desc);
        using GridwiseBinEltwise = ck::GridwiseBinaryElementwise_1D<ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    CDataType,
                                                                    AGridDesc_M,
                                                                    AGridDesc_M,
                                                                    AGridDesc_M,
                                                                    Add,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    1>;
        auto op                  = Add{};
        GridwiseBinEltwise::Run(a_t.data(), b_t.data(), c_t.data(), a_desc, a_desc, a_desc, op);
    }
}

} // namespace migraphx
#endif
