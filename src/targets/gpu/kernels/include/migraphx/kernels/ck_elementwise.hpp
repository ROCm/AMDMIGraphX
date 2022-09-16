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

#include <stdio.h>

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

using ADataType          = ck::half_t;
using BDataType          = ck::half_t;
using CDataType          = ck::half_t;
using ElementwiseFunctor = ck::half_t;

static constexpr auto I0 = ck::Number<0>{};

template <ck::index_t ndim>
struct CKBinaryElementwise
{
    template <class Desc_M>
    __device__ constexpr auto PadDescriptor_M_1d(Desc_M desc_m)
    {
        auto gridSize               = 72;
        auto blockSize              = 1024;
        auto MPerThread             = 8;
        const auto M                = desc_m.GetLength(I0);
        const ck::index_t loop_step = gridSize * blockSize * MPerThread;
        const auto pad              = ck::math::integer_least_multiple(M, loop_step) - M;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(ck::make_right_pad_transform(M, pad)),
                                        make_tuple(ck::Sequence<0>{}),
                                        make_tuple(ck::Sequence<0>{}));
        return desc_m_pad;
    }

    template <class L, class S>
    __device__ constexpr auto MakeDescriptor_M(const L& lengths, const S& strides)
    {
        auto tupleOfShape = generate_tuple(
            [&](auto I) { return static_cast<ck::index_t>(lengths[I]); }, ck::Number<ndim>{});
        auto tupleOfStride = generate_tuple(
            [&](auto I) { return static_cast<ck::index_t>(strides[I]); }, ck::Number<ndim>{});
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);
        // merge nd to 1d desc - [s0 * s1 * ...]
        if constexpr(ndim > 1)
        {
            const auto desc_m = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(tupleOfShape)),
                make_tuple(generate_sequence_v2([&](auto I) { return I; }, ck::Number<ndim>{})),
                make_tuple(ck::Sequence<0>{}));
            return PadDescriptor_M_1d(desc_m);
        }
        else
        {
            return PadDescriptor_M_1d(desc);
        }
    }
};

template <ck::index_t ndim>
struct CKBinaryElementwise2
{
    template <class Desc_M>
    /* constexpr */ __device__ auto PadDescriptor_M_1d(Desc_M desc_m)
    {
        auto gridSize               = 72;
        auto blockSize              = 1024;
        auto MPerThread             = 8;
        const auto M                = desc_m.GetLength(I0);
        const ck::index_t loop_step = gridSize * blockSize * MPerThread;
        const auto pad              = ck::math::integer_least_multiple(M, loop_step) - M;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(ck::make_right_pad_transform(M, pad)),
                                        make_tuple(ck::Sequence<0>{}),
                                        make_tuple(ck::Sequence<0>{}));
        return desc_m_pad;
    }

    template <class L, class S>
    /* constexpr */ __device__ auto MakeDescriptor_M(const L& lengths, const S& strides)
    {
        auto tupleOfShape = generate_tuple(
            [&](auto I) { return static_cast<ck::index_t>(lengths[I]); }, ck::Number<ndim>{});
        auto tupleOfStride = generate_tuple(
            [&](auto I) {
                printf("Stride %i: %i\n", int(I), int(strides[I]));
                return static_cast<ck::index_t>(strides[I]);
            },
            ck::Number<ndim>{});
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);
        // merge nd to 1d desc - [s0 * s1 * ...]
        if constexpr(ndim > 1)
        {
            const auto desc_m = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(tupleOfShape)),
                make_tuple(generate_sequence_v2([&](auto I) { return I; }, ck::Number<ndim>{})),
                make_tuple(ck::Sequence<0>{}));
            return PadDescriptor_M_1d(desc_m);
        }
        else
        {
            return PadDescriptor_M_1d(desc);
        }
    }
};

struct Add
{
    template <typename Y, typename X0, typename X1>
    __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        y = x0 + x1;
    };
};

struct Mul
{
    template <typename Y, typename X0, typename X1>
    __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        y = x0 * x1;
    };
};

struct Div
{
    template <typename Y, typename X0, typename X1>
    __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const
    {
        y = x0 / x1;
    };
};

template <class T, class U, class V>
__device__ void ck_elementwise(const T& a_t, const U& b_t, const V& c_t)
{
    // auto idx = make_index();
    constexpr auto a_lens        = get_shape_c<T>{}.lens;
    constexpr auto a_strides     = get_shape_c<T>{}.strides;
    constexpr ck::index_t a_ndim = a_lens.size(); // decltype(a_lens.size()){};
    // if (idx.global == 0)
    //     printf("a_ndim: %i\n", int(a_ndim));
    auto a_bin_op         = CKBinaryElementwise<a_ndim>{};
    constexpr auto a_desc = a_bin_op.MakeDescriptor_M(a_lens, a_strides);

    constexpr auto b_lens        = get_shape_c<U>{}.lens;
    constexpr auto b_strides     = get_shape_c<U>{}.strides;
    constexpr ck::index_t b_ndim = b_lens.size(); // decltype(b_lens.size()){};
    // if (idx.global == 0)
    //     printf("b_ndim: %i\n", int(b_ndim));
    auto b_bin_op         = CKBinaryElementwise<b_ndim>{};
    constexpr auto b_desc = b_bin_op.MakeDescriptor_M(b_lens, b_strides);

    constexpr auto c_lens        = get_shape_c<V>{}.lens;
    constexpr auto c_strides     = get_shape_c<V>{}.strides;
    constexpr ck::index_t c_ndim = c_lens.size(); // decltype(c_lens.size()){};
    auto c_bin_op                = CKBinaryElementwise<c_ndim>{};
    constexpr auto c_desc        = c_bin_op.MakeDescriptor_M(c_lens, c_strides);

    using AGridDesc_M                      = decltype(a_desc);
    using BGridDesc_M                      = decltype(b_desc);
    using CGridDesc_M                      = decltype(c_desc);
    constexpr ck::index_t MPerThread       = 8;
    constexpr ck::index_t AScalarPerVector = 8;
    constexpr ck::index_t BScalarPerVector = 8;
    constexpr ck::index_t CScalarPerVector = 8;
    using GridwiseBinEltwise               = ck::GridwiseBinaryElementwise_1D<ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                CDataType,
                                                                AGridDesc_M,
                                                                BGridDesc_M,
                                                                CGridDesc_M,
                                                                Add,
                                                                MPerThread,
                                                                AScalarPerVector,
                                                                BScalarPerVector,
                                                                CScalarPerVector>;
    auto op                                = Add{};
    GridwiseBinEltwise::Run(a_t.data(), b_t.data(), c_t.data(), a_desc, b_desc, c_desc, op);
}

} // namespace migraphx
#endif
