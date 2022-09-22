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

// #include "ck/device_utility/device_prop.hpp"
// #include "ck/device_utility/kernel_launch.hpp"
//#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include <ck/ck.hpp>
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_1d.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

namespace migraphx {

using ABDataType         = ck::half_t;
using CDataType          = ck::half_t;
using ElementwiseFunctor = ck::half_t;

static constexpr auto I0 = ck::Number<0>{};

// template <typename InDataTypeTuple,
//           typename OutDataTypeTuple,
//           typename ElementwiseOperation,
//           index_t NumDim,
//           index_t MPerThread,
//           typename InScalarPerVectorSeq,
//           typename OutScalarPerVectorSeq>
// struct CKDeviceElementwise
// {
//     __device__ constexpr auto GenerateInDataTypePointerTuple()
//     {
//         return generate_tuple(
//             [&](auto I) {
//                 using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;

//                 return static_cast<const DataType*>(nullptr);
//             },
//             Number<NumInput>{});
//     };

//     __device__ constexpr auto GenerateOutDataTypePointerTuple()
//     {
//         return generate_tuple(
//             [&](auto I) {
//                 using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;

//                 return static_cast<DataType*>(nullptr);
//             },
//             Number<NumOutput>{});
//     };

//     template <class Desc_M>
//     __device__ constexpr auto PadDescriptor_M_1d(Desc_M desc_m)
//     {
//         auto gridSize               = 72;
//         auto blockSize              = 1024;
//         auto MPerThread             = 8;
//         const auto M                = desc_m.GetLength(I0);
//         const ck::index_t loop_step = gridSize * blockSize * MPerThread;
//         const auto pad              = ck::math::integer_least_multiple(M, loop_step) - M;
//         const auto desc_m_pad =
//             transform_tensor_descriptor(desc_m,
//                                         make_tuple(ck::make_right_pad_transform(M, pad)),
//                                         make_tuple(ck::Sequence<0>{}),
//                                         make_tuple(ck::Sequence<0>{}));
//         return desc_m_pad;
//     }

//     template <class L, class S>
//     __device__ constexpr auto MakeDescriptor_M(const L& lengths, const S& strides)
//     {
//         auto tupleOfShape = generate_tuple(
//             [&](auto I) { return static_cast<ck::index_t>(lengths[I]); }, ck::Number<ndim>{});
//         auto tupleOfStride = generate_tuple(
//             [&](auto I) { return static_cast<ck::index_t>(strides[I]); }, ck::Number<ndim>{});
//         const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);
//         // merge nd to 1d desc - [s0 * s1 * ...]
//         if constexpr(ndim > 1)
//         {
//             const auto desc_m = transform_tensor_descriptor(
//                 desc,
//                 make_tuple(make_merge_transform(tupleOfShape)),
//                 make_tuple(generate_sequence_v2([&](auto I) { return I; }, ck::Number<ndim>{})),
//                 make_tuple(ck::Sequence<0>{}));
//             return PadDescriptor_M_1d(desc_m);
//         }
//         else
//         {
//             return PadDescriptor_M_1d(desc);
//         }
//     }

//     template <index_t TupleSize>
//     __device__ constexpr auto GenerateInOutGrid1dDescTuple(Number<TupleSize>)
//     {
//         return generate_tuple(
//             [&](auto) {
//                 if constexpr(NumDim > 1)
//                 {
//                     return MakeDescriptor_M({1, 1}, {1, 1}, 1, 1);
//                 }
//                 else
//                 {
//                     return MakeDescriptor_M({1}, {1}, 1, 1);
//                 };
//             },
//             Number<TupleSize>{});
//     };
// };

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

using InDataTypeTuple            = ck::Tuple<ABDataType, ABDataType>;
using OutDataTypeTuple           = ck::Tuple<CDataType>;
using ElementwiseOperation       = Add;
static constexpr auto MPerThread = 8;
using InScalarPerVectorSeq       = ck::Sequence<1, 8>;
using OutScalarPerVectorSeq      = ck::Sequence<8>;

// using DeviceElementwiseAddInstance =
//     ck::tensor_operation::device::DeviceElementwise<ck::Tuple<ABDataType, ABDataType>,
//                                                     ck::Tuple<CDataType>,
//                                                     Add,
//                                                     3,
//                                                     8,
//                                                     ck::Sequence<1, 8>,
//                                                     ck::Sequence<8>>;

template <class T, class U, class V>
__device__ void ck_elementwise(const T& a_t, const U& b_t, const V& c_t)
{
    // auto idx = make_index();
    constexpr auto a_lens        = get_shape_c<T>{}.lens;
    constexpr auto a_strides     = get_shape_c<T>{}.strides;
    constexpr ck::index_t ndim   = a_lens.size();
    constexpr auto b_lens        = get_shape_c<U>{}.lens;
    constexpr auto b_strides     = get_shape_c<U>{}.strides;
    constexpr ck::index_t b_ndim = b_lens.size();
    constexpr auto c_lens        = get_shape_c<V>{}.lens;
    constexpr auto c_strides     = get_shape_c<V>{}.strides;
    constexpr ck::index_t c_ndim = c_lens.size();
    assert(b_ndim == ndim and c_ndim == ndim);

    using DeviceElementwiseAddInstance =
        ck::tensor_operation::device::DeviceElementwise<ck::Tuple<ABDataType, ABDataType>,
                                                        ck::Tuple<CDataType>,
                                                        Add,
                                                        ndim,
                                                        8,
                                                        ck::Sequence<1, 8>,
                                                        ck::Sequence<8>>;
    using shapes_t = std::array<ck::index_t, 3>;
    // shapes_t lengths_abc;
    // copy(c_lens.begin(), c_lens.end(), lengths_abc);
    shapes_t lengths_abc = {c_lens[0], c_lens[1], c_lens[2]};
    // constexpr auto lengths_abc = static_cast<shapes_t>(c_lens[0], c_lens[1], c_lens[2]);
    constexpr auto strides_a = static_cast<shapes_t>(a_strides);
    constexpr auto strides_b = static_cast<shapes_t>(b_strides);
    constexpr auto strides_c = static_cast<shapes_t>(c_strides);

    std::array<const void*, 2> input = {a_t.data(), b_t.data()};
    std::array<void*, 1> output      = {c_t.data()};

    auto ck_add   = DeviceElementwiseAddInstance{};
    auto argument = ck_add.MakeArgumentPointer(
        lengths_abc, {strides_a, strides_b}, {strides_c}, input, output, Add{});

    using InGrid1dDescTuple  = decltype(ck_add.GenerateInOutGrid1dDescTuple(ck::Number<ndim>{}));
    using OutGrid1dDescTuple = decltype(ck_add.GenerateInOutGrid1dDescTuple(ck::Number<ndim>{}));
    using InDataTypePointerTuple  = decltype(ck_add.GenerateInDataTypePointerTuple());
    using OutDataTypePointerTuple = decltype(ck_add.GenerateOutDataTypePointerTuple());
    using GridwiseElementwise     = ck::GridwiseElementwise_1D<InGrid1dDescTuple,
                                                           OutGrid1dDescTuple,
                                                           InDataTypePointerTuple,
                                                           OutDataTypePointerTuple,
                                                           ElementwiseOperation,
                                                           MPerThread,
                                                           InScalarPerVectorSeq,
                                                           OutScalarPerVectorSeq>;

    GridwiseElementwise::Run(argument.in_grid_1d_desc_tuple_,
                             argument.out_grid_1d_desc_tuple_,
                             argument.in_dev_buffers_,
                             argument.out_dev_buffers_,
                             argument.elementwise_op_);
}

} // namespace migraphx
#endif
