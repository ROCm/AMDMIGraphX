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

#include <iostream>
#include <cstdlib>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

namespace migraphx {

// using F16 = ck::half_t;
// using F32 = float;

// using ABDataType = F16;
// using CDataType  = F16;

// using Add = ck::tensor_operation::element_wise::Add;

// using DeviceElementwiseAddInstance =
//     ck::tensor_operation::device::DeviceElementwise<ck::Tuple<ABDataType, ABDataType>,
//                                                     ck::Tuple<CDataType>,
//                                                     Add,
//                                                     1,
//                                                     8,
//                                                     ck::Sequence<8, 8>,
//                                                     ck::Sequence<8>>;

__host__ __device__ void
ck_elementwise(void* /* a_p */, void* /* b_p */, void* /* c_p */)
{
    // ck::index_t M = 1024;
    // std::array<const void*, 2> input = {a_p,
    //                                     b_p};
    // std::array<void*, 1> output      = {c_p};

    // std::array<ck::index_t, 1> abc_lengths = {M};
    // std::array<ck::index_t, 1> a_strides   = {1};
    // std::array<ck::index_t, 1> b_strides   = {1};
    // std::array<ck::index_t, 1> c_strides   = {1};

    // auto broadcastAdd = DeviceElementwiseAddInstance{};
    // auto argument     = broadcastAdd.MakeArgumentPointer(
    //     abc_lengths, {a_strides, b_strides}, {c_strides}, input, output, Add{});

    // broadcastAdd_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, false});
}

} // namespace migraphx

#endif
