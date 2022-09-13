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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_INCLUDES_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_INCLUDES_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp"
#include "ck/device_utility/device_prop.hpp"
#include "ck/device_utility/kernel_launch.hpp"

namespace migraphx {

static constexpr auto I0 = ck::Number<0>{};
static constexpr auto I1 = ck::Number<1>{};
static constexpr auto I2 = ck::Number<2>{};
static constexpr auto I3 = ck::Number<3>{};
static constexpr auto I4 = ck::Number<4>{};
static constexpr auto I5 = ck::Number<5>{};

static constexpr ck::index_t K1 = 1;
static constexpr auto K1Number  = ck::Number<K1>{};

using Row     = ck::tensor_layout::gemm::RowMajor;
using Col     = ck::tensor_layout::gemm::ColumnMajor;
using ALayout = Col;
using BLayout = Row;
using CLayout = Row;

using ADataType   = float;
using BDataType   = float;
using CDataType   = float;
using AccDataType = float;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// Values hard-coded by CK
static constexpr ck::index_t MPerBlock                         = 128;
static constexpr ck::index_t NPerBlock                         = 128;
static constexpr ck::index_t BlockSize                         = 256;
static constexpr ck::index_t K0PerBlock                        = 16;
static constexpr ck::index_t M1PerThread                       = 4;
static constexpr ck::index_t N1PerThread                       = 4;
static constexpr ck::index_t KPerThread                        = 1;
using M1N1ThreadClusterM1Xs                                    = S<8, 2>;
using M1N1ThreadClusterN1Xs                                    = S<8, 2>;
using ABlockTransferThreadSliceLengths_K0_M0_M1_K1             = S<2, 1, 4, 1>;
using ABlockTransferThreadClusterLengths_K0_M0_M1_K1           = S<8, 1, 32, 1>;
using ABlockTransferThreadClusterArrangeOrder                  = S<0, 3, 1, 2>;
using ABlockTransferSrcAccessOrder                             = S<0, 3, 1, 2>;
using ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1         = S<1, 1, 4, 1>;
using ABlockTransferSrcVectorTensorContiguousDimOrder          = S<0, 3, 1, 2>;
using ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1         = S<1, 1, 4, 1>;
using BBlockTransferThreadSliceLengths_K0_N0_N1_K1             = S<2, 1, 4, 1>;
using BBlockTransferThreadClusterLengths_K0_N0_N1_K1           = S<8, 1, 32, 1>;
using BBlockTransferThreadClusterArrangeOrder                  = S<0, 3, 1, 2>;
using BBlockTransferSrcAccessOrder                             = S<0, 3, 1, 2>;
using BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1         = S<1, 1, 4, 1>;
using BBlockTransferSrcVectorTensorContiguousDimOrder          = S<0, 3, 1, 2>;
using BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1         = S<1, 1, 4, 1>;
using CThreadTransferSrcDstAccessOrder                         = S<0, 1, 2, 3, 4, 5>;
static constexpr ck::index_t CThreadTransferSrcDstVectorDim    = 5;
static constexpr ck::index_t CThreadTransferDstScalarPerVector = 4;

static constexpr auto MakeAGridDescriptor_K0_M_K1(ck::index_t M, ck::index_t K, ck::index_t StrideA)
{
    assert(K % K1 == 0);

    const ck::index_t K0 = K / K1;

    const auto a_grid_desc_m_k = [&]() {
        if constexpr(is_same<ck::tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            return make_naive_tensor_descriptor(ck::make_tuple(M, K), ck::make_tuple(StrideA, I1));
        }
        else if constexpr(is_same<ck::tensor_layout::gemm::ColumnMajor, ALayout>::value)
        {
            return make_naive_tensor_descriptor(ck::make_tuple(M, K), ck::make_tuple(I1, StrideA));
        }
    }();

    if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNPadding)
    {
        const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

        return transform_tensor_descriptor(
            a_grid_desc_m_k,
            ck::make_tuple(ck::make_unmerge_transform(ck::make_tuple(K0, K1Number)),
                           ck::make_right_pad_transform(M, PadM)),
            ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
            ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));
    }
    else
    {
        return transform_tensor_descriptor(
            a_grid_desc_m_k,
            ck::make_tuple(ck::make_unmerge_transform(ck::make_tuple(K0, K1Number)),
                           ck::make_pass_through_transform(M)),
            ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
            ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));
    }
}

static constexpr auto MakeBGridDescriptor_K0_N_K1(ck::index_t K, ck::index_t N, ck::index_t StrideB)
{
    assert(K % K1 == 0);

    const ck::index_t K0 = K / K1;

    const auto b_grid_desc_k_n = [&]() {
        if constexpr(is_same<ck::tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            return make_naive_tensor_descriptor(ck::make_tuple(K, N), ck::make_tuple(StrideB, I1));
        }
        else if constexpr(is_same<ck::tensor_layout::gemm::ColumnMajor, BLayout>::value)
        {
            return make_naive_tensor_descriptor(ck::make_tuple(K, N), ck::make_tuple(I1, StrideB));
        }
    }();

    if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNPadding)
    {
        const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

        return transform_tensor_descriptor(
            b_grid_desc_k_n,
            ck::make_tuple(ck::make_unmerge_transform(ck::make_tuple(K0, K1Number)),
                           ck::make_right_pad_transform(N, PadN)),
            ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
            ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));
    }
    else
    {
        return transform_tensor_descriptor(
            b_grid_desc_k_n,
            ck::make_tuple(ck::make_unmerge_transform(ck::make_tuple(K0, K1Number)),
                           ck::make_pass_through_transform(N)),
            ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
            ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));
    }
}

static constexpr auto MakeCGridDescriptor_M_N(ck::index_t M, ck::index_t N, ck::index_t StrideC)
{
    const auto c_grid_desc_m_n = [&]() {
        if constexpr(is_same<ck::tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            return make_naive_tensor_descriptor(ck::make_tuple(M, N), ck::make_tuple(StrideC, I1));
        }
        else if constexpr(is_same<ck::tensor_layout::gemm::ColumnMajor, CLayout>::value)
        {
            return make_naive_tensor_descriptor(ck::make_tuple(M, N), ck::make_tuple(I1, StrideC));
        }
    }();

    if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNPadding)
    {
        const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
        const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

        return transform_tensor_descriptor(c_grid_desc_m_n,
                                           ck::make_tuple(ck::make_right_pad_transform(M, PadM),
                                                          ck::make_right_pad_transform(N, PadN)),
                                           ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                                           ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
    }
    else
    {

        return transform_tensor_descriptor(
            c_grid_desc_m_n,
            ck::make_tuple(ck::make_pass_through_transform(M), ck::make_pass_through_transform(N)),
            ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
            ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
    }
}

using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
using CGridDesc_M_N     = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

} // namespace migraphx
#endif
