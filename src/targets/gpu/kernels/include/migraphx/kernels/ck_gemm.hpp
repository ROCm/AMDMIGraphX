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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_GEMM_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_GEMM_HPP

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
static constexpr auto K1Number = ck::Number<K1>{};

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
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
static constexpr ck::index_t MPerBlock = 128; 
static constexpr ck::index_t NPerBlock = 128; 
static constexpr ck::index_t BlockSize = 256; 
static constexpr ck::index_t K0PerBlock = 16; 
static constexpr ck::index_t M1PerThread = 4; 
static constexpr ck::index_t N1PerThread = 4; 
static constexpr ck::index_t KPerThread = 1; 
using M1N1ThreadClusterM1Xs = S<8, 2>; 
using M1N1ThreadClusterN1Xs = S<8, 2>; 
using ABlockTransferThreadSliceLengths_K0_M0_M1_K1 = S<2, 1, 4, 1>;
using ABlockTransferThreadClusterLengths_K0_M0_M1_K1 = S<8, 1,  32, 1>;
using ABlockTransferThreadClusterArrangeOrder = S<0, 3, 1, 2>;
using ABlockTransferSrcAccessOrder = S<0, 3, 1, 2>;
using ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1 = S<1, 1, 4, 1>;
using ABlockTransferSrcVectorTensorContiguousDimOrder = S<0, 3, 1, 2>;
using ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1 = S<1, 1, 4, 1>;
using BBlockTransferThreadSliceLengths_K0_N0_N1_K1 = S<2, 1, 4, 1>;
using BBlockTransferThreadClusterLengths_K0_N0_N1_K1 = S<8, 1,  32, 1>;
using BBlockTransferThreadClusterArrangeOrder = S<0, 3, 1, 2>;
using BBlockTransferSrcAccessOrder = S<0, 3, 1, 2>;
using BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1 = S<1, 1, 4, 1>;
using BBlockTransferSrcVectorTensorContiguousDimOrder = S<0, 3, 1, 2>;
using BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1 = S<1, 1, 4, 1>;
using CThreadTransferSrcDstAccessOrder = S<0, 1, 2, 3, 4, 5>;
static constexpr ck::index_t CThreadTransferSrcDstVectorDim = 5;
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

        return transform_tensor_descriptor(
            c_grid_desc_m_n,
            ck::make_tuple(ck::make_right_pad_transform(M, PadM), ck::make_right_pad_transform(N, PadN)),
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

template <class T, class U, class V, class W>
__device__ void ck_gemm(const T& a_t, const U& b_t, const V& c_t, const W& p_t)
{
    constexpr auto alens    = get_shape_c<T>{}.lens;
    constexpr auto m        = alens[0];
    constexpr auto k        = alens[1];
    constexpr auto blens    = get_shape_c<U>{}.lens;
    constexpr auto n        = blens[1];
    constexpr auto astrides = get_shape_c<T>{}.strides;
    constexpr auto as       = astrides[0];
    constexpr auto bstrides = get_shape_c<U>{}.strides;
    constexpr auto bs       = bstrides[0];
    constexpr auto cstrides = get_shape_c<V>{}.strides;
    constexpr auto cs       = cstrides[0];
    auto idx = make_index();
    if (idx.global == 0)
        printf("%i %i %i, %i %i %i\n", int(m), int(n), int(k), int(as), int(bs), int(cs));
    
    auto a_grid_desc_k0_m_k1 = MakeAGridDescriptor_K0_M_K1(static_cast<ck::index_t>(m), static_cast<ck::index_t>(k), static_cast<ck::index_t>(as));
    auto b_grid_desc_k0_n_k1 = MakeBGridDescriptor_K0_N_K1(static_cast<ck::index_t>(k), static_cast<ck::index_t>(n), static_cast<ck::index_t>(bs));
    auto c_grid_desc_m_n =  MakeCGridDescriptor_M_N(static_cast<ck::index_t>(m), static_cast<ck::index_t>(n), static_cast<ck::index_t>(cs));
    using GridwiseGemm =
        ck::GridwiseGemmDl_km_kn_mn_v1r3<BlockSize,
                                        ADataType,
                                        AccDataType,
                                        CDataType,
                                        ck::InMemoryDataOperationEnum::Set,
                                        AGridDesc_K0_M_K1,
                                        BGridDesc_K0_N_K1,
                                        CGridDesc_M_N,
                                        MPerBlock,
                                        NPerBlock,
                                        K0PerBlock,
                                        M1PerThread,
                                        N1PerThread,
                                        KPerThread,
                                        M1N1ThreadClusterM1Xs,
                                        M1N1ThreadClusterN1Xs,
                                        ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                        ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                        ABlockTransferThreadClusterArrangeOrder,
                                        ABlockTransferSrcAccessOrder,
                                        ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                        ABlockTransferSrcVectorTensorContiguousDimOrder,
                                        ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                        BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
                                        BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
                                        BBlockTransferThreadClusterArrangeOrder,
                                        BBlockTransferSrcAccessOrder,
                                        BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
                                        BBlockTransferSrcVectorTensorContiguousDimOrder,
                                        BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
                                        CThreadTransferSrcDstAccessOrder,
                                        CThreadTransferSrcDstVectorDim,
                                        CThreadTransferDstScalarPerVector>;
    

    auto a_grid_desc_k0_m0_m1_k1 =
        GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(a_grid_desc_k0_m_k1);
    auto b_grid_desc_k0_n0_n1_k1 =
        GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(b_grid_desc_k0_n_k1);
    auto c_grid_desc_m0_m10_m11_n0_n10_n11 =
        GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(c_grid_desc_m_n);
    auto block_2_ctile_map = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n);


    constexpr bool HasMainKBlockLoop = true;
    constexpr bool HasDoubleTailKBlockLoop = true;
    GridwiseGemm::Run(a_t.data(), b_t.data(), c_t.data(), p_t.data(), a_grid_desc_k0_m0_m1_k1, b_grid_desc_k0_n0_n1_k1, c_grid_desc_m0_m10_m11_n0_n10_n11, block_2_ctile_map, ck::integral_constant<bool, HasMainKBlockLoop>{}, ck::integral_constant<bool, HasDoubleTailKBlockLoop>{});
}

} // namespace migraphx
#endif
