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

#include <ck/utility/common_header.hpp>
#include <ck/tensor_description/tensor_descriptor.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/device_gemm.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp>

namespace migraphx {

template <ck::index_t MPerBlock, ck::index_t NPerBlock, typename CGridDesc_M_N>
struct BlockToCTileMap_M00_N0_M01Adapt
{
    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};
    static constexpr auto I3 = ck::Number<3>{};

    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt() = default;

    __host__
        __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt(const CGridDesc_M_N& c_grid_desc_m_n,
                                                             ck::index_t M01 = 8)
        : M01_(M01), c_grid_desc_m_n_(c_grid_desc_m_n)
    {
    }

    __host__ __device__ constexpr ck::index_t
    CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = ck::math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = ck::math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const ck::index_t grid_size = M0 * N0;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        auto block_1d_id = idx_top[I0];

        const auto M0 = ck::math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I0), MPerBlock);
        const auto N0 = ck::math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I1), NPerBlock);

        block_1d_id = block_1d_id % (M0 * N0); // swallow batch index

        ck::index_t idx_N0 = block_1d_id % N0;
        ck::index_t idx_M0 = block_1d_id / N0;

        const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

        ck::index_t idx_M00          = idx_M0 / M01_;
        ck::index_t idx_M01          = idx_M0 % M01_;
        ck::index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

        return ck::make_tuple(idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                              idx_N0_M01_local / M01_adapt);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool constexpr ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                                       const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    __host__ __device__ constexpr bool
    CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
    {
        return true;
    }

    private:
    ck::index_t M01_;
    CGridDesc_M_N c_grid_desc_m_n_;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched = ck::make_default_loop_scheduler()
          >
struct CKDeviceGemm 
{
    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};
    static constexpr auto I3 = ck::Number<3>{};

    template<class Descriptor>
    static constexpr auto
    MakeAGridDescriptor_AK0_M_AK1(const Descriptor& a_grid_desc_mraw_kraw)
    {
        const auto MRaw = a_grid_desc_mraw_kraw.GetLength(I0);
        const auto KRaw = a_grid_desc_mraw_kraw.GetLength(I1);
        const auto M = ck::math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto K = ck::math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::MKPadding ||
                     GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                ck::make_tuple(ck::make_right_pad_transform(MRaw, MPad),
                               ck::make_right_pad_transform(KRaw, KPad)),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                ck::make_tuple(make_unmerge_transform(ck::make_tuple(AK0, AK1)),
                               ck::make_pass_through_transform(M)),
                ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
                ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::MPadding ||
                          GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                ck::make_tuple(make_unmerge_transform(ck::make_tuple(AK0, AK1)),
                               ck::make_right_pad_transform(MRaw, MPad)),
                ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
                ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::KPadding ||
                          GemmSpec == ck::tensor_operation::device::GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                ck::make_tuple(ck::make_pass_through_transform(MRaw),
                               ck::make_right_pad_transform(KRaw, KPad)),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                ck::make_tuple(make_unmerge_transform(ck::make_tuple(AK0, AK1)),
                               ck::make_pass_through_transform(MRaw)),
                ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
                ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                ck::make_tuple(make_unmerge_transform(ck::make_tuple(AK0, AK1)),
                               ck::make_pass_through_transform(MRaw)),
                ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
                ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
    }

    template<class Descriptor>
    static constexpr auto
    MakeBGridDescriptor_BK0_N_BK1(const Descriptor& b_grid_desc_nraw_kraw)
    {
        const auto NRaw = b_grid_desc_nraw_kraw.GetLength(I0);
        const auto KRaw = b_grid_desc_nraw_kraw.GetLength(I1);
        const auto N = ck::math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;
        const auto K = ck::math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::NKPadding ||
                     GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                ck::make_tuple(ck::make_right_pad_transform(NRaw, NPad),
                               ck::make_right_pad_transform(KRaw, KPad)),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                ck::make_tuple(make_unmerge_transform(ck::make_tuple(BK0, BK1)),
                               ck::make_pass_through_transform(N)),
                ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
                ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::NPadding ||
                          GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                ck::make_tuple(make_unmerge_transform(ck::make_tuple(BK0, BK1)),
                               ck::make_right_pad_transform(NRaw, NPad)),
                ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
                ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::KPadding ||
                          GemmSpec == ck::tensor_operation::device::GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                ck::make_tuple(ck::make_pass_through_transform(NRaw),
                               ck::make_right_pad_transform(KRaw, KPad)),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                ck::make_tuple(make_unmerge_transform(ck::make_tuple(BK0, BK1)),
                               ck::make_pass_through_transform(NRaw)),
                ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
                ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                ck::make_tuple(make_unmerge_transform(ck::make_tuple(BK0, BK1)),
                               ck::make_pass_through_transform(NRaw)),
                ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
                ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
    }

    template<class Descriptor>
    static constexpr auto
    MakeCGridDescriptor_M_N(const Descriptor& c_grid_desc_mraw_nraw)
    {
        const auto MRaw = c_grid_desc_mraw_nraw.GetLength(I0);
        const auto NRaw = c_grid_desc_mraw_nraw.GetLength(I1);
        const auto M = ck::math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto N = ck::math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == ck::tensor_operation::device::GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                ck::make_tuple(ck::make_right_pad_transform(MRaw, MPad),
                               ck::make_right_pad_transform(NRaw, NPad)),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
        }
        else if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::MPadding ||
                          GemmSpec == ck::tensor_operation::device::GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                ck::make_tuple(ck::make_right_pad_transform(MRaw, MPad),
                               ck::make_pass_through_transform(NRaw)),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
        }
        else if constexpr(GemmSpec == ck::tensor_operation::device::GemmSpecialization::NPadding ||
                          GemmSpec == ck::tensor_operation::device::GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                ck::make_tuple(ck::make_pass_through_transform(MRaw),
                               ck::make_right_pad_transform(NRaw, NPad)),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
    }

    // using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1<8, 8, 8>());
    // using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1<8, 8, 8>());
    // using CGridDesc_M_N       = decltype(MakeCGridDescriptor_M_N<8, 8, 8>());
    // using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1());
    // using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1());
    // using CGridDesc_M_N       = decltype(MakeCGridDescriptor_M_N());

        // return block_id to C matrix tile idx (m0, n0) mapping
    template<class CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    template<class AGridDesc_AK0_M_AK1, class BGridDesc_BK0_N_BK1, class CGridDesc_M_N> 
    using GridwiseGemm = ck::GridwiseGemm_k0mk1_k0nk1_mn_xdl_cshuffle_v1<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        ck::InMemoryDataOperationEnum::Set,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        CGridDesc_M_N,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;
    
    AElementwiseOperation a_element_op{};
    BElementwiseOperation b_element_op{};
    CElementwiseOperation c_element_op{};
};

} // namespace migraphx
#endif
