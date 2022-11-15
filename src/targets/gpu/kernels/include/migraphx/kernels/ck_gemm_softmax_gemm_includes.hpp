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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_GSG_INCLUDES_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_GSG_INCLUDES_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <ratio>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_xdl_cshuffle_v1.hpp"

#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"

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

// to track the points which need to be set to -inf on C0
// Note: no need to reset M padding value, because they will not be stored out.
struct C0MatrixMask
{
    __device__ C0MatrixMask(ck::index_t NRaw) : NRaw_(NRaw) {}

    __device__ bool IsUpperTriangle(ck::index_t m, ck::index_t n) const { return n > m; }

    __device__ bool IsNOutOfBound(/*ck::index_t m, */ ck::index_t n) const
    {
        return n >= NRaw_;
    }

    __device__ bool IsMaskedElement(ck::index_t m, ck::index_t n) const
    {
        return IsUpperTriangle(m, n) || IsNOutOfBound(n);
    }

    private:
    // ck::index_t MRaw_;
    ck::index_t NRaw_;
};

template <typename ALayout,
          typename BLayout, // B0Layout
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename B1DataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock, // Gemm0NPerBlock
          ck::index_t KPerBlock, // Gemm0KPerBlock
          ck::index_t Gemm1NPerBlock,
          ck::index_t Gemm1KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t B1K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          ck::index_t Gemm1NXdlPerWave,
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
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          ck::index_t B1BlockTransferSrcVectorDim,
          ck::index_t B1BlockTransferSrcScalarPerVector,
          ck::index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1BlockLdsExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          bool MaskOutUpperTriangle,
          typename Alpha,
          ck::LoopScheduler LoopSched = ck::LoopScheduler::Default>
struct CK_DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle
{
    static constexpr auto matrix_padder =
        ck::tensor_operation::device::GemmGemmPadder<GemmSpec, ck::index_t, ck::index_t, ck::index_t, ck::index_t>{
            MPerBlock, NPerBlock, KPerBlock, Gemm1NPerBlock};

    static constexpr auto get_AK1() { return AK1; };
    static constexpr auto get_BK1() { return BK1; };
    static constexpr auto get_B1K1() { return B1K1; };
    static constexpr auto get_mperblock() { return MPerBlock; };
    static constexpr auto get_gemm1nperblock() { return Gemm1NPerBlock; };

    static constexpr float alpha = float(Alpha::num) / Alpha::den;
    static constexpr auto get_alpha() { return alpha; };
    AElementwiseOperation a_element_op{};
    BElementwiseOperation b_element_op{};
    B1ElementwiseOperation b1_element_op{};
    CElementwiseOperation c_element_op{};
    AccElementwiseOperation acc_element_op{alpha};

    template<typename AGridDesc_AK0_M_AK1,
        typename BGridDesc_BK0_N_BK1,
        typename B1GridDesc_BK0_N_BK1,
        typename CGridDesc_M_N>
    struct rt_gridwisegemm 
    {
        // GridwiseGemm
        using GridwiseGemm = ck::GridwiseBatchedGemmSoftmaxGemm_Xdl_CShuffle<
            ADataType, // TODO: distinguish A/B datatype
            GemmAccDataType,
            CShuffleDataType,
            CDataType,
            AElementwiseOperation,
            BElementwiseOperation,
            AccElementwiseOperation,
            B1ElementwiseOperation,
            CElementwiseOperation,
            ck::InMemoryDataOperationEnum::Set,
            AGridDesc_AK0_M_AK1,
            BGridDesc_BK0_N_BK1,
            B1GridDesc_BK0_N_BK1,
            CGridDesc_M_N,
            NumGemmKPrefetchStage,
            BlockSize,
            MPerBlock,
            NPerBlock,
            KPerBlock,
            Gemm1NPerBlock,
            Gemm1KPerBlock,
            AK1,
            BK1,
            B1K1,
            MPerXDL,
            NPerXDL,
            MXdlPerWave,
            NXdlPerWave,
            Gemm1NXdlPerWave,
            ABlockTransferThreadClusterLengths_AK0_M_AK1,
            ABlockTransferThreadClusterArrangeOrder,
            ABlockTransferSrcAccessOrder,
            ABlockTransferSrcVectorDim,
            ABlockTransferSrcScalarPerVector,
            ABlockTransferDstScalarPerVector_AK1,
            true,
            ABlockLdsExtraM,
            BBlockTransferThreadClusterLengths_BK0_N_BK1,
            BBlockTransferThreadClusterArrangeOrder,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorDim,
            BBlockTransferSrcScalarPerVector,
            BBlockTransferDstScalarPerVector_BK1,
            true,
            BBlockLdsExtraN,
            B1BlockTransferThreadClusterLengths_BK0_N_BK1,
            B1BlockTransferThreadClusterArrangeOrder,
            B1BlockTransferSrcAccessOrder,
            B1BlockTransferSrcVectorDim,
            B1BlockTransferSrcScalarPerVector,
            B1BlockTransferDstScalarPerVector_BK1,
            false,
            B1BlockLdsExtraN,
            CShuffleMXdlPerWavePerShuffle,
            CShuffleNXdlPerWavePerShuffle,
            CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
            CShuffleBlockTransferScalarPerVector_NPerBlock,
            LoopSched,
            matrix_padder.PadN,
            MaskOutUpperTriangle>;
    };
};


} // namespace migraphx
#endif
