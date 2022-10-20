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
#include <ck/tensor_operation/gpu/device/matrix_padder.hpp>
#include <ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp>

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
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
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
          ck::index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          ck::index_t BBlockLdsExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched = ck::make_default_loop_scheduler()>
struct CK_DeviceGemmMultipleD
{
    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    // static constexpr auto I2 = ck::Number<2>{};
    // static constexpr auto I3 = ck::Number<3>{};
    // static constexpr auto I4 = ck::Number<4>{};
    // static constexpr auto I5 = ck::Number<5>{};
    // static constexpr auto I6 = ck::Number<6>{};
    // static constexpr auto I7 = ck::Number<7>{};

    ck::tensor_operation::device::MatrixPadder<GemmSpec, ck::index_t, ck::index_t, ck::index_t>
        matrix_padder{MPerBlock, NPerBlock, KPerBlock};

    // GridwiseGemm
    using GridwiseGemm = ck::GridwiseGemmMultipleD_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        ck::InMemoryDataOperationEnum::Set,
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
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;

    // return block_id to E matrix tile idx (m0, n0) mapping
    template <class EGridDesc_M_N>
    __device__ static constexpr auto
    MakeDefaultBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n_)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, EGridDesc_M_N>(
            e_grid_desc_m_n_);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename AGridDesc_M_K,
              typename BGridDesc_N_K,
              typename DsGridDesc_M_N,
              typename EGridDesc_M_N,
              typename Block2ETileMap>
    static constexpr bool CheckValidity(const AGridDesc_M_K& a_grid_desc_m_k,
                                        const BGridDesc_N_K& b_grid_desc_n_k,
                                        const DsGridDesc_M_N& ds_grid_desc_m_n,
                                        const EGridDesc_M_N& e_grid_desc_m_n,
                                        const Block2ETileMap& block_2_etile_map)
    {
        constexpr auto M = a_grid_desc_m_k.GetLength(I0);
        constexpr auto N = b_grid_desc_n_k.GetLength(I0);
        constexpr auto K = a_grid_desc_m_k.GetLength(I1);

        // check consistency of desc
        static_assert(M == e_grid_desc_m_n.GetLength(I0) && N == e_grid_desc_m_n.GetLength(I1));

        // check tile size
        static_assert(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0);

        // check block-to-E-tile
        static_assert(block_2_etile_map.CheckValidity(e_grid_desc_m_n));

        return GridwiseGemm::CheckValidity(
            a_grid_desc_m_k, b_grid_desc_n_k, ds_grid_desc_m_n, e_grid_desc_m_n, block_2_etile_map);
    }

    AElementwiseOperation a_element_op{};
    BElementwiseOperation b_element_op{};
    CDEElementwiseOperation cde_element_op{};
};

} // namespace migraphx
#endif
