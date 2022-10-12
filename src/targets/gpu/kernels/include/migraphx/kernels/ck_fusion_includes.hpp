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
//#include <migraphx/env.hpp>
#include <cstdlib>

// #include "ck/utility/common_header.hpp"
// #include "ck/tensor_description/tensor_descriptor.hpp"
// #include "ck/tensor_description/tensor_descriptor_helper.hpp"
// #include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/device_gemm.hpp"
// #include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
// #include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp"

// #include "ck/ck.hpp"
// #include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
// #include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_xdl_cshuffle.hpp"
// #include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

// #include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

// #include "ck/utility/common_header.hpp"
// #include "ck/tensor_description/tensor_descriptor.hpp"
// #include "ck/tensor_description/tensor_descriptor_helper.hpp"
// #include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
// #include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
// #include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
// #include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace migraphx {

static constexpr auto I0 = ck::Number<0>{};
static constexpr auto I1 = ck::Number<1>{};
static constexpr auto I2 = ck::Number<2>{};
static constexpr auto I3 = ck::Number<3>{};
static constexpr auto I4 = ck::Number<4>{};
static constexpr auto I5 = ck::Number<5>{};

static constexpr ck::index_t K1 = 1;
static constexpr auto K1Number  = ck::Number<K1>{};

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Row_Row_Tuple = ck::Tuple<Row, Row>;

using F16 = ck::half_t;
using F32 = float;
using F16_F16_Tuple = ck::Tuple<F16, F16>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddAddFastGelu = ck::tensor_operation::element_wise::AddAddFastGelu;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

//////////////////////////
// Remove after testing //
using namespace ck;     //
//////////////////////////

// Rows of column-vectors
// This C-tile map dynamically adjusts M01 when C-tile index is out of range
template <index_t MPerBlock, index_t NPerBlock, typename CGridDesc_M_N>
struct BlockToCTileMap_M00_N0_M01Adapt
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt() = default;

    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt(const CGridDesc_M_N& c_grid_desc_m_n,
                                                        index_t M01 = 8)
        : M01_(M01), c_grid_desc_m_n_(c_grid_desc_m_n)
    {
    }

    __host__ __device__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const index_t grid_size = M0 * N0;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        auto block_1d_id = idx_top[I0];

        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I1), NPerBlock);

        block_1d_id = block_1d_id % (M0 * N0); // swallow batch index

        index_t idx_N0 = block_1d_id % N0;
        index_t idx_M0 = block_1d_id / N0;

        const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

        index_t idx_M00          = idx_M0 / M01_;
        index_t idx_M01          = idx_M0 % M01_;
        index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

        return make_tuple(idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                          idx_N0_M01_local / M01_adapt);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ constexpr bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                             const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    __host__ __device__ constexpr  bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const { return true; }

    private:
    index_t M01_;
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
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          ck::index_t MRaw,
          ck::index_t KRaw,
          ck::index_t NRaw,
          ck::index_t StrideA,
          ck::index_t StrideB,
          ck::index_t StrideD0,
          ck::index_t StrideD1,
          ck::index_t StrideE,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct CK_DeviceGemmMultipleD
{
    static constexpr std::array<index_t, 2> MRaws{};
    static constexpr std::array<index_t, 2> NRaws{};
    static constexpr std::array<index_t, 2> DsStride{StrideD0, StrideD1};
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto matrix_padder =
        ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};
    
    constexpr static auto MakeAGridDescriptor_M_K()
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    constexpr static auto MakeBGridDescriptor_N_K()
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

    template <typename ELay>
    constexpr static auto MakeEGridDescriptor_M_N()
    {
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    template <typename ELay>
    constexpr static auto MakeEGridDescriptor_M_N0()
    {
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideD0, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideD0));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    template <typename ELay>
    constexpr static auto MakeEGridDescriptor_M_N1()
    {
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideD1, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELay>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideD1));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    constexpr static auto MakeDsGridDescriptor_M_N()
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return MakeEGridDescriptor_M_N<DLayout>();
            },
            Number<NumDTensor>{});
    }

    // desc for problem definition
    using AGridDesc_M_K  = decltype(MakeAGridDescriptor_M_K());
    using BGridDesc_N_K  = decltype(MakeBGridDescriptor_N_K());
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N())>;
    using EGridDesc_M_N  = decltype(MakeEGridDescriptor_M_N<ELayout>());

    template<class T, class U/* , class V */>
    constexpr static auto Populate_D_Ptr(T& p_ds_grid_, U& p_ds_grid/* , V& ds_grid_desc_m_n_ */)
    {
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            //using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

            // D pointer
            p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

            // D desc
            // if constexpr(i == 0)
            // {
            //     ds_grid_desc_m_n_.At(i) =
            //         MakeEGridDescriptor_M_N0<DLayout>();
            // }
            // else 
            // {
            //     ds_grid_desc_m_n_.At(i) =
            //         MakeEGridDescriptor_M_N1<DLayout>();
            // }
        });
        // return make_tuple(MakeEGridDescriptor_M_N0<remove_cvref_t<tuple_element_t<0, DsLayout>>>(), 
        //                   MakeEGridDescriptor_M_N1<remove_cvref_t<tuple_element_t<1, DsLayout>>>());
    }

    constexpr static auto MakeDsDescTuple()
    {
        return make_tuple(MakeEGridDescriptor_M_N0<remove_cvref_t<tuple_element_t<0, DsLayout>>>(), 
                          MakeEGridDescriptor_M_N1<remove_cvref_t<tuple_element_t<1, DsLayout>>>());
    }

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleD_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
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

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1                          = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1                          = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(BGridDesc_N_K{}))>;
    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DsGridDesc_M_N{}))>;
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock  = remove_cvref_t<decltype(
        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(EGridDesc_M_N{}))>;

    // return block_id to E matrix tile idx (m0, n0) mapping
    __device__ static constexpr auto
    MakeDefaultBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n_)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, EGridDesc_M_N>(
            e_grid_desc_m_n_);
    }

    // block-to-e-tile map
    using Block2ETileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;

    static constexpr DsGridDesc_M_N ds_grid_desc_m_n{};
    static constexpr EGridDesc_M_N e_grid_desc_m_n = MakeEGridDescriptor_M_N<ELayout>();
    static constexpr Block2ETileMap block_2_etile_map = MakeDefaultBlock2ETileMap(e_grid_desc_m_n);
    static constexpr GridwiseGemm gridwisegemm{};
    static constexpr AElementwiseOperation a_element_op{};
    static constexpr BElementwiseOperation b_element_op{};
    static constexpr CDEElementwiseOperation cde_element_op{};
};


} // namespace migraphx
#endif
