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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_GEMM_ADD_ADD_GELU_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_GEMM_ADD_ADD_GELU_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
//#include <migraphx/kernels/gemm_instance.hpp>

#include <migraphx/kernels/ck_fusion_inclusion.hpp>

namespace migraphx {

template <class A, class B, class C, class D, class E, class F, class G,
            class H, class I, class J, class K, class L, class M>
__device__ void fake_op(A, B, C, D, E, F, G, H, I, J, K, L, M)
{

}

// template <class G, class T, class U, class V, class W, class X>
// __device__ void ck_gemm_add_add_gelu(const T& , const U&, const V& , const V& , const W& , X& )
// {
    
    
// }

template <class G, class T, class U, class V, class W, class X>
__device__ void ck_gemm_add_add_gelu(const T& a_t, const U& b_t, const V& d0_t, const V& d1_t, const W& e_t, X& p_t)
{
    constexpr static G ckdg{};
    using GridwiseGemm      = decltype(ckdg.gridwisegemm);
    // tensor descriptors for problem definiton
    constexpr auto a_grid_desc_m_k = ckdg.MakeAGridDescriptor_M_K();
    constexpr auto b_grid_desc_n_k = ckdg.MakeBGridDescriptor_N_K();
    //constexpr auto ds_grid_desc_m_n = ckdg.ds_grid_desc_m_n;
    constexpr auto e_grid_desc_m_n = ckdg.e_grid_desc_m_n;

    // tensor descriptors for block/thread-wise copy
    constexpr auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k);
    constexpr auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k);

    // block-to-e-tile map
    constexpr auto block_2_etile_map = ckdg.block_2_etile_map;

    // element-wise op
    constexpr auto a_element_op = ckdg.a_element_op;
    constexpr auto b_element_op = ckdg.b_element_op;
    constexpr auto cde_element_op = ckdg.cde_element_op;

    constexpr std::size_t NumDTensor = 2;

    std::array<const void*, NumDTensor> p_ds_grid{d0_t.data(), d1_t.data()};

    typename GridwiseGemm::DsGridPointer p_ds_grid_{};
    ckdg.Populate_D_Ptr(p_ds_grid_, p_ds_grid/* , ds_grid_desc_m_n */);
    constexpr auto ds_grid_desc_m_n = ckdg.MakeDsDescTuple();

    // populate desc for Ds/E
    static_assert(GridwiseGemm::CheckValidity(a_grid_desc_m_k,
                                    b_grid_desc_n_k,
                                    ds_grid_desc_m_n,
                                    e_grid_desc_m_n,
                                    block_2_etile_map));
    constexpr auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
        GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            ds_grid_desc_m_n);

    constexpr auto e_grid_desc_mblock_mperblock_nblock_nperblock =
        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            e_grid_desc_m_n);

    constexpr auto K = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);
    if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
    {
        constexpr bool HasMainKBlockLoop = true;
        GridwiseGemm::template Run<HasMainKBlockLoop>(a_t.data(),
                                                  b_t.data(),
                                                  p_ds_grid_,
                                                  e_t.data(),
                                                  p_t.data(),
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_etile_map);
    }
    else
    {
        constexpr bool HasMainKBlockLoop = false;
        GridwiseGemm::template Run<HasMainKBlockLoop>(a_t.data(),
                                                  b_t.data(),
                                                  p_ds_grid_,
                                                  e_t.data(),
                                                  p_t.data(),
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_etile_map);
    }
    // constexpr auto K = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);
    // if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
    // {
    //     fake_op(a_t.data(),
    //                                               b_t.data(),
    //                                               p_ds_grid_,
    //                                               e_t.data(),
    //                                               p_t.data(),
    //                                               a_element_op,
    //                                               b_element_op,
    //                                               cde_element_op,
    //                                               a_grid_desc_ak0_m_ak1,
    //                                               b_grid_desc_bk0_n_bk1,
    //                                               ds_grid_desc_mblock_mperblock_nblock_nperblock,
    //                                               e_grid_desc_mblock_mperblock_nblock_nperblock,
    //                                               block_2_etile_map);
    // }
    // {
    //     fake_op(a_t.data(),
    //                                               b_t.data(),
    //                                               p_ds_grid_,
    //                                               e_t.data(),
    //                                               p_t.data(),
    //                                               a_element_op,
    //                                               b_element_op,
    //                                               cde_element_op,
    //                                               a_grid_desc_ak0_m_ak1,
    //                                               b_grid_desc_bk0_n_bk1,
    //                                               ds_grid_desc_mblock_mperblock_nblock_nperblock,
    //                                               e_grid_desc_mblock_mperblock_nblock_nperblock,
    //                                               block_2_etile_map);
    // }
    
}

} // namespace migraphx
#endif
