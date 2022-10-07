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

template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J, class K>
__device__ void fake_op(A, B, C, D, E, F, G, H, I, J, K)
{

}

template <class T, class U, class V, class W>
__device__ void ck_gemm_add_add_gelu(const T& a_t, const U& b_t, const V& d_t, const W& e_t)
{
    // constexpr static gemm ckdg{};
    // using GridwiseGemm      = decltype(ckdg.gridwisegemm);
    // constexpr auto alens    = get_shape_c<T>{}.lens;
    // constexpr auto m        = alens[0];
    // constexpr auto k        = alens[1];
    // constexpr auto blens    = get_shape_c<U>{}.lens;
    // constexpr auto n        = blens[1];
    // constexpr auto astrides = get_shape_c<T>{}.strides;
    // constexpr auto as       = astrides[0];
    // constexpr auto bstrides = get_shape_c<U>{}.strides;
    // constexpr auto bs       = bstrides[0];
    // constexpr auto cstrides = get_shape_c<V>{}.strides;
    // constexpr auto cs       = cstrides[0];

    // constexpr auto a_grid_desc_ak0_m_ak1 = ckdg.MakeAGridDescriptor_AK0_M_AK1<
    //     static_cast<ck::index_t>(m), static_cast<ck::index_t>(k), static_cast<ck::index_t>(as)>();
    // constexpr auto b_grid_desc_bk0_n_bk1 = ckdg.MakeBGridDescriptor_BK0_N_BK1<
    //     static_cast<ck::index_t>(k), static_cast<ck::index_t>(n), static_cast<ck::index_t>(bs)>();
    // constexpr auto c_grid_desc_m_n = ckdg.MakeCGridDescriptor_M_N<
    //     static_cast<ck::index_t>(m), static_cast<ck::index_t>(n), static_cast<ck::index_t>(cs)>();
    // constexpr auto block_2_ctile_map = ckdg.MakeDefaultBlock2CTileMap(c_grid_desc_m_n);

    // static_assert(GridwiseGemm::CheckValidity(
    //         a_grid_desc_ak0_m_ak1, b_grid_desc_bk0_n_bk1, c_grid_desc_m_n, block_2_ctile_map));
    
    // constexpr auto c_grid_desc_mblock_mperblock_nblock_nperblock =
    //     GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);

    // constexpr auto K      = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);
    // constexpr auto a_element_op = ckdg.a_element_op;
    // constexpr auto b_element_op = ckdg.b_element_op;
    // constexpr auto c_element_op = ckdg.c_element_op;

    // if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
    // {
    //     constexpr bool HasMainKBlockLoop = true;
    //     GridwiseGemm::template Run<HasMainKBlockLoop>(a_t.data(),
    //                                                   b_t.data(),
    //                                                   c_t.data(),
    //                                                   p_t.data(),
    //                                                   a_element_op,
    //                                                   b_element_op,
    //                                                   c_element_op,
    //                                                   a_grid_desc_ak0_m_ak1,
    //                                                   b_grid_desc_bk0_n_bk1,
    //                                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
    //                                                   block_2_ctile_map);
    // }
    // else
    // {
    //     constexpr bool HasMainKBlockLoop = false;
    //     GridwiseGemm::template Run<HasMainKBlockLoop>(a_t.data(),
    //                                                   b_t.data(),
    //                                                   c_t.data(),
    //                                                   p_t.data(),
    //                                                   a_element_op,
    //                                                   b_element_op,
    //                                                   c_element_op,
    //                                                   a_grid_desc_ak0_m_ak1,
    //                                                   b_grid_desc_bk0_n_bk1,
    //                                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
    //                                                   block_2_ctile_map);
    // }
}

} // namespace migraphx
#endif
