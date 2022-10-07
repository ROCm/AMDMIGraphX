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
#include <migraphx/kernels/ck_gemm_includes.hpp>

namespace migraphx {

template <class G, class T, class U, class V, class W>
__device__ void ck_gemm(const T& a_t, const U& b_t, const V& c_t, W& p_t)
{
    constexpr G ckdg{};
    using GridwiseGemm = decltype(ckdg.gridwisegemm);

    constexpr auto a_grid_desc_ak0_m_ak1 = ckdg.MakeAGridDescriptor_AK0_M_AK1();
    constexpr auto b_grid_desc_bk0_n_bk1 = ckdg.MakeBGridDescriptor_BK0_N_BK1();
    constexpr auto c_grid_desc_m_n       = ckdg.MakeCGridDescriptor_M_N();
    constexpr auto block_2_ctile_map     = ckdg.MakeDefaultBlock2CTileMap(c_grid_desc_m_n);

    // static_assert(GridwiseGemm::CheckValidity(
    //         a_grid_desc_ak0_m_ak1, b_grid_desc_bk0_n_bk1, c_grid_desc_m_n, block_2_ctile_map));

    constexpr auto c_grid_desc_mblock_mperblock_nblock_nperblock =
        GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);

    constexpr auto K = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);
    constexpr auto a_element_op = ckdg.a_element_op;
    constexpr auto b_element_op = ckdg.b_element_op;
    constexpr auto c_element_op = ckdg.c_element_op;

    if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
    {
        constexpr bool HasMainKBlockLoop = true;
        GridwiseGemm::template Run<HasMainKBlockLoop>(a_t.data(),
                                                      b_t.data(),
                                                      c_t.data(),
                                                      p_t.data(),
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op,
                                                      a_grid_desc_ak0_m_ak1,
                                                      b_grid_desc_bk0_n_bk1,
                                                      c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                      block_2_ctile_map);
    }
    else
    {
        constexpr bool HasMainKBlockLoop = false;
        GridwiseGemm::template Run<HasMainKBlockLoop>(a_t.data(),
                                                      b_t.data(),
                                                      c_t.data(),
                                                      p_t.data(),
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op,
                                                      a_grid_desc_ak0_m_ak1,
                                                      b_grid_desc_bk0_n_bk1,
                                                      c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                      block_2_ctile_map);
    }
}

} // namespace migraphx
#endif
