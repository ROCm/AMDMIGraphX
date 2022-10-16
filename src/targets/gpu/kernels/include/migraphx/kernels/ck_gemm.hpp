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
#include <migraphx/kernels/ck.hpp>
#include <migraphx/kernels/ck_gemm_includes.hpp>

namespace migraphx {

template <class G, class A, class B, class C>
__device__ void ck_gemm(const A& a, const B& b, const C& c)
{
    constexpr G gemm{};

    constexpr auto a_grid_desc_ak0_m_ak1 = gemm.MakeAGridDescriptor_AK0_M_AK1(to_ck_tensor<A>());
    constexpr auto b_grid_desc_bk0_n_bk1 = gemm.MakeBGridDescriptor_BK0_N_BK1(to_ck_tensor<B>());
    constexpr auto c_grid_desc_m_n       = gemm.MakeCGridDescriptor_M_N(to_ck_tensor<C>());
    constexpr auto block_2_ctile_map     = gemm.MakeDefaultBlock2CTileMap(c_grid_desc_m_n);

    using GridwiseGemm = typename G::template GridwiseGemm<decltype(a_grid_desc_ak0_m_ak1),
                                                           decltype(b_grid_desc_bk0_n_bk1),
                                                           decltype(c_grid_desc_m_n)>;
    // static_assert(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1, b_grid_desc_bk0_n_bk1,
    // c_grid_desc_m_n, block_2_ctile_map));

    constexpr auto c_grid_desc_mblock_mperblock_nblock_nperblock =
        GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(c_grid_desc_m_n);

    constexpr auto shared_block_size = GridwiseGemm::GetSharedMemoryNumberOfByte();
    __shared__ char p_shared_block[shared_block_size];

    constexpr bool HasMainKBlockLoop =
        GridwiseGemm::CalculateHasMainKBlockLoop(A{}.get_shape().elements());
    GridwiseGemm::template Run<HasMainKBlockLoop>(a.data(),
                                                  b.data(),
                                                  c.data(),
                                                  p_shared_block,
                                                  gemm.a_element_op,
                                                  gemm.b_element_op,
                                                  gemm.c_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_ctile_map);
}

} // namespace migraphx
#endif
