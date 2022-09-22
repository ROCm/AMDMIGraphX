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

#include <migraphx/kernels/ck_includes.hpp>

namespace migraphx {

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
    auto idx                = make_index();
    if(idx.global == 0)
        printf("%i %i %i, %i %i %i\n", int(m), int(n), int(k), int(as), int(bs), int(cs));

    auto a_grid_desc_k0_m_k1 = MakeAGridDescriptor_K0_M_K1(
        static_cast<ck::index_t>(m), static_cast<ck::index_t>(k), static_cast<ck::index_t>(as));
    auto b_grid_desc_k0_n_k1 = MakeBGridDescriptor_K0_N_K1(
        static_cast<ck::index_t>(k), static_cast<ck::index_t>(n), static_cast<ck::index_t>(bs));
    auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
        static_cast<ck::index_t>(m), static_cast<ck::index_t>(n), static_cast<ck::index_t>(cs));

    if(idx.global == 0)
    {
        printf("a_grid_desc_k0_m0_m1_k1{%i, %i, %i}\n", int(a_grid_desc_k0_m_k1.GetLength(I0)), int(a_grid_desc_k0_m_k1.GetLength(I1)), int(a_grid_desc_k0_m_k1.GetLength(I2)));
        printf("b_grid_desc_k0_n0_n1_k1{%i, %i, %i}\n", int(b_grid_desc_k0_n_k1.GetLength(I0)), int(b_grid_desc_k0_n_k1.GetLength(I1)), int(b_grid_desc_k0_n_k1.GetLength(I2)));
        printf("c_grid_desc_m_n{%i, %i}\n", int(c_grid_desc_m_n.GetLength(I0)), int(c_grid_desc_m_n.GetLength(I1)));
    }
    AGridDesc_K0_M0_M1_K1 a_grid_desc_k0_m0_m1_k1;
    BGridDesc_K0_N0_N1_K1 b_grid_desc_k0_n0_n1_k1;
    CGridDesc_M0_M10_M11_N0_N10_N11 c_grid_desc_m0_m10_m11_n0_n10_n11;
    DefaultBlock2CTileMap block_2_ctile_map;

    if(true or GridwiseGemm::CheckValidity(
                   a_grid_desc_k0_m_k1, b_grid_desc_k0_n_k1, c_grid_desc_m_n))
    {
        //printf("Is valid\n");
        a_grid_desc_k0_m0_m1_k1 =
            GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(a_grid_desc_k0_m_k1);
        b_grid_desc_k0_n0_n1_k1 =
            GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(b_grid_desc_k0_n_k1);
        c_grid_desc_m0_m10_m11_n0_n10_n11 =
            GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(c_grid_desc_m_n);
        block_2_ctile_map = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n);
    }
    else
    {
        //printf("Not valid\n");
    }

    if(idx.global == 0)
    {
        printf("a_grid_desc_k0_m0_m1_k1{%i, %i, %i}\n", int(a_grid_desc_k0_m0_m1_k1.GetLength(I0)), int(a_grid_desc_k0_m0_m1_k1.GetLength(I1)), int(a_grid_desc_k0_m0_m1_k1.GetLength(I2)));
        printf("b_grid_desc_k0_n0_n1_k1{%i, %i, %i}\n", int(b_grid_desc_k0_n0_n1_k1.GetLength(I0)), int(b_grid_desc_k0_n0_n1_k1.GetLength(I1)), int(b_grid_desc_k0_n0_n1_k1.GetLength(I2)));
        printf("c_grid_desc_m0_m10_m11_n0_n10_n11{%i, %i}\n", int(c_grid_desc_m0_m10_m11_n0_n10_n11.GetLength(I0)), int(c_grid_desc_m0_m10_m11_n0_n10_n11.GetLength(I1)));
    }

    const auto K0                    = a_grid_desc_k0_m0_m1_k1.GetLength(I0);
    const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K0);
    const bool has_double_tail_k_block_loop =
        GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K0);
    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        constexpr bool HasMainKBlockLoop = true;
        constexpr bool HasDoubleTailKBlockLoop = true;
        GridwiseGemm::Run(a_t.data(),
                        b_t.data(),
                        c_t.data(),
                        p_t.data(),
                        a_grid_desc_k0_m0_m1_k1,
                        b_grid_desc_k0_n0_n1_k1,
                        c_grid_desc_m0_m10_m11_n0_n10_n11,
                        block_2_ctile_map,
                        ck::integral_constant<bool, HasMainKBlockLoop>{},
                        ck::integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        constexpr bool HasMainKBlockLoop = true;
        constexpr bool HasDoubleTailKBlockLoop = false;
        GridwiseGemm::Run(a_t.data(),
                        b_t.data(),
                        c_t.data(),
                        p_t.data(),
                        a_grid_desc_k0_m0_m1_k1,
                        b_grid_desc_k0_n0_n1_k1,
                        c_grid_desc_m0_m10_m11_n0_n10_n11,
                        block_2_ctile_map,
                        ck::integral_constant<bool, HasMainKBlockLoop>{},
                        ck::integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        constexpr bool HasMainKBlockLoop = false;
        constexpr bool HasDoubleTailKBlockLoop = true;
        GridwiseGemm::Run(a_t.data(),
                        b_t.data(),
                        c_t.data(),
                        p_t.data(),
                        a_grid_desc_k0_m0_m1_k1,
                        b_grid_desc_k0_n0_n1_k1,
                        c_grid_desc_m0_m10_m11_n0_n10_n11,
                        block_2_ctile_map,
                        ck::integral_constant<bool, HasMainKBlockLoop>{},
                        ck::integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }
    else 
    {
        constexpr bool HasMainKBlockLoop = false;
        constexpr bool HasDoubleTailKBlockLoop = false;
        GridwiseGemm::Run(a_t.data(),
                        b_t.data(),
                        c_t.data(),
                        p_t.data(),
                        a_grid_desc_k0_m0_m1_k1,
                        b_grid_desc_k0_n0_n1_k1,
                        c_grid_desc_m0_m10_m11_n0_n10_n11,
                        block_2_ctile_map,
                        ck::integral_constant<bool, HasMainKBlockLoop>{},
                        ck::integral_constant<bool, HasDoubleTailKBlockLoop>{});
    }
}

} // namespace migraphx
#endif
