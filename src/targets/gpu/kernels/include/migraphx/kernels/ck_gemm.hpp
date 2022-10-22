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
#include <migraphx/kernels/gemm_batcher.hpp>

namespace migraphx {

// In CK, the B matrix is ordered as N,K instead of K,N
template <class Dims>
constexpr auto ck_transposeb_dims(Dims dims)
{
    return unpack(dims, [](auto k, auto n) { return make_const_array(n, k); });
}

template <class Tensor>
using ck_transposeb = decltype(make_shape(ck_transposeb_dims(get_shape_c<Tensor>{}.lens),
                                          ck_transposeb_dims(get_shape_c<Tensor>{}.strides)));

template <class G, class E, class A, class B, class... Ds>
__device__ void ck_gemm_matrix(E e, A a, B b, Ds... ds)
{
    constexpr const G gemm{};

    constexpr const auto a_grid_desc_m_k = gemm.matrix_padder.PadADescriptor_M_K(to_ck_tensor<A>());
    constexpr const auto b_grid_desc_n_k =
        gemm.matrix_padder.PadBDescriptor_N_K(to_ck_tensor<ck_transposeb<B>>());
    constexpr const auto e_grid_desc_m_n = gemm.matrix_padder.PadCDescriptor_M_N(to_ck_tensor<E>());
    constexpr const auto ds_grid_desc_m_n =
        ck::make_tuple(gemm.matrix_padder.PadCDescriptor_M_N(to_ck_tensor<Ds>())...);
    constexpr const auto block_2_etile_map = gemm.MakeDefaultBlock2ETileMap(e_grid_desc_m_n);

    using GridwiseGemm = typename G::GridwiseGemm;

    // tensor descriptors for block/thread-wise copy
    constexpr auto a_grid_desc_ak0_m_ak1 =
        GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k);
    constexpr auto b_grid_desc_bk0_n_bk1 =
        GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k);

    constexpr auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
        GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n);

    constexpr auto e_grid_desc_mblock_mperblock_nblock_nperblock =
        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(e_grid_desc_m_n);

    static_assert(GridwiseGemm::CheckValidity(
        a_grid_desc_m_k, b_grid_desc_n_k, ds_grid_desc_m_n, e_grid_desc_m_n, block_2_etile_map));

    __shared__ char p_shared_block[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    constexpr const bool HasMainKBlockLoop =
        GridwiseGemm::CalculateHasMainKBlockLoop(a_grid_desc_ak0_m_ak1.GetLength(ck::Number<0>{}) *
                                                 a_grid_desc_ak0_m_ak1.GetLength(ck::Number<2>{}));
    GridwiseGemm::template Run<HasMainKBlockLoop>(to_ck_const_pointer(a.data()),
                                                  to_ck_const_pointer(b.data()),
                                                  ck::make_tuple(to_ck_const_pointer(ds.data())...),
                                                  to_ck_pointer(e.data()),
                                                  p_shared_block,
                                                  gemm.a_element_op,
                                                  gemm.b_element_op,
                                                  gemm.cde_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_etile_map);
}

template <class G, index_int BlocksPerBatch, class... Ts>
__device__ void ck_gemm(Ts... xs)
{
    gemm_batch_args(make_index(), _c<BlocksPerBatch>, xs...)([](auto... ys) {
        ck_gemm_matrix<G>(ys...);
    });
}

} // namespace migraphx
#endif
