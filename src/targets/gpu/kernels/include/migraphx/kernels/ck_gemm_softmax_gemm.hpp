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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_GEMM_SOFTMAX_GEMM_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_GEMM_SOFTMAX_GEMM_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/ck.hpp>
#include <migraphx/kernels/ck_gemm_softmax_gemm_includes.hpp>
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

template <class G, class C, class A, class B, class B1>
__device__ void ck_gemm_softmax_gemm_matrix(C c, A a, B b, B1 b1)
{
    constexpr const G gemm{};

    constexpr const auto a_shape = get_shape_c<A>{};
    constexpr const auto m       = a_shape.lens[0];
    constexpr const auto k       = a_shape.lens[1];
    constexpr const auto sa      = a_shape.strides[0];
    constexpr const auto a_tensor =
        ck::make_naive_tensor_descriptor(ck::make_tuple(m, k), ck::make_tuple(sa, 1));
    constexpr const auto a_grid_desc_mraw_kraw = gemm.matrix_padder.PadADescriptor_M_K(a_tensor);

    constexpr const auto AK1 = gemm.get_AK1();
    constexpr const auto AK0 = k / AK1;

    constexpr const auto a_grid_desc_ak0_m_ak1 = ck::transform_tensor_descriptor(
        a_grid_desc_mraw_kraw,
        ck::make_tuple(ck::make_unmerge_transform(ck::make_tuple(AK0, AK1)),
                       ck::make_pass_through_transform(m)),
        ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
        ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

    constexpr const auto b_shape = get_shape_c<B>{};
    constexpr const auto n       = b_shape.lens[1];
    constexpr const auto sb      = b_shape.strides[1]; // col-major
    constexpr const auto BK1     = gemm.get_BK1();
    constexpr const auto BK0     = k / BK1;

    constexpr const auto b_tensor =
        ck::make_naive_tensor_descriptor(ck::make_tuple(n, k), ck::make_tuple(sb, 1));
    constexpr const auto b_grid_desc_nraw_kraw = gemm.matrix_padder.PadBDescriptor_N_K(b_tensor);
    constexpr const auto b_grid_desc_bk0_n_bk1 = ck::transform_tensor_descriptor(
        b_grid_desc_nraw_kraw,
        ck::make_tuple(ck::make_unmerge_transform(ck::make_tuple(BK0, BK1)),
                       ck::make_pass_through_transform(n)),
        ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
        ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

    constexpr const auto b1_shape = get_shape_c<B1>{};
    constexpr const auto k1       = b1_shape.lens[0];
    constexpr const auto n1       = b1_shape.lens[1];
    constexpr const auto sb1      = b1_shape.strides[0]; // row-major
    constexpr const auto B1K1     = gemm.get_B1K1();
    constexpr const auto B1K0     = k1 / B1K1;

    constexpr const auto b1_tensor =
        ck::make_naive_tensor_descriptor(ck::make_tuple(n1, k1), ck::make_tuple(1, sb1));
    constexpr const auto b1_grid_desc_nraw_kraw = gemm.matrix_padder.PadB1Descriptor_N_K(b1_tensor);
    constexpr const auto b1_grid_desc_bk0_n_bk1 = ck::transform_tensor_descriptor(
        b1_grid_desc_nraw_kraw,
        ck::make_tuple(ck::make_unmerge_transform(ck::make_tuple(B1K0, B1K1)),
                       ck::make_pass_through_transform(n1)),
        ck::make_tuple(ck::Sequence<1>{}, ck::Sequence<0>{}),
        ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));

    constexpr const auto c_shape = get_shape_c<C>{};
    constexpr const auto sc      = c_shape.strides[0];
    constexpr const auto c_tensor =
        ck::make_naive_tensor_descriptor(ck::make_tuple(m, n1), ck::make_tuple(sc, 1));
    constexpr const auto c_grid_desc_m_n = gemm.matrix_padder.PadCDescriptor_M_N(c_tensor);

    constexpr const auto MPerBlock      = gemm.get_mperblock();
    constexpr const auto Gemm1NPerBlock = gemm.get_gemm1nperblock();
    constexpr const auto MBlock         = m / MPerBlock;
    constexpr const auto NBlock         = n1 / Gemm1NPerBlock;
    constexpr const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
        ck::transform_tensor_descriptor(
            c_grid_desc_m_n,
            ck::make_tuple(
                ck::make_unmerge_transform(ck::make_tuple(MBlock, ck::Number<MPerBlock>{})),
                ck::make_unmerge_transform(ck::make_tuple(NBlock, ck::Number<Gemm1NPerBlock>{}))),
            ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
            ck::make_tuple(ck::Sequence<0, 1>{}, ck::Sequence<2, 3>{}));

    constexpr const auto block_2_ctile_map =
        BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, Gemm1NPerBlock, decltype(c_grid_desc_m_n)>(
            c_grid_desc_m_n);

    const C0MatrixMask c0_matrix_mask(n);

    const auto K = a_grid_desc_ak0_m_ak1.GetLength(ck::Number<0>{}) *
                   a_grid_desc_ak0_m_ak1.GetLength(ck::Number<2>{});

    using gridwise = typename G::template rt_gridwisegemm<decltype(a_grid_desc_ak0_m_ak1),
                                                          decltype(b_grid_desc_bk0_n_bk1),
                                                          decltype(b1_grid_desc_bk0_n_bk1),
                                                          decltype(c_grid_desc_m_n)>;

    using GridwiseGemm                     = typename gridwise::GridwiseGemm;
    constexpr const bool HasMainKBlockLoop = GridwiseGemm::CalculateHasMainKBlockLoop(K);

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    // static_assert(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1,
    //                                           b_grid_desc_bk0_n_bk1,
    //                                           b1_grid_desc_bk0_n_bk1,
    //                                           c_grid_desc_m_n,
    //                                           block_2_ctile_map));
    GridwiseGemm::template Run<HasMainKBlockLoop>(to_ck_const_pointer(a.data()),
                                                  to_ck_const_pointer(b.data()),
                                                  to_ck_const_pointer(b1.data()),
                                                  to_ck_pointer(c.data()),
                                                  p_shared,
                                                  gemm.a_element_op,
                                                  gemm.b_element_op,
                                                  gemm.acc_element_op,
                                                  gemm.b1_element_op,
                                                  gemm.c_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  b1_grid_desc_bk0_n_bk1,
                                                  c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_ctile_map,
                                                  c0_matrix_mask);
}

template <class G, index_int BlocksPerBatch, class... Ts>
__device__ void ck_gemm_softmax_gemm(Ts... xs)
{
    gemm_batch_args(make_index(), _c<BlocksPerBatch>, xs...)(
        [](auto... ys) { ck_gemm_softmax_gemm_matrix<G>(ys...); });
}

} // namespace migraphx
#endif
