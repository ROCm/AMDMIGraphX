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
#ifndef MIGRAPHX_GUARD_KERNELS_CK_BATCHED_GEMM_HPP
#define MIGRAPHX_GUARD_KERNELS_CK_BATCHED_GEMM_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/ck.hpp>
#include <migraphx/kernels/ck_batched_gemm_includes.hpp>
#include <migraphx/kernels/shape.hpp>

namespace migraphx {

template <class T0, class T1, class T2, class T3>
struct ck_batched_gemm_settings
{
    T0 batch_count{};
    T1 batchStrideA{};
    T2 batchStrideB{};
    T3 batchStrideC{};
};

template <class... Ts>
constexpr ck_batched_gemm_settings<Ts...> make_ck_batched_gemm_settings(Ts... xs)
{
    return {xs...};
}

template <ck::index_t NumDTensor>
struct ComputePtrOffsetOfStridedBatch
{
    __device__ ComputePtrOffsetOfStridedBatch(ck::index_t BatchStrideA,
                                              ck::index_t BatchStrideB,
                                              std::array<ck::index_t, NumDTensor> BatchStrideDs,
                                              ck::index_t BatchStrideE)
        : BatchStrideA_(BatchStrideA),
          BatchStrideB_(BatchStrideB),
          BatchStrideDs_(BatchStrideDs),
          BatchStrideE_(BatchStrideE)
    {
    }

    __host__ __device__ constexpr ck::long_index_t GetAPtrOffset(ck::index_t g_idx) const
    {
        return g_idx * static_cast<ck::long_index_t>(BatchStrideA_);
    }

    __host__ __device__ constexpr ck::long_index_t GetBPtrOffset(ck::index_t g_idx) const
    {
        return g_idx * static_cast<ck::long_index_t>(BatchStrideB_);
    }

    __host__ __device__ constexpr auto GetDsPtrOffset(ck::index_t g_idx) const
    {
        std::array<ck::long_index_t, NumDTensor> ds_offset;
        ck::static_for<0, NumDTensor, 1>{}([&](auto i) {
            ds_offset[i] = g_idx * static_cast<ck::long_index_t>(BatchStrideDs_[i]);
        });
        return ds_offset;
    }

    __host__ __device__ constexpr ck::long_index_t GetEPtrOffset(ck::index_t g_idx) const
    {
        return g_idx * static_cast<ck::long_index_t>(BatchStrideE_);
    }

    private:
    ck::index_t BatchStrideA_;
    ck::index_t BatchStrideB_;
    std::array<ck::index_t, NumDTensor> BatchStrideDs_;
    ck::index_t BatchStrideE_;
};

template <class G, class Settings, class A, class B, class E, class... Ds>
__device__ void ck_batched_gemm(Settings s, A a, B b, E e, Ds... ds)
{
    constexpr const G gemm{};

    constexpr const auto a_grid_desc_m_k =
        gemm.matrix_padder.PadADescriptor_M_K(to_ck_batched_tensor<A>());
    constexpr const auto b_grid_desc_n_k =
        gemm.matrix_padder.PadBDescriptor_N_K(to_ck_batched_tensor<B>());
    constexpr const auto e_grid_desc_m_n =
        gemm.matrix_padder.PadCDescriptor_M_N(to_ck_batched_tensor<E>());
    constexpr const auto ds_grid_desc_m_n =
        ck::make_tuple(gemm.matrix_padder.PadCDescriptor_M_N(to_ck_batched_tensor<Ds>())...);
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

    constexpr const bool HasMainKBlockLoop =
        GridwiseGemm::CalculateHasMainKBlockLoop(a_grid_desc_ak0_m_ak1.GetLength(ck::Number<0>{}) *
                                                 a_grid_desc_ak0_m_ak1.GetLength(ck::Number<2>{}));

    static constexpr ck::index_t NumDTensor = gemm.NumDTensor;
    std::array<ck::index_t, NumDTensor> batchStrideDs;
    ck::static_for<0, NumDTensor, 1>{}([&](auto i) { batchStrideDs[i] = s.batchStrideC; });
    const ComputePtrOffsetOfStridedBatch<NumDTensor> compute_ptr_offset_of_batch{
        s.batchStrideA, s.batchStrideB, batchStrideDs, s.batchStrideC};

    auto batch_count = s.batch_count;
    const ck::index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(ck::get_grid_size() / batch_count);
    const ck::index_t g_idx =
        __builtin_amdgcn_readfirstlane(ck::get_block_1d_id() / num_blocks_per_batch);

    const ck::long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<ck::long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const ck::long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<ck::long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const ck::long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<ck::long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    auto p_ds_grid_grp = ck::make_tuple(ds.data()...);

    ck::static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid_grp[i] + ds_batch_offset[i]; });

    GridwiseGemm::template Run<HasMainKBlockLoop>(a.data() + a_batch_offset,
                                                  b.data() + b_batch_offset,
                                                  p_ds_grid_grp,
                                                  e.data() + e_batch_offset,
                                                  p_shared,
                                                  gemm.a_element_op,
                                                  gemm.b_element_op,
                                                  gemm.cde_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_etile_map);
}

} // namespace migraphx
#endif
