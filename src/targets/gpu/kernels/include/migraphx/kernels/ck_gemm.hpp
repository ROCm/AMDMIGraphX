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

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_dl_v1r3.hpp"
#include "ck/device_utility/device_prop.hpp"
#include "ck/device_utility/kernel_launch.hpp"


namespace migraphx {

// static constexpr auto I0 = Number<0>{};
// static constexpr auto I1 = Number<1>{};
// static constexpr auto I2 = Number<2>{};
// static constexpr auto I3 = Number<3>{};
// static constexpr auto I4 = Number<4>{};
// static constexpr auto I5 = Number<5>{};

// static constexpr auto K1Number = Number<1>{};

// static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
// {
//     assert(K % K1 == 0);

//     const index_t K0 = K / K1;

//     const auto a_grid_desc_m_k = [&]() {
//         if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
//         {
//             return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
//         }
//         else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
//         {
//             return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
//         }
//     }();

//     if constexpr(GemmSpec == GemmSpecialization::MNPadding)
//     {
//         const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

//         return transform_tensor_descriptor(
//             a_grid_desc_m_k,
//             make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
//                         make_right_pad_transform(M, PadM)),
//             make_tuple(Sequence<1>{}, Sequence<0>{}),
//             make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
//     }
//     else
//     {
//         return transform_tensor_descriptor(
//             a_grid_desc_m_k,
//             make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
//                         make_pass_through_transform(M)),
//             make_tuple(Sequence<1>{}, Sequence<0>{}),
//             make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
//     }
// }

// static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
// {
//     assert(K % K1 == 0);

//     const index_t K0 = K / K1;

//     const auto b_grid_desc_k_n = [&]() {
//         if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
//         {
//             return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
//         }
//         else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
//         {
//             return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
//         }
//     }();

//     if constexpr(GemmSpec == GemmSpecialization::MNPadding)
//     {
//         const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

//         return transform_tensor_descriptor(
//             b_grid_desc_k_n,
//             make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
//                         make_right_pad_transform(N, PadN)),
//             make_tuple(Sequence<0>{}, Sequence<1>{}),
//             make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
//     }
//     else
//     {
//         return transform_tensor_descriptor(
//             b_grid_desc_k_n,
//             make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
//                         make_pass_through_transform(N)),
//             make_tuple(Sequence<0>{}, Sequence<1>{}),
//             make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
//     }
// }

// static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
// {
//     const auto c_grid_desc_m_n = [&]() {
//         if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
//         {
//             return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
//         }
//         else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
//         {
//             return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
//         }
//     }();

//     if constexpr(GemmSpec == GemmSpecialization::MNPadding)
//     {
//         const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
//         const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

//         return transform_tensor_descriptor(
//             c_grid_desc_m_n,
//             make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
//             make_tuple(Sequence<0>{}, Sequence<1>{}),
//             make_tuple(Sequence<0>{}, Sequence<1>{}));
//     }
//     else
//     {

//         return transform_tensor_descriptor(
//             c_grid_desc_m_n,
//             make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
//             make_tuple(Sequence<0>{}, Sequence<1>{}),
//             make_tuple(Sequence<0>{}, Sequence<1>{}));
//     }
// }

template <class T, class U, class V>
__device__ void ck_gemm(const T& /* a_t */, const U& /* b_t */, const V& /* c_t */)
{
    constexpr auto alens = get_shape_c<T>{}.lens;
    constexpr auto m = alens[0];
    constexpr auto k = alens[1];
    constexpr auto alens = get_shape_c<U>{}.lens;
    constexpr auto n = alens[1];
    constexpr auto astrides = get_shape_c<T>{}.strides;
    constexpr auto as = astrides[1];
    constexpr auto bstrides = get_shape_c<U>{}.strides;
    constexpr auto bs = bstrides[1];
    constexpr auto cstrides = get_shape_c<V>{}.strides;
    constexpr auto cs = cstrides[1];
    printf("%i %i %i, %i %i %i\n", int(m), int(n), int(k), int(as), int(bs), int(cs));
}

} // namespace migraphx
#endif
