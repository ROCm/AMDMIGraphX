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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
// static const char* const ck_gemm_kernel = R"__migraphx__(
// #include <migraphx/kernels/ck_gemm.hpp>
// #include <migraphx/kernels/ops.hpp>
// #include <migraphx/kernels/integral_constant.hpp>
// #include <migraphx/kernels/generic_constant.hpp>
// #include <args.hpp>

// #include <hip/hip_runtime_api.h>

// namespace migraphx {

// extern "C" {

// __global__ void ck_gemm_kernel(void* a_p, void* b_p, void* c_p)
// {
//     // hipDeviceProp_t hdp{};
//     // printf("Shared mem: %i\n", int(hdp.sharedMemPerBlock));
//     // make_tensors()(a_p, b_p, c_p)([](auto&&... xs) {
//     //     ck_gemm(xs...);
//     // });
//     make_tensors()(a_p, b_p, c_p)([](auto a_t, auto b_t, auto c_t) {
//         __shared__ float p_shared_block[512]; //[(a_t.get_shape().elements() +
//         b_t.get_shape().elements()) * 2]; ck_gemm(a_t, b_t, c_t, p_shared_block);
//         // make_tensors()(p_shared_block)([&](auto p_t) {
//         //     ck_gemm(a_t, b_t, c_t, p_t);
//         // });
//     });
// }

// }

// } // namespace migraphx

// )__migraphx__";

static const char* const ck_gemm_kernel = R"__migraphx__(
#include <migraphx/kernels/ck_includes.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

#include <hip/hip_runtime_api.h>

namespace migraphx {

extern "C" {

__global__ void ck_gemm_kernel(void* a_p, void* b_p, void* c_p) 
{
    make_tensors()(a_p, b_p, c_p)([](auto a_t, auto b_t, auto c_t) { 
        constexpr auto alens    = get_shape_c<decltype(a_t)>{}.lens;
        constexpr auto m        = alens[0];
        constexpr auto k        = alens[1];
        constexpr auto blens    = get_shape_c<decltype(b_t)>{}.lens;
        constexpr auto n        = blens[1];
        constexpr auto astrides = get_shape_c<decltype(a_t)>{}.strides;
        constexpr auto as       = astrides[0];
        constexpr auto bstrides = get_shape_c<decltype(b_t)>{}.strides;
        constexpr auto bs       = bstrides[0];
        constexpr auto cstrides = get_shape_c<decltype(c_t)>{}.strides;
        constexpr auto cs       = cstrides[0];

        auto a_grid_desc_k0_m_k1 = MakeAGridDescriptor_K0_M_K1(
            static_cast<ck::index_t>(m), static_cast<ck::index_t>(k), static_cast<ck::index_t>(as));
        auto b_grid_desc_k0_n_k1 = MakeBGridDescriptor_K0_N_K1(
            static_cast<ck::index_t>(k), static_cast<ck::index_t>(n), static_cast<ck::index_t>(bs));
        auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
            static_cast<ck::index_t>(m), static_cast<ck::index_t>(n), static_cast<ck::index_t>(cs));
        using GridwiseGemm =
            ck::GridwiseGemmDl_km_kn_mn_v1r3<BlockSize,
                                            ADataType,
                                            AccDataType,
                                            CDataType,
                                            ck::InMemoryDataOperationEnum::Set,
                                            AGridDesc_K0_M_K1,
                                            BGridDesc_K0_N_K1,
                                            CGridDesc_M_N,
                                            MPerBlock,
                                            NPerBlock,
                                            K0PerBlock,
                                            M1PerThread,
                                            N1PerThread,
                                            KPerThread,
                                            M1N1ThreadClusterM1Xs,
                                            M1N1ThreadClusterN1Xs,
                                            ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
                                            ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
                                            ABlockTransferThreadClusterArrangeOrder,
                                            ABlockTransferSrcAccessOrder,
                                            ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
                                            ABlockTransferSrcVectorTensorContiguousDimOrder,
                                            ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
                                            BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
                                            BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
                                            BBlockTransferThreadClusterArrangeOrder,
                                            BBlockTransferSrcAccessOrder,
                                            BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
                                            BBlockTransferSrcVectorTensorContiguousDimOrder,
                                            BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
                                            CThreadTransferSrcDstAccessOrder,
                                            CThreadTransferSrcDstVectorDim,
                                            CThreadTransferDstScalarPerVector>;

        auto a_grid_desc_k0_m0_m1_k1 =
            GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(a_grid_desc_k0_m_k1);
        auto b_grid_desc_k0_n0_n1_k1 =
            GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(b_grid_desc_k0_n_k1);
        auto c_grid_desc_m0_m10_m11_n0_n10_n11 =
            GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(c_grid_desc_m_n);
        auto block_2_ctile_map = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n);

        constexpr bool HasMainKBlockLoop       = true;
        constexpr bool HasDoubleTailKBlockLoop = true;
        constexpr ck::index_t shared_block_size =
            GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(float);
        __shared__ float p_shared_block[shared_block_size];
        GridwiseGemm::Run(a_t.data(),
                        b_t.data(),
                        c_t.data(),
                        p_shared_block,
                        a_grid_desc_k0_m0_m1_k1,
                        b_grid_desc_k0_n0_n1_k1,
                        c_grid_desc_m0_m10_m11_n0_n10_n11,
                        block_2_ctile_map,
                        ck::integral_constant<bool, HasMainKBlockLoop>{},
                        ck::integral_constant<bool, HasDoubleTailKBlockLoop>{});
        
        // using AGridDesc_K0_M0_M1_K1 =
        //     decltype(GridwiseGemm::MakeAGridDescriptor_K0_M0_M1_K1(AGridDesc_K0_M_K1{}));
        // using BGridDesc_K0_N0_N1_K1 =
        //     decltype(GridwiseGemm::MakeBGridDescriptor_K0_N0_N1_K1(BGridDesc_K0_N_K1{}));
        // using CGridDesc_M0_M10_M11_N0_N10_N11 =
        //     decltype(GridwiseGemm::MakeCGridDescriptor_M0_M10_M11_N0_N10_N11(CGridDesc_M_N{}));
        // using DefaultBlock2CTileMap =
        //     decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(CGridDesc_M_N{}));
        
        // const auto kernel = ck::kernel_gemm_dl_v1r3<GridwiseGemm,
        //                                 ADataType,
        //                                 CDataType,
        //                                 remove_reference_t<AGridDesc_K0_M0_M1_K1>,
        //                                 remove_reference_t<BGridDesc_K0_N0_N1_K1>,
        //                                 remove_reference_t<CGridDesc_M0_M10_M11_N0_N10_N11>,
        //                                 remove_reference_t<DefaultBlock2CTileMap>,
        //                                 true,
        //                                 true>;
        // kernel(a_t.data(),
        //     b_t.data(),
        //     c_t.data(),
        //     a_grid_desc_k0_m0_m1_k1,
        //     b_grid_desc_k0_n0_n1_k1,
        //     c_grid_desc_m0_m10_m11_n0_n10_n11,
        //     block_2_ctile_map);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct ck_gemm_compiler : compiler<ck_gemm_compiler>
{
    std::vector<std::string> names() const { return {"ck_gemm"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        auto out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "ck_gemm_kernel";
        options.virtual_inputs = inputs;

        return compile_hip_code_object(ck_gemm_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
