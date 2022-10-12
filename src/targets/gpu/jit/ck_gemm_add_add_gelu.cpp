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
#include <fstream>
#include <filesystem>
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
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const ck_gemm_aag_kernel = R"__migraphx__(
#include <migraphx/kernels/ck_fusion_inclusion.hpp>
#include <migraphx/kernels/ck_gemm_add_add_gelu.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

#include <hip/hip_runtime_api.h>

namespace migraphx {

using gemm_t = ${instance}, ${m}, ${k}, ${n}, ${sa}, ${sb}, ${sd0}, ${sd1}, ${se}>;

constexpr __device__ gemm_t ckdg{};
using GridwiseGemm = decltype(ckdg.gridwisegemm);

extern "C" {

__global__ void ck_gemm_aag_kernel(void* a_p, void* b_p, void* d0_p, void* d1_p, void* e_p)
{
    make_tensors()(a_p, b_p, d0_p, d1_p, e_p)([&](auto a_t, auto b_t, auto d0_t, auto d1_t, auto e_t) {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];
        make_tensors()(p_shared)([&](auto p_t){
            ck_gemm_add_add_gelu<gemm_t>(a_t, b_t, d0_t, d1_t, e_t, p_t);
        });
    });
}

}

} // namespace migraphx

)__migraphx__";

static std::size_t int_div_ceil(std::size_t x, std::size_t y)
{
    return (x + y - 1) / y;
}

static std::size_t get_grid_size(std::size_t m, std::size_t mpb, std::size_t n, std::size_t npb)
{
    return int_div_ceil(m, mpb) * int_div_ceil(n, npb);
}

struct block_settings
{
    int bs;
    int mpb;
    int npb;
};

struct ck_gemm_add_add_gelu_compiler : compiler<ck_gemm_add_add_gelu_compiler>
{
    const std::vector<std::string> instances{
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 256, 128, 32, 8, 2, 32, 32, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 256, 128, 32, 8, 8, 32, 32, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 256, 32, 8, 2, 32, 32, 2, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 128, 32, 8, 2, 32, 32, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 64, 32, 8, 2, 32, 32, 2, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16,16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 64, 32, 8, 8, 32, 32, 2, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 64, 128, 32, 8, 2, 32, 32, 1, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 64, 128, 32, 8, 8, 32, 32, 1, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 128, 128, 32, 8, 2, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 16, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 128, 128, 32, 8, 8, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 16, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 128, 64, 32, 8, 2, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 4>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 128, 64, 32, 8, 8, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 4>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 64, 128, 32, 8, 2, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 16, 1, 8>, 8",
        "CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 64, 128, 32, 8, 8, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 16, 1, 8>, 8"};

    const std::vector<block_settings> params {
        {256, 256, 128},
        {256, 256, 128},
        {256, 128, 256},
        {256, 128, 256},
        {256, 128, 128},
        {256, 128, 128},
        {256, 128, 64},
        {256, 128, 64},
        {256, 64, 128},
        {256, 64, 128},
        {128, 128, 128},
        {128, 128, 128},
        {128, 128, 64},
        {128, 128, 64},
        {128, 64, 128},
        {128, 64, 128}};
        
    std::vector<std::string> names() const { return {"ck_gemm_add_add_gelu", "gpu::ck_gemm_add_add_gelu"}; }

    operation compile_op(context& /* ctx */, const std::vector<shape>& inputs, const value& v) const
    {
        int i = 4;
        if (contains(v, "tuning_val"))
            i = v.at("tuning_val").to<int>();
        assert(i >= 0 and i < instances.size());

        hip_compile_options options;
        auto out_s = inputs.back();

        auto m = out_s.lens().front();
        auto n = out_s.lens().back();
        auto k = inputs.front().lens().back();
        std::string mnk = to_string(m) + " " + to_string(n) + " " + to_string(k);
        // auto itr = tuning_lookup.find(mnk);
        // std::cout << mnk << std::endl;
        // if (itr != tuning_lookup.end())
        // {
        //     i = tuning_lookup.at(mnk);
        //     std::cout << "  i: " << i << std::endl;
        // }
        // else    
        //     std::cout << "i: " << i << std::endl;

        auto b_s = params[i];
        auto block_size = b_s.bs;
        auto m_per_block = b_s.mpb;
        auto n_per_block = b_s.npb;
        
        auto grid_size = get_grid_size(m, m_per_block, n, n_per_block);

        printf("m, n, grid, global: %i, %i, %i, %i\n", int(m), int(n), int(grid_size), int(grid_size * block_size * 2));
        printf("out elm: %i\n", int(out_s.elements()));
        options.set_launch_params(v, grid_size * block_size, block_size);
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "ck_gemm_aag_kernel";
        options.virtual_inputs = inputs;

        auto sa = inputs.front().strides().front();
        auto sb = inputs.at(1).strides().front();
        //auto sc = inputs.back().strides().front();
        auto sd0 = inputs.at(2).strides().front();
        auto sd1 = inputs.at(3).strides().front();
        auto se = inputs.at(4).strides().front();
        printf("strides: %zu, %zu, %zu, %zu, %zu\n", sa, sb, sd0, sd1, se);
        auto src = interpolate_string(ck_gemm_aag_kernel, {{"instance", instances[i]},
                                                       {"m", to_string(m)},
                                                       {"k", to_string(k)},
                                                       {"n", to_string(n)},
                                                       {"sa", to_string(sa)},
                                                       {"sb", to_string(sb)},
                                                       {"sd0", to_string(sd0)},
                                                       {"sd1", to_string(sd1)},
                                                       {"se", to_string(se)}});
        
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
