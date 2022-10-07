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
static const char* const ck_gemm_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/ck_gemm2.hpp>

#include <hip/hip_runtime_api.h>

namespace migraphx {

using gemm_t = ${instance}, ${m}, ${k}, ${n}, ${sa}, ${sb}, ${sc}>;

constexpr __device__ gemm_t ckdg{};
using GridwiseGemm = decltype(ckdg.gridwisegemm);

extern "C" {

__global__ void ck_gemm_kernel(void* a_p, void* b_p, void* c_p)
{
    make_tensors()(a_p, b_p, c_p)([&](auto a_t, auto b_t, auto c_t) {
        constexpr ck::index_t shared_block_size =
            GridwiseGemm::GetSharedMemoryNumberOfByte();
        __shared__ char p_shared_block[shared_block_size];
        make_tensors()(p_shared_block)([&](auto p_t) {
            ck_gemm<gemm_t>(a_t, b_t, c_t, p_t);
        });
    });
}

}

} // namespace migraphx

)__migraphx__";


std::size_t int_div_ceil(std::size_t x, std::size_t y)
{
    return (x + y - 1) / y;
}

std::size_t get_grid_size(std::size_t m, std::size_t mpb, std::size_t n, std::size_t npb)
{
    return int_div_ceil(m, mpb) * int_div_ceil(n, npb);
}

struct block_settings
{
    int bs;
    int mpb;
    int npb;
};

namespace fs = std::filesystem;

struct ck_gemm_compiler : compiler<ck_gemm_compiler>
{
    const std::vector<std::string> instances{
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 256, 128, 32, 8, 2, 32, 32, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 256, 128, 32, 8, 8, 32, 32, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 128, 256, 32, 8, 2, 32, 32, 2, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 128, 128, 32, 8, 2, 32, 32, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 128, 64, 32, 8, 2, 32, 32, 2, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16,16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 128, 64, 32, 8, 8, 32, 32, 2, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 64, 128, 32, 8, 2, 32, 32, 1, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 64, 128, 32, 8, 8, 32, 32, 1, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 128, 128, 128, 32, 8, 2, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 16, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 128, 128, 128, 32, 8, 8, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 16, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 128, 128, 64, 32, 8, 2, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 4>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 128, 128, 64, 32, 8, 8, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 4>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 128, 64, 128, 32, 8, 2, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 16, 1, 8>, 8",
        "       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 128, 64, 128, 32, 8, 8, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 16, 1, 8>, 8"};

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

    std::vector<std::string> names() const { return {"ck_gemm"}; }

    operation compile_op(context& /* ctx */, const std::vector<shape>& inputs, const value& v) const
    {
        int i = 4;
        if (contains(v, "tuning_val"))
            i = v.at("tuning_val").to<int>();
        assert(i >= 0 and i < instances.size());

        hip_compile_options options;
        auto out_s = inputs.back();

        auto b_s = params[i];
        auto block_size = b_s.bs;
        auto m_per_block = b_s.mpb;
        auto n_per_block = b_s.npb;
        auto m = out_s.lens().front();
        auto n = out_s.lens().back();
        auto grid_size = get_grid_size(m, m_per_block, n, n_per_block);

        options.set_launch_params(v, grid_size * block_size, block_size);
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "ck_gemm_kernel";
        options.virtual_inputs = inputs;

        auto k = inputs.front().lens().back();
        auto sa = inputs.front().strides().front();
        auto sb = inputs.at(1).strides().front();
        auto sc = inputs.back().strides().front();
        auto src = interpolate_string(ck_gemm_kernel, {{"instance", instances[i]},
                                                       {"m", to_string(m)},
                                                       {"k", to_string(k)},
                                                       {"n", to_string(n)},
                                                       {"sa", to_string(sa)},
                                                       {"sb", to_string(sb)},
                                                       {"sc", to_string(sc)}});
        
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
