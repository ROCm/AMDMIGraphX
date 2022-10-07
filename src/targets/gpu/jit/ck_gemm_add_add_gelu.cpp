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
#include <migraphx/kernels/gemm_aag_instance.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

#include <hip/hip_runtime_api.h>

namespace migraphx {

extern "C" {

__global__ void ck_gemm_kernel(void* a_p, void* b_p, void* d_p, void* e_p)
{
    make_tensors()(a_p, b_p, d_p, e_p)([](auto a_t, auto b_t, auto d_t, auto e_t) {
        ck_gemm_add_add_gelu(a_t, b_t, c_t, p_t);
    });
}

}

} // namespace migraphx

)__migraphx__";

// std::string kernel_p1 = R"__migraphx__(
// #include <migraphx/kernels/ck_gemm_includes.hpp>
// #include <migraphx/kernels/ck_gemm2.hpp>
// #include <migraphx/kernels/ops.hpp>
// #include <migraphx/kernels/integral_constant.hpp>
// #include <migraphx/kernels/generic_constant.hpp>
// #include <args.hpp>

// #include <hip/hip_runtime_api.h>

// namespace migraphx {
// using gemm = CKDeviceGemm)__migraphx__";

// std::string tuning_vals = R"__migraphx__(< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 128, 128, 128, 32, 8, 2, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 16, 1, 8>, 8>;)__migraphx__";

// std::string kernel_p2 = R"__migraphx__(

// extern "C" {

// __global__ void ck_gemm_kernel(void* a_p, void* b_p, void* c_p)
// {
//     constexpr gemm htp{};
//     using hGridwiseGemm = decltype(htp.gg);
//     make_tensors()(a_p, b_p, c_p)([&](auto a_t, auto b_t, auto c_t) {
//         constexpr ck::index_t shared_block_size =
//             hGridwiseGemm::GetSharedMemoryNumberOfByte();
//         __shared__ char p_shared_block[shared_block_size];
//         make_tensors()(p_shared_block)([&](auto p_t) {
//             ck_gemm(a_t, b_t, c_t, p_t, htp);
//         });
//     });
// }

// }

// } // namespace migraphx

// )__migraphx__";

// std::string kernel_string = kernel_p1 + tuning_vals + kernel_p2;

static std::string gemm_aag_instance = R"__migraphx__(
#ifndef MIGRAPHX_GUARD_GEMM_AAG_INSTANCE_HPP
#define MIGRAPHX_GUARD_GEMM_AAG_INSTANCE_HPP
#include <migraphx/kernels/ck_fusion_inclusion.hpp>

namespace migraphx {

    using ck_op = 
    CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 256, 128, 32, 8, 2, 32, 32, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 256, 128, 32, 8, 8, 32, 32, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 256, 32, 8, 2, 32, 32, 2, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 128, 32, 8, 2, 32, 32, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 64, 32, 8, 2, 32, 32, 2, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<16,16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 128, 64, 32, 8, 8, 32, 32, 2, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 8, 1, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 64, 128, 32, 8, 2, 32, 32, 1, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 256, 64, 128, 32, 8, 8, 32, 32, 1, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8>;


    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 128, 128, 32, 8, 2, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 16, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 128, 128, 32, 8, 8, 32, 32, 4, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 16, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 128, 64, 32, 8, 2, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<8, 16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 32, 1, 4>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 128, 64, 32, 8, 8, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 4>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 64, 128, 32, 8, 2, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 1, S<1, 16, 1, 8>, 8>;
    // CK_DeviceGemmMultipleD< Row, Row, Row_Row_Tuple, Row, F16, F16, F32, F32, F16_F16_Tuple, F16, PassThrough, PassThrough, AddAddFastGelu, GemmDefault, 1, 128, 64, 128, 32, 8, 8, 32, 32, 2, 2, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 8, 1, 1, 1, S<1, 16, 1, 8>, 8>;


} // namespace migraphx
#endif
)__migraphx__";

// static const char* const ck_gemm_kernel = kernel_string.c_str();

namespace fs = std::filesystem;

std::size_t int_divide_ceil(std::size_t x, std::size_t y)
{
    return (x + y - 1) / y;
}

struct block_settings
{
    int bs;
    int mpb;
    int npb;
};

struct ck_gemm_compiler : compiler<ck_gemm_compiler>
{
    std::vector<std::string> names() const { return {"ck_gemm"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // create GEMM instance header
        std::string path = fs::absolute(__FILE__);
        path = path.substr(0, path.find_last_of("\\/"));
        path = path.substr(0, path.find_last_of("\\/"));
        path += "/kernels/include/migraphx/kernels/gemm_aag_instance.hpp";
        std::ofstream out(path);
        out << gemm_aag_instance;
        out.close();

        //std::cout << ck_gemm_kernel << std::endl;
        hip_compile_options options;
        auto out_s = inputs.back();
         block_settings b_s{256, 256, 128};
        // block_settings b_s{256, 256, 128};
        // block_settings b_s{256, 128, 256};
        // block_settings b_s{256, 128, 256};
        // block_settings b_s{256, 128, 128};
        // block_settings b_s{256, 128, 128};
        // block_settings b_s{256, 128, 64};
        // block_settings b_s{256, 128, 64};
        // block_settings b_s{256, 64, 128};
        // block_settings b_s{256, 64, 128};

        // block_settings b_s{128, 128, 128};
        // block_settings b_s{128, 128, 128};
        // block_settings b_s{128, 128, 64};
        // block_settings b_s{128, 128, 64};
        // block_settings b_s{128, 64, 128};
        // block_settings b_s{128, 64, 128};
        auto block_size = b_s.bs;
        auto m_per_block = b_s.mpb;
        auto n_per_block = b_s.npb;
        auto m = out_s.lens().front();
        auto n = out_s.lens().back();
        auto grid_size = int_divide_ceil(m, m_per_block) * int_divide_ceil(n, n_per_block);
        printf("m, n, grid, global: %i, %i, %i, %i\n", int(m), int(n), int(grid_size), int(grid_size * block_size * 2));
        printf("out elm: %i\n", int(out_s.elements()));
        options.set_launch_params(v, compute_global_for(ctx, grid_size * block_size, 2), block_size);
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "ck_gemm_aag_kernel";
        options.virtual_inputs = inputs;

        return compile_hip_code_object(ck_gemm_aag_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
