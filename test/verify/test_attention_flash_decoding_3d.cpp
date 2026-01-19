/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <migraphx::shape::type_t DType>
struct test_attention_flash_decoding_3d : verify_program<test_attention_flash_decoding_3d<DType>>
{
    migraphx::program create_program() const
    {
        // Q = [B, M, k]
        // K = [B, k, N]
        // V = [B, N, D]
        // B = 1
        // M = 64
        // k = 16
        // N = 256
        // D = 32
        migraphx::shape q_shape{DType, {1, 64, 16}};
        migraphx::shape k_shape{DType, {1, 16, 256}};
        migraphx::shape v_shape{DType, {1, 256, 32}};

        migraphx::program p1;
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("q", q_shape);
        auto b   = mm->add_parameter("k", k_shape);
        auto b1  = mm->add_parameter("v", v_shape);

        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b); // {1, 64, 256}
        auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}),
                                        gemm1); // {1, 64, 1}
        rmax       = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 64, 256}}}), rmax);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax); // {1, 64, 256}
        auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);         // {1, 64, 256}
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}),
                                        exp); // {1, 64, 1}
        rsum =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 64, 256}}}),
                                rsum);                                         // {1, 64, 256}
        auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum); // {1, 64, 256}
        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);   // {1, 64, 32}
        mm->add_return({gemm2});
        return p1;
    }
};

// These tests are not run by default currently; the env vars below need to be set:
// MIGRAPHX_FLASH_DECODING_NUM_SPLITS=2 # or another split factor
// MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention
template struct test_attention_flash_decoding_3d<migraphx::shape::half_type>;
template struct test_attention_flash_decoding_3d<migraphx::shape::bf16_type>;
template struct test_attention_flash_decoding_3d<migraphx::shape::float_type>;
