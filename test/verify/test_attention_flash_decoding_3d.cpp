/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
        // 3D Shape: [batch, sequence_length, head_dim]
        migraphx::shape s_3d{DType, {1, 256, 256}};

        migraphx::program p1;
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("q", s_3d);
        auto b   = mm->add_parameter("k", s_3d);
        auto b1  = mm->add_parameter("v", s_3d);
        b  = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), gemm1);
        rmax       = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), rmax);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
        auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), exp);
        rsum      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), rsum);
        auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
        mm->add_return({gemm2});
        return p1;
    }
};

// TODO: accuracy issue with fp16
// template struct test_attention_flash_decoding_3d<migraphx::shape::half_type>;
template struct test_attention_flash_decoding_3d<migraphx::shape::bf16_type>;

