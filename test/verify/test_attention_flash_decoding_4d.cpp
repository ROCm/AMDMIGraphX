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
struct test_attention_flash_decoding_4d : verify_program<test_attention_flash_decoding_4d<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s1{DType, {1, 12, 256, 256}};

        auto a  = mm->add_parameter("1", s1);
        auto b  = mm->add_parameter("2", s1);
        auto b1 = mm->add_parameter("3", s1);
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
        rmax = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}),
                                   rmax);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
        auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        rsum = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}),
                                   rsum);
        auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
        mm->add_return({gemm2});

        return p;
    }
};

// These tests are not run by default currently; the env vars below need to be set:
// MIGRAPHX_FLASH_DECODING_NUM_SPLITS=2 # or another split factor
// MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention
template struct test_attention_flash_decoding_4d<migraphx::shape::half_type>;
template struct test_attention_flash_decoding_4d<migraphx::shape::bf16_type>;
template struct test_attention_flash_decoding_4d<migraphx::shape::float_type>;
