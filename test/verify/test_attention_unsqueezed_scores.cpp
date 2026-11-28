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
#include <migraphx/instruction.hpp>

// Attention with a 3-D q.k^T gemm unsqueezed to 4-D (as in the dolly-v2-7b export), with
// enough q rows (32 heads * 256) to be fused
struct test_attention_unsqueezed_scores : verify_program<test_attention_unsqueezed_scores>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape qk_s{migraphx::shape::half_type, {32, 256, 64}};
        migraphx::shape v_s{migraphx::shape::half_type, {1, 32, 256, 64}};
        auto q = mm->add_parameter("q", qk_s);
        auto k = mm->add_parameter("k", qk_s);
        auto v = mm->add_parameter("v", v_s);
        auto kt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k);
        auto qk     = mm->add_instruction(migraphx::make_op("dot"), q, kt);
        auto scores = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), qk);
        auto scale  = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {0.125}});
        scale = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scores->get_shape().lens()}}), scale);
        scores       = mm->add_instruction(migraphx::make_op("mul"), scores, scale);
        auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), scores);
        mm->add_instruction(migraphx::make_op("dot"), softmax, v);
        return p;
    }
    std::string section() const { return "gemm"; }
};
