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
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

// MoE router on logits that keep the unit dims of the gemm: softmax -> squeeze
// -> topk -> values / sum(values), fused into one reduce kernel
template <migraphx::shape::type_t DType, unsigned int K, unsigned int N>
struct test_softmax_squeeze_topk_normalize
    : verify_program<test_softmax_squeeze_topk_normalize<DType, K, N>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {3, 1, N, 1, 1}};
        auto x       = mm->add_parameter("x", s);
        auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), x);
        auto squeeze =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 3, 4}}}), softmax);
        auto topk = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 1}, {"k", K}, {"largest", 1}}), squeeze);
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
        auto sum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), values);
        auto sumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", values->get_shape().lens()}}), sum);
        auto norm = mm->add_instruction(migraphx::make_op("div"), values, sumb);
        mm->add_return({norm, indices});
        return p;
    }
};

template struct test_softmax_squeeze_topk_normalize<migraphx::shape::float_type, 4, 128>;
template struct test_softmax_squeeze_topk_normalize<migraphx::shape::float_type, 1, 33>;
