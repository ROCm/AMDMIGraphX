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
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

// x - max(x) -> topk over a transposed input, fused into one reduce kernel
// whose outputs keep the transposed layout
template <migraphx::shape::type_t DType, unsigned int K, unsigned int N>
struct test_reduce_max_topk_transposed
    : verify_program<test_reduce_max_topk_transposed<DType, K, N>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {N, 2}};
        auto x = mm->add_parameter("x", s);
        auto t = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);
        auto max  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), t);
        auto maxb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", t->get_shape().lens()}}), max);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), t, maxb);
        auto topk = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 1}, {"k", K}, {"largest", true}}), sub);
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
        // The outputs are made contiguous after the topk, so the contiguous the
        // target inserts before it is removed and the kernel reads the transposed input
        auto cvalues  = mm->add_instruction(migraphx::make_op("contiguous"), values);
        auto cindices = mm->add_instruction(migraphx::make_op("contiguous"), indices);
        mm->add_return({cvalues, cindices});
        return p;
    }
};

template struct test_reduce_max_topk_transposed<migraphx::shape::float_type, 3, 257>;
template struct test_reduce_max_topk_transposed<migraphx::shape::half_type, 8, 1024>;
