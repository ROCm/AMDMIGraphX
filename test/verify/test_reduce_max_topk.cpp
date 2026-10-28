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

// x - max(x) -> topk, fused into one reduce kernel
template <migraphx::shape::type_t DType, unsigned int K, unsigned int N, bool Largest>
struct test_reduce_max_topk : verify_program<test_reduce_max_topk<DType, K, N, Largest>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {2, N}};
        auto x   = mm->add_parameter("x", s);
        auto max = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), x);
        auto maxb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), max);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), x, maxb);
        auto topk = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 1}, {"k", K}, {"largest", Largest}}), sub);
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
        mm->add_return({values, indices});
        return p;
    }
};

template struct test_reduce_max_topk<migraphx::shape::float_type, 3, 257, true>;
template struct test_reduce_max_topk<migraphx::shape::float_type, 512, 2048, true>;
template struct test_reduce_max_topk<migraphx::shape::half_type, 2, 33, false>;
template struct test_reduce_max_topk<migraphx::shape::half_type, 3, 9, true>;
