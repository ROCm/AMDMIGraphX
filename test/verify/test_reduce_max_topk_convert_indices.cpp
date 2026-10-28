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

// x - max(x) -> topk -> convert of the indices: the convert is folded into the
// topk, which is then fused into the reduce kernel with that index type
template <migraphx::shape::type_t IndexType, unsigned int K, unsigned int N>
struct test_reduce_max_topk_convert_indices
    : verify_program<test_reduce_max_topk_convert_indices<IndexType, K, N>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, N}};
        auto x   = mm->add_parameter("x", s);
        auto max = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), x);
        auto maxb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), max);
        auto sub  = mm->add_instruction(migraphx::make_op("sub"), x, maxb);
        auto topk = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 1}, {"k", K}, {"largest", true}}), sub);
        auto values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), topk);
        auto cindices = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", IndexType}}), indices);
        mm->add_return({values, cindices});
        return p;
    }
};

template struct test_reduce_max_topk_convert_indices<migraphx::shape::int32_type, 3, 257>;
template struct test_reduce_max_topk_convert_indices<migraphx::shape::int32_type, 8, 2048>;
template struct test_reduce_max_topk_convert_indices<migraphx::shape::uint16_type, 4, 1024>;
