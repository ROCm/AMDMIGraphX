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

// topk -> convert of the indices, which rewrite_topk folds into the topk. A
// long N exercises the split topk as well.
template <migraphx::shape::type_t IndexType, unsigned int K, unsigned int N>
struct test_topk_convert_indices : verify_program<test_topk_convert_indices<IndexType, K, N>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, N}};
        auto x = mm->add_parameter("x", s);
        auto r = mm->add_instruction(
            migraphx::make_op("topk", {{"axis", 1}, {"k", K}, {"largest", 1}}), x);
        auto values   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto indices  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        auto cindices = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", IndexType}}), indices);
        mm->add_return({values, cindices});
        return p;
    }
};

template struct test_topk_convert_indices<migraphx::shape::int32_type, 3, 9>;
template struct test_topk_convert_indices<migraphx::shape::int32_type, 30, 2400>;
template struct test_topk_convert_indices<migraphx::shape::uint16_type, 8, 240>;
template struct test_topk_convert_indices<migraphx::shape::int32_type, 8, 80000>;
template struct test_topk_convert_indices<migraphx::shape::int64_type, 8, 24000>;
