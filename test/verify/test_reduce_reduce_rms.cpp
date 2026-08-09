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
#include <migraphx/shape.hpp>

// Sequential reductions over the same fastest axis with pointwise operations
// in between, which fuse into a single nested reduce kernel
template <migraphx::shape::type_t DType>
struct test_reduce_reduce_rms : verify_program<test_reduce_reduce_rms<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {1, 8, 16}};
        migraphx::shape bs{DType, {1, 8}};
        auto x    = mm->add_parameter("x", s);
        auto b    = mm->add_parameter("b", bs);
        auto eps  = mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {1e-6f}});
        auto n    = mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {16.0f}});
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto sq   = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), rsum);
        auto sqf  = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), sq);
        auto bf = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), b);
        auto add = mm->add_instruction(migraphx::make_op("add"), sqf, bf);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), add, add);
        auto nb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 8}}}), n);
        auto div   = mm->add_instruction(migraphx::make_op("div"), mul, nb);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-1}}}), div);
        auto epsb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 1}}}), eps);
        auto add2  = mm->add_instruction(migraphx::make_op("add"), rsum2, epsb);
        auto rsqrt = mm->add_instruction(migraphx::make_op("rsqrt"), add2);
        mm->add_return({rsqrt});
        return p;
    };

    std::string section() const { return "reduce"; }
};

template struct test_reduce_reduce_rms<migraphx::shape::float_type>;
template struct test_reduce_reduce_rms<migraphx::shape::bf16_type>;
template struct test_reduce_reduce_rms<migraphx::shape::half_type>;
