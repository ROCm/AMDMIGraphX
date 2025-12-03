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

// Test pad followed by reduce_sum - tests index transformation with reduction fusion
template <migraphx::shape::type_t DType>
struct test_gen_pad_reduce_sum : verify_program<test_gen_pad_reduce_sum<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {4, 8, 16}};
        // Pad format: [before_d0, before_d1, before_d2, after_d0, after_d1, after_d2]
        std::vector<int64_t> pads = {0, 1, 0, 0, 1, 0};
        auto x      = mm->add_parameter("x", s);
        auto padded = mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), x);
        // Padded shape: {4, 10, 16}, reduce along last axis
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), padded);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_pad_reduce_sum<migraphx::shape::float_type>;
template struct test_gen_pad_reduce_sum<migraphx::shape::half_type>;

// Test pad followed by reduce_max
template <migraphx::shape::type_t DType>
struct test_gen_pad_reduce_max : verify_program<test_gen_pad_reduce_max<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {2, 4, 8}};
        std::vector<int64_t> pads = {1, 0, 0, 1, 0, 0};
        auto x      = mm->add_parameter("x", s);
        auto padded = mm->add_instruction(
            migraphx::make_op("pad", {{"pads", pads}, {"value", float{-1e10}}}), x);
        // Padded shape: {4, 4, 8}, reduce along axis 1
        mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), padded);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_pad_reduce_max<migraphx::shape::float_type>;
template struct test_gen_pad_reduce_max<migraphx::shape::half_type>;

// Test pad + pointwise + reduce fusion
template <migraphx::shape::type_t DType>
struct test_gen_pad_add_reduce : verify_program<test_gen_pad_add_reduce<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {4, 8, 16}};
        std::vector<int64_t> pads = {0, 0, 1, 0, 0, 1};
        auto x      = mm->add_parameter("x", s);
        auto padded = mm->add_instruction(migraphx::make_op("pad", {{"pads", pads}}), x);
        // Padded shape: {4, 8, 18}
        migraphx::shape s_padded{DType, {4, 8, 18}};
        auto y   = mm->add_parameter("y", s_padded);
        auto add = mm->add_instruction(migraphx::make_op("add"), padded, y);
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), add);
        return p;
    }
    std::string section() const { return "reduce"; }
};

template struct test_gen_pad_add_reduce<migraphx::shape::float_type>;
template struct test_gen_pad_add_reduce<migraphx::shape::half_type>;


