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

// Test reverse followed by pointwise add - verifies reverse index transformation fusion
template <migraphx::shape::type_t DType>
struct test_gen_reverse_add : verify_program<test_gen_reverse_add<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {4, 8, 16}};
        auto x                    = mm->add_parameter("x", s);
        std::vector<int64_t> axes = {1};
        auto reversed = mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), x);
        auto y        = mm->add_parameter("y", s);
        mm->add_instruction(migraphx::make_op("add"), reversed, y);
        return p;
    }
};

template struct test_gen_reverse_add<migraphx::shape::float_type>;
template struct test_gen_reverse_add<migraphx::shape::half_type>;

// Test reverse on multiple axes followed by subtract
template <migraphx::shape::type_t DType>
struct test_gen_reverse_multi_axis : verify_program<test_gen_reverse_multi_axis<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {4, 8, 16}};
        auto x                    = mm->add_parameter("x", s);
        std::vector<int64_t> axes = {0, 2};
        auto reversed = mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), x);
        auto y        = mm->add_parameter("y", s);
        mm->add_instruction(migraphx::make_op("sub"), reversed, y);
        return p;
    }
};

template struct test_gen_reverse_multi_axis<migraphx::shape::float_type>;
template struct test_gen_reverse_multi_axis<migraphx::shape::half_type>;

// Test reverse followed by relu - single argument pointwise
template <migraphx::shape::type_t DType>
struct test_gen_reverse_relu : verify_program<test_gen_reverse_relu<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {8, 16, 32}};
        auto x                    = mm->add_parameter("x", s);
        std::vector<int64_t> axes = {1};
        auto reversed = mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), x);
        mm->add_instruction(migraphx::make_op("relu"), reversed);
        return p;
    }
};

template struct test_gen_reverse_relu<migraphx::shape::float_type>;
template struct test_gen_reverse_relu<migraphx::shape::half_type>;

