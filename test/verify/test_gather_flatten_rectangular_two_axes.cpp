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
#include <migraphx/literal.hpp>

struct test_gather_flatten_rectangular_two_axes
    : verify_program<test_gather_flatten_rectangular_two_axes>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto data    = mm->add_parameter("X", {migraphx::shape::float_type, {1, 12}});
        auto flatten = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), data);

        migraphx::shape indices_shape{migraphx::shape::int32_type, {2, 3}};
        std::vector<int32_t> indices = {4, 5, 6, 8, 9, 10};
        auto indices_literal         = mm->add_literal(migraphx::literal{indices_shape, indices});

        auto gather = mm->add_instruction(migraphx::make_op("gather"), flatten, indices_literal);
        mm->add_return({gather});

        return p;
    }
};
