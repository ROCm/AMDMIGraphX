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

template <std::size_t N, migraphx::shape::type_t DType>
struct test_split_reduce_add : verify_program<test_split_reduce_add<N, DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {N, 32, 20, 16}};
        migraphx::shape bs{DType, {1, 32, 1, 16}, {1, 1, 1, 32}};
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", bs);
        auto reduce_mean =
            mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), x);
        auto add = mm->add_instruction(migraphx::make_op("add"), reduce_mean, y);
        mm->add_return({add});
        return p;
    };

    std::string section() const { return "reduce"; }
};

template struct test_split_reduce_add<14400, migraphx::shape::float_type>;
template struct test_split_reduce_add<3276, migraphx::shape::float_type>;
template struct test_split_reduce_add<3277, migraphx::shape::float_type>;
