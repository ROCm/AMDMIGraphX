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

struct test_propagate_precision_boundary : verify_program<test_propagate_precision_boundary>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        migraphx::shape s1{migraphx::shape::float_type, {1, 32}};
        auto* mm = p.get_main_module();

        auto x        = mm->add_parameter("x", s1);
        auto sig      = mm->add_instruction(migraphx::make_op("sigmoid"), x);
        auto cvt_bool = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), sig);
        auto n       = mm->add_instruction(migraphx::make_op("not"), cvt_bool);
        auto cvt_int = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), n);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), cvt_int);
        mm->add_return({rsum, sig});
        return p;
    };

    migraphx::parameter_map get_data_values() const
    {
        migraphx::shape s{migraphx::shape::float_type, {1, 32}};
        migraphx::argument arg{s};
        std::vector<float> data(32, 0.0f);
        arg.fill(data.begin(), data.end());
        return {{"x", arg}};
    };

    std::string section() const { return "reduce"; }
};
