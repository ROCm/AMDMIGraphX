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
#include <migraphx/literal.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// A loop body with more than 10 scan outputs: the output parameters from the 11th on are named
// with a padded index
struct test_loop_many_scan_outputs : verify_program<test_loop_many_scan_outputs>
{
    migraphx::program create_program() const
    {
        const std::size_t n = 11;
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape si{migraphx::shape::int64_type, {1}};
        migraphx::shape sc{migraphx::shape::bool_type, {1}};
        migraphx::shape s{migraphx::shape::float_type, {4}};
        auto in_iter = mm->add_literal(migraphx::literal(si, {2}));
        auto in_cond = mm->add_literal(migraphx::literal(sc, {true}));
        auto in_val  = mm->add_parameter("x", s);

        auto* body = p.create_module("loop_module");
        body->add_parameter("iter_num", si);
        auto cond                                      = body->add_parameter("cond", sc);
        auto prev                                      = body->add_parameter("prev", s);
        std::vector<migraphx::instruction_ref> returns = {cond, prev};
        for(std::size_t i = 0; i < n; ++i)
        {
            auto scale = body->add_literal(migraphx::literal(s, std::vector<float>(4, i + 2.0f)));
            returns.push_back(body->add_instruction(migraphx::make_op("mul"), prev, scale));
        }
        body->add_return(returns);

        auto rl = mm->add_instruction(
            migraphx::make_op("loop", {{"max_iterations", 4}}), {in_iter, in_cond, in_val}, {body});
        std::vector<migraphx::instruction_ref> outputs;
        for(std::size_t i = 0; i < n + 1; ++i)
            outputs.push_back(
                mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", i}}), rl));
        mm->add_return(outputs);

        return p;
    }
};
