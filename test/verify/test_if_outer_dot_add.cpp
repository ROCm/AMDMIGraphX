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

// Each branch of the if multiplies an input of the main module by a weight of the main module and
// adds a bias, so the dot and add the GPU fuses read values of the enclosing module
struct test_if_outer_dot_add : verify_program<test_if_outer_dot_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape xs{migraphx::shape::float_type, {2, 8}};
        auto cond = mm->add_parameter("cond", migraphx::shape{migraphx::shape::bool_type});
        auto x    = mm->add_parameter("x", xs);
        auto y    = mm->add_parameter("y", xs);
        auto w    = mm->add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {8, 16}}, 1));

        auto branch = [&](const std::string& name, migraphx::instruction_ref input, int seed) {
            auto* bm = p.create_module(name);
            auto b   = bm->add_literal(migraphx::generate_literal(
                migraphx::shape{migraphx::shape::float_type, {2, 16}}, seed));
            auto dot = bm->add_instruction(migraphx::make_op("dot"), input, w);
            bm->add_return({bm->add_instruction(migraphx::make_op("add"), dot, b)});
            return bm;
        };
        auto* then_mod = branch("If_0_if", x, 2);
        auto* else_mod = branch("If_0_else", y, 3);
        auto ret       = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return(
            {mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret)});
        return p;
    }
};
