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

struct test_concat_param_pointwise_concat : verify_program<test_concat_param_pointwise_concat>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {32, 8}};
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto z     = mm->add_parameter("z", s);
        auto c     = mm->add_literal(migraphx::generate_literal(sc, 1));
        auto exp   = mm->add_instruction(migraphx::make_op("exp"), y);
        auto inner = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), exp, z);
        auto add   = mm->add_instruction(migraphx::make_op("add"), inner, c);
        auto outer = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, add);
        mm->add_return({outer});
        return p;
    }
};

struct test_concat_param_pointwise_concat_mod
    : verify_program<test_concat_param_pointwise_concat_mod>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::int32_type, {16, 1}};
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto z     = mm->add_parameter("z", s);
        auto neg   = mm->add_instruction(migraphx::make_op("neg"), y);
        auto inner = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), neg, z);
        std::vector<std::size_t> lens = {32, 1};
        auto ten  = mm->add_literal(migraphx::literal{migraphx::shape{s.type()}, {10}});
        auto five = mm->add_literal(migraphx::literal{migraphx::shape{s.type()}, {5}});
        auto mod  = mm->add_instruction(
            migraphx::make_op("mod"),
            inner,
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), ten));
        auto add = mm->add_instruction(
            migraphx::make_op("add"),
            mod,
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), five));
        auto outer = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, add);
        mm->add_return({outer});
        return p;
    }
};
