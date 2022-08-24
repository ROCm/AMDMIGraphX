/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/op/reduce_mean.hpp>

migraphx::instruction_ref
add_layernorm(migraphx::module& m, migraphx::instruction_ref x, std::vector<size_t> dims)
{
    auto scale =
        m.add_parameter("scale", migraphx::shape{migraphx::shape::float_type, {dims.back()}});
    auto bias =
        m.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {dims.back()}});
    auto epsilon  = m.add_literal(1e-12f);
    auto exponent = m.add_literal(2.0f);

    auto mean = m.add_instruction(migraphx::op::reduce_mean({2}), x);
    auto mean_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), mean);
    auto sub = m.add_instruction(migraphx::make_op("sub"), x, mean_mbcast);
    auto exponent_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), exponent);
    auto pow            = m.add_instruction(migraphx::make_op("pow"), sub, exponent_mbcast);
    auto var            = m.add_instruction(migraphx::op::reduce_mean({2}), pow);
    auto epsilon_mbcast = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, dims.at(1), 1}}}), epsilon);
    auto add_epsilon = m.add_instruction(migraphx::make_op("add"), var, epsilon_mbcast);
    auto sqrt        = m.add_instruction(migraphx::make_op("sqrt"), add_epsilon);
    auto sqrt_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), sqrt);
    auto div = m.add_instruction(migraphx::make_op("div"), sub, sqrt_mbcast);
    auto scale_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), scale);
    auto mul = m.add_instruction(migraphx::make_op("mul"), scale_mbcast, div);
    auto bias_mbcast =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), bias);
    return m.add_instruction(migraphx::make_op("add"), mul, bias_mbcast);
}

struct test_layernorm : verify_program<test_layernorm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 2, 5};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }
};

struct test_layernorm2 : verify_program<test_layernorm2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 4, 24};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }
};

struct test_layernorm_triadd : verify_program<test_layernorm_triadd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 4, 24};
        auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        auto y    = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, dims});
        auto z    = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, dims});
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        add_layernorm(*mm, add2, dims);
        return p;
    }
};

struct test_layernorm_triadd_large : verify_program<test_layernorm_triadd_large>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 384, 1024};
        auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        auto y    = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, dims});
        auto z    = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, dims});
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        add_layernorm(*mm, add2, dims);
        return p;
    }
};
