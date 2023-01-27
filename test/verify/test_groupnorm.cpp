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
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

migraphx::instruction_ref
add_groupnorm(migraphx::module& m, migraphx::instruction_ref x, float eps = 1e-12f)
{
    auto lens      = x->get_shape().lens();
    auto reduce_op = migraphx::make_op("reduce_mean", {{"axes", {-1}}});
    auto reduce1   = m.add_instruction(reduce_op, x);
    auto sqdiff    = migraphx::add_common_op(m, migraphx::make_op("sqdiff"), {x, reduce1});
    auto reduce2   = m.add_instruction(reduce_op, sqdiff);
    auto sub       = migraphx::add_common_op(m, migraphx::make_op("sub"), {x, reduce1});
    auto epsilon   = m.add_literal(eps);
    auto add       = migraphx::add_common_op(m, migraphx::make_op("add"), {epsilon, reduce2});
    auto rsqrt     = m.add_instruction(migraphx::make_op("rsqrt"), add);
    return migraphx::add_common_op(m, migraphx::make_op("mul"), {rsqrt, sub});
}

struct test_groupnorm : verify_program<test_groupnorm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {2, 32, 40960};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_groupnorm(*mm, x);
        return p;
    }
};
