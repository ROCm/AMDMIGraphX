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

// The rsqrt scale is computed on the reshaped variance and broadcast back to
// the input shape before the final reduce, so it fuses into that reduce module
// as a pointwise instruction whose only input is the broadcast reduce result
struct test_reduce_broadcast_scalar_pointwise
    : verify_program<test_reduce_broadcast_scalar_pointwise>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 16, 1, 1}};
        migraphx::shape ss{migraphx::shape::float_type, {1}};
        std::vector<int64_t> axes = {0, 2, 3, 4};
        auto x                    = mm->add_parameter("x", s);
        auto scale                = mm->add_literal(migraphx::literal{ss, {1.0f / 16}});
        auto eps                  = mm->add_literal(migraphx::literal{ss, {0.001f}});
        auto broadcast = [&](migraphx::instruction_ref ins, std::vector<std::size_t> lens) {
            return mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                                       ins);
        };
        auto mean = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), x);
        auto diff = mm->add_instruction(migraphx::make_op("sub"), x, broadcast(mean, s.lens()));
        auto sq   = mm->add_instruction(migraphx::make_op("mul"), diff, diff);
        auto var  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sq);
        auto var3 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1}}}), var);
        auto scaled =
            mm->add_instruction(migraphx::make_op("mul"), var3, broadcast(scale, {1, 1, 1}));
        auto biased =
            mm->add_instruction(migraphx::make_op("add"), scaled, broadcast(eps, {1, 1, 1}));
        auto rsqrt = mm->add_instruction(migraphx::make_op("rsqrt"), biased);
        auto rsqrt5 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2, 4}}}), rsqrt);
        auto norm =
            mm->add_instruction(migraphx::make_op("mul"), diff, broadcast(rsqrt5, s.lens()));
        auto sum    = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), norm);
        auto result = mm->add_instruction(migraphx::make_op("sub"), norm, broadcast(sum, s.lens()));
        mm->add_return({result});
        return p;
    }

    std::string section() const { return "reduce"; }
};
