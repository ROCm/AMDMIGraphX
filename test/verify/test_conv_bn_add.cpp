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

struct test_conv_bn_add : verify_program<test_conv_bn_add>
{
    static migraphx::instruction_ref add_bn(migraphx::module& m,
                                            migraphx::instruction_ref x,
                                            std::size_t channels,
                                            std::size_t seed = 1)
    {
        migraphx::shape vars{migraphx::shape::float_type, {channels}};
        auto scale    = m.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1 + seed)));
        auto bias     = m.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2 + seed)));
        auto mean     = m.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3 + seed)));
        auto variance = m.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4 + seed)));
        return m.add_instruction(
            migraphx::make_op("batch_norm_inference"), x, scale, bias, mean, variance);
    }

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm              = p.get_main_module();
        std::size_t ichannels = 64;
        std::size_t ochannels = 256;
        auto x     = mm->add_parameter("x", {migraphx::shape::float_type, {1, ichannels, 56, 56}});
        auto w     = mm->add_literal(migraphx::generate_literal(
            {migraphx::shape::float_type, {ochannels, ichannels, 1, 1}}, 1));
        auto y     = mm->add_parameter("y", {migraphx::shape::float_type, {1, ichannels, 56, 56}});
        auto v     = mm->add_literal(migraphx::generate_literal(
            {migraphx::shape::float_type, {ochannels, ichannels, 1, 1}}, 2));
        auto relu1 = mm->add_instruction(migraphx::make_op("relu"), x);
        auto conv1 = mm->add_instruction(migraphx::make_op("convolution"), relu1, w);
        auto bn1   = add_bn(*mm, conv1, ochannels, 1);
        auto relu2 = mm->add_instruction(migraphx::make_op("relu"), y);
        auto conv2 = mm->add_instruction(migraphx::make_op("convolution"), relu2, v);
        auto bn2   = add_bn(*mm, conv2, ochannels, 1);
        auto sum   = mm->add_instruction(migraphx::make_op("add"), bn1, bn2);
        mm->add_instruction(migraphx::make_op("relu"), sum);
        return p;
    }
};
