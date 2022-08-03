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
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

#include <migraphx/op/batch_norm_inference.hpp>

struct test_batchnorm_1d_per_actv : verify_program<test_batchnorm_1d_per_actv>
{
    const size_t d1       = 5;
    const size_t channels = 2;
    const size_t batches  = 3;

    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, d1}};
        migraphx::shape vars{migraphx::shape::float_type, {channels, d1}};
        auto x        = mm->add_parameter("x", s);
        auto scale    = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = mm->add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        mm->add_instruction(
            migraphx::make_op(
                "batch_norm_inference",
                {{"epsilon", 1.0e-5},
                 {"momentum", 0.96f},
                 {"bn_mode",
                  migraphx::to_value(migraphx::op::batch_norm_inference::per_activation)}}),
            x,
            scale,
            bias,
            mean,
            variance);
        return p;
    }
};
