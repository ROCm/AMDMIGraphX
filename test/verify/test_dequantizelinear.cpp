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

struct test_dequantizelinear : verify_program<test_dequantizelinear>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto x    = mm->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 4}});
        auto w    = mm->add_parameter("w", {migraphx::shape::int8_type, {2, 8, 3, 3}});
        auto b    = mm->add_parameter("b", {migraphx::shape::int32_type, {1, 2, 2, 2}});
        auto conv = mm->add_instruction(migraphx::make_op("quant_convolution"), x, w);
        migraphx::shape ss{migraphx::shape::float_type, {1, 2, 2, 2}};
        migraphx::shape sz{migraphx::shape::int32_type, {1, 2, 2, 2}};
        //auto input2 = mm->add_parameter("x_scale", ss);
        std::vector<float> datax = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
        auto input2    = mm->add_literal(migraphx::literal(ss, datax));
        auto input3 = mm->add_parameter("x_zero_point", sz);
        auto dequant =
            mm->add_instruction(migraphx::make_op("dequantizelinear"), conv, input2, input3);
        // conv, input2);
        mm->add_return({dequant});
        // mm->add_return({conv});
        // auto add  = mm->add_instruction(migraphx::make_op("add"), conv, b);
        // mm->add_return({add});
        // auto r       = mm->add_instruction(migraphx::make_op("quantizelinear"), dequant, input2,
        // input3); mm->add_return({r});
        // auto s = migraphx::gpu::dump_mlir(m);

        // migraphx::shape sx{migraphx::shape::int8_type, {2, 2, 2}};
        // migraphx::shape ss{migraphx::shape::float_type, {2, 2, 2}};
        // migraphx::shape sz{migraphx::shape::int8_type, {2, 2, 2}};
        // auto input1 = mm->add_parameter("x", sx);
        // auto input2 = mm->add_parameter("x_scale", ss);
        // auto input3 = mm->add_parameter("x_zero_point", sz);
        // auto r = mm->add_instruction(migraphx::make_op("dequantizelinear"), input1, input2,
        // input3); mm->add_return({r});
        return p;
    };
};
