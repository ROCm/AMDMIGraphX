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

// A depthwise conv between two dense convs takes the dense convs' layout, so with the
// default channels-last selection it runs channels-last without any layout conversion
template <migraphx::shape::type_t DType>
struct test_channelwise_conv_nhwc_chain : verify_program<test_channelwise_conv_nhwc_chain<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{DType, {1, 8, 16, 16}});
        auto w1  = mm->add_literal(migraphx::generate_literal({DType, {8, 8, 1, 1}}, 1));
        auto w2  = mm->add_literal(migraphx::generate_literal({DType, {8, 1, 3, 3}}, 2));
        auto w3  = mm->add_literal(migraphx::generate_literal({DType, {8, 8, 1, 1}}, 3));
        auto c1  = mm->add_instruction(migraphx::make_op("convolution"), x, w1);
        auto r1  = mm->add_instruction(migraphx::make_op("relu"), c1);
        auto c2  = mm->add_instruction(
            migraphx::make_op("convolution", {{"group", 8}, {"padding", {1, 1}}}), r1, w2);
        auto r2 = mm->add_instruction(migraphx::make_op("relu"), c2);
        auto c3 = mm->add_instruction(migraphx::make_op("convolution"), r2, w3);
        mm->add_instruction(migraphx::make_op("relu"), c3);
        return p;
    }
    std::string section() const { return "conv"; }
};
template struct test_channelwise_conv_nhwc_chain<migraphx::shape::float_type>;
template struct test_channelwise_conv_nhwc_chain<migraphx::shape::half_type>;
