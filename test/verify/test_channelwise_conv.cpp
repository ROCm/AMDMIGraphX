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

template <migraphx::shape::type_t DType>
struct test_channelwise_conv_depthwise : verify_program<test_channelwise_conv_depthwise<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto input   = mm->add_parameter("x", migraphx::shape{DType, {2, 4, 8, 8}});
        auto weights = mm->add_parameter("w", migraphx::shape{DType, {4, 1, 3, 3}});
        mm->add_instruction(migraphx::make_op("convolution", {{"group", 4}}), input, weights);
        return p;
    }
    std::string section() const { return "conv"; }
};
template struct test_channelwise_conv_depthwise<migraphx::shape::float_type>;
template struct test_channelwise_conv_depthwise<migraphx::shape::half_type>;

template <migraphx::shape::type_t DType>
struct test_channelwise_conv_single_channel
    : verify_program<test_channelwise_conv_single_channel<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto input   = mm->add_parameter("x", migraphx::shape{DType, {2, 1, 8, 8}});
        auto weights = mm->add_parameter("w", migraphx::shape{DType, {4, 1, 3, 3}});
        mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        return p;
    }
    std::string section() const { return "conv"; }
};
template struct test_channelwise_conv_single_channel<migraphx::shape::float_type>;
template struct test_channelwise_conv_single_channel<migraphx::shape::half_type>;

template <migraphx::shape::type_t DType>
struct test_channelwise_conv_depthwise_5x5
    : verify_program<test_channelwise_conv_depthwise_5x5<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto input   = mm->add_parameter("x", migraphx::shape{DType, {1, 8, 12, 12}});
        auto weights = mm->add_parameter("w", migraphx::shape{DType, {8, 1, 5, 5}});
        mm->add_instruction(migraphx::make_op("convolution", {{"group", 8}}), input, weights);
        return p;
    }
    std::string section() const { return "conv"; }
};
template struct test_channelwise_conv_depthwise_5x5<migraphx::shape::float_type>;
template struct test_channelwise_conv_depthwise_5x5<migraphx::shape::half_type>;

template <migraphx::shape::type_t DType>
struct test_channelwise_conv_1d : verify_program<test_channelwise_conv_1d<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto input   = mm->add_parameter("x", migraphx::shape{DType, {2, 4, 16}});
        auto weights = mm->add_parameter("w", migraphx::shape{DType, {4, 1, 3}});
        mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}, {"group", 4}}),
            input,
            weights);
        return p;
    }
    std::string section() const { return "conv"; }
};
template struct test_channelwise_conv_1d<migraphx::shape::float_type>;
template struct test_channelwise_conv_1d<migraphx::shape::half_type>;
