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
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <sstream>
#include <migraphx/make_op.hpp>

#include <migraphx/serialize.hpp>

#include "test.hpp"

template <class... Ts>
void expect_shape(const migraphx::shape& expected, const migraphx::operation& op, Ts... xs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape> shapes{xs...};
    std::vector<migraphx::instruction_ref> args(shapes.size());
    std::transform(
        shapes.begin(), shapes.end(), args.begin(), [&](auto&& s) { return mm->add_outline(s); });
    mm->add_instruction(op, args);
    if(p.get_output_shapes().back() != expected)
    {
        std::cout << "FAILED: Incorrect shape for " << op << ": ";
        std::cout << expected << " != " << p.get_output_shapes().back() << std::endl;
        for(auto&& s : shapes)
            std::cout << "    " << s << std::endl;
    }
}

template <class... Ts>
void throws_shape(const migraphx::operation& op, Ts... xs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape> shapes{xs...};
    std::vector<migraphx::instruction_ref> args(shapes.size());
    std::transform(
        shapes.begin(), shapes.end(), args.begin(), [&](auto&& s) { return mm->add_outline(s); });
    bool thrown = test::throws([&] { mm->add_instruction(op, args); });
    if(not thrown)
    {
        std::cout << "FAILED: No error found for " << op.name() << ": ";
        for(auto&& s : shapes)
            std::cout << "    " << s << std::endl;
    }
}

template <class...>
struct always_false : std::false_type
{
};

template <class... Ts>
void throws_shape(const migraphx::shape&, Ts...)
{
    static_assert(always_false<Ts...>{},
                  "An expected shape should not be passed to throws_shape function");
}

TEST_CASE(binary_dyn_static_error)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 4, 4}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 1, 0}, {4, 4, 4}, {4, 4, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    throws_shape(migraphx::make_op("add"), a_shape, b_shape);
}

TEST_CASE(broadcast)
{
    {
        std::vector<std::size_t> lens{1, 1};
        migraphx::shape input{migraphx::shape::float_type, {1}, {0}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 1}, {0, 0}},
                     migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", lens}}),
                     input);
    }

    {
        std::vector<std::size_t> lens{1, 1};
        migraphx::shape input{migraphx::shape::float_type, {2}};
        throws_shape(migraphx::op::broadcast{1, lens}, input);
    }

    {
        std::vector<std::size_t> lens{2, 2};
        migraphx::shape input{migraphx::shape::float_type, {1, 2}};
        throws_shape(migraphx::op::broadcast{1, lens}, input);
    }

    {
        std::vector<std::size_t> lens{3, 2, 4, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 2, 4, 3}, {0, 0, 3, 1}},
                     migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", lens}}),
                     input);
    }

    {
        std::vector<std::size_t> lens{3, 2, 4, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 4}};
        throws_shape(migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", lens}}), input);
    }
}

TEST_CASE(broadcast_axis_out_of_range_error)
{
    std::vector<std::size_t> lens{1, 1};
    migraphx::shape input{migraphx::shape::float_type, {1}, {0}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 4}, {"out_lens", lens}}), input);
}

TEST_CASE(broadcast_2in_static_static)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4}, {1}};
    migraphx::shape b_input{migraphx::shape::float_type, {4, 4}, {4, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 4}, {1, 0}},
                 migraphx::make_op("broadcast", {{"axis", 0}}),
                 a_input,
                 b_input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 4}, {0, 1}},
                 migraphx::make_op("broadcast", {{"axis", 1}}),
                 a_input,
                 b_input);
    throws_shape(migraphx::make_op("broadcast", {{"axis", 2}}), a_input, b_input);
}

TEST_CASE(broadcast_2in_not_matching_error)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4}, {1}};
    migraphx::shape b_input{migraphx::shape::float_type, {2, 2}, {2, 1}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 1}}), a_input, b_input);
}

TEST_CASE(broadcast_2in_dynamic_s0_error1)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4, 2}, {2, 1}};
    migraphx::shape b_input{migraphx::shape::float_type, {{1, 4, 0}, {4, 4, 0}, {2, 2, 0}}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 0}}), b_input, a_input);
}

TEST_CASE(broadcast_2in_dynamic_s0_error2)
{
    std::vector<migraphx::shape::dynamic_dimension> dd{{4, 4, 0}};
    migraphx::shape a_input{migraphx::shape::float_type, dd};
    migraphx::shape b_input{migraphx::shape::float_type, {4, 4}, {4, 1}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 0}}), a_input, b_input);
}

TEST_CASE(broadcast_2in_static_dyn)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4}, {1}};
    migraphx::shape b_input{migraphx::shape::float_type, {{1, 4, 0}, {4, 4, 0}, {2, 2, 0}}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 0}}), a_input, b_input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4, 0}, {4, 4, 0}, {2, 2, 0}}},
                 migraphx::make_op("broadcast", {{"axis", 1}}),
                 a_input,
                 b_input);
    throws_shape(migraphx::make_op("broadcast", {{"axis", 2}}), a_input, b_input);
}

TEST_CASE(broadcast_2in_dyn_s0_ndim_greater_than_1_error)
{
    migraphx::shape a_input{migraphx::shape::float_type, {4, 2}};
    migraphx::shape b_input{migraphx::shape::float_type, {{1, 4, 0}, {4, 4, 0}, {2, 2, 0}}};
    throws_shape(migraphx::make_op("broadcast", {{"axis", 0}}), a_input, b_input);
}

TEST_CASE(convolution_shape)
{
    migraphx::shape output{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("convolution"), input, weights);
    throws_shape(migraphx::make_op("convolution"), input);
    throws_shape(
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input);

    migraphx::shape input2{migraphx::shape::float_type, {3, 3}};
    migraphx::shape weights2{migraphx::shape::float_type, {3, 3}};
    throws_shape(migraphx::make_op("convolution"), input2, weights2);
    throws_shape(migraphx::make_op("convolution"), input2, weights);

    // 1D convolution
    migraphx::shape output_1d{migraphx::shape::float_type, {4, 4, 1}};
    migraphx::shape input_1d{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape weights_1d{migraphx::shape::float_type, {4, 3, 3}};
    expect_shape(
        output_1d,
        migraphx::make_op("convolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input_1d,
        weights_1d);

    // channel numbers mismatch
    weights_1d = {migraphx::shape::float_type, {4, 8, 3}};
    throws_shape(migraphx::make_op("convolution"), input_1d, weights_1d);

    // 3D convolution
    migraphx::shape output_3d{migraphx::shape::float_type, {4, 4, 1, 1, 1}};
    migraphx::shape input_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    migraphx::shape weights_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    expect_shape(
        output_3d,
        migraphx::make_op("convolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        input_3d,
        weights_3d);

    throws_shape(migraphx::make_op("convolution"), input_3d, weights_3d);

    // dynamic batch
    migraphx::shape input_dyn_shape{migraphx::shape::float_type,
                                    {{1, 100, 0}, {3, 3, 0}, {5, 5, 0}, {5, 5, 0}}};
    migraphx::shape weights_shape{migraphx::shape::float_type, {1, 3, 3, 3}};
    migraphx::shape output_dyn_shape{migraphx::shape::float_type,
                                     {{
                                          1,
                                          100,
                                          0,
                                      },
                                      {1, 1, 0},
                                      {3, 3, 0},
                                      {3, 3, 0}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);

    // dynamic image
    input_dyn_shape = {migraphx::shape::float_type, {{1, 1, 0}, {3, 3, 0}, {5, 20, 0}, {5, 20, 0}}};
    weights_shape   = {migraphx::shape::float_type, {1, 3, 3, 3}};
    output_dyn_shape = {migraphx::shape::float_type,
                        {{
                             1,
                             1,
                             0,
                         },
                         {1, 1, 0},
                         {3, 18, 0},
                         {3, 18, 0}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);

    // dynamic weights
    input_dyn_shape  = {migraphx::shape::float_type, {1, 3, 10, 10}};
    weights_shape    = {migraphx::shape::float_type, {{1, 1, 0}, {3, 3, 0}, {2, 4, 0}, {2, 4, 0}}};
    output_dyn_shape = {migraphx::shape::float_type,
                        {{
                             1,
                             1,
                             0,
                         },
                         {1, 1, 0},
                         {7, 9, 0},
                         {7, 9, 0}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);

    // dynamic img and weights
    input_dyn_shape = {migraphx::shape::float_type, {{1, 1, 0}, {3, 3, 0}, {5, 20, 0}, {5, 20, 0}}};
    weights_shape   = {migraphx::shape::float_type, {{1, 1, 0}, {3, 3, 0}, {2, 4, 0}, {2, 4, 0}}};
    output_dyn_shape = {migraphx::shape::float_type,
                        {{
                             1,
                             1,
                             0,
                         },
                         {1, 1, 0},
                         {2, 19, 0},
                         {2, 19, 0}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);

    // input attr shape mismatch
    input_dyn_shape = {migraphx::shape::float_type,
                       {{1, 100, 0}, {3, 3, 0}, {5, 5, 0}, {5, 5, 0}, {5, 5, 0}}};
    weights_shape   = {migraphx::shape::float_type, {1, 3, 3, 3, 3}};
    throws_shape(migraphx::make_op("convolution",
                                   {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input_dyn_shape,
                 weights_shape);

    // auto_pad dynamic batch
    input_dyn_shape  = {migraphx::shape::float_type, {{1, 10, 0}, {3, 3, 0}, {5, 5, 0}, {5, 5, 0}}};
    weights_shape    = {migraphx::shape::float_type, {1, 3, 3, 3}};
    output_dyn_shape = {migraphx::shape::float_type, {{1, 10, 0}, {1, 1, 0}, {5, 5, 0}, {5, 5, 0}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"stride", {1, 1}},
                                    {"dilation", {1, 1}},
                                    {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
                 input_dyn_shape,
                 weights_shape);

    // auto_pad dynamic img
    input_dyn_shape = {migraphx::shape::float_type, {{1, 1, 0}, {3, 3, 0}, {5, 10, 0}, {5, 10, 0}}};
    weights_shape   = {migraphx::shape::float_type, {1, 3, 3, 3}};
    output_dyn_shape = {migraphx::shape::float_type,
                        {{1, 1, 0}, {1, 1, 0}, {5, 10, 0}, {5, 10, 0}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"stride", {1, 1}},
                                    {"dilation", {1, 1}},
                                    {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
                 input_dyn_shape,
                 weights_shape);

    // auto_pad dynamic kernel
    input_dyn_shape  = {migraphx::shape::float_type,
                       {{1, 1, 0}, {3, 3, 0}, {10, 10, 0}, {10, 10, 0}}};
    weights_shape    = {migraphx::shape::float_type, {{1, 1, 0}, {3, 3, 0}, {2, 4, 0}, {2, 4, 0}}};
    output_dyn_shape = {migraphx::shape::float_type,
                        {{1, 1, 0}, {1, 1, 0}, {10, 10, 0}, {10, 10, 0}}};
    expect_shape(output_dyn_shape,
                 migraphx::make_op("convolution",
                                   {{"stride", {1, 1}},
                                    {"dilation", {1, 1}},
                                    {"padding_mode", migraphx::op::padding_mode_t::same_lower}}),
                 input_dyn_shape,
                 weights_shape);
}

TEST_CASE(contiguous_shape)
{
    migraphx::shape output{migraphx::shape::float_type, {2, 2}};
    migraphx::shape input{migraphx::shape::float_type, {2, 2}, {1, 2}};
    expect_shape(output, migraphx::make_op("contiguous"), input);
    throws_shape(migraphx::make_op("contiguous"), input, input);

    migraphx::shape single{migraphx::shape::float_type, {2}};
    expect_shape(single, migraphx::make_op("contiguous"), single);
}

TEST_CASE(contiguous_dyn_shape)
{
    migraphx::shape s0{migraphx::shape::float_type, {{1, 4, 0}, {2, 2, 2}}};
    expect_shape(s0, migraphx::make_op("contiguous"), s0);
}

TEST_CASE(contiguous_shape_scalar)
{
    migraphx::shape output{migraphx::shape::float_type};
    migraphx::shape input{migraphx::shape::float_type};
    expect_shape(output, migraphx::make_op("contiguous"), input);
}

TEST_CASE(deconvolution_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 4, 1, 1}};
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("deconvolution"), input, weights);
    throws_shape(migraphx::make_op("deconvolution"), input);
    throws_shape(
        migraphx::make_op("deconvolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input);

    migraphx::shape input_1d{migraphx::shape::float_type, {4, 4, 1}};
    migraphx::shape output_1d{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape weights_1d{migraphx::shape::float_type, {4, 3, 3}};
    expect_shape(
        output_1d,
        migraphx::make_op("deconvolution", {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input_1d,
        weights_1d);

    migraphx::shape input_3d{migraphx::shape::float_type, {4, 4, 1, 1, 1}};
    migraphx::shape output_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    migraphx::shape weights_3d{migraphx::shape::float_type, {4, 3, 3, 3, 3}};
    expect_shape(
        output_3d,
        migraphx::make_op("deconvolution",
                          {{"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        input_3d,
        weights_3d);
}

TEST_CASE(flatten_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 4, 6, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 2 * 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 0}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 2 * 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", -4}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 1}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 4 * 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", -3}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4, 6 * 8}},
                 migraphx::make_op("flatten", {{"axis", 2}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4 * 6, 8}},
                 migraphx::make_op("flatten", {{"axis", 3}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {2 * 4 * 6 * 8, 1}},
                 migraphx::make_op("flatten", {{"axis", 4}}),
                 input);
    throws_shape(migraphx::make_op("flatten", {{"axis", 5}}), input);
    throws_shape(migraphx::make_op("flatten", {{"axis", -5}}), input);
}

TEST_CASE(gather)
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = 1;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 2, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {1}};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = -4;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 5}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = 3;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {3}};
        migraphx::shape indices{migraphx::shape::int32_type};
        int axis = 0;
        expect_shape(migraphx::shape{migraphx::shape::float_type},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {3}};
        migraphx::shape indices{migraphx::shape::int32_type, {1}};
        int axis = 0;
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1}},
                     migraphx::make_op("gather", {{"axis", axis}}),
                     input,
                     indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = 4;
        throws_shape(migraphx::make_op("gather", {{"axis", axis}}), input, indices);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        migraphx::shape indices{migraphx::shape::int32_type, {2, 3}};
        int axis = -5;
        throws_shape(migraphx::make_op("gather", {{"axis", axis}}), input, indices);
    }
}

// 3 input arguments
TEST_CASE(gemm)
{
    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {10, 8}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 6}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 8}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4, 8}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 6}};
        migraphx::shape s_m2{migraphx::shape::float_type, {2, 5, 8}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }
}

TEST_CASE(get_tuple_elem_test)
{
    migraphx::shape s0{migraphx::shape::bool_type, {1, 1}};
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::int32_type, {5, 6}};
    migraphx::shape s_tuple({s0, s1, s2});

    expect_shape(s0, migraphx::make_op("get_tuple_elem", {{"index", 0}}), s_tuple);
    expect_shape(s1, migraphx::make_op("get_tuple_elem", {{"index", 1}}), s_tuple);
    expect_shape(s2, migraphx::make_op("get_tuple_elem", {{"index", 2}}), s_tuple);
    throws_shape(migraphx::make_op("get_tuple_elem", {{"index", 3}}), s_tuple);
    throws_shape(migraphx::make_op("get_tuple_elem", {{"index", 0}}), s0);
    throws_shape(migraphx::make_op("get_tuple_elem", {{"index", 1}}), s1);
    throws_shape(migraphx::make_op("get_tuple_elem", {{"index", 0}}), s2);
}

TEST_CASE(gru)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "gru",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

TEST_CASE(inconsistent_attr_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(migraphx::make_op("convolution",
                                   {{"padding", {1, 1}}, {"stride", {2}}, {"dilation", {3, 3, 3}}}),
                 input,
                 weights);
    throws_shape(migraphx::make_op("deconvolution",
                                   {{"padding", {1, 1}}, {"stride", {2}}, {"dilation", {3, 3, 3}}}),
                 input,
                 weights);
    throws_shape(migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {1}},
                                    {"stride", {0}},
                                    {"lengths", {1, 1}}}),
                 input);
}

template <class T>
void test_softmax_variations()
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}}, T{0}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}}, T{1}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}}, T{2}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 5}}, T{3}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        int axis = 4;
        throws_shape(T{axis}, input);
    }
}
TEST_CASE(logsoftmax) { test_softmax_variations<migraphx::op::logsoftmax>(); }

TEST_CASE(lstm)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        throws_shape(
            migraphx::make_op(
                "lstm",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

// 2 inputs arguments
TEST_CASE(matmul)
{
    {
        migraphx::shape s_m1{migraphx::shape::float_type, {5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 2}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 4}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {4, 4}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {6, 5, 4}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {6, 1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {6, 5, 4}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {6, 1, 4}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 6, 1, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 6, 5, 4}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 6, 1, 4}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {5, 8}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 1}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 1}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 5, 7}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 4, 7}},
                     migraphx::make_op("dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 1, 5, 7}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::float_type, {1, 1, 4, 5}};
        migraphx::shape s_m2{migraphx::shape::float_type, {1, 2, 5, 7}};
        throws_shape(migraphx::make_op("dot"), s_m1, s_m2);
    }
}

TEST_CASE(multibroadcast)
{
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {2, 1, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 3, 0, 1}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {2, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 1, 0, 0}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {5, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 0, 1, 0}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {1, 0, 0, 0}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 2, 5, 3};
        migraphx::shape input{migraphx::shape::float_type, {3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 0, 0, 1}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 4, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 3}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {0, 3, 3, 1}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, lens, {1, 1, 1, 0}},
                     migraphx::make_op("multibroadcast", {{"out_lens", lens}}),
                     input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 3};
        migraphx::shape input{migraphx::shape::float_type, {4, 1, 1, 1}};
        throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{4, 1, 3};
        std::vector<std::size_t> empt = {};
        migraphx::shape input{migraphx::shape::float_type, empt};
        throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{2, 3, 4, 5};
        migraphx::shape input{migraphx::shape::float_type, {3, 4}};
        throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
    }
    {
        std::vector<std::size_t> lens{2, 3, 4, 5};
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4}};
        throws_shape(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), input);
    }
}

TEST_CASE(multibroadcast_2in_static_dyn0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 4, 0}, {4, 4, 4}, {4, 4, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4, 0}, {4, 4, 0}, {4, 4, 0}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4, 0}, {4, 4, 0}, {4, 4, 0}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn1)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{8, 8, 0}, {6, 6, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{8, 8, 0}, {6, 6, 0}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{8, 8, 0}, {6, 6, 0}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn2)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{8, 8, 0}, {6, 6, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{8, 8, 0}, {6, 6, 0}}},
                 migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(b)}}),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{8, 8, 0}, {6, 6, 0}}},
                 migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(b)}}),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn_error0)
{
    // doesn't match on first dimension
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 3, 0}, {6, 6, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    throws_shape(migraphx::make_op("multibroadcast"), a_shape, b_shape);
    throws_shape(migraphx::make_op("multibroadcast"), b_shape, a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn_error1)
{
    // doesn't match on first dimension
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 4, 0}, {6, 6, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    throws_shape(migraphx::make_op("multibroadcast"), a_shape, b_shape);
    throws_shape(migraphx::make_op("multibroadcast"), b_shape, a_shape);
}

TEST_CASE(multibroadcast_2in_static_dyn_error2)
{
    // doesn't match on first dimension
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    std::vector<migraphx::shape::dynamic_dimension> b{{1, 2, 0}, {6, 6, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    throws_shape(migraphx::make_op("multibroadcast"), a_shape, b_shape);
    throws_shape(migraphx::make_op("multibroadcast"), b_shape, a_shape);
}

TEST_CASE(multibroadcast_2in_dyn_dyn0)
{
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4, 0}, {2, 4, 2}, {2, 4, 0}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 4, 2}, {2, 4, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4, 0}, {2, 4, 2}, {2, 4, 0}}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4, 0}, {2, 4, 2}, {2, 4, 0}}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_dyn_dyn1)
{
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4, 0}, {2, 4, 2}, {2, 4, 0}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 4, 2}, {2, 4, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4, 0}, {2, 4, 2}, {2, 4, 0}}},
                 migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(a)}}),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {{1, 4, 0}, {2, 4, 2}, {2, 4, 0}}},
                 migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(a)}}),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_dyn_dyn_error0)
{
    // max doesn't match on second dimension of a
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4, 0}, {2, 4, 2}, {2, 4, 0}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 5, 2}, {2, 4, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    throws_shape(migraphx::make_op("multibroadcast"), a_shape, b_shape);
    throws_shape(migraphx::make_op("multibroadcast"), b_shape, a_shape);
}

TEST_CASE(multibroadcast_2in_dyn_dyn_error1)
{
    // opt doesn't match on second dimension of a
    std::vector<migraphx::shape::dynamic_dimension> a{{1, 4, 0}, {2, 4, 2}, {2, 4, 0}};
    migraphx::shape a_shape{migraphx::shape::float_type, a};
    std::vector<migraphx::shape::dynamic_dimension> b{{2, 4, 3}, {2, 4, 0}};
    migraphx::shape b_shape{migraphx::shape::float_type, b};
    throws_shape(migraphx::make_op("multibroadcast"), a_shape, b_shape);
    throws_shape(migraphx::make_op("multibroadcast"), b_shape, a_shape);
}

TEST_CASE(multibroadcast_2in_static_static0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 6}};
    migraphx::shape b_shape{migraphx::shape::float_type, {3, 6}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 6}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 6}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static1)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 8}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 8}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}, {0, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 8}, {8, 1}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static2)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 4, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 4, 8}, {0, 0, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {4, 4, 8}, {4, 1, 0}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static3)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 4}, {16, 4, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 4}, {0, 1, 0}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static4)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 1, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 1}};
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 4}, {4, 0, 1}},
                 migraphx::make_op("multibroadcast"),
                 a_shape,
                 b_shape);
    expect_shape(migraphx::shape{migraphx::shape::float_type, {3, 4, 4}, {0, 1, 0}},
                 migraphx::make_op("multibroadcast"),
                 b_shape,
                 a_shape);
}

TEST_CASE(multibroadcast_2in_static_static_error0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4, 4}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 3}};
    throws_shape(migraphx::make_op("multibroadcast"), a_shape, b_shape);
    throws_shape(migraphx::make_op("multibroadcast"), b_shape, a_shape);
}

TEST_CASE(multinomial)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 5}};
    int dtype = 0;

    throws_shape(migraphx::make_op("multinomial", {{"dtype", dtype}}), s, s);
}

TEST_CASE(nms_shape)
{
    // use_dyn_output == false
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    migraphx::shape max_out_s{migraphx::shape::int64_type, {1}};
    migraphx::shape iou_thres_s{migraphx::shape::float_type, {1}};
    migraphx::shape score_thres_s{migraphx::shape::float_type, {1}};
    migraphx::shape output_s{migraphx::shape::int64_type, {6, 3}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", false}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // use_dyn_output == true
    output_s = {migraphx::shape::int64_type, {{0, 6, 0}, {3, 3, 0}}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic batches
    boxes_s  = {migraphx::shape::float_type, {{1, 3, 0}, {6, 6, 0}, {4, 4, 0}}};
    scores_s = {migraphx::shape::float_type, {{1, 3, 0}, {1, 1, 0}, {6, 6, 0}}};
    output_s = {migraphx::shape::int64_type, {{0, 18, 0}, {3, 3, 0}}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic num boxes
    boxes_s  = {migraphx::shape::float_type, {{1, 1, 0}, {6, 20, 0}, {4, 4, 0}}};
    scores_s = {migraphx::shape::float_type, {{1, 1, 0}, {1, 1, 0}, {6, 20, 0}}};
    output_s = {migraphx::shape::int64_type, {{0, 20, 0}, {3, 3, 0}}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // use_dyn_output false with dynamic input shape
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", false}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic classes
    boxes_s  = {migraphx::shape::float_type, {{1, 1, 0}, {6, 6, 0}, {4, 4, 0}}};
    scores_s = {migraphx::shape::float_type, {{1, 1, 0}, {1, 3, 0}, {6, 6, 0}}};
    output_s = {migraphx::shape::int64_type, {{0, 6, 0}, {3, 3, 0}}};
    expect_shape(output_s,
                 migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // fixed mismatch batches
    boxes_s  = {migraphx::shape::float_type, {2, 6, 4}};
    scores_s = {migraphx::shape::float_type, {1, 1, 6}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // fixed mismatch num boxes
    boxes_s  = {migraphx::shape::float_type, {1, 6, 4}};
    scores_s = {migraphx::shape::float_type, {1, 1, 4}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic mismatch batches
    boxes_s  = {migraphx::shape::float_type, {{1, 4, 0}, {6, 6, 0}, {4, 4, 0}}};
    scores_s = {migraphx::shape::float_type, {{2, 8, 0}, {1, 1, 0}, {6, 6, 0}}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic mismatch num boxes
    boxes_s  = {migraphx::shape::float_type, {{1, 1, 0}, {6, 8, 0}, {4, 4, 0}}};
    scores_s = {migraphx::shape::float_type, {{1, 1, 0}, {1, 1, 0}, {3, 9, 0}}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);

    // dynamic number of classes, fixed boxes_s, mismatch batches
    boxes_s  = {migraphx::shape::float_type, {1, 6, 4}};
    scores_s = {migraphx::shape::float_type, {{1, 3, 0}, {1, 3, 0}, {6, 6, 0}}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);
    // dynamic number of classes, fixed boxes_s, mismatch num boxes
    boxes_s  = {migraphx::shape::float_type, {1, 6, 4}};
    scores_s = {migraphx::shape::float_type, {{1, 1, 0}, {1, 3, 0}, {4, 8, 0}}};
    throws_shape(migraphx::make_op("nonmaxsuppression",
                                   {{"center_point_box", true}, {"use_dyn_output", true}}),
                 boxes_s,
                 scores_s,
                 max_out_s,
                 iou_thres_s,
                 score_thres_s);
}

TEST_CASE(pooling_shape)
{
    migraphx::shape output{migraphx::shape::float_type, {4, 3, 1, 1}};
    migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {1}},
                                    {"stride", {0}},
                                    {"lengths", {1}}}),
                 input);
    expect_shape(output,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}}}),
                 input);

    migraphx::shape output1{migraphx::shape::float_type, {4, 3, 2, 2}};
    expect_shape(output1,
                 migraphx::make_op("pooling",
                                   {{"mode", migraphx::op::pooling_mode::max},
                                    {"padding", {0, 0}},
                                    {"stride", {3, 3}},
                                    {"lengths", {1, 1}},
                                    {"ceil_mode", true}}),
                 input);
}

TEST_CASE(prefix_scan_sum)
{
    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2, 3}};
        throws_shape(
            migraphx::make_op("prefix_scan_sum", {{"axis", 3}, {"exclusive", 0}, {"reverse", 0}}),
            s);
    }

    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2}};
        throws_shape(
            migraphx::make_op("prefix_scan_sum", {{"axis", -3}, {"exclusive", 0}, {"reverse", 0}}),
            s);
    }
}

TEST_CASE(quant_convolution_shape)
{
    migraphx::shape output{migraphx::shape::int32_type, {4, 4, 1, 1}};
    migraphx::shape input{migraphx::shape::int8_type, {4, 3, 3, 3}};
    migraphx::shape weights{migraphx::shape::int8_type, {4, 3, 3, 3}};
    expect_shape(output, migraphx::make_op("quant_convolution"), input, weights);
    throws_shape(migraphx::make_op("quant_convolution"), input);
    throws_shape(migraphx::make_op("quant_convolution",
                                   {{"padding", {0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
                 input,
                 weights);
    throws_shape(migraphx::make_op("quant_convolution",
                                   {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
                 input,
                 weights);

    migraphx::shape input2{migraphx::shape::int32_type, {3, 3}};
    migraphx::shape weights2{migraphx::shape::float_type, {3, 3}};
    throws_shape(migraphx::make_op("quant_convolution"), input2, weights2);
    throws_shape(migraphx::make_op("quant_convolution"), input2, weights);

    migraphx::shape input3{migraphx::shape::int32_type, {4, 3, 3, 3}};
    migraphx::shape weight3{migraphx::shape::float_type, {4, 3, 3, 3}};
    throws_shape(migraphx::make_op("quant_convolution"), input3, weights);
    throws_shape(migraphx::make_op("quant_convolution"), input, weight3);
    throws_shape(migraphx::make_op("quant_convolution"), input3, weight3);
}

// quant_dot
TEST_CASE(quant_dot_2args)
{
    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {4, 8}};
        expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 8}},
                     migraphx::make_op("quant_dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {3, 8}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {8, 7}};
        expect_shape(migraphx::shape{migraphx::shape::int32_type, {3, 7}},
                     migraphx::make_op("quant_dot"),
                     s_m1,
                     s_m2);
    }

    {
        migraphx::shape s_m1{migraphx::shape::int8_type, {2, 4}};
        migraphx::shape s_m2{migraphx::shape::int8_type, {8, 8}};
        throws_shape(migraphx::make_op("quant_dot"), s_m1, s_m2);
    }
}

template <class T>
void test_reduce_ops()
{
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1}}, T{}, input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(
            migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1}}, T{{0, 1, 2, 3}}, input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 1, 1}}, T{{2, 3}}, input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}}, T{{0}}, input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::float_type, {2, 3, 4, 1}}, T{{-1}}, input);
    }
    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(T{{4}}, input);
    }
}

TEST_CASE(reduce_mean) { test_reduce_ops<migraphx::op::reduce_mean>(); }
TEST_CASE(reduce_sum) { test_reduce_ops<migraphx::op::reduce_sum>(); }

TEST_CASE(reshape_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {24, 1, 1, 1}};
    for(auto&& new_shape :
        std::vector<std::vector<int64_t>>{{8, 3, 1, 1}, {1, 3, 4, 2}, {1, 3, 4, 2}})
    {
        std::vector<std::size_t> lens(new_shape.size());
        std::copy(new_shape.begin(), new_shape.end(), lens.begin());
        migraphx::shape output{migraphx::shape::float_type, lens};
        expect_shape(output, migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    }

    for(auto&& new_shape :
        std::vector<std::vector<int64_t>>{{8, 3, 2, 2}, {1, 3, -1, -1}, {3, 0, 0}, {3, 2, 0}})
    {
        throws_shape(migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    }

    std::vector<std::pair<std::vector<int64_t>, migraphx::shape>> minus1_tests{
        {{2, -1, 3}, {migraphx::shape::float_type, {2, 4, 3}}},
        {{0, -1, 0}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{2, -1, 0}, {migraphx::shape::float_type, {2, 12, 1}}},
        {{0, 0, -1}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{2, 0, -1}, {migraphx::shape::float_type, {2, 1, 12}}},
        {{-1, 2, 3}, {migraphx::shape::float_type, {4, 2, 3}}},
        {{-1, 0, 3}, {migraphx::shape::float_type, {8, 1, 3}}},
        {{-1, 0, 0}, {migraphx::shape::float_type, {24, 1, 1}}},
        {{-1, 3, 0}, {migraphx::shape::float_type, {8, 3, 1}}}};

    for(auto& it : minus1_tests)
    {
        expect_shape(it.second, migraphx::make_op("reshape", {{"dims", it.first}}), input);
    }
}

TEST_CASE(rnn)
{
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::reverse)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        expect_shape(
            migraphx::shape{migraphx::shape::float_type,
                            {seq_len, num_dirct, batch_size, hidden_size}},
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size + 1},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }

    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 2;
        std::size_t hidden_size = 4;
        std::size_t input_size  = 3;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};

        throws_shape(
            migraphx::make_op(
                "rnn",
                {{"hidden_size", hidden_size},
                 {"actv_func",
                  migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("tanh")})},
                 {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
                 {"clip", clip}}),
            in_shape,
            w_shape,
            r_shape,
            b_shape,
            ih_shape);
    }
}

TEST_CASE(slice_shape)
{
    migraphx::shape input{migraphx::shape::int32_type, {2, 2, 3}};
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}},
                 migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}},
                 migraphx::make_op(
                     "slice", {{"axes", {0, 1, 2}}, {"starts", {0, 0, 1}}, {"ends", {2, 2, 3}}}),
                 input);
    expect_shape(migraphx::shape{migraphx::shape::int32_type, {2, 2, 1}, {6, 3, 1}},
                 migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {10}}}),
                 input);
}

TEST_CASE(softmax) { test_softmax_variations<migraphx::op::softmax>(); }

TEST_CASE(test_argmax)
{
    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {1, 3, 4, 5}},
                     migraphx::make_op("argmax", {{"axis", 0}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 1, 4, 5}},
                     migraphx::make_op("argmax", {{"axis", 1}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 1, 5}},
                     migraphx::make_op("argmax", {{"axis", 2}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 1}},
                     migraphx::make_op("argmax", {{"axis", 3}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(migraphx::make_op("argmax", {{"axis", 4}}), input);
    }
}

TEST_CASE(test_argmin)
{
    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {1, 3, 4, 5}},
                     migraphx::make_op("argmin", {{"axis", 0}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 1, 4, 5}},
                     migraphx::make_op("argmin", {{"axis", 1}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 1, 5}},
                     migraphx::make_op("argmin", {{"axis", 2}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::half_type, {2, 3, 4, 5}};
        expect_shape(migraphx::shape{migraphx::shape::int64_type, {2, 3, 4, 1}},
                     migraphx::make_op("argmin", {{"axis", 3}}),
                     input);
    }

    {
        migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
        throws_shape(migraphx::make_op("argmin", {{"axis", 4}}), input);
    }
}

TEST_CASE(test_scalar)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}, {1}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4, 5}, {0, 0, 0, 0}};
    expect_shape(s2, migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 3, 4, 5}}}), s1);
}

TEST_CASE(test_scalar_nelemnts)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 3, 4, 5}};
    throws_shape(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 3, 4, 5}}}), input);
}

TEST_CASE(test_scatternd)
{
    {
        // k > r
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {8}};
        migraphx::shape is{itype, {4, 2}};
        migraphx::shape us{dtype, {4}};
        throws_shape(migraphx::make_op("scatternd_none"), ds, is, us);
    }

    {
        // update.lens != indices.lens[0:q-1] ++ data.lens[k:r-1]
        auto dtype = migraphx::shape::float_type;
        auto itype = migraphx::shape::int64_type;
        migraphx::shape ds{dtype, {8}};
        migraphx::shape is{itype, {4, 1}};
        migraphx::shape us{dtype, {2, 2}};
        throws_shape(migraphx::make_op("scatternd_none"), ds, is, us);
    }
}

TEST_CASE(test_squeeze)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {3}}}), s1);
}

TEST_CASE(test_squeeze_all)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}};
    migraphx::shape s2{migraphx::shape::float_type};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_squeeze_dyn)
{
    migraphx::shape s1{migraphx::shape::float_type,
                       {{1, 4, 0}, {1, 1, 0}, {3, 3, 0}, {1, 1, 0}, {3, 3, 0}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 4, 0}, {1, 1, 0}, {3, 3, 0}, {3, 3, 0}}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {3}}}), s1);

    migraphx::shape s3{migraphx::shape::float_type, {{1, 4, 0}, {3, 3, 0}, {3, 3, 0}}};
    expect_shape(s3, migraphx::make_op("squeeze"), s1);

    throws_shape(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_squeeze_dyn_neg_axes)
{
    migraphx::shape s1{migraphx::shape::float_type,
                       {{1, 4, 0}, {1, 1, 0}, {3, 3, 0}, {1, 1, 0}, {3, 3, 0}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 4, 0}, {1, 1, 0}, {3, 3, 0}, {3, 3, 0}}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {-2}}}), s1);

    migraphx::shape s3{migraphx::shape::float_type, {{1, 4, 0}, {3, 3, 0}, {3, 3, 0}}};
    expect_shape(s3, migraphx::make_op("squeeze", {{"axes", {-2, -4}}}), s1);
}

TEST_CASE(test_squeeze_transpose)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 4, 1}, {4, 1, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 4}, {4, 1}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_squeeze_multibroadcast)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 1, 4}, {0, 1, 1, 0}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4}, {0, 1, 0}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_squeeze_slice)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 1, 4}, {108, 36, 6, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4}, {108, 36, 1}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_squeeze_negative_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    expect_shape(s2, migraphx::make_op("squeeze", {{"axes", {-2}}}), s1);
}

TEST_CASE(test_squeeze_wrong_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3, 1, 3}};
    throws_shape(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_unsqueeze)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 5, 1, 3}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_dyn)
{
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4, 3}, {2, 5, 0}, {3, 3, 0}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 4, 3}, {2, 5, 0}, {1, 1, 0}, {3, 3, 0}}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);

    migraphx::shape s3{migraphx::shape::float_type,
                       {{1, 4, 3}, {2, 5, 0}, {1, 1, 0}, {3, 3, 0}, {1, 1, 0}}};
    expect_shape(s3, migraphx::make_op("unsqueeze", {{"axes", {2, 4}}}), s1);

    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {2, 4}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_dyn_neg_axes)
{
    migraphx::shape s1{migraphx::shape::float_type, {{1, 4, 3}, {2, 5, 0}, {3, 3, 0}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 4, 3}, {2, 5, 0}, {1, 1, 0}, {3, 3, 0}}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s1);

    migraphx::shape s3{migraphx::shape::float_type,
                       {{1, 4, 3}, {2, 5, 0}, {1, 1, 0}, {3, 3, 0}, {1, 1, 0}}};
    expect_shape(s3, migraphx::make_op("unsqueeze", {{"axes", {-1, -3}}}), s1);
}

TEST_CASE(test_unsqueeze_step)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 12}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 5, 2, 6}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_step_non_divisable)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_step_zero)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 12}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {0}}}), s1);
}

TEST_CASE(test_unsqueeze_step_at_end)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 12}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {3}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_mismatch_step_axis)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 12}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2, 3}}}), s1);
}

TEST_CASE(test_unsqueeze_negative_axis1)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 5, 1, 3}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s1);
}

TEST_CASE(test_unsqueeze_negative_axis2)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 5, 3, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-1}}}), s1);
}

TEST_CASE(test_unsqueeze_negative_axis3)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 5, 3}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-3}}}), s1);
}

TEST_CASE(test_unsqueeze_scalar)
{
    migraphx::shape s1{migraphx::shape::float_type, {1}, {0}};
    migraphx::shape s2{migraphx::shape::float_type, {1}, {1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_unsqueeze_scalar_tensor1)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 3, 3}, {0, 0, 0}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s);
}

TEST_CASE(test_unsqueeze_scalar_tensor2)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 1}, {0, 0, 0}};
    throws_shape(migraphx::make_op("unsqueeze", {{"axes", {-2}}}), s);
}

TEST_CASE(test_unsqueeze_transpose)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 4, 3}, {12, 1, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 4, 1, 3}, {12, 1, 12, 4}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_transpose_step)
{
    migraphx::shape s1{migraphx::shape::float_type, {4, 4, 6}, {24, 1, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 4, 2, 3}, {24, 1, 12, 4}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_multibroadcast)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}, {0, 1, 0}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 1, 4}, {0, 1, 0, 0}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_slice)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}, {108, 36, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 1, 4}, {108, 36, 4, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2}}}), s1);
}

TEST_CASE(test_unsqueeze_axis_zero)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 2, 3, 4}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0}}}), s1);
}

TEST_CASE(test_unsqueeze_axis_last)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {-1}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_1)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 2, 3, 4, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0, -1}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_2)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 1, 2, 3, 4}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_3)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4, 1, 5, 1, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2, 4, 5}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_4)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4, 1, 5, 1, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {5, 4, 2}}}), s1);
}

TEST_CASE(test_unsqueeze_multiple_axes_step)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 4, 10}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 4, 2, 5, 1, 1}};
    expect_shape(s2, migraphx::make_op("unsqueeze", {{"axes", {2, 4, 5}}, {"steps", {2}}}), s1);
}

TEST_CASE(transpose_shape)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 2}};
    migraphx::shape output{migraphx::shape::float_type, {2, 2}, {1, 2}};
    expect_shape(input, migraphx::make_op("transpose", {{"permutation", {0, 1}}}), input);
    expect_shape(output, migraphx::make_op("transpose", {{"permutation", {1, 0}}}), input);
    throws_shape(migraphx::make_op("transpose", {{"permutation", {1, 2}}}), input);
}

TEST_CASE(transpose_dyn_shape0)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4, 0}, {2, 2, 0}}};
    migraphx::shape output{migraphx::shape::float_type, {{2, 2, 0}, {1, 4, 0}}};
    expect_shape(input, migraphx::make_op("transpose", {{"permutation", {0, 1}}}), input);
    expect_shape(output, migraphx::make_op("transpose", {{"permutation", {1, 0}}}), input);
}

TEST_CASE(transpose_dyn_shape1)
{
    migraphx::shape input{migraphx::shape::float_type, {{1, 4, 0}, {4, 4, 0}, {4, 4, 0}}};
    migraphx::shape output{migraphx::shape::float_type, {{4, 4, 0}, {4, 4, 0}, {1, 4, 0}}};
    expect_shape(input, migraphx::make_op("transpose", {{"permutation", {0, 1, 2}}}), input);
    expect_shape(output, migraphx::make_op("transpose", {{"permutation", {2, 1, 0}}}), input);
}

TEST_CASE(transpose_axes_error)
{
    migraphx::shape input{migraphx::shape::float_type, {2, 2}};
    throws_shape(migraphx::make_op("transpose", {{"permutation", {1}}}), input);
}

TEST_CASE(step_test)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 2, 4}};
    {
        migraphx::shape s2{migraphx::shape::float_type, {1, 1, 2}, {8, 8, 3}};
        expect_shape(s2, migraphx::make_op("step", {{"axes", {1, 2}}, {"steps", {2, 3}}}), s1);
    }

    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2, 4}};
        throws_shape(migraphx::make_op("step", {{"axes", {1, 2}}, {"steps", {1}}}), s1);
    }

    {
        migraphx::shape s{migraphx::shape::float_type, {1, 2, 4}};
        throws_shape(migraphx::make_op("step", {{"axes", {2, 3}}, {"steps", {2, 3}}}), s1);
    }
}

TEST_CASE(unary_scalar_input)
{
    migraphx::shape ss{migraphx::shape::half_type};
    expect_shape(ss, migraphx::make_op("sin"), ss);

    migraphx::shape s{migraphx::shape::float_type, {1}};
    expect_shape(s, migraphx::make_op("sin"), s);
}

TEST_CASE(unary_broadcast_input)
{
    migraphx::shape ss{migraphx::shape::half_type, {2, 3}, {1, 0}};
    migraphx::shape s{migraphx::shape::half_type, {2, 3}};
    expect_shape(s, migraphx::make_op("sin"), ss);
}

TEST_CASE(where_broadcast_input)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 2}, {3, 0}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 2}};
    migraphx::shape s3{migraphx::shape::bool_type, {2, 2}};
    expect_shape(s2, migraphx::make_op("where"), s3, s1, s2);
}

TEST_CASE(roialign_test)
{
    migraphx::shape sx{migraphx::shape::float_type, {3, 4, 5, 6}};
    migraphx::shape srois{migraphx::shape::float_type, {2, 4}};
    migraphx::shape sbi{migraphx::shape::int64_type, {2}};
    migraphx::shape sout{migraphx::shape::float_type, {2, 4, 1, 1}};

    expect_shape(sout, migraphx::make_op("roialign"), sx, srois, sbi);

    migraphx::shape sbi1{migraphx::shape::int64_type, {2, 3}};
    throws_shape(migraphx::make_op("roialign"), sx, srois, sbi1);

    migraphx::shape sbi2{migraphx::shape::int64_type, {3}};
    throws_shape(migraphx::make_op("roialign"), sx, srois, sbi2);

    migraphx::shape srois1{migraphx::shape::float_type, {2, 4, 3}};
    throws_shape(migraphx::make_op("roialign"), sx, srois1, sbi);

    migraphx::shape srois2{migraphx::shape::float_type, {2, 3}};
    throws_shape(migraphx::make_op("roialign"), sx, srois2, sbi);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
