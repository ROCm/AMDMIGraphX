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

#include <cstddef>
#include <vector>
#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>

// tm::conv_transpose runs convolution_backwards unpadded, crops off the symmetric
// padding while keeping the output_padding elements, then adds the channel bias.
TEST_CASE(torch_kit_conv_transpose_op_builder_test)
{
    const auto f                           = migraphx::shape::float_type;
    std::vector<std::size_t> stride         = {2, 2};
    std::vector<std::size_t> padding        = {1, 1};
    std::vector<std::size_t> dilation       = {1, 1};
    std::vector<std::size_t> output_padding = {1, 1};
    migraphx::value options{{"stride", stride},
                            {"padding", padding},
                            {"dilation", dilation},
                            {"output_padding", output_padding},
                            {"group", 1}};

    migraphx::module mm;
    auto x    = mm.add_parameter("x", {f, {1, 3, 4, 4}});
    auto w    = mm.add_parameter("w", {f, {3, 4, 3, 3}});
    auto bias = mm.add_parameter("bias", {f, {4}});
    auto out  = mm.add_instruction(
        migraphx::make_op(
            "convolution_backwards",
            {{"stride", stride}, {"padding", {0, 0}}, {"dilation", dilation}, {"group", 1}}),
        x,
        w);
    auto cropped = mm.add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {9, 9}}}), out);
    auto b = mm.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 4, 8, 8}}}), bias);
    mm.add_instruction(migraphx::make_op("add"), cropped, b);

    EXPECT(mm == make_op_module("tm::conv_transpose", options, mm.get_parameters()));
}

// tm::conv_transpose with no output_padding passes padding straight to the op.
TEST_CASE(torch_kit_conv_transpose_no_crop_op_builder_test)
{
    const auto f                           = migraphx::shape::float_type;
    std::vector<std::size_t> stride         = {1, 1};
    std::vector<std::size_t> padding        = {1, 1};
    std::vector<std::size_t> dilation       = {1, 1};
    std::vector<std::size_t> output_padding = {0, 0};
    migraphx::value options{{"stride", stride},
                            {"padding", padding},
                            {"dilation", dilation},
                            {"output_padding", output_padding},
                            {"group", 1}};

    migraphx::module mm;
    auto x = mm.add_parameter("x", {f, {1, 3, 4, 4}});
    auto w = mm.add_parameter("w", {f, {3, 4, 3, 3}});
    mm.add_instruction(
        migraphx::make_op(
            "convolution_backwards",
            {{"stride", stride}, {"padding", padding}, {"dilation", dilation}, {"group", 1}}),
        x,
        w);

    EXPECT(mm == make_op_module("tm::conv_transpose", options, mm.get_parameters()));
}
