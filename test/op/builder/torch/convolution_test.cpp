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

// tm::convolution aliases the shared convolution builder (conv + fused channel bias). Note the
// builder's plural attribute names.
TEST_CASE(torch_kit_convolution_op_builder_test)
{
    const auto f                       = migraphx::shape::float_type;
    std::vector<std::size_t> strides   = {1, 1};
    std::vector<std::size_t> paddings  = {0, 0};
    std::vector<std::size_t> dilations = {1, 1};
    migraphx::value options{
        {"strides", strides}, {"paddings", paddings}, {"dilations", dilations}, {"group", 1}};
    migraphx::module mm;
    auto x    = mm.add_parameter("x", {f, {1, 3, 8, 8}});
    auto w    = mm.add_parameter("w", {f, {4, 3, 3, 3}});
    auto bias = mm.add_parameter("bias", {f, {4}});
    migraphx::op::builder::add("convolution", mm, {x, w, bias}, options);

    EXPECT(mm == make_op_module("tm::convolution", options, mm.get_parameters()));
}
