/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <onnx_test.hpp>

TEST_CASE(reducemax_dyn_test)
{
    // input shape with 4 dynamic dimensions
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "x", migraphx::shape{migraphx::shape::float_type, {{3, 5}, {4, 4}, {5, 5}, {6, 6}}});
    auto r0 = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), l0);
    auto r1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), r0);
    mm->add_return({r1});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = {{3, 5}, {4, 4}, {5, 5}, {6, 6}};
    auto prog                       = read_onnx("reducemax_dyn_test.onnx", options);

    EXPECT(p == prog);
}
