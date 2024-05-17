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

TEST_CASE(slice_var_input_dyn0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto data =
        mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {{3, 8}, {2, 2}}});
    auto starts = mm->add_parameter("starts", migraphx::shape{migraphx::shape::int32_type, {2}});
    auto ends   = mm->add_parameter("ends", migraphx::shape{migraphx::shape::int32_type, {2}});
    auto ret =
        mm->add_instruction(migraphx::make_op("slice", {{"axes", {0, 1}}}), data, starts, ends);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {3, 8};
    auto prog                     = read_onnx("slice_var_input_dyn0.onnx", options);
    EXPECT(p == prog);
}
