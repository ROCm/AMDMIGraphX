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

TEST_CASE(reducel1_dyn_test)
{
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // a shape with 4 dynamic dimensions
        auto l0      = mm->add_parameter("x",
                                    migraphx::shape{migraphx::shape::float_type,
                                                         {{3, 3}, {3, 5}, {4, 6, {5}}, {5, 7, {6}}}});
        auto abs_ins = mm->add_instruction(migraphx::make_op("abs"), l0);
        auto sum_ins =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {-2}}}), abs_ins);
        auto sq_ins = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {-2}}}), sum_ins);
        mm->add_return({sq_ins});

        migraphx::onnx_options options;
        options.map_dyn_input_dims["x"] = {{3, 3}, {3, 5}, {4, 6, {5}}, {5, 7, {6}}};
        auto prog                       = read_onnx("reducel1_dyn_test.onnx", options);

        EXPECT(p == prog);
    }

    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // No axes given in the onnx file.  Parser should default to all axes.
        auto l0      = mm->add_parameter("x",
                                    migraphx::shape{migraphx::shape::float_type,
                                                         {{3, 3}, {3, 5}, {4, 6, {5}}, {5, 7, {6}}}});
        auto abs_ins = mm->add_instruction(migraphx::make_op("abs"), l0);
        auto sum_ins =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1, 2, 3}}}), abs_ins);
        auto sq_ins =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1, 2, 3}}}), sum_ins);
        mm->add_return({sq_ins});

        migraphx::onnx_options options;
        options.map_dyn_input_dims["x"] = {{3, 3}, {3, 5}, {4, 6, {5}}, {5, 7, {6}}};
        auto prog                       = read_onnx("reducel1_dyn_noaxes_test.onnx", options);

        EXPECT(p == prog);
    }
}
