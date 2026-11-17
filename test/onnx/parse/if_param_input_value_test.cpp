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

#include <onnx_test.hpp>

// The branches return {2, 3} and {2, 4}, so the If only parses when its condition has a value
static migraphx::program read_with_cond(double cond)
{
    migraphx::onnx_options options;
    options.map_input_values["cond"] = {cond};
    return read_onnx("if_param_excp_test.onnx", options);
}

static bool has_op(const migraphx::program& p, const std::string& name)
{
    const auto* mm = p.get_main_module();
    return std::any_of(mm->begin(), mm->end(), [&](const auto& ins) { return ins.name() == name; });
}

TEST_CASE(if_param_input_value_then_test)
{
    auto p = read_with_cond(1);
    EXPECT(not has_op(p, "if"));
    EXPECT(has_op(p, "add"));
    EXPECT(not has_op(p, "mul"));
    EXPECT(not migraphx::contains(p.get_parameter_names(), "cond"));
    EXPECT(p.get_output_shapes().front() == migraphx::shape{migraphx::shape::float_type, {2, 3}});
}

TEST_CASE(if_param_input_value_else_test)
{
    auto p = read_with_cond(0);
    EXPECT(not has_op(p, "if"));
    EXPECT(has_op(p, "mul"));
    EXPECT(not has_op(p, "add"));
    EXPECT(p.get_output_shapes().front() == migraphx::shape{migraphx::shape::float_type, {2, 4}});
}

TEST_CASE(if_param_input_value_count_error_test)
{
    migraphx::onnx_options options;
    options.map_input_values["cond"] = {1, 0};
    EXPECT(test::throws([&] { read_onnx("if_param_excp_test.onnx", options); }));
}
