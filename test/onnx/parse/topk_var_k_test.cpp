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

// `k` is a runtime input (graph input, not an initializer), so the parser takes the var_k
// path: topk runs with k set to the axis dimension, then the outputs are sliced down to the
// runtime `k`.
TEST_CASE(topk_var_k_test)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("data", {migraphx::shape::float_type, {2, 4}});
    auto k    = mm->add_parameter("k", {migraphx::shape::int64_type, {1}});
    auto out  = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 4}, {"axis", 1}, {"largest", 1}}), data);
    auto val = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    val      = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0}}, {"axes", {1}}, {"mode", migraphx::value::array{"ends"}}}),
        val,
        k);
    ind = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0}}, {"axes", {1}}, {"mode", migraphx::value::array{"ends"}}}),
        ind,
        k);
    mm->add_return({val, ind});

    auto prog = read_onnx("topk_var_k_test.onnx");

    EXPECT(p == prog);
}

// Same model, but `data` is overridden to a dynamic shape. `k` stays a runtime input, so the
// var_k path still fires and sets the topk `k` to the axis dimension's max length.
TEST_CASE(topk_var_k_dynamic_test)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("data", {migraphx::shape::float_type, {{1, 4}, {2, 4}}});
    auto k    = mm->add_parameter("k", {migraphx::shape::int64_type, {1}});
    auto out  = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 4}, {"axis", 1}, {"largest", 1}}), data);
    auto val = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    val      = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0}}, {"axes", {1}}, {"mode", migraphx::value::array{"ends"}}}),
        val,
        k);
    ind = mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {0}}, {"axes", {1}}, {"mode", migraphx::value::array{"ends"}}}),
        ind,
        k);
    mm->add_return({val, ind});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["data"] = {{1, 4}, {2, 4}};
    auto prog                          = read_onnx("topk_var_k_test.onnx", options);

    EXPECT(p == prog);
}
