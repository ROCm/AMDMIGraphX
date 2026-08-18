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

#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <onnx_test.hpp>

// The parser names the symbol after the node: "<module>_<op type>_<instruction number>", where
// the instruction number is the module size when the TopK node is reached (the two parameters).
static migraphx::operation trim_to_k(int64_t axis, migraphx::sym::interval bounds)
{
    return migraphx::make_op(
        "dyn_slice",
        {{"axes", {axis}},
         {"starts", {0}},
         {"ends",
          migraphx::value::array{migraphx::to_value(migraphx::sym::var("main_TopK_2", bounds))}}});
}

// `k` is a runtime input (graph input, not an initializer), so the parser takes the var_k
// path: topk runs over the whole axis, then dyn_slice trims both outputs down to the runtime
// `k`, which the output shape carries as a symbol.
TEST_CASE(topk_var_k_test)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("data", {migraphx::shape::float_type, {2, 4}});
    auto k    = mm->add_parameter("k", {migraphx::shape::int64_type, {1}});
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto out  = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 4}, {"axis", 1}, {"largest", 1}}), data);
    auto val = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    auto ds  = trim_to_k(1, {0, 4});
    val      = mm->add_instruction(ds, val, zero, k);
    ind      = mm->add_instruction(ds, ind, zero, k);
    mm->add_return({val, ind});

    auto prog = read_onnx("topk_var_k_test.onnx");

    EXPECT(p == prog);
}

// Same model with `data` overridden to a symbolic shape. `k` stays a runtime input, so the
// var_k path still fires and sets the topk `k` to the axis dimension's max length.
TEST_CASE(topk_var_k_symbolic_test)
{
    using migraphx::sym::var;
    auto dims = [] { return sym_dims({var("n", {1, 4}), var("m", {2, 4})}); };

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, dims()});
    auto k    = mm->add_parameter("k", {migraphx::shape::int64_type, {1}});
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto out  = mm->add_instruction(
        migraphx::make_op("topk", {{"k", 4}, {"axis", 1}, {"largest", 1}}), data);
    auto val = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    auto ds  = trim_to_k(1, {0, 4});
    val      = mm->add_instruction(ds, val, zero, k);
    ind      = mm->add_instruction(ds, ind, zero, k);
    mm->add_return({val, ind});

    migraphx::onnx_options options;
    options.use_symbolic_shapes        = true;
    options.map_dyn_input_dims["data"] = dims();
    auto prog                          = read_onnx("topk_var_k_test.onnx", options);

    EXPECT(p == prog);
}

// A range-based dynamic shape has no symbol to slice against, so the runtime `k` is rejected.
TEST_CASE(topk_var_k_range_dynamic_error_test)
{
    migraphx::onnx_options options;
    options.map_dyn_input_dims["data"] = {{1, 4}, {2, 4}};

    EXPECT(test::throws([&] { read_onnx("topk_var_k_test.onnx", options); }));
}
