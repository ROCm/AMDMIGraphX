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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"
#include <read_onnx.hpp>

TEST_CASE(create_symbolic_dynamic_dimension)
{
    migraphx::dynamic_dimension sym("n", {{"n", migraphx::dynamic_dimension{1, 4}}});
    EXPECT(sym.is_symbolic());
    EXPECT(not sym.is_fixed());

    migraphx::dynamic_dimension sym_opt(
        "n", {{"n", migraphx::dynamic_dimension{1, 4, migraphx::optimals{1, 2, 4}}}});
    EXPECT(sym_opt.is_symbolic());

    migraphx::dynamic_dimension range{1, 4};
    EXPECT(not range.is_symbolic());
    EXPECT(range != sym);
}

TEST_CASE(symbolic_expression_compose)
{
    migraphx::dynamic_dimension product("n * 3", {{"n", migraphx::dynamic_dimension{1, 8}}});
    EXPECT(product.is_symbolic());

    migraphx::dynamic_dimension parsed("n + 1", {{"n", migraphx::dynamic_dimension{1, 8}}});
    EXPECT(parsed.is_symbolic());
}

TEST_CASE(create_symbolic_dynamic_shape)
{
    migraphx::dynamic_dimensions dyn_dims(
        migraphx::dynamic_dimension{"n", {{"n", migraphx::dynamic_dimension{1, 4}}}},
        migraphx::dynamic_dimension{3, 3});
    migraphx::shape s{migraphx_shape_float_type, dyn_dims};
    EXPECT(s.dynamic());
    EXPECT(s.dyn_dims()[0].is_symbolic());
    EXPECT(not s.dyn_dims()[1].is_symbolic());
}

TEST_CASE(parse_onnx_symbolic_dyn_input)
{
    migraphx::onnx_options options;
    migraphx::dynamic_dimensions dyn_dims(
        migraphx::dynamic_dimension{"n", {{"n", migraphx::dynamic_dimension{1, 8}}}},
        migraphx::dynamic_dimension{"m", {{"m", migraphx::dynamic_dimension{2, 16}}}});
    options.set_dyn_input_parameter_shape("0", dyn_dims);

    auto p      = read_onnx("dim_param_test.onnx", options);
    auto shapes = p.get_parameter_shapes();
    auto input  = shapes["0"];
    EXPECT(input.dynamic());
    auto dd = input.dyn_dims();
    EXPECT(dd[0].is_symbolic());
    EXPECT(dd[1].is_symbolic());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
