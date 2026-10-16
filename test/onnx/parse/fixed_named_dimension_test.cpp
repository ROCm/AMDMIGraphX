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

TEST_CASE(fixed_named_dimension_test)
{
    using migraphx::shape;
    using migraphx::sym::lit;
    using migraphx::sym::var;

    auto sequence_length = var("sequence_length", {1, 8}, {4});

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter(
        "x",
        shape{shape::float_type,
              sym_dims({lit(1), sequence_length})});
    auto output = mm->add_instruction(migraphx::make_op("identity"), x);
    mm->add_return({output});

    migraphx::onnx_options options;
    options.use_symbolic_shapes     = true;
    options.map_dyn_input_dims["x"] = {
        shape::dynamic_dimension{1, 1}, shape::dynamic_dimension{sequence_length}};

    EXPECT(p == read_onnx("fixed_named_dimension_test.onnx", options));
}
