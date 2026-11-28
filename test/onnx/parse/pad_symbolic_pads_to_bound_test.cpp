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

TEST_CASE(pad_symbolic_pads_to_bound_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    // N is at most 10 and the pads fill the rows up to 10, the bound of N: fixed_pad
    const auto n = var("N", {1, 10});
    const migraphx::shape x_shape{migraphx::shape::float_type, sym_dims({n, lit(4)})};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {1}});
    auto size  = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {10}});
    auto begin = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {2, 1}}, {0, 0}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {-1}});
    auto x = mm->add_parameter("x", x_shape);

    auto x_dims = mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 2}}), x);
    auto rows   = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), x_dims);
    auto row_pad = mm->add_instruction(migraphx::make_op("sub"), size, rows);
    auto end     = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), row_pad, zero);
    end          = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), end);
    auto table   = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), begin, end);
    table = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), table);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", {-1}}}), table);
    auto ret = mm->add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), x);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = x_shape.dyn_dims();
    options.use_symbolic_shapes     = true;
    auto prog                       = read_onnx("pad_symbolic_pads_test.onnx", options);
    EXPECT(p == prog);
    EXPECT(prog.get_main_module()->get_returns().front()->get_shape() ==
           migraphx::shape{migraphx::shape::float_type, {10, 4}});
}
