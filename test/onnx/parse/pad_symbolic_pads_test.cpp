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

TEST_CASE(pad_symbolic_pads_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
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
    auto pads = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {-1}}}), table);

    // The fill value, added last so it lands at the front of the module, where the parser puts it
    auto fill =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0}});
    auto fill_x_dims = mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 2}}), x);
    auto pad_rows    = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), pads);
    auto cols = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), fill_x_dims);
    auto fill_dims =
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), pad_rows, cols);
    // The pads resolve to 10 - N rows, so the padded output has the static size 10
    std::vector<migraphx::shape::dynamic_dimension> dims{
        migraphx::shape::dynamic_dimension{lit(10) - n},
        migraphx::shape::dynamic_dimension{lit(4)}};
    auto filled = mm->add_instruction(
        migraphx::make_op("broadcast_with_dims", {{"out_dyn_dims", migraphx::to_value(dims)}}),
        fill,
        fill_dims);
    auto ret = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, filled);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = x_shape.dyn_dims();
    options.use_symbolic_shapes     = true;
    auto prog                       = read_onnx("pad_symbolic_pads_test.onnx", options);
    EXPECT(p == prog);
}
