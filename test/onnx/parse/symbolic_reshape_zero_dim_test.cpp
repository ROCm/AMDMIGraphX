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

// Reshape with a runtime shape input of [0, 4]: the zero copies input dim 0, giving {batch, 4}.
TEST_CASE(symbolic_reshape_zero_dim_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto batch = var("batch", {1, 4});
    const migraphx::shape input_shape{migraphx::shape::float_type,
                                      sym_dims({batch, lit(int64_t{4})})};

    EXPECT(check_parse(
        "symbolic_reshape_zero_dim_test.onnx",
        {{"x", input_shape}},
        [&](migraphx::module& m, const std::vector<migraphx::instruction_ref>& args) {
            auto x = args.front();
            auto zero =
                m.insert_literal(x, migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
            auto width = m.add_instruction(
                migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 2}}), x);
            m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), zero, width);

            const auto expressions =
                migraphx::to_value(std::vector<migraphx::sym::expr>{batch, lit(4)});
            auto target_dims = m.add_instruction(
                migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}), x);
            auto allocation = m.add_instruction(
                migraphx::make_op("allocate", {{"shape", migraphx::to_value(input_shape)}}),
                target_dims);
            m.add_return({m.add_instruction(migraphx::make_op("reshape"), x, allocation)});
        }));
}
