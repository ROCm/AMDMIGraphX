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

TEST_CASE(symbolic_reshape_constant)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto batch = var("batch", {1, 4});
    const auto input_shape =
        migraphx::shape{migraphx::shape::float_type, sym_dims({batch, lit(int64_t{4})})};
    const migraphx::shape other_shape{migraphx::shape::float_type, {2}};

    EXPECT(check_parse(
        "symbolic_reshape_constant_test.onnx",
        {{"other", other_shape}, {"x", input_shape}},
        [&](auto& m, const auto& args) {
            m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2}},
                                            {int64_t{0}, int64_t{4}}});
            const auto expressions =
                migraphx::to_value(std::vector<migraphx::sym::expr>{batch, lit(4)});
            const auto output_shape =
                migraphx::shape{migraphx::shape::float_type, sym_dims({batch, lit(4)})};
            auto constant_dims = m.add_instruction(
                migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}),
                std::vector<migraphx::instruction_ref>{args[1], args[0]});
            auto constant_allocation = m.add_instruction(
                migraphx::make_op("allocate", {{"shape", migraphx::to_value(output_shape)}}),
                constant_dims);
            auto constant_output =
                m.add_instruction(migraphx::make_op("reshape"), args[1], constant_allocation);

            auto attribute_dims = m.add_instruction(
                migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}),
                std::vector<migraphx::instruction_ref>{args[1], args[0]});
            auto attribute_allocation = m.add_instruction(
                migraphx::make_op("allocate", {{"shape", migraphx::to_value(output_shape)}}),
                attribute_dims);
            auto attribute_output =
                m.add_instruction(migraphx::make_op("reshape"), args[1], attribute_allocation);
            m.add_return({constant_output, attribute_output});
        }));
}
