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

TEST_CASE(symbolic_reshape_cross_root_test)
{
    using dd = migraphx::shape::dynamic_dimension;
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto batch = var("batch", {1, 4});
    const migraphx::shape input_shape{migraphx::shape::float_type, {dd{batch}, dd{lit(4)}}};

    EXPECT(check_parse(
        "symbolic_reshape_cross_root_test.onnx",
        {{"x", input_shape}},
        [&](migraphx::module& m, const std::vector<migraphx::instruction_ref>& args) {
            auto x         = args.front();
            auto flattened = m.add_instruction(migraphx::make_op("flatten", {{"axis", 0}}), x);
            m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), x);
            const auto expressions =
                migraphx::to_value(std::vector<migraphx::sym::expr>{batch, lit(4)});
            auto target_dims = m.add_instruction(
                migraphx::make_op("eval_expr_from_shape", {{"expressions", expressions}}),
                flattened,
                x);
            auto allocation = m.add_instruction(
                migraphx::make_op("allocate", {{"shape", migraphx::to_value(input_shape)}}),
                target_dims);
            m.add_return({m.add_instruction(migraphx::make_op("reshape"), flattened, allocation)});
        }));
}
