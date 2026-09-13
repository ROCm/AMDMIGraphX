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

// A dynamic input pads the indices for the 4x2 maximum, so the trim is bounded by 8 rather than
// by the 4 elements the model itself declares.
TEST_CASE(nonzero_dyn_input_test)
{
    using migraphx::sym::var;
    migraphx::shape s{migraphx::shape::bool_type, {{1, 4}, {2, 2}}};
    EXPECT(check_parse("nonzero_dynamic_test.onnx", {{"data", s}}, [](auto& m, const auto& args) {
        auto nz      = m.add_instruction(migraphx::make_op("nonzero"), args[0]);
        auto indices = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nz);
        auto num_nonzero =
            m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nz);
        auto starts = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
        auto ends   = migraphx::value::array{migraphx::to_value(var("NonZero_1", {0, 8}))};
        auto r      = m.add_instruction(
            migraphx::make_op("dyn_slice", {{"axes", {1}}, {"starts", {0}}, {"ends", ends}}),
            indices,
            starts,
            num_nonzero);
        m.add_return({r});
    }));
}
