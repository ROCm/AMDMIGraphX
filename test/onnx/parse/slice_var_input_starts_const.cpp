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

TEST_CASE(slice_var_input_starts_const)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto data   = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 2}});
    auto ends   = mm->add_parameter("ends", migraphx::shape{migraphx::shape::int64_type, {2}});
    auto starts = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {2}}, {0, 0}});
    auto op     = migraphx::make_op(
        "dyn_slice",
        {{"axes", {0, 1}},
             {"starts",
              migraphx::value::array{migraphx::to_value(migraphx::sym::lit(0)),
                                 migraphx::to_value(migraphx::sym::lit(0))}},
             {"ends",
              migraphx::value::array{
              migraphx::to_value(migraphx::sym::var("main_Slice_3_end_0", {0, 3})),
              migraphx::to_value(migraphx::sym::var("main_Slice_3_end_1", {0, 2}))}}});
    mm->add_instruction(op, data, starts, ends);
    auto prog = optimize_onnx("slice_var_input_starts_const.onnx");

    EXPECT(p == prog);
}
