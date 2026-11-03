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

#include <limits>

// The dropout node names its unused optional mask output "". That name is not
// a tensor, so the clip node's omitted min input must still parse as missing
// (and be filled with the type's lowest value) rather than resolve to the mask.
TEST_CASE(empty_output_name_test)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto x       = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto max_val = mm->add_parameter("max_val", {migraphx::shape::float_type, {1}});
    mm->add_literal(migraphx::literal{{migraphx::shape::bool_type, {2, 3}}, {1, 1, 1, 1, 1, 1}});
    auto y = mm->add_instruction(migraphx::make_op("identity"), x);
    mm->add_instruction(migraphx::make_op("undefined"));
    auto min_val = mm->add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::float_type, {1}, {0}},
        {std::numeric_limits<float>::lowest()}});
    auto min_mb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), min_val);
    auto max_mb =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), max_val);
    mm->add_instruction(migraphx::make_op("clip"), y, min_mb, max_mb);

    auto prog = optimize_onnx("empty_output_name_test.onnx");
    EXPECT(p == prog);
}
