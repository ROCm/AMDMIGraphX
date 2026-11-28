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

#include <migraphx/sym.hpp>
#include <onnx_test.hpp>

// Run-time starts and ends over a static input become symbols bounded by their axis length.
TEST_CASE(slice_var_input_static0)
{
    using migraphx::sym::var;
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto data   = mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {3, 2}});
    auto starts = mm->add_parameter("starts", migraphx::shape{migraphx::shape::int32_type, {2}});
    auto ends   = mm->add_parameter("ends", migraphx::shape{migraphx::shape::int32_type, {2}});
    std::vector<migraphx::sym::expr> start_syms = {var("main_Slice_3_start0", {0, 3}),
                                                   var("main_Slice_3_start1", {0, 2})};
    std::vector<migraphx::sym::expr> end_syms   = {var("main_Slice_3_end0", {0, 3}),
                                                   var("main_Slice_3_end1", {0, 2})};
    mm->add_instruction(migraphx::make_op("dyn_slice",
                                          {{"axes", {0, 1}},
                                           {"starts", migraphx::to_value(start_syms)},
                                           {"ends", migraphx::to_value(end_syms)}}),
                        data,
                        starts,
                        ends);
    auto prog = optimize_onnx("slice_var_input_static0.onnx");

    EXPECT(p == prog);
}
