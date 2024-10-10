/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(roialign_default_test)
{
    migraphx::shape sx{migraphx::shape::float_type, {10, 4, 7, 8}};
    migraphx::shape srois{migraphx::shape::float_type, {8, 4}};
    migraphx::shape sbi{migraphx::shape::int64_type, {8}};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("x", sx);
    auto rois = mm->add_parameter("rois", srois);
    auto bi   = mm->add_parameter("batch_ind", sbi);

    // Depending on whether the model was built for Onnx opset 16 or earlier, the default
    // coordinate_transformation_mode is different.  These model files had explicit opset given
    // when they were created.
    auto r = mm->add_instruction(
        migraphx::make_op("roialign", {{"coordinate_transformation_mode", "half_pixel"}}),
        x,
        rois,
        bi);
    mm->add_return({r});
    auto prog = read_onnx("roialign_default_test.onnx");
    EXPECT(p == prog);

    // Opset 12 program
    migraphx::program p_12;
    auto* mm_12  = p_12.get_main_module();
    auto x_12    = mm_12->add_parameter("x", sx);
    auto rois_12 = mm_12->add_parameter("rois", srois);
    auto bi_12   = mm_12->add_parameter("batch_ind", sbi);

    auto r_12 = mm_12->add_instruction(
        migraphx::make_op("roialign", {{"coordinate_transformation_mode", "output_half_pixel"}}),
        x_12,
        rois_12,
        bi_12);
    mm_12->add_return({r_12});
    auto prog_12 = read_onnx("roialign_default_test_12.onnx");
    EXPECT(p_12 == prog_12);
}
