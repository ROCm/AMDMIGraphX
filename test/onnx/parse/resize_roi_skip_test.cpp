/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(resize_roi_skip_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sroi{migraphx::shape::float_type, {8}};
    std::vector<float> roid = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8};
    mm->add_literal(migraphx::literal(sroi, roid));

    migraphx::shape sscale{migraphx::shape::float_type, {4}};
    std::vector<float> scaled = {1, 1, 2, 2};
    auto scales = mm->add_literal(migraphx::literal(sscale, scaled));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto inx = mm->add_parameter("X", sx);

    auto r = mm->add_instruction(
        migraphx::make_op("resize",
                          {{"nearest_mode", "round_prefer_floor"},
                           {"coordinate_transformation_mode", "half_pixel"}}),
        inx,
        scales);
    mm->add_return({r});

    auto prog = read_onnx("resize_roi_skip_test.onnx");

    EXPECT(p == prog);
}
