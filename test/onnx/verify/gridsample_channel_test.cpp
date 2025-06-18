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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(gridsample_channel_test)
{
    migraphx::program p = read_onnx("gridsample_channel_test.onnx");
    migraphx::compile_options options;
    p.compile(migraphx::make_target("ref"));

    auto input_type = migraphx::shape::float_type;
    migraphx::shape data_shape{input_type, {1, 3, 4, 4}};
    migraphx::shape grid_shape{input_type, {1, 6, 6, 2}};
    std::vector<float> data = {
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
    };
    std::vector<float> grid = {-1., -1.,  -0.6, -1.,  -0.2, -1.,  0.2, -1.,  0.6, -1.,  1., -1.,
                               -1., -0.6, -0.6, -0.6, -0.2, -0.6, 0.2, -0.6, 0.6, -0.6, 1., -0.6,
                               -1., -0.2, -0.6, -0.2, -0.2, -0.2, 0.2, -0.2, 0.6, -0.2, 1., -0.2,
                               -1., 0.2,  -0.6, 0.2,  -0.2, 0.2,  0.2, 0.2,  0.6, 0.2,  1., 0.2,
                               -1., 0.6,  -0.6, 0.6,  -0.2, 0.6,  0.2, 0.6,  0.6, 0.6,  1., 0.6,
                               -1., 1.,   -0.6, 1.,   -0.2, 1.,   0.2, 1.,   0.6, 1.,   1., 1.};

    migraphx::parameter_map pp;
    pp["x"]    = migraphx::argument(data_shape, data.data());
    pp["grid"] = migraphx::argument(grid_shape, grid.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // clang-format off
    /*
      Generated with the following Python script:
        import torch
        from torch import nn   
        input = torch.tensor([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]]], dtype=torch.float)
        grid = torch.tensor([[[[-1, -1],  [-0.6, -1],  [-0.2, -1],  [0.2, -1],  [0.6, -1],  [1, -1]],
                                [[-1., -0.6], [-0.6, -0.6], [-0.2, -0.6], [0.2, -0.6], [0.6, -0.6], [1., -0.6]],
                                [[-1., -0.2], [-0.6, -0.2], [-0.2, -0.2], [0.2, -0.2], [0.6, -0.2], [1., -0.2]],
                                [[-1., 0.2],  [-0.6, 0.2],  [-0.2, 0.2],  [0.2, 0.2],  [0.6, 0.2],  [1., 0.2]],
                                [[-1., 0.6],  [-0.6, 0.6],  [-0.2, 0.6],  [0.2, 0.6],  [0.6, 0.6],  [1., 0.6]],
                                [[-1., 1.],   [-0.6, 1.],   [-0.2, 1.],   [0.2, 1.],   [0.6, 1.],   [1., 1.]]]], dtype=torch.float)   
        output = nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='border', align_corners=True)
        print(output)
    */
    // clang-format on
    std::vector<float> gold = {0.,  0.6,  1.2,  1.8,  2.4, 3.,   2.4, 3.,   3.6,  4.2,  4.8,  5.4,
                               4.8, 5.4,  6.,   6.6,  7.2, 7.8,  7.2, 7.8,  8.4,  9.,   9.6,  10.2,
                               9.6, 10.2, 10.8, 11.4, 12., 12.6, 12., 12.6, 13.2, 13.8, 14.4, 15.,

                               0.,  0.6,  1.2,  1.8,  2.4, 3.,   2.4, 3.,   3.6,  4.2,  4.8,  5.4,
                               4.8, 5.4,  6.,   6.6,  7.2, 7.8,  7.2, 7.8,  8.4,  9.,   9.6,  10.2,
                               9.6, 10.2, 10.8, 11.4, 12., 12.6, 12., 12.6, 13.2, 13.8, 14.4, 15.,

                               0.,  0.6,  1.2,  1.8,  2.4, 3.,   2.4, 3.,   3.6,  4.2,  4.8,  5.4,
                               4.8, 5.4,  6.,   6.6,  7.2, 7.8,  7.2, 7.8,  8.4,  9.,   9.6,  10.2,
                               9.6, 10.2, 10.8, 11.4, 12., 12.6, 12., 12.6, 13.2, 13.8, 14.4, 15.};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
