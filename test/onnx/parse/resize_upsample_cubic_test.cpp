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
#include <cmath>

// Cubic kernel (Keys cubic) with coefficient a
static float cubic_kernel(float s, float a = -0.75f)
{
    float abs_s = std::abs(s);
    if(abs_s < 1.0f)
        return (a + 2.0f) * abs_s * abs_s * abs_s - (a + 3.0f) * abs_s * abs_s + 1.0f;
    if(abs_s < 2.0f)
        return a * abs_s * abs_s * abs_s - 5.0f * a * abs_s * abs_s + 8.0f * a * abs_s - 4.0f * a;
    return 0.0f;
}

// Build cubic interpolation matrix for one dimension
static std::vector<float> build_cubic_matrix_1d(std::size_t in_size,
                                                 std::size_t out_size,
                                                 float scale,
                                                 float a = -0.75f)
{
    std::vector<float> matrix(in_size * out_size, 0.0f);

    for(std::size_t o_idx = 0; o_idx < out_size; ++o_idx)
    {
        // half_pixel coordinate transform
        float in_coord = (static_cast<float>(o_idx) + 0.5f) / scale - 0.5f;
        auto base      = static_cast<std::ptrdiff_t>(std::floor(in_coord)) - 1;

        for(std::size_t i = 0; i < 4; ++i)
        {
            auto pos           = base + static_cast<std::ptrdiff_t>(i);
            float t            = in_coord - static_cast<float>(pos);
            float weight       = cubic_kernel(t, a);
            auto clamped_pos   = std::max(std::ptrdiff_t{0},
                                        std::min(pos, static_cast<std::ptrdiff_t>(in_size - 1)));
            auto matrix_idx    = static_cast<std::size_t>(clamped_pos) * out_size + o_idx;
            matrix[matrix_idx] += weight;
        }
    }
    return matrix;
}

// Kronecker product of two matrices
static std::vector<float> kronecker_product(const std::vector<float>& a,
                                             std::size_t a_rows,
                                             std::size_t a_cols,
                                             const std::vector<float>& b,
                                             std::size_t b_rows,
                                             std::size_t b_cols)
{
    auto result_rows = a_rows * b_rows;
    auto result_cols = a_cols * b_cols;
    std::vector<float> result(result_rows * result_cols, 0.0f);

    for(std::size_t ai = 0; ai < a_rows; ++ai)
    {
        for(std::size_t aj = 0; aj < a_cols; ++aj)
        {
            float a_val = a[ai * a_cols + aj];
            if(std::abs(a_val) < 1e-12f)
                continue;

            for(std::size_t bi = 0; bi < b_rows; ++bi)
            {
                for(std::size_t bj = 0; bj < b_cols; ++bj)
                {
                    float b_val         = b[bi * b_cols + bj];
                    auto result_row     = ai * b_rows + bi;
                    auto result_col     = aj * b_cols + bj;
                    result[result_row * result_cols + result_col] = a_val * b_val;
                }
            }
        }
    }
    return result;
}

TEST_CASE(resize_upsample_cubic_test)
{
    // Test cubic upsample: [1, 1, 2, 2] -> [1, 1, 4, 4]
    // scales: [1.0, 1.0, 2.0, 2.0], mode: cubic, coordinate_transformation_mode: half_pixel
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Add scales literal (not used but present in ONNX)
    std::vector<float> ds = {1.0f, 1.0f, 2.0f, 2.0f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_literal(migraphx::literal{ss, ds});

    // Input parameter
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto inx = mm->add_parameter("X", sx);

    // Build interpolation matrix for spatial dimensions (H and W)
    // H: 2 -> 4, W: 2 -> 4
    auto h_matrix = build_cubic_matrix_1d(2, 4, 2.0f);
    auto w_matrix = build_cubic_matrix_1d(2, 4, 2.0f);

    // Kronecker product: [2, 4] ⊗ [2, 4] = [4, 16]
    auto interp_matrix = kronecker_product(h_matrix, 2, 4, w_matrix, 2, 4);

    migraphx::shape s_interp{migraphx::shape::float_type, {4, 16}};
    auto l_interp = mm->add_literal(migraphx::literal{s_interp, interp_matrix});

    // Reshape input: [1, 1, 2, 2] -> [1, 4]
    auto rsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4}}}), inx);

    // Dot product: [1, 4] @ [4, 16] = [1, 16]
    auto dot_out = mm->add_instruction(migraphx::make_op("dot"), rsp, l_interp);

    // Reshape output: [1, 16] -> [1, 1, 4, 4]
    auto r = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 4, 4}}}), dot_out);
    mm->add_return({r});

    auto prog = read_onnx("resize_upsample_cubic_test.onnx");

    EXPECT(p == prog);
}
