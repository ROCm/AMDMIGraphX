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

#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

static std::vector<float> run_ref_gridsample(const migraphx::shape& xs,
                                             const std::vector<float>& x_data,
                                             const migraphx::shape& gs,
                                             const std::vector<float>& grid_data,
                                             const std::string& mode,
                                             const std::string& padding_mode,
                                             bool align_corners)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x    = mm->add_literal(migraphx::literal{xs, x_data});
    auto grid = mm->add_literal(migraphx::literal{gs, grid_data});

    mm->add_instruction(
        migraphx::make_op(
            "gridsample",
            {{"mode", mode}, {"padding_mode", padding_mode}, {"align_corners", align_corners}}),
        x,
        grid);

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    return results_vector;
}

TEST_CASE(gridsample_nearest_align_corners_corners)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    std::vector<float> x_data{1, 2, 3, 4};
    std::vector<float> grid_data{-1, -1, 1, -1, -1, 1, 1, 1};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "nearest", "zeros", true);

    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_multi_channel)
{

    migraphx::shape xs{migraphx::shape::float_type, {1, 2, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    std::vector<float> x_data{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> grid_data{-1, -1, 1, -1, -1, 1, 1, 1};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "nearest", "zeros", true);


    std::vector<float> gold{1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_multi_batch)
{

    migraphx::shape xs{migraphx::shape::float_type, {2, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {2, 2, 2, 2}};

    std::vector<float> x_data{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> grid_data{-1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "nearest", "zeros", true);

    std::vector<float> gold{1, 2, 3, 4, 8, 7, 6, 5};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_padding_zeros)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    std::vector<float> x_data{1, 2, 3, 4};
    std::vector<float> grid_data{-3, -3, 3, -3, -3, 3, 3, 3};
    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "nearest", "zeros", true);
    std::vector<float> gold{0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_padding_border)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    std::vector<float> x_data{1, 2, 3, 4};
    std::vector<float> grid_data{-3, -3, 3, -3, -3, 3, 3, 3};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "nearest", "border", true);


    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_padding_reflection)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 4, 4}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    std::vector<float> x_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> grid_data{-1.5, -1.5, 1.5, -1.5, -1.5, 1.5, 1.5, 1.5};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "nearest", "reflection", true);

    std::vector<float> gold{6, 7, 10, 11};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_align_corners_false)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    std::vector<float> x_data{1, 2, 3, 4};
    std::vector<float> grid_data{-1, -1, 1, -1, -1, 1, 1, 1};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "nearest", "zeros", false);

    std::vector<float> gold{1, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_linear_midpoint)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 1, 1, 2}};

    std::vector<float> x_data{1, 2, 3, 4};
    std::vector<float> grid_data{0, 0};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "linear", "zeros", true);

    std::vector<float> gold{2.5};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_cubic_midpoint)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 4, 4}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 1, 1, 2}};

    std::vector<float> x_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> grid_data{0, 0};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "cubic", "zeros", true);

    std::vector<float> gold{8.5};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_singleton_dim_reflection)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 1, 1}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    std::vector<float> x_data{7};
    std::vector<float> grid_data{-1, -1, 1, -1, -1, 1, 1, 1};

    auto results = run_ref_gridsample(xs, x_data, gs, grid_data, "nearest", "reflection", true);

    // There is only one pixel to reach, so all four taps return it.
    std::vector<float> gold{7, 7, 7, 7};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}
