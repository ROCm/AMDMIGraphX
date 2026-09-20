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

// Direct GPU tests for the gridsample JIT kernel, following the shape of
// test/gpu/nonmaxsuppression.cpp: build the program, compile for the gpu
// target, copy parameters over, run, copy results back.
//
// This complements test/verify/test_gridsample.cpp. The verify tests assert
// that GPU and ref agree; these assert absolute values, so a fault common to
// both implementations is still caught here.

static std::vector<float> run_gpu_gridsample(migraphx::program p,
                                             const migraphx::parameter_map& host_params = {})
{
    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);

    migraphx::parameter_map gpu_params;
    for(auto&& x : p.get_parameter_shapes())
    {
        auto it = host_params.find(x.first);
        if(it != host_params.end())
            gpu_params[x.first] = t.copy_to(it->second);
        else
            gpu_params[x.first] = t.allocate(x.second);
    }

    auto result = t.copy_from(p.eval(gpu_params).back());

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    return results_vector;
}

// Large shapes cannot have a hand-written gold, so gridsample_gpu_large
// compares the device result against the reference operator instead.
static std::vector<float> run_ref_program(migraphx::program p)
{
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();

    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    return results_vector;
}

TEST_CASE(gridsample_gpu_nearest_align_corners_corners)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    auto x    = mm->add_literal(migraphx::literal{xs, {1, 2, 3, 4}});
    auto grid = mm->add_literal(migraphx::literal{gs, {-1, -1, 1, -1, -1, 1, 1, 1}});

    mm->add_instruction(
        migraphx::make_op(
            "gridsample",
            {{"mode", "nearest"}, {"padding_mode", "zeros"}, {"align_corners", true}}),
        x,
        grid);

    auto results = run_gpu_gridsample(p);

    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_gpu_linear)
{
    // Checks: the linear kernel instantiation. Mode is a template parameter,
    // so each mode compiles a separate device kernel -- covering nearest says
    // nothing about this branch. Same setup as gridsample_linear_midpoint in
    // test/ref, so both files must agree on the answer; a difference here is
    // device-specific.
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 1, 1, 2}};

    auto x    = mm->add_literal(migraphx::literal{xs, {1, 2, 3, 4}});
    auto grid = mm->add_literal(migraphx::literal{gs, {0, 0}});

    mm->add_instruction(
        migraphx::make_op("gridsample",
                          {{"mode", "linear"}, {"padding_mode", "zeros"}, {"align_corners", true}}),
        x,
        grid);

    auto results = run_gpu_gridsample(p);

    // The grid point sits at the image centre: 0 unnormalizes to 0.5 on both
    // axes, so each pixel carries a weight of 0.25 -- (1 + 2 + 3 + 4) / 4.
    std::vector<float> gold{2.5};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_gpu_cubic)
{
    // Checks: the cubic kernel instantiation and its 4x4 tap gather.
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 4, 4}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 1, 1, 2}};

    auto x = mm->add_literal(
        migraphx::literal{xs, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}});
    auto grid = mm->add_literal(migraphx::literal{gs, {0, 0}});

    mm->add_instruction(
        migraphx::make_op("gridsample",
                          {{"mode", "cubic"}, {"padding_mode", "zeros"}, {"align_corners", true}}),
        x,
        grid);

    auto results = run_gpu_gridsample(p);

    // x[row][col] = 4 * row + col + 1 is linear in both axes and cubic
    // reproduces a linear ramp exactly, so the sixteen weights need not be
    // summed by hand: 0 unnormalizes to 1.5, giving 4 * 1.5 + 1.5 + 1.
    std::vector<float> gold{8.5};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_gpu_padding_border)
{
    // Checks: border padding on device. PaddingMode is a template parameter,
    // so each padding mode is a distinct kernel.
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 2, 2}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    auto x    = mm->add_literal(migraphx::literal{xs, {1, 2, 3, 4}});
    auto grid = mm->add_literal(migraphx::literal{gs, {-3, -3, 3, -3, -3, 3, 3, 3}});

    mm->add_instruction(
        migraphx::make_op(
            "gridsample",
            {{"mode", "nearest"}, {"padding_mode", "border"}, {"align_corners", true}}),
        x,
        grid);

    auto results = run_gpu_gridsample(p);

    // -3 unnormalizes to -1.0 and 3 to 2.0; each axis clamps independently
    // into [0, 1], landing back on the four corners.
    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_gpu_padding_reflection)
{
    // Checks: reflection padding on device, including the zero-width span
    // guard in gridsample_reflect().
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 4, 4}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    auto x = mm->add_literal(
        migraphx::literal{xs, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}});
    auto grid =
        mm->add_literal(migraphx::literal{gs, {-1.5, -1.5, 1.5, -1.5, -1.5, 1.5, 1.5, 1.5}});

    mm->add_instruction(
        migraphx::make_op(
            "gridsample",
            {{"mode", "nearest"}, {"padding_mode", "reflection"}, {"align_corners", true}}),
        x,
        grid);

    auto results = run_gpu_gridsample(p);

    // -0.75 mirrors about 0 to +0.75 -> pixel 1; 3.75 overshoots the far edge
    // by 0.75 and bounces back to 2.25 -> pixel 2. Wrapping would give 0.
    std::vector<float> gold{6, 7, 10, 11};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_gpu_half)
{
    // Checks: half tensors, across all three modes. This type exposed a kernel
    // bug float could not -- both arms of a ternary wrapped in
    // implicit_conversion only collapse to one type when the tensor type is
    // float, so nearest failed to compile at all while float passed.
    auto make = [](const std::string& mode) {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape xs{migraphx::shape::half_type, {1, 1, 2, 2}};
        migraphx::shape gs{migraphx::shape::half_type, {1, 2, 2, 2}};

        auto x    = mm->add_literal(migraphx::literal{xs, {1, 2, 3, 4}});
        auto grid = mm->add_literal(migraphx::literal{gs, {-1, -1, 1, -1, -1, 1, 1, 1}});

        mm->add_instruction(
            migraphx::make_op("gridsample",
                              {{"mode", mode}, {"padding_mode", "zeros"}, {"align_corners", true}}),
            x,
            grid);
        return p;
    };

    // The grid hits the four pixel centres exactly, so interpolation collapses
    // to a straight pick and every mode must return the input unchanged.
    std::vector<float> gold{1, 2, 3, 4};
    for(const auto& mode : {"nearest", "linear", "cubic"})
    {
        auto results = run_gpu_gridsample(make(mode));
        EXPECT(migraphx::verify::verify_rms_range(results, gold));
    }
}

TEST_CASE(gridsample_gpu_nonstandard_input)
{
    // Checks: the kernel reads a permuted-stride view directly. The operator
    // dropped require_std_shape, so eliminate_contiguous removes the copy that
    // attribute used to pin in place and the kernel sees the transposed view.
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape xs{migraphx::shape::float_type, {1, 2, 2, 1}};
    migraphx::shape gs{migraphx::shape::float_type, {1, 2, 2, 2}};

    auto x = mm->add_literal(migraphx::literal{xs, {1, 2, 3, 4}});
    auto xt =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), x);
    auto grid = mm->add_literal(migraphx::literal{gs, {-1, -1, 1, -1, -1, 1, 1, 1}});

    mm->add_instruction(
        migraphx::make_op(
            "gridsample",
            {{"mode", "nearest"}, {"padding_mode", "zeros"}, {"align_corners", true}}),
        xt,
        grid);

    auto results = run_gpu_gridsample(p);

    // The transpose rearranges {1,2,3,4} back into [[1,2],[3,4]], so sampling
    // the four corners returns them in order -- same answer as the standard
    // layout, which is the point: only the strides differ.
    std::vector<float> gold{1, 2, 3, 4};
    EXPECT(migraphx::verify::verify_rms_range(results, gold));
}

TEST_CASE(gridsample_gpu_large)
{
    // Checks: a shape large enough to span several blocks, exercising
    // index.global_stride() rather than a single-block launch. 6144 outputs is
    // far too many for a hand-written gold, so this one compares against the
    // reference operator -- a launch-geometry bug shows up as a wrong or stale
    // value somewhere past the first block.
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 64, 64}};
    migraphx::shape gs{migraphx::shape::float_type, {2, 32, 32, 2}};

    std::vector<float> x_data(xs.elements());
    for(std::size_t i = 0; i < x_data.size(); ++i)
        x_data[i] = static_cast<float>(i % 251) / 251.0f;

    // Spans [-1.2, 1.2] so some taps fall outside the image.
    std::vector<float> grid_data(gs.elements());
    for(std::size_t i = 0; i < grid_data.size(); ++i)
        grid_data[i] = -1.2f + 2.4f * (static_cast<float>(i % 97) / 96.0f);

    auto make = [&] {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto x    = mm->add_literal(migraphx::literal{xs, x_data});
        auto grid = mm->add_literal(migraphx::literal{gs, grid_data});
        mm->add_instruction(
            migraphx::make_op(
                "gridsample",
                {{"mode", "linear"}, {"padding_mode", "border"}, {"align_corners", false}}),
            x,
            grid);
        return p;
    };

    auto gpu_results = run_gpu_gridsample(make());
    auto ref_results = run_ref_program(make());

    EXPECT(gpu_results.size() == 2 * 3 * 32 * 32);
    EXPECT(migraphx::verify::verify_rms_range(gpu_results, ref_results));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
