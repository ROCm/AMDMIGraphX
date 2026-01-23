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
#include <migraphx/rewrite_resize.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_resize{},
                         migraphx::dead_code_elimination{}});
}

migraphx::program make_resize_program(const migraphx::value& v, const migraphx::shape& input_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x = mm->add_parameter("x", input_shape);
    mm->add_instruction(migraphx::make_op("resize", v), x);
    return p;
}

auto check_resize(const migraphx::value& v, const migraphx::shape& input_shape)
{
    auto p1 = make_resize_program(v, input_shape);

    auto* mm = p1.get_main_module();
    run_pass(*mm);

    // After rewrite, should have gather instead of resize
    CHECK(std::none_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "resize"; }));
    CHECK(std::any_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "gather"; }));

    auto p2 = make_resize_program(v, input_shape);

    auto input = migraphx::iota_argument(input_shape);
    auto output1 = p1.eval({{"x", input}}).back();
    auto output2 = p2.eval({{"x", input}}).back();

    std::stringstream ss;
    ss << output1 << " == " << output2;

    return test::make_predicate(ss.str(), [=] {
        return output1 == output2;
    });
}


// Test nearest mode downsample with floor rounding (1-input mode with scales attribute)
TEST_CASE(rewrite_resize_nearest_downsample_floor)
{
    EXPECT(check_resize(
        {{"scales", {1.0f, 1.0f, 0.6f, 0.6f}},
         {"nearest_mode", "floor"},
         {"coordinate_transformation_mode", "asymmetric"}},
        {migraphx::shape::float_type, {1, 1, 2, 4}}));
}

// Test nearest mode upsample with round_prefer_floor rounding
TEST_CASE(rewrite_resize_nearest_upsample_pf)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto x = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("resize",
                                          {{"scales", {1.0f, 1.0f, 2.0f, 3.0f}},
                                           {"nearest_mode", "round_prefer_floor"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        x);

    run_pass(*mm);

    // After rewrite, should have gather instead of resize
    EXPECT(std::none_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "resize"; }));
    EXPECT(std::any_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "gather"; }));
}

// Test linear mode downsample
TEST_CASE(rewrite_resize_linear_downsample)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto x = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("resize",
                                          {{"scales", {1.0f, 1.0f, 0.6f, 0.5f}},
                                           {"mode", "linear"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        x);

    run_pass(*mm);

    // After rewrite, should have gather instead of resize
    EXPECT(std::none_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "resize"; }));
    EXPECT(std::any_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "gather"; }));
}

// Test that 2-input mode is not rewritten (handled by simplify_dyn_ops first)
TEST_CASE(rewrite_resize_2input_no_rewrite)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 5, 9}};
    auto x = mm->add_parameter("X", sx);

    std::vector<float> scales_data = {1.0f, 1.0f, 0.6f, 0.6f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    auto scales = mm->add_literal(migraphx::literal{ss, scales_data});

    mm->add_instruction(migraphx::make_op("resize",
                                          {{"nearest_mode", "floor"},
                                           {"coordinate_transformation_mode", "asymmetric"}}),
                        x,
                        scales);

    run_pass(*mm);

    // Should still have resize since rewrite_resize only handles 1-input mode
    EXPECT(std::any_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "resize"; }));
}

// Test linear mode with same input/output shapes is optimized away
TEST_CASE(rewrite_resize_linear_same_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sx{migraphx::shape::float_type, {1, 3, 5}};
    auto x = mm->add_parameter("X", sx);

    auto r =
        mm->add_instruction(migraphx::make_op("resize",
                                              {{"sizes", {1, 3, 5}},
                                               {"mode", "linear"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            x);
    mm->add_return({r});

    run_pass(*mm);

    // After rewrite, should not have resize or gather - just pass through
    EXPECT(std::none_of(mm->begin(), mm->end(), [](const auto& ins) {
        return ins.name() == "resize" or ins.name() == "gather";
    }));
}

// Test numerical correctness for nearest mode downsample
TEST_CASE(rewrite_resize_nearest_correctness)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto x1 = mm1->add_parameter("X", sx);

    mm1->add_instruction(migraphx::make_op("resize",
                                           {{"scales", {1.0f, 1.0f, 0.6f, 0.6f}},
                                            {"nearest_mode", "floor"},
                                            {"coordinate_transformation_mode", "asymmetric"}}),
                         x1);

    // Copy for comparison
    migraphx::program p2 = p1;

    // Apply rewrite to p1
    run_pass(*p1.get_main_module());

    // Compile both with ref target
    p1.compile(migraphx::make_target("ref"));
    p2.compile(migraphx::make_target("ref"));

    // Generate test data
    std::vector<float> data(sx.elements());
    std::iota(data.begin(), data.end(), 0.0f);

    migraphx::parameter_map params;
    params["X"] = migraphx::argument(sx, data.data());

    auto result1 = p1.eval(params).back();
    auto result2 = p2.eval(params).back();

    std::vector<float> r1, r2;
    result1.visit([&](auto output) { r1.assign(output.begin(), output.end()); });
    result2.visit([&](auto output) { r2.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(r1, r2));
}

// Test numerical correctness for linear mode
TEST_CASE(rewrite_resize_linear_correctness)
{
    migraphx::program p1;
    auto* mm1 = p1.get_main_module();

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto x1 = mm1->add_parameter("X", sx);

    mm1->add_instruction(migraphx::make_op("resize",
                                           {{"scales", {1.0f, 1.0f, 0.6f, 0.5f}},
                                            {"mode", "linear"},
                                            {"coordinate_transformation_mode", "half_pixel"}}),
                         x1);

    // Copy for comparison
    migraphx::program p2 = p1;

    // Apply rewrite to p1
    run_pass(*p1.get_main_module());

    // Compile both with ref target
    p1.compile(migraphx::make_target("ref"));
    p2.compile(migraphx::make_target("ref"));

    // Generate test data
    std::vector<float> data(sx.elements());
    std::iota(data.begin(), data.end(), 0.0f);

    migraphx::parameter_map params;
    params["X"] = migraphx::argument(sx, data.data());

    auto result1 = p1.eval(params).back();
    auto result2 = p2.eval(params).back();

    std::vector<float> r1, r2;
    result1.visit([&](auto output) { r1.assign(output.begin(), output.end()); });
    result2.visit([&](auto output) { r2.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(r1, r2));
}

// Test using sizes attribute instead of scales
TEST_CASE(rewrite_resize_sizes_attribute)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto x = mm->add_parameter("X", sx);

    mm->add_instruction(migraphx::make_op("resize",
                                          {{"sizes", {1, 1, 4, 6}},
                                           {"nearest_mode", "round_prefer_floor"},
                                           {"coordinate_transformation_mode", "half_pixel"}}),
                        x);

    run_pass(*mm);

    // After rewrite, should have gather instead of resize
    EXPECT(std::none_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "resize"; }));
    EXPECT(std::any_of(
        mm->begin(), mm->end(), [](const auto& ins) { return ins.name() == "gather"; }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
