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
#include <migraphx/op/pad.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(pad_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_test_asym)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 1, 1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(9);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{1, 2, 0, 3, 4, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_test_highest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op("pad",
                          {{"pads", {1, 1, 1, 1}}, {"value", std::numeric_limits<float>::max()}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    const float x = std::numeric_limits<migraphx::half>::max();
    std::vector<float> gold{x, x, x, x, x, 1, 2, x, x, 3, 4, x, x, x, x, x};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_test_lowest_half)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::half_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op(
            "pad", {{"pads", {1, 1, 1, 1}}, {"value", std::numeric_limits<float>::lowest()}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    const float x = std::numeric_limits<migraphx::half>::lowest();
    std::vector<float> gold{x, x, x, x, x, 1, 2, x, x, 3, 4, x, x, x, x, x};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2}}, {2, 4, {2}}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("pad", {{"pads", {1, 1, 1, 1}}}), x);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data = {1, 2, 3, 4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 2}};
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector(16);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_edge_1d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op("pad", {{"pads", {2, 3}}, {"mode", migraphx::op::pad::edge_pad}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // input:  [1, 2, 3, 4]
    // pad 2 before, 3 after with edge values
    // result: [1, 1, 1, 2, 3, 4, 4, 4, 4]
    std::vector<float> gold{1, 1, 1, 2, 3, 4, 4, 4, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_edge_2d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    // clang-format off
    auto l0 = mm->add_literal(migraphx::literal{s, {0, 1, 2,
                                                    3, 4, 5,
                                                    6, 7, 8}});
    // clang-format on
    mm->add_instruction(
        migraphx::make_op("pad", {{"pads", {1, 2, 1, 2}}, {"mode", migraphx::op::pad::edge_pad}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // pads: dim0 1 before/1 after, dim1 2 before/2 after -> shape {5, 7}
    // clang-format off
    std::vector<float> gold{0, 0, 0, 1, 2, 2, 2,
                            0, 0, 0, 1, 2, 2, 2,
                            3, 3, 3, 4, 5, 5, 5,
                            6, 6, 6, 7, 8, 8, 8,
                            6, 6, 6, 7, 8, 8, 8};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_edge_2d_asym_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op("pad", {{"pads", {0, 0, 1, 1}}, {"mode", migraphx::op::pad::edge_pad}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // pads: dim0 0 before/1 after, dim1 0 before/1 after -> shape {3, 3}
    // clang-format off
    std::vector<float> gold{1, 2, 2,
                            3, 4, 4,
                            3, 4, 4};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_reflect_1d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op("pad",
                          {{"pads", {2, 3}}, {"mode", migraphx::op::pad::reflect_pad}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // input:  [1, 2, 3, 4]
    // pad 2 before, 3 after with reflect
    // result: [3, 2, 1, 2, 3, 4, 3, 2, 1]
    std::vector<float> gold{3, 2, 1, 2, 3, 4, 3, 2, 1};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_reflect_2d_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3, 4}});
    mm->add_instruction(
        migraphx::make_op("pad",
                          {{"pads", {0, 2, 0, 1}}, {"mode", migraphx::op::pad::reflect_pad}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // input: [[1, 2], [3, 4]], pads: dim0 0/0, dim1 2 before/1 after -> shape {2, 5}
    // clang-format off
    std::vector<float> gold{1, 2, 1, 2, 1,
                            3, 4, 3, 4, 3};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_reflect_2d_sym_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 4}};
    // clang-format off
    auto l0 = mm->add_literal(migraphx::literal{s, { 1,  2,  3,  4,
                                                     5,  6,  7,  8,
                                                     9, 10, 11, 12,
                                                    13, 14, 15, 16}});
    // clang-format on
    mm->add_instruction(
        migraphx::make_op("pad",
                          {{"pads", {0, 2, 0, 2}}, {"mode", migraphx::op::pad::reflect_pad}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // pads: dim0 0/0, dim1 2 before/2 after -> shape {4, 8}
    // clang-format off
    std::vector<float> gold{ 3,  2,  1,  2,  3,  4,  3,  2,
                             7,  6,  5,  6,  7,  8,  7,  6,
                            11, 10,  9, 10, 11, 12, 11, 10,
                            15, 14, 13, 14, 15, 16, 15, 14};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(pad_reflect_multiaxis_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    // clang-format off
    auto l0 = mm->add_literal(migraphx::literal{s, {1, 2, 3,
                                                    4, 5, 6}});
    // clang-format on
    mm->add_instruction(
        migraphx::make_op("pad",
                          {{"pads", {0, 2, 2, 0}}, {"mode", migraphx::op::pad::reflect_pad}}),
        l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    // pads: dim0 0 before/2 after, dim1 2 before/0 after -> shape {4, 5}
    // clang-format off
    std::vector<float> gold{3, 2, 1, 2, 3,
                            6, 5, 4, 5, 6,
                            3, 2, 1, 2, 3,
                            6, 5, 4, 5, 6};
    // clang-format on
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
