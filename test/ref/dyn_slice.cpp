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
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

#include <numeric>
#include <vector>

#include <test.hpp>

using dd = migraphx::shape::dynamic_dimension;
using migraphx::sym::var;

TEST_CASE(dyn_slice_concrete_bounds_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}),
        l0,
        starts,
        ends);
    // Every bound is concrete, so the output shape is known at compile time.
    EXPECT(p.get_output_shapes().back() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> starts_data = {1};
    std::vector<int64_t> ends_data   = {3};
    migraphx::parameter_map params;
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data.data());

    auto result = p.eval(params).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    EXPECT(results_vector == gold);
    // The static output shape lets the compiler make the aliased view contiguous.
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}});
}

TEST_CASE(dyn_slice_sym_ends_test)
{
    // Symbolic `ends` bound with a variable input supplying its runtime value. The same
    // compiled program handles both values of the symbol.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {2}},
                           {"starts", {1}},
                           {"ends", migraphx::value::array{migraphx::to_value(var("n", {1, 3}))}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> starts_data = {1};
    std::vector<int64_t> ends_data0  = {3};
    std::vector<int64_t> ends_data1  = {2};
    migraphx::parameter_map params;
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data0.data());

    auto result0 = p.eval(params).back();
    std::vector<int> results_vector0;
    result0.visit([&](auto output) { results_vector0.assign(output.begin(), output.end()); });
    std::vector<int> gold0 = {1, 2, 4, 5, 7, 8, 10, 11};
    EXPECT(results_vector0 == gold0);
    EXPECT(result0.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});

    params["ends"] = migraphx::argument(bounds_shape, ends_data1.data());

    auto result1 = p.eval(params).back();
    std::vector<int> results_vector1;
    result1.visit([&](auto output) { results_vector1.assign(output.begin(), output.end()); });
    std::vector<int> gold1 = {1, 4, 7, 10};
    EXPECT(results_vector1 == gold1);
    EXPECT(result1.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 1}, {6, 3, 1}});
}

TEST_CASE(dyn_slice_sym_starts_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {2}},
                           {"starts", migraphx::value::array{migraphx::to_value(var("m", {0, 2}))}},
                           {"ends", {3}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> starts_data = {1};
    std::vector<int64_t> ends_data   = {3};
    migraphx::parameter_map params;
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data.data());

    auto result = p.eval(params).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    EXPECT(results_vector == gold);
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(dyn_slice_sym_both_bounds_test)
{
    // Both bounds symbolic, each with its own runtime value.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {2}},
                           {"starts", migraphx::value::array{migraphx::to_value(var("m", {0, 1}))}},
                           {"ends", migraphx::value::array{migraphx::to_value(var("n", {1, 3}))}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> starts_data = {1};
    std::vector<int64_t> ends_data   = {3};
    migraphx::parameter_map params;
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data.data());

    auto result = p.eval(params).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    EXPECT(results_vector == gold);
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(dyn_slice_sym_data_test)
{
    // Symbolic input shape sliced on a fixed axis: the output is symbolic after compiling and
    // resolves once the parameter is bound to a static shape.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type,
                      {dd{var("n", {1, 4})}, dd{migraphx::sym::lit(2)}, dd{migraphx::sym::lit(3)}}};
    auto x = mm->add_parameter("x", s);
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}),
        x,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(input_fixed_shape.elements());
    std::iota(data.begin(), data.end(), 0);
    std::vector<int64_t> starts_data = {1};
    std::vector<int64_t> ends_data   = {3};
    migraphx::parameter_map params;
    params["x"]      = migraphx::argument(input_fixed_shape, data.data());
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data.data());

    auto result = p.eval(params).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    EXPECT(results_vector == gold);
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(dyn_slice_sym_bounds_multi_axes_test)
{
    // Two symbolic end bounds over two axes, so each bound is clamped against its own axis
    // length and more than one axis is sliced.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {2}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1, 2}},
                           {"starts", {1, 0}},
                           {"ends",
                            migraphx::value::array{migraphx::to_value(var("n", {1, 2})),
                                                   migraphx::to_value(var("m", {1, 3}))}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> starts_data = {1, 0};
    std::vector<int64_t> ends_data   = {2, 2};
    migraphx::parameter_map params;
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data.data());

    auto result = p.eval(params).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold = {3, 4, 9, 10};
    EXPECT(results_vector == gold);
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 1, 2}, {6, 3, 1}});
}

TEST_CASE(dyn_slice_runtime_bounds_clamped_test)
{
    // A runtime end that is out of range and a negative runtime start are both resolved
    // against the axis length when the slice runs.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {2}},
                           {"starts", migraphx::value::array{migraphx::to_value(var("m", {0, 1}))}},
                           {"ends", migraphx::value::array{migraphx::to_value(var("n", {1, 8}))}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> starts_data = {-2};
    std::vector<int64_t> ends_data   = {100};
    migraphx::parameter_map params;
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data.data());

    auto result = p.eval(params).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    EXPECT(results_vector == gold);
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(dyn_slice_negative_axis_test)
{
    // The axes attribute is normalized when the program is compiled, so the runtime bounds are
    // applied to axis 2.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice", {{"axes", {-1}}, {"starts", {1}}, {"ends", {3}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> starts_data = {1};
    std::vector<int64_t> ends_data   = {3};
    migraphx::parameter_map params;
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data.data());

    auto result = p.eval(params).back();
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    EXPECT(results_vector == gold);
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}});
}

TEST_CASE(dyn_slice_end_before_start_error_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    migraphx::shape bounds_shape{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", bounds_shape);
    auto ends   = mm->add_parameter("ends", bounds_shape);
    mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {2}},
                           {"starts", migraphx::value::array{migraphx::to_value(var("m", {0, 3}))}},
                           {"ends", migraphx::value::array{migraphx::to_value(var("n", {0, 3}))}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> starts_data = {2};
    std::vector<int64_t> ends_data   = {1};
    migraphx::parameter_map params;
    params["starts"] = migraphx::argument(bounds_shape, starts_data.data());
    params["ends"]   = migraphx::argument(bounds_shape, ends_data.data());

    EXPECT(test::throws([&] { p.eval(params); }));
}
