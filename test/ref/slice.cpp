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
#include <migraphx/verify.hpp>

#include <test.hpp>

using dd = migraphx::shape::dynamic_dimension;
using migraphx::sym::lit;
using migraphx::sym::var;

TEST_CASE(slice_test_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}),
                        l0);
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
    auto result           = p.eval({}).back();
    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(slice_test_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {0, 1, 2}}, {"starts", {0, 0, 0}}, {"ends", {2, 2, 2}}}),
        l0);
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {4, 2, 1}};
    auto result           = p.eval({}).back();
    std::vector<int> gold = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(slice_var_inputs_static0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int32_t> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {2}}, {"mode", migraphx::value::array{"starts", "ends"}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int32_t> start_data = {1};
    std::vector<int32_t> end_data   = {3};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    auto result                     = p.eval(params).back();
    std::vector<int32_t> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int32_t> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_static1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int32_t> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {2}}, {"mode", migraphx::value::array{"starts", "ends"}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int32_t> start_data = {-2};
    std::vector<int32_t> end_data   = {2831};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    auto result                     = p.eval(params).back();
    std::vector<int32_t> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int32_t> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_static2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::float_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int64_type, {3}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    auto axes   = mm->add_parameter("axes", s1);
    mm->add_instruction(
        migraphx::make_op("slice", {{"mode", migraphx::value::array{"starts", "ends", "axes"}}}),
        l0,
        starts,
        ends,
        axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> start_data = {0, 0, 0};
    std::vector<int64_t> end_data   = {2, 2, 2};
    std::vector<int64_t> axes_data  = {0, 1, 2};
    params["starts"]                = migraphx::argument(s1, start_data.data());
    params["ends"]                  = migraphx::argument(s1, end_data.data());
    params["axes"]                  = migraphx::argument(s1, axes_data.data());
    auto result                     = p.eval(params).back();
    std::vector<float> gold         = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<float> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    mm->add_instruction(
        migraphx::make_op(
            "slice", {{"axes", {2}}, {"ends", {10}}, {"mode", migraphx::value::array{"starts"}}}),
        input,
        starts);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> start_data = {1};
    params["input"]             = migraphx::argument(s2, input_data.data());
    params["starts"]            = migraphx::argument(s1, start_data.data());
    auto result                 = p.eval(params).back();
    std::vector<int> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto ends = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op(
            "slice", {{"axes", {2}}, {"starts", {-5}}, {"mode", migraphx::value::array{"ends"}}}),
        input,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> ends_data = {3};
    params["input"]            = migraphx::argument(s2, input_data.data());
    params["ends"]             = migraphx::argument(s1, ends_data.data());
    auto result                = p.eval(params).back();
    std::vector<int> gold      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> results_vector(2 * 2 * 3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto axes = mm->add_parameter("axes", s1);
    mm->add_instruction(
        migraphx::make_op(
            "slice", {{"starts", {1}}, {"ends", {-1}}, {"mode", migraphx::value::array{"axes"}}}),
        input,
        axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> axes_data = {2};
    params["input"]            = migraphx::argument(s2, input_data.data());
    params["axes"]             = migraphx::argument(s1, axes_data.data());
    auto result                = p.eval(params).back();
    std::vector<int> gold      = {1, 4, 7, 10};
    std::vector<int> results_vector(2 * 2 * 1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {2}}, {"mode", migraphx::value::array{"starts", "ends"}}}),
        input,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> starts_data = {1};
    std::vector<int> ends_data   = {std::numeric_limits<int>::max()};
    params["input"]              = migraphx::argument(s2, input_data.data());
    params["starts"]             = migraphx::argument(s1, starts_data.data());
    params["ends"]               = migraphx::argument(s1, ends_data.data());
    auto result                  = p.eval(params).back();
    std::vector<int> gold        = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto axes   = mm->add_parameter("axes", s1);
    mm->add_instruction(migraphx::make_op("slice",
                                          {{"ends", {std::numeric_limits<int>::max()}},
                                           {"mode", migraphx::value::array{"starts", "axes"}}}),
                        input,
                        starts,
                        axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> starts_data = {1};
    std::vector<int> axes_data   = {2};
    params["input"]              = migraphx::argument(s2, input_data.data());
    params["starts"]             = migraphx::argument(s1, starts_data.data());
    params["axes"]               = migraphx::argument(s1, axes_data.data());
    auto result                  = p.eval(params).back();
    std::vector<int> gold        = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn5)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto ends = mm->add_parameter("ends", s1);
    auto axes = mm->add_parameter("axes", s1);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"starts", {-4}}, {"mode", migraphx::value::array{"ends", "axes"}}}),
        input,
        ends,
        axes);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> ends_data = {2};
    std::vector<int> axes_data = {2};
    params["input"]            = migraphx::argument(s2, input_data.data());
    params["ends"]             = migraphx::argument(s1, ends_data.data());
    params["axes"]             = migraphx::argument(s1, axes_data.data());
    auto result                = p.eval(params).back();
    std::vector<int> gold      = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_var_inputs_dyn6)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {{2, 4, {2, 4}}, {2, 4, {2, 4}}, {3, 8}}};
    auto input = mm->add_parameter("input", s0);
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {2}}, {"mode", migraphx::value::array{"starts", "ends"}}}),
        input,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    migraphx::shape s2{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> input_data(2 * 2 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    std::vector<int> start_data = {1};
    std::vector<int> end_data   = {3};
    params["input"]             = migraphx::argument(s2, input_data.data());
    params["starts"]            = migraphx::argument(s1, start_data.data());
    params["ends"]              = migraphx::argument(s1, end_data.data());
    auto result                 = p.eval(params).back();
    std::vector<int> gold       = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(slice_dyn_test0)
{
    // Slice a single dynamic dimension. ax1 slice limits are smaller than min; ax2 "ends" is
    // too large
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 3}, {2, 2}, {3, 3}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1, 2}}, {"starts", {0, 1}}, {"ends", {1, 6}}}), x);
    migraphx::shape s2{migraphx::shape::int32_type, {{2, 3}, {1, 1}, {2, 2}}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));

    //  the strides of sresult are those of the original shape, not
    // reduced to sliced size.
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 1, 2}, {6, 3, 1}};
    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    migraphx::parameter_map params;
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {1, 2, 7, 8};
    std::vector<int> results_vector(2 * 1 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(slice_dyn_test1)
{
    // Slice all three dynamic dimensions
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {{2, 2}, {2, 2}, {3, 3}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {0, 1, 2}}, {"starts", {0, 0, 0}}, {"ends", {2, 2, 2}}}),
        x);

    migraphx::shape s2{migraphx::shape::int32_type, {{2, 2}, {2, 2}, {2, 2}}};
    EXPECT(p.get_output_shapes().back() == s2);
    p.compile(migraphx::make_target("ref"));
    migraphx::shape sresult{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}};

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    migraphx::parameter_map params;
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<int> results_vector(2 * 2 * 2);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() == sresult);
}

TEST_CASE(slice_sym_data_test)
{
    // Symbolic input shape sliced on a fixed axis. The output is symbolic after compiling and
    // is demoted back to static when compute_shape is re-run with the runtime static shape.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {dd{var("n", {1, 4})}, dd{lit(2)}, dd{lit(3)}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(slice_sym_data_clamped_bounds_test)
{
    // Out-of-range bounds on the fixed axes of a symbolic shape are clamped against those axes'
    // lengths: axis 0 end 5 clips to 2, axis 2 start -2 resolves to 1 and end 10 clips to 3.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {dd{lit(2)}, dd{var("n", {1, 4})}, dd{lit(3)}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 2}}, {"starts", {0, -2}}, {"ends", {5, 10}}}), x);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(slice_sym_data_symbolic_axis_test)
{
    // Slice the non-fixed symbolic axis itself. The bounds cannot be normalized against a
    // compile-time length, so they are trusted and the extent is just ends - starts.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {dd{var("n", {2, 4})}, dd{lit(2)}, dd{lit(3)}}};
    auto x = mm->add_parameter("x", s);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {3}}}),
                        x);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {4, 2, 3}};
    std::vector<int> data(4 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold(2 * 2 * 3);
    std::iota(gold.begin(), gold.end(), 6);
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 3}, {6, 3, 1}});
}

TEST_CASE(slice_sym_data_transposed_test)
{
    // The slice aliases a transposed symbolic shape, so both the offset and the result strides
    // follow the permuted layout.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {dd{var("n", {1, 4})}, dd{lit(2)}, dd{lit(3)}}};
    auto x  = mm->add_parameter("x", s);
    auto tx = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), x);
    mm->add_instruction(migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {3}}}),
                        tx);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();

    std::vector<int> gold = {1, 4, 2, 5, 7, 10, 8, 11};
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 1, 3}});
}

TEST_CASE(slice_sym_ends_attr_test)
{
    // Symbolic `ends` bound with a variable input supplying its runtime value. Evaluating the
    // same compiled program twice shows the input drives the output, not the compile-time symbol.
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int64_type, {1}};
    auto ends = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op(
            "slice",
            {{"axes", {2}},
             {"starts", {1}},
             {"ends", migraphx::value::array{migraphx::to_value(dd{var("n", {1, 3})})}},
             {"mode", migraphx::value::array{"ends"}}}),
        l0,
        ends);
    p.compile(migraphx::make_target("ref"));

    std::vector<int64_t> ends_data0 = {3};
    migraphx::parameter_map params0;
    params0["ends"]        = migraphx::argument(s1, ends_data0.data());
    auto result0           = p.eval(params0).back();
    std::vector<int> gold0 = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector0;
    result0.visit([&](auto output) { results_vector0.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector0, gold0));
    EXPECT(result0.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});

    std::vector<int64_t> ends_data1 = {2};
    migraphx::parameter_map params1;
    params1["ends"]        = migraphx::argument(s1, ends_data1.data());
    auto result1           = p.eval(params1).back();
    std::vector<int> gold1 = {1, 4, 7, 10};
    std::vector<int> results_vector1;
    result1.visit([&](auto output) { results_vector1.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector1, gold1));
    EXPECT(result1.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 1}, {6, 3, 1}});
}

TEST_CASE(slice_sym_starts_attr_test)
{
    // Symbolic `starts` bound with a variable input supplying its runtime value.
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    mm->add_instruction(
        migraphx::make_op(
            "slice",
            {{"axes", {2}},
             {"starts", migraphx::value::array{migraphx::to_value(dd{var("m", {0, 2})})}},
             {"ends", {3}},
             {"mode", migraphx::value::array{"starts"}}}),
        l0,
        starts);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> starts_data = {1};
    params["starts"]                 = migraphx::argument(s1, starts_data.data());
    auto result                      = p.eval(params).back();
    std::vector<int> gold            = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(slice_sym_both_bounds_attr_test)
{
    // Both bounds symbolic, each with its own variable input.
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int64_type, {1}};
    auto starts = mm->add_parameter("starts", s1);
    auto ends   = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op(
            "slice",
            {{"axes", {2}},
             {"starts", migraphx::value::array{migraphx::to_value(dd{var("m", {0, 1})})}},
             {"ends", migraphx::value::array{migraphx::to_value(dd{var("n", {1, 3})})}},
             {"mode", migraphx::value::array{"starts", "ends"}}}),
        l0,
        starts,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> starts_data = {1};
    std::vector<int64_t> ends_data   = {3};
    params["starts"]                 = migraphx::argument(s1, starts_data.data());
    params["ends"]                   = migraphx::argument(s1, ends_data.data());
    auto result                      = p.eval(params).back();
    std::vector<int> gold            = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(slice_sym_bound_and_sym_data_test)
{
    // A symbolic bound on a symbolic input shape: the sliced axis is a fixed dimension of the
    // symbolic shape, so the bound is clamped against it while axis 0 stays symbolic.
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::int32_type, {dd{var("k", {1, 4})}, dd{lit(2)}, dd{lit(3)}}};
    auto x = mm->add_parameter("x", s0);
    migraphx::shape s1{migraphx::shape::int64_type, {1}};
    auto ends = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op(
            "slice",
            {{"axes", {2}},
             {"starts", {0}},
             {"ends", migraphx::value::array{migraphx::to_value(dd{var("n", {1, 3})})}},
             {"mode", migraphx::value::array{"ends"}}}),
        x,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_fixed_shape{migraphx::shape::int32_type, {2, 2, 3}};
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    std::vector<int64_t> ends_data = {2};
    params["x"]                    = migraphx::argument(input_fixed_shape, data.data());
    params["ends"]                 = migraphx::argument(s1, ends_data.data());
    auto result                    = p.eval(params).back();
    std::vector<int> gold          = {0, 1, 3, 4, 6, 7, 9, 10};
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}

TEST_CASE(slice_sym_bounds_multi_axes_test)
{
    // Two symbolic end bounds over two axes, so each bound is clamped against its own axis
    // length and lens_calc runs over more than one axis.
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int64_type, {2}};
    auto ends = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op("slice",
                          {{"axes", {1, 2}},
                           {"starts", {1, 0}},
                           {"ends",
                            migraphx::value::array{migraphx::to_value(dd{var("n", {1, 2})}),
                                                   migraphx::to_value(dd{var("m", {1, 3})})}},
                           {"mode", migraphx::value::array{"ends"}}}),
        l0,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> ends_data = {2, 2};
    params["ends"]                 = migraphx::argument(s1, ends_data.data());
    auto result                    = p.eval(params).back();
    std::vector<int> gold          = {3, 4, 9, 10};
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 1, 2}, {6, 3, 1}});
}

TEST_CASE(slice_sym_bound_clamped_runtime_test)
{
    // A concrete negative start (normalized to 1 at compile time) alongside a symbolic end whose
    // runtime value is out of range and has to be clamped to the axis length.
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<int> data(2 * 2 * 3);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s0{migraphx::shape::int32_type, {2, 2, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s0, data});
    migraphx::shape s1{migraphx::shape::int64_type, {1}};
    auto ends = mm->add_parameter("ends", s1);
    mm->add_instruction(
        migraphx::make_op(
            "slice",
            {{"axes", {2}},
             {"starts", {-2}},
             {"ends", migraphx::value::array{migraphx::to_value(dd{var("n", {1, 8})})}},
             {"mode", migraphx::value::array{"ends"}}}),
        l0,
        ends);
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    std::vector<int64_t> ends_data = {100};
    params["ends"]                 = migraphx::argument(s1, ends_data.data());
    auto result                    = p.eval(params).back();
    std::vector<int> gold          = {1, 2, 4, 5, 7, 8, 10, 11};
    std::vector<int> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    EXPECT(result.get_shape() ==
           migraphx::shape{migraphx::shape::int32_type, {2, 2, 2}, {6, 3, 1}});
}
