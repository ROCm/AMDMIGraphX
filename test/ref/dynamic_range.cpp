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
#include <migraphx/register_target.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>

#include <test.hpp>

using dd = migraphx::shape::dynamic_dimension;

TEST_CASE(dynamic_range_float_inc)
{
    // Start=0, Limit=5, Delta=1 -> [0, 1, 2, 3, 4]
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1}, {0}};
    auto start = mm->add_parameter("start", s);
    auto limit = mm->add_parameter("limit", s);
    auto delta = mm->add_parameter("delta", s);
    mm->add_instruction(migraphx::make_op("dynamic_range"), start, limit, delta);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> start_val = {0.0f};
    std::vector<float> limit_val = {5.0f};
    std::vector<float> delta_val = {1.0f};

    migraphx::parameter_map pp;
    pp["start"] = migraphx::argument(s, start_val.data());
    pp["limit"] = migraphx::argument(s, limit_val.data());
    pp["delta"] = migraphx::argument(s, delta_val.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(dynamic_range_float_step)
{
    // Start=0, Limit=5, Delta=2 -> [0, 2, 4]
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1}, {0}};
    auto start = mm->add_parameter("start", s);
    auto limit = mm->add_parameter("limit", s);
    auto delta = mm->add_parameter("delta", s);
    mm->add_instruction(migraphx::make_op("dynamic_range"), start, limit, delta);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> start_val = {0.0f};
    std::vector<float> limit_val = {5.0f};
    std::vector<float> delta_val = {2.0f};

    migraphx::parameter_map pp;
    pp["start"] = migraphx::argument(s, start_val.data());
    pp["limit"] = migraphx::argument(s, limit_val.data());
    pp["delta"] = migraphx::argument(s, delta_val.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.0f, 2.0f, 4.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(dynamic_range_float_dec)
{
    // Start=5, Limit=0, Delta=-1 -> [5, 4, 3, 2, 1]
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1}, {0}};
    auto start = mm->add_parameter("start", s);
    auto limit = mm->add_parameter("limit", s);
    auto delta = mm->add_parameter("delta", s);
    mm->add_instruction(migraphx::make_op("dynamic_range"), start, limit, delta);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> start_val = {5.0f};
    std::vector<float> limit_val = {0.0f};
    std::vector<float> delta_val = {-1.0f};

    migraphx::parameter_map pp;
    pp["start"] = migraphx::argument(s, start_val.data());
    pp["limit"] = migraphx::argument(s, limit_val.data());
    pp["delta"] = migraphx::argument(s, delta_val.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(dynamic_range_int)
{
    // Start=1, Limit=10, Delta=3 -> [1, 4, 7]
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {1}, {0}};
    auto start = mm->add_parameter("start", s);
    auto limit = mm->add_parameter("limit", s);
    auto delta = mm->add_parameter("delta", s);
    mm->add_instruction(migraphx::make_op("dynamic_range"), start, limit, delta);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> start_val = {1};
    std::vector<int32_t> limit_val = {10};
    std::vector<int32_t> delta_val = {3};

    migraphx::parameter_map pp;
    pp["start"] = migraphx::argument(s, start_val.data());
    pp["limit"] = migraphx::argument(s, limit_val.data());
    pp["delta"] = migraphx::argument(s, delta_val.data());

    auto result = p.eval(pp).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int32_t> gold = {1, 4, 7};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(dynamic_range_float_start_equals_limit)
{
    // Start=5, Limit=5, Delta=1 -> [] (empty output)
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1}, {0}};
    auto start = mm->add_parameter("start", s);
    auto limit = mm->add_parameter("limit", s);
    auto delta = mm->add_parameter("delta", s);
    mm->add_instruction(migraphx::make_op("dynamic_range"), start, limit, delta);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> start_val = {5.0f};
    std::vector<float> limit_val = {5.0f};
    std::vector<float> delta_val = {1.0f};

    migraphx::parameter_map pp;
    pp["start"] = migraphx::argument(s, start_val.data());
    pp["limit"] = migraphx::argument(s, limit_val.data());
    pp["delta"] = migraphx::argument(s, delta_val.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    EXPECT(result_vector.empty());
}

TEST_CASE(dynamic_range_symbolic_output)
{
    auto n = dd{migraphx::sym::var("n", {1, 8})};
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int64_type, {1}, {0}};
    auto start = mm->add_parameter("start", s);
    auto limit = mm->add_parameter("limit", s);
    auto delta = mm->add_parameter("delta", s);
    auto range = mm->add_instruction(
        migraphx::make_op("dynamic_range",
                          {{"max_output", 8}, {"output_dim", migraphx::to_value(n)}}),
        start,
        limit,
        delta);

    EXPECT(range->get_shape() == migraphx::shape{migraphx::shape::int64_type, {n}});
    p.compile(migraphx::make_target("ref"));

    std::int64_t start_value = 0;
    std::int64_t limit_value = 5;
    std::int64_t delta_value = 1;
    migraphx::parameter_map params;
    params["start"] = migraphx::argument{s, &start_value};
    params["limit"] = migraphx::argument{s, &limit_value};
    params["delta"] = migraphx::argument{s, &delta_value};

    auto result = p.eval(params).back();
    std::vector<std::int64_t> output;
    result.visit([&](auto values) { output.assign(values.begin(), values.end()); });
    EXPECT(output == std::vector<std::int64_t>{0, 1, 2, 3, 4});
}
