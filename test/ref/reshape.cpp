/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(reshape_lazy_test0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
    std::vector<int64_t> new_shape = {8, 3, 1, 1};
    mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, data));
}

TEST_CASE(reshape_lazy_test1)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
    std::vector<int64_t> new_shape = {1, 3, 4, 2};
    mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, data));
}

TEST_CASE(reshape_lazy_test2)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, data});
    std::vector<int64_t> new_shape = {1, 2, 3, 4};
    mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, data));
}

TEST_CASE(reshape_lazy_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    std::vector<int64_t> new_shape = {0, 8, 3, 1};
    auto input                     = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", new_shape}}), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(48);
    std::iota(data.begin(), data.end(), -3);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 24, 1, 1}};
    params["X"] = migraphx::argument(input_fixed_shape, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, data));
}

TEST_CASE(reshape_test0)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> gold(24);
    std::iota(gold.begin(), gold.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, gold});
    std::vector<int64_t> new_shape = {8, 3, 1, 1};
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reshape_test1)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> gold(24);
    std::iota(gold.begin(), gold.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, gold});
    std::vector<int64_t> new_shape = {1, 3, 4, 2};
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reshape_test2)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {24, 1, 1, 1}};
    std::vector<float> gold(24);
    std::iota(gold.begin(), gold.end(), -3);
    migraphx::program p;
    auto* mm                       = p.get_main_module();
    auto l                         = mm->add_literal(migraphx::literal{a_shape, gold});
    std::vector<int64_t> new_shape = {1, 2, 3, 4};
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reshape_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};
    std::vector<int64_t> new_shape = {0, 8, 3, 1};
    auto input                     = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("reshape", {{"dims", new_shape}}), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> gold(48);
    std::iota(gold.begin(), gold.end(), -3);
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 24, 1, 1}};
    params["X"] = migraphx::argument(input_fixed_shape, gold.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector{};
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
