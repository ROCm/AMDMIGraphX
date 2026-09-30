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
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/sym.hpp>

#include <test.hpp>

TEST_CASE(dyn_concat_gpu_middle_axis)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto x       = mm->add_parameter("x", {migraphx::shape::int32_type, {2, 3, 2}});
    auto y       = mm->add_parameter("y", {migraphx::shape::int32_type, {2, 2, 2}});
    auto x_count = mm->add_parameter("x_count", {migraphx::shape::int64_type, {1}});
    auto y_count = mm->add_parameter("y_count", {migraphx::shape::int64_type, {1}});
    auto result =
        mm->add_instruction(migraphx::make_op("dyn_concat", {{"axis", 1}}), x, y, x_count, y_count);
    auto buffer = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), result);
    auto count  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), result);
    mm->add_return({buffer, count});

    auto target = migraphx::make_target("gpu");
    p.compile(target);
    std::vector<int32_t> x_data       = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int32_t> y_data       = {100, 101, 102, 103, 104, 105, 106, 107};
    std::vector<int64_t> x_count_data = {2};
    std::vector<int64_t> y_count_data = {1};
    migraphx::parameter_map params;
    params["x"] =
        target.copy_to(migraphx::argument{{migraphx::shape::int32_type, {2, 3, 2}}, x_data.data()});
    params["y"] =
        target.copy_to(migraphx::argument{{migraphx::shape::int32_type, {2, 2, 2}}, y_data.data()});
    params["x_count"] =
        target.copy_to(migraphx::argument{{migraphx::shape::int64_type, {1}}, x_count_data.data()});
    params["y_count"] =
        target.copy_to(migraphx::argument{{migraphx::shape::int64_type, {1}}, y_count_data.data()});
    for(const auto& [name, s] : p.get_parameter_shapes())
        if(not contains(params, name))
            params[name] = target.allocate(s);

    auto outputs = p.eval(params);
    EXPECT(
        target.copy_from(outputs.at(0)).to_vector<int32_t>() ==
        std::vector<int32_t>{0, 1, 2, 3, 100, 101, 0, 0, 0, 0, 6, 7, 8, 9, 104, 105, 0, 0, 0, 0});
    EXPECT(target.copy_from(outputs.at(1)).to_vector<int64_t>() == std::vector<int64_t>{3});
}

TEST_CASE(dyn_concat_gpu_dynamic_zero_input)
{
    using migraphx::shape;
    using dynamic_dimension = migraphx::shape::dynamic_dimension;
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto n = var("n", {0, 2});
    const auto m = var("m", {0, 2});
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter(
        "x", {shape::float_type, std::vector<dynamic_dimension>{{lit(2)}, {n}, {lit(2)}}});
    auto y = mm->add_parameter(
        "y", {shape::float_type, std::vector<dynamic_dimension>{{lit(2)}, {m}, {lit(2)}}});
    auto x_count = mm->add_parameter("x_count", {shape::int64_type, {1}});
    auto y_count = mm->add_parameter("y_count", {shape::int64_type, {1}});
    auto result =
        mm->add_instruction(migraphx::make_op("dyn_concat", {{"axis", 1}}), x, y, x_count, y_count);
    auto buffer = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), result);
    auto count  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), result);
    mm->add_return({buffer, count});

    auto target = migraphx::make_target("gpu");
    p.compile(target);
    migraphx::argument x_arg{{shape::float_type, {2, 0, 2}}};
    std::vector<float> y_data         = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<int64_t> x_count_data = {0};
    std::vector<int64_t> y_count_data = {1};
    migraphx::parameter_map params;
    params["x"] = target.copy_to(x_arg);
    params["y"] = target.copy_to(migraphx::argument{{shape::float_type, {2, 1, 2}}, y_data.data()});
    params["x_count"] =
        target.copy_to(migraphx::argument{{shape::int64_type, {1}}, x_count_data.data()});
    params["y_count"] =
        target.copy_to(migraphx::argument{{shape::int64_type, {1}}, y_count_data.data()});
    for(const auto& [name, s] : p.get_parameter_shapes())
        if(not contains(params, name))
            params[name] = target.allocate(s);

    auto outputs = p.eval(params);
    EXPECT(target.copy_from(outputs.at(0)).to_vector<float>() == std::vector<float>{5.0f,
                                                                                    6.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    7.0f,
                                                                                    8.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    0.0f,
                                                                                    0.0f});
    EXPECT(target.copy_from(outputs.at(1)).to_vector<int64_t>() == std::vector<int64_t>{1});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
