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

#include <cstdint>

TEST_CASE(select_module_index_dispatch)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape index_s{migraphx::shape::int64_type, {1}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};

    auto create_sub = [&](float add_val, const std::string& name) {
        auto* sub = p.create_module(name);
        auto x    = sub->add_parameter("data", data_s);
        auto lit  = sub->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {add_val}});
        auto bc  = sub->add_instruction(migraphx::make_op("multibroadcast"), lit, x);
        auto add = sub->add_instruction(migraphx::make_op("add"), x, bc);
        sub->add_return({add});
        return sub;
    };
    auto* sub0 = create_sub(10.0f, "sub_0");
    auto* sub1 = create_sub(20.0f, "sub_1");

    auto index = mm->add_parameter("index", index_s);
    auto data  = mm->add_parameter("data", data_s);
    auto smi   = mm->add_instruction(
        migraphx::make_op("select_module_index"), {index, data}, {sub0, sub1});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), smi);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{1, 2, 3, 4};
    auto run = [&](std::int64_t which) {
        migraphx::parameter_map params;
        params["index"] = migraphx::argument(index_s, &which);
        params["data"]  = migraphx::argument(data_s, input_data.data());
        auto result     = p.eval(params).back();
        std::vector<float> out;
        result.visit([&](auto output) { out.assign(output.begin(), output.end()); });
        return out;
    };
    EXPECT(migraphx::verify::verify_rms_range(run(0), {11, 12, 13, 14}));
    EXPECT(migraphx::verify::verify_rms_range(run(1), {21, 22, 23, 24}));
}

TEST_CASE(select_module_index_map)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape index_s{migraphx::shape::int64_type, {1}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};

    auto create_sub = [&](float add_val, const std::string& name) {
        auto* sub = p.create_module(name);
        auto x    = sub->add_parameter("data", data_s);
        auto lit  = sub->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {add_val}});
        auto bc  = sub->add_instruction(migraphx::make_op("multibroadcast"), lit, x);
        auto add = sub->add_instruction(migraphx::make_op("add"), x, bc);
        sub->add_return({add});
        return sub;
    };
    auto* sub0 = create_sub(10.0f, "sub_0");
    auto* sub1 = create_sub(20.0f, "sub_1");

    auto index = mm->add_parameter("index", index_s);
    auto data  = mm->add_parameter("data", data_s);
    auto smi   = mm->add_instruction(
        migraphx::make_op("select_module_index", {{"index_map", std::vector<std::size_t>{4, 7}}}),
        {index, data},
        {sub0, sub1});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), smi);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{1, 2, 3, 4};
    migraphx::parameter_map params;
    std::int64_t which = 7;
    params["index"]    = migraphx::argument(index_s, &which);
    params["data"]     = migraphx::argument(data_s, input_data.data());
    auto result        = p.eval(params).back();
    std::vector<float> out;
    result.visit([&](auto output) { out.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(out, {21, 22, 23, 24}));
}
