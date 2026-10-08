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
    migraphx::shape index_s{migraphx::shape::int32_type};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};

    auto* sub0 = p.create_module("sub_0");
    auto x0    = sub0->add_parameter("data", data_s);
    sub0->add_return({sub0->add_instruction(migraphx::make_op("neg"), x0)});
    auto* sub1 = p.create_module("sub_1");
    auto x1    = sub1->add_parameter("data", data_s);
    sub1->add_return({sub1->add_instruction(migraphx::make_op("abs"), x1)});

    auto index = mm->add_parameter("index", index_s);
    auto data  = mm->add_parameter("data", data_s);
    auto smi =
        mm->add_instruction(migraphx::make_op("select_module_index"), {index, data}, {sub0, sub1});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), smi);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{1, -2, 3, -4};
    std::int32_t which = 0;
    migraphx::parameter_map params;
    params["index"] = migraphx::argument(index_s, &which);
    params["data"]  = migraphx::argument(data_s, input_data.data());
    std::vector<float> out0;
    p.eval(params).back().visit([&](auto output) { out0.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(out0, std::vector<float>{-1, 2, -3, 4}));

    which = 1;
    std::vector<float> out1;
    p.eval(params).back().visit([&](auto output) { out1.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(out1, std::vector<float>{1, 2, 3, 4}));

    which = -1;
    EXPECT(test::throws([&] { p.eval(params); }));
    which = 2;
    EXPECT(test::throws([&] { p.eval(params); }));
}

TEST_CASE(select_module_index_map)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape index_s{migraphx::shape::int64_type, {1}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};

    auto* sub0 = p.create_module("sub_0");
    auto x0    = sub0->add_parameter("data", data_s);
    sub0->add_return({sub0->add_instruction(migraphx::make_op("neg"), x0)});
    auto* sub1 = p.create_module("sub_1");
    auto x1    = sub1->add_parameter("data", data_s);
    sub1->add_return({sub1->add_instruction(migraphx::make_op("abs"), x1)});

    auto index = mm->add_parameter("index", index_s);
    auto data  = mm->add_parameter("data", data_s);
    auto smi   = mm->add_instruction(
        migraphx::make_op("select_module_index", {{"index_map", std::vector<std::size_t>{4, 7}}}),
        {index, data},
        {sub0, sub1});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), smi);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{1, -2, 3, -4};
    std::int64_t which = 7;
    migraphx::parameter_map params;
    params["index"] = migraphx::argument(index_s, &which);
    params["data"]  = migraphx::argument(data_s, input_data.data());
    std::vector<float> out7;
    p.eval(params).back().visit([&](auto output) { out7.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(out7, std::vector<float>{1, 2, 3, 4}));

    which = 4;
    std::vector<float> out4;
    p.eval(params).back().visit([&](auto output) { out4.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(out4, std::vector<float>{-1, 2, -3, 4}));

    which = 1;
    EXPECT(test::throws([&] { p.eval(params); }));
}
