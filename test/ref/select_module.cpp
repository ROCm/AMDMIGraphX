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
#include <migraphx/op/select_module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(select_module_add_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
    auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6}});

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto broadcast_lit =
            submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
        auto add_ins = submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
        submod->add_return({add_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 4}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{2, 14, 5, 10, 5, 14, 14, 2};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(select_module_reduce_test0)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {2, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-5, 12, 7, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(select_module_reduce_test1)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8, 8, -4, -4, 8, -1, 4, -1, 8, 8, -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {4, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    auto result    = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{-5, 12, 7, 4, -5, 12, 7, 4};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(select_module_symbolic_range_test)
{
    migraphx::program p;
    auto create_submodule = [&](std::size_t min, std::size_t max, const std::string& name) {
        auto* submod                                         = p.create_module(name);
        std::vector<migraphx::shape::dynamic_dimension> dims = {
            {migraphx::sym::var("n", {min, max})}, {migraphx::sym::lit(2)}};
        auto input =
            submod->add_parameter("data", migraphx::shape{migraphx::shape::float_type, dims});
        auto output =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), input);
        submod->add_return({output});
        return submod;
    };
    auto* first  = create_submodule(1, 2, "batch_1_2");
    auto* second = create_submodule(3, 4, "batch_3_4");

    auto* mm                                                   = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> input_dims = {{migraphx::sym::var("n", {1, 4})},
                                                                  {migraphx::sym::lit(2)}};
    auto input =
        mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, input_dims});
    migraphx::shape output_shape{
        std::vector<migraphx::shape>{migraphx::shape{migraphx::shape::float_type, {1, 1}}}};
    auto select = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
        {input},
        {first, second});
    auto output = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    mm->add_return({output});
    p.compile(migraphx::make_target("ref"));

    auto run_case = [&](std::size_t n, std::vector<float> data, float expected) {
        migraphx::parameter_map params;
        params["data"] =
            migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n, 2}}, data.data()};
        auto result = p.eval(params).back();
        std::vector<float> result_data;
        result.visit([&](auto values) { result_data.assign(values.begin(), values.end()); });
        EXPECT(migraphx::verify::verify_rms_range(result_data, std::vector<float>{expected}));
    };
    run_case(2, {1, 2, 3, 4}, 10);
    run_case(3, {1, 2, 3, 4, 5, 6}, 21);
}

TEST_CASE(select_module_prefers_min_range_clone)
{
    migraphx::program p;
    auto create_submodule =
        [&](std::size_t min_n, std::size_t max_n, const std::string& module_name) {
            auto* submod                                         = p.create_module(module_name);
            std::vector<migraphx::shape::dynamic_dimension> dims = {
                {migraphx::sym::var("n", {min_n, max_n})}, {migraphx::sym::lit(2)}};
            auto input =
                submod->add_parameter("data", migraphx::shape{migraphx::shape::float_type, dims});
            auto output =
                submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), input);
            submod->add_return({output});
            return submod;
        };
    auto* min_clone = create_submodule(1, 1, "min_clone");
    auto* max_clone = create_submodule(2, 64, "max_clone");

    auto* mm                                                   = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> input_dims = {
        {migraphx::sym::var("n", {1, 64})}, {migraphx::sym::lit(2)}};
    auto input =
        mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, input_dims});
    migraphx::shape output_shape{
        std::vector<migraphx::shape>{migraphx::shape{migraphx::shape::float_type, {1, 1}}}};
    auto select = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
        {input},
        {min_clone, max_clone});
    auto output = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    mm->add_return({output});
    p.compile(migraphx::make_target("ref"));

    auto run_case = [&](std::size_t n, std::vector<float> data, float expected) {
        migraphx::parameter_map params;
        params["data"] =
            migraphx::argument{migraphx::shape{migraphx::shape::float_type, {n, 2}}, data.data()};
        auto result = p.eval(params).back();
        std::vector<float> result_data;
        result.visit([&](auto values) { result_data.assign(values.begin(), values.end()); });
        EXPECT(migraphx::verify::verify_rms_range(result_data, std::vector<float>{expected}));
    };
    run_case(1, {3, 4}, 7);
    run_case(2, {3, 4, 5, 6}, 18);
}

TEST_CASE(select_module_tuple_output_maps_nonconsecutive_returns)
{
    migraphx::program p;
    auto* submod = p.create_module("fused");
    migraphx::shape scalar_shape{migraphx::shape::int32_type, {1}};
    migraphx::shape seq_shape{migraphx::shape::int32_type, {64}};
    migraphx::shape tuple_shape{{seq_shape, seq_shape}};
    submod->add_parameter("data", seq_shape);
    auto fused = submod->add_parameter("#output_fused", tuple_shape);
    auto zero  = submod->add_literal(migraphx::literal{scalar_shape, {0}});
    auto first =
        submod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
    auto second =
        submod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
    submod->add_return({zero, second, zero, first});

    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("data", seq_shape);
    migraphx::shape output_dyn{
        std::vector<migraphx::shape>{scalar_shape, seq_shape, scalar_shape, seq_shape}};
    auto output0 = mm->add_parameter("output0", scalar_shape);
    auto output1 = mm->add_parameter("output1", seq_shape);
    auto output2 = mm->add_parameter("output2", scalar_shape);
    auto output3 = mm->add_parameter("output3", seq_shape);
    auto select  = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(output_dyn)}}),
        {input, output0, output1, output2, output3},
        {submod});
    auto out0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto out1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), select);
    auto out2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), select);
    auto out3 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 3}}), select);
    mm->add_return({out0, out1, out2, out3});
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> data_vec(64, 1);
    std::vector<int32_t> scalar0(1, 0);
    std::vector<int32_t> seq1(64, 0);
    std::vector<int32_t> scalar2(1, 0);
    std::vector<int32_t> seq3(64, 0);
    migraphx::parameter_map params;
    params["data"]    = migraphx::argument{seq_shape, data_vec.data()};
    params["output0"] = migraphx::argument{scalar_shape, scalar0.data()};
    params["output1"] = migraphx::argument{seq_shape, seq1.data()};
    params["output2"] = migraphx::argument{scalar_shape, scalar2.data()};
    params["output3"] = migraphx::argument{seq_shape, seq3.data()};
    auto results      = p.eval(params);
    EXPECT(results.size() == 4);
    EXPECT(results[0].get_shape() == scalar_shape);
    EXPECT(results[1].get_shape() == seq_shape);
    EXPECT(results[2].get_shape() == scalar_shape);
    EXPECT(results[3].get_shape() == seq_shape);
}

TEST_CASE(select_module_nested_select_reordered_output_buffers)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {4}};

    auto* inner = p.create_module("inner");
    inner->add_parameter("data", s);
    auto inner_output0 = inner->add_parameter("#output_0", s);
    auto inner_output1 = inner->add_parameter("#output_1", s);
    inner->add_return({inner_output0, inner_output1});

    auto* outer     = p.create_module("outer");
    auto outer_data = outer->add_parameter("data", s);
    auto fused      = outer->add_parameter("#output_fused", migraphx::shape{{s, s}});
    auto first = outer->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
    auto second =
        outer->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
    auto inner_select = outer->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{{s, s}})}}),
        {outer_data, second, first},
        {inner});
    auto inner_result0 =
        outer->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), inner_select);
    auto inner_result1 =
        outer->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), inner_select);
    outer->add_return({inner_result0, inner_result1});

    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", s);
    auto output0 = mm->add_parameter("output0", s);
    auto output1 = mm->add_parameter("output1", s);
    auto select  = mm->add_instruction(
        migraphx::make_op("select_module",
                           {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{{s, s}})}}),
        {data, output0, output1},
        {outer});
    auto result0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto result1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), select);
    mm->add_return({result0, result1});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data_vec(s.elements(), 1);
    std::vector<float> output0_vec(s.elements(), 0);
    std::vector<float> output1_vec(s.elements(), 0);
    migraphx::parameter_map params;
    params["data"]    = migraphx::argument{s, data_vec.data()};
    params["output0"] = migraphx::argument{s, output0_vec.data()};
    params["output1"] = migraphx::argument{s, output1_vec.data()};
    auto results      = p.eval(params);
    EXPECT(results.size() == 2);
    EXPECT(results[0].data() == params.at("output0").data());
    EXPECT(results[1].data() == params.at("output1").data());
}

TEST_CASE(select_module_nested_tuple_output_parameter)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {4}};
    migraphx::shape nested_shape{{migraphx::shape{{s, s}}, s}};

    auto* submod = p.create_module("nested_tuple");
    submod->add_parameter("data", s);
    auto fused = submod->add_parameter("#output_fused", nested_shape);
    auto pair = submod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
    auto first = submod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pair);
    auto second =
        submod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), pair);
    auto third =
        submod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
    submod->add_return({third, second, first});

    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", s);
    auto output0 = mm->add_parameter("output0", s);
    auto output1 = mm->add_parameter("output1", s);
    auto output2 = mm->add_parameter("output2", s);
    auto select  = mm->add_instruction(
        migraphx::make_op("select_module",
                           {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{{s, s, s}})}}),
        {data, output0, output1, output2},
        {submod});
    auto result0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto result1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), select);
    auto result2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), select);
    mm->add_return({result0, result1, result2});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data_vec(s.elements(), 1);
    std::vector<float> output0_vec(s.elements(), 0);
    std::vector<float> output1_vec(s.elements(), 0);
    std::vector<float> output2_vec(s.elements(), 0);
    migraphx::parameter_map params;
    params["data"]    = migraphx::argument{s, data_vec.data()};
    params["output0"] = migraphx::argument{s, output0_vec.data()};
    params["output1"] = migraphx::argument{s, output1_vec.data()};
    params["output2"] = migraphx::argument{s, output2_vec.data()};
    auto results      = p.eval(params);
    EXPECT(results.size() == 3);
    EXPECT(results[0].data() == params.at("output0").data());
    EXPECT(results[1].data() == params.at("output1").data());
    EXPECT(results[2].data() == params.at("output2").data());
}

TEST_CASE(select_module_unaliased_output_error)
{
    migraphx::program p;
    auto* submod = p.create_module("sub");
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto data = submod->add_parameter("data", s);
    submod->add_parameter("#output_0", s);
    auto neg = submod->add_instruction(migraphx::make_op("neg"), data);
    submod->add_return({neg});

    auto* mm   = p.get_main_module();
    auto input = mm->add_parameter("data", s);
    migraphx::shape output_shape{std::vector<migraphx::shape>{s}};
    auto output = mm->add_parameter("output", s);
    auto select = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
        {input, output},
        {submod});
    auto out = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    mm->add_return({out});
    EXPECT(test::throws<migraphx::exception>([&] { p.compile(migraphx::make_target("ref")); },
                                             "does not alias a module output"));
}

TEST_CASE(select_module_repeated_symbol_mismatch_error)
{
    migraphx::program p;
    auto* submod = p.create_module("square");
    auto n       = migraphx::sym::var("n", {1, 4});
    auto submod_input =
        submod->add_parameter("data",
                              migraphx::shape{migraphx::shape::float_type,
                                              {migraphx::shape::dynamic_dimension{n},
                                               migraphx::shape::dynamic_dimension{n}}});
    auto submod_output =
        submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), submod_input);
    submod->add_return({submod_output});

    auto* mm = p.get_main_module();
    auto input =
        mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {1, 4}}});
    migraphx::shape output_shape{
        std::vector<migraphx::shape>{migraphx::shape{migraphx::shape::float_type, {1, 1}}}};
    auto select = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
        {input},
        {submod});
    auto output = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    mm->add_return({output});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{1, 2, 3, 4, 5, 6};
    migraphx::parameter_map params;
    params["data"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, {2, 2}}, data.data()};
    auto result = p.eval(params).back();
    std::vector<float> result_data;
    result.visit([&](auto output) { result_data.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_data, std::vector<float>{10}));

    params["data"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, {2, 3}}, data.data()};
    EXPECT(test::throws([&] { std::ignore = p.eval(params).back(); }));
}

TEST_CASE(select_module_static_stride_mismatch_error)
{
    migraphx::program p;
    auto* submod = p.create_module("nonstandard");
    migraphx::shape submod_shape{migraphx::shape::float_type, {2, 2}, {1, 2}};
    auto submod_input = submod->add_parameter("data", submod_shape);
    auto submod_output =
        submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), submod_input);
    submod->add_return({submod_output});

    auto* mm = p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type, {{2, 2}, {2, 2}}};
    auto input = mm->add_parameter("data", input_shape);
    migraphx::shape output_shape{
        std::vector<migraphx::shape>{migraphx::shape{migraphx::shape::float_type, {1, 1}}}};
    auto select = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
        {input},
        {submod});
    auto output = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    mm->add_return({output});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{1, 2, 3, 4};
    migraphx::parameter_map params;
    params["data"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, {2, 2}}, data.data()};
    EXPECT(test::throws([&] { std::ignore = p.eval(params).back(); }));
}

TEST_CASE(select_module_empty_stride_mismatch)
{
    migraphx::module submodule{"empty"};
    auto expected = migraphx::shape{migraphx::shape::int64_type, {1, 0}, {0, 1}};
    auto input    = submodule.add_parameter("data", expected);
    submodule.add_return({input});

    migraphx::op::select_module select;
    select.output_dyn_shapes = migraphx::shape{std::vector<migraphx::shape>{expected}};
    auto actual              = migraphx::shape{migraphx::shape::int64_type, {1, 0}, {100, 1}};
    bool invoked             = false;
    migraphx::context ctx;
    select.finalize(ctx, {}, {actual}, {&submodule});
    auto result              = select.compute(
        {}, {migraphx::argument{actual, nullptr}}, {&submodule}, [&](auto&, const auto& params) {
            invoked = true;
            EXPECT(params.at("data").get_shape() == expected);
            return std::vector<migraphx::argument>{params.at("data")};
        });
    EXPECT(invoked);
    EXPECT(result.get_sub_objects().front().get_shape() == expected);
}

TEST_CASE(select_module_unit_stride_mismatch)
{
    migraphx::module submodule{"unit"};
    auto expected = migraphx::shape{migraphx::shape::int64_type, {1, 1}, {1, 1}};
    auto input    = submodule.add_parameter("data", expected);
    submodule.add_return({input});

    migraphx::op::select_module select;
    select.output_dyn_shapes  = migraphx::shape{std::vector<migraphx::shape>{expected}};
    auto actual               = migraphx::shape{migraphx::shape::int64_type, {1, 1}, {4, 1}};
    std::vector<int64_t> data = {42};
    bool invoked              = false;
    migraphx::context ctx;
    select.finalize(ctx, {}, {actual}, {&submodule});
    auto result               = select.compute({},
                                               {migraphx::argument{actual, data.data()}},
                                               {&submodule},
                                               [&](auto&, const auto& params) {
                                     invoked = true;
                                     EXPECT(params.at("data").get_shape() == expected);
                                     return std::vector<migraphx::argument>{params.at("data")};
                                               });
    EXPECT(invoked);
    EXPECT(result.get_sub_objects().front().get_shape() == expected);
    EXPECT(result.get_sub_objects().front().at<int64_t>() == 42);
}

TEST_CASE(select_module_not_found_error)
{
    migraphx::program p;

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 2, 2}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto reduce_ins =
            submod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sm_input);
        auto squeeze_ins =
            submod->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), reduce_ins);
        submod->add_return({squeeze_ins});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}};
    auto input                              = mm->add_parameter("data", s);
    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {2, 2}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {batch1, batch2, batch3, batch4});
    auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    mm->add_return({ret});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-4, 8, -1, 4, -1, 8,  8,  -4, -4, 8,
                                  -1, 4, -1, 8, 8,  -4, -1, 8,  8,  -4};
    migraphx::parameter_map params;
    migraphx::shape input_fixed_shape{migraphx::shape::float_type, {5, 2, 2}};
    params["data"] = migraphx::argument(input_fixed_shape, input_data.data());
    EXPECT(test::throws([&] { std::ignore = p.eval(params).back(); }));
}

TEST_CASE(select_module_nested_output_buffers)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 4}};
    migraphx::shape output_shape{std::vector<migraphx::shape>{s, s}};

    auto* inner = p.create_module("inner");
    inner->add_parameter("data", s);
    auto inner_output0 = inner->add_parameter("inner:#output_0", s);
    auto inner_output1 = inner->add_parameter("inner:#output_1", s);
    inner->add_return({inner_output0, inner_output1});

    auto* outer        = p.create_module("outer");
    auto outer_data    = outer->add_parameter("data", s);
    auto outer_output0 = outer->add_parameter("outer:#output_0", s);
    auto outer_output1 = outer->add_parameter("outer:#output_1", s);
    auto inner_select  = outer->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
        {outer_data, outer_output0, outer_output1},
        {inner});
    auto inner_result0 =
        outer->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), inner_select);
    auto inner_result1 =
        outer->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), inner_select);
    outer->add_return({inner_result0, inner_result1});

    auto* mm          = p.get_main_module();
    auto data         = mm->add_parameter("data", s);
    auto buffer0      = mm->add_parameter("buffer0", s);
    auto buffer1      = mm->add_parameter("buffer1", s);
    auto outer_select = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
        {data, buffer0, buffer1},
        {outer});
    auto result0 =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), outer_select);
    auto result1 =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), outer_select);
    mm->add_return({result0, result1});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> data_values(s.elements(), 0);
    std::vector<float> buffer0_values(s.elements(), 1);
    std::vector<float> buffer1_values(s.elements(), 2);
    migraphx::parameter_map params;
    params["data"]    = migraphx::argument{s, data_values.data()};
    params["buffer0"] = migraphx::argument{s, buffer0_values.data()};
    params["buffer1"] = migraphx::argument{s, buffer1_values.data()};
    auto results      = p.eval(params);
    EXPECT(migraphx::verify::verify_rms_range(results.at(0).to_vector<float>(), buffer0_values));
    EXPECT(migraphx::verify::verify_rms_range(results.at(1).to_vector<float>(), buffer1_values));
}

TEST_CASE(select_module_zero_output_short_circuit)
{
    auto n = migraphx::sym::var("n", {0, 4});
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape input_shape{migraphx::shape::float_type,
                                {migraphx::shape::dynamic_dimension{n},
                                 migraphx::shape::dynamic_dimension{migraphx::sym::lit(2)}}};
    auto input                           = mm->add_parameter("data", input_shape);
    auto twice_n                         = n * migraphx::sym::lit(2);
    std::vector<migraphx::shape> outputs = {
        {migraphx::shape::float_type,
         {migraphx::shape::dynamic_dimension{twice_n},
          migraphx::shape::dynamic_dimension{migraphx::sym::lit(2)}}},
        {migraphx::shape::int64_type,
         {migraphx::shape::dynamic_dimension{migraphx::sym::lit(2)},
          migraphx::shape::dynamic_dimension{n}}}};
    auto select = mm->add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{outputs})},
             {"logical_output_dyn_shapes", migraphx::to_value(migraphx::shape{outputs})},
             {"num_inputs", 1}}),
        {input},
        {});
    auto output0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto output1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), select);
    mm->add_return({output0, output1});
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    params["data"] =
        migraphx::argument{migraphx::shape{migraphx::shape::float_type, {0, 2}}, nullptr};
    auto results = p.eval(params);
    EXPECT(results.at(0).get_shape() == migraphx::shape{migraphx::shape::float_type, {0, 2}});
    EXPECT(results.at(1).get_shape() == migraphx::shape{migraphx::shape::int64_type, {2, 0}});
}

TEST_CASE(select_module_mixed_empty_concat_runs)
{
    migraphx::program p;
    auto* submod = p.create_module("mixed_concat");
    auto data    = submod->add_parameter("data", {migraphx::shape::float_type, {2, 3}});
    auto empty   = submod->add_parameter("empty", {migraphx::shape::float_type, {0, 3}});
    auto concat  = submod->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), empty, data);
    submod->add_return({empty, concat});

    auto* mm        = p.get_main_module();
    auto main_data  = mm->add_parameter("data", {migraphx::shape::float_type, {2, 3}});
    auto main_empty = mm->add_parameter("empty", {migraphx::shape::float_type, {0, 3}});
    std::vector<migraphx::shape> outputs = {migraphx::shape{migraphx::shape::float_type, {0, 3}},
                                            migraphx::shape{migraphx::shape::float_type, {2, 3}}};
    auto select                          = mm->add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{outputs})},
             {"logical_output_dyn_shapes", migraphx::to_value(migraphx::shape{outputs})},
             {"num_inputs", 2}}),
        {main_data, main_empty},
        {submod});
    auto empty_output =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto data_output =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), select);
    mm->add_return({empty_output, data_output});
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data_values = {1, 2, 3, 4, 5, 6};
    migraphx::parameter_map params;
    params["data"]  = migraphx::argument{{migraphx::shape::float_type, {2, 3}}, data_values.data()};
    params["empty"] = migraphx::argument{{migraphx::shape::float_type, {0, 3}}, nullptr};
    auto results    = p.eval(params);
    EXPECT(results.at(0).get_shape() == migraphx::shape{migraphx::shape::float_type, {0, 3}});
    EXPECT(results.at(1).to_vector<float>() == data_values);
}
