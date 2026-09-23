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
