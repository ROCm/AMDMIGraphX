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
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/raw_data.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include <algorithm>
#include <unordered_map>

// Writes its first input into its last one and returns the last one, like a lowered kernel
// writing into an output buffer
struct copy_into_op
{
    std::string name() const { return "copy_into"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        return inputs.back();
    }

    migraphx::argument compute(const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        migraphx::visit_all(args.back(), args.front())(
            [](auto output, auto input) { std::copy(input.begin(), input.end(), output.begin()); });
        return args.back();
    }

    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>& inputs) const
    {
        return {inputs.size() - 1};
    }
};

TEST_CASE(select_module_output_parameter_reshapes_larger_buffer)
{
    migraphx::program p;
    std::vector<migraphx::module_ref> candidates;
    for(std::size_t n : {1, 2})
    {
        auto name = "dim_" + std::to_string(n);
        migraphx::shape s{migraphx::shape::float_type, {n, 4}};
        auto* submod = p.create_module(name);
        auto data    = submod->add_parameter("data", s);
        auto output  = submod->add_parameter(name + ":#output_0", s);
        auto neg     = submod->add_instruction(migraphx::make_op("neg"), data);
        submod->add_return({submod->add_instruction(copy_into_op{}, neg, output)});
        candidates.push_back(submod);
    }

    auto* mm = p.get_main_module();
    migraphx::shape dyn_s{migraphx::shape::float_type, {{1, 2}, {4, 4}}};
    migraphx::shape max_s{migraphx::shape::float_type, {2, 4}};
    auto data   = mm->add_parameter("data", dyn_s);
    auto output = mm->add_parameter("output", max_s);
    auto select = mm->add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{dyn_s}})}}),
        {data, output},
        candidates);
    mm->add_return(
        {mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select)});

    migraphx::shape data_s{migraphx::shape::float_type, {1, 4}};
    std::vector<float> data_vec{1, 2, 3, 4};
    std::vector<float> output_vec(max_s.elements(), 0);
    migraphx::parameter_map params;
    params["data"]   = migraphx::argument{data_s, data_vec.data()};
    params["output"] = migraphx::argument{max_s, output_vec.data()};
    auto results     = p.eval(params);
    EXPECT(results.size() == 1);
    EXPECT(results.front().get_shape() == data_s);
    EXPECT(results.front().data() == params.at("output").data());
    EXPECT(results.front().to_vector<float>() == std::vector<float>{-1, -2, -3, -4});
}

TEST_CASE(select_module_output_parameter_buffer_too_small_error)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 4}};
    auto* submod = p.create_module("dim_2");
    auto sdata   = submod->add_parameter("data", s);
    auto soutput = submod->add_parameter("dim_2:#output_0", s);
    submod->add_return({submod->add_instruction(copy_into_op{}, sdata, soutput)});

    auto* mm = p.get_main_module();
    migraphx::shape dyn_s{migraphx::shape::float_type, {{1, 2}, {4, 4}}};
    auto data   = mm->add_parameter("data", s);
    auto output = mm->add_parameter("output", dyn_s);
    auto select = mm->add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{dyn_s}})}}),
        {data, output},
        {submod});
    mm->add_return(
        {mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select)});

    migraphx::shape small_s{migraphx::shape::float_type, {1, 4}};
    std::vector<float> data_vec(s.elements(), 1);
    std::vector<float> output_vec(small_s.elements(), 0);
    migraphx::parameter_map params;
    params["data"]   = migraphx::argument{s, data_vec.data()};
    params["output"] = migraphx::argument{small_s, output_vec.data()};
    EXPECT(test::throws<migraphx::exception>([&] { std::ignore = p.eval(params); },
                                             "output buffer for \"dim_2:#output_0\""));
}

TEST_CASE(select_module_missing_output_allocations_error)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto* submod = p.create_module("sub");
    auto sdata   = submod->add_parameter("data", s);
    auto soutput = submod->add_parameter("sub:#output_0", s);
    submod->add_return({submod->add_instruction(copy_into_op{}, sdata, soutput)});

    auto* mm    = p.get_main_module();
    auto data   = mm->add_parameter("data", s);
    auto select = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes",
                            migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{s}})}}),
        {data},
        {submod});
    mm->add_return(
        {mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select)});

    std::vector<float> data_vec(s.elements(), 1);
    migraphx::parameter_map params;
    params["data"] = migraphx::argument{s, data_vec.data()};
    EXPECT(test::throws<migraphx::exception>([&] { std::ignore = p.eval(params); },
                                             "missing output allocations"));
}

// Each return writes an element of the tuple parameter through a view of it, and the elements
// are returned in the opposite order
TEST_CASE(select_module_tuple_output_parameter_through_alias)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto* submod = p.create_module("fused");
    auto sdata   = submod->add_parameter("data", s);
    auto fused   = submod->add_parameter("fused:#output_0", migraphx::shape{{s, s}});
    auto first =
        submod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
    auto second =
        submod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
    auto neg = submod->add_instruction(migraphx::make_op("neg"), sdata);
    submod->add_return({submod->add_instruction(copy_into_op{}, sdata, second),
                        submod->add_instruction(copy_into_op{}, neg, first)});

    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", s);
    auto output0 = mm->add_parameter("output0", s);
    auto output1 = mm->add_parameter("output1", s);
    auto select  = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{{s, s}})}}),
        {data, output0, output1},
        {submod});
    auto out0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    auto out1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), select);
    mm->add_return({out0, out1});

    std::vector<float> data_vec{1, 2, 3, 4};
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
    EXPECT(output0_vec == data_vec);
    EXPECT(output1_vec == std::vector<float>{-1, -2, -3, -4});
}

// One return is the tuple parameter itself and the other only reads it, so neither identifies the
// element it writes
TEST_CASE(select_module_tuple_output_parameter_unaliased_element_error)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto* submod = p.create_module("fused");
    submod->add_parameter("data", s);
    auto fused = submod->add_parameter("fused:#output_0", migraphx::shape{{s, s}});
    submod->add_return(
        {submod->add_instruction(pass_op{}, fused), submod->add_instruction(nop{}, fused)});

    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", s);
    auto output0 = mm->add_parameter("output0", s);
    auto output1 = mm->add_parameter("output1", s);
    auto select  = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{{s, s}})}}),
        {data, output0, output1},
        {submod});
    mm->add_return(
        {mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select)});

    std::vector<float> data_vec(s.elements(), 1);
    std::vector<float> output0_vec(s.elements(), 0);
    std::vector<float> output1_vec(s.elements(), 0);
    migraphx::parameter_map params;
    params["data"]    = migraphx::argument{s, data_vec.data()};
    params["output0"] = migraphx::argument{s, output0_vec.data()};
    params["output1"] = migraphx::argument{s, output1_vec.data()};
    EXPECT(test::throws<migraphx::exception>(
        [&] { std::ignore = p.eval(params); },
        "is not aliased by get_tuple_elem returns for every subobject"));
}

TEST_CASE(select_module_output_count_mismatch_error)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto* submod = p.create_module("sub");
    auto sdata   = submod->add_parameter("data", s);
    auto soutput = submod->add_parameter("sub:#output_0", s);
    submod->add_return({submod->add_instruction(copy_into_op{}, sdata, soutput)});

    auto* mm     = p.get_main_module();
    auto data    = mm->add_parameter("data", s);
    auto output0 = mm->add_parameter("output0", s);
    auto output1 = mm->add_parameter("output1", s);
    auto select  = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(migraphx::shape{{s, s}})}}),
        {data, output0, output1},
        {submod});
    mm->add_return(
        {mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select)});

    std::vector<float> data_vec(s.elements(), 1);
    std::vector<float> output0_vec(s.elements(), 0);
    std::vector<float> output1_vec(s.elements(), 0);
    migraphx::parameter_map params;
    params["data"]    = migraphx::argument{s, data_vec.data()};
    params["output0"] = migraphx::argument{s, output0_vec.data()};
    params["output1"] = migraphx::argument{s, output1_vec.data()};
    EXPECT(test::throws<migraphx::exception>([&] { std::ignore = p.eval(params); },
                                             "output allocation count does not match"));
}

// The operation interface selects the module matching the input shapes and passes its parameters
// by name
TEST_CASE(select_module_compute_maps_parameters_by_name)
{
    migraphx::shape small_s{migraphx::shape::float_type, {1, 4}};
    migraphx::shape s{migraphx::shape::float_type, {2, 4}};
    migraphx::shape max_s{migraphx::shape::float_type, {4, 4}};
    migraphx::module small{"small"};
    auto small_data   = small.add_parameter("data", small_s);
    auto small_output = small.add_parameter("small:#output_0", small_s);
    small.add_return({small.add_instruction(copy_into_op{}, small_data, small_output)});
    migraphx::module submod{"sub"};
    auto sdata   = submod.add_parameter("data", s);
    auto soutput = submod.add_parameter("sub:#output_0", s);
    submod.add_return({submod.add_instruction(copy_into_op{}, sdata, soutput)});

    migraphx::shape dyn_s{migraphx::shape::float_type, {{1, 2}, {4, 4}}};
    auto op = migraphx::make_op(
        "select_module",
        {{"output_dyn_shapes",
          migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{dyn_s}})}});
    std::vector<float> data_vec(s.elements(), 1);
    std::vector<float> output_vec(max_s.elements(), 0);
    migraphx::argument data_arg{s, data_vec.data()};
    migraphx::argument output_arg{max_s, output_vec.data()};
    std::unordered_map<std::string, migraphx::argument> seen;
    auto result = op.compute(migraphx::shape{std::vector<migraphx::shape>{s}},
                             {data_arg, output_arg},
                             {&small, &submod},
                             [&](const migraphx::module_ref& m, const auto& params) {
                                 EXPECT(m == &submod);
                                 seen = params;
                                 return std::vector<migraphx::argument>{params.at("sub:#output_0")};
                             });
    EXPECT(seen.size() == 2);
    EXPECT(seen.at("data").data() == data_arg.data());
    EXPECT(seen.at("sub:#output_0").get_shape() == s);
    EXPECT(seen.at("sub:#output_0").data() == output_arg.data());
    EXPECT(result.get_sub_objects().size() == 1);
    EXPECT(result.get_sub_objects().front().get_shape() == s);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
