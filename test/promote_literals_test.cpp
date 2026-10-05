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

#include <migraphx/promote_literals.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>

#include <algorithm>

static void run_promote(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::promote_literals{}, migraphx::dead_code_elimination{}});
}

static void run_promote_and_ecs(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::promote_literals{},
                          migraphx::dead_code_elimination{},
                          migraphx::eliminate_common_subexpression{},
                          migraphx::dead_code_elimination{}});
}

TEST_CASE(promote_only)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins = submod->add_literal(migraphx::literal{lit_s, {6}});
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            submod->add_return({add_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0                             = mm0->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm0->add_return({ret});
    }
    run_promote(p0);

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm1->add_literal(migraphx::literal{lit_s, {6}});

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p1.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            submod->add_return({add_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0 = mm1->insert_parameter(std::next(literal_ins), "data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm1->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm1->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm1->add_return({ret});
    }

    EXPECT(p0 == p1);
}

TEST_CASE(promote_and_ecs0)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins = submod->add_literal(migraphx::literal{lit_s, {6}});
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            submod->add_return({add_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0                             = mm0->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm0->add_return({ret});
    }
    run_promote_and_ecs(p0);

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins = mm1->add_literal(migraphx::literal{lit_s, {6}});

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p1.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            auto broadcast_lit =
                submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
            auto add_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
            submod->add_return({add_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");
        auto* dim3 = create_submodule(3, "dim_3");
        auto* dim4 = create_submodule(4, "dim_4");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0 = mm1->insert_parameter(std::next(literal_ins), "data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm1->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2, dim3, dim4});
        auto ret =
            mm1->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm1->add_return({ret});
    }

    EXPECT(p0 == p1);
}

TEST_CASE(promote_and_ecs1)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
            auto literal_ins0   = submod->add_literal(migraphx::literal{lit_s, {6}});
            auto literal_ins1   = submod->add_literal(migraphx::literal{lit_s, {2}});
            auto broadcast_lit0 = submod->add_instruction(
                migraphx::make_op("multibroadcast"), literal_ins0, sm_input);
            auto broadcast_lit1 = submod->add_instruction(
                migraphx::make_op("multibroadcast"), literal_ins1, sm_input);
            auto add_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit0);
            auto mul_ins =
                submod->add_instruction(migraphx::make_op("mul"), add_ins, broadcast_lit1);
            submod->add_return({mul_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0                             = mm0->add_parameter("data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm0->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2});
        auto ret =
            mm0->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm0->add_return({ret});
    }
    run_promote_and_ecs(p0);

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
        auto literal_ins1 = mm1->add_literal(migraphx::literal{lit_s, {2}});
        auto literal_ins0 = mm1->add_literal(migraphx::literal{lit_s, {6}});

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p1.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input       = submod->add_parameter("data", sm_shape);
            auto broadcast_lit0 = submod->add_instruction(
                migraphx::make_op("multibroadcast"), literal_ins0, sm_input);
            auto broadcast_lit1 = submod->add_instruction(
                migraphx::make_op("multibroadcast"), literal_ins1, sm_input);
            auto add_ins =
                submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit0);
            auto mul_ins =
                submod->add_instruction(migraphx::make_op("mul"), add_ins, broadcast_lit1);
            submod->add_return({mul_ins});
            return submod;
        };
        auto* dim1 = create_submodule(1, "dim_1");
        auto* dim2 = create_submodule(2, "dim_2");

        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input0 = mm1->insert_parameter(std::next(literal_ins1), "data", s);
        std::vector<migraphx::shape> sub_shapes = {};
        sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        migraphx::shape out_attr = migraphx::shape{sub_shapes};
        auto sm_ins              = mm1->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            {input0},
            {dim1, dim2});
        auto ret =
            mm1->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
        mm1->add_return({ret});
    }

    EXPECT(p0 == p1);
}

TEST_CASE(promote_grouped_literals)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape scalar_shape{migraphx::shape::float_type, {1}};
    migraphx::shape vector_shape{migraphx::shape::float_type, {4}};
    migraphx::literal shared_vector{vector_shape, {1.0f, 2.0f, 3.0f, 4.0f}};

    auto create_submodule = [&](std::size_t batch_size, const std::string& name) {
        auto* submodule = p.create_module(name);
        auto input =
            submodule->add_parameter("data", {migraphx::shape::float_type, {batch_size, 4}});
        auto scalar = submodule->add_literal(migraphx::literal{scalar_shape, {6.0f}});
        auto vector = submodule->add_literal(shared_vector);
        auto scalar_broadcast =
            submodule->add_instruction(migraphx::make_op("multibroadcast"), scalar, input);
        auto vector_broadcast =
            submodule->add_instruction(migraphx::make_op("multibroadcast"), vector, input);
        auto add_scalar =
            submodule->add_instruction(migraphx::make_op("add"), input, scalar_broadcast);
        auto add_vector =
            submodule->add_instruction(migraphx::make_op("add"), add_scalar, vector_broadcast);
        submodule->add_return({add_vector});
        return submodule;
    };

    auto* dim1 = create_submodule(1, "grouped_dim_1");
    auto* dim2 = create_submodule(2, "grouped_dim_2");
    migraphx::shape input_shape{migraphx::shape::float_type, {{1, 2}, {4, 4}}};
    auto input = mm->add_parameter("data", input_shape);
    migraphx::shape output_shape{std::vector<migraphx::shape>{
        migraphx::shape{migraphx::shape::float_type, {{1, 2}, {4, 4}}}}};
    auto selected = mm->add_instruction(
        migraphx::make_op("select_module",
                          {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
        {input},
        {dim1, dim2});
    auto output =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), selected);
    mm->add_return({output});

    run_promote(p);

    EXPECT(std::count_if(mm->begin(), mm->end(), [](const auto& ins) {
               return ins.name() == "@literal";
           }) == 2);
    EXPECT(std::count_if(mm->begin(), mm->end(), [&](const auto& ins) {
               return ins.name() == "@literal" and ins.get_shape() == scalar_shape;
           }) == 1);
    EXPECT(std::count_if(mm->begin(), mm->end(), [&](const auto& ins) {
               return ins.name() == "@literal" and ins.get_shape() == vector_shape;
           }) == 1);
    EXPECT(std::none_of(
        dim1->begin(), dim1->end(), [](const auto& ins) { return ins.name() == "@literal"; }));
    EXPECT(std::none_of(
        dim2->begin(), dim2->end(), [](const auto& ins) { return ins.name() == "@literal"; }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
