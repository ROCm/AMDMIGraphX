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
#include <migraphx/simplify_dyn_ops.hpp>
#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_dyn_ops{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(static_broadcast)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {4}}};
        auto literal_ins   = m0.add_literal(migraphx::literal{lit_s, {6, 5, 4, 3}});
        auto broadcast_lit = m0.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s.lens()}}), literal_ins);
        auto add_ins = m0.add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        m0.add_return({add_ins});
    }

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4}};
        auto input = m1.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {4}}};
        auto literal_ins = m1.add_literal(migraphx::literal{lit_s, {6, 5, 4, 3}});
        auto broadcast_lit =
            m1.add_instruction(migraphx::make_op("broadcast", {{"axis", 1}}), literal_ins, input);
        auto add_ins = m1.add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        m1.add_return({add_ins});
    }
    run_pass(m1);

    EXPECT(m0 == m1);
}

TEST_CASE(static_multibroadcast)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}, {0}}};
        auto literal_ins   = m0.add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit = m0.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), literal_ins);
        auto add_ins = m0.add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        m0.add_return({add_ins});
    }

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4}};
        auto input = m1.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}, {0}}};
        auto literal_ins = m1.add_literal(migraphx::literal{lit_s, {6}});
        auto broadcast_lit =
            m1.add_instruction(migraphx::make_op("multibroadcast"), literal_ins, input);
        auto add_ins = m1.add_instruction(migraphx::make_op("add"), input, broadcast_lit);
        m1.add_return({add_ins});
    }
    run_pass(m1);

    EXPECT(m0 == m1);
}

TEST_CASE(split_broadcast_match)
{
    migraphx::program p0;
    {
        auto* mm0 = p0.get_main_module();

        // create batch submodules
        auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
            auto* submod = p0.create_module(module_name);
            migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
            auto sm_input = submod->add_parameter("data", sm_shape);
            migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {4}}};
            auto literal_ins   = submod->add_literal(migraphx::literal{lit_s, {6, 5, 4, 3}});
            auto broadcast_lit = submod->add_instruction(
                migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", sm_shape.lens()}}),
                literal_ins);
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

    migraphx::program p1;
    {
        auto* mm1 = p1.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
        auto input1 = mm1->add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {4}}};
        auto literal_ins   = mm1->add_literal(migraphx::literal{lit_s, {6, 5, 4, 3}});
        auto broadcast_lit = mm1->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}}), literal_ins, input1);
        auto add_ins = mm1->add_instruction(migraphx::make_op("add"), input1, broadcast_lit);
        mm1->add_return({add_ins});
    }
    migraphx::run_passes(p1,
                         {migraphx::split_single_dyn_dim{},
                          migraphx::dead_code_elimination{},
                          migraphx::simplify_dyn_ops{}});

    EXPECT(p0 == p1);
}

TEST_CASE(const_slice_3input)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m0.add_parameter("data", s);
        auto slice_ins = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m0.add_return({slice_ins});
    }

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m1.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m1.add_literal(migraphx::literal{s1, {0}});
        auto input_ends   = m1.add_literal(migraphx::literal{s1, {3}});
        auto slice_ins    = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}}), input, input_starts, input_ends);
        m1.add_return({slice_ins});
    }
    run_pass(m1);

    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_4input)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m0.add_parameter("data", s);
        auto slice_ins = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m0.add_return({slice_ins});
    }

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m1.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m1.add_literal(migraphx::literal{s1, {0}});
        auto input_ends   = m1.add_literal(migraphx::literal{s1, {3}});
        auto input_axes   = m1.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins    = m1.add_instruction(
            migraphx::make_op("slice"), input, input_starts, input_ends, input_axes);
        m1.add_return({slice_ins});
    }
    run_pass(m1);

    EXPECT(m0 == m1);
}

TEST_CASE(static_dimensions_of)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4, 4}};
        m0.add_parameter("data", s);
        migraphx::shape lit_shape{migraphx::shape::int64_type, {3}};
        ;
        std::vector<int64_t> lit_data = {2, 4, 4};
        auto lit_ins                  = m0.add_literal(migraphx::literal{lit_shape, lit_data});
        m0.add_return({lit_ins});
    }

    // dead_code_elimination will get rid of atan
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4, 4}};
        auto input    = m1.add_parameter("data", s);
        auto atan_ins = m1.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins =
            m1.add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), atan_ins);
        m1.add_return({dimensions_of_ins});
    }
    run_pass(m1);

    EXPECT(m0 == m1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
