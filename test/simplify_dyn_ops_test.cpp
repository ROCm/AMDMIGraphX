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

TEST_CASE(resize)
{
    migraphx::module m0;
    {
        std::vector<float> ds = {1.f, 1.f, 0.601, 0.601};
        migraphx::shape ss{migraphx::shape::float_type, {4}};

        auto li = m0.add_literal(migraphx::literal{ss, ds});
        m0.add_instruction(migraphx::make_op("undefined"));

        migraphx::shape sx{migraphx::shape::float_type, {{1, 4, {1, 4}}, {1, 1}, {5, 5}, {9, 9}}};
        auto inx = m0.add_parameter("X", sx);

        auto r =
            m0.add_instruction(migraphx::make_op("resize",
                                                {{"mode", "nearest"},
                                                {"nearest_mode", "floor"},
                                                {"coordinate_transformation_mode", "asymmetric"}}),
                                inx,
                                li);

        m0.add_return({r});
    }
    run_pass(m0);
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

TEST_CASE(after_split_dyn_broadcast_match)
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

TEST_CASE(const_slice_2input_ends_axes)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"ends", {3}}, {"axes", {0}}}), input, input_starts);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_2input_starts_axes)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_ends = m0.add_literal(migraphx::literal{s1, {3}});
        auto slice_ins  = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"axes", {0}}}), input, input_ends);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_2input_starts_ends)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_axes = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins  = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}}), input, input_axes);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_3input_axes_only)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto input_ends   = m0.add_literal(migraphx::literal{s1, {3}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}}), input, input_starts, input_ends);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_3input_ends_only)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto input_axes   = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"ends", {3}}}), input, input_starts, input_axes);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_3inputs_starts_only)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_ends = m0.add_literal(migraphx::literal{s1, {3}});
        auto input_axes = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins  = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}}), input, input_ends, input_axes);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {6, 4, 4}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_2input_ends_axes_dyn)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {{6, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape s1{migraphx::shape::int32_type, {1}};
        auto input_starts = m0.add_literal(migraphx::literal{s1, {0}});
        auto slice_ins    = m0.add_instruction(
            migraphx::make_op("slice", {{"ends", {3}}, {"axes", {0}}}), input, input_starts);
        m0.add_return({slice_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {{6, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
        auto input     = m1.add_parameter("data", s);
        auto slice_ins = m1.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m1.add_return({slice_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(const_slice_3input_dyn)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {{6, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
        auto input     = m0.add_parameter("data", s);
        auto slice_ins = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {0}}, {"ends", {3}}, {"axes", {0}}}), input);
        m0.add_return({slice_ins});
    }

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {{6, 6}, {2, 4, {2, 4}}, {2, 4, {2, 4}}}};
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

TEST_CASE(static_dimensions_of0)
{
    // dead_code_elimination will get rid of atan
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4, 4}};
        auto input    = m0.add_parameter("data", s);
        auto atan_ins = m0.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins =
            m0.add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), atan_ins);
        m0.add_return({dimensions_of_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 4, 4}};
        m1.add_parameter("data", s);
        migraphx::shape lit_shape{migraphx::shape::int64_type, {3}};
        std::vector<int64_t> lit_data = {2, 4, 4};
        auto lit_ins                  = m1.add_literal(migraphx::literal{lit_shape, lit_data});
        m1.add_return({lit_ins});
    }
    EXPECT(m0 == m1);
}

TEST_CASE(static_dimensions_of1)
{
    // dead_code_elimination will get rid of atan
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2, 4}}, {4, 4}, {4, 4}}};
        auto input             = m0.add_parameter("data", s);
        auto atan_ins          = m0.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins = m0.add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}), atan_ins);
        m0.add_return({dimensions_of_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2, 4}}, {4, 4}, {4, 4}}};
        m1.add_parameter("data", s);
        migraphx::shape lit_shape{migraphx::shape::int64_type, {2}};
        std::vector<int64_t> lit_data = {4, 4};
        auto lit_ins                  = m1.add_literal(migraphx::literal{lit_shape, lit_data});
        m1.add_return({lit_ins});
    }

    EXPECT(m0 == m1);
}

// Does nothing because the dynamic_dimensions from start to end
// are not all fixed
TEST_CASE(static_dimensions_of_nonfixed)
{
    // dead_code_elimination will get rid of atan
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2, 4}}, {4, 8}, {4, 8}}};
        auto input             = m0.add_parameter("data", s);
        auto atan_ins          = m0.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins = m0.add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}), atan_ins);
        m0.add_return({dimensions_of_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4, {2, 4}}, {4, 8}, {4, 8}}};
        auto input             = m1.add_parameter("data", s);
        auto atan_ins          = m1.add_instruction(migraphx::make_op("atan"), input);
        auto dimensions_of_ins = m1.add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 3}}), atan_ins);
        m1.add_return({dimensions_of_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(constant_alloc_reshape)
{
    migraphx::module m0;
    {
        migraphx::shape s{migraphx::shape::float_type, {3, 32}};
        auto input = m0.add_parameter("data", s);
        migraphx::shape lit_s{migraphx::shape::int64_type, {3}};
        auto literal_ins = m0.add_literal(migraphx::literal{lit_s, {3, 4, 8}});
        auto alloc_ins   = m0.add_instruction(
            migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}),
            literal_ins);
        auto reshape_ins = m0.add_instruction(migraphx::make_op("reshape"), input, alloc_ins);
        m0.add_return({reshape_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {3, 32}};
        auto input = m1.add_parameter("data", s);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 4, 8}}}), input);
        m1.add_return({reshape_ins});
    }

    EXPECT(m0 == m1);
}

// A more contrived example to test static dimensions_of and constant reshape
TEST_CASE(static_dimensions_of_to_constant_alloc_reshape)
{
    migraphx::module m0;
    {
        migraphx::shape input_shape{migraphx::shape::float_type, {3, 4, 8}};
        auto x_param = m0.add_parameter("x", input_shape);
        auto dimensions_of_ins =
            m0.add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), x_param);
        migraphx::shape lit_shape{migraphx::shape::int64_type, {1}};
        auto lit0 = m0.add_literal(migraphx::literal{lit_shape, {0}});
        auto gather_ins =
            m0.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), dimensions_of_ins, lit0);
        auto slice_ins = m0.add_instruction(
            migraphx::make_op("slice", {{"starts", {1}}, {"ends", {3}}, {"axes", {0}}}),
            dimensions_of_ins);
        auto reduce_ins =
            m0.add_instruction(migraphx::make_op("reduce_prod", {{"axes", {0}}}), slice_ins);
        auto concat_ins =
            m0.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), gather_ins, reduce_ins);
        auto alloc_ins = m0.add_instruction(
            migraphx::make_op("allocate", {{"buf_type", migraphx::shape::float_type}}), concat_ins);
        auto reshape_ins = m0.add_instruction(migraphx::make_op("reshape"), x_param, alloc_ins);
        m0.add_return({reshape_ins});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {3, 4, 8}};
        auto x_param = m1.add_parameter("x", s);
        auto reshape_ins =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 32}}}), x_param);
        m1.add_return({reshape_ins});
    }

    EXPECT(m0 == m1);
}

TEST_CASE(const_alloc_fill)
{
    migraphx::module m0;
    {
        migraphx::shape val_shape{migraphx::shape::int64_type, {1}, {0}};
        std::vector<int64_t> lit_data = {3};
        auto value_lit                = m0.add_literal(migraphx::literal{val_shape, lit_data});
        migraphx::shape lit_s{migraphx::shape::int64_type, {3}};
        auto output_dim_lit = m0.add_literal(migraphx::literal{lit_s, {3, 4, 4}});
        auto alloc_ins      = m0.add_instruction(
            migraphx::make_op("allocate", {{"buf_type", migraphx::shape::int64_type}}),
            output_dim_lit);
        auto ret = m0.add_instruction(migraphx::make_op("fill"), value_lit, alloc_ins);
        m0.add_return({ret});
    }
    run_pass(m0);

    migraphx::module m1;
    {
        migraphx::shape lit_shape{migraphx::shape::int64_type, {3, 4, 4}};
        std::vector<int64_t> lit_data(3 * 4 * 4, 3);
        auto ret = m1.add_literal(migraphx::literal{lit_shape, lit_data});
        m1.add_return({ret});
    }
    EXPECT(m0 == m1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
