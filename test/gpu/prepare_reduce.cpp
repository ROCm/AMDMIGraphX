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

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/prepare_reduce.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::prepare_reduce{}, migraphx::dead_code_elimination{}});
}

// Helper to add the arg_reduce pattern: make_indices -> arg_reduce -> get_tuple_elem
static migraphx::instruction_ref add_arg_reduce(migraphx::module& m,
                                                migraphx::instruction_ref x,
                                                const std::string& op_name,
                                                int axis)
{
    auto indices = m.add_instruction(migraphx::make_op("gpu::make_indices"), {x});
    auto ar = m.add_instruction(
        migraphx::make_op(
            "gpu::arg_reduce",
            {{"op", migraphx::to_value(migraphx::make_op(op_name, {{"axis", axis}}))}}),
        x,
        indices);
    return m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), ar);
}

TEST_CASE(argmin_rewrite)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto argmin = m1.add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), x);
        m1.add_return({argmin});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto out = add_arg_reduce(m2, x, "argmin", 1);
        m2.add_return({out});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(argmax_rewrite)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto argmax = m1.add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), x);
        m1.add_return({argmax});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto out = add_arg_reduce(m2, x, "argmax", 2);
        m2.add_return({out});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(argmin_axis0)
{
    migraphx::shape s{migraphx::shape::float_type, {5, 3}};

    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto argmin = m1.add_instruction(migraphx::make_op("argmin", {{"axis", 0}}), x);
        m1.add_return({argmin});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto out = add_arg_reduce(m2, x, "argmin", 0);
        m2.add_return({out});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(parallel_reduce_two_sum)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s);
        auto r1 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto r2 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        m1.add_return({r1, r2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", s);
        auto y  = m2.add_parameter("y", s);
        auto pr = m2.add_instruction(
            migraphx::make_op(
                "gpu::parallel_reduce",
                {{"op", migraphx::to_value(migraphx::make_op("reduce_sum", {{"axes", {1}}}))}}),
            x,
            y);
        auto r1 = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pr);
        auto r2 = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), pr);
        m2.add_return({r1, r2});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(no_parallel_reduce_different_ops)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s);
        auto r1 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto r2 = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), y);
        m1.add_return({r1, r2});
    }
    auto m2 = m1;
    run_pass(m1);

    // Different reduce ops should NOT be fused - module unchanged
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(no_parallel_reduce_dependent)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto r1 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4}}}), r1);
        auto r2 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), bc);
        m1.add_return({r2});
    }
    auto m2 = m1;
    run_pass(m1);

    // Dependent reduces should NOT be fused - module unchanged
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(argmin_no_parallel_with_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto argmin = m1.add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), x);
        auto rsum   = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        m1.add_return({argmin, rsum});
    }
    run_pass(m1);

    // argmin gets rewritten, reduce_sum stays as is (no parallel fusion with single reduce)
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto y    = m2.add_parameter("y", s);
        auto out  = add_arg_reduce(m2, x, "argmin", 1);
        auto rsum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        m2.add_return({out, rsum});
    }

    EXPECT(m1.sort() == m2.sort());
}

static void run_program_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::prepare_reduce{}, migraphx::dead_code_elimination{}});
}

static migraphx::shape scalar(migraphx::shape::type_t t) { return migraphx::shape{t}; }

// The reduce module around an int4 unpack: unpack -> pointwise(pm) -> reduce_sum,
// with an optional per-element zero point tensor as the last pointwise input
static void add_unpack_reduce(migraphx::module& mm,
                              const migraphx::operation& unpack,
                              migraphx::module_ref pm,
                              bool zero_point_tensor = false)
{
    migraphx::shape s{migraphx::shape::half_type, {1, 4, 32}};
    auto packed = mm.add_parameter("x0", {migraphx::shape::uint8_type, {1, 4, 16}});
    std::vector<migraphx::instruction_ref> inputs = {
        mm.add_instruction(unpack, packed), mm.add_parameter("x1", s), mm.add_parameter("x2", s)};
    if(zero_point_tensor)
        inputs.push_back(mm.add_parameter("x3", {migraphx::shape::uint8_type, {1, 4, 32}}));
    auto pw = mm.add_instruction(migraphx::make_op("pointwise"), inputs, {pm});
    auto rs = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), pw);
    mm.add_return({rs});
}

static migraphx::operation unpack_int4_convert(double bias)
{
    return migraphx::make_op(
        "gpu::unpack_int4_convert",
        {{"axis", 2}, {"target_type", migraphx::shape::half_type}, {"bias", bias}});
}

// (convert(x0) + zp) * x1 * x2
static void add_dequant_pointwise(migraphx::module& pm, float zp)
{
    auto x0  = pm.add_parameter("x0", scalar(migraphx::shape::uint8_type));
    auto x1  = pm.add_parameter("x1", scalar(migraphx::shape::half_type));
    auto x2  = pm.add_parameter("x2", scalar(migraphx::shape::half_type));
    auto c   = pm.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x0);
    auto lit = pm.add_literal(migraphx::literal{scalar(migraphx::shape::half_type), {zp}});
    auto a   = pm.add_instruction(migraphx::make_op("add"), c, lit);
    auto m1  = pm.add_instruction(migraphx::make_op("mul"), a, x1);
    auto m2  = pm.add_instruction(migraphx::make_op("mul"), m1, x2);
    pm.add_return({m2});
}

// x0 * x1 * x2 with x0 already converted
static void add_folded_pointwise(migraphx::module& pm)
{
    auto x0 = pm.add_parameter("x0", scalar(migraphx::shape::half_type));
    auto x1 = pm.add_parameter("x1", scalar(migraphx::shape::half_type));
    auto x2 = pm.add_parameter("x2", scalar(migraphx::shape::half_type));
    auto m1 = pm.add_instruction(migraphx::make_op("mul"), x0, x1);
    auto m2 = pm.add_instruction(migraphx::make_op("mul"), m1, x2);
    pm.add_return({m2});
}

// (x0 - convert(x3)) * x1 * x2, with x0 converted unless already folded
static void add_zero_point_tensor_pointwise(migraphx::module& pm, bool folded)
{
    auto x0 = pm.add_parameter(
        "x0", scalar(folded ? migraphx::shape::half_type : migraphx::shape::uint8_type));
    auto x1 = pm.add_parameter("x1", scalar(migraphx::shape::half_type));
    auto x2 = pm.add_parameter("x2", scalar(migraphx::shape::half_type));
    auto x3 = pm.add_parameter("x3", scalar(migraphx::shape::uint8_type));
    auto c0 = x0;
    if(not folded)
        c0 = pm.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x0);
    auto c3 = pm.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x3);
    auto d  = pm.add_instruction(migraphx::make_op("sub"), c0, c3);
    auto m1 = pm.add_instruction(migraphx::make_op("mul"), d, x1);
    auto m2 = pm.add_instruction(migraphx::make_op("mul"), m1, x2);
    pm.add_return({m2});
}

TEST_CASE(unpack_int4_literal_zero_point)
{
    migraphx::program p1;
    {
        auto* pm = p1.create_module("pw");
        add_dequant_pointwise(*pm, -8);
        add_unpack_reduce(*p1.get_main_module(), migraphx::make_op("unpack_int4"), pm);
    }
    run_program_pass(p1);
    migraphx::program p2;
    {
        auto* pm = p2.create_module("pw:unpack");
        add_folded_pointwise(*pm);
        add_unpack_reduce(*p2.get_main_module(), unpack_int4_convert(-8), pm);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(unpack_int4_zero_point_tensor)
{
    migraphx::program p1;
    {
        auto* pm = p1.create_module("pw");
        add_zero_point_tensor_pointwise(*pm, false);
        add_unpack_reduce(*p1.get_main_module(), migraphx::make_op("unpack_int4"), pm, true);
    }
    run_program_pass(p1);
    migraphx::program p2;
    {
        auto* pm = p2.create_module("pw:unpack");
        add_zero_point_tensor_pointwise(*pm, true);
        add_unpack_reduce(*p2.get_main_module(), unpack_int4_convert(0), pm, true);
    }
    EXPECT(p1 == p2);
}

// A fractional zero point stays in the pointwise, only the convert folds
TEST_CASE(unpack_int4_fractional_zero_point)
{
    migraphx::program p1;
    {
        auto* pm = p1.create_module("pw");
        add_dequant_pointwise(*pm, -8.5);
        add_unpack_reduce(*p1.get_main_module(), migraphx::make_op("unpack_int4"), pm);
    }
    run_program_pass(p1);
    migraphx::program p2;
    {
        auto* pm = p2.create_module("pw:unpack");
        auto x0  = pm->add_parameter("x0", scalar(migraphx::shape::half_type));
        auto x1  = pm->add_parameter("x1", scalar(migraphx::shape::half_type));
        auto x2  = pm->add_parameter("x2", scalar(migraphx::shape::half_type));
        auto lit = pm->add_literal(migraphx::literal{scalar(migraphx::shape::half_type), {-8.5}});
        auto a   = pm->add_instruction(migraphx::make_op("add"), x0, lit);
        auto m1  = pm->add_instruction(migraphx::make_op("mul"), a, x1);
        auto m2  = pm->add_instruction(migraphx::make_op("mul"), m1, x2);
        pm->add_return({m2});
        add_unpack_reduce(*p2.get_main_module(), unpack_int4_convert(0), pm);
    }
    EXPECT(p1 == p2);
}

// The unpacked values are used unconverted, so nothing folds
TEST_CASE(unpack_int4_no_convert)
{
    migraphx::program p1;
    {
        auto* pm = p1.create_module("pw");
        auto px0 = pm->add_parameter("x0", scalar(migraphx::shape::uint8_type));
        auto px1 = pm->add_parameter("x1", scalar(migraphx::shape::uint8_type));
        auto mul = pm->add_instruction(migraphx::make_op("mul"), px0, px1);
        pm->add_return({mul});
        auto* mm    = p1.get_main_module();
        auto packed = mm->add_parameter("x0", {migraphx::shape::uint8_type, {1, 4, 16}});
        auto up     = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto x1     = mm->add_parameter("x1", {migraphx::shape::uint8_type, {1, 4, 32}});
        auto pw     = mm->add_instruction(migraphx::make_op("pointwise"), {up, x1}, {pm});
        auto rs     = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), pw);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
