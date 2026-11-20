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
#include <pointwise.hpp>

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
    auto ar      = m.add_instruction(
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

// Reductions over inputs of different lens read a different number of
// elements per output, so they are not run in parallel
TEST_CASE(no_parallel_reduce_different_lens)
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape ys{migraphx::shape::float_type, {2, 3, 1}};

    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", xs);
        auto y  = m1.add_parameter("y", ys);
        auto r1 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), x);
        auto r2 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), y);
        m1.add_return({r1, r2});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
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
    auto x0 = pm.add_parameter("x0", scalar(migraphx::shape::uint8_type));
    auto x1 = pm.add_parameter("x1", scalar(migraphx::shape::half_type));
    auto x2 = pm.add_parameter("x2", scalar(migraphx::shape::half_type));
    auto c  = pm.add_instruction(
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

// The unpacked values are used unconverted, so the unpack does not fold; the
// mul and reduce_sum are still rewritten to read the plain unpack
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
    run_program_pass(p1);
    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto packed = mm->add_parameter("x0", {migraphx::shape::uint8_type, {1, 4, 16}});
        auto up     = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto x1     = mm->add_parameter("x1", {migraphx::shape::uint8_type, {1, 4, 32}});
        auto mrs =
            mm->add_instruction(migraphx::make_op("gpu::mul_reduce_sum", {{"axes", {2}}}), up, x1);
        mm->add_return({mrs});
    }
    EXPECT(p1 == p2);
}

static migraphx::instruction_ref add_mul_reduce_sum(migraphx::module& m,
                                                    const std::vector<int64_t>& axes,
                                                    migraphx::instruction_ref x,
                                                    migraphx::instruction_ref y)
{
    return m.add_instruction(migraphx::make_op("gpu::mul_reduce_sum", {{"axes", axes}}), x, y);
}

TEST_CASE(mul_reduce_sum)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    run_program_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        mm->add_return({add_mul_reduce_sum(*mm, {1}, x, y)});
    }

    EXPECT(p1 == p2);
}

TEST_CASE(mul_reduce_sum_multiple_axes)
{
    migraphx::shape s{migraphx::shape::half_type, {2, 3, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), mul);
        mm->add_return({rs});
    }
    run_program_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        mm->add_return({add_mul_reduce_sum(*mm, {1, 2}, x, y)});
    }

    EXPECT(p1 == p2);
    EXPECT(p1.get_output_shapes().front() ==
           migraphx::shape{migraphx::shape::half_type, {2, 1, 1}});
}

// The operands follow the order of the mul in the pointwise module
TEST_CASE(mul_reduce_sum_swapped_operands)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x, y}, [](auto* pm, const auto& inputs) {
            return pm->add_instruction(migraphx::make_op("mul"), inputs[1], inputs[0]);
        });
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    run_program_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        mm->add_return({add_mul_reduce_sum(*mm, {1}, y, x)});
    }

    EXPECT(p1 == p2);
}

// A squared input is a pointwise with one input, which is read for both operands
TEST_CASE(mul_reduce_sum_square)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x}, [](auto* pm, const auto& inputs) {
            return pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[0]);
        });
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    run_program_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        mm->add_return({add_mul_reduce_sum(*mm, {1}, x, x)});
    }

    EXPECT(p1 == p2);
}

// A broadcast input is a value computed once per output, which the reduction
// cant read through its slices
TEST_CASE(mul_reduce_sum_broadcast_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::shape ys{migraphx::shape::float_type, {2, 1}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", ys);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), y);
        auto mul = add_pointwise(p1, "main:pointwise0", {x, yb}, single_pointwise("mul"));
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);

    EXPECT(p1 == p2);
}

// Reducing an axis of one element leaves the inputs at the reduced shape, so
// they are read as one value per output
TEST_CASE(mul_reduce_sum_unit_axis)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);

    EXPECT(p1 == p2);
}

// The pointwise is still needed by its other consumer
TEST_CASE(mul_reduce_sum_pointwise_used_twice)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({mul, rs});
    }
    run_program_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = add_pointwise(p2, "main:pointwise0", {x, y}, single_pointwise("mul"));
        mm->add_return({mul, add_mul_reduce_sum(*mm, {1}, x, y)});
    }

    EXPECT(p1 == p2);
}

// The rewritten reductions read two inputs each, so they are not fused into
// a gpu::parallel_reduce
TEST_CASE(mul_reduce_sum_no_parallel_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto mul1 = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto mul2 = add_pointwise(p1, "main:pointwise1", {x, z}, single_pointwise("mul"));
        auto rs1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul1);
        auto rs2  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul2);
        mm->add_return({rs1, rs2});
    }
    run_program_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto rs1 = add_mul_reduce_sum(*mm, {1}, x, y);
        auto rs2 = add_mul_reduce_sum(*mm, {1}, x, z);
        mm->add_return({rs1, rs2});
    }

    EXPECT(p1.sort() == p2.sort());
}

// A mul by a literal is a scale, which has no second tensor to read
TEST_CASE(mul_reduce_sum_literal_operand)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x}, [](auto* pm, const auto& inputs) {
            auto two = pm->add_literal(2.0f);
            return pm->add_instruction(migraphx::make_op("mul"), inputs[0], two);
        });
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(mul_reduce_sum_not_mul)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto add = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);

    EXPECT(p1 == p2);
}

// A product of three inputs is not a dot product of two tensors
TEST_CASE(mul_reduce_sum_three_operands)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto mul =
            add_pointwise(p1, "main:pointwise0", {x, y, z}, [](auto* pm, const auto& inputs) {
                auto xy = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("mul"), xy, inputs[2]);
            });
        auto rs = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);

    EXPECT(p1 == p2);
}

// The mul is not the result of the pointwise module
TEST_CASE(mul_reduce_sum_mul_not_last)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x, y}, [](auto* pm, const auto& inputs) {
            auto xy = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
            return pm->add_instruction(migraphx::make_op("relu"), xy);
        });
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);

    EXPECT(p1 == p2);
}

// Only a sum of products is a dot product
TEST_CASE(mul_reduce_sum_not_reduce_sum)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("mul"));
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);

    EXPECT(p1 == p2);
}

// The mul is not fused into a pointwise, so there is no module to match
TEST_CASE(mul_reduce_sum_not_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), x, y);
        auto rs  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rs});
    }
    migraphx::program p2 = p1;
    run_program_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(mul_reduce_sum_shape_requires_two_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    auto op = migraphx::make_op("gpu::mul_reduce_sum", {{"axes", {1}}});
    EXPECT(op.compute_shape({s, s}) == migraphx::shape{migraphx::shape::float_type, {2, 1}});
    EXPECT(test::throws([&] { op.compute_shape({s}); }));
    EXPECT(test::throws([&] { op.compute_shape({s, s, s}); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }

TEST_CASE(pointwise_broadcast_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), rsum);
        auto rsqrt = add_pointwise(p1, "main:pointwise0", {rsumb}, single_pointwise("rsqrt"));
        auto mul   = add_pointwise(p1, "main:pointwise1", {x, rsqrt}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    run_pass(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsqrt = add_pointwise(p2, "main:pointwise0", {rsum}, single_pointwise("rsqrt"));
        auto rsqrtb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), rsqrt);
        auto mul = add_pointwise(p2, "main:pointwise1", {x, rsqrtb}, single_pointwise("mul"));
        mm->add_return({mul});
    }

    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_broadcast_scalar_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};
    migraphx::shape ss{migraphx::shape::float_type, {1}};

    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", ss);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), rsum);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), y);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumb, yb}, single_pointwise("add"));
        auto mul = add_pointwise(p1, "main:pointwise1", {x, add}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    run_pass(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", ss);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 1}}}), y);
        auto add = add_pointwise(p2, "main:pointwise0", {rsum, yb}, single_pointwise("add"));
        auto addb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), add);
        auto mul = add_pointwise(p2, "main:pointwise1", {x, addb}, single_pointwise("mul"));
        mm->add_return({mul});
    }

    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_broadcast_and_tensor_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 8}};

    auto create_program = [&] {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8}}}), rsum);
        auto sub = add_pointwise(p, "main:pointwise0", {x, rsumb}, single_pointwise("sub"));
        mm->add_return({sub});
        return p;
    };

    auto p1 = create_program();
    run_pass(*p1.get_main_module());
    auto p2 = create_program();

    EXPECT(p1 == p2);
}
