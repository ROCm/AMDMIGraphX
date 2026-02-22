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
    auto reduce_axis_size = x->get_shape().lens().at(axis);
    auto indices          = m.add_instruction(migraphx::gpu::make_indices{reduce_axis_size});
    auto ar               = m.add_instruction(
        migraphx::gpu::arg_reduce{migraphx::make_op(op_name, {{"axis", axis}})}, x, indices);
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
            migraphx::gpu::parallel_reduce{migraphx::make_op("reduce_sum", {{"axes", {1}}})}, x, y);
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
