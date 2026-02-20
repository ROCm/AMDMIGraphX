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

static bool contains_op(const migraphx::module& m, const std::string& op_name)
{
    return std::any_of(m.begin(), m.end(), [&](const auto& ins) { return ins.name() == op_name; });
}

static std::size_t count_ops(const migraphx::module& m, const std::string& op_name)
{
    return std::count_if(
        m.begin(), m.end(), [&](const auto& ins) { return ins.name() == op_name; });
}

TEST_CASE(argmin_rewrite)
{
    // Test that argmin gets rewritten to make_indices -> arg_reduce -> get_tuple_elem
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m;
    {
        auto x      = m.add_parameter("x", s);
        auto argmin = m.add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), x);
        m.add_return({argmin});
    }
    auto expected_shape = m.get_output_shapes();

    run_pass(m);

    // argmin should be gone, replaced with gpu::arg_reduce pattern
    EXPECT(not contains_op(m, "argmin"));
    EXPECT(contains_op(m, "gpu::arg_reduce"));
    EXPECT(contains_op(m, "gpu::make_indices"));
    EXPECT(contains_op(m, "get_tuple_elem"));
    EXPECT(m.get_output_shapes() == expected_shape);
}

TEST_CASE(argmax_rewrite)
{
    // Test that argmax gets rewritten to make_indices -> arg_reduce -> get_tuple_elem
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m;
    {
        auto x      = m.add_parameter("x", s);
        auto argmax = m.add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), x);
        m.add_return({argmax});
    }
    auto expected_shape = m.get_output_shapes();

    run_pass(m);

    // argmax should be gone, replaced with gpu::arg_reduce pattern
    EXPECT(not contains_op(m, "argmax"));
    EXPECT(contains_op(m, "gpu::arg_reduce"));
    EXPECT(contains_op(m, "gpu::make_indices"));
    EXPECT(contains_op(m, "get_tuple_elem"));
    EXPECT(m.get_output_shapes() == expected_shape);
}

TEST_CASE(argmin_axis0)
{
    // Test argmin along axis 0
    migraphx::shape s{migraphx::shape::float_type, {5, 3}};

    migraphx::module m;
    {
        auto x      = m.add_parameter("x", s);
        auto argmin = m.add_instruction(migraphx::make_op("argmin", {{"axis", 0}}), x);
        m.add_return({argmin});
    }
    auto expected_shape = m.get_output_shapes();

    run_pass(m);

    EXPECT(not contains_op(m, "argmin"));
    EXPECT(contains_op(m, "gpu::arg_reduce"));
    EXPECT(contains_op(m, "gpu::make_indices"));
    EXPECT(m.get_output_shapes() == expected_shape);
}

TEST_CASE(parallel_reduce_two_sum)
{
    // Test that two independent reduce_sum operations get fused into parallel_reduce
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m;
    {
        auto x  = m.add_parameter("x", s);
        auto y  = m.add_parameter("y", s);
        auto r1 = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto r2 = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        m.add_return({r1, r2});
    }
    auto expected_shape = m.get_output_shapes();

    run_pass(m);

    // Two reduce_sum should be fused into one parallel_reduce
    EXPECT(not contains_op(m, "reduce_sum"));
    EXPECT(contains_op(m, "gpu::parallel_reduce"));
    EXPECT(count_ops(m, "gpu::parallel_reduce") == 1);
    EXPECT(count_ops(m, "get_tuple_elem") == 2);
    EXPECT(m.get_output_shapes() == expected_shape);
}

TEST_CASE(no_parallel_reduce_different_ops)
{
    // Test that reduce operations with different ops are not fused
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m;
    {
        auto x  = m.add_parameter("x", s);
        auto y  = m.add_parameter("y", s);
        auto r1 = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto r2 = m.add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), y);
        m.add_return({r1, r2});
    }
    auto expected_shape = m.get_output_shapes();

    run_pass(m);

    // Different reduce ops should NOT be fused
    EXPECT(not contains_op(m, "gpu::parallel_reduce"));
    EXPECT(contains_op(m, "reduce_sum"));
    EXPECT(contains_op(m, "reduce_max"));
    EXPECT(m.get_output_shapes() == expected_shape);
}

TEST_CASE(no_parallel_reduce_dependent)
{
    // Test that dependent reduce operations are not fused
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m;
    {
        auto x  = m.add_parameter("x", s);
        auto r1 = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto bc =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4}}}), r1);
        auto r2 = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), bc);
        m.add_return({r2});
    }
    auto expected_shape = m.get_output_shapes();

    run_pass(m);

    // Dependent reduces should NOT be fused
    EXPECT(not contains_op(m, "gpu::parallel_reduce"));
    EXPECT(count_ops(m, "reduce_sum") == 2);
    EXPECT(m.get_output_shapes() == expected_shape);
}

TEST_CASE(argmin_no_parallel_with_reduce)
{
    // Test that argmin (rewritten to arg_reduce) is not fused with reduce_sum
    // because arg_reduce is excluded from parallel fusion
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m;
    {
        auto x      = m.add_parameter("x", s);
        auto y      = m.add_parameter("y", s);
        auto argmin = m.add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), x);
        auto rsum   = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        m.add_return({argmin, rsum});
    }
    auto expected_shape = m.get_output_shapes();

    run_pass(m);

    // argmin rewritten to arg_reduce, reduce_sum stays separate (only 1 reduce, no fusion)
    EXPECT(not contains_op(m, "argmin"));
    EXPECT(contains_op(m, "gpu::arg_reduce"));
    EXPECT(contains_op(m, "reduce_sum"));
    EXPECT(not contains_op(m, "gpu::parallel_reduce"));
    EXPECT(m.get_output_shapes() == expected_shape);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
