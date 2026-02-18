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
#include <migraphx/register_op.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

// Mirror the ops from prepare_reduce.cpp for testing
struct arg_reduce
{
    migraphx::operation op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::arg_reduce"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        auto reduced_shape = op.compute_shape({inputs.front()});
        return migraphx::shape{{reduced_shape, reduced_shape.with_type(migraphx::shape::int64_type)}};
    }
};

struct make_indices
{
    std::size_t size = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.size, "size"));
    }

    std::string name() const { return "gpu::make_indices"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        return migraphx::shape{migraphx::shape::int64_type, {size}};
    }
};

struct parallel_reduce
{
    migraphx::operation op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::parallel_reduce"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        std::vector<migraphx::shape> result;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(result), [&](auto input) {
            return op.compute_shape({input});
        });
        return migraphx::shape{result};
    }
};

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::prepare_reduce{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(argmin_rewrite)
{
    // Test that argmin gets rewritten to make_indices -> arg_reduce -> get_tuple_elem
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
        auto x         = m2.add_parameter("x", s);
        auto indices   = m2.add_instruction(make_indices{3});
        auto arg_red   = m2.add_instruction(
            arg_reduce{migraphx::make_op("argmin", {{"axis", 1}})}, x, indices);
        auto result = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), arg_red);
        m2.add_return({result});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(argmax_rewrite)
{
    // Test that argmax gets rewritten to make_indices -> arg_reduce -> get_tuple_elem
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
        auto x         = m2.add_parameter("x", s);
        auto indices   = m2.add_instruction(make_indices{4});
        auto arg_red   = m2.add_instruction(
            arg_reduce{migraphx::make_op("argmax", {{"axis", 2}})}, x, indices);
        auto result = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), arg_red);
        m2.add_return({result});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(argmin_axis0)
{
    // Test argmin along axis 0
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
        auto x         = m2.add_parameter("x", s);
        auto indices   = m2.add_instruction(make_indices{5});
        auto arg_red   = m2.add_instruction(
            arg_reduce{migraphx::make_op("argmin", {{"axis", 0}})}, x, indices);
        auto result = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), arg_red);
        m2.add_return({result});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(parallel_reduce_two_sum)
{
    // Test that two independent reduce_sum operations get fused into parallel_reduce
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
            parallel_reduce{migraphx::make_op("reduce_sum", {{"axes", {1}}})}, x, y);
        auto r1 = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), pr);
        auto r2 = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), pr);
        m2.add_return({r1, r2});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(no_parallel_reduce_different_ops)
{
    // Test that reduce operations with different ops are not fused
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto y  = m1.add_parameter("y", s);
        auto r1 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto r2 = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), y);
        m1.add_return({r1, r2});
    }

    migraphx::module m2 = m1; // Copy before running pass
    run_pass(m1);

    // Should remain unchanged - no parallel_reduce
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(no_parallel_reduce_dependent)
{
    // Test that dependent reduce operations are not fused
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 1, 4}};

    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", s);
        auto r1 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto bc = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4}}}), r1);
        auto r2 = m1.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), bc);
        m1.add_return({r2});
    }

    migraphx::module m2 = m1; // Copy before running pass
    run_pass(m1);

    // Should remain unchanged - r2 depends on r1
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(argmin_no_parallel_with_reduce)
{
    // Test that argmin (rewritten to arg_reduce) is not fused with reduce_sum
    // because arg_reduce is excluded from parallel fusion
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

    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", s);
        auto y         = m2.add_parameter("y", s);
        // argmin gets rewritten
        auto indices   = m2.add_instruction(make_indices{3});
        auto arg_red   = m2.add_instruction(
            arg_reduce{migraphx::make_op("argmin", {{"axis", 1}})}, x, indices);
        auto argmin_result = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), arg_red);
        // reduce_sum remains unchanged (only 1 reduce, no parallel fusion)
        auto rsum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        m2.add_return({argmin_result, rsum});
    }

    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }

