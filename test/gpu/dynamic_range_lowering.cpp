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

#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>

static void run_lowering(migraphx::module& m, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        m, {migraphx::gpu::lowering{&ctx, offload_copy}, migraphx::dead_code_elimination{}});
}

static migraphx::operation bounded_range()
{
    const migraphx::shape::dynamic_dimension length{migraphx::sym::var("n", {0, 8})};
    return migraphx::make_op("dynamic_range",
                             {{"max_output", 8}, {"output_dim", migraphx::to_value(length)}});
}

// A bounded dynamic_range runs on the host from its inputs copied back from the GPU, into an
// output allocated for its bound.
TEST_CASE(dynamic_range_lowering_bounded)
{
    migraphx::shape s{migraphx::shape::int64_type, {1}};
    migraphx::module m1;
    {
        auto start = m1.add_parameter("start", s);
        auto limit = m1.add_parameter("limit", s);
        auto delta = m1.add_parameter("delta", s);
        auto r     = m1.add_instruction(bounded_range(), start, limit, delta);
        m1.add_return({r});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto start  = m2.add_parameter("start", s);
        auto limit  = m2.add_parameter("limit", s);
        auto delta  = m2.add_parameter("delta", s);
        auto out_s  = bounded_range().compute_shape({s, s, s});
        auto output = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto copy_start = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), start);
        auto copy_limit = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), limit);
        auto copy_delta = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), delta);
        auto sync       = m2.add_instruction(
            migraphx::make_op("hip::sync_stream"), copy_start, copy_limit, copy_delta);
        auto r   = m2.add_instruction(bounded_range(), sync, copy_limit, copy_delta);
        auto out = m2.add_instruction(migraphx::make_op("hip::copy_to_gpu"), r, output);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

// An unbounded dynamic_range cannot be allocated, so it is not lowered.
TEST_CASE(dynamic_range_lowering_unbounded)
{
    migraphx::shape s{migraphx::shape::int64_type, {1}};
    migraphx::module m1;
    {
        auto start = m1.add_parameter("start", s);
        auto limit = m1.add_parameter("limit", s);
        auto delta = m1.add_parameter("delta", s);
        auto r     = m1.add_instruction(migraphx::make_op("dynamic_range"), start, limit, delta);
        m1.add_return({r});
    }
    auto m2 = m1;
    run_lowering(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
