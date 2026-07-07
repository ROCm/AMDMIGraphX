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
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <test.hpp>

static void run_lowering(migraphx::module& m, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        m, {migraphx::gpu::lowering{&ctx, offload_copy}, migraphx::dead_code_elimination{}});
}

// dimensions_of reads the input's runtime lengths on host, so lowering should
// sync the stream, run the op on host, and copy the result to the gpu.
TEST_CASE(dimensions_of_lowering_default)
{
    migraphx::shape in_s{migraphx::shape::float_type, {{1, 4, {2, 4}}, {3, 3}, {4, 4}}};
    migraphx::shape out_s{migraphx::shape::int64_type, {3}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", in_s);
        auto d = m1.add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), x);
        m1.add_return({d});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", in_s);
        auto output = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto sync     = m2.add_instruction(migraphx::make_op("hip::sync_stream"), x);
        auto host_out = m2.add_instruction(migraphx::make_op("dimensions_of", {{"end", 3}}), sync);
        auto gpu_out  = m2.add_instruction(migraphx::make_op("hip::copy_to_gpu"), host_out, output);
        m2.add_return({gpu_out});
    }
    EXPECT(m1 == m2);
}

// A sub-range of the input dimensions should be handled the same way, honoring
// both the start and end attributes on the host op.
TEST_CASE(dimensions_of_lowering_start_end)
{
    migraphx::shape in_s{migraphx::shape::float_type, {{1, 4, {1, 4}}, {3, 3}, {3, 8}, {3, 8}}};
    migraphx::shape out_s{migraphx::shape::int64_type, {2}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", in_s);
        auto d =
            m1.add_instruction(migraphx::make_op("dimensions_of", {{"start", 2}, {"end", 4}}), x);
        m1.add_return({d});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", in_s);
        auto output = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto sync     = m2.add_instruction(migraphx::make_op("hip::sync_stream"), x);
        auto host_out = m2.add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 2}, {"end", 4}}), sync);
        auto gpu_out = m2.add_instruction(migraphx::make_op("hip::copy_to_gpu"), host_out, output);
        m2.add_return({gpu_out});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
