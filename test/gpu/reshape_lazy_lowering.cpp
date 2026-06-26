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
#include <test.hpp>

static void run_lowering(migraphx::module& m, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        m, {migraphx::gpu::lowering{&ctx, offload_copy}, migraphx::dead_code_elimination{}});
}

TEST_CASE(reshape_lazy_lowering_static_dims)
{
    migraphx::shape in_s{migraphx::shape::float_type, {6, 4}};
    migraphx::shape out_s{migraphx::shape::float_type, {2, 3, 4}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", in_s);
        auto r = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4}}}), x);
        m1.add_return({r});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto x            = m2.add_parameter("x", in_s);
        auto before_alloc = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(in_s)}}));
        auto before_contig =
            m2.add_instruction(migraphx::make_op("gpu::contiguous"), x, before_alloc);
        auto rl = m2.add_instruction(
            migraphx::make_op("reshape_lazy", {{"dims", std::vector<int64_t>{2, 3, 4}}}),
            before_contig);
        auto after_alloc = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto after_contig =
            m2.add_instruction(migraphx::make_op("gpu::contiguous"), rl, after_alloc);
        m2.add_return({after_contig});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(reshape_lazy_lowering_skip_zero)
{
    migraphx::shape in_s{migraphx::shape::float_type, {{1, 4}, {24, 24}, {1, 1}, {1, 1}}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", in_s);
        auto r = m1.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<int64_t>{0, 8, 3, 1}}}), x);
        m1.add_return({r});
    }
    auto m2 = m1;
    run_lowering(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(reshape_lazy_lowering_skip_neg_one)
{
    migraphx::shape in_s{migraphx::shape::float_type, {24, 1, 1, 1}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", in_s);
        auto r = m1.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<int64_t>{-1, 1, 1, 24}}}), x);
        m1.add_return({r});
    }
    auto m2 = m1;
    run_lowering(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
