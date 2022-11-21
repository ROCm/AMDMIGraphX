/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include "make_precompile_op.hpp"
#include <migraphx/program.hpp>

// Treat some operators as compilable to enable lowering
MIGRAPHX_GPU_TEST_PRECOMPILE("add", "mul")

TEST_CASE(simple_program_from_gpu)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    auto ones = mm->add_literal(migraphx::literal{s, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}});
    auto sum  = mm->add_instruction(migraphx::make_op("add"), ones, ones);
    mm->add_return({sum});

    p.compile(migraphx::gpu::target{});

    auto result = migraphx::gpu::from_gpu(p.eval({}).back());
    EXPECT(true);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
