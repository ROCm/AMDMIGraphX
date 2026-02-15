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
#include <migraphx/gpu/gen/codegen.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/stringutils.hpp>
#include <test.hpp>

using migraphx::make_op;
using migraphx::shape;

TEST_CASE(gen_function_pointwise_add)
{
    migraphx::module m;
    auto x   = m.add_parameter("x", {shape::float_type});
    auto y   = m.add_parameter("y", {shape::float_type});
    auto add = m.add_instruction(make_op("add"), x, y);
    m.add_return({add});

    auto code = migraphx::gpu::gen::generate_gen_function(m);

    EXPECT(migraphx::contains(code, "gen_func"));
    EXPECT(migraphx::contains(code, "__device__"));
}

TEST_CASE(gen_kernel_structure)
{
    migraphx::module m;
    auto x   = m.add_parameter("x", {shape::float_type});
    auto y   = m.add_parameter("y", {shape::float_type});
    auto add = m.add_instruction(make_op("add"), x, y);
    m.add_return({add});

    auto code = migraphx::gpu::gen::generate_gen_kernel(m, "test_kernel", 3);

    EXPECT(migraphx::contains(code, "test_kernel"));
    EXPECT(migraphx::contains(code, "make_index"));
    EXPECT(migraphx::contains(code, "make_tensors"));
    EXPECT(migraphx::contains(code, "gen_func"));
    EXPECT(migraphx::contains(code, "migraphx/kernels/gen.hpp"));
}

TEST_CASE(gen_function_mul_add)
{
    migraphx::module m;
    auto x   = m.add_parameter("x", {shape::float_type});
    auto y   = m.add_parameter("y", {shape::float_type});
    auto z   = m.add_parameter("z", {shape::float_type});
    auto mul = m.add_instruction(make_op("mul"), x, y);
    auto add = m.add_instruction(make_op("add"), mul, z);
    m.add_return({add});

    auto code = migraphx::gpu::gen::generate_gen_function(m);
    EXPECT(migraphx::contains(code, "gen_func"));
}

TEST_CASE(gen_function_gen_ops)
{
    migraphx::module m;
    auto x   = m.add_parameter("x", shape{shape::float_type, {64}});
    auto z   = m.add_parameter("z", shape{shape::float_type, {64}});
    auto gid = m.add_instruction(make_op("gpu::gen::global_id"));
    auto ld  = m.add_instruction(make_op("gpu::gen::vector_load", {{"size", 4}}), x, gid);
    auto st  = m.add_instruction(make_op("gpu::gen::vector_store", {{"size", 4}}), z, gid, ld);
    m.add_return({st});

    auto code = migraphx::gpu::gen::generate_gen_function(m);

    EXPECT(migraphx::contains(code, "gen_func"));
    EXPECT(migraphx::contains(code, "idx.global"));
    EXPECT(migraphx::contains(code, "gen::vec_load<4>"));
    EXPECT(migraphx::contains(code, "gen::vec_store<4>"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
