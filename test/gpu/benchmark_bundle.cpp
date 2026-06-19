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

#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <test.hpp>

// No context-requiring ops (builtins + a context-free op) -> bundle 1.
TEST_CASE(bundle_zero_ops)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto x  = m.add_parameter("x", s);
    auto id = m.add_instruction(migraphx::make_op("identity"), x);
    m.add_return({id});

    EXPECT(migraphx::gpu::compute_benchmark_bundle(m) == 1);
}

// One context-requiring op -> bundle 2.
TEST_CASE(bundle_one_op)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto x  = m.add_parameter("x", s);
    auto id = m.add_instruction(migraphx::make_op("identity"), x);
    auto alloc =
        m.add_instruction(migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s)}}));
    m.add_return({id, alloc});

    EXPECT(migraphx::gpu::compute_benchmark_bundle(m) == 2);
}

// Two context-requiring ops (e.g. kernel + prefill) -> bundle 6.
TEST_CASE(bundle_two_ops)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto alloc =
        m.add_instruction(migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s)}}));
    auto fill = m.add_instruction(migraphx::make_op("hip::fill", {{"value", 0}}), alloc);
    m.add_return({fill});

    EXPECT(migraphx::gpu::compute_benchmark_bundle(m) == 6);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
