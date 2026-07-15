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

#include <vector>
#include <migraphx/gpu/prepare_mlir.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::prepare_mlir{}, migraphx::dead_code_elimination{}});
}

// A non-standard-strided literal (as folded from a transposed constant, which mlir rejects) is
// rewritten to a standard shape, preserving the logical values.
TEST_CASE(nonstandard_literal_normalized)
{
    const auto f = migraphx::shape::float_type;

    migraphx::module m1;
    {
        migraphx::shape s{f, {2, 2}, {1, 2}};
        auto lit = m1.add_literal(migraphx::literal{s, {1, 3, 2, 4}});
        m1.add_return({lit});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto lit = m2.add_literal(migraphx::literal{migraphx::shape{f, {2, 2}}, {1, 3, 2, 4}});
        m2.add_return({lit});
    }

    EXPECT(m1.sort() == m2.sort());
}

// An already-standard literal is left untouched.
TEST_CASE(standard_literal_unchanged)
{
    const auto f = migraphx::shape::float_type;

    migraphx::module m1;
    {
        auto lit = m1.add_literal(migraphx::literal{migraphx::shape{f, {2, 2}}, {1, 2, 3, 4}});
        m1.add_return({lit});
    }
    auto m2 = m1;
    run_pass(m1);

    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
