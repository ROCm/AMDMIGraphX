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
#include <migraphx/errors.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <test.hpp>

TEST_CASE(two_input_reshape_is_rejected)
{
    migraphx::shape input_shape{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape output_shape{migraphx::shape::float_type, {6, 4}};

    migraphx::module m;
    auto input  = m.add_parameter("input", input_shape);
    auto output = m.add_parameter("output", output_shape);
    auto result = m.add_instruction(migraphx::make_op("reshape"), input, output);
    m.add_return({result});

    auto ctx = migraphx::gpu::context{};
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::run_passes(m, {migraphx::gpu::lowering{&ctx, false}}); },
        "gpu::lowering: two-input reshape must be eliminated before lowering"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
