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
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <test.hpp>

static bool rejects_non_pointwise(const migraphx::operation& op, const migraphx::shape& input_shape)
{
    migraphx::module m;
    auto input  = m.add_parameter("input", input_shape);
    auto result = m.add_instruction(op, input);
    m.add_return({result});
    return test::throws<migraphx::exception>([&] { migraphx::gpu::validate_pointwise_module(m); },
                                             op.name());
}

TEST_CASE(pointwise_module_accepts_pointwise_literal_param_and_return)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    auto input = m.add_parameter("input", s);
    auto one   = m.add_literal(migraphx::literal{s, {1.0f, 1.0f, 1.0f, 1.0f}});
    auto add   = m.add_instruction(migraphx::make_op("add"), input, one);
    m.add_return({add});

    migraphx::gpu::validate_pointwise_module(m);
}

TEST_CASE(pointwise_module_rejects_shape_operations)
{
    const migraphx::shape reshape_input{migraphx::shape::float_type, {1, 1, 8, 8}};
    const migraphx::shape reshape_output{migraphx::shape::float_type, {1, 1, 4, 2, 8}};
    EXPECT(rejects_non_pointwise(migraphx::make_op("reshape", {{"dims", reshape_output.lens()}}),
                                 reshape_input));
    EXPECT(rejects_non_pointwise(
        migraphx::make_op("reshape_lazy", {{"dims", reshape_output.lens()}}), reshape_input));
    EXPECT(rejects_non_pointwise(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2, 4}}}),
                                 reshape_output));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
