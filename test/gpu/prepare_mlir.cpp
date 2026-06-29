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
#include <migraphx/gpu/prepare_mlir.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::prepare_mlir{}});
}

// fast_mm emits fp16-input/fp32-output quant_dot; prepare_mlir rewrites it back into a
// plain fp16 dot followed by a convert to fp32 on the output for the MLIR pipeline.
TEST_CASE(fp16_quant_dot_rewritten)
{
    migraphx::shape as{migraphx::shape::half_type, {2, 8}};
    migraphx::shape bs{migraphx::shape::half_type, {8, 4}};

    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", as);
        auto b   = m1.add_parameter("b", bs);
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), a, b);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a   = m2.add_parameter("a", as);
        auto b   = m2.add_parameter("b", bs);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a, b);
        auto out = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(fp16_quant_convolution_rewritten)
{
    migraphx::shape xs{migraphx::shape::half_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::half_type, {4, 3, 3, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_parameter("w", ws);
        auto conv = m1.add_instruction(migraphx::make_op("quant_convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", xs);
        auto w    = m2.add_parameter("w", ws);
        auto conv = m2.add_instruction(migraphx::make_op("convolution"), x, w);
        auto out  = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), conv);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

// int8 quant ops are the real quantized path and must be left for MLIR to handle.
TEST_CASE(int8_quant_dot_unchanged)
{
    migraphx::shape as{migraphx::shape::int8_type, {2, 8}};
    migraphx::shape bs{migraphx::shape::int8_type, {8, 4}};

    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", as);
        auto b   = m1.add_parameter("b", bs);
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), a, b);
        m1.add_return({dot});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
