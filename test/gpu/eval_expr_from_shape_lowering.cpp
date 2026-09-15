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
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>

static void run_lowering(migraphx::module& m)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(m,
                         {migraphx::gpu::lowering{&ctx, false}, migraphx::dead_code_elimination{}});
}

TEST_CASE(eval_expr_from_shape_lowering_single_input)
{
    using dd = migraphx::shape::dynamic_dimension;
    auto n   = migraphx::sym::var("N", {1, 4});
    migraphx::shape input_shape{migraphx::shape::float_type, {dd{n}, dd{migraphx::sym::lit(3)}}};
    migraphx::shape output_shape{migraphx::shape::int64_type, {1}};
    auto op = migraphx::make_op(
        "eval_expr_from_shape",
        {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{n})}});

    migraphx::module m1;
    {
        auto input  = m1.add_parameter("input", input_shape);
        auto result = m1.add_instruction(op, input);
        m1.add_return({result});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto input  = m2.add_parameter("input", input_shape);
        auto output = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(output_shape)}}));
        auto host_result = m2.add_instruction(op, input);
        auto gpu_result =
            m2.add_instruction(migraphx::make_op("hip::copy_to_gpu"), host_result, output);
        m2.add_return({gpu_result});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(eval_expr_from_shape_lowering_multi_input)
{
    using dd = migraphx::shape::dynamic_dimension;
    auto m   = migraphx::sym::var("M", {1, 4});
    auto n   = migraphx::sym::var("N", {1, 8});
    migraphx::shape a_shape{migraphx::shape::float_type, {dd{m}, dd{migraphx::sym::lit(3)}}};
    migraphx::shape b_shape{migraphx::shape::float_type, {dd{migraphx::sym::lit(2)}, dd{n}}};
    migraphx::shape output_shape{migraphx::shape::int64_type, {3}};
    auto op = migraphx::make_op(
        "eval_expr_from_shape",
        {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{m + n, m, n})}});

    migraphx::module m1;
    {
        auto a      = m1.add_parameter("a", a_shape);
        auto b      = m1.add_parameter("b", b_shape);
        auto result = m1.add_instruction(op, a, b);
        m1.add_return({result});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto a      = m2.add_parameter("a", a_shape);
        auto b      = m2.add_parameter("b", b_shape);
        auto output = m2.add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(output_shape)}}));
        auto host_result = m2.add_instruction(op, a, b);
        auto gpu_result =
            m2.add_instruction(migraphx::make_op("hip::copy_to_gpu"), host_result, output);
        m2.add_return({gpu_result});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(eval_expr_from_shape_lowering_slice_metadata_stays_on_host)
{
    using dd  = migraphx::shape::dynamic_dimension;
    auto n    = migraphx::sym::var("N", {1, 4});
    auto zero = migraphx::sym::lit(0);
    migraphx::shape source_shape{migraphx::shape::float_type, {dd{n}, dd{migraphx::sym::lit(3)}}};
    migraphx::shape data_shape{migraphx::shape::float_type, {4, 2}};
    auto start_op = migraphx::make_op(
        "eval_expr_from_shape",
        {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{zero})}});
    auto end_op = migraphx::make_op(
        "eval_expr_from_shape",
        {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{n})}});
    auto slice_op =
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", migraphx::to_value(std::vector<migraphx::sym::expr>{zero})},
                           {"ends", migraphx::to_value(std::vector<migraphx::sym::expr>{n})}});

    migraphx::module m1;
    auto source = m1.add_parameter("source", source_shape);
    auto data   = m1.add_parameter("data", data_shape);
    auto start  = m1.add_instruction(start_op, source);
    auto end    = m1.add_instruction(end_op, source);
    auto slice  = m1.add_instruction(slice_op, data, start, end);
    m1.add_return({slice});

    auto m2 = m1;
    run_lowering(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
