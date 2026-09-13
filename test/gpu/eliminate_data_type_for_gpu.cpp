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
#include <migraphx/gpu/eliminate_data_type_for_gpu.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        m, {migraphx::gpu::eliminate_data_type_for_gpu{.disable_64bit = true, .ctx = &ctx}});
}

TEST_CASE(materialize_returned_slice_of_tuple)
{
    using dd  = migraphx::shape::dynamic_dimension;
    auto n    = migraphx::sym::var("N", {1, 4});
    auto opt  = migraphx::sym::var("opt", {1, 4});
    auto zero = migraphx::sym::lit(0);
    migraphx::shape source_shape{migraphx::shape::float_type, {dd{n}}};
    migraphx::shape data_shape{migraphx::shape::float_type, {dd{opt}, dd{migraphx::sym::lit(2)}}};
    migraphx::shape tuple_shape{std::vector<migraphx::shape>{data_shape}};
    auto start_op = migraphx::make_op(
        "eval_expr_from_shape",
        {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{zero})}});
    auto end_op = migraphx::make_op(
        "eval_expr_from_shape",
        {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{n})}});
    auto slice_op = migraphx::make_op(
        "dyn_slice",
        {{"axes", {0}},
         {"starts", migraphx::to_value(std::vector<migraphx::sym::expr>{zero})},
         {"ends",
          migraphx::to_value(std::vector<migraphx::sym::expr>{migraphx::sym::min(n, opt)})}});

    migraphx::module m1;
    {
        auto source = m1.add_parameter("source", source_shape);
        auto tuple  = m1.add_parameter("tuple", tuple_shape);
        auto data  = m1.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), tuple);
        auto start = m1.add_instruction(start_op, source);
        auto end   = m1.add_instruction(end_op, source);
        auto slice = m1.add_instruction(slice_op, data, start, end);
        m1.add_return({slice});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto source = m2.add_parameter("source", source_shape);
        auto tuple  = m2.add_parameter("tuple", tuple_shape);
        auto data  = m2.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), tuple);
        auto start = m2.add_instruction(start_op, source);
        auto end   = m2.add_instruction(end_op, source);
        auto slice = m2.add_instruction(slice_op, data, start, end);
        auto materialized = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), slice);
        m2.add_return({materialized});
    }

    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
