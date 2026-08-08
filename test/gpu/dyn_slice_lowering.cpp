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
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>

static void run_lowering(migraphx::module& m, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        m, {migraphx::gpu::lowering{&ctx, offload_copy}, migraphx::dead_code_elimination{}});
}

// After lowering, a slice with runtime inputs should have hip::copy_from_gpu
// and hip::sync_stream inserted for the metadata inputs.
TEST_CASE(dyn_slice_lowering_runtime_inputs)
{
    migraphx::shape data_s{migraphx::shape::float_type, {{2, 4}, {2, 4}, {3, 8}}};
    migraphx::shape idx_s{migraphx::shape::int32_type, {1}};

    migraphx::module m1;
    {
        auto data   = m1.add_parameter("data", data_s);
        auto starts = m1.add_parameter("starts", idx_s);
        auto ends   = m1.add_parameter("ends", idx_s);
        auto sl =
            m1.add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), data, starts, ends);
        m1.add_return({sl});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto data        = m2.add_parameter("data", data_s);
        auto starts      = m2.add_parameter("starts", idx_s);
        auto ends        = m2.add_parameter("ends", idx_s);
        auto copy_starts = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), starts);
        auto copy_ends   = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), ends);
        auto sync =
            m2.add_instruction(migraphx::make_op("hip::sync_stream"), copy_starts, copy_ends);
        auto sl =
            m2.add_instruction(migraphx::make_op("slice", {{"axes", {2}}}), data, sync, copy_ends);
        m2.add_return({sl});
    }
    EXPECT(m1 == m2);
}

// dyn_slice always has runtime bound inputs, so both of them are copied to the host.
TEST_CASE(dyn_slice_lowering_dyn_slice_op)
{
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2, 4}};
    migraphx::shape idx_s{migraphx::shape::int64_type, {1}};
    auto op = migraphx::make_op("dyn_slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}});

    migraphx::module m1;
    {
        auto data   = m1.add_parameter("data", data_s);
        auto starts = m1.add_parameter("starts", idx_s);
        auto ends   = m1.add_parameter("ends", idx_s);
        auto sl     = m1.add_instruction(op, data, starts, ends);
        m1.add_return({sl});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto data        = m2.add_parameter("data", data_s);
        auto starts      = m2.add_parameter("starts", idx_s);
        auto ends        = m2.add_parameter("ends", idx_s);
        auto copy_starts = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), starts);
        auto copy_ends   = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), ends);
        auto sync =
            m2.add_instruction(migraphx::make_op("hip::sync_stream"), copy_starts, copy_ends);
        auto sl = m2.add_instruction(op, data, sync, copy_ends);
        m2.add_return({sl});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(dyn_slice_lowering_mixed_host_and_device_metadata)
{
    using dd  = migraphx::shape::dynamic_dimension;
    auto n    = migraphx::sym::var("N", {1, 4});
    auto zero = migraphx::sym::lit(0);
    migraphx::shape source_shape{migraphx::shape::float_type, {dd{n}}};
    migraphx::shape data_shape{migraphx::shape::float_type, {4}};
    migraphx::shape index_shape{migraphx::shape::int64_type, {1}};
    auto eval_op = migraphx::make_op(
        "eval_expr_from_shape",
        {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{zero})}});
    auto slice_op =
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", migraphx::to_value(std::vector<migraphx::sym::expr>{zero})},
                           {"ends", migraphx::to_value(std::vector<migraphx::sym::expr>{n})}});

    migraphx::module m1;
    {
        auto source = m1.add_parameter("source", source_shape);
        auto data   = m1.add_parameter("data", data_shape);
        auto ends   = m1.add_parameter("ends", index_shape);
        auto starts = m1.add_instruction(eval_op, source);
        auto slice  = m1.add_instruction(slice_op, data, starts, ends);
        m1.add_return({slice});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto source    = m2.add_parameter("source", source_shape);
        auto data      = m2.add_parameter("data", data_shape);
        auto ends      = m2.add_parameter("ends", index_shape);
        auto starts    = m2.add_instruction(eval_op, source);
        auto copy_ends = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), ends);
        auto sync      = m2.add_instruction(migraphx::make_op("hip::sync_stream"), copy_ends);
        auto slice     = m2.add_instruction(slice_op, data, starts, sync);
        m2.add_return({slice});
    }
    EXPECT(m1 == m2);
}

// A slice with only 1 input (all attributes inline) should not be modified
// by the dynamic slice lowering.
TEST_CASE(dyn_slice_lowering_single_input)
{
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2, 4}};

    migraphx::module m1;
    {
        auto data = m1.add_parameter("data", data_s);
        auto sl   = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), data);
        m1.add_return({sl});
    }
    auto m2 = m1;
    run_lowering(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
