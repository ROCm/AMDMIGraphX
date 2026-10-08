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
#include <test.hpp>

static void run_lowering(migraphx::module& m, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        m, {migraphx::gpu::lowering{&ctx, offload_copy}, migraphx::dead_code_elimination{}});
}

// scan_slice's index is a genuine runtime value (normalize_compute_shape never inspects it), so
// a non-literal index reaches lowering unfolded whenever it isn't constant-propagated away first
// - e.g. a convert inserted by eliminate_data_type_for_gpu on a runtime loop counter. The convert
// has no GPU compiler, so it must be rewritten to run on a host copy of its source rather than
// left to dereference GPU memory directly.
TEST_CASE(scan_slice_lowering_converted_runtime_index)
{
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2, 2}};
    migraphx::shape idx64_s{migraphx::shape::int64_type, {1}};
    auto scan_slice_op = migraphx::make_op("scan_slice", {{"axis", 0}, {"direction", 0}});
    auto convert_op = migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}});

    migraphx::module m1;
    {
        auto data   = m1.add_parameter("data", data_s);
        auto idx64  = m1.add_parameter("idx64", idx64_s);
        auto idx32  = m1.add_instruction(convert_op, idx64);
        auto result = m1.add_instruction(scan_slice_op, data, idx32);
        m1.add_return({result});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto data       = m2.add_parameter("data", data_s);
        auto idx64      = m2.add_parameter("idx64", idx64_s);
        auto cpu_src    = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), idx64);
        auto sync       = m2.add_instruction(migraphx::make_op("hip::sync_stream"), cpu_src);
        auto idx32_host = m2.add_instruction(convert_op, sync);
        auto result     = m2.add_instruction(scan_slice_op, data, idx32_host);
        m2.add_return({result});
    }
    EXPECT(m1 == m2);
}

// A scan_slice index that is already a plain runtime parameter (no intervening host op) should
// be copied directly, matching the original pre-fix single-copy behavior.
TEST_CASE(scan_slice_lowering_plain_runtime_index)
{
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2, 2}};
    migraphx::shape idx_s{migraphx::shape::int64_type, {1}};
    auto scan_slice_op = migraphx::make_op("scan_slice", {{"axis", 0}, {"direction", 0}});

    migraphx::module m1;
    {
        auto data   = m1.add_parameter("data", data_s);
        auto idx    = m1.add_parameter("idx", idx_s);
        auto result = m1.add_instruction(scan_slice_op, data, idx);
        m1.add_return({result});
    }
    run_lowering(m1);

    migraphx::module m2;
    {
        auto data    = m2.add_parameter("data", data_s);
        auto idx     = m2.add_parameter("idx", idx_s);
        auto cpu_idx = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), idx);
        auto sync    = m2.add_instruction(migraphx::make_op("hip::sync_stream"), cpu_idx);
        auto result  = m2.add_instruction(scan_slice_op, data, sync);
        m2.add_return({result});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
