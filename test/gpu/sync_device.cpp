/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/sync_device.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::gpu::sync_device{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(inserts_sync_after_copy_from_gpu)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m1;
    auto input  = m1.add_parameter("input", s);
    auto output = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), input);
    m1.add_return({output});
    run_pass(m1);

    migraphx::module m2;
    auto expected_input = m2.add_parameter("input", s);
    auto expected_output =
        m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_input);
    auto expected_sync = m2.add_instruction(migraphx::make_op("hip::sync_stream"), expected_output);
    m2.add_return({expected_sync});
    EXPECT(m1 == m2);
}

TEST_CASE(removes_redundant_syncs_after_multiple_copies)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m1;
    auto x      = m1.add_parameter("x", s);
    auto y      = m1.add_parameter("y", s);
    auto copy_x = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), x);
    auto copy_y = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), y);
    auto sum    = m1.add_instruction(migraphx::make_op("add"), copy_x, copy_y);
    m1.add_return({sum});
    run_pass(m1);

    migraphx::module m2;
    auto expected_x      = m2.add_parameter("x", s);
    auto expected_y      = m2.add_parameter("y", s);
    auto expected_copy_x = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_x);
    auto expected_copy_y = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_y);
    auto expected_sync_y =
        m2.add_instruction(migraphx::make_op("hip::sync_stream"), expected_copy_y);
    auto expected_sum =
        m2.add_instruction(migraphx::make_op("add"), expected_copy_x, expected_sync_y);
    m2.add_return({expected_sum});
    EXPECT(m1 == m2);
}

TEST_CASE(preserves_sync_before_runtime_slice)
{
    migraphx::shape data_shape{migraphx::shape::float_type, {4}};
    migraphx::shape index_shape{migraphx::shape::int64_type, {1}};
    auto slice_op = migraphx::make_op("dyn_slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}});

    migraphx::module m1;
    auto data      = m1.add_parameter("data", data_shape);
    auto starts    = m1.add_parameter("starts", index_shape);
    auto ends      = m1.add_parameter("ends", index_shape);
    auto copy_ends = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), ends);
    auto slice     = m1.add_instruction(slice_op, data, starts, copy_ends);
    auto output    = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), slice);
    m1.add_return({output});
    run_pass(m1);

    migraphx::module m2;
    auto expected_data   = m2.add_parameter("data", data_shape);
    auto expected_starts = m2.add_parameter("starts", index_shape);
    auto expected_ends   = m2.add_parameter("ends", index_shape);
    auto expected_copy_ends =
        m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_ends);
    auto expected_ends_sync =
        m2.add_instruction(migraphx::make_op("hip::sync_stream"), expected_copy_ends);
    auto expected_slice =
        m2.add_instruction(slice_op, expected_data, expected_starts, expected_ends_sync);
    auto expected_output =
        m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_slice);
    auto expected_output_sync =
        m2.add_instruction(migraphx::make_op("hip::sync_stream"), expected_output);
    m2.add_return({expected_output_sync});
    EXPECT(m1 == m2);
}

TEST_CASE(preserves_syncs_separated_by_contextful_instructions)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m1;
    auto x      = m1.add_parameter("x", s);
    auto y      = m1.add_parameter("y", s);
    auto copy_x = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), x);
    auto fill_y = m1.add_instruction(migraphx::make_op("hip::fill"), y);
    auto copy_y = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), fill_y);
    auto sum    = m1.add_instruction(migraphx::make_op("add"), copy_x, copy_y);
    m1.add_return({sum});
    run_pass(m1);

    migraphx::module m2;
    auto expected_x      = m2.add_parameter("x", s);
    auto expected_y      = m2.add_parameter("y", s);
    auto expected_copy_x = m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_x);
    auto expected_sync_x =
        m2.add_instruction(migraphx::make_op("hip::sync_stream"), expected_copy_x);
    auto expected_fill_y = m2.add_instruction(migraphx::make_op("hip::fill"), expected_y);
    auto expected_copy_y =
        m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_fill_y);
    auto expected_sync_y =
        m2.add_instruction(migraphx::make_op("hip::sync_stream"), expected_copy_y);
    auto expected_sum =
        m2.add_instruction(migraphx::make_op("add"), expected_sync_x, expected_sync_y);
    m2.add_return({expected_sum});
    EXPECT(m1 == m2);
}

TEST_CASE(preserves_implicit_copy_return)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m1;
    auto input = m1.add_parameter("input", s);
    m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), input);
    run_pass(m1);

    migraphx::module m2;
    auto expected_input = m2.add_parameter("input", s);
    auto expected_output =
        m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_input);
    auto expected_sync = m2.add_instruction(migraphx::make_op("hip::sync_stream"), expected_output);
    m2.add_return({expected_sync});
    EXPECT(m1 == m2);
}

TEST_CASE(sync_device_is_idempotent)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m;
    auto input  = m.add_parameter("input", s);
    auto output = m.add_instruction(migraphx::make_op("hip::copy_from_gpu"), input);
    m.add_return({output});
    run_pass(m);
    auto expected = m;
    run_pass(m);
    EXPECT(m == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
