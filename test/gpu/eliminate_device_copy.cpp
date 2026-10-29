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
#include <migraphx/gpu/eliminate_device_copy.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m, {migraphx::gpu::eliminate_device_copy{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(copy_to_then_from_gpu_is_eliminated)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m1;
    auto input = m1.add_parameter("input", s);
    auto output =
        m1.add_instruction(migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s)}}));
    auto to_gpu   = m1.add_instruction(migraphx::make_op("hip::copy_to_gpu"), input, output);
    auto from_gpu = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), to_gpu);
    m1.add_return({from_gpu});
    run_pass(m1);

    migraphx::module m2;
    auto expected = m2.add_parameter("input", s);
    m2.add_return({expected});
    EXPECT(m1 == m2);
}

TEST_CASE(copy_to_external_destination_before_copy_from_is_preserved)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m1;
    auto input    = m1.add_parameter("input", s);
    auto output   = m1.add_parameter("output", s);
    auto to_gpu   = m1.add_instruction(migraphx::make_op("hip::copy_to_gpu"), input, output);
    auto from_gpu = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), to_gpu);
    m1.add_return({from_gpu});
    run_pass(m1);

    migraphx::module m2;
    auto expected_input  = m2.add_parameter("input", s);
    auto expected_output = m2.add_parameter("output", s);
    auto expected_to_gpu =
        m2.add_instruction(migraphx::make_op("hip::copy_to_gpu"), expected_input, expected_output);
    auto expected_from_gpu =
        m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_to_gpu);
    m2.add_return({expected_from_gpu});
    EXPECT(m1 == m2);
}

TEST_CASE(copy_from_then_to_gpu_is_eliminated)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m1;
    auto input    = m1.add_parameter("input", s);
    auto from_gpu = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), input);
    auto output =
        m1.add_instruction(migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s)}}));
    auto to_gpu = m1.add_instruction(migraphx::make_op("hip::copy_to_gpu"), from_gpu, output);
    m1.add_return({to_gpu});
    run_pass(m1);

    migraphx::module m2;
    auto expected = m2.add_parameter("input", s);
    m2.add_return({expected});
    EXPECT(m1 == m2);
}

TEST_CASE(copy_to_external_destination_is_preserved)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};

    migraphx::module m1;
    auto input    = m1.add_parameter("input", s);
    auto output   = m1.add_parameter("output", s);
    auto from_gpu = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), input);
    auto to_gpu   = m1.add_instruction(migraphx::make_op("hip::copy_to_gpu"), from_gpu, output);
    m1.add_return({to_gpu});
    run_pass(m1);

    migraphx::module m2;
    auto expected_input  = m2.add_parameter("input", s);
    auto expected_output = m2.add_parameter("output", s);
    auto expected_copy =
        m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_input);
    auto expected_result =
        m2.add_instruction(migraphx::make_op("hip::copy_to_gpu"), expected_copy, expected_output);
    m2.add_return({expected_result});
    EXPECT(m1 == m2);
}

TEST_CASE(eliminates_redundant_copy_before_consumer)
{
    migraphx::shape data_shape{migraphx::shape::float_type, {4}};
    migraphx::shape index_shape{migraphx::shape::int64_type, {1}};
    auto slice_op = migraphx::make_op("dyn_slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}});

    migraphx::module m1;
    auto data        = m1.add_parameter("data", data_shape);
    auto host_starts = m1.add_parameter("starts", index_shape);
    auto ends        = m1.add_parameter("ends", index_shape);
    auto output      = m1.add_instruction(
        migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(index_shape)}}));
    auto starts_to_gpu =
        m1.add_instruction(migraphx::make_op("hip::copy_to_gpu"), host_starts, output);
    auto starts_from_gpu =
        m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), starts_to_gpu);
    auto copy_ends = m1.add_instruction(migraphx::make_op("hip::copy_from_gpu"), ends);
    auto slice     = m1.add_instruction(slice_op, data, starts_from_gpu, copy_ends);
    m1.add_return({slice});
    run_pass(m1);

    migraphx::module m2;
    auto expected_data   = m2.add_parameter("data", data_shape);
    auto expected_starts = m2.add_parameter("starts", index_shape);
    auto expected_ends   = m2.add_parameter("ends", index_shape);
    auto expected_copy_ends =
        m2.add_instruction(migraphx::make_op("hip::copy_from_gpu"), expected_ends);
    auto expected_slice =
        m2.add_instruction(slice_op, expected_data, expected_starts, expected_copy_ends);
    m2.add_return({expected_slice});
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
