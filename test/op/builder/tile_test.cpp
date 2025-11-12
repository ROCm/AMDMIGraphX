/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <op_builder_test_utils.hpp>

#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

namespace {
std::vector<float>
run_with_data(migraphx::module m, const migraphx::shape& input_shape, std::vector<float> data)
{
    migraphx::program p{m};
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(input_shape, data.data());

    migraphx::argument result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    return result_vector;
}
} // namespace

TEST_CASE(tile_op_builder_test)
{
    migraphx::module mm;
    auto input = mm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    auto unsq  = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2}}}), input);
    auto mbcast =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 2, 2}}}), unsq);
    mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), mbcast);

    EXPECT(mm == make_op_module("tile", {{"repeats", {3, 2}}}, mm.get_parameters()));
}

TEST_CASE(tile_repeats_size_mismatch_op_builder_test)
{
    migraphx::module mm;
    mm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});

    EXPECT(test::throws<migraphx::exception>(
        [&] { make_op_module("tile", {{"repeats", {2}}}, mm.get_parameters()); },
        "repeats size mismatch with input shape"));
}

TEST_CASE(tile_verify_2d_op_builder_test)
{
    migraphx::module mm;
    const migraphx::shape sh_data = migraphx::shape{migraphx::shape::float_type, {2, 2}};

    auto a0 = mm.add_parameter("x", sh_data);
    migraphx::op::builder::add("tile", mm, {a0}, {{"repeats", {3, 2}}});

    const std::vector<float> result_vector = run_with_data(mm, sh_data, {1.0, 2.0, 3.0, 4.0});

    /*
    from:
    [ 1.0, 2.0,
      3.0, 4.0 ]

    to:
    [ 1.0, 2.0,   1.0, 2.0,
      3.0, 4.0,   3.0, 4.0,

      1.0, 2.0,   1.0, 2.0,
      3.0, 4.0,   3.0, 4.0,

      1.0, 2.0,   1.0, 2.0,
      3.0, 4.0,   3.0, 4.0 ]
    */

    const std::vector<float> expected_result = {
        1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0,
        3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0,
    };

    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected_result));
}

TEST_CASE(tile_verify_1d_op_builder_test)
{
    migraphx::module mm;
    const migraphx::shape sh_data = migraphx::shape{migraphx::shape::float_type, {1}};

    auto a0 = mm.add_parameter("x", sh_data);
    migraphx::op::builder::add("tile", mm, {a0}, {{"repeats", {5}}});
    const std::vector<float> result_vector   = run_with_data(mm, sh_data, {1.0});
    const std::vector<float> expected_result = {1.0, 1.0, 1.0, 1.0, 1.0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected_result));
}
