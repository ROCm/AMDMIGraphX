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

#include <op_builder_test_utils.hpp>

#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/common.hpp>
#include <migraphx/op/builder/insert.hpp>

#include <numeric>

namespace {
std::vector<float>
run_with_data(migraphx::module m, const migraphx::shape& input_shape, std::vector<float> data)
{
    migraphx::program p{std::move(m)};
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

TEST_CASE(tile_verify_dynamic_2d_op_builder_test)
{
    migraphx::module mm;
    const migraphx::shape sh_param = migraphx::shape{
        migraphx::shape::float_type, {{1, 1}, {1, 1}, {1, 1}, {2, 65}}};

    auto a0 = mm.add_parameter("x", sh_param);
    migraphx::op::builder::add("tile", mm, {a0}, {{"repeats", {1, 15, 1, 1}}});

    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0};
    const migraphx::shape sh_data{migraphx::shape::float_type, {1, 1, 1, 4}};

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(sh_data, input_data.data());

    migraphx::argument result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> expected_result;
    expected_result.reserve(15 * input_data.size());
    for(std::size_t i = 0; i < 15; ++i)
        expected_result.insert(expected_result.end(), input_data.begin(), input_data.end());

    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected_result));
}

TEST_CASE(tile_dynamic_op_builder_test)
{
    migraphx::module mm;
    auto input = mm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {{2, 2}, {2, 2}}});
    auto unsq  = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2}}}), input);
    std::vector<migraphx::shape::dynamic_dimension> bcast_dims{
        {3, 3}, {2, 2}, {2, 2}, {2, 2}};
    auto mbcast = mm.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(bcast_dims)}}),
        unsq,
        unsq);
    std::vector<migraphx::shape::dynamic_dimension> reshape_dims{
        migraphx::shape::dynamic_dimension{6, 6}, migraphx::shape::dynamic_dimension{4, 4}};
    mm.add_instruction(
        migraphx::make_op("reshape", {{"dims", migraphx::to_value(reshape_dims)}}), mbcast);
    EXPECT(mm == make_op_module("tile", {{"repeats", {3, 2}}}, mm.get_parameters()));
}

TEST_CASE(tile_verify_dynamic_rank2_op_builder_test)
{
    migraphx::module mm;
    const migraphx::shape sh_param = migraphx::shape{migraphx::shape::float_type, {{3, 3}, {2, 2}}};

    auto a0 = mm.add_parameter("x", sh_param);
    migraphx::op::builder::add("tile", mm, {a0}, {{"repeats", {2, 3}}});

    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const migraphx::shape sh_data{migraphx::shape::float_type, {3, 2}};

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(sh_data, input_data.data());

    migraphx::argument result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> expected_result;
    for(std::size_t rep0 = 0; rep0 < 2; ++rep0)
    {
        for(std::size_t row = 0; row < 3; ++row)
        {
            for(std::size_t rep1 = 0; rep1 < 3; ++rep1)
            {
                for(std::size_t col = 0; col < 2; ++col)
                    expected_result.push_back(input_data[row * 2 + col]);
            }
        }
    }

    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected_result));
}

TEST_CASE(tile_verify_dynamic_variable_width_op_builder_test)
{
    const migraphx::shape sh_param = migraphx::shape{
        migraphx::shape::float_type, {{1, 1}, {1, 1}, {1, 1}, {2, 65}}};

    for(std::size_t width : {3, 5})
    {
        std::vector<float> input_data(width);
        std::iota(input_data.begin(), input_data.end(), 1.f);
        const migraphx::shape sh_data{migraphx::shape::float_type, {1, 1, 1, width}};

        migraphx::module mm_run;
        auto param = mm_run.add_parameter("x", sh_param);
        migraphx::op::builder::add("tile", mm_run, {param}, {{"repeats", {1, 1, 1, 3}}});

        migraphx::program p{std::move(mm_run)};
        p.compile(migraphx::make_target("ref"));

        migraphx::parameter_map pp;
        pp["x"] = migraphx::argument(sh_data, input_data.data());

        migraphx::argument result = p.eval(pp).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

        std::vector<float> expected_result;
        expected_result.reserve(3 * width);
        for(std::size_t i = 0; i < 3; ++i)
            expected_result.insert(expected_result.end(), input_data.begin(), input_data.end());

        EXPECT(migraphx::verify::verify_rms_range(result_vector, expected_result));
    }
}

TEST_CASE(tile_dynamic_range_dims_shape_error)
{
    migraphx::module mm;
    auto input = mm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {{2, 4}, {2, 4}}});
    migraphx::op::builder::add("tile", mm, {input}, {{"repeats", {2, 3}}});

    migraphx::program p{std::move(mm)};
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const migraphx::shape sh_data{migraphx::shape::float_type, {3, 2}};

    EXPECT(test::throws<migraphx::exception>(
        [&] {
            migraphx::parameter_map pp;
            pp["x"] = migraphx::argument(sh_data, input_data.data());
            std::ignore = p.eval(pp);
        },
        "Reshape: Dimensions for reshape can only have one -1 dim"));
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
